// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The device: `libttsim` through `dlopen`, one TLB window, DRAM and L1 access,
//! and starting a core on an image.
//!
//! `libttsim` is a process-wide singleton with no context handle. Every contract
//! violation terminates the process (`_Exit`), so nothing here hands it an
//! address it does not decode: the window geometry is checked before each call,
//! the TLB configuration is the one the library accepts, and L1 access is
//! bounded by [`tt_isa::tensix::L1_SIZE`].
//!
//! The simulator is loaded from `$TT_TTSIM_LIB`, or `libttsim_bh.so` in the
//! current directory. It is initialized once per process; [`Device`] can be
//! dropped and reopened without re-initializing it.

use std::ffi::{OsStr, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;

use libloading::{Library, Symbol};

use crate::ir::Error;
use tt_isa::dram::{CHANNEL_BYTES, Dram, DramChannel};
use tt_isa::l1;
use tt_isa::mailbox::{self, status};
use tt_isa::noc::niu::Endpoint;
use tt_isa::noc::{Noc0, NocCoord};
use tt_isa::tensix::{self, Core};

/// The environment variable that names the simulator's shared object.
pub const LIB_PATH_ENV: &str = "TT_TTSIM_LIB";

/// Where `libttsim` decodes chip 0's BARs, and where the TLB configuration
/// array lives. These are compile-time constants inside the library, not
/// values derived from configuration space.
const BAR0_BASE: u64 = 0x1_0000_0000;
const TLB_CONFIG_BASE: u64 = 0x1FC0_0000;

/// The one 2 MiB window this driver uses. Window 201 belongs to the kernel
/// driver, so it is never taken.
const WINDOW: u16 = 0;
const WINDOW_SIZE: u64 = 2 * 1024 * 1024;

/// Device ID of a Blackhole.
const VENDOR_ID_TENSTORRENT: u16 = 0x1E52;
const DEVICE_ID_BLACKHOLE: u16 = 0xB140;

/// Simulated cycles one core may take to answer before it is declared stuck.
pub const BUDGET: u64 = 400_000;

/// Simulated cycles advanced between two polls of a mailbox word.
const CYCLES_PER_POLL: u32 = 512;

type DmaRead = unsafe extern "C" fn(u64, *mut c_void, u32);
type DmaWrite = unsafe extern "C" fn(u64, *const c_void, u32);
type FnInit = unsafe extern "C" fn();
type FnSetDma = unsafe extern "C" fn(DmaRead, DmaWrite);
type FnConfigRd32 = unsafe extern "C" fn(u32, u32) -> u32;
type FnMemRd = unsafe extern "C" fn(u64, *mut c_void, u32);
type FnMemWr = unsafe extern "C" fn(u64, *const c_void, u32);
type FnClock = unsafe extern "C" fn(u32);

/// The loaded simulator, with the entry points this driver calls.
struct Lib {
    pci_config_rd32: FnConfigRd32,
    pci_mem_rd_bytes: FnMemRd,
    pci_mem_wr_bytes: FnMemWr,
    clock: FnClock,
    /// Kept alive alongside the function pointers above.
    _library: Library,
}

// The library is loaded once and its entry points are plain `extern "C"`
// functions. There is no device state in this struct, so moving it between
// threads is sound; the device state lives behind the runtime's mutex.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

impl Lib {
    /// `dlopen` the simulator and resolve the entry points. The library is
    /// initialized on the first successful load, once per process.
    fn open(path: &OsStr) -> Result<&'static Lib, Error> {
        static LIB: OnceLock<Result<&'static Lib, Error>> = OnceLock::new();
        LIB.get_or_init(|| Self::load(path))
            .as_ref()
            .map(|lib| *lib)
            .map_err(|e| *e)
    }

    fn load(path: &OsStr) -> Result<&'static Lib, Error> {
        // SAFETY: the caller named a libttsim shared object; loading a shared
        // object runs its initializers, which is what the caller asked for.
        let library = unsafe { Library::new(path) }.map_err(|e| {
            let _ = (e, PathBuf::from(path));
            Error::TT_ERROR_DLOPEN
        })?;
        macro_rules! sym {
            ($name:literal) => {{
                let s: Symbol<'_, _> = unsafe { library.get(concat!($name, "\0").as_bytes()) }
                    .map_err(|e| {
                        let _ = (e, $name);
                        Error::TT_ERROR_DLSYM
                    })?;
                *s
            }};
        }
        let set_dma: FnSetDma = sym!("libttsim_set_pci_dma_mem_callbacks");
        let init: FnInit = sym!("libttsim_init");
        let lib = Lib {
            pci_config_rd32: sym!("libttsim_pci_config_rd32"),
            pci_mem_rd_bytes: sym!("libttsim_pci_mem_rd_bytes"),
            pci_mem_wr_bytes: sym!("libttsim_pci_mem_wr_bytes"),
            clock: sym!("libttsim_clock"),
            _library: library,
        };
        // Callbacks must be installed before `init`, and there is no way to
        // tell the library "no": the baseline performs no DMA, so a callback
        // here means a descriptor named host memory by mistake.
        // SAFETY: both are `extern "C"` functions of the required signatures,
        // and the simulator is not running yet.
        unsafe { set_dma(dma_read_unexpected, dma_write_unexpected) };
        // SAFETY: this is the first and only init of the process.
        unsafe { init() };
        Ok(Box::leak(Box::new(lib)))
    }

    fn config_rd32(&self, bdf: u32, offset: u32) -> u32 {
        // SAFETY: the simulator is initialized and an absent slot reads
        // all-ones rather than faulting.
        unsafe { (self.pci_config_rd32)(bdf, offset) }
    }

    fn mem_write(&self, paddr: u64, bytes: &[u8]) {
        debug_assert!(!bytes.is_empty() && bytes.len() <= u32::MAX as usize);
        // SAFETY: every caller has validated the address against the decode map.
        unsafe { (self.pci_mem_wr_bytes)(paddr, bytes.as_ptr().cast(), bytes.len() as u32) };
    }

    fn mem_read(&self, paddr: u64, out: &mut [u8]) {
        debug_assert!(!out.is_empty() && out.len() <= u32::MAX as usize);
        // SAFETY: as above.
        unsafe { (self.pci_mem_rd_bytes)(paddr, out.as_mut_ptr().cast(), out.len() as u32) };
    }

    fn clock(&self, n: u32) {
        // SAFETY: the simulator is initialized and calls are serialized.
        unsafe { (self.clock)(n) };
    }
}

/// A misdirected DMA is a bug in the code that programmed the device, and
/// there is no way to report it to the library. Stop.
unsafe extern "C" fn dma_read_unexpected(paddr: u64, _dst: *mut c_void, size: u32) {
    eprintln!("libTT: the device attempted a DMA read of {size} bytes at host address {paddr:#x}");
    std::process::abort()
}

unsafe extern "C" fn dma_write_unexpected(paddr: u64, _src: *const c_void, size: u32) {
    eprintln!("libTT: the device attempted a DMA write of {size} bytes at host address {paddr:#x}");
    std::process::abort()
}

/// One simulated Blackhole, reached through a TLB window.
pub struct Device {
    lib: &'static Lib,
    /// The configuration the window is known to hold. The registers are
    /// write-only, so this is the only record of the current mapping.
    shadow: Option<[u32; 3]>,
    /// The Tensix tile a 1x1 launch runs on.
    tile: NocCoord<Noc0>,
    /// DRAM channels that can hold data.
    dram: Dram,
    /// First byte of DRAM this device has not handed out.
    dram_used: u64,
    /// Is `exec_b` resident and looping on the tile?
    exec_started: bool,
    /// Programs already placed at their L1 address: `(address, words, hash)`.
    /// The program cache keeps them, and a launch only names the address; an
    /// identical program is not rewritten.
    placed: Vec<(u32, u64, u64)>,
}

impl Device {
    /// Load the simulator if needed and take the device.
    pub fn open() -> Result<Device, Error> {
        let path = std::env::var_os(LIB_PATH_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("libttsim_bh.so"));
        let lib = Lib::open(path.as_os_str())?;

        // Chip 0 must answer configuration space, and it must be a Blackhole:
        // every address constant above is Blackhole's.
        let id = lib.config_rd32(0, 0);
        if id == u32::MAX {
            return Err(Error::TT_ERROR_NO_DEVICE);
        }
        let vendor = (id & 0xFFFF) as u16;
        let device = (id >> 16) as u16;
        if vendor != VENDOR_ID_TENSTORRENT || device != DEVICE_ID_BLACKHOLE {
            let _ = (vendor, device);
            return Err(Error::TT_ERROR_NOT_BLACKHOLE);
        }

        let tile = tt_isa::noc::grid::Tensix::FULL
            .tiles::<Noc0>()
            .next()
            .ok_or(Error::TT_ERROR_NO_TENSIX_TILE)?;
        let mut device = Device {
            lib,
            shadow: None,
            tile,
            dram: Dram::FULL,
            dram_used: 0x10_0000,
            exec_started: false,
            placed: Vec::new(),
        };
        // A previous process (`ttRelease` only drops this struct) leaves its
        // cores running and its mailboxes written. Reopening is the 0 -> 1
        // transition, so it cannot disturb a live caller: hold the five cores
        // and clear the launch state here, making the next launch start from a
        // known tile.
        device.quiesce()?;
        Ok(device)
    }

    /// The tile a 1x1 launch runs on.
    pub fn tile(&self) -> NocCoord<Noc0> {
        self.tile
    }

    /// A GDDR channel that can hold data.
    pub fn dram_channel(&self) -> DramChannel {
        self.dram
            .channel(0)
            .expect("channel 0 is usable on a full chip")
    }

    /// Reserve `bytes` of DRAM, 64-byte aligned, and return its channel-0
    /// offset. This is what a caller passes back to `ttMemcpy` and `ttLaunch`.
    pub fn alloc_dram(&mut self, bytes: usize) -> Result<u64, Error> {
        if bytes == 0 {
            return Err(Error::TT_ERROR_ALLOC_ZERO);
        }
        let at = self.dram_used.next_multiple_of(64);
        let end = at
            .checked_add(bytes as u64)
            .ok_or(Error::TT_ERROR_DRAM_ADDRESS_OVERFLOW)?;
        if end > CHANNEL_BYTES {
            let _ = (bytes, at);
            return Err(Error::TT_ERROR_DRAM_FULL);
        }
        self.dram_used = end;
        Ok(at)
    }

    /// The endpoint high word a program names for DRAM on channel 0.
    pub fn dram_hi(&self) -> u32 {
        let channel = self.dram_channel();
        let ep = self.channel_endpoint(channel);
        Endpoint {
            x: ep.x(),
            y: ep.y(),
            addr: 0,
        }
        .hi()
    }

    /// Write `bytes` to DRAM at a channel-0 offset.
    pub fn dram_write(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        let channel = self.dram_channel();
        let range = channel.range(offset, bytes.len() as u64).ok_or_else(|| {
            let _ = (bytes.len(), offset);
            Error::TT_ERROR_DRAM_RANGE
        })?;
        let ep = self.channel_endpoint(channel);
        self.window_write(ep, range.offset(), bytes)?;
        // A write is visible through the port it went out of; a reader may use
        // any endpoint NoC #0 owns, so read the last word back through each.
        if bytes.len() >= 4 {
            let last = range.offset() + bytes.len() as u64 - 4;
            let mut word = [0u8; 4];
            for port in 0..tt_isa::dram::PORTS {
                if let Some(via) = channel.endpoint(tt_isa::noc::niu::Niu::Noc0, port) {
                    self.window_read(via, last, &mut word)?;
                }
            }
        }
        Ok(())
    }

    /// Read DRAM at a channel-0 offset.
    pub fn dram_read(&mut self, offset: u64, out: &mut [u8]) -> Result<(), Error> {
        let channel = self.dram_channel();
        channel.range(offset, out.len() as u64).ok_or_else(|| {
            let _ = out.len();
            Error::TT_ERROR_DRAM_RANGE
        })?;
        let ep = self.channel_endpoint(channel);
        self.window_read(ep, offset, out)
    }

    /// Write into this tile's L1.
    pub fn l1_write(&mut self, address: u64, bytes: &[u8]) -> Result<(), Error> {
        self.check_l1(address, bytes.len())?;
        let tile = self.tile;
        self.window_write(tile, address, bytes)
    }

    /// Read from this tile's L1.
    pub fn l1_read(&mut self, address: u64, out: &mut [u8]) -> Result<(), Error> {
        self.check_l1(address, out.len())?;
        let tile = self.tile;
        self.window_read(tile, address, out)
    }

    /// Read one dword from the tile (an L1 word or a core register).
    pub fn read32(&mut self, address: u64) -> Result<u32, Error> {
        let mut b = [0u8; 4];
        let tile = self.tile;
        self.window_read(tile, address, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// Write one dword to the tile (an L1 word or a core register).
    pub fn write32(&mut self, address: u64, value: u32) -> Result<(), Error> {
        let tile = self.tile;
        self.window_write(tile, address, &value.to_le_bytes())
    }

    /// Advance simulated time. Nothing happens on the device outside this.
    pub fn clock(&self, n: u32) {
        self.lib.clock(n);
    }

    /// Load `image` into L1 and start `core` on it.
    pub fn load_and_start(&mut self, core: Core, image: &[u8], at: u64) -> Result<(), Error> {
        self.load_held(core, image, at)?;
        self.set_core_reset(core, false)
    }

    /// Load `image` at `at` and leave `core` held in reset.
    ///
    /// Everything [`Device::load_and_start`] does except the release: the image,
    /// the reset PC and a cleared status word are in place, so one later write to
    /// `SOFT_RESET_0` starts this core and every other held one at once.
    fn load_held(&mut self, core: Core, image: &[u8], at: u64) -> Result<(), Error> {
        self.set_core_reset(core, true)?;
        self.l1_write(at, image)?;
        // A stale status word would satisfy the wait before the image runs.
        self.write32(mailbox::STATUS, 0)?;
        match core.reset_pc_register() {
            Some(pc) => {
                self.write32(pc, at as u32)?;
                let (reg, bit) = core
                    .reset_pc_override()
                    .expect("a core with a reset PC has an override");
                let current = self.read32(reg)?;
                self.write32(reg, current | bit)?;
            }
            None if at == u64::from(core.default_reset_pc()) => {}
            None => {
                let _ = (core, at);
                return Err(Error::TT_ERROR_RESET_PC);
            }
        }
        Ok(())
    }

    /// Load every image and release all their cores with **one** write to
    /// `SOFT_RESET_0`, so cores meant to run alongside each other do.
    pub fn load_and_start_together(
        &mut self,
        images: &[crate::images::Image],
    ) -> Result<(), Error> {
        let mut release = 0u32;
        for &(core, image, at) in images {
            self.load_held(core, image, at)?;
            release |= core.soft_reset_mask();
        }
        let current = self.read32(tensix::SOFT_RESET_0)?;
        self.write32(tensix::SOFT_RESET_0, current & !release)
    }

    /// Release the Tensix backend — unpackers, packers, Matrix Unit, Vector Unit
    /// and the rest — from soft reset, leaving the five RISC-V cores alone.
    pub fn release_tensix_backend(&mut self) -> Result<(), Error> {
        let current = self.read32(tensix::SOFT_RESET_0)?;
        self.write32(tensix::SOFT_RESET_0, current & !tensix::BACKEND_RESET_MASK)
    }

    /// Wait until `predicate` accepts the word at `address`.
    pub fn wait_for(
        &mut self,
        address: u64,
        mut predicate: impl FnMut(u32) -> bool,
    ) -> Result<u32, Error> {
        let mut waited = 0u64;
        loop {
            let value = self.read32(address)?;
            if predicate(value) {
                return Ok(value);
            }
            if value == status::PANICKED {
                let code = self.read32(mailbox::PANIC_CODE)?;
                let _ = code;
                return Err(Error::TT_ERROR_FIRMWARE_PANIC);
            }
            if waited >= BUDGET {
                let _ = value;
                return Err(Error::TT_ERROR_FIRMWARE_TIMEOUT);
            }
            self.clock(CYCLES_PER_POLL);
            waited += u64::from(CYCLES_PER_POLL);
        }
    }

    /// Stop every core this driver can start and clear the launch mailboxes it
    /// wrote, so a failed launch leaves nothing polling the L1 that the next
    /// launch rewrites.
    ///
    /// The failure path has no `ttReset` (user decision): this is the driver's
    /// own cleanup. Every step is attempted even if one fails, and the first
    /// failure is returned, so a wedged tile is reported rather than hidden.
    pub fn quiesce(&mut self) -> Result<(), Error> {
        let mut first: Result<(), Error> = Ok(());
        for core in [Core::B, Core::T0, Core::T1, Core::T2] {
            if let Err(e) = self.set_core_reset(core, true) {
                first = first.and(Err(e));
            }
        }
        for mb in Self::launch_mailboxes() {
            // Generation first: a core released before the rest lands must not
            // run the named program with a stale start.
            for at in [
                mb.generation(),
                mb.program_addr(),
                mb.program_len(),
                mb.args(),
                mb.ack(),
                mb.result(),
                mb.status(),
                mb.panic_code(),
            ] {
                if let Err(e) = self.write32(at, 0) {
                    first = first.and(Err(e));
                }
            }
        }
        // The executor on B is gone with the reset; the next launch stages it
        // again instead of waiting on a mailbox nobody will write.
        self.exec_started = false;
        self.placed.clear();
        first
    }

    /// The four launch mailboxes: RISCV B and the three Tensix threads.
    fn launch_mailboxes() -> [mailbox::launch::Mailbox; 4] {
        [
            mailbox::launch::Mailbox::b(),
            mailbox::launch::Mailbox::of(0),
            mailbox::launch::Mailbox::of(1),
            mailbox::launch::Mailbox::of(2),
        ]
    }

    /// Hold `core` in reset, stopping whatever it is running.
    pub fn hold(&mut self, core: Core) -> Result<(), Error> {
        self.set_core_reset(core, true)
    }

    /// Boot the launch executor on RISCV B, once.
    pub fn start_executor(&mut self) -> Result<(), Error> {
        if self.exec_started {
            return Ok(());
        }
        self.load_and_start(
            Core::B,
            crate::images::EXEC_B,
            u64::from(Core::B.default_reset_pc()),
        )?;
        let mb = mailbox::launch::Mailbox::b();
        self.wait_for(mb.status(), |s| s == status::RUNNING)?;
        // A previous process leaves a program named and a generation set; a new
        // executor must not replay it.
        self.clear_launch(mb)?;
        self.exec_started = true;
        Ok(())
    }

    /// Clear one launch message: the generation first, then the program and
    /// control words.
    fn clear_launch(&mut self, mb: mailbox::launch::Mailbox) -> Result<(), Error> {
        self.write32(mb.generation(), 0)?;
        for at in [
            mb.program_addr(),
            mb.program_len(),
            mb.args(),
            mb.ack(),
            mb.result(),
            mb.status(),
            mb.panic_code(),
        ] {
            self.write32(at, 0)?;
        }
        Ok(())
    }

    /// Place `words` at `at` in the program cache, unless this device put the
    /// same words there already. The launch message names the address; it does
    /// not carry the program (tt-metal's `kernel_text_offset`).
    pub fn place(&mut self, at: u32, words: &[u32]) -> Result<(), Error> {
        if words.is_empty() {
            return Err(Error::TT_ERROR_EMPTY_PROGRAM);
        }
        if words.len() as u64 > u64::from(mailbox::MAX_WORDS) {
            let _ = words.len();
            return Err(Error::TT_ERROR_PROGRAM_TOO_LONG);
        }
        // FNV-1a: enough to skip rewriting an unchanged program, not a
        // correctness check on the words.
        let hash = words.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, word| {
            (h ^ u64::from(*word)).wrapping_mul(0x0000_0100_0000_01b3)
        });
        let len = words.len() as u64;
        if self
            .placed
            .iter()
            .any(|&(a, l, h)| a == at && l == len && h == hash)
        {
            return Ok(());
        }
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.l1_write(u64::from(at), &code)?;
        self.placed.retain(|&(a, _, _)| a != at);
        self.placed.push((at, len, hash));
        Ok(())
    }

    /// Write one launch message and start it: the text address, its length, the
    /// arguments base, then a fresh generation. Returns that generation.
    pub fn launch(
        &mut self,
        mb: mailbox::launch::Mailbox,
        text: u32,
        words: u32,
        args: u32,
    ) -> Result<u32, Error> {
        self.write32(mb.program_addr(), text)?;
        self.write32(mb.program_len(), words)?;
        self.write32(mb.args(), args)?;
        self.write32(mb.ack(), 0)?;
        let generation = self.read32(mb.generation())?.wrapping_add(1).max(1);
        self.write32(mb.generation(), generation)?;
        Ok(generation)
    }

    /// Wait for a launch's `ACK`, or report its panic or a timeout.
    pub fn await_launch(
        &mut self,
        mb: mailbox::launch::Mailbox,
        generation: u32,
    ) -> Result<(), Error> {
        let mut waited = 0u64;
        loop {
            if self.read32(mb.ack())? == generation {
                return Ok(());
            }
            if self.read32(mb.status())? == status::PANICKED {
                return Err(Error::TT_ERROR_PROGRAM_REFUSED);
            }
            if waited >= BUDGET {
                return Err(Error::TT_ERROR_FIRMWARE_TIMEOUT);
            }
            self.clock(CYCLES_PER_POLL);
            waited += u64::from(CYCLES_PER_POLL);
        }
    }

    /// Place a RISC-V program and its launch arguments, run it on RISCV B to
    /// completion, and report a refusal.
    pub fn run_program(
        &mut self,
        program_at: u32,
        args_at: u32,
        words: &[u32],
        args: &[u32],
    ) -> Result<u32, Error> {
        if words.is_empty() {
            return Err(Error::TT_ERROR_EMPTY_PROGRAM);
        }
        if (program_at as u64) + 4 * words.len() as u64 > args_at as u64 {
            let _ = (program_at, args_at);
            return Err(Error::TT_ERROR_PROGRAM_OVERLAPS_ARGS);
        }
        self.place(program_at, words)?;
        let args_bytes: Vec<u8> = args.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.l1_write(u64::from(args_at), &args_bytes)?;
        let mb = mailbox::launch::Mailbox::b();
        let generation = self.launch(mb, program_at, words.len() as u32, args_at)?;
        self.await_launch(mb, generation)?;
        self.read32(mb.result())
    }

    /// Point the window at `address` in `tile`, unless it already is.
    fn retarget(&mut self, tile: NocCoord<Noc0>, address: u64) -> Result<(), Error> {
        let words = tlb_words(address, tile)?;
        if self.shadow == Some(words) {
            return Ok(());
        }
        for (i, word) in words.iter().enumerate() {
            self.bar_write32(TLB_CONFIG_BASE + 4 * i as u64, *word)?;
        }
        self.shadow = Some(words);
        Ok(())
    }

    fn window_write(
        &mut self,
        tile: NocCoord<Noc0>,
        address: u64,
        bytes: &[u8],
    ) -> Result<(), Error> {
        let mut done = 0usize;
        while done < bytes.len() {
            let at = address + done as u64;
            let base = at & !(WINDOW_SIZE - 1);
            let offset = at - base;
            let here = (bytes.len() - done).min((WINDOW_SIZE - offset) as usize);
            self.retarget(tile, base)?;
            let bar = window_bar_offset() + offset;
            check_bar0(bar, here)?;
            self.lib
                .mem_write(BAR0_BASE + bar, &bytes[done..done + here]);
            done += here;
        }
        Ok(())
    }

    fn window_read(
        &mut self,
        tile: NocCoord<Noc0>,
        address: u64,
        out: &mut [u8],
    ) -> Result<(), Error> {
        let mut done = 0usize;
        while done < out.len() {
            let at = address + done as u64;
            let base = at & !(WINDOW_SIZE - 1);
            let offset = at - base;
            let here = (out.len() - done).min((WINDOW_SIZE - offset) as usize);
            self.retarget(tile, base)?;
            let bar = window_bar_offset() + offset;
            check_bar0(bar, here)?;
            self.lib
                .mem_read(BAR0_BASE + bar, &mut out[done..done + here]);
            done += here;
        }
        Ok(())
    }

    fn bar_write32(&self, offset: u64, value: u32) -> Result<(), Error> {
        check_bar0(offset, 4)?;
        self.lib.mem_write(BAR0_BASE + offset, &value.to_le_bytes());
        Ok(())
    }

    fn channel_endpoint(&self, channel: DramChannel) -> NocCoord<Noc0> {
        channel
            .endpoint(
                tt_isa::noc::niu::Niu::Noc0,
                channel.port_for(tt_isa::noc::niu::Niu::Noc0, 0),
            )
            .expect("NoC #0's first endpoint")
    }

    fn check_l1(&self, address: u64, len: usize) -> Result<(), Error> {
        let end = address
            .checked_add(len as u64)
            .ok_or(Error::TT_ERROR_L1_ADDRESS_OVERFLOW)?;
        if end > tensix::L1_SIZE {
            let _ = (address, end);
            return Err(Error::TT_ERROR_L1_ACCESS);
        }
        Ok(())
    }

    fn set_core_reset(&mut self, core: Core, held: bool) -> Result<(), Error> {
        let current = self.read32(tensix::SOFT_RESET_0)?;
        let updated = if held {
            current | core.soft_reset_mask()
        } else {
            current & !core.soft_reset_mask()
        };
        if updated != current {
            self.write32(tensix::SOFT_RESET_0, updated)?;
        }
        Ok(())
    }
}

/// The byte offset of this driver's window within BAR0.
const fn window_bar_offset() -> u64 {
    WINDOW as u64 * WINDOW_SIZE
}

/// Refuse an access BAR0 would not decode: past the window aperture, or into
/// the configuration array.
fn check_bar0(offset: u64, len: usize) -> Result<(), Error> {
    if len == 0 {
        return Err(Error::TT_ERROR_BAR0_ZERO_LENGTH);
    }
    let end = offset
        .checked_add(len as u64)
        .ok_or(Error::TT_ERROR_BAR0_OVERFLOW)?;
    let in_aperture = end <= 202 * WINDOW_SIZE;
    let in_config = offset >= TLB_CONFIG_BASE && end <= TLB_CONFIG_BASE + 210 * 12;
    if !in_aperture && !in_config {
        let _ = (offset, end);
        return Err(Error::TT_ERROR_BAR0_NOT_DECODED);
    }
    Ok(())
}

/// Encode the three dwords of a 2 MiB unicast window on NoC #0, strict ordering.
fn tlb_words(base_address: u64, tile: NocCoord<Noc0>) -> Result<[u32; 3], Error> {
    if base_address % WINDOW_SIZE != 0 {
        let _ = base_address;
        return Err(Error::TT_ERROR_WINDOW_ALIGNMENT);
    }
    // Field positions, `HostToDeviceTLBs.md`: local offset 43 bits, then the
    // unicast end coordinate, NoC index and ordering.
    const OFFSET_BITS: u32 = 43;
    const X_END: u32 = 43;
    const Y_END: u32 = 49;
    const NOC: u32 = 67;
    const ORDERING: u32 = 70;
    /// Strict AXI response ordering.
    const STRICT_AXI: u64 = 1;

    let mut words = [0u32; 3];
    place(
        &mut words,
        0,
        OFFSET_BITS,
        base_address >> (64 - OFFSET_BITS),
    );
    place(&mut words, X_END, 6, u64::from(tile.x()));
    place(&mut words, Y_END, 6, u64::from(tile.y()));
    // NoC #0: the only NoC libttsim decodes.
    place(&mut words, NOC, 1, 0);
    place(&mut words, ORDERING, 2, STRICT_AXI);
    Ok(words)
}

/// Set `count` bits of `value` at `first_bit`, in a 96-bit little-endian field.
fn place(words: &mut [u32; 3], first_bit: u32, count: u32, value: u64) {
    debug_assert!(first_bit + count <= 96);
    debug_assert!(count == 64 || value < (1u64 << count));
    let mut bits = value;
    let mut bit = first_bit;
    let mut remaining = count;
    while remaining > 0 {
        let word = (bit / 32) as usize;
        let shift = bit % 32;
        let here = remaining.min(32 - shift);
        let mask = if here == 32 {
            u32::MAX
        } else {
            ((1u32 << here) - 1) << shift
        };
        words[word] = (words[word] & !mask) | (((bits as u32) << shift) & mask);
        bits >>= here;
        bit += here;
        remaining -= here;
    }
}

/// Re-exported so the runtime need not depend on the location of the program
/// arena; this is where a staged program lives in L1.
pub const PROGRAM_AREA: u64 = l1::PROGRAM_CACHE.base;
