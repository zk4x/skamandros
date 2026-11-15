// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The device: `libttsim` through `dlopen`, one TLB window, DRAM and L1 access,
//! and starting a core on an image.
//!
//! `libttsim` is a process-wide singleton with no context handle. Every contract
//! violation terminates the process (`_Exit`), so nothing here hands it an
//! address it does not decode: the window geometry is checked before each call,
//! the TLB configuration is the one the library accepts, and L1 access is
//! bounded by [`tt_isa::tensix::L1_SIZE`].
//!
//! The simulator is loaded from `$TT_METAL_SIMULATOR` — the name tt-metal's own
//! runtime and the ttsim project use — or `libttsim_bh.so` in the current
//! directory. It is initialized once per process; [`Device`] can be dropped and
//! reopened without re-initializing it.

use std::ffi::{OsStr, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;

use libloading::{Library, Symbol};

use crate::kmd::{DEVICE_ID_BLACKHOLE, Kmd, VENDOR_ID_TENSTORRENT};
use crate::status::ttStatus;
use tt_isa::dram::{CHANNEL_BYTES, Dram, DramChannel};
use tt_isa::l1;
use tt_isa::mailbox::{self, status};
use tt_isa::noc::NocCoord;
use tt_isa::noc::niu::Endpoint;
use tt_isa::tensix::{self, Core};

/// The environment variable that names the simulator's shared object:
/// tt-metal's own (`TT_METAL_SIMULATOR`, `TTSIM.md`).
pub const LIB_PATH_ENV: &str = "TT_METAL_SIMULATOR";

/// Where `libttsim` decodes chip 0's BARs, and where the TLB configuration
/// array lives. These are compile-time constants inside the library, not
/// values derived from configuration space.
const BAR0_BASE: u64 = 0x1_0000_0000;
const TLB_CONFIG_BASE: u64 = 0x1FC0_0000;

/// The one 2 MiB window this driver uses. Window 201 belongs to the kernel
/// driver, so it is never taken.
const WINDOW: u16 = 0;
const WINDOW_SIZE: u64 = 2 * 1024 * 1024;

/// The ARC's APB window: `ARC_APB_BAR0_XBAR_OFFSET_START`
/// (`blackhole_implementation.hpp:222`), the last MiB of BAR0, and the same
/// window over the NoC (`ARC_NOC_XBAR_ADDRESS_START`, `:237`).
///
/// Which one a write goes through is tt-metal's rule: the APB is reachable over
/// AXI only when the PCIe tile is at X 11
/// (`BlackholeTTDevice::is_arc_available_over_axi`, `blackhole_tt_device.cpp:296-303`).
const ARC_APB_BAR0: u64 = 0x1FF0_0000;
const ARC_APB_NOC: u64 = 0x8000_0000;

/// `ARC_FW_INT_ADDR` / `ARC_FW_INT_VAL` (`blackhole_implementation.hpp:255-256`):
/// the doorbell that wakes the ARC's message-queue handler.
const ARC_FW_INT_ADDR: u64 = 0x0003_0100;
const ARC_FW_INT_VAL: u32 = 65_536;

/// `NIU_CFG_NOC0_BAR_PCIE_ADDR` (`blackhole_implementation.hpp:266`) plus the
/// 0x100 tt-metal adds (`blackhole_tt_device.cpp:129`). Bit 14 is the
/// coordinate-translation enable; the low six bits are the PCIe tile's X.
const NIU_CFG_0: u64 = 0x1FD0_4100;

/// `BlackholeArcMessageQueue` (`blackhole_arc_message_queue.hpp:51-58`): an
/// eight-word header of double-wrapping pointers, then two rings of eight-word
/// entries, the requests and then the responses.
const ARC_QUEUE_HEADER_BYTES: u64 = 32;
const ARC_QUEUE_ENTRY_BYTES: u64 = 32;
/// The header's pointers, in words from the queue's base: request write,
/// response read, request read, response write.
const ARC_REQUEST_WPTR: u64 = 0;
const ARC_RESPONSE_RPTR: u64 = 1;
const ARC_REQUEST_RPTR: u64 = 4;
const ARC_RESPONSE_WPTR: u64 = 5;
/// The queue tt-metal sends on (`BlackholeArcMessageQueueIndex::APPLICATION`,
/// `blackhole_arc.hpp:49-54`).
const ARC_APPLICATION_QUEUE: u64 = 3;
/// `ARC_MSG_RESPONSE_OK_LIMIT` (`blackhole_implementation.hpp:258`): a response
/// status below it is a success, and its high half is the message's exit code.
const ARC_RESPONSE_OK_LIMIT: u32 = 240;

/// `AICLK_GO_BUSY` and `AICLK_GO_LONG_IDLE` (`arc_message_type`,
/// `blackhole_implementation.hpp:59-67`).
const AICLK_GO_BUSY: u32 = 0x52;
const AICLK_GO_LONG_IDLE: u32 = 0x54;
/// `AICLK_IDLE_VAL` (`blackhole_implementation.hpp:273`): the clock an
/// `AICLK_GO_LONG_IDLE` is confirmed against.
const AICLK_IDLE: u32 = 800;

/// `ARC_MESSAGE_TIMEOUT` and `AICLK_TIMEOUT` (`umd/device/utils/timeouts.hpp:12,25`).
const ARC_MESSAGE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(1_000);
const AICLK_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(200);

/// First DRAM byte this driver hands out.
///
/// tt-metal reserves the bottom of every channel for the DRAM barrier (one
/// alignment word at offset 0), the profiler block (empty in a build without
/// Tracy) and the device-print dispatch region (`DRAM_ALIGNMENT + 1 MiB`), so
/// its `UNRESERVED` base is `0x1_0080` (`bh_hal.cpp:33-83`, `:329-332`). This
/// driver writes none of those regions, but starting above them keeps its data
/// out of anything the card's own dispatch would touch.
const DRAM_FIRST: u64 = 0x10_0080;

/// Simulated cycles one core may take to answer before it is declared stuck.
///
/// On silicon there are no simulated cycles, so the same number is read as
/// nanoseconds and floored at [`SILICON_MIN_WAIT`] (`tt-device`
/// `core_control.rs:489-521`).
pub const BUDGET: u64 = 400_000;

/// The shortest wall-clock wait on silicon, whatever the budget says.
///
/// A budget is written in simulated cycles, and converted at a nominal 1 GHz —
/// so the gates' 400 000 cycles would be 0.4 ms, less than a handful of MMIO
/// round trips through a TLB window. The floor makes a silicon timeout mean "the
/// core is not doing this" rather than "the host asked too quickly".
pub const SILICON_MIN_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

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
    fn open(path: &OsStr) -> Result<&'static Lib, ttStatus> {
        static LIB: OnceLock<Result<&'static Lib, ttStatus>> = OnceLock::new();
        LIB.get_or_init(|| Self::load(path))
            .as_ref()
            .map(|lib| *lib)
            .map_err(|e| *e)
    }

    fn load(path: &OsStr) -> Result<&'static Lib, ttStatus> {
        // SAFETY: the caller named a libttsim shared object; loading a shared
        // object runs its initializers, which is what the caller asked for.
        let library = unsafe { Library::new(path) }.map_err(|e| {
            let _ = (e, PathBuf::from(path));
            ttStatus::TT_ERROR_DLOPEN
        })?;
        macro_rules! sym {
            ($name:literal) => {{
                let s: Symbol<'_, _> = unsafe { library.get(concat!($name, "\0").as_bytes()) }
                    .map_err(|e| {
                        let _ = (e, $name);
                        ttStatus::TT_ERROR_DLSYM
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
    eprintln!(
        "libskamandros: the device attempted a DMA read of {size} bytes at host address {paddr:#x}"
    );
    std::process::abort()
}

/// Which transport a [`Device`] sits on.
///
/// The two differ in four calls — configuration read, window read, window write
/// and a clock step — and in nothing else, so the choice is one field rather
/// than two device types. The driver variant is the `tt-kmd` port recorded in
/// `TODO.md`; until that lands it cannot be constructed, and the four arms say
/// so.
enum Link {
    /// The simulator: `dlopen`ed `libttsim_bh.so`, entry points resolved once.
    Sim(&'static Lib),
    /// `/dev/tenstorrent/N` through tt-kmd, opened by [`Device::open_hardware`].
    Driver(Kmd),
}

impl Link {
    fn mem_write(&self, paddr: u64, bytes: &[u8]) {
        match self {
            Link::Sim(lib) => lib.mem_write(paddr, bytes),
            Link::Driver(kmd) => {
                let offset = bar0_offset(paddr).expect("a BAR0 address this driver wrote");
                kmd.bar0_write(offset, bytes)
                    .expect("a BAR0 address this driver wrote");
            }
        }
    }

    fn mem_read(&self, paddr: u64, out: &mut [u8]) {
        match self {
            Link::Sim(lib) => lib.mem_read(paddr, out),
            Link::Driver(kmd) => {
                let offset = bar0_offset(paddr).expect("a BAR0 address this driver read");
                kmd.bar0_read(offset, out)
                    .expect("a BAR0 address this driver read");
            }
        }
    }

    /// The wait budget, and how the device advances time.
    ///
    /// The simulator advances only inside [`Link::clock`]; silicon advances on its
    /// own, so the same budget is read as nanoseconds and floored at
    /// [`SILICON_MIN_WAIT`]. That is the port of `tt-device`'s rule
    /// (`core_control.rs:489-521`): a budget written in simulated cycles is a
    /// fraction of a millisecond on silicon — less than a handful of MMIO round
    /// trips — and a caller needs a wall-clock deadline instead.
    fn clock(&self, n: u32) {
        match self {
            Link::Sim(lib) => lib.clock(n),
            // Nothing to do: time passes by itself on silicon.
            Link::Driver(_) => {}
        }
    }

    /// The TLB window this session writes: window 0 in the simulator, and the
    /// index the driver arbitrated on the card.
    fn window(&self) -> u16 {
        match self {
            Link::Sim(_) => WINDOW,
            Link::Driver(kmd) => kmd.window() as u16,
        }
    }

    fn is_simulated(&self) -> bool {
        matches!(self, Link::Sim(_))
    }
}

impl Device {
    /// Is this device the simulator?
    ///
    /// The two targets differ in what a program may write: the unpacker's
    /// tile-descriptor tail is written on the card and refused by `ttsim`, so
    /// a launch stages [`crate::module::Module::silicon_cores`] here and
    /// [`crate::module::Module::cores`] on the simulator.
    pub fn is_simulated(&self) -> bool {
        self.link.is_simulated()
    }
}

/// A BAR0-relative offset from the address the window accessors use.
///
/// Everything above the link names a *physical* BAR0 address — `BAR0_BASE` plus
/// the BAR offset — because that is what `libttsim` decodes. The driver maps BAR0
/// itself, so the base is subtracted once, here, with the range checked: an
/// address off the mapping is a bug in this driver, not a device fault.
fn bar0_offset(paddr: u64) -> Result<u64, ttStatus> {
    let offset = paddr.wrapping_sub(BAR0_BASE);
    if paddr < BAR0_BASE || offset >= crate::kmd::BAR0_BYTES {
        let _ = paddr;
        return Err(ttStatus::TT_ERROR_BAR0_NOT_DECODED);
    }
    Ok(offset)
}

unsafe extern "C" fn dma_write_unexpected(paddr: u64, _src: *const c_void, size: u32) {
    eprintln!(
        "libskamandros: the device attempted a DMA write of {size} bytes at host address {paddr:#x}"
    );
    std::process::abort()
}

/// One image: the core it runs on, its bytes, and where it is loaded in L1.
///
/// The bytes are the firmware `build.rs` produced from `firmware/` and judged
/// instruction by instruction, so an image here is known to be executable before
/// anything runs it. That is what makes `libskamandros.so` self-contained: the consumer's
/// build needs neither the RISC-V target nor `llvm-objcopy`, only this crate's
/// build did.
pub type Image = (Core, &'static [u8], u64);

/// Where `link.x` places an image: RISCV T0's default reset PC, which every image
/// but the role images and the per-core probes uses.
#[cfg(test)]
pub const LOAD_ADDRESS: u64 = Core::T0.default_reset_pc() as u64;

/// The three role images -- unpack on T0, math on T1, pack on T2 -- each linked
/// at its own slot (`tt_isa::l1::roles`), so all three fit in one tile's L1 at
/// once and each starts through its reset-PC override.
#[cfg(test)]
pub const ROLES: [Image; 3] = [
    (
        Core::T0,
        include_bytes!(env!("FIRMWARE_ROLE_T0")),
        tt_isa::l1::roles::BASE[0],
    ),
    (
        Core::T1,
        include_bytes!(env!("FIRMWARE_ROLE_T1")),
        tt_isa::l1::roles::BASE[1],
    ),
    (
        Core::T2,
        include_bytes!(env!("FIRMWARE_ROLE_T2")),
        tt_isa::l1::roles::BASE[2],
    ),
];

/// The program executor: reads a staged program's address, length and arguments
/// from the mailbox, invalidates the I-cache and calls it.
pub const EXEC_B: &[u8] = include_bytes!(env!("FIRMWARE_EXEC_B"));

/// The three role executors -- unpack on T0, math on T1, pack on T2 -- each
/// linked at its role slot. They read their own role mailbox and call the
/// program with `a0` at its arguments, the way tt-metal's firmware calls a
/// kernel; nothing is pushed.
pub const EXEC_ROLES: [Image; 3] = [
    (
        Core::T0,
        include_bytes!(env!("FIRMWARE_EXEC_T0")),
        tt_isa::l1::roles::BASE[0],
    ),
    (
        Core::T1,
        include_bytes!(env!("FIRMWARE_EXEC_T1")),
        tt_isa::l1::roles::BASE[1],
    ),
    (
        Core::T2,
        include_bytes!(env!("FIRMWARE_EXEC_T2")),
        tt_isa::l1::roles::BASE[2],
    ),
];

/// RISCV NC's executor: the writer's program, called the way the TRISCs call
/// theirs. NC cannot invalidate its own cache, so `exec_b` invalidates it before
/// each launch; the image runs from `dm_nc`'s slot behind a jump at NC's reset
/// PC.
pub const EXEC_NC: &[u8] = include_bytes!(env!("FIRMWARE_EXEC_NC"));

/// The four subordinate executors, loaded held and released together after
/// RISCV B has invalidated every instruction cache.
pub const EXEC_SUBORDINATES: [Image; 4] = [
    EXEC_ROLES[0],
    EXEC_ROLES[1],
    EXEC_ROLES[2],
    (Core::NC, EXEC_NC, tt_isa::dm::nc::IMAGE_BASE),
];

/// RISCV NC's mover image (`tt_isa::dm::nc`), reached through the stub at NC's
/// reset PC.
#[cfg(test)]
pub const DM_NC: &[u8] = include_bytes!(env!("FIRMWARE_DM_NC"));

/// RISCV B's data mover (`tt_isa::dm`), loaded at L1 offset 0.
#[cfg(test)]
pub const DM_B: &[u8] = include_bytes!(env!("FIRMWARE_DM_B"));

/// Every image this build produced, by name, with its load address.
///
/// The list exists so a test can name every image the build gate judged, rather
/// than trusting that the one it embeds was among them.
#[cfg(test)]
pub fn all_images() -> Vec<(&'static str, &'static [u8])> {
    vec![
        ("exec_b", EXEC_B),
        ("exec_t0", EXEC_ROLES[0].1),
        ("exec_t1", EXEC_ROLES[1].1),
        ("exec_t2", EXEC_ROLES[2].1),
        ("exec_nc", EXEC_NC),
        ("dm_nc", DM_NC),
        ("dm_b", DM_B),
        ("role_t0", ROLES[0].1),
        ("role_t1", ROLES[1].1),
        ("role_t2", ROLES[2].1),
    ]
}

/// One Blackhole, reached through a TLB window.
///
/// The transport is one field: [`Link::Sim`] is the `dlopen`ed simulator, and
/// [`Link::Driver`] will be `/dev/tenstorrent/N`. Everything else here is device
/// state, discovered through the same window either way.
pub struct Device {
    link: Link,
    /// The configuration the window is known to hold. The registers are
    /// write-only, so this is the only record of the current mapping.
    shadow: Option<[u32; 3]>,
    /// The tile the low-level L1 accessors currently target. Everything is
    /// per tile, so a caller selects one before a per-tile step.
    tile: NocCoord,
    /// Every Tensix tile the part has, in row-major order (logical order). A
    /// module's `(x, y)` is an index into this.
    tiles: Vec<NocCoord>,
    /// Tensix columns in [`Device::tiles`]: the width of the logical grid.
    columns: usize,
    /// DRAM channels that can hold data.
    dram: Dram,
    /// First byte of DRAM this device has not handed out.
    dram_used: u64,
    /// Is `exec_b` resident and looping on the tile?
    exec_started: Vec<NocCoord>,
    /// The last generation handed to any core. Device-wide, so a value a core
    /// has run never repeats: the setup program and the launch that follows it
    /// are two different generations even though they are the first two runs.
    launch_seq: u32,
    /// Programs already placed at their L1 address: `(tile, address, words,
    /// hash)`. The program cache is per tile, and a launch only names the
    /// address; an identical program on the same tile is not rewritten.
    placed: Vec<(NocCoord, u32, u64, u64)>,
}

impl Device {
    /// Load the simulator if needed and take the device.
    pub fn open() -> Result<Device, ttStatus> {
        let path = std::env::var_os(LIB_PATH_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("libttsim_bh.so"));
        let lib = Lib::open(path.as_os_str())?;

        // Chip 0 must answer configuration space, and it must be a Blackhole:
        // every address constant above is Blackhole's. Only the simulator has a
        // configuration-space path: tt-kmd exposes no ioctl for it, and the card's
        // identity check is `GET_DEVICE_INFO` at open (`Kmd::open`).
        let id = lib.config_rd32(0, 0);
        if id == u32::MAX {
            return Err(ttStatus::TT_ERROR_NO_DEVICE);
        }
        let vendor = (id & 0xFFFF) as u16;
        let device = (id >> 16) as u16;
        if vendor != VENDOR_ID_TENSTORRENT || device != DEVICE_ID_BLACKHOLE {
            let _ = (vendor, device);
            return Err(ttStatus::TT_ERROR_NOT_BLACKHOLE);
        }

        let link = Link::Sim(lib);

        // The live Tensix grid comes from the ARC's telemetry, never from the
        // part's population: ttsim v1.16 (and a harvested chip) refuses accesses
        // to a fused-off tile's registers, and a guessed full grid writes
        // exactly there.
        let mut device = Device {
            link,
            shadow: None,
            tile: tt_isa::arc::arc_tile(),
            tiles: Vec::new(),
            columns: 0,
            dram: Dram::FULL,
            dram_used: DRAM_FIRST,
            exec_started: Vec::new(),
            launch_seq: 0,
            placed: Vec::new(),
        };
        // Before any tile is named: with translation off, a logical column number
        // is a physical one, and the first n of them are the wrong set.
        if !device.translation_enabled()? {
            eprintln!(
                "libskamandros: the NIU is not translating coordinates; this driver addresses \
                 tensix tiles logically and would name a fused-off column"
            );
            return Err(ttStatus::TT_ERROR_DEVICE);
        }
        let tensix = device.probe_tensix()?;
        device.tiles = tensix.tiles().collect();
        device.columns = tensix.enabled_column_count();
        device.tile = *device
            .tiles
            .first()
            .ok_or(ttStatus::TT_ERROR_NO_TENSIX_TILE)?;
        // A previous process (`ttRelease` only drops this struct) leaves its
        // cores running and its mailboxes written. Reopening is the 0 -> 1
        // transition, so it cannot disturb a live caller: hold the five cores
        // and clear the launch state here, making the next launch start from a
        // known tile.
        device.quiesce()?;
        Ok(device)
    }

    /// Take the card through `/dev/tenstorrent/0`.
    ///
    /// The same `Device` as [`Device::open`], over the driver link: BAR0 mapped
    /// through tt-kmd, one window the driver arbitrated, and the card's telemetry
    /// consulted before any Tensix tile is named.
    ///
    /// Device 0 only (user decision 2026-11-15): on a multi-card host the node is
    /// not chosen, so a second card is simply not reached. Opening it also requires
    /// `TT_HARDWARE=1` for now — a bring-up gate that is dropped once a kernel has
    /// run on the card — so nothing reaches silicon by accident.
    pub fn open_hardware() -> Result<Device, ttStatus> {
        if std::env::var_os("TT_HARDWARE").as_deref() != Some(OsStr::new("1")) {
            eprintln!(
                "libskamandros: refusing to open the card: TT_HARDWARE=1 is required for now \
                 (and TT_METAL_SIMULATOR is unset)"
            );
            return Err(ttStatus::TT_ERROR_DEVICE);
        }
        let kmd = Kmd::open(0)?;
        let link = Link::Driver(kmd);

        // The live Tensix grid comes from the ARC's telemetry, never from the
        // part's population: a harvested chip refuses accesses to a fused-off
        // tile's registers, and a guessed full grid writes exactly there.
        let mut device = Device {
            link,
            shadow: None,
            tile: tt_isa::arc::arc_tile(),
            tiles: Vec::new(),
            columns: 0,
            dram: Dram::FULL,
            dram_used: DRAM_FIRST,
            exec_started: Vec::new(),
            launch_seq: 0,
            placed: Vec::new(),
        };
        if !device.translation_enabled()? {
            eprintln!(
                "libskamandros: the NIU is not translating coordinates; this driver addresses \
                 tensix tiles logically and would name a fused-off column"
            );
            return Err(ttStatus::TT_ERROR_DEVICE);
        }
        let tensix = device.probe_tensix()?;
        device.tiles = tensix.tiles().collect();
        device.columns = tensix.enabled_column_count();
        device.tile = *device
            .tiles
            .first()
            .ok_or(ttStatus::TT_ERROR_NO_TENSIX_TILE)?;
        device.dram = device.probe_dram()?;

        // Nothing on the card is ours until now, and a previous process leaves
        // its cores running: hold the five on every tile, clear the launch state,
        // and register the one write that survives this process dying. The driver
        // performs that write when the descriptor closes, whatever closed it
        // (`tt_cdev_release_noc_cleanup`, `chardev.c:1021-1030`).
        device.quiesce()?;
        let held = Core::B.soft_reset_mask()
            | Core::T0.soft_reset_mask()
            | Core::T1.soft_reset_mask()
            | Core::T2.soft_reset_mask()
            | Core::NC.soft_reset_mask();
        let first = device.tile;
        if let Link::Driver(kmd) = &device.link {
            kmd.hold_on_close(first.x(), first.y(), tensix::SOFT_RESET_0, held)?;
        }
        // tt-metal puts the chip at its busy operating point once the device is
        // up (`Cluster::deassert_resets_and_set_power_state`, `cluster.cpp:1080-1099`).
        // At the idle point the Matrix Unit's `Src` reads are unreliable on
        // silicon, so this belongs to bringing the device up.
        device.set_power_state(true)?;
        Ok(device)
    }

    /// Does the NIU translate coordinates?
    ///
    /// Every Tensix tile this driver names is a *logical* coordinate: with
    /// translation on, the NIU maps it onto a surviving physical column, which is
    /// why the first `n` columns of the grid are the right ones on a harvested chip
    /// ([`tt_isa::noc::grid::Tensix`]). With translation off those numbers are
    /// physical and the first `n` are the wrong set — a write to a fused-off tile,
    /// which is a hung NoC.
    ///
    /// tt-metal reads the same bit before building any coordinate
    /// (`BlackholeTTDevice::get_noc_translation_enabled`, `blackhole_tt_device.cpp:123-135`):
    /// BAR0 at the PCIe NIU's configuration register, bit 14. ttsim decodes the same
    /// BAR0 window (`PCIE_NIU0_BASE 0x1FD04000`) and reports the bit set, so this is
    /// checked on both targets rather than assumed.
    fn translation_enabled(&mut self) -> Result<bool, ttStatus> {
        let mut word = [0u8; 4];
        self.link.mem_read(BAR0_BASE + NIU_CFG_0, &mut word);
        let cfg = u32::from_le_bytes(word);
        // A read the NoC did not answer comes back all ones, whose bit 14 is
        // set: reading that as "translation is on" would be a sentinel mistaken
        // for an answer, and every tile named afterwards would be a guess.
        if cfg == u32::MAX {
            eprintln!(
                "libskamandros: the PCIe NIU's configuration register at 0x{NIU_CFG_0:08X} \
                 did not answer; refusing to name any tile"
            );
            return Err(ttStatus::TT_ERROR_DEVICE);
        }
        Ok((cfg >> 14) & 1 != 0)
    }

    /// Which GDDR channels can hold data, from the ARC's telemetry.
    ///
    /// tt-metal reads `ENABLED_GDDR` and inverts it to obtain the *harvested*
    /// mask (`BlackholeTTDevice::get_chip_info`,
    /// `blackhole_tt_device.cpp:146-149`, `~read_entry & 0xFF`); the driver's own
    /// sysfs cannot answer it, because `tt_telemetry_probe` caches only the tags
    /// that back an attribute. The tag itself is the enabled mask, so this walk
    /// must not invert it again.
    fn probe_dram(&mut self) -> Result<Dram, ttStatus> {
        use tt_isa::arc;
        let arc_tile = arc::arc_tile();
        let table = u64::from(self.arc_word(arc_tile, arc::TELEMETRY_PTR)?);
        let values = u64::from(self.arc_word(arc_tile, arc::TELEMETRY_DATA)?);
        if !arc::is_within_csm(table, 8) || !arc::is_within_csm(values, 4) {
            let _ = (table, values);
            return Err(ttStatus::TT_ERROR_TELEMETRY);
        }
        let version = self.arc_word(arc_tile, table)?;
        let entries = self.arc_word(arc_tile, table + 4)?;
        let tag_table = 8 + 4 * u64::from(entries);
        // The header word is `major << 16 | minor << 8 | patch`, so the major
        // version is its top byte, exactly as the driver reads it
        // (`blackhole.c:485-497`). Comparing the whole word would refuse every
        // firmware whose version is not literally 1.
        if ((version >> 16) & 0xFF) > arc::TELEMETRY_MAX_MAJOR_VERSION
            || entries > arc::TELEMETRY_MAX_ENTRIES
            || !arc::is_within_csm(table, tag_table)
        {
            let _ = (version, entries);
            return Err(ttStatus::TT_ERROR_TELEMETRY);
        }
        let mut enabled = None;
        for i in 0..u64::from(entries) {
            let entry = self.arc_word(arc_tile, table + 8 + 4 * i)?;
            if entry & 0xFFFF != u32::from(arc::tag::ENABLED_GDDR) {
                continue;
            }
            let at = values + 4 * u64::from(entry >> 16);
            if !arc::is_within_csm(at, 4) {
                let _ = at;
                return Err(ttStatus::TT_ERROR_TELEMETRY);
            }
            enabled = Some(self.arc_word(arc_tile, at)?);
        }
        // An absent tag names nothing, and guessing a full part from firmware
        // that does not publish it is the mistake this walk exists to avoid. The
        // word is the *enabled* mask, which is the shape `Dram` is built from;
        // tt-metal's inversion to a harvested mask
        // (`blackhole_tt_device.cpp:146-149`) is its own bookkeeping, not this
        // one's.
        let mask = enabled.ok_or(ttStatus::TT_ERROR_TELEMETRY)?;
        Dram::from_enabled_mask(mask as u8).ok_or(ttStatus::TT_ERROR_TELEMETRY)
    }

    /// This chip's live Tensix columns, from the ARC's telemetry table.
    ///
    /// tt-metal's `ArcTelemetryReader` (`umd/device/arc/arc_telemetry_reader.cpp`)
    /// takes the table and value pointers from the ARC's reset-unit scratch, then
    /// the entry count and the tag table from the table, whose words are
    /// `tag | offset << 16` and whose values live in the value block at that
    /// offset. The tag this driver needs is `ENABLED_TENSIX_COL`, whose bits
    /// index the harvesting order rather than physical columns. It is read the
    /// way tt-metal reads it ([`tt_isa::noc::grid::Tensix::from_enabled_column_mask`]),
    /// and the survivors are a prefix of the translated columns whatever the bit
    /// order (`NoC/Coordinates.md:54`). A chip that will not name its columns,
    /// or names none at all, is refused: assuming a full part writes to a
    /// fused-off tile, which ttsim v1.16 treats as undefined.
    fn probe_tensix(&mut self) -> Result<tt_isa::noc::grid::Tensix, ttStatus> {
        use tt_isa::arc;
        let arc_tile = arc::arc_tile();
        let table = u64::from(self.arc_word(arc_tile, arc::TELEMETRY_PTR)?);
        let values = u64::from(self.arc_word(arc_tile, arc::TELEMETRY_DATA)?);
        // The pointers come from firmware: a CSM read is a NoC read, and an
        // address outside the CSM is an unmapped ARC read, which hangs.
        if !arc::is_within_csm(table, 8) || !arc::is_within_csm(values, 4) {
            let _ = (table, values);
            return Err(ttStatus::TT_ERROR_TELEMETRY);
        }
        let version = self.arc_word(arc_tile, table)?;
        let entries = self.arc_word(arc_tile, table + 4)?;
        // The header is two words and the tag table follows it.
        let tag_table = 8 + 4 * u64::from(entries);
        // The header word is `major << 16 | minor << 8 | patch`, so the major
        // version is its top byte, exactly as the driver reads it
        // (`blackhole.c:485-497`). Comparing the whole word would refuse every
        // firmware whose version is not literally 1.
        if ((version >> 16) & 0xFF) > arc::TELEMETRY_MAX_MAJOR_VERSION
            || entries > arc::TELEMETRY_MAX_ENTRIES
            || !arc::is_within_csm(table, tag_table)
        {
            let _ = (version, entries);
            return Err(ttStatus::TT_ERROR_TELEMETRY);
        }
        let mut enabled = None;
        for i in 0..u64::from(entries) {
            let entry = self.arc_word(arc_tile, table + 8 + 4 * i)?;
            if entry & 0xFFFF != u32::from(arc::tag::ENABLED_TENSIX_COL) {
                continue;
            }
            let at = values + 4 * u64::from(entry >> 16);
            if !arc::is_within_csm(at, 4) {
                let _ = at;
                return Err(ttStatus::TT_ERROR_TELEMETRY);
            }
            enabled = Some(self.arc_word(arc_tile, at)?);
        }
        // The tag is the *enabled* word; `from_enabled_column_mask` reads it the
        // way tt-metal does (complement over the fourteen harvestable bits, then
        // the count), which is also where bits above the fourteen columns are
        // masked off.
        let mask = enabled.ok_or(ttStatus::TT_ERROR_TELEMETRY)?;
        tt_isa::noc::grid::Tensix::from_enabled_column_mask(mask)
            .ok_or(ttStatus::TT_ERROR_TELEMETRY)
    }

    /// One word of the ARC's address space, through the window: its reset-unit
    /// scratch and its CSM are ordinary tile addresses.
    fn arc_word(&mut self, arc_tile: NocCoord, address: u64) -> Result<u32, ttStatus> {
        let mut bytes = [0u8; 4];
        self.window_read(arc_tile, address, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    /// One ARC telemetry tag's value, or `None` if the table does not carry it.
    ///
    /// The walk is `probe_dram`'s: the pointers come from the ARC's reset-unit
    /// scratch, the table's two header words give the version and the entry
    /// count, and each tag entry is `tag | offset << 16` into the value block.
    /// Every address firmware supplied is bounded by the CSM first, because a
    /// CSM read is a NoC read and an unmapped ARC address hangs the host.
    fn arc_tag(&mut self, tag: u16) -> Result<Option<u32>, ttStatus> {
        use tt_isa::arc;
        let arc_tile = arc::arc_tile();
        let table = u64::from(self.arc_word(arc_tile, arc::TELEMETRY_PTR)?);
        let values = u64::from(self.arc_word(arc_tile, arc::TELEMETRY_DATA)?);
        if !arc::is_within_csm(table, 8) || !arc::is_within_csm(values, 4) {
            let _ = (table, values);
            return Err(ttStatus::TT_ERROR_TELEMETRY);
        }
        let version = self.arc_word(arc_tile, table)?;
        let entries = self.arc_word(arc_tile, table + 4)?;
        if ((version >> 16) & 0xFF) > arc::TELEMETRY_MAX_MAJOR_VERSION
            || entries > arc::TELEMETRY_MAX_ENTRIES
            || !arc::is_within_csm(table, 8 + 4 * u64::from(entries))
        {
            let _ = (version, entries);
            return Err(ttStatus::TT_ERROR_TELEMETRY);
        }
        let mut found = None;
        for i in 0..u64::from(entries) {
            let entry = self.arc_word(arc_tile, table + 8 + 4 * i)?;
            if entry & 0xFFFF != u32::from(tag) {
                continue;
            }
            let at = values + 4 * u64::from(entry >> 16);
            if !arc::is_within_csm(at, 4) {
                let _ = at;
                return Err(ttStatus::TT_ERROR_TELEMETRY);
            }
            found = Some(self.arc_word(arc_tile, at)?);
        }
        Ok(found)
    }

    /// One message on the ARC's application queue, exactly as tt-metal sends
    /// one (`BlackholeArcMessageQueue::send_message`,
    /// `blackhole_arc_message_queue.cpp:101-142`), and its exit code.
    ///
    /// The queue control block is the address SCRATCH_RAM[11] publishes; its
    /// first word is the first queue's ARC address and bits 32-39 are the entries
    /// in each queue, with queue [`ARC_APPLICATION_QUEUE`] at
    /// `base + index * (32 + 2 * entries * 32)` bytes
    /// (`get_blackhole_arc_message_queue`, `blackhole_arc_message_queue.cpp:144-167`).
    /// A queue is an eight-word header of double-wrapping read/write pointers
    /// followed by a request ring and a response ring of eight-word entries. The
    /// request is written at the write pointer, the pointer is bumped, and the
    /// ARC is rung with the APB doorbell; the response is taken from the response
    /// ring once its write pointer moves, and its read pointer is bumped.
    fn arc_msg(&mut self, code: u32) -> Result<u32, ttStatus> {
        use tt_isa::arc;
        let arc_tile = arc::arc_tile();
        let qcb = u64::from(self.arc_word(arc_tile, arc::reset_scratch(11))?);
        if !arc::is_within_csm(qcb, 8) {
            eprintln!(
                "libskamandros: the ARC's message-queue control block is at {qcb:#x}, outside \
                 its CSM"
            );
            return Err(ttStatus::TT_ERROR_TELEMETRY);
        }
        let mut block = [0u8; 8];
        self.window_read(arc_tile, qcb, &mut block)?;
        let base = u64::from(u32::from_le_bytes(
            block[0..4].try_into().expect("four bytes"),
        ));
        let entries =
            u64::from(u32::from_le_bytes(block[4..8].try_into().expect("four bytes")) & 0xFF);
        if base == 0 || entries == 0 {
            eprintln!(
                "libskamandros: the ARC published no message queue (base {base:#x}, {entries} \
                 entries)"
            );
            return Err(ttStatus::TT_ERROR_TELEMETRY);
        }
        let queue_bytes = ARC_QUEUE_HEADER_BYTES + 2 * entries * ARC_QUEUE_ENTRY_BYTES;
        let queue = base + ARC_APPLICATION_QUEUE * queue_bytes;
        if !arc::is_within_csm(queue, queue_bytes) {
            eprintln!(
                "libskamandros: the ARC's application queue at {queue:#x} is outside its CSM"
            );
            return Err(ttStatus::TT_ERROR_TELEMETRY);
        }
        let ring = 2 * entries;
        let started = std::time::Instant::now();

        // Room in the request ring: it is full when the read pointer is one ring
        // behind the write pointer (`push_request`,
        // `blackhole_arc_message_queue.cpp:53-70`).
        let write_at = queue + 4 * ARC_REQUEST_WPTR;
        let mut wptr = u64::from(self.arc_word(arc_tile, write_at)?);
        loop {
            let rptr = u64::from(self.arc_word(arc_tile, queue + 4 * ARC_REQUEST_RPTR)?);
            if (rptr as i64 - wptr as i64).unsigned_abs() != entries {
                break;
            }
            if started.elapsed() >= ARC_MESSAGE_TIMEOUT {
                eprintln!("libskamandros: the ARC's request queue is full");
                return Err(ttStatus::TT_ERROR_DEVICE);
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        let mut request = [0u8; ARC_QUEUE_ENTRY_BYTES as usize];
        request[0..4].copy_from_slice(&code.to_le_bytes());
        let entry = queue + ARC_QUEUE_HEADER_BYTES + (wptr % entries) * ARC_QUEUE_ENTRY_BYTES;
        self.window_write(arc_tile, entry, &request)?;
        wptr = (wptr + 1) % ring;
        self.window_write(arc_tile, write_at, &(wptr as u32).to_le_bytes())?;

        // Ring the doorbell: tt-metal writes `ARC_FW_INT_VAL` to `ARC_FW_INT_ADDR`
        // through the APB, which is BAR0 when the PCIe tile is at X 11 and the
        // ARC's NoC aperture otherwise (`write_to_arc_apb`,
        // `blackhole_tt_device.cpp:237-257`).
        let mut niu = [0u8; 4];
        self.link.mem_read(BAR0_BASE + NIU_CFG_0, &mut niu);
        if u32::from_le_bytes(niu) & 0x3F == 11 {
            self.bar_write32(ARC_APB_BAR0 + ARC_FW_INT_ADDR, ARC_FW_INT_VAL)?;
        } else {
            self.window_write(
                arc_tile,
                ARC_APB_NOC + ARC_FW_INT_ADDR,
                &ARC_FW_INT_VAL.to_le_bytes(),
            )?;
        }

        // The response: its ring's write pointer moves off the read pointer
        // (`pop_response`, `blackhole_arc_message_queue.cpp:72-99`).
        let read_at = queue + 4 * ARC_RESPONSE_RPTR;
        let mut rptr = u64::from(self.arc_word(arc_tile, read_at)?);
        loop {
            let wptr = u64::from(self.arc_word(arc_tile, queue + 4 * ARC_RESPONSE_WPTR)?);
            if wptr != rptr {
                break;
            }
            if started.elapsed() >= ARC_MESSAGE_TIMEOUT {
                eprintln!("libskamandros: no ARC response to message {code:#04x}");
                return Err(ttStatus::TT_ERROR_DEVICE);
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        let mut response = [0u8; ARC_QUEUE_ENTRY_BYTES as usize];
        let at =
            queue + ARC_QUEUE_HEADER_BYTES + (entries + rptr % entries) * ARC_QUEUE_ENTRY_BYTES;
        self.window_read(arc_tile, at, &mut response)?;
        rptr = (rptr + 1) % ring;
        self.window_write(arc_tile, read_at, &(rptr as u32).to_le_bytes())?;

        let word = u32::from_le_bytes(response[0..4].try_into().expect("four bytes"));
        let status = word & 0xFF;
        if status >= ARC_RESPONSE_OK_LIMIT {
            eprintln!("libskamandros: the ARC refused message {code:#04x}: status {status:#04x}");
            return Err(ttStatus::TT_ERROR_DEVICE);
        }
        Ok(word >> 16)
    }

    /// Put the chip at its busy or idle operating point, and wait for telemetry
    /// to show it: tt-metal's `Chip::set_power_state` (`chip.cpp:216-238`) and
    /// `wait_for_aiclk_value` (`chip.cpp:240-296`).
    ///
    /// The message is `AICLK_GO_BUSY` or `AICLK_GO_LONG_IDLE`; the target is
    /// telemetry's `AICLK_LIMIT_MAX` (`get_max_clock_freq`,
    /// `firmware_info_provider.cpp:197`) or `AICLK_IDLE`. A message the ARC
    /// refuses is an error. A clock that does not settle within `AICLK_TIMEOUT`
    /// is reported and not fatal, as tt-metal's warning is -- but the consequence
    /// is printed, because at the idle point the Matrix Unit's `Src` reads are
    /// unreliable on silicon.
    ///
    /// Nothing here resets the board: no `RESET_DEVICE` ioctl, no ARC reset
    /// message, no backend pulse. It only moves the clock.
    pub fn set_power_state(&mut self, busy: bool) -> Result<(), ttStatus> {
        if self.link.is_simulated() {
            return Ok(());
        }
        // tt-metal skips the idle message when the chip is already there
        // (`LocalChip::close_device`, `local_chip.cpp:195-211`).
        if !busy && self.arc_tag(tt_isa::arc::tag::AICLK)? == Some(AICLK_IDLE) {
            return Ok(());
        }
        let target = if busy {
            self.arc_tag(tt_isa::arc::tag::AICLK_LIMIT_MAX)?
        } else {
            Some(AICLK_IDLE)
        };
        let code = if busy {
            AICLK_GO_BUSY
        } else {
            AICLK_GO_LONG_IDLE
        };
        let exit = self.arc_msg(code)?;
        if exit != 0 {
            eprintln!(
                "libskamandros: the ARC returned exit code {exit} for AICLK message {code:#04x}"
            );
            return Err(ttStatus::TT_ERROR_DEVICE);
        }
        let Some(target) = target else {
            eprintln!(
                "libskamandros: telemetry carries no AICLK limit; the operating point cannot be \
                 confirmed"
            );
            return Ok(());
        };
        let started = std::time::Instant::now();
        loop {
            let aiclk = self.arc_tag(tt_isa::arc::tag::AICLK)?;
            if aiclk == Some(target) {
                return Ok(());
            }
            if started.elapsed() >= AICLK_TIMEOUT {
                eprintln!(
                    "libskamandros: AICLK reads {aiclk:?} after {AICLK_TIMEOUT:?}, not {target}; \
                     at the idle operating point the Matrix Unit's Src reads are unreliable on \
                     silicon"
                );
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// Target `tile` for the low-level accessors (`l1_write`, `read32`, the
    /// launch messages, the program cache). One launch selects each of its
    /// tiles in turn.
    pub fn select(&mut self, tile: NocCoord) {
        self.tile = tile;
    }

    /// Tensix rows in the logical grid (`grid.y` runs `0..rows`).
    pub fn rows(&self) -> usize {
        self.tiles.len() / self.columns
    }

    /// The physical tile at logical `(x, y)`. `tt_metal`'s logical grid is the
    /// worker grid: `absolute_logical_x/y` are these coordinates, and the
    /// physical tile follows the column list.
    pub fn tile_at(&self, x: u32, y: u32) -> Result<NocCoord, ttStatus> {
        if x as usize >= self.columns || y as usize >= self.rows() {
            let _ = (x, y, self.columns, self.rows());
            return Err(ttStatus::TT_ERROR_GRID);
        }
        Ok(self.tiles[y as usize * self.columns + x as usize])
    }

    /// A GDDR channel that can hold data: the chip's first, in the logical
    /// numbering of [`Dram`].
    pub fn dram_channel(&self) -> DramChannel {
        self.dram
            .channel(0)
            .expect("a chip the probe accepted has a first channel")
    }

    /// The chip's `ENABLED_GDDR` word, for the data mover's `USABLE` mailbox.
    pub fn dram_mask(&self) -> u32 {
        u32::from(self.dram.enabled_mask())
    }

    /// Reserve `bytes` of DRAM, 64-byte aligned, and return its channel-0
    /// offset. This is what a caller passes back to `ttMemcpy` and `ttLaunch`.
    pub fn alloc_dram(&mut self, bytes: usize) -> Result<u64, ttStatus> {
        if bytes == 0 {
            return Err(ttStatus::TT_ERROR_ALLOC_ZERO);
        }
        let at = self.dram_used.next_multiple_of(64);
        let end = at
            .checked_add(bytes as u64)
            .ok_or(ttStatus::TT_ERROR_DRAM_ADDRESS_OVERFLOW)?;
        if end > CHANNEL_BYTES {
            let _ = (bytes, at);
            return Err(ttStatus::TT_ERROR_DRAM_FULL);
        }
        self.dram_used = end;
        Ok(at)
    }

    /// The endpoint high word a program names for DRAM on the first channel.
    pub fn dram_hi_for(&self, niu: tt_isa::noc::niu::Niu) -> u32 {
        let channel = self.dram_channel();
        let ep = channel.preferred(niu);
        Endpoint {
            x: ep.x(),
            y: ep.y(),
            addr: 0,
        }
        .hi()
    }

    /// Write `bytes` to DRAM at a channel-0 offset.
    pub fn dram_write(&mut self, offset: u64, bytes: &[u8]) -> Result<(), ttStatus> {
        let channel = self.dram_channel();
        let range = channel.range(offset, bytes.len() as u64).ok_or_else(|| {
            let _ = (bytes.len(), offset);
            ttStatus::TT_ERROR_DRAM_RANGE
        })?;
        let ep = channel.preferred(tt_isa::noc::niu::Niu::Noc0);
        self.window_write(ep, range.offset(), bytes)?;
        // A write is visible through the endpoint it went out of: read the last
        // word back through it.
        if bytes.len() >= 4 {
            let last = range.offset() + bytes.len() as u64 - 4;
            let mut word = [0u8; 4];
            self.window_read(ep, last, &mut word)?;
        }
        Ok(())
    }

    /// Read DRAM at a channel-0 offset.
    pub fn dram_read(&mut self, offset: u64, out: &mut [u8]) -> Result<(), ttStatus> {
        let channel = self.dram_channel();
        channel.range(offset, out.len() as u64).ok_or_else(|| {
            let _ = out.len();
            ttStatus::TT_ERROR_DRAM_RANGE
        })?;
        let ep = channel.preferred(tt_isa::noc::niu::Niu::Noc0);
        self.window_read(ep, offset, out)
    }

    /// Write into this tile's L1.
    pub fn l1_write(&mut self, address: u64, bytes: &[u8]) -> Result<(), ttStatus> {
        self.check_l1(address, bytes.len())?;
        let tile = self.tile;
        self.window_write(tile, address, bytes)
    }

    /// Read one dword from the tile (an L1 word or a core register).
    pub fn read32(&mut self, address: u64) -> Result<u32, ttStatus> {
        let mut b = [0u8; 4];
        let tile = self.tile;
        self.window_read(tile, address, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// Write one dword to the tile (an L1 word or a core register).
    pub fn write32(&mut self, address: u64, value: u32) -> Result<(), ttStatus> {
        let tile = self.tile;
        self.window_write(tile, address, &value.to_le_bytes())
    }

    /// Advance simulated time. Nothing happens on the device outside this.
    pub fn clock(&self, n: u32) {
        self.link.clock(n);
    }

    /// Load `image` into L1 and start `core` on it.
    pub fn load_and_start(&mut self, core: Core, image: &[u8], at: u64) -> Result<(), ttStatus> {
        self.load_held(core, image, at)?;
        self.set_core_reset(core, false)
    }

    /// Load `image` at `at` and leave `core` held in reset.
    ///
    /// Everything [`Device::load_and_start`] does except the release: the image,
    /// the reset PC and a cleared status word are in place, so one later write to
    /// `SOFT_RESET_0` starts this core and every other held one at once.
    fn load_held(&mut self, core: Core, image: &[u8], at: u64) -> Result<(), ttStatus> {
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
                return Err(ttStatus::TT_ERROR_RESET_PC);
            }
        }
        Ok(())
    }

    /// Release the Tensix backend — unpackers, packers, Matrix Unit, Vector Unit
    /// and the rest — from soft reset, leaving the five RISC-V cores alone.
    pub fn release_tensix_backend(&mut self) -> Result<(), ttStatus> {
        let current = self.read32(tensix::SOFT_RESET_0)?;
        self.write32(tensix::SOFT_RESET_0, current & !tensix::BACKEND_RESET_MASK)
    }

    /// Wait until `predicate` accepts the word at `address`.
    ///
    /// On the simulator nothing happens outside [`Device::clock`], so the budget
    /// is simulated cycles. On silicon time passes by itself and the same number
    /// is a wall-clock deadline, floored at [`SILICON_MIN_WAIT`] (`tt-device`
    /// `core_control.rs:489-521`).
    pub fn wait_for(
        &mut self,
        address: u64,
        mut predicate: impl FnMut(u32) -> bool,
    ) -> Result<u32, ttStatus> {
        let mut waited = 0u64;
        let started = std::time::Instant::now();
        let limit = std::time::Duration::from_nanos(BUDGET).max(SILICON_MIN_WAIT);
        loop {
            let value = self.read32(address)?;
            if predicate(value) {
                return Ok(value);
            }
            if value == status::PANICKED {
                let code = self.read32(mailbox::PANIC_CODE)?;
                let _ = code;
                return Err(ttStatus::TT_ERROR_FIRMWARE_PANIC);
            }
            let expired = if self.link.is_simulated() {
                waited >= BUDGET
            } else {
                started.elapsed() >= limit
            };
            if expired {
                let _ = value;
                return Err(ttStatus::TT_ERROR_FIRMWARE_TIMEOUT);
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
    pub fn quiesce(&mut self) -> Result<(), ttStatus> {
        let mut first: Result<(), ttStatus> = Ok(());
        for i in 0..self.tiles.len() {
            let tile = self.tiles[i];
            self.select(tile);
            for core in [Core::B, Core::T0, Core::T1, Core::T2, Core::NC] {
                if let Err(e) = self.set_core_reset(core, true) {
                    first = first.and(Err(e));
                }
            }
            for mb in Self::launch_mailboxes() {
                // Generation first: a core released before the rest lands must
                // not run the named program with a stale start.
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
        }
        // The executor on every tile is gone with the reset; the next launch
        // stages it again instead of waiting on a mailbox nobody will write.
        self.exec_started.clear();
        self.placed.clear();
        first
    }

    /// The five launch mailboxes: RISCV B, NC and the three Tensix threads.
    fn launch_mailboxes() -> [mailbox::launch::Mailbox; 5] {
        [
            mailbox::launch::Mailbox::b(),
            mailbox::launch::Mailbox::nc(),
            mailbox::launch::Mailbox::of(0),
            mailbox::launch::Mailbox::of(1),
            mailbox::launch::Mailbox::of(2),
        ]
    }

    /// Boot the resident executors, once: all five, the way tt-metal starts a
    /// tile's firmware.
    ///
    /// The four subordinates are loaded while held; RISCV B is started and its
    /// executor invalidates every instruction cache before it publishes
    /// `RUNNING`, so the images are fetched through clean caches when the host
    /// then releases the other four (tt-metal: BRISC's mask store, then
    /// `deassert_all_reset`).
    pub fn start_executors(&mut self) -> Result<(), ttStatus> {
        if self
            .exec_started
            .iter()
            .any(|t| t.x() == self.tile.x() && t.y() == self.tile.y())
        {
            return Ok(());
        }
        // NC cannot take a reset-PC override (ttsim refuses it), so its image
        // starts through a two-word jump at its reset PC, as every NC image
        // has since the mover.
        for &(core, image, at) in &EXEC_SUBORDINATES {
            if core == Core::NC {
                self.load_held_nc(image, at)?;
            } else {
                self.load_held(core, image, at)?;
            }
        }
        self.load_and_start(Core::B, EXEC_B, u64::from(Core::B.default_reset_pc()))?;
        let mb = mailbox::launch::Mailbox::b();
        self.wait_for(mb.status(), |s| s == status::RUNNING)?;
        // A previous process leaves programs named and generations set; a new
        // executor must not replay them.
        for mb in Self::launch_mailboxes() {
            self.clear_launch(mb)?;
        }
        for &(core, _, _) in &EXEC_SUBORDINATES {
            self.set_core_reset(core, false)?;
        }
        self.exec_started.push(self.tile);
        Ok(())
    }

    /// Load NC's image at `at` and leave NC held, with the jump that starts it
    /// at its reset PC.
    fn load_held_nc(&mut self, image: &[u8], at: u64) -> Result<(), ttStatus> {
        self.set_core_reset(Core::NC, true)?;
        self.l1_write(at, image)?;
        let stub = tt_isa::dm::nc::stub(at);
        let words: Vec<u8> = stub.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.l1_write(tt_isa::dm::nc::STUB_AT, &words)?;
        Ok(())
    }

    /// Clear one launch message: the generation first, then the program and
    /// control words.
    fn clear_launch(&mut self, mb: mailbox::launch::Mailbox) -> Result<(), ttStatus> {
        self.write32(mb.generation(), 0)?;
        for at in [
            mb.program_addr(),
            mb.program_len(),
            mb.args(),
            mb.enables(),
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
    pub fn place(&mut self, at: u32, words: &[u32]) -> Result<(), ttStatus> {
        if words.is_empty() {
            return Err(ttStatus::TT_ERROR_EMPTY_PROGRAM);
        }
        if words.len() as u64 > u64::from(mailbox::MAX_WORDS) {
            let _ = words.len();
            return Err(ttStatus::TT_ERROR_PROGRAM_TOO_LONG);
        }
        // FNV-1a: enough to skip rewriting an unchanged program, not a
        // correctness check on the words.
        let hash = words.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, word| {
            (h ^ u64::from(*word)).wrapping_mul(0x0000_0100_0000_01b3)
        });
        let len = words.len() as u64;
        if self.placed.iter().any(|&(t, a, l, h)| {
            t.x() == self.tile.x() && t.y() == self.tile.y() && a == at && l == len && h == hash
        }) {
            return Ok(());
        }
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.l1_write(u64::from(at), &code)?;
        self.placed
            .retain(|&(t, a, _, _)| t.x() != self.tile.x() || t.y() != self.tile.y() || a != at);
        self.placed.push((self.tile, at, len, hash));
        Ok(())
    }

    /// Write a launch message without starting it: the text address, its length,
    /// the arguments base, and a cleared generation and `ACK`. The dispatcher
    /// names a subordinate's program this way, then writes the generation that
    /// starts it.
    pub fn prepare(
        &mut self,
        mb: mailbox::launch::Mailbox,
        text: u32,
        words: u32,
        args: u32,
    ) -> Result<(), ttStatus> {
        self.write32(mb.program_addr(), text)?;
        self.write32(mb.program_len(), words)?;
        self.write32(mb.args(), args)?;
        self.write32(mb.ack(), 0)?;
        self.write32(mb.generation(), 0)?;
        Ok(())
    }

    /// Start whatever launch message `mb` holds with a fresh generation, and
    /// return it. Device-wide and monotonic, so a core never sees a value equal
    /// to the one it last ran.
    pub fn start(&mut self, mb: mailbox::launch::Mailbox) -> Result<u32, ttStatus> {
        self.launch_seq = self.launch_seq.wrapping_add(1).max(1);
        let generation = self.launch_seq;
        self.write32(mb.ack(), 0)?;
        self.write32(mb.generation(), generation)?;
        Ok(generation)
    }

    /// Write one launch message and start it: the text address, its length, the
    /// arguments base, which processors it enables, then a fresh generation.
    /// Only RISCV B's enables word is read, because B dispatches to the rest.
    pub fn launch(
        &mut self,
        mb: mailbox::launch::Mailbox,
        text: u32,
        words: u32,
        args: u32,
        enables: u32,
    ) -> Result<u32, ttStatus> {
        self.prepare(mb, text, words, args)?;
        self.write32(mb.enables(), enables)?;
        self.start(mb)
    }

    /// Wait for a launch's `ACK`, or report its panic or a timeout.
    ///
    /// The budget is simulated cycles on the simulator and wall clock on silicon,
    /// as [`Device::wait_for`] describes.
    pub fn await_launch(
        &mut self,
        mb: mailbox::launch::Mailbox,
        generation: u32,
    ) -> Result<(), ttStatus> {
        let mut waited = 0u64;
        let started = std::time::Instant::now();
        let limit = std::time::Duration::from_nanos(BUDGET).max(SILICON_MIN_WAIT);
        loop {
            if self.read32(mb.ack())? == generation {
                return Ok(());
            }
            if self.read32(mb.status())? == status::PANICKED {
                return Err(ttStatus::TT_ERROR_PROGRAM_REFUSED);
            }
            let expired = if self.link.is_simulated() {
                waited >= BUDGET
            } else {
                started.elapsed() >= limit
            };
            if expired {
                return Err(ttStatus::TT_ERROR_FIRMWARE_TIMEOUT);
            }
            self.clock(CYCLES_PER_POLL);
            waited += u64::from(CYCLES_PER_POLL);
        }
    }

    /// Place a RISC-V program and its launch arguments, run it on RISCV B to
    /// completion, and report a refusal.
    #[allow(dead_code)]
    pub fn run_program(
        &mut self,
        program_at: u32,
        args_at: u32,
        words: &[u32],
        args: &[u32],
    ) -> Result<u32, ttStatus> {
        if words.is_empty() {
            return Err(ttStatus::TT_ERROR_EMPTY_PROGRAM);
        }
        if (program_at as u64) + 4 * words.len() as u64 > args_at as u64 {
            let _ = (program_at, args_at);
            return Err(ttStatus::TT_ERROR_PROGRAM_OVERLAPS_ARGS);
        }
        self.place(program_at, words)?;
        let args_bytes: Vec<u8> = args.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.l1_write(u64::from(args_at), &args_bytes)?;
        let mb = mailbox::launch::Mailbox::b();
        let generation = self.launch(
            mb,
            program_at,
            words.len() as u32,
            args_at,
            mailbox::launch::ENABLE_B,
        )?;
        self.await_launch(mb, generation)?;
        self.read32(mb.result())
    }

    /// Point the window at `address` in `tile`, unless it already is.
    fn retarget(&mut self, tile: NocCoord, address: u64) -> Result<(), ttStatus> {
        let words = tlb_words(address, tile)?;
        if self.shadow == Some(words) {
            return Ok(());
        }
        // The configuration array is 12 bytes per window on Blackhole
        // (`blackhole.c:153-154`, `TLB_REG_SIZE`; UMD `pci_device.cpp:884-890`
        // with `TLB_CFG_REG_SIZE_BYTES`), so *this* session's window is
        // configured at `12 * window`. Writing window 0's register while
        // addressing window `n` would retarget another process's window and
        // leave ours unconfigured.
        let base = TLB_CONFIG_BASE + 12 * u64::from(self.link.window());
        for (i, word) in words.iter().enumerate() {
            self.bar_write32(base + 4 * i as u64, *word)?;
        }
        self.shadow = Some(words);
        Ok(())
    }

    fn window_write(&mut self, tile: NocCoord, address: u64, bytes: &[u8]) -> Result<(), ttStatus> {
        let mut done = 0usize;
        while done < bytes.len() {
            let at = address + done as u64;
            let base = at & !(WINDOW_SIZE - 1);
            let offset = at - base;
            let here = (bytes.len() - done).min((WINDOW_SIZE - offset) as usize);
            self.retarget(tile, base)?;
            let bar = self.window_bar_offset() + offset;
            check_bar0(bar, here)?;
            self.link
                .mem_write(BAR0_BASE + bar, &bytes[done..done + here]);
            done += here;
        }
        Ok(())
    }

    fn window_read(
        &mut self,
        tile: NocCoord,
        address: u64,
        out: &mut [u8],
    ) -> Result<(), ttStatus> {
        let mut done = 0usize;
        while done < out.len() {
            let at = address + done as u64;
            let base = at & !(WINDOW_SIZE - 1);
            let offset = at - base;
            let here = (out.len() - done).min((WINDOW_SIZE - offset) as usize);
            self.retarget(tile, base)?;
            let bar = self.window_bar_offset() + offset;
            check_bar0(bar, here)?;
            self.link
                .mem_read(BAR0_BASE + bar, &mut out[done..done + here]);
            done += here;
        }
        Ok(())
    }

    fn bar_write32(&self, offset: u64, value: u32) -> Result<(), ttStatus> {
        check_bar0(offset, 4)?;
        self.link
            .mem_write(BAR0_BASE + offset, &value.to_le_bytes());
        Ok(())
    }

    fn check_l1(&self, address: u64, len: usize) -> Result<(), ttStatus> {
        let end = address
            .checked_add(len as u64)
            .ok_or(ttStatus::TT_ERROR_L1_ADDRESS_OVERFLOW)?;
        if end > tensix::L1_SIZE {
            let _ = (address, end);
            return Err(ttStatus::TT_ERROR_L1_ACCESS);
        }
        Ok(())
    }

    fn set_core_reset(&mut self, core: Core, held: bool) -> Result<(), ttStatus> {
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

/// The byte offset of this session's window within BAR0.
impl Device {
    fn window_bar_offset(&self) -> u64 {
        u64::from(self.link.window()) * WINDOW_SIZE
    }
}

/// Refuse an access BAR0 would not decode: past the window aperture, or into
/// the configuration array.
fn check_bar0(offset: u64, len: usize) -> Result<(), ttStatus> {
    if len == 0 {
        return Err(ttStatus::TT_ERROR_BAR0_ZERO_LENGTH);
    }
    let end = offset
        .checked_add(len as u64)
        .ok_or(ttStatus::TT_ERROR_BAR0_OVERFLOW)?;
    let in_aperture = end <= 202 * WINDOW_SIZE;
    let in_config = offset >= TLB_CONFIG_BASE && end <= TLB_CONFIG_BASE + 210 * 12;
    // The ARC's APB window, the last MiB of BAR0: `ARC_APB_BAR0_XBAR_OFFSET_START`
    // (`blackhole_implementation.hpp:222`), where the message-queue doorbell is.
    let in_arc_apb = offset >= ARC_APB_BAR0 && end <= crate::kmd::BAR0_BYTES;
    if !in_aperture && !in_config && !in_arc_apb {
        let _ = (offset, end);
        return Err(ttStatus::TT_ERROR_BAR0_NOT_DECODED);
    }
    Ok(())
}

/// Encode the three dwords of a 2 MiB unicast window on NoC #0, strict ordering.
fn tlb_words(base_address: u64, tile: NocCoord) -> Result<[u32; 3], ttStatus> {
    if base_address % WINDOW_SIZE != 0 {
        let _ = base_address;
        return Err(ttStatus::TT_ERROR_WINDOW_ALIGNMENT);
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
/// arena; this is where every program lives in L1, named by its launch message.
pub const PROGRAM_AREA: u64 = l1::PROGRAM_CACHE.base;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_image_is_present_and_linked_where_it_is_loaded() {
        for (name, bytes) in all_images() {
            assert!(!bytes.is_empty(), "{name} is empty");
        }
        let [t0, t1, t2] = ROLES;
        assert_eq!(t0.0, Core::T0);
        assert_eq!(t1.0, Core::T1);
        assert_eq!(t2.0, Core::T2);
        for (i, (_, bytes, at)) in ROLES.iter().enumerate() {
            assert_eq!(
                *at,
                tt_isa::l1::roles::BASE[i],
                "role {i} is loaded at its slot"
            );
            assert!(
                bytes.len() as u64 <= tt_isa::l1::roles::SLOT,
                "role {i} is {} bytes for a {} byte slot",
                bytes.len(),
                tt_isa::l1::roles::SLOT
            );
        }
    }

    #[test]
    fn the_executor_is_the_image_the_mailbox_starts() {
        // The executor is linked at T0's reset PC, which `link.x` also uses for the
        // images loaded through a plain reset.
        assert_eq!(LOAD_ADDRESS, u64::from(Core::T0.default_reset_pc()));
        assert!(!EXEC_B.is_empty());
    }
}
