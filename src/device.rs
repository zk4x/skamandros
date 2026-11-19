// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The device: `libttsim` through `dlopen`, one TLB window, DRAM and L1 access,
//! and starting a core on an image.
//!
//! `libttsim` is a process-wide singleton with no context handle. Every contract
//! violation terminates the process (`_Exit`), so nothing here hands it an
//! address it does not decode: the window geometry is checked before each call,
//! the TLB configuration is the one the library accepts, and L1 access is
//! bounded by [`tt_isa::arch::blackhole::Blackhole::L1_SIZE`].
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
use tt_isa::arch::Arch;
use tt_isa::arch::blackhole::Blackhole;
use tt_isa::arch::blackhole::dram::{CHANNEL_BYTES, Dram, DramChannel};
use tt_isa::arch::blackhole::l1;
use tt_isa::arch::blackhole::mailbox::{self, status};
use tt_isa::arch::blackhole::noc::NocCoord;
use tt_isa::arch::blackhole::noc::niu::Endpoint;
use tt_isa::tensix::Core;

/// `RISC_SOFT_RESET_0`: the one register every Blackhole core and the backend
/// share. The crate names it only through [`Arch::reset`], which also holds the
/// bit of each core, so the two are looked up together.
fn soft_reset_register() -> u64 {
    Blackhole::reset(Core::B)
        .expect("Blackhole has a reset bit for B")
        .0
}

/// The reset bit of `core`. Every core this driver names is Blackhole's, so the
/// architecture's refusal is unreachable here.
fn soft_reset_mask(core: Core) -> u32 {
    Blackhole::soft_reset_mask(core).expect("Blackhole's core set is the driver's")
}

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
/// coordinate-translation enable. The low twelve bits are reserved — they carry
/// no coordinate (`NoC/MemoryMap.md:228-238`), so nothing reads a tile's X here.
const NIU_CFG_0: u64 = 0x1FD0_4100;

/// The PCIe tile NIU's `NOC_NODE_ID`: `NIU_CFG_NOC0_BAR_PCIE_ADDR + 0x44`
/// (`blackhole_implementation.hpp:298`, and `get_read_checking_offset`,
/// `:419-420`). Bits 0-5 are the NIU's X coordinate (`NoC/MemoryMap.md:174-186`),
/// and this is the register tt-metal reads with `get_pcie_x_coordinate`
/// (`blackhole_tt_device.cpp:290-292`) to decide whether the ARC answers over
/// AXI (`is_arc_available_over_axi`, `:296-303`).
const NIU_NODE_ID: u64 = 0x1FD0_4044;
// Both are registers of the same NIU, whose window starts at 0x1FD0_4000.
const _: () = assert!(NIU_NODE_ID + 0xBC == NIU_CFG_0);

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
/// `DRAM_TRAINING_TIMEOUT` (`umd/device/utils/timeouts.hpp:22`), the bound on
/// waiting for a channel's firmware training.
const DRAM_TRAINING_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
/// A trained GDDR channel's two-bit state in `GDDR_STATUS`
/// (`FirmwareInfoProvider::get_modern_dram_statuses`,
/// `firmware_info_provider.cpp:470-488`); `0b10` and `0b11` are failures, which
/// is what testing the high bit finds, and `0b00` is still training.
const GDDR_TRAINED: u32 = 0b01;
const GDDR_FAILED: u32 = 0b10;

/// The cadence of a silicon poll: tt-metal's `poll_until` busy-polls for
/// `busy_poll_window` and then sleeps `poll_interval` between reads
/// (`umd/device/common/utils.hpp:136-159`; the ARC startup uses 1 ms and 10 µs,
/// `blackhole_tt_device.cpp:184-195`). Every read is a PCIe round trip on the
/// tile whose cores the poll is waiting for, so an unpaced loop is traffic
/// against the kernel it is waiting on.
const POLL_BUSY_WINDOW: std::time::Duration = std::time::Duration::from_millis(1);
const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_micros(10);

/// How long the ARC is given to come up: tt-kmd's `ARC_MSG_READY_MS`
/// (`blackhole.c:69`).
///
/// tt-metal waits up to `ARC_STARTUP_TIMEOUT` = 300 s for the same status
/// (`timeouts.hpp:14`). Half a second is longer than tt-kmd's own bound allows
/// for a card the driver has already brought up, and a card still booting that
/// long after the module let us open is reported rather than waited on.
const ARC_READY_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);
/// The interval between two reads of the ARC's boot-status word.
const ARC_READY_POLL: std::time::Duration = std::time::Duration::from_micros(100);

/// The two values a host-to-device L1 barrier alternates through, tt-metal's
/// `MemBarFlag { SET = 0xaa, RESET = 0xbb }`
/// (`umd/device/types/cluster_types.hpp:106-109`) written to
/// [`tt_isa::arch::blackhole::mailbox::BARRIER`].
const L1_BARRIER_SET: u32 = 0xAA;
const L1_BARRIER_RESET: u32 = 0xBB;
/// How many times the L1 barrier word is read before the tile is declared not to
/// answer. A flush is one or two round trips.
const L1_BARRIER_POLLS: u32 = 1000;

/// First DRAM byte this driver hands out.
/// tt-metal reserves the bottom of every channel for the DRAM barrier (one
/// alignment word at offset 0), the profiler block (empty in a build without
/// Tracy) and the device-print dispatch region (`DRAM_ALIGNMENT + 1 MiB`), so
/// its `UNRESERVED` base is `0x10_0080` — 128 bytes and a mebibyte
/// (`bh_hal.cpp:33-83`, `:329-332`). This driver writes none of those regions,
/// but starting above them keeps its data out of anything the card's own
/// dispatch would touch.
const DRAM_FIRST: u64 = 0x10_0080;

/// Bit 63 of a handle: a striped tensor, not a DRAM address.
///
/// A DRAM address is `channel * CHANNEL_BYTES + offset` with at most eight
/// channels of 4 GiB, so bit 63 is never set on a real address.
const INTERLEAVED_TAG: u64 = 1 << 63;

/// A tensor whose pages are striped across the DRAM channels. See
/// [`Device::alloc_interleaved`].
pub struct Interleaved {
    /// Bytes in one page.
    pub page: u32,
    /// The first page's offset in each channel, in channel order.
    pub base: Vec<u64>,
}

/// The most one PCIe DMA transfer moves: UMD's "512KB is the empirical sweet
/// spot for per-chunk DMA throughput" (`pci_device.cpp:1045-1050`).
const DMA_CHUNK: usize = 512 * 1024;

/// Transfers at or above this many bytes go through the DMA engine. Below it
/// the fixed cost of a doorbell and a completion poll is more than the window
/// copy it replaces.
const DMA_MIN: usize = 4096;

/// Whether host transfers use the PCIe DMA engine.
///
/// The transcription of `BlackholeDmaTransfer::h2d_start` is in
/// [`crate::kmd::Kmd::dma_h2d`], but the destination address it is given is the
/// window's AXI address, and a 2 MiB window is a BAR0 aperture: the engine,
/// which reaches the card through BAR4, does not complete a transfer aimed
/// there (measured: the completion marker never lands). UMD's DMA path
/// allocates its own 4 GiB BAR4 window for exactly this reason
/// (`get_dma_tlb_size`, `pcie_protocol.cpp:548-560`), and this driver does not
/// map or configure that class yet.
const DMA_ENABLED: bool = false;

/// How many times a DMA window's configuration is read back before the card is
/// declared not to have taken it.
const CONFIG_POLLS: u32 = 1000;

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
/// than two device types. Each variant is constructed by its own `open`.
enum Link {
    /// The simulator: `dlopen`ed `libttsim_bh.so`, entry points resolved once.
    Sim(&'static Lib),
    /// `/dev/tenstorrent/N` through tt-kmd, opened by [`Device::open_hardware`].
    Driver(Kmd),
}

impl Link {
    /// Device registers: the TLB configuration array, the PCIe NIU's
    /// configuration word, the ARC's APB doorbell. Uncached on the card: these
    /// are registers, and a merged or reordered store is silent corruption.
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

    /// Window bytes: L1 and DRAM through the tile-addressed TLB aperture.
    /// Write-combined on the card, and fenced by the link once a write is out.
    fn bulk_write(&self, paddr: u64, bytes: &[u8]) {
        match self {
            Link::Sim(lib) => lib.mem_write(paddr, bytes),
            Link::Driver(kmd) => {
                let offset = bar0_offset(paddr).expect("a BAR0 address this driver wrote");
                kmd.window_write(offset, bytes)
                    .expect("a BAR0 address this driver wrote");
            }
        }
    }

    fn bulk_read(&self, paddr: u64, out: &mut [u8]) {
        match self {
            Link::Sim(lib) => lib.mem_read(paddr, out),
            Link::Driver(kmd) => {
                let offset = bar0_offset(paddr).expect("a BAR0 address this driver read");
                kmd.window_read(offset, out)
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
#[cfg(feature = "firmware")]
pub type Image = (Core, &'static [u8], u64);

/// Where `link.x` places an image: RISCV T0's default reset PC, which every image
/// but the role images and the per-core probes uses.
///
/// A literal because `Arch::default_reset_pc` is not a `const fn`; the test
/// below pins it to the architecture's answer.
#[cfg(test)]
pub const LOAD_ADDRESS: u64 = 0x0_6000;

/// The three role images -- unpack on T0, math on T1, pack on T2 -- each linked
/// at its own slot (`tt_isa::arch::blackhole::l1::roles`), so all three fit in one tile's L1 at
/// once and each starts through its reset-PC override.
#[cfg(all(test, feature = "firmware"))]
pub const ROLES: [Image; 3] = [
    (
        Core::T0,
        include_bytes!(env!("FIRMWARE_ROLE_T0")),
        tt_isa::arch::blackhole::l1::roles::BASE[0],
    ),
    (
        Core::T1,
        include_bytes!(env!("FIRMWARE_ROLE_T1")),
        tt_isa::arch::blackhole::l1::roles::BASE[1],
    ),
    (
        Core::T2,
        include_bytes!(env!("FIRMWARE_ROLE_T2")),
        tt_isa::arch::blackhole::l1::roles::BASE[2],
    ),
];

/// The program executor: reads a staged program's address, length and arguments
/// from the mailbox, invalidates the I-cache and calls it.
#[cfg(feature = "firmware")]
pub const EXEC_B: &[u8] = include_bytes!(env!("FIRMWARE_EXEC_B"));

/// The three role executors -- unpack on T0, math on T1, pack on T2 -- each
/// linked at its role slot. They read their own role mailbox and call the
/// program with `a0` at its arguments, the way tt-metal's firmware calls a
/// kernel; nothing is pushed.
#[cfg(feature = "firmware")]
pub const EXEC_ROLES: [Image; 3] = [
    (
        Core::T0,
        include_bytes!(env!("FIRMWARE_EXEC_T0")),
        tt_isa::arch::blackhole::l1::roles::BASE[0],
    ),
    (
        Core::T1,
        include_bytes!(env!("FIRMWARE_EXEC_T1")),
        tt_isa::arch::blackhole::l1::roles::BASE[1],
    ),
    (
        Core::T2,
        include_bytes!(env!("FIRMWARE_EXEC_T2")),
        tt_isa::arch::blackhole::l1::roles::BASE[2],
    ),
];

/// RISCV NC's executor: the writer's program, called the way the TRISCs call
/// theirs. NC cannot invalidate its own cache, so `exec_b` invalidates it before
/// each launch; the image runs from `dm_nc`'s slot behind a jump at NC's reset
/// PC.
#[cfg(feature = "firmware")]
pub const EXEC_NC: &[u8] = include_bytes!(env!("FIRMWARE_EXEC_NC"));

/// The four subordinate executors, loaded held and released together after
/// RISCV B has invalidated every instruction cache.
#[cfg(feature = "firmware")]
pub const EXEC_SUBORDINATES: [Image; 4] = [
    EXEC_ROLES[0],
    EXEC_ROLES[1],
    EXEC_ROLES[2],
    (
        Core::NC,
        EXEC_NC,
        tt_isa::arch::blackhole::dm::nc::IMAGE_BASE,
    ),
];

/// RISCV NC's mover image (`tt_isa::arch::blackhole::dm::nc`), reached through the stub at NC's
/// reset PC.
#[cfg(all(test, feature = "firmware"))]
pub const DM_NC: &[u8] = include_bytes!(env!("FIRMWARE_DM_NC"));

/// RISCV B's data mover (`tt_isa::arch::blackhole::dm`), loaded at [`tt_isa::arch::blackhole::dm::IMAGE_BASE`]
/// and reached through the jump the loader writes at its hardwired reset PC.
#[cfg(all(test, feature = "firmware"))]
pub const DM_B: &[u8] = include_bytes!(env!("FIRMWARE_DM_B"));

/// Every image this build produced, by name, with its load address.
///
/// The list exists so a test can name every image the build gate judged, rather
/// than trusting that the one it embeds was among them.
#[cfg(all(test, feature = "firmware"))]
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

/// FNV-1a over words: enough to skip rewriting an unchanged program or setup,
/// not a correctness check on the words.
pub(crate) fn hash_words(words: &[u32]) -> u64 {
    words.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, word| {
        (h ^ u64::from(*word)).wrapping_mul(0x0000_0100_0000_01b3)
    })
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
    /// First byte of each channel this device has not handed out, in logical
    /// channel order.
    dram_used: Vec<u64>,
    /// The pinned host buffer the PCIe DMA engine reads, once a bulk transfer
    /// has needed one. `None` on the simulator, whose transfers go through the
    /// window like a register access.
    dma: Option<crate::kmd::DmaBuffer>,
    /// Tensors whose pages are striped across the channels, in allocation
    /// order. A tagged address names one of these.
    interleaved: Vec<Interleaved>,
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
    /// Host-staged words already at their tile address, keyed by
    /// `(tile x, tile y, address)`. A launch compares what it would write with
    /// this and writes only the runs that differ, so a repeated launch has no
    /// argument traffic at all and a first launch writes each block once.
    staged: std::collections::HashMap<(u8, u8, u64), Vec<u32>>,
    /// The value the next `l1_barrier` writes. Alternating means the word can
    /// never already hold the value being written, so one write and one
    /// read-back prove the write landed (tt-metal's two-phase
    /// `insert_host_to_device_barrier` is this written out without knowing the
    /// previous value).
    barrier_flag: u32,
    /// Tiles whose Tensix backend has been released from soft reset. Releasing
    /// is idempotent, and the register read it costs is per launch otherwise.
    backend_released: Vec<NocCoord>,
    /// Has the host written DRAM since the last barrier? A launch only needs
    /// the DRAM barrier if it has: the barrier exists to make posted host
    /// writes visible to the device, and a launch after no host write has
    /// nothing to flush.
    dram_dirty: bool,
    /// The module whose programs, arguments, CB interface template and
    /// dispatcher descriptor are staged on every tile of its grid
    /// (`Module::hash` with the resolved grid). A launch of that same module
    /// again can be fanned out on the card from one tile instead of staged from
    /// the host. Cleared whenever the tiles are reset or their executors
    /// restart.
    installed: Option<u64>,
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
            tile: tt_isa::arch::blackhole::arc::arc_tile(),
            tiles: Vec::new(),
            columns: 0,
            dram: Dram::FULL,
            dram_used: vec![DRAM_FIRST; usize::from(Dram::FULL.channels().count())],
            dma: None,
            interleaved: Vec::new(),
            exec_started: Vec::new(),
            launch_seq: 0,
            placed: Vec::new(),
            staged: std::collections::HashMap::new(),
            barrier_flag: L1_BARRIER_SET,
            backend_released: Vec::new(),
            dram_dirty: false,
            installed: None,
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
    /// Device 0 only: on a multi-card host the node is not chosen, so a second
    /// card is not reached. Opening the card also requires `TT_HARDWARE=1`, so
    /// nothing reaches silicon by accident.
    pub fn open_hardware() -> Result<Device, ttStatus> {
        if std::env::var_os("TT_HARDWARE").as_deref() != Some(OsStr::new("1")) {
            eprintln!(
                "libskamandros: refusing to open the card: TT_HARDWARE=1 is required \
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
            tile: tt_isa::arch::blackhole::arc::arc_tile(),
            tiles: Vec::new(),
            columns: 0,
            dram: Dram::FULL,
            dram_used: vec![DRAM_FIRST; usize::from(Dram::FULL.channels().count())],
            dma: None,
            interleaved: Vec::new(),
            exec_started: Vec::new(),
            launch_seq: 0,
            placed: Vec::new(),
            staged: std::collections::HashMap::new(),
            barrier_flag: L1_BARRIER_SET,
            backend_released: Vec::new(),
            dram_dirty: false,
            installed: None,
        };
        // The ARC publishes every address the rest of this reads, so it is
        // waited for first, as tt-metal waits before it creates its messenger or
        // its telemetry reader.
        device.wait_arc_ready()?;
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
        device.dram_used = vec![DRAM_FIRST; usize::from(device.dram.channels().count())];

        // Firmware trains each GDDR channel after power-up, and tt-metal waits
        // for every enabled channel before it touches DRAM (`LocalChip`'s
        // constructor -> `wait_chip_to_be_ready`, `local_chip.cpp:107-112`,
        // `chip.cpp:81-95`). The status is `GDDR_STATUS`'s two bits per channel
        // (`firmware_info_provider.cpp:470-488`). A channel that failed is
        // refused rather than retrained: tt-metal's recovery is
        // `TOGGLE_GDDR_RESET`, an ARC message that disturbs the card's DRAM, and
        // this driver sends no reset. tt-metal ends its wait when the tag is
        // absent, and so does this.
        let enabled = device.dram.enabled_mask();
        let started = std::time::Instant::now();
        loop {
            let Some(status) = device.arc_tag(tt_isa::arch::blackhole::arc::tag::GDDR_STATUS)?
            else {
                eprintln!(
                    "libskamandros: telemetry carries no GDDR training status; not waiting for \
                     training"
                );
                break;
            };
            let mut failed = None;
            let mut trained = true;
            for bank in 0..u32::from(tt_isa::arch::blackhole::dram::CHANNELS) {
                if enabled & (1 << bank) == 0 {
                    continue;
                }
                let state = (status >> (2 * bank)) & 0x3;
                if state == GDDR_TRAINED {
                    continue;
                }
                trained = false;
                if state & GDDR_FAILED != 0 {
                    failed = Some(bank);
                }
            }
            if let Some(bank) = failed {
                eprintln!(
                    "libskamandros: GDDR channel {bank} failed training (status word {status:#010x}); \
                     refusing to use DRAM, and not retraining it"
                );
                return Err(ttStatus::TT_ERROR_TELEMETRY);
            }
            if trained {
                break;
            }
            if started.elapsed() >= DRAM_TRAINING_TIMEOUT {
                eprintln!(
                    "libskamandros: GDDR training still reads {status:#010x} after \
                     {DRAM_TRAINING_TIMEOUT:?}; refusing to use DRAM"
                );
                return Err(ttStatus::TT_ERROR_TELEMETRY);
            }
        }

        // Nothing on the card is ours until now, and a previous process leaves
        // its cores running: hold the five on every tile, clear the launch state,
        // and register the one write that survives this process dying. The driver
        // performs that write when the descriptor closes, whatever closed it
        // (`tt_cdev_release_noc_cleanup`, `chardev.c:1021-1030`).
        device.quiesce()?;
        let held = soft_reset_mask(Core::B)
            | soft_reset_mask(Core::T0)
            | soft_reset_mask(Core::T1)
            | soft_reset_mask(Core::T2)
            | soft_reset_mask(Core::NC);
        let first = device.tile;
        if let Link::Driver(kmd) = &device.link {
            kmd.hold_on_close(first.x(), first.y(), soft_reset_register(), held)?;
        }
        // tt-metal puts the chip at its busy operating point once the device is
        // up (`Cluster::deassert_resets_and_set_power_state`, `cluster.cpp:1080-1099`).
        // At the idle point the Matrix Unit's `Src` reads are unreliable on
        // silicon, so this belongs to bringing the device up.
        device.set_power_state(true)?;
        // Which target this is: the same C ABI drives ttsim, which prints
        // nothing, so this line is what tells a run it reached silicon.
        let clock = device
            .arc_tag(tt_isa::arch::blackhole::arc::tag::AICLK)?
            .map_or_else(|| "unknown".to_string(), |mhz| format!("{mhz} MHz"));
        let limit = device
            .arc_tag(tt_isa::arch::blackhole::arc::tag::AICLK_LIMIT_MAX)?
            .map_or_else(|| "unknown".to_string(), |mhz| format!("{mhz} MHz"));
        eprintln!(
            "libskamandros: running on hardware: {}/{}, {} Tensix columns, {} tiles, \
             DRAM mask {:#04x}, AICLK {clock} (limit {limit})",
            crate::kmd::DEV_DIR,
            0,
            device.columns,
            device.tiles.len(),
            device.dram.enabled_mask(),
        );
        Ok(device)
    }

    /// Wait for the ARC to report that it is up, before anything reads its
    /// memory.
    ///
    /// tt-metal's device init reads one scratch word (`probe_arc`) and then
    /// waits here *before* it creates either its message queue or its telemetry
    /// reader (`TTDevice::init_tt_device`, `tt_device.cpp:119-123`):
    /// `BlackholeTTDevice::wait_arc_core_start` polls `SCRATCH_RAM_2` until its
    /// low three bits read `0x5` (`blackhole_tt_device.cpp:182-203`). Everything
    /// this driver learns about the chip — the harvesting masks, the DRAM mask,
    /// the clock — is read from addresses the ARC publishes, so the same wait
    /// belongs before the first of those reads.
    ///
    /// All ones is the NoC not answering, the state tt-kmd turns into `-EIO`
    /// (`blackhole.c:543-545`); it is reported rather than polled.
    fn wait_arc_ready(&mut self) -> Result<(), ttStatus> {
        use tt_isa::arch::blackhole::arc;
        let arc_tile = arc::arc_tile();
        let started = std::time::Instant::now();
        loop {
            let status = self.arc_word(arc_tile, arc::BOOT_STATUS)?;
            if status == u32::MAX {
                eprintln!(
                    "libskamandros: the ARC's boot-status word at {:#x} did not answer; the \
                     NoC is not routing to the ARC tile",
                    arc::BOOT_STATUS
                );
                return Err(ttStatus::TT_ERROR_DEVICE);
            }
            if status & 0x7 == arc::BOOT_STATUS_READY {
                return Ok(());
            }
            if started.elapsed() >= ARC_READY_TIMEOUT {
                eprintln!(
                    "libskamandros: the ARC's boot status still reads {status:#010x} after \
                     {ARC_READY_TIMEOUT:?}; refusing to read its memory"
                );
                return Err(ttStatus::TT_ERROR_TELEMETRY);
            }
            std::thread::sleep(ARC_READY_POLL);
        }
    }

    /// Does the NIU translate coordinates?
    ///
    /// Every Tensix tile this driver names is a *logical* coordinate: with
    /// translation on, the NIU maps it onto a surviving physical column, which is
    /// why the first `n` columns of the grid are the right ones on a harvested chip
    /// ([`tt_isa::arch::blackhole::noc::grid::Tensix`]). With translation off those numbers are
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
        use tt_isa::arch::blackhole::arc;
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
    /// way tt-metal reads it ([`tt_isa::arch::blackhole::noc::grid::Tensix::from_enabled_column_mask`]),
    /// and the survivors are a prefix of the translated columns whatever the bit
    /// order (`NoC/Coordinates.md:54`). A chip that will not name its columns,
    /// or names none at all, is refused: assuming a full part writes to a
    /// fused-off tile, which ttsim v1.16 treats as undefined.
    fn probe_tensix(&mut self) -> Result<tt_isa::arch::blackhole::noc::grid::Tensix, ttStatus> {
        use tt_isa::arch::blackhole::arc;
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
        tt_isa::arch::blackhole::noc::grid::Tensix::from_enabled_column_mask(mask)
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
        use tt_isa::arch::blackhole::arc;
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
        use tt_isa::arch::blackhole::arc;
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
        // `blackhole_tt_device.cpp:237-257`). The X comes from the NIU's node-id
        // register, as tt-metal reads it; `NIU_CFG_0`'s low bits are reserved.
        let mut node = [0u8; 4];
        self.link.mem_read(BAR0_BASE + NIU_NODE_ID, &mut node);
        if u32::from_le_bytes(node) & 0x3F == 11 {
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
        if !busy && self.arc_tag(tt_isa::arch::blackhole::arc::tag::AICLK)? == Some(AICLK_IDLE) {
            return Ok(());
        }
        let target = if busy {
            self.arc_tag(tt_isa::arch::blackhole::arc::tag::AICLK_LIMIT_MAX)?
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
            let aiclk = self.arc_tag(tt_isa::arch::blackhole::arc::tag::AICLK)?;
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

    /// The chip's `ENABLED_GDDR` word, for the data mover's `USABLE` mailbox.
    pub fn dram_mask(&self) -> u32 {
        u32::from(self.dram.enabled_mask())
    }

    /// Every channel, in logical order.
    pub fn dram_channels(&self) -> Vec<DramChannel> {
        self.dram.channels().collect()
    }

    /// The channel a flat DRAM address lives on, and its offset inside that
    /// channel.
    ///
    /// An address is `channel * CHANNEL_BYTES + offset`, the same shape
    /// tt-metal gives a bank: `bank_id * bank_size + offset`, with the bank's
    /// NoC endpoint resolved from the bank id (`get_noc_addr_from_bank_id`,
    /// `dataflow_api_addrgen.h:516-533`). Splitting it needs no table: the
    /// channel is the quotient and the offset the remainder.
    fn dram_at(&self, address: u64) -> Result<(DramChannel, u64), ttStatus> {
        if address & INTERLEAVED_TAG != 0 {
            // A striped tensor has no single channel; its pages do.
            return Err(ttStatus::TT_ERROR_DRAM_RANGE);
        }
        let channel = address / CHANNEL_BYTES;
        let offset = address % CHANNEL_BYTES;
        let channel = u8::try_from(channel)
            .ok()
            .and_then(|index| self.dram.channel(index))
            .ok_or(ttStatus::TT_ERROR_DRAM_RANGE)?;
        Ok((channel, offset))
    }

    /// Reserve `bytes` of DRAM, 64-byte aligned, and return its flat address
    /// (`channel * CHANNEL_BYTES + offset`). This is what a caller passes back
    /// to `ttMemcpy` and `ttLaunch`.
    /// The channel is the one with the most free space, as tt-metal's
    /// `ChannelAllocator` picks a bank — `free_size` per bank and the largest
    /// first (`allocator.cpp:57-100`, `bank_manager.cpp:417-446`). No range is
    /// ever freed here, so "most free" is "least used".
    pub fn alloc_dram(&mut self, bytes: usize) -> Result<u64, ttStatus> {
        if bytes == 0 {
            return Err(ttStatus::TT_ERROR_ALLOC_ZERO);
        }
        let bytes = bytes as u64;
        let mut best: Option<(u8, u64)> = None;
        for channel in self.dram.channels() {
            let used = self.dram_used[usize::from(channel.index())];
            let at = used.next_multiple_of(64);
            let fits = at
                .checked_add(bytes)
                .is_some_and(|end| end <= CHANNEL_BYTES);
            if fits && best.is_none_or(|(_, free)| used < free) {
                best = Some((channel.index(), used));
            }
        }
        let Some((index, used)) = best else {
            let _ = bytes;
            return Err(ttStatus::TT_ERROR_DRAM_FULL);
        };
        let at = used.next_multiple_of(64);
        self.dram_used[usize::from(index)] = at + bytes;
        Ok(u64::from(index) * CHANNEL_BYTES + at)
    }

    /// Reserve a tensor whose pages are striped across the channels, and
    /// return its handle.
    ///
    /// Page `p` lives on channel `p % channels` at
    /// `base[channel] + (p / channels) * page`, which is tt-metal's
    /// interleaved layout: the allocator gives every bank
    /// `ceil(pages / num_banks)` pages and the page's bank is its index modulo
    /// the bank count (`AllocatorImpl::allocate` for an interleaved buffer, and
    /// the matching address generator on the core,
    /// `dataflow_api_addrgen.h:516-533`). Each channel's share comes from that
    /// channel's cursor, so a striped tensor is spread over every bank the chip
    /// has.
    ///
    /// The handle is a tag and an index: no DRAM address has bit 63 set, so a
    /// handle can be told from one, and [`Device::dram_write`] and the launch
    /// read the layout back from it.
    pub fn alloc_interleaved(&mut self, bytes: usize, page: usize) -> Result<u64, ttStatus> {
        if bytes == 0 || page == 0 {
            return Err(ttStatus::TT_ERROR_ALLOC_ZERO);
        }
        if page as u64 > CHANNEL_BYTES {
            return Err(ttStatus::TT_ERROR_DRAM_RANGE);
        }
        let pages = bytes.div_ceil(page);
        let channels = self.dram.channels().count();
        let per_channel = pages.div_ceil(channels) * page;
        let mut base = Vec::with_capacity(channels);
        for channel in self.dram.channels() {
            let used = self.dram_used[usize::from(channel.index())].next_multiple_of(64);
            let end = used
                .checked_add(per_channel as u64)
                .ok_or(ttStatus::TT_ERROR_DRAM_ADDRESS_OVERFLOW)?;
            if end > CHANNEL_BYTES {
                let _ = (bytes, page);
                return Err(ttStatus::TT_ERROR_DRAM_FULL);
            }
            base.push(used);
        }
        for (channel, at) in self.dram.channels().zip(&base) {
            self.dram_used[usize::from(channel.index())] = at + per_channel as u64;
        }
        let index = self.interleaved.len() as u64;
        self.interleaved.push(Interleaved {
            page: page as u32,
            base,
        });
        Ok(INTERLEAVED_TAG | index)
    }

    /// The layout a striped tensor's handle names, if `address` is a handle.
    pub fn interleaved_at(&self, address: u64) -> Option<&Interleaved> {
        if address & INTERLEAVED_TAG == 0 {
            return None;
        }
        self.interleaved.get((address & !INTERLEAVED_TAG) as usize)
    }

    /// One striped tensor's pages, host side: page `p` goes to channel
    /// `p % channels`, as the core's address arithmetic expects.
    fn dram_write_interleaved(&mut self, handle: u64, bytes: &[u8]) -> Result<(), ttStatus> {
        let (page, base) = match self.interleaved_at(handle) {
            Some(layout) => (layout.page as usize, layout.base.clone()),
            None => return Err(ttStatus::TT_ERROR_DRAM_RANGE),
        };
        let channels = base.len();
        for (index, chunk) in bytes.chunks(page).enumerate() {
            let bank = index % channels;
            let at = u64::from(bank as u32) * CHANNEL_BYTES
                + base[bank]
                + (index / channels) as u64 * page as u64;
            self.dram_write(at, chunk)?;
        }
        Ok(())
    }

    fn dram_read_interleaved(&mut self, handle: u64, out: &mut [u8]) -> Result<(), ttStatus> {
        let (page, base) = match self.interleaved_at(handle) {
            Some(layout) => (layout.page as usize, layout.base.clone()),
            None => return Err(ttStatus::TT_ERROR_DRAM_RANGE),
        };
        let channels = base.len();
        for (index, chunk) in out.chunks_mut(page).enumerate() {
            let bank = index % channels;
            let at = u64::from(bank as u32) * CHANNEL_BYTES
                + base[bank]
                + (index / channels) as u64 * page as u64;
            self.dram_read(at, chunk)?;
        }
        Ok(())
    }

    /// The endpoint high word a program names for the channel `address` lives
    /// on.
    ///
    /// A DRAM address is a bank id and an offset; the NoC endpoint is the
    /// bank's, resolved per NIU (`get_noc_addr_from_bank_id` adds
    /// `bank_to_dram_offset[bank_id]` to the offset and the packed endpoint to
    /// the high word, `dataflow_api_addrgen.h:516-533`). The reader and the
    /// writer use opposite NoCs, and the endpoints differ between them.
    pub fn dram_hi(
        &self,
        address: u64,
        niu: tt_isa::arch::blackhole::noc::niu::Niu,
    ) -> Result<u32, ttStatus> {
        let (channel, _) = self.dram_at(address)?;
        let ep = channel.preferred(niu);
        Ok(Endpoint {
            x: ep.x(),
            y: ep.y(),
            addr: 0,
        }
        .hi())
    }

    /// Write `bytes` to DRAM at a flat address.
    pub fn dram_write(&mut self, address: u64, bytes: &[u8]) -> Result<(), ttStatus> {
        self.dram_dirty = true;
        if address & INTERLEAVED_TAG != 0 {
            return self.dram_write_interleaved(address, bytes);
        }
        let (channel, offset) = self.dram_at(address)?;
        let range = channel.range(offset, bytes.len() as u64).ok_or_else(|| {
            let _ = (bytes.len(), address);
            ttStatus::TT_ERROR_DRAM_RANGE
        })?;
        let ep = channel.preferred(tt_isa::arch::blackhole::noc::niu::Niu::Noc0);
        if DMA_ENABLED && bytes.len() >= DMA_MIN && self.dma_write(ep, range.offset(), bytes)? {
            // The engine's completion flag is the transfer's witness; the window
            // read is a second one, from the endpoint the data went out of.
            if bytes.len() >= 4 {
                let last = range.offset() + bytes.len() as u64 - 4;
                let mut word = [0u8; 4];
                self.window_read(ep, last, &mut word)?;
            }
            return Ok(());
        }
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

    /// Has the host written DRAM since the last launch's barrier?
    pub fn dram_dirty(&self) -> bool {
        self.dram_dirty
    }

    /// Record that every host DRAM write so far has been made visible, so a
    /// launch with no new writes does not pay for another barrier.
    pub fn mark_dram_clean(&mut self) {
        self.dram_dirty = false;
    }

    /// Read DRAM at a flat address.
    pub fn dram_read(&mut self, address: u64, out: &mut [u8]) -> Result<(), ttStatus> {
        if address & INTERLEAVED_TAG != 0 {
            return self.dram_read_interleaved(address, out);
        }
        let (channel, offset) = self.dram_at(address)?;
        channel.range(offset, out.len() as u64).ok_or_else(|| {
            let _ = (out.len(), address);
            ttStatus::TT_ERROR_DRAM_RANGE
        })?;
        let ep = channel.preferred(tt_isa::arch::blackhole::noc::niu::Niu::Noc0);
        self.window_read(ep, offset, out)
    }

    /// Flush this tile's L1: write the barrier word and wait for it to read
    /// back.
    ///
    /// tt-metal's `l1_membar` -> `insert_host_to_device_barrier`
    /// (`local_chip.cpp:472-508`) writes `0xAA`, waits for the tile to show it,
    /// then does the same with `0xBB`, so a read that returns the value just
    /// written cannot be a leftover from the previous barrier. This keeps that
    /// guarantee with one phase by alternating the value across calls: the word
    /// never already holds the one being written, so a read-back of it is proof
    /// the write landed and, behind it, every earlier L1 write.
    ///
    /// The word is [`tt_isa::arch::blackhole::mailbox::BARRIER`], which no core reads: tt-metal's
    /// own `MEM_L1_BARRIER` would land inside this driver's B image.
    pub fn l1_barrier(&mut self) -> Result<(), ttStatus> {
        let flag = self.barrier_flag;
        self.barrier_flag = if flag == L1_BARRIER_SET {
            L1_BARRIER_RESET
        } else {
            L1_BARRIER_SET
        };
        self.write32(mailbox::BARRIER, flag)?;
        let mut polls = 0u32;
        loop {
            if self.read32(mailbox::BARRIER)? == flag {
                break;
            }
            polls += 1;
            if polls == L1_BARRIER_POLLS {
                eprintln!(
                    "libskamandros: the L1 barrier word at {:#x} of tile ({}, {}) did not \
                     show {flag:#04x}; the host's writes are not reaching the tile",
                    mailbox::BARRIER,
                    self.tile.x(),
                    self.tile.y()
                );
                return Err(ttStatus::TT_ERROR_DEVICE);
            }
        }
        Ok(())
    }

    /// Write into this tile's L1.
    pub fn l1_write(&mut self, address: u64, bytes: &[u8]) -> Result<(), ttStatus> {
        // A direct write invalidates what `stage_words` remembered over this
        // range: the cache may only claim what is actually in L1. A staging
        // range and a later partial write of the same words overlap, so every
        // entry the write touches goes, not just one at the same address.
        self.forget_overlapping(address, bytes.len() as u64);
        self.l1_write_uncached(address, bytes)
    }

    /// Drop every staging-memory entry of this tile whose bytes overlap
    /// `[at, at + len)`.
    fn forget_overlapping(&mut self, at: u64, len: u64) {
        let tile = self.tile;
        self.staged.retain(|&(x, y, address), words| {
            let end = address + 4 * words.len() as u64;
            (x, y) != (tile.x(), tile.y()) || end <= at || at + len <= address
        });
    }

    fn l1_write_uncached(&mut self, address: u64, bytes: &[u8]) -> Result<(), ttStatus> {
        self.check_l1(address, bytes.len())?;
        let tile = self.tile;
        if DMA_ENABLED && bytes.len() >= DMA_MIN && self.dma_write(tile, address, bytes)? {
            return Ok(());
        }
        self.window_write(tile, address, bytes)
    }

    /// Write `words` at `at` unless they are already there, and then write only
    /// the runs that changed.
    ///
    /// The launch path stages the same core info, argument arrays and launch
    /// words on every tile of every launch. Their contents barely change between
    /// launches (an argument array differs only in the tile's own coordinate
    /// word for a fixed grid), and every unchanged dword is a PCIe store that
    /// need not happen. The cache is per tile and address, and a direct
    /// [`Device::l1_write`] to the same address drops the entry.
    pub fn stage_words(&mut self, at: u64, words: &[u32]) -> Result<(), ttStatus> {
        if words.is_empty() {
            return Ok(());
        }
        self.check_l1(at, 4 * words.len())?;
        let tile = self.tile;
        let key = (tile.x(), tile.y(), at);
        let mut runs: Vec<(usize, usize)> = Vec::new();
        match self.staged.get(&key).filter(|w| w.len() == words.len()) {
            Some(cached) => {
                let mut start = None;
                for j in 0..=words.len() {
                    let differs = j < words.len() && cached[j] != words[j];
                    match (differs, start) {
                        (true, None) => start = Some(j),
                        (false, Some(s)) => {
                            runs.push((s, j));
                            start = None;
                        }
                        _ => {}
                    }
                }
            }
            None => runs.push((0, words.len())),
        }
        if runs.is_empty() {
            return Ok(());
        }
        self.staged.remove(&key);
        // A partial write leaves the rest of an overlapping range unknown.
        self.forget_overlapping(at, 4 * words.len() as u64);
        for (start, end) in runs {
            let bytes: Vec<u8> = words[start..end]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            self.l1_write_uncached(at + 4 * start as u64, &bytes)?;
        }
        self.staged.insert(key, words.to_vec());
        Ok(())
    }

    /// The module whose staging is complete on every tile of its grid, if any
    /// (`Module::hash` with the resolved grid).
    pub fn installed(&self) -> Option<u64> {
        self.installed
    }

    /// Record that `fingerprint`'s module is staged on every tile of its grid,
    /// so its next launch can be fanned out on the card.
    pub fn note_installed(&mut self, fingerprint: u64) {
        self.installed = Some(fingerprint);
    }

    /// The tiles no longer hold a complete installation: the next launch has to
    /// stage it from the host again.
    pub fn forget_installed(&mut self) {
        self.installed = None;
    }

    /// Forget what `stage_words` remembered about every tile but `tile`: the
    /// on-card dispatcher writes the workers' argument blocks over the NoC,
    /// while this tile's are the ones the host wrote here.
    pub fn forget_staged_except(&mut self, tile: NocCoord) {
        self.staged
            .retain(|&(x, y, _), _| (x, y) == (tile.x(), tile.y()));
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
    #[cfg(feature = "firmware")]
    pub fn load_and_start(&mut self, core: Core, image: &[u8], at: u64) -> Result<(), ttStatus> {
        self.load_held(core, image, at)?;
        self.set_core_reset(core, false)
    }

    /// Load `image` at `at` and leave `core` held in reset.
    ///
    /// Everything [`Device::load_and_start`] does except the release: the image,
    /// the reset PC and a cleared status word are in place, so one later write to
    /// `SOFT_RESET_0` starts this core and every other held one at once.
    ///
    /// A core with no reset-PC register is hardwired to its reset PC and is
    /// started by one `jal x0, at` written there, which is how tt-metal starts
    /// BRISC (`generate_risc_startup_addr`, `llrt/hal.cpp:126-149`).
    #[cfg(feature = "firmware")]
    fn load_held(&mut self, core: Core, image: &[u8], at: u64) -> Result<(), ttStatus> {
        self.set_core_reset(core, true)?;
        self.l1_write(at, image)?;
        // A stale status word would satisfy the wait before the image runs.
        self.write32(mailbox::STATUS, 0)?;
        let reset_pc =
            Blackhole::reset_pc_register(core).map_err(|_| ttStatus::TT_ERROR_RESET_PC)?;
        match reset_pc {
            Some(pc) => {
                self.write32(pc, at as u32)?;
                let override_ = Blackhole::reset_pc_override(core)
                    .map_err(|_| ttStatus::TT_ERROR_RESET_PC)?
                    .expect("a core with a reset PC has an override");
                let (reg, bit) = override_;
                let current = self.read32(reg)?;
                self.write32(reg, current | bit)?;
            }
            None if at
                == u64::from(
                    Blackhole::default_reset_pc(core).map_err(|_| ttStatus::TT_ERROR_RESET_PC)?,
                ) => {}
            None => {
                let entry = i32::try_from(at).ok().filter(|n| *n < 0x8_0000);
                let Some(entry) = entry else {
                    let _ = at;
                    return Err(ttStatus::TT_ERROR_RESET_PC);
                };
                let jump = crate::riscv::jal(0, entry);
                let entry_pc =
                    Blackhole::default_reset_pc(core).map_err(|_| ttStatus::TT_ERROR_RESET_PC)?;
                self.l1_write(u64::from(entry_pc), &jump.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Release the Tensix backend — unpackers, packers, Matrix Unit, Vector Unit
    /// and the rest — from soft reset, leaving the five RISC-V cores alone.
    ///
    /// Idempotent, so a tile whose backend is already out of reset costs
    /// nothing after the first launch.
    pub fn release_tensix_backend(&mut self) -> Result<(), ttStatus> {
        if self.backend_released.contains(&self.tile) {
            return Ok(());
        }
        let current = self.read32(soft_reset_register())?;
        self.write32(
            soft_reset_register(),
            current & !Blackhole::BACKEND_RESET_MASK,
        )?;
        self.backend_released.push(self.tile);
        Ok(())
    }

    /// Wait until `predicate` accepts the word at `address`.
    ///
    /// On the simulator nothing happens outside [`Device::clock`], so the budget
    /// is simulated cycles. On silicon time passes by itself and the same number
    /// is a wall-clock deadline, floored at [`SILICON_MIN_WAIT`] (`tt-device`
    /// `core_control.rs:489-521`).
    #[cfg(feature = "firmware")]
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
                let elapsed = started.elapsed();
                // tt-metal's cadence: tight for the first millisecond, then a
                // sleep between reads (`poll_until`, `utils.hpp:136-159`).
                if elapsed > POLL_BUSY_WINDOW {
                    std::thread::sleep(POLL_INTERVAL);
                }
                elapsed >= limit
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
    /// The failure path has no `ttReset`: this is the driver's own cleanup.
    /// Every step is attempted even if one fails, and the first failure is
    /// returned, so a wedged tile is reported rather than hidden.
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
        // What was staged is only valid until the cores are reset, and the
        // semaphores are no longer known to hold the setup program's values.
        self.staged.clear();
        self.installed = None;
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
        // A build without the `firmware` feature carries no images to start,
        // so a launch in it is refused instead of leaving held cores behind.
        #[cfg(not(feature = "firmware"))]
        return Err(ttStatus::TT_ERROR_UNSUPPORTED);
        #[cfg(feature = "firmware")]
        {
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
            self.load_and_start(Core::B, EXEC_B, tt_isa::arch::blackhole::dm::IMAGE_BASE)?;
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
            // This boot cleared the tile's mailboxes behind the staging cache's
            // back.
            let tile = self.tile;
            self.staged
                .retain(|&(x, y, _), _| (x, y) != (tile.x(), tile.y()));
            // The tile booted again, so nothing of a module is staged on it.
            self.installed = None;
            self.exec_started.push(self.tile);
            Ok(())
        }
    }

    /// Load NC's image at `at` and leave NC held, with the jump that starts it
    /// at its reset PC.
    #[cfg(feature = "firmware")]
    fn load_held_nc(&mut self, image: &[u8], at: u64) -> Result<(), ttStatus> {
        self.set_core_reset(Core::NC, true)?;
        self.l1_write(at, image)?;
        let stub = tt_isa::arch::blackhole::dm::nc::stub(at);
        let words: Vec<u8> = stub.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.l1_write(tt_isa::arch::blackhole::dm::nc::STUB_AT, &words)?;
        Ok(())
    }

    /// Clear one launch message: the generation first, then the program and
    /// control words.
    #[cfg(feature = "firmware")]
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
        // A program is bounded by the L1 program cache, not by a fixed slot:
        // its address is named by the launch message. tt-metal's bound is the
        // same region (`MEM_MAX_KERNEL_SIZE`, 1497 KiB).
        if 4 * words.len() as u64 > Blackhole::L1_SIZE - u64::from(at) {
            let _ = words.len();
            return Err(ttStatus::TT_ERROR_PROGRAM_TOO_LONG);
        }
        // FNV-1a: enough to skip rewriting an unchanged program, not a
        // correctness check on the words.
        let hash = hash_words(words);
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

    /// Write a launch message without starting it: the text address, its length
    /// and the arguments base.
    ///
    /// The generation and `ACK` are not cleared here: the generation a launch
    /// writes is new, so a stale `ACK` can never equal it, and clearing it is a
    /// store nobody reads. A subordinate is named before the dispatcher writes
    /// the generation that starts it.
    pub fn prepare(
        &mut self,
        mb: mailbox::launch::Mailbox,
        text: u32,
        words: u32,
        args: u32,
    ) -> Result<(), ttStatus> {
        self.stage_words(mb.program_addr(), &[text])?;
        self.stage_words(mb.program_len(), &[words])?;
        self.stage_words(mb.args(), &[args])?;
        // A thread left in `SEMWAIT` by a failed run survives the reset of its
        // core and would block the next program at the same semaphore
        // (`crate::arch::blackhole::mailbox::UNWEDGE`). Silicon has no other way to
        // release it: the runner posts every semaphore that reads zero before it
        // pushes. The simulator models no such stuck thread, and its `SEMPOST`
        // from RISC-V is not part of what it emulates.
        let unwedge = u32::from(matches!(self.link, Link::Driver(_)));
        self.stage_words(mb.unwedge(), &[unwedge])?;
        Ok(())
    }

    /// Start whatever launch message `mb` holds with a fresh generation, and
    /// return it. Device-wide and monotonic, so a core never sees a value equal
    /// to the one it last ran.
    pub fn start(&mut self, mb: mailbox::launch::Mailbox) -> Result<u32, ttStatus> {
        self.launch_seq = self.launch_seq.wrapping_add(1).max(1);
        let generation = self.launch_seq;
        // One store: the generation is the word the core polls, and it is new,
        // so it cannot be mistaken for the run before it.
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
        self.stage_words(mb.enables(), &[enables])?;
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
        // Asked once: the loop runs per poll and must stay two reads.
        let debug = crate::runtime::debug_enabled();
        loop {
            if self.read32(mb.ack())? == generation {
                return Ok(());
            }
            if self.read32(mb.status())? == status::PANICKED {
                if debug {
                    eprintln!(
                        "libskamandros: tile ({}, {}) panicked with code {}",
                        self.tile.x(),
                        self.tile.y(),
                        self.read32(mb.panic_code())?
                    );
                }
                return Err(ttStatus::TT_ERROR_PROGRAM_REFUSED);
            }
            let expired = if self.link.is_simulated() {
                waited >= BUDGET
            } else {
                let elapsed = started.elapsed();
                // As in `wait_for`: tt-metal's busy window, then its interval.
                if elapsed > POLL_BUSY_WINDOW {
                    std::thread::sleep(POLL_INTERVAL);
                }
                elapsed >= limit
            };
            if expired {
                if debug {
                    eprintln!(
                        "libskamandros: tile ({}, {}) reset {:#x} enables {:#x} launch timeout: gen {} ack {} status {} \
                         program {:#x} len {} args {:#x} enables {:#x} desc {:#x?}",
                        self.tile.x(),
                        self.tile.y(),
                        self.read32(soft_reset_register())?,
                        self.read32(mb.enables())?,
                        self.read32(mb.generation())?,
                        self.read32(mb.ack())?,
                        self.read32(mb.status())?,
                        self.read32(mb.program_addr())?,
                        self.read32(mb.program_len())?,
                        self.read32(mb.args())?,
                        self.read32(mb.enables())?,
                        {
                            let mut d = [0u32; 4];
                            for (k, w) in d.iter_mut().enumerate() {
                                *w = self.read32(mb.dump_offset(0, k as u32))?;
                            }
                            d
                        }
                    );
                    // Every core's own launch mailbox: which subordinate is
                    // holding the join, and what it thinks it is doing.
                    for other in Self::launch_mailboxes() {
                        eprintln!(
                            "  mailbox {:#x}: gen {} ack {} status {:#x} panic {}",
                            other.generation(),
                            self.read32(other.generation())?,
                            self.read32(other.ack())?,
                            self.read32(other.status())?,
                            self.read32(other.panic_code())?,
                        );
                    }
                }
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
    fn retarget(&mut self, tile: NocCoord, address: u64, write: bool) -> Result<(), ttStatus> {
        let words = tlb_words(address, tile, write)?;
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
            self.retarget(tile, base, true)?;
            let bar = self.window_bar_offset() + offset;
            check_bar0(bar, here)?;
            self.link
                .bulk_write(BAR0_BASE + bar, &bytes[done..done + here]);
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
            self.retarget(tile, base, false)?;
            let bar = self.window_bar_offset() + offset;
            check_bar0(bar, here)?;
            self.link
                .bulk_read(BAR0_BASE + bar, &mut out[done..done + here]);
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

    fn bar_read32(&self, offset: u64) -> Result<u32, ttStatus> {
        check_bar0(offset, 4)?;
        let mut word = [0u8; 4];
        self.link.mem_read(BAR0_BASE + offset, &mut word);
        Ok(u32::from_le_bytes(word))
    }

    /// Point the window at `base` and confirm the card took it.
    ///
    /// A host store through the window is ordered by the CPU's own fence, but a
    /// DMA transfer is read by the *engine*, which is not behind this core's
    /// stores. UMD allocates its DMA window with `verify_config=true` for this
    /// reason (`pcie_protocol.cpp:548-560`): every configure reads the words
    /// back before the engine is started.
    fn retarget_verified(
        &mut self,
        tile: NocCoord,
        base: u64,
        write: bool,
    ) -> Result<(), ttStatus> {
        self.retarget(tile, base, write)?;
        let words = tlb_words(base, tile, write)?;
        let at = TLB_CONFIG_BASE + 12 * u64::from(self.link.window());
        for attempt in 0..CONFIG_POLLS {
            let landed =
                (0..3).all(|i| self.bar_read32(at + 4 * i).ok() == Some(words[i as usize]));
            if landed {
                return Ok(());
            }
            if attempt + 1 == CONFIG_POLLS {
                break;
            }
        }
        Err(ttStatus::TT_ERROR_DEVICE)
    }

    /// The pinned DMA buffer, allocated on first use.
    fn ensure_dma(&mut self) -> Result<(), ttStatus> {
        if self.dma.is_some() {
            return Ok(());
        }
        let Link::Driver(kmd) = &self.link else {
            return Ok(());
        };
        self.dma = Some(kmd.allocate_dma(DMA_CHUNK)?);
        Ok(())
    }

    /// Copy `bytes` to `address` on `tile` through the PCIe DMA engine.
    ///
    /// Returns `Ok(false)` when there is no engine (the simulator), so the
    /// caller falls back to the window.
    fn dma_write(&mut self, tile: NocCoord, address: u64, bytes: &[u8]) -> Result<bool, ttStatus> {
        self.ensure_dma()?;
        let Some(mut dma) = self.dma.take() else {
            return Ok(false);
        };
        let result = self.dma_write_chunks(tile, address, bytes, &mut dma);
        self.dma = Some(dma);
        result.map(|()| true)
    }

    fn dma_write_chunks(
        &mut self,
        tile: NocCoord,
        address: u64,
        bytes: &[u8],
        dma: &mut crate::kmd::DmaBuffer,
    ) -> Result<(), ttStatus> {
        let mut done = 0usize;
        while done < bytes.len() {
            let at = address + done as u64;
            let base = at & !(WINDOW_SIZE - 1);
            let offset = at - base;
            let room = (WINDOW_SIZE - offset) as usize;
            let here = (bytes.len() - done).min(room).min(dma.size);
            self.retarget_verified(tile, base, true)?;
            // The engine reads the window at its AXI address, which is the
            // window's offset in BAR0 (`architecture_tlbs.cpp:22-36`).
            let window_axi = self.window_bar_offset() + offset;
            let axi = u32::try_from(window_axi).map_err(|_| ttStatus::TT_ERROR_ARGUMENT)?;
            let Link::Driver(kmd) = &self.link else {
                return Err(ttStatus::TT_ERROR_UNSUPPORTED);
            };
            kmd.dma_h2d(axi, &bytes[done..done + here], dma)?;
            done += here;
        }
        Ok(())
    }

    fn check_l1(&self, address: u64, len: usize) -> Result<(), ttStatus> {
        let end = address
            .checked_add(len as u64)
            .ok_or(ttStatus::TT_ERROR_L1_ADDRESS_OVERFLOW)?;
        if end > Blackhole::L1_SIZE {
            let _ = (address, end);
            return Err(ttStatus::TT_ERROR_L1_ACCESS);
        }
        Ok(())
    }

    fn set_core_reset(&mut self, core: Core, held: bool) -> Result<(), ttStatus> {
        let current = self.read32(soft_reset_register())?;
        let updated = if held {
            current | soft_reset_mask(core)
        } else {
            current & !soft_reset_mask(core)
        };
        if updated != current {
            self.write32(soft_reset_register(), updated)?;
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

/// Encode the three dwords of a 2 MiB unicast window on NoC #0.
///
/// This is UMD's `tlb_data` for Blackhole (`blackhole_implementation.hpp:24-43`,
/// `tlb.hpp`) plus the two settings a *transfer* window gets:
/// `TlbWindow::make_tlb_config` (`tlb_window.cpp:43-61`) fills the static
/// virtual channel from `ArchitectureTlbs::get_static_vc(flags)`
/// (`architecture_tlbs.cpp:42-57`: `static_vc = 1`, and the buddy is 1 for a
/// read window and 0 for a write), and `configure` defaults to
/// `IoOrdering::Strict`. A window without a static VC is arbitrated per
/// transaction, which is what made every host store wait for its own round
/// trip.
fn tlb_words(base_address: u64, tile: NocCoord, write: bool) -> Result<[u32; 3], ttStatus> {
    if base_address % WINDOW_SIZE != 0 {
        let _ = base_address;
        return Err(ttStatus::TT_ERROR_WINDOW_ALIGNMENT);
    }
    // Field positions, `HostToDeviceTLBs.md`: local offset 43 bits, then the
    // unicast end coordinate, NoC index, ordering and the static channel.
    const OFFSET_BITS: u32 = 43;
    const X_END: u32 = 43;
    const Y_END: u32 = 49;
    const NOC: u32 = 67;
    const ORDERING: u32 = 70;
    const STATIC_VC: u32 = 73;
    const STATIC_VC_BUDDY: u32 = 75;
    const STATIC_VC_CLASS: u32 = 76;
    /// Strict AXI response ordering (`IoOrdering::Strict`).
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
    place(&mut words, STATIC_VC, 1, 1);
    place(&mut words, STATIC_VC_BUDDY, 1, u64::from(!write));
    place(&mut words, STATIC_VC_CLASS, 2, 0);
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
    #[cfg(feature = "firmware")]
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
                tt_isa::arch::blackhole::l1::roles::BASE[i],
                "role {i} is loaded at its slot"
            );
            assert!(
                bytes.len() as u64 <= tt_isa::arch::blackhole::l1::roles::SLOT,
                "role {i} is {} bytes for a {} byte slot",
                bytes.len(),
                tt_isa::arch::blackhole::l1::roles::SLOT
            );
        }
    }

    #[test]
    #[cfg(feature = "firmware")]
    fn the_executor_is_the_image_the_mailbox_starts() {
        // The executor is linked at T0's reset PC, which `link.x` also uses for the
        // images loaded through a plain reset.
        assert_eq!(
            LOAD_ADDRESS,
            u64::from(Blackhole::default_reset_pc(Core::T0).unwrap())
        );
        assert!(!EXEC_B.is_empty());
    }
}
