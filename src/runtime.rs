// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The runtime: a device, DRAM buffers, host transfers, and a launch.
//!
//! A launch is one dispatch: RISCV B starts the reader (on B), the writer (on
//! NC) and the three compute cores together, then joins them, and the host waits
//! for B. That is tt-metal's shape -- BRISC runs the reader, NCRISC the writer,
//! the TRISCs the compute, all from one launch message -- and it is what lets
//! the movers overlap the compute instead of running before and after it.
//!
//! The reader and the writer use opposite NoCs (tt-metal's `brisc_noc_id = 1 -
//! ncrisc's`), so each is handed its own arguments array: a DRAM address in an
//! argument names that NoC's endpoint of the channel.
//!
//! Every core has a launch message (`mailbox::launch::Mailbox`): the program's
//! address in `l1::PROGRAM_CACHE`, its length, the arguments base and a
//! generation. A program is placed once and later launches only name it; the
//! resident executor on the core runs the generation it sees
//! (tt-metal's launch message and kernel-text offset).

use std::sync::Mutex;

use crate::compile::Module;
use crate::device::{Architecture, Device};
use crate::status::ttStatus;
use tt_isa::arch::blackhole::dram::CHANNEL_BYTES;
use tt_isa::arch::blackhole::mailbox;
use tt_isa::arch::blackhole::noc::NocCoord;
use tt_isa::arch::blackhole::noc::niu::Niu;
use tt_isa::arch::quasar::mailbox as quasar_mailbox;
use tt_isa::tensix::Core;

/// The one open device, if [`init`] has been called.
static DEVICE: Mutex<Option<Device>> = Mutex::new(None);

/// Is the debug output enabled (`SKAM_DEBUG`)? Read once: the launch path must
/// not scan the environment per launch.
pub fn debug_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SKAM_DEBUG").is_some())
}

/// Has the on-card dispatcher been turned off (`SKAM_NO_DISPATCH`), keeping
/// every launch on the host path? Read once, as [`debug_enabled`].
fn dispatch_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("SKAM_NO_DISPATCH").is_some())
}

/// First byte of `l1::PROGRAM_CACHE`, where every program is placed once.
const PROGRAM_AREA: u32 = crate::device::PROGRAM_AREA as u32;
/// Bytes in one program-cache slot: `mailbox::MAX_WORDS` words.
const SLOT_BYTES: u32 = 0x8000;
/// The barrier word of a GDDR channel: offset 0, as tt-metal's
/// `DRAM_BARRIER_BASE` (`bh_hal.cpp:34-37`, `DRAM_BARRIER_SIZE` = 64 bytes).
/// It lies below [`crate::device`]'s first allocation, so a launch never reads
/// a tensor there and no tensor is ever written there.
const DRAM_BARRIER: u64 = 0;

/// The two values tt-metal alternates through the barrier word (`MemBarFlag`,
/// `umd/device/types/cluster_types.hpp:106-109`): a barrier writes one and
/// reads it back, so every earlier write to the channel has been committed,
/// and a value the previous barrier left behind can never be mistaken for this
/// one because every barrier uses the other value first.
const BARRIER_SET: u32 = 0xAA;
const BARRIER_RESET: u32 = 0xBB;

/// How many times the barrier word is read before the channel is declared not
/// to answer. A flush is one or two PCIe round trips; a thousand reads is far
/// past that and still bounded, so a channel that is not answering is reported
/// rather than spun on.
const BARRIER_POLLS: u32 = 1000;

/// Slot 0 and 1: RISCV B's reader and writer. 3..6: T0/T1/T2. The arguments
/// follow the last slot.
const SLOT_READER: u32 = 0;
const SLOT_WRITER: u32 = 1;
/// The setup program's slot. It is not one of the role slots: T0 runs both, and
/// sharing a slot would evict the role program every launch.
const SLOT_SETUP: u32 = 2;
const ROLE_SLOT: [u32; 3] = [3, 4, 5];
pub(crate) const ARGS_AT: u32 = PROGRAM_AREA + 6 * SLOT_BYTES;
/// The writer's array, one kilobyte after the reader's: its DRAM endpoints are
/// NoC #1's.
const ARGS_WRITER: u32 = ARGS_AT + 0x400;

/// The fan-out lists, which the tree in `launch_steps` stages: the root's
/// (every other column leader, then its own column's workers) and one per
/// column leader (its column's other tiles). One packed endpoint per word
/// (`endpoint_of`), `GROUPS_STRIDE` bytes apart.
const GROUPS_AT: u32 = ARGS_WRITER + 0x400;
/// Bytes per fan-out list: sixty-four endpoints of room. The root's is the
/// widest (every other column leader plus its own column), and the slots must
/// not overlap: an 0x40 stride truncated the root's list and started the wrong
/// tiles.
const GROUPS_STRIDE: u32 = 0x100;

// The dispatcher's descriptor, in RISCV B's dump block -- which that core's
// executor never writes, because it has no `Dst` to dump. `firmware/src/
// dispatch.rs` holds the same field numbers and the regions of the block; the
// two must be changed together.
/// L1 address of this tile's fan-out list ([`GROUPS_AT`]).
const DESC_WORKERS: u32 = 0;
/// Words in this tile's fan-out list.
const DESC_COUNT: u32 = 1;
/// Words in each argument block.
const DESC_ARGS_WORDS: u32 = 2;
/// The writer's block, as a byte offset from [`ARGS_AT`].
const DESC_WRITER_OFFSET: u32 = 3;
/// Words in the CB interface template.
const DESC_IFACE_WORDS: u32 = 4;
/// The semaphore setup program's address; zero for none.
const DESC_SETUP_ADDR: u32 = 5;
/// Words in the setup program.
const DESC_SETUP_WORDS: u32 = 6;
/// This tile's packed endpoint.
const DESC_SELF: u32 = 7;
/// The dispatcher's packed endpoint; zero on the dispatcher itself.
const DESC_DISPATCHER: u32 = 8;
/// How many argument blocks the module has: the reader's, plus the writer's
/// when it has one.
const DESC_BLOCKS: u32 = 9;
/// Words in the descriptor itself, before the interface template.
const DESC_FIELDS: u32 = 16;
/// The CB interface template's first word in the same block.
const DESC_IFACE_AT: u32 = 16;
/// Words the template may hold (sixteen rings of eight words).
const DESC_IFACE_CAPACITY: u32 = 128;
/// Words an argument block may hold for the dispatcher path.
const DESC_SCRATCH_CAPACITY: u32 = 64;

/// Word `k` of RISCV B's descriptor, in its mailbox's dump block.
fn descriptor_at(k: u32) -> u64 {
    mailbox::launch::Mailbox::b().dump_offset(k / 16, k % 16)
}

/// A tile's NoC endpoint, packed into one word: the coordinate a NoC request
/// names, not the logical grid coordinate a module sees.
fn endpoint_of(tile: NocCoord) -> u32 {
    u32::from(tile.x()) | (u32::from(tile.y()) << 8)
}

/// `index`'s slot in the program cache.
const fn slot(index: u32) -> u32 {
    PROGRAM_AREA + index * SLOT_BYTES
}

/// Open the device. Idempotent; the simulator is initialized once per process.
///
/// [`Device::open`] chooses the transport the way tt-metal chooses: if
/// `$TT_METAL_SIMULATOR` names a shared object, the simulator is used, and
/// otherwise the card. `TT_HARDWARE=1` is additionally required before this
/// driver will touch silicon — a bring-up gate that goes away once a kernel has
/// run on the card — so with neither variable set, [`init`] refuses rather than
/// guessing which of the two the caller wanted.
pub fn init() -> Result<(), ttStatus> {
    let mut device = lock()?;
    if device.is_none() {
        *device = Some(Device::open()?);
    }
    Ok(())
}

/// Close the device. DRAM allocations from this device are gone; a later
/// [`init`] opens a fresh one over the same simulator.
///
/// On the card, the chip is returned to its idle operating point first, as
/// tt-metal does when it closes a device (`LocalChip::close_device` ->
/// `set_power_state(LONG_IDLE)`, `local_chip.cpp:195-211`). This path only moves
/// the clock: it never resets the board — no `RESET_DEVICE` ioctl, no ARC reset
/// message, no backend pulse. tt-metal's `close_device` also calls
/// `assert_risc_reset(RiscType::ALL)`; that is a soft hold of the five cores, not
/// a board reset, and this driver already has its own holds (`Device::quiesce`
/// at open and `Kmd::hold_on_close` at descriptor close), so nothing here
/// repeats it.
pub fn release() -> Result<(), ttStatus> {
    let mut device = lock()?;
    let result = match device.as_mut() {
        Some(device) => device.set_power_state(false),
        None => Ok(()),
    };
    *device = None;
    result
}

/// Reserve `bytes` of DRAM and return its address.
pub fn alloc(bytes: usize) -> Result<u64, ttStatus> {
    with_device(|d| d.alloc_dram(bytes))
}

/// Reserve a tensor striped across the DRAM channels and return its handle.
pub fn alloc_interleaved(bytes: usize, page: usize) -> Result<u64, ttStatus> {
    with_device(|d| d.alloc_interleaved(bytes, page))
}

/// Copy host bytes into DRAM at `address`.
pub fn h2d(address: u64, bytes: &[u8]) -> Result<(), ttStatus> {
    with_device(|d| d.dram_write(address, bytes))
}

/// Copy DRAM at `address` into host bytes.
pub fn d2h(address: u64, out: &mut [u8]) -> Result<(), ttStatus> {
    with_device(|d| d.dram_read(address, out))
}

/// Run a module over its grid with one launch argument per parameter.
///
/// A tensor parameter's argument is its DRAM address; a scalar parameter's is
/// its value, as CUDA's kernel parameter buffer holds values for scalars and
/// addresses for pointers. A grid extent may be a launch-bound value too, so it
/// is resolved here, before any core starts. The launch arguments are built
/// here, not by the caller: the grid, this tile's coordinate, and each
/// parameter's word(s). A count that disagrees with the module is refused.
/// Every failure is a [`ttStatus`] variant; there is no second error channel
/// and no string error.
pub fn launch(module: &Module, params: &[u64]) -> Result<(), ttStatus> {
    if module.params.len() != params.len() {
        return Err(ttStatus::TT_ERROR_ARGUMENT);
    }
    let grid = resolve_grid(module, params)?;
    if module.reader.is_empty()
        && module.writer.is_empty()
        && module.cores.iter().all(Vec::is_empty)
    {
        return Err(ttStatus::TT_ERROR_UNSUPPORTED);
    }
    with_device(|device| run(device, module, params, grid))
}

fn run(
    device: &mut Device,
    module: &Module,
    params: &[u64],
    grid: [u32; 2],
) -> Result<(), ttStatus> {
    // The two chips launch differently and neither path can stand in for the
    // other: Blackhole runs its own firmware on five cores, Quasar has no such
    // firmware and runs the two movers the module carries for it.
    if device.arch() == Architecture::Quasar {
        let result = quasar_launch(device, module, params, grid);
        if result.is_err() {
            device.quasar_quiesce()?;
        }
        return result;
    }
    let result = launch_steps(device, module, params, grid);
    if result.is_err() {
        // A failed launch is a status, not a wedged tile: stop the cores this
        // launch started and clear the mailboxes it wrote before returning, so
        // the next launch starts from a known state. No `ttReset` exists; this
        // is the driver's own cleanup.
        device.quiesce()?;
    }
    result
}

/// Where a Quasar mover's boot stub lives: the image its reset vector jumps to.
///
/// All eight movers fetch L1 word 0 and every one of them is released the same
/// way, so word 0 holds a jump to one stub and the stub picks the image by
/// `mhartid`. The images are one [`QUASAR_IMAGE_STRIDE`] apart from
/// [`QUASAR_IMAGE_BASE`].
///
/// The stub sits exactly at `MEM_DM_FIRMWARE_BASE`
/// (`tt_isa::arch::quasar::l1::DM_FIRMWARE`), which is where a released mover
/// fetches and is already reserved for mover code. The images are placed at the
/// top of the 4 MiB L1, above every region the launch uses: the CB rings are
/// laid out in the data arena ([`crate::alloc::plan_cbs`], `0x2_0000` up to
/// Blackhole's `mailbox::MAILBOX_BASE`) and the program cache and argument
/// blocks follow it, all below `0x1_8000_0`. One [`QUASAR_IMAGE_STRIDE`] per
/// mover leaves room for the largest program the launch accepts.
const QUASAR_STUB: u32 = 0xF000;
/// The word the unpack hart sets once it has seeded the semaphores, and the
/// math and pack harts spin on before their first one.
///
/// A semaphore nobody has seeded has a maximum of zero, so the two consumers
/// cannot wait on one; the launch stages this word cleared, and the flag is what
/// makes the seed visible before either consumer takes a semaphore. It sits
/// above every region a launch stages (the rings, the program cache, the
/// argument blocks, all below `0x1_8000_0`) and below the movers' images.
pub const QUASAR_READY_AT: u32 = 0x10_0000;
const QUASAR_IMAGE_BASE: u32 = 0x3C_0000;
const QUASAR_IMAGE_STRIDE: u32 = 0x2000;
/// The two movers a launch runs: a tile's movers are 0..=7, and tt-metal keeps
/// 0 and 1 for its own ISR and remapper, so a kernel's movers start at 2.
const QUASAR_READER_HART: u8 = 2;
const QUASAR_WRITER_HART: u8 = 3;
/// The three compute harts of engine 0, in tt-metal's thread order: 0 unpacks,
/// 1 does the math, 2 packs, 3 isolates the SFPU (`ckernel_trisc_id.h:10-14`).
const QUASAR_UNPACK_TRISC: u8 = 0;
const QUASAR_MATH_TRISC: u8 = 1;
const QUASAR_PACK_TRISC: u8 = 2;
/// Where a compute hart's image is staged.
///
/// tt-metal puts a TRISC's firmware at `MEM_TRISC0_FIRMWARE_BASE` with a 5 KiB
/// stride (`dev_mem_map.h:114-117`), but that is the *firmware* slot and an
/// unfolded kernel's program does not fit in it: a launch stages its own image,
/// so it takes its own slots, above the rings and argument blocks (all below
/// `0x1_8000_0`) and below the movers' images at `0x3C_0000`.
const QUASAR_COMPUTE_BASE: u32 = 0x20_0000;
/// 32 KiB per hart: an unfolded lane kernel's program is a few thousand words
/// (gelu's math image is 23 KiB), far past tt-metal's 5 KiB firmware slot.
const QUASAR_TRISC_IMAGE_STRIDE: u32 = 0x8000;

/// The image of the mover with hart id `hart`.
fn quasar_image(hart: u8) -> u32 {
    QUASAR_IMAGE_BASE + QUASAR_IMAGE_STRIDE * u32::from(hart)
}

/// The stub every mover runs first: read the hart id, jump to that hart's image,
/// and hand it the launch's argument block in `a0`.
///
/// A Quasar mover has no launch message of its own: it fetches at L1 word 0,
/// and `quasar_start_at` points that word here. `mhartid` (CSR 0xF14) is the
/// only thing that tells the eight movers apart, and the simulator was measured
/// to return 0..=7 there.
fn quasar_stub() -> Vec<u32> {
    use crate::riscv::reg;
    let base = QUASAR_IMAGE_BASE >> 12;
    let mut words = Vec::with_capacity(6);
    words.push(crate::riscv::csrr(reg::T0, crate::riscv::MHARTID));
    words.push(crate::riscv::slli(
        reg::T0,
        reg::T0,
        QUASAR_IMAGE_STRIDE.trailing_zeros(),
    ));
    words.push(crate::riscv::lui(reg::T1, base));
    words.push(crate::riscv::add(reg::T0, reg::T0, reg::T1));
    // `li a0, ARGS_AT`: the argument block, as the launch's `a0` on every other
    // chip would carry it.
    let high = ARGS_AT.wrapping_add(0x800) >> 12;
    let low = ARGS_AT as i32 - ((high as i32) << 12);
    words.push(crate::riscv::lui(reg::A0, high));
    words.push(crate::riscv::addi(reg::A0, reg::A0, low));
    words.push(crate::riscv::jalr(reg::ZERO, reg::T0, 0));
    words
}

/// Turn a mover program into a self-contained image.
///
/// The rendered program ends in `ret`, because on Blackhole a resident
/// executor calls it. A Quasar mover is not called by anything: the host starts
/// it at its first word, so `ret` would jump to whatever `ra` happens to hold.
/// The last word is therefore replaced by the launch's epilogue — publish the
/// generation, or park — and the hart never returns.
fn quasar_finish(mut words: Vec<u32>, ack: bool) -> Vec<u32> {
    use crate::riscv::reg;
    if words.last() == Some(&crate::riscv::ret()) {
        words.pop();
    }
    if ack {
        words.extend(quasar_ack());
    } else {
        words.push(crate::riscv::jal(reg::ZERO, 0));
    }
    words
}

/// The words that publish the launch generation once a mover's program is done.
///
/// A Quasar launch has no resident runner to do it: the program is an image the
/// host starts, so the epilogue is appended to the image of the mover that
/// finishes last. `a0` and the caller-saved registers are free by then.
fn quasar_ack() -> Vec<u32> {
    use crate::riscv::reg;
    vec![
        crate::riscv::lw(reg::T1, reg::ZERO, quasar_mailbox::GENERATION as i32),
        crate::riscv::sw(reg::T1, reg::ZERO, quasar_mailbox::ACK as i32),
        crate::riscv::jal(reg::ZERO, 0),
    ]
}

/// Run a module on a Quasar: stage the two movers' images and the CB interface,
/// start them, and wait for the one that reports.
///
/// This is the whole launch. There is no dispatch firmware on this chip, no
/// fan-out tree and no resident executor: the host writes the images and the
/// arguments, points the reset vector at the stub, releases the two movers and
/// waits for the generation. A module with datapath work has no Quasar program
/// (its unpacker and packer configuration is still Blackhole's) and is refused
/// rather than run with a program that configures the wrong registers.
fn quasar_launch(
    device: &mut Device,
    module: &Module,
    params: &[u64],
    grid: [u32; 2],
) -> Result<(), ttStatus> {
    let Some(program) = &module.quasar else {
        eprintln!(
            "libskamandros: this module has no Quasar program ({})",
            match module.quasar_refusal {
                Some(ttStatus::TT_ERROR_UNSUPPORTED) =>
                    "a part of its datapath has no Quasar lowering".to_string(),
                Some(reason) => format!("the lowering was refused: {reason:?}"),
                None => "none was compiled".to_string(),
            }
        );
        return Err(ttStatus::TT_ERROR_UNSUPPORTED);
    };
    if program.reader.is_empty() && program.writer.is_empty() {
        return Err(ttStatus::TT_ERROR_EMPTY_MODULE);
    }
    // The logical grid the module asks for, as physical tiles, row-major. Each
    // tile runs the whole program against its own argument block: no dispatch
    // firmware exists to fan the launch out, so the host stages and starts every
    // tile itself.
    let mut tiles = Vec::with_capacity(grid[0] as usize * grid[1] as usize);
    for y in 0..grid[1] {
        for x in 0..grid[0] {
            tiles.push(((x, y), device.tile_at(x, y)?));
        }
    }
    // Stage and start every tile before waiting for any of them: a tile's harts
    // are its own, and the grid is meant to run at once.
    let mut started: Vec<(NocCoord, u32)> = Vec::with_capacity(tiles.len());
    for ((x, y), tile) in &tiles {
        device.select(*tile);
        // This core's logical coordinate: `%x = grid.x` and `%y = grid.y` read
        // it, the same host-written `core_info` Blackhole's dispatch stages.
        device.forget_staged(u64::from(crate::mir::QUASAR_CORE_INFO), 8);
        device.stage_words(u64::from(crate::mir::QUASAR_CORE_INFO), &[*x, *y])?;
        let args = args_for(device, module, params, grid, *tile)?[0].1.clone();
        device.stage_words(u64::from(ARGS_AT), &args)?;
        // The last launch's harts advanced this table's counters and set the
        // ready flag, so the cached words for those ranges are not what is in L1
        // any more.
        device.forget_staged(u64::from(module.iface), 4 * module.iface_words.len() as u64);
        device.forget_staged(u64::from(QUASAR_READY_AT), 4);
        if !module.iface_words.is_empty() {
            device.stage_words(u64::from(module.iface), &module.iface_words)?;
        }
        device.stage_words(u64::from(QUASAR_STUB), &quasar_stub())?;
        device.stage_words(u64::from(QUASAR_READY_AT), &[0])?;
        // The mover that reports: the writer drains what the reader pushed, so
        // it finishes last; with no writer the reader is the whole launch.
        let mut harts: Vec<(u8, u32)> = Vec::new();
        if !program.reader.is_empty() {
            harts.push((QUASAR_READER_HART, quasar_image(QUASAR_READER_HART)));
        }
        if !program.writer.is_empty() {
            harts.push((QUASAR_WRITER_HART, quasar_image(QUASAR_WRITER_HART)));
        }
        for (hart, at) in &harts {
            let words = match *hart {
                h if h == QUASAR_READER_HART => &program.reader,
                _ => &program.writer,
            };
            let ack = harts.last().map(|(h, _)| *h) == Some(*hart);
            let image = quasar_finish(words.clone(), ack);
            if 4 * image.len() as u32 > QUASAR_IMAGE_STRIDE {
                eprintln!(
                    "libskamandros: a mover's image is {} words; a Quasar image slot holds {}",
                    image.len(),
                    QUASAR_IMAGE_STRIDE / 4
                );
                return Err(ttStatus::TT_ERROR_PROGRAM_AREA_SIZE);
            }
            device.stage_words(u64::from(*at), &image)?;
        }
        // The three compute harts, each in its own firmware slot, held in reset
        // until its image is staged (`MEM_TRISC0_FIRMWARE_BASE` and the 5 KiB
        // stride, `dev_mem_map.h:114-117`).
        let mut triscs: Vec<(Core, u32)> = Vec::new();
        for (trisc, words) in [
            (QUASAR_UNPACK_TRISC, &program.unpack),
            (QUASAR_MATH_TRISC, &program.math),
            (QUASAR_PACK_TRISC, &program.pack),
        ] {
            if words.is_empty() {
                continue;
            }
            let core = Core::Math { engine: 0, trisc };
            let at = QUASAR_COMPUTE_BASE + QUASAR_TRISC_IMAGE_STRIDE * u32::from(trisc);
            let image = quasar_compute_image(words);
            if 4 * image.len() as u32 > QUASAR_TRISC_IMAGE_STRIDE {
                eprintln!(
                    "libskamandros: TRISC{trisc}'s image is {} words; a Quasar image slot holds {}",
                    image.len(),
                    QUASAR_TRISC_IMAGE_STRIDE / 4
                );
                return Err(ttStatus::TT_ERROR_PROGRAM_AREA_SIZE);
            }
            device.stage_words(u64::from(at), &image)?;
            triscs.push((core, at));
        }
        for (core, at) in &triscs {
            // A TRISC holds reset with the bit set and starts when it is
            // cleared, so it is held before its reset PC is pointed at the
            // image.
            device.quasar_set_reset(*core, true)?;
            device.quasar_start_at(*core, u64::from(*at))?;
        }
        // The reset vector: word 0 of L1, which every released mover fetches.
        device.quasar_start_at(Core::Dm(QUASAR_READER_HART), u64::from(QUASAR_STUB))?;
        // Everything staged is visible before a hart is released.
        device.l1_barrier()?;
        let generation = device.quasar_start()?;
        for (core, _) in &triscs {
            device.quasar_set_reset(*core, false)?;
        }
        for (hart, _) in &harts {
            device.quasar_set_reset(Core::Dm(*hart), false)?;
        }
        started.push((*tile, generation));
    }
    let mut result = Ok(());
    for (tile, generation) in started {
        device.select(tile);
        let done = device.quasar_await(generation);
        for hart in 0..8u8 {
            let _ = device.quasar_set_reset(Core::Dm(hart), true);
        }
        for trisc in [QUASAR_UNPACK_TRISC, QUASAR_MATH_TRISC, QUASAR_PACK_TRISC] {
            let _ = device.quasar_set_reset(Core::Math { engine: 0, trisc }, true);
        }
        if done.is_err() && result.is_ok() {
            result = done;
        }
    }
    result
}

/// Turn a compute program into an image a Quasar TRISC can be started at.
///
/// `a0` is the launch's argument block, which the movers' stub supplies and a
/// TRISC has no supplier for, so the image starts with `li a0, ARGS_AT`. Its
/// trailing `ret` becomes a park for the same reason it does for a mover: no
/// executor calls this hart.
fn quasar_compute_image(words: &[u32]) -> Vec<u32> {
    use crate::riscv::reg;
    let high = ARGS_AT.wrapping_add(0x800) >> 12;
    let low = ARGS_AT as i32 - ((high as i32) << 12);
    let mut image = Vec::with_capacity(words.len() + 4);
    image.push(crate::riscv::lui(reg::A0, high));
    image.push(crate::riscv::addi(reg::A0, reg::A0, low));
    image.extend_from_slice(words);
    if image.last() == Some(&crate::riscv::ret()) {
        image.pop();
    }
    image.push(crate::riscv::jal(reg::ZERO, 0));
    image
}

/// The grid extent of one axis, from the launch's scalar arguments. Zero, a
/// value beyond the part and a malformed expression are refused.
fn resolve_grid(module: &Module, params: &[u64]) -> Result<[u32; 2], ttStatus> {
    let mut grid = [0u32; 2];
    for (axis, term) in module.grid.iter().enumerate() {
        let value = grid_eval(term, module, params)?;
        grid[axis] = u32::try_from(value)
            .ok()
            .filter(|n| *n > 0)
            .ok_or(ttStatus::TT_ERROR_GRID)?;
    }
    Ok(grid)
}

fn grid_eval(
    term: &crate::compile::GridTerm,
    module: &Module,
    params: &[u64],
) -> Result<i64, ttStatus> {
    use crate::compile::GridTerm;
    use crate::ir::BOp;
    let value = match term {
        GridTerm::Const(n) => i64::from(*n),
        GridTerm::Param(index) => {
            let index = *index as usize;
            if module.params.get(index) != Some(&crate::compile::ParamKind::Scalar) {
                return Err(ttStatus::TT_ERROR_GRID);
            }
            let word = *params.get(index).ok_or(ttStatus::TT_ERROR_ARGUMENT)?;
            // A scalar is an `i32`: its low word, read as one.
            i64::from(word as u32 as i32)
        }
        GridTerm::Binary(op, a, b) => {
            let a = grid_eval(a, module, params)?;
            let b = grid_eval(b, module, params)?;
            match op {
                BOp::Add => a.checked_add(b).ok_or(ttStatus::TT_ERROR_GRID)?,
                BOp::Sub => a.checked_sub(b).ok_or(ttStatus::TT_ERROR_GRID)?,
                BOp::Mul => a.checked_mul(b).ok_or(ttStatus::TT_ERROR_GRID)?,
                BOp::Div => a.checked_div(b).ok_or(ttStatus::TT_ERROR_GRID)?,
                BOp::Mod => a.checked_rem(b).ok_or(ttStatus::TT_ERROR_GRID)?,
                BOp::Pow => a
                    .checked_pow(u32::try_from(b).map_err(|_| ttStatus::TT_ERROR_GRID)?)
                    .ok_or(ttStatus::TT_ERROR_GRID)?,
                BOp::Max => a.max(b),
                BOp::Min => a.min(b),
                BOp::BitAnd => a & b,
                BOp::BitOr => a | b,
                BOp::BitXor => a ^ b,
                BOp::BitShiftLeft => a
                    .checked_shl(u32::try_from(b).map_err(|_| ttStatus::TT_ERROR_GRID)?)
                    .ok_or(ttStatus::TT_ERROR_GRID)?,
                BOp::BitShiftRight => a
                    .checked_shr(u32::try_from(b).map_err(|_| ttStatus::TT_ERROR_GRID)?)
                    .ok_or(ttStatus::TT_ERROR_GRID)?,
                _ => {
                    return Err(ttStatus::TT_ERROR_GRID);
                }
            }
        }
    };
    Ok(value)
}

/// The arguments base of each data mover's array.
fn args_at(niu: Niu) -> u32 {
    if niu == Niu::Noc1 {
        ARGS_WRITER
    } else {
        ARGS_AT
    }
}

/// The two argument arrays for one tile: a DRAM endpoint is a NoC coordinate,
/// and the reader on B and the writer on NC use opposite NoCs (tt-metal's
/// BRISC NoC #0, NCRISC NoC #1). The compute cores read the reader's array;
/// they never name a DRAM endpoint.
fn args_for(
    device: &Device,
    module: &Module,
    params: &[u64],
    grid: [u32; 2],
    tile: NocCoord,
) -> Result<[(Niu, Vec<u32>); 2], ttStatus> {
    let mut arrays = [(Niu::Noc0, Vec::new()), (Niu::Noc1, Vec::new())];
    for (niu, args) in arrays.iter_mut() {
        *args = vec![grid[0], grid[1], module.iface, me_hi(tile)];
        for (index, kind) in module.params.iter().enumerate() {
            let word = params[index];
            // A DRAM address is a channel and an offset inside it, and the two
            // chips' channels are not the same size.
            let channel_bytes = match device.arch() {
                Architecture::Blackhole => tt_isa::arch::blackhole::dram::CHANNEL_BYTES,
                Architecture::Quasar => tt_isa::arch::quasar::dram::CHANNEL_BYTES,
            };
            if *kind == crate::compile::ParamKind::Interleaved {
                // A striped tensor: a pair per channel, then the count and the
                // page, exactly what the mover's address arithmetic reads.
                let layout = device
                    .interleaved_at(word)
                    .ok_or(ttStatus::TT_ERROR_ARGUMENT)?;
                let mut pairs = 0u32;
                for (bank, base) in layout.base.iter().enumerate() {
                    let flat = u64::from(bank as u32) * channel_bytes + base;
                    args.push((flat % channel_bytes) as u32);
                    args.push(device.dram_hi(flat, *niu)?);
                    pairs += 1;
                }
                for _ in pairs..crate::mover::MAX_BANKS as u32 {
                    args.push(0);
                    args.push(0);
                }
                args.push(pairs);
                args.push(layout.page);
                continue;
            }
            let (low, high) = match kind {
                crate::compile::ParamKind::Scalar => (word as u32, 0),
                // A tensor names its channel's endpoint in the high word and
                // the offset inside that channel in the low one, as tt-metal's
                // `get_noc_addr_from_bank_id` does: the endpoint comes from the
                // bank, the address is the bank-local offset
                // (`dataflow_api_addrgen.h:516-533`).
                _ => {
                    let hi = device.dram_hi(word, *niu)?;
                    ((word % channel_bytes) as u32, hi)
                }
            };
            args.push(low);
            args.push(high);
        }
        if (args_at(*niu) as u64) + 4 * args.len() as u64
            > tt_isa::arch::blackhole::l1::PROGRAM_CACHE.end
        {
            let _ = args.len();
            return Err(ttStatus::TT_ERROR_PROGRAM_AREA_SIZE);
        }
    }
    Ok(arrays)
}

fn launch_steps(
    device: &mut Device,
    module: &Module,
    params: &[u64],
    grid: [u32; 2],
) -> Result<(), ttStatus> {
    // The host's writes to DRAM are posted, so a launch that reads them must
    // flush first: tt-metal inserts its DRAM barrier before launching a kernel,
    // "to ensure metadata and data to compute on are committed" and because
    // "writes to device are not blocking unless strict TLB ordering is used"
    // (`Cluster::dram_barrier`, `tt_cluster.cpp:1021-1031`). The barrier is the
    // channel's own word, written and read back twice with different values
    // (`LocalChip::insert_host_to_device_barrier` -> `set_membar_flag`,
    // `local_chip.cpp:472-508`).
    //
    // Every channel, as tt-metal's `dram_barrier` does: it collects all DRAM
    // channels and inserts the barrier on each channel's DRAM core
    // (`tt_cluster.cpp:1201-1208` -> `LocalChip::dram_membar`,
    // `local_chip.cpp:446-476`), because a host write is posted and may land on
    // any bank after the launch reads it. An address is `channel *
    // CHANNEL_BYTES`, so the same word is written on each channel in turn.
    //
    // Only when this process has written DRAM since the last launch: the
    // barrier exists for posted host writes, and a launch with none has nothing
    // to flush. The kernel's own DRAM writes are ordered by its NoC requests.
    if device.dram_dirty() {
        for flag in [BARRIER_SET, BARRIER_RESET] {
            for channel in device.dram_channels() {
                let at = u64::from(channel) * CHANNEL_BYTES + DRAM_BARRIER;
                device.dram_write(at, &flag.to_le_bytes())?;
                let mut back = [0u8; 4];
                let mut polls = 0u32;
                loop {
                    device.dram_read(at, &mut back)?;
                    if u32::from_le_bytes(back) == flag {
                        break;
                    }
                    polls += 1;
                    if polls == BARRIER_POLLS {
                        eprintln!(
                            "libskamandros: the DRAM barrier word at offset {DRAM_BARRIER:#x} of \
                             channel {channel} still reads {:#010x}, not {flag:#010x}: the host's \
                             writes are not reaching that channel",
                            u32::from_le_bytes(back)
                        );
                        return Err(ttStatus::TT_ERROR_DEVICE);
                    }
                }
            }
        }
        device.mark_dram_clean();
    }

    // The logical grid the module asks for, as physical tiles, row-major. Each
    // tile's logical coordinate is what `grid.x`/`grid.y` read.
    let mut tiles = Vec::with_capacity(grid[0] as usize * grid[1] as usize);
    for y in 0..grid[1] {
        for x in 0..grid[0] {
            tiles.push(((x, y), device.tile_at(x, y)?));
        }
    }
    for (name, words) in [("reader", &module.reader), ("writer", &module.writer)] {
        if words.is_empty() {
            continue;
        }
        if 4 * words.len() as u32 > SLOT_BYTES {
            let _ = (name, words.len());
            return Err(ttStatus::TT_ERROR_PROGRAM_AREA_SIZE);
        }
    }
    // Which programs this target can run: the card writes the unpacker's
    // tile-descriptor tail, which `ttsim` refuses, so the compiler emitted both
    // sets and the launch picks one (`Module::silicon_cores`).
    let cores = if device.is_simulated() {
        &module.cores
    } else {
        &module.silicon_cores
    };
    let compute = cores.iter().any(|words| !words.is_empty());
    // Which processors the launch starts. Only RISCV B reads it: it starts the
    // reader's and the writer's cores and the three Tensors threads itself.
    let mut enables = 0u32;
    if !module.reader.is_empty() {
        enables |= mailbox::launch::ENABLE_B;
    }
    if !module.writer.is_empty() {
        enables |= mailbox::launch::ENABLE_NC;
    }
    for (thread, words) in cores.iter().enumerate() {
        if !words.is_empty() {
            enables |= ROLE_ENABLE[thread];
        }
    }

    // The CB interface is host state: tt-metal's
    // `setup_local_cb_read_write_interfaces` writes each `LocalCBInterface`
    // before the roles start, and every mover and compute role reads its ring
    // pointers from there. Bytes, not tt-metal's 16-byte units (`riscv::cb`).
    // The words are identical on every tile, and each tile's B installs them
    // from its own copy before it starts the roles (`firmware/src/dispatch.rs`,
    // tt-metal's worker firmware does the same): a launch carries the template
    // once per module instead of the table per tile per launch, because a role
    // advances the ring pointers in it.
    let interface = &module.iface_words;
    let args_words = module.args_words as usize;
    if 4 * args_words as u64 > u64::from(ARGS_WRITER - ARGS_AT) {
        return Err(ttStatus::TT_ERROR_PROGRAM_AREA_SIZE);
    }
    // The fan-out tree. One core issuing 120 workers' blocks and generations
    // is ~40 us of NoC issues on card 0; two levels are ~10. Each physical
    // column of the grid has a leader, the grid's first tile is the root, and
    // the root starts the other leaders *and* its own column's workers. A tile
    // reports to its parent when its own join is done, so the root's `ACK`
    // still means the whole grid has finished.
    //
    // The lists are packed NoC endpoints (`endpoint_of`): physical coordinates,
    // which is what a NoC request names, not the logical grid a module sees.
    let mut columns: Vec<Vec<u32>> = Vec::new();
    for (_, tile) in &tiles {
        let endpoint = endpoint_of(*tile);
        match columns.iter_mut().find(|column| {
            // Same physical column: both endpoints decode to the same x.
            (column[0] & 0xff) == (endpoint & 0xff)
        }) {
            Some(column) => column.push(endpoint),
            None => columns.push(vec![endpoint]),
        }
    }
    let root = columns[0][0];
    // The root's list: every other leader, then its own column's workers.
    let mut root_list: Vec<u32> = columns.iter().skip(1).map(|column| column[0]).collect();
    root_list.extend(columns[0].iter().copied().filter(|tile| *tile != root));
    let mut lists: Vec<Vec<u32>> = vec![root_list];
    for column in columns.iter().skip(1) {
        lists.push(column[1..].to_vec());
    }
    // Each list's slot, and the parent of every tile, in `tiles` order.
    let mut plan: Vec<(usize, u32)> = Vec::with_capacity(tiles.len());
    for (_, tile) in &tiles {
        let endpoint = endpoint_of(*tile);
        let column = columns
            .iter()
            .position(|column| (column[0] & 0xff) == (endpoint & 0xff))
            .ok_or(ttStatus::TT_ERROR_GRID)?;
        if endpoint == root {
            plan.push((0, 0));
        } else if columns[column][0] == endpoint {
            // A leader: it starts its column and reports to the root.
            plan.push((column, root));
        } else {
            // A worker: started by its column's leader, reports to it.
            plan.push((usize::MAX, columns[column][0]));
        }
    }
    if u64::from(GROUPS_AT) + (lists.len() as u64) * u64::from(GROUPS_STRIDE)
        > tt_isa::arch::blackhole::l1::PROGRAM_CACHE.end
    {
        return Err(ttStatus::TT_ERROR_PROGRAM_AREA_SIZE);
    }
    // The tile whose B fans the next launch out from: the grid's first tile,
    // whose B is resident like every other tile's. It costs no tile of the
    // grid, which is the point -- tt-metal's dispatcher takes a whole column,
    // because its dispatch core is a full-time ring processor and cannot also
    // compute.
    let dispatcher = tiles[0].1;
    // The installation is only reusable while the programs, the rings, the
    // parameters and the resolved grid are all the same.
    let fingerprint = module.hash.rotate_left(17) ^ (u64::from(grid[1]) << 32) ^ u64::from(grid[0]);
    let dispatchable =
        interface.len() as u32 <= DESC_IFACE_CAPACITY && args_words as u32 <= DESC_SCRATCH_CAPACITY;
    // A launch of a module this device has already staged on every tile runs
    // from the dispatcher: the host writes one generation and waits for one
    // `ACK` (`firmware/src/dispatch.rs`). `SKAM_NO_DISPATCH` keeps every launch
    // on the host path, for measuring the two against each other.
    if dispatchable && device.installed() == Some(fingerprint) && !dispatch_disabled() {
        return dispatch_launch(device, module, params, grid, dispatcher);
    }

    // Every tile is staged. Then the grid is started from one tile, which fans
    // the launch out over the NoC to the rest (`firmware/src/dispatch.rs`);
    // a module too wide for the descriptor's block scratch is started on every
    // tile by the host instead.
    let mut launched: Vec<(NocCoord, u32)> = Vec::new();
    for (index, ((x, y), tile)) in tiles.iter().enumerate() {
        device.select(*tile);
        // This core's logical coordinate, host-written like tt-metal's
        // `core_info`: `%x = grid.x` and `%y = grid.y` read it.
        device.stage_words(u64::from(crate::mir::CORE_INFO), &[*x, *y])?;
        // The mover's `USABLE` mailbox: the chip's `ENABLED_GDDR` word, in the
        // compacted logical channel numbering its decoder uses
        // (`tt_isa::arch::blackhole::dm::Descriptor::decode`). The mover reads it once when it
        // starts, so it lands before the executor is released.
        let mask = device.dram_mask();
        device.stage_words(tt_isa::arch::blackhole::dm::USABLE, &[mask])?;
        if compute {
            device.release_tensix_backend()?;
        }
        device.start_executors()?;
        let [args, writer_args] = args_for(device, module, params, grid, *tile)?.map(|(_, v)| v);
        // Both arrays before the first core starts, not only as a side effect of
        // a mover run. Staged: a repeat launch writes only the words that differ.
        device.stage_words(u64::from(ARGS_AT), &args)?;
        device.stage_words(u64::from(ARGS_WRITER), &writer_args)?;
        // The semaphore setup is placed here but not run: the descriptor names
        // it, and B runs it on T0 before it starts the roles, as BRISC
        // initialises the semaphores before it releases the TRISCs. It runs on
        // every launch, not once per tile: a launch does not necessarily leave
        // every semaphore at the value the setup wrote, and re-initialising is
        // what puts the next launch's pacing back (skipping it measured a
        // kernel ~20% slower, as well as relying on an end state no code
        // guarantees).
        if compute && !module.setup.is_empty() {
            device.place(slot(SLOT_SETUP), &module.setup)?;
        }

        // The descriptor: the tiles this one fans out to, the width of the
        // argument blocks, the setup program, and the CB interface template it
        // installs from its own L1. It lives in B's dump block, which its
        // executor never writes (`firmware/src/dispatch.rs` has the same field
        // numbers).
        let (list_slot, parent) = plan[index];
        let list = if list_slot == usize::MAX {
            None
        } else {
            Some(&lists[list_slot])
        };
        let setup = if compute && !module.setup.is_empty() {
            (slot(SLOT_SETUP), module.setup.len() as u32)
        } else {
            (0, 0)
        };
        let mut desc = vec![0u32; DESC_FIELDS as usize];
        desc[DESC_WORKERS as usize] =
            list.map_or(0, |_| GROUPS_AT + list_slot as u32 * GROUPS_STRIDE);
        // Zero when the host starts every tile itself: then no tile fans out,
        // and a launch the host has not finished staging is never started.
        desc[DESC_COUNT as usize] = if dispatchable {
            list.map_or(0, |list| list.len() as u32)
        } else {
            0
        };
        desc[DESC_ARGS_WORDS as usize] = args_words as u32;
        desc[DESC_WRITER_OFFSET as usize] = ARGS_WRITER - ARGS_AT;
        desc[DESC_IFACE_WORDS as usize] = interface.len() as u32;
        desc[DESC_SETUP_ADDR as usize] = setup.0;
        desc[DESC_SETUP_WORDS as usize] = setup.1;
        desc[DESC_BLOCKS as usize] = 1 + u32::from(!module.writer.is_empty());
        desc[DESC_SELF as usize] = endpoint_of(*tile);
        // Zero on the root: it reports to nobody, and a tile that reports
        // increments its parent's completion counter.
        desc[DESC_DISPATCHER as usize] = parent;
        device.stage_words(descriptor_at(0), &desc)?;
        device.stage_words(descriptor_at(DESC_IFACE_AT), interface)?;
        if let Some(list) = list {
            device.stage_words(
                u64::from(GROUPS_AT + list_slot as u32 * GROUPS_STRIDE),
                list,
            )?;
        }

        // Name every program of the launch. B is dispatched once and starts
        // the enabled cores itself; the reader on B and the writer on NC
        // therefore run with the compute cores, not before or after them.
        if !module.reader.is_empty() {
            device.place(slot(SLOT_READER), &module.reader)?;
            device.prepare(
                mailbox::launch::Mailbox::b(),
                slot(SLOT_READER),
                module.reader.len() as u32,
                ARGS_AT,
            )?;
        }
        if !module.writer.is_empty() {
            device.place(slot(SLOT_WRITER), &module.writer)?;
            device.prepare(
                mailbox::launch::Mailbox::nc(),
                slot(SLOT_WRITER),
                module.writer.len() as u32,
                ARGS_WRITER,
            )?;
        }
        for (thread, words) in cores.iter().enumerate() {
            if words.is_empty() {
                continue;
            }
            stage_role(device, thread, words, ARGS_AT)?;
        }
        // B's launch message, on every tile: the host names the reader here,
        // and the generation below is the only word a launch writes. The
        // dispatcher never rewrites it, so a tile started over the NoC runs the
        // same message the host staged.
        let mb = mailbox::launch::Mailbox::b();
        let (text, words) = if module.reader.is_empty() {
            (0, 0)
        } else {
            (slot(SLOT_READER), module.reader.len() as u32)
        };
        device.prepare(mb, text, words, ARGS_AT)?;
        device.stage_words(mb.enables(), &[enables])?;
        if !dispatchable {
            // Every program and launch word of this tile is staged; the
            // generation below is the store that lets a core read them, so the
            // tile is made to show them first (`l1_membar`, as before the setup
            // run).
            device.l1_barrier()?;
            let generation = device.start(mb)?;
            launched.push((*tile, generation));
        }
    }
    // Every tile is staged: start the grid from the root, or from every tile
    // when the module does not fit the dispatcher's descriptor, then join it.
    let mb = mailbox::launch::Mailbox::b();
    if dispatchable {
        device.select(dispatcher);
        if debug_enabled() {
            eprintln!(
                "libskamandros: staging launch from root ({}, {}): {} lists, dispatchable {}",
                dispatcher.x(),
                dispatcher.y(),
                lists.len(),
                dispatchable
            );
        }
        // Every program, argument word and descriptor of every tile is staged;
        // the generation is the store that lets B read this tile's, so the tile
        // is made to show them first (`l1_membar`).
        device.l1_barrier()?;
        let generation = device.start(mb)?;
        device.await_launch(mb, generation)?;
    } else {
        for (tile, generation) in launched {
            device.select(tile);
            device.await_launch(mb, generation)?;
        }
    }
    // A launch of this module can now be fanned out on the card: every tile has
    // its programs, arguments, interface template and descriptor.
    if dispatchable {
        device.note_installed(fingerprint);
    } else {
        device.forget_installed();
    }
    Ok(())
}

/// A repeat launch of a module this device has already staged: one generation
/// to the dispatcher tile's B, one `ACK` back.
///
/// The host writes only that tile's argument blocks; the dispatcher patches
/// each worker's own coordinate word in, writes the blocks over the NoC and
/// then the worker's generation, and joins them all before it acknowledges
/// (`firmware/src/dispatch.rs`). Nothing else changes between two launches of
/// the same module: the programs, the roles' launch messages, the CB interface
/// template and the descriptor are resident.
fn dispatch_launch(
    device: &mut Device,
    module: &Module,
    params: &[u64],
    grid: [u32; 2],
    dispatcher: NocCoord,
) -> Result<(), ttStatus> {
    device.select(dispatcher);
    if debug_enabled() {
        eprintln!(
            "libskamandros: launch fanned out from tile ({}, {}) over {} tiles",
            dispatcher.x(),
            dispatcher.y(),
            grid[0] * grid[1]
        );
    }
    let [args, writer_args] = args_for(device, module, params, grid, dispatcher)?.map(|(_, v)| v);
    device.stage_words(u64::from(ARGS_AT), &args)?;
    device.stage_words(u64::from(ARGS_WRITER), &writer_args)?;
    // The generation is the store that lets B read what the host staged.
    device.l1_barrier()?;
    let mb = mailbox::launch::Mailbox::b();
    let generation = device.start(mb)?;
    device.await_launch(mb, generation)?;
    // The workers' argument blocks were written by the dispatcher, behind the
    // staging cache's back, so what it claims to have staged is no longer
    // guaranteed to be in L1. The dispatcher's own tile is the exception: its
    // blocks are the ones written here.
    device.forget_staged_except(dispatcher);
    Ok(())
}

/// The enables bit of each role, in thread order.
const ROLE_ENABLE: [u32; 3] = [
    mailbox::launch::ENABLE_T0,
    mailbox::launch::ENABLE_T1,
    mailbox::launch::ENABLE_T2,
];

/// Place `words` in `thread`'s launch program slot and name it in that role's
/// launch message, with no generation: RISCV B writes the one that starts it.
fn stage_role(
    device: &mut Device,
    thread: usize,
    words: &[u32],
    args_at: u32,
) -> Result<(), ttStatus> {
    stage_at(device, thread, ROLE_SLOT[thread], words, args_at)
}

/// [`stage_role`], in an explicit program-cache slot: the setup program runs on
/// T0 too, but out of a slot of its own.
fn stage_at(
    device: &mut Device,
    thread: usize,
    slot_index: u32,
    words: &[u32],
    args_at: u32,
) -> Result<(), ttStatus> {
    if words.is_empty() || words.len() as u32 > mailbox::MAX_WORDS {
        let _ = (thread, words.len());
        return Err(ttStatus::TT_ERROR_ROLE_PROGRAM_SIZE);
    }
    if 4 * words.len() as u32 > SLOT_BYTES {
        let _ = (thread, words.len());
        return Err(ttStatus::TT_ERROR_PROGRAM_AREA_SIZE);
    }
    let mb = mailbox::launch::Mailbox::of(thread as u32);
    let text = slot(slot_index);
    device.place(text, words)?;
    // The role's configuration words are host state the executor never
    // touches, so staging them costs nothing on a repeat launch.
    for (at, value) in [
        (mb.thread_index(), thread as u32),
        (mb.dst_access_fmt(), 0),
        (mb.dump_row_first(), 0),
        (mb.dump_row_count(), 0),
        (mb.trace(), 0),
        (mb.push_window(), 0),
        (mb.unwedge(), 0),
        (mb.mop_cfg_valid(), 0),
    ] {
        device.stage_words(at, &[value])?;
    }
    // `status` is written by the executor at start and finish, so clearing it
    // is a direct write the staging cache must not remember.
    device.write32(mb.status(), 0)?;
    device.prepare(mb, text, words.len() as u32, args_at)
}

/// This tile's NOC address-high word: the coordinate packed as `x | y << 6`.
fn me_hi(tile: NocCoord) -> u32 {
    (tile.x() as u32 & 0x3F) | ((tile.y() as u32 & 0x3F) << 6)
}

fn lock() -> Result<std::sync::MutexGuard<'static, Option<Device>>, ttStatus> {
    DEVICE.lock().map_err(|_| ttStatus::TT_ERROR_DEVICE_LOCK)
}

fn with_device<T>(f: impl FnOnce(&mut Device) -> Result<T, ttStatus>) -> Result<T, ttStatus> {
    let mut device = lock()?;
    let device = device.as_mut().ok_or(ttStatus::TT_ERROR_NO_DEVICE)?;
    f(device)
}
