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
use crate::device::Device;
use crate::status::ttStatus;
use tt_isa::mailbox;
use tt_isa::noc::NocCoord;
use tt_isa::noc::niu::Niu;

/// The one open device, if [`init`] has been called.
static DEVICE: Mutex<Option<Device>> = Mutex::new(None);

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
const ROLE_SLOT: [u32; 3] = [3, 4, 5];
const ARGS_AT: u32 = PROGRAM_AREA + 6 * SLOT_BYTES;
/// The writer's array, one kilobyte after the reader's: its DRAM endpoints are
/// NoC #1's.
const ARGS_WRITER: u32 = ARGS_AT + 0x400;

/// `index`'s slot in the program cache.
const fn slot(index: u32) -> u32 {
    PROGRAM_AREA + index * SLOT_BYTES
}

/// Open the device. Idempotent; the simulator is initialized once per process.
///
/// The transport is chosen the way tt-metal chooses: if `$TT_METAL_SIMULATOR`
/// names a shared object, the simulator is used, and otherwise the card
/// (`Device::open_hardware`). `TT_HARDWARE=1` is additionally required before this
/// driver will touch silicon — a bring-up gate that goes away once a kernel has
/// run on the card — so with neither variable set, [`init`] refuses rather than
/// guessing which of the two the caller wanted.
pub fn init() -> Result<(), ttStatus> {
    let mut device = lock()?;
    if device.is_none() {
        let simulator = std::env::var_os(crate::device::LIB_PATH_ENV).filter(|p| !p.is_empty());
        *device = Some(match simulator {
            Some(_) => Device::open()?,
            None => Device::open_hardware()?,
        });
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
    let result = launch_steps(device, module, params, grid);
    if result.is_err() {
        // A failed launch is a status, not a wedged tile: stop the cores this
        // launch started and clear the mailboxes it wrote before returning, so
        // the next launch starts from a known state (user requirement). No
        // `ttReset` exists; this is the driver's own cleanup.
        device.quiesce()?;
    }
    result
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
        let hi = device.dram_hi_for(*niu);
        for (index, kind) in module.params.iter().enumerate() {
            let word = params[index];
            let (low, high) = match kind {
                crate::compile::ParamKind::Scalar => (word as u32, 0),
                crate::compile::ParamKind::Tensor => (
                    u32::try_from(word).map_err(|_| ttStatus::TT_ERROR_PARAMETER_ADDRESS)?,
                    hi,
                ),
            };
            args.push(low);
            args.push(high);
        }
        if (args_at(*niu) as u64) + 4 * args.len() as u64 > tt_isa::l1::PROGRAM_CACHE.end {
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
    // Channel 0 only: every allocation this driver hands out is an offset in
    // channel 0 (`Device::alloc_dram`), so one channel's barrier covers every
    // byte a launch can read. A chip-wide barrier would have to visit all
    // seven survivors.
    for flag in [BARRIER_SET, BARRIER_RESET] {
        device.dram_write(DRAM_BARRIER, &flag.to_le_bytes())?;
        let mut back = [0u8; 4];
        let mut polls = 0u32;
        loop {
            device.dram_read(DRAM_BARRIER, &mut back)?;
            if u32::from_le_bytes(back) == flag {
                break;
            }
            polls += 1;
            if polls == BARRIER_POLLS {
                eprintln!(
                    "libskamandros: the DRAM barrier word at offset {DRAM_BARRIER:#x} of channel 0 \
                     still reads {:#010x}, not {flag:#010x}: the host's writes are not reaching DRAM",
                    u32::from_le_bytes(back)
                );
                return Err(ttStatus::TT_ERROR_DEVICE);
            }
        }
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

    // Every tile is staged, then all of them are started and joined, so the
    // grid runs concurrently rather than one core at a time.
    let mut launched: Vec<(NocCoord, u32)> = Vec::new();
    for ((x, y), tile) in &tiles {
        device.select(*tile);
        // The CB interface is host state: tt-metal's
        // `setup_local_cb_read_write_interfaces` writes each
        // `LocalCBInterface` before the roles start, and every mover and
        // compute role reads its ring pointers from there. Bytes, not
        // tt-metal's 16-byte units (`riscv::cb`).
        if !module.cbs.is_empty() {
            let mut words = Vec::with_capacity(module.cbs.len() * 8);
            for ring in &module.cbs {
                words.extend_from_slice(&crate::riscv::cb::interface(
                    ring.base,
                    ring.page_bytes.checked_mul(ring.len).unwrap_or(0),
                    ring.page_bytes,
                    ring.len,
                )?);
            }
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            device.l1_write(u64::from(module.iface), &bytes)?;
        }
        // This core's logical coordinate, host-written like tt-metal's
        // `core_info`: `%x = grid.x` and `%y = grid.y` read it.
        let mut info = Vec::with_capacity(8);
        info.extend_from_slice(&x.to_le_bytes());
        info.extend_from_slice(&y.to_le_bytes());
        device.l1_write(u64::from(crate::mir::CORE_INFO), &info)?;
        // The mover's `USABLE` mailbox: the chip's `ENABLED_GDDR` word, in the
        // compacted logical channel numbering its decoder uses
        // (`tt_isa::dm::Descriptor::decode`). The mover reads it once when it
        // starts, so it lands before the executor is released.
        let mask = device.dram_mask();
        device.write32(tt_isa::dm::USABLE, mask)?;
        if compute {
            device.release_tensix_backend()?;
        }
        device.start_executors()?;
        let [args, writer_args] = args_for(device, module, params, grid, *tile)?.map(|(_, v)| v);
        // Write both arrays before the first core starts, not only as a side
        // effect of a mover run.
        for (at, words) in [(ARGS_AT, &args), (ARGS_WRITER, &writer_args)] {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            device.l1_write(u64::from(at), &bytes)?;
        }
        // The semaphore and unpacker setup runs on T0 first and to completion,
        // as BRISC initialises the semaphores before it releases the TRISCs. A
        // launch with no compute program has no semaphores to set up.
        if compute && !module.setup.is_empty() {
            stage_role(device, 0, &module.setup, ARGS_AT)?;
            let mb = mailbox::launch::Mailbox::of(0);
            let generation = device.start(mb)?;
            device.await_launch(mb, generation)?;
        }

        // Name every program of the launch. B is dispatched once and starts
        // the enabled cores itself; the reader on B and the writer on NC
        // therefore run with the compute cores, not before or after them.
        let mut enables = 0u32;
        if !module.reader.is_empty() {
            device.place(slot(SLOT_READER), &module.reader)?;
            device.prepare(
                mailbox::launch::Mailbox::b(),
                slot(SLOT_READER),
                module.reader.len() as u32,
                ARGS_AT,
            )?;
            enables |= mailbox::launch::ENABLE_B;
        }
        if !module.writer.is_empty() {
            device.place(slot(SLOT_WRITER), &module.writer)?;
            device.prepare(
                mailbox::launch::Mailbox::nc(),
                slot(SLOT_WRITER),
                module.writer.len() as u32,
                ARGS_WRITER,
            )?;
            enables |= mailbox::launch::ENABLE_NC;
        }
        for (thread, words) in cores.iter().enumerate() {
            if words.is_empty() {
                continue;
            }
            stage_role(device, thread, words, ARGS_AT)?;
            enables |= ROLE_ENABLE[thread];
        }
        let mb = mailbox::launch::Mailbox::b();
        let (text, words) = if module.reader.is_empty() {
            (0, 0)
        } else {
            (slot(SLOT_READER), module.reader.len() as u32)
        };
        let generation = device.launch(mb, text, words, ARGS_AT, enables)?;
        launched.push((*tile, generation));
    }
    // Every tile is running; join them all.
    let mb = mailbox::launch::Mailbox::b();
    for (tile, generation) in launched {
        device.select(tile);
        device.await_launch(mb, generation)?;
    }
    Ok(())
}

/// The enables bit of each role, in thread order.
const ROLE_ENABLE: [u32; 3] = [
    mailbox::launch::ENABLE_T0,
    mailbox::launch::ENABLE_T1,
    mailbox::launch::ENABLE_T2,
];

/// Place `words` at `thread`'s fixed cache slot and name it in that role's
/// launch message, with no generation: RISCV B writes the one that starts it.
fn stage_role(
    device: &mut Device,
    thread: usize,
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
    let text = slot(ROLE_SLOT[thread]);
    device.place(text, words)?;
    device.write32(mb.thread_index(), thread as u32)?;
    device.write32(mb.dst_access_fmt(), 0)?;
    device.write32(mb.dump_row_first(), 0)?;
    device.write32(mb.dump_row_count(), 0)?;
    device.write32(mb.trace(), 0)?;
    device.write32(mb.push_window(), 0)?;
    device.write32(mb.unwedge(), 0)?;
    device.write32(mb.mop_cfg_valid(), 0)?;
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
