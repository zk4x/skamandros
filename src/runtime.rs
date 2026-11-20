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
use tt_isa::arch::blackhole::dram::CHANNEL_BYTES;
use tt_isa::arch::blackhole::mailbox;
use tt_isa::arch::blackhole::noc::NocCoord;
use tt_isa::arch::blackhole::noc::niu::Niu;

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
/// Last byte of the program cache: the top of L1.
const CACHE_END: u32 = tt_isa::arch::blackhole::l1::PROGRAM_CACHE.end as u32;
/// Bytes reserved for one mover's argument array, and for the writer's.
const ARGS_BLOCK: u32 = 0x400;
/// Bytes reserved above the arguments for the fan-out lists.
const GROUPS_BYTES: u32 = 0x2000;
/// The fan-out lists sit at the top of the program cache, at a fixed address:
/// a repeat launch writes only the arguments and the dispatcher's generation,
/// and it must know where they are without rebuilding the program layout.
const GROUPS_AT: u32 = CACHE_END - GROUPS_BYTES;
/// The reader's argument array, and the writer's (its DRAM endpoints are NoC
/// #1's).
const ARGS_WRITER: u32 = GROUPS_AT - ARGS_BLOCK;
const ARGS_AT: u32 = ARGS_WRITER - ARGS_BLOCK;
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

/// Slot 0 and 1: RISCV B's reader and writer. 3..6: T0/T1/T2.
const SLOT_READER: u32 = 0;
const SLOT_WRITER: u32 = 1;
/// The setup program's slot. It is not one of the role slots: T0 runs both, and
/// sharing a slot would evict the role program every launch.
const SLOT_SETUP: u32 = 2;
const ROLE_SLOT: [u32; 3] = [3, 4, 5];
/// The fan-out lists, which the tree in `launch_steps` stages: the root's
/// (every other column leader, then its own column's workers) and one per
/// column leader (its column's other tiles). One packed endpoint per word
/// (`endpoint_of`), `GROUPS_STRIDE` bytes apart.

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

/// This launch's program slots, packed from the bottom of the program cache.
///
/// The address goes through the role's launch message, so a slot need not be
/// fixed or of a fixed size: each program gets the room it actually needs, and
/// the five together may use the whole cache below the arguments. tt-metal has
/// no per-role slot either — a kernel binary is bounded by the L1 program
/// region (`MEM_MAX_KERNEL_SIZE` is 1497 KiB on Blackhole,
/// `dev_mem_map.h:79-80`) and shares L1 with the CBs at run time — so a fixed
/// 32 KiB slot would be a limit of this driver, not of the hardware.
struct Slots {
    at: [u32; 6],
}

impl Slots {
    /// Pack the six programs (reader, writer, setup, T0, T1, T2) in that
    /// order, each at the next 16-byte boundary, and refuse a launch whose
    /// programs would run into the arguments.
    fn new(module: &Module, cores: &[Vec<u32>; 3]) -> Result<Self, ttStatus> {
        let sizes = [
            module.reader.len(),
            module.writer.len(),
            module.setup.len(),
            cores[0].len(),
            cores[1].len(),
            cores[2].len(),
        ];
        let mut at = [0u32; 6];
        let mut next = PROGRAM_AREA;
        for (slot, words) in at.iter_mut().zip(sizes) {
            *slot = next;
            let bytes = u32::try_from(words)
                .ok()
                .and_then(|n| n.checked_mul(4))
                .ok_or(ttStatus::TT_ERROR_PROGRAM_AREA_SIZE)?;
            next = next
                .checked_add(bytes)
                .map(|end| end.div_ceil(16) * 16)
                .ok_or(ttStatus::TT_ERROR_PROGRAM_AREA_SIZE)?;
            if next > ARGS_AT {
                return Err(ttStatus::TT_ERROR_PROGRAM_AREA_SIZE);
            }
        }
        Ok(Self { at })
    }
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
pub fn alloc(bytes: usize, page: usize) -> Result<u64, ttStatus> {
    with_device(|d| d.alloc(bytes, page))
}

/// Reserve a tensor striped across the DRAM channels and return its handle.
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
        // the next launch starts from a known state. No `ttReset` exists; this
        // is the driver's own cleanup.
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
        // The collective tables: the sites every `expand` owns, and the
        // logical-to-physical coordinate map a rectangle or a neighbour is
        // named from. Neither depends on the tile, but both are per-tile
        // addresses staged in L1, so they ride with the launch arguments the
        // movers already read.
        *args = vec![
            grid[0],
            grid[1],
            module.iface,
            me_hi(tile),
            module.mcast_base,
            module.coords,
            grid[0],
            grid[1],
        ];
        for (index, kind) in module.params.iter().enumerate() {
            let word = params[index];
            let channel_bytes = tt_isa::arch::blackhole::dram::CHANNEL_BYTES;
            let crate::compile::ParamKind::Tensor { page } = *kind else {
                args.push(word as u32);
                args.push(0);
                continue;
            };
            // A tensor is a table of channel bases, then the count and the
            // page: exactly what the mover's address arithmetic reads. A
            // buffer that lives in one channel -- an ordinary allocation --
            // is one entry, and the same arithmetic degenerates to
            // `base + page * page_size`, so the program never names a layout.
            let (bases, channels, page): (Vec<u64>, u32, u32) = match device.interleaved_at(word) {
                Some(layout) => (
                    layout.base.iter().map(|base| *base).collect(),
                    layout.base.len() as u32,
                    layout.page,
                ),
                None => (vec![word], 1, page),
            };
            let mut pairs = 0u32;
            for base in bases {
                let flat = u64::from(pairs) * channel_bytes + base;
                args.push((flat % channel_bytes) as u32);
                args.push(device.dram_hi(flat, *niu)?);
                pairs += 1;
            }
            for _ in pairs..crate::mover::MAX_BANKS as u32 {
                args.push(0);
                args.push(0);
            }
            args.push(channels);
            args.push(page);
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
                let at = u64::from(channel.index()) * CHANNEL_BYTES + DRAM_BARRIER;
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
                             channel {} still reads {:#010x}, not {flag:#010x}: the host's writes \
                             are not reaching that channel",
                            channel.index(),
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
    // Which programs this target can run: the card writes the unpacker's
    // tile-descriptor tail, which `ttsim` refuses, so the compiler emitted both
    // sets and the launch picks one (`Module::silicon_cores`).
    let cores = if device.is_simulated() {
        &module.cores
    } else {
        &module.silicon_cores
    };
    let compute = cores.iter().any(|words| !words.is_empty());
    // Where this launch's programs go: packed, at their own sizes.
    let slots = Slots::new(module, cores)?;
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
            device.place(slots.at[SLOT_SETUP as usize], &module.setup)?;
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
            (slots.at[SLOT_SETUP as usize], module.setup.len() as u32)
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
        // The multicast sites' kernel semaphores: every tile starts with the
        // credit and data words at zero and the four-byte semaphore at one.
        // They return to that state after a launch that ran to completion (the
        // sender drains its credits, the receivers clear their data words), so
        // this is staged once with the programs, not rewritten per launch.
        if module.mcast_sites != 0 {
            // `arrive`, `credit`, `ret` start at zero and `one` at one; the
            // rest of the block is the geometry's scratch.
            let site_words = (crate::mir::McastSite::BYTES / 4) as usize;
            let mut words = Vec::with_capacity(module.mcast_sites as usize * site_words);
            for _ in 0..module.mcast_sites {
                let mut site = vec![0u32; site_words];
                site[(crate::mir::McastSite { base: 0 }.one() / 4) as usize] = 1;
                words.extend_from_slice(&site);
            }
            device.stage_words(u64::from(module.mcast_base), &words)?;
            // The logical-to-physical map a rectangle or a neighbour is named
            // from. It is a property of the chip, not of the launch, so it is
            // staged with the programs; a coordinate the grid cannot hold
            // reads zero and is never named by a verified program.
            let mut table = vec![0u32; crate::mir::COORDS_WORDS];
            for y in 0..crate::mir::COORD_ROWS {
                for x in 0..crate::mir::COORD_COLUMNS {
                    let at = (y * crate::mir::COORD_COLUMNS + x) as usize;
                    if let Ok(tile) = device.tile_at(x, y) {
                        table[at] = u32::from(tile.packed());
                    }
                }
            }
            device.stage_words(u64::from(module.coords), &table)?;
        }
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
            device.place(slots.at[SLOT_READER as usize], &module.reader)?;
            device.prepare(
                mailbox::launch::Mailbox::b(),
                slots.at[SLOT_READER as usize],
                module.reader.len() as u32,
                ARGS_AT,
            )?;
        }
        if !module.writer.is_empty() {
            device.place(slots.at[SLOT_WRITER as usize], &module.writer)?;
            device.prepare(
                mailbox::launch::Mailbox::nc(),
                slots.at[SLOT_WRITER as usize],
                module.writer.len() as u32,
                ARGS_WRITER,
            )?;
        }
        for (thread, words) in cores.iter().enumerate() {
            if words.is_empty() {
                continue;
            }
            stage_role(
                device,
                thread,
                slots.at[ROLE_SLOT[thread] as usize],
                words,
                ARGS_AT,
            )?;
        }
        // B's launch message, on every tile: the host names the reader here,
        // and the generation below is the only word a launch writes. The
        // dispatcher never rewrites it, so a tile started over the NoC runs the
        // same message the host staged.
        let mb = mailbox::launch::Mailbox::b();
        let (text, words) = if module.reader.is_empty() {
            (0, 0)
        } else {
            (slots.at[SLOT_READER as usize], module.reader.len() as u32)
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
    at: u32,
    words: &[u32],
    args_at: u32,
) -> Result<(), ttStatus> {
    stage_at(device, thread, at, words, args_at)
}

/// [`stage_role`], at an explicit program-cache address.
fn stage_at(
    device: &mut Device,
    thread: usize,
    text: u32,
    words: &[u32],
    args_at: u32,
) -> Result<(), ttStatus> {
    // The only limit is the program cache: a role's program may use every byte
    // between its address and the launch arguments.
    let room = u64::from(CACHE_END - text);
    if words.is_empty() || 4 * words.len() as u64 > room {
        if std::env::var_os("SKAM_DEBUG").is_some() {
            eprintln!(
                "stage: thread {thread} program is {} words ({:#x} bytes), {room:#x} bytes of room",
                words.len(),
                4 * words.len() as u32
            );
        }
        return Err(ttStatus::TT_ERROR_ROLE_PROGRAM_SIZE);
    }
    let mb = mailbox::launch::Mailbox::of(thread as u32);
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
