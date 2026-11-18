//! On-card launch fan-out: RISCV B of one tile starts every other tile's B
//! across the NoC, so a repeat launch costs the host one store and one poll.
//!
//! tt-metal's dispatcher core does this from `cq_dispatch`: a go-signal
//! multicast wakes the workers (`process_go_signal_mcast_cmd`,
//! `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp:1187`) and each worker's
//! launch message names the text, the length and its arguments. This is the
//! same shape with unicast NoC writes per worker instead of the multicast
//! rectangle -- the write path is `tt_isa::noc::niu::Command::Write`, an
//! L1-to-remote-L1 copy -- and the workers' completion is one NoC atomic
//! increment each rather than tt-metal's completion queue.
//!
//! # The descriptor
//!
//! Everything the fan-out needs is staged in RISCV B's dump block, which its
//! executor never writes: the executor has no `Dst` to dump, and the host
//! stages the same block for the first launch. `src/runtime.rs` holds the
//! matching field numbers (`DESC_*`) and stages the block; the two must be
//! changed together.
//!
//! Per launch, a tile's B, in this order:
//!
//! 1. if it is the dispatcher, resets its completion counter and writes every
//!    worker's two argument blocks and then its generation (`fan_out`);
//! 2. installs the CB interface from its own template, which is what tt-metal's
//!    worker firmware does in `setup_local_cb_read_write_interfaces`, so the
//!    table is not part of the launch message;
//! 3. runs the semaphore setup on T0 if the module has one, as BRISC
//!    initialises the semaphores before it releases the TRISCs;
//! 4. starts its own subordinates (`corpus::run_exec`), joins them and, as the
//!    dispatcher, joins the workers that have reported (`join_workers`);
//! 5. reports to the dispatcher if one started it, and then acknowledges the
//!    host -- in that order, so a report is never still in flight when the host
//!    sees the `ACK` and starts the next launch.

use crate::corpus::fail_in;
use crate::{l1_read32, l1_write32, noc, publish};
use tt_isa::mailbox::launch::Mailbox;
use tt_isa::mailbox::{DUMP_ROW_WORDS, panic_code, status};
use tt_isa::noc::niu::{Command, Endpoint, Niu, TxnId};

/// The fan-out's own transaction ID: B's mover uses [`tt_isa::dm::TXN_B`] and
/// NC's [`tt_isa::dm::TXN_NC`], and no request of one ID can be mistaken for
/// another's when the counters are drained.
const TXN: TxnId = match TxnId::new(5) {
    Some(t) => t,
    None => panic!(),
};

/// Issue one request of the fan-out. NoC #0 alone: alternating the two NoCs
/// measured four times slower on card 0 (fan-out 9.6 us -> 41.0 us at 120
/// tiles), which is a cost of this card's NoC #1 for L1-to-L1 writes, not
/// something to build on.
///
/// `inline(always)`: out of line this costs three times the issue (38.9 us
/// against 9.6 us of fan-out at 120 tiles), because the per-request register
/// writes stop being folded into the loop.
#[inline(always)]
fn issue(cmd: &Command, me: (u8, u8)) -> bool {
    noc::issue(Niu::Noc0, cmd, me, TXN).is_ok()
}

/// Wait until the fan-out's requests have completed.
fn drain() {
    noc::wait(TXN);
}

/// The setup run's generation is the launch's with this bit set, so T0 sees a
/// value different from both the role generation that follows and the one the
/// previous launch acknowledged. The host keeps its generations below the bit.
const SETUP_GENERATION: u32 = 0x8000_0000;

// --- the descriptor's fields, from the start of B's dump block ---------------

/// L1 address of the worker list.
pub const WORKERS: u32 = 0;
/// Words in the worker list.
pub const COUNT: u32 = 1;
/// Words in each argument block.
pub const ARGS_WORDS: u32 = 2;
/// The writer's block, as a byte offset from the arguments base.
pub const WRITER_OFFSET: u32 = 3;
/// Words in the CB interface template.
pub const IFACE_WORDS: u32 = 4;
/// The semaphore setup program's address in the program cache; zero for none.
pub const SETUP_ADDR: u32 = 5;
/// Words in the setup program.
pub const SETUP_WORDS: u32 = 6;
/// This tile's endpoint, packed as [`endpoint`].
pub const SELF: u32 = 7;
/// The dispatcher's endpoint, packed; zero on the dispatcher itself and on a
/// launch the host started on this tile directly.
pub const DISPATCHER: u32 = 8;
/// How many argument blocks the module has: the reader's, plus the writer's
/// when it has one.
pub const BLOCKS: u32 = 9;

// --- the regions of the same block ------------------------------------------

/// The CB interface template each worker copies into its own interface.
pub const IFACE_AT: u32 = 16;
/// Words the template may hold: sixteen rings of eight words.
pub const IFACE_CAPACITY: u32 = 128;
/// The patched argument block the fan-out writes from, one block at a time.
pub const SCRATCH_AT: u32 = 144;
/// Words an argument block may hold (the host refuses the dispatcher path for a
/// wider one).
pub const SCRATCH_CAPACITY: u32 = 64;
/// The word a worker's completion atomic leaves its old value in.
pub const RET_AT: u32 = 15;
/// How many workers have finished this launch's work.
pub const DONE_AT: u32 = 208;

/// The last word of an argument block: this tile's own NoC high word
/// (`x | y << 6`), which `args_for` places there and which differs per worker.
const ARGS_TILE_WORD: u32 = 3;

fn at(mb: Mailbox, k: u32) -> u32 {
    mb.dump_offset(k / DUMP_ROW_WORDS, k % DUMP_ROW_WORDS) as u32
}

fn get(mb: Mailbox, k: u32) -> u32 {
    // SAFETY: fixed word `k` of this tile's own dump block; the host stages it
    // before the generation that starts this run.
    unsafe { l1_read32(u64::from(at(mb, k))) }
}

fn put(mb: Mailbox, k: u32, value: u32) {
    // SAFETY: as `get`, and the block is B's own L1.
    unsafe { l1_write32(u64::from(at(mb, k)), value) };
}

fn endpoint(packed: u32) -> (u8, u8) {
    (packed as u8, (packed >> 8) as u8)
}

/// Install this tile's CB interface from the template the host staged, so a
/// launch does not carry the table. tt-metal's BRISC does the same from the
/// launch message before it releases the TRISCs
/// (`setup_local_cb_read_write_interfaces`); the values are the host's
/// (`riscv::cb::interface`) and are identical for every tile.
pub fn install_iface(mb: Mailbox) {
    let words = get(mb, IFACE_WORDS);
    if words == 0 {
        return;
    }
    if words > IFACE_CAPACITY {
        fail_in(mb, panic_code::EXPLICIT);
    }
    // The arguments' third word is the address the compiler gave the CB
    // interface (`args_for` in `src/runtime.rs`).
    // SAFETY: the arguments word of this tile's mailbox, and word 2 of the
    // block it names.
    let base = unsafe { l1_read32(mb.args()) };
    let interface = unsafe { l1_read32(u64::from(base) + 8) };
    for k in 0..words {
        let value = get(mb, IFACE_AT + k);
        // SAFETY: inside the tile's L1 -- the interface the compiler emitted
        // there, whose width the host staged as `words`.
        unsafe { l1_write32(u64::from(interface) + 4 * u64::from(k), value) };
    }
    publish();
}

/// Run the module's semaphore setup on T0 and wait for it, the way BRISC
/// initialises the semaphores before it releases the TRISCs.
///
/// T0's launch message is borrowed for the run: its program address and length
/// are saved and put back, so the role launch that follows is the host's, and
/// its arguments are already the same base. Only B calls this, before it starts
/// any subordinate.
pub fn run_setup(mb: Mailbox, generation: u32) {
    let program = get(mb, SETUP_ADDR);
    let words = get(mb, SETUP_WORDS);
    if words == 0 {
        return;
    }
    let t0 = Mailbox::of(0);
    // SAFETY: T0's launch message, in the same L1, written by the host before
    // this run's generation.
    let (role_program, role_words, args) = unsafe {
        (
            l1_read32(t0.program_addr()),
            l1_read32(t0.program_len()),
            l1_read32(mb.args()),
        )
    };
    let run = generation | SETUP_GENERATION;
    // SAFETY: as above; the role program is put back before the generation that
    // would run it is written.
    unsafe {
        l1_write32(t0.program_addr(), program);
        l1_write32(t0.program_len(), words);
        l1_write32(t0.args(), args);
        l1_write32(t0.generation(), run);
    }
    publish();
    loop {
        publish();
        // SAFETY: as above.
        let ack = unsafe { l1_read32(t0.ack()) };
        if ack == run {
            break;
        }
        // SAFETY: as above.
        if unsafe { l1_read32(t0.status()) } == status::PANICKED {
            fail_in(mb, panic_code::EXPLICIT);
        }
    }
    // SAFETY: as above.
    unsafe {
        l1_write32(t0.program_addr(), role_program);
        l1_write32(t0.program_len(), role_words);
    }
    publish();
}

/// Fan this launch out to the workers: their two argument blocks, with each
/// worker's own coordinate word patched in, then its generation.
///
/// Returns how many workers were started, which is what [`join_workers`] waits
/// for. The blocks are written before any generation is, and each pass is
/// drained, so no worker can read a half-written block: the generation is the
/// word that lets a worker read its arguments.
pub fn fan_out(mb: Mailbox) -> u32 {
    let count = get(mb, COUNT);
    if count == 0 {
        return 0;
    }
    let list = u64::from(get(mb, WORKERS));
    let args_words = get(mb, ARGS_WORDS);
    if args_words > SCRATCH_CAPACITY {
        fail_in(mb, panic_code::EXPLICIT);
    }
    let writer_offset = get(mb, WRITER_OFFSET);
    let me = endpoint(get(mb, SELF));
    // SAFETY: the arguments word of this tile's mailbox.
    let args_at = unsafe { l1_read32(mb.args()) };
    // This launch's counter starts empty: anything an earlier launch's workers
    // reported is cleared here, before this fan-out can start one.
    put(mb, DONE_AT, 0);
    publish();
    let mut started = 0u32;
    for i in 0..count {
        // SAFETY: word `i` of the worker list the host staged, whose length is
        // `count`.
        let worker = unsafe { l1_read32(list + 4 * u64::from(i)) };
        if worker != get(mb, SELF) {
            started += 1;
        }
    }
    if args_words != 0 {
        for block_index in 0..get(mb, BLOCKS) {
            let block = if block_index == 0 {
                args_at
            } else {
                args_at.wrapping_add(writer_offset)
            };
            // The patched copy: the block's own tile word differs per worker.
            for k in 0..args_words {
                // SAFETY: the block the host staged at `args_at`, and the same
                // number of words of the scratch region.
                let value = unsafe { l1_read32(u64::from(block) + 4 * u64::from(k)) };
                put(mb, SCRATCH_AT + k, value);
            }
            for i in 0..count {
                // SAFETY: as above.
                let worker = unsafe { l1_read32(list + 4 * u64::from(i)) };
                if worker == get(mb, SELF) {
                    continue;
                }
                let (x, y) = endpoint(worker);
                put(mb, SCRATCH_AT + ARGS_TILE_WORD, u32::from(x) | (u32::from(y) << 6));
                publish();
                if !issue(
                    &Command::Write {
                        from_local: at(mb, SCRATCH_AT),
                        to: Endpoint {
                            x,
                            y,
                            addr: block,
                        },
                        len: 4 * args_words,
                    },
                    me,
                ) {
                    fail_in(mb, panic_code::EXPLICIT);
                }
            }
            // Before the scratch region is reused for the next block.
            drain();
        }
    }
    // The generation each worker is to see is the one this launch put in this
    // tile's own mailbox, so the NoC reads it from there: an L1 copy's two
    // addresses must be congruent mod 16, and the mailbox word is both the
    // source and the target offset.
    for i in 0..count {
        // SAFETY: as above.
        let worker = unsafe { l1_read32(list + 4 * u64::from(i)) };
        if worker == get(mb, SELF) {
            continue;
        }
        let (x, y) = endpoint(worker);
        if !issue(
            &Command::Write {
                from_local: Mailbox::b().generation() as u32,
                to: Endpoint {
                    x,
                    y,
                    addr: Mailbox::b().generation() as u32,
                },
                len: 4,
            },
            me,
        ) {
            fail_in(mb, panic_code::EXPLICIT);
        }
    }
    drain();
    started
}

/// Wait until `target` workers have reported this launch complete.
///
/// A worker that panics never reports, so this waits until the host gives up
/// rather than failing at once with the panic: the host's timeout is the error
/// a repeat launch reports for a panicked tile. The host path does better (it
/// polls every tile's status), which is the cost of the tiles not being read
/// over PCIe.
pub fn join_workers(mb: Mailbox, target: u32) {
    if target == 0 {
        return;
    }
    loop {
        publish();
        if get(mb, DONE_AT) >= target {
            return;
        }
    }
}

/// Tell the dispatcher this tile has finished. A no-op on the dispatcher
/// itself and on a launch the host started here directly, which is exactly what
/// a zero [`DISPATCHER`] field says.
pub fn report(mb: Mailbox) {
    let dispatcher = get(mb, DISPATCHER);
    if dispatcher == 0 {
        return;
    }
    let (x, y) = endpoint(dispatcher);
    let me = endpoint(get(mb, SELF));
    // One atomic increment of the dispatcher's counter: the counter is in the
    // dispatcher's L1, at the same offset in every tile's dump block.
    let _ = noc::issue(
        Niu::Noc0,
        &Command::AtomicIncrement {
            to: Endpoint {
                x,
                y,
                addr: at(mb, DONE_AT),
            },
            value: 1,
            ret_local: at(mb, RET_AT),
        },
        me,
        TXN,
    );
    noc::wait(TXN);
}
