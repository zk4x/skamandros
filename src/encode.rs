// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! Machine IR to Tensix instructions.
//!
//! One [`MirOp`] is one or two instructions: a lane-register load is one
//! `SFPLOAD`, a hand-off is `SEMWAIT`+`SEMGET` or a drain wait + `SEMPOST`, and a
//! [`MirOp::Repeat`] is expanded into its body (the replay encoding is a later
//! step; the expansion is the same program, only longer).
//!
//! The unpacker and packer configuration is emitted here, whole, per unit: the
//! `ThreadConfig` entries, the `Config` words, the address counters and the
//! `UNPACR`/`PACR` itself. Nothing above this module names a descriptor, a
//! stride or a `Dst` address.

use crate::ir::DType;
use crate::mir::{
    Bcast, BinarySfpOp, Cb, Core, Dst, ElwOp, IOp, MachineKernel, MirOp, PackMask, UnarySfpOp,
};
use crate::riscv::{self as rv, Asm, reg};
use crate::status::ttStatus;
use tt_isa::arch::Arch;
use tt_isa::arch::blackhole::Blackhole;
use tt_isa::arch::blackhole::backend::{
    self, Before, ConfigWords, ThreadConfigEntry, block, cond, wait_for_packer, wait_for_unpacker0,
};
use tt_isa::arch::blackhole::noc::niu::Niu;
use tt_isa::arch::blackhole::sync::{self, Semaphore};
use tt_isa::cfg::blackhole::{alu, global, pack0, thcon, thread, unpack0, unpack1};
use tt_isa::isa::Instruction;
use tt_isa::isa::blackhole::{defs, encode};

/// The GPR the configuration program stages each word through.
pub const SCRATCH_GPR: u32 = 8;
/// Bytes in one row of FP32 `Dst`: the packer's input `Ystride` per row.
pub const DST_ROW_BYTES: u32 = 16 * 4;
/// The `AddrMod` entry that steps the packer's input Y by four rows.
pub const PACK_ADDR_MOD_NEXT_GROUP: u32 = 1;

/// One entry of a role program: a Tensix instruction (emitted as `.ttinsn`) or
/// a RISC-V instruction (emitted as itself).
///
/// A role is RISC-V code with the Tensix words inline, so a program that has to
/// read a ring pointer or run a CB sync mixes the two in one stream.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Elem {
    Tensix(Instruction),
    Riscv(u32),
}

/// A role's program under construction.
#[derive(Debug, Clone, Default)]
pub struct Stream {
    pub(crate) elems: Vec<Elem>,
    /// The callee-saved registers this program uses: the third loop level's
    /// `s0`/`s1`, and the mover's request temporaries. The executor calls a
    /// program as a function and keeps its own frame in that bank, so a program
    /// that uses any of them saves and restores them around its body, exactly
    /// as a compiled callee would.
    saved: Vec<u8>,
}

impl Stream {
    /// Take the callee-saved pair from the third loop level on, so [`words`]
    /// brackets the program with the save and the restore.
    fn save_if_callee(&mut self, pair: (u8, u8), depth: usize) {
        if depth >= 2 {
            self.keep_callee(&[pair.0, pair.1]);
        }
    }

    /// Declare callee-saved registers the program's temporaries use, so the
    /// bracket saves and restores them.
    fn keep_callee(&mut self, regs: &[u8]) {
        for reg in regs {
            if !self.saved.contains(reg) {
                self.saved.push(*reg);
            }
        }
    }

    fn push(&mut self, instruction: Instruction) {
        self.elems.push(Elem::Tensix(instruction));
    }

    fn extend(&mut self, instructions: impl IntoIterator<Item = Instruction>) {
        for instruction in instructions {
            self.push(instruction);
        }
    }

    /// Append one already-encoded RISC-V word.
    fn rv(&mut self, word: u32) {
        self.elems.push(Elem::Riscv(word));
    }

    /// Append an assembled RISC-V sequence.
    fn asm(
        &mut self,
        build: impl FnOnce(&mut Asm) -> Result<(), ttStatus>,
    ) -> Result<(), ttStatus> {
        let mut a = Asm::default();
        build(&mut a)?;
        for word in a.finish().map_err(|_| ttStatus::TT_ERROR_MOVER_PROGRAM)? {
            self.rv(word);
        }
        Ok(())
    }

    /// Words appended so far. One elem is one RISC-V word, Tensix or not, so a
    /// branch distance is a difference of these.
    fn len(&self) -> usize {
        self.elems.len()
    }

    /// Replace an already-appended word. Used to close a branch whose target
    /// is only known later (a loop's forward skip).
    fn patch(&mut self, at: usize, word: u32) {
        self.elems[at] = Elem::Riscv(word);
    }

    /// `li rd, value`: the same two words [`Asm::li`] emits.
    fn li(&mut self, rd: u8, value: u32) {
        let imm = value as i32;
        let hi = (imm.wrapping_add(0x800) >> 12) as u32;
        let lo = imm - (hi as i32) * 4096;
        self.rv(rv::lui(rd, hi & 0xf_ffff));
        self.rv(rv::addi(rd, rd, lo));
    }

    /// The program as the core executes it: the ABI bracket a third loop level
    /// needs, the ops, the closing restore and `ret`. One list, so the debug
    /// listing and the encoded words can never show different programs.
    pub(crate) fn rendered(&self) -> Vec<Elem> {
        let mut out = Vec::with_capacity(self.elems.len() + 8);
        // The save goes in front and the restore at the end, where every
        // forward skip already lands: branches were patched with relative
        // distances, so shifting the whole body stays correct.
        out.extend(
            crate::riscv::callee_save(&self.saved)
                .into_iter()
                .map(Elem::Riscv),
        );
        out.extend_from_slice(&self.elems);
        out.extend(
            crate::riscv::callee_restore(&self.saved)
                .into_iter()
                .map(Elem::Riscv),
        );
        out.push(Elem::Riscv(crate::riscv::ret()));
        out
    }

    /// The encoded words, ending in `ret`. An empty program stays empty: it is
    /// not started, and a lone `ret` would make it look non-empty.
    fn words(&self) -> Result<Vec<u32>, ttStatus> {
        if self.elems.is_empty() {
            return Ok(Vec::new());
        }
        self.rendered()
            .into_iter()
            .map(|elem| match elem {
                Elem::Tensix(i) => i.ttinsn_word().ok_or_else(|| {
                    let _ = i;
                    ttStatus::TT_ERROR_INSTRUCTION_NOT_INLINE
                }),
                Elem::Riscv(w) => Ok(w),
            })
            .collect()
    }
}

/// One open [`MirOp::Loop`]: the register that counts up from zero, the register
/// holding the limit, the word the zero-trip skip is patched into, and the word
/// the body starts at.
struct OpenLoop {
    index: u8,
    limit: u8,
    top: usize,
    skip: usize,
}

/// A loop's `(index, limit)` pair per nesting level, shared with the lowering so
/// the two can never disagree: the compute roles count in `t4`/`t5`, then
/// `a7`/`a1`, then `s0`/`s1`; the movers in `a1`/`a2`, `a3`/`a4`, `s0`/`s1`
/// (their `t5` is the NIU base).
use crate::lower::{COMPUTE_PAIRS, MOVER_PAIRS};

/// Open a loop: `reg = count; if reg == 0 goto end; top:`.
///
/// The skip's target is the end, which is not known until [`close_loop`], so
/// that one word is patched there. One caller-saved register per nesting level:
/// `T4` then `T5`. A deeper loop is a concrete future item, not a guess at a
/// register.
fn open_loop(out: &mut Stream, loops: &mut Vec<OpenLoop>, count: u32) -> Result<(), ttStatus> {
    open_loop_with(out, loops, count, &COMPUTE_PAIRS)
}

/// [`open_loop`] with an explicit counter set: the movers keep `t5` for their
/// NIU base, so their counters are `a1..a4`.
fn open_loop_with(
    out: &mut Stream,
    loops: &mut Vec<OpenLoop>,
    count: u32,
    pairs: &[(u8, u8)],
) -> Result<(), ttStatus> {
    let depth = loops.len();
    let (index, limit) = *pairs.get(depth).ok_or(ttStatus::TT_TODO_LOOP_DEPTH)?;
    out.save_if_callee((index, limit), depth);
    out.li(index, 0);
    out.li(limit, count);
    let skip = out.len();
    // Both edges are long branches: a body is not bounded by the +-4 KiB a
    // B-type branch reaches (`exp2`'s lane sequence is several KiB), so a
    // conditional skips over a `jal` that carries the displacement.
    out.rv(rv::bne(limit, reg::ZERO, 8));
    out.rv(rv::jal(reg::ZERO, 0));
    let top = out.len();
    loops.push(OpenLoop {
        index,
        limit,
        top,
        skip,
    });
    Ok(())
}

/// Close a loop: `index = index + 1; if index < limit goto top; end:`, and the
/// zero-trip skip of [`open_loop`] patched to the end.
fn close_loop(out: &mut Stream, loops: &mut Vec<OpenLoop>) -> Result<(), ttStatus> {
    let loop_ = loops.pop().ok_or(ttStatus::TT_ERROR_ENDLOOP_MACHINE)?;
    out.rv(rv::addi(loop_.index, loop_.index, 1));
    let here = out.len();
    // Long edges: `bgeu` skips the `jal` back to `top` once the index reaches
    // the limit. The zero-trip skip's `jal` (the word after its `bne`) targets
    // the end.
    out.rv(rv::bgeu(loop_.index, loop_.limit, 8));
    out.rv(rv::jal(reg::ZERO, 0));
    let after = out.len();
    let back = ((loop_.top as i32) - (here as i32 + 1)) * 4;
    out.patch(here + 1, rv::jal(reg::ZERO, back));
    let forward = ((after as i32) - (loop_.skip as i32 + 1)) * 4;
    out.patch(loop_.skip + 1, rv::jal(reg::ZERO, forward));
    Ok(())
}

/// Open a loop whose limit is already in `reg` (a dynamic bound). Same shape as
/// [`open_loop`]; the limit arrives in a register instead of an immediate.
fn open_loop_reg(
    out: &mut Stream,
    loops: &mut Vec<OpenLoop>,
    reg: u8,
    pairs: &[(u8, u8)],
) -> Result<(), ttStatus> {
    let depth = loops.len();
    let (index, limit) = *pairs.get(depth).ok_or(ttStatus::TT_TODO_LOOP_DEPTH)?;
    out.save_if_callee((index, limit), depth);
    if reg != limit {
        out.rv(rv::addi(limit, reg, 0));
    }
    out.li(index, 0);
    let skip = out.len();
    out.rv(rv::bne(limit, reg::ZERO, 8));
    out.rv(rv::jal(reg::ZERO, 0));
    let top = out.len();
    loops.push(OpenLoop {
        index,
        limit,
        top,
        skip,
    });
    Ok(())
}

/// Encode one integer-scratch op. Returns `false` when `op` is not one, so a
/// role encoder can fall through to its own instruction set.
///
/// These are the ops `lower` emits before a [`MirOp::LoopReg`] to compute a
/// bound that is not a literal: a launch argument (a scalar parameter's value),
/// this core's coordinate, or arithmetic over them. They are ordinary RISC-V,
/// the way a C kernel computes its loop bound.
///
/// `pairs` is the calling role's counter pairs: the counters a `LoopReg` opens
/// with are the role's own, so a mover's loop never lands in `t5` (its NIU
/// base) and a compute role's never lands in `a1`/`a2` unless it is the second
/// level.
fn encode_int(
    op: &MirOp,
    out: &mut Stream,
    loops: &mut Vec<OpenLoop>,
    pairs: &[(u8, u8)],
) -> Result<bool, ttStatus> {
    match *op {
        MirOp::Li { reg, value } => out.li(reg, value),
        MirOp::LoadArg { reg, arg } => out.rv(rv::lw(reg, reg::A0, arg)),
        MirOp::Lw { reg, base, at } => out.rv(rv::lw(reg, base, at)),
        MirOp::LoadCoord { reg, axis } => {
            out.li(reg, crate::mir::CORE_INFO + u32::from(axis) * 4);
            out.rv(rv::lw(reg, reg, 0));
        }
        MirOp::IArith { op, d, a, b } => match op {
            IOp::Add => out.rv(rv::add(d, a, b)),
            IOp::Sub => out.rv(rv::sub(d, a, b)),
            IOp::Mul => out.rv(rv::mul(d, a, b)),
            IOp::Div => out.rv(rv::div(d, a, b)),
            IOp::Rem => out.rv(rv::rem(d, a, b)),
            IOp::And => out.rv(rv::and(d, a, b)),
            IOp::Or => out.rv(rv::or(d, a, b)),
            IOp::Xor => out.rv(rv::xor(d, a, b)),
            IOp::Sll => out.rv(rv::sll(d, a, b)),
            IOp::Srl => out.rv(rv::srl(d, a, b)),
            IOp::Max => out.rv(rv::max(d, a, b)),
            IOp::Min => out.rv(rv::min(d, a, b)),
            IOp::Slt => out.rv(rv::slt(d, a, b)),
            // `eq`/`ne` are two words: the difference, then "is it zero" /
            // "is it not zero". A bool scalar is `1`/`0`, so no widening step.
            IOp::Eq => {
                out.rv(rv::xor(d, a, b));
                out.rv(rv::sltiu(d, d, 1));
            }
            IOp::Ne => {
                out.rv(rv::xor(d, a, b));
                out.rv(rv::sltu(d, reg::ZERO, d));
            }
        },
        MirOp::Cmov { cond, a, b, d } => {
            // `d` may alias either arm (`eval` passes `d == a`), so neither arm
            // may be written before it is read: branch over both.
            out.rv(rv::bne(cond, reg::ZERO, 12));
            out.rv(rv::addi(d, b, 0));
            out.rv(rv::jal(reg::ZERO, 8));
            out.rv(rv::addi(d, a, 0));
        }
        MirOp::LoopReg { reg } => open_loop_reg(out, loops, reg, pairs)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// The absolute `Dst` row an access names: the register's base plus the offset
/// inside it, plus a repeat's iteration offset.
fn dst_row(reg: Dst, at: u32, row_delta: u32) -> u32 {
    u32::from(reg) * Dst::ROWS + at + row_delta
}

/// One instruction stream per compute core.
#[derive(Debug, Clone, Default)]
pub struct Words {
    reader: Stream,
    t0: Stream,
    t1: Stream,
    t2: Stream,
    writer: Stream,
}

impl Words {
    /// The five encoded programs, in [`crate::mir::Core`] order, for a debug
    /// listing: each keeps the tag that says whether a word is a Tensix
    /// instruction or RISC-V, which the raw words lose.
    pub(crate) fn programs(&self) -> [(&'static str, &Stream); 5] {
        [
            ("reader", &self.reader),
            ("unpack", &self.t0),
            ("math", &self.t1),
            ("pack", &self.t2),
            ("writer", &self.writer),
        ]
    }

    /// The five encoded programs, in [`crate::mir::Core`] order.
    pub fn words(&self) -> Result<[Vec<u32>; 5], ttStatus> {
        Ok([
            self.reader.words()?,
            self.t0.words()?,
            self.t1.words()?,
            self.t2.words()?,
            self.writer.words()?,
        ])
    }
}

/// Encode all five programs.
///
/// `tail` decides whether the unpacker's tile-descriptor words 2 and 3 are
/// written. They are zero for a tile this compiler stages (no blobs, no
/// explicit `W` dimension), and `ttsim` refuses a write of the all-zero tail,
/// so the card's variant is emitted separately ([`crate::module::Module::silicon_cores`]).
pub fn encode(m: &MachineKernel, tail: bool) -> Result<Words, ttStatus> {
    let mut out = Words::default();
    encode_mover(m, Core::Reader, false, &mut out.reader)?;
    encode_mover(m, Core::Writer, true, &mut out.writer)?;
    encode_unpack(
        &m.stream(Core::Unpack).collect::<Vec<_>>(),
        &mut out.t0,
        m.dst_is_16_bit,
        tail,
    )?;
    encode_math(
        &m.stream(Core::Math).collect::<Vec<_>>(),
        &mut out.t1,
        m.dst_is_16_bit,
    )?;
    encode_pack(
        &m.stream(Core::Pack).collect::<Vec<_>>(),
        &mut out.t2,
        m.dst_is_16_bit,
    )?;
    Ok(out)
}

/// One mover's program: its prologue, then the ops lower emitted for it.
///
/// The prologue is tt-metal's: `t5` = this NIU's base, `t3` = this tile's NOC
/// high word, `a6` = the CB interface table. The reader runs on RISCV B and the
/// writer on RISCV NC, and the two data movers use opposite NoCs
/// (`brisc_noc_id = 1 - ncrisc's`).
fn encode_mover(
    m: &MachineKernel,
    core: Core,
    outbound: bool,
    out: &mut Stream,
) -> Result<(), ttStatus> {
    let ops: Vec<MirOp> = m.stream(core).collect();
    if ops.is_empty() {
        return Ok(());
    }
    // The mover's request temporaries come from the callee-saved bank too (a
    // request that resolves its page through the launch's channel table needs
    // more registers than the caller-saved set has), so its run is bracketed.
    out.keep_callee(&[reg::S2, reg::S3, reg::S4]);
    let niu = if outbound { Niu::Noc1 } else { Niu::Noc0 };
    out.asm(|a| {
        a.li(reg::T5, niu.base() as u32);
        a.word(rv::lw(reg::T3, reg::A0, 12));
        a.word(rv::lw(reg::A6, reg::A0, crate::mover::ARG_IFACE as i32 * 4));
        Ok(())
    })?;
    let mut loops = Vec::new();
    // Each open `IfNonZero`'s unconditional skip word (`jal` to the end).
    let mut ifs: Vec<usize> = Vec::new();
    for op in &ops {
        if encode_int(op, out, &mut loops, &MOVER_PAIRS)? {
            continue;
        }
        match *op {
            MirOp::Loop { count } => open_loop_with(out, &mut loops, count, &MOVER_PAIRS)?,
            MirOp::EndLoop => close_loop(out, &mut loops)?,
            // A conditional body, opened the way a loop is: `bne` skips the
            // `jal` that jumps over the body, so the body may be long.
            MirOp::IfNonZero { cond } => {
                out.rv(rv::bne(cond, reg::ZERO, 8));
                ifs.push(out.len());
                out.rv(rv::jal(reg::ZERO, 0));
            }
            MirOp::EndIf => {
                let at = ifs
                    .pop()
                    .ok_or_else(|| crate::status::internal_bug("encode: an endif has no if"))?;
                let forward = ((out.len() as i32) - (at as i32)) * 4;
                out.patch(at, rv::jal(reg::ZERO, forward));
            }
            // A word store, and the NoC's spin: one instruction and one fixed
            // pair, so neither is a macro op.
            MirOp::Sw { value, base, at } => out.rv(rv::sw(value, base, at)),
            MirOp::NiuPoll { reg, field } => {
                let top = out.len();
                out.rv(rv::lw(reg, reg::T5, field));
                let here = out.len();
                out.rv(rv::bne(reg, reg::ZERO, ((top as i32) - (here as i32)) * 4));
            }
            // A kernel semaphore written by the NoC: tt-metal's
            // `noc_semaphore_wait` / `noc_semaphore_wait_min`, whose every poll
            // invalidates the L0 data cache first (`dataflow_api.h:2114-2154`).
            MirOp::L1PollNonZero { base, reg } => {
                let top = out.len();
                out.rv(0x0ff0_000f);
                out.rv(rv::lw(reg, base, 0));
                // `reg = (reg == 0)`; loop while it is 1.
                out.rv(rv::sltiu(reg, reg, 1));
                let here = out.len();
                out.rv(rv::bne(reg, reg::ZERO, ((top as i32) - (here as i32)) * 4));
            }
            MirOp::L1PollBelow { base, reg, bound } => {
                let top = out.len();
                out.rv(0x0ff0_000f);
                out.rv(rv::lw(reg, base, 0));
                let here = out.len();
                out.rv(rv::bltu(reg, bound, ((top as i32) - (here as i32)) * 4));
            }
            // The ring hand-offs are ordinary ops of the machine kernel: the
            // lowering puts each one on the core that performs it.
            MirOp::CbReserve { cb, n } => out.asm(|a| {
                rv::cb::reserve_back(a, reg::A6, u32::from(cb.0), n, [reg::T0, reg::T1, reg::T2])?;
                Ok(())
            })?,
            MirOp::CbPush { cb, n } => out.asm(|a| {
                rv::cb::push_back(
                    a,
                    reg::A6,
                    u32::from(cb.0),
                    n,
                    [reg::T0, reg::T1, reg::T2, reg::T6],
                )?;
                Ok(())
            })?,
            MirOp::CbWait { cb, n } => out.asm(|a| {
                rv::cb::wait_front(a, reg::A6, u32::from(cb.0), n, [reg::T0, reg::T1, reg::T2])?;
                Ok(())
            })?,
            MirOp::CbPop { cb, n } => out.asm(|a| {
                rv::cb::pop_front(
                    a,
                    reg::A6,
                    u32::from(cb.0),
                    n,
                    [reg::T0, reg::T1, reg::T2, reg::T6],
                )?;
                Ok(())
            })?,
            other => {
                let _ = other;
                return Err(ttStatus::TT_ERROR_MOVER_PROGRAM);
            }
        }
    }
    if !loops.is_empty() {
        return Err(ttStatus::TT_ERROR_ENDLOOP_MACHINE);
    }
    if !ifs.is_empty() {
        return Err(crate::status::internal_bug(
            "encode: a mover program ends inside an if",
        ));
    }
    Ok(())
}

/// The setup program thread 0 pushes before the three roles are released: the
/// `SEMINIT`s for the semaphores the compute programs hand off through.
///
/// tt-metal does exactly this from the core that runs first: BRISC's
/// `device_setup` (`tt_llk_*/tests/helpers/include/boot.h`) calls
/// `t6_semaphore_init(sem, 0, 1)` for each semaphore and only then releases the
/// three TRISCs. Initialising from one core *before* the others start is what
/// makes it race-free; a `SEMINIT` issued by a role itself could reset a value
/// another role has already posted. The table is the machine IR's `Semset` ops,
/// which `reg_sync` derives with the seams: an edge that paces a loop starts with
/// one token per hand-off the body performs, every other edge at zero.
pub fn setup(sems: &[(u8, u8)]) -> Result<Vec<u32>, ttStatus> {
    let mut out: Vec<u32> = semaphore_init(sems)?
        .into_iter()
        .map(|i| {
            i.ttinsn_word().ok_or_else(|| {
                let _ = i;
                ttStatus::TT_ERROR_INSTRUCTION_NOT_INLINE
            })
        })
        .collect::<Result<_, _>>()?;
    // A program is RISC-V code ending in `ret` (NOTES, "Role programs are
    // RISC-V code"); the setup run is called by T0's executor like any other.
    out.push(crate::riscv::ret());
    Ok(out)
}

/// `SEMINIT` each semaphore in the machine IR's table. The `Max` is the Sync
/// Unit's for every edge: back-pressure comes from the `SemWaitFree` before each
/// post, not from a per-edge maximum.
fn semaphore_init(sems: &[(u8, u8)]) -> Result<Vec<Instruction>, ttStatus> {
    let mut out = Vec::new();
    for &(index, value) in sems {
        let sem = semaphore(index)?;
        // The `Dst` lock is a one-token semaphore: `MathLock` waits until it is
        // below its maximum, and the packer's release is the get that takes it
        // back to zero. Every other seam is only a counter.
        let max = if index == crate::lower::SEM_MATH_PACK {
            1
        } else {
            sync::MAX_VALUE
        };
        out.push(sync::init(sem, value, max).map_err(|_| {
            let _ = (index, value);
            ttStatus::TT_ERROR_SEMINIT
        })?);
    }
    Ok(out)
}

/// Announce, through `sem`, that every pipeline the condition mask selects has
/// drained: a `STALLWAIT` that holds every unit of this thread, then the
/// `SEMPOST`. tt-isa's `sync::post_after` is this for one unit; a role whose
/// producers are several (`SFPU` and `MATRIX`, unpacker 0 and 1) needs the
/// combined mask.
fn post_after_drained(drained: u32, sem: Semaphore) -> Result<[Instruction; 2], ttStatus> {
    Ok([
        backend::stallwait(Before::EVERYTHING.mask(), drained)
            .map_err(|e| crate::status::internal_bug(format_args!("encode: stallwait: {e}")))?,
        sync::post(sem),
    ])
}

fn semaphore(index: u8) -> Result<Semaphore, ttStatus> {
    Semaphore::new(index).ok_or_else(|| {
        let _ = index;
        ttStatus::TT_ERROR_SEMAPHORE
    })
}

// ---------------------------------------------------------------------------
// The unpack core (T0)
// ---------------------------------------------------------------------------

/// What one unpacker's `Config` bank currently holds, so a repeated
/// [`MirOp::UnpackConfig`] costs nothing. A `Src` config depends only on the
/// format and the haloize transpose; a `Dst` config also carries the run and the
/// row it lands in, because the unpacker bakes both into that bank.
#[derive(Clone, Copy, PartialEq, Eq)]
enum UnpackCfg {
    Src {
        dtype: DType,
        haloize: bool,
        add_dest_cntr: bool,
        ch1_z_stride: u32,
    },
    Dst {
        dtype: DType,
        datums: u32,
        row: u32,
        dst_is_16_bit: bool,
    },
}

/// What the next `UNPACR` is going to read and where it lands. The `SET*`
/// machine ops before it set these; [`MirOp::UnpackConfig`] writes the bank and
/// the descriptor, and the transfer reads them.
#[derive(Default)]
struct UnpackState {
    /// The `Dst` row an unpack with `to_src` clear lands in.
    dst_row: u32,
    /// Per unpacker (0 = `SrcA`, 1 = `SrcB`): the face `SETADCZW` chose and the
    /// datum run `SETADCX` chose.
    face: [u32; 2],
    x0: [u32; 2],
    x1: [u32; 2],
    /// Per unpacker: the `Config` bank in effect.
    cfg: [Option<UnpackCfg>; 2],
    /// Per unpacker: the page the base-address word holds, as `(ring, offset,
    /// extra)`. `UNPACR` reads the word; it does not move it, so a second read
    /// of the same page needs no new write -- tt-metal's `llk_unpack_AB_matmul`
    /// configures both operand addresses once and then issues its `TTI_UNPACR`s
    /// (`llk_unpack_AB_matmul.h:374-379`). A hand-off moves a pointer, so every
    /// `cb_wait`/`cb_pop` forgets it.
    base: [Option<(Cb, u32, Option<u8>)>; 2],
    /// The `SRCA_SET_SetOvrdWithAddr` word as last written. It is its own
    /// runtime state, not part of an unpacker's config bank: a `Dst` unpack
    /// clears it, so a `Src` unpack that reuses a cached config still has to
    /// set it again. The prologue wrote it to zero (`thread_config`).
    override_set: Option<bool>,
}

fn encode_unpack(
    ops: &[MirOp],
    out: &mut Stream,
    dst_is_16_bit: bool,
    tail: bool,
) -> Result<(), ttStatus> {
    if ops.is_empty() {
        return Ok(());
    }
    // The `ThreadConfig` entries come first, as at the start of every tt-metal
    // kernel: `CFG_STATE_ID` decides which `Config` bank later writes land in, so
    // no config store may race with them.
    thread_config(out)?;
    alu_acc_ctrl(out, dst_is_16_bit)?;
    // Every role is called with `a0` = the launch arguments; word 2 is the CB
    // interface table (`tt-metal`'s `ARG_IFACE`).
    out.rv(rv::lw(reg::A6, reg::A0, crate::mover::ARG_IFACE as i32 * 4));
    let mut state = UnpackState {
        // `thread_config` writes the toggle to zero before anything runs.
        override_set: Some(false),
        ..UnpackState::default()
    };
    let mut loops = Vec::new();
    for op in ops {
        if encode_int(op, out, &mut loops, &COMPUTE_PAIRS)? {
            continue;
        }
        match *op {
            MirOp::Loop { count } => {
                // The unpacker's config bank is runtime state and the cache
                // below is straight-line: at a loop head the previous execution
                // is the loop's own tail, not the instruction before it in the
                // stream, so every config in the body is rewritten on entry. A
                // body that ends with a `Src` unpack and starts with a `Dst` one
                // is how this went wrong: the second iteration's first `Src`
                // write was elided and `SRCA_SET_SetOvrdWithAddr` stayed clear
                // under an unpack to `Src`, which ttsim refuses. The toggle is
                // the same runtime state, so it is forgotten with the bank.
                state.cfg = [None, None];
                state.override_set = None;
                state.base = [None, None];
                open_loop(out, &mut loops, count)?;
            }
            MirOp::EndLoop => {
                // The loop may run zero times, so nothing its body configured
                // is in effect after it: the bank, the `Src` override and the
                // base address are all forgotten. A zero-trip loop whose body
                // configured `SrcB` and a peeled tail that reads `SrcB` right
                // after it is how this went wrong -- the tail's `UnpackConfig`
                // was elided and ttsim refused the `UNPACR` with
                // `is_uncompressed=0`.
                state.cfg = [None, None];
                state.override_set = None;
                state.base = [None, None];
                close_loop(out, &mut loops)?;
            }
            MirOp::Setdst { reg } => state.dst_row = u32::from(reg) * Dst::ROWS,
            MirOp::Setadcx { unit, first, count } => {
                if count == 0 {
                    return Err(ttStatus::TT_ERROR_UNPACK_ZERO_RANGE);
                }
                let u = usize::from(unit);
                state.x0[u] = first;
                state.x1[u] = first + count - 1;
            }
            MirOp::Setadczw { u0, u1, z0, z1 } => {
                let mut entry = encode::Setadczw::ZERO;
                if u0 {
                    entry = entry.u0(1);
                }
                if u1 {
                    entry = entry.u1(1);
                }
                if let Some(face) = z0 {
                    entry = entry.z0(1).z0_val(face);
                    if u0 {
                        state.face[0] = face;
                    }
                    if u1 {
                        state.face[1] = face;
                    }
                }
                if let Some(face) = z1 {
                    entry = entry.z1(1).z1_val(face);
                }
                out.push(entry.encode().map_err(|_| ttStatus::TT_ERROR_SETADCZW)?);
            }
            MirOp::UnpackConfig {
                unit,
                to_src,
                dtype,
                haloize,
                add_dest_cntr,
                ch1_z_stride,
            } => {
                let u = usize::from(unit);
                let cfg = if to_src {
                    UnpackCfg::Src {
                        dtype,
                        haloize,
                        add_dest_cntr,
                        ch1_z_stride,
                    }
                } else {
                    // A `Dst` config bakes the page's datum count and landing
                    // row into the descriptor, so both come from the `SETADCX`
                    // and `SETDST` in effect.
                    let datums = state.x1[u]
                        .checked_sub(state.x0[u])
                        .and_then(|span| span.checked_add(1))
                        .filter(|datums| *datums != 0)
                        .ok_or_else(|| {
                            let _ = (unit, u);
                            ttStatus::TT_ERROR_UNPACK_ZERO_DATUMS
                        })?;
                    UnpackCfg::Dst {
                        dtype,
                        datums,
                        row: state.dst_row,
                        dst_is_16_bit,
                    }
                };
                // `SRCA_SET_SetOvrdWithAddr`: the Src path wants it set, the Dst
                // path clear, and the unpacker reads it at execution time, so a
                // role that does both toggles it here. ttsim refuses either
                // combination otherwise. The toggle is written whenever it is
                // not already what this config wants, even when the config
                // bank itself is cached: a `Dst` unpack between two `Src`
                // unpacks clears it, and the second `Src` unpack would
                // otherwise read a page through an override it never enabled.
                if state.override_set != Some(to_src) {
                    out.push(thread_entry(
                        thread::SRCA_SET_SetOvrdWithAddr,
                        u16::from(to_src),
                    )?);
                    state.override_set = Some(to_src);
                }
                if state.cfg[u] == Some(cfg) {
                    continue;
                }
                let mut words = ConfigWords::new();
                match cfg {
                    UnpackCfg::Src {
                        dtype,
                        haloize,
                        add_dest_cntr,
                        ch1_z_stride,
                    } => src_unpack_config(
                        &mut words,
                        unit,
                        dtype,
                        haloize,
                        add_dest_cntr,
                        ch1_z_stride,
                        tail,
                    )?,
                    UnpackCfg::Dst { datums, row, .. } => {
                        unpack_config(&mut words, datums, row, dtype, dst_is_16_bit, tail)?
                    }
                }
                out.extend(config_program(&words)?);
                out.push(stallwait(
                    block::UNPACKER | block::CONFIG,
                    cond::CONFIG_BUSY,
                )?);
                state.cfg[u] = Some(cfg);
                continue;
            }
            MirOp::SetUnpackBase {
                cb,
                unit,
                offset,
                extra,
            } => {
                // The page is at the ring's read pointer, not a compile-time L1
                // address: write that pointer into the unpacker's base register
                // after draining the previous unpack. `offset` is the page's
                // byte offset past the pointer, tt-metal's
                // `get_tile_l1_byte_address` (`cb_api.h:143-151`).
                //
                // The drain is unconditional; the write is not. `UNPACR` reads
                // the base word without moving it (the face walk is the address
                // counters'), so several unpacks of one page share it -- tt-metal
                // configures the addresses once per `llk_unpack_AB_matmul` and
                // issues its `TTI_UNPACR`s after (`llk_unpack_AB_matmul.h`).
                cb_base_wait(out, unit)?;
                let slot = usize::from(unit);
                // A register-valued page offset may hold a *different* value
                // than it did at the previous call even though the op names the
                // same register, so only a constant offset may be elided.
                if extra.is_none() && state.base[slot] == Some((cb, offset, extra)) {
                    continue;
                }
                let config = match unit {
                    0 => thcon::THCON_SEC0_REG3_Base_address.addr32(),
                    1 => thcon::THCON_SEC1_REG3_Base_address.addr32(),
                    _ => {
                        let _ = unit;
                        return Err(ttStatus::TT_ERROR_UNPACR);
                    }
                };
                cb_base(out, cb, rv::cb::RD_PTR, config, offset, extra)?;
                state.base[slot] = Some((cb, offset, extra));
            }
            MirOp::Unpacr { unit, count, flip } => {
                if count == 0 {
                    return Err(ttStatus::TT_ERROR_UNPACK_ZERO_DATUMS);
                }
                let u = usize::from(unit);
                state.x0[u] = 0;
                state.x1[u] = count - 1;
                out.push(set_adc_x_unpack(unit, state.x0[u], state.x1[u])?);
                // Always named: the address counters outlive the program, so a
                // face a previous kernel left behind would be read again.
                out.push(set_adc_z_unpack(unit, state.face[u])?);
                out.push(unpack_instruction(unit, flip)?);
            }
            MirOp::CbWait { cb, n } => {
                // A hand-off can move a ring's pointer; the base words no longer
                // name the pages they were computed from.
                state.base = [None, None];
                out.asm(|a| {
                    rv::cb::wait_front(a, reg::A6, u32::from(cb), n, [reg::T0, reg::T1, reg::T2])
                })?;
            }
            MirOp::CbPop { cb, n } => {
                state.base = [None, None];
                // The page is only free once both unpackers have read it.
                out.push(
                    wait_for_unpacker0(Before::EVERYTHING)
                        .map_err(|_| ttStatus::TT_ERROR_STALLWAIT)?,
                );
                // A page the Matrix Unit read is released only once its own
                // unpacker has drained too.
                out.push(
                    tt_isa::arch::blackhole::backend::wait_for_unpacker1(Before::EVERYTHING)
                        .map_err(|_| ttStatus::TT_ERROR_STALLWAIT)?,
                );
                out.asm(|a| {
                    // `pop_front` is RISC-V `lw`/`sh`, which a pushed
                    // `STALLWAIT` does not hold back: without the coprocessor
                    // drain the page is acked while the `UNPACR` may still be
                    // reading it, and the producer refills it under the
                    // unpacker. The packer's push drains the same way.
                    rv::coprocessor_done(a);
                    rv::cb::pop_front(
                        a,
                        reg::A6,
                        u32::from(cb),
                        n,
                        [reg::T0, reg::T1, reg::T2, reg::T6],
                    )
                })?;
            }
            MirOp::SemPost { sem } => {
                // The data a post announces must have *finished*, not merely
                // been issued: `SEMPOST` can run as soon as the Sync Unit sees
                // it, so the pipelines that write this role's operands drain
                // first (tt-isa's `post_after`; tt-metal's
                // `tile_regs_commit`'s `STALLWAIT(PACK, MATH)`). Both unpackers
                // write `Src` here, so both drain.
                out.extend(post_after_drained(
                    cond::UNPACKER0_BUSY | cond::UNPACKER1_BUSY,
                    semaphore(sem)?,
                )?);
            }
            MirOp::SemWait { sem, .. } => {
                out.extend(sync::take(semaphore(sem)?, Before::EVERYTHING));
            }
            MirOp::SemWaitNoGet { sem, .. } => {
                out.push(sync::wait_nonzero(semaphore(sem)?, Before::EVERYTHING));
            }
            MirOp::SemGet { sem } => out.push(sync::get(semaphore(sem)?)),
            MirOp::SemWaitFree { sem } => {
                out.push(sync::wait_below_max(semaphore(sem)?, Before::EVERYTHING));
            }
            other => {
                let _ = other;
                return Err(ttStatus::TT_ERROR_UNPACK_OP);
            }
        }
    }
    if !loops.is_empty() {
        return Err(ttStatus::TT_ERROR_ENDLOOP_MACHINE);
    }
    Ok(())
}

/// The Tensix GPR the ring base is staged through before its `WRCFG`. GPR *n*
/// is 16-bit halves `2n` and `2n + 1` (`SETDMAREG_Immediate.md`).
const BASE_GPR: u32 = 24;
/// `SETDMAREG`'s opcode, fixed bits and field positions (`SETDMAREG_Immediate`):
/// `(0x45 << 24) | (NewValue << 8) | ResultHalfReg`.
const SETDMAREG: u32 = 0x4500_0000;
/// `INSTRN_BUF_BASE`: the store target that pushes a Tensix instruction. RISC-V
/// T0/T1/T2 all push their own thread through this address
/// (`PushTensixInstruction.md`); a `.ttinsn` is the same store.
const INSTRN_BUF: u32 = Blackhole::INSTRN_BUF_BASE as u32;

/// Drain an unpacker before its next base write or read: every earlier
/// `UNPACR` of this thread must have left the pipeline.
fn cb_base_wait(out: &mut Stream, unit: u8) -> Result<(), ttStatus> {
    if unit > 1 {
        let _ = unit;
        return Err(ttStatus::TT_ERROR_UNPACR);
    }
    out.push(wait_for_unpacker0(Before::EVERYTHING).map_err(|_| ttStatus::TT_ERROR_STALLWAIT)?);
    if unit == 1 {
        out.push(
            tt_isa::arch::blackhole::backend::wait_for_unpacker1(Before::EVERYTHING)
                .map_err(|_| ttStatus::TT_ERROR_STALLWAIT)?,
        );
    }
    Ok(())
}

/// The packer's output base: the ring's write pointer plus the page's byte
/// offset, same 16-byte conversion as the unpacker's read side.
fn cb_base_pack(
    out: &mut Stream,
    cb: Cb,
    offset: u32,
    extra: Option<u8>,
) -> Result<(), ttStatus> {
    out.push(wait_for_packer(Before::EVERYTHING).map_err(|_| ttStatus::TT_ERROR_STALLWAIT)?);
    cb_base(
        out,
        cb,
        rv::cb::WR_PTR,
        thcon::THCON_SEC0_REG1_L1_Dest_addr.addr32(),
        offset,
        extra,
    )
}

/// Write the ring page into a `Config` word from a Tensix GPR, never from a
/// RISC-V `sw` into `Config`.
///
/// A plain `sw` to `Config` is processed by the LSU, not the Tensix instruction
/// FIFO, so it overtakes the `UNPACR`/`PACR` already reading the same word.
/// tt-metal's own unpacker writes the base that way
/// (`_llk_unpack_configure_addresses_`) and holds the unpacker with
/// `TTI_STALLWAIT(STALL_UNPACK, TRISC_CFG)` (C10, "the RISC-V core's `Config`
/// write has not been processed"); on this card that store still overtakes the
/// `UNPACR` and the page lags by one, so the write is a pushed instruction
/// instead: `SETDMAREG` stages the dynamic value, then `WRCFG` copies it, in
/// FIFO order with the backend. This is tt-metal's `program_packer_destination`
/// (`SETDMAREG` x 2, `STALLWAIT(STALL_CFG, THCON)`, `WRCFG`, `DMANOP`).
///
/// The caller's `wait_for_unpacker0`/`wait_for_packer` now reaches the write:
/// because the write is a pushed instruction, `STALLWAIT`'s CONFIG block holds
/// it until the previous unit has drained.
fn cb_base(
    out: &mut Stream,
    cb: Cb,
    field: i32,
    config: u16,
    offset: u32,
    extra: Option<u8>,
) -> Result<(), ttStatus> {
    // The config word is in 16-byte units, and so is the page offset the caller
    // named: tt-metal's `get_tile_l1_byte_address` adds `fifo_page_size *
    // tile_index` to `fifo_rd_ptr` before the whole address is shifted into the
    // interface's units.
    if offset % 16 != 0 {
        return Err(ttStatus::TT_ERROR_BUFFER_OFFSET_OVERFLOW);
    }
    let offset = (offset / 16) as i32;
    out.asm(|a| {
        // The address is built in `T0`, and a caller may have named `T0` as the
        // page register itself (the lowering's per-tile offsets use `T0`/`T6`).
        // Stage the index first, or `add T0, T0, T0` would double the base
        // instead of adding the page.
        let extra = extra.map(|index| {
            a.word(rv::addi(reg::T1, index, 0));
            reg::T1
        });
        a.word(rv::lw(
            reg::T0,
            reg::A6,
            (u32::from(cb) * rv::cb::STRIDE) as i32 + field,
        ));
        a.word(rv::srli(reg::T0, reg::T0, 4));
        a.word(rv::addi(reg::T0, reg::T0, -1));
        if offset != 0 {
            // `ADDI`'s immediate is twelve *signed* bits, so a page further than
            // 32 KiB into its ring does not fit: `(32 KiB / 16)` is 2048, one
            // past the largest positive immediate, and the encoder's mask would
            // wrap it to -2048. A 24-page f32 ring names page 8 at exactly
            // 2048, which put the unpacker a whole 64 KiB below the page it
            // meant (into the operand ring's data).
            if (-2048..=2047).contains(&offset) {
                a.word(rv::addi(reg::T0, reg::T0, offset));
            } else {
                a.li(reg::T2, offset as u32);
                a.word(rv::add(reg::T0, reg::T0, reg::T2));
            }
        }
        // A run-time page index (tt-metal's `matmul_tiles` argument) has already
        // been scaled to the word's 16-byte units by the lowering.
        if let Some(index) = extra {
            a.word(rv::add(reg::T0, reg::T0, index));
        }
        // `SETDMAREG` with a run-time immediate: the word is built here and
        // pushed through the instruction FIFO, exactly as tt-metal's
        // `TT_SETDMAREG` macro does (`instrn_buffer[0] = ...`).
        a.word(rv::slli(reg::T1, reg::T0, 16));
        a.word(rv::srli(reg::T1, reg::T1, 16));
        a.word(rv::slli(reg::T1, reg::T1, 8));
        a.li(reg::T2, SETDMAREG);
        a.word(rv::or(reg::T1, reg::T1, reg::T2));
        a.word(rv::ori(reg::T1, reg::T1, (BASE_GPR * 2) as i32));
        a.li(reg::T2, INSTRN_BUF);
        a.word(rv::sw(reg::T1, reg::T2, 0));
        a.word(rv::srli(reg::T1, reg::T0, 16));
        a.word(rv::slli(reg::T1, reg::T1, 8));
        a.li(reg::T2, SETDMAREG);
        a.word(rv::or(reg::T1, reg::T1, reg::T2));
        a.word(rv::ori(reg::T1, reg::T1, (BASE_GPR * 2 + 1) as i32));
        a.li(reg::T2, INSTRN_BUF);
        a.word(rv::sw(reg::T1, reg::T2, 0));
        Ok(())
    })?;
    // The Scalar Unit fills the GPR; `WRCFG` reads it. `WRCFG.md` also wants a
    // non-consumer after the write, which the trailing NOP gives it.
    out.push(
        backend::stallwait(block::CONFIG, cond::SCALAR_OUTSTANDING)
            .map_err(|_| ttStatus::TT_ERROR_STALLWAIT)?,
    );
    out.push(backend::write_word(BASE_GPR, config).map_err(|e| {
        let _ = e;
        ttStatus::TT_ERROR_STALLWAIT
    })?);
    out.push(backend::nop());
    Ok(())
}

/// `ALU_ACC_CTRL`: the `Dst` width the whole program is built for, written
/// before any unit runs. `Fp32_enabled` is the 32-bit bank; a bf16 kernel clears
/// it. The packer writes the same value with its own config, but the SFPU and
/// the Matrix Unit read it long before the first pack — and it outlives the
/// program, so every role sets it rather than inheriting it from whatever ran
/// before.
fn alu_acc_ctrl(out: &mut Stream, dst_is_16_bit: bool) -> Result<(), ttStatus> {
    let set = config_set;
    let mut words = ConfigWords::new();
    set(
        &mut words,
        alu::ALU_ACC_CTRL_Fp32_enabled,
        u32::from(!dst_is_16_bit),
    )?;
    // `MVMUL.md`: the "keep SrcB denormals" mode must not be used.
    set(&mut words, alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 0)?;
    out.extend(config_program(&words)?);
    Ok(())
}

/// `SETC16.md`: the first `ThreadConfig` writes any thread must make. Emitted
/// once per role program, before anything reads the configuration bank.
fn thread_config(out: &mut Stream) -> Result<(), ttStatus> {
    for (field, value) in [
        (thread::CFG_STATE_ID_StateID, 0),
        (thread::SRCA_SET_SetOvrdWithAddr, 0),
        (thread::UNPACK_MISC_CFG_CfgContextOffset_0, 0),
    ] {
        out.push(thread_entry(field, value)?);
    }
    Ok(())
}

fn thread_entry(
    field: tt_isa::cfg::ThreadConfigField,
    value: u16,
) -> Result<Instruction, ttStatus> {
    let entry = match ThreadConfigEntry::zeroed(field.addr32()).set(field, value) {
        Ok(entry) => entry,
        Err(_) => {
            return Err(crate::status::internal_bug(format_args!(
                "encode: thread config value {value} does not fit {field:?}"
            )));
        }
    };
    match entry.encode() {
        Ok(instruction) => Ok(instruction),
        Err(e) => Err(crate::status::internal_bug(format_args!(
            "encode: a thread entry is not encodable: {e}"
        ))),
    }
}

/// Place one `Config` word, reporting a value that does not fit its field.
fn config_set(
    words: &mut ConfigWords,
    field: tt_isa::cfg::ConfigField,
    value: u32,
) -> Result<(), ttStatus> {
    words
        .set(field, value)
        .map(|_| ())
        .map_err(|e| crate::status::internal_bug(format_args!("encode: {e}")))
}

/// Stage one tile-descriptor word into the `Config` bank.
fn config_seed(words: &mut ConfigWords, at: u16, value: u32) -> Result<(), ttStatus> {
    words
        .seed(at, value)
        .map(|_| ())
        .map_err(|e| crate::status::internal_bug(format_args!("encode: {e}")))
}

/// The instructions that write `words` into `Config`, sized to fit.
fn config_program(words: &ConfigWords) -> Result<Vec<Instruction>, ttStatus> {
    let mut buf = vec![tt_isa::arch::blackhole::sfpu::nop(); words.program_len()];
    let n = match words.program(SCRATCH_GPR, &mut buf) {
        Ok(n) => n,
        Err(e) => {
            return Err(crate::status::internal_bug(format_args!(
                "encode: the config program does not fit its buffer: {e}"
            )));
        }
    };
    buf.truncate(n);
    Ok(buf)
}

fn stallwait(block_mask: u32, condition: u32) -> Result<Instruction, ttStatus> {
    backend::stallwait(block_mask, condition).map_err(|e| {
        let _ = e;
        ttStatus::TT_ERROR_STALLWAIT
    })
}

/// The `SFPLOAD`/`SFPSTORE` lane format of one datum: the conversion the lane
/// makes on the way in or out.
fn lane_format(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => tt_isa::arch::blackhole::sfpu::mod0_fmt::FP32,
        DType::I32 | DType::Bool => tt_isa::arch::blackhole::sfpu::mod0_fmt::INT32,
        DType::Bf16 => tt_isa::arch::blackhole::sfpu::mod0_fmt::BF16,
    }
}

/// The L1 format of one buffer datum: a bf16 datum is two bytes in L1, every
/// other datum is carried as 32 opaque bits.
fn l1_format(dtype: DType) -> u32 {
    match dtype {
        DType::Bf16 => DATA_FORMAT_BF16,
        _ => DATA_FORMAT_FP32,
    }
}

/// The `Dst` format of the whole kernel: the 16-bit bank is a bf16 datum
/// (`DST_ACCUM_MODE`'s 16-bit half), the 32-bit bank is an opaque 32-bit one.
/// The unpacker's `Out_data_format`, the packer's `In_data_format` and the
/// packer's 16-bit read path all follow it.
fn dst_format(dst_is_16_bit: bool) -> u32 {
    if dst_is_16_bit {
        DATA_FORMAT_BF16
    } else {
        DATA_FORMAT_FP32
    }
}

/// The 4-bit `InDataFormat`/`OutDataFormat` codes, tt-metal's `DataFormat`:
/// `Float32` 0, `Float16` 1, `Tf32` 4, `BFloat16` 5, `Int32` 8.
///
/// `MEASURED` on ttsim rather than transcribed
/// (`metalium-rs/docs/learnings/ttsim-divergence.md` rows G and H): exactly
/// `(0, 0)`, `(0, 4)` and `(8, 8)` run to `Dst` out of the 256 pairs, and the
/// `Src` path separates 1 from 5 by the mantissa cut. The measurement agrees
/// with tt-metal's enum, which was not consulted to find it.
const DATA_FORMAT_FP32: u32 = 0;
const DATA_FORMAT_TF32: u32 = 4;
const DATA_FORMAT_BF16: u32 = 5;

/// The unpacker's tile descriptor, tt-metal's `unpack_tile_descriptor_t`
/// (`tt_llk_wormhole_b0/common/inc/cunpack_common.h:22-53`): four words.
///
/// ```text
/// word 0  in_data_format:4  uncompressed:1  reserved:3  blobs_per_xy_plane:4
///         reserved:4  x_dim:16
/// word 1  y_dim:16  z_dim:16
/// word 2  w_dim:16  blobs_y_start_lo:16
/// word 3  blobs_y_start_hi:16  digest_type:8  digest_size:8
/// ```
///
/// A tile this compiler stages is uncompressed and carries no blobs, so only
/// the dims and the format are ever set.
fn tile_descriptor(in_data_format: u32, x_dim: u16, y_dim: u16, z_dim: u16) -> [u32; 4] {
    [
        in_data_format | (1 << 4) | (u32::from(x_dim) << 16),
        u32::from(y_dim) | (u32::from(z_dim) << 16),
        0,
        0,
    ]
}

/// The unpacker's `Config` for one unit: base, destination, X dimension, FP32
/// in and out. The base is `base/16 - 1` because L1 holds the datum block with
/// no 16-byte tile header, while the unpacker adds one unconditionally.
fn unpack_config(
    words: &mut ConfigWords,
    datums: u32,
    dst_row: u32,
    dtype: DType,
    dst_is_16_bit: bool,
    tail: bool,
) -> Result<(), ttStatus> {
    // `In_data_format` is the L1 datum's own format; the `Dst` side is the
    // kernel's one width. The lane's format (`SFPLOAD`/`SFPSTORE`) is the third
    // choice and lives in the ops.
    let fmt = l1_format(dtype);
    let dst = dst_format(dst_is_16_bit);
    config_set(words, thcon::THCON_SEC0_REG7_Offset_address, 0)?;
    config_set(words, thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0, datums)?;
    config_set(words, thcon::THCON_SEC0_REG2_Out_data_format, dst)?;
    config_set(words, thcon::THCON_SEC0_REG2_Ovrd_data_format, 0)?;
    config_set(words, thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx0, 1)?;
    config_set(words, thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx0, 1)?;
    config_set(words, thcon::THCON_SEC0_REG2_Haloize_mode, 0)?;
    config_set(words, thcon::THCON_SEC0_REG2_Tileize_mode, 0)?;
    config_set(words, thcon::THCON_SEC0_REG2_Throttle_mode, 0)?;
    config_set(words, thcon::THCON_SEC0_REG2_Upsample_rate, 0)?;
    config_set(words, thcon::THCON_SEC0_REG2_Upsample_and_interleave, 0)?;
    config_set(words, thcon::THCON_SEC0_REG2_Shift_amount_cntx0, 0)?;
    config_set(words, thcon::THCON_SEC0_REG2_Force_shared_exp, 0)?;
    config_set(
        words,
        thcon::THCON_SEC0_REG5_Dest_cntx0_address,
        dst_address(dst_row),
    )?;
    config_set(words, unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride, 0)?;
    config_set(words, unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride, 0)?;
    let descriptor = tile_descriptor(fmt, datums as u16, 1, 0);
    // Words 0 and 1 always; 2 and 3 only when they carry something — or when
    // `tail` says the card is the target, which needs the whole four-word
    // descriptor written because `Config` survives the program that wrote it.
    // ttsim refuses a write of the all-zero tail, which is why the two
    // variants exist.
    for (i, word) in descriptor.iter().enumerate() {
        if i < 2 || *word != 0 || tail {
            config_seed(
                words,
                thcon::THCON_SEC0_REG0_TileDescriptor.addr32() + i as u16,
                *word,
            )?;
        }
    }
    Ok(())
}

/// Configure unpacker `unit` to move a 32x32 tile's four 16x16 faces from L1
/// into `SrcA` (`unit` 0) or `SrcB` (`unit` 1), as the Matrix Unit reads them.
///
/// The `Src` side is not `Dst`: `Unpack_if_sel_cntx0` clear, no `SETADCX` run,
/// and the descriptor names four Z planes of one 16x16 face each. What comes
/// out of the unpacker is `TF32` for an `f32` operand (FP32 *into* `Src` is
/// undefined) and `BF16` for a bf16 one; the Matrix Unit learns it from the
/// implied format the handover sets, not from `ALU_FORMAT_SPEC`.
fn src_unpack_config(
    words: &mut ConfigWords,
    unit: u8,
    dtype: DType,
    haloize: bool,
    add_dest_cntr: bool,
    ch1_z_stride: u32,
    tail: bool,
) -> Result<(), ttStatus> {
    let out = match dtype {
        DType::Bf16 => DATA_FORMAT_BF16,
        _ => DATA_FORMAT_TF32,
    };
    let haloize = u32::from(haloize);
    let add_dest_cntr = u32::from(add_dest_cntr);
    let descriptor = tile_descriptor(l1_format(dtype), 16, 16, 4);
    let span = match unit {
        0 => {
            config_set(words, thcon::THCON_SEC0_REG7_Offset_address, 0)?;
            config_set(words, thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0, 16)?;
            config_set(words, thcon::THCON_SEC0_REG2_Out_data_format, out)?;
            config_set(words, thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx0, 0)?;
            config_set(words, thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx0, 1)?;
            config_set(words, thcon::THCON_SEC0_REG2_Haloize_mode, haloize)?;
            config_set(words, thcon::THCON_SEC0_REG5_Dest_cntx0_address, 64)?;
            config_set(
                words,
                unpack0::UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr,
                add_dest_cntr,
            )?;
            config_set(words, unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride, 0)?;
            config_set(
                words,
                unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride,
                ch1_z_stride,
            )?;
            thcon::THCON_SEC0_REG0_TileDescriptor
        }
        _ => {
            // Unpacker 1 takes far fewer fields: no `Dest_cntx` term, no
            // `Tile_x_dim_cntx` (the descriptor carries the width), and its
            // `Offset_address` cannot be written (reset zero).
            config_set(words, thcon::THCON_SEC1_REG2_Out_data_format, out)?;
            config_set(words, thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx0, 1)?;
            config_set(words, thcon::THCON_SEC1_REG2_Haloize_mode, haloize)?;
            if add_dest_cntr != 0 {
                config_set(
                    words,
                    unpack1::UNP1_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr,
                    add_dest_cntr,
                )?;
            }
            config_set(
                words,
                unpack1::UNP1_ADDR_CTRL_ZW_REG_1_Zstride,
                ch1_z_stride,
            )?;
            thcon::THCON_SEC1_REG0_TileDescriptor
        }
    };
    // As `unpack_config`: the card needs words 2 and 3 written even when they
    // are zero, because `Config` outlives the program; ttsim refuses that write.
    for (i, word) in descriptor.iter().enumerate() {
        if *word != 0 || tail {
            config_seed(words, span.addr32() + i as u16, *word)?;
        }
    }
    Ok(())
}

/// The `REG5_Dest_cntx0_address` that lands an unpack's first datum in `Dst`
/// row `row`: `Dst` row is `OutAddr / 16 - 4`.
pub const fn dst_address(row: u32) -> u32 {
    (row + 4) * 16
}

fn set_adc_x_unpack(unit: u8, first: u32, last: u32) -> Result<Instruction, ttStatus> {
    let base = encode::Setadcxx::ZERO;
    let base = match unit {
        0 => base.u0(1),
        1 => base.u1(1),
        _ => {
            let _ = unit;
            return Err(ttStatus::TT_ERROR_SETADCXX);
        }
    };
    base.x0_val(first).x1_val(last).encode().map_err(|e| {
        let _ = e;
        ttStatus::TT_ERROR_SETADCXX
    })
}

fn set_adc_z_unpack(unit: u8, face: u32) -> Result<Instruction, ttStatus> {
    let base = encode::Setadczw::ZERO;
    let base = match unit {
        0 => base.u0(1),
        1 => base.u1(1),
        _ => {
            let _ = unit;
            return Err(ttStatus::TT_ERROR_SETADCZW);
        }
    };
    base.z0(1).z0_val(face).encode().map_err(|e| {
        let _ = e;
        ttStatus::TT_ERROR_SETADCZW
    })
}

/// `UNPACR` into `Dst` (unit 0, no `flip`) or into `SrcA`/`SrcB` (`flip` hands
/// the bank to the Matrix Unit).
fn unpack_instruction(unit: u8, flip: bool) -> Result<Instruction, ttStatus> {
    // Bit 0 is undocumented and ttsim refuses the instruction without it.
    let base = encode::UnpacrRegular::ZERO
        .which_unpacker(u32::from(unit))
        .multi_context_mode(1)
        .flip_src(u32::from(flip))
        .encode()
        .map_err(|e| {
            let _ = e;
            ttStatus::TT_ERROR_UNPACR
        })?;
    Ok(Instruction::new(base.word() | 1, &defs::UNPACR_Regular))
}

// ---------------------------------------------------------------------------
// The math core (T1)
// ---------------------------------------------------------------------------

fn encode_math(ops: &[MirOp], out: &mut Stream, dst_is_16_bit: bool) -> Result<(), ttStatus> {
    if ops.is_empty() {
        return Ok(());
    }
    alu_acc_ctrl(out, dst_is_16_bit)?;
    let mut at = 0usize;
    let mut loops = Vec::new();
    while at < ops.len() {
        if encode_int(&ops[at], out, &mut loops, &COMPUTE_PAIRS)? {
            at += 1;
            continue;
        }
        match ops[at] {
            MirOp::Repeat { len, count } => {
                let body = ops.get(at + 1..at + 1 + len as usize).ok_or_else(|| {
                    let _ = at;
                    ttStatus::TT_ERROR_MATH_REPEAT_BODY
                })?;
                for iteration in 0..u32::from(count) {
                    for op in body {
                        encode_math_op(op, iteration * 2, out)?;
                    }
                }
                at += 1 + len as usize;
                continue;
            }
            MirOp::Loop { count } => {
                open_loop(out, &mut loops, count)?;
                at += 1;
                continue;
            }
            MirOp::EndLoop => {
                close_loop(out, &mut loops)?;
                at += 1;
                continue;
            }
            op => encode_math_op(&op, 0, out)?,
        }
        at += 1;
    }
    if !loops.is_empty() {
        return Err(ttStatus::TT_ERROR_ENDLOOP_MACHINE);
    }
    Ok(())
}

/// One math op at `row_delta` (a repeat's iteration offset, two address units per
/// iteration).
fn encode_math_op(op: &MirOp, row_delta: u32, out: &mut Stream) -> Result<(), ttStatus> {
    fn bad(e: impl std::fmt::Display, op: &MirOp) -> ttStatus {
        let _ = (e, op);
        ttStatus::TT_ERROR_ENCODE
    }
    let bad_isa = |e: tt_isa::isa::EncodeError| bad(e, op);
    match *op {
        MirOp::Sfpload {
            lreg,
            reg,
            at,
            dtype,
        } => {
            let i = encode::sfpload(
                u32::from(lreg),
                lane_format(dtype),
                0,
                dst_row(reg, at, row_delta),
            )
            .map_err(bad_isa)?;
            out.push(i);
        }
        MirOp::Sfpstore {
            lreg,
            reg,
            at,
            dtype,
        } => {
            let i = encode::sfpstore(
                u32::from(lreg),
                lane_format(dtype),
                0,
                dst_row(reg, at, row_delta),
            )
            .map_err(bad_isa)?;
            out.push(i);
        }
        MirOp::LaneConst { lreg, bits } => {
            let halves = tt_isa::arch::blackhole::sfpu::load_f32(u32::from(lreg), bits)
                .map_err(|e| bad(e, op))?;
            for i in halves {
                out.push(i);
            }
        }
        MirOp::UnarySfp {
            op: UnarySfpOp::Neg,
            a,
            d,
        } => {
            out.push(encode::sfpmov(u32::from(a), u32::from(d), 1).map_err(bad_isa)?);
        }
        MirOp::UnarySfp {
            op: UnarySfpOp::Abs,
            a,
            d,
        } => {
            out.push(encode::sfpabs(u32::from(a), u32::from(d), 1).map_err(bad_isa)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a,
            b,
            d,
        } => {
            out.push(
                tt_isa::arch::blackhole::sfpu::add(u32::from(a), u32::from(b), u32::from(d))
                    .map_err(|e| bad(e, op))?,
            );
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::Sub,
            a,
            b,
            d,
        } => {
            out.push(
                tt_isa::arch::blackhole::sfpu::sub(u32::from(a), u32::from(b), u32::from(d))
                    .map_err(|e| bad(e, op))?,
            );
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a,
            b,
            d,
        } => {
            out.push(
                tt_isa::arch::blackhole::sfpu::mul(u32::from(a), u32::from(b), u32::from(d))
                    .map_err(|e| bad(e, op))?,
            );
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::BitAnd,
            a,
            b,
            d,
        } => {
            out.push(encode::sfpand(u32::from(b), u32::from(a), u32::from(d), 1).map_err(bad_isa)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::BitOr,
            a,
            b,
            d,
        } => {
            out.push(encode::sfpor(u32::from(b), u32::from(a), u32::from(d), 1).map_err(bad_isa)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::BitXor,
            a,
            b,
            d,
        } => {
            out.push(encode::sfpmov(u32::from(a), u32::from(d), 0).map_err(bad_isa)?);
            out.push(encode::sfpxor(u32::from(b), u32::from(d)).map_err(bad_isa)?);
        }
        MirOp::UnarySfp {
            op: UnarySfpOp::BitNot,
            a,
            d,
        } => {
            out.push(encode::sfpnot(u32::from(a), u32::from(d)).map_err(bad_isa)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::Shl,
            a,
            b,
            d,
        } => {
            // `SFPSHFT` shifts its destination in place, so the value is moved
            // in first and the amount's lane is the `VC` operand.
            out.push(encode::sfpmov(u32::from(a), u32::from(d), 0).map_err(bad_isa)?);
            out.push(encode::sfpshft(0, u32::from(b), u32::from(d), 0).map_err(bad_isa)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::Shr,
            a,
            b,
            d,
        } => {
            out.push(encode::sfpmov(u32::from(a), u32::from(d), 0).map_err(bad_isa)?);
            // A negative amount is the logical right shift; `SFPIADD`'s
            // two's-complement form negates the amount in its own lane
            // (`VC - VD`, with `VC` the integer zero), the same step zyx's
            // `tt_sfp_integer` takes.
            out.push(
                encode::sfpiadd(0, tt_isa::arch::blackhole::sfpu::LREG_ZERO, u32::from(b), 6)
                    .map_err(bad_isa)?,
            );
            out.push(encode::sfpshft(0, u32::from(b), u32::from(d), 0).map_err(bad_isa)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::IAdd,
            a,
            b,
            d,
        } => {
            // `SFPIADD` adds into its destination, so one operand is moved in:
            // `VC + VD`. Modifier 4 is `CC_NONE`: the lane flags are not this
            // instruction's business.
            out.push(encode::sfpmov(u32::from(b), u32::from(d), 0).map_err(bad_isa)?);
            out.push(encode::sfpiadd(0, u32::from(a), u32::from(d), 4).map_err(bad_isa)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::ISub,
            a,
            b,
            d,
        } => {
            // `VC - VD` in two's complement, with the flags left alone (6 is
            // `ARG_2SCOMP_LREG_DST | CC_NONE`).
            out.push(encode::sfpmov(u32::from(b), u32::from(d), 0).map_err(bad_isa)?);
            out.push(encode::sfpiadd(0, u32::from(a), u32::from(d), 6).map_err(bad_isa)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::Setexp,
            a,
            b,
            d,
        } => {
            // The exponent comes from the destination lane's low eight bits, so
            // it is moved in first; the sign and mantissa then come from `a`.
            out.push(encode::sfpmov(u32::from(b), u32::from(d), 0).map_err(bad_isa)?);
            out.push(encode::sfpsetexp(0, u32::from(a), u32::from(d), 0).map_err(bad_isa)?);
        }
        MirOp::SfpGt { mod1, vc, vd } => {
            out.push(encode::sfpgt(u32::from(vc), u32::from(vd), mod1).map_err(bad_isa)?);
        }
        MirOp::SfpLe { mod1, vc, vd } => {
            out.push(encode::sfple(u32::from(vc), u32::from(vd), mod1).map_err(bad_isa)?);
        }
        MirOp::SfpMov { a, d } => {
            out.push(encode::sfpmov(u32::from(a), u32::from(d), 0).map_err(bad_isa)?);
        }
        MirOp::SfpAbs { mod1, a, d } => {
            out.push(encode::sfpabs(u32::from(a), u32::from(d), mod1).map_err(bad_isa)?);
        }
        MirOp::SfpCast { mod1, a, d } => {
            out.push(encode::sfpcast(u32::from(a), u32::from(d), mod1).map_err(bad_isa)?);
        }
        MirOp::SfpExexp { mod1, a, d } => {
            out.push(encode::sfpexexp(u32::from(a), u32::from(d), mod1).map_err(bad_isa)?);
        }
        MirOp::SfpExman { mod1, a, d } => {
            out.push(encode::sfpexman(u32::from(a), u32::from(d), mod1).map_err(bad_isa)?);
        }
        MirOp::SfpIadd { imm, a, d, mod1 } => {
            out.push(encode::sfpiadd(imm, u32::from(a), u32::from(d), mod1).map_err(bad_isa)?);
        }
        MirOp::SfpShft { mod1, amount, a, d } => {
            // `SFPSHFT` shifts its destination in place, so the value moves in
            // first and the amount's lane is `VC`.
            out.push(encode::sfpmov(u32::from(a), u32::from(d), 0).map_err(bad_isa)?);
            out.push(encode::sfpshft(0, u32::from(amount), u32::from(d), mod1).map_err(bad_isa)?);
        }
        MirOp::SfpSetcc { mod1, a } => {
            // `VD` is only the backdoor-load gate, not a destination.
            out.push(encode::sfpsetcc(0, u32::from(a), 0, mod1).map_err(bad_isa)?);
        }
        MirOp::SfpEncc {
            first,
            second,
            mod1,
        } => {
            out.push(encode::sfpencc(first, second, mod1).map_err(bad_isa)?);
        }
        MirOp::SfpArecip { mod1, a, d } => {
            // `VC` is the value; `VB` is only read by the conditional mode, so
            // it is left zero. The simulator refuses a non-zero `VB` in the
            // unconditional mode (`tensix_sfparecip: imm12_math=4`), which is
            // what made the first `mod` attempt abort.
            out.push(encode::sfparecip(0, u32::from(a), u32::from(d), mod1).map_err(bad_isa)?);
        }
        MirOp::SfpMul24 { upper, a, b, d } => {
            // `VC` must be the constant zero for the multiply to be a plain
            // 24x24 product (`SFPMUL24.md`: the other modes are noncontractual).
            out.push(
                encode::Sfpmul24::ZERO
                    .va(u32::from(a))
                    .vb(u32::from(b))
                    .vc(tt_isa::arch::blackhole::sfpu::LREG_ZERO)
                    .vd(u32::from(d))
                    .mod1(u32::from(upper))
                    .encode()
                    .map_err(bad_isa)?,
            );
        }
        MirOp::SfpMad { mod1, a, b, c, d } => {
            out.push(
                tt_isa::arch::blackhole::sfpu::mad(
                    u32::from(a),
                    u32::from(b),
                    u32::from(c),
                    u32::from(d),
                    mod1,
                )
                .map_err(|e| bad(e, op))?,
            );
        }
        MirOp::SfpStochRnd { mod1, a, d } => {
            // `VC` is the value. `VB` is the stalling work-around the page
            // describes (repeat `VC`), but the simulator refuses anything but
            // zero there, and it is not read for the result.
            out.push(
                encode::Sfpstochrnd::ZERO
                    .rounding_mode(0)
                    .vb(0)
                    .vc(u32::from(a))
                    .vd(u32::from(d))
                    .mod1(mod1)
                    .encode()
                    .map_err(bad_isa)?,
            );
        }
        MirOp::Mvmul {
            reg,
            at,
            addr_mod,
            flip_a,
            flip_b,
        } => {
            let i = encode::Mvmul::ZERO
                .dst_row(dst_row(reg, at, row_delta))
                .addr_mod(addr_mod)
                .flip_src_a(u32::from(flip_a))
                .flip_src_b(u32::from(flip_b))
                .broadcast_src_b_row(0)
                .encode()
                .map_err(bad_isa)?;
            out.push(i);
        }
        MirOp::Elw {
            op,
            bcast,
            addr_mod,
            reg,
            at,
        } => {
            // The two broadcast bits are the instruction's: a row broadcast
            // reads `SrcB`'s row 0, a column broadcast its column 0, and both
            // together are the one-datum scalar broadcast.
            let row = dst_row(reg, at, row_delta);
            let (brow, col) = match bcast {
                Bcast::None => (0, 0),
                Bcast::Row => (1, 0),
                Bcast::Col => (0, 1),
                Bcast::Scalar => (1, 1),
            };
            let encoded = match op {
                ElwOp::Add => encode::Elwadd::ZERO
                    .add_dst(0)
                    .broadcast_src_b_row(brow)
                    .broadcast_src_b_col0(col)
                    .addr_mod(addr_mod)
                    .dst_row(row)
                    .encode(),
                ElwOp::Sub => encode::Elwsub::ZERO
                    .add_dst(0)
                    .broadcast_src_b_row(brow)
                    .broadcast_src_b_col0(col)
                    .addr_mod(addr_mod)
                    .dst_row(row)
                    .encode(),
                ElwOp::Mul => encode::Elwmul::ZERO
                    .broadcast_src_b_row(brow)
                    .broadcast_src_b_col0(col)
                    .addr_mod(addr_mod)
                    .dst_row(row)
                    .encode(),
            };
            out.push(encoded.map_err(bad_isa)?);
        }
        MirOp::Gapool {
            flip_src_b,
            flip_src_a,
            addr_mod,
            reg,
            at,
        } => {
            out.push(
                encode::gapool(
                    u32::from(flip_src_b),
                    u32::from(flip_src_a),
                    addr_mod,
                    dst_row(reg, at, row_delta),
                )
                .map_err(bad_isa)?,
            );
        }
        MirOp::Gmpool {
            flip_src_b,
            flip_src_a,
            addr_mod,
            arg_max,
            reg,
            at,
        } => {
            out.push(
                encode::Gmpool::ZERO
                    .flip_src_b(u32::from(flip_src_b))
                    .flip_src_a(u32::from(flip_src_a))
                    .addr_mod(addr_mod)
                    .arg_max(arg_max)
                    .dst_row(dst_row(reg, at, row_delta))
                    .encode()
                    .map_err(bad_isa)?,
            );
        }
        MirOp::Movd2b {
            use_dst32b_lo,
            src_row,
            addr_mod,
            move4_rows,
            reg,
            at,
        } => {
            out.push(
                encode::Movd2B::ZERO
                    .use_dst32b_lo(u32::from(use_dst32b_lo))
                    .src_row(src_row)
                    .addr_mod(addr_mod)
                    .move4_rows(u32::from(move4_rows))
                    .dst_row(dst_row(reg, at, row_delta))
                    .encode()
                    .map_err(bad_isa)?,
            );
        }
        MirOp::Movb2d {
            use_dst32b_lo,
            src_row,
            addr_mod,
            move4_rows,
            reg,
            at,
        } => {
            out.push(
                encode::Movb2D::ZERO
                    .use_dst32b_lo(u32::from(use_dst32b_lo))
                    .src_row(src_row)
                    .addr_mod(addr_mod)
                    .move4_rows(u32::from(move4_rows))
                    .dst_row(dst_row(reg, at, row_delta))
                    .encode()
                    .map_err(bad_isa)?,
            );
        }
        MirOp::Movb2a {
            src_a_row,
            addr_mod,
            move4_rows,
            src_b_row,
        } => {
            out.push(
                encode::movb2_a(src_a_row, addr_mod, u32::from(move4_rows), src_b_row)
                    .map_err(bad_isa)?,
            );
        }
        MirOp::Mova2d {
            use_dst32b_lo,
            src_row,
            addr_mod,
            move8_rows,
            reg,
            at,
        } => {
            out.push(
                encode::Mova2D::ZERO
                    .use_dst32b_lo(u32::from(use_dst32b_lo))
                    .src_row(src_row)
                    .addr_mod(addr_mod)
                    .move8_rows(u32::from(move8_rows))
                    .dst_row(dst_row(reg, at, row_delta))
                    .encode()
                    .map_err(bad_isa)?,
            );
        }
        MirOp::Trnspsrcb => {
            out.push(encode::trnspsrcb().map_err(bad_isa)?);
        }
        MirOp::Zerosrc {
            zero_val,
            write_mode,
            bank_mask,
            clear_src_b,
            clear_src_a,
        } => {
            out.push(
                encode::Zerosrc::ZERO
                    .negative_inf_src_a(u32::from(zero_val))
                    .single_bank_matrix_unit(u32::from(write_mode))
                    .both_banks(u32::from(bank_mask))
                    .clear_src_b(u32::from(clear_src_b))
                    .clear_src_a(u32::from(clear_src_a))
                    .encode()
                    .map_err(bad_isa)?,
            );
        }
        MirOp::AluAccCtrl {
            fp32,
            zero_flag_src,
        } => {
            // The `Dst` width the moves address and the denormal policy of the
            // `Src` reads: tt-metal clears `Fp32_enabled` around the high-half
            // `MOVD2B`/`MOVB2D` transposes on Blackhole (issue #449), and keeps
            // source denormals so the low-half bits survive the moves.
            let mut words = ConfigWords::new();
            config_set(&mut words, alu::ALU_ACC_CTRL_Fp32_enabled, u32::from(fp32))?;
            config_set(
                &mut words,
                alu::ALU_ACC_CTRL_Zero_Flag_disabled_src,
                u32::from(zero_flag_src),
            )?;
            out.extend(config_program(&words)?);
        }
        MirOp::SrcAFormat { format } => {
            // The moves take `SrcA`'s format while
            // `DISABLE_IMPLIED_SRCA_FMT` is set; the override pair is the
            // field the format switch writes.
            let mut words = ConfigWords::new();
            config_set(&mut words, alu::ALU_FORMAT_SPEC_REG_SrcA_override, 1)?;
            config_set(&mut words, alu::ALU_FORMAT_SPEC_REG_SrcA_val, format)?;
            out.extend(config_program(&words)?);
        }
        MirOp::Setrwc {
            src_a,
            src_b,
            dst,
            flip_src_a,
            flip_src_b,
            dst_cr,
        } => {
            let i = encode::Setrwc::ZERO
                .src_a(1)
                .src_a_val(src_a)
                .src_b(1)
                .src_b_val(src_b)
                .dst(1)
                .dst_val(dst)
                .fidelity(1)
                .flip_src_a(u32::from(flip_src_a))
                .flip_src_b(u32::from(flip_src_b))
                .dst_cr(u32::from(dst_cr))
                .encode()
                .map_err(bad_isa)?;
            out.push(i);
        }
        MirOp::Incrwc { src_a, src_b, dst } => {
            out.push(
                encode::Incrwc::ZERO
                    .dst_cr(0)
                    .src_b_cr(0)
                    .src_a_cr(0)
                    .dst_inc(dst)
                    .src_b_inc(src_b)
                    .src_a_inc(src_a)
                    .encode()
                    .map_err(bad_isa)?,
            );
        }
        MirOp::ThreadWord { addr, value } => {
            // A whole `ThreadConfig` word: `SETC16` carries the composed value
            // and `0` leaves every other field of the entry clear.
            out.push(backend::set_thread_entry(addr, value).map_err(|e| bad(e, op))?);
        }
        MirOp::WaitMatrix => {
            out.push(backend::wait_for_matrix(Before::EVERYTHING).map_err(|e| bad(e, op))?);
        }
        MirOp::SemWaitFree { sem } => {
            out.push(sync::wait_below_max(semaphore(sem)?, Before::EVERYTHING));
        }
        MirOp::SemPost { sem } => {
            // Both the SFPU and the Matrix Unit (the FPU's `ELW`s and the
            // matmul/pool ops) write `Dst`, so both pipelines drain before the
            // section is announced; a post waiting on the SFPU alone lets an
            // `ELW`'s result be packed while it is still in flight.
            out.extend(post_after_drained(
                cond::SFPU_BUSY | cond::MATRIX_BUSY,
                semaphore(sem)?,
            )?);
        }
        MirOp::SemWait { sem, .. } => {
            let s = semaphore(sem)?;
            out.extend(sync::take(s, Before::EVERYTHING));
        }
        MirOp::SemWaitNoGet { sem, .. } => {
            out.push(sync::wait_nonzero(semaphore(sem)?, Before::EVERYTHING));
        }
        MirOp::SemGet { sem } => out.push(sync::get(semaphore(sem)?)),
        other => {
            let _ = other;
            return Err(ttStatus::TT_ERROR_NOT_MATH);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The pack core (T2)
// ---------------------------------------------------------------------------

fn encode_pack(ops: &[MirOp], out: &mut Stream, dst_is_16_bit: bool) -> Result<(), ttStatus> {
    if ops.is_empty() {
        return Ok(());
    }
    // The `ThreadConfig` entries come first, as at the start of every tt-metal
    // kernel: `CFG_STATE_ID` decides which `Config` bank later writes land in, so
    // no config store may race with them.
    thread_config(out)?;
    // `a0` = the launch arguments; word 2 is the CB interface table.
    out.rv(rv::lw(reg::A6, reg::A0, crate::mover::ARG_IFACE as i32 * 4));
    // What the packer's `Config` bank and its `Dst` offset already hold, so a
    // pack that wants the same values again writes nothing. Both caches are
    // dropped at a loop head, where the previous execution is the loop's own
    // tail rather than the instruction before it (the unpacker's
    // `UnpackState.cfg` keeps the same rule).
    let mut last_config: Option<(DType, PackMask)> = None;
    let mut last_offset: Option<(Dst, u32)> = None;
    let mut loops = Vec::new();
    for op in ops {
        if encode_int(op, out, &mut loops, &COMPUTE_PAIRS)? {
            continue;
        }
        match *op {
            MirOp::Loop { count } => {
                last_config = None;
                last_offset = None;
                open_loop(out, &mut loops, count)?;
            }
            MirOp::EndLoop => close_loop(out, &mut loops)?,
            MirOp::SemWait { sem, .. } => {
                out.extend(sync::take(semaphore(sem)?, Before::EVERYTHING));
            }
            MirOp::SemWaitNoGet { sem, .. } => {
                out.push(sync::wait_nonzero(semaphore(sem)?, Before::EVERYTHING));
            }
            MirOp::SemGet { sem } => out.push(sync::get(semaphore(sem)?)),
            MirOp::SemPost { sem } => {
                out.extend(sync::post_after(sync::Unit::Packer, semaphore(sem)?));
            }
            MirOp::SemWaitFree { sem } => {
                out.push(sync::wait_below_max(semaphore(sem)?, Before::EVERYTHING));
            }
            MirOp::CbReserve { cb, n } => {
                out.asm(|a| {
                    rv::cb::reserve_back(a, reg::A6, u32::from(cb), n, [reg::T0, reg::T1, reg::T2])
                })?;
            }
            MirOp::CbPush { cb, n } => {
                // The page is only visible to the writer once the pack has
                // written it: the `STALLWAIT` holds back later pushed work, and
                // the coprocessor check holds this core's store back until the
                // packer has retired and its L1 writes have landed.
                out.push(wait_for_packer(Before::EVERYTHING).map_err(|e| {
                    crate::status::internal_bug(format_args!("encode: packer wait: {e}"))
                })?);
                out.asm(|a| {
                    rv::coprocessor_done(a);
                    Ok(())
                })?;
                out.asm(|a| {
                    rv::cb::push_back(
                        a,
                        reg::A6,
                        u32::from(cb),
                        n,
                        [reg::T0, reg::T1, reg::T2, reg::T6],
                    )
                })?;
            }
            MirOp::SetPackBase { cb, offset, extra } => {
                // The destination is the ring's write pointer, converted to
                // 16-byte units minus the header unit the packer adds.
                cb_base_pack(out, cb, offset, extra)?;
            }
            MirOp::SetPackConfig { dtype, mask } => {
                // A bf16 page out of a 32-bit `Dst` is the packer's own early
                // conversion (`pack_fmt_conv_mode = 0x105`, tt-metal's
                // `is_packer_to_L1_early_conversion_supported`: `Float32` ->
                // `Float16_b`). `pack_config` already writes the two formats and
                // `Read_32b_data`, so the mode is the one the hardware takes;
                // `ttsim` refuses to model it (`TENSIX_EXECUTE_PACR`'s accepted
                // list has no `0x105`), so a launch would abort there. The
                // compiler still emits it: the silicon path exists and only the
                // simulator cannot run it.
                if last_config == Some((dtype, mask)) {
                    continue;
                }
                let mut words = ConfigWords::new();
                pack_config(&mut words, dtype, dst_is_16_bit, mask)?;
                out.extend(config_program(&words)?);
                out.push(stallwait(block::PACKER | block::CONFIG, cond::CONFIG_BUSY)?);
                last_config = Some((dtype, mask));
            }
            MirOp::SetPackOffset { reg, at } => {
                if last_offset == Some((reg, at)) {
                    continue;
                }
                let mut words = ConfigWords::new();
                config_set(
                    &mut words,
                    global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset,
                    dst_row(reg, at, 0),
                )?;
                out.extend(config_program(&words)?);
                out.push(stallwait(block::PACKER | block::CONFIG, cond::CONFIG_BUSY)?);
                last_offset = Some((reg, at));
            }
            MirOp::PackRows { rows, last } => {
                if rows == 0 {
                    return Err(ttStatus::TT_ERROR_PACK_ZERO_ROWS);
                }
                pack_rows(out, rows, last)?;
            }
            other => {
                let _ = other;
                return Err(ttStatus::TT_ERROR_PACK_OP);
            }
        }
    }
    if !loops.is_empty() {
        return Err(ttStatus::TT_ERROR_ENDLOOP_MACHINE);
    }
    Ok(())
}

/// The packer's `Config` for one unit: read `Dst` as 32-bit, write FP32 to the
/// ring's page. The output address is in 16-byte units and the generator always
/// adds the one 16-byte tile header unit it never writes; the base store takes
/// the same `- 1`.
///
/// The `Dst` row is *not* written here: `DEST_TARGET_REG_CFG_PACK_SEC0_Offset`
/// is [`MirOp::SetPackOffset`], so a pack changes one word of the bank rather
/// than all of it, and the fields below (formats, strides, the edge mask) are
/// only rewritten when a pack's dtype or mask differs from the last one.
/// (`Dst` addresses are 16-word rows, and tt-metal's own values for that field —
/// 0 and `DEST_REGISTER_HALF_SIZE`, 512 — are row counts. tt-metal splits the
/// same job: the field holds the half base, and `set_dst_write_addr(tile_index)`
/// puts the tile within it in the packer's `W` address counter
/// (`TT_SETADC(PAC, CH_0, SET_W, tile_index)`). The simulator does not model
/// that counter — a `SETADC` there leaves the pack reading row 0 — so the row is
/// written in that field, which is also what zyx did.)
fn pack_config(
    words: &mut ConfigWords,
    dtype: DType,
    dst_is_16_bit: bool,
    mask: PackMask,
) -> Result<(), ttStatus> {
    // `Out_data_format` is the L1 datum's own format; `In_data_format` is the
    // kernel's `Dst` width. In 16-bit mode the packer reads a 16-bit `Dst` (two
    // bytes per datum, so half the row stride) and its conversion is the
    // identity; in 32-bit mode it reads a 32-bit one and converts, which is how
    // a bf16 buffer is written out of an f32 `Dst`.
    let out = l1_format(dtype);
    let in_ = dst_format(dst_is_16_bit);
    let narrow = dst_is_16_bit;
    config_set(words, thcon::THCON_SEC0_REG1_In_data_format, in_)?;
    config_set(words, thcon::THCON_SEC0_REG1_Out_data_format, out)?;
    config_set(words, thcon::THCON_SEC0_REG1_Source_interface_selection, 0)?;
    config_set(words, thcon::THCON_SEC0_REG1_Disable_zero_compress, 1)?;
    config_set(words, thcon::THCON_SEC0_REG1_Add_l1_dest_addr_offset, 0)?;
    config_set(words, thcon::THCON_SEC0_REG1_Enable_out_fifo, 0)?;
    config_set(words, thcon::THCON_SEC0_REG1_Exp_section_size, 0)?;
    config_set(words, thcon::THCON_SEC0_REG1_Row_start_section_size, 0)?;
    config_set(words, thcon::THCON_SEC0_REG1_Downsample_rate, 0)?;
    config_set(words, thcon::THCON_SEC0_REG1_Downsample_mask, 0)?;
    config_set(words, thcon::THCON_SEC0_REG1_Exp_threshold_en, 0)?;
    config_set(words, thcon::THCON_SEC0_REG1_Pack_L1_Acc, 0)?;
    config_set(words, thcon::THCON_SEC0_REG1_pack_start_intf_pos, 0)?;
    config_set(words, pack0::PCK0_ADDR_BASE_REG_1_Base, 0)?;
    config_set(words, pack0::PCK0_ADDR_CTRL_XY_REG_1_Ystride, 0)?;
    config_set(words, pack0::PCK0_ADDR_CTRL_ZW_REG_1_Zstride, 0)?;
    config_set(words, pack0::PCK0_ADDR_CTRL_XY_REG_0_Xstride, 0)?;
    config_set(
        words,
        pack0::PCK0_ADDR_CTRL_XY_REG_0_Ystride,
        if narrow {
            DST_ROW_BYTES / 2
        } else {
            DST_ROW_BYTES
        },
    )?;
    config_set(words, pack0::PCK0_ADDR_CTRL_ZW_REG_0_Zstride, 0)?;
    config_set(
        words,
        pack0::PCK_DEST_RD_CTRL_Read_32b_data,
        u32::from(!narrow),
    )?;
    config_set(words, pack0::PCK_DEST_RD_CTRL_Read_unsigned, 0)?;
    config_set(words, pack0::PCK_DEST_RD_CTRL_Read_int8, 0)?;
    config_set(words, pack0::PCK_DEST_RD_CTRL_Round_10b_mant, 0)?;
    config_set(words, pack0::PACK_COUNTERS_SEC0_pack_reads_per_xy_plane, 16)?;
    config_set(words, pack0::PCK_EDGE_MODE_mode, 0)?;
    config_set(
        words,
        pack0::PCK_EDGE_OFFSET_SEC0_mask,
        match mask {
            PackMask::None => 0xFFFF,
            PackMask::Col | PackMask::Row => 0x0000,
        },
    )?;
    config_set(
        words,
        pack0::PCK_EDGE_OFFSET_SEC1_mask,
        match mask {
            PackMask::None => 0xFFFF,
            PackMask::Col => 0xFFFF,
            // tt-metal's `REDUCE_ROW`: only datum 0 of the row survives.
            PackMask::Row => 0x0001,
        },
    )?;
    config_set(
        words,
        pack0::PCK_EDGE_TILE_ROW_SET_SELECT_select,
        match mask {
            PackMask::None => 0,
            // Each packer's two-bit field is the mapping register it
            // reads: packers 0 and 1 both use `TILE_ROW_SET_MAPPING_1`.
            PackMask::Col => 0b0101,
            // `REDUCE_ROW` gives every packer mapping 1.
            PackMask::Row => 0b0101_0101,
        },
    )?;
    config_set(
        words,
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_0,
        match mask {
            PackMask::None => 0,
            PackMask::Col => 1,
            // Every one of a face's sixteen rows selects the `SEC1`
            // mask, i.e. keeps column 0.
            PackMask::Row => 1,
        },
    )?;
    config_set(words, alu::ALU_ACC_CTRL_Fp32_enabled, u32::from(!narrow))?;
    if mask == PackMask::Row {
        for field in [
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_1,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_2,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_3,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_4,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_5,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_6,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_7,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_8,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_9,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_10,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_11,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_12,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_13,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_14,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_15,
        ] {
            config_set(words, field, 1)?;
        }
    }
    Ok(())
}

/// Pack `rows` rows of `Dst` from the offset [`MirOp::SetPackOffset`] set, to
/// the page [`MirOp::SetPackBase`] named.
///
/// The address generators' per-program setup (`ADDR_MOD_PACK_*_YsrcIncr`,
/// `SETADCXX`) is in `encode_pack`'s prologue, like every other invariant
/// `Config` word; only the Y reset is per pack, because the `PACR`s leave Y at
/// the end of the run they moved.
fn pack_rows(out: &mut Stream, rows: u32, last: bool) -> Result<(), ttStatus> {
    out.push(thread_entry(thread::ADDR_MOD_PACK_SEC0_YsrcIncr, 0)?);
    out.push(thread_entry(thread::ADDR_MOD_PACK_SEC1_YsrcIncr, 4)?);
    out.push(
        encode::Setadcxx::ZERO
            .pk(1)
            .x0_val(0)
            .x1_val(15)
            .encode()
            .map_err(|e| {
                let _ = e;
                ttStatus::TT_ERROR_SETADCXX_PACK
            })?,
    );
    out.push(
        encode::Setadcxy::ZERO
            .pk(1)
            .y0(1)
            .y0_val(0)
            .encode()
            .map_err(|e| {
                let _ = e;
                ttStatus::TT_ERROR_SETADCXY_PACK
            })?,
    );
    let groups = rows.div_ceil(4);
    for group in 0..groups {
        let remaining = rows - 4 * group;
        let mask = if remaining >= 4 {
            0b1111
        } else {
            (1 << remaining) - 1
        };
        let is_last = last && group + 1 == groups;
        out.push(
            encode::Pacr::ZERO
                .read_intf_sel(mask)
                .addr_mod(if is_last { 0 } else { PACK_ADDR_MOD_NEXT_GROUP })
                .last(u32::from(is_last))
                .encode()
                .map_err(|e| {
                    let _ = e;
                    ttStatus::TT_ERROR_PACR
                })?,
        );
    }
    Ok(())
}

/// Wait for the packer to drain, then post. Used by a program that has to know
/// its own output is complete.
#[allow(dead_code)]
fn pack_done(sem: Semaphore) -> Result<[Instruction; 2], ttStatus> {
    Ok([
        wait_for_packer(Before::EVERYTHING)
            .map_err(|e| crate::status::internal_bug(format_args!("encode: packer wait: {e}")))?,
        sync::post(sem),
    ])
}

#[allow(dead_code)]
fn unpack_done() -> Result<Instruction, ttStatus> {
    wait_for_unpacker0(Before::EVERYTHING)
        .map_err(|e| crate::status::internal_bug(format_args!("encode: unpacker wait: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A packer program with `packs` identical packs, each of the same 32x32
    /// `f32` tile shape with a different `Dst` row.
    fn packs(count: u8) -> MachineKernel {
        let mut m = MachineKernel::default();
        for pack in 0..count {
            let at = u32::from(pack) * 2;
            m.push(Core::Pack, MirOp::CbReserve { cb: Cb(0), n: 1 });
            m.push(
                Core::Pack,
                MirOp::SetPackBase {
                    cb: Cb(0),
                    offset: 0,
                    extra: None,
                },
            );
            m.push(
                Core::Pack,
                MirOp::SetPackConfig {
                    dtype: DType::F32,
                    mask: PackMask::None,
                },
            );
            m.push(Core::Pack, MirOp::SetPackOffset { reg: Dst(0), at });
            m.push(
                Core::Pack,
                MirOp::PackRows {
                    rows: Dst::ROWS,
                    last: true,
                },
            );
            m.push(Core::Pack, MirOp::CbPush { cb: Cb(0), n: 1 });
        }
        m
    }

    fn pack_words(count: u8) -> usize {
        let words = encode(&packs(count), false)
            .expect("the packer program encodes")
            .words()
            .expect("the words are inline");
        words[3].len()
    }

    /// The packer's `Config` bank is written once for a run of packs that share
    /// a format and edge mask: a pack that changes only its `Dst` row rewrites
    /// that one field, not the 32 the bank holds (`pack_config`). This is the
    /// point of the split into `SetPackConfig`/`SetPackOffset`/`PackRows`; with
    /// one `Pack` op every pack paid for the whole bank.
    #[test]
    fn a_second_pack_rewrites_only_the_row() {
        let one = pack_words(1);
        let two = pack_words(2);
        let whole_bank = pack_config_words() * 3 + 1;
        // Measured: a pack is 131 words, a second identical one 82 (its base,
        // its row, the Y reset and the sixteen `PACR`s), and the bank alone is
        // 130 — so a run of packs no longer pays for the bank per pack.
        let second = two - one;
        assert!(
            second < whole_bank,
            "the second pack costs {second} words, the bank alone is {whole_bank}"
        );
    }

    /// The words `pack_config` seeds, i.e. what a full bank write costs before
    /// the `SETDMAREG` pairs are counted.
    fn pack_config_words() -> usize {
        let mut words = ConfigWords::new();
        pack_config(&mut words, DType::F32, false, PackMask::None);
        words.program_len()
    }

    /// The one thing the two target variants differ in: the unpacker's tile
    /// descriptor. Words 2 and 3 are zero for a tile this compiler stages, so
    /// the simulator's program omits them (`ttsim` refuses the write) and the
    /// card's writes them (`Config` outlives the program that wrote it). Both
    /// config paths — `Dst` and `Src` — must agree.
    #[test]
    fn the_tile_descriptor_tail_is_written_only_for_the_card() {
        let base = thcon::THCON_SEC0_REG0_TileDescriptor.addr32();
        for tail in [false, true] {
            let mut dst = ConfigWords::new();
            unpack_config(&mut dst, 1024, 0, DType::F32, false, tail).unwrap();
            let mut src = ConfigWords::new();
            src_unpack_config(&mut src, 0, DType::F32, false, false, 0, tail).unwrap();
            for (name, words) in [("dst", &dst), ("src", &src)] {
                let tail_words: Vec<(u16, u32)> = words
                    .words()
                    .iter()
                    .copied()
                    .filter(|&(at, _)| at == base + 2 || at == base + 3)
                    .collect();
                if tail {
                    assert_eq!(
                        tail_words,
                        [(base + 2, 0), (base + 3, 0)],
                        "{name}: the card needs the whole descriptor"
                    );
                } else {
                    assert!(
                        tail_words.is_empty(),
                        "{name}: the simulator's variant omits the all-zero tail"
                    );
                }
            }
        }
    }
}
