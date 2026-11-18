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
    Bcast, BinarySfpOp, Cb, Core, Dst, ElwOp, IOp, MachineKernel, MirOp, PackMask, Target,
    UnarySfpOp,
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
use tt_isa::arch::quasar::issue;
use tt_isa::arch::quasar::{Quasar, backend as qbackend, sfpu as qsfpu};
use tt_isa::cfg::blackhole::{alu, global, pack0, thcon, thread, unpack0, unpack1};
use tt_isa::cfg::quasar::{alu as qalu, thcon as qthcon};
use tt_isa::isa::Instruction;
use tt_isa::isa::blackhole::{defs, encode};
use tt_isa::isa::quasar::encode as qencode;

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
enum Elem {
    Tensix(Instruction),
    Riscv(u32),
}

/// A role's program under construction.
#[derive(Debug, Clone, Default)]
pub struct Stream {
    elems: Vec<Elem>,
}

impl Stream {
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

    /// The encoded words, ending in `ret`. An empty program stays empty: it is
    /// not started, and a lone `ret` would make it look non-empty.
    fn words(&self) -> Result<Vec<u32>, ttStatus> {
        if self.elems.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(self.elems.len() + 1);
        for elem in &self.elems {
            out.push(match *elem {
                Elem::Tensix(i) => i.ttinsn_word().ok_or_else(|| {
                    let _ = i;
                    ttStatus::TT_ERROR_INSTRUCTION_NOT_INLINE
                })?,
                Elem::Riscv(w) => w,
            });
        }
        out.push(crate::riscv::ret());
        Ok(out)
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

/// A loop's `(index, limit)` pair per nesting level: the compute roles count in
/// `t4` against `t5`, then `a7` against `a1`; the movers in `a1`/`a2` then
/// `a3`/`a4` (their `t5` is the NIU base).
const COMPUTE_PAIRS: [(u8, u8); 2] = [(reg::T4, reg::T5), (reg::A7, reg::A1)];
const MOVER_PAIRS: [(u8, u8); 2] = [(reg::A1, reg::A2), (reg::A3, reg::A4)];

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
    let (index, limit) = *pairs.get(loops.len()).ok_or(ttStatus::TT_TODO_LOOP_DEPTH)?;
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
    let (index, limit) = *pairs.get(loops.len()).ok_or(ttStatus::TT_TODO_LOOP_DEPTH)?;
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
    let niu = if outbound { Niu::Noc1 } else { Niu::Noc0 };
    out.asm(|a| {
        if m.target == Target::Blackhole {
            // `t5` is the initiator's register block, `t3` this tile's own NoC
            // word; a Quasar mover has neither, so it only reads its ring table.
            a.li(reg::T5, niu.base() as u32);
            a.word(rv::lw(reg::T3, reg::A0, 12));
        }
        a.word(rv::lw(reg::A6, reg::A0, crate::mover::ARG_IFACE as i32 * 4));
        Ok(())
    })?;
    let mut loops = Vec::new();
    for op in &ops {
        if encode_int(op, out, &mut loops, &MOVER_PAIRS)? {
            continue;
        }
        match *op {
            MirOp::Loop { count } => open_loop_with(out, &mut loops, count, &MOVER_PAIRS)?,
            MirOp::EndLoop => close_loop(out, &mut loops)?,
            // A word store, and the NoC's spin: one instruction and one fixed
            // pair, so neither is a macro op.
            MirOp::Sw { value, base, at } => out.rv(rv::sw(value, base, at)),
            MirOp::NiuPoll { reg, field } => {
                let top = out.len();
                out.rv(rv::lw(reg, reg::T5, field));
                let here = out.len();
                out.rv(rv::bne(reg, reg::ZERO, ((top as i32) - (here as i32)) * 4));
            }
            // A Quasar mover's NoC: the command buffer is written through RoCC
            // instructions, and the completion test is the overlay's own counter.
            MirOp::CmdbufWrReg {
                buffer,
                register,
                value,
            } => out.rv(issue::cmdbuf_wr_reg(buffer, register, value)),
            MirOp::CmdbufIssue { buffer } => out.rv(issue::cmdbuf_issue(buffer)),
            MirOp::RoccPoll { write, reg } => {
                let top = out.len();
                out.rv(if write {
                    issue::scmdbuf_wr_sent(reg)
                } else {
                    issue::scmdbuf_tr_ack(reg)
                });
                let here = out.len();
                out.rv(rv::bne(reg, reg::ZERO, ((top as i32) - (here as i32)) * 4));
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
                open_loop(out, &mut loops, count)?;
            }
            MirOp::EndLoop => close_loop(out, &mut loops)?,
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
            MirOp::SetUnpackBase { cb, unit } => {
                // The page is at the ring's read pointer, not a compile-time L1
                // address: write that pointer into the unpacker's base register
                // after draining the previous unpack.
                cb_base_unpack(out, cb, unit)?;
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
                out.asm(|a| {
                    rv::cb::wait_front(a, reg::A6, u32::from(cb), n, [reg::T0, reg::T1, reg::T2])
                })?;
            }
            MirOp::CbPop { cb, n } => {
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

/// The unpacker's read base: the ring's read pointer in 16-byte units, minus the
/// header unit the unpacker adds (`UNPACR_Regular.md`: `InAddr = (Base_address +
/// 1 + DigestSize) * 16`).
fn cb_base_unpack(out: &mut Stream, cb: Cb, unit: u8) -> Result<(), ttStatus> {
    out.push(wait_for_unpacker0(Before::EVERYTHING).map_err(|_| ttStatus::TT_ERROR_STALLWAIT)?);
    if unit == 1 {
        out.push(
            tt_isa::arch::blackhole::backend::wait_for_unpacker1(Before::EVERYTHING)
                .map_err(|_| ttStatus::TT_ERROR_STALLWAIT)?,
        );
    }
    let config = match unit {
        0 => thcon::THCON_SEC0_REG3_Base_address.addr32(),
        1 => thcon::THCON_SEC1_REG3_Base_address.addr32(),
        _ => {
            let _ = unit;
            return Err(ttStatus::TT_ERROR_UNPACR);
        }
    };
    cb_base(out, cb, rv::cb::RD_PTR, config)
}

/// The packer's output base: the ring's write pointer, same 16-byte conversion.
fn cb_base_pack(out: &mut Stream, cb: Cb) -> Result<(), ttStatus> {
    out.push(wait_for_packer(Before::EVERYTHING).map_err(|_| ttStatus::TT_ERROR_STALLWAIT)?);
    cb_base(
        out,
        cb,
        rv::cb::WR_PTR,
        thcon::THCON_SEC0_REG1_L1_Dest_addr.addr32(),
    )
}

/// Write the ring page into a `Config` word from a Tensix GPR, never from a
/// RISC-V `sw` into `Config`.
///
/// A plain `sw` to `Config` is processed by the LSU, not the Tensix instruction
/// FIFO, so it overtakes the `UNPACR`/`PACR` already reading the same word; the
/// role's per-iteration page then lags by one (`firmware/src/ttinsn.rs`,
/// `STEP_STORE`). The write has to be a pushed Tensix instruction:
/// `SETDMAREG` stages the dynamic value, then `WRCFG` copies it, in FIFO order
/// with the backend. This is tt-metal's `program_packer_destination`
/// (`SETDMAREG` × 2, `STALLWAIT(STALL_CFG, THCON)`, `WRCFG`, `DMANOP`) and
/// `_llk_unpack_configure_addresses_`' ordering requirement.
///
/// The caller's `wait_for_unpacker0`/`wait_for_packer` now reaches the write:
/// because the write is a pushed instruction, `STALLWAIT`'s CONFIG block holds
/// it until the previous unit has drained.
fn cb_base(out: &mut Stream, cb: Cb, field: i32, config: u16) -> Result<(), ttStatus> {
    out.asm(|a| {
        a.word(rv::lw(
            reg::T0,
            reg::A6,
            (u32::from(cb) * rv::cb::STRIDE) as i32 + field,
        ));
        a.word(rv::srli(reg::T0, reg::T0, 4));
        a.word(rv::addi(reg::T0, reg::T0, -1));
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
            MirOp::SetPackBase { cb } => {
                // The destination is the ring's write pointer, converted to
                // 16-byte units minus the header unit the packer adds.
                cb_base_pack(out, cb)?;
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

// ---------------------------------------------------------------------------
// Quasar's datapath
// ---------------------------------------------------------------------------
//
// Quasar's unpacker and packer are buffer-descriptor driven. Where Blackhole
// bakes a page's shape into the unit's `Config` bank (`SETADCX`/`SETADCZW`, the
// `THCON_SEC*_REG*` descriptor) and reads from a base address, Quasar keeps a
// table of 32 descriptors in the `Config` array and the transfer names one:
//
//   * word 0 holds the L1 base in 16-byte units and the L1 data format,
//   * word 1 the X dimension, word 2 the Y and Z dimensions,
//   * the unpack TRISC owns ids 0..16 and the pack TRISC 16..24
//     (`llk_bfd_alloc.h`),
//   * `UNPACR_DEST_TILE_INC`/`PACR0_TILE_INC` name the id and the tile-count
//     increments.
//
// So the shape is written once, and what changes per page is a *tile index* into
// the descriptor, set by `SET_SRC`/`SET_DST_TILE_FACE_ROW_IDX` and consumed by
// the next transfer. tt-metal spells the page the same way
// (`_llk_unpack_unary_operand_to_dest_tile_` and `_llk_pack_` both take an
// `l1_tile_idx`), which is why this encoder never rewrites the descriptor's base
// while the engine may still be reading it.

/// `CFG_REGS_BASE`: the `Config` array in each hart's local map
/// (`tt_t6_trisc_map.h:31`).
///
/// A Quasar TRISC writes `Config` with a plain store — tt-metal's
/// `bd_table[i] = ...` (`ckernel_buf_desc.h:132-153`) and
/// `cfg[ctrl_regs[...]] = ...` (`ckernel.h:774-800`) are both word stores — and
/// the hardware keeps such a store ordered against the Tensix instructions that
/// read the same word: `TENSIX_TRISC_SYNC[n]`'s `DISABLETRACKGLOBALCFG` resets to
/// zero, i.e. tracking is on, and `ckernel.h:270-288` says so
/// (`t6_debug_map.h:874-886`). Blackhole has no such tracking, which is why its
/// page pointer has to be pushed as `SETDMAREG`+`WRCFG` (`cb_base`).
const QUASAR_CFG: u32 = Quasar::TENSIX_CFG_BASE as u32;
/// `IBUFFER_BASE`: the store target that pushes a Tensix instruction
/// (`tt_t6_trisc_map.h:21`), as `cb_base` uses it.
const QUASAR_INSTRN_BUF: u32 = Quasar::INSTRN_BUF_BASE as u32;
/// `BUFFER_DESCRIPTOR_TABLE_REG0_L1_BASE_ADDR_ADDR32`: the table's first word;
/// one descriptor is four words of which three are written
/// (`cfg_defines.h:1469-1490`, `ckernel_buf_desc.h:195-201`).
const QUASAR_BD: u32 = 108;
/// The unpack TRISC's descriptor partition, `[0, 16)` (`llk_bfd_alloc.h`).
const QUASAR_UNPACK_BD: u32 = 0;
/// The pack TRISC's, `[16, 24)`.
const QUASAR_PACK_BD: u32 = 16;
/// A 32x32 tile's descriptor: `x=16, y=16, z=4` is the only shape the table
/// calls a tile, and a face is 16 datums by 16 rows (`llk_bfd_alloc.h:62-90`).
const QUASAR_BD_X: u32 = 16;
const QUASAR_BD_Y: u32 = 16;
const QUASAR_BD_Z: u32 = 4;
/// `Tile_Face_Row_Sel`'s `TILE_SEL`: set only the tile counter.
const QUASAR_TILE_SEL: u32 = 0;
/// `EngineSel` of unpacker 0 and packer 0 (`p_unpacr::UNP_A`, `p_pacr::PACK0`).
///
/// `UNP_DEST` is driven off `UNP_A`'s counters, not its own
/// (`llk_unpack_unary_operand_to_dest.h:143-160`), so both the L1 index and the
/// `Dst` index of an unpack-to-`Dst` are `UNP_A`'s.
const QUASAR_ENGINE_UNP_A: u32 = 0;
const QUASAR_ENGINE_UNP_B: u32 = 1;
const QUASAR_ENGINE_PACK0: u32 = 3;
/// Datums in one `Dst` tile: `UNPACR_DEST_TILE_INC` moves exactly one tile.
const QUASAR_TILE_DATUMS: u32 = 32 * 32;
/// Datums in one face: a 32x32 tile is four 16x16 faces, and a `Src` unpack
/// may move one at a time (`UNPACR{0,1}_FACE`).
const QUASAR_FACE_DATUMS: u32 = 16 * 16;
/// Datums in one face row: the `UNPACR{0,1}_ROW` run, and the scaler the
/// AB-reduce math reads (`llk_unpack_reduce.h`'s per-face scaler unpack).
const QUASAR_ROW_DATUMS: u32 = 16;
/// The address-modifier slot the SFPU's `Src`/`Dst` accesses name: `ADDR_MOD_7`,
/// which `_sfpu_configure_addrmod_` (`ckernel_sfpu.h:32-45`) leaves with every
/// counter increment at zero.
const QUASAR_SFPU_ADDR_MOD: u32 = 7;
/// The `EngineSel` of unpacker `unit`: `UNP_A` is 0 and `UNP_B` is 1
/// (`SET_SRC_TILE_FACE_ROW_IDX`'s table, `assembly.yaml`).
fn quasar_engine(unit: u8) -> Result<u32, ttStatus> {
    match unit {
        0 => Ok(QUASAR_ENGINE_UNP_A),
        1 => Ok(QUASAR_ENGINE_UNP_B),
        _ => Err(quasar_shape("an unpacker other than 0 or 1")),
    }
}

/// The buffer-descriptor id unpacker `unit` owns: the unpack TRISC's partition
/// is `[0, 16)`, one id per engine (`llk_bfd_alloc.h`).
fn quasar_unpack_bd(unit: u8) -> u32 {
    u32::from(unit)
}

/// The register-file format a `Src` unpack writes: `Tf32` for a 32-bit operand
/// and `Float16_b` for a 16-bit one.
///
/// This is `unpack_dst_format` (`jit_build/genfiles.cpp:1074-1079`): with
/// `fp32_dest_acc_en` and every buffer `Float32` the conditional format is
/// `Tf32` (4), and a bf16 buffer unpacks as itself (5). It is *not* the `Dst`
/// format: a 32-bit accumulator is `Float32` (0).
fn src_register_format(dtype: DType) -> u32 {
    match dtype {
        DType::Bf16 => DATA_FORMAT_BF16,
        _ => DATA_FORMAT_TF32,
    }
}

/// One `Src` unpack: a face at a time, or a whole tile in one instruction.
///
/// tt-metal's `_llk_unpack_unary_operand_mop_config_` issues one
/// `UNPACR0_TILE_INC(0, 1, buf_desc_id, 1)` per tile (`llk_unpack_unary_operand.h:97-118`),
/// while the per-face sequences — the matmul operand walk and the transposing
/// unpack — name the source and destination face explicitly with
/// `UNPACR0_FACE`/`UNPACR1_FACE`. Both are the same hardware path; only the
/// counters they use differ, and `set_dat_valid` is the bank handover the
/// Matrix Unit waits for.
fn quasar_unpack_src(
    out: &mut Stream,
    unit: u8,
    count: u32,
    set_dvalid: bool,
    (src_face, dst_face): (u32, u32),
    transpose: bool,
    have_bd: bool,
) -> Result<(), ttStatus> {
    if !have_bd {
        return Err(quasar_shape("a `Src` unpack with no descriptor"));
    }
    let bd = quasar_unpack_bd(unit);
    let dvalid = u32::from(set_dvalid);
    match count {
        QUASAR_FACE_DATUMS => {
            let i = match unit {
                0 => qencode::Unpacr0Face::ZERO
                    .dst_face_idx(dst_face)
                    .src_face_idx(src_face)
                    .dst_tile_offset_idx_inc(0)
                    .src_tile_offset_idx_inc(0)
                    .buffer_descriptor_table_sel(bd)
                    .set_dat_valid(dvalid)
                    .encode(),
                _ => qencode::Unpacr1Face::ZERO
                    .dst_face_idx(dst_face)
                    .src_face_idx(src_face)
                    .dst_tile_offset_idx_inc(0)
                    .src_tile_offset_idx_inc(0)
                    .buffer_descriptor_table_sel(bd)
                    .set_dat_valid(dvalid)
                    .encode(),
            };
            out.push(i.map_err(quasar_unusable)?);
        }
        // One row of a face. The reduce's scaler is exactly this: tt-metal's
        // `_llk_unpack_reduce_mop_config_` hands the scaler face to the
        // AB-reduce math one face at a time, and the row form is the one
        // `_llk_unpack_unary_broadcast_operands_mop_config_` uses for a scalar
        // broadcast (`UNPACR1_ROW(0, 0, 0, 0, 0, 0, buf_desc_id, 1)`).
        //
        // The destination is the reader's own row 0, not the source face's:
        // `src_face` names the L1 face the row comes from, and a reader that
        // takes one row per face hands the `Src` bank back between faces
        // (tt-metal's reduce clears both banks after every face), so each face's
        // row lands where that reader addresses it — row 0.
        QUASAR_ROW_DATUMS => {
            let i = match unit {
                0 => qencode::Unpacr0Row::ZERO
                    .dst_row_idx(0)
                    .src_row_idx(0)
                    .dst_face_idx(0)
                    .src_face_idx(src_face)
                    .dst_tile_offset_idx_inc(0)
                    .src_tile_offset_idx_inc(0)
                    .buffer_descriptor_table_sel(bd)
                    .set_dat_valid(dvalid)
                    .encode(),
                _ => qencode::Unpacr1Row::ZERO
                    .dst_row_idx(0)
                    .src_row_idx(0)
                    .dst_face_idx(0)
                    .src_face_idx(src_face)
                    .dst_tile_offset_idx_inc(0)
                    .src_tile_offset_idx_inc(0)
                    .buffer_descriptor_table_sel(bd)
                    .set_dat_valid(dvalid)
                    .encode(),
            };
            out.push(i.map_err(quasar_unusable)?);
        }
        QUASAR_TILE_DATUMS if !transpose => {
            let i = match unit {
                0 => qencode::unpacr0_tile_inc(0, 1, bd, dvalid),
                _ => qencode::unpacr1_tile_inc(0, 1, bd, dvalid),
            };
            out.push(i.map_err(quasar_unusable)?);
        }
        QUASAR_TILE_DATUMS => {
            return Err(quasar_shape(
                "a whole-tile transposing unpack (the faces must be permuted one at a time)",
            ));
        }
        _ => {
            return Err(quasar_shape(
                "a `Src` run that is neither a face nor a tile",
            ));
        }
    }
    Ok(())
}

/// The stall resource classes one compute hart's own instructions belong to.
///
/// Every mask names [`qbackend::block::SYNC`] as well: a `SEMGET` that follows a
/// `SEMWAIT` must be held until the wait clears, and the get is itself a Sync Unit
/// instruction (`SyncUnit.md`'s rule, which qsr enforces — it refuses a `SEMGET`
/// whose thread's `SEMWAIT` "block mask lacks B1"). tt-metal reaches the same
/// place from the other side, with `t6_semaphore_get<kIdle..>` putting a
/// `STALLWAIT(STALL_SYNC, ..)` in front of the get.
fn quasar_block(core: Core) -> u32 {
    let own = match core {
        Core::Unpack => qbackend::block::UNPACK,
        Core::Pack => qbackend::block::PACK,
        _ => qbackend::block::MATH | qbackend::block::SFPU,
    };
    own | qbackend::block::SYNC
}

/// The units a hart drains before it posts or takes a semaphore: tt-metal's
/// per-thread `hw::t6_semaphore_*` idle list
/// (`hw/inc/api/compute/experimental/semaphore_compute_impl.h:113-137`).
fn quasar_idle(core: Core) -> &'static [u32] {
    match core {
        Core::Unpack => &[qbackend::res::UNPACK0, qbackend::res::UNPACK1],
        Core::Pack => &[qbackend::res::PACK0],
        _ => &[qbackend::res::MATH, qbackend::res::SFPU],
    }
}

/// Refuse a Quasar lowering, naming the op under `SKAM_DEBUG`: the next slice
/// starts from the first thing a kernel needs that this backend does not have.
fn quasar_unsupported(op: &MirOp) -> ttStatus {
    if std::env::var_os("SKAM_DEBUG").is_some() {
        eprintln!("libskamandros: no Quasar lowering for {op:?}");
    }
    ttStatus::TT_ERROR_UNSUPPORTED
}

/// Refuse a shape this backend cannot express yet, naming it under `SKAM_DEBUG`.
fn quasar_shape(what: &str) -> ttStatus {
    if std::env::var_os("SKAM_DEBUG").is_some() {
        eprintln!("libskamandros: no Quasar lowering for {what}");
    }
    ttStatus::TT_ERROR_UNSUPPORTED
}

fn quasar_unusable(e: impl std::fmt::Display) -> ttStatus {
    let _ = e;
    ttStatus::TT_ERROR_ENCODE
}

fn quasar_semaphore(index: u8) -> Result<qbackend::Semaphore, ttStatus> {
    qbackend::Semaphore::new(index).ok_or(ttStatus::TT_ERROR_SEMAPHORE)
}

/// Write `Config` words with plain stores. See [`QUASAR_CFG`].
fn quasar_config(out: &mut Stream, words: &[(u16, u32)]) -> Result<(), ttStatus> {
    out.asm(|a| {
        for &(addr32, value) in words {
            a.li(reg::T0, QUASAR_CFG + 4 * u32::from(addr32));
            a.li(reg::T1, value);
            a.word(rv::sw(reg::T1, reg::T0, 0));
        }
        Ok(())
    })
}

fn quasar_config_fields(out: &mut Stream, words: &ConfigWords) -> Result<(), ttStatus> {
    quasar_config(out, words.words())
}

/// One buffer descriptor, as `construct_buf_desc` builds it
/// (`ckernel_buf_desc.h:132-153`): the L1 base in 16-byte units plus the data
/// format in word 0, the X dimension in word 1, and Y and Z in word 2.
fn quasar_bd(out: &mut Stream, id: u32, base: u32, format: u32) -> Result<(), ttStatus> {
    let word0 = (base >> 4) | (format << 20);
    let at = QUASAR_BD + 4 * id;
    quasar_config(
        out,
        &[
            (at as u16, word0),
            ((at + 1) as u16, QUASAR_BD_X << 20),
            ((at + 2) as u16, QUASAR_BD_Y | (QUASAR_BD_Z << 8)),
        ],
    )
}

/// Push `SET_SRC`/`SET_DST_TILE_FACE_ROW_IDX` naming this page's tile.
///
/// The descriptor's base is the ring's first page, so the page is a tile index
/// into it, `(pointer - base) / page_bytes`. `opcode_base` is the instruction
/// word with every field but the counter already placed: the counter is a
/// run-time value, so the word is built in a register and pushed, the same way
/// `cb_base` pushes `SETDMAREG` on Blackhole.
fn quasar_page(
    out: &mut Stream,
    opcode_base: u32,
    cb: Cb,
    info: &crate::mir::CbInfo,
    field: i32,
) -> Result<(), ttStatus> {
    if !info.page_bytes.is_power_of_two() {
        return Err(ttStatus::TT_ERROR_UNSUPPORTED);
    }
    out.asm(|a| {
        a.word(rv::lw(
            reg::T0,
            reg::A6,
            (u32::from(cb) * rv::cb::STRIDE) as i32 + field,
        ));
        a.li(reg::T1, info.base);
        a.word(rv::sub(reg::T0, reg::T0, reg::T1));
        a.word(rv::srli(reg::T0, reg::T0, info.page_bytes.trailing_zeros()));
        a.li(reg::T1, opcode_base);
        a.word(rv::or(reg::T1, reg::T1, reg::T0));
        a.li(reg::T2, QUASAR_INSTRN_BUF);
        a.word(rv::sw(reg::T1, reg::T2, 0));
        Ok(())
    })
}

/// The instruction word of a `SET_*_TILE_FACE_ROW_IDX` with every field but the
/// counter placed.
const fn set_src_tile_word(engine: u32) -> u32 {
    (0x06 << 24) | (QUASAR_TILE_SEL << 21) | (engine << 18)
}

const fn set_dst_tile_word(engine: u32) -> u32 {
    (0x0d << 24) | (QUASAR_TILE_SEL << 21) | (engine << 18)
}

/// `SEMINIT` the machine IR's semaphores, then publish the flag that releases
/// the other two compute harts.
///
/// One hart seeds them all, exactly as Blackhole's thread-0 setup program does,
/// because a `SEMINIT` from a consumer could reset a value a producer has
/// already posted. The other two spin on [`crate::runtime::QUASAR_READY_AT`]
/// rather than on a semaphore: a semaphore nobody has seeded has a maximum of
/// zero, so a post could never raise it.
fn quasar_seed(out: &mut Stream, m: &MachineKernel) -> Result<(), ttStatus> {
    let idle = quasar_idle(Core::Unpack);
    for &(index, value) in &m.sems {
        let sem = quasar_semaphore(index)?;
        // The `Dst` lock is a one-token semaphore (`semaphore_init`).
        let max = if index == crate::lower::SEM_MATH_PACK {
            1
        } else {
            qbackend::MAX_VALUE
        };
        out.extend(qbackend::init_after(sem, value, max, idle).map_err(quasar_unusable)?);
    }
    out.asm(|a| {
        a.li(reg::T0, crate::runtime::QUASAR_READY_AT);
        a.li(reg::T1, 1);
        a.word(rv::sw(reg::T1, reg::T0, 0));
        Ok(())
    })
}

/// Wait for the unpack hart to publish the seeded semaphores.
fn quasar_await_ready(out: &mut Stream) -> Result<(), ttStatus> {
    out.asm(|a| {
        a.li(reg::T0, crate::runtime::QUASAR_READY_AT);
        let top = a.fresh_label();
        a.label(top.clone());
        a.word(rv::lw(reg::T1, reg::T0, 0));
        a.beq_label(reg::T1, reg::ZERO, top);
        Ok(())
    })
}

/// What the unpack hart's `Config` bank and counters hold, so a repeated
/// configuration costs nothing. It is dropped at a loop head, where the
/// previous execution is the loop's own tail rather than the instruction
/// before it.
#[derive(Clone, Copy, Default)]
struct QuasarUnpackState {
    /// Per unit: the whole `UNPACKERn_REG0` word last written. It carries both
    /// the register-file format (bits 0..8) and `TRANSPOSE` (bit 8), and a
    /// Quasar TRISC writes `Config` a word at a time, so the two fields are
    /// composed rather than written separately.
    reg0: [Option<u32>; 2],
    /// Per unit: the ring whose descriptor is in that unit's table slot.
    bd: [Option<(Cb, u32)>; 2],
    /// Per unit: `(source, destination)` face the last `SETADCZW` named.
    faces: [(u32, u32); 2],
    /// Per unit: whether the last `UnpackConfig` sent its page to `Src` rather
    /// than `Dst`.
    to_src: [bool; 2],
    /// Per unit: whether that `Src` unpack transposes each face.
    src_transpose: [bool; 2],
}

/// The unpack hart's program.
///
/// A page reaches either `Dst` (`UNP_DEST`, one whole tile per
/// `UNPACR_DEST_TILE_INC`) or `SrcA`/`SrcB` (`UNP_A`/`UNP_B`, one
/// `UNPACR{0,1}_FACE` per face so the Matrix Unit can take a face at a time).
/// The `Dst` path is ordered by the semaphore sections; the `Src` path hands the
/// bank over with the instruction's `SetDatValid`, which is what the Matrix
/// Unit's reads wait for.
fn encode_unpack_quasar(
    m: &MachineKernel,
    ops: &[MirOp],
    out: &mut Stream,
) -> Result<(), ttStatus> {
    if ops.is_empty() {
        return Ok(());
    }
    // `a0` = the launch arguments; word 2 is the CB interface table.
    out.rv(rv::lw(reg::A6, reg::A0, crate::mover::ARG_IFACE as i32 * 4));
    quasar_seed(out, m)?;
    let idle = quasar_idle(Core::Unpack);
    let block = quasar_block(Core::Unpack);
    let mut state = QuasarUnpackState::default();
    let mut loops = Vec::new();
    for op in ops {
        if encode_int(op, out, &mut loops, &COMPUTE_PAIRS)? {
            continue;
        }
        match *op {
            MirOp::Loop { count } => {
                state = QuasarUnpackState::default();
                open_loop(out, &mut loops, count)?;
            }
            MirOp::EndLoop => close_loop(out, &mut loops)?,
            MirOp::Setdst { reg } => {
                // The `Dst` tile an unpack-to-`Dst` lands in.
                out.push(
                    qencode::set_dst_tile_face_row_idx(
                        QUASAR_TILE_SEL,
                        QUASAR_ENGINE_UNP_A,
                        u32::from(reg),
                    )
                    .map_err(quasar_unusable)?,
                );
            }
            MirOp::Setadcx { unit, first, count } => {
                // A descriptor-decoded unpack covers its own x/y/z: one whole
                // tile from datum 0. A narrower run would need a descriptor per
                // shape, which nothing lowers yet.
                if unit != 0 || first != 0 || count != QUASAR_TILE_DATUMS {
                    return Err(quasar_shape("a datum run narrower than a tile"));
                }
            }
            MirOp::Setadczw { u0, u1, z0, z1 } => {
                // The face the next unpack reads, and the face it is written
                // to. On `Src` these are the `UNPACR{0,1}_FACE` operands, so
                // they are remembered rather than written to a counter; a
                // Blackhole bank bakes them into `SETADCZW`, which Quasar's
                // descriptor does not have.
                for unit in [0usize, 1] {
                    if (unit == 0 && !u0) || (unit == 1 && !u1) {
                        continue;
                    }
                    let src = z0.unwrap_or(state.faces[unit].0);
                    let dst = z1.unwrap_or(src);
                    state.faces[unit] = (src, dst);
                }
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
                state.to_src[u] = to_src;
                state.src_transpose[u] = to_src && haloize;
                // The address-mod fields a Blackhole unpacker bakes into its
                // bank describe the face a tile's destination starts at. On
                // Quasar that face is the `UNPACR_FACE` operand, so the two
                // fields are carried by `SETADCZW` and nothing is written here;
                // a `Dst` unpack has no face counter to steer at all.
                if !to_src && (haloize || add_dest_cntr || ch1_z_stride != 0) {
                    return Err(quasar_shape(
                        "an unpack into `Dst` with a face counter or a transpose",
                    ));
                }
                // `_llk_unpack_hw_configure_` (`llk_unpack_common.h:34-54`):
                // `UNP_A` and `UNP_DEST` share unpacker 0's `REG0`, `UNP_B` has
                // its own. A `Src` unpack writes the *register file's* format —
                // tf32 for a 32-bit operand, bf16 for a 16-bit one
                // (`genfiles.cpp:1074-1079`: `unpack_conditional_dst_format` is
                // `Tf32` when every buffer is f32 and dest accumulation is on) —
                // while a `Dst` unpack writes the accumulator's own format, which
                // is what the packer reads back. `TRANSPOSE` shares that word
                // (`_llk_unpack_unary_operand_init_`): only a transposing `Src`
                // unpack sets it.
                let (field, format) = if to_src {
                    let field = if u == 0 {
                        qthcon::THCON_UNPACKER0_REG0_OUT_DATA_FORMAT
                    } else {
                        qthcon::THCON_UNPACKER1_REG0_OUT_DATA_FORMAT
                    };
                    (field, src_register_format(dtype))
                } else {
                    (
                        qthcon::THCON_UNPACKER0_REG0_OUT_DATA_FORMAT,
                        dst_format(m.dst_is_16_bit),
                    )
                };
                let transpose = if to_src && haloize {
                    qthcon::THCON_UNPACKER0_REG0_TRANSPOSE.mask()
                } else {
                    0
                };
                let word = format | transpose;
                if state.reg0[u] != Some(word) {
                    quasar_config(out, &[(field.addr32(), word)])?;
                    state.reg0[u] = Some(word);
                }
            }
            MirOp::SetUnpackBase { cb, unit } => {
                if unit > 1 {
                    return Err(quasar_shape("an unpacker other than 0 or 1"));
                }
                let u = usize::from(unit);
                let info = &m.cbs[cb];
                let format = l1_format(info.dtype);
                if state.bd[u] != Some((cb, format)) {
                    quasar_bd(out, quasar_unpack_bd(unit), info.base, format)?;
                    state.bd[u] = Some((cb, format));
                }
                // The page is a tile index into the descriptor; the engine's own
                // tile counter carries it (`_llk_unpack_unary_operand_` sets the
                // source counter to `l1_tile_idx` and the destination counter to
                // zero before running the MOP).
                quasar_page(
                    out,
                    set_src_tile_word(quasar_engine(unit)?),
                    cb,
                    info,
                    rv::cb::RD_PTR,
                )?;
                if state.to_src[u] {
                    out.push(
                        qencode::set_dst_tile_face_row_idx(
                            QUASAR_TILE_SEL,
                            quasar_engine(unit)?,
                            0,
                        )
                        .map_err(quasar_unusable)?,
                    );
                }
            }
            MirOp::Unpacr { unit, count, flip } => {
                if unit > 1 {
                    return Err(quasar_shape("an unpacker other than 0 or 1"));
                }
                let u = usize::from(unit);
                if state.to_src[u] {
                    quasar_unpack_src(
                        out,
                        unit,
                        count,
                        flip,
                        state.faces[u],
                        state.src_transpose[u],
                        state.bd[u].is_some(),
                    )?;
                } else {
                    if unit != 0 || flip || count != QUASAR_TILE_DATUMS {
                        return Err(quasar_shape("an unpack that is not one tile into `Dst`"));
                    }
                    // `_llk_unpack_unary_operand_to_dest_mop_config_`: both the
                    // L1 and the `Dst` tile counters advance by one, so
                    // back-to-back runs land consecutive tiles.
                    out.push(
                        qencode::unpacr_dest_tile_inc(1, 1, QUASAR_UNPACK_BD, 0)
                            .map_err(quasar_unusable)?,
                    );
                }
            }
            MirOp::CbWait { cb, n } => {
                out.asm(|a| {
                    rv::cb::wait_front(a, reg::A6, u32::from(cb), n, [reg::T0, reg::T1, reg::T2])
                })?;
            }
            MirOp::CbPop { cb, n } => {
                out.push(
                    qbackend::stallwait(qbackend::block::SYNC, idle).map_err(quasar_unusable)?,
                );
                out.asm(|a| {
                    // The ring's pop is a store, which a pushed `STALLWAIT` does
                    // not hold back: `tensix_sync` retires this hart's queued
                    // instructions before the page is handed back.
                    rv::coprocessor_done_quasar(a);
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
                out.extend(qbackend::post_after(quasar_semaphore(sem)?, idle));
            }
            MirOp::SemWait { sem, .. } => {
                out.extend(qbackend::take_after(
                    quasar_semaphore(sem)?,
                    qbackend::cond::STALL_ON_ZERO,
                    block,
                    idle,
                ));
            }
            MirOp::SemWaitNoGet { sem, .. } => {
                out.extend(qbackend::wait_after(
                    quasar_semaphore(sem)?,
                    qbackend::cond::STALL_ON_ZERO,
                    block,
                    idle,
                ));
            }
            MirOp::SemGet { sem } => {
                out.extend(qbackend::get_after(quasar_semaphore(sem)?, idle));
            }
            MirOp::SemWaitFree { sem } => {
                out.extend(qbackend::wait_after(
                    quasar_semaphore(sem)?,
                    qbackend::cond::STALL_ON_MAX,
                    block,
                    idle,
                ));
            }
            other => return Err(quasar_unsupported(&other)),
        }
    }
    if !loops.is_empty() {
        return Err(ttStatus::TT_ERROR_ENDLOOP_MACHINE);
    }
    Ok(())
}

/// The math hart's program.
///
/// The SFPU reads and writes `Dst` at an address relative to the section base
/// `DEST_TARGET_REG_CFG_MATH_SEC1_Offset` holds — one section per `Dst` register,
/// 64 address units apart (`_set_dst_write_addr_`, `cmath_common.h:236-249`,
/// `64 = tile_index << get_dest_tile_size_log2(Tile32x32)`). Within a section the
/// SFPU's own counter steps by two per row pair, which is exactly the `at` the
/// lowering names, so the address the loop below emits is Blackhole's.
fn encode_math_quasar(m: &MachineKernel, ops: &[MirOp], out: &mut Stream) -> Result<(), ttStatus> {
    if ops.is_empty() {
        return Ok(());
    }
    quasar_await_ready(out)?;
    // `_configure_alu_formats_` (`llk_math_common.h:80-115`) writes both width
    // bits: `Fp32_enabled` for the FPU and `SFPU_Fp32_enabled` for the SFPU's own
    // view of `Dst`. `MEASURED`: qsr refuses an SFPU `Dst` access with the second
    // bit clear (`sfpu_dest_mode: SFPU_Fp32_enabled=0`).
    let mut words = ConfigWords::new();
    config_set(
        &mut words,
        qalu::ALU_ACC_CTRL_Fp32_enabled,
        u32::from(!m.dst_is_16_bit),
    )?;
    config_set(
        &mut words,
        qalu::ALU_ACC_CTRL_SFPU_Fp32_enabled,
        u32::from(!m.dst_is_16_bit),
    )?;
    // `ALU_ACC_CTRL_Zero_Flag_disabled_src` (word 3) is *not* written: it resets
    // to zero, which is the value the comment above requires, and qsr refuses any
    // store to that word from a TRISC (`tensix_cfg_wr32: reg=3`,
    // `UnimplementedFunctionality`) — a simulator gap, so a program that writes
    // the same zero back would be untestable here for no gain.
    quasar_config_fields(out, &words)?;
    // `_sfpu_configure_addrmod_` (`ckernel_sfpu.h:32-45`): the SFPU's own
    // address modifier, `ADDR_MOD_7`, leaves every counter where it is — chosen
    // there precisely so it does not conflict with the A2D/matmul slots 0 and
    // 1. Every `Src`/`Dst` access names it, and it is written here rather than
    // trusted: the FPU's matmul leaves slot 0 stepping `Dst` by a row group,
    // and a `Config` word outlives the launch that wrote it.
    for field in [
        thread::ADDR_MOD_AB_SEC7_SrcAIncr,
        thread::ADDR_MOD_DST_SEC7_DestIncr,
    ] {
        quasar_config(out, &[(quasar_thread_word(field.addr32())?, 0)])?;
    }
    let idle = quasar_idle(Core::Math);
    let block = quasar_block(Core::Math);
    let mut section: Option<Dst> = None;
    let mut at = 0usize;
    let mut loops = Vec::new();
    while at < ops.len() {
        if encode_int(&ops[at], out, &mut loops, &COMPUTE_PAIRS)? {
            at += 1;
            continue;
        }
        match ops[at] {
            MirOp::Repeat { len, count } => {
                let body = ops.get(at + 1..at + 1 + len as usize).ok_or({
                    let _ = at;
                    ttStatus::TT_ERROR_MATH_REPEAT_BODY
                })?;
                for iteration in 0..u32::from(count) {
                    for op in body {
                        quasar_math_op(op, iteration * 2, &mut section, out)?;
                    }
                }
                at += 1 + len as usize;
                continue;
            }
            MirOp::Loop { count } => {
                section = None;
                open_loop(out, &mut loops, count)?;
            }
            MirOp::EndLoop => close_loop(out, &mut loops)?,
            MirOp::SemPost { sem } => {
                out.extend(qbackend::post_after(quasar_semaphore(sem)?, idle));
            }
            MirOp::SemWait { sem, .. } => {
                out.extend(qbackend::take_after(
                    quasar_semaphore(sem)?,
                    qbackend::cond::STALL_ON_ZERO,
                    block,
                    idle,
                ));
            }
            MirOp::SemWaitNoGet { sem, .. } => {
                out.extend(qbackend::wait_after(
                    quasar_semaphore(sem)?,
                    qbackend::cond::STALL_ON_ZERO,
                    block,
                    idle,
                ));
            }
            MirOp::SemGet { sem } => {
                out.extend(qbackend::get_after(quasar_semaphore(sem)?, idle));
            }
            MirOp::SemWaitFree { sem } => {
                out.extend(qbackend::wait_after(
                    quasar_semaphore(sem)?,
                    qbackend::cond::STALL_ON_MAX,
                    block,
                    idle,
                ));
            }
            op => quasar_math_op(&op, 0, &mut section, out)?,
        }
        at += 1;
    }
    if !loops.is_empty() {
        return Err(ttStatus::TT_ERROR_ENDLOOP_MACHINE);
    }
    Ok(())
}

/// Point the math hart's SFPU at `reg`'s section, when it is not there already.
fn quasar_section(out: &mut Stream, section: &mut Option<Dst>, reg: Dst) -> Result<(), ttStatus> {
    if *section == Some(reg) {
        return Ok(());
    }
    quasar_config(
        out,
        &[(
            qalu::DEST_TARGET_REG_CFG_MATH_SEC1_Offset.addr32(),
            u32::from(reg) * Dst::ROWS,
        )],
    )?;
    *section = Some(reg);
    Ok(())
}

fn quasar_math_op(
    op: &MirOp,
    row_delta: u32,
    section: &mut Option<Dst>,
    out: &mut Stream,
) -> Result<(), ttStatus> {
    match *op {
        MirOp::Sfpload {
            lreg,
            reg,
            at,
            dtype,
        } => {
            quasar_section(out, section, reg)?;
            out.push(
                qencode::Sfpload::ZERO
                    .lreg_ind(u32::from(lreg))
                    .instr_mod0(lane_format(dtype))
                    .sfpu_addr_mode(QUASAR_SFPU_ADDR_MOD)
                    .done(0)
                    .dest_reg_addr(at + row_delta)
                    .encode()
                    .map_err(quasar_unusable)?,
            );
        }
        MirOp::Sfpstore {
            lreg,
            reg,
            at,
            dtype,
        } => {
            quasar_section(out, section, reg)?;
            out.push(
                qencode::Sfpstore::ZERO
                    .lreg_ind(u32::from(lreg))
                    .instr_mod0(lane_format(dtype))
                    .sfpu_addr_mode(QUASAR_SFPU_ADDR_MOD)
                    .done(0)
                    .dest_reg_addr(at + row_delta)
                    .encode()
                    .map_err(quasar_unusable)?,
            );
        }
        MirOp::UnarySfp {
            op: UnarySfpOp::Neg,
            a,
            d,
        } => out.push(qencode::sfpmov(u32::from(a), u32::from(d), 1).map_err(quasar_unusable)?),
        MirOp::UnarySfp {
            op: UnarySfpOp::Abs,
            a,
            d,
        } => out.push(qencode::sfpabs(u32::from(a), u32::from(d), 1).map_err(quasar_unusable)?),
        MirOp::UnarySfp {
            op: UnarySfpOp::BitNot,
            a,
            d,
        } => out.push(qencode::sfpnot(u32::from(a), u32::from(d)).map_err(quasar_unusable)?),
        MirOp::LaneConst { lreg, bits } => {
            for i in tt_isa::arch::quasar::sfpu::load_f32(u32::from(lreg), bits)
                .map_err(quasar_unusable)?
            {
                out.push(i);
            }
        }
        // The lane-register set. tt-metal builds its SFPU elementwise ops from
        // these, and `SFPMAD` covers add, subtract and multiply too (with `VC`
        // the constant one, or zero for the product), so these sequences follow
        // Blackhole's encoder instruction for instruction.
        MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a,
            b,
            d,
        } => {
            out.push(qsfpu::add(u32::from(a), u32::from(b), u32::from(d)).map_err(quasar_unusable)?)
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::Sub,
            a,
            b,
            d,
        } => {
            out.push(qsfpu::sub(u32::from(a), u32::from(b), u32::from(d)).map_err(quasar_unusable)?)
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a,
            b,
            d,
        } => {
            out.push(qsfpu::mul(u32::from(a), u32::from(b), u32::from(d)).map_err(quasar_unusable)?)
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::BitAnd,
            a,
            b,
            d,
        } => {
            // Quasar's `SFPAND` is the in-place form `VD = VD & VC`: move the
            // first operand in, then and the second into it.
            out.push(qencode::sfpmov(u32::from(a), u32::from(d), 0).map_err(quasar_unusable)?);
            out.push(qencode::sfpand(u32::from(b), u32::from(d)).map_err(quasar_unusable)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::BitOr,
            a,
            b,
            d,
        } => {
            out.push(qencode::sfpmov(u32::from(a), u32::from(d), 0).map_err(quasar_unusable)?);
            out.push(qencode::sfpor(u32::from(b), u32::from(d)).map_err(quasar_unusable)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::BitXor,
            a,
            b,
            d,
        } => {
            out.push(qencode::sfpmov(u32::from(a), u32::from(d), 0).map_err(quasar_unusable)?);
            out.push(qencode::sfpxor(u32::from(b), u32::from(d)).map_err(quasar_unusable)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::Shl,
            a,
            b,
            d,
        } => {
            out.push(qencode::sfpmov(u32::from(a), u32::from(d), 0).map_err(quasar_unusable)?);
            out.push(qencode::sfpshft(0, u32::from(b), u32::from(d), 0).map_err(quasar_unusable)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::Shr,
            a,
            b,
            d,
        } => {
            out.push(qencode::sfpmov(u32::from(a), u32::from(d), 0).map_err(quasar_unusable)?);
            // A negative amount is the logical right shift; `SFPIADD`'s
            // two's-complement form negates the amount in its own lane.
            out.push(
                qencode::sfpiadd(0, qsfpu::LREG_ZERO, u32::from(b), 6).map_err(quasar_unusable)?,
            );
            out.push(qencode::sfpshft(0, u32::from(b), u32::from(d), 0).map_err(quasar_unusable)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::IAdd,
            a,
            b,
            d,
        } => {
            out.push(qencode::sfpmov(u32::from(b), u32::from(d), 0).map_err(quasar_unusable)?);
            out.push(qencode::sfpiadd(0, u32::from(a), u32::from(d), 4).map_err(quasar_unusable)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::ISub,
            a,
            b,
            d,
        } => {
            out.push(qencode::sfpmov(u32::from(a), u32::from(d), 0).map_err(quasar_unusable)?);
            out.push(qencode::sfpiadd(0, u32::from(b), u32::from(d), 6).map_err(quasar_unusable)?);
        }
        MirOp::BinarySfp {
            op: BinarySfpOp::Setexp,
            a,
            b,
            d,
        } => {
            out.push(qencode::sfpmov(u32::from(b), u32::from(d), 0).map_err(quasar_unusable)?);
            out.push(
                qencode::sfpsetexp(0, u32::from(a), u32::from(d), 0).map_err(quasar_unusable)?,
            );
        }
        MirOp::SfpMov { a, d } => {
            out.push(qencode::sfpmov(u32::from(a), u32::from(d), 0).map_err(quasar_unusable)?);
        }
        MirOp::SfpAbs { mod1, a, d } => {
            out.push(qencode::sfpabs(u32::from(a), u32::from(d), mod1).map_err(quasar_unusable)?);
        }
        MirOp::SfpCast { mod1, a, d } => {
            out.push(
                qencode::sfpcast(u32::from(a), u32::from(d), quasar_cast_mode(mod1))
                    .map_err(quasar_unusable)?,
            );
        }
        MirOp::SfpGt { mod1, vc, vd } => {
            // Quasar's `SFPGT` names the `imm12_math` operand Blackhole's does
            // not; the lowering's comparison uses the register form, so it is
            // zero.
            out.push(
                qencode::sfpgt(0, u32::from(vc), u32::from(vd), mod1).map_err(quasar_unusable)?,
            );
        }
        MirOp::SfpExexp { mod1, a, d } => {
            out.push(qencode::sfpexexp(u32::from(a), u32::from(d), mod1).map_err(quasar_unusable)?);
        }
        MirOp::SfpExman { mod1, a, d } => {
            out.push(qencode::sfpexman(u32::from(a), u32::from(d), mod1).map_err(quasar_unusable)?);
        }
        MirOp::SfpIadd { imm, a, d, mod1 } => {
            out.push(
                qencode::sfpiadd(imm, u32::from(a), u32::from(d), mod1).map_err(quasar_unusable)?,
            );
        }
        MirOp::SfpShft { mod1, amount, a, d } => {
            out.push(qencode::sfpmov(u32::from(a), u32::from(d), 0).map_err(quasar_unusable)?);
            out.push(
                qencode::sfpshft(0, u32::from(amount), u32::from(d), mod1)
                    .map_err(quasar_unusable)?,
            );
        }
        MirOp::SfpSetcc { mod1, a } => {
            out.push(qencode::sfpsetcc(0, u32::from(a), mod1).map_err(quasar_unusable)?);
        }
        MirOp::SfpEncc {
            first,
            second,
            mod1,
        } => {
            // Quasar's `SFPENCC` names one 12-bit `imm12_math` where Blackhole
            // names a 2-bit `Imm2` and a `VD`; the lowering's `second` is the
            // zero `VD`, so the same bits are written.
            if second != 0 {
                return Err(quasar_unsupported(op));
            }
            out.push(qencode::sfpencc(first, mod1).map_err(quasar_unusable)?);
        }
        // `SFPARECIP` has no Quasar encoding in the LLK's table at all, so a
        // reciprocal stays refused rather than guessed: `other` below reports it.
        MirOp::SfpMad { mod1, a, b, c, d } => {
            out.push(
                qencode::Sfpmad::ZERO
                    .lreg_a(u32::from(a))
                    .lreg_b(u32::from(b))
                    .lreg_c(u32::from(c))
                    .lreg_dest(u32::from(d))
                    .instr_mod1(mod1)
                    .encode()
                    .map_err(quasar_unusable)?,
            );
        }
        // The Matrix Unit. `MVMUL` carries the destination row, the
        // address-modifier index, the `instr_mod19` code and the two
        // `clear_dvalid` bits, and the lowering's fields map across one for
        // one. `instr_mod19` is 0 — a full four-row-band write. The *walk* is
        // Quasar's, and lives in the lowering's Quasar arm (`quasar_matmul`):
        // one `MVMUL` writes eight `Dst` rows and does not step `Dst` by
        // itself, so the tile's rows are laid out by address modifiers 0..5
        // rather than by the `at` the MIR carries.
        MirOp::Mvmul {
            reg,
            at,
            addr_mod,
            flip_a,
            flip_b,
        } => {
            let clear = u32::from(flip_a) | (u32::from(flip_b) << 1);
            out.push(
                qencode::mvmul(clear, 0, addr_mod, dst_row(reg, at, row_delta))
                    .map_err(quasar_unusable)?,
            );
        }
        MirOp::Elw {
            op,
            bcast,
            addr_mod,
            reg,
            at,
        } => {
            // `instr_mod19` is the broadcast code, the same four values as
            // Blackhole's two broadcast bits (`ELW_ARGS`, `assembly.yaml`):
            // 0 none, 1 `SrcB` column 0, 2 `SrcB` row 0, 3 both.
            let code = match bcast {
                Bcast::None => 0,
                Bcast::Row => 2,
                Bcast::Col => 1,
                Bcast::Scalar => 3,
            };
            let row = dst_row(reg, at, row_delta);
            let encoded = match op {
                ElwOp::Add => qencode::Elwadd::ZERO
                    .clear_dvalid(0)
                    .dest_accum_en(0)
                    .instr_mod19(code)
                    .addr_mode(addr_mod)
                    .dst(row)
                    .encode(),
                ElwOp::Sub => qencode::Elwsub::ZERO
                    .clear_dvalid(0)
                    .dest_accum_en(0)
                    .instr_mod19(code)
                    .addr_mode(addr_mod)
                    .dst(row)
                    .encode(),
                ElwOp::Mul => qencode::Elwmul::ZERO
                    .clear_dvalid(0)
                    .dest_accum_en(0)
                    .instr_mod19(code)
                    .addr_mode(addr_mod)
                    .dst(row)
                    .encode(),
            };
            out.push(encoded.map_err(quasar_unusable)?);
        }
        MirOp::Gapool {
            flip_src_b,
            flip_src_a,
            addr_mod,
            reg,
            at,
        } => {
            let clear = u32::from(flip_src_a) | (u32::from(flip_src_b) << 1);
            out.push(
                qencode::Gapool::ZERO
                    .clear_dvalid(clear)
                    // 1 is the 16x16 pool the reducer uses; Blackhole's encoding
                    // fixes the same bit (`isa::blackhole::defs::GAPOOL`'s
                    // constant field at bit 19).
                    .instr_mod19(1)
                    .pool_addr_mode(addr_mod)
                    .rsvd(0)
                    .dst(dst_row(reg, at, row_delta))
                    .encode()
                    .map_err(quasar_unusable)?,
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
            // Quasar's `GMPOOL` has no argument-index field: bit 14 is reserved
            // and must be zero, where Blackhole carries `ArgMax` there.
            if arg_max != 0 {
                return Err(quasar_shape("an arg-max pool"));
            }
            let clear = u32::from(flip_src_a) | (u32::from(flip_src_b) << 1);
            out.push(
                qencode::Gmpool::ZERO
                    .clear_dvalid(clear)
                    .instr_mod19(1)
                    .pool_addr_mode(addr_mod)
                    .rsvd(0)
                    .dst(dst_row(reg, at, row_delta))
                    .encode()
                    .map_err(quasar_unusable)?,
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
                qencode::Mova2D::ZERO
                    .dest_32b_lo(u32::from(use_dst32b_lo))
                    .src(src_row)
                    .addr_mode(addr_mod)
                    .instr_mod(quasar_move(MoveKind::EightOrNone, move8_rows))
                    .dst(dst_row(reg, at, row_delta))
                    .encode()
                    .map_err(quasar_unusable)?,
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
                qencode::Movd2B::ZERO
                    .dest_32b_lo(u32::from(use_dst32b_lo))
                    .src(src_row)
                    .addr_mode(addr_mod)
                    .instr_mod(quasar_move(MoveKind::FourOrNone, move4_rows))
                    .transpose(0)
                    .dst(dst_row(reg, at, row_delta))
                    .encode()
                    .map_err(quasar_unusable)?,
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
                qencode::Movb2D::ZERO
                    .dest_32b_lo(u32::from(use_dst32b_lo))
                    .src(src_row)
                    .addr_mode(addr_mod)
                    .transfer_sz(quasar_move(MoveKind::FourOrNone, move4_rows))
                    .bcast_datum0(0)
                    .dst(dst_row(reg, at, row_delta))
                    .encode()
                    .map_err(quasar_unusable)?,
            );
        }
        MirOp::Movb2a {
            src_a_row,
            addr_mod,
            move4_rows,
            src_b_row,
        } => {
            out.push(
                qencode::movb2_a(
                    src_a_row,
                    addr_mod,
                    quasar_move(MoveKind::FourOrNone, move4_rows),
                    src_b_row,
                )
                .map_err(quasar_unusable)?,
            );
        }
        MirOp::Zerosrc {
            zero_val,
            write_mode,
            bank_mask,
            clear_src_b,
            clear_src_a,
        } => {
            let src_mask = u32::from(clear_src_a) | (u32::from(clear_src_b) << 1);
            out.push(
                qencode::Zerosrc::ZERO
                    // The three format bits only qualify the `-inf` pattern; an
                    // fp32 `-inf` is the 8-bit-exponent form, which is what
                    // `zero_val` names here.
                    .packed_fmt(0)
                    .int_fmt(0)
                    .exp_bias(u32::from(zero_val))
                    .zero_val(u32::from(zero_val))
                    .write_mode(u32::from(write_mode))
                    .bank_mask(u32::from(bank_mask))
                    .src_mask(src_mask)
                    .encode()
                    .map_err(quasar_unusable)?,
            );
        }
        MirOp::Setrwc {
            src_a,
            src_b,
            dst,
            flip_src_a,
            flip_src_b,
            dst_cr,
        } => {
            // Quasar's `SETRWC` writes one `rwc_val` to *every* counter its
            // mask names (`SETRWC_ARGS`, `assembly.yaml`), where Blackhole's
            // carries a separate value per counter. The MIR's three values are
            // always equal (a reset), and anything else has no encoding here.
            if src_a != src_b || src_b != dst {
                let _ = (src_a, src_b, dst);
                return Err(quasar_shape("a `SETRWC` with per-counter values"));
            }
            // `bit_mask` names the counters: `SrcA`, `SrcB`, `Dst` and the
            // fidelity counter — tt-metal's `p_setrwc::SET_ABD_F`, which is
            // what `_reset_counters_` and every Blackhole `SETRWC` here set.
            // `clear_ab_vld` is the pair of data-valid clears, `rwc_cr` names
            // the Dest carriage return.
            out.push(
                qencode::setrwc(
                    u32::from(flip_src_a) | (u32::from(flip_src_b) << 1),
                    if dst_cr { 0b0100 } else { 0 },
                    src_a,
                    0b1111,
                )
                .map_err(quasar_unusable)?,
            );
        }
        MirOp::Incrwc { src_a, src_b, dst } => {
            out.push(qencode::incrwc(0, src_a, src_b, dst).map_err(quasar_unusable)?);
        }
        MirOp::AluAccCtrl {
            fp32,
            zero_flag_src,
        } => {
            // `_configure_alu_formats_` (`llk_math_common.h:81-116`) writes the
            // whole ALU block at once, the zero flags included; Quasar's word 3
            // is not modeled. `MEASURED`: qsr refuses a TRISC store to it
            // (`tensix_cfg_wr32: reg=3`), so only the width word is written,
            // which is the field these sequences switch around a `MOVD2B`/
            // `MOVB2D` pair. The deviation is the simulator's, not a choice:
            // word 3 is where tt-metal's own block write lands too.
            let _ = zero_flag_src;
            quasar_config(
                out,
                &[(qalu::ALU_ACC_CTRL_Fp32_enabled.addr32(), u32::from(fp32))],
            )?;
        }
        MirOp::SrcAFormat { format } => {
            // `ALU_FORMAT_SPEC_REG_SrcA_override`/`_val`: the override is what
            // makes the manual switch take effect (word 0 on both chips).
            quasar_config(
                out,
                &[
                    (qalu::ALU_FORMAT_SPEC_REG_SrcA_override.addr32(), 1),
                    (qalu::ALU_FORMAT_SPEC_REG_SrcA_val.addr32(), format),
                ],
            )?;
        }
        MirOp::ThreadWord { addr, value } => {
            let at = quasar_thread_word(addr)?;
            quasar_config(out, &[(at, u32::from(value))])?;
        }
        MirOp::WaitMatrix => {
            // `STALLWAIT(STALL_SFPU, 0, 0, MATH)`, exactly what
            // `_llk_math_sfpu_start_` issues before the SFPU touches rows the
            // Matrix Unit has just written (`ckernel_sfpu.h:49-56`). The class
            // mask is held back until the unit drains; a semaphore hand-off
            // after this one adds its own `STALLWAIT(STALL_SYNC, MATH)`.
            out.push(
                qbackend::stallwait(qbackend::block::SFPU, &[qbackend::res::MATH])
                    .map_err(quasar_unusable)?,
            );
        }
        other => return Err(quasar_unsupported(&other)),
    }
    Ok(())
}

/// How many rows a `MOV*` moves, as Quasar's `instr_mod`/`transfer_sz` codes
/// (0 single row, 1 four rows, 2 eight rows; `assembly.yaml`'s `MOVD_REG`).
enum MoveKind {
    /// `MOVA2D`: eight rows when the bit is set, otherwise one.
    EightOrNone,
    /// `MOVD2A`/`MOVD2B`/`MOVB2D`: four rows when the bit is set, otherwise one.
    FourOrNone,
}

fn quasar_move(kind: MoveKind, wide: bool) -> u32 {
    match (kind, wide) {
        (MoveKind::EightOrNone, false) | (MoveKind::FourOrNone, false) => 0,
        (MoveKind::EightOrNone, true) => 2,
        (MoveKind::FourOrNone, true) => 1,
    }
}

/// The Quasar `instr_mod1` for a cast the lowering spells in Blackhole's
/// numbering.
///
/// `SFPCAST`'s table is the same on both chips for the rounding modes (`0`
/// int32 to fp32 nearest-even, `1` stochastic) and, `MEASURED` on qsr, carries
/// the sign-magnitude pair the other way round: Blackhole's `3` (two's
/// complement to sign-magnitude, `InstrModCast::INT32_2S_COMP_TO_INT_SIGN_MAGN`
/// is `2` in the enum but the hardware takes `3`, which is what
/// `lower.rs`'s integer widen uses and what the Blackhole simulator accepts)
/// is Quasar's `2`.
fn quasar_cast_mode(mod1: u32) -> u32 {
    match mod1 {
        3 => 2,
        other => other,
    }
}

/// The Quasar `Config` word a Blackhole `SETC16` `ThreadConfig` address names,
/// on the math hart.
///
/// The bit layouts are the same on both chips (both tables are generated from
/// the same field definitions), but the words are not: Blackhole gives the
/// address modifiers words 12..=19 and 28..=35 in a per-thread bank, while
/// Quasar lays 64 `ADDR_MOD_AB_SEC*`/`ADDR_MOD_DST_SEC*` words out from 264 and
/// 328 and gives thread `n` the eight slots at `8n` (`addr_mod_t::set`,
/// `ckernel_addrmod.h:258-266`: `addr_mod_src_reg_addr[i] + NUM_MATH_ADDR_MODS *
/// thread_id`). The math hart is TRISC1.
fn quasar_thread_word(addr: u16) -> Result<u16, ttStatus> {
    /// The math hart's thread id (`ckernel_trisc_id.h`: 0 unpacks, 1 does the
    /// math, 2 packs).
    const MATH: u16 = 1;
    /// `ADDR_MOD_AB_SEC0`, `ADDR_MOD_DST_SEC0` and one thread's eight slots.
    const AB0: u16 = 264;
    const DST0: u16 = 328;
    const SLOTS: u16 = 8;
    /// `FIDELITY_BASE_Phase` (`cfg::blackhole::thread`) and Quasar's per-thread
    /// `FIDELITY_BASE_SEC*`.
    const BH_FIDELITY: u16 = 11;
    const QS_FIDELITY: u16 = 260;
    match addr {
        BH_FIDELITY => Ok(QS_FIDELITY + MATH),
        12..=19 => Ok(AB0 + SLOTS * MATH + (addr - 12)),
        28..=35 => Ok(DST0 + SLOTS * MATH + (addr - 28)),
        _ => Err(quasar_shape(
            "a `ThreadConfig` word with no Quasar counterpart",
        )),
    }
}

/// The pack hart's program: one `PACR0_TILE_INC` per tile out of `Dst`
/// (`_llk_pack_mop_config_`, `llk_pack.h:27-40`).
fn encode_pack_quasar(m: &MachineKernel, ops: &[MirOp], out: &mut Stream) -> Result<(), ttStatus> {
    if ops.is_empty() {
        return Ok(());
    }
    out.rv(rv::lw(reg::A6, reg::A0, crate::mover::ARG_IFACE as i32 * 4));
    quasar_await_ready(out)?;
    let idle = quasar_idle(Core::Pack);
    let mut in_format: Option<u32> = None;
    let mut pack_mask = PackMask::None;
    let mut bd: Option<(Cb, u32)> = None;
    let mut loops = Vec::new();
    for op in ops {
        if encode_int(op, out, &mut loops, &COMPUTE_PAIRS)? {
            continue;
        }
        match *op {
            MirOp::Loop { count } => {
                in_format = None;
                bd = None;
                open_loop(out, &mut loops, count)?;
            }
            MirOp::EndLoop => close_loop(out, &mut loops)?,
            MirOp::CbReserve { cb, n } => {
                out.asm(|a| {
                    rv::cb::reserve_back(a, reg::A6, u32::from(cb), n, [reg::T0, reg::T1, reg::T2])
                })?;
            }
            MirOp::SetPackBase { cb } => {
                let info = &m.cbs[cb];
                let format = l1_format(info.dtype);
                if bd != Some((cb, format)) {
                    quasar_bd(out, QUASAR_PACK_BD, info.base, format)?;
                    bd = Some((cb, format));
                }
                quasar_page(
                    out,
                    set_dst_tile_word(QUASAR_ENGINE_PACK0),
                    cb,
                    info,
                    rv::cb::WR_PTR,
                )?;
            }
            MirOp::SetPackConfig { dtype, mask } => {
                // The packer's input is the math `Dst`; its L1 output format is
                // the descriptor's (`llk_pack_common.h:105-124`). The output is
                // the `Dst`'s own format for now: a converting pack has no
                // Quasar arm yet.
                if l1_format(dtype) != dst_format(m.dst_is_16_bit) {
                    return Err(quasar_shape("a converting pack"));
                }
                let format = dst_format(m.dst_is_16_bit);
                if in_format != Some(format) {
                    quasar_config(
                        out,
                        &[(qthcon::THCON_PACKER0_REG0_IN_DATA_FORMAT.addr32(), format)],
                    )?;
                    in_format = Some(format);
                }
                // `_llk_pack_reduce_mask_config_`/`_llk_pack_reduce_mask_clear_`
                // (`llk_pack_common.h:176-239`): a reduce packs a partial tile by
                // masking datums to zero. Mask0 covers every datum of a row and
                // mask1 the datums that survive, selected per face-row by the
                // two-bit-per-row fields in words 56..59.
                if pack_mask != mask {
                    // Words 52 and 54 also hold the pack strides; the pack path
                    // writes whole words, so the zeros of the untouched fields
                    // are part of the value written here.
                    let (mask0, mask1) = match mask {
                        PackMask::None | PackMask::Col => (0xFFFF, 0x0000),
                        // `EDGE_MASK_ROW_DATUMS_EXCEPT_0`: column 0 survives.
                        PackMask::Row => (0xFFFF, 0xFFFE),
                    };
                    let select = match mask {
                        PackMask::None => 0,
                        // `EDGE_MASK_FACE_ROW0_MASK_1`: row 0 of every face.
                        PackMask::Col => 0x0000_0001,
                        // `EDGE_MASK_FACE_ALL_ROWS_MASK_1`: every row.
                        PackMask::Row => 0x5555_5555,
                    };
                    quasar_config(
                        out,
                        &[
                            (52, mask0 | (mask1 << 16)),
                            // `EDGE_MASK_MODE_ZERO`: masked datums pack as
                            // zero (`cpack_common.h:55`). The rest of the word
                            // is the untilize strides, which stay 0.
                            (54, 0),
                            (56, select),
                            (57, select),
                            (58, select),
                            (59, select),
                        ],
                    )?;
                    pack_mask = mask;
                }
            }
            MirOp::SetPackOffset { reg, at } => {
                // The `Dst` tile the packer reads; a partial tile is not lowered.
                if at != 0 {
                    return Err(quasar_shape("a pack that starts inside a tile"));
                }
                out.push(
                    qencode::set_src_tile_face_row_idx(
                        QUASAR_TILE_SEL,
                        QUASAR_ENGINE_PACK0,
                        u32::from(reg),
                    )
                    .map_err(quasar_unusable)?,
                );
            }
            MirOp::PackRows { rows, .. } => {
                if rows != Dst::ROWS {
                    return Err(quasar_shape("a pack shorter than a tile"));
                }
                out.push(
                    qencode::pacr0_tile_inc(1, 0, QUASAR_PACK_BD, 0).map_err(quasar_unusable)?,
                );
            }
            MirOp::CbPush { cb, n } => {
                out.push(
                    qbackend::stallwait(qbackend::block::SYNC, idle).map_err(quasar_unusable)?,
                );
                out.asm(|a| {
                    rv::coprocessor_done_quasar(a);
                    rv::cb::push_back(
                        a,
                        reg::A6,
                        u32::from(cb),
                        n,
                        [reg::T0, reg::T1, reg::T2, reg::T6],
                    )
                })?;
            }
            MirOp::SemPost { sem } => {
                out.extend(qbackend::post_after(quasar_semaphore(sem)?, idle));
            }
            MirOp::SemWait { sem, .. } => {
                out.extend(qbackend::take_after(
                    quasar_semaphore(sem)?,
                    qbackend::cond::STALL_ON_ZERO,
                    quasar_block(Core::Pack),
                    idle,
                ));
            }
            MirOp::SemWaitNoGet { sem, .. } => {
                out.extend(qbackend::wait_after(
                    quasar_semaphore(sem)?,
                    qbackend::cond::STALL_ON_ZERO,
                    quasar_block(Core::Pack),
                    idle,
                ));
            }
            MirOp::SemGet { sem } => {
                out.extend(qbackend::get_after(quasar_semaphore(sem)?, idle));
            }
            MirOp::SemWaitFree { sem } => {
                out.extend(qbackend::wait_after(
                    quasar_semaphore(sem)?,
                    qbackend::cond::STALL_ON_MAX,
                    quasar_block(Core::Pack),
                    idle,
                ));
            }
            other => return Err(quasar_unsupported(&other)),
        }
    }
    if !loops.is_empty() {
        return Err(ttStatus::TT_ERROR_ENDLOOP_MACHINE);
    }
    Ok(())
}

/// Every program a Quasar launch stages: the two movers and the three compute
/// harts.
pub fn quasar_programs(m: &MachineKernel) -> Result<crate::module::QuasarProgram, ttStatus> {
    let mut out = Words::default();
    encode_mover(m, Core::Reader, false, &mut out.reader)?;
    encode_mover(m, Core::Writer, true, &mut out.writer)?;
    let unpack: Vec<MirOp> = m.stream(Core::Unpack).collect();
    let math: Vec<MirOp> = m.stream(Core::Math).collect();
    let pack: Vec<MirOp> = m.stream(Core::Pack).collect();
    encode_unpack_quasar(m, &unpack, &mut out.t0)?;
    encode_math_quasar(m, &math, &mut out.t1)?;
    encode_pack_quasar(m, &pack, &mut out.t2)?;
    let [reader, t0, t1, t2, writer] = out.words()?;
    Ok(crate::module::QuasarProgram {
        reader,
        writer,
        unpack: t0,
        math: t1,
        pack: t2,
    })
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
            m.push(Core::Pack, MirOp::SetPackBase { cb: Cb(0) });
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
