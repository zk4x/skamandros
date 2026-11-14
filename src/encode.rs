// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

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

use crate::ir::{DType, Error};
use crate::mir::{
    Bcast, BinarySfpOp, Cb, CmpOp, Core, Dst, ElwOp, IOp, MachineKernel, MirOp, PackMask,
    UnarySfpOp,
};
use crate::riscv::{self as rv, Asm, reg};
use tt_isa::backend::{
    self, Before, ConfigWords, ThreadConfigEntry, block, cond, wait_for_packer, wait_for_unpacker0,
};
use tt_isa::cfg::generated::{alu, global, pack0, thcon, thread, unpack1};
use tt_isa::isa::Instruction;
use tt_isa::isa::generated::{defs, encode};
use tt_isa::noc::niu::Niu;
use tt_isa::sync::{self, Semaphore};

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
    fn asm(&mut self, build: impl FnOnce(&mut Asm)) -> Result<(), Error> {
        let mut a = Asm::default();
        build(&mut a);
        for word in a.finish().map_err(|_| Error::TT_ERROR_MOVER_PROGRAM)? {
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
    fn words(&self) -> Result<Vec<u32>, Error> {
        if self.elems.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(self.elems.len() + 1);
        for elem in &self.elems {
            out.push(match *elem {
                Elem::Tensix(i) => i.ttinsn_word().ok_or_else(|| {
                    let _ = i;
                    Error::TT_ERROR_INSTRUCTION_NOT_INLINE
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
fn open_loop(out: &mut Stream, loops: &mut Vec<OpenLoop>, count: u32) -> Result<(), Error> {
    open_loop_with(out, loops, count, &COMPUTE_PAIRS)
}

/// [`open_loop`] with an explicit counter set: the movers keep `t5` for their
/// NIU base, so their counters are `a1..a4`.
fn open_loop_with(
    out: &mut Stream,
    loops: &mut Vec<OpenLoop>,
    count: u32,
    pairs: &[(u8, u8)],
) -> Result<(), Error> {
    let (index, limit) = *pairs.get(loops.len()).ok_or(Error::TT_TODO_LOOP_DEPTH)?;
    out.li(index, 0);
    out.li(limit, count);
    let skip = out.len();
    out.rv(rv::beq(limit, reg::ZERO, 0));
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
fn close_loop(out: &mut Stream, loops: &mut Vec<OpenLoop>) -> Result<(), Error> {
    let loop_ = loops.pop().ok_or(Error::TT_ERROR_ENDLOOP_MACHINE)?;
    out.rv(rv::addi(loop_.index, loop_.index, 1));
    let here = out.len();
    out.rv(rv::bltu(loop_.index, loop_.limit, 0));
    let after = out.len();
    // The back branch is emitted at `here` and targets `top`.
    let back = ((loop_.top as i32) - (here as i32)) * 4;
    out.patch(here, rv::bltu(loop_.index, loop_.limit, back));
    let forward = ((after as i32) - (loop_.skip as i32)) * 4;
    out.patch(loop_.skip, rv::beq(loop_.limit, reg::ZERO, forward));
    Ok(())
}

/// Open a loop whose limit is already in `reg` (a dynamic bound). Same shape as
/// [`open_loop`]; the limit arrives in a register instead of an immediate.
fn open_loop_reg(
    out: &mut Stream,
    loops: &mut Vec<OpenLoop>,
    reg: u8,
    pairs: &[(u8, u8)],
) -> Result<(), Error> {
    let (index, limit) = *pairs.get(loops.len()).ok_or(Error::TT_TODO_LOOP_DEPTH)?;
    if reg != limit {
        out.rv(rv::addi(limit, reg, 0));
    }
    out.li(index, 0);
    let skip = out.len();
    out.rv(rv::beq(limit, reg::ZERO, 0));
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
) -> Result<bool, Error> {
    match *op {
        MirOp::Li { reg, value } => out.li(reg, value),
        MirOp::LoadArg { reg, arg } => out.rv(rv::lw(reg, reg::A0, arg)),
        MirOp::Lw { reg, base, at } => out.rv(rv::lw(reg, base, at)),
        MirOp::LoadCoord { reg, axis } => {
            out.li(reg, crate::mir::CORE_INFO + u32::from(axis) * 4);
            out.rv(rv::lw(reg, reg, 0));
        }
        MirOp::IArith { op, d, a, b } => out.rv(match op {
            IOp::Add => rv::add(d, a, b),
            IOp::Sub => rv::sub(d, a, b),
            IOp::Mul => rv::mul(d, a, b),
            IOp::Div => rv::div(d, a, b),
            IOp::Rem => rv::rem(d, a, b),
            IOp::And => rv::and(d, a, b),
            IOp::Or => rv::or(d, a, b),
            IOp::Xor => rv::xor(d, a, b),
            IOp::Sll => rv::sll(d, a, b),
            IOp::Sra => rv::sra(d, a, b),
        }),
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
    pub fn words(&self) -> Result<[Vec<u32>; 5], Error> {
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
pub fn encode(m: &MachineKernel) -> Result<Words, Error> {
    let mut out = Words::default();
    encode_mover(m, Core::Reader, false, &mut out.reader)?;
    encode_mover(m, Core::Writer, true, &mut out.writer)?;
    encode_unpack(
        &m.stream(Core::Unpack).collect::<Vec<_>>(),
        &mut out.t0,
        m.dst_is_16_bit,
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
) -> Result<(), Error> {
    let ops: Vec<MirOp> = m.stream(core).collect();
    if ops.is_empty() {
        return Ok(());
    }
    let niu = if outbound { Niu::Noc1 } else { Niu::Noc0 };
    out.asm(|a| {
        a.li(reg::T5, niu.base() as u32);
        a.word(rv::lw(reg::T3, reg::A0, 12));
        a.word(rv::lw(reg::A6, reg::A0, crate::mover::ARG_IFACE as i32 * 4));
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
            // The ring hand-offs are ordinary ops of the machine kernel: the
            // lowering puts each one on the core that performs it.
            MirOp::CbReserve { cb } => out.asm(|a| {
                rv::cb::reserve_back(a, reg::A6, u32::from(cb.0), 1, [reg::T0, reg::T1, reg::T2]);
            })?,
            MirOp::CbPush { cb } => out.asm(|a| {
                rv::cb::push_back(
                    a,
                    reg::A6,
                    u32::from(cb.0),
                    1,
                    [reg::T0, reg::T1, reg::T2, reg::T6],
                );
            })?,
            MirOp::CbWait { cb } => out.asm(|a| {
                rv::cb::wait_front(a, reg::A6, u32::from(cb.0), 1, [reg::T0, reg::T1, reg::T2]);
            })?,
            MirOp::CbPop { cb } => out.asm(|a| {
                rv::cb::pop_front(
                    a,
                    reg::A6,
                    u32::from(cb.0),
                    1,
                    [reg::T0, reg::T1, reg::T2, reg::T6],
                );
            })?,
            other => {
                let _ = other;
                return Err(Error::TT_ERROR_MOVER_PROGRAM);
            }
        }
    }
    if !loops.is_empty() {
        return Err(Error::TT_ERROR_ENDLOOP_MACHINE);
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
pub fn setup(sems: &[(u8, u8)]) -> Result<Vec<u32>, Error> {
    let mut out: Vec<u32> = semaphore_init(sems)?
        .into_iter()
        .map(|i| {
            i.ttinsn_word().ok_or_else(|| {
                let _ = i;
                Error::TT_ERROR_INSTRUCTION_NOT_INLINE
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
fn semaphore_init(sems: &[(u8, u8)]) -> Result<Vec<Instruction>, Error> {
    let mut out = Vec::new();
    for &(index, value) in sems {
        let sem = semaphore(index)?;
        out.push(sync::init(sem, value, sync::MAX_VALUE).map_err(|_| {
            let _ = (index, value);
            Error::TT_ERROR_SEMINIT
        })?);
    }
    Ok(out)
}

/// `SFPENCC`: gate lanes by the lane flags (on) or stop doing so (off), and
/// reset every lane's flag to true. Bit 0 of `Imm2` is `E`, bit 1 is `R`; `Mod1`
/// 10 is `EI|RI`, as tt-metal's boot sequence uses.
fn cc_mask(on: bool) -> Result<Instruction, tt_isa::isa::EncodeError> {
    encode::sfpencc(if on { 3 } else { 2 }, 0, 10)
}

/// Announce, through `sem`, that every pipeline the condition mask selects has
/// drained: a `STALLWAIT` that holds every unit of this thread, then the
/// `SEMPOST`. tt-isa's `sync::post_after` is this for one unit; a role whose
/// producers are several (`SFPU` and `MATRIX`, unpacker 0 and 1) needs the
/// combined mask.
fn post_after_drained(drained: u32, sem: Semaphore) -> [Instruction; 2] {
    [
        backend::stallwait(Before::EVERYTHING.mask(), drained).expect("named bits"),
        sync::post(sem),
    ]
}

fn semaphore(index: u8) -> Result<Semaphore, Error> {
    Semaphore::new(index).ok_or_else(|| {
        let _ = index;
        Error::TT_ERROR_SEMAPHORE
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
}

fn encode_unpack(ops: &[MirOp], out: &mut Stream, dst_is_16_bit: bool) -> Result<(), Error> {
    if ops.is_empty() {
        return Ok(());
    }
    // The `ThreadConfig` entries come first, as at the start of every tt-metal
    // kernel: `CFG_STATE_ID` decides which `Config` bank later writes land in, so
    // no config store may race with them.
    thread_config(out);
    alu_acc_ctrl(out, dst_is_16_bit);
    // Every role is called with `a0` = the launch arguments; word 2 is the CB
    // interface table (`tt-metal`'s `ARG_IFACE`).
    out.rv(rv::lw(reg::A6, reg::A0, crate::mover::ARG_IFACE as i32 * 4));
    let mut state = UnpackState::default();
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
                // under an unpack to `Src`, which ttsim refuses.
                state.cfg = [None, None];
                open_loop(out, &mut loops, count)?;
            }
            MirOp::EndLoop => close_loop(out, &mut loops)?,
            MirOp::Setdst { reg } => state.dst_row = u32::from(reg) * Dst::ROWS,
            MirOp::Setadcx { unit, first, count } => {
                if count == 0 {
                    return Err(Error::TT_ERROR_UNPACK_ZERO_RANGE);
                }
                let u = usize::from(unit);
                state.x0[u] = first;
                state.x1[u] = first + count - 1;
            }
            MirOp::Setadczw { unit, face } => state.face[usize::from(unit)] = face,
            MirOp::UnpackConfig {
                unit,
                to_src,
                dtype,
                haloize,
            } => {
                let u = usize::from(unit);
                let cfg = if to_src {
                    UnpackCfg::Src { dtype, haloize }
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
                            Error::TT_ERROR_UNPACK_ZERO_DATUMS
                        })?;
                    UnpackCfg::Dst {
                        dtype,
                        datums,
                        row: state.dst_row,
                        dst_is_16_bit,
                    }
                };
                if state.cfg[u] == Some(cfg) {
                    continue;
                }
                // `SRCA_SET_SetOvrdWithAddr`: the Src path wants it set, the Dst
                // path clear, and the unpacker reads it at execution time, so a
                // role that does both toggles it here. ttsim refuses either
                // combination otherwise.
                out.push(thread_entry(
                    thread::SRCA_SET_SetOvrdWithAddr,
                    u16::from(to_src),
                ));
                let mut words = ConfigWords::new();
                match cfg {
                    UnpackCfg::Src { dtype, haloize } => {
                        src_unpack_config(&mut words, unit, dtype, haloize)
                    }
                    UnpackCfg::Dst { datums, row, .. } => {
                        unpack_config(&mut words, datums, row, dtype, dst_is_16_bit)
                    }
                }
                out.extend(config_program(&words));
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
                    return Err(Error::TT_ERROR_UNPACK_ZERO_DATUMS);
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
            MirOp::CbWait { cb } => {
                out.asm(|a| {
                    rv::cb::wait_front(a, reg::A6, u32::from(cb), 1, [reg::T0, reg::T1, reg::T2])
                })?;
            }
            MirOp::CbPop { cb } => {
                // The page is only free once both unpackers have read it.
                out.push(
                    wait_for_unpacker0(Before::EVERYTHING)
                        .map_err(|_| Error::TT_ERROR_STALLWAIT)?,
                );
                // A page the Matrix Unit read is released only once its own
                // unpacker has drained too.
                out.push(
                    tt_isa::backend::wait_for_unpacker1(Before::EVERYTHING)
                        .map_err(|_| Error::TT_ERROR_STALLWAIT)?,
                );
                out.asm(|a| {
                    rv::cb::pop_front(
                        a,
                        reg::A6,
                        u32::from(cb),
                        1,
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
                ));
            }
            MirOp::SemWait { sem, .. } => {
                out.extend(sync::take(semaphore(sem)?, Before::EVERYTHING));
            }
            MirOp::SemWaitFree { sem } => {
                out.push(sync::wait_below_max(semaphore(sem)?, Before::EVERYTHING));
            }
            other => {
                let _ = other;
                return Err(Error::TT_ERROR_UNPACK_OP);
            }
        }
    }
    if !loops.is_empty() {
        return Err(Error::TT_ERROR_ENDLOOP_MACHINE);
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
const INSTRN_BUF: u32 = tt_isa::tensix::INSTRN_BUF_BASE as u32;

/// The unpacker's read base: the ring's read pointer in 16-byte units, minus the
/// header unit the unpacker adds (`UNPACR_Regular.md`: `InAddr = (Base_address +
/// 1 + DigestSize) * 16`).
fn cb_base_unpack(out: &mut Stream, cb: Cb, unit: u8) -> Result<(), Error> {
    out.push(wait_for_unpacker0(Before::EVERYTHING).map_err(|_| Error::TT_ERROR_STALLWAIT)?);
    if unit == 1 {
        out.push(
            tt_isa::backend::wait_for_unpacker1(Before::EVERYTHING)
                .map_err(|_| Error::TT_ERROR_STALLWAIT)?,
        );
    }
    let config = match unit {
        0 => thcon::THCON_SEC0_REG3_Base_address.addr32(),
        1 => thcon::THCON_SEC1_REG3_Base_address.addr32(),
        _ => {
            let _ = unit;
            return Err(Error::TT_ERROR_UNPACR);
        }
    };
    cb_base(out, cb, rv::cb::RD_PTR, config)
}

/// The packer's output base: the ring's write pointer, same 16-byte conversion.
fn cb_base_pack(out: &mut Stream, cb: Cb) -> Result<(), Error> {
    out.push(wait_for_packer(Before::EVERYTHING).map_err(|_| Error::TT_ERROR_STALLWAIT)?);
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
fn cb_base(out: &mut Stream, cb: Cb, field: i32, config: u16) -> Result<(), Error> {
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
    })?;
    // The Scalar Unit fills the GPR; `WRCFG` reads it. `WRCFG.md` also wants a
    // non-consumer after the write, which the trailing NOP gives it.
    out.push(
        backend::stallwait(block::CONFIG, cond::SCALAR_OUTSTANDING)
            .map_err(|_| Error::TT_ERROR_STALLWAIT)?,
    );
    out.push(backend::write_word(BASE_GPR, config).map_err(|e| {
        let _ = e;
        Error::TT_ERROR_STALLWAIT
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
fn alu_acc_ctrl(out: &mut Stream, dst_is_16_bit: bool) {
    let mut words = ConfigWords::new();
    words
        .set(alu::ALU_ACC_CTRL_Fp32_enabled, u32::from(!dst_is_16_bit))
        .unwrap()
        // `MVMUL.md`: the "keep SrcB denormals" mode must not be used.
        .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 0)
        .unwrap();
    out.extend(config_program(&words));
}

/// `SETC16.md`: the first `ThreadConfig` writes any thread must make. Emitted
/// once per role program, before anything reads the configuration bank.
fn thread_config(out: &mut Stream) {
    for (field, value) in [
        (thread::CFG_STATE_ID_StateID, 0),
        (thread::SRCA_SET_SetOvrdWithAddr, 0),
        (thread::UNPACK_MISC_CFG_CfgContextOffset_0, 0),
    ] {
        out.push(thread_entry(field, value));
    }
}

fn thread_entry(field: tt_isa::cfg::ThreadConfigField, value: u16) -> Instruction {
    ThreadConfigEntry::zeroed(field.addr32())
        .set(field, value)
        .expect("the value fits its field")
        .encode()
        .expect("a thread entry is encodable")
}

/// The instructions that write `words` into `Config`, sized to fit.
fn config_program(words: &ConfigWords) -> Vec<Instruction> {
    let mut buf = vec![tt_isa::sfpu::nop(); words.program_len()];
    let n = words
        .program(SCRATCH_GPR, &mut buf)
        .expect("the buffer is the program's own length");
    buf.truncate(n);
    buf
}

fn stallwait(block_mask: u32, condition: u32) -> Result<Instruction, Error> {
    backend::stallwait(block_mask, condition).map_err(|e| {
        let _ = e;
        Error::TT_ERROR_STALLWAIT
    })
}

/// The `SFPLOAD`/`SFPSTORE` lane format of one datum: the conversion the lane
/// makes on the way in or out.
fn lane_format(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => tt_isa::sfpu::mod0_fmt::FP32,
        DType::I32 | DType::Bool => tt_isa::sfpu::mod0_fmt::INT32,
        DType::Bf16 => tt_isa::sfpu::mod0_fmt::BF16,
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
) {
    // `In_data_format` is the L1 datum's own format; the `Dst` side is the
    // kernel's one width. The lane's format (`SFPLOAD`/`SFPSTORE`) is the third
    // choice and lives in the ops.
    let fmt = l1_format(dtype);
    let dst = dst_format(dst_is_16_bit);
    words
        .set(thcon::THCON_SEC0_REG7_Offset_address, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0, datums)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Out_data_format, dst)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Ovrd_data_format, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx0, 1)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx0, 1)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Haloize_mode, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Tileize_mode, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Throttle_mode, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Upsample_rate, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Upsample_and_interleave, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Shift_amount_cntx0, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Force_shared_exp, 0)
        .unwrap()
        .set(
            thcon::THCON_SEC0_REG5_Dest_cntx0_address,
            dst_address(dst_row),
        )
        .unwrap()
        .set(unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride, 0)
        .unwrap()
        .set(unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride, 0)
        .unwrap();
    let descriptor = tile_descriptor(fmt, datums as u16, 1, 0);
    // Words 0 and 1 always; 2 and 3 only when they carry something (ttsim
    // refuses a write of the all-zero tail).
    for (i, word) in descriptor.iter().enumerate() {
        if i < 2 || *word != 0 {
            words
                .seed(
                    thcon::THCON_SEC0_REG0_TileDescriptor.addr32() + i as u16,
                    *word,
                )
                .unwrap();
        }
    }
}

/// Configure unpacker `unit` to move a 32x32 tile's four 16x16 faces from L1
/// into `SrcA` (`unit` 0) or `SrcB` (`unit` 1), as the Matrix Unit reads them.
///
/// The `Src` side is not `Dst`: `Unpack_if_sel_cntx0` clear, no `SETADCX` run,
/// and the descriptor names four Z planes of one 16x16 face each. What comes
/// out of the unpacker is `TF32` for an `f32` operand (FP32 *into* `Src` is
/// undefined) and `BF16` for a bf16 one; the Matrix Unit learns it from the
/// implied format the handover sets, not from `ALU_FORMAT_SPEC`.
fn src_unpack_config(words: &mut ConfigWords, unit: u8, dtype: DType, haloize: bool) {
    let out = match dtype {
        DType::Bf16 => DATA_FORMAT_BF16,
        _ => DATA_FORMAT_TF32,
    };
    let haloize = u32::from(haloize);
    let descriptor = tile_descriptor(l1_format(dtype), 16, 16, 4);
    let span = match unit {
        0 => {
            words
                .set(thcon::THCON_SEC0_REG7_Offset_address, 0)
                .unwrap()
                .set(thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0, 16)
                .unwrap()
                .set(thcon::THCON_SEC0_REG2_Out_data_format, out)
                .unwrap()
                // `Src`, not `Dst`.
                .set(thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx0, 0)
                .unwrap()
                .set(thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx0, 1)
                .unwrap()
                // `THCON_SEC0_REG2_Haloize_mode`: tt-metal's
                // `_llk_unpack_AB_reduce_init_` enables it for `REDUCE_ROW`, so
                // the unpacker transposes each face on the way into `SrcA`.
                .set(thcon::THCON_SEC0_REG2_Haloize_mode, haloize)
                .unwrap()
                // With `UnpackToDst` clear this *replaces* the output address;
                // `SrcA` row 0 is `OutAddr / 16 - 4`, so 64 is row 0.
                .set(thcon::THCON_SEC0_REG5_Dest_cntx0_address, 64)
                .unwrap()
                .set(unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride, 0)
                .unwrap()
                .set(unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride, 0)
                .unwrap();
            thcon::THCON_SEC0_REG0_TileDescriptor
        }
        _ => {
            // Unpacker 1 takes far fewer fields: no `Dest_cntx` term, no
            // `Tile_x_dim_cntx` (the descriptor carries the width), and its
            // `Offset_address` cannot be written (reset zero).
            words
                .set(thcon::THCON_SEC1_REG2_Out_data_format, out)
                .unwrap()
                .set(thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx0, 1)
                .unwrap()
                .set(thcon::THCON_SEC1_REG2_Haloize_mode, haloize)
                .unwrap()
                .set(unpack1::UNP1_ADDR_CTRL_ZW_REG_1_Zstride, 0)
                .unwrap();
            thcon::THCON_SEC1_REG0_TileDescriptor
        }
    };
    for (i, word) in descriptor.iter().enumerate() {
        if *word != 0 {
            words.seed(span.addr32() + i as u16, *word).unwrap();
        }
    }
}

/// The `REG5_Dest_cntx0_address` that lands an unpack's first datum in `Dst`
/// row `row`: `Dst` row is `OutAddr / 16 - 4`.
pub const fn dst_address(row: u32) -> u32 {
    (row + 4) * 16
}

fn set_adc_x_unpack(unit: u8, first: u32, last: u32) -> Result<Instruction, Error> {
    let base = encode::Setadcxx::ZERO;
    let base = match unit {
        0 => base.u0(1),
        1 => base.u1(1),
        _ => {
            let _ = unit;
            return Err(Error::TT_ERROR_SETADCXX);
        }
    };
    base.x0_val(first).x1_val(last).encode().map_err(|e| {
        let _ = e;
        Error::TT_ERROR_SETADCXX
    })
}

fn set_adc_z_unpack(unit: u8, face: u32) -> Result<Instruction, Error> {
    let base = encode::Setadczw::ZERO;
    let base = match unit {
        0 => base.u0(1),
        1 => base.u1(1),
        _ => {
            let _ = unit;
            return Err(Error::TT_ERROR_SETADCZW);
        }
    };
    base.z0(1).z0_val(face).encode().map_err(|e| {
        let _ = e;
        Error::TT_ERROR_SETADCZW
    })
}

/// `UNPACR` into `Dst` (unit 0, no `flip`) or into `SrcA`/`SrcB` (`flip` hands
/// the bank to the Matrix Unit).
fn unpack_instruction(unit: u8, flip: bool) -> Result<Instruction, Error> {
    // Bit 0 is undocumented and ttsim refuses the instruction without it.
    let base = encode::UnpacrRegular::ZERO
        .which_unpacker(u32::from(unit))
        .multi_context_mode(1)
        .flip_src(u32::from(flip))
        .encode()
        .map_err(|e| {
            let _ = e;
            Error::TT_ERROR_UNPACR
        })?;
    Ok(Instruction::new(base.word() | 1, &defs::UNPACR_Regular))
}

// ---------------------------------------------------------------------------
// The math core (T1)
// ---------------------------------------------------------------------------

fn encode_math(ops: &[MirOp], out: &mut Stream, dst_is_16_bit: bool) -> Result<(), Error> {
    if ops.is_empty() {
        return Ok(());
    }
    alu_acc_ctrl(out, dst_is_16_bit);
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
                    Error::TT_ERROR_MATH_REPEAT_BODY
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
        return Err(Error::TT_ERROR_ENDLOOP_MACHINE);
    }
    Ok(())
}

/// One math op at `row_delta` (a repeat's iteration offset, two address units per
/// iteration).
fn encode_math_op(op: &MirOp, row_delta: u32, out: &mut Stream) -> Result<(), Error> {
    fn bad(e: impl std::fmt::Display, op: &MirOp) -> Error {
        let _ = (e, op);
        Error::TT_ERROR_ENCODE
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
            let halves = tt_isa::sfpu::load_f32(u32::from(lreg), bits).map_err(|e| bad(e, op))?;
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
                tt_isa::sfpu::add(u32::from(a), u32::from(b), u32::from(d))
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
                tt_isa::sfpu::sub(u32::from(a), u32::from(b), u32::from(d))
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
                tt_isa::sfpu::mul(u32::from(a), u32::from(b), u32::from(d))
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
                encode::sfpiadd(0, tt_isa::sfpu::LREG_ZERO, u32::from(b), 6).map_err(bad_isa)?,
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
        MirOp::UnarySfp {
            op: UnarySfpOp::Exexp,
            a,
            d,
        } => {
            // `SFPEXEXP`'s `NODEBIAS`: the exponent as it sits in the word, not
            // minus 127 (tt-metal's `ExponentMode::NoDebias`).
            out.push(encode::sfpexexp(u32::from(a), u32::from(d), 1).map_err(bad_isa)?);
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
        MirOp::Compare { op: cmp, a, b, d } => {
            let set_vd = 8u32;
            let (first, second) = match cmp {
                CmpOp::Lt => (b, a),
                CmpOp::Gt => (a, b),
                CmpOp::Ge => (b, a),
            };
            out.push(encode::sfpmov(u32::from(first), u32::from(d), 0).map_err(bad_isa)?);
            let i = match cmp {
                CmpOp::Lt | CmpOp::Gt => {
                    encode::sfpgt(u32::from(second), u32::from(d), set_vd).map_err(bad_isa)?
                }
                CmpOp::Ge => {
                    encode::sfple(u32::from(second), u32::from(d), set_vd).map_err(bad_isa)?
                }
            };
            out.push(i);
        }
        MirOp::Select { cond, t, f, d } => {
            // `d = f` in every lane, then `d = t` only where the condition's
            // lane is true. `SFPGT` sets the lane flags from the condition
            // (true means the sign-magnitude `0 > cond`, which is exactly a
            // bool stored as `-1`), and `SFPMOV` is predicated on them.
            out.push(encode::sfpmov(u32::from(f), u32::from(d), 0).map_err(bad_isa)?);
            out.push(cc_mask(true).map_err(bad_isa)?);
            out.push(
                encode::sfpgt(
                    u32::from(cond),
                    tt_isa::sfpu::LREG_ZERO,
                    1, // SFPGT_MOD1_SET_CC
                )
                .map_err(bad_isa)?,
            );
            out.push(encode::sfpmov(u32::from(t), u32::from(d), 0).map_err(bad_isa)?);
            out.push(cc_mask(false).map_err(bad_isa)?);
        }
        MirOp::Cast { from, to, a, d } => {
            cast_lanes(from, to, a.0 as u8, d.0 as u8, out)?;
        }
        MirOp::Max { a, b, d } => {
            out.push(encode::sfpmov(u32::from(b), u32::from(d), 0).map_err(bad_isa)?);
            out.push(cc_mask(true).map_err(bad_isa)?);
            out.push(
                encode::sfpgt(u32::from(b), u32::from(a), 1) // flags = a > b
                    .map_err(bad_isa)?,
            );
            out.push(encode::sfpmov(u32::from(a), u32::from(d), 0).map_err(bad_isa)?);
            out.push(cc_mask(false).map_err(bad_isa)?);
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
        MirOp::AluAccCtrl {
            fp32,
            zero_flag_src,
        } => {
            // The `Dst` width the moves address and the denormal policy of the
            // `Src` reads: tt-metal clears `Fp32_enabled` around the high-half
            // `MOVD2B`/`MOVB2D` transposes on Blackhole (issue #449), and keeps
            // source denormals so the low-half bits survive the moves.
            let mut words = ConfigWords::new();
            words
                .set(alu::ALU_ACC_CTRL_Fp32_enabled, u32::from(fp32))
                .expect("the value fits its field")
                .set(
                    alu::ALU_ACC_CTRL_Zero_Flag_disabled_src,
                    u32::from(zero_flag_src),
                )
                .expect("the value fits its field");
            out.extend(config_program(&words));
        }
        MirOp::SrcAFormat { dtype } => {
            // The moves take `SrcA`'s format while
            // `DISABLE_IMPLIED_SRCA_FMT` is set; the override pair is the
            // field the format switch writes.
            let format = match dtype {
                DType::F32 => DATA_FORMAT_FP32,
                DType::Bf16 => DATA_FORMAT_BF16,
                other => {
                    let _ = other;
                    return Err(Error::TT_ERROR_ENCODE);
                }
            };
            let mut words = ConfigWords::new();
            words
                .set(alu::ALU_FORMAT_SPEC_REG_SrcA_override, 1)
                .expect("the value fits its field")
                .set(alu::ALU_FORMAT_SPEC_REG_SrcA_val, format)
                .expect("the value fits its field");
            out.extend(config_program(&words));
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
            ));
        }
        MirOp::SemWait { sem, .. } => {
            let s = semaphore(sem)?;
            out.extend(sync::take(s, Before::EVERYTHING));
        }
        other => {
            let _ = other;
            return Err(Error::TT_ERROR_NOT_MATH);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/// Push one conversion word. The immediate and register fields are in range by
/// construction, so an encoding failure is a bug, not a caller's mistake.
fn place<E: std::fmt::Display>(out: &mut Stream, instruction: Result<Instruction, E>) {
    out.push(instruction.unwrap_or_else(|e| panic!("a conversion's word encodes: {e}")));
}

/// One lanewise conversion in the SFPU's 32-bit lane, with the rounding the C
/// type has: `(int)f` truncates toward zero, `(float)i` rounds to nearest even,
/// `(bool)x` is `x != 0`. `bool`'s true is `-1.0f` (`0xBF80_0000`), the value a
/// comparison writes; a cast *from* `bool` reads any negative lane as true.
///
/// `f32` to `i32` is tt-metal's `_calculate_typecast_fp32_to_int32_`: the
/// exponent picks the mantissa's shift, the bits below the binary point are
/// dropped (truncation), and a negative input is negated into two's complement.
/// A magnitude of `2^31` or more saturates to `INT_MIN`, as that kernel does;
/// CUDA leaves the result undefined there.
fn cast_lanes(from: DType, to: DType, a: u8, d: u8, out: &mut Stream) -> Result<(), Error> {
    use tt_isa::sfpu::{LREG_ZERO, loadi, loadi_mode};
    let a = u32::from(a);
    let d = u32::from(d);
    // Mod1 values from the vendored `sfpi_constants.h` (Blackhole).
    const EXEXP_CC_SGN_COMP: u32 = 2 | 8;
    const IADD_ARG_IMM: u32 = 1;
    const IADD_2SCOMP_LREG_DST: u32 = 2;
    const IADD_CC_NONE: u32 = 4;
    const SETCC_NE0: u32 = 2;
    const ENCC_ALL: u32 = 0;
    const SHFT_IMM_USE_VC: u32 = 1 | 4;
    const CAST_INT32_TO_FP32_RNE: u32 = 0;
    let scratch = (0u8..crate::alloc::LREGS)
        .find(|r| u32::from(*r) != a && u32::from(*r) != d)
        .ok_or(Error::TT_ERROR_LANE_REGISTER)?;
    let t = u32::from(scratch);
    match (from, to) {
        (DType::F32, DType::F32) | (DType::I32, DType::I32) | (DType::Bool, DType::Bool) => {
            place(out, encode::sfpmov(a, d, 0))
        }
        (DType::I32, DType::F32) => {
            // Two's complement to sign-magnitude, then sign-magnitude to FP32
            // (round to nearest even) -- the pair tt-metal's
            // `_calculate_typecast_int32_to_fp32_` splits into abs/cast/setsgn.
            place(out, encode::sfpcast(a, t, 3));
            place(out, encode::sfpcast(t, d, CAST_INT32_TO_FP32_RNE));
        }
        (DType::F32, DType::I32) => {
            // The exponent's lane flags decide which lanes saturate, so
            // conditional execution is on for the whole sequence.
            place(out, cc_mask(true));
            place(out, loadi(d, loadi_mode::USHORT, 0));
            place(out, encode::sfpexexp(a, t, EXEXP_CC_SGN_COMP));
            // Saturated lanes keep `INT_MIN` (`0x8000_0000`).
            place(out, loadi(d, loadi_mode::FLOATB, 0x8000));
            place(
                out,
                encode::sfpiadd((-31i32 & 0xfff) as u32, t, t, IADD_ARG_IMM),
            );
            place(out, encode::sfpiadd(8, t, t, IADD_ARG_IMM | IADD_CC_NONE));
            place(out, encode::sfpexman(a, d, 0));
            place(out, encode::sfpshft(0, t, d, 0));
            place(out, encode::sfpencc(0, 0, ENCC_ALL));
            place(out, encode::sfpsetcc(0, a, 0, 0));
            place(
                out,
                encode::sfpiadd(0, LREG_ZERO, d, IADD_2SCOMP_LREG_DST | IADD_CC_NONE),
            );
            place(out, cc_mask(false));
        }
        // A comparison's true has the sign bit set and its false is zero, so a
        // logical shift by 31 is the C value: one or zero.
        (DType::Bool, DType::I32) => place(
            out,
            encode::sfpshft((-31i32 & 0xfff) as u32, a, d, SHFT_IMM_USE_VC),
        ),
        (DType::Bool, DType::F32) => {
            place(
                out,
                encode::sfpshft((-31i32 & 0xfff) as u32, a, d, SHFT_IMM_USE_VC),
            );
            place(out, encode::sfpcast(d, d, CAST_INT32_TO_FP32_RNE));
        }
        (from, DType::Bool) => {
            place(out, cc_mask(true));
            if from == DType::F32 {
                // `|x|` first: `-0.0` is zero and a NaN is not.
                place(out, encode::sfpabs(a, t, 0));
            }
            place(out, encode::sfpmov(LREG_ZERO, d, 0));
            place(
                out,
                encode::sfpsetcc(0, if from == DType::F32 { t } else { a }, 0, SETCC_NE0),
            );
            place(out, loadi(d, loadi_mode::UPPER, 0xBF80));
            place(out, cc_mask(false));
        }
        (DType::Bf16, _) | (_, DType::Bf16) => {
            let _ = (from, to);
            return Err(Error::TT_TODO_CAST);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The pack core (T2)
// ---------------------------------------------------------------------------

fn encode_pack(ops: &[MirOp], out: &mut Stream, dst_is_16_bit: bool) -> Result<(), Error> {
    if ops.is_empty() {
        return Ok(());
    }
    // The `ThreadConfig` entries come first, as at the start of every tt-metal
    // kernel: `CFG_STATE_ID` decides which `Config` bank later writes land in, so
    // no config store may race with them.
    thread_config(out);
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
            MirOp::SemPost { sem } => {
                out.extend(sync::post_after(sync::Unit::Packer, semaphore(sem)?));
            }
            MirOp::SemWaitFree { sem } => {
                out.push(sync::wait_below_max(semaphore(sem)?, Before::EVERYTHING));
            }
            MirOp::CbReserve { cb } => {
                out.asm(|a| {
                    rv::cb::reserve_back(a, reg::A6, u32::from(cb), 1, [reg::T0, reg::T1, reg::T2])
                })?;
            }
            MirOp::CbPush { cb } => {
                // The page is only visible to the writer once the pack has
                // written it: the `STALLWAIT` holds back later pushed work, and
                // the coprocessor check holds this core's store back until the
                // packer has retired and its L1 writes have landed.
                out.push(wait_for_packer(Before::EVERYTHING).expect("named bits"));
                out.asm(rv::coprocessor_done)?;
                out.asm(|a| {
                    rv::cb::push_back(
                        a,
                        reg::A6,
                        u32::from(cb),
                        1,
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
                if last_config == Some((dtype, mask)) {
                    continue;
                }
                let mut words = ConfigWords::new();
                pack_config(&mut words, dtype, dst_is_16_bit, mask);
                out.extend(config_program(&words));
                out.push(stallwait(block::PACKER | block::CONFIG, cond::CONFIG_BUSY)?);
                last_config = Some((dtype, mask));
            }
            MirOp::SetPackOffset { reg, at } => {
                if last_offset == Some((reg, at)) {
                    continue;
                }
                let mut words = ConfigWords::new();
                words
                    .set(
                        global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset,
                        dst_row(reg, at, 0),
                    )
                    .expect("the row fits its field");
                out.extend(config_program(&words));
                out.push(stallwait(block::PACKER | block::CONFIG, cond::CONFIG_BUSY)?);
                last_offset = Some((reg, at));
            }
            MirOp::PackRows { rows, last } => {
                if rows == 0 {
                    return Err(Error::TT_ERROR_PACK_ZERO_ROWS);
                }
                pack_rows(out, rows, last)?;
            }
            other => {
                let _ = other;
                return Err(Error::TT_ERROR_PACK_OP);
            }
        }
    }
    if !loops.is_empty() {
        return Err(Error::TT_ERROR_ENDLOOP_MACHINE);
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
fn pack_config(words: &mut ConfigWords, dtype: DType, dst_is_16_bit: bool, mask: PackMask) {
    // `Out_data_format` is the L1 datum's own format; `In_data_format` is the
    // kernel's `Dst` width. In 16-bit mode the packer reads a 16-bit `Dst` (two
    // bytes per datum, so half the row stride) and its conversion is the
    // identity; in 32-bit mode it reads a 32-bit one and converts, which is how
    // a bf16 buffer is written out of an f32 `Dst`.
    let out = l1_format(dtype);
    let in_ = dst_format(dst_is_16_bit);
    let narrow = dst_is_16_bit;
    words
        .set(thcon::THCON_SEC0_REG1_In_data_format, in_)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Out_data_format, out)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Source_interface_selection, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Disable_zero_compress, 1)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Add_l1_dest_addr_offset, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Enable_out_fifo, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Exp_section_size, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Row_start_section_size, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Downsample_rate, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Downsample_mask, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Exp_threshold_en, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Pack_L1_Acc, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_pack_start_intf_pos, 0)
        .unwrap()
        .set(pack0::PCK0_ADDR_BASE_REG_1_Base, 0)
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_XY_REG_1_Ystride, 0)
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_ZW_REG_1_Zstride, 0)
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_XY_REG_0_Xstride, 0)
        .unwrap()
        .set(
            pack0::PCK0_ADDR_CTRL_XY_REG_0_Ystride,
            if narrow {
                DST_ROW_BYTES / 2
            } else {
                DST_ROW_BYTES
            },
        )
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_ZW_REG_0_Zstride, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_32b_data, u32::from(!narrow))
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_unsigned, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_int8, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Round_10b_mant, 0)
        .unwrap()
        .set(pack0::PACK_COUNTERS_SEC0_pack_reads_per_xy_plane, 16)
        .unwrap()
        .set(pack0::PCK_EDGE_MODE_mode, 0)
        .unwrap()
        // The edge mask is rewritten on every pack: the packer's `Config` bank
        // outlives one pack, so a reduce's mask must not leak into the next
        // pack. `None` is tt-metal's `_llk_pack_reduce_mask_clear_`: both masks
        // pass everything and no row selects the second one.
        .set(
            pack0::PCK_EDGE_OFFSET_SEC0_mask,
            match mask {
                PackMask::None => 0xFFFF,
                PackMask::Col | PackMask::Row => 0x0000,
            },
        )
        .unwrap()
        .set(
            pack0::PCK_EDGE_OFFSET_SEC1_mask,
            match mask {
                PackMask::None => 0xFFFF,
                PackMask::Col => 0xFFFF,
                // tt-metal's `REDUCE_ROW`: only datum 0 of the row survives.
                PackMask::Row => 0x0001,
            },
        )
        .unwrap()
        .set(
            pack0::PCK_EDGE_TILE_ROW_SET_SELECT_select,
            match mask {
                PackMask::None => 0,
                // Each packer's two-bit field is the mapping register it
                // reads: packers 0 and 1 both use `TILE_ROW_SET_MAPPING_1`.
                PackMask::Col => 0b0101,
                // `REDUCE_ROW` gives every packer mapping 1.
                PackMask::Row => 0b0101_0101,
            },
        )
        .unwrap()
        .set(
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_0,
            match mask {
                PackMask::None => 0,
                PackMask::Col => 1,
                // Every one of a face's sixteen rows selects the `SEC1`
                // mask, i.e. keeps column 0.
                PackMask::Row => 1,
            },
        )
        .unwrap()
        .set(alu::ALU_ACC_CTRL_Fp32_enabled, u32::from(!narrow))
        .unwrap();
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
            words.set(field, 1).unwrap();
        }
    }
}

/// Pack `rows` rows of `Dst` from the offset [`MirOp::SetPackOffset`] set, to
/// the page [`MirOp::SetPackBase`] named.
///
/// The address generators' per-program setup (`ADDR_MOD_PACK_*_YsrcIncr`,
/// `SETADCXX`) is in `encode_pack`'s prologue, like every other invariant
/// `Config` word; only the Y reset is per pack, because the `PACR`s leave Y at
/// the end of the run they moved.
fn pack_rows(out: &mut Stream, rows: u32, last: bool) -> Result<(), Error> {
    out.push(thread_entry(thread::ADDR_MOD_PACK_SEC0_YsrcIncr, 0));
    out.push(thread_entry(thread::ADDR_MOD_PACK_SEC1_YsrcIncr, 4));
    out.push(
        encode::Setadcxx::ZERO
            .pk(1)
            .x0_val(0)
            .x1_val(15)
            .encode()
            .map_err(|e| {
                let _ = e;
                Error::TT_ERROR_SETADCXX_PACK
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
                Error::TT_ERROR_SETADCXY_PACK
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
                    Error::TT_ERROR_PACR
                })?,
        );
    }
    Ok(())
}

/// Wait for the packer to drain, then post. Used by a program that has to know
/// its own output is complete.
#[allow(dead_code)]
fn pack_done(sem: Semaphore) -> [Instruction; 2] {
    [
        wait_for_packer(Before::EVERYTHING).expect("named bits"),
        sync::post(sem),
    ]
}

#[allow(dead_code)]
fn unpack_done() -> Instruction {
    wait_for_unpacker0(Before::EVERYTHING).expect("named bits")
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
            m.push(Core::Pack, MirOp::CbReserve { cb: Cb(0) });
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
            m.push(Core::Pack, MirOp::CbPush { cb: Cb(0) });
        }
        m
    }

    fn pack_words(count: u8) -> usize {
        let words = encode(&packs(count))
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
}
