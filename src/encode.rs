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
use crate::mir::{CmpOp, IOp, Machine, MirOp, SfpOp};
use crate::riscv::{self as rv, Asm, reg};
use tt_isa::backend::{
    self, Before, ConfigWords, ThreadConfigEntry, block, cond, wait_for_packer, wait_for_unpacker0,
};
use tt_isa::cfg::generated::{alu, global, pack0, thcon, thread, unpack1};
use tt_isa::isa::Instruction;
use tt_isa::isa::generated::{defs, encode};
use tt_isa::sync::{self, Semaphore};
use tt_isa::tile::{L1Format, TileDescriptor};

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
        let mut a = Asm::new();
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

/// One open [`MirOp::Loop`]: the register holding the remaining trip count, the
/// word the forward skip is patched into, and the word the body starts at.
struct OpenLoop {
    reg: u8,
    top: usize,
    skip: usize,
}

/// Open a loop: `reg = count; if reg == 0 goto end; top:`.
///
/// The skip's target is the end, which is not known until [`close_loop`], so
/// that one word is patched there. One caller-saved register per nesting level:
/// `T4` then `T5`. A deeper loop is a concrete future item, not a guess at a
/// register.
fn open_loop(out: &mut Stream, loops: &mut Vec<OpenLoop>, count: u32) -> Result<(), Error> {
    let reg = match loops.len() {
        0 => reg::T4,
        1 => reg::T5,
        _ => return Err(Error::TT_TODO_LOOP_DEPTH),
    };
    out.li(reg, count);
    let skip = out.len();
    out.rv(rv::beq(reg, reg::ZERO, 0));
    let top = out.len();
    loops.push(OpenLoop { reg, top, skip });
    Ok(())
}

/// Close a loop: `addi reg, reg, -1; bne reg, zero, top; end:` and the patched
/// forward skip of [`open_loop`].
fn close_loop(out: &mut Stream, loops: &mut Vec<OpenLoop>) -> Result<(), Error> {
    let loop_ = loops.pop().ok_or(Error::TT_ERROR_ENDLOOP_MACHINE)?;
    out.rv(rv::addi(loop_.reg, loop_.reg, -1));
    let here = out.len();
    out.rv(rv::bne(loop_.reg, reg::ZERO, 0));
    let after = out.len();
    // The back branch is emitted at `here` and targets `top`.
    let back = ((loop_.top as i32) - (here as i32)) * 4;
    out.patch(here, rv::bne(loop_.reg, reg::ZERO, back));
    let forward = ((after as i32) - (loop_.skip as i32)) * 4;
    out.patch(loop_.skip, rv::beq(loop_.reg, reg::ZERO, forward));
    Ok(())
}

/// Open a loop whose count is already in `reg` (a dynamic bound). The branch
/// and the counter are the same as [`open_loop`]'s; only the `li` is missing.
fn open_loop_reg(out: &mut Stream, loops: &mut Vec<OpenLoop>, reg: u8) -> Result<(), Error> {
    if loops.len() >= 2 {
        return Err(Error::TT_TODO_LOOP_DEPTH);
    }
    let skip = out.len();
    out.rv(rv::beq(reg, reg::ZERO, 0));
    let top = out.len();
    loops.push(OpenLoop { reg, top, skip });
    Ok(())
}

/// Encode one integer-scratch op. Returns `false` when `op` is not one, so a
/// role encoder can fall through to its own instruction set.
///
/// These are the ops `lower` emits before a [`MirOp::LoopReg`] to compute a
/// bound that is not a literal: a launch argument (a scalar parameter's value),
/// this core's coordinate, or arithmetic over them. They are ordinary RISC-V,
/// the way a C kernel computes its loop bound.
fn encode_int(op: &MirOp, out: &mut Stream, loops: &mut Vec<OpenLoop>) -> Result<bool, Error> {
    match *op {
        MirOp::Li { reg, value } => out.li(reg, value),
        MirOp::LoadArg { reg, arg } => out.rv(rv::lw(reg, reg::A0, arg)),
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
        MirOp::LoopReg { reg } => open_loop_reg(out, loops, reg)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// One instruction stream per compute core.
#[derive(Debug, Clone, Default)]
pub struct Words {
    t0: Stream,
    t1: Stream,
    t2: Stream,
}

impl Words {
    /// The encoded words, as the program slot takes them.
    pub fn words(&self) -> Result<[Vec<u32>; 3], Error> {
        Ok([self.t0.words()?, self.t1.words()?, self.t2.words()?])
    }
}

/// Encode only the math core.
pub fn math(m: &Machine) -> Result<Vec<u32>, Error> {
    let mut out = Stream::default();
    encode_math(&m.t1, &mut out)?;
    out.words()
}

/// Encode all three compute programs.
pub fn encode(m: &Machine) -> Result<Words, Error> {
    let mut out = Words::default();
    encode_unpack(&m.t0, &mut out.t0)?;
    encode_math(&m.t1, &mut out.t1)?;
    encode_pack(&m.t2, &mut out.t2)?;
    Ok(out)
}

/// The setup program thread 0 pushes before the three roles are released: the
/// `SEMINIT`s for the semaphores the compute programs hand off through.
///
/// tt-metal does exactly this from the core that runs first: BRISC's
/// `device_setup` (`tt_llk_*/tests/helpers/include/boot.h`) calls
/// `t6_semaphore_init(sem, 0, 1)` for each semaphore and only then releases the
/// three TRISCs. Initialising from one core *before* the others start is what
/// makes it race-free; a `SEMINIT` issued by a role itself could reset a value
/// another role has already posted. The semaphore that counts free `Dst` slots
/// starts at one instead of zero, because the first load has a slot to fill.
pub fn setup(slots: u8) -> Result<Vec<u32>, Error> {
    let mut out: Vec<u32> = semaphore_init(slots)?
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

/// `SEMINIT` each semaphore with max 1 — the boot sequence's shape.
fn semaphore_init(slots: u8) -> Result<Vec<Instruction>, Error> {
    let mut out = Vec::new();
    let max = sync::MAX_VALUE;
    for (index, value) in [
        (crate::lower::SEM_UNPACKED, 0),
        (crate::lower::SEM_READY, 0),
        // Every `Dst` slot this kernel uses is free at the start; the packer
        // hands each back. The count is the allocation's peak liveness, not the
        // whole bank: with more tokens than the program can have live units, the
        // unpacker would run a second unit into a slot the packer is still
        // reading.
        (crate::lower::SEM_SLOT, slots),
    ] {
        let sem = semaphore(index)?;
        out.push(sync::init(sem, value, max).map_err(|_| {
            let _ = index;
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

fn semaphore(index: u8) -> Result<Semaphore, Error> {
    Semaphore::new(index).ok_or_else(|| {
        let _ = index;
        Error::TT_ERROR_SEMAPHORE
    })
}

// ---------------------------------------------------------------------------
// The unpack core (T0)
// ---------------------------------------------------------------------------

/// What the next `UNPACR` is going to read and where it lands. The `SET*`
/// machine ops before it set these; the descriptor is written just before the
/// unpack that uses it.
#[derive(Default)]
struct UnpackState {
    dst_row: u32,
    x0: u32,
    x1: u32,
    face: u32,
}

fn encode_unpack(ops: &[MirOp], out: &mut Stream) -> Result<(), Error> {
    if ops.is_empty() {
        return Ok(());
    }
    // The `ThreadConfig` entries come first, as at the start of every tt-metal
    // kernel: `CFG_STATE_ID` decides which `Config` bank later writes land in, so
    // no config store may race with them.
    thread_config(out);
    // Every role is called with `a0` = the launch arguments; word 2 is the CB
    // interface table (`tt-metal`'s `ARG_IFACE`).
    out.rv(rv::lw(reg::A6, reg::A0, crate::mover::ARG_IFACE as i32 * 4));
    let mut state = UnpackState::default();
    let mut loops = Vec::new();
    for op in ops {
        if encode_int(op, out, &mut loops)? {
            continue;
        }
        match *op {
            MirOp::Loop { count } => open_loop(out, &mut loops, count)?,
            MirOp::EndLoop => close_loop(out, &mut loops)?,
            MirOp::Setdst { row } => state.dst_row = row,
            MirOp::Setadcx { first, count, .. } => {
                if count == 0 {
                    return Err(Error::TT_ERROR_UNPACK_ZERO_RANGE);
                }
                state.x0 = first;
                state.x1 = first + count - 1;
            }
            MirOp::Setadczw { face, .. } => state.face = face,
            MirOp::Unpacr {
                cb, count, dtype, ..
            } => {
                if count == 0 {
                    return Err(Error::TT_ERROR_UNPACK_ZERO_DATUMS);
                }
                state.x0 = 0;
                state.x1 = count - 1;
                // The page is at the ring's read pointer, not a compile-time L1
                // address: write that pointer into the unpacker's base register
                // after draining the previous unpack.
                cb_base_unpack(out, cb)?;
                let mut words = ConfigWords::new();
                unpack_config(&mut words, count, state.dst_row, dtype);
                out.extend(config_program(&words));
                out.push(stallwait(
                    block::UNPACKER | block::CONFIG,
                    cond::CONFIG_BUSY,
                )?);
                out.push(set_adc_x_unpack(state.x0, state.x1)?);
                if state.face != 0 {
                    out.push(set_adc_z_unpack(state.face)?);
                }
                out.push(unpack_instruction()?);
            }
            MirOp::CbWait { cb } => {
                out.asm(|a| rv::cb::wait_front(a, reg::A6, cb, 1, [reg::T0, reg::T1, reg::T2]))?;
            }
            MirOp::CbPop { cb } => {
                // The page is only free once the unpacker has read it.
                out.push(
                    wait_for_unpacker0(Before::EVERYTHING)
                        .map_err(|_| Error::TT_ERROR_STALLWAIT)?,
                );
                out.asm(|a| {
                    rv::cb::pop_front(a, reg::A6, cb, 1, [reg::T0, reg::T1, reg::T2, reg::T6])
                })?;
            }
            MirOp::SemPost { sem } => {
                out.extend(sync::post_after(sync::Unit::Unpacker0, semaphore(sem)?));
            }
            MirOp::SemWait { sem, .. } => {
                out.extend(sync::take(semaphore(sem)?, Before::EVERYTHING));
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
fn cb_base_unpack(out: &mut Stream, cb: u32) -> Result<(), Error> {
    out.push(wait_for_unpacker0(Before::EVERYTHING).map_err(|_| Error::TT_ERROR_STALLWAIT)?);
    cb_base(
        out,
        cb,
        rv::cb::RD_PTR,
        thcon::THCON_SEC0_REG3_Base_address.addr32(),
    )
}

/// The packer's output base: the ring's write pointer, same 16-byte conversion.
fn cb_base_pack(out: &mut Stream, cb: u32) -> Result<(), Error> {
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
/// role's per-iteration page then lags by one (`firmware/tt-firmware/src/
/// ttinsn.rs`, `STEP_STORE`). The write has to be a pushed Tensix instruction:
/// `SETDMAREG` stages the dynamic value, then `WRCFG` copies it, in FIFO order
/// with the backend. This is tt-metal's `program_packer_destination`
/// (`SETDMAREG` × 2, `STALLWAIT(STALL_CFG, THCON)`, `WRCFG`, `DMANOP`) and
/// `_llk_unpack_configure_addresses_`' ordering requirement.
///
/// The caller's `wait_for_unpacker0`/`wait_for_packer` now reaches the write:
/// because the write is a pushed instruction, `STALLWAIT`'s CONFIG block holds
/// it until the previous unit has drained.
fn cb_base(out: &mut Stream, cb: u32, field: i32, config: u16) -> Result<(), Error> {
    out.asm(|a| {
        a.word(rv::lw(
            reg::T0,
            reg::A6,
            (cb * rv::cb::STRIDE) as i32 + field,
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
fn l1_format(dtype: DType) -> L1Format {
    match dtype {
        DType::Bf16 => L1Format::Bf16,
        _ => L1Format::Fp32,
    }
}

/// The unpacker's `Config` for one unit: base, destination, X dimension, FP32
/// in and out. The base is `base/16 - 1` because L1 holds the datum block with
/// no 16-byte tile header, while the unpacker adds one unconditionally.
fn unpack_config(words: &mut ConfigWords, datums: u32, dst_row: u32, dtype: DType) {
    // The datum is opaque 32 bits at this layer: `Int32` in the packed output
    // requires the `Dst` access format to match it, and the role descriptor's is
    // 32-bit. The lane's own format is `dtype`'s, which decides the conversion.
    let fmt = l1_format(dtype);
    words
        .set(thcon::THCON_SEC0_REG7_Offset_address, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0, datums)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Out_data_format, fmt.code().unwrap())
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
    let descriptor = TileDescriptor::zeroed()
        .with_x_dim(datums)
        .with_y_dim(1)
        .with_z_dim(0)
        .with_w_dim(0)
        .with_is_uncompressed(true)
        .with_in_data_format_raw(fmt.code().unwrap());
    // Words 0 and 1 always; 2 and 3 only when they carry something (ttsim
    // refuses a write of the all-zero tail).
    for (i, word) in descriptor.words().iter().enumerate() {
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

/// The `REG5_Dest_cntx0_address` that lands an unpack's first datum in `Dst`
/// row `row`: `Dst` row is `OutAddr / 16 - 4`.
pub const fn dst_address(row: u32) -> u32 {
    (row + 4) * 16
}

fn set_adc_x_unpack(first: u32, last: u32) -> Result<Instruction, Error> {
    encode::Setadcxx::ZERO
        .u0(1)
        .x0_val(first)
        .x1_val(last)
        .encode()
        .map_err(|e| {
            let _ = e;
            Error::TT_ERROR_SETADCXX
        })
}

fn set_adc_z_unpack(face: u32) -> Result<Instruction, Error> {
    encode::Setadczw::ZERO
        .u0(1)
        .z0(1)
        .z0_val(face)
        .encode()
        .map_err(|e| {
            let _ = e;
            Error::TT_ERROR_SETADCZW
        })
}

fn unpack_instruction() -> Result<Instruction, Error> {
    // Bit 0 is undocumented and ttsim refuses the instruction without it.
    let base = encode::UnpacrRegular::ZERO
        .which_unpacker(0)
        .multi_context_mode(1)
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

fn encode_math(ops: &[MirOp], out: &mut Stream) -> Result<(), Error> {
    let mut at = 0usize;
    let mut loops = Vec::new();
    while at < ops.len() {
        if encode_int(&ops[at], out, &mut loops)? {
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
        MirOp::Sfpload { lreg, row, dtype } => {
            let i = encode::sfpload(u32::from(lreg), lane_format(dtype), 0, row + row_delta)
                .map_err(bad_isa)?;
            out.push(i);
        }
        MirOp::Sfpstore { lreg, row, dtype } => {
            let i = encode::sfpstore(u32::from(lreg), lane_format(dtype), 0, row + row_delta)
                .map_err(bad_isa)?;
            out.push(i);
        }
        MirOp::Sfp {
            op: SfpOp::Neg,
            a,
            b: None,
            d,
        } => {
            out.push(encode::sfpmov(u32::from(a), u32::from(d), 1).map_err(bad_isa)?);
        }
        MirOp::Sfp {
            op: SfpOp::Abs,
            a,
            b: None,
            d,
        } => {
            out.push(encode::sfpabs(u32::from(a), u32::from(d), 1).map_err(bad_isa)?);
        }
        MirOp::Sfp {
            op: SfpOp::Add,
            a,
            b: Some(b),
            d,
        } => {
            out.push(
                tt_isa::sfpu::add(u32::from(a), u32::from(b), u32::from(d))
                    .map_err(|e| bad(e, op))?,
            );
        }
        MirOp::Sfp {
            op: SfpOp::Sub,
            a,
            b: Some(b),
            d,
        } => {
            out.push(
                tt_isa::sfpu::sub(u32::from(a), u32::from(b), u32::from(d))
                    .map_err(|e| bad(e, op))?,
            );
        }
        MirOp::Sfp {
            op: SfpOp::Mul,
            a,
            b: Some(b),
            d,
        } => {
            out.push(
                tt_isa::sfpu::mul(u32::from(a), u32::from(b), u32::from(d))
                    .map_err(|e| bad(e, op))?,
            );
        }
        MirOp::Sfp {
            op: SfpOp::BitAnd,
            a,
            b: Some(b),
            d,
        } => {
            out.push(encode::sfpand(u32::from(b), u32::from(a), u32::from(d), 1).map_err(bad_isa)?);
        }
        MirOp::Sfp {
            op: SfpOp::BitOr,
            a,
            b: Some(b),
            d,
        } => {
            out.push(encode::sfpor(u32::from(b), u32::from(a), u32::from(d), 1).map_err(bad_isa)?);
        }
        MirOp::Sfp {
            op: SfpOp::BitXor,
            a,
            b: Some(b),
            d,
        } => {
            out.push(encode::sfpmov(u32::from(a), u32::from(d), 0).map_err(bad_isa)?);
            out.push(encode::sfpxor(u32::from(b), u32::from(d)).map_err(bad_isa)?);
        }
        MirOp::Sfp {
            op: SfpOp::BitNot,
            a,
            b: None,
            d,
        } => {
            out.push(encode::sfpnot(u32::from(a), u32::from(d)).map_err(bad_isa)?);
        }
        MirOp::Sfp { .. } => {
            let _ = op;
            return Err(Error::TT_ERROR_NO_ENCODING);
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
            cast_lanes(from, to, a, d, out)?;
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
        MirOp::SemPost { sem } => {
            let s = semaphore(sem)?;
            out.extend(sync::post_after(sync::Unit::Sfpu, s));
        }
        MirOp::SemWait { sem, .. } => {
            let s = semaphore(sem)?;
            out.extend(sync::take(s, Before::EVERYTHING));
        }
        MirOp::SemWaitData { sem } => {
            let s = semaphore(sem)?;
            out.push(sync::wait_nonzero(s, Before::EVERYTHING));
        }
        MirOp::SemGet { sem } => {
            out.push(sync::get(semaphore(sem)?));
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

fn encode_pack(ops: &[MirOp], out: &mut Stream) -> Result<(), Error> {
    if ops.is_empty() {
        return Ok(());
    }
    // The `ThreadConfig` entries come first, as at the start of every tt-metal
    // kernel: `CFG_STATE_ID` decides which `Config` bank later writes land in, so
    // no config store may race with them.
    thread_config(out);
    // `a0` = the launch arguments; word 2 is the CB interface table.
    out.rv(rv::lw(reg::A6, reg::A0, crate::mover::ARG_IFACE as i32 * 4));
    let mut loops = Vec::new();
    for op in ops {
        if encode_int(op, out, &mut loops)? {
            continue;
        }
        match *op {
            MirOp::Loop { count } => open_loop(out, &mut loops, count)?,
            MirOp::EndLoop => close_loop(out, &mut loops)?,
            MirOp::SemWait { sem, .. } => {
                out.extend(sync::take(semaphore(sem)?, Before::EVERYTHING));
            }
            MirOp::SemWaitData { sem } => {
                out.push(sync::wait_nonzero(semaphore(sem)?, Before::EVERYTHING));
            }
            MirOp::SemGet { sem } => {
                // The claim is released only once the pack has read `Dst`, the
                // order `_llk_pack_dest_section_done_` uses.
                out.push(wait_for_packer(Before::EVERYTHING).expect("named bits"));
                out.push(sync::get(semaphore(sem)?));
            }
            MirOp::SemPost { sem } => {
                out.extend(sync::post_after(sync::Unit::Packer, semaphore(sem)?));
            }
            MirOp::CbReserve { cb } => {
                out.asm(|a| rv::cb::reserve_back(a, reg::A6, cb, 1, [reg::T0, reg::T1, reg::T2]))?;
            }
            MirOp::CbPush { cb } => {
                // The page is only visible to the writer once the pack has
                // written it: the `STALLWAIT` holds back later pushed work, and
                // the coprocessor check holds this core's store back until the
                // packer has retired and its L1 writes have landed.
                out.push(wait_for_packer(Before::EVERYTHING).expect("named bits"));
                out.asm(rv::coprocessor_done)?;
                out.asm(|a| {
                    rv::cb::push_back(a, reg::A6, cb, 1, [reg::T0, reg::T1, reg::T2, reg::T6])
                })?;
            }
            MirOp::Pack {
                cb,
                rows,
                row,
                last,
                dtype,
            } => {
                if rows == 0 {
                    return Err(Error::TT_ERROR_PACK_ZERO_ROWS);
                }
                // The destination is the ring's write pointer, converted to
                // 16-byte units minus the header unit the packer adds.
                cb_base_pack(out, cb)?;
                let mut words = ConfigWords::new();
                pack_config(&mut words, dtype, row);
                out.extend(config_program(&words));
                out.push(stallwait(block::PACKER | block::CONFIG, cond::CONFIG_BUSY)?);
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

/// The packer's `Config` for one unit: read `Dst` as 32-bit from row `row`,
/// write FP32 to the ring's page. The output address is in 16-byte units and
/// the generator always adds the one 16-byte tile header unit it never writes;
/// the base store takes the same `- 1`.
///
/// `DEST_TARGET_REG_CFG_PACK_SEC0_Offset` is the `Dst` row the packer reads
/// from (`Dst` addresses are 16-word rows, and tt-metal's own values for this
/// field — 0 and `DEST_REGISTER_HALF_SIZE`, 512 — are row counts). tt-metal
/// splits the same job: this field holds the half base, and
/// `set_dst_write_addr(tile_index)` puts the tile within it in the packer's `W`
/// address counter (`TT_SETADC(PAC, CH_0, SET_W, tile_index)`). The simulator
/// does not model that counter — a `SETADC` there leaves the pack reading row
/// 0 — so the row is written here, per pack, which is also what zyx did.
fn pack_config(words: &mut ConfigWords, dtype: DType, row: u32) {
    // The datum's own format decides the read path: a bf16 tile is read from a
    // 16-bit `Dst` (two bytes per datum, so half the row stride) and written as
    // bf16; every other datum is opaque 32 bits.
    let fmt = l1_format(dtype);
    let bf16 = dtype == DType::Bf16;
    words
        .set(thcon::THCON_SEC0_REG1_In_data_format, fmt.code().unwrap())
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Out_data_format, fmt.code().unwrap())
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
            if bf16 {
                DST_ROW_BYTES / 2
            } else {
                DST_ROW_BYTES
            },
        )
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_ZW_REG_0_Zstride, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_32b_data, u32::from(!bf16))
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_unsigned, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_int8, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Round_10b_mant, 0)
        .unwrap()
        .set(pack0::PACK_COUNTERS_SEC0_pack_reads_per_xy_plane, 4)
        .unwrap()
        .set(global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset, row)
        .unwrap()
        .set(pack0::PCK_EDGE_MODE_mode, 0)
        .unwrap()
        .set(pack0::PCK_EDGE_OFFSET_SEC0_mask, 0xFFFF)
        .unwrap()
        .set(alu::ALU_ACC_CTRL_Fp32_enabled, u32::from(!bf16))
        .unwrap();
}

/// Pack `rows` rows of FP32 `Dst` from the row `pack_config` set, to the run it
/// set up.
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
