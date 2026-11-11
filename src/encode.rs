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

use crate::ir::Error;
use crate::mir::{CmpOp, Machine, MirOp, SfpOp};
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

/// One instruction stream per compute core.
#[derive(Debug, Clone, Default)]
pub struct Words {
    pub t0: Vec<Instruction>,
    pub t1: Vec<Instruction>,
    pub t2: Vec<Instruction>,
}

impl Words {
    /// The encoded words, as the program slot takes them.
    pub fn words(&self) -> [Vec<u32>; 3] {
        [
            self.t0.iter().map(|i| i.word()).collect(),
            self.t1.iter().map(|i| i.word()).collect(),
            self.t2.iter().map(|i| i.word()).collect(),
        ]
    }
}

/// Encode only the math core.
pub fn math(m: &Machine) -> Result<Vec<Instruction>, Error> {
    let mut out = Vec::new();
    encode_math(&m.t1, &mut out)?;
    Ok(out)
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
    Ok(semaphore_init(slots)?
        .into_iter()
        .map(|i| i.word())
        .collect())
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
        out.push(
            sync::init(sem, value, max)
                .map_err(|e| Error(format!("SEMINIT on semaphore {index}: {e}")))?,
        );
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
    Semaphore::new(index).ok_or_else(|| Error(format!("semaphore {index} does not exist")))
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
    started: bool,
}

fn encode_unpack(ops: &[MirOp], out: &mut Vec<Instruction>) -> Result<(), Error> {
    let mut state = UnpackState::default();
    for op in ops {
        match *op {
            MirOp::Setdst { row } => state.dst_row = row,
            MirOp::Setadcx { first, count, .. } => {
                if count == 0 {
                    return Err(Error("an unpack with a zero X range".into()));
                }
                state.x0 = first;
                state.x1 = first + count - 1;
            }
            MirOp::Setadczw { face, .. } => state.face = face,
            MirOp::Unpacr { src, count, .. } => {
                if !state.started {
                    thread_config(out);
                    state.started = true;
                }
                if count == 0 {
                    return Err(Error("an unpack of zero datums".into()));
                }
                state.x0 = 0;
                state.x1 = count - 1;
                let mut words = ConfigWords::new();
                unpack_config(&mut words, src, count, state.dst_row);
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
            MirOp::SemPost { sem } => {
                out.extend(sync::post_after(sync::Unit::Unpacker0, semaphore(sem)?));
            }
            MirOp::SemWait { sem, .. } => {
                out.extend(sync::take(semaphore(sem)?, Before::EVERYTHING));
            }
            other => return Err(Error(format!("the unpack program cannot encode {other:?}"))),
        }
    }
    Ok(())
}

/// `SETC16.md`: the first `ThreadConfig` writes any thread must make. Emitted
/// once per role program, before anything reads the configuration bank.
fn thread_config(out: &mut Vec<Instruction>) {
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
    backend::stallwait(block_mask, condition).map_err(|e| Error(format!("stallwait: {e}")))
}

/// The unpacker's `Config` for one unit: base, destination, X dimension, FP32
/// in and out. The base is `base/16 - 1` because L1 holds the datum block with
/// no 16-byte tile header, while the unpacker adds one unconditionally.
fn unpack_config(words: &mut ConfigWords, src: u32, datums: u32, dst_row: u32) {
    let base = src / 16 - 1;
    words
        .set(thcon::THCON_SEC0_REG3_Base_address, base)
        .unwrap()
        .set(thcon::THCON_SEC0_REG7_Offset_address, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0, datums)
        .unwrap()
        .set(
            thcon::THCON_SEC0_REG2_Out_data_format,
            L1Format::Fp32.code().unwrap(),
        )
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
        .with_in_data_format_raw(L1Format::Fp32.code().unwrap());
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
        .map_err(|e| Error(format!("SETADCXX: {e}")))
}

fn set_adc_z_unpack(face: u32) -> Result<Instruction, Error> {
    encode::Setadczw::ZERO
        .u0(1)
        .z0(1)
        .z0_val(face)
        .encode()
        .map_err(|e| Error(format!("SETADCZW: {e}")))
}

fn unpack_instruction() -> Result<Instruction, Error> {
    // Bit 0 is undocumented and ttsim refuses the instruction without it.
    let base = encode::UnpacrRegular::ZERO
        .which_unpacker(0)
        .multi_context_mode(1)
        .encode()
        .map_err(|e| Error(format!("UNPACR: {e}")))?;
    Ok(Instruction::new(base.word() | 1, &defs::UNPACR_Regular))
}

// ---------------------------------------------------------------------------
// The math core (T1)
// ---------------------------------------------------------------------------

fn encode_math(ops: &[MirOp], out: &mut Vec<Instruction>) -> Result<(), Error> {
    let mut at = 0usize;
    while at < ops.len() {
        match ops[at] {
            MirOp::Repeat { len, count } => {
                let body = ops.get(at + 1..at + 1 + len as usize).ok_or_else(|| {
                    Error(format!("math[{at}]: repeat body runs past the program"))
                })?;
                for iteration in 0..u32::from(count) {
                    for op in body {
                        encode_math_op(op, iteration * 2, out)?;
                    }
                }
                at += 1 + len as usize;
                continue;
            }
            op => encode_math_op(&op, 0, out)?,
        }
        at += 1;
    }
    Ok(())
}

/// One math op at `row_delta` (a repeat's iteration offset, two address units per
/// iteration).
fn encode_math_op(op: &MirOp, row_delta: u32, out: &mut Vec<Instruction>) -> Result<(), Error> {
    fn bad(e: impl std::fmt::Display, op: &MirOp) -> Error {
        Error(format!("{op:?}: {e}"))
    }
    let bad_isa = |e: tt_isa::isa::EncodeError| bad(e, op);
    match *op {
        MirOp::Sfpload { lreg, row, int } => {
            let i = encode::sfpload(
                u32::from(lreg),
                if int {
                    tt_isa::sfpu::mod0_fmt::INT32
                } else {
                    tt_isa::sfpu::mod0_fmt::FP32
                },
                0,
                row + row_delta,
            )
            .map_err(bad_isa)?;
            out.push(i);
        }
        MirOp::Sfpstore { lreg, row, int } => {
            let i = encode::sfpstore(
                u32::from(lreg),
                if int {
                    tt_isa::sfpu::mod0_fmt::INT32
                } else {
                    tt_isa::sfpu::mod0_fmt::FP32
                },
                0,
                row + row_delta,
            )
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
        MirOp::Sfp { .. } => return Err(Error(format!("{op:?} has no encoding yet"))),
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
        other => return Err(Error(format!("{other:?} is not a math instruction"))),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The pack core (T2)
// ---------------------------------------------------------------------------

fn encode_pack(ops: &[MirOp], out: &mut Vec<Instruction>) -> Result<(), Error> {
    let mut started = false;
    for op in ops {
        match *op {
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
            MirOp::Pack { dst, rows, last } => {
                if !started {
                    thread_config(out);
                    started = true;
                }
                if rows == 0 {
                    return Err(Error("a pack of zero rows".into()));
                }
                let mut words = ConfigWords::new();
                pack_config(&mut words, dst);
                out.extend(config_program(&words));
                out.push(stallwait(block::PACKER | block::CONFIG, cond::CONFIG_BUSY)?);
                pack_rows(out, rows, last)?;
            }
            other => return Err(Error(format!("the pack program cannot encode {other:?}"))),
        }
    }
    Ok(())
}

/// The packer's `Config` for one unit: read `Dst` as 32-bit, write FP32 to
/// `dst`. The output address is in 16-byte units and the generator always adds
/// the one 16-byte tile header unit it never writes; `dst/16 - 1` compensates.
fn pack_config(words: &mut ConfigWords, dst: u32) {
    words
        .set(thcon::THCON_SEC0_REG1_L1_Dest_addr, dst / 16 - 1)
        .unwrap()
        .set(
            thcon::THCON_SEC0_REG1_In_data_format,
            L1Format::Fp32.code().unwrap(),
        )
        .unwrap()
        .set(
            thcon::THCON_SEC0_REG1_Out_data_format,
            L1Format::Fp32.code().unwrap(),
        )
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
        .set(pack0::PCK0_ADDR_CTRL_XY_REG_0_Ystride, DST_ROW_BYTES)
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_ZW_REG_0_Zstride, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_32b_data, 1)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_unsigned, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_int8, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Round_10b_mant, 0)
        .unwrap()
        .set(pack0::PACK_COUNTERS_SEC0_pack_reads_per_xy_plane, 4)
        .unwrap()
        .set(global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset, 0)
        .unwrap()
        .set(pack0::PCK_EDGE_MODE_mode, 0)
        .unwrap()
        .set(pack0::PCK_EDGE_OFFSET_SEC0_mask, 0xFFFF)
        .unwrap()
        .set(alu::ALU_ACC_CTRL_Fp32_enabled, 1)
        .unwrap();
}

/// Pack `rows` rows of FP32 `Dst`, from row 0, to the run `pack_config` set up.
fn pack_rows(out: &mut Vec<Instruction>, rows: u32, last: bool) -> Result<(), Error> {
    out.push(thread_entry(thread::ADDR_MOD_PACK_SEC0_YsrcIncr, 0));
    out.push(thread_entry(thread::ADDR_MOD_PACK_SEC1_YsrcIncr, 4));
    out.push(
        encode::Setadcxx::ZERO
            .pk(1)
            .x0_val(0)
            .x1_val(15)
            .encode()
            .map_err(|e| Error(format!("SETADCXX (pack): {e}")))?,
    );
    out.push(
        encode::Setadcxy::ZERO
            .pk(1)
            .y0(1)
            .y0_val(0)
            .encode()
            .map_err(|e| Error(format!("SETADCXY (pack): {e}")))?,
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
                .map_err(|e| Error(format!("PACR: {e}")))?,
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
