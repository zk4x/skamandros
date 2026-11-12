// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Unified program to machine IR: the first lowering step.
//!
//! `load` configures and fills the unpacker on T0, the register ops become
//! load/op/store bodies on T1, `store_to` configures and drains the packer on
//! T2. `copy_to` crosses the L1 boundary and is emitted by [`crate::mover`], not
//! here. The three cores run as a ring paced by three semaphores: the packer
//! hands a `Dst` slot back, the unpacker publishes each loaded unit, and the
//! math core publishes each stored one. One slot is in flight at a time, so a
//! program that loads a second unit before storing the first is refused rather
//! than left to wait for a slot that never comes.
//!
//! Addresses come from the L1 plan, `Dst` slots from the allocation; the caller
//! named none of them.

use crate::alloc::Alloc;
use crate::ir::{BinOp, Dim, Error, Kernel, Op, RegShape, Types, UnOp, unit_of};
use crate::mir::{CmpOp, IOp, Machine, MirOp, SfpOp};
use crate::planner::Plan;
use crate::riscv::reg;
use tt_isa::cfg::generated::thread;

/// `Dst` rows one 32x32 F32 tile occupies (`tt_isa`'s SFPU row is 16 datums).
pub const DST_ROWS: u32 = 64;
/// Datums in one 32x32 tile.
pub const TILE_DATUMS: u32 = 1024;
/// Semaphore the unpacker publishes a loaded unit on.
pub const SEM_UNPACKED: u8 = 0;
/// Semaphore the math core publishes a stored unit on.
pub const SEM_READY: u8 = 1;
/// Semaphore the packer hands a `Dst` slot back on. It starts at 1: the first
/// load has one slot to fill.
pub const SEM_SLOT: u8 = 2;

/// The RISC-V register a loop counts in, by nesting depth. `encode` uses the
/// same two for a constant bound, so a dynamic bound leaves its count there.
const COUNTERS: [u8; 2] = [reg::T4, reg::T5];

/// Registers a role owns for evaluating a bound expression. They are disjoint
/// from `a0` (the launch arguments), `a6` (the CB interface), the loop counters
/// and the sync scratch `encode` uses.
const SCRATCH: [u8; 6] = [reg::A1, reg::A2, reg::A3, reg::A4, reg::A5, reg::T3];

/// Lower `k` to the three compute programs.
pub fn lower(k: &Kernel, types: &Types, plan: &Plan, alloc: &Alloc) -> Result<Machine, Error> {
    // The `Dst` bank holds one datum width for the whole kernel: tt-metal's
    // `DST_ACCUM_MODE` (`fp32_dest_acc_en`, `tt_metal/jit_build/genfiles.cpp`).
    // A value that needs more than 16 bits forces 32-bit mode; otherwise every
    // value in `Dst` is bf16 or a `bool` lane and the bank is 16-bit, which is
    // the native path: the SFPU widens on load and truncates on store, so bf16
    // arithmetic needs no rounding glue and a unit costs half the rows.
    //
    // `bool` does not force 32-bit mode: its lane is the sign-magnitude `-1`,
    // and the bf16 store keeps exactly that pattern (`0xBF80`), so a comparison
    // in a bf16 kernel round-trips.
    let wide = (0..types.values.len()).any(|v| {
        alloc.place[v].is_some()
            && matches!(
                types.values[v].dtype,
                crate::ir::DType::F32 | crate::ir::DType::I32
            )
    });
    // The dtype a value occupies `Dst` with. 16-bit mode stores every value as
    // a bf16 datum, the `bool` lane's `-1` pattern included; 32-bit mode holds
    // bf16 as the f32 it widens to, so the packer is the only thing that rounds
    // it, at the bf16 buffer it lands in.
    let storage = |dtype: crate::ir::DType| {
        if !wide {
            crate::ir::DType::Bf16
        } else if dtype == crate::ir::DType::Bf16 {
            crate::ir::DType::F32
        } else {
            dtype
        }
    };
    let mut m = Machine {
        dst_is_16_bit: !wide,
        ..Machine::default()
    };
    let mut def: Vec<Option<usize>> = vec![None; types.values.len()];
    for (at, op) in k.ops.iter().enumerate() {
        if let Some(value) = op.defined() {
            def[value as usize] = Some(at);
        }
    }
    // The launch argument index of every parameter, in program order: a scalar
    // parameter's value is loaded from its pair in the arguments array.
    let mut param: Vec<Option<u32>> = vec![None; types.values.len()];
    let mut next = 0u32;
    for op in &k.ops {
        if let Op::Param { value, .. } = *op {
            param[value as usize] = Some(next);
            next += 1;
        }
    }
    // The induction value of every loop being lowered, innermost last: a bound
    // may name an outer loop's index, whose value is its counter register.
    let mut loops: Vec<crate::ir::ValueId> = Vec::new();
    // A bitcast is the same bits under another name, so it has no program of its
    // own and no `Dst` slot: every reader of the bitcast reads its source.
    let mut alias: Vec<Option<crate::ir::ValueId>> = vec![None; types.values.len()];
    for op in &k.ops {
        if let Op::Bitcast { value, src, .. } = *op {
            alias[value as usize] = Some(resolve(&alias, src));
        }
    }
    // Who reads each value, and where it last does: a `matmul` reads its
    // operands' CB pages straight into `SrcA`/`SrcB`, so a page stays reserved
    // until the last of its readers, and only then is popped.
    let mut last_use: Vec<Option<usize>> = vec![None; types.values.len()];
    let mut matmul_use: Vec<bool> = vec![false; types.values.len()];
    let mut non_matmul_use: Vec<bool> = vec![false; types.values.len()];
    for (at, op) in k.ops.iter().enumerate() {
        for v in op.used() {
            last_use[v as usize] = Some(at);
            match *op {
                Op::Matmul { a, b, .. } if v == a || v == b => matmul_use[v as usize] = true,
                _ => non_matmul_use[v as usize] = true,
            }
        }
    }
    // The page of a load that a `matmul` also reads is released at the last
    // reader, not at the load: the ring's read pointer must not move while the
    // Matrix Unit's unpack still points at it.
    let mut pop_here: Vec<Vec<u32>> = vec![Vec::new(); k.ops.len()];
    for (at, op) in k.ops.iter().enumerate() {
        match *op {
            Op::Const { .. }
            | Op::Param { .. }
            | Op::Buffer { .. }
            | Op::Acc { .. }
            | Op::Grid { .. }
            | Op::CopyTo { .. } => {}
            // A scalar is index arithmetic: it names a `Dim`, and the mover
            // evaluates it in RISC-V. The datapath has no scalar values, so
            // nothing here has to produce one.
            Op::GridCoord { .. } => {}
            Op::Cast { value, .. }
            | Op::Bitcast { value, .. }
            | Op::Unary { value, .. }
            | Op::Binary { value, .. }
            | Op::Select { value, .. }
                if types.of(value).shape == RegShape::Scalar => {}
            Op::Loop { value, bound } => {
                let depth = loops.len();
                let counter = *COUNTERS.get(depth).ok_or(Error::TT_TODO_LOOP_DEPTH)?;
                match bound_const(k, bound) {
                    Some(count) => {
                        m.t0.push(MirOp::Loop { count });
                        m.t1.push(MirOp::Loop { count });
                        m.t2.push(MirOp::Loop { count });
                    }
                    None => {
                        // A symbolic bound is RISC-V arithmetic in the role,
                        // like the C loop of a tt-metal kernel; the count ends
                        // in the counter register and the branch reads it.
                        for role in [&mut m.t0, &mut m.t1, &mut m.t2] {
                            let mut scratch = SCRATCH.iter().copied();
                            scalar(k, types, role, &param, &loops, bound, counter, &mut scratch)?;
                            role.push(MirOp::LoopReg { reg: counter });
                        }
                    }
                }
                loops.push(value);
            }
            Op::EndLoop => {
                loops.pop();
                m.t0.push(MirOp::EndLoop);
                m.t1.push(MirOp::EndLoop);
                m.t2.push(MirOp::EndLoop);
            }
            Op::Load {
                value,
                buffer,
                offset,
            } => {
                let info = types.of(buffer);
                let Op::Buffer { shape, dtype, .. } = k.ops[def[buffer as usize].expect("defined")]
                else {
                    let _ = buffer;
                    return Err(Error::TT_ERROR_NOT_A_BUFFER);
                };
                let (unit_shape, _, _) = unit_of(shape, dtype)?;
                let rows = rows_of(unit_shape, at)?;
                let cb = plan.cb[buffer as usize].ok_or_else(|| {
                    let _ = buffer;
                    Error::TT_ERROR_BUFFER_NOT_PLACED
                })?;
                let row = dst_row(alloc, resolve(&alias, value));
                // A value only a `matmul` reads never enters `Dst`: the unpack
                // goes straight into `SrcA`/`SrcB` at the matmul, and this load
                // only reserves the page.
                let lands_in_dst = non_matmul_use[value as usize] || !matmul_use[value as usize];
                if lands_in_dst {
                    // A bf16 page can only be unpacked into a bf16 `Dst`
                    // (`BF16 -> BF16` is the unpacker's only Dst-path conversion
                    // for it), so a bf16 buffer only enters a 16-bit kernel.
                    // Reaching one from a 32-bit kernel needs SrcA/SrcB and the
                    // widening `BF16 -> TF32` path.
                    if wide && dtype == crate::ir::DType::Bf16 {
                        let _ = (at, buffer);
                        return Err(Error::TT_ERROR_BF16_WIDEN);
                    }
                    // The unpacker writes `Dst`: it waits for a slot the packer
                    // has given back, so the first unit's result cannot be
                    // overwritten before it has been packed. The page itself
                    // comes from the ring: wait for the producer's push, unpack,
                    // then release the page.
                    m.t0.push(MirOp::SemWait {
                        sem: SEM_SLOT,
                        before: 0,
                    });
                    m.t0.push(MirOp::CbWait { cb });
                    m.t0.push(MirOp::Setdst { row });
                    m.t0.push(MirOp::Setadczw { unit: 0, face: 0 });
                    m.t0.push(MirOp::Setadcx {
                        unit: 0,
                        first: 0,
                        count: unit_shape.datums(),
                    });
                    m.t0.push(MirOp::Unpacr {
                        cb,
                        unit: 0,
                        count: unit_shape.datums(),
                        dtype,
                        to_src: false,
                        flip: false,
                    });
                    m.t0.push(MirOp::SemPost { sem: SEM_UNPACKED });
                    m.t1.push(MirOp::SemWait {
                        sem: SEM_UNPACKED,
                        before: 0,
                    });
                } else {
                    // The Matrix Unit's operand: the page is reserved here and
                    // unpacked at the matmul; the bank handover gates the rest.
                    m.t0.push(MirOp::CbWait { cb });
                }
                // A page the Matrix Unit still reads is released after that
                // read, not here.
                if matmul_use[value as usize] {
                    let last = last_use[value as usize].expect("a read value has a last use");
                    pop_here[last].push(cb);
                } else {
                    m.t0.push(MirOp::CbPop { cb });
                }
                let _ = info;
                let _ = rows;
                let _ = offset;
            }
            Op::StoreTo {
                value,
                buffer,
                offset,
            } => {
                // An accumulator is written in place in `Dst`, not packed: the
                // only form the verifier admits is a fill from an f32 constant,
                // and the SFPU writes it row by row.
                if types.of(buffer).kind == crate::ir::ValueKind::Acc {
                    let bits = match k.ops[def[value as usize].expect("defined")] {
                        Op::Const { bits, .. } => bits,
                        _ => {
                            let _ = value;
                            return Err(Error::TT_TODO_ACC_STORE);
                        }
                    };
                    let acc_dtype = types.of(buffer).dtype;
                    let (unit_shape, _, _) = unit_of([32, 32], acc_dtype)?;
                    let rows = rows_of(unit_shape, at)?;
                    let base = dst_row(alloc, buffer);
                    m.t1.push(MirOp::LaneConst { lreg: 0, bits });
                    // An SFPU store covers a row pair (`SFPSTORE` refuses an odd
                    // row and writes two), the same step the datapath uses.
                    for row in (0..rows).step_by(2) {
                        m.t1.push(MirOp::Sfpstore {
                            lreg: 0,
                            row: base + row,
                            dtype: acc_dtype,
                        });
                    }
                    let _ = (offset, value);
                    // An accumulator store never reads a CB page, but a page
                    // this instruction was the last reader of is released here
                    // exactly like the packed path's.
                    for cb in std::mem::take(&mut pop_here[at]) {
                        m.t0.push(MirOp::CbPop { cb });
                    }
                    continue;
                }
                let Op::Buffer { shape, dtype, .. } = k.ops[def[buffer as usize].expect("defined")]
                else {
                    let _ = buffer;
                    return Err(Error::TT_ERROR_NOT_A_BUFFER);
                };
                let (unit_shape, _, _) = unit_of(shape, dtype)?;
                let rows = rows_of(unit_shape, at)?;
                let cb = plan.cb[buffer as usize].ok_or_else(|| {
                    let _ = buffer;
                    Error::TT_ERROR_BUFFER_NOT_PLACED
                })?;
                m.t1.push(MirOp::SemPost { sem: SEM_READY });
                // The packer claims the unit, packs it, and only then releases
                // the semaphore — so the math core cannot overwrite `Dst` until
                // the pack has actually read it (`_llk_pack_dest_section_done_`).
                m.t2.push(MirOp::SemWaitData { sem: SEM_READY });
                // The page comes from the ring: reserve a free one, pack into
                // it, then publish it to the writer.
                m.t2.push(MirOp::CbReserve { cb });
                m.t2.push(MirOp::Pack {
                    cb,
                    rows,
                    row: dst_row(alloc, resolve(&alias, value)),
                    last: true,
                    dtype,
                });
                m.t2.push(MirOp::SemGet { sem: SEM_READY });
                m.t2.push(MirOp::CbPush { cb });
                // The pack has retired, so the `Dst` slot is free again.
                m.t2.push(MirOp::SemPost { sem: SEM_SLOT });
                let _ = offset;
            }
            Op::Cast { value, src, dtype } => {
                let from = types.of(src).dtype;
                let shape = types.of(src).shape;
                let rows = rows_of(shape, at)?;
                let s = dst_row(alloc, resolve(&alias, src));
                let out = dst_row(alloc, resolve(&alias, value));
                let (la, ld) = alloc.lreg[resolve(&alias, value) as usize];
                // The conversion happens in the lane. 16-bit mode stores every
                // value as a bf16 datum, so a `bool` lane *is* `-1.0`/`0.0` there
                // and an `f32` one is already bf16 precision; 32-bit mode holds
                // bf16 as the f32 it widens to. Either way the recipe is the
                // 32-bit lane's, so `cast_lanes` never sees `Bf16`.
                let lane = |dtype: crate::ir::DType| {
                    if dtype == crate::ir::DType::Bf16 {
                        crate::ir::DType::F32
                    } else {
                        dtype
                    }
                };
                for row in (0..rows).step_by(2) {
                    m.t1.push(MirOp::Sfpload {
                        lreg: la,
                        row: s + row,
                        dtype: storage(from),
                    });
                    m.t1.push(MirOp::Cast {
                        from: lane(from),
                        to: lane(dtype),
                        a: la,
                        d: ld,
                    });
                    m.t1.push(MirOp::Sfpstore {
                        lreg: ld,
                        row: out + row,
                        dtype: storage(dtype),
                    });
                }
            }
            // A bitcast is resolved, not lowered: every reader reads the source.
            Op::Bitcast { .. } => {}
            Op::Unary { value, op, a } => {
                let shape = types.of(a).shape;
                let rows = rows_of(shape, at)?;
                let src = dst_row(alloc, resolve(&alias, a));
                let out = dst_row(alloc, resolve(&alias, value));
                let (la, ld) = alloc.lreg[resolve(&alias, value) as usize];
                let sfp = unary(op, at)?;
                let dtype = storage(types.of(a).dtype);
                for row in (0..rows).step_by(2) {
                    m.t1.push(MirOp::Sfpload {
                        lreg: la,
                        row: src + row,
                        dtype,
                    });
                    m.t1.push(MirOp::Sfp {
                        op: sfp,
                        a: la,
                        b: None,
                        d: ld,
                    });
                    m.t1.push(MirOp::Sfpstore {
                        lreg: ld,
                        row: out + row,
                        dtype,
                    });
                }
            }
            Op::Binary { value, op, a, b } => {
                let left = *types.of(a);
                let right = *types.of(b);
                if left.shape != right.shape {
                    let _ = (left.shape, right.shape);
                    return Err(Error::TT_TODO_BROADCAST);
                }
                let dtype = match left.dtype {
                    // bf16 runs the same lane ops as f32: in 16-bit mode the
                    // SFPU widens on load and the store truncates, so each op
                    // lands at bf16 precision; in 32-bit mode the value is the
                    // f32 it widens to and the packer rounds at the buffer.
                    crate::ir::DType::F32 | crate::ir::DType::Bf16 => storage(left.dtype),
                    crate::ir::DType::I32
                        if matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor) =>
                    {
                        crate::ir::DType::I32
                    }
                    // A bool lane is `-1`/`0`, so the bitwise and/or *is* the
                    // logical one (this is what zyx's `tt_sfp_integer` does for
                    // `BOp::And`/`Or` too).
                    crate::ir::DType::Bool if matches!(op, BinOp::And | BinOp::Or) => {
                        crate::ir::DType::Bool
                    }
                    other => {
                        let _ = (op, other);
                        return Err(Error::TT_TODO_BINARY_DTYPE);
                    }
                };
                let rows = rows_of(left.shape, at)?;
                let sa = dst_row(alloc, resolve(&alias, a));
                let sb = dst_row(alloc, resolve(&alias, b));
                let out = dst_row(alloc, resolve(&alias, value));
                let (la, ld) = alloc.lreg[resolve(&alias, value) as usize];
                let lb = 2u8;
                // A comparison writes `-1`/`0` lanes, so its result is stored in
                // the integer format; everything else keeps the operands'.
                let out_dtype = if matches!(op, BinOp::Cmplt | BinOp::Cmpgt | BinOp::Cmpge) {
                    storage(crate::ir::DType::Bool)
                } else {
                    dtype
                };
                for row in (0..rows).step_by(2) {
                    m.t1.push(MirOp::Sfpload {
                        lreg: la,
                        row: sa + row,
                        dtype,
                    });
                    m.t1.push(MirOp::Sfpload {
                        lreg: lb,
                        row: sb + row,
                        dtype,
                    });
                    match op {
                        BinOp::Cmplt | BinOp::Cmpgt | BinOp::Cmpge => {
                            let cmp = match op {
                                BinOp::Cmplt => CmpOp::Lt,
                                BinOp::Cmpgt => CmpOp::Gt,
                                _ => CmpOp::Ge,
                            };
                            m.t1.push(MirOp::Compare {
                                op: cmp,
                                a: la,
                                b: lb,
                                d: ld,
                            });
                        }
                        BinOp::Max => m.t1.push(MirOp::Max {
                            a: la,
                            b: lb,
                            d: ld,
                        }),
                        _ => m.t1.push(MirOp::Sfp {
                            op: binary(op, at)?,
                            a: la,
                            b: Some(lb),
                            d: ld,
                        }),
                    }
                    m.t1.push(MirOp::Sfpstore {
                        lreg: ld,
                        row: out + row,
                        dtype: out_dtype,
                    });
                }
            }
            Op::Select {
                value,
                cond,
                if_true,
                if_false,
            } => {
                let shape = types.of(if_true).shape;
                let rows = rows_of(shape, at)?;
                let sc = dst_row(alloc, resolve(&alias, cond));
                let st = dst_row(alloc, resolve(&alias, if_true));
                let sf = dst_row(alloc, resolve(&alias, if_false));
                let out = dst_row(alloc, resolve(&alias, value));
                let ld = alloc.lreg[resolve(&alias, value) as usize].1;
                let lc = 3u8;
                let lt = 0u8;
                let lf = 2u8;
                if types.of(cond).dtype != crate::ir::DType::Bool {
                    let _ = at;
                    return Err(Error::TT_ERROR_CONDITION_NOT_BOOL);
                }
                for row in (0..rows).step_by(2) {
                    m.t1.push(MirOp::Sfpload {
                        lreg: lc,
                        row: sc + row,
                        dtype: storage(crate::ir::DType::Bool),
                    });
                    m.t1.push(MirOp::Sfpload {
                        lreg: lt,
                        row: st + row,
                        dtype: storage(types.of(if_true).dtype),
                    });
                    m.t1.push(MirOp::Sfpload {
                        lreg: lf,
                        row: sf + row,
                        dtype: storage(types.of(if_false).dtype),
                    });
                    m.t1.push(MirOp::Select {
                        cond: lc,
                        t: lt,
                        f: lf,
                        d: ld,
                    });
                    m.t1.push(MirOp::Sfpstore {
                        lreg: ld,
                        row: out + row,
                        dtype: storage(types.of(value).dtype),
                    });
                }
            }
            Op::Matmul { acc, a, b } => {
                let dt_a = types.of(a).dtype;
                let dt_b = types.of(b).dtype;
                let dtype = types.of(acc).dtype;
                // The accumulator is the `Dst` width; a `bf16` accumulator in a
                // module whose bank is 32-bit would accumulate in FP32 instead.
                if dtype == crate::ir::DType::Bf16 && wide {
                    let _ = at;
                    return Err(Error::TT_ERROR_MATMUL_ACCUM_WIDE);
                }
                let cb_a = matmul_page(k, plan, &def, &alias, a, at)?;
                let cb_b = matmul_page(k, plan, &def, &alias, b, at)?;
                let base = dst_row(alloc, acc);
                // The Matrix Unit's address modifiers, fidelity phase and row
                // counters are this thread's `ThreadConfig` and RWC state. Every
                // word goes out as a whole `SETC16`, so an entry carrying several
                // fields of one word is composed here.
                for entry in [
                    vec![(thread::ADDR_MOD_AB_SEC0_SrcAIncr, 0u16)],
                    vec![(thread::ADDR_MOD_DST_SEC0_DestIncr, 0)],
                    vec![(thread::ADDR_MOD_AB_SEC1_SrcAIncr, 0)],
                    vec![(thread::ADDR_MOD_DST_SEC1_FidelityIncr, 1)],
                    // End of a group's first `SrcB` half: `SrcB` on eight rows,
                    // phase back to 0.
                    vec![
                        (thread::ADDR_MOD_AB_SEC2_SrcBIncr, 8),
                        (thread::ADDR_MOD_AB_SEC2_SrcBClear, 0),
                    ],
                    vec![
                        (thread::ADDR_MOD_DST_SEC2_DestIncr, 0),
                        (thread::ADDR_MOD_DST_SEC2_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC2_FidelityClear, 1),
                    ],
                    // End of a group that is not the `fi`'s last: phase and
                    // `SrcB` back to 0.
                    vec![
                        (thread::ADDR_MOD_AB_SEC3_SrcBIncr, 0),
                        (thread::ADDR_MOD_AB_SEC3_SrcBClear, 1),
                    ],
                    vec![
                        (thread::ADDR_MOD_DST_SEC3_DestIncr, 0),
                        (thread::ADDR_MOD_DST_SEC3_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC3_FidelityClear, 1),
                    ],
                    // End of an `fi`: as 3, and the `Dst` base on 32 rows for
                    // the next `fi`.
                    vec![
                        (thread::ADDR_MOD_AB_SEC4_SrcBIncr, 0),
                        (thread::ADDR_MOD_AB_SEC4_SrcBClear, 1),
                    ],
                    vec![
                        (thread::ADDR_MOD_DST_SEC4_DestIncr, 32),
                        (thread::ADDR_MOD_DST_SEC4_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC4_FidelityClear, 1),
                    ],
                    vec![(thread::FIDELITY_BASE_Phase, 0)],
                ] {
                    m.t1.push(thread_word(&entry));
                }
                // Src rows and the `Dst` counter start at 0.
                m.t1.push(MirOp::Setrwc {
                    src_a: 0,
                    src_b: 0,
                    dst: 0,
                });
                // The accumulator's rows are added to, never cleared: an `acc`
                // holds whatever the program last put there, and zeroing it is a
                // `store_to %acc from %zero` the program writes itself.
                // One `(fi, k)` group: A's face `(fi, k)` into `SrcB`, B's two
                // faces `(k, 0)` and `(k, 1)` into `SrcA`. The unpacker and the
                // Matrix Unit alternate through the bank handover, so the two
                // roles interleave by ownership, not by instruction position.
                for fi in 0..2u32 {
                    for kk in 0..2u32 {
                        m.t0.push(MirOp::Setadczw {
                            unit: 1,
                            face: 2 * fi + kk,
                        });
                        m.t0.push(MirOp::Unpacr {
                            cb: cb_a,
                            unit: 1,
                            count: TILE_DATUMS,
                            dtype: dt_a,
                            to_src: true,
                            flip: true,
                        });
                        m.t0.push(MirOp::Setadczw {
                            unit: 0,
                            face: 2 * kk,
                        });
                        m.t0.push(MirOp::Unpacr {
                            cb: cb_b,
                            unit: 0,
                            count: TILE_DATUMS,
                            dtype: dt_b,
                            to_src: true,
                            flip: true,
                        });
                        mvmul_group(&mut m.t1, base);
                        m.t1.push(MirOp::Mvmul {
                            row: base + 8,
                            addr_mod: MATMUL_AM_GROUP,
                            flip_a: true,
                            flip_b: false,
                        });
                        m.t0.push(MirOp::Setadczw {
                            unit: 0,
                            face: 2 * kk + 1,
                        });
                        m.t0.push(MirOp::Unpacr {
                            cb: cb_b,
                            unit: 0,
                            count: TILE_DATUMS,
                            dtype: dt_b,
                            to_src: true,
                            flip: true,
                        });
                        mvmul_group(&mut m.t1, base + 16);
                        let end = if kk == 0 {
                            MATMUL_AM_GROUP
                        } else {
                            MATMUL_AM_FI
                        };
                        m.t1.push(MirOp::Mvmul {
                            row: base + 24,
                            addr_mod: end,
                            flip_a: true,
                            flip_b: true,
                        });
                    }
                }
                // The Matrix Unit advanced the thread's `Dst` counter as it
                // walked the accumulator, and an `SFPLOAD`/`SFPSTORE` address is
                // relative to it, so the counter goes back to 0. Its writes are
                // still in flight, so the accumulator is not readable (the pack
                // follows on T2) until the unit drains.
                m.t1.push(MirOp::WaitMatrix);
                m.t1.push(MirOp::Setrwc {
                    src_a: 0,
                    src_b: 0,
                    dst: 0,
                });
            }
        }
        // A page a `matmul` read is released once the last reader of the loaded
        // value has consumed it, which is after the matmul's own unpacks.
        for cb in std::mem::take(&mut pop_here[at]) {
            m.t0.push(MirOp::CbPop { cb });
        }
    }
    Ok(m)
}

/// `MVMUL`'s `ADDR_MOD` entries, one per column-group boundary: the next
/// fidelity phase, the end of a first `SrcB` half, the end of a group, and the
/// end of an `fi`.
const MATMUL_AM_PHASE: u32 = 1;
const MATMUL_AM_HALF: u32 = 2;
const MATMUL_AM_GROUP: u32 = 3;
const MATMUL_AM_FI: u32 = 4;

/// One `MVMUL` group into `Dst` rows `base..base + 16`: both `SrcB` halves and
/// every fidelity phase, all but the group's last multiply, which the caller
/// issues with its flips.
fn mvmul_group(role: &mut Vec<MirOp>, base: u32) {
    for half in 0..2u32 {
        let phases = if half == 0 { 4 } else { 3 };
        for phase in 0..phases {
            let addr_mod = if half == 0 && phase + 1 == 4 {
                MATMUL_AM_HALF
            } else {
                MATMUL_AM_PHASE
            };
            role.push(MirOp::Mvmul {
                row: base + 8 * half,
                addr_mod,
                flip_a: false,
                flip_b: false,
            });
        }
    }
}

/// One whole `SETC16` `ThreadConfig` word composed from `fields`.
fn thread_word(fields: &[(tt_isa::cfg::ThreadConfigField, u16)]) -> MirOp {
    let mut entry = tt_isa::backend::ThreadConfigEntry::zeroed(fields[0].0.addr32());
    for &(field, value) in fields {
        entry = entry
            .set(field, value)
            .expect("the matmul's thread entry is encodable");
    }
    MirOp::ThreadWord {
        addr: entry.addr32(),
        value: entry.value(),
    }
}

/// The CB a `matmul` operand reads: the value must be a `load` (possibly through
/// a bitcast), because the Matrix Unit's `SrcA`/`SrcB` are filled by the
/// unpacker from a page, not from `Dst`.
fn matmul_page(
    k: &Kernel,
    plan: &Plan,
    def: &[Option<usize>],
    alias: &[Option<crate::ir::ValueId>],
    value: crate::ir::ValueId,
    at: usize,
) -> Result<u32, Error> {
    let id = resolve(alias, value);
    let Some(op_at) = def[id as usize] else {
        let _ = (at, value);
        return Err(Error::TT_ERROR_MATMUL_OPERAND);
    };
    let Op::Load { buffer, .. } = k.ops[op_at] else {
        let _ = (at, value);
        return Err(Error::TT_ERROR_MATMUL_OPERAND);
    };
    plan.cb[buffer as usize].ok_or_else(|| {
        let _ = buffer;
        Error::TT_ERROR_BUFFER_NOT_PLACED
    })
}

/// The literal trip count of a bound, if it has one.
fn bound_const(k: &Kernel, bound: Dim) -> Option<u32> {
    match bound {
        Dim::Const(n) => Some(n),
        Dim::Value(v) => match k.const_i32(v) {
            Some(n) if n >= 0 => Some(n as u32),
            _ => None,
        },
    }
}

/// Emit `dim` into `dest` as role RISC-V integer ops: a literal, or an index
/// expression over literals, scalar parameters, grid coordinates, enclosing
/// loop indices and ordinary arithmetic.
#[allow(clippy::too_many_arguments)]
fn scalar(
    k: &Kernel,
    types: &Types,
    role: &mut Vec<MirOp>,
    param: &[Option<u32>],
    loops: &[crate::ir::ValueId],
    dim: Dim,
    dest: u8,
    scratch: &mut impl Iterator<Item = u8>,
) -> Result<(), Error> {
    match dim {
        Dim::Const(n) => {
            role.push(MirOp::Li {
                reg: dest,
                value: n,
            });
            Ok(())
        }
        Dim::Value(value) => eval(k, types, role, param, loops, value, dest, scratch),
    }
}

/// Emit the value `value` into `dest`; `scratch` supplies the temporaries a
/// right-hand operand needs.
#[allow(clippy::too_many_arguments)]
fn eval(
    k: &Kernel,
    types: &Types,
    role: &mut Vec<MirOp>,
    param: &[Option<u32>],
    loops: &[crate::ir::ValueId],
    value: crate::ir::ValueId,
    dest: u8,
    scratch: &mut impl Iterator<Item = u8>,
) -> Result<(), Error> {
    // An enclosing loop's induction value is in that loop's counter register.
    if let Some(depth) = loops.iter().rposition(|v| *v == value) {
        let a = *COUNTERS.get(depth).ok_or(Error::TT_TODO_LOOP_DEPTH)?;
        role.push(MirOp::IArith {
            op: IOp::Add,
            d: dest,
            a,
            b: reg::ZERO,
        });
        return Ok(());
    }
    let op = k
        .ops
        .iter()
        .find(|op| op.defined() == Some(value))
        .ok_or(Error::TT_TODO_LOOP_ADDRESS)?;
    match *op {
        Op::Const {
            dtype: crate::ir::DType::I32,
            bits,
            ..
        } => {
            role.push(MirOp::Li {
                reg: dest,
                value: bits,
            });
            Ok(())
        }
        Op::Param { value: p, .. } if types.of(p).shape == RegShape::Scalar => {
            let index = param[p as usize].ok_or(Error::TT_TODO_LOOP_ADDRESS)?;
            role.push(MirOp::LoadArg {
                reg: dest,
                arg: crate::mover::param_arg(index).0,
            });
            Ok(())
        }
        Op::GridCoord { axis, .. } => {
            role.push(MirOp::LoadCoord { reg: dest, axis });
            Ok(())
        }
        Op::Binary { op, a, b, .. } => {
            let op = int_op(op).ok_or(Error::TT_TODO_LOOP_ADDRESS)?;
            eval(k, types, role, param, loops, a, dest, scratch)?;
            let right = scratch.next().ok_or(Error::TT_TODO_LOOP_ADDRESS)?;
            eval(k, types, role, param, loops, b, right, scratch)?;
            role.push(MirOp::IArith {
                op,
                d: dest,
                a: dest,
                b: right,
            });
            Ok(())
        }
        _ => Err(Error::TT_TODO_LOOP_ADDRESS),
    }
}

/// The RISC-V form of an index arithmetic op; a comparison, `max` or `pow` is
/// not one.
fn int_op(op: BinOp) -> Option<IOp> {
    Some(match op {
        BinOp::Add => IOp::Add,
        BinOp::Sub => IOp::Sub,
        BinOp::Mul => IOp::Mul,
        BinOp::Div => IOp::Div,
        BinOp::Mod => IOp::Rem,
        BinOp::BitAnd => IOp::And,
        BinOp::BitOr => IOp::Or,
        BinOp::BitXor => IOp::Xor,
        BinOp::BitShiftLeft => IOp::Sll,
        BinOp::BitShiftRight => IOp::Sra,
        _ => return None,
    })
}

fn rows_of(shape: RegShape, at: usize) -> Result<u32, Error> {
    match shape {
        RegShape::Scalar => {
            let _ = at;
            Err(Error::TT_ERROR_SCALAR_NO_ROWS)
        }
        RegShape::Row | RegShape::Column => Ok(2),
        RegShape::Tile => Ok(DST_ROWS),
    }
}

fn dst_row(alloc: &Alloc, value: crate::ir::ValueId) -> u32 {
    match alloc.place_of(value) {
        crate::alloc::Place::Dst(slot) => u32::from(slot) * DST_ROWS,
    }
}

/// Follow a bitcast to the value that owns the `Dst` slot. The alias table is
/// fully resolved when it is built, so one lookup is enough.
fn resolve(alias: &[Option<crate::ir::ValueId>], value: crate::ir::ValueId) -> crate::ir::ValueId {
    alias[value as usize].unwrap_or(value)
}

fn unary(op: UnOp, at: usize) -> Result<SfpOp, Error> {
    match op {
        UnOp::Neg => Ok(SfpOp::Neg),
        UnOp::Abs => Ok(SfpOp::Abs),
        UnOp::BitNot => Ok(SfpOp::BitNot),
        other => {
            let _ = (at, other);
            Err(Error::TT_TODO_UNARY_SFPU)
        }
    }
}

fn binary(op: BinOp, at: usize) -> Result<SfpOp, Error> {
    match op {
        BinOp::Add => Ok(SfpOp::Add),
        BinOp::Sub => Ok(SfpOp::Sub),
        BinOp::Mul => Ok(SfpOp::Mul),
        BinOp::And | BinOp::BitAnd => Ok(SfpOp::BitAnd),
        BinOp::Or | BinOp::BitOr => Ok(SfpOp::BitOr),
        BinOp::BitXor => Ok(SfpOp::BitXor),
        other => {
            let _ = (at, other);
            Err(Error::TT_TODO_BINARY_SFPU)
        }
    }
}
