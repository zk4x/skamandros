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
use crate::ir::{
    BinOp, DType, Dim, Error, Kernel, Op, RegShape, Types, UnOp, access_shape, unit_of,
};
use crate::mir::{Bcast, CmpOp, ElwOp, IOp, Machine, MirOp, PackMask, SfpOp};
use crate::planner::Plan;
use crate::riscv::reg;
use tt_isa::cfg::generated::thread;

/// `Dst` rows one 32x32 F32 tile occupies (`tt_isa`'s SFPU row is 16 datums).
pub const DST_ROWS: u32 = 64;
/// Datums in one 32x32 tile.
pub const TILE_DATUMS: u32 = 1024;
/// Datums in one 16x16 face: one `Src` unpack's run.
pub const FACE_DATUMS: u32 = TILE_DATUMS / 4;
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
    // The edge mask a pack of each value carries follows the value's shape: a
    // reduce's result keeps only its folded positions, and every value with the
    // same narrow shape keeps the same part — a `1,32` row 0, a `32,1` column 0.
    let mut acc_mask: Vec<PackMask> = types
        .values
        .iter()
        .map(|v| match v.shape {
            RegShape::Row => PackMask::Col,
            RegShape::Column => PackMask::Row,
            _ => PackMask::None,
        })
        .collect();
    // Who reads each value, and where it last does: a `matmul` reads its
    // operands' CB pages straight into `SrcA`/`SrcB`, so a page stays reserved
    // until the last of its readers, and only then is popped.
    let mut last_use: Vec<Option<usize>> = vec![None; types.values.len()];
    let mut matmul_use: Vec<bool> = vec![false; types.values.len()];
    let mut non_matmul_use: Vec<bool> = vec![false; types.values.len()];
    // A value read only as a copy's index is the mover's: it is an address
    // expression, or a row of bases the mover reads one datum of per lane.
    let mut index_use: Vec<bool> = vec![false; types.values.len()];
    let mut other_use: Vec<bool> = vec![false; types.values.len()];
    for (at, op) in k.ops.iter().enumerate() {
        for v in op.used() {
            last_use[v as usize] = Some(at);
            // A copy's ends are its operands; its other reads are indices.
            if matches!(*op, Op::CopyTo { dst, src, .. } if v != dst && v != src) {
                index_use[v as usize] = true;
                continue;
            }
            other_use[v as usize] = true;
            if let Some((tile, narrow)) = crate::ir::broadcast_pair(types, op)
                && (v == tile || v == narrow)
            {
                // The FPU's broadcast reads both operands from `Src` pages.
                matmul_use[v as usize] = true;
                continue;
            }
            match *op {
                Op::Matmul { a, b, .. } if v == a || v == b => matmul_use[v as usize] = true,
                // A reduce's operands are Src operands too: the fold runs on
                // the Src path from their pages, exactly as `matmul`'s do.
                Op::Sum { x, scaler, .. } if v == x || v == scaler => {
                    matmul_use[v as usize] = true;
                }
                Op::Max { x, .. } if v == x => matmul_use[v as usize] = true,
                _ => non_matmul_use[v as usize] = true,
            }
        }
    }
    // The page of a load that a `matmul` also reads is released at the last
    // reader, not at the load: the ring's read pointer must not move while the
    // Matrix Unit's unpack still points at it.
    let mut pop_here: Vec<Vec<u32>> = vec![Vec::new(); k.ops.len()];
    for (at, op) in k.ops.iter().enumerate() {
        // A reduce's axis is its accumulator's shape (`ReduceDim::of`); the
        // verifier has already refused a `32,32` acc there, and a `matmul`
        // needs that same `32,32`.
        let reduce_dim = match *op {
            Op::Sum { acc, .. } | Op::Max { acc, .. } => {
                Some(crate::ir::ReduceDim::of(types.of(acc).shape).expect("a reduce acc folds"))
            }
            _ => None,
        };
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
                shape: access,
            } => {
                // A scalar load is an index read: the mover evaluates it from
                // the page it copied, in RISC-V (see `mover::eval_value`).
                if types.of(value).shape == RegShape::Scalar {
                    continue;
                }
                // A vector load that nothing reads but a copy's index is the
                // mover's lane bases: it reads one datum per lane out of the
                // page it just copied, so no `Dst` slot ever holds it.
                if index_use[value as usize] && !other_use[value as usize] {
                    continue;
                }
                let info = types.of(buffer);
                let Op::Buffer { dtype, .. } = k.ops[def[buffer as usize].expect("defined")] else {
                    let _ = buffer;
                    return Err(Error::TT_ERROR_NOT_A_BUFFER);
                };
                let unit_shape = access_shape(access)?;
                let row = dst_row(alloc, resolve(&alias, value));
                // A value only an `Src` reader consumes never enters `Dst`: the
                // unpack goes straight into `SrcA`/`SrcB` at the op that reads
                // it, and this load only reserves the page.
                let lands_in_dst = non_matmul_use[value as usize] || !matmul_use[value as usize];
                // Only a tile (one whole page) is lowered into the datapath
                // today; a row, a column or a datum needs the narrowed unpack,
                // which is not written. A narrow access that only an `Src`
                // reader consumes needs no unpack at all: that op reads the
                // page's own layout.
                if unit_shape != RegShape::Tile && lands_in_dst {
                    let _ = (at, access);
                    return Err(Error::TT_TODO_ACCESS_SHAPE);
                }
                let rows = rows_of(unit_shape, at)?;
                let cb = plan.cb[buffer as usize].ok_or_else(|| {
                    let _ = buffer;
                    Error::TT_ERROR_BUFFER_NOT_PLACED
                })?;
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
                    m.t0.push(MirOp::UnpackConfig {
                        unit: 0,
                        to_src: false,
                        dtype,
                        haloize: false,
                    });
                    m.t0.push(MirOp::Unpacr {
                        cb,
                        unit: 0,
                        count: unit_shape.datums(),
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
                shape: access,
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
                    // The fill writes `Dst` rows. If those rows were packed
                    // since their last write, the packer may still be reading
                    // them, so the math thread waits for the section to drain
                    // first (`wait_math_semaphores`).
                    m.t1.push(MirOp::SemWaitFree { sem: SEM_READY });
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
                let Op::Buffer { dtype, .. } = k.ops[def[buffer as usize].expect("defined")] else {
                    let _ = buffer;
                    return Err(Error::TT_ERROR_NOT_A_BUFFER);
                };
                let unit_shape = access_shape(access)?;
                // Only a tile is lowered today; a narrower pack needs the
                // packer's `SETADCX` narrowing.
                if unit_shape != RegShape::Tile {
                    let _ = (at, access);
                    return Err(Error::TT_TODO_ACCESS_SHAPE);
                }
                let rows = rows_of(unit_shape, at)?;
                let cb = plan.cb[buffer as usize].ok_or_else(|| {
                    let _ = buffer;
                    Error::TT_ERROR_BUFFER_NOT_PLACED
                })?;
                // One `Dst` section is in flight at a time, as in tt_llk: the
                // math thread waits until the packer has released the previous
                // one before it offers this one. Without the wait two posts
                // coalesce in the one-deep pack hand-off and the packer waits
                // for a section that never comes.
                m.t1.push(MirOp::SemWaitFree { sem: SEM_READY });
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
                    // A pack of a reduced accumulator carries the reduce's edge
                    // mask; every other pack keeps the whole tile.
                    mask: acc_mask[resolve(&alias, value) as usize],
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
                let rows = sfp_rows(shape, at)?;
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
                for row in rows {
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
                let rows = sfp_rows(shape, at)?;
                let src = dst_row(alloc, resolve(&alias, a));
                let out = dst_row(alloc, resolve(&alias, value));
                let (la, ld) = alloc.lreg[resolve(&alias, value) as usize];
                let dtype = storage(types.of(a).dtype);
                // `exp2` is the one transcendental no IR op spells: its
                // exponent assembly is `SFPEXEXP`/`SFPSETEXP`, so it is a body
                // here rather than a line in `src/unfold.rs`.
                if op == UnOp::Exp2 {
                    exp2_body(&mut m.t1, src, out, &rows, la, dtype);
                    continue;
                }
                let sfp = unary(op, at)?;
                for row in rows {
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
                // A scalar constant operand is a lane immediate, not a tile:
                // every coefficient and mask `src/unfold.rs` writes is a
                // literal, and a lane load is two instructions where a filled
                // tile would be a `Dst` slot and a sweep. Any other scalar
                // (an index expression) still has no lane and is refused.
                let ca = scalar_const(k, &def, a);
                let cb = scalar_const(k, &def, b);
                if left.shape != right.shape
                    && !(left.shape == RegShape::Scalar && ca.is_some())
                    && !(right.shape == RegShape::Scalar && cb.is_some())
                {
                    // A broadcast is an FPU op, and the FPU's `Src` path takes
                    // `f32` and `bf16` only. Anything else (`i32`, `bool`) needs
                    // the SFPU route, which is not lowered: say so rather than
                    // report the FPU form as merely pending.
                    if !matches!(left.dtype, DType::F32 | DType::Bf16)
                        || !matches!(right.dtype, DType::F32 | DType::Bf16)
                    {
                        let _ = (left.dtype, right.dtype);
                        return Err(Error::TT_TODO_BROADCAST_SFPU);
                    }
                    // The FPU's `ELWADD`/`ELWSUB`/`ELWMUL` broadcast `SrcB`:
                    // one operand is the tile, the other its row, column or
                    // scalar. Both come from pages, so each is a `load`'s page
                    // or the compiler's staged one.
                    if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) {
                        let _ = (op, left.shape, right.shape);
                        return Err(Error::TT_TODO_BROADCAST);
                    }
                    let (tile, narrow) = match (left.shape, right.shape) {
                        (RegShape::Tile, _) => (a, b),
                        (_, RegShape::Tile) => (b, a),
                        _ => {
                            let _ = (left.shape, right.shape);
                            return Err(Error::TT_TODO_BROADCAST);
                        }
                    };
                    // The instruction is `SrcA op SrcB` and the tile is the
                    // one that can be `SrcA`, so a difference with the narrow
                    // value on the left has no lowering.
                    if op == BinOp::Sub && tile != a {
                        let _ = (a, b);
                        return Err(Error::TT_TODO_BROADCAST);
                    }
                    let bcast = match types.of(narrow).shape {
                        RegShape::Row => Bcast::Row,
                        RegShape::Column => Bcast::Col,
                        RegShape::Scalar => Bcast::Scalar,
                        RegShape::Tile => unreachable!("the tile is the other operand"),
                    };
                    fpu_broadcast(
                        &mut m, k, types, plan, alloc, &def, &alias, &acc_mask, value, op, tile,
                        narrow, bcast, at,
                    )?;
                    // A page whose last reader this broadcast was is released
                    // here; `continue` skips the common block below.
                    for cb in std::mem::take(&mut pop_here[at]) {
                        m.t0.push(MirOp::CbPop { cb });
                    }
                    continue;
                }
                let shape = if left.shape == RegShape::Scalar {
                    right.shape
                } else {
                    left.shape
                };
                let dtype = match left.dtype {
                    // bf16 runs the same lane ops as f32: in 16-bit mode the
                    // SFPU widens on load and the store truncates, so each op
                    // lands at bf16 precision; in 32-bit mode the value is the
                    // f32 it widens to and the packer rounds at the buffer.
                    crate::ir::DType::F32 | crate::ir::DType::Bf16 => storage(left.dtype),
                    // The bit and shift ops are raw 32-bit words, and the
                    // comparisons order sign-magnitude, which an `i32` lane is
                    // only where its datum is non-negative (TODO.md's recorded
                    // integer-comparison gap). The unfold's `!= 0` tests a
                    // masked 0/1, so that is exact.
                    crate::ir::DType::I32
                        if matches!(
                            op,
                            BinOp::Add
                                | BinOp::Sub
                                | BinOp::BitAnd
                                | BinOp::BitOr
                                | BinOp::BitXor
                                | BinOp::BitShiftLeft
                                | BinOp::BitShiftRight
                                | BinOp::Cmplt
                                | BinOp::Cmpgt
                                | BinOp::Cmpge
                                | BinOp::Eq
                                | BinOp::NotEq
                        ) =>
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
                let rows = sfp_rows(shape, at)?;
                let out = dst_row(alloc, resolve(&alias, value));
                let (la, ld) = alloc.lreg[resolve(&alias, value) as usize];
                let lb = 2u8;
                // A comparison writes `-1`/`0` lanes, so its result is stored in
                // the integer format; everything else keeps the operands'.
                let out_dtype = if matches!(
                    op,
                    BinOp::Cmplt | BinOp::Cmpgt | BinOp::Cmpge | BinOp::Eq | BinOp::NotEq
                ) {
                    storage(crate::ir::DType::Bool)
                } else {
                    dtype
                };
                // The constant lanes live for the whole sweep, so they are set
                // once, before the first row. The one exception is a right
                // shift's amount: its encoding negates the amount in its own
                // lane (there is no other way to ask `SFPSHFT` for the other
                // direction), so it is re-loaded per row.
                let reload_amount = matches!(op, BinOp::BitShiftRight);
                if let Some(bits) = ca {
                    m.t1.push(MirOp::LaneConst { lreg: la, bits });
                }
                if let Some(bits) = cb
                    && !reload_amount
                {
                    m.t1.push(MirOp::LaneConst { lreg: lb, bits });
                }
                for row in rows {
                    if ca.is_none() {
                        m.t1.push(MirOp::Sfpload {
                            lreg: la,
                            row: dst_row(alloc, resolve(&alias, a)) + row,
                            dtype,
                        });
                    }
                    if cb.is_none() {
                        m.t1.push(MirOp::Sfpload {
                            lreg: lb,
                            row: dst_row(alloc, resolve(&alias, b)) + row,
                            dtype,
                        });
                    } else if reload_amount {
                        m.t1.push(MirOp::LaneConst {
                            lreg: lb,
                            bits: cb.expect("a re-loaded amount is a constant"),
                        });
                    }
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
                        // `a == b` is `!(a > b || a < b)`, two of the ordered
                        // tests this compiler already has, combined as
                        // sign-magnitude `-1`/`0` lanes. This is zyx's
                        // `tt_sfp_compare` shape without its NaN and `-0`
                        // specials, which the ordered tests do not carry either.
                        BinOp::Eq | BinOp::NotEq => {
                            m.t1.push(MirOp::Compare {
                                op: CmpOp::Gt,
                                a: la,
                                b: lb,
                                d: 3,
                            });
                            m.t1.push(MirOp::Compare {
                                op: CmpOp::Lt,
                                a: la,
                                b: lb,
                                d: 4,
                            });
                            m.t1.push(MirOp::Sfp {
                                op: SfpOp::BitOr,
                                a: 3,
                                b: Some(4),
                                d: ld,
                            });
                            if matches!(op, BinOp::Eq) {
                                m.t1.push(MirOp::Sfp {
                                    op: SfpOp::BitNot,
                                    a: ld,
                                    b: None,
                                    d: ld,
                                });
                            }
                        }
                        BinOp::Max => m.t1.push(MirOp::Max {
                            a: la,
                            b: lb,
                            d: ld,
                        }),
                        _ => m.t1.push(MirOp::Sfp {
                            // An `i32` lane is a raw 32-bit word, so its add and
                            // subtract are the integer instruction rather than
                            // `SFPMAD`'s float ones.
                            op: if left.dtype == crate::ir::DType::I32 {
                                match op {
                                    BinOp::Add => SfpOp::IAdd,
                                    BinOp::Sub => SfpOp::ISub,
                                    other => binary(other, at)?,
                                }
                            } else {
                                binary(op, at)?
                            },
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
                let rows = sfp_rows(shape, at)?;
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
                for row in rows {
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
            // `sum row` is the Matrix Unit's `x @ scaler`: tt-metal's
            // `ReduceDim::ROW` with `PoolType::SUM` is a matmul against the
            // scaler's column of ones, not a pool instruction, so it shares
            // this sequence.
            Op::Matmul { .. } | Op::Sum { .. }
                if matches!(*op, Op::Matmul { .. })
                    || reduce_dim == Some(crate::ir::ReduceDim::Row) =>
            {
                let (acc, a, b) = match *op {
                    Op::Matmul { acc, a, b } => (acc, a, b),
                    Op::Sum { acc, x, scaler } => (acc, x, scaler),
                    _ => unreachable!("the arm pattern is matmul and sum row"),
                };
                let dt_a = types.of(a).dtype;
                let dt_b = types.of(b).dtype;
                let dtype = types.of(acc).dtype;
                // The accumulator is the `Dst` width; a `bf16` accumulator in a
                // module whose bank is 32-bit would accumulate in FP32 instead.
                if dtype == crate::ir::DType::Bf16 && wide {
                    let _ = at;
                    return Err(Error::TT_ERROR_MATMUL_ACCUM_WIDE);
                }
                let (cb_a, stage_a) =
                    fpu_page(k, plan, &def, &alias, a, at, Error::TT_ERROR_MATMUL_OPERAND)?;
                let (cb_b, stage_b) =
                    fpu_page(k, plan, &def, &alias, b, at, Error::TT_ERROR_MATMUL_OPERAND)?;
                // A computed operand has no page: the compiler packs it into its
                // own and the Matrix Unit reads that.
                if stage_a {
                    stage(&mut m, types, alloc, &alias, &acc_mask, a, cb_a);
                }
                if stage_b {
                    stage(&mut m, types, alloc, &alias, &acc_mask, b, cb_b);
                }
                let base = dst_row(alloc, acc);
                // The Matrix Unit writes the accumulator's rows. If they were
                // packed since their last write, the packer may still be
                // reading them; wait for the section to drain
                // (`_llk_math_wait_for_dest_available_`).
                m.t1.push(MirOp::SemWaitFree { sem: SEM_READY });
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
                    flip_src_a: false,
                    flip_src_b: false,
                    dst_cr: false,
                });
                // The accumulator's rows are added to, never cleared: an `acc`
                // holds whatever the program last put there, and zeroing it is a
                // `store_to %acc from %zero` the program writes itself.
                // One `(fi, k)` group: A's face `(fi, k)` into `SrcB`, B's two
                // faces `(k, 0)` and `(k, 1)` into `SrcA`. The unpacker and the
                // Matrix Unit alternate through the bank handover, so the two
                // roles interleave by ownership, not by instruction position.
                // A staged operand's page is published by the packer (T2) before
                // the unpacker reads it, and released after the last face.
                for (cb, staged) in [(cb_a, stage_a), (cb_b, stage_b)] {
                    if staged {
                        m.t0.push(MirOp::CbWait { cb });
                    }
                }
                for fi in 0..2u32 {
                    for kk in 0..2u32 {
                        m.t0.push(MirOp::Setadczw {
                            unit: 1,
                            face: 2 * fi + kk,
                        });
                        m.t0.push(MirOp::UnpackConfig {
                            unit: 1,
                            to_src: true,
                            dtype: dt_a,
                            haloize: false,
                        });
                        m.t0.push(MirOp::Unpacr {
                            cb: cb_a,
                            unit: 1,
                            count: FACE_DATUMS,
                            flip: true,
                        });
                        m.t0.push(MirOp::Setadczw {
                            unit: 0,
                            face: 2 * kk,
                        });
                        m.t0.push(MirOp::UnpackConfig {
                            unit: 0,
                            to_src: true,
                            dtype: dt_b,
                            haloize: false,
                        });
                        m.t0.push(MirOp::Unpacr {
                            cb: cb_b,
                            unit: 0,
                            count: FACE_DATUMS,
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
                            count: FACE_DATUMS,
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
                for (cb, staged) in [(cb_a, stage_a), (cb_b, stage_b)] {
                    if staged {
                        m.t0.push(MirOp::CbPop { cb });
                    }
                }
                m.t1.push(MirOp::Setrwc {
                    src_a: 0,
                    src_b: 0,
                    dst: 0,
                    flip_src_a: false,
                    flip_src_b: false,
                    dst_cr: false,
                });
            }
            // `sum col`/`max col`: `GAPOOL`/`GMPOOL` fold each 16x16 face over
            // its rows into the destination row, one value per column, and the
            // pack's edge mask drops everything below row 0 — tt-metal's
            // `_llk_math_reduce_` for `REDUCE_COL`.
            Op::Sum { .. } | Op::Max { .. } => {
                let (acc, x, scaler, max) = match *op {
                    Op::Sum { acc, x, scaler } => (acc, x, Some(scaler), false),
                    Op::Max { acc, x } => (acc, x, None, true),
                    _ => unreachable!("the arm pattern is the two reduce ops"),
                };
                let dim = reduce_dim.expect("a reduce's acc has a fold");
                if dim == crate::ir::ReduceDim::Scalar {
                    let _ = dim;
                    return Err(Error::TT_TODO_REDUCE);
                }
                // `max` needs a `SrcB` row of exactly 1.0 (`GMPOOL` scales by
                // the exponent). It names no scaler, so the compiler's own
                // page (`plan.ones`) is the operand.
                let dt_x = types.of(x).dtype;
                let dtype = types.of(acc).dtype;
                if dtype == crate::ir::DType::Bf16 && wide {
                    let _ = at;
                    return Err(Error::TT_ERROR_MATMUL_ACCUM_WIDE);
                }
                let (cb_x, stage_x) =
                    fpu_page(k, plan, &def, &alias, x, at, Error::TT_ERROR_REDUCE_OPERAND)?;
                let (cb_b, stage_b, dt_b) = match scaler {
                    Some(scaler) => {
                        let (cb, staged) = fpu_page(
                            k,
                            plan,
                            &def,
                            &alias,
                            scaler,
                            at,
                            Error::TT_ERROR_REDUCE_OPERAND,
                        )?;
                        (cb, staged, types.of(scaler).dtype)
                    }
                    None => {
                        let ones = plan.ones.ok_or_else(|| {
                            let _ = at;
                            Error::TT_TODO_REDUCE
                        })?;
                        if ones.dtype != dt_x {
                            let _ = (ones.dtype, dt_x);
                            return Err(Error::TT_ERROR_REDUCE_DTYPE);
                        }
                        (ones.cb, false, ones.dtype)
                    }
                };
                // The 1.0 page is pushed once at kernel start; a `max` waits
                // for it before the first face and never pops it, so every
                // `max` in the program re-reads the same page.
                if scaler.is_none() {
                    m.t0.push(MirOp::CbWait { cb: cb_b });
                }
                // A computed operand has no page: the compiler stages it, and
                // the unpacker waits for that page before the first face.
                if stage_x {
                    stage(&mut m, types, alloc, &alias, &acc_mask, x, cb_x);
                    m.t0.push(MirOp::CbWait { cb: cb_x });
                }
                if stage_b {
                    if let Some(scaler) = scaler {
                        stage(&mut m, types, alloc, &alias, &acc_mask, scaler, cb_b);
                    }
                    m.t0.push(MirOp::CbWait { cb: cb_b });
                }
                let base = dst_row(alloc, acc);
                // The accumulator's rows are written, so wait for the packer to
                // give the section back.
                m.t1.push(MirOp::SemWaitFree { sem: SEM_READY });
                // `reduce_configure_addrmod`: entry 0 clears the phase, 1 and 2
                // step `SrcB`/`Dst`, and 3 steps only the phase.
                for entry in [
                    // `ADDR_MOD_0`: no increments, and the phase is cleared.
                    vec![
                        (thread::ADDR_MOD_AB_SEC0_SrcAIncr, 0u16),
                        (thread::ADDR_MOD_AB_SEC0_SrcBIncr, 0),
                        (thread::ADDR_MOD_AB_SEC0_SrcBClear, 0),
                    ],
                    vec![
                        (thread::ADDR_MOD_DST_SEC0_DestIncr, 0),
                        (thread::ADDR_MOD_DST_SEC0_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC0_FidelityClear, 1),
                    ],
                    // `ADDR_MOD_1`: one `SrcB` row and one `Dst` row.
                    vec![(thread::ADDR_MOD_AB_SEC1_SrcBIncr, 1)],
                    vec![(thread::ADDR_MOD_DST_SEC1_DestIncr, 1)],
                    // `ADDR_MOD_2`: eight of each.
                    vec![(thread::ADDR_MOD_AB_SEC2_SrcBIncr, 8)],
                    vec![(thread::ADDR_MOD_DST_SEC2_DestIncr, 8)],
                    // `ADDR_MOD_3`: no increments, the next phase.
                    vec![(thread::ADDR_MOD_AB_SEC3_SrcBIncr, 0)],
                    vec![
                        (thread::ADDR_MOD_DST_SEC3_DestIncr, 0),
                        (thread::ADDR_MOD_DST_SEC3_FidelityIncr, 1),
                    ],
                    vec![(thread::FIDELITY_BASE_Phase, 0)],
                ] {
                    m.t1.push(thread_word(&entry));
                }
                m.t1.push(MirOp::Setrwc {
                    src_a: 0,
                    src_b: 0,
                    dst: 0,
                    flip_src_a: false,
                    flip_src_b: false,
                    dst_cr: false,
                });
                if dim == crate::ir::ReduceDim::Row {
                    // tt-metal's `_llk_math_reduce_` for `REDUCE_ROW`: each
                    // face pair folds into one row of the accumulator (the
                    // haloized unpack presents the face transposed, so the fold
                    // runs along the tile's original rows), and
                    // `reduce_row_perform_transpose` turns that row into the
                    // result's column 0. Only `max` reaches this arm: `sum row`
                    // is the Matrix Unit's matmul above.
                    m.t0.push(MirOp::UnpackConfig {
                        unit: 0,
                        to_src: true,
                        dtype: dt_x,
                        haloize: true,
                    });
                    m.t0.push(MirOp::UnpackConfig {
                        unit: 1,
                        to_src: true,
                        dtype: dt_b,
                        haloize: false,
                    });
                    for group in 0..2u32 {
                        let dst = base + 32 * group;
                        for half in 0..2u32 {
                            let face = 2 * group + half;
                            // One face into `SrcA` and one row of the scaler
                            // into `SrcB` (16 datums): tt-metal's AB-reduce
                            // unpack.
                            m.t0.push(MirOp::Setadczw { unit: 0, face });
                            m.t0.push(MirOp::Unpacr {
                                cb: cb_x,
                                unit: 0,
                                count: FACE_DATUMS,
                                flip: true,
                            });
                            m.t0.push(MirOp::Setadczw { unit: 1, face });
                            m.t0.push(MirOp::Unpacr {
                                cb: cb_b,
                                unit: 1,
                                count: 16,
                                flip: true,
                            });
                            m.t0.push(MirOp::SemPost { sem: SEM_UNPACKED });
                            m.t1.push(MirOp::SemWait {
                                sem: SEM_UNPACKED,
                                before: 0,
                            });
                            // The first face of a pair hands both banks back
                            // (`CLR_AB`); the second leaves them, because
                            // `TRNSPSRCB` reads the `SrcB` bank the fold just
                            // consumed (tt-metal's `CLR_AB` then `CLR_NONE`).
                            m.t1.push(MirOp::Gmpool {
                                flip_src_b: half == 0,
                                flip_src_a: half == 0,
                                addr_mod: 0,
                                arg_max: 0,
                                dst_row: dst,
                            });
                        }
                        // `reduce_row_perform_transpose<true>`: the fold's row
                        // goes to `SrcB` rows 16..31, is transposed there, and
                        // comes back as `Dst` column 0. A 32-bit `Dst` datum is
                        // moved as its two 16-bit halves — the only way a
                        // 16-bit `SrcB` carries all of it. The moves read the
                        // format tt-metal's transpose names explicitly
                        // (`DISABLE_IMPLIED_SRCA_FMT`, then `Float32`), because
                        // the implied `TF32` of the unpacked operand has no
                        // 16-bit half to move.
                        //
                        // The low halves are saved in `SrcA` first: writing the
                        // high halves zeroes them in `Dst`, exactly as
                        // tt-metal's `transpose_dest_32b` orders it.
                        m.t1.push(thread_word(&[(thread::DISABLE_IMPLIED_SRCA_FMT_Base, 1)]));
                        m.t1.push(MirOp::SrcAFormat {
                            dtype: crate::ir::DType::F32,
                        });
                        m.t1.push(MirOp::AluAccCtrl {
                            fp32: true,
                            zero_flag_src: true,
                        });
                        // The transposed low halves, parked in `SrcA` rows
                        // 0..15.
                        m.t1.push(MirOp::Movd2b {
                            use_dst32b_lo: true,
                            src_row: 16,
                            addr_mod: 0,
                            move4_rows: false,
                            dst_row: dst,
                        });
                        m.t1.push(MirOp::Trnspsrcb);
                        for chunk in 0..4u32 {
                            m.t1.push(MirOp::Movb2a {
                                src_a_row: 4 * chunk,
                                addr_mod: 0,
                                move4_rows: true,
                                src_b_row: 16 + 4 * chunk,
                            });
                        }
                        // The high halves, back into the `Dst` column.
                        m.t1.push(MirOp::Movd2b {
                            use_dst32b_lo: false,
                            src_row: 16,
                            addr_mod: 0,
                            move4_rows: false,
                            dst_row: dst,
                        });
                        m.t1.push(MirOp::Trnspsrcb);
                        m.t1.push(MirOp::Movd2b {
                            use_dst32b_lo: false,
                            src_row: 16,
                            addr_mod: 0,
                            move4_rows: false,
                            dst_row: dst,
                        });
                        for chunk in 0..4u32 {
                            m.t1.push(MirOp::Movb2d {
                                use_dst32b_lo: false,
                                src_row: 16 + 4 * chunk,
                                addr_mod: 0,
                                move4_rows: true,
                                dst_row: dst + 4 * chunk,
                            });
                        }
                        // The low halves, restored from the parked column.
                        for half in 0..2u32 {
                            m.t1.push(MirOp::Mova2d {
                                use_dst32b_lo: true,
                                src_row: 8 * half,
                                addr_mod: 0,
                                move8_rows: true,
                                dst_row: dst + 8 * half,
                            });
                        }
                        m.t1.push(MirOp::AluAccCtrl {
                            fp32: true,
                            zero_flag_src: false,
                        });
                        m.t1.push(thread_word(&[(thread::DISABLE_IMPLIED_SRCA_FMT_Base, 0)]));
                        // `SETRWC(CLR_AB, 0, 0, 0, 0, SET_BD)`: hand both banks
                        // back before the next pair unpacks.
                        m.t1.push(MirOp::Setrwc {
                            src_a: 0,
                            src_b: 0,
                            dst: 0,
                            flip_src_a: true,
                            flip_src_b: true,
                            dst_cr: false,
                        });
                    }
                    m.t1.push(MirOp::WaitMatrix);
                    // The pack's edge mask keeps column 0 of every row, and the
                    // fold's column lands at rows 0..15 and 32..47; the rows
                    // between (16..31 and 48..63) would pack stale datums, so
                    // they are written as the zeros the result defines.
                    m.t1.push(MirOp::LaneConst {
                        lreg: 0,
                        bits: 0.0f32.to_bits(),
                    });
                    for row in (16..32).chain(48..DST_ROWS).step_by(2) {
                        m.t1.push(MirOp::Sfpstore {
                            lreg: 0,
                            row: base + row,
                            dtype: storage(crate::ir::DType::F32),
                        });
                    }
                    m.t1.push(MirOp::Setrwc {
                        src_a: 0,
                        src_b: 0,
                        dst: 0,
                        flip_src_a: false,
                        flip_src_b: false,
                        dst_cr: false,
                    });
                    acc_mask[acc as usize] = PackMask::Row;
                } else {
                    // Four faces, `SrcA` (256 datums) and the scaler's row (16
                    // datums) each: tt-metal's AB-reduce unpack, then the fold.
                    // The scaler carries the factor in row 0 of each face
                    // (`prepare_reduce_scaler`'s row-0 fill).
                    for face in 0..4u32 {
                        // One face into `SrcA` and one row of the scaler into
                        // `SrcB` (16 datums): tt-metal's AB-reduce unpack.
                        m.t0.push(MirOp::Setadczw { unit: 0, face });
                        m.t0.push(MirOp::UnpackConfig {
                            unit: 0,
                            to_src: true,
                            dtype: dt_x,
                            haloize: false,
                        });
                        m.t0.push(MirOp::Unpacr {
                            cb: cb_x,
                            unit: 0,
                            count: FACE_DATUMS,
                            flip: true,
                        });
                        m.t0.push(MirOp::Setadczw { unit: 1, face });
                        m.t0.push(MirOp::UnpackConfig {
                            unit: 1,
                            to_src: true,
                            dtype: dt_b,
                            haloize: false,
                        });
                        m.t0.push(MirOp::Unpacr {
                            cb: cb_b,
                            unit: 1,
                            count: 16,
                            flip: true,
                        });
                        m.t0.push(MirOp::SemPost { sem: SEM_UNPACKED });
                        m.t1.push(MirOp::SemWait {
                            sem: SEM_UNPACKED,
                            before: 0,
                        });
                        // Faces 0 and 2 fold into the first half of row 0, faces 1
                        // and 3 into the second (tt-metal's `SETRWC(CR_D, 8)`
                        // twice).
                        let dst = base + if face % 2 == 0 { 0 } else { 16 };
                        if max {
                            m.t1.push(MirOp::Gmpool {
                                flip_src_b: false,
                                flip_src_a: false,
                                addr_mod: 0,
                                arg_max: 0,
                                dst_row: dst,
                            });
                        } else {
                            // Four fidelity phases, the last clearing the phase
                            // (tt-metal's `reduce_configure_mop`).
                            for addr_mod in [3u32, 3, 3, 0] {
                                m.t1.push(MirOp::Gapool {
                                    flip_src_b: false,
                                    flip_src_a: false,
                                    addr_mod,
                                    dst_row: dst,
                                });
                            }
                        }
                        // `SETRWC(CLR_AB, 0, 0, 0, 0, SET_AD)`: hand both banks back
                        // and leave the counters where the face started.
                        m.t1.push(MirOp::Setrwc {
                            src_a: 0,
                            src_b: 0,
                            dst: 0,
                            flip_src_a: true,
                            flip_src_b: true,
                            dst_cr: false,
                        });
                    }
                    m.t1.push(MirOp::WaitMatrix);
                    // The pack's edge mask keeps the first row of *every* face,
                    // and the fold only writes faces 0 and 1; the packer cannot
                    // select faces, so faces 2 and 3 (the lower half of the tile)
                    // are written as the zeros the result defines.
                    m.t1.push(MirOp::LaneConst {
                        lreg: 0,
                        bits: 0.0f32.to_bits(),
                    });
                    for row in (32..DST_ROWS).step_by(2) {
                        m.t1.push(MirOp::Sfpstore {
                            lreg: 0,
                            row: base + row,
                            dtype: storage(crate::ir::DType::F32),
                        });
                    }
                    m.t1.push(MirOp::Setrwc {
                        src_a: 0,
                        src_b: 0,
                        dst: 0,
                        flip_src_a: false,
                        flip_src_b: false,
                        dst_cr: false,
                    });
                    acc_mask[acc as usize] = PackMask::Col;
                }
                // A staged operand's page is the compiler's: release it once
                // the fold has drained it.
                for (cb, staged) in [(cb_x, stage_x), (cb_b, stage_b)] {
                    if staged {
                        m.t0.push(MirOp::CbPop { cb });
                    }
                }
            }
            // The arms above cover every `Op`; the matmul/sum-row one is
            // guarded, so the compiler cannot see it does, and a new variant
            // lands here loudly rather than in a silent default.
            _ => unreachable!("every op is handled above"),
        }
        // A page a `matmul` read is released once the last reader of the loaded
        // value has consumed it, which is after the matmul's own unpacks.
        for cb in std::mem::take(&mut pop_here[at]) {
            m.t0.push(MirOp::CbPop { cb });
        }
    }
    // The compiler's 1.0 page: written once, at the start of the kernel, into
    // the last `Dst` slot (the allocator keeps it free), and packed into the
    // page a `max` reduce unpacks. The fills cover the face rows the packer
    // copies into each L1 face's row 0; the rest of the packed tile is the
    // scratch slot's old contents, which nothing reads.
    if let Some(ones) = plan.ones {
        let row = u32::from(crate::alloc::DST_SLOTS - 1) * DST_ROWS;
        let mut t1 = vec![
            MirOp::SemWaitFree { sem: SEM_READY },
            MirOp::LaneConst {
                lreg: 0,
                bits: 1.0f32.to_bits(),
            },
        ];
        for row_off in (0..DST_ROWS).step_by(2) {
            t1.push(MirOp::Sfpstore {
                lreg: 0,
                row: row + row_off,
                dtype: storage(ones.dtype),
            });
        }
        t1.push(MirOp::SemPost { sem: SEM_READY });
        // The slot is reused by the program, so the math does not proceed
        // until the packer has retired this section.
        t1.push(MirOp::SemWaitFree { sem: SEM_READY });
        let mut t2 = vec![
            MirOp::SemWaitData { sem: SEM_READY },
            MirOp::CbReserve { cb: ones.cb },
            MirOp::Pack {
                cb: ones.cb,
                rows: DST_ROWS,
                row,
                last: true,
                dtype: ones.dtype,
                mask: PackMask::None,
            },
            MirOp::SemGet { sem: SEM_READY },
            MirOp::CbPush { cb: ones.cb },
        ];
        t1.append(&mut m.t1);
        t2.append(&mut m.t2);
        m.t1 = t1;
        m.t2 = t2;
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

/// The CB an `Src`-read operand takes, and whether it had to be staged: a
/// `load` already has a page, while a computed value is packed into the
/// compiler's page for it (see [`stage`]). `err` is the operand rule the
/// calling op reports when the value has neither.
fn fpu_page(
    k: &Kernel,
    plan: &Plan,
    def: &[Option<usize>],
    alias: &[Option<crate::ir::ValueId>],
    value: crate::ir::ValueId,
    at: usize,
    err: Error,
) -> Result<(u32, bool), Error> {
    let id = resolve(alias, value);
    let Some(op_at) = def[id as usize] else {
        let _ = (at, value);
        return Err(err);
    };
    if let Op::Load { buffer, .. } = k.ops[op_at] {
        return Ok((
            plan.cb[buffer as usize].ok_or_else(|| {
                let _ = buffer;
                Error::TT_ERROR_BUFFER_NOT_PLACED
            })?,
            false,
        ));
    }
    // Not a `load`: the value lives in `Dst` and the `Src` path reads L1, so
    // the compiler packs it into a page it owns. The verifier admits only
    // f32/bf16 operands, which is also what a staging `Pack` takes.
    let _ = at;
    let cb = plan.staged[id as usize].ok_or_else(|| {
        let _ = (id, value);
        err
    })?;
    Ok((cb, true))
}

/// `SrcA op SrcB -> Dst` with `SrcB` broadcast: tt-metal's
/// `eltwise_binary_configure_addrmod` and its `ELWADD`/`ELWSUB`/`ELWMUL` MOP.
/// The tile is `SrcA` and the narrow value `SrcB`; `ADDR_MOD_0` steps `SrcA` and
/// `Dst` eight rows per instruction and, except for a row or scalar broadcast,
/// steps `SrcB` too, and the end op hands both banks back.
///
/// Both operands are unpacked from L1 pages: a `load`'s page, or the compiler's
/// staged page for a computed value (or for an `acc`, which an FPU op never
/// reads from `Dst`).
#[allow(clippy::too_many_arguments)]
fn fpu_broadcast(
    m: &mut Machine,
    k: &Kernel,
    types: &Types,
    plan: &Plan,
    alloc: &Alloc,
    def: &[Option<usize>],
    alias: &[Option<crate::ir::ValueId>],
    acc_mask: &[PackMask],
    value: crate::ir::ValueId,
    op: BinOp,
    tile: crate::ir::ValueId,
    narrow: crate::ir::ValueId,
    bcast: Bcast,
    at: usize,
) -> Result<(), Error> {
    let (cb_a, stage_a) = fpu_page(k, plan, def, alias, tile, at, Error::TT_TODO_BROADCAST)?;
    let (cb_b, stage_b) = fpu_page(k, plan, def, alias, narrow, at, Error::TT_TODO_BROADCAST)?;
    if stage_a {
        stage(m, types, alloc, alias, acc_mask, tile, cb_a);
    }
    if stage_b {
        stage(m, types, alloc, alias, acc_mask, narrow, cb_b);
    }
    let dt_a = types.of(tile).dtype;
    let dt_b = types.of(narrow).dtype;
    let out = dst_row(alloc, resolve(alias, value));
    // The `Src` banks are filled from the pages the unpacker owns.
    for (cb, staged) in [(cb_a, stage_a), (cb_b, stage_b)] {
        if staged {
            m.t0.push(MirOp::CbWait { cb });
        }
    }
    // The counters start at 0 and `ADDR_MOD_0` steps `SrcA` and `Dst` eight
    // rows per instruction, and `SrcB` too unless the broadcast holds it still.
    m.t1.push(MirOp::Setrwc {
        src_a: 0,
        src_b: 0,
        dst: 0,
        flip_src_a: false,
        flip_src_b: false,
        dst_cr: false,
    });
    m.t1.push(thread_word(&[
        (thread::ADDR_MOD_AB_SEC0_SrcAIncr, 8),
        (
            thread::ADDR_MOD_AB_SEC0_SrcBIncr,
            match bcast {
                Bcast::Col => 8u16,
                Bcast::Row | Bcast::Scalar => 0,
            },
        ),
        (thread::ADDR_MOD_AB_SEC0_SrcBClear, 0),
        (thread::ADDR_MOD_AB_SEC0_SrcAClear, 0),
    ]));
    m.t1.push(thread_word(&[
        (thread::ADDR_MOD_DST_SEC0_DestIncr, 0),
        (thread::ADDR_MOD_DST_SEC0_DestCR, 0),
    ]));
    // The section is written, never accumulated into.
    m.t1.push(MirOp::SemWaitFree { sem: SEM_READY });
    // `ELWMUL`'s word carries bit 21 set, so the hardware accumulates into the
    // destination (tt-isa's measured encoding; `ELWADD`/`ELWSUB` overwrite).
    // The result section is zeroed first so the accumulate is the plain product.
    if matches!(op, BinOp::Mul) {
        m.t1.push(MirOp::LaneConst {
            lreg: 0,
            bits: 0.0f32.to_bits(),
        });
        for row in (0..DST_ROWS).step_by(2) {
            m.t1.push(MirOp::Sfpstore {
                lreg: 0,
                row: out + row,
                dtype: if m.dst_is_16_bit {
                    crate::ir::DType::Bf16
                } else {
                    crate::ir::DType::F32
                },
            });
        }
    }
    for face in 0..4u32 {
        m.t0.push(MirOp::Setadczw { unit: 0, face });
        m.t0.push(MirOp::UnpackConfig {
            unit: 0,
            to_src: true,
            dtype: dt_a,
            haloize: false,
        });
        m.t0.push(MirOp::Unpacr {
            cb: cb_a,
            unit: 0,
            count: FACE_DATUMS,
            flip: true,
        });
        // `SrcB`: the page face the FPU reads for `SrcA`'s face. A narrow
        // value's page keeps only its own part, so both halves of the tile read
        // the same face: a row's faces 0 and 1 (row 0) serve the tile's four
        // faces two at a time, and a column's faces 0 and 2 (column 0) serve
        // each pair of face rows.
        let b_face = match bcast {
            Bcast::Row => face & 1,
            Bcast::Col => face & !1,
            Bcast::Scalar => 0,
        };
        m.t0.push(MirOp::Setadczw {
            unit: 1,
            face: b_face,
        });
        m.t0.push(MirOp::UnpackConfig {
            unit: 1,
            to_src: true,
            dtype: dt_b,
            haloize: false,
        });
        m.t0.push(MirOp::Unpacr {
            cb: cb_b,
            unit: 1,
            count: FACE_DATUMS,
            flip: true,
        });
        m.t0.push(MirOp::SemPost { sem: SEM_UNPACKED });
        m.t1.push(MirOp::SemWait {
            sem: SEM_UNPACKED,
            before: 0,
        });
        // One face is two instructions of eight rows.
        for half in 0..2u32 {
            m.t1.push(MirOp::Elw {
                op: match op {
                    BinOp::Add => ElwOp::Add,
                    BinOp::Sub => ElwOp::Sub,
                    BinOp::Mul => ElwOp::Mul,
                    other => unreachable!("the caller admits add, sub and mul, not {other:?}"),
                },
                bcast,
                addr_mod: 0,
                dst_row: out + 16 * face + 8 * half,
            });
        }
        // `SETRWC(CLR_AB, .., SET_AB)`: the banks go back before the next
        // face's unpack, which is the only way the unpacker may refill them
        // (tt-metal's reduce clears after every face for the same reason). The
        // unpacker refills each face from the bank's start, so the counters stay
        // at 0 and the `SFPU` afterwards addresses `Dst` relative to 0.
        m.t1.push(MirOp::Setrwc {
            src_a: 0,
            src_b: 0,
            dst: 0,
            flip_src_a: true,
            flip_src_b: true,
            dst_cr: false,
        });
    }
    // The counters are back to 0 for the `SFPU`, which addresses `Dst` relative
    // to them.
    m.t1.push(MirOp::Setrwc {
        src_a: 0,
        src_b: 0,
        dst: 0,
        flip_src_a: false,
        flip_src_b: false,
        dst_cr: false,
    });
    for (cb, staged) in [(cb_a, stage_a), (cb_b, stage_b)] {
        if staged {
            m.t0.push(MirOp::CbPop { cb });
        }
    }
    Ok(())
}

/// Pack a computed value's `Dst` rows into the compiler's page for it, so the
/// FPU can read it from `Src`. This is the store path's hand-off without a
/// caller buffer: the math core publishes the section, the packer drains it
/// into the page and publishes the page, and the unpacker waits for that page.
fn stage(
    m: &mut Machine,
    types: &Types,
    alloc: &Alloc,
    alias: &[Option<crate::ir::ValueId>],
    acc_mask: &[PackMask],
    value: crate::ir::ValueId,
    cb: u32,
) {
    let id = resolve(alias, value);
    let dtype = types.of(id).dtype;
    m.t1.push(MirOp::SemWaitFree { sem: SEM_READY });
    m.t1.push(MirOp::SemPost { sem: SEM_READY });
    m.t2.push(MirOp::SemWaitData { sem: SEM_READY });
    m.t2.push(MirOp::CbReserve { cb });
    m.t2.push(MirOp::Pack {
        cb,
        rows: DST_ROWS,
        row: dst_row(alloc, id),
        last: true,
        dtype,
        mask: acc_mask[id as usize],
    });
    m.t2.push(MirOp::SemGet { sem: SEM_READY });
    m.t2.push(MirOp::CbPush { cb });
    // The pack has retired, so the `Dst` slot is free again.
    m.t2.push(MirOp::SemPost { sem: SEM_SLOT });
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

/// The bits a scalar constant operand carries, if `v` is one: `lower` puts
/// them in a lane instead of reading a `Dst` tile, which is what lets the
/// unfold write its coefficients and masks as ordinary `const` values.
fn scalar_const(k: &Kernel, def: &[Option<usize>], v: crate::ir::ValueId) -> Option<u32> {
    match k.ops[def[v as usize]?] {
        Op::Const { bits, .. } => Some(bits),
        _ => None,
    }
}

/// `2^x` for one tile of `Dst` rows, in lane registers.
///
/// tt-metal's `ckernel_sfpu_exp2.h` with zyx's polynomial: `x = n + f` by the
/// `1.5 * 2^23` rounding trick, `2^f` by the `exp(f ln 2)` Horner chain, and
/// the exponent assembled by `SFPEXEXP`/`SFPSETEXP` — `n` is added to the
/// exponent field, which is the step no IR op spells.
///
/// Lanes: `la` holds the input, 2 the rounding constant, 3 a temporary, 4 the
/// fractional part and then the polynomial, 5 the product (and the result), 6
/// `-bits(1.5 * 2^23)` for the integer subtraction that produces `n`, 7 the
/// coefficient of the moment. Constants are re-loaded per row, which is what
/// the allocator pass will fold.
fn exp2_body(
    role: &mut Vec<MirOp>,
    src: u32,
    out: u32,
    rows: &[u32],
    la: u8,
    dtype: crate::ir::DType,
) {
    const C231: u32 = 0x4B40_0000; // 1.5 * 2^23, exact
    const NEG_C231: u32 = 0xB4C0_0000; // the same bits, two's complement negated
    debug_assert!(la < 2, "the body's scratch lanes start at 2");
    // `n = bits(x + c231) - bits(c231)`: the integer subtraction tt-metal's
    // `_sfpu_round_to_nearest_int32_` does, and the exponent it adds.
    role.push(MirOp::LaneConst {
        lreg: 2,
        bits: C231,
    });
    role.push(MirOp::LaneConst {
        lreg: 6,
        bits: NEG_C231,
    });
    for &row in rows {
        role.push(MirOp::Sfpload {
            lreg: la,
            row: src + row,
            dtype,
        });
        // tmp = x + c231, nf = tmp - c231, f = x - nf.
        role.push(MirOp::Sfp {
            op: SfpOp::Add,
            a: la,
            b: Some(2),
            d: 3,
        });
        role.push(MirOp::Sfp {
            op: SfpOp::Sub,
            a: 3,
            b: Some(2),
            d: 4,
        });
        role.push(MirOp::Sfp {
            op: SfpOp::Sub,
            a: la,
            b: Some(4),
            d: 5,
        });
        // n = bits(tmp) - bits(c231), as a two's complement integer.
        role.push(MirOp::Sfp {
            op: SfpOp::IAdd,
            a: 3,
            b: Some(6),
            d: la,
        });
        // s = f ln 2, then the Horner chain into lane 4.
        role.push(MirOp::LaneConst {
            lreg: 7,
            bits: std::f32::consts::LN_2.to_bits(),
        });
        role.push(MirOp::Sfp {
            op: SfpOp::Mul,
            a: 5,
            b: Some(7),
            d: 3,
        });
        role.push(MirOp::LaneConst {
            lreg: 4,
            bits: (1.0f32 / 5040.0).to_bits(),
        });
        for c in [
            1.0f32 / 720.0,
            1.0 / 120.0,
            1.0 / 24.0,
            1.0 / 6.0,
            0.5,
            1.0,
            1.0,
        ] {
            role.push(MirOp::LaneConst {
                lreg: 7,
                bits: c.to_bits(),
            });
            role.push(MirOp::Sfp {
                op: SfpOp::Mul,
                a: 4,
                b: Some(3),
                d: 5,
            });
            role.push(MirOp::Sfp {
                op: SfpOp::Add,
                a: 5,
                b: Some(7),
                d: 4,
            });
        }
        // The result's exponent is the polynomial's plus n.
        role.push(MirOp::Sfp {
            op: SfpOp::Exexp,
            a: 4,
            b: None,
            d: 3,
        });
        role.push(MirOp::Sfp {
            op: SfpOp::IAdd,
            a: 3,
            b: Some(la),
            d: 5,
        });
        role.push(MirOp::Sfp {
            op: SfpOp::Setexp,
            a: 4,
            b: Some(5),
            d: 5,
        });
        role.push(MirOp::Sfpstore {
            lreg: 5,
            row: out + row,
            dtype,
        });
    }
}

/// The `Dst` rows an SFPU body walks. An instruction covers 32 lanes, two
/// 16-datum rows, so a tile is its 64 rows two at a time. A narrow value keeps
/// the acc's narrow part (a `1,32` in rows 0 and 16, a `32,1` in column 0), and
/// its lane layout in the lane — sixteen meaningful lanes of each — is not
/// lowered: a value that reaches the datapath with that shape is refused rather
/// than walked with a tile's addressing.
fn sfp_rows(shape: RegShape, at: usize) -> Result<Vec<u32>, Error> {
    Ok(match shape {
        RegShape::Tile => (0..DST_ROWS).step_by(2).collect(),
        RegShape::Row | RegShape::Column => {
            let _ = (at, shape);
            return Err(Error::TT_TODO_ACCESS_SHAPE);
        }
        RegShape::Scalar => {
            let _ = at;
            return Err(Error::TT_ERROR_SCALAR_NO_ROWS);
        }
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
        BinOp::BitShiftLeft => Ok(SfpOp::Shl),
        BinOp::BitShiftRight => Ok(SfpOp::Shr),
        other => {
            let _ = (at, other);
            Err(Error::TT_TODO_BINARY_SFPU)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `matmul`, the fill and the SFPU all write `Dst` rows the packer reads.
    /// Once a `store_to` posts its section, the next `Dst` write must wait for
    /// the packer to release it — otherwise the two posts coalesce in the
    /// one-deep hand-off and the write lands while the pack is still reading.
    #[test]
    fn a_dst_write_after_a_post_waits_for_the_packer() {
        const PROGRAM: &str = "\
version 9
grid 1,1
%pa = param f32 1024
%pb = param f32 1024
%out = param f32 2048
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %ba [0] = %pa [0]
copy 32,32 %bb [0] = %pb [0]
%m = acc f32 32,32
%zero = const f32 0
store 32,32 %m = %zero
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
matmul %m %a %b
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
store 32,32 %m = %zero
matmul %m %a %b
store 32,32 %bo [0] = %m
copy 32,32 %out [1] = %bo [0]
";
        let k = Kernel::parse(PROGRAM).expect("it parses");
        let types = k.verify().expect("it verifies");
        let k = crate::mover::add_movers(&k, &types).expect("the movers are added");
        let types = k.verify().expect("it verifies");
        let plan = crate::planner::place(&k, &types).expect("it places");
        let alloc = crate::alloc::allocate(&k, &types).expect("it allocates");
        let m = lower(&k, &types, &plan, &alloc).expect("it lowers");

        let mut released = false;
        let mut posts = 0;
        let mut writes = 0;
        for op in &m.t1 {
            match *op {
                MirOp::SemPost { sem } if sem == SEM_READY => {
                    posts += 1;
                    released = false;
                }
                MirOp::SemWaitFree { sem } if sem == SEM_READY => released = true,
                MirOp::Mvmul { .. } | MirOp::Sfpstore { .. } => {
                    writes += 1;
                    assert!(
                        released,
                        "a Dst write follows a section post with no drain wait: {op:?}"
                    );
                }
                _ => {}
            }
        }
        assert_eq!(posts, 2, "one post per pack");
        assert!(writes > 2, "the program writes Dst");
    }

    /// A broadcast is an FPU op, and the FPU's `Src` path takes `f32` and
    /// `bf16` only: an integer broadcast has to go through the SFPU, which is
    /// not lowered. It must be reported as that route's TODO, never routed
    /// through the FPU.
    #[test]
    fn an_integer_broadcast_is_refused_as_an_sfpu_todo() {
        const PROGRAM: &str = "\
version 9
grid 1,1
%pi = param i32 1024
%po = param i32 1024
%bi = buffer i32 32,32
%bo = buffer i32 32,32
copy 32,32 %bi [0] = %pi [0]
load 32,32 %t = %bi [0]
load 1,1 %s = %bi [0]
%d = add %t %s
store 32,32 %bo [0] = %d
copy 32,32 %po [0] = %bo [0]
";
        let k = Kernel::parse(PROGRAM).expect("it parses");
        let types = k.verify().expect("it verifies");
        let k = crate::mover::add_movers(&k, &types).expect("the movers are added");
        let types = k.verify().expect("it verifies");
        let plan = crate::planner::place(&k, &types).expect("it places");
        let alloc = crate::alloc::allocate(&k, &types).expect("it allocates");
        assert_eq!(
            lower(&k, &types, &plan, &alloc).expect_err("an integer broadcast has no path"),
            Error::TT_TODO_BROADCAST_SFPU
        );
    }
}
