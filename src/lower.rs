// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Unified program to machine IR: the first lowering step.
//!
//! `load` configures and fills the unpacker on T0, the register ops become
//! load/op/store bodies on T1, `store_to` configures and drains the packer on
//! T2. `copy_to` crosses the L1 boundary and is emitted by [`crate::mover`], not
//! here. The three cores are paced by two semaphores: the unpacker publishes
//! each loaded unit, and the math core publishes each stored one.
//!
//! Addresses come from the L1 plan, `Dst` slots from the allocation; the caller
//! named none of them.

use crate::alloc::Alloc;
use crate::ir::{BinOp, Dim, Error, Kernel, Op, RegShape, Types, UnOp, unit_of};
use crate::mir::{Machine, MirOp, SfpOp};
use crate::planner::Plan;

/// `Dst` rows one 32x32 F32 tile occupies (`tt_isa`'s SFPU row is 16 datums).
pub const DST_ROWS: u32 = 64;
/// Datums in one 32x32 tile.
pub const TILE_DATUMS: u32 = 1024;
/// Semaphore the unpacker publishes a loaded unit on.
pub const SEM_UNPACKED: u8 = 0;
/// Semaphore the math core publishes a stored unit on.
pub const SEM_READY: u8 = 1;

/// Lower `k` to the three compute programs.
pub fn lower(k: &Kernel, types: &Types, plan: &Plan, alloc: &Alloc) -> Result<Machine, Error> {
    let mut m = Machine::default();
    let mut def: Vec<Option<usize>> = vec![None; types.values.len()];
    for (at, op) in k.ops.iter().enumerate() {
        if let Some(value) = op.defined() {
            def[value as usize] = Some(at);
        }
    }
    for (at, op) in k.ops.iter().enumerate() {
        match *op {
            Op::Const { .. }
            | Op::Param { .. }
            | Op::Buffer { .. }
            | Op::Grid { .. }
            | Op::CopyTo { .. } => {}
            Op::Loop { .. } | Op::EndLoop => {
                return Err(Error(format!(
                    "op {at}: loops are not lowered yet; the parser and verifier accept them"
                )));
            }
            Op::Load {
                value,
                buffer,
                offset,
            } => {
                let info = types.of(buffer);
                let Op::Buffer { shape, dtype, .. } = k.ops[def[buffer as usize].expect("defined")]
                else {
                    return Err(Error(format!("op {at}: %{buffer} is not a buffer")));
                };
                if dtype != crate::ir::DType::F32 {
                    return Err(Error(format!(
                        "op {at}: the register datapath is FP32 only for now, not {}",
                        dtype.name()
                    )));
                }
                let (unit_shape, _, _) = unit_of(shape, dtype)?;
                let rows = rows_of(unit_shape, at)?;
                let offset = const_offset(offset, at)?;
                let base = plan.buf[buffer as usize]
                    .ok_or_else(|| Error(format!("op {at}: buffer %{buffer} was not placed")))?;
                let src = base + offset * dtype.bytes();
                let row = match alloc.place_of(value) {
                    crate::alloc::Place::Dst(slot) => u32::from(slot) * DST_ROWS,
                };
                m.t0.push(MirOp::Setdst { row });
                m.t0.push(MirOp::Setadczw { unit: 0, face: 0 });
                m.t0.push(MirOp::Setadcx {
                    unit: 0,
                    first: 0,
                    count: unit_shape.datums(),
                });
                m.t0.push(MirOp::Unpacr {
                    src,
                    unit: 0,
                    count: unit_shape.datums(),
                });
                m.t0.push(MirOp::SemPost { sem: SEM_UNPACKED });
                m.t1.push(MirOp::SemWait {
                    sem: SEM_UNPACKED,
                    before: 0,
                });
                let _ = info;
                let _ = rows;
            }
            Op::StoreTo {
                value,
                buffer,
                offset,
            } => {
                let Op::Buffer { shape, dtype, .. } = k.ops[def[buffer as usize].expect("defined")]
                else {
                    return Err(Error(format!("op {at}: %{buffer} is not a buffer")));
                };
                if dtype != crate::ir::DType::F32 {
                    return Err(Error(format!(
                        "op {at}: the register datapath is FP32 only for now, not {}",
                        dtype.name()
                    )));
                }
                let (unit_shape, _, _) = unit_of(shape, dtype)?;
                let rows = rows_of(unit_shape, at)?;
                let offset = const_offset(offset, at)?;
                let base = plan.buf[buffer as usize]
                    .ok_or_else(|| Error(format!("op {at}: buffer %{buffer} was not placed")))?;
                let dst = base + offset * dtype.bytes();
                m.t1.push(MirOp::SemPost { sem: SEM_READY });
                m.t2.push(MirOp::SemWait {
                    sem: SEM_READY,
                    before: 0,
                });
                m.t2.push(MirOp::Pack {
                    dst,
                    rows,
                    last: true,
                });
                let _ = value;
            }
            Op::Cast { .. } => {
                return Err(Error(format!(
                    "op {at}: cast is not lowered yet; the parser and verifier accept it"
                )));
            }
            Op::Unary { value, op, a } => {
                let shape = types.of(a).shape;
                let rows = rows_of(shape, at)?;
                let src = dst_row(alloc, a);
                let out = dst_row(alloc, value);
                let (la, ld) = alloc.lreg[value as usize];
                let sfp = unary(op, at)?;
                for row in (0..rows).step_by(2) {
                    m.t1.push(MirOp::Sfpload {
                        lreg: la,
                        row: src + row,
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
                    });
                }
            }
            Op::Binary { value, op, a, b } => {
                let left = types.of(a);
                let right = types.of(b);
                if left.shape != right.shape {
                    return Err(Error(format!(
                        "op {at}: broadcast of a {} and a {} is not lowered yet",
                        right.shape.name(),
                        left.shape.name()
                    )));
                }
                if left.dtype != crate::ir::DType::F32 {
                    return Err(Error(format!(
                        "op {at}: the register datapath is FP32 only for now, not {}",
                        left.dtype.name()
                    )));
                }
                let rows = rows_of(left.shape, at)?;
                let sa = dst_row(alloc, a);
                let sb = dst_row(alloc, b);
                let out = dst_row(alloc, value);
                let (la, ld) = alloc.lreg[value as usize];
                let lb = 2u8;
                let sfp = binary(op, at)?;
                for row in (0..rows).step_by(2) {
                    m.t1.push(MirOp::Sfpload {
                        lreg: la,
                        row: sa + row,
                    });
                    m.t1.push(MirOp::Sfpload {
                        lreg: lb,
                        row: sb + row,
                    });
                    m.t1.push(MirOp::Sfp {
                        op: sfp,
                        a: la,
                        b: Some(lb),
                        d: ld,
                    });
                    m.t1.push(MirOp::Sfpstore {
                        lreg: ld,
                        row: out + row,
                    });
                }
            }
            Op::Matmul { .. } => {
                return Err(Error(format!(
                    "op {at}: the matrix path is not lowered yet"
                )));
            }
        }
    }
    Ok(m)
}

fn rows_of(shape: RegShape, at: usize) -> Result<u32, Error> {
    match shape {
        RegShape::Scalar => Err(Error(format!(
            "op {at}: a scalar register value has no row body"
        ))),
        RegShape::Row | RegShape::Column => Ok(2),
        RegShape::Tile => Ok(DST_ROWS),
    }
}

fn dst_row(alloc: &Alloc, value: crate::ir::ValueId) -> u32 {
    match alloc.place_of(value) {
        crate::alloc::Place::Dst(slot) => u32::from(slot) * DST_ROWS,
    }
}

fn const_offset(dim: Dim, at: usize) -> Result<u32, Error> {
    match dim {
        Dim::Const(n) => Ok(n),
        Dim::Value(v) => Err(Error(format!(
            "op {at}: the offset %{v} is symbolic; only a literal offset is lowered yet"
        ))),
    }
}

fn unary(op: UnOp, at: usize) -> Result<SfpOp, Error> {
    match op {
        UnOp::Neg => Ok(SfpOp::Neg),
        UnOp::Abs => Ok(SfpOp::Abs),
        UnOp::Relu => Err(Error(format!("op {at}: relu has no encoding yet"))),
    }
}

fn binary(op: BinOp, at: usize) -> Result<SfpOp, Error> {
    match op {
        BinOp::Add => Ok(SfpOp::Add),
        BinOp::Sub => Ok(SfpOp::Sub),
        BinOp::Mul => Ok(SfpOp::Mul),
        other => Err(Error(format!(
            "op {at}: {} has no SFPU encoding yet",
            other.name()
        ))),
    }
}
