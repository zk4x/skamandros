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
use crate::ir::{BinOp, Error, Kernel, Op, RegShape, Types, UnOp, unit_of};
use crate::mir::{CmpOp, Machine, MirOp, SfpOp};
use crate::planner::Plan;

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
                let _ = at;
                return Err(Error::TT_TODO_LOOP);
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
                let int = match dtype {
                    crate::ir::DType::F32 => false,
                    crate::ir::DType::I32 => true,
                    other => {
                        let _ = other;
                        return Err(Error::TT_ERROR_DATAPATH_DTYPE);
                    }
                };
                let (unit_shape, _, _) = unit_of(shape, dtype)?;
                let rows = rows_of(unit_shape, at)?;
                let cb = plan.cb[buffer as usize].ok_or_else(|| {
                    let _ = buffer;
                    Error::TT_ERROR_BUFFER_NOT_PLACED
                })?;
                let row = match alloc.place_of(value) {
                    crate::alloc::Place::Dst(slot) => u32::from(slot) * DST_ROWS,
                };
                // The unpacker writes `Dst`: it waits for a slot the packer has
                // given back, so the first unit's result cannot be overwritten
                // before it has been packed. The page itself comes from the ring:
                // wait for the producer's push, unpack, then release the page.
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
                    int,
                });
                m.t0.push(MirOp::CbPop { cb });
                m.t0.push(MirOp::SemPost { sem: SEM_UNPACKED });
                m.t1.push(MirOp::SemWait {
                    sem: SEM_UNPACKED,
                    before: 0,
                });
                let _ = info;
                let _ = rows;
                let _ = offset;
            }
            Op::StoreTo {
                value,
                buffer,
                offset,
            } => {
                let Op::Buffer { shape, dtype, .. } = k.ops[def[buffer as usize].expect("defined")]
                else {
                    let _ = buffer;
                    return Err(Error::TT_ERROR_NOT_A_BUFFER);
                };
                let int = match dtype {
                    crate::ir::DType::F32 => false,
                    crate::ir::DType::I32 => true,
                    other => {
                        let _ = other;
                        return Err(Error::TT_ERROR_DATAPATH_DTYPE);
                    }
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
                    last: true,
                    int,
                });
                m.t2.push(MirOp::SemGet { sem: SEM_READY });
                m.t2.push(MirOp::CbPush { cb });
                // The pack has retired, so the `Dst` slot is free again.
                m.t2.push(MirOp::SemPost { sem: SEM_SLOT });
                let _ = value;
                let _ = offset;
            }
            Op::Cast { .. } => {
                let _ = at;
                return Err(Error::TT_TODO_CAST);
            }
            Op::Unary { value, op, a } => {
                let shape = types.of(a).shape;
                let rows = rows_of(shape, at)?;
                let src = dst_row(alloc, a);
                let out = dst_row(alloc, value);
                let (la, ld) = alloc.lreg[value as usize];
                let sfp = unary(op, at)?;
                let int = types.of(a).dtype == crate::ir::DType::I32;
                for row in (0..rows).step_by(2) {
                    m.t1.push(MirOp::Sfpload {
                        lreg: la,
                        row: src + row,
                        int,
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
                        int,
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
                let int = match left.dtype {
                    crate::ir::DType::F32 => false,
                    crate::ir::DType::I32
                        if matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor) =>
                    {
                        true
                    }
                    // A bool lane is `-1`/`0`, so the bitwise and/or *is* the
                    // logical one (this is what zyx's `tt_sfp_integer` does for
                    // `BOp::And`/`Or` too).
                    crate::ir::DType::Bool if matches!(op, BinOp::And | BinOp::Or) => true,
                    other => {
                        let _ = (op, other);
                        return Err(Error::TT_TODO_BINARY_DTYPE);
                    }
                };
                let rows = rows_of(left.shape, at)?;
                let sa = dst_row(alloc, a);
                let sb = dst_row(alloc, b);
                let out = dst_row(alloc, value);
                let (la, ld) = alloc.lreg[value as usize];
                let lb = 2u8;
                // A comparison writes `-1`/`0` lanes, so its result is stored in
                // the integer format; everything else here is FP32.
                let int_out = int || matches!(op, BinOp::Cmplt | BinOp::Cmpgt | BinOp::Cmpge);
                for row in (0..rows).step_by(2) {
                    m.t1.push(MirOp::Sfpload {
                        lreg: la,
                        row: sa + row,
                        int,
                    });
                    m.t1.push(MirOp::Sfpload {
                        lreg: lb,
                        row: sb + row,
                        int,
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
                        int: int_out,
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
                let sc = dst_row(alloc, cond);
                let st = dst_row(alloc, if_true);
                let sf = dst_row(alloc, if_false);
                let out = dst_row(alloc, value);
                let ld = alloc.lreg[value as usize].1;
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
                        int: true,
                    });
                    m.t1.push(MirOp::Sfpload {
                        lreg: lt,
                        row: st + row,
                        int: false,
                    });
                    m.t1.push(MirOp::Sfpload {
                        lreg: lf,
                        row: sf + row,
                        int: false,
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
                        int: false,
                    });
                }
            }
            Op::Matmul { .. } => {
                let _ = at;
                return Err(Error::TT_TODO_MATMUL);
            }
        }
    }
    Ok(m)
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
