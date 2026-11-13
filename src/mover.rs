// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The boundary programs: DRAM <-> L1 over the NoC.
//!
//! Every `copy` between a parameter and a buffer becomes one NoC request per
//! lane — one for a scalar index, one per datum of a vector index — and the ten
//! initiator registers are written in `tt_isa`'s order, `CMD_CTRL` triggers it,
//! and the program waits on that word and then on the transaction ID's
//! outstanding counter before the next copy. The parameter's address is its
//! launch argument plus the copy's datum offset; the local address is the
//! planned L1 buffer plus the same datum offset. Both sides are 64-byte aligned
//! and add the same byte offset, so a request's two addresses are congruent.
//!
//! `TARG_ADDR` is the request's source and `RET_ADDR` its destination, so a
//! read names the parameter as target and L1 as return, and a write the other
//! way round. Which copies belong to the inbound program and which to the
//! outbound one is decided here from the operands, not by the caller.

use std::collections::BTreeMap;

use crate::ir::{BinOp, Dim, Error, Kernel, Op, OpId, Types, ValueId};
use crate::planner::Plan;
use crate::riscv::{self as rv, Asm, reg};
use tt_isa::noc::niu::{
    CMD_RD, CMD_WR, Niu, RESP_MARKED, STATIC_VC_1, TxnId, initiator, reqs_outstanding,
};

/// Words of launch arguments before the per-parameter pairs: grid x, grid y,
/// reserved, this tile's NOC high word.
pub const ARG_HEAD: usize = 4;

/// The launch argument word with the CB interface table (tt-metal's
/// `ARG_IFACE`); word 2 is the slot that used to be reserved.
pub const ARG_IFACE: usize = 2;

/// Transaction ID every request uses. The programs issue in order and drain
/// after each copy, so one ID is enough.
const TXN: TxnId = match TxnId::new(1) {
    Some(t) => t,
    None => panic!("transaction 1 is valid"),
};

/// The launch argument words: one `(low offset, endpoint high word)` pair per
/// parameter, after [`ARG_HEAD`].
pub const fn param_arg(param: u32) -> (i32, i32) {
    let lo = (ARG_HEAD + 2 * param as usize) * 4;
    (lo as i32, lo as i32 + 4)
}

/// One copy, resolved against the plan.
struct Copy {
    param: u32,
    /// Where in the parameter this copy reads or writes.
    offset: ParamOffset,
    /// The CB whose ring holds the L1 side of the copy.
    cb: u32,
    bytes: u32,
}

/// The parameter side of a copy: a literal byte offset, or a value computed
/// each iteration of an enclosing loop (the address a loop index names), or one
/// base per lane read out of a page the mover copied (a gather's row of ids).
enum ParamOffset {
    Const(u32),
    /// `value` is the index expression, in units of the access; `dtype_bytes`
    /// scales it to bytes.
    Index {
        value: ValueId,
        dtype_bytes: u32,
    },
    /// `load` names a vector read out of L1: `lanes` bases, each in units of the
    /// access, one NoC request per base.
    Lanes {
        load: ValueId,
        lanes: u32,
        dtype_bytes: u32,
    },
}

/// One item of a boundary program, in program order. The copies between a
/// [`Item::Loop`] and its [`Item::EndLoop`] run once per iteration; the loop is
/// a RISC-V branch, not an unroll, so the trip count can be any constant.
enum Item {
    Copy(Copy),
    Loop { index: ValueId, count: LoopCount },
    EndLoop,
}

/// A loop's trip count: a literal, or an index expression the program computes
/// into the limit register before the branch.
#[derive(Clone, Copy)]
enum LoopCount {
    Const(u32),
    Value(ValueId),
}

/// The parameter-side offset [`emit_copy`] adds: a literal, or the register an
/// enclosing loop's index expression left it in.
enum CopyOffset {
    Const(u32),
    Reg(u8),
}

/// One open loop in a boundary program: the register pair the branch counts in,
/// and the label it branches back to.
struct MoverLoop {
    index: ValueId,
    index_reg: u8,
    limit_reg: u8,
    top: String,
}

/// The inbound program: every copy from a parameter into a buffer, in program
/// order. Empty when the program has none.
pub fn inbound(k: &Kernel, types: &Types, plan: &Plan) -> Result<Vec<u32>, Error> {
    program(k, types, plan, true)
}

/// The outbound program: every copy from a buffer into a parameter.
pub fn outbound(k: &Kernel, types: &Types, plan: &Plan) -> Result<Vec<u32>, Error> {
    program(k, types, plan, false)
}

/// The boundary `tt_add_movers` draws, on skamandros's IR: a `load` or
/// `store_to` against a parameter is staged through a fresh one-unit L1 buffer,
/// with the inbound/outbound copy emitted around it. The caller names the
/// parameter and the offset; the compiler adds the traffic, exactly as zyx's
/// `tt_add_movers` turns a `Load`/`Store` against a `Global` into a `Copy` into a
/// circular buffer plus the register access.
///
/// One unit per access: the staged buffer is the access's own shape, so a tile
/// load gets a `32,32` buffer and a scalar one a `1,1`. Every program after this
/// pass sees only buffers and copies; nothing downstream knows the parameter was
/// named directly.
pub fn add_movers(k: &Kernel, types: &Types) -> Result<Kernel, Error> {
    let direct = |op: &Op| match *op {
        Op::Load { buffer, .. } | Op::StoreTo { buffer, .. } => {
            types.of(buffer).kind == crate::ir::ValueKind::Param
        }
        _ => false,
    };
    if !k.ops().any(|(_, op)| direct(op)) {
        return Ok(k.clone());
    }

    let mut out = Kernel::default();
    // Old value id -> new value id. Definitions are emitted in the same order,
    // so a use only needs an already-emitted definition.
    let mut map: BTreeMap<OpId, OpId> = BTreeMap::new();
    let mut stage = 0usize;

    for (_, op) in k.ops() {
        match *op {
            Op::Load {
                value,
                buffer,
                offset,
                shape: _,
            } if types.of(buffer).kind == crate::ir::ValueKind::Param => {
                let info = *types.of(buffer);
                let shape = stage_shape(info.shape, buffer)?;
                let staged = fresh(&mut out, k, &mut stage);
                out.push(Op::Buffer {
                    value: staged,
                    dtype: info.dtype,
                    shape,
                });
                let src = map[&buffer];
                out.push(Op::CopyTo {
                    dst: staged,
                    src,
                    dst_offset: Dim::Const(0),
                    src_offset: move_dim(offset, &map),
                    shape,
                });
                let id = out.ops.max_id();
                out.push(Op::Load {
                    value: id,
                    buffer: staged,
                    offset: Dim::Const(0),
                    shape,
                });
                map.insert(value, id);
                name(&mut out, k, value, id);
            }
            Op::StoreTo {
                value,
                buffer,
                offset,
                shape: _,
            } if types.of(buffer).kind == crate::ir::ValueKind::Param => {
                let info = *types.of(buffer);
                let shape = stage_shape(info.shape, buffer)?;
                let staged = fresh(&mut out, k, &mut stage);
                out.push(Op::Buffer {
                    value: staged,
                    dtype: info.dtype,
                    shape,
                });
                let value = map[&value];
                out.push(Op::StoreTo {
                    value,
                    buffer: staged,
                    offset: Dim::Const(0),
                    shape,
                });
                let dst = map[&buffer];
                out.push(Op::CopyTo {
                    dst,
                    src: staged,
                    dst_offset: move_dim(offset, &map),
                    src_offset: Dim::Const(0),
                    shape,
                });
            }
            _ => {
                let mut op = op.clone();
                let id = out.ops.max_id();
                if let Some(old) = op.defined() {
                    map.insert(old, id);
                    name(&mut out, k, old, id);
                }
                crate::ir::renumber(&mut op, &map)?;
                out.push(op);
            }
        }
    }
    Ok(out)
}

/// The staged buffer's shape for a direct parameter access. A tensor parameter
/// is a tile; a scalar one would need the 64-byte span read, which the movers do
/// not do yet.
fn stage_shape(shape: crate::ir::RegShape, param: ValueId) -> Result<[u32; 2], Error> {
    match shape {
        crate::ir::RegShape::Tile => Ok([32, 32]),
        other => {
            let _ = (param, other);
            Err(Error::TT_TODO_SCALAR_PARAMETER)
        }
    }
}

/// Define one fresh staging value, with a name no existing value has, at the id
/// its `Buffer` will get.
fn fresh(out: &mut Kernel, k: &Kernel, stage: &mut usize) -> ValueId {
    let value = out.ops.max_id();
    loop {
        let name = format!("stage{}", *stage);
        *stage += 1;
        let taken = k.names.values().any(|n| *n == name) || out.names.values().any(|n| *n == name);
        if !taken {
            out.names.insert(value, name);
            return value;
        }
    }
}

/// Give a copied definition its name under its new id.
fn name(out: &mut Kernel, k: &Kernel, old: ValueId, new: ValueId) {
    if let Some(name) = k.names.get(&old) {
        out.names.insert(new, name.clone());
    }
}

/// The same bracket with every value it names renamed.
fn move_dim(dim: Dim, map: &BTreeMap<OpId, OpId>) -> Dim {
    match dim {
        Dim::Const(n) => Dim::Const(n),
        Dim::Value(v) => Dim::Value(map[&v]),
    }
}

/// The same op with every value it reads renamed, and (for a definition) its
/// own new id installed.
#[allow(dead_code)]
fn remap(op: &Op, map: &BTreeMap<OpId, OpId>, value: Option<ValueId>) -> Op {
    let v = |x: ValueId| map[&x];
    let d = |x: Dim| move_dim(x, map);
    let own = || value.expect("a definition has a new id");
    match *op {
        Op::Const { dtype, bits, .. } => Op::Const {
            value: own(),
            dtype,
            bits,
        },
        Op::Param {
            dtype, ref dims, ..
        } => Op::Param {
            value: own(),
            dtype,
            dims: dims.iter().map(|x| d(*x)).collect(),
        },
        Op::Buffer { dtype, shape, .. } => Op::Buffer {
            value: own(),
            dtype,
            shape,
        },
        Op::Acc { dtype, shape, .. } => Op::Acc {
            value: own(),
            dtype,
            shape,
        },
        Op::Grid { columns, rows } => Op::Grid {
            columns: d(columns),
            rows: d(rows),
        },
        Op::Loop { bound, .. } => Op::Loop {
            value: own(),
            bound: d(bound),
        },
        Op::EndLoop => Op::EndLoop,
        Op::GridCoord { axis, .. } => Op::GridCoord { value: own(), axis },
        Op::CopyTo {
            dst,
            src,
            dst_offset,
            src_offset,
            shape,
        } => Op::CopyTo {
            dst: v(dst),
            src: v(src),
            dst_offset: d(dst_offset),
            src_offset: d(src_offset),
            shape,
        },
        Op::Load {
            buffer,
            offset,
            shape,
            ..
        } => Op::Load {
            value: own(),
            buffer: v(buffer),
            offset: d(offset),
            shape,
        },
        Op::StoreTo {
            value: src,
            buffer,
            offset,
            shape,
        } => Op::StoreTo {
            value: v(src),
            buffer: v(buffer),
            offset: d(offset),
            shape,
        },
        Op::Cast { src, dtype, .. } => Op::Cast {
            value: own(),
            src: v(src),
            dtype,
        },
        Op::Bitcast { src, dtype, .. } => Op::Bitcast {
            value: own(),
            src: v(src),
            dtype,
        },
        Op::Unary { op, a, .. } => Op::Unary {
            value: own(),
            op,
            a: v(a),
        },
        Op::Binary { op, a, b, .. } => Op::Binary {
            value: own(),
            op,
            a: v(a),
            b: v(b),
        },
        Op::Select {
            cond,
            if_true,
            if_false,
            ..
        } => Op::Select {
            value: own(),
            cond: v(cond),
            if_true: v(if_true),
            if_false: v(if_false),
        },
        Op::Matmul { acc, a, b } => Op::Matmul {
            acc: v(acc),
            a: v(a),
            b: v(b),
        },
        Op::Sum { acc, x, scaler } => Op::Sum {
            acc: v(acc),
            x: v(x),
            scaler: v(scaler),
        },
        Op::Max { acc, x } => Op::Max {
            acc: v(acc),
            x: v(x),
        },
    }
}

fn program(k: &Kernel, types: &Types, plan: &Plan, inbound: bool) -> Result<Vec<u32>, Error> {
    let items = collect(k, types, plan, inbound)?;
    if !items.iter().any(|item| matches!(item, Item::Copy(_))) {
        return Ok(Vec::new());
    }
    let mut a = Asm::default();
    prologue(&mut a, if inbound { Niu::Noc0 } else { Niu::Noc1 });
    // The sync scratch leaves `t3` (this tile's high word), `t5` (the NOC
    // initiator) and `a6` (the interface) alone.
    let sync3 = [reg::T0, reg::T1, reg::T2];
    let sync4 = [reg::T0, reg::T1, reg::T2, reg::T6];
    // Temporaries for an index expression. An expression's right spine needs one
    // per level (`%t = add %site %step`, `%step = mul %i %nsites`, `%nsites =
    // mul %nx %ny` is three), and at this point the CB counters are in memory and
    // the sync scratch has been dropped, so the three sync words are free again.
    let mut eval_temps: [u8; 5] = [reg::T6, reg::T4, reg::T0, reg::T1, reg::T2];
    let param_index = param_indices(k);
    let mut loops: Vec<MoverLoop> = Vec::new();
    let mut at = 0usize;
    for item in &items {
        match *item {
            Item::Loop { index, count } => {
                // `a0` holds the arguments, `a5` stages the offset, `t3`/`t5`
                // hold the NoC words. Two nesting levels fit in `a1..a4`.
                let (index_reg, limit_reg) = match loops.len() {
                    0 => (reg::A1, reg::A2),
                    1 => (reg::A3, reg::A4),
                    _ => return Err(Error::TT_TODO_LOOP_DEPTH),
                };
                a.li(index_reg, 0);
                match count {
                    LoopCount::Const(n) => a.li(limit_reg, n),
                    LoopCount::Value(value) => eval_value(
                        &mut a,
                        k,
                        plan,
                        &param_index,
                        &loops,
                        value,
                        limit_reg,
                        &mut eval_temps,
                    )?,
                }
                let top = a.fresh_label();
                a.label(top.clone());
                loops.push(MoverLoop {
                    index,
                    index_reg,
                    limit_reg,
                    top,
                });
            }
            Item::EndLoop => {
                let loop_ = loops.pop().ok_or(Error::TT_ERROR_ENDLOOP_MACHINE)?;
                a.word(rv::addi(loop_.index_reg, loop_.index_reg, 1));
                a.bltu_label(loop_.index_reg, loop_.limit_reg, loop_.top);
            }
            Item::Copy(ref copy) => {
                let cb = copy.cb;
                if inbound {
                    rv::cb::reserve_back(&mut a, reg::A6, cb, 1, sync3);
                } else {
                    rv::cb::wait_front(&mut a, reg::A6, cb, 1, sync3);
                }
                match copy.offset {
                    ParamOffset::Lanes {
                        load,
                        lanes,
                        dtype_bytes,
                    } => emit_lanes(
                        &mut a,
                        k,
                        plan,
                        at,
                        copy,
                        load,
                        lanes,
                        dtype_bytes,
                        loops.len(),
                        inbound,
                    )?,
                    ref single => {
                        let offset = match *single {
                            ParamOffset::Const(bytes) => CopyOffset::Const(bytes),
                            ParamOffset::Index { value, dtype_bytes } => {
                                eval_value(
                                    &mut a,
                                    k,
                                    plan,
                                    &param_index,
                                    &loops,
                                    value,
                                    reg::A5,
                                    &mut eval_temps,
                                )?;
                                if dtype_bytes != 1 {
                                    a.li(reg::T4, dtype_bytes);
                                    a.word(rv::mul(reg::A5, reg::A5, reg::T4));
                                }
                                CopyOffset::Reg(reg::A5)
                            }
                            ParamOffset::Lanes { .. } => unreachable!("the arm above took it"),
                        };
                        emit_copy(
                            &mut a, at, copy.param, offset, None, cb, copy.bytes, inbound,
                        )?;
                    }
                }
                if inbound {
                    rv::cb::push_back(&mut a, reg::A6, cb, 1, sync4);
                } else {
                    rv::cb::pop_front(&mut a, reg::A6, cb, 1, sync4);
                }
                at += 1;
            }
        }
    }
    if !loops.is_empty() {
        return Err(Error::TT_ERROR_ENDLOOP_MACHINE);
    }
    a.word(rv::ret());
    a.finish().map_err(|_| Error::TT_ERROR_MOVER_PROGRAM)
}

/// The launch argument index of every parameter, in program order.
fn param_indices(k: &Kernel) -> BTreeMap<OpId, u32> {
    let mut index: BTreeMap<OpId, u32> = BTreeMap::new();
    let mut next = 0u32;
    for (_, op) in k.ops() {
        if let Op::Param { value, .. } = *op {
            index.insert(value, next);
            next += 1;
        }
    }
    index
}

/// Compute `value` (an `i32` index expression) into `dest` with ordinary
/// RV32IM, the way a C kernel's loop address is computed by the RISC-V
/// compiler.
///
/// Only expressions a RISC-V instruction can evaluate are accepted: constants,
/// an enclosing loop's index, and `+ - * / % & | ^ << >>` over those. Anything
/// else (a comparison, `max`, `pow`, a cast) would need a decision this pass
/// must not make, so it is a concrete refusal.
///
/// `temps` are the registers a right operand may use; the left accumulates in
/// `dest`. `A1..A4` are the loop indices and are only read.
fn eval_value(
    a: &mut Asm,
    k: &Kernel,
    plan: &Plan,
    param_index: &BTreeMap<OpId, u32>,
    loops: &[MoverLoop],
    value: ValueId,
    dest: u8,
    temps: &mut [u8],
) -> Result<(), Error> {
    if let Some(loop_) = loops.iter().rev().find(|l| l.index == value) {
        let index_reg = loop_.index_reg;
        a.word(rv::add(dest, index_reg, reg::ZERO));
        return Ok(());
    }
    let Some(op) = k.op(value) else {
        let _ = value;
        return Err(Error::TT_TODO_LOOP_ADDRESS);
    };
    match *op {
        Op::Const {
            dtype: crate::ir::DType::I32,
            bits,
            ..
        } => {
            a.li(dest, bits);
            Ok(())
        }
        // A scalar parameter's launch argument is its value, not an address:
        // the array the host wrote holds the value word (`get_arg_val`).
        Op::Param { ref dims, .. } if dims.is_empty() => {
            let index = *param_index.get(&value).ok_or(Error::TT_TODO_LOOP_ADDRESS)?;
            a.word(rv::lw(dest, reg::A0, param_arg(index).0));
            Ok(())
        }
        // This core's logical coordinate, host-written in `core_info`.
        Op::GridCoord { axis, .. } => {
            a.li(dest, crate::mir::CORE_INFO + u32::from(axis) * 4);
            a.word(rv::lw(dest, dest, 0));
            Ok(())
        }
        Op::Binary {
            op, a: lhs, b: rhs, ..
        } => {
            if !matches!(
                op,
                BinOp::Add
                    | BinOp::Sub
                    | BinOp::Mul
                    | BinOp::Div
                    | BinOp::Mod
                    | BinOp::BitAnd
                    | BinOp::BitOr
                    | BinOp::BitXor
                    | BinOp::BitShiftLeft
                    | BinOp::BitShiftRight
            ) {
                let _ = op;
                return Err(Error::TT_TODO_LOOP_ADDRESS);
            }
            eval_value(a, k, plan, param_index, loops, lhs, dest, temps)?;
            let Some((right, rest)) = temps.split_first_mut() else {
                return Err(Error::TT_TODO_LOOP_ADDRESS);
            };
            let right = *right;
            eval_value(a, k, plan, param_index, loops, rhs, right, rest)?;
            a.word(match op {
                BinOp::Add => rv::add(dest, dest, right),
                BinOp::Sub => rv::sub(dest, dest, right),
                BinOp::Mul => rv::mul(dest, dest, right),
                BinOp::Div => rv::div(dest, dest, right),
                BinOp::Mod => rv::rem(dest, dest, right),
                BinOp::BitAnd => rv::and(dest, dest, right),
                BinOp::BitOr => rv::or(dest, dest, right),
                BinOp::BitXor => rv::xor(dest, dest, right),
                BinOp::BitShiftLeft => rv::sll(dest, dest, right),
                // The unroll folded `i32::checked_shr`, an arithmetic shift.
                BinOp::BitShiftRight => rv::sra(dest, dest, right),
                _ => unreachable!("the match above refused the rest"),
            });
            Ok(())
        }
        // A scalar read out of an L1 page: an index the mover brought in
        // itself, read back as a register (tt-metal's `input_l1_ptr[index]`,
        // zyx's `value_into`). It is a plain `lw`, not a NoC request, so no
        // congruence rule applies. The page address is a compile-time constant
        // only on a one-page ring: the pointer wraps back to the base after
        // every push.
        Op::Load {
            buffer, ref offset, ..
        } => {
            let dtype = match k.at(buffer) {
                Op::Buffer { dtype, .. } => *dtype,
                _ => {
                    let _ = buffer;
                    return Err(Error::TT_ERROR_NOT_A_BUFFER);
                }
            };
            let cb = *plan.cb.get(&buffer).ok_or_else(|| {
                let _ = buffer;
                Error::TT_ERROR_BUFFER_NOT_PLACED
            })?;
            let ring = plan.cbs[cb as usize];
            if ring.pages != 1 {
                let _ = ring.pages;
                return Err(Error::TT_TODO_LOOP_ADDRESS);
            }
            let el = dtype.bytes();
            a.li(dest, ring.base);
            match *offset {
                Dim::Const(n) => {
                    let bytes = n
                        .checked_mul(el)
                        .ok_or(Error::TT_ERROR_PARAM_OFFSET_OVERFLOW)?;
                    if bytes != 0 {
                        a.word(rv::addi(dest, dest, bytes as i32));
                    }
                }
                Dim::Value(v) => {
                    let Some((off, rest)) = temps.split_first_mut() else {
                        return Err(Error::TT_TODO_LOOP_ADDRESS);
                    };
                    let off = *off;
                    eval_value(a, k, plan, param_index, loops, v, off, rest)?;
                    if el != 1 {
                        let Some((scale, _)) = rest.split_first_mut() else {
                            return Err(Error::TT_TODO_LOOP_ADDRESS);
                        };
                        a.li(*scale, el);
                        a.word(rv::mul(off, off, *scale));
                    }
                    a.word(rv::add(dest, dest, off));
                }
            }
            a.word(rv::lw(dest, dest, 0));
            Ok(())
        }
        _ => Err(Error::TT_TODO_LOOP_ADDRESS),
    }
}

fn collect(k: &Kernel, types: &Types, plan: &Plan, inbound: bool) -> Result<Vec<Item>, Error> {
    let param_index = param_indices(k);
    let mut items = Vec::new();
    for (at, op) in k.ops() {
        match *op {
            Op::Loop { value, bound } => {
                let count = match bound {
                    Dim::Const(n) => LoopCount::Const(n),
                    Dim::Value(v) => match k.const_i32(v) {
                        Some(n) if n >= 0 => LoopCount::Const(n as u32),
                        _ => LoopCount::Value(v),
                    },
                };
                items.push(Item::Loop {
                    index: value,
                    count,
                });
                continue;
            }
            Op::EndLoop => {
                items.push(Item::EndLoop);
                continue;
            }
            Op::CopyTo {
                dst,
                src,
                dst_offset,
                src_offset,
                shape: access,
            } => {
                let (param, buffer, param_offset, buffer_offset) =
                    match (types.of(dst).kind, types.of(src).kind) {
                        (crate::ir::ValueKind::Buffer, crate::ir::ValueKind::Param) => {
                            if !inbound {
                                continue;
                            }
                            (src, dst, src_offset, dst_offset)
                        }
                        (crate::ir::ValueKind::Param, crate::ir::ValueKind::Buffer) => {
                            if inbound {
                                continue;
                            }
                            (dst, src, dst_offset, src_offset)
                        }
                        _ => {
                            let _ = (at, dst, src);
                            return Err(Error::TT_ERROR_COPY_ENDS);
                        }
                    };
                let dtype = match k.at(buffer) {
                    Op::Buffer { dtype, .. } => *dtype,
                    _ => unreachable!("the type says it is a buffer"),
                };
                let cb = *plan.cb.get(&buffer).ok_or_else(|| {
                    let _ = (at, buffer);
                    Error::TT_ERROR_BUFFER_NOT_PLACED
                })?;
                // The index is in access units, so one index step is the unit's
                // bytes: a tile for `32,32`, a row for `1,32`.
                let unit = (access[0] * access[1])
                    .checked_mul(dtype.bytes())
                    .ok_or(Error::TT_ERROR_PARAM_OFFSET_OVERFLOW)?;
                // A vector index is one base per datum: a gather reads them out
                // of the page it copied, one datum per lane.
                let lanes = match param_offset {
                    Dim::Const(_) => 1,
                    Dim::Value(v) => crate::ir::lanes(types.of(v).shape),
                };
                let offset = match param_offset {
                    Dim::Const(n) => ParamOffset::Const(n.checked_mul(unit).ok_or_else(|| {
                        let _ = (at, n);
                        Error::TT_ERROR_PARAM_OFFSET_OVERFLOW
                    })?),
                    Dim::Value(value) => match k.const_i32(value) {
                        // A constant written as a value is still a literal.
                        Some(n) if n >= 0 => ParamOffset::Const(
                            (n as u32)
                                .checked_mul(unit)
                                .ok_or(Error::TT_ERROR_PARAM_OFFSET_OVERFLOW)?,
                        ),
                        _ if lanes > 1 => ParamOffset::Lanes {
                            load: value,
                            lanes,
                            dtype_bytes: unit,
                        },
                        _ => {
                            // The value depends on a loop index; the loop
                            // carries it in a register and the copy scales it
                            // by the unit size.
                            ParamOffset::Index {
                                value,
                                dtype_bytes: unit,
                            }
                        }
                    },
                };
                // The L1 side is the ring's next page, so the declared buffer
                // offset only has to fit when it is a literal; the ring pointer
                // is the address (`planner`): a page is one unit and the
                // pointers advance in program order.
                if let Dim::Const(n) = buffer_offset {
                    n.checked_mul(unit)
                        .ok_or(Error::TT_ERROR_BUFFER_OFFSET_OVERFLOW)?;
                }
                items.push(Item::Copy(Copy {
                    param: *param_index.get(&param).expect("a parameter has an index"),
                    offset,
                    cb,
                    bytes: unit,
                }));
            }
            _ => continue,
        }
    }
    Ok(items)
}

/// `t5` = this NIU's base, `t3` = this tile's NOC high word (the launch's third
/// argument pair, read from `a0`), `a6` = the CB interface table (argument 2).
///
/// The reader runs on RISCV B and the writer on RISCV NC, and the two data
/// movers use opposite NoCs -- tt-metal's `brisc_noc_id = 1 - ncrisc's`
/// (`program.cpp`). On one NoC they would share the initiator's registers: NC's
/// requests are programmed through B's (`ttsim-divergence.md` row 73), so a
/// reader and a writer in flight would overwrite each other's command.
fn prologue(a: &mut Asm, niu: Niu) {
    a.li(reg::T5, niu.base() as u32);
    a.word(rv::lw(reg::T3, reg::A0, 12));
    a.word(rv::lw(reg::A6, reg::A0, ARG_IFACE as i32 * 4));
}

/// One NoC request per lane: a copy whose parameter index names `lanes` bases.
///
/// The bases are data, not an expression: they sit in the page the mover copied
/// (a row of ids), so each lane reads its own datum out of L1 and names a span
/// of `dtype_bytes`. The L1 side is the ring's page, so the lane's span starts
/// one request past the one before it.
#[allow(clippy::too_many_arguments)]
fn emit_lanes(
    a: &mut Asm,
    k: &Kernel,
    plan: &Plan,
    at: usize,
    copy: &Copy,
    load: ValueId,
    lanes: u32,
    dtype_bytes: u32,
    depth: usize,
    inbound: bool,
) -> Result<(), Error> {
    let (base, el) = lane_page(k, plan, load, at)?;
    // A lane loop is one more nesting level: it takes the next counter pair, so
    // two program loops around a lane copy do not fit.
    let (counter, limit) = match depth {
        0 => (reg::A1, reg::A2),
        1 => (reg::A3, reg::A4),
        _ => return Err(Error::TT_TODO_LOOP_DEPTH),
    };
    a.li(counter, 0);
    a.li(limit, lanes);
    let top = format!("lane{at}");
    a.label(top.clone());
    // This lane's base: the id row's datum, in units of the access.
    a.li(reg::T1, base);
    a.li(reg::T0, el);
    a.word(rv::mul(reg::T0, counter, reg::T0));
    a.word(rv::add(reg::T1, reg::T1, reg::T0));
    a.word(rv::lw(reg::T1, reg::T1, 0));
    a.li(reg::T0, dtype_bytes);
    a.word(rv::mul(reg::A5, reg::T1, reg::T0));
    // And this lane's span of the ring's page.
    a.li(reg::T0, copy.bytes);
    a.word(rv::mul(reg::T0, counter, reg::T0));
    emit_copy(
        a,
        at,
        copy.param,
        CopyOffset::Reg(reg::A5),
        Some(reg::T0),
        copy.cb,
        copy.bytes,
        inbound,
    )?;
    a.word(rv::addi(counter, counter, 1));
    a.bltu_label(counter, limit, top);
    Ok(())
}

/// The L1 address of the vector a lane index reads and its datum width: the
/// page the `load` names, on a one-page ring. A deeper ring's pointer moves at
/// run time, so its page is refused rather than read from the wrong one.
fn lane_page(k: &Kernel, plan: &Plan, load: ValueId, at: usize) -> Result<(u32, u32), Error> {
    let Some(op) = k.op(load) else {
        let _ = (at, load);
        return Err(Error::TT_TODO_MULTI_LANE);
    };
    let Op::Load {
        buffer, ref offset, ..
    } = *op
    else {
        let _ = (at, load);
        return Err(Error::TT_TODO_MULTI_LANE);
    };
    let dtype = match k.at(buffer) {
        Op::Buffer { dtype, .. } => *dtype,
        _ => {
            let _ = buffer;
            return Err(Error::TT_ERROR_NOT_A_BUFFER);
        }
    };
    let cb = *plan.cb.get(&buffer).ok_or_else(|| {
        let _ = (at, buffer);
        Error::TT_ERROR_BUFFER_NOT_PLACED
    })?;
    let ring = plan.cbs[cb as usize];
    if ring.pages != 1 {
        let _ = (at, ring.pages);
        return Err(Error::TT_TODO_LOOP_ADDRESS);
    }
    let units = match *offset {
        Dim::Const(n) => n,
        Dim::Value(v) => k
            .const_i32(v)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| {
                let _ = (at, v);
                Error::TT_TODO_LOOP_ADDRESS
            })?,
    };
    let el = dtype.bytes();
    let bytes = units
        .checked_mul(el)
        .ok_or(Error::TT_ERROR_PARAM_OFFSET_OVERFLOW)?;
    let base = ring
        .base
        .checked_add(bytes)
        .ok_or(Error::TT_ERROR_PARAM_OFFSET_OVERFLOW)?;
    Ok((base, el))
}

fn emit_copy(
    a: &mut Asm,
    at: usize,
    param: u32,
    offset: CopyOffset,
    l1_offset: Option<u8>,
    cb: u32,
    bytes: u32,
    read: bool,
) -> Result<(), Error> {
    if bytes == 0 || bytes > tt_isa::noc::niu::MAX_REQUEST_BYTES {
        let _ = (at, bytes);
        return Err(Error::TT_ERROR_NOC_REQUEST_BYTES);
    }
    let (lo_arg, hi_arg) = param_arg(param);
    // Target: the parameter's range base plus the copy's byte offset.
    a.word(rv::lw(reg::T1, reg::A0, lo_arg));
    match offset {
        CopyOffset::Const(0) => {}
        CopyOffset::Const(n) => {
            a.li(reg::T4, n);
            a.word(rv::add(reg::T1, reg::T1, reg::T4));
        }
        CopyOffset::Reg(reg) => {
            a.word(rv::add(reg::T1, reg::T1, reg));
        }
    }
    a.word(rv::lw(reg::T2, reg::A0, hi_arg));
    // The local side is the ring's write pointer for a publish and its read
    // pointer for a drain: `reserve_back` made the page the writer's,
    // `wait_front` made it the reader's.
    let field = if read { rv::cb::WR_PTR } else { rv::cb::RD_PTR };
    a.word(rv::lw(
        reg::T4,
        reg::A6,
        (cb * rv::cb::STRIDE) as i32 + field,
    ));
    if let Some(reg) = l1_offset {
        a.word(rv::add(reg::T4, reg::T4, reg));
    }
    // `TARG_ADDR` is the request's source and `RET_ADDR` its destination: a
    // read has the parameter as source and L1 as destination, a write the
    // other way round.
    if read {
        store_reg(a, initiator::TARG_ADDR_LO, reg::T1);
        store_zero(a, initiator::TARG_ADDR_MID);
        store_reg(a, initiator::TARG_ADDR_HI, reg::T2);
        store_reg(a, initiator::RET_ADDR_LO, reg::T4);
        store_zero(a, initiator::RET_ADDR_MID);
        store_reg(a, initiator::RET_ADDR_HI, reg::T3);
    } else {
        store_reg(a, initiator::TARG_ADDR_LO, reg::T4);
        store_zero(a, initiator::TARG_ADDR_MID);
        store_reg(a, initiator::TARG_ADDR_HI, reg::T3);
        store_reg(a, initiator::RET_ADDR_LO, reg::T1);
        store_zero(a, initiator::RET_ADDR_MID);
        store_reg(a, initiator::RET_ADDR_HI, reg::T2);
    }
    a.li(reg::T4, (TXN.index() as u32) << 10);
    store_reg(a, initiator::PACKET_TAG, reg::T4);
    let ctrl = if read { CMD_RD } else { CMD_WR } | RESP_MARKED | STATIC_VC_1;
    a.li(reg::T4, ctrl);
    store_reg(a, initiator::CTRL, reg::T4);
    a.li(reg::T4, bytes);
    store_reg(a, initiator::AT_LEN_BE, reg::T4);
    store_zero(a, initiator::AT_DATA);
    // Trigger, then wait until the hardware takes the request.
    a.word(rv::addi(reg::T4, reg::ZERO, 1));
    store_reg(a, initiator::CMD_CTRL, reg::T4);
    let poll = format!("poll{at}");
    a.label(&poll);
    a.word(rv::lw(reg::T6, reg::T5, initiator::CMD_CTRL as i32));
    a.bne_label(reg::T6, reg::ZERO, poll.clone());
    // Then wait for the transaction to complete: read the trigger word back
    // first (`Counters.md`), then poll the ID's outstanding counter.
    a.word(rv::lw(reg::T6, reg::T5, initiator::CMD_CTRL as i32));
    let drain = format!("drain{at}");
    a.label(&drain);
    a.word(rv::lw(reg::T6, reg::T5, reqs_outstanding(TXN) as i32));
    a.bne_label(reg::T6, reg::ZERO, drain);
    Ok(())
}

fn store_reg(a: &mut Asm, at: u64, src: u8) {
    a.word(rv::sw(src, reg::T5, at as i32));
}

fn store_zero(a: &mut Asm, at: u64) {
    a.word(rv::sw(reg::ZERO, reg::T5, at as i32));
}

#[allow(dead_code)]
fn _unused(_: ValueId) {}
