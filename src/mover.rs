// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The boundary programs: DRAM <-> L1 over the NoC.
//!
//! Every `copy_to` between a parameter and a buffer becomes one NOC request:
//! the ten initiator registers are written in `tt_isa`'s order, `CMD_CTRL`
//! triggers it, and the program waits on that word and then on the transaction
//! ID's outstanding counter before the next copy. The parameter's address is its
//! launch argument plus the copy's datum offset; the local address is the
//! planned L1 buffer plus the same datum offset. Both sides are 64-byte aligned
//! and add the same byte offset, so a request's two addresses are congruent.
//!
//! `TARG_ADDR` is the request's source and `RET_ADDR` its destination, so a
//! read names the parameter as target and L1 as return, and a write the other
//! way round. Which copies belong to the inbound program and which to the
//! outbound one is decided here from the operands, not by the caller.

use crate::ir::{Dim, Error, Kernel, Op, Types, ValueId, unit_of};
use crate::planner::Plan;
use crate::riscv::{self as rv, Asm, reg};
use tt_isa::noc::niu::{
    CMD_RD, CMD_WR, Niu, RESP_MARKED, STATIC_VC_1, TxnId, initiator, reqs_outstanding,
};

/// Words of launch arguments before the per-parameter pairs: grid x, grid y,
/// reserved, this tile's NOC high word.
pub const ARG_HEAD: usize = 4;

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
    /// Byte offset into the parameter, and into the buffer.
    offset: u32,
    l1: u32,
    bytes: u32,
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
    if !k.ops.iter().any(direct) {
        return Ok(k.clone());
    }

    let mut out = Kernel {
        ops: Vec::with_capacity(k.ops.len()),
        names: Vec::with_capacity(k.names.len()),
    };
    // Old value id -> new value id. Definitions are emitted in the same order,
    // so a use only needs an already-emitted definition.
    let mut map: Vec<ValueId> = vec![u32::MAX; k.names.len()];
    let mut stage = 0usize;

    for op in &k.ops {
        match *op {
            Op::Load {
                value,
                buffer,
                offset,
            } if types.of(buffer).kind == crate::ir::ValueKind::Param => {
                let info = *types.of(buffer);
                let shape = stage_shape(info.shape, buffer)?;
                let staged = fresh(&mut out, &k.names, &mut stage);
                out.ops.push(Op::Buffer {
                    value: staged,
                    dtype: info.dtype,
                    shape,
                });
                out.ops.push(Op::CopyTo {
                    dst: staged,
                    src: map[buffer as usize],
                    offset: remap_dim(offset, &map),
                });
                let value = define(&mut out, k, value, &mut map);
                out.ops.push(Op::Load {
                    value,
                    buffer: staged,
                    offset: Dim::Const(0),
                });
            }
            Op::StoreTo {
                value,
                buffer,
                offset,
            } if types.of(buffer).kind == crate::ir::ValueKind::Param => {
                let info = *types.of(buffer);
                let shape = stage_shape(info.shape, buffer)?;
                let staged = fresh(&mut out, &k.names, &mut stage);
                out.ops.push(Op::Buffer {
                    value: staged,
                    dtype: info.dtype,
                    shape,
                });
                out.ops.push(Op::StoreTo {
                    value: map[value as usize],
                    buffer: staged,
                    offset: Dim::Const(0),
                });
                out.ops.push(Op::CopyTo {
                    dst: map[buffer as usize],
                    src: staged,
                    offset: remap_dim(offset, &map),
                });
            }
            _ => {
                let value = op.defined().map(|_| out.names.len() as ValueId);
                let new = remap(op, &map, value);
                if let Some(old) = op.defined() {
                    map[old as usize] = value.expect("a definition has a new id");
                    out.names.push(k.names[old as usize].clone());
                }
                out.ops.push(new);
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
        other => Err(Error(format!(
            "parameter %{param} is a {}; only a tile access can be staged yet",
            other.name()
        ))),
    }
}

/// Define one fresh value with a name no existing value has.
fn fresh(out: &mut Kernel, names: &[String], stage: &mut usize) -> ValueId {
    let value = out.names.len() as ValueId;
    loop {
        let name = format!("stage{}", *stage);
        *stage += 1;
        if !names.contains(&name) && !out.names.contains(&name) {
            out.names.push(name);
            return value;
        }
    }
}

/// Give an existing definition its new id and copy its name.
fn define(out: &mut Kernel, k: &Kernel, old: ValueId, map: &mut [ValueId]) -> ValueId {
    let value = out.names.len() as ValueId;
    map[old as usize] = value;
    out.names.push(k.names[old as usize].clone());
    value
}

fn remap_dim(dim: Dim, map: &[ValueId]) -> Dim {
    match dim {
        Dim::Const(n) => Dim::Const(n),
        Dim::Value(v) => Dim::Value(map[v as usize]),
    }
}

/// The same op with every value it reads renamed, and (for a definition) its
/// own new id installed.
fn remap(op: &Op, map: &[ValueId], value: Option<ValueId>) -> Op {
    let v = |x: ValueId| map[x as usize];
    let d = |x: Dim| remap_dim(x, map);
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
        Op::Grid { columns, rows } => Op::Grid {
            columns: d(columns),
            rows: d(rows),
        },
        Op::Loop { bound, .. } => Op::Loop {
            value: own(),
            bound: d(bound),
        },
        Op::EndLoop => Op::EndLoop,
        Op::CopyTo { dst, src, offset } => Op::CopyTo {
            dst: v(dst),
            src: v(src),
            offset: d(offset),
        },
        Op::Load { buffer, offset, .. } => Op::Load {
            value: own(),
            buffer: v(buffer),
            offset: d(offset),
        },
        Op::StoreTo {
            value: src,
            buffer,
            offset,
        } => Op::StoreTo {
            value: v(src),
            buffer: v(buffer),
            offset: d(offset),
        },
        Op::Cast { src, dtype, .. } => Op::Cast {
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
        Op::Matmul { a, b, .. } => Op::Matmul {
            value: own(),
            a: v(a),
            b: v(b),
        },
    }
}

fn program(k: &Kernel, types: &Types, plan: &Plan, inbound: bool) -> Result<Vec<u32>, Error> {
    let copies = collect(k, types, plan, inbound)?;
    if copies.is_empty() {
        return Ok(Vec::new());
    }
    let mut a = Asm::new();
    prologue(&mut a);
    for (at, copy) in copies.iter().enumerate() {
        emit_copy(
            &mut a,
            at,
            copy.param,
            copy.offset,
            copy.l1,
            copy.bytes,
            inbound,
        )?;
    }
    a.word(rv::ret());
    a.finish().map_err(Error)
}

fn collect(k: &Kernel, types: &Types, plan: &Plan, inbound: bool) -> Result<Vec<Copy>, Error> {
    let mut param_index: Vec<Option<u32>> = vec![None; types.values.len()];
    let mut def: Vec<Option<usize>> = vec![None; types.values.len()];
    let mut next = 0u32;
    for (at, op) in k.ops.iter().enumerate() {
        if let Some(value) = op.defined() {
            def[value as usize] = Some(at);
        }
        if let Op::Param { value, .. } = *op {
            param_index[value as usize] = Some(next);
            next += 1;
        }
    }
    let mut copies = Vec::new();
    for (at, op) in k.ops.iter().enumerate() {
        let Op::CopyTo { dst, src, offset } = *op else {
            continue;
        };
        let (param, buffer) = match (types.of(dst).kind, types.of(src).kind) {
            (crate::ir::ValueKind::Buffer, crate::ir::ValueKind::Param) => {
                if !inbound {
                    continue;
                }
                (src, dst)
            }
            (crate::ir::ValueKind::Param, crate::ir::ValueKind::Buffer) => {
                if inbound {
                    continue;
                }
                (dst, src)
            }
            _ => {
                return Err(Error(format!(
                    "op {at}: a copy moves between a parameter and a buffer"
                )));
            }
        };
        let Op::Buffer { shape, dtype, .. } = k.ops[def[buffer as usize].expect("defined")] else {
            unreachable!("the type says it is a buffer");
        };
        let (_, _, unit_bytes) = unit_of(shape, dtype)?;
        let offset = match offset {
            Dim::Const(n) => n,
            Dim::Value(v) => {
                return Err(Error(format!(
                    "op {at}: the offset %{v} is symbolic; only a literal offset is lowered yet"
                )));
            }
        };
        let base = plan.buf[buffer as usize]
            .ok_or_else(|| Error(format!("op {at}: buffer %{buffer} was not placed")))?;
        let byte_offset = offset
            .checked_mul(dtype.bytes())
            .ok_or_else(|| Error(format!("op {at}: the offset overflows")))?;
        copies.push(Copy {
            param: param_index[param as usize].expect("a parameter has an index"),
            offset: byte_offset,
            l1: base
                .checked_add(byte_offset)
                .ok_or_else(|| Error(format!("op {at}: the buffer address overflows")))?,
            bytes: unit_bytes,
        });
    }
    Ok(copies)
}

/// `t5` = this NIU's base, `t3` = this tile's NOC high word (the launch's third
/// argument pair, read from `a0`).
fn prologue(a: &mut Asm) {
    a.li(reg::T5, Niu::Noc0.base() as u32);
    a.word(rv::lw(reg::T3, reg::A0, 12));
}

fn emit_copy(
    a: &mut Asm,
    at: usize,
    param: u32,
    offset: u32,
    l1: u32,
    bytes: u32,
    read: bool,
) -> Result<(), Error> {
    if bytes == 0 || bytes > tt_isa::noc::niu::MAX_REQUEST_BYTES {
        return Err(Error(format!(
            "copy {at}: {bytes} bytes is not one NOC request (1..={})",
            tt_isa::noc::niu::MAX_REQUEST_BYTES
        )));
    }
    let (lo_arg, hi_arg) = param_arg(param);
    // Target: the parameter's range base plus the copy's byte offset.
    a.word(rv::lw(reg::T1, reg::A0, lo_arg));
    if offset != 0 {
        a.li(reg::T4, offset);
        a.word(rv::add(reg::T1, reg::T1, reg::T4));
    }
    a.word(rv::lw(reg::T2, reg::A0, hi_arg));
    // `TARG_ADDR` is the request's source and `RET_ADDR` its destination: a
    // read has the parameter as source and L1 as destination, a write the
    // other way round.
    a.li(reg::T4, l1);
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
