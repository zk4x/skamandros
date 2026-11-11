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

use crate::ir::{Dim, Error, Kernel, Op, Types, ValueId};
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
    /// Byte offset into the parameter.
    offset: u32,
    /// The CB whose ring holds the L1 side of the copy.
    cb: u32,
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
                    dst_offset: Dim::Const(0),
                    src_offset: remap_dim(offset, &map),
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
                    dst_offset: remap_dim(offset, &map),
                    src_offset: Dim::Const(0),
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

/// `tt_add_movers`'s invariant: a circular buffer has one producer and one
/// consumer. The reader publishes and the unpacker consumes an input buffer; the
/// packer publishes and the writer consumes an output buffer. A buffer used in
/// both directions would share one counter pair and one page ring, which the
/// hardware's CB cannot do, so it is refused rather than guessed at.
pub fn check_roles(k: &Kernel, types: &Types) -> Result<(), Error> {
    use crate::ir::ValueKind;
    let mut producer: Vec<Option<&str>> = vec![None; types.values.len()];
    let mut consumer: Vec<Option<&str>> = vec![None; types.values.len()];
    for (at, op) in k.ops.iter().enumerate() {
        let (buffer, role, publishes) = match *op {
            Op::CopyTo { dst, src, .. } => match (types.of(dst).kind, types.of(src).kind) {
                (ValueKind::Buffer, ValueKind::Param) => (dst, "the reader", true),
                (ValueKind::Param, ValueKind::Buffer) => (src, "the writer", false),
                _ => continue,
            },
            Op::Load { buffer, .. } => (buffer, "the unpacker", false),
            Op::StoreTo { buffer, .. } => (buffer, "the packer", true),
            _ => continue,
        };
        let slot = if publishes {
            &mut producer
        } else {
            &mut consumer
        };
        match slot[buffer as usize] {
            None => slot[buffer as usize] = Some(role),
            Some(previous) if previous == role => {}
            Some(previous) => {
                let _ = (at, previous, role, publishes);
                return Err(Error::TT_TODO_MULTI_PRODUCER_OR_CONSUMER);
            }
        }
    }
    for (value, info) in types.values.iter().enumerate() {
        if info.kind != ValueKind::Buffer {
            continue;
        }
        let pair = (producer[value], consumer[value]);
        match pair {
            (Some("the reader"), Some("the unpacker"))
            | (Some("the packer"), Some("the writer"))
            // A pure transfer (DRAM to DRAM through one page) has no compute:
            // the reader publishes and the writer drains the same buffer.
            | (Some("the reader"), Some("the writer")) => {}
            _ => {
                let _ = (value, pair);
                return Err(Error::TT_TODO_CIRCULAR_BUFFER_ROLE);
            }
        }
    }
    Ok(())
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
        Op::CopyTo {
            dst,
            src,
            dst_offset,
            src_offset,
        } => Op::CopyTo {
            dst: v(dst),
            src: v(src),
            dst_offset: d(dst_offset),
            src_offset: d(src_offset),
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
    // The sync scratch leaves `t3` (this tile's high word), `t5` (the NOC
    // initiator) and `a6` (the interface) alone.
    let sync3 = [reg::T0, reg::T1, reg::T2];
    let sync4 = [reg::T0, reg::T1, reg::T2, reg::T6];
    for (at, copy) in copies.iter().enumerate() {
        let cb = copy.cb;
        if inbound {
            rv::cb::reserve_back(&mut a, reg::A6, cb, 1, sync3);
        } else {
            rv::cb::wait_front(&mut a, reg::A6, cb, 1, sync3);
        }
        emit_copy(&mut a, at, copy.param, copy.offset, cb, copy.bytes, inbound)?;
        if inbound {
            rv::cb::push_back(&mut a, reg::A6, cb, 1, sync4);
        } else {
            rv::cb::pop_front(&mut a, reg::A6, cb, 1, sync4);
        }
    }
    a.word(rv::ret());
    a.finish().map_err(|_| Error::TT_ERROR_MOVER_PROGRAM)
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
        let Op::CopyTo {
            dst,
            src,
            dst_offset,
            src_offset,
        } = *op
        else {
            continue;
        };
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
        let Op::Buffer { dtype, .. } = k.ops[def[buffer as usize].expect("defined")] else {
            unreachable!("the type says it is a buffer");
        };
        let literal = |offset: Dim, what: &str| match offset {
            Dim::Const(n) => Ok(n),
            Dim::Value(v) => {
                let _ = (at, what, v);
                Err(Error::TT_TODO_SYMBOLIC_OFFSET)
            }
        };
        let param_offset = literal(param_offset, "parameter")?;
        let buffer_offset = literal(buffer_offset, "buffer")?;
        let cb = plan.cb[buffer as usize].ok_or_else(|| {
            let _ = (at, buffer);
            Error::TT_ERROR_BUFFER_NOT_PLACED
        })?;
        let param_bytes = param_offset.checked_mul(dtype.bytes()).ok_or_else(|| {
            let _ = (at, param_offset);
            Error::TT_ERROR_PARAM_OFFSET_OVERFLOW
        })?;
        let buffer_bytes = buffer_offset.checked_mul(dtype.bytes()).ok_or_else(|| {
            let _ = (at, buffer_offset);
            Error::TT_ERROR_BUFFER_OFFSET_OVERFLOW
        })?;
        // The ring position, not the literal offset, is the L1 address: a page is
        // one unit and the pointers advance in program order. The offset is still
        // checked (it must be a literal that fits) but it names the parameter's
        // side only.
        let _ = buffer_bytes;
        copies.push(Copy {
            param: param_index[param as usize].expect("a parameter has an index"),
            offset: param_bytes,
            cb,
            bytes: plan.cbs[cb as usize].page,
        });
    }
    Ok(copies)
}

/// `t5` = this NIU's base, `t3` = this tile's NOC high word (the launch's third
/// argument pair, read from `a0`), `a6` = the CB interface table (argument 2).
fn prologue(a: &mut Asm) {
    a.li(reg::T5, Niu::Noc0.base() as u32);
    a.word(rv::lw(reg::T3, reg::A0, 12));
    a.word(rv::lw(reg::A6, reg::A0, ARG_IFACE as i32 * 4));
}

fn emit_copy(
    a: &mut Asm,
    at: usize,
    param: u32,
    offset: u32,
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
    if offset != 0 {
        a.li(reg::T4, offset);
        a.word(rv::add(reg::T1, reg::T1, reg::T4));
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
