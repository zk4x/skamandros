// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The semantic kernel IR: one unified SSA program, machine programs out.
//!
//! What the caller writes is what the kernel *means*. There are no reader,
//! compute and writer segments in what it writes: a DRAM parameter, an L1
//! buffer, a register value and the operations on them are all named in one
//! list, and the compiler classifies each instruction by the operands it
//! touches. There is no `unpack`/`pack` and no buffer layout: how L1 becomes a
//! register and how a register becomes L1 is the compiler's business.
//!
//! There is one id space: an op's [`OpId`] *is* the value it defines, so a
//! shape term, an index, a loop bound and an operand are all `OpId`s. A shape
//! whose extent is not a value is either a [`Dim`] on the op that owns it (a
//! fixed 2-D shape: an accumulator, a ring page, a copy's unit) or an
//! `Op::Shape` of term ops.
//!
//! dtypes and shapes are not a side table: a leaf op carries its own, and every
//! other op's is [`Kernel::dtype`]/[`Kernel::shape`], recomputed on demand.
//!
//! The text form is the only program form, and [`VERSION`] its version line.

use core::fmt;
use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::slab::{Slab, SlabId};
use crate::status::ttStatus;

/// One operation of the kernel's list.
///
/// The list is a slab with a doubly linked order (zyx's `OpLinked`): ids never
/// move, removing is O(1), and a pass inserts by id instead of by position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpLinked {
    pub prev: OpId,
    pub next: OpId,
    pub op: Op,
}

/// Identifier of an op, and of the value it defines: there is one id space
/// (zyx's model), so a value *is* its defining op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpId(pub(crate) u32);

impl OpId {
    /// The sentinel for "no op": no previous/next link, no index.
    pub const NULL: Self = Self(u32::MAX);
    /// Whether this is [`OpId::NULL`].
    pub const fn is_null(self) -> bool {
        self.0 == u32::MAX
    }
}

impl fmt::Display for OpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl From<usize> for OpId {
    fn from(value: usize) -> Self {
        OpId(value as u32)
    }
}

impl From<OpId> for usize {
    fn from(value: OpId) -> Self {
        value.0 as usize
    }
}

impl SlabId for OpId {
    const ZERO: Self = Self(0);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

/// The fixed dtype of every index, shape and bracket term.
pub const INDEX: DType = DType::I32;

/// Element type.
///
/// `Bool` is what a comparison and a logical op produce, and what `sel` takes.
/// A parameter or a buffer may name it in the format, but neither is lowered
/// yet (the verifier says so loudly).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
    I32,
    Bool,
}

impl DType {
    /// Bytes of one datum.
    pub const fn bytes(self) -> u32 {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::Bf16 | DType::Bool => 2,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::Bf16)
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::Bf16 => "bf16",
            DType::I32 => "i32",
            DType::Bool => "bool",
        }
    }
}

/// A concrete extent: a shape's dimension, a ring's page count, a copy's unit.
///
/// A *symbolic* extent is never a `Dim`: it is an [`OpId`] naming the value that
/// computes it, exactly as zyx models a shape term. `Dim` is only the resolved
/// number, which is what fixed shapes (an accumulator, a ring page, a copy's
/// unit) hold.
pub type Dim = u32;

/// The shape of a register value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegShape {
    /// One datum.
    Scalar,
    /// 32 datums in a column.
    Column,
    /// 32 datums in a row.
    Row,
    /// A 32x32 tile.
    Tile,
}

impl RegShape {
    /// Datums in one value of this shape.
    pub const fn datums(self) -> u32 {
        match self {
            RegShape::Scalar => 1,
            RegShape::Column | RegShape::Row => 32,
            RegShape::Tile => 32 * 32,
        }
    }
}

/// Which axis a reduce folds, and where its result lands: tt-metal's
/// `ReduceDim`, and zyx's `TileDim` under the same names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceDim {
    /// One value per row, in the result tile's column 0.
    Row,
    /// One value per column, in the result tile's row 0.
    Col,
    /// One value, in the result tile's face 0 `[0,0]`.
    Scalar,
}

impl ReduceDim {
    /// The fold an accumulator's shape asks for. An `acc` is one `Dst` tile
    /// region whatever its shape; the shape says which part is the result, so
    /// it is also the fold: a `1,32` row is one value per column, a `32,1`
    /// column one per row, a `1,1` datum the whole tile. `32,32` names no
    /// fold — it is the matmul accumulator, and a reduce over it is refused.
    pub fn of(shape: RegShape) -> Option<ReduceDim> {
        Some(match shape {
            RegShape::Row => ReduceDim::Col,
            RegShape::Column => ReduceDim::Row,
            RegShape::Scalar => ReduceDim::Scalar,
            RegShape::Tile => return None,
        })
    }
}

/// What a value names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    /// A DRAM parameter. Mutable, not SSA.
    Param,
    /// An L1 buffer. Mutable, not SSA.
    Buffer,
    /// A literal.
    Const,
    /// A loop's induction value.
    Induction,
    /// A register value.
    Register,
    /// A named accumulator: mutable, one `Dst` tile region.
    Acc,
    /// A shape: its members are extents.
    Shape,
}

/// Everything the compiler knows about one value, recomputed on demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueInfo {
    pub kind: ValueKind,
    pub dtype: DType,
    pub shape: RegShape,
}

/// The binary operations the unified program has, the same set as zyx's `BOp`.
///
/// Comparisons produce a `bool`; `Or`/`And` are logical (bool operands and
/// result); `BitOr`/`BitAnd`/`BitXor`/`BitShiftLeft`/`BitShiftRight`/`Mod` are
/// integer (`i32`); the rest operate on two values of the same dtype.
///
/// A shift moves the two's-complement bit pattern, filling with zeros:
/// `BitShiftRight` is a *logical* shift even on a negative value, unlike
/// Rust's `i32 >>`, which sign-extends. The tile path (`SFPSHFT`), the scalar
/// path (`srl`/`sll`) and the constant fold all agree on that, and the amount
/// is the hardware's low five bits rather than Rust's panic on `>= 32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Cmplt,
    Cmpgt,
    Cmpge,
    Max,
    Min,
    Or,
    And,
    BitXor,
    BitOr,
    BitAnd,
    BitShiftLeft,
    BitShiftRight,
    NotEq,
    Eq,
}

impl BOp {
    fn from_name(name: &str) -> Option<BOp> {
        Some(match name {
            "add" => BOp::Add,
            "sub" => BOp::Sub,
            "mul" => BOp::Mul,
            "div" => BOp::Div,
            "pow" => BOp::Pow,
            "mod" => BOp::Mod,
            "cmplt" => BOp::Cmplt,
            "cmpgt" => BOp::Cmpgt,
            "cmpge" => BOp::Cmpge,
            "max" => BOp::Max,
            "min" => BOp::Min,
            "or" => BOp::Or,
            "and" => BOp::And,
            "bitxor" => BOp::BitXor,
            "bitor" => BOp::BitOr,
            "bitand" => BOp::BitAnd,
            "bitshiftleft" => BOp::BitShiftLeft,
            "bitshiftright" => BOp::BitShiftRight,
            "noteq" => BOp::NotEq,
            "eq" => BOp::Eq,
            _ => return None,
        })
    }

    /// A comparison: the result is a `bool`.
    pub const fn returns_bool(self) -> bool {
        matches!(
            self,
            BOp::Cmplt | BOp::Cmpgt | BOp::Cmpge | BOp::NotEq | BOp::Eq
        )
    }

    /// The logical ops take and return a `bool`.
    pub const fn is_logical(self) -> bool {
        matches!(self, BOp::Or | BOp::And)
    }
}

/// A unary register op, the same set as zyx's `UOp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UOp {
    Neg,
    Not,
    BitNot,
    Exp2,
    Log2,
    Reciprocal,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Floor,
    Trunc,
    Abs,
}

impl UOp {
    fn from_name(name: &str) -> Option<UOp> {
        Some(match name {
            "neg" => UOp::Neg,
            "not" => UOp::Not,
            "bitnot" => UOp::BitNot,
            "exp2" => UOp::Exp2,
            "log2" => UOp::Log2,
            "reciprocal" => UOp::Reciprocal,
            "sqrt" => UOp::Sqrt,
            "rsqrt" => UOp::Rsqrt,
            "sin" => UOp::Sin,
            "cos" => UOp::Cos,
            "floor" => UOp::Floor,
            "trunc" => UOp::Trunc,
            "abs" => UOp::Abs,
            _ => return None,
        })
    }
}

/// One instruction of the unified program.
///
/// A producing op's result is its own [`OpId`], so no variant carries a
/// `value` field. `EndLoop`, `EndReader`, `EndCompute`, `CopyTo`, `StoreTo`,
/// `Matmul`, `Sum`, `Max`, `Grid`, `Wait`, `Reserve`, `Push` and `Pop` produce
/// nothing (an accumulator is written in place).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// A literal. `bits` holds the two's-complement `i32` or the `f32` bits.
    /// Its shape is the empty one: a literal is a scalar.
    Const { dtype: DType, bits: u32 },
    /// A shape: its members are the term values, in dimension order. A term is
    /// any value of the index dtype, so a shape may be symbolic; an extent that
    /// is fixed (a tile, a ring page) is a `[Dim; 2]` on the op that owns it.
    Shape(Box<[OpId]>),
    /// A tensor parameter in DRAM, with its logical shape.
    /// A tensor parameter: `shape` is its element count, which the caller's
    /// DRAM buffer holds. `interleaved` is the buffer's layout: its pages are
    /// striped across the device's DRAM channels (tt-metal's interleaved
    /// buffer, `InterleavedBuffer`), instead of lying in one channel.
    Param {
        dtype: DType,
        shape: OpId,
        interleaved: bool,
    },
    /// A scalar parameter: one word in DRAM, the value a launch argument
    /// carries (the shape `0` in the text form). Its shape is the empty one.
    Variable { dtype: DType },
    /// A CB ring in L1: a page shape and the number of pages. Its CB id is its
    /// own `OpId`; the machine side names the ring by [`crate::mir::Cb`].
    Cb {
        dtype: DType,
        /// One page's shape: a tile, a row or a column.
        page: [Dim; 2],
        /// Pages in the ring.
        len: Dim,
    },
    /// A named accumulator: one `Dst` region that `matmul`, `sum`/`max` and
    /// `store_to` write, read back by `store_to` into a buffer. It is mutable —
    /// the whole point of it is that a K loop adds into the same rows, and a
    /// reduce's fold accumulates — and it starts unspecified: a program that
    /// needs zeros stores them (`store_to %acc from %zero`). Its shape is which
    /// part of the region the program means, and for a reduce it is the fold:
    /// `1,32`, `32,1` or `1,1`, while `32,32` is `matmul`'s.
    Acc { dtype: DType, shape: [Dim; 2] },
    /// The launch grid, as two terms: Tensix cores are a 2-D grid.
    Grid { x: OpId, y: OpId },
    /// `%i = loop %bound`.
    Loop { bound: OpId },
    /// `endloop`.
    EndLoop,
    /// The end of the reader's segment: the ops before it run on the inbound
    /// mover, and those after the last marker on the writer.
    EndReader,
    /// The end of the compute segments (unpack, math, pack).
    EndCompute,
    /// `%x = grid.x` / `%y = grid.y`: this core's logical coordinate on axis 0
    /// (x) or 1 (y), written by the host before the launch.
    GridCoord { axis: u8 },
    /// DRAM into L1, or L1 into DRAM: exactly one side is a parameter, the other
    /// a buffer. Each side has its own datum offset, because a circular buffer
    /// names its page by the ring pointer while the parameter names the access's
    /// own index (zyx's `Op::Copy` carries `src_idx`/`dst_idx`). `shape` is the
    /// unit one base moves: `32,32` a tile, `1,32` a row, `32,1` a column, `1,1`
    /// one datum. A copy's unit has to be a whole NoC request, so it is a tile
    /// or a row and nothing narrower. A vector index on the parameter side names
    /// one base per datum, and the lanes fill the buffer's page: a `1,32` copy
    /// with 32 bases is 32 rows, one `32,32` page.
    Copy {
        dst: OpId,
        src: OpId,
        dst_offset: OpId,
        src_offset: OpId,
        shape: [Dim; 2],
    },
    /// L1 buffer into a register value. `shape` is the unit the index's base
    /// names; the result has that shape.
    Load {
        buffer: OpId,
        offset: OpId,
        shape: [Dim; 2],
    },
    /// A register value into an L1 buffer.
    Store {
        src: OpId,
        buffer: OpId,
        offset: OpId,
        shape: [Dim; 2],
    },
    /// The only conversion.
    Cast { dtype: DType, src: OpId },
    /// The same bits under another name, between dtypes of equal width.
    Bitcast { dtype: DType, src: OpId },
    /// A 32x32 tile's transpose: `%y = transpose %x`. Datum `(r, c)` of the
    /// result is datum `(c, r)` of the source, in the same page layout, and the
    /// dtype is unchanged.
    Transpose { src: OpId },
    /// One unary register op.
    Unary { op: UOp, a: OpId },
    /// One binary register op, with implicit broadcast.
    Binary { op: BOp, a: OpId, b: OpId },
    /// Branchless select: `cond ? if_true : if_false`. `cond` is a `bool`; the
    /// two values have the same dtype and the result has their shape.
    Select {
        cond: OpId,
        if_true: OpId,
        if_false: OpId,
    },
    /// `a x b` accumulating into a named accumulator. The operands are
    /// `SrcA`/`SrcB` tiles, multiplied and summed over the K faces into `acc`,
    /// which the op adds to rather than replaces: a K loop is one `matmul` under
    /// one `loop`, and two products in a row are two `matmul`s on one `acc`.
    /// The accumulator's dtype is the `acc`'s, so this op names no dtype.
    Matmul { acc: OpId, a: OpId, b: OpId },
    /// `sum %acc %x %s`: fold `x` with an add, into `acc`, scaled by the factors
    /// in `s` (tt-metal's `PoolType::SUM`).
    ///
    /// The accumulator's shape picks the axis and the result's home: `1,32`
    /// folds the rows of each column into row 0 (tt-metal's `ReduceDim::Col`),
    /// `32,1` folds the columns of each row into column 0 (`ReduceDim::Row`,
    /// zyx's `TileDim` under the same names), and `1,1` folds the whole tile
    /// into face 0's `[0,0]`. `x` and `s` are `load` pages, because the folding
    /// happens on the Src path: `Row` is the Matrix Unit's `x @ s` (tt-metal's
    /// split — `SUM` with `ReduceDim::ROW` is a matmul against the scaler's
    /// column of ones), and `Col`/`Scalar` are `GAPOOL`, whose scaler is its
    /// `SrcB` operand. The scaler's layout follows the hardware: column 0 for
    /// `Row`, row 0 otherwise.
    Sum { acc: OpId, x: OpId, s: OpId },
    /// `max %acc %x`: fold `x` with a maximum into `acc` (tt-metal's
    /// `PoolType::MAX`, `GMPOOL`). The accumulator's shape picks the axis, as
    /// it does for `sum`.
    ///
    /// No scaler: `GMPOOL` applies only the exponent of a non-unity scaler
    /// (`exp2(floor(log2(|s|)))`), so tt-metal always reduces `MAX` with 1.0
    /// and scales the packed result afterwards with an SFPU multiply. The
    /// compiler supplies the 1.0 tile this op's `SrcB` still needs.
    Max { acc: OpId, x: OpId },
    /// `wait %cb`: the unpacker waits for a published page of the ring.
    Wait { cb: OpId, n: Dim },
    /// `reserve %cb`: the packer takes a free page of the ring.
    Reserve { cb: OpId, n: Dim },
    /// `push %cb`: the page just filled is published.
    Push { cb: OpId, n: Dim },
    /// `pop %cb`: the page just consumed is released.
    Pop { cb: OpId, n: Dim },
    /// The math thread takes `Dst` back from the packer.
    MathLock,
    /// The math thread hands its `Dst` section to the packer.
    MathUnlock,
    /// The packer waits for the math thread's `Dst` section.
    PackLock,
    /// The packer hands `Dst` back to the math thread.
    PackUnlock,
}

impl Op {
    /// The `idx`-th operand, in the order [`Op::parameters`] walks them.
    pub fn param_at(&self, idx: usize) -> Option<OpId> {
        match *self {
            Op::Const { .. }
            | Op::Variable { .. }
            | Op::Cb { .. }
            | Op::Acc { .. }
            | Op::EndLoop
            | Op::EndReader
            | Op::EndCompute
            | Op::MathLock
            | Op::MathUnlock
            | Op::PackLock
            | Op::PackUnlock
            | Op::GridCoord { .. } => None,
            Op::Shape(ref ops) => ops.get(idx).copied(),
            Op::Param { shape, .. } => [shape].get(idx).copied(),
            Op::Grid { x, y } => [x, y].get(idx).copied(),
            Op::Loop { bound } => [bound].get(idx).copied(),
            Op::Copy {
                dst,
                src,
                dst_offset,
                src_offset,
                ..
            } => [dst, src, dst_offset, src_offset].get(idx).copied(),
            Op::Load { buffer, offset, .. } => [buffer, offset].get(idx).copied(),
            Op::Store {
                src,
                buffer,
                offset,
                ..
            } => [src, buffer, offset].get(idx).copied(),
            Op::Cast { src, .. } | Op::Bitcast { src, .. } | Op::Unary { a: src, .. } => {
                [src].get(idx).copied()
            }
            Op::Transpose { src } => [src].get(idx).copied(),
            Op::Binary { a, b, .. } => [a, b].get(idx).copied(),
            Op::Select {
                cond,
                if_true,
                if_false,
            } => [cond, if_true, if_false].get(idx).copied(),
            Op::Matmul { acc, a, b } => [acc, a, b].get(idx).copied(),
            Op::Sum { acc, x, s } => [acc, x, s].get(idx).copied(),
            Op::Max { acc, x } => [acc, x].get(idx).copied(),
            Op::Wait { cb, .. }
            | Op::Reserve { cb, .. }
            | Op::Push { cb, .. }
            | Op::Pop { cb, .. } => [cb].get(idx).copied(),
        }
    }

    /// The operands this op reads, in program order.
    pub fn parameters(&self) -> Params<'_> {
        Params { op: self, idx: 0 }
    }

    /// The mutable counterpart of [`Op::parameters`], for a pass that rewrites
    /// operands (the allocator's reuse step).
    pub fn parameters_mut(&mut self) -> ParamsMut<'_> {
        match self {
            Op::Const { .. }
            | Op::Variable { .. }
            | Op::Cb { .. }
            | Op::Acc { .. }
            | Op::EndLoop
            | Op::EndReader
            | Op::EndCompute
            | Op::MathLock
            | Op::MathUnlock
            | Op::PackLock
            | Op::PackUnlock
            | Op::GridCoord { .. } => ParamsMut::Empty,
            Op::Shape(ops) => ParamsMut::Slice(ops.iter_mut()),
            Op::Param { shape, .. } => ParamsMut::One(Some(shape)),
            Op::Grid { x, y } => ParamsMut::Two(Some(x), Some(y)),
            Op::Loop { bound } => ParamsMut::One(Some(bound)),
            Op::Copy {
                dst,
                src,
                dst_offset,
                src_offset,
                ..
            } => ParamsMut::Four([dst, src, dst_offset, src_offset].into_iter()),
            Op::Load { buffer, offset, .. } => ParamsMut::Two(Some(buffer), Some(offset)),
            Op::Store {
                src,
                buffer,
                offset,
                ..
            } => ParamsMut::Three(Some(src), Some(buffer), Some(offset)),
            Op::Cast { src, .. } | Op::Bitcast { src, .. } | Op::Unary { a: src, .. } => {
                ParamsMut::One(Some(src))
            }
            Op::Transpose { src } => ParamsMut::One(Some(src)),
            Op::Binary { a, b, .. } => ParamsMut::Two(Some(a), Some(b)),
            Op::Select {
                cond,
                if_true,
                if_false,
            } => ParamsMut::Three(Some(cond), Some(if_true), Some(if_false)),
            Op::Matmul { acc, a, b } => ParamsMut::Three(Some(acc), Some(a), Some(b)),
            Op::Sum { acc, x, s } => ParamsMut::Three(Some(acc), Some(x), Some(s)),
            Op::Max { acc, x } => ParamsMut::Two(Some(acc), Some(x)),
            Op::Wait { cb, .. }
            | Op::Reserve { cb, .. }
            | Op::Push { cb, .. }
            | Op::Pop { cb, .. } => ParamsMut::One(Some(cb)),
        }
    }
}

/// [`Op::parameters`]: an index walk over the operand list.
pub struct Params<'a> {
    op: &'a Op,
    idx: usize,
}

impl Iterator for Params<'_> {
    type Item = OpId;

    fn next(&mut self) -> Option<OpId> {
        let param = self.op.param_at(self.idx)?;
        self.idx += 1;
        Some(param)
    }
}

/// [`Op::parameters_mut`]: the mutable operand lists. Each arm splits the
/// `&mut Op` once into disjoint fields, so no operand is borrowable twice.
pub enum ParamsMut<'a> {
    Empty,
    One(Option<&'a mut OpId>),
    Two(Option<&'a mut OpId>, Option<&'a mut OpId>),
    Three(
        Option<&'a mut OpId>,
        Option<&'a mut OpId>,
        Option<&'a mut OpId>,
    ),
    Four(std::array::IntoIter<&'a mut OpId, 4>),
    Slice(std::slice::IterMut<'a, OpId>),
}

impl<'a> Iterator for ParamsMut<'a> {
    type Item = &'a mut OpId;

    fn next(&mut self) -> Option<&'a mut OpId> {
        match self {
            ParamsMut::Empty => None,
            ParamsMut::One(a) => a.take(),
            ParamsMut::Two(a, b) => a.take().or_else(|| b.take()),
            ParamsMut::Three(a, b, c) => a.take().or_else(|| b.take()).or_else(|| c.take()),
            ParamsMut::Four(it) => it.next(),
            ParamsMut::Slice(it) => it.next(),
        }
    }
}

/// A parsed kernel. [`Kernel::verify`] is the only thing that checks it; a
/// program that fails it must never reach the compiler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    /// The op slab: ids never move, so a pass can hold one across an edit.
    pub ops: Slab<OpId, OpLinked>,
    /// Head of the list, in program order.
    pub head: OpId,
    /// Tail of the list.
    pub tail: OpId,
    /// The name of every value that has one, by the id of its defining op.
    pub names: BTreeMap<OpId, String>,
}

impl Default for Kernel {
    fn default() -> Self {
        Self {
            ops: Slab::new(),
            head: OpId::NULL,
            tail: OpId::NULL,
            names: BTreeMap::new(),
        }
    }
}

impl Kernel {
    /// The op `id`, and the only way to read one: a stale id is a panic, not a
    /// neighbouring op.
    pub fn at(&self, id: OpId) -> &Op {
        &self.ops[id].op
    }

    /// The op `id`, if it is live.
    pub fn op(&self, id: OpId) -> Option<&Op> {
        self.ops.get(id).map(|linked| &linked.op)
    }

    /// One op's program order, as the slab's `(id, op)` pairs. The order is the
    /// linked list, not the slab's storage order.
    pub fn ops(&self) -> impl Iterator<Item = (OpId, &Op)> + '_ {
        let mut id = self.head;
        std::iter::from_fn(move || {
            if id.is_null() {
                return None;
            }
            let here = id;
            id = self.ops[here].next;
            Some((here, &self.ops[here].op))
        })
    }

    /// The op after `id` in program order.
    #[allow(dead_code)]
    pub fn next_op(&self, id: OpId) -> OpId {
        self.ops[id].next
    }

    /// The op before `id` in program order.
    #[allow(dead_code)]
    pub fn prev_op(&self, id: OpId) -> OpId {
        self.ops[id].prev
    }

    /// How many ops the program has.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        usize::from(self.ops.len())
    }

    /// Whether the program has no ops.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append `op` to the list and return its id.
    pub fn push(&mut self, op: Op) -> OpId {
        let id = self.ops.push(OpLinked {
            prev: self.tail,
            next: OpId::NULL,
            op,
        });
        if self.tail.is_null() {
            self.head = id;
        } else {
            self.ops[self.tail].next = id;
        }
        self.tail = id;
        id
    }

    /// Insert `op` before `at`. Part of the editing API the slab exists for.
    #[allow(dead_code)]
    pub fn insert_before(&mut self, at: OpId, op: Op) -> OpId {
        let prev = self.ops[at].prev;
        let id = self.ops.push(OpLinked { prev, next: at, op });
        self.ops[at].prev = id;
        if prev.is_null() {
            self.head = id;
        } else {
            self.ops[prev].next = id;
        }
        id
    }

    /// Insert `op` after `at`. Part of the editing API the slab exists for.
    #[allow(dead_code)]
    pub fn insert_after(&mut self, at: OpId, op: Op) -> OpId {
        let next = self.ops[at].next;
        let id = self.ops.push(OpLinked { prev: at, next, op });
        self.ops[at].next = id;
        if next.is_null() {
            self.tail = id;
        } else {
            self.ops[next].prev = id;
        }
        id
    }
}

/// The two operands of an FPU broadcast, in `(tile, narrow)` order: an `add`,
/// `sub` or `mul` of a tile with a row, a column or a datum goes through the
/// FPU's `BroadcastType`, and both sides come from `Src` pages.
pub(crate) fn broadcast_pair(k: &Kernel, op: &Op) -> Option<(OpId, OpId)> {
    let Op::Binary { op, a, b } = *op else {
        return None;
    };
    if !matches!(op, BOp::Add | BOp::Sub | BOp::Mul) {
        return None;
    }
    let (x, y) = (k.info(a), k.info(b));
    if !matches!(x.dtype, DType::F32 | DType::Bf16) || !matches!(y.dtype, DType::F32 | DType::Bf16)
    {
        return None;
    }
    match (x.shape, y.shape) {
        (RegShape::Tile, RegShape::Row | RegShape::Column | RegShape::Scalar) => Some((a, b)),
        (RegShape::Row | RegShape::Column | RegShape::Scalar, RegShape::Tile) => Some((b, a)),
        _ => None,
    }
}

/// Rewrite every operand of `op` through `map`, the id translation a pass
/// builds as it copies a program (zyx's `renumber`).
pub fn renumber(op: &mut Op, map: &BTreeMap<OpId, OpId>) -> Result<(), ttStatus> {
    for param in op.parameters_mut() {
        let to = map
            .get(param)
            .copied()
            .ok_or(ttStatus::TT_ERROR_VALUE_UNDEFINED)?;
        *param = to;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// The register shape of one buffer unit, how many units the buffer holds, and
/// its byte size.
///
/// A `1,N` buffer is a row vector and a `N,1` buffer a column; otherwise both
/// dimensions are multiples of 32 and each 32x32 block is a tile. The extra
/// units are the pages of a deeper buffer (double or triple buffering).
pub fn unit_of(shape: [Dim; 2], dtype: DType) -> Result<(RegShape, u32, u32), ttStatus> {
    let [a, b] = shape;
    if a == 0 || b == 0 {
        return Err(ttStatus::TT_ERROR_ZERO_DIMENSION);
    }
    let reg = if a == 1 && b == 1 {
        RegShape::Scalar
    } else if a == 1 {
        RegShape::Row
    } else if b == 1 {
        RegShape::Column
    } else if a % 32 == 0 && b % 32 == 0 {
        RegShape::Tile
    } else {
        return Err(ttStatus::TT_ERROR_BUFFER_SHAPE);
    };
    let datums = if reg == RegShape::Scalar {
        1
    } else if reg == RegShape::Tile {
        32 * 32
    } else {
        32
    };
    let units = (a * b) / datums;
    let bytes = a * b * dtype.bytes();
    Ok((reg, units, bytes))
}

/// A ring's page shape and page count from the buffer's declared shape
/// (`%bi = buffer f32 32,64`: a `32,32` page, two of them).
fn cb_ring(shape: [Dim; 2], dtype: DType) -> Result<([Dim; 2], Dim), ttStatus> {
    let (unit, units, _) = unit_of(shape, dtype)?;
    let page = match unit {
        RegShape::Scalar => [1, 1],
        RegShape::Row => [1, 32],
        RegShape::Column => [32, 1],
        RegShape::Tile => [32, 32],
    };
    if units == 0 {
        return Err(ttStatus::TT_ERROR_ZERO_DIMENSION);
    }
    Ok((page, units))
}

/// The register shape a copy's or an access's `[Dim; 2]` unit names.
pub(crate) fn access_shape(shape: [Dim; 2]) -> Result<RegShape, ttStatus> {
    let [a, b] = shape;
    Ok(match (a, b) {
        (0, _) | (_, 0) => return Err(ttStatus::TT_ERROR_ZERO_DIMENSION),
        (1, 1) => RegShape::Scalar,
        (1, _) if b == 32 => RegShape::Row,
        (_, 1) if a == 32 => RegShape::Column,
        _ if a == 32 && b == 32 => RegShape::Tile,
        _ => return Err(ttStatus::TT_ERROR_ACCESS_SHAPE),
    })
}

impl Kernel {
    /// The `i32` a value folds to, when it is a constant or constant-only
    /// arithmetic. A value that depends on a loop index is not constant.
    ///
    /// The folds are exact two's-complement `i32`; a division by zero or a
    /// shift outside `0..32` is `None`, never a guess.
    pub fn const_i32(&self, value: OpId) -> Option<i32> {
        match *self.op(value)? {
            Op::Const {
                dtype: DType::I32,
                bits,
            } => Some(bits as i32),
            Op::Binary { op, a, b } => {
                let a = self.const_i32(a)?;
                let b = self.const_i32(b)?;
                match op {
                    BOp::Add => Some(a.wrapping_add(b)),
                    BOp::Sub => Some(a.wrapping_sub(b)),
                    BOp::Mul => Some(a.wrapping_mul(b)),
                    BOp::Div => a.checked_div(b),
                    BOp::Mod => a.checked_rem(b),
                    BOp::Pow => a.checked_pow(u32::try_from(b).ok()?),
                    BOp::Max => Some(a.max(b)),
                    BOp::Min => Some(a.min(b)),
                    BOp::BitAnd => Some(a & b),
                    BOp::BitOr => Some(a | b),
                    BOp::BitXor => Some(a ^ b),
                    BOp::BitShiftLeft => a.checked_shl(u32::try_from(b).ok()?),
                    // A shift is over the two's-complement bit pattern: the
                    // right shift fills with zeros, a *logical* shift.
                    BOp::BitShiftRight => (a as u32)
                        .checked_shr(u32::try_from(b).ok()?)
                        .map(|v| v as i32),
                    BOp::Cmplt
                    | BOp::Cmpgt
                    | BOp::Cmpge
                    | BOp::Eq
                    | BOp::NotEq
                    | BOp::Or
                    | BOp::And => None,
                }
            }
            _ => None,
        }
    }

    /// What `value` is, recomputed from its defining op.
    ///
    /// The walk is bottom-up over the ops in program order, so a chain of `n`
    /// operations costs `n`, not the number of paths through its DAG: a value
    /// that is read many times (`silu`'s `x`, `gelu`'s `x`) would otherwise be
    /// re-derived once per use and the cost would be exponential in the chain's
    /// depth.
    pub fn info(&self, value: OpId) -> ValueInfo {
        use Op::*;
        let mut memo: Vec<Option<ValueInfo>> = vec![None; usize::from(self.ops.max_id()) + 1];
        for (id, op) in self.ops() {
            let of = |v: OpId| -> ValueInfo {
                memo[usize::from(v)].expect("a value's operand is defined before it")
            };
            let (kind, dtype, shape) = match *op {
                Const { dtype, .. } => (ValueKind::Const, dtype, RegShape::Scalar),
                Shape(_) => (ValueKind::Shape, INDEX, RegShape::Scalar),
                Param { dtype, .. } => (ValueKind::Param, dtype, RegShape::Tile),
                Variable { dtype } => (ValueKind::Param, dtype, RegShape::Scalar),
                Cb { dtype, .. } => (ValueKind::Buffer, dtype, RegShape::Tile),
                Acc { dtype, shape } => (ValueKind::Acc, dtype, acc_reg_shape(shape)),
                GridCoord { .. } | Loop { .. } => (ValueKind::Induction, INDEX, RegShape::Scalar),
                MathLock | MathUnlock | PackLock | PackUnlock => {
                    (ValueKind::Const, INDEX, RegShape::Scalar)
                }
                Load { buffer, shape, .. } => (
                    ValueKind::Register,
                    of(buffer).dtype,
                    access_shape(shape).unwrap_or(RegShape::Tile),
                ),
                Cast { dtype, .. } | Bitcast { dtype, .. } => {
                    (ValueKind::Register, dtype, of(self.cast_source(id)).shape)
                }
                Unary { a, .. } => (ValueKind::Register, of(a).dtype, of(a).shape),
                Transpose { src } => (ValueKind::Register, of(src).dtype, of(src).shape),
                Binary { op, a, b } => {
                    let dtype = if op.returns_bool() {
                        DType::Bool
                    } else {
                        of(a).dtype
                    };
                    let sa = of(a).shape;
                    let sb = of(b).shape;
                    let shape = if sa == sb {
                        sa
                    } else if sb == RegShape::Scalar {
                        sa
                    } else if sa == RegShape::Scalar {
                        sb
                    } else {
                        RegShape::Tile
                    };
                    (ValueKind::Register, dtype, shape)
                }
                Select {
                    if_true, if_false, ..
                } => {
                    let shape = of(if_true).shape;
                    let _ = if_false;
                    (ValueKind::Register, of(if_true).dtype, shape)
                }
                Matmul { acc, .. } | Sum { acc, .. } | Max { acc, .. } => {
                    (ValueKind::Acc, of(acc).dtype, of(acc).shape)
                }
                Store { buffer, .. } => (ValueKind::Buffer, of(buffer).dtype, RegShape::Tile),
                Grid { .. } => (ValueKind::Shape, INDEX, RegShape::Scalar),
                Copy { .. }
                | EndLoop
                | EndReader
                | EndCompute
                | Wait { .. }
                | Reserve { .. }
                | Push { .. }
                | Pop { .. } => {
                    let _ = id;
                    (ValueKind::Register, INDEX, RegShape::Scalar)
                }
            };
            memo[usize::from(id)] = Some(ValueInfo { kind, dtype, shape });
            if id == value {
                break;
            }
        }
        memo[usize::from(value)].expect("the value being asked about is defined")
    }

    /// The dtype of `value`.
    pub fn dtype(&self, value: OpId) -> DType {
        self.info(value).dtype
    }

    /// The shape of `value`, in dimensions. A register value has one or two; a
    /// scalar and a shape value have none.
    ///
    /// A `Shape` op's own value is its members, so [`Kernel::shape`] of one is
    /// that list, resolved to numbers: a symbolic term folds through
    /// [`Kernel::const_i32`]. A term that does not fold is this compiler's own
    /// bug, not a user program: [`Kernel::verify`] admits only constant terms
    /// where a consumer needs numbers, so the failure is reported and returned.
    #[allow(dead_code)]
    pub fn shape(&self, value: OpId) -> Result<Cow<'_, [Dim]>, ttStatus> {
        Ok(match *self.at(value) {
            Op::Shape(ref ops) => {
                let mut dims: Vec<Dim> = Vec::with_capacity(ops.len());
                for op in ops {
                    let Some(term) = self.const_i32(*op) else {
                        return Err(crate::status::internal_bug(
                            "a shape term is not a constant",
                        ));
                    };
                    let Ok(dim) = u32::try_from(term) else {
                        return Err(crate::status::internal_bug(
                            "a shape term does not fit a Dim",
                        ));
                    };
                    dims.push(dim);
                }
                Cow::Owned(dims)
            }
            Op::Param { shape, .. } => self.shape(shape)?,
            Op::Cb { page, .. } => Cow::Owned(page.to_vec()),
            Op::Acc { shape, .. } => Cow::Owned(shape.to_vec()),
            Op::Load { shape, .. } => Cow::Owned(shape.to_vec()),
            _ => Cow::Borrowed(&[]),
        })
    }

    /// The value a cast's operand names, following a chain of casts.
    fn cast_source(&self, value: OpId) -> OpId {
        match *self.at(value) {
            Op::Cast { src, .. } | Op::Bitcast { src, .. } => self.cast_source(src),
            _ => value,
        }
    }

    /// Everything a text program can get wrong: its references, scopes,
    /// dtypes, shapes and its grid. [`Kernel::parse`] is the caller that
    /// reports these as the caller's errors; [`Kernel::verify`] runs the same
    /// checks after the passes, where a failure is this compiler's own bug.
    pub(crate) fn check(&self) -> Result<(), ttStatus> {
        let slots = usize::from(self.ops.max_id());
        let mut values: Vec<Option<ValueInfo>> = vec![None; slots];
        let mut visible: Vec<bool> = vec![false; slots];
        let mut loops: Vec<usize> = Vec::new();
        let mut grid: Option<usize> = None;

        for (id, op) in self.ops() {
            let at: usize = id.into();
            for v in op.parameters() {
                let slot = usize::from(v);
                if values.get(slot).and_then(|v| *v).is_none() {
                    let _ = (at, v);
                    return Err(ttStatus::TT_ERROR_VALUE_UNDEFINED);
                }
                if !visible.get(slot).copied().unwrap_or(false) {
                    let _ = (at, v);
                    return Err(ttStatus::TT_ERROR_VALUE_OUT_OF_SCOPE);
                }
            }
            match *op {
                Op::Const { dtype, .. } => {
                    // A `bf16` constant is one datum of the format, the way
                    // zyx's `Constant::BF16` is: the 16 bits of the value (a
                    // decimal literal is truncated into them, `bf16::from_f32`'s
                    // `(x.to_bits() >> 16) as u16`), widened to the f32 image
                    // wherever a lane reads it. A `bool` has no datum to write.
                    if dtype == DType::Bool {
                        return Err(ttStatus::TT_ERROR_CONST_BOOL);
                    }
                }
                Op::Shape(ref ops) => {
                    if ops.is_empty() {
                        return Err(ttStatus::TT_ERROR_ZERO_DIMENSION);
                    }
                    for term in ops.iter() {
                        self.expect_index_scalar(*term, at)?;
                    }
                }
                Op::Param { dtype, shape, .. } => {
                    if dtype == DType::Bool {
                        return Err(ttStatus::TT_TODO_BOOL_PARAMETER);
                    }
                    self.expect_index_scalar(shape, at)?;
                    self.expect_terms(shape, at)?;
                }
                Op::Variable { dtype } => {
                    if dtype == DType::Bool {
                        return Err(ttStatus::TT_TODO_BOOL_PARAMETER);
                    }
                }
                Op::Cb { dtype, page, len } => {
                    if dtype == DType::Bool {
                        return Err(ttStatus::TT_TODO_BOOL_BUFFER);
                    }
                    if len == 0 {
                        return Err(ttStatus::TT_ERROR_ZERO_DIMENSION);
                    }
                    unit_of(page, dtype)?;
                }
                Op::Acc { dtype, shape } => {
                    acc_reg_shape_checked(shape, at, dtype)?;
                }
                Op::Grid { x, y } => {
                    if grid.is_some() {
                        return Err(ttStatus::TT_ERROR_GRID);
                    }
                    self.expect_index_scalar(x, at)?;
                    self.expect_index_scalar(y, at)?;
                    grid = Some(at);
                }
                Op::GridCoord { axis } => {
                    if axis > 1 {
                        return Err(ttStatus::TT_ERROR_GRID_AXIS);
                    }
                }
                Op::Loop { bound } => {
                    self.expect_index_scalar(bound, at)?;
                    loops.push(at);
                    visible[at] = true;
                }
                Op::EndLoop => {
                    let Some(start) = loops.pop() else {
                        return Err(ttStatus::TT_ERROR_ENDLOOP_WITHOUT_LOOP);
                    };
                    for slot in start..values.len().min(visible.len()) {
                        visible[slot] = false;
                    }
                }
                Op::Copy {
                    dst,
                    src,
                    dst_offset,
                    src_offset,
                    shape,
                } => {
                    let d = self.info(dst);
                    let s = self.info(src);
                    let (buffer, param, buffer_offset, param_offset) = match (d.kind, s.kind) {
                        (ValueKind::Buffer, ValueKind::Param) => (dst, src, dst_offset, src_offset),
                        (ValueKind::Param, ValueKind::Buffer) => (src, dst, src_offset, dst_offset),
                        _ => return Err(ttStatus::TT_ERROR_COPY_ENDS),
                    };
                    if self.info(param).shape == RegShape::Scalar {
                        return Err(ttStatus::TT_ERROR_COPY_SCALAR_PARAM);
                    }
                    if self.dtype(buffer) != self.dtype(param) {
                        return Err(ttStatus::TT_ERROR_COPY_CAST);
                    }
                    let unit = access_shape(shape)?;
                    if !matches!(unit, RegShape::Row | RegShape::Tile) {
                        return Err(ttStatus::TT_ERROR_COPY_UNIT);
                    }
                    let lanes = self.expect_offset(param_offset, at)?;
                    let page = self.expect_offset(buffer_offset, at)?;
                    if page != 1 {
                        return Err(ttStatus::TT_TODO_MULTI_LANE);
                    }
                    let (page, len) = match *self.at(buffer) {
                        Op::Cb { page, len, .. } => (page, len),
                        _ => return Err(ttStatus::TT_ERROR_NOT_A_BUFFER),
                    };
                    let (unit_shape, _, _) = unit_of(page, self.dtype(buffer))?;
                    if lanes.checked_mul(shape[0] * shape[1]) != Some(unit_shape.datums()) {
                        return Err(ttStatus::TT_ERROR_COPY_LANES);
                    }
                    let _ = len;
                    let _ = unit;
                }
                Op::Load {
                    buffer,
                    offset,
                    shape,
                } => {
                    let b = self.info(buffer);
                    let unit = access_shape(shape)?;
                    if self.expect_offset(offset, at)? != 1 {
                        return Err(ttStatus::TT_TODO_MULTI_LANE);
                    }
                    let datums = shape[0] * shape[1];
                    if b.kind == ValueKind::Param {
                        if unit != RegShape::Tile || b.shape != RegShape::Tile {
                            return Err(ttStatus::TT_ERROR_LOAD_PARAM_NOT_TILE);
                        }
                    } else {
                        if b.kind != ValueKind::Buffer {
                            return Err(ttStatus::TT_ERROR_NOT_A_BUFFER);
                        }
                        let (page, _) = match *self.at(buffer) {
                            Op::Cb { page, len, .. } => (page, len),
                            _ => return Err(ttStatus::TT_ERROR_NOT_A_BUFFER),
                        };
                        let (unit_shape, _, _) = unit_of(page, b.dtype)?;
                        if unit_shape.datums() < datums {
                            return Err(ttStatus::TT_ERROR_ACCESS_SHAPE);
                        }
                    }
                }
                Op::Store {
                    src,
                    buffer,
                    offset,
                    shape,
                } => {
                    let v = self.info(src);
                    let b = self.info(buffer);
                    let unit = access_shape(shape)?;
                    if self.expect_offset(offset, at)? != 1 {
                        return Err(ttStatus::TT_TODO_MULTI_LANE);
                    }
                    let datums = shape[0] * shape[1];
                    if b.kind == ValueKind::Acc {
                        if unit != RegShape::Tile {
                            return Err(ttStatus::TT_ERROR_ACC_SHAPE);
                        }
                        if self.const_i32(offset) != Some(0) {
                            return Err(ttStatus::TT_ERROR_ACC_OFFSET);
                        }
                        if v.kind != ValueKind::Const || v.dtype != b.dtype {
                            return Err(ttStatus::TT_TODO_ACC_STORE);
                        }
                        continue;
                    }
                    if v.kind != ValueKind::Register && v.kind != ValueKind::Acc {
                        return Err(ttStatus::TT_ERROR_STORE_VALUE_NOT_REGISTER);
                    }
                    if b.kind == ValueKind::Param {
                        if unit != RegShape::Tile || b.shape != RegShape::Tile {
                            return Err(ttStatus::TT_ERROR_STORE_PARAM_NOT_TILE);
                        }
                        if v.dtype != b.dtype {
                            return Err(ttStatus::TT_ERROR_STORE_PARAM_CAST);
                        }
                        if v.shape != RegShape::Tile && v.kind != ValueKind::Acc {
                            return Err(ttStatus::TT_ERROR_STORE_PARAM_VALUE_NOT_TILE);
                        }
                        continue;
                    }
                    if b.kind != ValueKind::Buffer {
                        return Err(ttStatus::TT_ERROR_NOT_A_BUFFER);
                    }
                    if v.dtype != b.dtype {
                        return Err(ttStatus::TT_ERROR_STORE_BUFFER_CAST);
                    }
                    let (page, _) = match *self.at(buffer) {
                        Op::Cb { page, len, .. } => (page, len),
                        _ => return Err(ttStatus::TT_ERROR_NOT_A_BUFFER),
                    };
                    let (unit_shape, _, _) = unit_of(page, b.dtype)?;
                    if unit_shape.datums() < datums {
                        return Err(ttStatus::TT_ERROR_ACCESS_SHAPE);
                    }
                    if v.shape != unit && v.kind != ValueKind::Acc {
                        return Err(ttStatus::TT_ERROR_STORE_BUFFER_SHAPE);
                    }
                }
                Op::Cast { src, dtype } => {
                    let s = self.info(src);
                    if s.kind != ValueKind::Register && s.kind != ValueKind::Const {
                        return Err(ttStatus::TT_ERROR_CAST_SOURCE);
                    }
                    if s.dtype.bytes() == 0 || dtype.bytes() == 0 {
                        let _ = dtype;
                        return Err(ttStatus::TT_ERROR_CAST_SOURCE);
                    }
                }
                Op::Bitcast { src, dtype } => {
                    if self.dtype(src).bytes() != dtype.bytes() {
                        let _ = (src, dtype);
                        return Err(ttStatus::TT_ERROR_BITCAST_WIDTH);
                    }
                }
                Op::Transpose { src } => {
                    let s = self.info(src);
                    if !is_value(&s) || s.shape != RegShape::Tile {
                        let _ = src;
                        return Err(ttStatus::TT_ERROR_TRANSPOSE_OPERAND);
                    }
                }
                Op::Unary { a, .. } => {
                    if !is_value(&self.info(a)) {
                        return Err(ttStatus::TT_ERROR_UNARY_OPERAND);
                    }
                }
                Op::Binary { op, a, b } => {
                    let (ia, ib) = (self.info(a), self.info(b));
                    if !is_value(&ia) || !is_value(&ib) {
                        return Err(ttStatus::TT_ERROR_BINARY_OPERANDS);
                    }
                    if op.is_logical() && (ia.dtype != DType::Bool || ib.dtype != DType::Bool) {
                        return Err(ttStatus::TT_ERROR_BINARY_LOGICAL_NOT_BOOL);
                    }
                    if ia.dtype != ib.dtype {
                        return Err(ttStatus::TT_ERROR_NO_IMPLICIT_CAST);
                    }
                }
                Op::Select {
                    cond,
                    if_true,
                    if_false,
                } => {
                    if self.dtype(cond) != DType::Bool {
                        return Err(ttStatus::TT_ERROR_CONDITION_NOT_BOOL);
                    }
                    if self.dtype(if_true) != self.dtype(if_false) {
                        return Err(ttStatus::TT_ERROR_SELECT_CAST);
                    }
                    if self.info(if_true).shape != self.info(if_false).shape {
                        return Err(ttStatus::TT_ERROR_SELECT_BROADCAST);
                    }
                }
                Op::Matmul { acc, a, b } => {
                    if self.info(acc).kind != ValueKind::Acc
                        || self.info(acc).shape != RegShape::Tile
                    {
                        return Err(ttStatus::TT_ERROR_MATMUL_ACC);
                    }
                    if self.info(a).shape != RegShape::Tile || self.info(b).shape != RegShape::Tile
                    {
                        return Err(ttStatus::TT_ERROR_MATMUL_OPERAND);
                    }
                    // The operands share a dtype; the accumulator may be wider
                    // (bf16 operands accumulate in f32), never narrower.
                    if !self.info(a).dtype.is_float()
                        || self.info(a).dtype != self.info(b).dtype
                        || (self.info(a).dtype == DType::F32 && self.info(acc).dtype == DType::Bf16)
                    {
                        return Err(ttStatus::TT_ERROR_MATMUL_DTYPE);
                    }
                }
                Op::Sum { acc, x, s } => {
                    self.expect_reduce(acc, x, at)?;
                    let _ = s;
                }
                Op::Max { acc, x } => {
                    self.expect_reduce(acc, x, at)?;
                }
                Op::Wait { cb, .. }
                | Op::Reserve { cb, .. }
                | Op::Push { cb, .. }
                | Op::Pop { cb, .. } => {
                    if !matches!(*self.at(cb), Op::Cb { .. }) {
                        return Err(ttStatus::TT_ERROR_NOT_A_BUFFER);
                    }
                }
                Op::EndReader
                | Op::EndCompute
                | Op::MathLock
                | Op::MathUnlock
                | Op::PackLock
                | Op::PackUnlock => {}
            }
            if let Some(slot) = values.get_mut(at) {
                if !matches!(
                    *op,
                    Op::Copy { .. }
                        | Op::Store { .. }
                        | Op::EndLoop
                        | Op::EndReader
                        | Op::EndCompute
                        | Op::MathLock
                        | Op::MathUnlock
                        | Op::PackLock
                        | Op::PackUnlock
                        | Op::Grid { .. }
                        | Op::Matmul { .. }
                        | Op::Sum { .. }
                        | Op::Max { .. }
                        | Op::Wait { .. }
                        | Op::Reserve { .. }
                        | Op::Push { .. }
                        | Op::Pop { .. }
                ) {
                    *slot = Some(self.info(id));
                    visible[at] = true;
                }
            }
        }

        if grid.is_none() {
            return Err(ttStatus::TT_ERROR_GRID);
        }
        if !loops.is_empty() {
            return Err(ttStatus::TT_ERROR_ENDLOOP_WITHOUT_LOOP);
        }
        Ok(())
    }

    /// An index: a value of the index dtype, whose shape is how many lanes it
    /// names (a row or a column is 32 bases, a tile 1024).
    fn expect_offset(&self, value: OpId, at: usize) -> Result<u32, ttStatus> {
        let info = self.info(value);
        if info.dtype != INDEX {
            let _ = at;
            return Err(ttStatus::TT_ERROR_OFFSET_NOT_INDEX);
        }
        Ok(match info.shape {
            RegShape::Scalar => 1,
            RegShape::Row | RegShape::Column => 32,
            RegShape::Tile => 1024,
        })
    }

    /// A shape term must be a scalar of the index dtype.
    fn expect_index_scalar(&self, value: OpId, at: usize) -> Result<(), ttStatus> {
        let _ = at;
        let info = self.info(value);
        if info.dtype != INDEX || info.shape != RegShape::Scalar {
            return Err(ttStatus::TT_ERROR_DIMENSION_NOT_INDEX);
        }
        Ok(())
    }

    /// A reduce's accumulator and operand: the accumulator folds one tile
    /// region, its shape picking the axis (see [`ReduceDim::of`]), and the
    /// operand is a tile of the accumulator's dtype.
    fn expect_reduce(&self, acc: OpId, x: OpId, at: usize) -> Result<(), ttStatus> {
        let a = self.info(acc);
        if a.kind != ValueKind::Acc {
            let _ = at;
            return Err(ttStatus::TT_ERROR_REDUCE_ACC);
        }
        if ReduceDim::of(a.shape).is_none() {
            let _ = at;
            return Err(ttStatus::TT_ERROR_REDUCE_ACC_SHAPE);
        }
        let x = self.info(x);
        if x.shape != RegShape::Tile {
            let _ = at;
            return Err(ttStatus::TT_ERROR_REDUCE_OPERAND);
        }
        // The accumulator's declared dtype is the dtype the fold accumulates
        // into: it may be wider than the tile (a bf16 tile into an f32 `Dst`),
        // never narrower.
        if x.dtype != a.dtype && !(x.dtype == DType::Bf16 && a.dtype == DType::F32) {
            let _ = (x.dtype, a.dtype);
            return Err(ttStatus::TT_ERROR_REDUCE_DTYPE);
        }
        Ok(())
    }

    /// Every member of the shape op `value` must be a non-zero index scalar.
    fn expect_terms(&self, value: OpId, at: usize) -> Result<(), ttStatus> {
        let Op::Shape(ref ops) = *self.at(value) else {
            let _ = at;
            return Err(ttStatus::TT_ERROR_DIMENSION_NOT_INDEX);
        };
        for term in ops.iter() {
            self.expect_index_scalar(*term, at)?;
            if self.const_i32(*term) == Some(0) {
                return Err(ttStatus::TT_ERROR_ZERO_DIMENSION);
            }
        }
        Ok(())
    }
}

/// The values the datapath may take as an operand: a literal, a loop index, a
/// grid coordinate or a register value. Parameters, buffers, accumulators and
/// shapes are named by their own ops and never flow through the datapath.
fn is_value(info: &ValueInfo) -> bool {
    match info.kind {
        ValueKind::Const | ValueKind::Induction | ValueKind::Register => true,
        // A scalar parameter carries its value, so the movers can compute with
        // it, and an accumulator is a `Dst` region the datapath reads like any
        // value (a reduce's result is broadcast back over the tile).
        ValueKind::Param => info.shape == RegShape::Scalar,
        ValueKind::Acc => true,
        ValueKind::Buffer | ValueKind::Shape => false,
    }
}

/// An accumulator's shape and dtype, checked: the four shapes the hardware
/// has, and a float datum.
fn acc_reg_shape_checked(shape: [Dim; 2], at: usize, dtype: DType) -> Result<RegShape, ttStatus> {
    let reg = match shape {
        [1, 32] | [32, 1] | [1, 1] | [32, 32] => acc_reg_shape(shape),
        _ => {
            let _ = at;
            return Err(ttStatus::TT_ERROR_ACC_SHAPE);
        }
    };
    if !matches!(dtype, DType::F32 | DType::Bf16) {
        let _ = (at, dtype);
        return Err(ttStatus::TT_ERROR_ACC_DTYPE);
    }
    Ok(reg)
}

/// The register shape an accumulator's fixed shape names.
fn acc_reg_shape(shape: [Dim; 2]) -> RegShape {
    match shape {
        [1, 32] => RegShape::Row,
        [32, 1] => RegShape::Column,
        [1, 1] => RegShape::Scalar,
        _ => RegShape::Tile,
    }
}

// ---------------------------------------------------------------------------
// Forms
// ---------------------------------------------------------------------------

/// The IR version this build reads.
pub const VERSION: u32 = 9;

impl Kernel {
    /// Parse the text form.
    ///
    /// The grammar is the one the tests write:
    ///
    /// ```text
    /// version 9
    /// grid 1,1
    /// %in = param f32 2048
    /// %s  = param i32 0
    /// %bi = buffer f32 32,64
    /// %a  = acc f32 32,32
    /// %c  = const f32 1.0
    /// %i  = loop 2
    /// %o  = mul %i %s
    /// %x  = load 32,32 %bi [%o]
    /// %y  = neg %x
    /// copy 32,32 %bi [%o] = %in [%o]
    /// store 32,32 %bo [%o] = %y
    /// inventory: matmul, sum, max, sel, cast, bitcast, wait, reserve, push,
    /// pop, endreader, endcompute, endloop
    /// ```
    ///
    /// Every inline literal — a bracket index, a shape term, a loop bound — is
    /// interned as an `Op::Const` of the index dtype, and a multi-term
    /// parameter shape becomes an `Op::Shape` of those terms, so the program's
    /// own ops are the only place an extent lives.
    pub fn parse(text: &str) -> Result<Kernel, ttStatus> {
        let mut p = Parser::default();
        p.run(text)
    }
}

impl Parser {
    fn value(&self, t: &mut Tokens<'_>) -> Result<OpId, ttStatus> {
        t.value(&self.symbols)
    }

    /// Parse every line, in order.
    fn run(&mut self, text: &str) -> Result<Kernel, ttStatus> {
        let mut version = false;
        for line in text.lines() {
            // `#` starts a comment wherever it appears and runs to the end of
            // the line, so a statement can be annotated after it. Cutting it
            // off here, before the line is tokenised, keeps the rule out of
            // every op's parser and makes a `#` inside a word a comment too,
            // never a token.
            let line = line.split_once('#').map_or(line, |(before, _)| before);
            let line = line.trim();
            if line.is_empty() || line.starts_with(';') || line.starts_with("//") {
                continue;
            }
            let mut t = Tokens::new(line);
            if !version {
                if t.word() != Some("version") {
                    return Err(ttStatus::TT_ERROR_VERSION);
                }
                if t.number()? != VERSION {
                    return Err(ttStatus::TT_ERROR_VERSION);
                }
                version = true;
                continue;
            }
            if let Some(word) = t.peek_word() {
                match word {
                    "grid" => {
                        t.word();
                        let x = self.term(&mut t)?;
                        t.comma()?;
                        let y = self.term(&mut t)?;
                        t.end()?;
                        self.push(Op::Grid { x, y });
                        continue;
                    }
                    "copy" => {
                        t.word();
                        let shape = t.shape2()?;
                        let dst = self.value(&mut t)?;
                        let dst_offset = t.bracket(self)?;
                        t.eq()?;
                        let src = self.value(&mut t)?;
                        let src_offset = if t.peek_char() == Some('[') {
                            t.bracket(self)?
                        } else {
                            dst_offset
                        };
                        t.end()?;
                        self.push(Op::Copy {
                            dst,
                            src,
                            dst_offset,
                            src_offset,
                            shape,
                        });
                        continue;
                    }
                    "store" => {
                        t.word();
                        let shape = t.shape2()?;
                        let buffer = self.value(&mut t)?;
                        let offset = if t.peek_char() == Some('[') {
                            t.bracket(self)?
                        } else {
                            self.intern_i32(0)
                        };
                        t.eq()?;
                        let src = self.value(&mut t)?;
                        t.end()?;
                        self.push(Op::Store {
                            src,
                            buffer,
                            offset,
                            shape,
                        });
                        continue;
                    }
                    "load" => {
                        t.word();
                        let shape = t.shape2()?;
                        let name = t.name()?;
                        t.eq()?;
                        let buffer = self.value(&mut t)?;
                        let offset = t.bracket(self)?;
                        t.end()?;
                        let id = self.ops.len();
                        self.define(name, id)?;
                        self.push(Op::Load {
                            buffer,
                            offset,
                            shape,
                        });
                        continue;
                    }
                    "matmul" => {
                        t.word();
                        let acc = self.value(&mut t)?;
                        let a = self.value(&mut t)?;
                        let b = self.value(&mut t)?;
                        t.end()?;
                        self.push(Op::Matmul { acc, a, b });
                        continue;
                    }
                    "sum" => {
                        t.word();
                        let acc = self.value(&mut t)?;
                        let x = self.value(&mut t)?;
                        let s = self.value(&mut t)?;
                        t.end()?;
                        self.push(Op::Sum { acc, x, s });
                        continue;
                    }
                    "max" => {
                        t.word();
                        let acc = self.value(&mut t)?;
                        let x = self.value(&mut t)?;
                        t.end()?;
                        self.push(Op::Max { acc, x });
                        continue;
                    }
                    "wait" | "reserve" | "push" | "pop" => {
                        let Some(word) = t.word() else {
                            return Err(crate::status::internal_bug(
                                "a peeked hand-off word was not there",
                            ));
                        };
                        let cb = self.value(&mut t)?;
                        let n = match t.peek_char() {
                            Some(c) if c.is_ascii_digit() => t.number()?,
                            _ => 1,
                        };
                        t.end()?;
                        self.push(match word {
                            "wait" => Op::Wait { cb, n },
                            "reserve" => Op::Reserve { cb, n },
                            "push" => Op::Push { cb, n },
                            _ => Op::Pop { cb, n },
                        });
                        continue;
                    }
                    "endloop" => {
                        t.word();
                        t.end()?;
                        self.push(Op::EndLoop);
                        continue;
                    }
                    "endreader" => {
                        t.word();
                        t.end()?;
                        self.push(Op::EndReader);
                        continue;
                    }
                    "endcompute" => {
                        t.word();
                        t.end()?;
                        self.push(Op::EndCompute);
                        continue;
                    }
                    _ => {}
                }
            }
            // Everything else is a definition: `%name = ...`.
            self.definition(&mut t)?;
        }
        if !version {
            return Err(ttStatus::TT_ERROR_VERSION);
        }
        let k = self.take()?;
        k.check()?;
        // The ring hand-offs are the compiler's, not the program's: `sync_cbs`
        // places them after it has split the program into its roles, and a text
        // program that names one has stepped into the machine representation.
        // Refuse it as the caller error it is, at the boundary the caller
        // crossed, rather than letting the pass report an internal bug later.
        if k.ops().any(|(_, op)| {
            matches!(
                *op,
                Op::Wait { .. } | Op::Reserve { .. } | Op::Push { .. } | Op::Pop { .. }
            )
        }) {
            return Err(ttStatus::TT_ERROR_PROGRAM);
        }
        Ok(k)
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut p = Printer {
            k: self,
            f: f as &mut dyn fmt::Write,
            loop_depth: 0,
        };
        writeln!(p.f, "version {VERSION}")?;
        for (id, op) in self.ops() {
            p.op(id, op)?;
        }
        Ok(())
    }
}

/// The state of a parse: the ops in order and the names defined so far.
#[derive(Default)]
struct Parser {
    ops: Vec<Op>,
    names: BTreeMap<OpId, String>,
    symbols: BTreeMap<String, OpId>,
}

impl Parser {
    fn push(&mut self, op: Op) -> OpId {
        let id = OpId(self.ops.len() as u32);
        self.ops.push(op);
        id
    }

    fn define(&mut self, name: String, id: usize) -> Result<(), ttStatus> {
        if self.symbols.contains_key(&name) {
            return Err(ttStatus::TT_ERROR_NAME_DUPLICATE);
        }
        let id = OpId(id as u32);
        self.names.insert(id, name.clone());
        self.symbols.insert(name, id);
        Ok(())
    }

    /// A literal index, as a const op of the index dtype.
    fn intern_i32(&mut self, value: i32) -> OpId {
        self.push(Op::Const {
            dtype: INDEX,
            bits: value as u32,
        })
    }

    fn term(&mut self, t: &mut Tokens<'_>) -> Result<OpId, ttStatus> {
        if t.peek_char() == Some('%') {
            return t.value(&self.symbols);
        }
        let n = t.number()?;
        Ok(self.intern_i32(n as i32))
    }

    fn definition(&mut self, t: &mut Tokens<'_>) -> Result<(), ttStatus> {
        let name = t.name()?;
        t.eq()?;
        let keyword = t.word().ok_or(ttStatus::TT_ERROR_PARSE)?.to_owned();
        let key = keyword.as_str();
        match key {
            "const" => {
                let dtype = t.dtype()?;
                let bits = t.literal(dtype)?;
                t.end()?;
                let id = self.ops.len();
                self.define(name, id)?;
                self.push(Op::Const { dtype, bits });
            }
            "param" => {
                let dtype = t.dtype()?;
                let scalar = t.peek_char() == Some('0') && t.rest().trim() == "0";
                let shape = if scalar {
                    t.number()?;
                    OpId::NULL
                } else {
                    let mut terms = vec![self.term(t)?];
                    while t.peek_char() == Some(',') {
                        t.comma()?;
                        terms.push(self.term(t)?);
                    }
                    self.push(Op::Shape(terms.into_boxed_slice()))
                };
                let interleaved = t.peek_word() == Some("interleaved");
                if interleaved {
                    t.word();
                }
                t.end()?;
                let id = self.ops.len();
                self.define(name, id)?;
                if scalar {
                    self.push(Op::Variable { dtype });
                } else {
                    self.push(Op::Param {
                        dtype,
                        shape,
                        interleaved,
                    });
                }
            }
            "buffer" => {
                let dtype = t.dtype()?;
                let a = t.number()?;
                t.comma()?;
                let b = t.number()?;
                t.end()?;
                let (page, len) = cb_ring([a, b], dtype)?;
                let id = self.ops.len();
                self.define(name, id)?;
                self.push(Op::Cb { dtype, page, len });
            }
            "acc" => {
                let dtype = t.dtype()?;
                let a = t.number()?;
                t.comma()?;
                let b = t.number()?;
                t.end()?;
                let id = self.ops.len();
                self.define(name, id)?;
                self.push(Op::Acc {
                    dtype,
                    shape: [a, b],
                });
            }
            "loop" => {
                let bound = self.term(t)?;
                t.end()?;
                let id = self.ops.len();
                self.define(name, id)?;
                self.push(Op::Loop { bound });
            }
            "grid" => {
                t.dot()?;
                let axis = match t.word() {
                    Some("x") => 0,
                    Some("y") => 1,
                    _ => return Err(ttStatus::TT_ERROR_GRID_AXIS),
                };
                t.end()?;
                let id = self.ops.len();
                self.define(name, id)?;
                self.push(Op::GridCoord { axis });
            }
            "cast" | "bitcast" => {
                let src = self.value(t)?;
                if t.word() != Some("to") {
                    return Err(ttStatus::TT_ERROR_PARSE);
                }
                let dtype = t.dtype()?;
                t.end()?;
                let id = self.ops.len();
                self.define(name, id)?;
                self.push(if key == "cast" {
                    Op::Cast { dtype, src }
                } else {
                    Op::Bitcast { dtype, src }
                });
            }
            "sel" => {
                let cond = t.value(&self.symbols)?;
                let if_true = t.value(&self.symbols)?;
                let if_false = t.value(&self.symbols)?;
                t.end()?;
                let id = self.ops.len();
                self.define(name, id)?;
                self.push(Op::Select {
                    cond,
                    if_true,
                    if_false,
                });
            }
            "transpose" => {
                let src = t.value(&self.symbols)?;
                t.end()?;
                let id = self.ops.len();
                self.define(name, id)?;
                self.push(Op::Transpose { src });
            }
            other => {
                let id = self.ops.len();
                if let Some(op) = UOp::from_name(other) {
                    let a = t.value(&self.symbols)?;
                    t.end()?;
                    self.define(name, id)?;
                    self.push(Op::Unary { op, a });
                } else if let Some(op) = BOp::from_name(other) {
                    let a = t.value(&self.symbols)?;
                    let b = t.value(&self.symbols)?;
                    t.end()?;
                    self.define(name, id)?;
                    self.push(Op::Binary { op, a, b });
                } else {
                    return Err(ttStatus::TT_ERROR_PARSE);
                }
            }
        }
        Ok(())
    }

    fn take(&mut self) -> Result<Kernel, ttStatus> {
        let mut k = Kernel::default();
        for op in std::mem::take(&mut self.ops) {
            k.push(op);
        }
        k.names = std::mem::take(&mut self.names);
        Ok(k)
    }
}

/// The text form's printer.
struct Printer<'a, 'b> {
    k: &'a Kernel,
    f: &'b mut dyn fmt::Write,
    #[allow(dead_code)]
    loop_depth: usize,
}

impl Printer<'_, '_> {
    fn name(&self, value: OpId) -> String {
        match self.k.names.get(&value) {
            Some(name) => format!("%{name}"),
            None => match *self.k.at(value) {
                Op::Const { .. } => format!("{}", self.k.const_i32(value).unwrap_or_default()),
                _ => format!("%{value}"),
            },
        }
    }

    fn op(&mut self, id: OpId, op: &Op) -> fmt::Result {
        let k = self.k;
        let named = k.names.contains_key(&id);
        match *op {
            Op::Shape(_) => Ok(()),
            Op::Const { dtype, bits } => writeln!(
                self.f,
                "%{} = const {} {}",
                k.names.get(&id).map(String::as_str).unwrap_or("?"),
                dtype.name(),
                match literal(dtype, bits) {
                    Ok(text) => text,
                    Err(_) => return Err(fmt::Error),
                }
            ),
            Op::Param {
                dtype,
                shape,
                interleaved,
            } => {
                write!(self.f, "{} = param {} ", self.name(id), dtype.name())?;
                let terms: Vec<OpId> = match *k.at(shape) {
                    Op::Shape(ref ops) => ops.to_vec(),
                    _ => Vec::new(),
                };
                let text: Vec<String> = terms.iter().map(|t| self.name(*t)).collect();
                writeln!(
                    self.f,
                    "{}{}",
                    text.join(","),
                    if interleaved { " interleaved" } else { "" }
                )
            }
            Op::Variable { dtype } => {
                writeln!(self.f, "{} = param {} 0", self.name(id), dtype.name())
            }
            Op::Cb { dtype, page, len } => writeln!(
                self.f,
                "{} = buffer {} {},{}",
                self.name(id),
                dtype.name(),
                page[0],
                page[1] * len
            ),
            Op::Acc { dtype, shape } => writeln!(
                self.f,
                "{} = acc {} {},{}",
                self.name(id),
                dtype.name(),
                shape[0],
                shape[1]
            ),
            Op::Grid { x, y } => writeln!(self.f, "grid {},{}", self.name(x), self.name(y)),
            Op::Loop { bound } => {
                self.loop_depth += 1;
                writeln!(self.f, "{} = loop {}", self.name(id), self.name(bound))
            }
            Op::EndLoop => {
                self.loop_depth = self.loop_depth.saturating_sub(1);
                writeln!(self.f, "endloop")
            }
            Op::EndReader => writeln!(self.f, "endreader"),
            Op::EndCompute => writeln!(self.f, "endcompute"),
            Op::MathLock => writeln!(self.f, "mathlock"),
            Op::MathUnlock => writeln!(self.f, "mathunlock"),
            Op::PackLock => writeln!(self.f, "packlock"),
            Op::PackUnlock => writeln!(self.f, "packunlock"),
            Op::GridCoord { axis } => writeln!(
                self.f,
                "{} = grid.{}",
                self.name(id),
                if axis == 0 { "x" } else { "y" }
            ),
            Op::Copy {
                dst,
                src,
                dst_offset,
                src_offset,
                shape,
            } => writeln!(
                self.f,
                "copy {},{} {} [{}] = {} [{}]",
                shape[0],
                shape[1],
                self.name(dst),
                self.name(dst_offset),
                self.name(src),
                self.name(src_offset)
            ),
            Op::Load {
                buffer,
                offset,
                shape,
            } => writeln!(
                self.f,
                "load {},{} {} = {} [{}]",
                shape[0],
                shape[1],
                self.name(id),
                self.name(buffer),
                self.name(offset)
            ),
            Op::Store {
                src,
                buffer,
                offset,
                shape,
            } => writeln!(
                self.f,
                "store {},{} {} [{}] = {}",
                shape[0],
                shape[1],
                self.name(buffer),
                self.name(offset),
                self.name(src)
            ),
            Op::Cast { dtype, src } => writeln!(
                self.f,
                "{} = cast {} {}",
                self.name(id),
                dtype.name(),
                self.name(src)
            ),
            Op::Bitcast { dtype, src } => writeln!(
                self.f,
                "{} = bitcast {} {}",
                self.name(id),
                dtype.name(),
                self.name(src)
            ),
            Op::Unary { op, a } => writeln!(
                self.f,
                "{} = {} {}",
                self.name(id),
                unary_name(op),
                self.name(a)
            ),
            Op::Transpose { src } => {
                writeln!(self.f, "{} = transpose {}", self.name(id), self.name(src))
            }
            Op::Binary { op, a, b } => writeln!(
                self.f,
                "{} = {} {} {}",
                self.name(id),
                binary_name(op),
                self.name(a),
                self.name(b)
            ),
            Op::Select {
                cond,
                if_true,
                if_false,
            } => writeln!(
                self.f,
                "{} = sel {} {} {}",
                self.name(id),
                self.name(cond),
                self.name(if_true),
                self.name(if_false)
            ),
            Op::Matmul { acc, a, b } => writeln!(
                self.f,
                "matmul {} {} {}",
                self.name(acc),
                self.name(a),
                self.name(b)
            ),
            Op::Sum { acc, x, s } => writeln!(
                self.f,
                "sum {} {} {}",
                self.name(acc),
                self.name(x),
                self.name(s)
            ),
            Op::Max { acc, x } => writeln!(self.f, "max {} {}", self.name(acc), self.name(x)),
            Op::Wait { cb, n } => writeln!(self.f, "wait {}{}", self.name(cb), pages(n)),
            Op::Reserve { cb, n } => writeln!(self.f, "reserve {}{}", self.name(cb), pages(n)),
            Op::Push { cb, n } => writeln!(self.f, "push {}{}", self.name(cb), pages(n)),
            Op::Pop { cb, n } => writeln!(self.f, "pop {}{}", self.name(cb), pages(n)),
        }?;
        let _ = named;
        Ok(())
    }
}

/// A hand-off's page count as text: omitted when it is one page, the common
/// case.
fn pages(n: Dim) -> String {
    if n == 1 {
        String::new()
    } else {
        format!(" {n}")
    }
}

fn literal(dtype: DType, bits: u32) -> Result<String, ttStatus> {
    match dtype {
        DType::I32 => Ok((bits as i32).to_string()),
        DType::F32 => {
            let v = f32::from_bits(bits);
            if v.fract() == 0.0 {
                Ok(format!("{v:.1}"))
            } else {
                Ok(format!("{v}"))
            }
        }
        DType::Bf16 => {
            // The 16-bit datum is the f32 it widens to; printing that image
            // truncates back to the same sixteen bits.
            let v = f32::from_bits((bits & 0xffff) << 16);
            if v.fract() == 0.0 {
                Ok(format!("{v:.1}"))
            } else {
                Ok(format!("{v}"))
            }
        }
        DType::Bool => Err(crate::status::internal_bug(
            "a bool constant reached the printer",
        )),
    }
}

fn unary_name(op: UOp) -> &'static str {
    match op {
        UOp::Neg => "neg",
        UOp::Not => "not",
        UOp::BitNot => "bitnot",
        UOp::Exp2 => "exp2",
        UOp::Log2 => "log2",
        UOp::Reciprocal => "reciprocal",
        UOp::Sqrt => "sqrt",
        UOp::Rsqrt => "rsqrt",
        UOp::Sin => "sin",
        UOp::Cos => "cos",
        UOp::Floor => "floor",
        UOp::Trunc => "trunc",
        UOp::Abs => "abs",
    }
}

fn binary_name(op: BOp) -> &'static str {
    match op {
        BOp::Add => "add",
        BOp::Sub => "sub",
        BOp::Mul => "mul",
        BOp::Div => "div",
        BOp::Pow => "pow",
        BOp::Mod => "mod",
        BOp::Cmplt => "cmplt",
        BOp::Cmpgt => "cmpgt",
        BOp::Cmpge => "cmpge",
        BOp::Max => "max",
        BOp::Min => "min",
        BOp::Or => "or",
        BOp::And => "and",
        BOp::BitXor => "bitxor",
        BOp::BitOr => "bitor",
        BOp::BitAnd => "bitand",
        BOp::BitShiftLeft => "bitshiftleft",
        BOp::BitShiftRight => "bitshiftright",
        BOp::NotEq => "noteq",
        BOp::Eq => "eq",
    }
}

/// One line's tokens, read left to right. The grammar is whitespace-separated
/// words, `%names`, numbers and the punctuation `, = . [ ]`.
struct Tokens<'a> {
    text: &'a str,
    at: usize,
}

impl<'a> Tokens<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, at: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.text[self.at..]
    }

    fn skip_space(&mut self) {
        while let Some(c) = self.rest().chars().next() {
            if c.is_whitespace() {
                self.at += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn peek_char(&mut self) -> Option<char> {
        self.skip_space();
        self.rest().chars().next()
    }

    fn peek_word(&mut self) -> Option<&'a str> {
        self.skip_space();
        let rest = self.rest();
        let end = rest
            .find(|c: char| {
                c.is_whitespace() || c == ',' || c == '[' || c == ']' || c == '=' || c == '.'
            })
            .unwrap_or(rest.len());
        if end == 0 { None } else { Some(&rest[..end]) }
    }

    /// One whitespace-delimited token, punctuation included: a literal like
    /// `0.5` has a `.` in it.
    fn token(&mut self) -> Option<&'a str> {
        self.skip_space();
        let rest = self.rest();
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        if end == 0 {
            return None;
        }
        let token = &rest[..end];
        self.at += end;
        Some(token)
    }

    fn word(&mut self) -> Option<&'a str> {
        let word = self.peek_word()?;
        self.at += word.len();
        Some(word)
    }

    fn name(&mut self) -> Result<String, ttStatus> {
        self.skip_space();
        if self.peek_char() != Some('%') {
            return Err(ttStatus::TT_ERROR_PARSE);
        }
        self.at += 1;
        let word = self.word().ok_or(ttStatus::TT_ERROR_PARSE)?;
        Ok(word.to_owned())
    }

    fn value(&mut self, symbols: &BTreeMap<String, OpId>) -> Result<OpId, ttStatus> {
        let name = self.name()?;
        symbols
            .get(&name)
            .copied()
            .ok_or(ttStatus::TT_ERROR_VALUE_UNDEFINED)
    }

    fn number(&mut self) -> Result<u32, ttStatus> {
        self.skip_space();
        let rest = self.rest();
        let end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        if end == 0 {
            return Err(ttStatus::TT_ERROR_PARSE);
        }
        let n: u32 = rest[..end].parse().map_err(|_| ttStatus::TT_ERROR_PARSE)?;
        self.at += end;
        Ok(n)
    }

    fn comma(&mut self) -> Result<(), ttStatus> {
        self.skip_space();
        if self.peek_char() != Some(',') {
            return Err(ttStatus::TT_ERROR_PARSE);
        }
        self.at += 1;
        Ok(())
    }

    fn dot(&mut self) -> Result<(), ttStatus> {
        self.skip_space();
        if self.peek_char() != Some('.') {
            return Err(ttStatus::TT_ERROR_PARSE);
        }
        self.at += 1;
        Ok(())
    }

    fn eq(&mut self) -> Result<(), ttStatus> {
        self.skip_space();
        if self.peek_char() != Some('=') {
            return Err(ttStatus::TT_ERROR_PARSE);
        }
        self.at += 1;
        Ok(())
    }

    fn bracket(&mut self, p: &mut Parser) -> Result<OpId, ttStatus> {
        self.skip_space();
        if self.peek_char() != Some('[') {
            return Err(ttStatus::TT_ERROR_PARSE);
        }
        self.at += 1;
        let value = p.term(self)?;
        self.skip_space();
        if self.peek_char() != Some(']') {
            return Err(ttStatus::TT_ERROR_PARSE);
        }
        self.at += 1;
        Ok(value)
    }

    fn shape2(&mut self) -> Result<[Dim; 2], ttStatus> {
        let a = self.number()?;
        self.comma()?;
        let b = self.number()?;
        Ok([a, b])
    }

    fn dtype(&mut self) -> Result<DType, ttStatus> {
        match self.word() {
            Some("f32") => Ok(DType::F32),
            Some("bf16") => Ok(DType::Bf16),
            Some("i32") => Ok(DType::I32),
            Some("bool") => Ok(DType::Bool),
            // A real format with no path yet: named, never guessed.
            Some("tf32") => Err(ttStatus::TT_TODO_TF32),
            _ => Err(ttStatus::TT_ERROR_PARSE),
        }
    }

    fn literal(&mut self, dtype: DType) -> Result<u32, ttStatus> {
        let word = self.token().ok_or(ttStatus::TT_ERROR_PARSE)?;
        match dtype {
            DType::I32 => word
                .parse::<i32>()
                .map(|v| v as u32)
                .map_err(|_| ttStatus::TT_ERROR_PARSE),
            DType::F32 => word
                .parse::<f32>()
                .map(|v| v.to_bits())
                .map_err(|_| ttStatus::TT_ERROR_PARSE),
            // zyx's `bf16::from_f32`: the high half of the literal's f32 bits,
            // kept as the bf16 datum itself (the op's own two bytes).
            DType::Bf16 => word
                .parse::<f32>()
                .map(|v| (v.to_bits() >> 16) & 0xffff)
                .map_err(|_| ttStatus::TT_ERROR_PARSE),
            DType::Bool => Err(ttStatus::TT_ERROR_CONST_BOOL),
        }
    }

    fn end(&mut self) -> Result<(), ttStatus> {
        self.skip_space();
        if self.rest().is_empty() {
            Ok(())
        } else {
            Err(ttStatus::TT_ERROR_TRAILING_INPUT)
        }
    }
}
