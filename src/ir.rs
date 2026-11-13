// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The semantic kernel IR: one unified SSA program, machine programs out.
//!
//! What the caller writes is what the kernel *means*. There are no reader,
//! compute and writer segments: a DRAM parameter, an L1 buffer, a register
//! value and the operations on them are all named in one list, and the compiler
//! classifies each instruction by the operands it touches. There is no
//! `unpack`/`pack` and no buffer layout: how L1 becomes a register and how a
//! register becomes L1 is the compiler's business.
//!
//! Parameters and buffers are mutable, not SSA: `copy_to` (in either direction),
//! `load` and `store_to` all name the same parameter or buffer. SSA covers the
//! register values produced by `const`, `cast`, the arithmetic ops, `load` and
//! `matmul`.
//!
//! A shape is a comma-separated list of terms, and a term is a literal or a
//! value of the fixed index dtype. There is no inline arithmetic: a product is
//! built by an op first. A parameter may be arbitrary-dimensional and may use
//! symbolic terms; a scalar parameter is the shape `0`. A buffer is always
//! two-dimensional, always constant, and each dimension is `1` or a multiple
//! of 32. The grid is a shape of exactly two terms and may be symbolic too.
//!
//! Both forms are canonical: the text form and the byte form ([`MAGIC`],
//! [`VERSION`]). Each refuses the other's version.

use core::fmt;
use std::collections::BTreeMap;

/// Identifier of an SSA value. Equals the definition order.
pub type ValueId = u32;

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

    fn from_tag(tag: u8) -> Option<DType> {
        match tag {
            0 => Some(DType::F32),
            1 => Some(DType::Bf16),
            2 => Some(DType::I32),
            3 => Some(DType::Bool),
            _ => None,
        }
    }
}

/// One term of a shape, a grid or a bracket: a literal or an SSA value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    Const(u32),
    Value(ValueId),
}

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
///
/// `Row` folds the columns of each row into column 0 of that row, `Col` folds
/// the rows of each column into row 0 of that column, and `Scalar` folds the
/// whole tile into face 0's `[0,0]`. Everything else in the result tile is
/// zero: the pool writes only the folded values, and a pack out of a reduced
/// accumulator carries the packer's edge mask so the rest leaves as zeros.
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
    fn from_name(name: &str) -> Option<ReduceDim> {
        Some(match name {
            "row" => ReduceDim::Row,
            "col" => ReduceDim::Col,
            "scalar" => ReduceDim::Scalar,
            _ => return None,
        })
    }

    fn from_tag(tag: u8) -> Option<ReduceDim> {
        Some(match tag {
            0 => ReduceDim::Row,
            1 => ReduceDim::Col,
            2 => ReduceDim::Scalar,
            _ => return None,
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
    /// This core's logical coordinate on one grid axis.
    GridCoord,
    /// A register value.
    Register,
    /// A named `Dst` accumulator: mutable, one 32x32 tile.
    Acc,
}

/// Everything the compiler knows about one value.
///
/// A scalar parameter has [`RegShape::Scalar`]; any other parameter has
/// [`RegShape::Tile`], and the `Param` op carries its logical shape.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
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

impl BinOp {
    fn from_name(name: &str) -> Option<BinOp> {
        Some(match name {
            "add" => BinOp::Add,
            "sub" => BinOp::Sub,
            "mul" => BinOp::Mul,
            "div" => BinOp::Div,
            "pow" => BinOp::Pow,
            "mod" => BinOp::Mod,
            "cmplt" => BinOp::Cmplt,
            "cmpgt" => BinOp::Cmpgt,
            "cmpge" => BinOp::Cmpge,
            "max" => BinOp::Max,
            "or" => BinOp::Or,
            "and" => BinOp::And,
            "bitxor" => BinOp::BitXor,
            "bitor" => BinOp::BitOr,
            "bitand" => BinOp::BitAnd,
            "bitshiftleft" => BinOp::BitShiftLeft,
            "bitshiftright" => BinOp::BitShiftRight,
            "noteq" => BinOp::NotEq,
            "eq" => BinOp::Eq,
            _ => return None,
        })
    }

    fn from_tag(tag: u8) -> Option<BinOp> {
        Some(match tag {
            0 => BinOp::Add,
            1 => BinOp::Sub,
            2 => BinOp::Mul,
            3 => BinOp::Div,
            4 => BinOp::Pow,
            5 => BinOp::Mod,
            6 => BinOp::Cmplt,
            7 => BinOp::Cmpgt,
            8 => BinOp::Cmpge,
            9 => BinOp::Max,
            10 => BinOp::Or,
            11 => BinOp::And,
            12 => BinOp::BitXor,
            13 => BinOp::BitOr,
            14 => BinOp::BitAnd,
            15 => BinOp::BitShiftLeft,
            16 => BinOp::BitShiftRight,
            17 => BinOp::NotEq,
            18 => BinOp::Eq,
            _ => return None,
        })
    }

    /// A comparison: the result is a `bool`.
    pub const fn returns_bool(self) -> bool {
        matches!(
            self,
            BinOp::Cmplt | BinOp::Cmpgt | BinOp::Cmpge | BinOp::NotEq | BinOp::Eq
        )
    }

    /// The ops that only make sense on an integer dtype.
    pub const fn is_integer_only(self) -> bool {
        matches!(
            self,
            BinOp::Mod
                | BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::BitXor
                | BinOp::BitShiftLeft
                | BinOp::BitShiftRight
        )
    }

    /// The logical ops take and return a `bool`.
    pub const fn is_logical(self) -> bool {
        matches!(self, BinOp::Or | BinOp::And)
    }
}

/// A unary register op, the same set as zyx's `UOp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
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

impl UnOp {
    fn from_name(name: &str) -> Option<UnOp> {
        Some(match name {
            "neg" => UnOp::Neg,
            "not" => UnOp::Not,
            "bitnot" => UnOp::BitNot,
            "exp2" => UnOp::Exp2,
            "log2" => UnOp::Log2,
            "reciprocal" => UnOp::Reciprocal,
            "sqrt" => UnOp::Sqrt,
            "rsqrt" => UnOp::Rsqrt,
            "sin" => UnOp::Sin,
            "cos" => UnOp::Cos,
            "floor" => UnOp::Floor,
            "trunc" => UnOp::Trunc,
            "abs" => UnOp::Abs,
            _ => return None,
        })
    }

    fn from_tag(tag: u8) -> Option<UnOp> {
        Some(match tag {
            0 => UnOp::Neg,
            1 => UnOp::Not,
            2 => UnOp::BitNot,
            3 => UnOp::Exp2,
            4 => UnOp::Log2,
            5 => UnOp::Reciprocal,
            6 => UnOp::Sqrt,
            7 => UnOp::Rsqrt,
            8 => UnOp::Sin,
            9 => UnOp::Cos,
            10 => UnOp::Floor,
            11 => UnOp::Trunc,
            12 => UnOp::Abs,
            _ => return None,
        })
    }
}

/// One instruction of the unified program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// A literal. `bits` holds the two's-complement `i32` or the `f32` bits.
    Const {
        value: ValueId,
        dtype: DType,
        bits: u32,
    },
    /// A DRAM parameter. Empty `dims` is the scalar shape `0`.
    Param {
        value: ValueId,
        dtype: DType,
        dims: Vec<Dim>,
    },
    /// An L1 buffer: two constant dimensions.
    Buffer {
        value: ValueId,
        dtype: DType,
        shape: [u32; 2],
    },
    /// A named `Dst` accumulator: one 32x32 tile that `matmul` and `store_to`
    /// write, read back by `store_to` into a buffer. It is mutable — the whole
    /// point of it is that a K loop adds into the same rows — and it starts
    /// unspecified: a program that needs zeros stores them (`store_to %acc from
    /// %zero`). Every name is its own tile; a name is never redefined.
    Acc {
        value: ValueId,
        dtype: DType,
        shape: [u32; 2],
    },
    /// The launch grid, as a shape of two terms.
    Grid { columns: Dim, rows: Dim },
    /// `%i = loop %bound`.
    Loop { value: ValueId, bound: Dim },
    /// `endloop`.
    EndLoop,
    /// `%x = grid.x` / `%y = grid.y`: this core's logical coordinate on axis 0
    /// (x) or 1 (y), written by the host before the launch.
    GridCoord { value: ValueId, axis: u8 },
    /// DRAM into L1, or L1 into DRAM: exactly one side is a parameter, the other
    /// a buffer. Each side has its own datum offset, because a circular buffer
    /// names its page by the ring pointer while the parameter names the access's
    /// own index (zyx's `Op::Copy` carries `src_idx`/`dst_idx`). `shape` is the
    /// unit one base moves: `32,32` a tile, `1,32` a row, `32,1` a column, `1,1`
    /// one datum. A copy's unit has to be a whole NoC request, so it is a tile
    /// or a row and nothing narrower. A vector index on the parameter side names
    /// one base per datum, and the lanes fill the buffer's page: a `1,32` copy
    /// with 32 bases is 32 rows, one `32,32` page.
    CopyTo {
        dst: ValueId,
        src: ValueId,
        dst_offset: Dim,
        src_offset: Dim,
        shape: [u32; 2],
    },
    /// L1 buffer into a register value. `shape` is the unit the index's base
    /// names; the result has that shape.
    Load {
        value: ValueId,
        buffer: ValueId,
        offset: Dim,
        shape: [u32; 2],
    },
    /// A register value into an L1 buffer.
    StoreTo {
        value: ValueId,
        buffer: ValueId,
        offset: Dim,
        shape: [u32; 2],
    },
    /// The only conversion.
    Cast {
        value: ValueId,
        src: ValueId,
        dtype: DType,
    },
    /// The same bits under another name, between dtypes of equal width.
    Bitcast {
        value: ValueId,
        src: ValueId,
        dtype: DType,
    },
    /// One unary register op.
    Unary {
        value: ValueId,
        op: UnOp,
        a: ValueId,
    },
    /// One binary register op, with implicit broadcast.
    Binary {
        value: ValueId,
        op: BinOp,
        a: ValueId,
        b: ValueId,
    },
    /// Branchless select: `cond ? if_true : if_false`. `cond` is a `bool`; the
    /// two values have the same dtype and the result has their shape.
    Select {
        value: ValueId,
        cond: ValueId,
        if_true: ValueId,
        if_false: ValueId,
    },
    /// `a x b` accumulating into a named `Dst` tile. The operands are
    /// `SrcA`/`SrcB` tiles, multiplied and summed over the K faces into `acc`,
    /// which the op adds to rather than replaces: a K loop is one `matmul` under
    /// one `loop`, and two products in a row are two `matmul`s on one `acc`.
    /// The accumulator's dtype is the `acc`'s, so this op names no dtype.
    Matmul {
        acc: ValueId,
        a: ValueId,
        b: ValueId,
    },
    /// `sum <dim> %acc %x %scaler`: fold `x` with an add, into `acc`, scaled
    /// by the factors in `scaler` (tt-metal's `PoolType::SUM`).
    ///
    /// `x` and `scaler` are `load` pages, because the folding happens on the
    /// Src path: `Row` is the Matrix Unit's `x @ scaler` (tt-metal's split —
    /// `SUM` with `ReduceDim::ROW` is a matmul against the scaler's column of
    /// ones), and `Col`/`Scalar` are `GAPOOL`, whose scaler is its `SrcB`
    /// operand. The scaler's layout follows the hardware: column 0 for `Row`,
    /// row 0 otherwise.
    Sum {
        acc: ValueId,
        dim: ReduceDim,
        x: ValueId,
        scaler: ValueId,
    },
    /// `max <dim> %acc %x`: fold `x` with a maximum into `acc` (tt-metal's
    /// `PoolType::MAX`, `GMPOOL`).
    ///
    /// No scaler: `GMPOOL` applies only the exponent of a non-unity scaler
    /// (`exp2(floor(log2(|s|)))`), so tt-metal always reduces `MAX` with 1.0
    /// and scales the packed result afterwards with an SFPU multiply. The
    /// compiler supplies the 1.0 tile this op's `SrcB` still needs.
    Max {
        acc: ValueId,
        dim: ReduceDim,
        x: ValueId,
    },
}

impl Op {
    /// The value this op defines, if any.
    pub fn defined(&self) -> Option<ValueId> {
        match *self {
            Op::Const { value, .. }
            | Op::Param { value, .. }
            | Op::Buffer { value, .. }
            | Op::Acc { value, .. }
            | Op::Loop { value, .. }
            | Op::GridCoord { value, .. }
            | Op::Load { value, .. }
            | Op::Cast { value, .. }
            | Op::Bitcast { value, .. }
            | Op::Unary { value, .. }
            | Op::Binary { value, .. }
            | Op::Select { value, .. } => Some(value),
            Op::Grid { .. }
            | Op::EndLoop
            | Op::CopyTo { .. }
            | Op::StoreTo { .. }
            | Op::Matmul { .. }
            | Op::Sum { .. }
            | Op::Max { .. } => None,
        }
    }

    /// The values this op reads.
    pub fn used(&self) -> Vec<ValueId> {
        fn dim(d: &Dim) -> Option<ValueId> {
            match *d {
                Dim::Value(v) => Some(v),
                Dim::Const(_) => None,
            }
        }
        match self {
            Op::CopyTo {
                dst,
                src,
                dst_offset,
                src_offset,
                ..
            } => {
                let mut v = vec![*dst, *src];
                v.extend(dim(dst_offset));
                v.extend(dim(src_offset));
                v
            }
            Op::Load { buffer, offset, .. } => {
                let mut v = vec![*buffer];
                v.extend(dim(offset));
                v
            }
            Op::StoreTo {
                value,
                buffer,
                offset,
                ..
            } => {
                let mut v = vec![*value, *buffer];
                v.extend(dim(offset));
                v
            }
            Op::Cast { src, .. } | Op::Bitcast { src, .. } => vec![*src],
            Op::Unary { a, .. } => vec![*a],
            Op::Binary { a, b, .. } => vec![*a, *b],
            Op::Select {
                cond,
                if_true,
                if_false,
                ..
            } => vec![*cond, *if_true, *if_false],
            Op::Matmul { acc, a, b } => vec![*acc, *a, *b],
            Op::Sum { acc, x, scaler, .. } => vec![*acc, *x, *scaler],
            Op::Max { acc, x, .. } => vec![*acc, *x],
            Op::Loop { bound, .. } => dim(bound).into_iter().collect(),
            Op::Grid { columns, rows } => dim(columns).into_iter().chain(dim(rows)).collect(),
            Op::Param { dims, .. } => dims.iter().filter_map(dim).collect(),
            Op::Const { .. }
            | Op::Buffer { .. }
            | Op::Acc { .. }
            | Op::EndLoop
            | Op::GridCoord { .. } => Vec::new(),
        }
    }
}

/// A parsed kernel. [`Kernel::verify`] is the only thing that fills in what a
/// value means; a program that fails it must never reach the compiler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    pub ops: Vec<Op>,
    /// The name of every value, by id.
    pub names: Vec<String>,
}

/// The meaning of every value, by id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Types {
    pub values: Vec<ValueInfo>,
    /// The grid terms. A program must have them.
    pub grid: [Dim; 2],
}

impl Types {
    pub fn of(&self, value: ValueId) -> &ValueInfo {
        &self.values[value as usize]
    }
}

/// The crate's error: the one C ABI enum, [`crate::status::ttStatus`]. The
/// `Error` name is an alias for it, not a second error type and not a second
/// enum.
pub use crate::status::ttStatus as Error;

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Const(n) => write!(f, "{n}"),
            Dim::Value(v) => write!(f, "%{v}"),
        }
    }
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
pub fn unit_of(shape: [u32; 2], dtype: DType) -> Result<(RegShape, u32, u32), Error> {
    let [a, b] = shape;
    if a == 0 || b == 0 {
        return Err(Error::TT_ERROR_ZERO_DIMENSION);
    }
    let ok = |n: u32| n == 1 || n % 32 == 0;
    if !ok(a) || !ok(b) {
        return Err(Error::TT_ERROR_BUFFER_SHAPE);
    }
    let elements = a
        .checked_mul(b)
        .ok_or(Error::TT_ERROR_BUFFER_SIZE_OVERFLOW)?;
    let bytes = elements
        .checked_mul(dtype.bytes())
        .ok_or(Error::TT_ERROR_BUFFER_BYTES_OVERFLOW)?;
    let (unit, datums) = match (a, b) {
        (1, 1) => (RegShape::Scalar, 1),
        (1, _) => (RegShape::Row, 32),
        (_, 1) => (RegShape::Column, 32),
        _ => (RegShape::Tile, 1024),
    };
    Ok((unit, elements / datums, bytes))
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

fn lookup<'a>(
    values: &'a [ValueInfo],
    visible: &[bool],
    v: ValueId,
    _at: usize,
) -> Result<&'a ValueInfo, Error> {
    let info = values
        .get(v as usize)
        .ok_or(Error::TT_ERROR_VALUE_UNDEFINED)?;
    if !visible.get(v as usize).copied().unwrap_or(false) {
        return Err(Error::TT_ERROR_VALUE_OUT_OF_SCOPE);
    }
    Ok(info)
}

impl Kernel {
    /// The `i32` a value folds to, when it is a constant or constant-only
    /// arithmetic. This is how a `Dim::Value` that names a literal (`%two`,
    /// `%step`) becomes a `Dim::Const` without unrolling the loop that carries
    /// it; a value that depends on a loop index is not constant.
    ///
    /// The folds are exact two's-complement `i32`; a division by zero or a
    /// shift outside `0..32` is `None`, never a guess.
    pub fn const_i32(&self, value: ValueId) -> Option<i32> {
        let op = self.ops.iter().find(|op| op.defined() == Some(value))?;
        match *op {
            Op::Const {
                dtype: DType::I32,
                bits,
                ..
            } => Some(bits as i32),
            Op::Binary { op, a, b, .. } => {
                let a = self.const_i32(a)?;
                let b = self.const_i32(b)?;
                match op {
                    BinOp::Add => Some(a.wrapping_add(b)),
                    BinOp::Sub => Some(a.wrapping_sub(b)),
                    BinOp::Mul => Some(a.wrapping_mul(b)),
                    BinOp::Div => a.checked_div(b),
                    BinOp::Mod => a.checked_rem(b),
                    BinOp::Pow => a.checked_pow(u32::try_from(b).ok()?),
                    BinOp::Max => Some(a.max(b)),
                    BinOp::BitAnd => Some(a & b),
                    BinOp::BitOr => Some(a | b),
                    BinOp::BitXor => Some(a ^ b),
                    BinOp::BitShiftLeft => a.checked_shl(u32::try_from(b).ok()?),
                    BinOp::BitShiftRight => a.checked_shr(u32::try_from(b).ok()?),
                    BinOp::Cmplt
                    | BinOp::Cmpgt
                    | BinOp::Cmpge
                    | BinOp::Eq
                    | BinOp::NotEq
                    | BinOp::Or
                    | BinOp::And => None,
                }
            }
            _ => None,
        }
    }

    /// Check every reference, dtype, shape and scope, and return what each
    /// value means.
    pub fn verify(&self) -> Result<Types, Error> {
        let mut values: Vec<ValueInfo> = Vec::with_capacity(self.names.len());
        let mut visible: Vec<bool> = Vec::with_capacity(self.names.len());
        let mut loops: Vec<usize> = Vec::new();
        let mut grid: Option<([Dim; 2], usize)> = None;
        let mut seen: Vec<&str> = Vec::with_capacity(self.names.len());
        for name in &self.names {
            if seen.contains(&name.as_str()) {
                return Err(Error::TT_ERROR_NAME_DUPLICATE);
            }
            seen.push(name);
        }
        // The op that defines each value. Value ids are definition order, not op
        // indices: a `grid`, a `copy_to` or an `endloop` takes an op slot and
        // defines nothing.
        let mut def: Vec<usize> = vec![usize::MAX; self.names.len()];
        for (at, op) in self.ops.iter().enumerate() {
            if let Some(value) = op.defined() {
                def[value as usize] = at;
            }
        }

        for (at, op) in self.ops.iter().enumerate() {
            for v in op.used() {
                lookup(&values, &visible, v, at)?;
            }
            if let Some(value) = op.defined() {
                if value as usize != values.len() {
                    return Err(Error::TT_ERROR_VALUE_ORDER);
                }
                if self.names.get(value as usize).is_none() {
                    return Err(Error::TT_ERROR_VALUE_UNNAMED);
                }
            }
            let info = match *op {
                Op::Const { dtype, bits, .. } => {
                    if dtype == DType::Bf16 {
                        return Err(Error::TT_ERROR_CONST_BF16);
                    }
                    if dtype == DType::Bool {
                        return Err(Error::TT_ERROR_CONST_BOOL);
                    }
                    let _ = bits;
                    ValueInfo {
                        kind: ValueKind::Const,
                        dtype,
                        shape: RegShape::Scalar,
                    }
                }
                Op::Param {
                    dtype, ref dims, ..
                } => {
                    if dtype == DType::Bool {
                        todo!("op {at}: a bool parameter is not implemented");
                    }
                    for (k, dim) in dims.iter().enumerate() {
                        match *dim {
                            Dim::Const(0) => {
                                let _ = k;
                                return Err(Error::TT_ERROR_ZERO_DIMENSION);
                            }
                            Dim::Const(_) => {}
                            Dim::Value(v) => {
                                let i = lookup(&values, &visible, v, at)?;
                                if i.dtype != INDEX || i.shape != RegShape::Scalar {
                                    let _ = (k, i);
                                    return Err(Error::TT_ERROR_DIMENSION_NOT_INDEX);
                                }
                            }
                        }
                    }
                    ValueInfo {
                        kind: ValueKind::Param,
                        dtype,
                        // A tensor parameter's logical shape lives in the op;
                        // only the scalar/tensor distinction matters here.
                        shape: if dims.is_empty() {
                            RegShape::Scalar
                        } else {
                            RegShape::Tile
                        },
                    }
                }
                Op::Buffer { dtype, shape, .. } => {
                    if dtype == DType::Bool {
                        todo!("op {at}: a bool buffer is not implemented");
                    }
                    unit_of(shape, dtype)?;
                    ValueInfo {
                        kind: ValueKind::Buffer,
                        dtype,
                        shape: RegShape::Tile,
                    }
                }
                Op::Acc { dtype, shape, .. } => {
                    if shape != [32, 32] {
                        let _ = (at, shape);
                        return Err(Error::TT_ERROR_ACC_SHAPE);
                    }
                    if !matches!(dtype, DType::F32 | DType::Bf16) {
                        let _ = (at, dtype);
                        return Err(Error::TT_ERROR_ACC_DTYPE);
                    }
                    ValueInfo {
                        kind: ValueKind::Acc,
                        dtype,
                        shape: RegShape::Tile,
                    }
                }
                Op::Grid { columns, rows } => {
                    if grid.is_some() {
                        return Err(Error::TT_ERROR_GRID);
                    }
                    grid = Some(([columns, rows], at));
                    continue;
                }
                Op::GridCoord { axis, .. } => {
                    if axis > 1 {
                        let _ = (at, axis);
                        return Err(Error::TT_ERROR_GRID_AXIS);
                    }
                    ValueInfo {
                        kind: ValueKind::GridCoord,
                        dtype: INDEX,
                        shape: RegShape::Scalar,
                    }
                }
                Op::Loop { bound, .. } => {
                    if let Dim::Value(v) = bound {
                        let i = lookup(&values, &visible, v, at)?;
                        if i.dtype != INDEX || i.shape != RegShape::Scalar {
                            let _ = i;
                            return Err(Error::TT_ERROR_LOOP_BOUND_NOT_INDEX);
                        }
                    }
                    loops.push(values.len());
                    values.push(ValueInfo {
                        kind: ValueKind::Induction,
                        dtype: INDEX,
                        shape: RegShape::Scalar,
                    });
                    visible.push(true);
                    continue;
                }
                Op::EndLoop => {
                    let Some(start) = loops.pop() else {
                        return Err(Error::TT_ERROR_ENDLOOP_WITHOUT_LOOP);
                    };
                    for v in start..values.len() {
                        visible[v] = false;
                    }
                    continue;
                }
                Op::CopyTo {
                    dst,
                    src,
                    dst_offset,
                    src_offset,
                    shape: access,
                } => {
                    let d = *lookup(&values, &visible, dst, at)?;
                    let s = *lookup(&values, &visible, src, at)?;
                    let (buffer, param, buffer_offset, param_offset) = match (d.kind, s.kind) {
                        (ValueKind::Buffer, ValueKind::Param) => (dst, src, dst_offset, src_offset),
                        (ValueKind::Param, ValueKind::Buffer) => (src, dst, src_offset, dst_offset),
                        _ => {
                            let _ = (kind_name(d.kind), kind_name(s.kind));
                            return Err(Error::TT_ERROR_COPY_ENDS);
                        }
                    };
                    let b = *lookup(&values, &visible, buffer, at)?;
                    let p = *lookup(&values, &visible, param, at)?;
                    if p.shape == RegShape::Scalar {
                        let _ = param;
                        return Err(Error::TT_ERROR_COPY_SCALAR_PARAM);
                    }
                    if b.dtype != p.dtype {
                        let _ = (b.dtype, p.dtype);
                        return Err(Error::TT_ERROR_COPY_CAST);
                    }
                    // A copy's unit is one NoC request, so it is a row or a
                    // tile: a datum or a column is not 64 bytes.
                    let unit = access_shape(access)?;
                    if !matches!(unit, RegShape::Row | RegShape::Tile) {
                        return Err(Error::TT_ERROR_COPY_UNIT);
                    }
                    // The parameter side names one base per datum of its index,
                    // so a vector index is one request per lane; the L1 side
                    // names the ring's page, which is one.
                    let lanes = check_index(
                        &values,
                        &visible,
                        param_offset,
                        at,
                        Error::TT_ERROR_OFFSET_NOT_INDEX,
                    )?;
                    let page = check_index(
                        &values,
                        &visible,
                        buffer_offset,
                        at,
                        Error::TT_ERROR_OFFSET_NOT_INDEX,
                    )?;
                    if page != 1 {
                        return Err(Error::TT_TODO_MULTI_LANE);
                    }
                    let Op::Buffer { shape, .. } = self.ops[def[buffer as usize]] else {
                        let _ = buffer;
                        return Err(Error::TT_ERROR_NOT_A_BUFFER);
                    };
                    let (unit_shape, _, _) = unit_of(shape, b.dtype)?;
                    let datums = access[0] * access[1];
                    // One lane moves one unit, and the lanes fill exactly one
                    // L1 page: `page_datums / lanes` datums each.
                    if lanes.checked_mul(datums) != Some(unit_shape.datums()) {
                        let _ = (lanes, datums, unit_shape);
                        return Err(Error::TT_ERROR_COPY_LANES);
                    }
                    let total = shape[0] * shape[1];
                    check_offset(buffer_offset, total, datums, at)?;
                    if let Some(count) = param_elements(self, &def, param) {
                        check_offset(param_offset, count, datums, at)?;
                    }
                    continue;
                }
                Op::Load {
                    buffer,
                    offset,
                    value,
                    shape: access,
                } => {
                    let b = *lookup(&values, &visible, buffer, at)?;
                    let unit = access_shape(access)?;
                    if check_index(
                        &values,
                        &visible,
                        offset,
                        at,
                        Error::TT_ERROR_OFFSET_NOT_INDEX,
                    )? != 1
                    {
                        return Err(Error::TT_TODO_MULTI_LANE);
                    }
                    let datums = access[0] * access[1];
                    if b.kind == ValueKind::Param {
                        // A direct parameter access: one tile, which
                        // `mover::add_movers` stages through a `32,32` buffer.
                        if unit != RegShape::Tile || b.shape != RegShape::Tile {
                            let _ = (buffer, access);
                            return Err(Error::TT_ERROR_LOAD_PARAM_NOT_TILE);
                        }
                        if let Some(count) = param_elements(self, &def, buffer) {
                            check_offset(offset, count, 1024, at)?;
                        }
                        let _ = value;
                        ValueInfo {
                            kind: ValueKind::Register,
                            dtype: b.dtype,
                            shape: RegShape::Tile,
                        }
                    } else {
                        if b.kind != ValueKind::Buffer {
                            let _ = buffer;
                            return Err(Error::TT_ERROR_NOT_A_BUFFER);
                        }
                        let Op::Buffer { shape, .. } = self.ops[def[buffer as usize]] else {
                            let _ = buffer;
                            return Err(Error::TT_ERROR_NOT_A_BUFFER);
                        };
                        let (unit_shape, _, _) = unit_of(shape, b.dtype)?;
                        if unit_shape.datums() < datums {
                            let _ = (unit_shape, access);
                            return Err(Error::TT_ERROR_ACCESS_SHAPE);
                        }
                        let total = shape[0] * shape[1];
                        check_offset(offset, total, datums, at)?;
                        let _ = value;
                        ValueInfo {
                            kind: ValueKind::Register,
                            dtype: b.dtype,
                            shape: unit,
                        }
                    }
                }
                Op::StoreTo {
                    value,
                    buffer,
                    offset,
                    shape: access,
                } => {
                    let v = *lookup(&values, &visible, value, at)?;
                    let b = *lookup(&values, &visible, buffer, at)?;
                    let unit = access_shape(access)?;
                    if check_index(
                        &values,
                        &visible,
                        offset,
                        at,
                        Error::TT_ERROR_OFFSET_NOT_INDEX,
                    )? != 1
                    {
                        return Err(Error::TT_TODO_MULTI_LANE);
                    }
                    let datums = access[0] * access[1];
                    if b.kind == ValueKind::Acc {
                        // An accumulator is one tile and is written in place: the
                        // only form implemented is a fill from an f32 constant,
                        // which is how a program zeroes it.
                        if unit != RegShape::Tile {
                            let _ = access;
                            return Err(Error::TT_ERROR_ACC_SHAPE);
                        }
                        if !matches!(offset, Dim::Const(0)) {
                            let _ = offset;
                            return Err(Error::TT_ERROR_ACC_OFFSET);
                        }
                        if v.kind != ValueKind::Const || v.dtype != DType::F32 {
                            let _ = (value, v.kind, v.dtype);
                            return Err(Error::TT_TODO_ACC_STORE);
                        }
                        continue;
                    }
                    if v.kind != ValueKind::Register && v.kind != ValueKind::Acc {
                        let _ = value;
                        return Err(Error::TT_ERROR_STORE_VALUE_NOT_REGISTER);
                    }
                    if b.kind == ValueKind::Param {
                        // The direct form; `mover::add_movers` stages it.
                        if unit != RegShape::Tile || b.shape != RegShape::Tile {
                            let _ = (buffer, access);
                            return Err(Error::TT_ERROR_STORE_PARAM_NOT_TILE);
                        }
                        if v.dtype != b.dtype {
                            let _ = (v.dtype, b.dtype);
                            return Err(Error::TT_ERROR_STORE_PARAM_CAST);
                        }
                        if v.shape != RegShape::Tile {
                            let _ = v.shape;
                            return Err(Error::TT_ERROR_STORE_PARAM_VALUE_NOT_TILE);
                        }
                        if let Some(count) = param_elements(self, &def, buffer) {
                            check_offset(offset, count, 1024, at)?;
                        }
                        continue;
                    }
                    if b.kind != ValueKind::Buffer {
                        let _ = buffer;
                        return Err(Error::TT_ERROR_NOT_A_BUFFER);
                    }
                    if v.dtype != b.dtype {
                        let _ = (v.dtype, b.dtype);
                        return Err(Error::TT_ERROR_STORE_BUFFER_CAST);
                    }
                    let Op::Buffer { shape, .. } = self.ops[def[buffer as usize]] else {
                        let _ = buffer;
                        return Err(Error::TT_ERROR_NOT_A_BUFFER);
                    };
                    let (unit_shape, _, _) = unit_of(shape, b.dtype)?;
                    if unit_shape.datums() < datums {
                        let _ = (unit_shape, access);
                        return Err(Error::TT_ERROR_ACCESS_SHAPE);
                    }
                    if v.shape != unit {
                        let _ = (unit, v.shape);
                        return Err(Error::TT_ERROR_STORE_BUFFER_SHAPE);
                    }
                    let total = shape[0] * shape[1];
                    check_offset(offset, total, datums, at)?;
                    continue;
                }
                Op::Cast { src, dtype, .. } => {
                    let s = *lookup(&values, &visible, src, at)?;
                    if !is_value(s.kind) {
                        let _ = src;
                        return Err(Error::TT_ERROR_CAST_SOURCE);
                    }
                    // Every pair converts, C-style (`(int)f`, `(bool)f`,
                    // `__float2bfloat16`): a scalar of one dtype to another, in
                    // the lane, with the rounding the C type has.
                    ValueInfo {
                        kind: ValueKind::Register,
                        dtype,
                        shape: s.shape,
                    }
                }
                Op::Bitcast { src, dtype, .. } => {
                    let s = *lookup(&values, &visible, src, at)?;
                    if !is_value(s.kind) {
                        let _ = src;
                        return Err(Error::TT_ERROR_CAST_SOURCE);
                    }
                    // A reinterpretation, not a conversion: the bits are the
                    // value, so the two dtypes must be the same width.
                    if s.dtype.bytes() != dtype.bytes() {
                        let _ = (s.dtype, dtype);
                        return Err(Error::TT_ERROR_BITCAST_WIDTH);
                    }
                    ValueInfo {
                        kind: ValueKind::Register,
                        dtype,
                        shape: s.shape,
                    }
                }
                Op::Unary { op, a, .. } => {
                    let x = *lookup(&values, &visible, a, at)?;
                    if !is_index_operand(x) {
                        let _ = a;
                        return Err(Error::TT_ERROR_UNARY_OPERAND);
                    }
                    let dtype = match op {
                        UnOp::Not => {
                            if x.dtype != DType::Bool {
                                let _ = x.dtype;
                                return Err(Error::TT_ERROR_UNARY_NOT_BOOL);
                            }
                            DType::Bool
                        }
                        UnOp::BitNot => {
                            if x.dtype != DType::I32 {
                                let _ = x.dtype;
                                return Err(Error::TT_ERROR_UNARY_BITNOT_NOT_I32);
                            }
                            DType::I32
                        }
                        _ => {
                            if !x.dtype.is_float() {
                                let _ = (op, x.dtype);
                                return Err(Error::TT_ERROR_UNARY_NOT_FLOAT);
                            }
                            x.dtype
                        }
                    };
                    ValueInfo {
                        kind: ValueKind::Register,
                        dtype,
                        shape: x.shape,
                    }
                }
                Op::Binary { op, a, b, .. } => {
                    let x = *lookup(&values, &visible, a, at)?;
                    let y = *lookup(&values, &visible, b, at)?;
                    if !is_index_operand(x) || !is_index_operand(y) {
                        let _ = (a, b);
                        return Err(Error::TT_ERROR_BINARY_OPERANDS);
                    }
                    if x.dtype != y.dtype {
                        let _ = op;
                        return Err(Error::TT_ERROR_NO_IMPLICIT_CAST);
                    }
                    if op.is_logical() && x.dtype != DType::Bool {
                        let _ = (op, x.dtype);
                        return Err(Error::TT_ERROR_BINARY_LOGICAL_NOT_BOOL);
                    }
                    if op.is_integer_only() && x.dtype != INDEX {
                        let _ = (op, x.dtype);
                        return Err(Error::TT_ERROR_BINARY_NOT_INTEGER);
                    }
                    if !op.is_logical() && x.dtype == DType::Bool {
                        let _ = op;
                        return Err(Error::TT_ERROR_BINARY_BOOL);
                    }
                    let shape = broadcast(x.shape, y.shape).ok_or_else(|| {
                        let _ = (x.shape, y.shape);
                        Error::TT_ERROR_BINARY_BROADCAST
                    })?;
                    let dtype = if op.returns_bool() {
                        DType::Bool
                    } else {
                        x.dtype
                    };
                    ValueInfo {
                        kind: ValueKind::Register,
                        dtype,
                        shape,
                    }
                }
                Op::Select {
                    cond,
                    if_true,
                    if_false,
                    ..
                } => {
                    let c = *lookup(&values, &visible, cond, at)?;
                    let t = *lookup(&values, &visible, if_true, at)?;
                    let f = *lookup(&values, &visible, if_false, at)?;
                    if !is_value(c.kind) || !is_value(t.kind) || !is_value(f.kind) {
                        let _ = (cond, if_true, if_false);
                        return Err(Error::TT_ERROR_SELECT_OPERANDS);
                    }
                    if c.dtype != DType::Bool {
                        let _ = cond;
                        return Err(Error::TT_ERROR_CONDITION_NOT_BOOL);
                    }
                    if t.dtype != f.dtype {
                        let _ = (t.dtype, f.dtype);
                        return Err(Error::TT_ERROR_SELECT_CAST);
                    }
                    let shape = broadcast(t.shape, f.shape).ok_or_else(|| {
                        let _ = (t.shape, f.shape);
                        Error::TT_ERROR_SELECT_BROADCAST
                    })?;
                    let shape = broadcast(shape, c.shape).ok_or_else(|| {
                        let _ = (c.shape, shape);
                        Error::TT_ERROR_SELECT_CONDITION_BROADCAST
                    })?;
                    ValueInfo {
                        kind: ValueKind::Register,
                        dtype: t.dtype,
                        shape,
                    }
                }
                Op::Matmul { acc, a, b } => {
                    let c = *lookup(&values, &visible, acc, at)?;
                    let x = *lookup(&values, &visible, a, at)?;
                    let y = *lookup(&values, &visible, b, at)?;
                    if c.kind != ValueKind::Acc {
                        let _ = (at, acc);
                        return Err(Error::TT_ERROR_MATMUL_ACC);
                    }
                    if !is_value(x.kind) || !is_value(y.kind) {
                        let _ = (a, b);
                        return Err(Error::TT_ERROR_MATMUL_OPERANDS);
                    }
                    if x.shape != RegShape::Tile || y.shape != RegShape::Tile {
                        let _ = (x.shape, y.shape);
                        return Err(Error::TT_ERROR_MATMUL_NOT_TILES);
                    }
                    if x.dtype != y.dtype {
                        let _ = (x.dtype, y.dtype);
                        return Err(Error::TT_ERROR_MATMUL_CAST);
                    }
                    // The Matrix Unit's Src path takes `f32` (TF32 in Src) and
                    // `bf16` only, and the accumulator is the `Dst` width: a
                    // 32-bit `Src` cannot accumulate into a 16-bit `Dst`.
                    let ok_operand = matches!(x.dtype, DType::F32 | DType::Bf16);
                    let ok_accum = matches!(c.dtype, DType::F32 | DType::Bf16)
                        && !(x.dtype == DType::F32 && c.dtype == DType::Bf16);
                    if !ok_operand || !ok_accum {
                        let _ = (x.dtype, c.dtype);
                        return Err(Error::TT_ERROR_MATMUL_DTYPE);
                    }
                    // The accumulator is written in place: no value is defined.
                    continue;
                }
                ref reduce @ (Op::Sum { .. } | Op::Max { .. }) => {
                    let (acc, x, scaler) = match *reduce {
                        Op::Sum { acc, x, scaler, .. } => (acc, x, Some(scaler)),
                        Op::Max { acc, x, .. } => (acc, x, None),
                        _ => unreachable!("the arm pattern is the two reduce ops"),
                    };
                    let c = *lookup(&values, &visible, acc, at)?;
                    if c.kind != ValueKind::Acc {
                        let _ = (at, acc);
                        return Err(Error::TT_ERROR_REDUCE_ACC);
                    }
                    // The folding happens on the Src path, from an L1 page:
                    // both operands must be `load`s, as a `matmul`'s are.
                    let loaded = |value: ValueId| -> bool {
                        matches!(self.ops.get(def[value as usize]), Some(Op::Load { .. }))
                    };
                    let v = *lookup(&values, &visible, x, at)?;
                    if !is_value(v.kind) || v.shape != RegShape::Tile || !loaded(x) {
                        let _ = (at, x);
                        return Err(Error::TT_ERROR_REDUCE_OPERAND);
                    }
                    if !matches!(v.dtype, DType::F32 | DType::Bf16)
                        || !matches!(c.dtype, DType::F32 | DType::Bf16)
                        || (v.dtype == DType::F32 && c.dtype == DType::Bf16)
                    {
                        let _ = (v.dtype, c.dtype);
                        return Err(Error::TT_ERROR_REDUCE_DTYPE);
                    }
                    if let Some(scaler) = scaler {
                        let s = *lookup(&values, &visible, scaler, at)?;
                        if !is_value(s.kind) || s.shape != RegShape::Tile || !loaded(scaler) {
                            let _ = (at, scaler);
                            return Err(Error::TT_ERROR_REDUCE_OPERAND);
                        }
                        if s.dtype != v.dtype {
                            let _ = (s.dtype, v.dtype);
                            return Err(Error::TT_ERROR_REDUCE_DTYPE);
                        }
                    }
                    // The accumulator is written in place: no value is defined.
                    continue;
                }
            };
            values.push(info);
            visible.push(true);
        }
        if !loops.is_empty() {
            return Err(Error::TT_ERROR_LOOP_UNTERMINATED);
        }
        if values.len() != self.names.len() {
            return Err(Error::TT_ERROR_VALUE_COUNT);
        }
        let Some((grid, at)) = grid else {
            return Err(Error::TT_ERROR_GRID);
        };
        check_dim(
            &values,
            &visible,
            grid[0],
            at,
            Error::TT_ERROR_GRID_DIMENSION_NOT_INDEX,
        )?;
        check_dim(
            &values,
            &visible,
            grid[1],
            at,
            Error::TT_ERROR_GRID_DIMENSION_NOT_INDEX,
        )?;
        Ok(Types { values, grid })
    }
}

/// A value an arithmetic op can read: a computed register value, a literal or a
/// loop induction value. Parameters and buffers are memory, not operands.
fn is_value(kind: ValueKind) -> bool {
    matches!(
        kind,
        ValueKind::Register | ValueKind::Const | ValueKind::Induction
    )
}

/// Can this value be an operand of index arithmetic? A register value, a
/// literal, an induction value or a grid coordinate; a scalar parameter is
/// index arithmetic too, while a tensor parameter is L1 and must be `load`ed.
fn is_index_operand(info: ValueInfo) -> bool {
    is_value(info.kind)
        || info.kind == ValueKind::GridCoord
        || (info.kind == ValueKind::Param && info.shape == RegShape::Scalar)
}

fn kind_name(kind: ValueKind) -> &'static str {
    match kind {
        ValueKind::Param => "parameter",
        ValueKind::Buffer => "buffer",
        ValueKind::Const => "constant",
        ValueKind::Induction => "induction value",
        ValueKind::GridCoord => "grid coordinate",
        ValueKind::Register => "register value",
        ValueKind::Acc => "accumulator",
    }
}

fn broadcast(a: RegShape, b: RegShape) -> Option<RegShape> {
    use RegShape::*;
    Some(match (a, b) {
        (x, y) if x == y => x,
        (Scalar, y) => y,
        (x, Scalar) => x,
        (Tile, Row | Column) => Tile,
        (Row | Column, Tile) => Tile,
        _ => return None,
    })
}

/// A shape term that must name an i32 scalar: a literal, or a visible value.
///
/// `bad` is the variant for the concrete context the term appears in (a grid
/// dimension, an offset, ...), so a caller can tell them apart.
fn check_dim(
    values: &[ValueInfo],
    visible: &[bool],
    dim: Dim,
    at: usize,
    bad: Error,
) -> Result<(), Error> {
    if let Dim::Value(v) = dim {
        let i = lookup(values, visible, v, at)?;
        if i.dtype != INDEX || i.shape != RegShape::Scalar {
            let _ = i;
            return Err(bad);
        }
    }
    Ok(())
}

/// An access shape: the unit one base moves, and nothing between the four.
pub(crate) fn access_shape(shape: [u32; 2]) -> Result<RegShape, Error> {
    Ok(match shape {
        [1, 1] => RegShape::Scalar,
        [1, 32] => RegShape::Row,
        [32, 1] => RegShape::Column,
        [32, 32] => RegShape::Tile,
        _ => return Err(Error::TT_ERROR_ACCESS_SHAPE),
    })
}

/// An access's index, and how many bases it names. A literal is one base; a
/// value is one base per datum of its shape, so a `1,32` vector names 32.
fn check_index(
    values: &[ValueInfo],
    visible: &[bool],
    dim: Dim,
    at: usize,
    bad: Error,
) -> Result<u32, Error> {
    let Dim::Value(v) = dim else {
        return Ok(1);
    };
    let i = lookup(values, visible, v, at)?;
    if i.dtype != INDEX {
        let _ = i;
        return Err(bad);
    }
    Ok(lanes(i.shape))
}

/// How many bases an index of this shape names: one per datum.
pub(crate) fn lanes(shape: RegShape) -> u32 {
    match shape {
        RegShape::Scalar => 1,
        RegShape::Row | RegShape::Column => 32,
        RegShape::Tile => 1024,
    }
}

/// A bracket must leave room for one unit. A symbolic offset cannot be checked
/// here; it is checked at run time.
fn check_offset(offset: Dim, total: u32, unit: u32, at: usize) -> Result<(), Error> {
    if let Dim::Const(n) = offset {
        if n.checked_add(unit).is_none_or(|end| end > total) {
            let _ = (n, unit, total, at);
            return Err(Error::TT_ERROR_OFFSET);
        }
    }
    Ok(())
}

/// A parameter's element count when every dimension is a literal.
fn param_elements(k: &Kernel, def: &[usize], param: ValueId) -> Option<u32> {
    let Op::Param { dims, .. } = &k.ops[def[param as usize]] else {
        return None;
    };
    if dims.is_empty() {
        return None;
    }
    dims.iter().try_fold(1u32, |acc, d| match *d {
        Dim::Const(n) => acc.checked_mul(n),
        Dim::Value(_) => None,
    })
}

// ---------------------------------------------------------------------------
// Byte form
// ---------------------------------------------------------------------------

/// The byte form's magic.
pub const MAGIC: [u8; 4] = *b"TTIR";
/// The IR version this build reads.
pub const VERSION: u32 = 8;

impl Kernel {
    /// Read the byte form. A wrong magic, an unknown version or a truncated
    /// record is refused, never guessed at.
    pub fn from_bytes(bytes: &[u8]) -> Result<Kernel, Error> {
        let mut r = Cursor { at: 0, bytes };
        if r.take(4)? != MAGIC {
            return Err(Error::TT_ERROR_NOT_TTIR);
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(Error::TT_ERROR_VERSION);
        }
        let count = r.u32()? as usize;
        let mut ops = Vec::with_capacity(count);
        let mut names: Vec<String> = Vec::new();
        for _ in 0..count {
            let tag = r.u8()?;
            let op = match tag {
                0 => {
                    let name = r.name()?;
                    let dtype = r.dtype()?;
                    let bits = r.u32()?;
                    let value = define(&mut names, name)?;
                    Op::Const { value, dtype, bits }
                }
                1 => {
                    let name = r.name()?;
                    let dtype = r.dtype()?;
                    let n = r.u32()? as usize;
                    let mut dims = Vec::with_capacity(n);
                    for _ in 0..n {
                        dims.push(r.dim()?);
                    }
                    let value = define(&mut names, name)?;
                    Op::Param { value, dtype, dims }
                }
                2 => {
                    let name = r.name()?;
                    let dtype = r.dtype()?;
                    let shape = [r.u32()?, r.u32()?];
                    let value = define(&mut names, name)?;
                    Op::Buffer {
                        value,
                        dtype,
                        shape,
                    }
                }
                3 => Op::Grid {
                    columns: r.dim()?,
                    rows: r.dim()?,
                },
                4 => {
                    let name = r.name()?;
                    let bound = r.dim()?;
                    let value = define(&mut names, name)?;
                    Op::Loop { value, bound }
                }
                5 => Op::EndLoop,
                14 => {
                    let name = r.name()?;
                    let axis = r.u8()?;
                    let value = define(&mut names, name)?;
                    Op::GridCoord { value, axis }
                }
                6 => Op::CopyTo {
                    dst: r.u32()?,
                    src: r.u32()?,
                    dst_offset: r.dim()?,
                    src_offset: r.dim()?,
                    shape: [r.u32()?, r.u32()?],
                },
                7 => {
                    let name = r.name()?;
                    let buffer = r.u32()?;
                    let offset = r.dim()?;
                    let shape = [r.u32()?, r.u32()?];
                    let value = define(&mut names, name)?;
                    Op::Load {
                        value,
                        buffer,
                        offset,
                        shape,
                    }
                }
                8 => Op::StoreTo {
                    buffer: r.u32()?,
                    value: r.u32()?,
                    offset: r.dim()?,
                    shape: [r.u32()?, r.u32()?],
                },
                9 => {
                    let name = r.name()?;
                    let src = r.u32()?;
                    let dtype = r.dtype()?;
                    let value = define(&mut names, name)?;
                    Op::Cast { value, src, dtype }
                }
                15 => {
                    let name = r.name()?;
                    let src = r.u32()?;
                    let dtype = r.dtype()?;
                    let value = define(&mut names, name)?;
                    Op::Bitcast { value, src, dtype }
                }
                10 => {
                    let name = r.name()?;
                    let op = UnOp::from_tag(r.u8()?).ok_or(Error::TT_ERROR_UNARY_TAG)?;
                    let a = r.u32()?;
                    let value = define(&mut names, name)?;
                    Op::Unary { value, op, a }
                }
                11 => {
                    let name = r.name()?;
                    let op = BinOp::from_tag(r.u8()?).ok_or(Error::TT_ERROR_BINARY_TAG)?;
                    let a = r.u32()?;
                    let b = r.u32()?;
                    let value = define(&mut names, name)?;
                    Op::Binary { value, op, a, b }
                }
                12 => {
                    let acc = r.u32()?;
                    let a = r.u32()?;
                    let b = r.u32()?;
                    Op::Matmul { acc, a, b }
                }
                16 => {
                    let name = r.name()?;
                    let dtype = r.dtype()?;
                    let shape = [r.u32()?, r.u32()?];
                    let value = define(&mut names, name)?;
                    Op::Acc {
                        value,
                        dtype,
                        shape,
                    }
                }
                13 => {
                    let name = r.name()?;
                    let cond = r.u32()?;
                    let if_true = r.u32()?;
                    let if_false = r.u32()?;
                    let value = define(&mut names, name)?;
                    Op::Select {
                        value,
                        cond,
                        if_true,
                        if_false,
                    }
                }
                17 => {
                    let acc = r.u32()?;
                    let x = r.u32()?;
                    let scaler = r.u32()?;
                    let dim =
                        ReduceDim::from_tag(r.u8()?).ok_or(Error::TT_ERROR_INSTRUCTION_TAG)?;
                    Op::Sum {
                        acc,
                        dim,
                        x,
                        scaler,
                    }
                }
                18 => {
                    let acc = r.u32()?;
                    let x = r.u32()?;
                    let dim =
                        ReduceDim::from_tag(r.u8()?).ok_or(Error::TT_ERROR_INSTRUCTION_TAG)?;
                    Op::Max { acc, dim, x }
                }
                other => {
                    let _ = other;
                    return Err(Error::TT_ERROR_INSTRUCTION_TAG);
                }
            };
            ops.push(op);
        }
        if r.at != r.bytes.len() {
            return Err(Error::TT_ERROR_TRAILING_INPUT);
        }
        let k = Kernel { ops, names };
        k.verify()?;
        Ok(k)
    }
}

fn define(names: &mut Vec<String>, name: String) -> Result<ValueId, Error> {
    if names.iter().any(|n| *n == name) {
        return Err(Error::TT_ERROR_NAME_DUPLICATE);
    }
    let value = names.len() as ValueId;
    names.push(name);
    Ok(value)
}

struct Cursor<'a> {
    at: usize,
    bytes: &'a [u8],
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], Error> {
        let end = self.at + n;
        if end > self.bytes.len() {
            return Err(Error::TT_ERROR_TRUNCATED);
        }
        let s = &self.bytes[self.at..end];
        self.at = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn dtype(&mut self) -> Result<DType, Error> {
        let tag = self.u8()?;
        DType::from_tag(tag).ok_or(Error::TT_ERROR_DTYPE_TAG)
    }

    fn name(&mut self) -> Result<String, Error> {
        let n = self.u32()? as usize;
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::TT_ERROR_NAME_UTF8)
    }

    fn dim(&mut self) -> Result<Dim, Error> {
        match self.u8()? {
            0 => Ok(Dim::Const(self.u32()?)),
            1 => Ok(Dim::Value(self.u32()?)),
            other => {
                let _ = other;
                Err(Error::TT_ERROR_SHAPE_TERM_TAG)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Text form
// ---------------------------------------------------------------------------

impl Kernel {
    /// Parse the text form. Line-oriented; `//` starts a comment.
    pub fn parse(text: &str) -> Result<Kernel, Error> {
        let mut p = Parser::default();
        let mut version: Option<u32> = None;
        for (line_no, raw) in text.lines().enumerate() {
            let at = line_no + 1;
            let line = match raw.find("//") {
                Some(i) => &raw[..i],
                None => raw,
            }
            .trim();
            if line.is_empty() {
                continue;
            }
            let tokens = tokenize(line).map_err(|e| {
                let _ = e;
                Error::TT_ERROR_PARSE
            })?;
            let mut t = Tokens {
                tok: &tokens,
                at: 0,
                line: at,
            };
            if matches!(t.peek(), Some(Token::Name(_))) {
                let op = parse_definition(&mut t, &mut p)?;
                p.ops.push(op);
                continue;
            }
            let first = t.ident()?;
            if first == "version" {
                if version.is_some() {
                    return Err(t.err("more than one version line"));
                }
                version = Some(t.number_u32()?);
                t.end()?;
                continue;
            }
            if version.is_none() {
                return Err(t.err("the version line comes first"));
            }
            if first == "grid" {
                let columns = t.term(&p.symbols)?;
                t.comma()?;
                let rows = t.term(&p.symbols)?;
                t.end()?;
                p.ops.push(Op::Grid { columns, rows });
                continue;
            }
            if first == "copy" {
                let shape = t.shape()?;
                let dst = t.value(&p.symbols)?;
                let dst_offset = t.bracket(&p.symbols)?;
                t.expect_eq()?;
                let src = t.value(&p.symbols)?;
                // The second bracket names the source offset. One bracket is
                // shorthand for both sides carrying the same index.
                let src_offset = if matches!(t.peek(), Some(Token::LBracket)) {
                    t.bracket(&p.symbols)?
                } else {
                    dst_offset
                };
                t.end()?;
                p.ops.push(Op::CopyTo {
                    dst,
                    src,
                    dst_offset,
                    src_offset,
                    shape,
                });
                continue;
            }
            if first == "store" {
                let shape = t.shape()?;
                let buffer = t.value(&p.symbols)?;
                // An `acc` has no offset to name; a buffer's bracket is still
                // required by every program that has one, and a missing one
                // reads as zero.
                let offset = if matches!(t.peek(), Some(Token::LBracket)) {
                    t.bracket(&p.symbols)?
                } else {
                    Dim::Const(0)
                };
                t.expect_eq()?;
                let value = t.value(&p.symbols)?;
                t.end()?;
                p.ops.push(Op::StoreTo {
                    value,
                    buffer,
                    offset,
                    shape,
                });
                continue;
            }
            if first == "load" {
                let shape = t.shape()?;
                let name = t.name()?;
                t.expect_eq()?;
                let buffer = t.value(&p.symbols)?;
                let offset = t.bracket(&p.symbols)?;
                t.end()?;
                let value = p.define(name)?;
                p.ops.push(Op::Load {
                    value,
                    buffer,
                    offset,
                    shape,
                });
                continue;
            }
            if first == "matmul" {
                let acc = t.value(&p.symbols)?;
                let a = t.value(&p.symbols)?;
                let b = t.value(&p.symbols)?;
                t.end()?;
                p.ops.push(Op::Matmul { acc, a, b });
                continue;
            }
            if first == "sum" || first == "max" {
                let word = t.ident()?;
                let dim =
                    ReduceDim::from_name(&word).ok_or_else(|| t.err("no reduce dimension"))?;
                let acc = t.value(&p.symbols)?;
                let x = t.value(&p.symbols)?;
                let op = if first == "sum" {
                    let scaler = t.value(&p.symbols)?;
                    Op::Sum {
                        acc,
                        dim,
                        x,
                        scaler,
                    }
                } else {
                    Op::Max { acc, dim, x }
                };
                t.end()?;
                p.ops.push(op);
                continue;
            }
            if first == "endloop" {
                t.end()?;
                p.ops.push(Op::EndLoop);
                continue;
            }
            return Err(t.err(format!("no instruction starts with `{first}`")));
        }
        match version {
            Some(v) if v == VERSION => {}
            Some(v) => {
                let _ = v;
                return Err(Error::TT_ERROR_VERSION);
            }
            None => return Err(Error::TT_ERROR_VERSION),
        }
        let k = Kernel {
            ops: p.ops,
            names: p.names,
        };
        k.verify()?;
        Ok(k)
    }
}

fn literal(dtype: DType, bits: u32) -> String {
    match dtype {
        DType::I32 => (bits as i32).to_string(),
        DType::F32 => {
            let v = f32::from_bits(bits);
            if v.fract() == 0.0 {
                format!("{v:.1}")
            } else {
                format!("{v}")
            }
        }
        DType::Bf16 => unreachable!("bf16 constants are refused"),
        DType::Bool => unreachable!("bool constants are refused"),
    }
}

/// The parser state that spans lines: the program and the names defined so far.
#[derive(Default)]
struct Parser {
    ops: Vec<Op>,
    names: Vec<String>,
    symbols: BTreeMap<String, ValueId>,
}

impl Parser {
    fn define(&mut self, name: String) -> Result<ValueId, Error> {
        if self.symbols.contains_key(&name) {
            return Err(Error::TT_ERROR_NAME_DUPLICATE);
        }
        let value = self.names.len() as ValueId;
        self.symbols.insert(name.clone(), value);
        self.names.push(name);
        Ok(value)
    }
}

/// `%name = ...`: the only lines that define a value.
fn parse_definition(t: &mut Tokens, p: &mut Parser) -> Result<Op, Error> {
    let name = t.name()?;
    t.expect_eq()?;
    let keyword = t.ident()?;
    let value = p.define(name)?;
    let op = match keyword.as_str() {
        "const" => {
            let dtype = t.dtype()?;
            let bits = t.literal(dtype)?;
            t.end()?;
            Op::Const { value, dtype, bits }
        }
        "param" => {
            let dtype = t.dtype()?;
            // The scalar shape is a single `0` and nothing else.
            let dims = if matches!(t.peek(), Some(Token::Int(0))) && t.tok.len() == t.at + 1 {
                t.at += 1;
                Vec::new()
            } else {
                let mut dims = vec![t.term(&p.symbols)?];
                while matches!(t.peek(), Some(Token::Comma)) {
                    t.comma()?;
                    dims.push(t.term(&p.symbols)?);
                }
                dims
            };
            t.end()?;
            Op::Param { value, dtype, dims }
        }
        "buffer" => {
            let dtype = t.dtype()?;
            let a = t.number_u32()?;
            t.comma()?;
            let b = t.number_u32()?;
            t.end()?;
            Op::Buffer {
                value,
                dtype,
                shape: [a, b],
            }
        }
        "acc" => {
            let dtype = t.dtype()?;
            let a = t.number_u32()?;
            t.comma()?;
            let b = t.number_u32()?;
            t.end()?;
            Op::Acc {
                value,
                dtype,
                shape: [a, b],
            }
        }
        "loop" => {
            let bound = t.term(&p.symbols)?;
            t.end()?;
            Op::Loop { value, bound }
        }
        "grid" => {
            t.dot()?;
            let axis = match t.ident()?.as_str() {
                "x" => 0,
                "y" => 1,
                other => return Err(t.err(format!("a grid coordinate is x or y, not `{other}`"))),
            };
            t.end()?;
            Op::GridCoord { value, axis }
        }
        "cast" => {
            let src = t.value(&p.symbols)?;
            t.expect("to")?;
            let dtype = t.dtype()?;
            t.end()?;
            Op::Cast { value, src, dtype }
        }
        "bitcast" => {
            let src = t.value(&p.symbols)?;
            t.expect("to")?;
            let dtype = t.dtype()?;
            t.end()?;
            Op::Bitcast { value, src, dtype }
        }
        "sel" => {
            let cond = t.value(&p.symbols)?;
            let if_true = t.value(&p.symbols)?;
            let if_false = t.value(&p.symbols)?;
            t.end()?;
            Op::Select {
                value,
                cond,
                if_true,
                if_false,
            }
        }
        word if BinOp::from_name(word).is_some() => {
            let op = BinOp::from_name(word).expect("checked");
            let a = t.value(&p.symbols)?;
            let b = t.value(&p.symbols)?;
            t.end()?;
            Op::Binary { value, op, a, b }
        }
        word if UnOp::from_name(word).is_some() => {
            let op = UnOp::from_name(word).expect("checked");
            let a = t.value(&p.symbols)?;
            t.end()?;
            Op::Unary { value, op, a }
        }
        other => return Err(t.err(format!("no definition starts with `{other}`"))),
    };
    Ok(op)
}

struct Tokens<'a> {
    tok: &'a [Token],
    at: usize,
    /// The line, kept for a future diagnostic that is not an enum variant.
    #[allow(dead_code)]
    line: usize,
}

impl<'a> Tokens<'a> {
    fn err(&self, msg: impl Into<String>) -> Error {
        let _ = msg;
        Error::TT_ERROR_PARSE
    }

    fn peek(&self) -> Option<Token> {
        self.tok.get(self.at).cloned()
    }

    fn next(&mut self) -> Result<Token, Error> {
        let t = self
            .tok
            .get(self.at)
            .cloned()
            .ok_or_else(|| self.err("the line ends early"))?;
        self.at += 1;
        Ok(t)
    }

    fn end(&self) -> Result<(), Error> {
        match self.peek() {
            None => Ok(()),
            Some(t) => Err(self.err(format!("unexpected `{}`", t.text()))),
        }
    }

    fn ident(&mut self) -> Result<String, Error> {
        match self.next()? {
            Token::Ident(s) => Ok(s),
            t => Err(self.err(format!("expected a word, found `{}`", t.text()))),
        }
    }

    fn expect(&mut self, word: &str) -> Result<(), Error> {
        let got = self.ident()?;
        if got == word {
            Ok(())
        } else {
            Err(self.err(format!("expected `{word}`, found `{got}`")))
        }
    }

    fn value(&mut self, symbols: &BTreeMap<String, ValueId>) -> Result<ValueId, Error> {
        let name = self.name()?;
        symbols
            .get(&name)
            .copied()
            .ok_or_else(|| self.err(format!("unknown value %{name}")))
    }

    fn name(&mut self) -> Result<String, Error> {
        match self.next()? {
            Token::Name(s) => Ok(s),
            t => Err(self.err(format!("expected a %value, found `{}`", t.text()))),
        }
    }

    fn comma(&mut self) -> Result<(), Error> {
        match self.next()? {
            Token::Comma => Ok(()),
            t => Err(self.err(format!("expected `,`, found `{}`", t.text()))),
        }
    }

    fn bracket(&mut self, symbols: &BTreeMap<String, ValueId>) -> Result<Dim, Error> {
        match self.next()? {
            Token::LBracket => {}
            t => {
                return Err(self.err(format!("expected `[`, found `{}`", t.text())));
            }
        }
        let dim = self.term(symbols)?;
        match self.next()? {
            Token::RBracket => Ok(dim),
            t => Err(self.err(format!("expected `]`, found `{}`", t.text()))),
        }
    }

    fn term(&mut self, symbols: &BTreeMap<String, ValueId>) -> Result<Dim, Error> {
        match self.next()? {
            Token::Name(s) => Ok(Dim::Value(
                *symbols
                    .get(&s)
                    .ok_or_else(|| self.err(format!("unknown value %{s}")))?,
            )),
            Token::Int(n) => u32::try_from(n)
                .map(Dim::Const)
                .map_err(|_| self.err(format!("{n} is not a non-negative index"))),
            t => Err(self.err(format!(
                "expected a literal or a %value, found `{}`",
                t.text()
            ))),
        }
    }

    fn number_u32(&mut self) -> Result<u32, Error> {
        match self.next()? {
            Token::Int(n) => u32::try_from(n).map_err(|_| self.err(format!("{n} is not a u32"))),
            t => Err(self.err(format!("expected a number, found `{}`", t.text()))),
        }
    }

    /// An access shape: `<rows>,<cols>`.
    fn shape(&mut self) -> Result<[u32; 2], Error> {
        let rows = self.number_u32()?;
        self.comma()?;
        let cols = self.number_u32()?;
        Ok([rows, cols])
    }

    fn dtype(&mut self) -> Result<DType, Error> {
        let name = self.ident()?;
        match name.as_str() {
            "f32" => Ok(DType::F32),
            "bf16" => Ok(DType::Bf16),
            "i32" => Ok(DType::I32),
            other => Err(self.err(format!("unknown dtype `{other}`"))),
        }
    }

    fn expect_eq(&mut self) -> Result<(), Error> {
        match self.next()? {
            Token::Eq => Ok(()),
            t => Err(self.err(format!("expected `=`, found `{}`", t.text()))),
        }
    }

    fn dot(&mut self) -> Result<(), Error> {
        match self.next()? {
            Token::Dot => Ok(()),
            t => Err(self.err(format!("expected `.`, found `{}`", t.text()))),
        }
    }

    /// A constant's bits, in its dtype.
    fn literal(&mut self, dtype: DType) -> Result<u32, Error> {
        if dtype == DType::Bf16 {
            return Err(self.err("a bf16 constant is not exact; write it as f32 and cast"));
        }
        match self.next()? {
            Token::Int(n) if dtype == DType::I32 => {
                Ok(u32::try_from(n).map_err(|_| self.err(format!("{n} is not an i32")))?)
            }
            Token::Int(n) if dtype == DType::F32 => Ok((n as f32).to_bits()),
            Token::Float(bits) if dtype == DType::F32 => Ok(bits),
            t => Err(self.err(format!(
                "expected a {} literal, found `{}`",
                dtype.name(),
                t.text()
            ))),
        }
    }
}

#[derive(Debug, Clone)]
enum Token {
    Name(String),
    Ident(String),
    Int(i64),
    Float(u32),
    Comma,
    Dot,
    LBracket,
    RBracket,
    Eq,
}

impl Token {
    fn text(&self) -> String {
        match self {
            Token::Name(n) => format!("%{n}"),
            Token::Ident(s) => s.clone(),
            Token::Int(n) => n.to_string(),
            Token::Float(bits) => literal(DType::F32, *bits),
            Token::Comma => ",".to_string(),
            Token::Dot => ".".to_string(),
            Token::LBracket => "[".to_string(),
            Token::RBracket => "]".to_string(),
            Token::Eq => "=".to_string(),
        }
    }
}

fn tokenize(line: &str) -> Result<Vec<Token>, String> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            ',' => {
                out.push(Token::Comma);
                i += 1;
            }
            '[' => {
                out.push(Token::LBracket);
                i += 1;
            }
            ']' => {
                out.push(Token::RBracket);
                i += 1;
            }
            '=' => {
                out.push(Token::Eq);
                i += 1;
            }
            '.' => {
                out.push(Token::Dot);
                i += 1;
            }
            '%' => {
                let start = i + 1;
                i += 1;
                let mut end = i;
                while end < bytes.len()
                    && ((bytes[end] as char).is_ascii_alphanumeric() || bytes[end] == b'_')
                {
                    end += 1;
                }
                if end == start {
                    return Err("a `%` must be followed by a name".into());
                }
                out.push(Token::Name(line[start..end].to_string()));
                i = end;
            }
            c if c.is_ascii_digit() || c == '-' => {
                let start = i;
                i += 1;
                let mut float = false;
                while i < bytes.len() {
                    let d = bytes[i] as char;
                    if d.is_ascii_digit() {
                        i += 1;
                    } else if d == '.' {
                        float = true;
                        i += 1;
                    } else {
                        break;
                    }
                }
                let text = &line[start..i];
                if float {
                    let v: f32 = text
                        .parse()
                        .map_err(|_| format!("`{text}` is not a number"))?;
                    out.push(Token::Float(v.to_bits()));
                } else {
                    let v: i64 = text
                        .parse()
                        .map_err(|_| format!("`{text}` is not an integer"))?;
                    out.push(Token::Int(v));
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < bytes.len()
                    && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] == b'_')
                {
                    i += 1;
                }
                out.push(Token::Ident(line[start..i].to_string()));
            }
            other => return Err(format!("unexpected `{other}`")),
        }
    }
    Ok(out)
}
