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

    fn tag(self) -> u8 {
        match self {
            DType::F32 => 0,
            DType::Bf16 => 1,
            DType::I32 => 2,
            DType::Bool => 3,
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

    pub fn name(self) -> &'static str {
        match self {
            RegShape::Scalar => "scalar",
            RegShape::Column => "column",
            RegShape::Row => "row",
            RegShape::Tile => "tile",
        }
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
    pub fn name(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::Div => "div",
            BinOp::Pow => "pow",
            BinOp::Mod => "mod",
            BinOp::Cmplt => "cmplt",
            BinOp::Cmpgt => "cmpgt",
            BinOp::Cmpge => "cmpge",
            BinOp::Max => "max",
            BinOp::Or => "or",
            BinOp::And => "and",
            BinOp::BitXor => "bitxor",
            BinOp::BitOr => "bitor",
            BinOp::BitAnd => "bitand",
            BinOp::BitShiftLeft => "bitshiftleft",
            BinOp::BitShiftRight => "bitshiftright",
            BinOp::NotEq => "noteq",
            BinOp::Eq => "eq",
        }
    }

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

    fn tag(self) -> u8 {
        match self {
            BinOp::Add => 0,
            BinOp::Sub => 1,
            BinOp::Mul => 2,
            BinOp::Div => 3,
            BinOp::Pow => 4,
            BinOp::Mod => 5,
            BinOp::Cmplt => 6,
            BinOp::Cmpgt => 7,
            BinOp::Cmpge => 8,
            BinOp::Max => 9,
            BinOp::Or => 10,
            BinOp::And => 11,
            BinOp::BitXor => 12,
            BinOp::BitOr => 13,
            BinOp::BitAnd => 14,
            BinOp::BitShiftLeft => 15,
            BinOp::BitShiftRight => 16,
            BinOp::NotEq => 17,
            BinOp::Eq => 18,
        }
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
    pub fn name(self) -> &'static str {
        match self {
            UnOp::Neg => "neg",
            UnOp::Not => "not",
            UnOp::BitNot => "bitnot",
            UnOp::Exp2 => "exp2",
            UnOp::Log2 => "log2",
            UnOp::Reciprocal => "reciprocal",
            UnOp::Sqrt => "sqrt",
            UnOp::Rsqrt => "rsqrt",
            UnOp::Sin => "sin",
            UnOp::Cos => "cos",
            UnOp::Floor => "floor",
            UnOp::Trunc => "trunc",
            UnOp::Abs => "abs",
        }
    }

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

    fn tag(self) -> u8 {
        match self {
            UnOp::Neg => 0,
            UnOp::Not => 1,
            UnOp::BitNot => 2,
            UnOp::Exp2 => 3,
            UnOp::Log2 => 4,
            UnOp::Reciprocal => 5,
            UnOp::Sqrt => 6,
            UnOp::Rsqrt => 7,
            UnOp::Sin => 8,
            UnOp::Cos => 9,
            UnOp::Floor => 10,
            UnOp::Trunc => 11,
            UnOp::Abs => 12,
        }
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
    /// The launch grid, as a shape of two terms.
    Grid { columns: Dim, rows: Dim },
    /// `%i = loop %bound`.
    Loop { value: ValueId, bound: Dim },
    /// `endloop`.
    EndLoop,
    /// DRAM into L1, or L1 into DRAM: exactly one side is a parameter, the other
    /// a buffer. Each side has its own datum offset, because a circular buffer
    /// names its page by the ring pointer while the parameter names the access's
    /// own index (zyx's `Op::Copy` carries `src_idx`/`dst_idx`).
    CopyTo {
        dst: ValueId,
        src: ValueId,
        dst_offset: Dim,
        src_offset: Dim,
    },
    /// L1 buffer into a register value.
    Load {
        value: ValueId,
        buffer: ValueId,
        offset: Dim,
    },
    /// A register value into an L1 buffer.
    StoreTo {
        value: ValueId,
        buffer: ValueId,
        offset: Dim,
    },
    /// The only conversion.
    Cast {
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
    /// `a x b` accumulating into a tile.
    Matmul {
        value: ValueId,
        a: ValueId,
        b: ValueId,
    },
}

impl Op {
    /// The value this op defines, if any.
    pub fn defined(&self) -> Option<ValueId> {
        match *self {
            Op::Const { value, .. }
            | Op::Param { value, .. }
            | Op::Buffer { value, .. }
            | Op::Loop { value, .. }
            | Op::Load { value, .. }
            | Op::Cast { value, .. }
            | Op::Unary { value, .. }
            | Op::Binary { value, .. }
            | Op::Select { value, .. }
            | Op::Matmul { value, .. } => Some(value),
            Op::Grid { .. } | Op::EndLoop | Op::CopyTo { .. } | Op::StoreTo { .. } => None,
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
            } => {
                let mut v = vec![*value, *buffer];
                v.extend(dim(offset));
                v
            }
            Op::Cast { src, .. } => vec![*src],
            Op::Unary { a, .. } => vec![*a],
            Op::Binary { a, b, .. } => vec![*a, *b],
            Op::Select {
                cond,
                if_true,
                if_false,
                ..
            } => vec![*cond, *if_true, *if_false],
            Op::Matmul { a, b, .. } => vec![*a, *b],
            Op::Loop { bound, .. } => dim(bound).into_iter().collect(),
            Op::Grid { columns, rows } => dim(columns).into_iter().chain(dim(rows)).collect(),
            Op::Param { dims, .. } => dims.iter().filter_map(dim).collect(),
            Op::Const { .. } | Op::Buffer { .. } | Op::EndLoop => Vec::new(),
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

/// What can be wrong with a kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl Error {
    fn new(msg: impl Into<String>) -> Self {
        Error(msg.into())
    }
}

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
        return Err(Error::new(format!(
            "buffer shape {a},{b} has a zero dimension"
        )));
    }
    let ok = |n: u32| n == 1 || n % 32 == 0;
    if !ok(a) || !ok(b) {
        return Err(Error::new(format!(
            "buffer shape {a},{b}: each dimension is 1 or a multiple of 32"
        )));
    }
    let elements = a
        .checked_mul(b)
        .ok_or_else(|| Error::new(format!("buffer shape {a},{b} overflows")))?;
    let bytes = elements
        .checked_mul(dtype.bytes())
        .ok_or_else(|| Error::new(format!("buffer shape {a},{b} overflows in bytes")))?;
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
    at: usize,
) -> Result<&'a ValueInfo, Error> {
    let info = values
        .get(v as usize)
        .ok_or_else(|| Error::new(format!("op {at} uses %{v}, which is not defined")))?;
    if !visible.get(v as usize).copied().unwrap_or(false) {
        return Err(Error::new(format!(
            "op {at} uses %{v}, which is out of scope"
        )));
    }
    Ok(info)
}

impl Kernel {
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
                return Err(Error::new(format!("the name %{name} is defined twice")));
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
                    return Err(Error::new(format!(
                        "op {at} defines %{value} but the next value is %{}",
                        values.len()
                    )));
                }
                if self.names.get(value as usize).is_none() {
                    return Err(Error::new(format!("op {at}: value %{value} has no name")));
                }
            }
            let info = match *op {
                Op::Const { dtype, bits, .. } => {
                    if dtype == DType::Bf16 {
                        return Err(Error::new(format!(
                            "op {at}: a bf16 constant is not exact; write it as f32 and cast"
                        )));
                    }
                    if dtype == DType::Bool {
                        return Err(Error::new(format!(
                            "op {at}: a bool constant is not a datum; write a comparison"
                        )));
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
                                return Err(Error::new(format!(
                                    "op {at}: dimension {k} is zero; the scalar shape is a single 0 and nothing else"
                                )));
                            }
                            Dim::Const(_) => {}
                            Dim::Value(v) => {
                                let i = lookup(&values, &visible, v, at)?;
                                if i.dtype != INDEX || i.shape != RegShape::Scalar {
                                    return Err(Error::new(format!(
                                        "op {at}: dimension {k} is %{v}, a {} {}; a dimension needs an i32 scalar",
                                        i.dtype.name(),
                                        i.shape.name()
                                    )));
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
                Op::Grid { columns, rows } => {
                    if grid.is_some() {
                        return Err(Error::new("the program has more than one grid"));
                    }
                    grid = Some(([columns, rows], at));
                    continue;
                }
                Op::Loop { bound, .. } => {
                    if let Dim::Value(v) = bound {
                        let i = lookup(&values, &visible, v, at)?;
                        if i.dtype != INDEX || i.shape != RegShape::Scalar {
                            return Err(Error::new(format!(
                                "op {at}: the loop bound %{v} is not an i32 scalar"
                            )));
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
                        return Err(Error::new(format!("op {at}: endloop without a loop")));
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
                } => {
                    let d = *lookup(&values, &visible, dst, at)?;
                    let s = *lookup(&values, &visible, src, at)?;
                    let (buffer, param, buffer_offset, param_offset) = match (d.kind, s.kind) {
                        (ValueKind::Buffer, ValueKind::Param) => (dst, src, dst_offset, src_offset),
                        (ValueKind::Param, ValueKind::Buffer) => (src, dst, src_offset, dst_offset),
                        _ => {
                            return Err(Error::new(format!(
                                "op {at}: a copy moves between a parameter and a buffer, not a {} and a {}",
                                kind_name(d.kind),
                                kind_name(s.kind)
                            )));
                        }
                    };
                    let b = *lookup(&values, &visible, buffer, at)?;
                    let p = *lookup(&values, &visible, param, at)?;
                    if p.shape == RegShape::Scalar {
                        return Err(Error::new(format!(
                            "op {at}: %{param} is a scalar parameter; it cannot hold a buffer unit"
                        )));
                    }
                    if b.dtype != p.dtype {
                        return Err(Error::new(format!(
                            "op {at}: the buffer is {} and the parameter {}; there is no implicit cast",
                            b.dtype.name(),
                            p.dtype.name()
                        )));
                    }
                    check_dim(
                        &values,
                        &visible,
                        dst_offset,
                        at,
                        "the destination copy offset",
                    )?;
                    check_dim(&values, &visible, src_offset, at, "the source copy offset")?;
                    let Op::Buffer { shape, .. } = self.ops[def[buffer as usize]] else {
                        return Err(Error::new(format!("op {at}: %{buffer} is not a buffer")));
                    };
                    let (unit_shape, _, _) = unit_of(shape, b.dtype)?;
                    let total = shape[0] * shape[1];
                    check_offset(buffer_offset, total, unit_shape.datums(), at, "buffer")?;
                    if let Some(count) = param_elements(self, &def, param) {
                        check_offset(param_offset, count, unit_shape.datums(), at, "parameter")?;
                    }
                    continue;
                }
                Op::Load {
                    buffer,
                    offset,
                    value,
                } => {
                    let b = *lookup(&values, &visible, buffer, at)?;
                    if b.kind == ValueKind::Param {
                        // A direct parameter access: one tile, which
                        // `mover::add_movers` stages through a `32,32` buffer.
                        if b.shape != RegShape::Tile {
                            return Err(Error::new(format!(
                                "op {at}: %{buffer} is a {} parameter; a load reads a tile",
                                b.shape.name()
                            )));
                        }
                        check_dim(&values, &visible, offset, at, "the load offset")?;
                        if let Some(count) = param_elements(self, &def, buffer) {
                            check_offset(offset, count, 1024, at, "parameter")?;
                        }
                        let _ = value;
                        ValueInfo {
                            kind: ValueKind::Register,
                            dtype: b.dtype,
                            shape: RegShape::Tile,
                        }
                    } else {
                        if b.kind != ValueKind::Buffer {
                            return Err(Error::new(format!("op {at}: %{buffer} is not a buffer")));
                        }
                        check_dim(&values, &visible, offset, at, "the load offset")?;
                        let Op::Buffer { shape, .. } = self.ops[def[buffer as usize]] else {
                            return Err(Error::new(format!("op {at}: %{buffer} is not a buffer")));
                        };
                        let (unit_shape, _, _) = unit_of(shape, b.dtype)?;
                        let total = shape[0] * shape[1];
                        check_offset(offset, total, unit_shape.datums(), at, "buffer")?;
                        let _ = value;
                        ValueInfo {
                            kind: ValueKind::Register,
                            dtype: b.dtype,
                            shape: unit_shape,
                        }
                    }
                }
                Op::StoreTo {
                    value,
                    buffer,
                    offset,
                } => {
                    let v = *lookup(&values, &visible, value, at)?;
                    let b = *lookup(&values, &visible, buffer, at)?;
                    if v.kind != ValueKind::Register {
                        return Err(Error::new(format!(
                            "op {at}: %{value} is not a register value"
                        )));
                    }
                    if b.kind == ValueKind::Param {
                        // The direct form; `mover::add_movers` stages it.
                        if b.shape != RegShape::Tile {
                            return Err(Error::new(format!(
                                "op {at}: %{buffer} is a {} parameter; a store writes a tile",
                                b.shape.name()
                            )));
                        }
                        if v.dtype != b.dtype {
                            return Err(Error::new(format!(
                                "op {at}: the value is {} and the parameter {}; there is no implicit cast",
                                v.dtype.name(),
                                b.dtype.name()
                            )));
                        }
                        if v.shape != RegShape::Tile {
                            return Err(Error::new(format!(
                                "op {at}: a parameter holds a tile, the value is a {}",
                                v.shape.name()
                            )));
                        }
                        check_dim(&values, &visible, offset, at, "the store offset")?;
                        if let Some(count) = param_elements(self, &def, buffer) {
                            check_offset(offset, count, 1024, at, "parameter")?;
                        }
                        continue;
                    }
                    if b.kind != ValueKind::Buffer {
                        return Err(Error::new(format!("op {at}: %{buffer} is not a buffer")));
                    }
                    if v.dtype != b.dtype {
                        return Err(Error::new(format!(
                            "op {at}: the value is {} and the buffer {}; there is no implicit cast",
                            v.dtype.name(),
                            b.dtype.name()
                        )));
                    }
                    let Op::Buffer { shape, .. } = self.ops[def[buffer as usize]] else {
                        return Err(Error::new(format!("op {at}: %{buffer} is not a buffer")));
                    };
                    let (unit_shape, _, _) = unit_of(shape, b.dtype)?;
                    if v.shape != unit_shape {
                        return Err(Error::new(format!(
                            "op {at}: the buffer holds a {}, the value is a {}",
                            unit_shape.name(),
                            v.shape.name()
                        )));
                    }
                    check_dim(&values, &visible, offset, at, "the store offset")?;
                    let total = shape[0] * shape[1];
                    check_offset(offset, total, unit_shape.datums(), at, "buffer")?;
                    continue;
                }
                Op::Cast { src, dtype, .. } => {
                    let s = *lookup(&values, &visible, src, at)?;
                    if !is_value(s.kind) {
                        return Err(Error::new(format!(
                            "op {at}: %{src} is not a register value"
                        )));
                    }
                    if s.dtype == DType::Bool || dtype == DType::Bool {
                        todo!("op {at}: a cast to or from bool is not implemented");
                    }
                    ValueInfo {
                        kind: ValueKind::Register,
                        dtype,
                        shape: s.shape,
                    }
                }
                Op::Unary { op, a, .. } => {
                    let x = *lookup(&values, &visible, a, at)?;
                    if !is_value(x.kind) {
                        return Err(Error::new(format!("op {at}: %{a} is not a register value")));
                    }
                    let dtype = match op {
                        UnOp::Not => {
                            if x.dtype != DType::Bool {
                                return Err(Error::new(format!(
                                    "op {at}: not needs a bool, not {}",
                                    x.dtype.name()
                                )));
                            }
                            DType::Bool
                        }
                        UnOp::BitNot => {
                            if x.dtype != DType::I32 {
                                return Err(Error::new(format!(
                                    "op {at}: bitnot is integer (i32), not {}",
                                    x.dtype.name()
                                )));
                            }
                            DType::I32
                        }
                        _ => {
                            if !x.dtype.is_float() {
                                return Err(Error::new(format!(
                                    "op {at}: {} needs a float value, not {}",
                                    op.name(),
                                    x.dtype.name()
                                )));
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
                    if !is_value(x.kind) || !is_value(y.kind) {
                        return Err(Error::new(format!(
                            "op {at}: {} needs two register values",
                            op.name()
                        )));
                    }
                    if x.dtype != y.dtype {
                        return Err(Error::new(format!(
                            "op {at}: {} of {} and {}; there is no implicit cast",
                            op.name(),
                            x.dtype.name(),
                            y.dtype.name()
                        )));
                    }
                    if op.is_logical() && x.dtype != DType::Bool {
                        return Err(Error::new(format!(
                            "op {at}: {} is logical (bool), not {}",
                            op.name(),
                            x.dtype.name()
                        )));
                    }
                    if op.is_integer_only() && x.dtype != INDEX {
                        return Err(Error::new(format!(
                            "op {at}: {} is integer math (i32), not {}",
                            op.name(),
                            x.dtype.name()
                        )));
                    }
                    if !op.is_logical() && x.dtype == DType::Bool {
                        return Err(Error::new(format!(
                            "op {at}: {} is not a bool operation",
                            op.name()
                        )));
                    }
                    let shape = broadcast(x.shape, y.shape).ok_or_else(|| {
                        Error::new(format!(
                            "op {at}: cannot broadcast a {} and a {}",
                            x.shape.name(),
                            y.shape.name()
                        ))
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
                        return Err(Error::new(format!(
                            "op {at}: sel needs three register values"
                        )));
                    }
                    if c.dtype != DType::Bool {
                        return Err(Error::new(format!(
                            "op {at}: the condition %{cond} is {}, not bool",
                            c.dtype.name()
                        )));
                    }
                    if t.dtype != f.dtype {
                        return Err(Error::new(format!(
                            "op {at}: sel of {} and {}; there is no implicit cast",
                            t.dtype.name(),
                            f.dtype.name()
                        )));
                    }
                    let shape = broadcast(t.shape, f.shape).ok_or_else(|| {
                        Error::new(format!(
                            "op {at}: cannot broadcast a {} and a {}",
                            t.shape.name(),
                            f.shape.name()
                        ))
                    })?;
                    let shape = broadcast(shape, c.shape).ok_or_else(|| {
                        Error::new(format!(
                            "op {at}: cannot broadcast the condition's {} with the values' {}",
                            c.shape.name(),
                            shape.name()
                        ))
                    })?;
                    ValueInfo {
                        kind: ValueKind::Register,
                        dtype: t.dtype,
                        shape,
                    }
                }
                Op::Matmul { a, b, .. } => {
                    let x = *lookup(&values, &visible, a, at)?;
                    let y = *lookup(&values, &visible, b, at)?;
                    if !is_value(x.kind) || !is_value(y.kind) {
                        return Err(Error::new(format!(
                            "op {at}: matmul needs two register values"
                        )));
                    }
                    if x.shape != RegShape::Tile || y.shape != RegShape::Tile {
                        return Err(Error::new(format!(
                            "op {at}: matmul needs two tiles, not a {} and a {}",
                            x.shape.name(),
                            y.shape.name()
                        )));
                    }
                    if x.dtype != y.dtype {
                        return Err(Error::new(format!(
                            "op {at}: matmul of {} and {}; there is no implicit cast",
                            x.dtype.name(),
                            y.dtype.name()
                        )));
                    }
                    ValueInfo {
                        kind: ValueKind::Register,
                        dtype: x.dtype,
                        shape: RegShape::Tile,
                    }
                }
            };
            values.push(info);
            visible.push(true);
        }
        if !loops.is_empty() {
            return Err(Error::new("the program ends inside a loop"));
        }
        if values.len() != self.names.len() {
            return Err(Error::new(format!(
                "the program defines {} values but names {}",
                values.len(),
                self.names.len()
            )));
        }
        let Some((grid, at)) = grid else {
            return Err(Error::new("the program has no grid"));
        };
        check_dim(&values, &visible, grid[0], at, "the grid's columns")?;
        check_dim(&values, &visible, grid[1], at, "the grid's rows")?;
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

fn kind_name(kind: ValueKind) -> &'static str {
    match kind {
        ValueKind::Param => "parameter",
        ValueKind::Buffer => "buffer",
        ValueKind::Const => "constant",
        ValueKind::Induction => "induction value",
        ValueKind::Register => "register value",
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
fn check_dim(
    values: &[ValueInfo],
    visible: &[bool],
    dim: Dim,
    at: usize,
    what: &str,
) -> Result<(), Error> {
    if let Dim::Value(v) = dim {
        let i = lookup(values, visible, v, at)?;
        if i.dtype != INDEX || i.shape != RegShape::Scalar {
            return Err(Error::new(format!(
                "op {at}: {what} is %{v}, a {} {}; it must be an i32 scalar",
                i.dtype.name(),
                i.shape.name()
            )));
        }
    }
    Ok(())
}

/// A bracket must leave room for one unit. A symbolic offset cannot be checked
/// here; it is checked at run time.
fn check_offset(offset: Dim, total: u32, unit: u32, at: usize, what: &str) -> Result<(), Error> {
    if let Dim::Const(n) = offset {
        if n.checked_add(unit).is_none_or(|end| end > total) {
            return Err(Error::new(format!(
                "op {at}: {n} + {unit} datums runs past the {what} ({total} datums)"
            )));
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
/// The IR version this build reads and writes.
pub const VERSION: u32 = 4;

impl Kernel {
    /// The byte form: the same program as [`Kernel::to_text`], little-endian and
    /// length-prefixed. Values are referenced by id; every definition carries
    /// its name.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        put_u32(&mut b, VERSION);
        put_u32(&mut b, self.ops.len() as u32);
        for op in &self.ops {
            match op {
                Op::Const { value, dtype, bits } => {
                    b.push(0);
                    put_name(&mut b, &self.names[*value as usize]);
                    b.push(dtype.tag());
                    put_u32(&mut b, *bits);
                }
                Op::Param { value, dtype, dims } => {
                    b.push(1);
                    put_name(&mut b, &self.names[*value as usize]);
                    b.push(dtype.tag());
                    put_u32(&mut b, dims.len() as u32);
                    for d in dims {
                        put_dim(&mut b, *d);
                    }
                }
                Op::Buffer {
                    value,
                    dtype,
                    shape,
                } => {
                    b.push(2);
                    put_name(&mut b, &self.names[*value as usize]);
                    b.push(dtype.tag());
                    put_u32(&mut b, shape[0]);
                    put_u32(&mut b, shape[1]);
                }
                Op::Grid { columns, rows } => {
                    b.push(3);
                    put_dim(&mut b, *columns);
                    put_dim(&mut b, *rows);
                }
                Op::Loop { value, bound } => {
                    b.push(4);
                    put_name(&mut b, &self.names[*value as usize]);
                    put_dim(&mut b, *bound);
                }
                Op::EndLoop => b.push(5),
                Op::CopyTo {
                    dst,
                    src,
                    dst_offset,
                    src_offset,
                } => {
                    b.push(6);
                    put_u32(&mut b, *dst);
                    put_u32(&mut b, *src);
                    put_dim(&mut b, *dst_offset);
                    put_dim(&mut b, *src_offset);
                }
                Op::Load {
                    value,
                    buffer,
                    offset,
                } => {
                    b.push(7);
                    put_name(&mut b, &self.names[*value as usize]);
                    put_u32(&mut b, *buffer);
                    put_dim(&mut b, *offset);
                }
                Op::StoreTo {
                    value,
                    buffer,
                    offset,
                } => {
                    b.push(8);
                    put_u32(&mut b, *buffer);
                    put_u32(&mut b, *value);
                    put_dim(&mut b, *offset);
                }
                Op::Cast { value, src, dtype } => {
                    b.push(9);
                    put_name(&mut b, &self.names[*value as usize]);
                    put_u32(&mut b, *src);
                    b.push(dtype.tag());
                }
                Op::Unary { value, op, a } => {
                    b.push(10);
                    put_name(&mut b, &self.names[*value as usize]);
                    b.push(op.tag());
                    put_u32(&mut b, *a);
                }
                Op::Binary {
                    value,
                    op,
                    a,
                    b: rhs,
                } => {
                    b.push(11);
                    put_name(&mut b, &self.names[*value as usize]);
                    b.push(op.tag());
                    put_u32(&mut b, *a);
                    put_u32(&mut b, *rhs);
                }
                Op::Select {
                    value,
                    cond,
                    if_true,
                    if_false,
                } => {
                    b.push(13);
                    put_name(&mut b, &self.names[*value as usize]);
                    put_u32(&mut b, *cond);
                    put_u32(&mut b, *if_true);
                    put_u32(&mut b, *if_false);
                }
                Op::Matmul { value, a, b: rhs } => {
                    b.push(12);
                    put_name(&mut b, &self.names[*value as usize]);
                    put_u32(&mut b, *a);
                    put_u32(&mut b, *rhs);
                }
            }
        }
        b
    }

    /// Read the byte form. A wrong magic, an unknown version or a truncated
    /// record is refused, never guessed at.
    pub fn from_bytes(bytes: &[u8]) -> Result<Kernel, Error> {
        let mut r = Cursor { at: 0, bytes };
        if r.take(4)? != MAGIC {
            return Err(Error::new("not a TTIR byte program"));
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(Error::new(format!(
                "byte program version {version} is not {VERSION}"
            )));
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
                6 => Op::CopyTo {
                    dst: r.u32()?,
                    src: r.u32()?,
                    dst_offset: r.dim()?,
                    src_offset: r.dim()?,
                },
                7 => {
                    let name = r.name()?;
                    let buffer = r.u32()?;
                    let offset = r.dim()?;
                    let value = define(&mut names, name)?;
                    Op::Load {
                        value,
                        buffer,
                        offset,
                    }
                }
                8 => Op::StoreTo {
                    buffer: r.u32()?,
                    value: r.u32()?,
                    offset: r.dim()?,
                },
                9 => {
                    let name = r.name()?;
                    let src = r.u32()?;
                    let dtype = r.dtype()?;
                    let value = define(&mut names, name)?;
                    Op::Cast { value, src, dtype }
                }
                10 => {
                    let name = r.name()?;
                    let op = UnOp::from_tag(r.u8()?)
                        .ok_or_else(|| Error::new("unknown unary op tag"))?;
                    let a = r.u32()?;
                    let value = define(&mut names, name)?;
                    Op::Unary { value, op, a }
                }
                11 => {
                    let name = r.name()?;
                    let op = BinOp::from_tag(r.u8()?)
                        .ok_or_else(|| Error::new("unknown binary op tag"))?;
                    let a = r.u32()?;
                    let b = r.u32()?;
                    let value = define(&mut names, name)?;
                    Op::Binary { value, op, a, b }
                }
                12 => {
                    let name = r.name()?;
                    let a = r.u32()?;
                    let b = r.u32()?;
                    let value = define(&mut names, name)?;
                    Op::Matmul { value, a, b }
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
                other => return Err(Error::new(format!("unknown instruction tag {other}"))),
            };
            ops.push(op);
        }
        if r.at != r.bytes.len() {
            return Err(Error::new(format!(
                "{} trailing bytes",
                r.bytes.len() - r.at
            )));
        }
        let k = Kernel { ops, names };
        k.verify()?;
        Ok(k)
    }
}

fn define(names: &mut Vec<String>, name: String) -> Result<ValueId, Error> {
    if names.iter().any(|n| *n == name) {
        return Err(Error::new(format!("the name %{name} is defined twice")));
    }
    let value = names.len() as ValueId;
    names.push(name);
    Ok(value)
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_name(out: &mut Vec<u8>, name: &str) {
    put_u32(out, name.len() as u32);
    out.extend_from_slice(name.as_bytes());
}

fn put_dim(out: &mut Vec<u8>, dim: Dim) {
    match dim {
        Dim::Const(n) => {
            out.push(0);
            put_u32(out, n);
        }
        Dim::Value(v) => {
            out.push(1);
            put_u32(out, v);
        }
    }
}

struct Cursor<'a> {
    at: usize,
    bytes: &'a [u8],
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], Error> {
        let end = self.at + n;
        if end > self.bytes.len() {
            return Err(Error::new("byte program ends mid-record"));
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
        DType::from_tag(tag).ok_or_else(|| Error::new(format!("unknown dtype tag {tag}")))
    }

    fn name(&mut self) -> Result<String, Error> {
        let n = self.u32()? as usize;
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::new("a name is not UTF-8"))
    }

    fn dim(&mut self) -> Result<Dim, Error> {
        match self.u8()? {
            0 => Ok(Dim::Const(self.u32()?)),
            1 => Ok(Dim::Value(self.u32()?)),
            other => Err(Error::new(format!("unknown shape-term tag {other}"))),
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
            let tokens = tokenize(line).map_err(|e| Error::new(format!("line {at}: {e}")))?;
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
            if first == "copy_to" {
                let dst = t.value(&p.symbols)?;
                t.expect("from")?;
                let src = t.value(&p.symbols)?;
                let dst_offset = t.bracket(&p.symbols)?;
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
                });
                continue;
            }
            if first == "store_to" {
                let buffer = t.value(&p.symbols)?;
                t.expect("from")?;
                let value = t.value(&p.symbols)?;
                let offset = t.bracket(&p.symbols)?;
                t.end()?;
                p.ops.push(Op::StoreTo {
                    value,
                    buffer,
                    offset,
                });
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
                return Err(Error::new(format!("IR version {v} is not {VERSION}")));
            }
            None => return Err(Error::new("the program has no version line")),
        }
        let k = Kernel {
            ops: p.ops,
            names: p.names,
        };
        k.verify()?;
        Ok(k)
    }

    /// The canonical text form.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("version {VERSION}\n"));
        for op in &self.ops {
            let name = |v: ValueId| self.names[v as usize].as_str();
            match op {
                Op::Const { value, dtype, bits } => {
                    s.push_str(&format!(
                        "%{} = const {} {}\n",
                        name(*value),
                        dtype.name(),
                        literal(*dtype, *bits)
                    ));
                }
                Op::Param { value, dtype, dims } => {
                    let shape = if dims.is_empty() {
                        "0".to_string()
                    } else {
                        dims.iter()
                            .map(|d| self.dim_text(*d))
                            .collect::<Vec<_>>()
                            .join(",")
                    };
                    s.push_str(&format!(
                        "%{} = param {} {shape}\n",
                        name(*value),
                        dtype.name()
                    ));
                }
                Op::Buffer {
                    value,
                    dtype,
                    shape,
                } => {
                    s.push_str(&format!(
                        "%{} = buffer {} {},{}\n",
                        name(*value),
                        dtype.name(),
                        shape[0],
                        shape[1]
                    ));
                }
                Op::Grid { columns, rows } => {
                    s.push_str(&format!(
                        "grid {},{}\n",
                        self.dim_text(*columns),
                        self.dim_text(*rows)
                    ));
                }
                Op::Loop { value, bound } => {
                    s.push_str(&format!(
                        "%{} = loop {}\n",
                        name(*value),
                        self.dim_text(*bound)
                    ));
                }
                Op::EndLoop => s.push_str("endloop\n"),
                Op::CopyTo {
                    dst,
                    src,
                    dst_offset,
                    src_offset,
                } => {
                    s.push_str(&format!(
                        "copy_to %{} from %{} [{}]",
                        name(*dst),
                        name(*src),
                        self.dim_text(*dst_offset)
                    ));
                    if dst_offset != src_offset {
                        s.push_str(&format!(" [{}]", self.dim_text(*src_offset)));
                    }
                    s.push('\n');
                }
                Op::Load {
                    value,
                    buffer,
                    offset,
                } => {
                    s.push_str(&format!(
                        "%{} = load %{} [{}]\n",
                        name(*value),
                        name(*buffer),
                        self.dim_text(*offset)
                    ));
                }
                Op::StoreTo {
                    value,
                    buffer,
                    offset,
                } => {
                    s.push_str(&format!(
                        "store_to %{} from %{} [{}]\n",
                        name(*buffer),
                        name(*value),
                        self.dim_text(*offset)
                    ));
                }
                Op::Cast { value, src, dtype } => {
                    s.push_str(&format!(
                        "%{} = cast %{} to {}\n",
                        name(*value),
                        name(*src),
                        dtype.name()
                    ));
                }
                Op::Unary { value, op, a } => {
                    s.push_str(&format!(
                        "%{} = {} %{}\n",
                        name(*value),
                        op.name(),
                        name(*a)
                    ));
                }
                Op::Binary { value, op, a, b } => {
                    s.push_str(&format!(
                        "%{} = {} %{} %{}\n",
                        name(*value),
                        op.name(),
                        name(*a),
                        name(*b)
                    ));
                }
                Op::Select {
                    value,
                    cond,
                    if_true,
                    if_false,
                } => {
                    s.push_str(&format!(
                        "%{} = sel %{} %{} %{}\n",
                        name(*value),
                        name(*cond),
                        name(*if_true),
                        name(*if_false)
                    ));
                }
                Op::Matmul { value, a, b } => {
                    s.push_str(&format!(
                        "%{} = matmul %{} %{}\n",
                        name(*value),
                        name(*a),
                        name(*b)
                    ));
                }
            }
        }
        s
    }

    fn dim_text(&self, dim: Dim) -> String {
        match dim {
            Dim::Const(n) => n.to_string(),
            Dim::Value(v) => format!("%{}", self.names[v as usize]),
        }
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
            return Err(Error::new(format!("the name %{name} is defined twice")));
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
        "loop" => {
            let bound = t.term(&p.symbols)?;
            t.end()?;
            Op::Loop { value, bound }
        }
        "load" => {
            let buffer = t.value(&p.symbols)?;
            let offset = t.bracket(&p.symbols)?;
            t.end()?;
            Op::Load {
                value,
                buffer,
                offset,
            }
        }
        "cast" => {
            let src = t.value(&p.symbols)?;
            t.expect("to")?;
            let dtype = t.dtype()?;
            t.end()?;
            Op::Cast { value, src, dtype }
        }
        "matmul" => {
            let a = t.value(&p.symbols)?;
            let b = t.value(&p.symbols)?;
            t.end()?;
            Op::Matmul { value, a, b }
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
    line: usize,
}

impl<'a> Tokens<'a> {
    fn err(&self, msg: impl Into<String>) -> Error {
        Error::new(format!("line {}: {}", self.line, msg.into()))
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
