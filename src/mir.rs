// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The machine IR: what each of the three compute cores executes.

use crate::ir::DType;

/// One SFPU instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfpOp {
    Neg,
    Abs,
    Add,
    Sub,
    Mul,
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
}

/// A comparison a [`MirOp::Compare`] performs; the result is `-1` for true
/// and `0` for false, lanewise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Gt,
    Ge,
}

/// An integer operation a role computes with RISC-V, for a loop bound that is
/// not a literal. The set is what an index expression may contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Sll,
    Sra,
}

/// This core's `core_info`: the logical grid coordinate the host wrote before
/// the launch. `x` is at this address, `y` four bytes after it.
pub const CORE_INFO: u32 = tt_isa::mailbox::MAILBOX_BASE as u32 + 0x100;

/// One instruction of a compute core's program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirOp {
    /// `SFPLOAD`: `Dst` row into a lane register. `dtype` picks the lane's
    /// format: `INT32` for a `bool` or `i32` value, `BF16` for a 16-bit `Dst`
    /// datum, `FP32` otherwise.
    Sfpload { lreg: u8, row: u32, dtype: DType },
    /// `SFPSTORE`: lane register into a `Dst` row.
    Sfpstore { lreg: u8, row: u32, dtype: DType },
    /// One SFPU arithmetic instruction; `b` is `None` for a unary op.
    Sfp {
        op: SfpOp,
        a: u8,
        b: Option<u8>,
        d: u8,
    },
    /// A lanewise comparison, `-1`/`0` into `d`.
    Compare { op: CmpOp, a: u8, b: u8, d: u8 },
    /// A lanewise conversion from `from` to `to`, in the SFPU's 32-bit lane,
    /// with the rounding the C type has: `(int)f` truncates, `i32` to `f32`
    /// rounds to nearest, `(bool)x` is `x != 0`.
    Cast {
        from: DType,
        to: DType,
        a: u8,
        d: u8,
    },
    /// `cond ? t : f`, lanewise, through the SFPU lane flags.
    Select { cond: u8, t: u8, f: u8, d: u8 },
    /// The lanewise maximum, through the SFPU lane flags.
    Max { a: u8, b: u8, d: u8 },
    /// `UNPACR`: the page at the head of `cb`'s ring into unpacker `unit`'s
    /// destination. `dtype` picks the L1 format: `Bf16` for a 16-bit datum,
    /// `Fp32` otherwise (an `i32` datum is opaque 32 bits). The base is read
    /// from the ring's read pointer at run time.
    Unpacr {
        cb: u32,
        unit: u8,
        count: u32,
        dtype: DType,
    },
    /// `SETADCZW`: which face the unpacker reads next.
    Setadczw { unit: u8, face: u32 },
    /// `SETADCX`: the datum run the unpacker walks.
    Setadcx { unit: u8, first: u32, count: u32 },
    /// The base `Dst` row an unpack writes to.
    Setdst { row: u32 },
    /// The packer drains into the page at the tail of `cb`'s ring. The base is
    /// read from the ring's write pointer at run time; `row` is the `Dst` row
    /// the first packed row comes from, the packer's `W` counter. `dtype` picks
    /// the read path: `Bf16` reads a 16-bit `Dst`, everything else 32-bit.
    Pack {
        cb: u32,
        rows: u32,
        row: u32,
        last: bool,
        dtype: DType,
    },
    /// `cb_wait_front`: wait for a page to be published on this ring.
    CbWait { cb: u32 },
    /// `cb_pop_front`: release the page [`MirOp::CbWait`] waited for, after the
    /// unpacker has drained it.
    CbPop { cb: u32 },
    /// `cb_reserve_back`: wait for a free page on this ring.
    CbReserve { cb: u32 },
    /// `cb_push_back`: publish the page [`MirOp::CbReserve`] reserved, after the
    /// packer has drained `Dst`.
    CbPush { cb: u32 },
    /// `SEMPOST` on this core.
    SemPost { sem: u8 },
    /// `SEMWAIT` on this core, holding `before`'s blocks.
    SemWait { sem: u8, before: u8 },
    /// `SEMWAIT` C0 only: wait until `sem` is non-zero, without taking it. The
    /// consumer's claim; the matching [`MirOp::SemGet`] comes after the work.
    SemWaitData { sem: u8 },
    /// `SEMGET`: release what [`MirOp::SemWaitData`] claimed.
    SemGet { sem: u8 },
    /// The next `len` ops form one row body run `count` times, stepping the
    /// `Dst` row by one per iteration. Placed by [`crate::optimize::fuse`].
    Repeat { len: u16, count: u16 },
    /// Start of a loop: the ops up to the matching [`MirOp::EndLoop`] run
    /// `count` times, as a RISC-V branch around the body. An IR loop that was
    /// constant-bound is this, per role.
    Loop { count: u32 },
    /// Start of a loop whose trip count is already in `reg`: the dynamic bound
    /// was computed by the [`MirOp::Li`]/[`MirOp::LoadArg`]/[`MirOp::IArith`]
    /// ops just before it.
    LoopReg { reg: u8 },
    /// `reg = value`.
    Li { reg: u8, value: u32 },
    /// `reg = *(a0 + arg)`: a launch argument word, which for a scalar
    /// parameter is its value.
    LoadArg { reg: u8, arg: i32 },
    /// `reg = core_info[axis]`: this core's logical grid coordinate.
    LoadCoord { reg: u8, axis: u8 },
    /// `d = a op b`, in RISC-V.
    IArith { op: IOp, d: u8, a: u8, b: u8 },
    /// The branch back to the matching [`MirOp::Loop`] or [`MirOp::LoopReg`].
    EndLoop,
}

/// The three compute programs. The movers' RISC-V programs are emitted
/// separately (`crate::mover`), because they are code, not Tensix instructions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Machine {
    pub t0: Vec<MirOp>,
    pub t1: Vec<MirOp>,
    pub t2: Vec<MirOp>,
}

impl Machine {
    /// Every core's program in order: T0, T1, T2.
    pub fn cores(&self) -> [&[MirOp]; 3] {
        [&self.t0, &self.t1, &self.t2]
    }
}
