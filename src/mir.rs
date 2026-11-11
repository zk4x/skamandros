// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The machine IR: what each of the three compute cores executes.

/// One SFPU instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfpOp {
    Neg,
    Abs,
    Add,
    Sub,
    Mul,
}

/// The comparison a [`MirOp::Compare`] performs; the result is `-1` for true
/// and `0` for false, lanewise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Gt,
    Ge,
}

/// One instruction of a compute core's program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirOp {
    /// `SFPLOAD`: `Dst` row into a lane register. `int` selects the `INT32`
    /// format instead of `FP32`, which is what a `bool` or `i32` value needs.
    Sfpload { lreg: u8, row: u32, int: bool },
    /// `SFPSTORE`: lane register into a `Dst` row.
    Sfpstore { lreg: u8, row: u32, int: bool },
    /// One SFPU arithmetic instruction; `b` is `None` for a unary op.
    Sfp {
        op: SfpOp,
        a: u8,
        b: Option<u8>,
        d: u8,
    },
    /// A lanewise comparison, `-1`/`0` into `d`.
    Compare { op: CmpOp, a: u8, b: u8, d: u8 },
    /// `cond ? t : f`, lanewise, through the SFPU lane flags.
    Select { cond: u8, t: u8, f: u8, d: u8 },
    /// The lanewise maximum, through the SFPU lane flags.
    Max { a: u8, b: u8, d: u8 },
    /// `UNPACR`: the tile image at L1 `src` into unpacker `unit`'s destination.
    Unpacr { src: u32, unit: u8, count: u32 },
    /// `SETADCZW`: which face the unpacker reads next.
    Setadczw { unit: u8, face: u32 },
    /// `SETADCX`: the datum run the unpacker walks.
    Setadcx { unit: u8, first: u32, count: u32 },
    /// The base `Dst` row an unpack writes to.
    Setdst { row: u32 },
    /// The L1 address the packer drains to, and how many rows.
    Pack { dst: u32, rows: u32, last: bool },
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
