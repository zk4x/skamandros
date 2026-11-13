// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The compiled program: what a launch runs.

use crate::ir::BinOp;

/// What a launch argument for one parameter is: a value or an address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// A scalar parameter: the argument is its value.
    Scalar,
    /// A tensor parameter: the argument is its DRAM address.
    Tensor,
}

/// One grid extent, resolved at launch from the scalar arguments. A `Dim` is a
/// literal or an `i32` value; this is the value's expression, so the runtime can
/// evaluate it once the launch arguments are known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GridTerm {
    Const(u32),
    /// The value of parameter `i` (program order), which must be a scalar.
    Param(u32),
    Binary(BinOp, Box<GridTerm>, Box<GridTerm>),
}

impl Default for GridTerm {
    fn default() -> Self {
        GridTerm::Const(1)
    }
}

/// One compiled kernel: the mover programs, the three compute programs, the
/// grid they run over, and the parameters the launch supplies.
#[derive(Debug, Clone, Default)]
pub struct Module {
    /// The grid extents, in tiles along x and y.
    pub grid: [GridTerm; 2],
    /// One launch argument per parameter, in program order.
    pub params: Vec<ParamKind>,
    /// The inbound mover program (DRAM into L1), words.
    pub reader: Vec<u32>,
    /// The outbound mover program (L1 into DRAM), words.
    pub writer: Vec<u32>,
    /// The three compute programs, T0/T1/T2, words.
    pub cores: [Vec<u32>; 3],
    /// The setup program thread 0 runs before the three roles are released: the
    /// semaphore initialisation, sized to the `Dst` slots this kernel uses.
    pub setup: Vec<u32>,
    /// The CB interface table the host writes before the launch.
    pub iface: u32,
    /// The rings the table describes, in base-address order.
    pub cbs: Vec<crate::mir::CbInfo>,
}
