// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `compile`: unified program to loadable machine programs.
//!
//! The whole pipeline in one place, in the order the design fixes: verify the
//! program, place its L1, allocate its register homes, lower to the machine IR,
//! fuse and verify that, encode the three compute programs, and emit the mover
//! programs. The reader/compute/writer split is not in the input; it is what
//! falls out of which instructions touch a parameter, a buffer or a register.

use crate::alloc::allocate;
use crate::encode;
use crate::ir::{BinOp, Dim, Error, Kernel, Op};
use crate::lower::lower;
use crate::mover;
use crate::optimize;
use crate::planner::place;
use crate::unfold;

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
    /// The rings the table describes, in cb-id order.
    pub cbs: Vec<crate::planner::Cb>,
}

/// Compile one program. Every refusal is an error, never a partial module.
pub fn compile(k: &Kernel) -> Result<Module, Error> {
    k.verify()?;
    let types = k.verify()?;
    // The transcendentals become primitives here, before anything is placed:
    // every value the unfold creates gets its own `Dst` slot, which is correct
    // and slow, and the allocator pass in `TODO.md` is what fuses the chain.
    let k = unfold::unfold(k, &types)?;
    let k = &k;
    let types = k.verify()?;
    // `tt_add_movers`: a direct parameter access gets its staging and copy.
    let k = mover::add_movers(k, &types)?;
    let k = &k;
    let types = k.verify()?;
    mover::check_roles(k, &types)?;
    let plan = place(k, &types)?;
    let alloc = allocate(k, &types)?;
    let mut machine = lower(k, &types, &plan, &alloc)?;
    optimize::fuse(&mut machine);
    optimize::verify(&machine)?;
    let words = encode::encode(&machine)?;
    let setup = encode::setup(alloc.dst_peak.max(1))?;
    let reader = mover::inbound(k, &types, &plan)?;
    let writer = mover::outbound(k, &types, &plan)?;
    let params = parameter_kinds(k, &types);
    let grid = [grid_term(k, types.grid[0])?, grid_term(k, types.grid[1])?];
    let module = Module {
        grid,
        params,
        reader,
        writer,
        cores: words.words()?,
        setup,
        iface: plan.iface,
        cbs: plan.cbs.clone(),
    };
    if module.reader.is_empty()
        && module.writer.is_empty()
        && module.cores.iter().all(Vec::is_empty)
    {
        return Err(Error::TT_ERROR_EMPTY_MODULE);
    }
    Ok(module)
}

/// One launch argument kind per parameter, in program order.
fn parameter_kinds(k: &Kernel, types: &crate::ir::Types) -> Vec<ParamKind> {
    let mut kinds = Vec::new();
    for op in &k.ops {
        if let Op::Param { value, .. } = *op {
            kinds.push(if types.of(value).shape == crate::ir::RegShape::Scalar {
                ParamKind::Scalar
            } else {
                ParamKind::Tensor
            });
        }
    }
    kinds
}

/// Compile one grid extent. A literal (or an expression that folds to one) is a
/// constant; anything else is an expression over the launch's scalar parameters,
/// evaluated by the runtime before any core starts.
fn grid_term(k: &Kernel, dim: Dim) -> Result<GridTerm, Error> {
    let mut param: Vec<Option<u32>> = vec![None; k.names.len()];
    let mut next = 0u32;
    for op in &k.ops {
        if let Op::Param { value, .. } = *op {
            param[value as usize] = Some(next);
            next += 1;
        }
    }
    match dim {
        Dim::Const(n) => Ok(GridTerm::Const(n)),
        Dim::Value(value) => match k.const_i32(value) {
            Some(n) => u32::try_from(n)
                .map(GridTerm::Const)
                .map_err(|_| Error::TT_ERROR_GRID),
            None => grid_value(k, &param, value),
        },
    }
}

fn grid_value(
    k: &Kernel,
    param: &[Option<u32>],
    value: crate::ir::ValueId,
) -> Result<GridTerm, Error> {
    let op = k
        .ops
        .iter()
        .find(|op| op.defined() == Some(value))
        .ok_or(Error::TT_ERROR_VALUE_UNDEFINED)?;
    match *op {
        Op::Const {
            dtype: crate::ir::DType::I32,
            bits,
            ..
        } => Ok(GridTerm::Const(bits)),
        Op::Param {
            value: p, ref dims, ..
        } if dims.is_empty() => Ok(GridTerm::Param(
            param[p as usize].ok_or(Error::TT_ERROR_ARGUMENT)?,
        )),
        Op::Binary { op, a, b, .. } if grid_op(op) => Ok(GridTerm::Binary(
            op,
            Box::new(grid_value(k, param, a)?),
            Box::new(grid_value(k, param, b)?),
        )),
        Op::GridCoord { .. } => Err(Error::TT_ERROR_GRID_COORD),
        _ => Err(Error::TT_TODO_SYMBOLIC_GRID),
    }
}

/// The arithmetic a grid extent may contain: the same set `Kernel::const_i32`
/// folds.
fn grid_op(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Add
            | BinOp::Sub
            | BinOp::Mul
            | BinOp::Div
            | BinOp::Mod
            | BinOp::Pow
            | BinOp::Max
            | BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::BitXor
            | BinOp::BitShiftLeft
            | BinOp::BitShiftRight
    )
}
