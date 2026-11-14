// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Rendering a machine kernel into the module a launch runs.

use crate::encode;
use crate::ir::{Error, Kernel};
use crate::mir::{CbInfo, MachineKernel};

pub use crate::module::{GridTerm, Module, ParamKind};

/// Compile one program: the virtual passes, the lowering, the machine passes,
/// the allocation and the render — in that order, and nothing else.
///
/// Every refusal is an error, never a partial module.
pub fn compile(k: &Kernel) -> Result<Module, Error> {
    k.verify()?;
    let k = crate::unfold::unfold(k)?;
    let k = &k;
    k.verify()?;
    let k = crate::mover::add_movers(k)?;
    let k = &k;
    k.verify()?;
    let mut machine = k.lower()?;
    // The hand-offs are derived from the accesses, before fusion replaces the
    // cross-core order with the per-core streams.
    crate::optimize::reg_sync(&mut machine);
    crate::optimize::fuse(&mut machine);
    // The allocator is what brings the lane registers and `Dst` ids inside the
    // hardware's limits, so it runs before the check that enforces them.
    machine.alloc()?;
    machine.verify()?;
    machine.render()
}

/// Render one machine kernel: the five programs as words, the setup program
/// thread 0 runs, and the launch metadata the lowering carried over.
///
/// `render` is read-only: everything that had to be decided (the rings' places,
/// the registers, the grid and the parameters) was decided before it.
pub fn render(m: &MachineKernel) -> Result<Module, Error> {
    let words = encode::encode(m)?.words()?;
    let setup = encode::setup(&m.sems)?;
    let mut cbs: Vec<CbInfo> = m.cbs.iter().map(|(_, info)| *info).collect();
    cbs.sort_by_key(|info| info.base);
    let iface = m
        .cbs
        .iter()
        .map(|(_, info)| info.iface_entry)
        .min()
        .unwrap_or(0);
    let module = Module {
        grid: m.grid.clone(),
        params: m.params.clone(),
        reader: words[0].clone(),
        writer: words[4].clone(),
        cores: [words[1].clone(), words[2].clone(), words[3].clone()],
        setup,
        iface,
        cbs,
    };
    if module.reader.is_empty()
        && module.writer.is_empty()
        && module.cores.iter().all(Vec::is_empty)
    {
        return Err(Error::TT_ERROR_EMPTY_MODULE);
    }
    Ok(module)
}

/// The launch's parameter kinds, in program order.
pub fn parameter_kinds(k: &Kernel) -> Vec<ParamKind> {
    let mut kinds = Vec::new();
    for (_, op) in k.ops() {
        match *op {
            crate::ir::Op::Param { .. } => kinds.push(ParamKind::Tensor),
            crate::ir::Op::Variable { .. } => kinds.push(ParamKind::Scalar),
            _ => {}
        }
    }
    kinds
}

/// The launch's grid: the two terms of the program's `grid` op.
pub fn grid_terms(k: &Kernel) -> Result<[GridTerm; 2], Error> {
    let mut grid = None;
    for (_, op) in k.ops() {
        if let crate::ir::Op::Grid { x, y } = *op {
            grid = Some((x, y));
        }
    }
    let Some((x, y)) = grid else {
        return Err(Error::TT_ERROR_GRID);
    };
    Ok([term(k, x)?, term(k, y)?])
}

/// One grid term: a foldable expression is a constant, a scalar parameter is
/// the launch's argument, and anything else is refused.
fn term(k: &Kernel, value: crate::ir::OpId) -> Result<GridTerm, Error> {
    if let Some(n) = k.const_i32(value) {
        return u32::try_from(n)
            .map(GridTerm::Const)
            .map_err(|_| Error::TT_ERROR_GRID);
    }
    if matches!(*k.at(value), crate::ir::Op::Variable { .. }) {
        let ordinal = k
            .ops()
            .filter(|(_, op)| {
                matches!(
                    **op,
                    crate::ir::Op::Param { .. } | crate::ir::Op::Variable { .. }
                )
            })
            .position(|(id, _)| id == value)
            .ok_or(Error::TT_ERROR_GRID)?;
        return Ok(GridTerm::Param(ordinal as u32));
    }
    Err(Error::TT_ERROR_GRID)
}
