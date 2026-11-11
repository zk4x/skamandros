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
use crate::ir::{Dim, Error, Kernel, Op};
use crate::lower::lower;
use crate::mover;
use crate::optimize;
use crate::planner::place;

/// One compiled kernel: the mover programs, the three compute programs, the
/// grid they run over, and how many parameters the launch supplies.
#[derive(Debug, Clone, Default)]
pub struct Module {
    pub grid: [u32; 2],
    /// Kernel parameters the launch supplies an address for, in program order.
    pub params: u32,
    /// The inbound mover program (DRAM into L1), words.
    pub reader: Vec<u32>,
    /// The outbound mover program (L1 into DRAM), words.
    pub writer: Vec<u32>,
    /// The three compute programs, T0/T1/T2, words.
    pub cores: [Vec<u32>; 3],
    /// The setup program thread 0 runs before the three roles are released: the
    /// semaphore initialisation, sized to the `Dst` slots this kernel uses.
    pub setup: Vec<u32>,
}

/// Compile one program. Every refusal is an error, never a partial module.
pub fn compile(k: &Kernel) -> Result<Module, Error> {
    k.verify()?;
    let k = crate::unroll::constant_loops(k)?;
    let k = &k;
    let types = k.verify()?;
    let grid = match types.grid {
        [Dim::Const(columns), Dim::Const(rows)] => [columns, rows],
        _ => {
            return Err(Error(
                "a symbolic grid is not lowered yet; the parser and verifier accept it".into(),
            ));
        }
    };
    let plan = place(k, &types)?;
    let alloc = allocate(k, &types)?;
    let mut machine = lower(k, &types, &plan, &alloc)?;
    optimize::fuse(&mut machine);
    optimize::verify(&machine)?;
    let words = encode::encode(&machine)?;
    let setup = encode::setup(alloc.dst_peak.max(1))?;
    let reader = mover::inbound(k, &types, &plan)?;
    let writer = mover::outbound(k, &types, &plan)?;
    let params = k
        .ops
        .iter()
        .filter(|op| matches!(op, Op::Param { .. }))
        .count();
    let params = u32::try_from(params).map_err(|_| Error("too many parameters".into()))?;
    let module = Module {
        grid,
        params,
        reader,
        writer,
        cores: words.words(),
        setup,
    };
    if module.reader.is_empty()
        && module.writer.is_empty()
        && module.cores.iter().all(Vec::is_empty)
    {
        return Err(Error(
            "the program does no data movement and no compute".into(),
        ));
    }
    Ok(module)
}
