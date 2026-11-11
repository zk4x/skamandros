// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `compile`: semantic program to a loadable module.
//!
//! The whole pipeline in one place, in the order the design fixes: verify the
//! program, place its L1, allocate its homes and hand-offs, lower to the machine
//! IR, fuse and verify that, encode the three compute programs, and emit the two
//! mover programs. The result is a [`Module`] the runtime holds; a caller only
//! ever sees an opaque handle to it.

use crate::alloc::allocate;
use crate::encode;
use crate::ir::{Error, Kernel};
use crate::lower::lower;
use crate::mir::Machine;
use crate::mover;
use crate::optimize;
use crate::planner::place;

/// One compiled kernel: five programs, the grid they run over, and how many
/// launch arguments they take.
#[derive(Debug, Clone, Default)]
pub struct Module {
    pub grid: [u32; 2],
    /// Kernel parameters the launch supplies an address for, in program order.
    pub params: u32,
    /// The reader program (RISCV B), words.
    pub reader: Vec<u32>,
    /// The writer program (RISCV B), words.
    pub writer: Vec<u32>,
    /// The three compute programs, T0/T1/T2, words.
    pub cores: [Vec<u32>; 3],
}

/// Compile one semantic program. Every refusal is an error, never a partial
/// module.
pub fn compile(k: &Kernel) -> Result<Module, Error> {
    k.verify()?;
    let plan = place(k)?;
    let alloc = allocate(k)?;
    let mut machine: Machine = lower(k, &plan, &alloc)?;
    optimize::fuse(&mut machine);
    optimize::verify(&machine)?;
    let words = encode::encode(&machine)?;
    Ok(Module {
        grid: k.grid,
        params: k.params.len() as u32,
        reader: mover::reader(k, &plan)?,
        writer: mover::writer(k, &plan)?,
        cores: words.words(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const COPY: &str = "\
version 1
grid 1 1
param in f32 8192
param out f32 8192
buf 0 4096 plain rm
buf 1 4096 plain rm
copy_in param 0 offset 0 len 4096 buf 0
copy_out buf 1 param 1 offset 64 len 4096
";

    #[test]
    fn a_copy_kernel_has_both_boundary_programs_and_no_compute() {
        let k = Kernel::parse(COPY).unwrap();
        let m = compile(&k).unwrap();
        assert_eq!(m.params, 2);
        assert!(!m.reader.is_empty());
        assert!(!m.writer.is_empty());
        assert!(m.cores.iter().all(|c| c.is_empty()), "no compute in a copy");
    }
}
