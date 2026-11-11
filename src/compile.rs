// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `compile`: semantic program to a loadable module.
//!
//! The whole pipeline in one place, in the order the design fixes: verify the
//! program, place its L1, allocate its homes and hand-offs, lower to the machine
//! IR, fuse and verify that, encode the three compute programs, and emit the two
//! mover programs. The result is bytes: this is what a driver hands back across
//! the C ABI.

use crate::alloc::allocate;
use crate::encode;
use crate::ir::{Error, Kernel};
use crate::lower::lower;
use crate::mir::Machine;
use crate::mover;
use crate::optimize;
use crate::planner::place;

/// One compiled kernel: five programs and the grid they run over.
#[derive(Debug, Clone, Default)]
pub struct Module {
    pub grid: [u32; 2],
    /// The reader program (RISCV B), words.
    pub reader: Vec<u32>,
    /// The writer program (RISCV NC), words.
    pub writer: Vec<u32>,
    /// The three compute programs, T0/T1/T2, words.
    pub cores: [Vec<u32>; 3],
}

/// The byte form's magic.
pub const MAGIC: [u8; 4] = *b"TTMD";
/// The byte form's version.
pub const VERSION: u32 = 1;

impl Module {
    /// The module bytes: `TTMD`, version, grid, then the five programs
    /// (reader, writer, T0, T1, T2), each a word count and its words.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&self.grid[0].to_le_bytes());
        b.extend_from_slice(&self.grid[1].to_le_bytes());
        for words in [&self.reader, &self.writer, &self.cores[0], &self.cores[1], &self.cores[2]] {
            b.extend_from_slice(&(words.len() as u32).to_le_bytes());
            for w in words {
                b.extend_from_slice(&w.to_le_bytes());
            }
        }
        b
    }

    /// Read the module bytes back. A wrong magic or version is refused.
    pub fn from_bytes(bytes: &[u8]) -> Result<Module, Error> {
        if bytes.len() < 16 || bytes[..4] != MAGIC {
            return Err(Error("not a TTMD module".into()));
        }
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let version = u32_at(4);
        if version != VERSION {
            return Err(Error(format!("module version {version} is not {VERSION}")));
        }
        let mut m = Module { grid: [u32_at(8), u32_at(12)], ..Module::default() };
        let mut at = 16usize;
        for slot in 0..5 {
            let n = u32_at(at) as usize;
            at += 4;
            let end = at + 4 * n;
            if end > bytes.len() {
                return Err(Error("module ends mid-program".into()));
            }
            let mut words = Vec::with_capacity(n);
            while at < end {
                words.push(u32_at(at));
                at += 4;
            }
            match slot {
                0 => m.reader = words,
                1 => m.writer = words,
                other => m.cores[other - 2] = words,
            }
        }
        if at != bytes.len() {
            return Err(Error(format!("{} trailing bytes", bytes.len() - at)));
        }
        Ok(m)
    }
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
    fn the_module_round_trips() {
        let k = Kernel::parse(COPY).unwrap();
        let m = compile(&k).unwrap();
        assert!(!m.reader.is_empty());
        assert!(!m.writer.is_empty());
        assert!(m.cores.iter().all(|c| c.is_empty()), "no compute in a copy");
        let back = Module::from_bytes(&m.to_bytes()).unwrap();
        assert_eq!(back.reader, m.reader);
        assert_eq!(back.writer, m.writer);
    }
}
