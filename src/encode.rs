// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Machine IR to Tensix instructions.
//!
//! One [`MirOp`] is one or two instructions: a lane-register load is one
//! `SFPLOAD`, a hand-off is `SEMWAIT`+`SEMGET` or a drain wait + `SEMPOST`, and a
//! [`MirOp::Repeat`] is expanded into its body (the replay encoding is a later
//! step; the expansion is the same program, only longer).
//!
//! The three cores are encoded separately, and the mover cores' RISC-V programs
//! are not here: they are code, not Tensix instructions.

use crate::ir::Error;
use crate::mir::{Machine, MirOp, SfpOp};
use tt_isa::sync::Unit;
use tt_isa::isa::Instruction;
use tt_isa::isa::generated::encode;
use tt_isa::sfpu::mod0_fmt;
use tt_isa::sync::{self, Semaphore};

/// One instruction stream per compute core.
#[derive(Debug, Clone, Default)]
pub struct Words {
    pub t0: Vec<Instruction>,
    pub t1: Vec<Instruction>,
    pub t2: Vec<Instruction>,
}

impl Words {
    /// The encoded words, as the program slot takes them.
    pub fn words(&self) -> [Vec<u32>; 3] {
        [
            self.t0.iter().map(|i| i.word()).collect(),
            self.t1.iter().map(|i| i.word()).collect(),
            self.t2.iter().map(|i| i.word()).collect(),
        ]
    }
}

/// Encode only the math core. The other two cores are encoded by [`encode`];
/// this is here because the math path is the one whose encoding is complete, and
/// a test of it should not depend on the others.
pub fn math(m: &Machine) -> Result<Vec<Instruction>, Error> {
    let mut out = Vec::new();
    encode_math(&m.t1, &mut out)?;
    Ok(out)
}

/// Encode all three compute programs.
pub fn encode(m: &Machine) -> Result<Words, Error> {
    let mut out = Words::default();
    encode_math(&m.t1, &mut out.t1)?;
    for op in &m.t0 {
        match *op {
            MirOp::SemPost { sem } => out.t0.extend(sync::post_after(Unit::Unpacker0, semaphore(sem)?)),
            MirOp::SemWait { sem, .. } => out.t0.extend(sync::take(semaphore(sem)?, tt_isa::backend::Before::EVERYTHING)),
            other => return Err(Error(format!("the unpack program is not encoded yet: {other:?}"))),
        }
    }
    for op in &m.t2 {
        match *op {
            MirOp::SemPost { sem } => out.t2.extend(sync::post_after(Unit::Packer, semaphore(sem)?)),
            MirOp::SemWait { sem, .. } => out.t2.extend(sync::take(semaphore(sem)?, tt_isa::backend::Before::EVERYTHING)),
            other => return Err(Error(format!("the pack program is not encoded yet: {other:?}"))),
        }
    }
    Ok(out)
}

fn semaphore(index: u8) -> Result<Semaphore, Error> {
    Semaphore::new(index).ok_or_else(|| Error(format!("semaphore {index} does not exist")))
}

fn encode_math(ops: &[MirOp], out: &mut Vec<Instruction>) -> Result<(), Error> {
    let mut at = 0usize;
    while at < ops.len() {
        match ops[at] {
            MirOp::Repeat { len, count } => {
                let body = ops
                    .get(at + 1..at + 1 + len as usize)
                    .ok_or_else(|| Error(format!("math[{at}]: repeat body runs past the program")))?;
                for iteration in 0..u32::from(count) {
                    for op in body {
                        encode_math_op(op, iteration * 2, out)?;
                    }
                }
                at += 1 + len as usize;
                continue;
            }
            op => encode_math_op(&op, 0, out)?,
        }
        at += 1;
    }
    Ok(())
}

/// One math op at `row_delta` (a repeat's iteration offset, two address units per
/// iteration).
fn encode_math_op(op: &MirOp, row_delta: u32, out: &mut Vec<Instruction>) -> Result<(), Error> {
    let bad = |e: tt_isa::isa::EncodeError| Error(format!("{op:?}: {e}"));
    match *op {
        MirOp::Sfpload { lreg, row } => {
            let i = encode::sfpload(u32::from(lreg), mod0_fmt::FP32, 0, row + row_delta).map_err(bad)?;
            out.push(i);
        }
        MirOp::Sfpstore { lreg, row } => {
            let i = encode::sfpstore(u32::from(lreg), mod0_fmt::FP32, 0, row + row_delta).map_err(bad)?;
            out.push(i);
        }
        MirOp::Sfp { op: SfpOp::Neg, a, b: None, d } => {
            let i = encode::sfpmov(u32::from(a), u32::from(d), 1).map_err(bad)?;
            out.push(i);
        }
        MirOp::Sfp { op: SfpOp::Abs, a, b: None, d } => {
            let i = encode::sfpabs(u32::from(a), u32::from(d), 1).map_err(bad)?;
            out.push(i);
        }
        MirOp::Sfp { .. } => return Err(Error(format!("{op:?} has no encoding yet"))),
        MirOp::SemPost { sem } => {
            let s = Semaphore::new(sem).ok_or_else(|| Error(format!("semaphore {sem} does not exist")))?;
            out.extend(sync::post_after(Unit::Sfpu, s));
        }
        MirOp::SemWait { sem, .. } => {
            let s = Semaphore::new(sem).ok_or_else(|| Error(format!("semaphore {sem} does not exist")))?;
            out.extend(sync::take(s, tt_isa::backend::Before::EVERYTHING));
        }
        other => return Err(Error(format!("{other:?} is not a math instruction"))),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::allocate;
    use crate::ir::Kernel;
    use crate::lower::lower;
    use crate::optimize::fuse;
    use crate::planner::place;

    const NEG: &str = "\
version 1
grid 1 1
param in f32 4096
param out f32 4096
buf 0 4096 plain
buf 1 4096 plain
copy_in param 0 offset 0 len 4096 buf 0
%0 = unpack buf 0
%1 = sfpu neg %0
pack %1 buf 1
copy_out buf 1 param 1 offset 0 len 4096
";

    fn math_program(fused: bool) -> Vec<String> {
        let k = Kernel::parse(NEG).unwrap();
        let plan = place(&k).unwrap();
        let alloc = allocate(&k).unwrap();
        let mut m = lower(&k, &plan, &alloc).unwrap();
        if fused {
            fuse(&mut m);
        }
        math(&m).unwrap().iter().map(|i| i.def().mnemonic().to_string()).collect()
    }

    #[test]
    fn the_math_body_is_a_load_negate_store_per_address() {
        let k = Kernel::parse(NEG).unwrap();
        let plan = place(&k).unwrap();
        let alloc = allocate(&k).unwrap();
        let m = lower(&k, &plan, &alloc).unwrap();
        let words = math(&m).unwrap();
        let m: Vec<String> = words.iter().map(|i| i.def().mnemonic().to_string()).collect();
        // `SEMWAIT`+`SEMGET` first, `SEMWAIT`+`SEMPOST` last.
        assert_eq!(&m[0..2], ["SEMWAIT", "SEMGET"]);
        assert_eq!(&m[2..5], ["SFPLOAD", "SFPMOV", "SFPSTORE"]);
        assert_eq!(m.iter().filter(|x| *x == "SFPLOAD").count(), 32);
        assert_eq!(m.iter().filter(|x| *x == "SFPMOV").count(), 32);
        assert_eq!(m.iter().filter(|x| *x == "SFPSTORE").count(), 32);
        assert_eq!(m[m.len() - 1], "SEMPOST");
    }

    #[test]
    fn fusion_and_expansion_encode_the_same_program() {
        assert_eq!(math_program(true), math_program(false));
    }

    #[test]
    fn the_unpack_and_pack_programs_are_not_guessed() {
        let k = Kernel::parse(NEG).unwrap();
        let plan = place(&k).unwrap();
        let alloc = allocate(&k).unwrap();
        let m = lower(&k, &plan, &alloc).unwrap();
        let err = encode(&m).expect_err("the unpack program is not encoded yet");
        assert!(err.to_string().contains("not encoded yet"), "{err}");
    }
}
