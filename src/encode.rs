// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Machine IR to tensix words: the five programs, and the run's setup.
//!
//! One function per core: the reader and the writer are RISC-V programs, and
//! the unpacker, the math core and the packer are the Tensix programs the
//! `tt_isa` encoders describe. Every one of them reads its addresses from the
//! kernel's slabs — [`crate::mir::CbInfo`] for a ring's base and page, the
//! `Dst` id for a register — so nothing here decides placement.
//!
//! TODO(the IR re-model): the body is the next port. The encoders are unchanged
//! and live in the commit before the re-model (`741d58f`); what changes is the
//! address source (`MachineKernel::cbs`/`dsts` instead of a `Plan`), the core
//! set (five, not three) and the `Pack`/`Unpack` splits that are already in.

use crate::ir::Error;
use crate::mir::MachineKernel;

/// The five programs, in [`crate::mir::Core`] order.
pub type Words = [Vec<u32>; 5];

/// Encode every core's program.
pub fn encode(m: &MachineKernel) -> Result<Words, Error> {
    let _ = m;
    todo!("the encoder is being re-pointed at the machine kernel's slabs")
}

/// The setup program thread 0 runs before the five cores are released: one
/// semaphore initialisation per entry of the table.
pub fn setup(sems: &[(u8, u8)]) -> Result<Vec<u32>, Error> {
    let _ = sems;
    todo!("the setup program is being re-pointed at the semaphore table")
}
