// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Unified program to machine IR: the first lowering step.
//!
//! `lower` is the *only* step that goes from the SSA program to the machine
//! IR, and it runs the simplest possible allocator: one `Dst` per value, one
//! lane register per temporary, one ring per buffer — no reuse, no compacting,
//! no physical limit. The ids it mints are names, and [`crate::alloc::alloc`]
//! is the pass that turns them into the registers the hardware has.
//!
//! `load` configures and fills the unpacker, the register ops become
//! load/op/store bodies on the math core, `store_to` configures and drains the
//! packer, `copy_to` is the movers' segment, and the `wait`/`reserve`/`push`/
//! `pop` the CB pass inserted become the ring hand-offs. The cores run as a
//! ring paced by semaphores: the packer hands a `Dst` register back, the
//! unpacker publishes each loaded unit, and the math core publishes each stored
//! one.
//!
//! TODO(the IR re-model): the body is the next port. What it must read from the
//! virtual IR is now recomputed rather than tabulated (`Kernel::dtype`,
//! `Kernel::info`, `Kernel::shape`, `Kernel::const_i32`), and what it must mint
//! is `MachineKernel::dsts`/`lregs`/`cbs`, not a `Plan`/`Alloc` pair. The
//! lowering itself — the instruction sequences — is unchanged; the commit
//! before the re-model (`741d58f`) carries it.

use crate::ir::{Error, Kernel};
use crate::mir::MachineKernel;

impl Kernel {
    /// Lower the verified SSA program to the machine IR, with one register per
    /// value and no hardware limit applied.
    pub fn lower(&self) -> Result<MachineKernel, Error> {
        todo!("lowering is being re-pointed at the recomputed IR")
    }
}
