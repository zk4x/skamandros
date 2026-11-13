// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The passes over the machine kernel.
//!
//! Three of them are ports of zyx's five: [`reg_sync`] is zyx's per-epoch rule
//! (one publication per `Dst` section, one take before the next writer), [`fuse`]
//! is `tt_reclaim_round_trips`/`tt_dst_reuse`, and [`verify`] is the check every
//! pass ends with. The CB passes (`sync_cbs`, `dedup_pushes`, `place_pops`)
//! belong to the *virtual* IR and live in [`crate::mover`], because they are
//! what decides the reader/compute/writer segments.
//!
//! TODO(the IR re-model): the bodies are the next port. They work on the same
//! ops, and the only structural change is that a register is a slab id
//! (`MachineKernel::dsts`) rather than a flat row.

use crate::ir::Error;
use crate::mir::MachineKernel;

/// Derive the cross-core semaphores from the `Dst` accesses.
pub fn reg_sync(m: &mut MachineKernel) {
    let _ = m;
    todo!("reg_sync is being re-pointed at the machine kernel")
}

/// Fuse the `Dst` round trips a chain of elementwise ops makes.
pub fn fuse(m: &mut MachineKernel) {
    let _ = m;
    todo!("fuse is being re-pointed at the machine kernel")
}

/// Check that every op names a live slab entry, and that an op's rows fit the
/// register it names.
pub fn verify(m: &MachineKernel) -> Result<(), Error> {
    let _ = m;
    todo!("verify is being re-pointed at the machine kernel")
}
