// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The movers: the virtual-IR pass that draws the segments and the CB sync.
//!
//! A `load`/`store_to` against a parameter is staged through a fresh one-page
//! L1 ring, with the copy that fills or drains it (`Op::CopyTo`), exactly as
//! zyx's `tt_add_movers` turns an access against a `Global` into a `Copy` into a
//! circular buffer plus the register access.
//!
//! The same pass is what splits the one list into three segments — the reader's
//! ops, the compute ops, the writer's — with `Op::EndReader`/`Op::EndCompute`,
//! and what inserts the ring hand-offs (`Op::Wait`/`Reserve`/`Push`/`Pop`)
//! where a page crosses a core. That is zyx's `tt_sync_cbs`/`dedup_pushes`/
//! `place_pops`, and it happens here, in the virtual IR, before any lowering:
//! a page a core reads is named by exactly one wait, and a page it writes by
//! exactly one reserve/push/pop triple.
//!
//! TODO(the IR re-model): the bodies are the next port. `add_movers` is close
//! (it already rewrites the op list); the segment split has to extract and
//! duplicate loops, which is the new part.

use crate::ir::{Error, Kernel};

/// The inbound launch argument: the DRAM address a parameter's page is read
/// from.
pub const ARG_HEAD: usize = 4;
/// The launch argument word that holds the CB interface table's address.
pub const ARG_IFACE: usize = 2;

/// The pair of launch argument words a parameter's address occupies, and the
/// number of words the address arithmetic walks.
pub const fn param_arg(param: u32) -> (i32, i32) {
    let at = ARG_HEAD as i32 + 2 * param as i32;
    (at, at + 1)
}

/// Stage every parameter access through an L1 ring, split the list into the
/// reader/compute/writer segments, and insert the ring hand-offs.
pub fn add_movers(k: &Kernel) -> Result<Kernel, Error> {
    let _ = k;
    todo!("the mover pass is being re-pointed at the new op set")
}
