// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Automatic allocation: register values to `Dst` slots and lane registers.
//!
//! The caller never names a physical resource. A value's home follows from
//! liveness: slots are reused as soon as the value is dead, and the lane
//! registers of an op's body are its own temporaries. The cross-core hand-offs
//! are derived by [`crate::optimize::reg_sync`] from the accesses.
//!
//! Parameters and buffers are not allocated here: they are L1, placed by
//! [`crate::planner`], and their contents are mutated in place.

use std::collections::BTreeMap;

use crate::ir::{Error, Kernel, Op, OpId, Types, ValueId};

/// `Dst` slots the hardware has (32-bit tiles).
pub const DST_SLOTS: u8 = 8;
/// Writable lane registers (`SFPLOAD`/`SFPSTORE` address `L0..L7`).
pub const LREGS: u8 = 8;

/// Where a register value lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    Dst(u8),
}

/// The allocation of one kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alloc {
    /// Home of each value, if it is a register value.
    pub place: BTreeMap<OpId, Place>,
    /// Lane registers a value's own body uses: `(input, output)`.
    pub lreg: BTreeMap<OpId, (u8, u8)>,
    /// Peak `Dst` slots in use.
    pub dst_peak: u8,
}

impl Alloc {
    /// The home of `value`. Only a register value has one.
    pub fn place_of(&self, value: ValueId) -> Place {
        self.place[&value]
    }
}

/// Allocate homes for the register values of `k`.
pub fn allocate(k: &Kernel, types: &Types) -> Result<Alloc, Error> {
    let mut place: BTreeMap<OpId, Place> = BTreeMap::new();
    let mut lreg: BTreeMap<OpId, (u8, u8)> = BTreeMap::new();
    // A kernel with a `max` reduce gives the compiler the last slot for its
    // 1.0 page (see `lower`): the page is written at the start of the kernel
    // and packed before any allocated value is live, so the slot must not be
    // handed out.
    let slots = if k.ops().any(|(_, op)| matches!(op, Op::Max { .. })) {
        DST_SLOTS - 1
    } else {
        DST_SLOTS
    };
    let mut free: Vec<bool> = vec![false; slots as usize];
    let mut last_use: BTreeMap<OpId, OpId> = BTreeMap::new();
    let mut freed: Vec<OpId> = Vec::new();
    let mut live = 0usize;
    let mut peak = 0usize;

    for (id, op) in k.ops() {
        for v in op.used() {
            last_use.insert(v, id);
        }
    }
    // A bitcast has no `Dst` slot: its readers read its source, so the source
    // stays live until the last reader of the bitcast. Walking backwards lets a
    // chain of bitcasts settle in one pass.
    let order: Vec<OpId> = k.ops().map(|(id, _)| id).collect();
    for id in order.into_iter().rev() {
        let op = k.at(id);
        if let Op::Bitcast { value, src, .. } = *op {
            let Some(end) = last_use.get(&value).copied() else {
                continue;
            };
            let use_at = last_use.entry(src).or_insert(end);
            *use_at = (*use_at).max(end);
        }
    }

    for (id, op) in k.ops() {
        // Free the homes of values that died before this instruction, so an
        // output can reuse its input's slot.
        for (value, home) in place.iter() {
            if freed.contains(value) {
                continue;
            }
            if last_use.get(value).is_some_and(|u| *u <= id) {
                let Place::Dst(slot) = *home;
                free[slot as usize] = false;
                live = live.saturating_sub(1);
                freed.push(*value);
            }
        }
        let Some(value) = op.defined() else { continue };
        if !matches!(
            op,
            Op::Acc { .. }
                | Op::Load { .. }
                | Op::Cast { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Select { .. }
        ) {
            continue;
        }
        // A scalar is index arithmetic (it names a `Dim`), not a datapath
        // value: the movers evaluate it in RISC-V and it has no `Dst` home. An
        // `acc` is the exception: a `1,1` acc still folds in a whole `Dst` tile
        // region and is packed from it.
        if types.of(value).shape == crate::ir::RegShape::Scalar && !matches!(op, Op::Acc { .. }) {
            continue;
        }
        let Some(slot) = free.iter().position(|in_use| !in_use) else {
            let _ = (value, slots);
            return Err(Error::TT_ERROR_DST_SLOTS_EXHAUSTED);
        };
        free[slot] = true;
        place.insert(value, Place::Dst(slot as u8));
        lreg.insert(value, (0, 1));
        live += 1;
        peak = peak.max(live);
    }
    Ok(Alloc {
        place,
        lreg,
        dst_peak: peak as u8,
    })
}
