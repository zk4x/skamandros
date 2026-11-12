// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Automatic allocation: register values to `Dst` slots and lane registers.
//!
//! The caller never names a physical resource. A value's home follows from
//! liveness: slots are reused as soon as the value is dead, and the lane
//! registers of an op's body are its own temporaries. The cross-core hand-offs
//! are emitted by [`crate::lower`] at the `load`/`store_to` that need them.
//!
//! Parameters and buffers are not allocated here: they are L1, placed by
//! [`crate::planner`], and their contents are mutated in place.

use crate::ir::{Error, Kernel, Op, Types, ValueId};

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
    pub place: Vec<Option<Place>>,
    /// Lane registers a value's own body uses: `(input, output)`.
    pub lreg: Vec<(u8, u8)>,
    /// Peak `Dst` slots in use.
    pub dst_peak: u8,
}

impl Alloc {
    /// The home of `value`. Only a register value has one.
    pub fn place_of(&self, value: ValueId) -> Place {
        self.place[value as usize].expect("a register value has a home")
    }
}

/// Allocate homes for the register values of `k`.
pub fn allocate(k: &Kernel, types: &Types) -> Result<Alloc, Error> {
    let n = types.values.len();
    let mut place: Vec<Option<Place>> = vec![None; n];
    let mut lreg: Vec<(u8, u8)> = vec![(0, 1); n];
    let mut free: Vec<bool> = vec![false; DST_SLOTS as usize];
    let mut last_use: Vec<Option<usize>> = vec![None; n];
    let mut freed: Vec<bool> = vec![false; n];
    let mut live = 0usize;
    let mut peak = 0usize;

    for (at, op) in k.ops.iter().enumerate() {
        for v in op.used() {
            last_use[v as usize] = Some(at);
        }
    }

    for (at, op) in k.ops.iter().enumerate() {
        // Free the homes of values that died before this instruction, so an
        // output can reuse its input's slot.
        for id in 0..n {
            let Some(home) = place[id] else { continue };
            if !freed[id] && last_use[id].is_some_and(|u| u <= at) {
                let Place::Dst(slot) = home;
                free[slot as usize] = false;
                live = live.saturating_sub(1);
                freed[id] = true;
            }
        }
        let Some(value) = op.defined() else { continue };
        if !matches!(
            op,
            Op::Load { .. }
                | Op::Cast { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Select { .. }
                | Op::Matmul { .. }
        ) {
            continue;
        }
        // A scalar is index arithmetic (it names a `Dim`), not a datapath
        // value: the movers evaluate it in RISC-V and it has no `Dst` home.
        if types.of(value).shape == crate::ir::RegShape::Scalar {
            continue;
        }
        let Some(slot) = free.iter().position(|in_use| !in_use) else {
            let _ = (value, DST_SLOTS);
            return Err(Error::TT_ERROR_DST_SLOTS_EXHAUSTED);
        };
        free[slot] = true;
        place[value as usize] = Some(Place::Dst(slot as u8));
        lreg[value as usize] = (0, 1);
        live += 1;
        peak = peak.max(live);
    }
    Ok(Alloc {
        place,
        lreg,
        dst_peak: peak as u8,
    })
}
