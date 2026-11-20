// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! `unroll`: a `loop ... unroll` body written out at compile time.
//!
//! A walk whose page indices have to be constants -- the ring hand-off reads
//! the pages off the accesses (`sync_cbs`) -- is then one line per copy in the
//! program, not one line per trip: the program states the shape (`obh x kb`
//! tiles, `nisub x njsub` subblocks) and the loop says how it is walked, and
//! this pass writes the walk out, folding the induction variable into every
//! offset. `Kernel::const_i32` folds the arithmetic that substitution leaves
//! behind, so the lowered program is the one the text would have spelled out.
//!
//! A bound that does not fold to a non-negative constant is refused
//! ([`ttStatus::TT_ERROR_LOOP_BOUND_SYMBOLIC`]): a run-time trip count cannot
//! be written out.
//!
//! The pass runs on the unified IR before `unfold`.

use std::collections::BTreeMap;

use crate::ir::{self, Kernel, Op, OpId};
use crate::status::ttStatus;

/// Write every `loop ... unroll` body out, one copy per trip.
pub fn unroll(k: &Kernel) -> Result<Kernel, ttStatus> {
    let ops: Vec<(OpId, Op)> = k.ops().map(|(id, op)| (id, op.clone())).collect();
    let mut u = Unroll {
        old: k,
        out: Kernel::default(),
    };
    let mut map: BTreeMap<OpId, OpId> = BTreeMap::new();
    u.emit(&ops, 0, ops.len(), &mut map)?;
    Ok(u.out)
}

struct Unroll<'a> {
    old: &'a Kernel,
    out: Kernel,
}

impl Unroll<'_> {
    fn emit(
        &mut self,
        ops: &[(OpId, Op)],
        from: usize,
        to: usize,
        map: &mut BTreeMap<OpId, OpId>,
    ) -> Result<(), ttStatus> {
        let mut at = from;
        while at < to {
            let (id, op) = &ops[at];
            match op {
                Op::Loop {
                    bound,
                    unroll: true,
                } => {
                    let end = matching_end(ops, at, to)?;
                    // The bound may itself be a value the enclosing trip folded
                    // to a constant, so the copied kernel is asked first.
                    let copied = map.get(bound).copied().unwrap_or(*bound);
                    let Some(trips) = self
                        .out
                        .const_i32(copied)
                        .or_else(|| self.old.const_i32(*bound))
                    else {
                        return Err(ttStatus::TT_ERROR_LOOP_BOUND_SYMBOLIC);
                    };
                    if trips < 0 {
                        return Err(ttStatus::TT_ERROR_LOOP_BOUND_SYMBOLIC);
                    }
                    for trip in 0..trips {
                        let value = self.out.push(Op::Const {
                            dtype: ir::INDEX,
                            bits: trip as u32,
                        });
                        // One trip's values are dead once the trip is written:
                        // seed the body's map with the enclosing scope and drop
                        // it, so the next trip gets its own copies.
                        let mut inner = map.clone();
                        inner.insert(*id, value);
                        self.emit(ops, at + 1, end, &mut inner)?;
                    }
                    at = end + 1;
                }
                Op::Loop {
                    bound,
                    unroll: false,
                } => {
                    let end = matching_end(ops, at, to)?;
                    let bound = *map
                        .get(bound)
                        .ok_or(ttStatus::TT_ERROR_VALUE_UNDEFINED)?;
                    let new = self.out.push(Op::Loop {
                        bound,
                        unroll: false,
                    });
                    map.insert(*id, new);
                    self.emit(ops, at + 1, end, map)?;
                    self.out.push(Op::EndLoop);
                    at = end + 1;
                }
                Op::EndLoop => return Err(ttStatus::TT_ERROR_LOOP_UNMATCHED),
                other => {
                    let mut op = other.clone();
                    ir::renumber(&mut op, map)?;
                    let new = self.out.push(op);
                    if let Some(name) = self.old.names.get(id) {
                        self.out.names.insert(new, name.clone());
                    }
                    map.insert(*id, new);
                    at += 1;
                }
            }
        }
        Ok(())
    }
}

/// The index of the `endloop` matching the loop at `at`.
fn matching_end(ops: &[(OpId, Op)], at: usize, to: usize) -> Result<usize, ttStatus> {
    let mut depth = 0usize;
    for (i, (_, op)) in ops.iter().enumerate().take(to).skip(at) {
        match op {
            Op::Loop { .. } => depth += 1,
            Op::EndLoop => {
                let Some(rest) = depth.checked_sub(1) else {
                    return Err(ttStatus::TT_ERROR_LOOP_UNMATCHED);
                };
                depth = rest;
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => {}
        }
    }
    Err(ttStatus::TT_ERROR_LOOP_UNTERMINATED)
}
