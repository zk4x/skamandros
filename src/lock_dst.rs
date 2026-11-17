// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The four `Dst` locks, placed on the virtual IR.
//!
//! The math thread and the packer share the `Dst` register file, and tt-metal
//! gives them four hand-offs: `tile_regs_acquire` (math takes it back),
//! `tile_regs_commit` (math hands its section over), `tile_regs_wait` (the
//! packer waits for that section) and `tile_regs_release` (the packer hands
//! `Dst` back). They are [`Op::MathLock`], [`Op::MathUnlock`], [`Op::PackLock`]
//! and [`Op::PackUnlock`] here.
//!
//! Placement is lazy, as in zyx's `tt_lock_dst`: the math cone opens at its
//! first compute op, the packer replaces it at a store to a ring, and a cone
//! that never packs emits no lock at all (a load into `Dst` that only the math
//! thread reads needs no hand-off). Both close at a loop or section boundary,
//! so every loop iteration takes and returns `Dst` once.

use crate::ir::{Kernel, Op, OpId};
use crate::status::ttStatus;

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Unlocked,
    Pack,
}

/// Does `op` run on the math thread? A load unpacks into `Dst`, the arithmetic
/// writes it, and a store to an accumulator writes it in place.
fn is_math(k: &Kernel, op: &Op) -> bool {
    match *op {
        // A `load` writes `Dst` on the unpack thread, and a bitcast emits no
        // instruction at all: neither starts a math section. The unpacker's
        // page is already ordered by the ring hand-offs `sync_cbs` placed.
        Op::Cast { .. }
        | Op::Unary { .. }
        | Op::Binary { .. }
        | Op::Select { .. }
        | Op::Transpose { .. }
        | Op::Matmul { .. }
        | Op::Sum { .. }
        | Op::Max { .. } => true,
        Op::Store { buffer, .. } => matches!(*k.at(buffer), Op::Acc { .. }),
        _ => false,
    }
}

/// Does `op` drain `Dst` to a ring page? That is the packer's work; the
/// reserve/push around it belong to the same cone.
fn is_pack(k: &Kernel, op: &Op) -> bool {
    match *op {
        Op::Reserve { .. } | Op::Push { .. } => true,
        Op::Store { buffer, .. } => matches!(*k.at(buffer), Op::Cb { .. }),
        _ => false,
    }
}

/// Insert the four locks into `k`, which must already have its ring hand-offs.
pub fn lock_dst(k: &mut Kernel) -> Result<(), ttStatus> {
    let mut in_compute = false;
    let mut state = State::Unlocked;
    // The first math op of a cone that has not packed yet: its `MathLock` is
    // only worth inserting if a pack follows.
    let mut pending: Option<OpId> = None;
    // Whether a compute op has run since the last pack. A program that only
    // copies (unpack straight to pack) has none, and needs no `Dst` lock.
    let mut has_math = false;
    // `(at, op)` insertions; applied from the back so positions stay valid.
    let mut edits: Vec<(OpId, Op)> = Vec::new();
    fn close(state: &mut State, at: OpId, edits: &mut Vec<(OpId, Op)>) {
        if *state == State::Pack {
            edits.push((at, Op::PackUnlock));
        }
        *state = State::Unlocked;
    }
    for (id, op) in k.ops() {
        match *op {
            Op::EndReader => {
                in_compute = true;
            }
            Op::EndCompute => {
                if in_compute {
                    close(&mut state, id, &mut edits);
                }
                pending = None;
                in_compute = false;
            }
            Op::Loop { .. } => {
                // A cone does not span a loop: the body runs its own hand-offs
                // every iteration, and a lock hoisted past the back edge would
                // be taken only once.
                if in_compute {
                    close(&mut state, id, &mut edits);
                }
                pending = None;
            }
            Op::EndLoop => {
                if in_compute {
                    close(&mut state, id, &mut edits);
                }
                pending = None;
            }
            _ if in_compute => {
                if is_pack(k, op) {
                    match state {
                        State::Unlocked => {
                            // A cone that has done no compute is a straight
                            // unpack -> pack copy; its page is already ordered
                            // by the ring hand-offs, so it takes no `Dst` lock.
                            if has_math {
                                if let Some(start) = pending.take() {
                                    edits.push((start, Op::MathLock));
                                }
                                // The math thread commits the section the pack
                                // reads, even when the last writer was a loop
                                // that ended before this store.
                                edits.push((id, Op::MathUnlock));
                                edits.push((id, Op::PackLock));
                                state = State::Pack;
                                has_math = false;
                            }
                        }
                        State::Pack => {}
                    }
                } else if is_math(k, op) {
                    has_math = true;
                    match state {
                        // The get is the packer's release: `Dst` is ours again.
                        // A lock is only worth taking if this cone packs.
                        State::Pack => {
                            edits.push((id, Op::PackUnlock));
                            state = State::Unlocked;
                            pending = Some(id);
                        }
                        State::Unlocked => {
                            if pending.is_none() {
                                pending = Some(id);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    // Inserting from the back means a `MathLock` before `pending` and a
    // `MathUnlock` before the pack are both applied after later edits, so the
    // relative order at one position is the order in `edits` reversed. Push
    // each position's ops so the program order survives the reversal.
    edits.sort_by(|a, b| a.0.cmp(&b.0));
    for (at, op) in edits {
        k.insert_before(at, op);
    }
    Ok(())
}
