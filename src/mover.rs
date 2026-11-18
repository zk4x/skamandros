// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The movers: the virtual-IR pass that stages parameter accesses and draws the
//! three segments.
//!
//! A `load`/`store_to` against a parameter is staged through a fresh one-page
//! L1 ring, with the copy that fills or drains it (`Op::CopyTo`), exactly as
//! zyx's `tt_add_movers` turns an access against a `Global` into a `Copy` into a
//! circular buffer plus the register access.
//!
//! The same pass splits the one list into three segments — the reader's ops,
//! the compute ops, the writer's — with `Op::EndReader`/`Op::EndCompute`. The
//! ring hand-offs (`Op::Wait`/`Reserve`/`Push`/`Pop`) are not this pass's:
//! [`crate::sync_cbs`] places them, on the assembled program, before lowering.
//!
//! The segments are *copies* of the program's loops, not a reordering: a copy
//! that lives inside two loops appears in the reader segment inside two loops
//! of its own, with their own induction values and the index expression rebuilt
//! over them. That is what lets the reader run its traffic at its own pace
//! while the compute segment keeps the same indices.

use std::collections::BTreeMap;

use crate::ir::{Kernel, Op, OpId, ValueKind};
use crate::status::ttStatus;
use tt_isa::noc::niu::TxnId;

/// Transaction ID every request uses. A mover issues in order and drains after
/// each copy, so one ID is enough.
pub const TXN: TxnId = match TxnId::new(1) {
    Some(txn) => txn,
    None => panic!("transaction id 1 exists"),
};

/// The inbound launch argument: the DRAM address a parameter's page is read
/// from.
pub const ARG_HEAD: usize = 4;
/// The launch argument word that holds the CB interface table's address.
pub const ARG_IFACE: usize = 2;

/// The byte offsets of the launch-argument block a parameter occupies.
///
/// Every parameter's block is [`INTERLEAVED_WORDS`] wide, whatever its kind:
/// one `(offset, endpoint)` pair per DRAM channel, then the channel count and
/// the page size. A scalar or a contiguous tensor uses the first pair and a
/// count of one; an interleaved tensor lets the mover resolve a page's channel
/// on the core, which is tt-metal's `InterleavedAddrGen` /
/// `get_noc_addr_from_bank_id`. A fixed width keeps the layout a function of
/// the parameter's index alone, on both the compiler's side and the runtime's.
pub fn param_arg(param: u32) -> (i32, i32) {
    let lo = param_block(param as usize);
    (lo, lo + 4)
}

/// The most channels a parameter can be striped over: every channel a
/// Blackhole can have.
pub const MAX_BANKS: usize = tt_isa::dram::CHANNELS as usize;

/// Words a parameter's block occupies: a pair per channel, then the channel
/// count and the page size.
pub const INTERLEAVED_WORDS: usize = 2 * MAX_BANKS + 2;

/// The byte offset of parameter `index`'s block in the launch arguments.
pub fn param_block(index: usize) -> i32 {
    ((ARG_HEAD + INTERLEAVED_WORDS * index) * 4) as i32
}

/// The byte offset, inside a parameter's block, of its channel count and its
/// page size.
pub const fn interleaved_banks() -> i32 {
    (2 * MAX_BANKS * 4) as i32
}

pub const fn interleaved_page() -> i32 {
    interleaved_banks() + 4
}

/// One operand of a copy, with the loops it lives under.
#[derive(Clone)]
struct Copy {
    /// The enclosing loops, outermost first: the old loop op and its induction.
    path: Vec<(OpId, OpId)>,
    /// The copy itself, with the offsets still naming the old program.
    op: Op,
    /// Into a ring (the reader segment) or out of one (the writer segment).
    reader: bool,
}

/// Stage every parameter access, split the list into the three segments, and
/// insert the ring hand-offs.
pub fn add_movers(k: &Kernel) -> Result<Kernel, ttStatus> {
    let mut m = Mover {
        old: k,
        prologue: Vec::new(),
        staged: 0,
        created: 0,
        compute: Vec::new(),
        copies: Vec::new(),
        prologue_of: BTreeMap::new(),
        ring_of: BTreeMap::new(),
        staged_of: BTreeMap::new(),
        map: BTreeMap::new(),
        loops: Vec::new(),
        names: Vec::new(),
    };
    m.walk()?;
    m.assemble()
}

struct Mover<'a> {
    old: &'a Kernel,
    /// The parameters, variables, rings, shapes and literals, before any
    /// segment, each with the old id it was defined at.
    prologue: Vec<(OpId, Op)>,
    staged: usize,
    /// How many values this pass created, for [`Mover::fresh`].
    created: u32,
    /// The compute segment's ops, with their loop structure, and the old id
    /// each was defined at.
    compute: Vec<(OpId, Op)>,
    copies: Vec<Copy>,
    /// Old prologue value to its new id.
    prologue_of: BTreeMap<OpId, OpId>,
    /// Old ring value to its new id.
    ring_of: BTreeMap<OpId, OpId>,
    /// Old parameter to the ring its access was staged through.
    staged_of: BTreeMap<OpId, OpId>,
    /// Old induction to the new loop op being built, inside one segment.
    map: BTreeMap<OpId, OpId>,
    /// The loop nesting being walked, outermost first.
    loops: Vec<(OpId, OpId)>,
    /// The names of the values this pass created.
    names: Vec<(OpId, String)>,
}

impl Mover<'_> {
    /// One pass over the program: the prologue is emitted, the compute ops are
    /// collected, and the copies are recorded with their loop path.
    fn walk(&mut self) -> Result<(), ttStatus> {
        for (id, op) in self.old.ops() {
            match *op {
                Op::Param { .. } | Op::Variable { .. } | Op::Cb { .. } => {
                    self.prologue.push((id, op.clone()));
                }
                // A shape and the literals it names are the prologue's too: a
                // parameter's shape operand must be defined before it.
                Op::Const { .. } | Op::Shape(_) => {
                    self.prologue.push((id, op.clone()));
                }
                Op::Loop { .. } => {
                    self.compute.push((id, op.clone()));
                    self.loops.push((id, id));
                }
                Op::EndLoop => {
                    self.compute.push((id, op.clone()));
                    self.loops.pop();
                }
                Op::Copy {
                    dst,
                    src,
                    dst_offset,
                    src_offset,
                    shape,
                } => {
                    let reader = self.old.info(src).kind == ValueKind::Param;
                    let writer = self.old.info(dst).kind == ValueKind::Param;
                    if reader == writer {
                        let _ = (dst, src);
                        return Err(ttStatus::TT_ERROR_COPY_ENDS);
                    }
                    self.copies.push(Copy {
                        path: self.loops.clone(),
                        op: Op::Copy {
                            dst,
                            src,
                            dst_offset,
                            src_offset,
                            shape,
                        },
                        reader,
                    });
                }
                Op::Load {
                    buffer,
                    offset,
                    shape,
                } if self.old.info(buffer).kind == ValueKind::Param => {
                    let ring = self.stage(buffer)?;
                    let zero = self.zero()?;
                    self.copies.push(Copy {
                        path: self.loops.clone(),
                        op: Op::Copy {
                            dst: ring,
                            src: buffer,
                            dst_offset: zero,
                            src_offset: offset,
                            shape,
                        },
                        reader: true,
                    });
                    self.compute.push((
                        id,
                        Op::Load {
                            buffer: ring,
                            offset: zero,
                            shape,
                        },
                    ));
                }
                Op::Store {
                    src,
                    buffer,
                    offset,
                    shape,
                } if self.old.info(buffer).kind == ValueKind::Param => {
                    let ring = self.stage(buffer)?;
                    let zero = self.zero()?;
                    self.copies.push(Copy {
                        path: self.loops.clone(),
                        op: Op::Copy {
                            dst: buffer,
                            src: ring,
                            dst_offset: offset,
                            src_offset: zero,
                            shape,
                        },
                        reader: false,
                    });
                    self.compute.push((
                        id,
                        Op::Store {
                            src,
                            buffer: ring,
                            offset: zero,
                            shape,
                        },
                    ));
                }
                _ => self.compute.push((id, op.clone())),
            }
        }
        Ok(())
    }

    /// Define the staging ring of a parameter access, one page of the
    /// parameter's datum.
    fn stage(&mut self, param: OpId) -> Result<OpId, ttStatus> {
        if let Some(ring) = self.staged_of.get(&param).copied() {
            return Ok(ring);
        }
        let info = self.old.info(param);
        if info.shape != crate::ir::RegShape::Tile {
            let _ = param;
            return Err(ttStatus::TT_TODO_SCALAR_PARAMETER);
        }
        let name = format!("stage{}", self.staged);
        self.staged += 1;
        let ring = self.fresh()?;
        self.prologue.push((
            ring,
            Op::Cb {
                dtype: info.dtype,
                page: [32, 32],
                len: 1,
            },
        ));
        self.names.push((ring, name));
        self.staged_of.insert(param, ring);
        Ok(ring)
    }

    /// An id no old op has, for a value this pass creates.
    ///
    /// Ids are handed out downwards from `u32::MAX - 1`, away from the old
    /// program's, so a pass-created value never collides with a parsed one.
    fn fresh(&mut self) -> Result<OpId, ttStatus> {
        let Some(left) = u32::MAX
            .checked_sub(1)
            .and_then(|top| top.checked_sub(self.created))
        else {
            return Err(crate::status::internal_bug(
                "mover: the pass created more values than an OpId can hold",
            ));
        };
        self.created += 1;
        Ok(OpId(left))
    }

    /// A zero offset.
    fn zero(&mut self) -> Result<OpId, ttStatus> {
        let id = self.fresh()?;
        self.prologue.push((
            id,
            Op::Const {
                dtype: crate::ir::INDEX,
                bits: 0,
            },
        ));
        Ok(id)
    }

    /// Assemble the three segments around the prologue.
    fn assemble(&mut self) -> Result<Kernel, ttStatus> {
        let mut out = Kernel::default();
        let mut map: BTreeMap<OpId, OpId> = BTreeMap::new();
        let prologue = std::mem::take(&mut self.prologue);
        for (old, mut op) in prologue {
            crate::ir::renumber(&mut op, &map)?;
            let id = out.push(op);
            map.insert(old, id);
            if matches!(self.old.op(old), Some(Op::Cb { .. })) || self.is_staged(old) {
                self.ring_of.insert(old, id);
            }
            if let Some((_, name)) = self.names.iter().find(|(at, _)| *at == old) {
                out.names.insert(id, name.clone());
            }
        }
        // The segments name the prologue's values, so they are given the map.
        self.prologue_of = map.clone();
        self.segment(&mut out, true)?;
        out.push(Op::EndReader);
        // The compute segment's operands are the old ids; the prologue and the
        // ops already copied into this kernel have new ones, so the map is
        // carried through the stream.
        let compute = std::mem::take(&mut self.compute);
        for (old, op) in compute {
            let mut op = op;
            crate::ir::renumber(&mut op, &map)?;
            let id = out.push(op);
            if !old.is_null() {
                map.insert(old, id);
            }
        }
        out.push(Op::EndCompute);
        // The writer's copies name the prologue and their own clones only: a
        // value the compute segment defines lives inside its loops.
        self.segment(&mut out, false)?;
        Ok(out)
    }

    /// One mover segment: its copies, each inside a copy of the loops it lived
    /// under, with the ring hand-offs around it.
    fn segment(&mut self, into: &mut Kernel, reader: bool) -> Result<(), ttStatus> {
        let mut groups: BTreeMap<Vec<(OpId, OpId)>, Vec<Copy>> = BTreeMap::new();
        for copy in &self.copies {
            if copy.reader != reader {
                continue;
            }
            groups
                .entry(copy.path.clone())
                .or_default()
                .push(copy.clone());
        }
        for (path, group) in groups {
            self.map.clear();
            self.nest(into, &path, &group)?;
        }
        Ok(())
    }

    /// Emit `path`'s loop nest, with `group`'s copies in the innermost body.
    fn nest(
        &mut self,
        into: &mut Kernel,
        path: &[(OpId, OpId)],
        group: &[Copy],
    ) -> Result<(), ttStatus> {
        let Some(((old_loop, old_ind), rest)) = path.split_first() else {
            return self.bodies(into, group);
        };
        let Op::Loop { bound } = *self.old.at(*old_loop) else {
            return Err(ttStatus::TT_ERROR_ENDLOOP_WITHOUT_LOOP);
        };
        let bound = self.clone_value(into, bound)?;
        let new = into.push(Op::Loop { bound });
        self.map.insert(*old_ind, new);
        self.nest(into, rest, group)?;
        into.push(Op::EndLoop);
        Ok(())
    }

    /// Emit one group's copies.
    fn bodies(&mut self, into: &mut Kernel, group: &[Copy]) -> Result<(), ttStatus> {
        for copy in group {
            let Op::Copy {
                dst,
                src,
                dst_offset,
                src_offset,
                shape,
            } = copy.op
            else {
                return Err(crate::status::internal_bug(
                    "mover: a copy group holds a non-copy op",
                ));
            };
            // The copy's ring must be one this pass placed.
            self.new_ring(if copy.reader { dst } else { src })?;
            let dst = self.clone_value(into, dst)?;
            let src = self.clone_value(into, src)?;
            let dst_offset = self.clone_value(into, dst_offset)?;
            let src_offset = self.clone_value(into, src_offset)?;
            // The producer reserves a free page and publishes it; the consumer
            // waits for a published page and releases it (tt-metal's
            // `writer_unary_1.cpp` is `cb_wait_front`/`cb_pop_front`). Those
            // hand-offs are `sync_cbs`'s: this pass only places the copy.
            into.push(Op::Copy {
                dst,
                src,
                dst_offset,
                src_offset,
                shape,
            });
        }
        Ok(())
    }

    /// Whether `id` is a value this pass created (rather than one of the old
    /// program's).
    fn is_staged(&self, id: OpId) -> bool {
        self.staged_of.values().any(|ring| *ring == id)
    }

    /// The new id of a ring.
    fn new_ring(&self, old: OpId) -> Result<OpId, ttStatus> {
        self.ring_of
            .get(&old)
            .copied()
            .ok_or(ttStatus::TT_ERROR_NOT_A_BUFFER)
    }

    /// Rebuild `old`'s value inside the segment being built: a prologue value
    /// is shared, an induction comes from the loop copy, and anything else the
    /// segment needs is rebuilt from its definition.
    fn clone_value(&mut self, into: &mut Kernel, old: OpId) -> Result<OpId, ttStatus> {
        if let Some(new) = self
            .prologue_of
            .get(&old)
            .or_else(|| self.ring_of.get(&old))
            .or_else(|| self.map.get(&old))
            .copied()
        {
            return Ok(new);
        }
        match self.old.at(old).clone() {
            Op::Const { dtype, bits } => Ok(into.push(Op::Const { dtype, bits })),
            Op::GridCoord { axis } => Ok(into.push(Op::GridCoord { axis })),
            Op::Unary { op, a } => {
                let a = self.clone_value(into, a)?;
                Ok(into.push(Op::Unary { op, a }))
            }
            Op::Binary { op, a, b } => {
                let a = self.clone_value(into, a)?;
                let b = self.clone_value(into, b)?;
                Ok(into.push(Op::Binary { op, a, b }))
            }
            Op::Cast { dtype, src } => {
                let src = self.clone_value(into, src)?;
                Ok(into.push(Op::Cast { dtype, src }))
            }
            Op::Bitcast { dtype, src } => {
                let src = self.clone_value(into, src)?;
                Ok(into.push(Op::Bitcast { dtype, src }))
            }
            Op::Select {
                cond,
                if_true,
                if_false,
            } => {
                let cond = self.clone_value(into, cond)?;
                let if_true = self.clone_value(into, if_true)?;
                let if_false = self.clone_value(into, if_false)?;
                Ok(into.push(Op::Select {
                    cond,
                    if_true,
                    if_false,
                }))
            }
            // A vector index's bases live in the page the program copied, so
            // the load that names it has to be in the moved program too: the
            // lowering reads the page's address out of it per lane. The load
            // itself stays a definition — nothing unpacks it — because a value
            // only a copy's index reads never enters a `Dst` slot.
            Op::Load {
                buffer,
                offset,
                shape,
            } => {
                let buffer = self.clone_value(into, buffer)?;
                let offset = self.clone_value(into, offset)?;
                Ok(into.push(Op::Load {
                    buffer,
                    offset,
                    shape,
                }))
            }
            other => {
                let _ = other;
                Err(ttStatus::TT_ERROR_VALUE_UNDEFINED)
            }
        }
    }
}
