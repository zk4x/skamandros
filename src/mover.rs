// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

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
//! exactly one reserve/push pair.
//!
//! The segments are *copies* of the program's loops, not a reordering: a copy
//! that lives inside two loops appears in the reader segment inside two loops
//! of its own, with their own induction values and the index expression rebuilt
//! over them. That is what lets the reader run its traffic at its own pace
//! while the compute segment keeps the same indices.

use std::collections::BTreeMap;

use crate::ir::{Error, Kernel, Op, OpId, ValueKind};
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

/// The byte offsets of the launch-argument pair a parameter's address occupies,
/// after [`ARG_HEAD`] words.
pub const fn param_arg(param: u32) -> (i32, i32) {
    let lo = (ARG_HEAD + 2 * param as usize) * 4;
    (lo as i32, lo as i32 + 4)
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
pub fn add_movers(k: &Kernel) -> Result<Kernel, Error> {
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
    fn walk(&mut self) -> Result<(), Error> {
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
                Op::CopyTo {
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
                        return Err(Error::TT_ERROR_COPY_ENDS);
                    }
                    self.copies.push(Copy {
                        path: self.loops.clone(),
                        op: Op::CopyTo {
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
                    let zero = self.zero();
                    self.copies.push(Copy {
                        path: self.loops.clone(),
                        op: Op::CopyTo {
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
                Op::StoreTo {
                    src,
                    buffer,
                    offset,
                    shape,
                } if self.old.info(buffer).kind == ValueKind::Param => {
                    let ring = self.stage(buffer)?;
                    let zero = self.zero();
                    self.copies.push(Copy {
                        path: self.loops.clone(),
                        op: Op::CopyTo {
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
                        Op::StoreTo {
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
    fn stage(&mut self, param: OpId) -> Result<OpId, Error> {
        if let Some(ring) = self.staged_of.get(&param).copied() {
            return Ok(ring);
        }
        let info = self.old.info(param);
        if info.shape != crate::ir::RegShape::Tile {
            let _ = param;
            return Err(Error::TT_TODO_SCALAR_PARAMETER);
        }
        let name = format!("stage{}", self.staged);
        self.staged += 1;
        let ring = self.fresh();
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
    fn fresh(&mut self) -> OpId {
        let id = OpId(u32::MAX - 1 - u32::try_from(self.created).expect("few values"));
        self.created += 1;
        id
    }

    /// A zero offset.
    fn zero(&mut self) -> OpId {
        let id = self.fresh();
        self.prologue.push((
            id,
            Op::Const {
                dtype: crate::ir::INDEX,
                bits: 0,
            },
        ));
        id
    }

    /// Assemble the three segments around the prologue.
    fn assemble(&mut self) -> Result<Kernel, Error> {
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
        for (old, op) in self.handoffs(compute)? {
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
    fn segment(&mut self, into: &mut Kernel, reader: bool) -> Result<(), Error> {
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
    ) -> Result<(), Error> {
        let Some(((old_loop, old_ind), rest)) = path.split_first() else {
            return self.bodies(into, group);
        };
        let Op::Loop { bound } = *self.old.at(*old_loop) else {
            return Err(Error::TT_ERROR_ENDLOOP_WITHOUT_LOOP);
        };
        let bound = self.clone_value(into, bound)?;
        let new = into.push(Op::Loop { bound });
        self.map.insert(*old_ind, new);
        self.nest(into, rest, group)?;
        into.push(Op::EndLoop);
        Ok(())
    }

    /// Emit one group's copies, with their ring hand-offs.
    fn bodies(&mut self, into: &mut Kernel, group: &[Copy]) -> Result<(), Error> {
        for copy in group {
            let Op::CopyTo {
                dst,
                src,
                dst_offset,
                src_offset,
                shape,
            } = copy.op
            else {
                unreachable!("a copy is a CopyTo")
            };
            let ring = if copy.reader { dst } else { src };
            let ring = self.new_ring(ring)?;
            let dst = self.clone_value(into, dst)?;
            let src = self.clone_value(into, src)?;
            let dst_offset = self.clone_value(into, dst_offset)?;
            let src_offset = self.clone_value(into, src_offset)?;
            // The producer reserves a free page and publishes it; the consumer
            // waits for a published page and releases it (tt-metal's
            // `writer_unary_1.cpp` is `cb_wait_front`/`cb_pop_front`).
            if copy.reader {
                into.push(Op::Reserve { cb: ring });
            } else {
                into.push(Op::Wait { cb: ring });
            }
            into.push(Op::CopyTo {
                dst,
                src,
                dst_offset,
                src_offset,
                shape,
            });
            if copy.reader {
                into.push(Op::Push { cb: ring });
            } else {
                into.push(Op::Pop { cb: ring });
            }
        }
        Ok(())
    }

    /// Whether `id` is a value this pass created (rather than one of the old
    /// program's).
    fn is_staged(&self, id: OpId) -> bool {
        self.staged_of.values().any(|ring| *ring == id)
    }

    /// The new id of a ring.
    fn new_ring(&self, old: OpId) -> Result<OpId, Error> {
        self.ring_of
            .get(&old)
            .copied()
            .ok_or(Error::TT_ERROR_NOT_A_BUFFER)
    }

    /// Rebuild `old`'s value inside the segment being built: a prologue value
    /// is shared, an induction comes from the loop copy, and anything else the
    /// segment needs is rebuilt from its definition.
    fn clone_value(&mut self, into: &mut Kernel, old: OpId) -> Result<OpId, Error> {
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
                Err(Error::TT_ERROR_VALUE_UNDEFINED)
            }
        }
    }

    /// Wrap every read of a ring with a `Wait` and its last one with a `Pop`,
    /// and every write with a `Reserve`/`Push` pair. The rings are named by the
    /// *old* ids here; [`Mover::assemble`] renumbers the whole stream.
    fn handoffs(&mut self, compute: Vec<(OpId, Op)>) -> Result<Vec<(OpId, Op)>, Error> {
        // When a page is released. A `load` that lands in `Dst` has consumed its
        // page by the time the op ends; a `load` whose value the Matrix Unit
        // reads straight from `Src` has not, so its page is released after the
        // `matmul`/reduce that reads it.
        let mut release: BTreeMap<usize, Vec<OpId>> = BTreeMap::new();
        for (at, (id, op)) in compute.iter().enumerate() {
            let Some(ring) = self.read_ring(op) else {
                continue;
            };
            let consumed_at = if matches!(*op, Op::Load { .. }) {
                compute
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, (_, later))| reads_source(later, *id))
                    .map(|(at, _)| at)
                    .unwrap_or(at)
            } else {
                at
            };
            release.entry(consumed_at).or_default().push(ring);
        }
        let mut out = Vec::with_capacity(compute.len());
        for (at, (id, op)) in compute.into_iter().enumerate() {
            let read = self.read_ring(&op);
            let write = self.write_ring(&op);
            if let Some(ring) = read {
                out.push((OpId::NULL, Op::Wait { cb: ring }));
            }
            if let Some(ring) = write {
                out.push((OpId::NULL, Op::Reserve { cb: ring }));
            }
            out.push((id, op));
            if let Some(ring) = write {
                out.push((OpId::NULL, Op::Push { cb: ring }));
            }
            for ring in release.remove(&at).unwrap_or_default() {
                out.push((OpId::NULL, Op::Pop { cb: ring }));
            }
        }
        Ok(out)
    }
}

/// Whether `op` reads `value`'s page straight from `Src`: the Matrix Unit's
/// operands are unpacked by the `matmul`/reduce itself, not by the `load`.
fn reads_source(op: &Op, value: OpId) -> bool {
    match *op {
        Op::Matmul { a, b, .. } => a == value || b == value,
        Op::Sum { x, s, .. } => x == value || s == value,
        Op::Max { x, .. } => x == value,
        _ => false,
    }
}

impl Mover<'_> {
    /// Whether `value` names a ring of the program being assembled: one the old
    /// program declared, or one this pass staged.
    fn is_ring(&self, value: OpId) -> bool {
        matches!(self.old.op(value), Some(Op::Cb { .. }))
            || self.staged_of.values().any(|ring| *ring == value)
    }

    /// The ring an op reads, if it reads one.
    fn read_ring(&self, op: &Op) -> Option<OpId> {
        match *op {
            Op::Load { buffer, .. } if self.is_ring(buffer) => Some(buffer),
            Op::CopyTo { src, .. } if self.is_ring(src) => Some(src),
            _ => None,
        }
    }

    /// The ring an op writes, if it writes one.
    fn write_ring(&self, op: &Op) -> Option<OpId> {
        match *op {
            Op::StoreTo { buffer, .. } if self.is_ring(buffer) => Some(buffer),
            Op::CopyTo { dst, .. } if self.is_ring(dst) => Some(dst),
            _ => None,
        }
    }
}
