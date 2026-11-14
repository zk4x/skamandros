// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The passes over the machine IR: role synchronization, fusion and
//! verification.
//!
//! Allocation (`Dst` slots, `Src` banks, lane registers) happens in
//! [`crate::alloc`], before lowering, because it is an SSA property. What is
//! left to do after the machine IR exists is the row body: the lowering emits
//! one load/op/store triple per `Dst` row, and fusion collapses a run of them
//! into one body plus a repeat, which is what the hardware's replay expander and
//! row-address modifier execute. The hand-offs between the three roles are
//! derived here too ([`reg_sync`]), from the accesses themselves.

use crate::ir::Error;
use crate::mir::{Core, Dst, MachineKernel, MirOp};

/// The three compute cores, as 0/1/2: the movers have no `Dst` accesses.
fn compute_slot(core: Core) -> Option<usize> {
    match core {
        Core::Unpack => Some(0),
        Core::Math => Some(1),
        Core::Pack => Some(2),
        Core::Reader | Core::Writer => None,
    }
}

/// One `Dst` access, in program order: the position of the op in
/// [`MachineKernel::ops`], the role that issues it, and whether it writes.
#[derive(Debug, Clone, Copy)]
struct Access {
    at: usize,
    role: Core,
    write: bool,
}

/// An ordered `(producer, consumer)` role edge: one Sync Unit semaphore carries
/// every seam between those two roles.
type Edge = (Core, Core);

/// The role hand-offs over the `Dst` slots.
///
/// The hardware has no `Dst` lock: the Sync Unit holds eight flat 4-bit
/// counters, and the only ordering between the unpacker, the math thread and
/// the packer is one `SEMPOST`/`SEMWAIT` per direction. So the pass derives,
/// per physical slot, who writes it and who reads it, in program order, and
/// emits the seams that are actually needed:
///
/// - a **data** seam when a reader in another role follows a writer: the writer
///   posts after its last write, the reader takes before its first read;
/// - a **reuse** seam when a later epoch overwrites a slot some other role read:
///   the reader posts after its last read, the next writer takes before its
///   first write;
/// - a **loop** seam when both ends are in the same loop body: the body runs
///   `count` times, so the writer may not overwrite the slot until the reader
///   has taken the previous iteration's. Such an edge starts with one token per
///   hand-off the body performs, so the first iteration's writes have nothing to
///   wait for.
///
/// All three are counting hand-offs, so one semaphore serves one ordered edge —
/// every slot and epoch on that edge shares it. The consumer's takes are counted
/// against the producer's posts up to the one it needs, so a consumer that reads
/// the second slot the producer wrote first takes two tokens at once rather than
/// waiting for a post that already happened. Reads by the role that wrote the
/// slot need no seam: they are the same stream. Initial values are zero unless
/// the edge paces a loop; a producer always posts before any take of its edge.
///
/// This must run before [`fuse`]: it reads the cross-role program order, which
/// fusion replaces with the three per-role streams.
pub fn reg_sync(m: &mut MachineKernel) {
    // The pass inserts hand-offs between ops, so it works on the order as a
    // vector and rebuilds the slab at the end.
    let mut ops: Vec<(Core, MirOp)> = m.ops.iter().map(|(_, entry)| *entry).collect();
    // The unpacker's `Dst` target and, per unit, whether it is configured for
    // `Src`: an `UNPACR` writes `Dst` only when its unit is on the `Dst` path.
    let mut target: Option<Dst> = None;
    let mut to_src = [false; 2];
    let mut touches: Vec<Vec<Access>> = vec![Vec::new(); m.dst_slots() as usize];
    for (at, (role, op)) in ops.iter().enumerate() {
        let mut touched: Vec<(Dst, bool)> = Vec::new();
        match *op {
            MirOp::Setdst { reg } => target = Some(reg),
            MirOp::UnpackConfig {
                unit, to_src: src, ..
            } => to_src[unit as usize] = src,
            MirOp::Unpacr { unit, .. } => {
                if !to_src[unit as usize] {
                    let reg = target.expect("the Dst path names its register with Setdst");
                    touched.push((reg, true));
                }
            }
            MirOp::Sfpstore { reg, .. } => touched.push((reg, true)),
            MirOp::Sfpload { reg, .. } => touched.push((reg, false)),
            // The Matrix Unit's accumulator: `MVMUL` carries the register it
            // accumulates into.
            MirOp::Mvmul { reg, .. } => touched.push((reg, true)),
            MirOp::Elw { reg, .. }
            | MirOp::Gapool { reg, .. }
            | MirOp::Gmpool { reg, .. }
            | MirOp::Mova2d { reg, .. }
            | MirOp::Movb2d { reg, .. } => touched.push((reg, true)),
            // `MOVD2B` reads `Dst` into `SrcB`.
            MirOp::Movd2b { reg, .. } => touched.push((reg, false)),
            // `SetPackOffset` only names the register; the transfer that reads
            // it is the `PACR`s of `PackRows`, and that is where the read
            // completes (so the release seam goes after them, not after the
            // setting).
            MirOp::SetPackOffset { reg, .. } => target = Some(reg),
            MirOp::PackRows { .. } => {
                let reg = target.expect("a pack names its register with SetPackOffset");
                touched.push((reg, false));
            }
            _ => {}
        }
        for (reg, write) in touched {
            touches[usize::from(reg.0)].push(Access {
                at,
                role: *role,
                write,
            });
        }
    }
    // The loop bodies. Every role emits a marker for every IR loop at the same
    // point in the program, so the n-th loop a role opens is the n-th loop every
    // other role opens: pairing the roles by that ordinal gives one region per
    // IR loop, spanning all three roles' body ops. (A symbolic bound puts the
    // role's RISC-V count computation between the markers, so the three markers
    // of one loop are not adjacent in the list.)
    let mut levels: Vec<(usize, usize)> = Vec::new();
    let mut stacks: Vec<Vec<usize>> = vec![Vec::new(); 3];
    let mut opened = [0usize; 3];
    for (at, (role, op)) in ops.iter().enumerate() {
        let Some(r) = compute_slot(*role) else {
            continue;
        };
        match op {
            MirOp::Loop { .. } | MirOp::LoopReg { .. } => {
                let ordinal = opened[r];
                opened[r] += 1;
                stacks[r].push(ordinal);
                if ordinal == levels.len() {
                    levels.push((at, 0));
                } else {
                    levels[ordinal].0 = levels[ordinal].0.min(at);
                }
            }
            MirOp::EndLoop => {
                let ordinal = stacks[r].pop().expect("EndLoop matches a Loop");
                levels[ordinal].1 = levels[ordinal].1.max(at);
            }
            _ => {}
        }
    }
    // A data seam whose two ends are in different bodies happens once for the
    // whole loop, not once per iteration: the post goes after the innermost body
    // that holds the writer but not the reader, the take before the innermost
    // body that holds the reader but not the writer. Positions are `insert`
    // indices into `MachineKernel::ops`.
    let hoist_end = |writer: usize, reader: usize| -> usize {
        let mut bodies: Vec<(usize, usize)> = levels
            .iter()
            .filter(|(start, end)| *start < writer && writer < *end)
            .copied()
            .collect();
        bodies.sort_by(|a, b| b.0.cmp(&a.0));
        match bodies
            .into_iter()
            .find(|(start, end)| !(*start < reader && reader < *end))
        {
            Some((_, end)) => end + 1,
            None => writer + 1,
        }
    };
    let hoist_begin = |writer: usize, reader: usize| -> usize {
        let mut bodies: Vec<(usize, usize)> = levels
            .iter()
            .filter(|(start, end)| *start < reader && reader < *end)
            .copied()
            .collect();
        bodies.sort_by(|a, b| b.0.cmp(&a.0));
        match bodies
            .into_iter()
            .find(|(start, end)| !(*start < writer && writer < *end))
        {
            Some((start, _)) => start,
            None => reader,
        }
    };
    // The innermost body containing `at`.
    let scope = |at: usize| -> Option<(usize, usize)> {
        levels
            .iter()
            .filter(|(start, end)| *start < at && at < *end)
            .max_by_key(|(start, _)| *start)
            .copied()
    };
    // A seam names the ordered edge it carries; the semaphore index is assigned
    // once all edges are known. A data seam counts its posts, because the
    // consumer's take count must match them (see below); a reuse or pacing seam
    // is one post and one take, in role order.
    let mut data: Vec<(Edge, usize, usize, usize)> = Vec::new();
    let mut posts: Vec<(usize, Edge)> = Vec::new();
    let mut takes: Vec<(usize, Edge)> = Vec::new();
    // Loop-pacing edges, counted per body: the semaphore starts with one token
    // per hand-off the body performs.
    let mut reuse: Vec<(Edge, usize, u8)> = Vec::new();
    for events in &touches {
        // One epoch per write: the data an epoch publishes is what the reads
        // that follow it (before the next write) consume.
        let mut epochs: Vec<((Core, usize), Vec<(Core, usize, usize)>)> = Vec::new();
        for &Access { at, role, write } in events {
            if write {
                epochs.push(((role, at), Vec::new()));
            } else if let Some(((writer, _), readers)) = epochs.last_mut() {
                if *writer != role {
                    match readers.iter_mut().find(|r| r.0 == role) {
                        Some(r) => r.2 = at,
                        None => readers.push((role, at, at)),
                    }
                }
            }
        }
        for (n, ((writer, write_at), readers)) in epochs.iter().enumerate() {
            for &(reader, first, _) in readers {
                data.push((
                    (*writer, reader),
                    hoist_end(*write_at, first),
                    hoist_begin(*write_at, first),
                    *write_at,
                ));
            }
            // The reuse edge: a slot written again in the next epoch must not be
            // overwritten while this epoch's readers still need it. Only within
            // one body: a next write in another body is the loop wrap, which the
            // pacing seam below covers once per iteration.
            if let Some(((next_writer, next_write), _)) = epochs.get(n + 1) {
                for &(reader, _, last) in readers {
                    if reader != *next_writer && scope(last) == scope(*next_write) {
                        posts.push((last + 1, (reader, *next_writer)));
                        takes.push((*next_write, (reader, *next_writer)));
                    }
                }
            }
        }
        // A body runs its accesses `count` times, so the previous iteration's
        // content is still being read when the first write of the next one
        // starts. The reader that releases the slot is the body's last reader,
        // and the write that overwrites it is the body's first write.
        for (start, end) in &levels {
            let in_body = |a: &Access| *start < a.at && a.at < *end;
            let first_write = events.iter().find(|a| in_body(a) && a.write);
            let last_read = events.iter().rev().find(|a| in_body(a) && !a.write);
            let (Some(w), Some(r)) = (first_write, last_read) else {
                continue;
            };
            if w.role == r.role {
                continue;
            }
            posts.push((r.at + 1, (r.role, w.role)));
            takes.push((w.at, (r.role, w.role)));
            match reuse
                .iter_mut()
                .find(|(edge, body, _)| *edge == (r.role, w.role) && *body == *start)
            {
                Some(count) => count.2 += 1,
                None => reuse.push(((r.role, w.role), *start, 1)),
            }
        }
    }
    if data.is_empty() && posts.is_empty() && takes.is_empty() {
        return;
    }
    let mut edges: Vec<Edge> = Vec::new();
    for &(edge, _, _, _) in &data {
        if !edges.contains(&edge) {
            edges.push(edge);
        }
    }
    for &(_, edge) in posts.iter().chain(takes.iter()) {
        if !edges.contains(&edge) {
            edges.push(edge);
        }
    }
    edges.sort_unstable();
    assert!(
        edges.len() <= tt_isa::sync::Semaphore::COUNT as usize,
        "reg_sync: {} role hand-off edges need more than the {} semaphores of the Sync Unit",
        edges.len(),
        tt_isa::sync::Semaphore::COUNT
    );
    let sem = |edge: Edge| -> u8 {
        edges
            .iter()
            .position(|e| *e == edge)
            .expect("every seam's edge is in `edges`") as u8
    };
    // A counting hand-off pairs posts with takes in the producer's order, but a
    // consumer can read a slot the producer wrote *second* before one it wrote
    // first: `div` reads the divisor (the second page) to form the reciprocal
    // before it reads the dividend. So a take needs as many tokens as there are
    // posts up to the one it needs, and the takes are batched at the read that
    // needs them. Within one body the ranks are local; a seam whose ends are in
    // different bodies is a single shot (`hoist_end`/`hoist_begin` have already
    // moved it out), so one token is its whole count.
    //
    // The rank of each data post on its edge within its body: the n-th post of
    // that edge in that scope.
    let mut rank: Vec<(usize, Option<(usize, usize)>, usize)> = Vec::new();
    let mut counted: Vec<(Edge, Option<(usize, usize)>, usize)> = Vec::new();
    let mut data_posts: Vec<&(Edge, usize, usize, usize)> = data.iter().collect();
    data_posts.sort_by_key(|(_, post, _, _)| *post);
    for (edge, post, _, epoch) in &data_posts {
        let at = scope(*post);
        let n = match counted.iter().position(|(e, s, _)| e == edge && *s == at) {
            Some(i) => {
                counted[i].2 += 1;
                counted[i].2
            }
            None => {
                counted.push((*edge, at, 1));
                1
            }
        };
        rank.push((*epoch, at, n));
    }
    let mut taken: Vec<(Edge, Option<(usize, usize)>, usize)> = Vec::new();
    let mut data_takes: Vec<&(Edge, usize, usize, usize)> = data.iter().collect();
    data_takes.sort_by_key(|(_, _, take, _)| *take);
    let mut inserts: Vec<(usize, (Core, MirOp))> = Vec::new();
    for (edge, post, take, epoch) in &data_takes {
        let at = scope(*take);
        let hops = *post == *epoch + 1 && scope(*epoch) == at;
        let need = if hops {
            rank.iter()
                .find(|(e, s, _)| e == epoch && *s == at)
                .map(|(_, _, n)| *n)
                .unwrap_or(1)
        } else {
            1
        };
        let seen = match taken.iter().position(|(e, s, _)| e == edge && *s == at) {
            Some(i) => &mut taken[i].2,
            None => {
                taken.push((*edge, at, 0));
                let i = taken.len() - 1;
                &mut taken[i].2
            }
        };
        for _ in *seen..need {
            inserts.push((
                *take,
                (
                    edge.1,
                    MirOp::SemWait {
                        sem: sem(*edge),
                        before: 0,
                    },
                ),
            ));
        }
        *seen = (*seen).max(need);
    }
    for (at, edge) in takes {
        inserts.push((
            at,
            (
                edge.1,
                MirOp::SemWait {
                    sem: sem(edge),
                    before: 0,
                },
            ),
        ));
    }
    for (at, edge) in posts {
        // Inserted in this order because `insert(at, ..)` puts the new op in
        // front of what is already there: the post goes in first, so the guard
        // ends up before it.
        inserts.push((at, (edge.0, MirOp::SemPost { sem: sem(edge) })));
        inserts.push((at, (edge.0, MirOp::SemWaitFree { sem: sem(edge) })));
    }
    // Each data post is one token, guarded like every other post.
    for (edge, post, _, _) in &data {
        inserts.push((*post, (edge.0, MirOp::SemPost { sem: sem(*edge) })));
        inserts.push((*post, (edge.0, MirOp::SemWaitFree { sem: sem(*edge) })));
    }
    // Apply from the back, so the earlier positions stay valid.
    inserts.sort_by(|a, b| b.0.cmp(&a.0));
    for (at, op) in inserts {
        ops.insert(at, op);
    }
    // `Semset` is not an instruction: `encode::setup` folds the table below into
    // the run's semaphore inits, before any role is released.
    let mut inits: Vec<(Core, MirOp)> = Vec::new();
    for (index, edge) in edges.iter().enumerate() {
        let value = reuse
            .iter()
            .filter(|(e, _, _)| e == edge)
            .map(|(_, _, count)| *count)
            .max()
            .unwrap_or(0);
        assert!(
            value <= tt_isa::sync::MAX_VALUE,
            "reg_sync: edge {edge:?} performs {value} loop hand-offs per iteration, more than the {} a semaphore can start with",
            tt_isa::sync::MAX_VALUE
        );
        inits.push((
            Core::Unpack,
            MirOp::Semset {
                sem: index as u8,
                value,
            },
        ));
    }
    m.sems = inits
        .iter()
        .filter_map(|(_, op)| match *op {
            MirOp::Semset { sem, value } => Some((sem, value)),
            _ => None,
        })
        .collect();
    inits.extend(ops);
    let mut fresh = crate::slab::Slab::new();
    for entry in inits {
        fresh.push(entry);
    }
    m.ops = fresh;
}

/// Collapse every run of identical consecutive row triples into one body plus a
/// [`MirOp::Repeat`].
///
/// A triple is fused with the next when the instructions are identical and both
/// `Dst` rows advance by exactly one, so the body is the row-0 form and the
/// repeat walks the rest.
pub fn fuse(m: &mut MachineKernel) {
    let entries: Vec<(Core, MirOp)> = m.ops.iter().map(|(_, entry)| *entry).collect();
    let mut fused: Vec<(Core, MirOp)> = Vec::with_capacity(entries.len());
    // The semaphore inits are not instructions and have no stream; they stay at
    // the front, which is where `reg_sync` put them.
    fused.extend(
        entries
            .iter()
            .filter(|(_, op)| matches!(op, MirOp::Semset { .. }))
            .copied(),
    );
    // The movers' segments are not row bodies: they stay as they are, on their
    // own two cores.
    for role in [Core::Reader, Core::Writer] {
        fused.extend(entries.iter().filter(|(core, _)| *core == role).copied());
    }
    for role in [Core::Unpack, Core::Math, Core::Pack] {
        let ops: Vec<MirOp> = entries
            .iter()
            .filter(|(core, _)| *core == role)
            .map(|(_, op)| *op)
            .collect();
        let mut out: Vec<MirOp> = Vec::with_capacity(ops.len());
        let mut i = 0usize;
        while i < ops.len() {
            let Some(body) = triple(&ops, i) else {
                out.push(ops[i]);
                i += 1;
                continue;
            };
            let mut count = 1u16;
            let mut next = i + 3;
            let mut prev = body;
            while let Some(following) = triple(&ops, next) {
                if !same_shifted(&prev, &following) {
                    break;
                }
                prev = following;
                count += 1;
                next += 3;
            }
            if count > 1 {
                out.push(MirOp::Repeat { len: 3, count });
            }
            out.extend_from_slice(&ops[i..i + 3]);
            i = next;
        }
        for op in out {
            fused.push((role, op));
        }
    }
    let mut fresh = crate::slab::Slab::new();
    for entry in fused {
        fresh.push(entry);
    }
    m.ops = fresh;
}

/// The three ops at `at` if they are a lane-register row body.
fn triple(ops: &[MirOp], at: usize) -> Option<[MirOp; 3]> {
    let [a, b, c] = [*ops.get(at)?, *ops.get(at + 1)?, *ops.get(at + 2)?];
    match (a, b, c) {
        (
            MirOp::Sfpload { .. },
            MirOp::UnarySfp { .. } | MirOp::BinarySfp { .. },
            MirOp::Sfpstore { .. },
        ) => Some([a, b, c]),
        _ => None,
    }
}

/// Is `following` `body` with both `Dst` addresses two rows further on, and the
/// lane op itself identical?
fn same_shifted(body: &[MirOp; 3], following: &[MirOp; 3]) -> bool {
    match (body, following) {
        (
            [
                MirOp::Sfpload {
                    lreg: la,
                    reg: ra,
                    at: aa,
                    dtype: ia,
                },
                op_a,
                MirOp::Sfpstore {
                    lreg: sa,
                    reg: sra,
                    at: saa,
                    dtype: isa,
                },
            ],
            [
                MirOp::Sfpload {
                    lreg: lb,
                    reg: rb,
                    at: ab,
                    dtype: ib,
                },
                op_b,
                MirOp::Sfpstore {
                    lreg: sb,
                    reg: srb,
                    at: sab,
                    dtype: isb,
                },
            ],
        ) => {
            la == lb
                && ra == rb
                && ia == ib
                && *ab == *aa + 2
                && op_a == op_b
                && sa == sb
                && sra == srb
                && isa == isb
                && *sab == *saa + 2
        }
        _ => false,
    }
}

/// Check every machine-IR invariant. A kernel that fails this must not be
/// encoded.
pub fn verify(m: &MachineKernel) -> Result<(), Error> {
    // The semaphore table: every index a stream names must be initialised, and
    // every init names a semaphore the Sync Unit has.
    let sems = m.sems.clone();
    for (i, (sem, value)) in sems.iter().enumerate() {
        if tt_isa::sync::Semaphore::new(*sem).is_none() || *value > tt_isa::sync::MAX_VALUE {
            let _ = (sem, value);
            return Err(Error::TT_ERROR_SEMAPHORE);
        }
        if sems[..i].iter().any(|(s, _)| s == sem) {
            let _ = sem;
            return Err(Error::TT_ERROR_SEMAPHORE);
        }
    }
    for (name, ops) in [
        ("unpack", m.stream(Core::Unpack).collect::<Vec<_>>()),
        ("math", m.stream(Core::Math).collect::<Vec<_>>()),
        ("pack", m.stream(Core::Pack).collect::<Vec<_>>()),
    ] {
        let mut at = 0usize;
        let mut depth = 0usize;
        // The `Dst` row the last `SetPackOffset` named, so `PackRows` can check
        // that the run it moves stays inside one register.
        let mut pack_at: Option<u32> = None;
        while at < ops.len() {
            match ops[at] {
                MirOp::Sfpload { lreg, reg, .. } => {
                    if usize::from(lreg.0) >= usize::from(crate::alloc::LREGS) {
                        let _ = (name, at, lreg);
                        return Err(Error::TT_ERROR_LANE_REGISTER);
                    }
                    if reg.0 >= m.dst_slots() {
                        let _ = (name, at, reg);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                }
                MirOp::Sfpstore {
                    lreg, reg, at: row, ..
                } => {
                    if usize::from(lreg.0) >= usize::from(crate::alloc::LREGS) {
                        let _ = (name, at, lreg);
                        return Err(Error::TT_ERROR_LANE_REGISTER);
                    }
                    if reg.0 >= m.dst_slots() || row >= Dst::ROWS {
                        let _ = (name, at, reg, row);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                    // `SFPSTORE`'s `dest_reg_addr` names a row *pair*: the
                    // hardware refuses an odd row (`tensix.cpp`:
                    // `TTSIM_VERIFY(!(dest_reg_addr & 1))`) and writes two.
                    if row & 1 != 0 {
                        let _ = (name, at, row);
                        return Err(Error::TT_ERROR_DST_ROW_ODD);
                    }
                }
                MirOp::LaneConst { lreg, .. } => {
                    if usize::from(lreg.0) >= usize::from(crate::alloc::LREGS) {
                        let _ = (name, at, lreg);
                        return Err(Error::TT_ERROR_LANE_REGISTER);
                    }
                }
                MirOp::UnarySfp { a, d, .. } => {
                    for lreg in [a, d] {
                        if usize::from(lreg.0) >= usize::from(crate::alloc::LREGS) {
                            let _ = (name, at, lreg);
                            return Err(Error::TT_ERROR_LANE_REGISTER);
                        }
                    }
                }
                MirOp::BinarySfp { a, b, d, .. } => {
                    for lreg in [a, b, d] {
                        if usize::from(lreg.0) >= usize::from(crate::alloc::LREGS) {
                            let _ = (name, at, lreg);
                            return Err(Error::TT_ERROR_LANE_REGISTER);
                        }
                    }
                }
                MirOp::Cast { a, d, .. } => {
                    for lreg in [a, d] {
                        if usize::from(lreg.0) >= usize::from(crate::alloc::LREGS) {
                            let _ = (name, at, lreg);
                            return Err(Error::TT_ERROR_LANE_REGISTER);
                        }
                    }
                }
                MirOp::Setdst { reg } => {
                    if reg.0 >= m.dst_slots() {
                        let _ = (name, at, reg);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                }
                MirOp::SemPost { sem }
                | MirOp::SemWait { sem, .. }
                | MirOp::SemWaitFree { sem } => {
                    if tt_isa::sync::Semaphore::new(sem).is_none()
                        || !sems.iter().any(|(s, _)| *s == sem)
                    {
                        let _ = (name, at, sem);
                        return Err(Error::TT_ERROR_SEMAPHORE);
                    }
                }
                MirOp::Semset { .. } => {
                    unreachable!("a semaphore init is not an instruction and has no role stream")
                }
                MirOp::Repeat { len, count } => {
                    if len == 0 || count == 0 {
                        let _ = (name, at);
                        return Err(Error::TT_ERROR_REPEAT_EMPTY);
                    }
                    if at + 1 + len as usize > ops.len() {
                        let _ = (name, at);
                        return Err(Error::TT_ERROR_REPEAT_BODY);
                    }
                }
                MirOp::Compare { a, b, d, .. } => {
                    for lreg in [a, b, d] {
                        if usize::from(lreg.0) >= usize::from(crate::alloc::LREGS) {
                            let _ = (name, at, lreg);
                            return Err(Error::TT_ERROR_LANE_REGISTER);
                        }
                    }
                }
                MirOp::Select { cond, t, f, d } => {
                    for lreg in [cond, t, f, d] {
                        if usize::from(lreg.0) >= usize::from(crate::alloc::LREGS) {
                            let _ = (name, at, lreg);
                            return Err(Error::TT_ERROR_LANE_REGISTER);
                        }
                    }
                }
                MirOp::Max { a, b, d } => {
                    for lreg in [a, b, d] {
                        if usize::from(lreg.0) >= usize::from(crate::alloc::LREGS) {
                            let _ = (name, at, lreg);
                            return Err(Error::TT_ERROR_LANE_REGISTER);
                        }
                    }
                }
                MirOp::SetUnpackBase { cb, .. }
                | MirOp::SetPackBase { cb }
                | MirOp::CbWait { cb }
                | MirOp::CbPop { cb }
                | MirOp::CbReserve { cb }
                | MirOp::CbPush { cb } => {
                    let _ = cb;
                }
                MirOp::Unpacr { unit, count, .. } => {
                    if count == 0 {
                        let _ = (name, at, unit);
                        return Err(Error::TT_ERROR_UNPACK_ZERO_DATUMS);
                    }
                }
                MirOp::SetPackConfig { .. } => {}
                MirOp::SetPackOffset { reg, at: row } => {
                    if reg.0 >= m.dst_slots() || row >= Dst::ROWS {
                        let _ = (name, at, reg, row);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                    pack_at = Some(row);
                }
                MirOp::PackRows { rows, .. } => {
                    if rows == 0 {
                        let _ = (name, at, rows);
                        return Err(Error::TT_ERROR_PACK_ZERO_ROWS);
                    }
                    let row = pack_at.expect("a pack names its row with SetPackOffset");
                    if row + rows > Dst::ROWS {
                        let _ = (name, at, row, rows);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                }
                MirOp::Setadczw { .. } | MirOp::Setadcx { .. } => {}
                MirOp::Mvmul { reg, addr_mod, .. } => {
                    if reg.0 >= m.dst_slots() {
                        let _ = (name, at, reg);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                    if addr_mod > 7 {
                        let _ = (name, at, addr_mod);
                        return Err(Error::TT_ERROR_ADDR_MOD);
                    }
                }
                MirOp::Gapool { reg, addr_mod, .. }
                | MirOp::Gmpool { reg, addr_mod, .. }
                | MirOp::Elw { reg, addr_mod, .. } => {
                    if reg.0 >= m.dst_slots() {
                        let _ = (name, at, reg);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                    if addr_mod > 7 {
                        let _ = (name, at, addr_mod);
                        return Err(Error::TT_ERROR_ADDR_MOD);
                    }
                }
                MirOp::Movd2b {
                    src_row,
                    reg,
                    addr_mod,
                    ..
                }
                | MirOp::Movb2d {
                    src_row,
                    reg,
                    addr_mod,
                    ..
                } => {
                    // `SrcB` is 64 rows of 16 datums.
                    if src_row >= 64 {
                        let _ = (name, at, src_row);
                        return Err(Error::TT_ERROR_SRC_ROW);
                    }
                    if reg.0 >= m.dst_slots() {
                        let _ = (name, at, reg);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                    if addr_mod > 7 {
                        let _ = (name, at, addr_mod);
                        return Err(Error::TT_ERROR_ADDR_MOD);
                    }
                }
                MirOp::Movb2a {
                    src_a_row,
                    addr_mod,
                    src_b_row,
                    ..
                } => {
                    // `SrcA` and `SrcB` are 64 rows of 16 datums.
                    if src_a_row >= 64 || src_b_row >= 64 {
                        let _ = (name, at, src_a_row, src_b_row);
                        return Err(Error::TT_ERROR_SRC_ROW);
                    }
                    if addr_mod > 7 {
                        let _ = (name, at, addr_mod);
                        return Err(Error::TT_ERROR_ADDR_MOD);
                    }
                }
                MirOp::Mova2d {
                    src_row,
                    reg,
                    addr_mod,
                    ..
                } => {
                    if src_row >= 64 {
                        let _ = (name, at, src_row);
                        return Err(Error::TT_ERROR_SRC_ROW);
                    }
                    if reg.0 >= m.dst_slots() {
                        let _ = (name, at, reg);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                    if addr_mod > 7 {
                        let _ = (name, at, addr_mod);
                        return Err(Error::TT_ERROR_ADDR_MOD);
                    }
                }
                MirOp::Setrwc {
                    src_a, src_b, dst, ..
                } => {
                    // Each counter's field is four bits: sixteen rows.
                    if src_a >= 16 || src_b >= 16 || dst >= 16 {
                        let _ = (name, at, src_a, src_b, dst);
                        return Err(Error::TT_ERROR_SETRWC);
                    }
                }
                MirOp::ThreadWord { .. } | MirOp::WaitMatrix => {}
                MirOp::Trnspsrcb
                | MirOp::AluAccCtrl { .. }
                | MirOp::SrcAFormat { .. }
                | MirOp::UnpackConfig { .. } => {}
                MirOp::Loop { count } => {
                    let _ = count;
                    depth += 1;
                    if depth > 2 {
                        let _ = (name, at, depth);
                        return Err(Error::TT_TODO_LOOP_DEPTH);
                    }
                }
                MirOp::LoopReg { reg } => {
                    let _ = reg;
                    depth += 1;
                    if depth > 2 {
                        let _ = (name, at, depth);
                        return Err(Error::TT_TODO_LOOP_DEPTH);
                    }
                }
                MirOp::Li { .. }
                | MirOp::LoadArg { .. }
                | MirOp::LoadCoord { .. }
                | MirOp::IArith { .. } => {}
                MirOp::Copy { bytes, .. } => {
                    if bytes == 0 {
                        let _ = (name, at, bytes);
                        return Err(Error::TT_ERROR_NOC_REQUEST_BYTES);
                    }
                }
                MirOp::EndLoop => {
                    depth = depth.checked_sub(1).ok_or_else(|| {
                        let _ = (name, at);
                        Error::TT_ERROR_ENDLOOP_MACHINE
                    })?;
                }
            }
            at += 1;
        }
        if depth != 0 {
            let _ = (name, depth);
            return Err(Error::TT_ERROR_ENDLOOP_MACHINE);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Kernel;
    use crate::mir::LReg;

    const NEG: &str = "\
version 9
grid 1,1
%in = param f32 4096
%out = param f32 4096
%b0 = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
load 32,32 %x = %b0 [0]
%y = neg %x
store 32,32 %b0 [0] = %y
copy 32,32 %out [0] = %b0 [0]
";

    fn machine() -> MachineKernel {
        let k = Kernel::parse(NEG).unwrap();
        k.verify().unwrap();
        let k = crate::unfold::unfold(&k).unwrap();
        let k = crate::mover::add_movers(&k).unwrap();
        let mut m = k.lower().unwrap();
        // The same order `compile` uses: the seams are derived from the
        // accesses, the allocations are brought inside the hardware's limits,
        // then fusion collapses the row bodies around them.
        reg_sync(&mut m);
        m.alloc().expect("the registers fit");
        m
    }

    /// The math core's program, which is what the row-body tests look at.
    fn math(m: &MachineKernel) -> Vec<MirOp> {
        m.stream(Core::Math).collect()
    }

    /// The IR index of the `n`th math op.
    fn math_at(m: &MachineKernel, n: usize) -> usize {
        m.ops
            .iter()
            .enumerate()
            .filter(|(_, (_, (role, _)))| *role == Core::Math)
            .nth(n)
            .map(|(i, _)| i)
            .expect("the math stream has that many ops")
    }

    #[test]
    fn sixty_four_row_triples_become_one_body() {
        let mut m = machine();
        let before = math(&m).len();
        fuse(&mut m);
        // One repeat marker plus the body, around the two hand-offs and the
        // packer drain wait.
        assert_eq!(math(&m).len(), 7, "got {:?}", math(&m));
        assert!(matches!(math(&m)[1], MirOp::Repeat { len: 3, count: 32 }));
        assert!(matches!(
            math(&m)[2],
            MirOp::Sfpload {
                lreg: LReg(0),
                reg: Dst(0),
                at: 0,
                dtype: crate::ir::DType::F32
            }
        ));
        assert_eq!(math(&m).len(), before - (32 - 1) * 3 + 1);
    }

    #[test]
    fn a_hand_off_breaks_a_run() {
        let mut m = machine();
        // Cut the body: a semaphore in the middle must stop fusion across it.
        let at = math_at(&m, 3 * 16);
        let mut entries: Vec<(Core, MirOp)> = m.ops.iter().map(|(_, e)| *e).collect();
        entries.insert(at, (Core::Math, MirOp::SemPost { sem: 7 }));
        let mut fresh = crate::slab::Slab::new();
        for e in entries {
            fresh.push(e);
        }
        m.ops = fresh;
        fuse(&mut m);
        let repeats = math(&m)
            .iter()
            .filter(|op| matches!(op, MirOp::Repeat { .. }))
            .count();
        assert_eq!(repeats, 2, "one repeat on each side of the hand-off");
    }

    #[test]
    fn a_different_op_is_not_fused() {
        let mut m = machine();
        // Row 1 negated with a different lane register: not the same body.
        let at = math_at(&m, 5);
        let mut entries: Vec<(Core, MirOp)> = m.ops.iter().map(|(_, e)| *e).collect();
        if let MirOp::UnarySfp { d, .. } = &mut entries[at].1 {
            *d = LReg(5);
        }
        let mut fresh = crate::slab::Slab::new();
        for e in entries {
            fresh.push(e);
        }
        m.ops = fresh;
        fuse(&mut m);
        let repeats: Vec<u16> = math(&m)
            .iter()
            .filter_map(|op| match op {
                MirOp::Repeat { count, .. } => Some(*count),
                _ => None,
            })
            .collect();
        assert_eq!(repeats, vec![30], "the changed row splits the run");
    }

    #[test]
    fn the_machine_verifies_before_and_after_fusion() {
        let mut m = machine();
        verify(&m).expect("lowered machine is valid");
        fuse(&mut m);
        verify(&m).expect("fused machine is valid");
    }

    #[test]
    fn a_bad_lane_register_is_refused() {
        let mut m = machine();
        m.push(
            Core::Math,
            MirOp::Sfpload {
                lreg: LReg(9),
                reg: Dst(0),
                at: 0,
                dtype: crate::ir::DType::F32,
            },
        );
        assert!(verify(&m).is_err());
    }

    #[test]
    fn a_dst_row_past_the_last_tile_is_refused() {
        let mut m = machine();
        m.push(
            Core::Math,
            MirOp::Sfpstore {
                lreg: LReg(0),
                reg: Dst(m.dst_slots()),
                at: 0,
                dtype: crate::ir::DType::F32,
            },
        );
        assert!(verify(&m).is_err());
    }

    #[test]
    fn an_odd_sfpstore_row_is_refused() {
        // `SFPSTORE`'s address names a row pair; the hardware refuses an odd
        // one, so the machine IR must too, before it reaches the encoder.
        let mut m = machine();
        m.push(
            Core::Math,
            MirOp::Sfpstore {
                lreg: LReg(0),
                reg: Dst(0),
                at: 1,
                dtype: crate::ir::DType::F32,
            },
        );
        assert_eq!(
            verify(&m).expect_err("an odd row"),
            Error::TT_ERROR_DST_ROW_ODD
        );
    }

    #[test]
    fn a_repeat_that_runs_past_its_program_is_refused() {
        let mut m = machine();
        m.push(Core::Math, MirOp::Repeat { len: 99, count: 2 });
        assert!(verify(&m).is_err());
    }
}
