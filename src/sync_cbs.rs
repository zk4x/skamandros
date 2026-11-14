// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The ring hand-offs: the virtual-IR pass that places `wait`/`reserve`/`push`/
//! `pop` where a page crosses a core.
//!
//! It runs on the program `add_movers` assembled — the reader, compute and
//! writer sections, the staged parameter accesses, the loops — and mutates it in
//! place. The rule is tt-metal's:
//!
//! - A producer reserves a page before its write and publishes it after, in the
//!   loop body the write lives in (tt-metal's reader is
//!   `cb_reserve_back`; `noc_async_read_page`; `cb_push_back`). One write is one
//!   page: the ring's write pointer moves only on the push, so the write after
//!   it lands on the next page.
//! - A consumer waits immediately before the read that takes the page: the
//!   unpacker's `load`, or the writer's copy out of the ring.
//! - A page is released after the last op that reads *the page itself*: after
//!   the copy for the writer, after the `matmul`/`sum`/`max` that reads the
//!   loaded value straight from `Src`, and after the `load` itself otherwise
//!   (the unpack has taken the page into `Dst` by then; the encoder drains the
//!   unpacker before the pop).
//! - A page produced outside the loops that consume it — tt-metal's
//!   transformation matrix, its `cos`/`sin` under `RELOAD_IMPL == 0` — is
//!   waited for once where it is produced and released after the loop that
//!   consumes it, never once per iteration.
//!
//! `Op::Cb::page` says what a page is and `Op::Cb::len` how many pages the ring
//! holds; a hand-off's `n` is pages throughout, like `cb_wait_front`.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{Kernel, Op, OpId};
use crate::status::ttStatus;

/// One read of a page, as the scan sees it.
struct Consumer {
    ring: OpId,
    at: OpId,
    /// The loops enclosing the read, outermost first.
    stack: Vec<OpId>,
    /// A copy out of the ring (the writer) rather than a `load` (the unpack).
    writer: bool,
}

/// Insert the ring hand-offs into `k`, which must be a program `add_movers`
/// assembled and no hand-offs yet.
pub fn sync_cbs(k: &mut Kernel) -> Result<(), ttStatus> {
    let mut rings: BTreeSet<OpId> = BTreeSet::new();
    for (id, op) in k.ops() {
        match *op {
            Op::Cb { .. } => {
                rings.insert(id);
            }
            Op::Wait { .. } | Op::Reserve { .. } | Op::Push { .. } | Op::Pop { .. } => {
                eprintln!(
                    "libskamandros: sync_cbs was given a program that already has ring hand-offs (op {id})"
                );
                return Err(ttStatus::TT_INTERNAL_BUG);
            }
            _ => {}
        }
    }

    let mut consumers: Vec<Consumer> = Vec::new();
    let mut end_of: BTreeMap<OpId, OpId> = BTreeMap::new();
    let mut loops: Vec<OpId> = Vec::new();
    // Where each ring is written: the loop depths of its writes. A ring may be
    // filled in more than one place (a page before a loop and a page in it).
    let mut producer_depths: BTreeMap<OpId, std::collections::BTreeSet<usize>> = BTreeMap::new();
    // How many writes each ring has: a hoisted page must be produced exactly
    // once.
    let mut writes: BTreeMap<OpId, usize> = BTreeMap::new();
    // The loads whose page a later `Src` reader still needs, that reader, and
    // the loop stack the reader sits in (so a read that repeats is released
    // after its loop, not once per iteration).
    let mut pending: Vec<(OpId, u8)> = Vec::new();
    let mut last_src: BTreeMap<OpId, (OpId, Vec<OpId>)> = BTreeMap::new();
    let mut position: BTreeMap<OpId, usize> = BTreeMap::new();
    let mut section = 0u8;

    for (at, op) in k.ops() {
        position.insert(at, position.len());
        for (load, load_section) in &pending {
            if *load_section == section
                && (reads_source(op, *load)
                    // An FPU broadcast reads both operands from `Src` pages, so
                    // its op is the page's last reader even though the IR names
                    // it as an ordinary binary op. Without this the pop lands at
                    // the load, before the unpack, and the producer may refill
                    // the page under the unpacker.
                    || crate::ir::broadcast_pair(k, op).is_some_and(|(tile, narrow)| {
                        // A narrow operand that is a scalar immediate is a lane
                        // constant, not a page (the same rule `lower::scalar_const`
                        // folds): the op reads no `Src` page of its own.
                        (tile == *load || narrow == *load) && !is_scalar_immediate(k, narrow)
                    }))
            {
                last_src.insert(*load, (at, loops.clone()));
            }
        }
        match *op {
            Op::EndReader => {
                section = 1;
            }
            Op::EndCompute => {
                section = 2;
            }
            Op::Loop { .. } => loops.push(at),
            Op::EndLoop => {
                let opened = loops.pop().ok_or(ttStatus::TT_ERROR_ENDLOOP_WITHOUT_LOOP)?;
                end_of.insert(opened, at);
            }
            Op::CopyTo { dst, src, .. } => {
                if rings.contains(&src) {
                    consumers.push(Consumer {
                        ring: src,
                        at,
                        stack: loops.clone(),
                        writer: true,
                    });
                } else if rings.contains(&dst) {
                    producer_depths.entry(dst).or_default().insert(loops.len());
                    *writes.entry(dst).or_default() += 1;
                }
            }
            Op::Load { buffer, .. } if section == 1 && rings.contains(&buffer) => {
                consumers.push(Consumer {
                    ring: buffer,
                    at,
                    stack: loops.clone(),
                    writer: false,
                });
                pending.push((at, section));
            }
            Op::StoreTo { buffer, .. } if rings.contains(&buffer) => {
                producer_depths
                    .entry(buffer)
                    .or_default()
                    .insert(loops.len());
                *writes.entry(buffer).or_default() += 1;
            }
            _ => {}
        }
        pending.retain(|(_, load_section)| *load_section == section);
    }

    // A load whose page is still needed by a `Src` reader cannot have another
    // read of the same ring in between: that read would take the same page.
    for (load, (reader, _)) in &last_src {
        let ring = consumer_ring(&consumers, *load)?;
        let (Some(&from), Some(&to)) = (position.get(load), position.get(reader)) else {
            eprintln!("libskamandros: sync_cbs: a load or its `Src` reader has no position");
            return Err(ttStatus::TT_INTERNAL_BUG);
        };
        for consumer in &consumers {
            if consumer.ring != ring {
                continue;
            }
            let Some(&at) = position.get(&consumer.at) else {
                eprintln!("libskamandros: sync_cbs: a consumer has no position");
                return Err(ttStatus::TT_INTERNAL_BUG);
            };
            if at > from && at < to {
                return Err(ttStatus::TT_TODO_RING_READ_BEFORE_RELEASE);
            }
        }
    }

    // `(at, op, before)` insertions, applied after the scan so the ids stay
    // valid. A write is one page, so `n` is one throughout.
    let mut edits: Vec<(OpId, Op, bool)> = Vec::new();
    // Producer hand-offs, in program order, around each write.
    for (at, op) in k.ops() {
        if let Some(ring) = write_ring(op, &rings) {
            edits.push((at, Op::Reserve { cb: ring, n: 1 }, true));
            edits.push((at, Op::Push { cb: ring, n: 1 }, false));
        }
    }
    // Consumer hand-offs. A page produced at the same depth is waited for and
    // released per read; a page produced *outside* the reading loops is hoisted
    // to the producer's scope and shared by the whole loop.
    let mut hoisted: BTreeMap<OpId, OpId> = BTreeMap::new();
    for consumer in &consumers {
        let Some(depths) = producer_depths.get(&consumer.ring) else {
            eprintln!(
                "libskamandros: sync_cbs: ring {} is read at {} but never written",
                consumer.ring, consumer.at
            );
            return Err(ttStatus::TT_INTERNAL_BUG);
        };
        let depth = consumer.stack.len();
        let Some(&producer_depth) = depths.last() else {
            eprintln!(
                "libskamandros: sync_cbs: ring {} has no producer depth",
                consumer.ring
            );
            return Err(ttStatus::TT_INTERNAL_BUG);
        };
        if depths.contains(&depth) {
            if hoisted.contains_key(&consumer.ring) {
                return Err(ttStatus::TT_TODO_RING_IN_AND_OUT_OF_LOOP);
            }
            edits.push((
                consumer.at,
                Op::Wait {
                    cb: consumer.ring,
                    n: 1,
                },
                true,
            ));
            // A page produced at the same depth is released after its last
            // reader. When that reader sits inside a loop the load is outside
            // of, the read repeats, so the release belongs after the loop.
            let pop_at = if consumer.writer || !last_src.contains_key(&consumer.at) {
                consumer.at
            } else {
                let (reader, stack) = &last_src[&consumer.at];
                match stack.get(depth) {
                    Some(outer) => {
                        let Some(&end) = end_of.get(outer) else {
                            eprintln!("libskamandros: sync_cbs: loop {outer} has no end");
                            return Err(ttStatus::TT_INTERNAL_BUG);
                        };
                        end
                    }
                    None => *reader,
                }
            };
            edits.push((
                pop_at,
                Op::Pop {
                    cb: consumer.ring,
                    n: 1,
                },
                false,
            ));
        } else if depth > producer_depth {
            if writes.get(&consumer.ring) != Some(&1) {
                return Err(ttStatus::TT_TODO_RING_MULTI_WRITE_OUTSIDE_LOOP);
            }
            let Some(&outer) = consumer.stack.get(producer_depth) else {
                eprintln!(
                    "libskamandros: sync_cbs: ring {} has a producer depth past its stack",
                    consumer.ring
                );
                return Err(ttStatus::TT_INTERNAL_BUG);
            };
            match hoisted.insert(consumer.ring, outer) {
                Some(previous) if previous != outer => {
                    return Err(ttStatus::TT_TODO_RING_TWO_LOOPS);
                }
                _ => {}
            }
        } else {
            return Err(ttStatus::TT_TODO_RING_READ_OUTSIDE_LOOP);
        }
    }
    for (ring, outer) in hoisted {
        let Some(&end) = end_of.get(&outer) else {
            eprintln!("libskamandros: sync_cbs: loop {outer} has no end");
            return Err(ttStatus::TT_INTERNAL_BUG);
        };
        edits.push((outer, Op::Wait { cb: ring, n: 1 }, true));
        edits.push((end, Op::Pop { cb: ring, n: 1 }, false));
    }

    for (at, op, before) in edits {
        if before {
            k.insert_before(at, op);
        } else {
            k.insert_after(at, op);
        }
    }
    dedup_pushes(k)?;
    Ok(())
}

/// Merge adjacent publishes of one ring into one publish of their sum.
///
/// A ring's write pointer moves on the `push`, so a write between two publishes
/// keeps them apart: the second write lands on the next page *because* the push
/// before it happened. Two `push`es with no op between them therefore publish
/// the same page twice, and only that shape may collapse; the pass never moves
/// a publish past a write or a read.
fn dedup_pushes(k: &mut Kernel) -> Result<(), ttStatus> {
    let mut sums: BTreeMap<OpId, u32> = BTreeMap::new();
    let mut removed: BTreeSet<OpId> = BTreeSet::new();
    // The run being accumulated: its first `push`, the ring, and the pages so
    // far.
    let mut run: Option<(OpId, OpId, u32)> = None;
    for (id, op) in k.ops() {
        match *op {
            Op::Push { cb, n } => {
                if let Some((head, seen, total)) = run {
                    if seen == cb {
                        let Some(sum) = total.checked_add(n) else {
                            return Err(crate::status::internal_bug(
                                "sync_cbs: a merged publish count does not fit a page count",
                            ));
                        };
                        removed.insert(id);
                        sums.insert(head, sum);
                        run = Some((head, cb, sum));
                        continue;
                    }
                }
                run = Some((id, cb, n));
            }
            _ => run = None,
        }
    }
    if removed.is_empty() {
        return Ok(());
    }
    // The slab's ids are the program's order, so the merge is a rebuild of the
    // list with the merged `push`es left out and the surviving one carrying the
    // sum. `renumber` rewrites every operands' reference through the map.
    let mut out = Kernel::default();
    let mut map: BTreeMap<OpId, OpId> = BTreeMap::new();
    for (old, op) in k.ops() {
        if removed.contains(&old) {
            continue;
        }
        let mut op = op.clone();
        if let Some(sum) = sums.get(&old) {
            if let Op::Push { n, .. } = &mut op {
                *n = *sum;
            }
        }
        crate::ir::renumber(&mut op, &map)?;
        let id = out.push(op);
        if let Some(name) = k.names.get(&old) {
            out.names.insert(id, name.clone());
        }
        map.insert(old, id);
    }
    *k = out;
    Ok(())
}

/// The ring a `load` reads.
fn consumer_ring(consumers: &[Consumer], load: OpId) -> Result<OpId, ttStatus> {
    consumers
        .iter()
        .find(|consumer| consumer.at == load)
        .map(|consumer| consumer.ring)
        .ok_or_else(|| {
            eprintln!("libskamandros: sync_cbs: load {load} is not a consumer");
            ttStatus::TT_INTERNAL_BUG
        })
}

/// The ring an op writes, if it writes one.
fn write_ring(op: &Op, rings: &BTreeSet<OpId>) -> Option<OpId> {
    match *op {
        Op::StoreTo { buffer, .. } if rings.contains(&buffer) => Some(buffer),
        Op::CopyTo { dst, .. } if rings.contains(&dst) => Some(dst),
        _ => None,
    }
}

/// A scalar operand that is a lane immediate, not a page: a `const`, or a
/// `cast`/`bitcast` chain ending at one (the lowering folds the cast into the
/// immediate).
fn is_scalar_immediate(k: &Kernel, value: OpId) -> bool {
    if k.info(value).shape != crate::ir::RegShape::Scalar {
        return false;
    }
    let mut at = value;
    loop {
        match *k.at(at) {
            Op::Cast { src, .. } | Op::Bitcast { src, .. } => at = src,
            Op::Const { .. } => return true,
            _ => return false,
        }
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
