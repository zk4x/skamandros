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
    /// The page the read names inside the ring's block (`load ... = %ba [3]`:
    /// page 3), when that is a constant. The pages one loop body's reads name
    /// are the block the consumer waits for and releases together.
    page: Option<u32>,
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
    // Every write to a ring: its ring, the loop stack it sits in, and the page
    // it names (`copy 32,32 %ba [3]`: page 3), when that is a constant. The
    // pages one loop body's writes name are the block the ring must hold.
    // The `bool` says the write is a `copy` into the ring (the mover's NoC read)
    // rather than a `store` (the packer's L1 write).
    let mut write_info: BTreeMap<OpId, (OpId, Vec<OpId>, Option<u32>, bool)> = BTreeMap::new();
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
            Op::Copy {
                dst,
                src,
                dst_offset,
                src_offset,
                ..
            } => {
                if rings.contains(&src) {
                    consumers.push(Consumer {
                        ring: src,
                        at,
                        stack: loops.clone(),
                        writer: true,
                        page: k.const_i32(src_offset).and_then(|n| u32::try_from(n).ok()),
                    });
                } else if rings.contains(&dst) {
                    producer_depths.entry(dst).or_default().insert(loops.len());
                    *writes.entry(dst).or_default() += 1;
                    let page = k.const_i32(dst_offset).and_then(|n| u32::try_from(n).ok());
                    write_info.insert(at, (dst, loops.clone(), page, true));
                }
            }
            // An `expand` publishes a page of the ring it fans out: the page
            // the leader's `copy` filled travels to every core in the span, so
            // the fan-out is the write on the receiver as much as on the
            // sender. Its page joins the block for the same reason a copy's
            // does.
            Op::Expand { src, pages, .. } if rings.contains(&src) => {
                producer_depths.entry(src).or_default().insert(loops.len());
                *writes.entry(src).or_default() += 1;
                // The fan-out publishes the whole reservation, so the block it
                // names is `pages` long: the last page is `pages - 1`.
                let page = k
                    .const_i32(pages)
                    .and_then(|n| u32::try_from(n).ok())
                    .and_then(|n| n.checked_sub(1));
                write_info.insert(at, (src, loops.clone(), page, true));
            }
            Op::Load { buffer, offset, .. } if section == 1 && rings.contains(&buffer) => {
                consumers.push(Consumer {
                    ring: buffer,
                    at,
                    stack: loops.clone(),
                    writer: false,
                    page: k.const_i32(offset).and_then(|n| u32::try_from(n).ok()),
                });
                pending.push((at, section));
            }
            Op::Store { buffer, offset, .. } if rings.contains(&buffer) => {
                producer_depths
                    .entry(buffer)
                    .or_default()
                    .insert(loops.len());
                *writes.entry(buffer).or_default() += 1;
                let page = k.const_i32(offset).and_then(|n| u32::try_from(n).ok());
                write_info.insert(at, (buffer, loops.clone(), page, false));
            }
            _ => {}
        }
        pending.retain(|(_, load_section)| *load_section == section);
    }

    // The scope of every write, as the folded bound of each enclosing loop
    // (`None` for a run-time bound). A read pairs with the writer scope that
    // encloses it: the k loop on both sides folds to the same `None`, while the
    // peel and the subblock loops name their constants. Positions say nothing
    // -- `add_movers` emits all reader copies before all compute ops.
    let mut producer_paths: BTreeMap<OpId, Vec<Vec<Option<i32>>>> = BTreeMap::new();
    for (_, (ring, stack, _, _)) in &write_info {
        let path = stack
            .iter()
            .map(|loop_op| scope_key(k, *loop_op))
            .collect::<Result<Vec<Option<i32>>, ttStatus>>()?;
        producer_paths.entry(*ring).or_default().push(path);
    }
    // The block: the pages one hand-off of a ring moves. A `copy` moves the
    // page it names; a block `load`/`store` moves its `m * n` tile pages, all
    // at the hand-off's base. tt-metal's reader reserves a block, issues every
    // read of it, barriers once and pushes once
    // (`reader_bmm_tile_layout.cpp:76-113`), and its compute waits the same
    // block and walks the tiles by index (`bmm_large_block_zm.cpp:46-126`).
    //
    // The block is one number for the ring, not one per side: the write pointer
    // and the read pointer only stay in step if both advance the same amount
    // per hand-off. It is the highest page either side names; a page index that
    // is a loop value names no maximum, so such a ring takes the other side's
    // count.
    let named_pages = |pages: &[(OpId, Option<u32>, u32)]| -> BTreeMap<OpId, Option<u32>> {
        let mut named: BTreeMap<OpId, Option<u32>> = BTreeMap::new();
        for (ring, page, moves) in pages {
            let entry = named.entry(*ring).or_insert(None);
            // An access whose page index is a loop value names no maximum, but
            // it does not erase the pages another access on the same side does
            // name: the block is the extent the side names, and the unknown
            // walk stays inside it. A side whose every access is unknown is
            // `None`, and then the other side's extent is the ring's.
            if let Some(p) = page {
                let known = entry.unwrap_or(0);
                *entry = Some(known.max(p.saturating_add(*moves)));
            }
        }
        named
    };
    let write_pages = named_pages(
        &write_info
            .iter()
            .map(|(at, (ring, _, page, _))| Ok((*ring, *page, access_pages(k, *at)?)))
            .collect::<Result<Vec<_>, ttStatus>>()?,
    );
    let read_pages = named_pages(
        &consumers
            .iter()
            .map(|consumer| Ok((consumer.ring, consumer.page, access_pages(k, consumer.at)?)))
            .collect::<Result<Vec<_>, ttStatus>>()?,
    );
    let mut block: BTreeMap<OpId, u32> = BTreeMap::new();
    {
        let rings: BTreeSet<OpId> = write_pages
            .keys()
            .chain(read_pages.keys())
            .copied()
            .collect();
        for ring in rings {
            let written = write_pages.get(&ring).copied().flatten();
            let read = read_pages.get(&ring).copied().flatten();
            let pages = match (written, read) {
                // Both sides name pages: the same block on both, or the two
                // pointers would advance by different amounts.
                (Some(a), Some(b)) if a != b => {
                    let _ = (ring, a, b);
                    return Err(ttStatus::TT_ERROR_CB_BLOCK);
                }
                (Some(a), Some(_)) => a,
                // One side's page index is a loop value: the other side's
                // count is the ring's.
                (Some(a), None) | (None, Some(a)) => a,
                // Neither side names a page: the hand-off count would be a
                // guess.
                (None, None) => {
                    let _ = ring;
                    return Err(ttStatus::TT_ERROR_CB_BLOCK);
                }
            };
            if pages > 1 {
                block.insert(ring, pages);
            }
        }
    }
    // The lowering places a loop's page index inside the block, so the derived
    // number goes back into the ring op.
    for (ring, pages) in &block {
        if let Op::Cb {
            dtype, page, len, ..
        } = *k.at(*ring)
        {
            k.ops[*ring].op = Op::Cb {
                dtype,
                page,
                len,
                block: Some(*pages),
            };
        }
    }
    // A block's hand-off lives in the scope that publishes the block, not in
    // the loops below it: the `expand` that fans the reservation out sits at
    // the block's own scope while the pages are filled by the walk under it.
    // A run is the writes that fill one block, ending at its publish (the
    // `expand`, or the write that reaches the block's last page); the peeled
    // prologue and epilogue are runs of their own because each publishes its
    // own block.
    struct Run {
        first: OpId,
        publish: OpId,
        /// The shallowest loop stack the run's writes sit in: the block's scope.
        level: Vec<OpId>,
        /// The stack of the run's first write, which is where its walk starts.
        first_stack: Vec<OpId>,
    }
    let mut runs: BTreeMap<OpId, Vec<Run>> = BTreeMap::new();
    {
        let mut by_ring: BTreeMap<OpId, Vec<OpId>> = BTreeMap::new();
        for (at, (ring, _, _, _)) in &write_info {
            by_ring.entry(*ring).or_default().push(*at);
        }
        for (ring, list) in by_ring {
            let Some(&pages) = block.get(&ring) else {
                continue;
            };
            let mut at = 0usize;
            while at < list.len() {
                let first = list[at];
                let first_stack = write_info[&first].1.clone();
                let mut level = first_stack.clone();
                let mut extent = 0u32;
                let (publish, next) = {
                    let mut next = at;
                    loop {
                        let id = list[next];
                        let (_, stack, page, _) = &write_info[&id];
                        if stack.len() < level.len() {
                            level = stack.clone();
                        }
                        let moved = access_pages(k, id)?;
                        if let Some(page) = page {
                            extent = extent.max(page.saturating_add(moved));
                        }
                        if matches!(*k.at(id), Op::Expand { .. }) {
                            break (id, next + 1);
                        }
                        // A copy that reaches the block's last page still
                        // leaves the publish to the `expand` that follows it.
                        let followed = list.get(next + 1).copied();
                        let followed_by_expand = followed.is_some_and(|f| {
                            matches!(*k.at(f), Op::Expand { src, .. } if src == ring)
                        });
                        if extent >= pages && !followed_by_expand {
                            break (id, next + 1);
                        }
                        next += 1;
                        if next >= list.len() {
                            if extent < pages {
                                // The program's writes do not add up to the
                                // block it states.
                                return Err(ttStatus::TT_ERROR_CB_BLOCK);
                            }
                            break (id, next);
                        }
                    }
                };
                runs.entry(ring).or_default().push(Run {
                    first,
                    publish,
                    level,
                    first_stack,
                });
                at = next;
            }
        }
    }
    let mut publish_paths: BTreeMap<OpId, Vec<Vec<Option<i32>>>> = BTreeMap::new();
    for (ring, list) in &runs {
        for run in list {
            let path = run
                .level
                .iter()
                .map(|loop_op| scope_key(k, *loop_op))
                .collect::<Result<Vec<Option<i32>>, ttStatus>>()?;
            publish_paths.entry(*ring).or_default().push(path);
        }
    }

    // A load whose page is still needed by a `Src` reader cannot have another
    // read of *the same page* in between: that read would take the page the
    // first one is still using. A read of a different page is the whole point
    // of a block -- tt-metal's compute waits for the block and walks its tiles
    // (`matmul_tiles(in0_cb, in1_cb, in0_index, in1_index, ...)`,
    // `bmm_large_block_zm.cpp`) -- so only a read that names the same page (or
    // names none) is refused.
    let page_of: BTreeMap<OpId, Option<u32>> = consumers
        .iter()
        .map(|consumer| (consumer.at, consumer.page))
        .collect();
    for (load, (reader, _)) in &last_src {
        let ring = consumer_ring(&consumers, *load)?;
        // A ring that states its block is walked by the program: the pages are
        // the indices the program names inside that block, and whether two of
        // them are the same page is the program's business, not something the
        // compiler can read off a loop value. tt-metal's compute does exactly
        // this -- `matmul_tiles(in0_index, in1_index, ...)` with the indices
        // from the subblock loops (`bmm_large_block_zm.cpp`) -- and the block
        // hand-off is one per block whatever the indices are.
        if block.contains_key(&ring) {
            continue;
        }
        let (Some(&from), Some(&to)) = (position.get(load), position.get(reader)) else {
            eprintln!("libskamandros: sync_cbs: a load or its `Src` reader has no position");
            return Err(ttStatus::TT_INTERNAL_BUG);
        };
        let load_page = page_of.get(load).copied().flatten();
        for consumer in &consumers {
            if consumer.ring != ring {
                continue;
            }
            let Some(&at) = position.get(&consumer.at) else {
                eprintln!("libskamandros: sync_cbs: a consumer has no position");
                return Err(ttStatus::TT_INTERNAL_BUG);
            };
            if at > from && at < to {
                let same_page = match (load_page, consumer.page) {
                    (Some(a), Some(b)) => a == b,
                    // A read that names no page takes the read pointer's, which
                    // is the page the first read still holds.
                    _ => true,
                };
                if same_page {
                    return Err(ttStatus::TT_TODO_RING_READ_BEFORE_RELEASE);
                }
            }
        }
    }

    // `(at, op, before)` insertions, applied after the scan so the ids stay
    // valid. A write is one page, so `n` is one throughout.
    let mut edits: Vec<(OpId, Op, bool)> = Vec::new();
    // Producer hand-offs. A block ring's hand-off is its runs': the reserve
    // sits at the block's level, before the first op that leads into the walk,
    // and the push at the publish. A ring without a stated block keeps one
    // reserve and push per write, and an `expand` that follows a copy publishes
    // the copy's page rather than a new one.
    for (ring, list) in &runs {
        let n = block[ring];
        for run in list {
            if run.first == run.publish && matches!(*k.at(run.publish), Op::Expand { .. }) {
                // A publish with nothing before it reserves nothing.
                continue;
            }
            // The walk's outermost loop when the pages are filled below the
            // block's scope; the first write itself when they are not.
            let anchor = match run.first_stack.get(run.level.len()) {
                Some(loop_op) => *loop_op,
                None => run.first,
            };
            edits.push((anchor, Op::Reserve { cb: *ring, n }, true));
            edits.push((run.publish, Op::Push { cb: *ring, n }, false));
        }
    }
    let writes_in_order: Vec<(OpId, Op)> = k.ops().map(|(at, op)| (at, op.clone())).collect();
    for (i, (at, op)) in writes_in_order.iter().enumerate() {
        let Some(ring) = write_ring(op, &rings) else {
            continue;
        };
        if block.contains_key(&ring) {
            continue;
        }
        let expand_publishes = matches!(op, Op::Expand { .. });
        // The expand that publishes this page is the next op to touch the ring;
        // the values it is computed from may sit in between, and they are not
        // ring traffic.
        let mut next_publishes_it = false;
        if !expand_publishes {
            for (_, next) in writes_in_order.iter().skip(i + 1) {
                match next {
                    Op::Expand { src, .. } if *src == ring => {
                        next_publishes_it = true;
                        break;
                    }
                    Op::EndReader | Op::EndCompute | Op::EndLoop => break,
                    other if other.parameters().any(|v| v == ring) => break,
                    _ => {}
                }
            }
        }
        let n = 1;
        if !expand_publishes {
            edits.push((*at, Op::Reserve { cb: ring, n }, true));
        }
        if !next_publishes_it {
            edits.push((*at, Op::Push { cb: ring, n }, false));
        }
    }
    // The read side waits and releases the same block the producer hands off
    // (`block`, derived above from both sides): every reader of the ring counts,
    // the writer's copies out included, because a block the packer published as
    // a whole is released as a whole.
    // The first read of each block and the op after which its pages are free:
    // the block's last `Src` reader (or the end of the loop it sits in, so a
    // reader that repeats is released once).
    let mut read_first: BTreeMap<(OpId, Vec<OpId>), OpId> = BTreeMap::new();
    let mut read_last: BTreeMap<(OpId, Vec<OpId>), OpId> = BTreeMap::new();
    let mut read_pop: BTreeMap<(OpId, Vec<OpId>), (usize, OpId)> = BTreeMap::new();
    for consumer in consumers.iter() {
        let Some(_) = block.get(&consumer.ring) else {
            continue;
        };
        let key = (consumer.ring, consumer.stack.clone());
        read_first.entry(key.clone()).or_insert(consumer.at);
        read_last.insert(key.clone(), consumer.at);
        let pop_at = match last_src.get(&consumer.at) {
            Some((reader, stack)) => match stack.get(consumer.stack.len()) {
                Some(outer) => {
                    let Some(&end) = end_of.get(outer) else {
                        eprintln!("libskamandros: sync_cbs: loop {outer} has no end");
                        return Err(ttStatus::TT_INTERNAL_BUG);
                    };
                    end
                }
                None => *reader,
            },
            None => consumer.at,
        };
        let at = position
            .get(&pop_at)
            .copied()
            .ok_or(ttStatus::TT_INTERNAL_BUG)?;
        let entry = read_pop.entry(key).or_insert((at, pop_at));
        if at > entry.0 {
            *entry = (at, pop_at);
        }
    }
    // Consumer hand-offs. A page produced at the same depth is waited for and
    // released per read; a page produced *outside* the reading loops is hoisted
    // to the producer's scope and shared by the whole loop.
    // A ring whose block is read across a nest of loops below the loop that
    // produces it: where its one wait/pop pair sits, and how many pages it
    // hands off.
    // One hand-off pair per (ring, scope): tt-metal's kernel reads one operand
    // ring in a peeled prologue, in the k loop and in a peeled epilogue, and
    // each of those is its own wait/release with the same block.
    let mut hoisted: BTreeMap<(OpId, OpId), u32> = BTreeMap::new();
    for consumer in &consumers {
        let Some(_depths) = producer_depths.get(&consumer.ring) else {
            eprintln!(
                "libskamandros: sync_cbs: ring {} is read at {} but never written",
                consumer.ring, consumer.at
            );
            return Err(ttStatus::TT_INTERNAL_BUG);
        };
        let depth = consumer.stack.len();
        // The hand-off's level is the scope that publishes the block: the
        // `expand` sits at the block's scope while the pages are filled in the
        // loops below it, so the wait and pop belong to that scope whether or
        // not the read's nest happens to have the writer's depth. A read that
        // sits below the publishing scope is hoisted to it; a read directly in
        // it hands off per read.
        let read_keys = consumer
            .stack
            .iter()
            .map(|loop_op| scope_key(k, *loop_op))
            .collect::<Result<Vec<Option<i32>>, ttStatus>>()?;
        let paths = if block.contains_key(&consumer.ring) {
            publish_paths.get(&consumer.ring)
        } else {
            producer_paths.get(&consumer.ring)
        };
        let Some(paths) = paths else {
            eprintln!(
                "libskamandros: sync_cbs: ring {} is read at {} but never written",
                consumer.ring, consumer.at
            );
            return Err(ttStatus::TT_INTERNAL_BUG);
        };
        let producer_depth = (0..=read_keys.len()).rev().find(|depth| {
            paths
                .iter()
                .any(|path| path.len() == *depth && path[..] == read_keys[..*depth])
        });
        let Some(producer_depth) = producer_depth else {
            eprintln!(
                "libskamandros: sync_cbs: ring {} is read before it is written",
                consumer.ring
            );
            return Err(ttStatus::TT_INTERNAL_BUG);
        };
        if producer_depth < read_keys.len() {
            // The read is below the scope that publishes the block: its
            // hand-off is hoisted to that scope and shared by the whole loop.
            let pages = block.get(&consumer.ring).copied();
            if pages.is_none() && writes.get(&consumer.ring) != Some(&1) {
                return Err(ttStatus::TT_TODO_RING_MULTI_WRITE_OUTSIDE_LOOP);
            }
            let n = pages.unwrap_or(1);
            match hoisted.insert((consumer.ring, consumer.stack[producer_depth]), n) {
                Some(previous) if previous != n => {
                    return Err(ttStatus::TT_ERROR_CB_BLOCK);
                }
                _ => {}
            }
            continue;
        }
        {
            // A block is waited for at its first read and released after its
            // last `Src` reader; a ring without one keeps a hand-off per read.
            let key = (consumer.ring, consumer.stack.clone());
            let pages = block.get(&consumer.ring).copied();
            let n = pages.unwrap_or(1);
            let first = pages.map_or(true, |_| read_first.get(&key) == Some(&consumer.at));
            let last = pages.map_or(true, |_| read_last.get(&key) == Some(&consumer.at));
            if first {
                edits.push((
                    consumer.at,
                    Op::Wait {
                        cb: consumer.ring,
                        n,
                    },
                    true,
                ));
            }
            // A page produced at the same depth is released after its last
            // reader. When that reader sits inside a loop the load is outside
            // of, the read repeats, so the release belongs after the loop.
            let pop_at = match pages {
                Some(_) => read_pop[&key].1,
                None => {
                    if consumer.writer || !last_src.contains_key(&consumer.at) {
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
                    }
                }
            };
            if last {
                edits.push((
                    pop_at,
                    Op::Pop {
                        cb: consumer.ring,
                        n,
                    },
                    false,
                ));
            }
        }
    }
    for ((ring, outer), n) in hoisted {
        let Some(&end) = end_of.get(&outer) else {
            eprintln!("libskamandros: sync_cbs: loop {outer} has no end");
            return Err(ttStatus::TT_INTERNAL_BUG);
        };
        edits.push((outer, Op::Wait { cb: ring, n }, true));
        edits.push((end, Op::Pop { cb: ring, n }, false));
    }

    for (at, op, before) in edits {
        if before {
            k.insert_before(at, op);
        } else {
            k.insert_after(at, op);
        }
    }
    dedup_pushes(k)?;
    check_rings_balanced(k)?;
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

/// Prove that no ring can deadlock.
///
/// A producer blocks in `reserve` while its ring is full; a consumer blocks in
/// `wait` while the ring is empty. Neither can block forever unless one side
/// finishes while the other still needs a page, so what has to hold, per ring,
/// is that the pages its producers push and its consumers pop are the same
/// number as a function of their enclosing trip counts. Then at every point the
/// pages in flight stay inside the ring, the side that runs out of work is the
/// side the other has already released, and a blocked role always has a
/// counterpart that can still move. The allowance at the end is the ring's own
/// depth — a producer one page ahead of its consumer is a pipeline, not a leak
/// — which is the same allowance `verify`'s FIFO makes.
///
/// A page count is a sum of monomials: every hand-off contributes its pages
/// times the product of its enclosing trip counts, and a product is a multiset
/// of bound values. Bounds are compared structurally ([`Bounds`]) because the
/// movers rebuild a bound for every segment they copy a loop into, so the same
/// loop's bound has a different `OpId` on each core.
///
/// This is the guarantee, not a diagnostic. It is a compile-time refusal: a
/// program that gets past it cannot have a role waiting for a page that is
/// never published.
fn check_rings_balanced(k: &Kernel) -> Result<(), ttStatus> {
    let mut bounds = Bounds::default();
    let mut labels: Vec<u32> = Vec::new();
    let mut traffic: BTreeMap<OpId, BTreeMap<Vec<u32>, i64>> = BTreeMap::new();
    let mut depth: BTreeMap<OpId, i64> = BTreeMap::new();
    for (id, op) in k.ops() {
        match *op {
            Op::Cb { len, .. } => {
                depth.insert(id, i64::from(len));
            }
            Op::Loop { bound, .. } => labels.push(bounds.key(k, bound)?),
            Op::EndLoop => {
                if labels.pop().is_none() {
                    return Err(crate::status::internal_bug(
                        "sync_cbs: an end-of-loop has no open loop",
                    ));
                }
            }
            Op::Push { cb, n } => {
                let monomial = monomial(&labels);
                *traffic.entry(cb).or_default().entry(monomial).or_default() += i64::from(n);
            }
            Op::Pop { cb, n } => {
                let monomial = monomial(&labels);
                *traffic.entry(cb).or_default().entry(monomial).or_default() -= i64::from(n);
            }
            _ => {}
        }
    }
    for (cb, monomials) in traffic {
        let depth = depth.get(&cb).copied().unwrap_or(0);
        for (monomial, left) in monomials {
            let holds = if monomial.is_empty() {
                // Pages no loop multiplies can only be a constant, and only a
                // pipeline's leftover is allowed: never more than the ring
                // holds, and never a consumer ahead of its producer.
                left >= 0 && left <= depth
            } else {
                // A term in any trip count means the imbalance grows without
                // bound, so some launch trips it.
                left == 0
            };
            if !holds {
                let _ = (cb, left);
                return Err(ttStatus::TT_ERROR_RING_UNBALANCED);
            }
        }
    }
    Ok(())
}

/// A product of trip counts as the multiset of the bounds it multiplies: the
/// product is commutative, so the order the loops are nested in does not
/// matter, only which bounds and how many times each appears.
fn monomial(labels: &[u32]) -> Vec<u32> {
    let mut monomial = labels.to_vec();
    monomial.sort_unstable();
    monomial
}

/// The scenario a bound is, independent of the `OpId` a segment gives it.
#[derive(Default)]
struct Bounds {
    /// Each value's key, once it has one.
    of: BTreeMap<OpId, u32>,
    /// The keys themselves: a value's op and its operands' keys.
    sigs: BTreeMap<Sig, u32>,
    next: u32,
}

/// What a value's key is computed from.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Sig {
    /// A folded `i32`, so two spellings of one constant agree.
    Int(i32),
    /// A leaf the movers share across the segments they build, so its own id
    /// is its identity.
    Leaf(OpId),
    Coord(u8),
    Unary(u8, u32),
    Binary(u8, u32, u32),
    Cast(u8, bool, u32),
    Select(u32, u32, u32),
}

impl Bounds {
    /// The key of `value`, interning it on first sight.
    fn key(&mut self, k: &Kernel, value: OpId) -> Result<u32, ttStatus> {
        if let Some(key) = self.of.get(&value) {
            return Ok(*key);
        }
        let sig = if let Some(n) = k.const_i32(value) {
            Sig::Int(n)
        } else {
            match *k.at(value) {
                Op::GridCoord { axis } => Sig::Coord(axis),
                Op::Unary { op, a } => Sig::Unary(op as u8, self.key(k, a)?),
                Op::Binary { op, a, b } => Sig::Binary(op as u8, self.key(k, a)?, self.key(k, b)?),
                Op::Cast { dtype, src } => Sig::Cast(dtype as u8, false, self.key(k, src)?),
                Op::Bitcast { dtype, src } => Sig::Cast(dtype as u8, true, self.key(k, src)?),
                Op::Select {
                    cond,
                    if_true,
                    if_false,
                } => Sig::Select(
                    self.key(k, cond)?,
                    self.key(k, if_true)?,
                    self.key(k, if_false)?,
                ),
                // A parameter or a variable is the prologue's: every segment
                // names the same op.
                _ => Sig::Leaf(value),
            }
        };
        let key = match self.sigs.get(&sig) {
            Some(key) => *key,
            None => {
                let key = self.next;
                self.next += 1;
                self.sigs.insert(sig, key);
                key
            }
        };
        self.of.insert(value, key);
        Ok(key)
    }
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

/// A loop's scope key: its bound's folded value, or `None` when the bound is
/// run-time. Two spellings that fold alike are the same scope; the clones the
/// mover makes of a run-time bound are the same `None`.
fn scope_key(k: &Kernel, loop_op: OpId) -> Result<Option<i32>, ttStatus> {
    let Op::Loop { bound, .. } = *k.at(loop_op) else {
        eprintln!("libskamandros: sync_cbs: {loop_op} is not a loop");
        return Err(ttStatus::TT_INTERNAL_BUG);
    };
    Ok(k.const_i32(bound))
}

/// The pages one access moves: a `copy` moves the page it names, a block
/// `load`/`store` moves its `m * n` tile pages, and an `expand` publishes the
/// pages it names (its page index is already the last of them).
fn access_pages(k: &Kernel, at: OpId) -> Result<u32, ttStatus> {
    Ok(match *k.at(at) {
        Op::Copy { shape, .. } | Op::Load { shape, .. } | Op::Store { shape, .. } => {
            let (m, n) = crate::ir::access_shape(shape)?.tiles();
            m * n
        }
        _ => 1,
    })
}

/// The ring an op writes, if it writes one.
fn write_ring(op: &Op, rings: &BTreeSet<OpId>) -> Option<OpId> {
    match *op {
        Op::Store { buffer, .. } if rings.contains(&buffer) => Some(buffer),
        Op::Copy { dst, .. } if rings.contains(&dst) => Some(dst),
        Op::Expand { src, .. } if rings.contains(&src) => Some(src),
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
        Op::Transpose { src } => src == value,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a program, give it its movers and place its hand-offs. The check
    /// under test is the last step of [`sync_cbs`], so a refusal here is the
    /// balance, not the placement.
    fn balance(text: &str) -> Result<(), ttStatus> {
        let k = Kernel::parse(text).expect("the program parses");
        let mut k = crate::mover::add_movers(&k).expect("the movers are added");
        sync_cbs(&mut k)
    }

    /// One loop, both rings written and read inside it: the shape every
    /// pipeline has, and the one the whole suite is built on.
    const BALANCED: &str = "\
grid 1,1
%n = param i32 0
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,64
%bo = buffer f32 32,64
%step = const i32 1
%i = loop %n
%o = mul %i %step
copy 32,32 %bi [0] = %in [%o]
load 32,32 %x = %bi [0]
%y = neg %x
store 32,32 %bo [0] = %y
copy 32,32 %out [%o] = %bo [0]
endloop
";

    #[test]
    fn a_balanced_pipeline_is_accepted() {
        assert_eq!(balance(BALANCED), Ok(()));
    }

    /// The producer's loop and the consumer's are different loops with
    /// different launch arguments. Nothing proves they trip the same number of
    /// times, so for some arguments one role waits for a page that never comes.
    #[test]
    fn a_ring_moved_by_two_different_bounds_is_refused() {
        let text = "\
grid 1,1
%n = param i32 0
%m = param i32 0
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,64
%bo = buffer f32 32,64
%i = loop %n
copy 32,32 %bi [0] = %in [0]
endloop
%j = loop %m
load 32,32 %x = %bi [0]
%y = neg %x
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
endloop
";
        assert_eq!(
            balance(text),
            Err(ttStatus::TT_ERROR_RING_UNBALANCED),
            "a producer loop and a consumer loop with different bounds"
        );
    }

    /// The bound is an expression, so the movers rebuild it once per segment:
    /// the two cores hold two distinct `OpId`s for the same trip count. The
    /// bound's identity is structural, or this legal pipeline is refused.
    #[test]
    fn a_bound_rebuilt_per_segment_is_still_one_bound() {
        let text = "\
grid 1,1
%a = param i32 0
%b = param i32 0
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,64
%bo = buffer f32 32,64
%per = mul %a %b
%step = const i32 1
%i = loop %per
%o = mul %i %step
copy 32,32 %bi [0] = %in [%o]
load 32,32 %x = %bi [0]
%y = neg %x
store 32,32 %bo [0] = %y
copy 32,32 %out [%o] = %bo [0]
endloop
";
        let k = Kernel::parse(text).expect("the program parses");
        let mut k = crate::mover::add_movers(&k).expect("the movers are added");
        let bounds: Vec<OpId> = k
            .ops()
            .filter_map(|(_, op)| match *op {
                Op::Loop { bound, .. } => Some(bound),
                _ => None,
            })
            .collect();
        assert!(
            bounds.len() >= 2,
            "the loop is copied into more than one segment"
        );
        assert_ne!(
            bounds[0], bounds[1],
            "the movers rebuilt the bound, so identity would refuse it"
        );
        assert_eq!(sync_cbs(&mut k), Ok(()));
    }

    /// The same product of trip counts, with the loops nested in the opposite
    /// order: a product is commutative, and only the multiset of bounds counts.
    #[test]
    fn nested_bounds_match_in_either_order() {
        let text = "\
grid 1,1
%n = param i32 0
%m = param i32 0
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,64
%bo = buffer f32 32,64
%i = loop %n
%j = loop %m
copy 32,32 %bi [0] = %in [0]
endloop
endloop
%k = loop %m
%l = loop %n
load 32,32 %x = %bi [0]
%y = neg %x
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
endloop
endloop
";
        assert_eq!(balance(text), Ok(()));
    }

    /// A ring may end holding pages — a producer one page ahead of its consumer
    /// is a pipeline — as long as they fit in the ring.
    #[test]
    fn a_leftover_that_fits_the_ring_is_accepted() {
        let text = "\
grid 1,1
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,64
%bo = buffer f32 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = neg %x
copy 32,32 %bi [0] = %in [0]
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";
        assert_eq!(balance(text), Ok(()));
    }

    /// One page more than the ring holds is a producer that blocks in
    /// `reserve` with no consumer left to free a page.
    #[test]
    fn a_leftover_deeper_than_the_ring_is_refused() {
        let text = "\
grid 1,1
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = neg %x
copy 32,32 %bi [0] = %in [0]
copy 32,32 %bi [0] = %in [0]
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";
        assert_eq!(
            balance(text),
            Err(ttStatus::TT_ERROR_RING_UNBALANCED),
            "two pages left in a one-page ring"
        );
    }
}
