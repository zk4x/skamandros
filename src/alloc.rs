// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The machine-side allocation: what `lower` minted without a limit, brought
//! inside the limits the hardware has.
//!
//! `lower` names one `Dst` per value and one lane register per temporary, with
//! no reuse at all: correctness first, and the ids are only names. This module
//! is the pass that keeps a whole chain of row bodies in lane registers
//! ([`fuse_bodies`]), decides which of the remaining names may share a physical
//! `Dst` or lane register, by liveness over the one machine order, and places
//! the circular buffers in L1.
//!
//! Nothing here runs in the virtual IR: the SSA program has no allocation at
//! all.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::DType;
use crate::mir::{Cb, CbInfo, Core, Dst, DstInfo, LReg, MachineKernel, MirOp};
use crate::status::ttStatus;

/// Writable lane registers (`SFPLOAD`/`SFPSTORE` address `L0..L7`).
pub const LREGS: u8 = 8;
/// Alignment every ring starts at: the NoC requires the two addresses of a
/// request to be congruent, and every copy's L1 side is a page start.
const ALIGN: u32 = 64;

/// Bring the machine kernel's `Dst` registers and lane registers inside the
/// hardware's limits, in place.
pub fn alloc(m: &mut MachineKernel) -> Result<(), ttStatus> {
    alloc_dsts(m)?;
    alloc_lregs(m)?;
    Ok(())
}

/// Lane registers `8..16` are the hardware's read-only constants: a body reads
/// them (`LREG_ZERO` is 9, `LREG_ONE` 10) and they never take part in the
/// allocation. Virtual lane names start above them, in a space wider than the
/// eight work registers, so a body can be rewritten without collisions and the
/// scan below turns every virtual into `0..7`.
const VIRTUAL: u16 = 16;

/// The one optimizing pass over the machine IR: a whole chain of row bodies
/// becomes a single row body, every interior value stays in a lane register,
/// and the body's virtuals take the eight physical lanes by linear scan.
///
/// `lower` emits one sweep per IR op, each sweep a `Sfpload`, the lane sequence
/// and an `Sfpstore` repeated for all 32 `Dst` rows. A chain therefore writes a
/// tile and immediately reads it back, one `Dst` slot per intermediate. This
/// pass turns a run of sweeps into one body: the interior stores and loads
/// disappear, their values cross in a lane register through an `SFPMOV`, and
/// the merged body is run 32 times by a [`MirOp::Repeat`]. One load and one
/// store per tile remain — the traffic of a hand-written activation.
///
/// It is exact: a value is interior only when exactly one sweep writes it, no
/// other op names its `Dst`, and every read is a later sweep of the same run.
/// When the merged body needs more than eight live lanes the scan spills values
/// around the over-budget windows (one store, one reload each) as many times as
/// it takes, as zyx's `tt_lreg_alloc` does. Spilling to L1 when a body still
/// cannot be placed is not built yet; that is `TT_ERROR_LANE_REGISTER`, never a
/// silently unfused cone.
///
/// It runs before [`crate::optimize::reg_sync`]: removing an interior `Dst`
/// removes the hand-off it would have needed, and the seams are derived from
/// the fused stream.
pub fn fuse_bodies(m: &mut MachineKernel) -> Result<(), ttStatus> {
    let entries = m.ops.clone();
    // Who writes, who reads and who merely names each `Dst`, over the whole
    // program. A value whose `Dst` any other op names — a pack, a `MOVD2B`, the
    // Matrix Unit — is never interior, whatever the run says.
    let mut stores: BTreeMap<Dst, usize> = BTreeMap::new();
    let mut loads: BTreeMap<Dst, usize> = BTreeMap::new();
    let mut other: BTreeSet<Dst> = BTreeSet::new();
    for (_, op) in &entries {
        match *op {
            MirOp::Sfpstore { reg, .. } => *stores.entry(reg).or_default() += 1,
            MirOp::Sfpload { reg, .. } => *loads.entry(reg).or_default() += 1,
            _ => {
                if let Some(reg) = named_dst(op) {
                    other.insert(reg);
                }
            }
        }
    }

    let mut out: Vec<(Core, MirOp)> = Vec::with_capacity(entries.len());
    let mut at = 0usize;
    while at < entries.len() {
        // A section is a maximal run of row-body ops on the math core; every
        // hand-off, `Loop`, FPU op and mover op ends it.
        let start = at;
        while at < entries.len() && entries[at].0 == Core::Math && is_row_op(&entries[at].1) {
            at += 1;
        }
        if at == start {
            out.push(entries[at]);
            at += 1;
            continue;
        }
        let section: Vec<MirOp> = entries[start..at].iter().map(|(_, op)| *op).collect();
        match parse_sweeps(&section) {
            Some(sweeps) => {
                for op in fuse_section(m, &sweeps, &stores, &loads, &other)? {
                    out.push((Core::Math, op));
                }
            }
            // A shape the pass cannot read (a partial sweep, a body it did not
            // mint): leave it exactly as it is.
            None => out.extend(entries[start..at].iter().copied()),
        }
    }
    m.ops = out;
    Ok(())
}

/// Every op that is part of a row body: the SFPU sequence, its load and store,
/// and the lane constant a body carries.
fn is_row_op(op: &MirOp) -> bool {
    matches!(
        op,
        MirOp::Sfpload { .. }
            | MirOp::Sfpstore { .. }
            | MirOp::LaneConst { .. }
            | MirOp::UnarySfp { .. }
            | MirOp::BinarySfp { .. }
            | MirOp::SfpMov { .. }
            | MirOp::SfpAbs { .. }
            | MirOp::SfpCast { .. }
            | MirOp::SfpExexp { .. }
            | MirOp::SfpExman { .. }
            | MirOp::SfpIadd { .. }
            | MirOp::SfpShft { .. }
            | MirOp::SfpSetcc { .. }
            | MirOp::SfpEncc { .. }
            | MirOp::SfpArecip { .. }
            | MirOp::SfpMul24 { .. }
            | MirOp::SfpMad { .. }
            | MirOp::SfpStochRnd { .. }
            | MirOp::SfpGt { .. }
            | MirOp::SfpLe { .. }
    )
}

/// The `Dst` an op other than a lane load or store names, if it names one.
fn named_dst(op: &MirOp) -> Option<Dst> {
    match *op {
        MirOp::Setdst { reg }
        | MirOp::Mvmul { reg, .. }
        | MirOp::Elw { reg, .. }
        | MirOp::Gapool { reg, .. }
        | MirOp::Gmpool { reg, .. }
        | MirOp::Mova2d { reg, .. }
        | MirOp::Movb2d { reg, .. }
        | MirOp::Movd2b { reg, .. }
        | MirOp::SetPackOffset { reg, .. } => Some(reg),
        _ => None,
    }
}

/// One op sweep: the lane constants hoisted before it and its 32 row bodies.
struct Sweep {
    gap: Vec<MirOp>,
    rows: Vec<Vec<MirOp>>,
}

/// Read a section as a sequence of 32-row sweeps. Anything that does not line
/// up exactly is `None`, and the caller leaves the section alone.
fn parse_sweeps(section: &[MirOp]) -> Option<Vec<Sweep>> {
    let mut sweeps: Vec<Sweep> = Vec::new();
    let mut at = 0usize;
    while at < section.len() {
        let mut gap: Vec<MirOp> = Vec::new();
        while at < section.len() && matches!(section[at], MirOp::LaneConst { .. }) {
            gap.push(section[at]);
            at += 1;
        }
        let mut rows: Vec<Vec<MirOp>> = Vec::with_capacity(32);
        for _ in 0..32 {
            if !matches!(section.get(at), Some(MirOp::Sfpload { .. })) {
                return None;
            }
            let start = at;
            while at < section.len() && !matches!(section[at], MirOp::Sfpstore { .. }) {
                at += 1;
            }
            if at >= section.len() {
                return None;
            }
            at += 1;
            rows.push(section[start..at].to_vec());
        }
        let base = row_at(&rows[0])?;
        if base != 0 {
            return None;
        }
        for (k, row) in rows.iter().enumerate() {
            if row_at(row)? != 2 * k as u32 || !rows_match(&rows[0], row, 2 * k as u32) {
                return None;
            }
        }
        sweeps.push(Sweep { gap, rows });
    }
    if sweeps.is_empty() {
        None
    } else {
        Some(sweeps)
    }
}

/// The `Dst` row a row body starts at.
fn row_at(row: &[MirOp]) -> Option<u32> {
    match row.first()? {
        MirOp::Sfpload { at, .. } => Some(*at),
        _ => None,
    }
}

/// Is `b` the row body `a` with its `Dst` rows `delta` further on?
fn rows_match(a: &[MirOp], b: &[MirOp], delta: u32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).all(|(a, b)| match (a, b) {
        (
            MirOp::Sfpload {
                lreg: la,
                reg: ra,
                at: aa,
                dtype: da,
            },
            MirOp::Sfpload {
                lreg: lb,
                reg: rb,
                at: ab,
                dtype: db,
            },
        )
        | (
            MirOp::Sfpstore {
                lreg: la,
                reg: ra,
                at: aa,
                dtype: da,
            },
            MirOp::Sfpstore {
                lreg: lb,
                reg: rb,
                at: ab,
                dtype: db,
            },
        ) => la == lb && ra == rb && da == db && *ab == *aa + delta,
        _ => a == b,
    })
}

/// Fuse the sweeps of one section, in order.
fn fuse_section(
    m: &mut MachineKernel,
    sweeps: &[Sweep],
    stores: &BTreeMap<Dst, usize>,
    loads: &BTreeMap<Dst, usize>,
    other: &BTreeSet<Dst>,
) -> Result<Vec<MirOp>, ttStatus> {
    let mut out: Vec<MirOp> = Vec::new();
    // Who reads each value, over the whole section: the cone is the
    // reader-closure of its first sweep, so a value defined early and used at
    // the end pulls the sweeps in between with it.
    let mut readers: BTreeMap<Dst, Vec<usize>> = BTreeMap::new();
    for (u, sweep) in sweeps.iter().enumerate() {
        for op in &sweep.rows[0] {
            if let MirOp::Sfpload { reg, .. } = *op {
                readers.entry(reg).or_default().push(u);
            }
        }
    }
    let mut at = 0usize;
    while at < sweeps.len() {
        let mut end = at + 1;
        loop {
            let mut next = end;
            for sweep in &sweeps[at..end] {
                for op in &sweep.rows[0] {
                    if let MirOp::Sfpstore { reg, .. } = *op {
                        if let Some(uses) = readers.get(&reg) {
                            for &u in uses {
                                next = next.max(u + 1);
                            }
                        }
                    }
                }
            }
            if next <= end {
                break;
            }
            end = next.min(sweeps.len());
        }
        let body = build_group(m, &sweeps[at..end], stores, loads, other)?;
        out.push(MirOp::Repeat {
            len: body.len() as u16,
            count: 32,
        });
        out.extend(body);
        at = end;
    }
    Ok(out)
}

/// Merge a group of sweeps into one row body in virtual lane space, then
/// allocate it.
fn build_group(
    m: &mut MachineKernel,
    group: &[Sweep],
    global_stores: &BTreeMap<Dst, usize>,
    global_loads: &BTreeMap<Dst, usize>,
    other: &BTreeSet<Dst>,
) -> Result<Vec<MirOp>, ttStatus> {
    // Which sweep writes and reads each value.
    let mut producers: BTreeMap<Dst, Vec<usize>> = BTreeMap::new();
    let mut readers: BTreeMap<Dst, Vec<usize>> = BTreeMap::new();
    for (u, sweep) in group.iter().enumerate() {
        for op in &sweep.rows[0] {
            match *op {
                MirOp::Sfpstore { reg, .. } => producers.entry(reg).or_default().push(u),
                MirOp::Sfpload { reg, .. } => readers.entry(reg).or_default().push(u),
                _ => {}
            }
        }
    }

    // A value that crosses from one sweep to another gets one lane name. A
    // value read once and written outside the group keeps its plain load.
    let mut cross: BTreeMap<Dst, LReg> = BTreeMap::new();
    let mut next = VIRTUAL;
    for (&reg, uses) in &readers {
        let produced_here = producers.contains_key(&reg);
        if produced_here || uses.len() > 1 {
            cross.insert(reg, LReg(next));
            next += 1;
        }
    }
    // Interior: one producer, every read later in this group, and no other op
    // anywhere names the `Dst`. Its store disappears. The global counts are per
    // row (lowering writes and reads every one of the 32), the group's are per
    // sweep.
    const ROWS: usize = 32;
    let interior = |reg: Dst| -> bool {
        let Some(ps) = producers.get(&reg) else {
            return false;
        };
        let Some(rs) = readers.get(&reg) else {
            return false;
        };
        ps.len() == 1
            && !other.contains(&reg)
            && global_stores.get(&reg).copied().unwrap_or(0) == ROWS
            && global_loads.get(&reg).copied().unwrap_or(0) == rs.len() * ROWS
            && rs.iter().all(|&r| r > ps[0])
    };

    let dtype = group
        .iter()
        .flat_map(|sweep| sweep.rows[0].iter())
        .find_map(|op| match *op {
            MirOp::Sfpload { dtype, .. } | MirOp::Sfpstore { dtype, .. } => Some(dtype),
            _ => None,
        })
        .unwrap_or(DType::F32);

    let mut body: Vec<MirOp> = Vec::new();
    // Which cross values have had their defining load or store yet, in body
    // order. A value read twice in one sweep is loaded once and moved the
    // second time; the sweep index alone cannot tell the two apart.
    let mut defined: BTreeSet<Dst> = BTreeSet::new();
    // One virtual-name counter for the whole merged body: two sweeps must never
    // receive the same name, or the allocator sees two values as one.
    let mut fresh = next;
    for sweep in group {
        // Every lane this sweep touches becomes a name of its own, so two
        // sweeps cannot collide no matter what physical lanes they used.
        let mut map: BTreeMap<LReg, LReg> = BTreeMap::new();
        for op in sweep.gap.iter().chain(sweep.rows[0].iter()) {
            for (lreg, _) in op.lreg_params() {
                if (8..16).contains(&lreg.0) {
                    continue;
                }
                map.entry(lreg).or_insert_with(|| {
                    let name = LReg(fresh);
                    fresh += 1;
                    name
                });
            }
        }
        for op in &sweep.gap {
            let mut op = *op;
            rewrite_lregs(&mut op, &map);
            body.push(op);
        }
        for op in &sweep.rows[0] {
            match *op {
                MirOp::Sfpload {
                    lreg, reg, dtype, ..
                } => {
                    let lane = map[&lreg];
                    match cross.get(&reg) {
                        Some(&value) if !defined.contains(&reg) => {
                            defined.insert(reg);
                            body.push(MirOp::Sfpload {
                                lreg: lane,
                                reg,
                                at: 0,
                                dtype,
                            });
                            body.push(MirOp::SfpMov { a: lane, d: value });
                        }
                        Some(&value) => {
                            body.push(MirOp::SfpMov { a: value, d: lane });
                        }
                        None => body.push(MirOp::Sfpload {
                            lreg: lane,
                            reg,
                            at: 0,
                            dtype,
                        }),
                    }
                }
                MirOp::Sfpstore {
                    lreg, reg, dtype, ..
                } => {
                    let lane = map[&lreg];
                    match cross.get(&reg) {
                        Some(&value) => {
                            defined.insert(reg);
                            if interior(reg) {
                                body.push(MirOp::SfpMov { a: lane, d: value });
                            } else {
                                body.push(MirOp::Sfpstore {
                                    lreg: lane,
                                    reg,
                                    at: 0,
                                    dtype,
                                });
                                body.push(MirOp::SfpMov { a: lane, d: value });
                            }
                        }
                        None => body.push(MirOp::Sfpstore {
                            lreg: lane,
                            reg,
                            at: 0,
                            dtype,
                        }),
                    }
                }
                mut op => {
                    rewrite_lregs(&mut op, &map);
                    body.push(op);
                }
            }
        }
    }
    allocate_body(m, &mut body, dtype)?;
    Ok(body)
}

/// Assign the eight physical lanes to a body's virtual names. The scan gives
/// every value as much reuse as liveness allows; when a body still needs more
/// than eight at once, the longest-lived value that is not touched in the
/// over-budget window goes to a `Dst` (one store, one reload) and the scan runs
/// again, as many times as it takes.
fn allocate_body(
    m: &mut MachineKernel,
    body: &mut Vec<MirOp>,
    dtype: DType,
) -> Result<(), ttStatus> {
    loop {
        let regs: Vec<Vec<(u16, bool, bool)>> = body
            .iter()
            .map(|op| {
                op.lreg_params()
                    .into_iter()
                    .map(|(lreg, use_)| (lreg.0, use_.reads(), use_.writes()))
                    .collect()
            })
            .collect();
        if let Some(assign) = linear_scan(&regs, usize::from(LREGS)) {
            let mut rename: BTreeMap<LReg, LReg> = BTreeMap::new();
            for (name, slot) in assign {
                rename.insert(LReg(name), LReg(u16::from(slot)));
            }
            for op in body.iter_mut() {
                rewrite_lregs(op, &rename);
            }
            body.retain(|op| !matches!(op, MirOp::SfpMov { a, d } if a == d));
            return Ok(());
        }
        let (before, after) = mask_states(body);
        let Some((name, a, b)) = split_at_pressure(&regs, &before, &after, usize::from(LREGS))
        else {
            return Err(ttStatus::TT_ERROR_LANE_REGISTER);
        };
        let fresh = regs
            .iter()
            .flatten()
            .map(|(id, _, _)| *id)
            .filter(|id| *id >= VIRTUAL)
            .max()
            .map_or(VIRTUAL, |id| id + 1);
        let tile = m.dsts.push(DstInfo {
            dtype,
            shape: [32, 32],
        });
        // One store before the over-budget run, one reload after it, and every
        // later use reads the reload. The value inside the window keeps its
        // register; that is what makes it a window and not a reload per use.
        // The lane moves are raw 32-bit words (`INT32`), which is what makes
        // them bit-exact for both `f32` and `i32` bodies.
        body.insert(
            a,
            MirOp::Sfpstore {
                lreg: LReg(name),
                reg: tile,
                at: 0,
                dtype: DType::I32,
            },
        );
        let after = b + 1;
        body.insert(
            after + 1,
            MirOp::Sfpload {
                lreg: LReg(fresh),
                reg: tile,
                at: 0,
                dtype: DType::I32,
            },
        );
        let mut rename: BTreeMap<LReg, LReg> = BTreeMap::new();
        rename.insert(LReg(name), LReg(fresh));
        for op in body.iter_mut().skip(after + 2) {
            rewrite_lregs(op, &rename);
        }
    }
}

/// Whether every lane is enabled before and after each instruction: the only
/// moments a spill's store and reload are bit-exact, because a lane-masked
/// `SFPSTORE` copies the enabled lanes only. `SFPENCC` resets the mask, a
/// condition set narrows it; every other instruction leaves it alone.
fn mask_states(body: &[MirOp]) -> (Vec<bool>, Vec<bool>) {
    let mut full = true;
    let mut before = Vec::with_capacity(body.len());
    let mut after = Vec::with_capacity(body.len());
    for op in body {
        before.push(full);
        match *op {
            MirOp::SfpEncc { .. } => full = true,
            MirOp::SfpSetcc { .. } => full = false,
            _ => {}
        }
        after.push(full);
    }
    (before, after)
}

/// Liveness over one straight-line body: a name is live from its definition
/// (or the body's start, when a read comes first) to its last touch. The scan
/// is exact because a row body is one block.
fn linear_scan(body: &[Vec<(u16, bool, bool)>], budget: usize) -> Option<BTreeMap<u16, u8>> {
    let (def, last) = live_ranges(body);
    let mut order: Vec<(u16, i64, i64)> = last
        .iter()
        .filter(|(name, _)| **name >= VIRTUAL)
        .map(|(&name, &end)| (name, def[&name], end))
        .collect();
    order.sort_by_key(|&(name, start, _)| (start, name));
    let mut assign: BTreeMap<u16, u8> = BTreeMap::new();
    let mut active: Vec<(u16, u8, i64)> = Vec::new();
    for (name, start, end) in order {
        active.retain(|&(_, _, live_to)| live_to >= start);
        if active.len() >= budget {
            return None;
        }
        let used = active.iter().fold(0u16, |m, &(_, slot, _)| m | (1 << slot));
        let slot = (0..budget as u8).find(|slot| used & (1 << slot) == 0)?;
        assign.insert(name, slot);
        active.push((name, slot, end));
    }
    Some(assign)
}

/// A value's definition and last touch, over a straight-line body. A name read
/// before it is written has no definition: it is live from the start.
fn live_ranges(body: &[Vec<(u16, bool, bool)>]) -> (BTreeMap<u16, i64>, BTreeMap<u16, i64>) {
    let mut def: BTreeMap<u16, i64> = BTreeMap::new();
    let mut last: BTreeMap<u16, i64> = BTreeMap::new();
    let mut first_use: BTreeMap<u16, i64> = BTreeMap::new();
    for (at, touches) in body.iter().enumerate() {
        for &(name, reads, writes) in touches {
            if name < VIRTUAL {
                continue;
            }
            if writes {
                def.entry(name).or_insert(at as i64);
            }
            if reads {
                first_use.entry(name).or_insert(at as i64);
            }
            last.insert(name, at as i64);
        }
    }
    for (&name, &use_at) in &first_use {
        if def.get(&name).is_none_or(|&def_at| use_at < def_at) {
            def.insert(name, -1);
        }
    }
    (def, last)
}

/// The value to spill and the window to spill it over, `(name, a, b)`: store it
/// before the body's `a`, reload it after its `b`. Candidate: a value live at
/// the body's pressure peak whose definition and every use lie outside the
/// over-budget run, with a store point (a full lane mask) after its last touch
/// before the run and a reload point (also full) before its next use. The
/// widest window wins, because one store and one reload then cover the whole
/// run instead of one reload per use.
fn split_at_pressure(
    body: &[Vec<(u16, bool, bool)>],
    before: &[bool],
    after: &[bool],
    budget: usize,
) -> Option<(u16, usize, usize)> {
    let (def, last) = live_ranges(body);
    let names: Vec<u16> = last
        .keys()
        .copied()
        .filter(|name| *name >= VIRTUAL)
        .collect();
    let counts: Vec<usize> = (0..body.len())
        .map(|at| {
            names
                .iter()
                .filter(|name| {
                    let def_at = def.get(name).copied().unwrap_or(-1);
                    def_at <= at as i64 && last.get(name).copied().unwrap_or(-1) >= at as i64
                })
                .count()
        })
        .collect();
    let (peak, pressure) = counts.iter().enumerate().max_by_key(|(_, count)| **count)?;
    if *pressure <= budget {
        return None;
    }
    let (mut ra, mut rb) = (peak, peak);
    while ra > 0 && counts[ra - 1] > budget {
        ra -= 1;
    }
    while rb + 1 < counts.len() && counts[rb + 1] > budget {
        rb += 1;
    }
    let mut best: Option<(usize, u16, usize, usize)> = None;
    for &name in &names {
        let def_at = def.get(&name).copied().unwrap_or(-1);
        let end = last.get(&name).copied().unwrap_or(-1);
        if def_at < 0 || def_at > peak as i64 || end < peak as i64 {
            continue;
        }
        let uses: Vec<usize> = (0..body.len())
            .filter(|at| body[*at].iter().any(|&(touched, _, _)| touched == name))
            .collect();
        if uses.iter().any(|&at| at >= ra && at <= rb) {
            continue;
        }
        let prev = uses
            .iter()
            .copied()
            .filter(|&at| at < ra)
            .max()
            .unwrap_or(0);
        let next = uses
            .iter()
            .copied()
            .filter(|&at| at > rb)
            .min()
            .unwrap_or(body.len());
        let Some(a) = (prev + 1..=ra).find(|&at| before.get(at).copied().unwrap_or(false)) else {
            continue;
        };
        let Some(b) = (rb..next)
            .rev()
            .find(|&at| after.get(at).copied().unwrap_or(false))
        else {
            continue;
        };
        if a >= peak || b < peak {
            continue;
        }
        let coverage = b - a + 1;
        if best.is_none_or(|(c, _, _, _)| coverage > c) {
            best = Some((coverage, name, a, b));
        }
    }
    best.map(|(_, name, a, b)| (name, a, b))
}

/// One register's live range over the machine order: the first op that writes
/// it and the last op that touches it.
fn ranges<Id: Copy + Ord + std::hash::Hash>(
    m: &MachineKernel,
    mut touched: impl FnMut(&MirOp) -> Vec<(Id, bool)>,
) -> BTreeMap<Id, (usize, usize)> {
    let mut live: BTreeMap<Id, (usize, usize)> = BTreeMap::new();
    for (at, (_, op)) in m.ops.iter().enumerate() {
        for (id, _write) in touched(op) {
            live.entry(id)
                .and_modify(|(_, last)| *last = at)
                .or_insert((at, at));
        }
    }
    live
}

/// The `Dst` registers an op touches: the register and whether it writes.
fn dst_touches(op: &MirOp) -> Vec<(Dst, bool)> {
    match *op {
        MirOp::Sfpload { reg, .. } => vec![(reg, false)],
        MirOp::Sfpstore { reg, .. } => vec![(reg, true)],
        MirOp::Setdst { reg } => vec![(reg, true)],
        MirOp::Mvmul { reg, .. } => vec![(reg, true)],
        MirOp::Elw { reg, .. }
        | MirOp::Gapool { reg, .. }
        | MirOp::Gmpool { reg, .. }
        | MirOp::Mova2d { reg, .. }
        | MirOp::Movb2d { reg, .. } => vec![(reg, true)],
        MirOp::Movd2b { reg, .. } => vec![(reg, false)],
        MirOp::SetPackOffset { reg, .. } => vec![(reg, false)],
        _ => Vec::new(),
    }
}

/// The loop bodies of the machine order, one span per IR loop: the n-th loop a
/// role opens is the n-th loop every other role opens, the same pairing
/// `optimize::reg_sync` uses for its seams.
fn loop_spans(m: &MachineKernel) -> Result<Vec<(usize, usize)>, ttStatus> {
    let slot = |role: Core| match role {
        Core::Unpack => Some(0),
        Core::Math => Some(1),
        Core::Pack => Some(2),
        Core::Reader | Core::Writer => None,
    };
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut stacks: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut opened = [0usize; 3];
    for (at, (role, op)) in m.ops.iter().enumerate() {
        let Some(r) = slot(*role) else {
            continue;
        };
        match op {
            MirOp::Loop { .. } | MirOp::LoopReg { .. } => {
                let ordinal = opened[r];
                opened[r] += 1;
                stacks[r].push(ordinal);
                if ordinal == spans.len() {
                    spans.push((at, 0));
                } else {
                    spans[ordinal].0 = spans[ordinal].0.min(at);
                }
            }
            MirOp::EndLoop => {
                let Some(ordinal) = stacks[r].pop() else {
                    return Err(crate::status::internal_bug(
                        "alloc: an end-of-loop has no open loop",
                    ));
                };
                spans[ordinal].1 = spans[ordinal].1.max(at);
            }
            _ => {}
        }
    }
    Ok(spans)
}

/// Assign the physical `Dst` registers, rewriting every op that names one.
///
/// A linear scan over the order: a register is free from the op after its last
/// touch, and the scan hands it to the next live range. The slab is rebuilt so
/// that its ids *are* the physical registers; `alloc` is the only step that
/// makes a `Dst` id mean a hardware register.
fn alloc_dsts(m: &mut MachineKernel) -> Result<(), ttStatus> {
    let mut live = ranges(m, dst_touches);
    // An unpacker's target is refilled by the next iteration as soon as the
    // math has *read* it (`reg_sync`'s pacing seam), so the slot must not be
    // shared with a later value of the same body: the math may still write it
    // as scratch after its last read, and that write would land on the next
    // iteration's tile.
    let spans = loop_spans(m)?;
    for (at, (role, op)) in m.ops.iter().enumerate() {
        let MirOp::Setdst { reg } = *op else {
            continue;
        };
        if *role != Core::Unpack {
            continue;
        }
        let Some((_, end)) = spans
            .iter()
            .filter(|(start, end)| *start < at && at < *end)
            .max_by_key(|(start, _)| *start)
        else {
            continue;
        };
        if let Some((_, last)) = live.get_mut(&reg) {
            *last = (*last).max(*end);
        }
    }
    let slots = m.dst_slots() as usize;
    // The order of first use, so the scan is deterministic.
    let mut order: Vec<(Dst, (usize, usize))> = live.into_iter().collect();
    order.sort_by_key(|(_, (first, last))| (*first, *last));

    let mut rename: BTreeMap<Dst, Dst> = BTreeMap::new();
    // Each physical register's end of life, by physical id.
    let mut ends: Vec<Option<usize>> = vec![None; slots];
    let all = order.clone();
    for (id, (first, last)) in order {
        let free = ends
            .iter()
            .position(|end| end.map(|end| end < first).unwrap_or(true));
        let Some(slot) = free else {
            if std::env::var_os("SKAM_DEBUG").is_some() {
                eprintln!("alloc: dst {id:?} {first}..{last} has no slot; live = {all:?}");
            }
            return Err(ttStatus::TT_ERROR_DST_SLOTS_EXHAUSTED);
        };
        ends[slot] = Some(last);
        rename.insert(id, Dst(slot as u16));
    }

    for (_, op) in m.ops.iter_mut() {
        rewrite_dsts(op, &rename);
    }
    // The slab becomes the physical registers: one entry per physical id, in
    // id order, with the dtype and shape of the first value that used it.
    let mut infos: Vec<Option<DstInfo>> = vec![None; slots];
    for (from, info) in m.dsts.iter() {
        let Some(to) = rename.get(&from).copied() else {
            continue;
        };
        let slot = usize::from(to);
        if infos.get(slot).map(|i| i.is_none()).unwrap_or(false) {
            infos[slot] = Some(*info);
        }
    }
    let mut fresh = crate::slab::Slab::<Dst, DstInfo>::new();
    for info in infos.into_iter().flatten() {
        fresh.push(info);
    }
    m.dsts = fresh;
    Ok(())
}

/// Assign the physical lane registers, rewriting every op that names one.
fn alloc_lregs(m: &mut MachineKernel) -> Result<(), ttStatus> {
    let live = ranges(m, |op| {
        op.lreg_params()
            .into_iter()
            .map(|(lreg, use_)| (lreg, use_.writes()))
            .collect()
    });
    let slots = usize::from(crate::alloc::LREGS);
    // Lane registers 8 upwards are the hardware's own constants (`LREG_ZERO`
    // is 9, `LREG_ONE` 10), not allocation slots: an op that names one keeps
    // it, so a sequence can read a constant without spending a work lane.
    let mut order: Vec<(LReg, (usize, usize))> = live
        .into_iter()
        .filter(|(id, _)| usize::from(*id) < slots)
        .collect();
    order.sort_by_key(|(_, (first, last))| (*first, *last));

    let mut rename: BTreeMap<LReg, LReg> = BTreeMap::new();
    let mut ends: Vec<Option<usize>> = vec![None; slots];
    for (id, (first, last)) in order {
        let free = ends
            .iter()
            .position(|end| end.map(|end| end < first).unwrap_or(true));
        let Some(slot) = free else {
            let _ = (id, first);
            return Err(ttStatus::TT_ERROR_LANE_REGISTER);
        };
        ends[slot] = Some(last);
        rename.insert(id, LReg(slot as u16));
    }

    for (_, op) in m.ops.iter_mut() {
        rewrite_lregs(op, &rename);
    }
    let mut dtys: Vec<Option<DType>> = vec![None; slots];
    for (from, dtype) in m.lregs.iter() {
        let Some(to) = rename.get(&from).copied() else {
            continue;
        };
        let slot = usize::from(to);
        if dtys.get(slot).map(|i| i.is_none()).unwrap_or(false) {
            dtys[slot] = Some(*dtype);
        }
    }
    let mut fresh = crate::slab::Slab::<LReg, DType>::new();
    for dtype in dtys.into_iter().flatten() {
        fresh.push(dtype);
    }
    m.lregs = fresh;
    Ok(())
}

/// Rewrite every `Dst` an op names.
fn rewrite_dsts(op: &mut MirOp, rename: &BTreeMap<Dst, Dst>) {
    let map = |reg: &mut Dst| {
        if let Some(to) = rename.get(reg) {
            *reg = *to;
        }
    };
    match op {
        MirOp::Sfpload { reg, .. }
        | MirOp::Sfpstore { reg, .. }
        | MirOp::Setdst { reg }
        | MirOp::Mvmul { reg, .. }
        | MirOp::Elw { reg, .. }
        | MirOp::Gapool { reg, .. }
        | MirOp::Gmpool { reg, .. }
        | MirOp::Mova2d { reg, .. }
        | MirOp::Movb2d { reg, .. }
        | MirOp::Movd2b { reg, .. }
        | MirOp::SetPackOffset { reg, .. } => map(reg),
        _ => {}
    }
}

/// Rewrite every lane register an op names.
fn rewrite_lregs(op: &mut MirOp, rename: &BTreeMap<LReg, LReg>) {
    op.map_lregs(|reg| {
        if let Some(to) = rename.get(reg) {
            *reg = *to;
        }
    });
}

/// Place every ring in L1, in id order, and the `LocalCBInterface` table after
/// them. Every ring is `ALIGN`-aligned, so the two sides of a copy are
/// congruent by construction.
pub fn plan_cbs(m: &mut MachineKernel) -> Result<(), ttStatus> {
    let mut at = tt_isa::arch::blackhole::l1::DATA.base as u32;
    let mut infos: Vec<(Cb, CbInfo)> = m.cbs.iter().map(|(id, info)| (id, *info)).collect();
    infos.sort_by_key(|(id, _)| *id);
    let count = infos.len() as u32;
    for (_, info) in infos.iter_mut() {
        let bytes = info
            .page_bytes
            .checked_mul(info.len)
            .ok_or(ttStatus::TT_ERROR_L1_OVERFLOW)?;
        let base = at.div_ceil(ALIGN) * ALIGN;
        let end = base
            .checked_add(bytes)
            .ok_or(ttStatus::TT_ERROR_L1_OVERFLOW)?;
        if u64::from(end) > tt_isa::arch::blackhole::l1::DATA.end {
            let _ = (base, end);
            return Err(ttStatus::TT_ERROR_L1_ARENA);
        }
        info.base = base;
        at = end;
    }
    let iface = at.div_ceil(ALIGN) * ALIGN;
    let table = crate::riscv::cb::COUNT
        .checked_mul(crate::riscv::cb::STRIDE)
        .ok_or(ttStatus::TT_ERROR_L1_OVERFLOW)?;
    let end = iface
        .checked_add(table)
        .ok_or(ttStatus::TT_ERROR_L1_OVERFLOW)?;
    if u64::from(end) > tt_isa::arch::blackhole::l1::DATA.end {
        let _ = (iface, end);
        return Err(ttStatus::TT_ERROR_L1_ARENA);
    }
    // A multicast copy's kernel semaphores follow the interface table: the
    // same four words in every tile, so a sender can name a receiver's word and
    // a receiver can name its sender's.
    let mcast = end.div_ceil(ALIGN) * ALIGN;
    let sites = u32::try_from(m.mcast.len()).map_err(|_| ttStatus::TT_ERROR_L1_OVERFLOW)?;
    let mcast_end = sites
        .checked_mul(crate::mir::McastSite::BYTES)
        .and_then(|bytes| mcast.checked_add(bytes))
        .ok_or(ttStatus::TT_ERROR_L1_OVERFLOW)?;
    if u64::from(mcast_end) > tt_isa::arch::blackhole::l1::DATA.end {
        let _ = (mcast, mcast_end);
        return Err(ttStatus::TT_ERROR_L1_ARENA);
    }
    for (i, site) in m.mcast.iter_mut().enumerate() {
        site.base = mcast + (i as u32) * crate::mir::McastSite::BYTES;
    }
    // The coordinate table follows the sites: one packed NoC coordinate per
    // logical `(x, y)`, indexed `y * COORD_COLUMNS + x`. A harvested chip's
    // logical columns are not its physical ones, so a rectangle cannot be
    // computed from the logical span alone.
    if !m.mcast.is_empty() {
        let at = mcast_end.div_ceil(ALIGN) * ALIGN;
        let bytes = crate::mir::COORDS_WORDS as u32 * 4;
        let end = at
            .checked_add(bytes)
            .ok_or(ttStatus::TT_ERROR_L1_OVERFLOW)?;
        if u64::from(end) > tt_isa::arch::blackhole::l1::DATA.end {
            let _ = (at, end);
            return Err(ttStatus::TT_ERROR_L1_ARENA);
        }
        m.coords = at;
    }
    for (k, (_, info)) in infos.iter_mut().enumerate() {
        let stride = u32::try_from(k)
            .ok()
            .and_then(|k| crate::riscv::cb::STRIDE.checked_mul(k))
            .ok_or(ttStatus::TT_ERROR_TOO_MANY_PARAMETERS)?;
        info.iface_entry = iface
            .checked_add(stride)
            .ok_or(ttStatus::TT_ERROR_L1_OVERFLOW)?;
    }
    let _ = count;
    for (id, info) in infos {
        m.cbs[id] = info;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::Core;

    /// The program header every activation test shares; `OP` is the chain.
    fn activation(op: &str) -> String {
        format!(
            "\
version 9
grid 1,1
%in = param f32 1024
%out = param f32 1024
%b0 = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
load 32,32 %x = %b0 [0]
{op}
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
"
        )
    }

    const SIGMOID_CHAIN: &str = "\
%nx = neg %x
%k = const f32 1.442695
%p = mul %nx %k
%e = exp2 %p
%one = const f32 1
%d = add %e %one
%r = reciprocal %d";

    const SILU_CHAIN: &str = "\
%nx = neg %x
%k = const f32 1.442695
%p = mul %nx %k
%e = exp2 %p
%one = const f32 1
%d = add %e %one
%s = reciprocal %d
%r = mul %x %s";

    const TANH_CHAIN: &str = "\
%doubled = add %x %x
%nx = neg %doubled
%k = const f32 1.442695
%p = mul %nx %k
%e = exp2 %p
%one = const f32 1
%d = add %e %one
%s = reciprocal %d
%two = const f32 2
%m = mul %s %two
%r = sub %m %one";

    const GELU_CHAIN: &str = "\
%half = const f32 0.5
%xh = mul %x %half
%x2 = mul %x %x
%x3 = mul %x2 %x
%c = const f32 0.044715
%cx3 = mul %x3 %c
%inner = add %x %cx3
%kk = const f32 0.7978846
%arg = mul %inner %kk
%doubled = add %arg %arg
%narg = neg %doubled
%k = const f32 1.442695
%p = mul %narg %k
%e = exp2 %p
%one = const f32 1
%d = add %e %one
%s = reciprocal %d
%two = const f32 2
%m = mul %s %two
%t = sub %m %one
%t1 = add %t %one
%r = mul %xh %t1";

    /// `elu = max(x, 0) - alpha * max(1 - exp2(x * log2(e)), 0)`: the shape
    /// `Tensor::elu` composes, written with the exponent the SFPU has.
    const ELU_CHAIN: &str = "\
%z = const f32 0
%pos = max %x %z
%l2e = const f32 1.442695
%p = mul %x %l2e
%e = exp2 %p
%one = const f32 1
%d = sub %one %e
%neg = max %d %z
%alpha = const f32 1
%t = mul %neg %alpha
%r = sub %pos %t";

    /// One program through the same machine passes `compile` runs.
    fn lowered(text: &str) -> MachineKernel {
        let k = crate::ir::Kernel::parse(text).expect("the program parses");
        k.verify().expect("the program verifies");
        let k = crate::unfold::unfold(&k).expect("the unfold");
        let mut k = crate::mover::add_movers(&k).expect("the movers");
        crate::sync_cbs::sync_cbs(&mut k).expect("the rings");
        k.verify().expect("the moved program verifies");
        let mut m = k.lower().expect("the lowering");
        fuse_bodies(&mut m).expect("the fusion");
        crate::optimize::reg_sync(&mut m).expect("the seams");
        crate::optimize::fuse(&mut m);
        m.alloc().expect("the registers fit");
        m
    }

    /// The row-body traffic of one core: repeats, lane loads, lane stores.
    fn traffic(m: &MachineKernel, core: Core) -> (usize, usize, usize) {
        let mut repeats = 0;
        let mut loads = 0;
        let mut stores = 0;
        for op in m.stream(core) {
            match op {
                MirOp::Repeat { .. } => repeats += 1,
                MirOp::Sfpload { .. } => loads += 1,
                MirOp::Sfpstore { .. } => stores += 1,
                _ => {}
            }
        }
        (repeats, loads, stores)
    }

    #[test]
    fn activations_are_one_row_body() {
        for (name, chain) in [
            ("sigmoid", SIGMOID_CHAIN),
            ("silu", SILU_CHAIN),
            ("tanh", TANH_CHAIN),
            ("gelu", GELU_CHAIN),
            ("elu", ELU_CHAIN),
        ] {
            let m = lowered(&activation(chain));
            assert_eq!(
                traffic(&m, Core::Math),
                (1, 1, 1),
                "{name} must be one body, one load, one store; got {:?}",
                traffic(&m, Core::Math)
            );
        }
    }

    /// Nine values alive at once, eight registers: the scan spills the one that
    /// is merely live through the peak, not used there, and the body fits.
    #[test]
    fn a_body_over_eight_lanes_spills_through_a_dst() {
        let mut m = MachineKernel::default();
        m.dst_is_16_bit = false;
        let mut body: Vec<MirOp> = Vec::new();
        for i in 0..9u16 {
            body.push(MirOp::LaneConst {
                lreg: LReg(VIRTUAL + i),
                bits: u32::from(i),
            });
        }
        body.push(MirOp::SfpMad {
            mod1: 0,
            a: LReg(VIRTUAL),
            b: LReg(VIRTUAL + 1),
            c: LReg(VIRTUAL + 2),
            d: LReg(VIRTUAL + 3),
        });
        body.push(MirOp::SfpMad {
            mod1: 0,
            a: LReg(VIRTUAL + 4),
            b: LReg(VIRTUAL + 5),
            c: LReg(VIRTUAL + 6),
            d: LReg(VIRTUAL + 7),
        });
        body.push(MirOp::BinarySfp {
            op: crate::mir::BinarySfpOp::Add,
            a: LReg(VIRTUAL + 8),
            b: LReg(VIRTUAL),
            d: LReg(VIRTUAL + 9),
        });
        allocate_body(&mut m, &mut body, DType::F32).expect("the spill makes it fit");
        assert_eq!(usize::from(m.dsts.len()), 1, "one spill tile");
        assert_eq!(
            body.iter()
                .filter(|op| matches!(op, MirOp::Sfpstore { .. }))
                .count(),
            1,
            "one store per spilled value: {body:?}"
        );
        assert_eq!(
            body.iter()
                .filter(|op| matches!(op, MirOp::Sfpload { .. }))
                .count(),
            1,
            "one reload per spilled value: {body:?}"
        );
        let used: BTreeSet<u16> = body
            .iter()
            .flat_map(|op| op.lreg_params())
            .map(|(lreg, _)| lreg.0)
            .filter(|id| !(8..16).contains(id))
            .collect();
        assert!(
            used.iter().all(|id| *id < u16::from(LREGS)),
            "every lane is physical after the scan: {used:?}"
        );
    }

    /// Two values that do not overlap share one physical register.
    #[test]
    fn two_disjoint_ranges_share_a_register() {
        let mut m = MachineKernel::default();
        let a = m.dsts.push(DstInfo {
            dtype: DType::F32,
            shape: [32, 32],
        });
        let b = m.dsts.push(DstInfo {
            dtype: DType::F32,
            shape: [32, 32],
        });
        assert_ne!(a, b);
        m.push(
            Core::Math,
            MirOp::Sfpstore {
                lreg: LReg(0),
                reg: a,
                at: 0,
                dtype: DType::F32,
            },
        );
        m.push(
            Core::Math,
            MirOp::Sfpload {
                lreg: LReg(0),
                reg: a,
                at: 0,
                dtype: DType::F32,
            },
        );
        m.push(
            Core::Math,
            MirOp::Sfpstore {
                lreg: LReg(0),
                reg: b,
                at: 0,
                dtype: DType::F32,
            },
        );
        alloc(&mut m).expect("they fit");
        assert_eq!(usize::from(m.dsts.len()), 1);
        assert_eq!(
            m.dsts.iter().next().expect("one register").1.shape,
            [32, 32]
        );
    }
}
