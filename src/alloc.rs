// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The machine-side allocation: what `lower` minted without a limit, brought
//! inside the limits the hardware has.
//!
//! `lower` names one `Dst` per value and one lane register per temporary, with
//! no reuse at all: correctness first, and the ids are only names. This module
//! is the pass that decides which of those names may share a physical register,
//! by liveness over the one machine order, and that places the circular buffers
//! in L1.
//!
//! Nothing here runs in the virtual IR: the SSA program has no allocation at
//! all.

use std::collections::BTreeMap;

use crate::ir::{DType, Error};
use crate::mir::{Cb, CbInfo, Dst, DstInfo, LReg, MachineKernel, MirOp};

/// Writable lane registers (`SFPLOAD`/`SFPSTORE` address `L0..L7`).
pub const LREGS: u8 = 8;
/// Alignment every ring starts at: the NoC requires the two addresses of a
/// request to be congruent, and every copy's L1 side is a page start.
const ALIGN: u32 = 64;

/// Bring the machine kernel's `Dst` registers and lane registers inside the
/// hardware's limits, in place.
pub fn alloc(m: &mut MachineKernel) -> Result<(), Error> {
    alloc_dsts(m)?;
    alloc_lregs(m)?;
    Ok(())
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

/// The lane registers an op touches: the register and whether it writes.
fn lreg_touches(op: &MirOp) -> Vec<(LReg, bool)> {
    match *op {
        MirOp::Sfpload { lreg, .. } | MirOp::LaneConst { lreg, .. } => vec![(lreg, true)],
        MirOp::Sfpstore { lreg, .. } => vec![(lreg, false)],
        MirOp::UnarySfp { a, d, .. } | MirOp::Cast { a, d, .. } => vec![(a, false), (d, true)],
        MirOp::BinarySfp { a, b, d, .. }
        | MirOp::Compare { a, b, d, .. }
        | MirOp::Max { a, b, d } => vec![(a, false), (b, false), (d, true)],
        MirOp::Select { cond, t, f, d } => {
            vec![(cond, false), (t, false), (f, false), (d, true)]
        }
        _ => Vec::new(),
    }
}

/// Assign the physical `Dst` registers, rewriting every op that names one.
///
/// A linear scan over the order: a register is free from the op after its last
/// touch, and the scan hands it to the next live range. The slab is rebuilt so
/// that its ids *are* the physical registers; `alloc` is the only step that
/// makes a `Dst` id mean a hardware register.
fn alloc_dsts(m: &mut MachineKernel) -> Result<(), Error> {
    let live = ranges(m, dst_touches);
    let slots = m.dst_slots() as usize;
    // The order of first use, so the scan is deterministic.
    let mut order: Vec<(Dst, (usize, usize))> = live.into_iter().collect();
    order.sort_by_key(|(_, (first, last))| (*first, *last));

    let mut rename: BTreeMap<Dst, Dst> = BTreeMap::new();
    // Each physical register's end of life, by physical id.
    let mut ends: Vec<Option<usize>> = vec![None; slots];
    for (id, (first, last)) in order {
        let free = ends
            .iter()
            .position(|end| end.map(|end| end < first).unwrap_or(true));
        let Some(slot) = free else {
            let _ = (id, first);
            return Err(Error::TT_ERROR_DST_SLOTS_EXHAUSTED);
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
fn alloc_lregs(m: &mut MachineKernel) -> Result<(), Error> {
    let live = ranges(m, lreg_touches);
    let slots = 8usize;
    let mut order: Vec<(LReg, (usize, usize))> = live.into_iter().collect();
    order.sort_by_key(|(_, (first, last))| (*first, *last));

    let mut rename: BTreeMap<LReg, LReg> = BTreeMap::new();
    let mut ends: Vec<Option<usize>> = vec![None; slots];
    for (id, (first, last)) in order {
        let free = ends
            .iter()
            .position(|end| end.map(|end| end < first).unwrap_or(true));
        let Some(slot) = free else {
            let _ = (id, first);
            return Err(Error::TT_ERROR_LANE_REGISTER);
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
    let map = |reg: &mut LReg| {
        if let Some(to) = rename.get(reg) {
            *reg = *to;
        }
    };
    match op {
        MirOp::Sfpload { lreg, .. }
        | MirOp::Sfpstore { lreg, .. }
        | MirOp::LaneConst { lreg, .. } => map(lreg),
        MirOp::UnarySfp { a, d, .. } | MirOp::Cast { a, d, .. } => {
            map(a);
            map(d);
        }
        MirOp::BinarySfp { a, b, d, .. }
        | MirOp::Compare { a, b, d, .. }
        | MirOp::Max { a, b, d } => {
            map(a);
            map(b);
            map(d);
        }
        MirOp::Select { cond, t, f, d } => {
            map(cond);
            map(t);
            map(f);
            map(d);
        }
        _ => {}
    }
}

/// Place every ring in L1, in id order, and the `LocalCBInterface` table after
/// them. Every ring is `ALIGN`-aligned, so the two sides of a copy are
/// congruent by construction.
pub fn plan_cbs(m: &mut MachineKernel) -> Result<(), Error> {
    let mut at = tt_isa::l1::DATA.base as u32;
    let mut infos: Vec<(Cb, CbInfo)> = m.cbs.iter().map(|(id, info)| (id, *info)).collect();
    infos.sort_by_key(|(id, _)| *id);
    let count = infos.len() as u32;
    for (_, info) in infos.iter_mut() {
        let bytes = info
            .page_bytes
            .checked_mul(info.len)
            .ok_or(Error::TT_ERROR_L1_OVERFLOW)?;
        let base = at.div_ceil(ALIGN) * ALIGN;
        let end = base.checked_add(bytes).ok_or(Error::TT_ERROR_L1_OVERFLOW)?;
        if u64::from(end) > tt_isa::l1::DATA.end {
            let _ = (base, end);
            return Err(Error::TT_ERROR_L1_ARENA);
        }
        info.base = base;
        at = end;
    }
    let iface = at.div_ceil(ALIGN) * ALIGN;
    let table = crate::riscv::cb::COUNT
        .checked_mul(crate::riscv::cb::STRIDE)
        .ok_or(Error::TT_ERROR_L1_OVERFLOW)?;
    let end = iface
        .checked_add(table)
        .ok_or(Error::TT_ERROR_L1_OVERFLOW)?;
    if u64::from(end) > tt_isa::l1::DATA.end {
        let _ = (iface, end);
        return Err(Error::TT_ERROR_L1_ARENA);
    }
    for (k, (_, info)) in infos.iter_mut().enumerate() {
        let stride = u32::try_from(k)
            .ok()
            .and_then(|k| crate::riscv::cb::STRIDE.checked_mul(k))
            .ok_or(Error::TT_ERROR_TOO_MANY_PARAMETERS)?;
        info.iface_entry = iface
            .checked_add(stride)
            .ok_or(Error::TT_ERROR_L1_OVERFLOW)?;
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
