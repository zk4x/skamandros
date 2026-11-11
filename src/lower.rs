// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! SSA to machine IR: the first lowering step.
//!
//! Each semantic op becomes the instructions of the core that runs it: `unpack`
//! configures and fills the unpacker on T0, `sfpu` becomes one load/op/store
//! triple per `Dst` row on T1, `pack` configures and drains on T2. Addresses come
//! from the L1 plan, `Dst` slots from the allocation, and the hand-offs from the
//! allocation's semaphores -- the caller wrote none of them.

use crate::alloc::{Alloc, Place, Role, Sync, SyncKind};
use crate::ir::{Error, Kernel, Op};
use crate::mir::{Machine, MirOp, SfpOp};
use crate::planner::Plan;

/// `Dst` rows one 32x32 F32 tile occupies (`tt_isa`'s SFPU row is 16 datums).
pub const DST_ROWS: u32 = 64;
/// Datums in one tile.
pub const TILE_DATUMS: u32 = 1024;

/// Lower `k` to the three compute programs. The mover programs are separate.
pub fn lower(k: &Kernel, plan: &Plan, alloc: &Alloc) -> Result<Machine, Error> {
    let mut m = Machine::default();
    // Hand-offs are emitted in program order around the compute ops they pace;
    // the allocation lists them as a whole-program protocol.
    let mut syncs: Vec<&Sync> = alloc.sync.iter().collect();
    for (at, op) in k.ops.iter().enumerate() {
        match *op {
            Op::Unpack { value, buf } => {
                let place = alloc.place_of(value);
                let row = match place {
                    Place::Dst(slot) => u32::from(slot) * DST_ROWS,
                    Place::SrcA(_) | Place::SrcB(_) => {
                        return Err(Error(format!(
                            "op {at}: unpack %{value} lands in a Src bank; the Src path is not lowered yet"
                        )));
                    }
                };
                m.t0.push(MirOp::Setdst { row });
                m.t0.push(MirOp::Setadczw { unit: 0, face: 0 });
                m.t0.push(MirOp::Setadcx { unit: 0, first: 0, count: TILE_DATUMS });
                m.t0.push(MirOp::Unpacr { src: plan.buf[buf as usize], unit: 0, count: TILE_DATUMS });
                emit_sync(&mut m, &mut syncs, Role::Unpack, true);
            }
            Op::Sfpu { value, kind, a, b } => {
                if b.is_some() {
                    return Err(Error(format!("op {at}: binary SFPU ops are not lowered yet")));
                }
                let (src_row, dst_row) = match (alloc.place_of(a), alloc.place_of(value)) {
                    (Place::Dst(s), Place::Dst(d)) => (u32::from(s) * DST_ROWS, u32::from(d) * DST_ROWS),
                    _ => return Err(Error(format!("op {at}: an SFPU value is not in Dst"))),
                };
                let (lreg_in, lreg_out) = alloc.lreg[value as usize];
                // Wait for the tile before reading `Dst`, publish the result
                // after writing it.
                emit_sync(&mut m, &mut syncs, Role::Math, false);
                // One `SFPLOAD`/`SFPSTORE` reaches four `Dst` rows and eight of
                // the sixteen columns (`SFPLOAD.md`: `row = (addr & !3) +
                // lane/8`, `col = (lane & 7) * 2 + (addr & 2)`), so a 64-row
                // tile is 32 addresses, stepping by two: bit 1 is the odd half.
                for at in (0..DST_ROWS).step_by(2) {
                    m.t1.push(MirOp::Sfpload { lreg: lreg_in, row: src_row + at });
                    m.t1.push(MirOp::Sfp { op: SfpOp::from(kind), a: lreg_in, b: None, d: lreg_out });
                    m.t1.push(MirOp::Sfpstore { lreg: lreg_out, row: dst_row + at });
                }
                emit_sync(&mut m, &mut syncs, Role::Math, true);
            }
            Op::Matmul { .. } => {
                return Err(Error(format!("op {at}: the matrix path is not lowered yet")));
            }
            Op::Pack { value, buf } => {
                if !matches!(alloc.place_of(value), Place::Dst(_)) {
                    return Err(Error(format!("op {at}: packing a value that is not in Dst")));
                }
                emit_sync(&mut m, &mut syncs, Role::Pack, false);
                m.t2.push(MirOp::Pack { dst: plan.buf[buf as usize], rows: DST_ROWS, last: true });
                emit_sync(&mut m, &mut syncs, Role::Pack, true);
            }
            // Boundary traffic and the explicit syncs are not compute.
            Op::CopyIn { .. } | Op::CopyOut { .. } | Op::Reserve { .. } | Op::Push { .. } | Op::Wait { .. } | Op::Pop { .. } => {}
        }
    }
    if !syncs.is_empty() {
        return Err(Error(format!(
            "{} hand-offs were never placed on a core",
            syncs.len()
        )));
    }
    Ok(m)
}

/// Emit the hand-offs this role owns at the current point: the posts when
/// `posts` is true, the waits when it is false.
fn emit_sync(m: &mut Machine, syncs: &mut Vec<&Sync>, role: Role, posts: bool) {
    while let Some(pos) = syncs.iter().position(|s| {
        s.role == role && matches!((s.kind, posts), (SyncKind::Post, true) | (SyncKind::Wait { .. }, false))
    }) {
        let s = *syncs.remove(pos);
        let op = match s.kind {
            SyncKind::Post => MirOp::SemPost { sem: s.sem.index() },
            SyncKind::Wait { before } => MirOp::SemWait { sem: s.sem.index(), before },
        };
        match role {
            Role::Unpack => m.t0.push(op),
            Role::Math => m.t1.push(op),
            Role::Pack => m.t2.push(op),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::allocate;
    use crate::ir::Kernel;
    use crate::planner::place;

    const ELEMENTWISE: &str = "\
version 1
grid 1 1
param in f32 4096
param out f32 4096
buf 0 4096 plain rm
buf 1 4096 plain rm
copy_in param 0 offset 0 len 4096 buf 0
%0 = unpack buf 0
%1 = sfpu neg %0
pack %1 buf 1
copy_out buf 1 param 1 offset 0 len 4096
";

    #[test]
    fn the_sfpu_body_is_one_row_triple_per_dst_row() {
        let k = Kernel::parse(ELEMENTWISE).unwrap();
        let plan = place(&k).unwrap();
        let alloc = allocate(&k).unwrap();
        let m = lower(&k, &plan, &alloc).unwrap();
        let math: Vec<MirOp> = m.t1.iter().copied().filter(|op| !matches!(op, MirOp::SemWait { .. } | MirOp::SemPost { .. })).collect();
        assert_eq!(math.len(), (DST_ROWS / 2 * 3) as usize);
        assert!(matches!(math[0], MirOp::Sfpload { lreg: 0, row: 0 }));
        assert!(matches!(math[1], MirOp::Sfp { op: SfpOp::Neg, a: 0, b: None, d: 1 }));
        assert!(matches!(math[2], MirOp::Sfpstore { lreg: 1, row: 0 }));
        // The last triple is the last address, in place.
        assert!(matches!(math[math.len() - 3], MirOp::Sfpload { row, .. } if row == DST_ROWS - 2));
    }

    #[test]
    fn the_unpack_reads_the_planned_l1_buffer() {
        let k = Kernel::parse(ELEMENTWISE).unwrap();
        let plan = place(&k).unwrap();
        let alloc = allocate(&k).unwrap();
        let m = lower(&k, &plan, &alloc).unwrap();
        assert!(m.t0.iter().any(|op| matches!(op, MirOp::Unpacr { src, .. } if *src == plan.buf[0])));
        assert!(m.t0.iter().any(|op| matches!(op, MirOp::Setdst { row: 0 })));
    }

    #[test]
    fn the_pack_drains_to_the_planned_l1_buffer() {
        let k = Kernel::parse(ELEMENTWISE).unwrap();
        let plan = place(&k).unwrap();
        let alloc = allocate(&k).unwrap();
        let m = lower(&k, &plan, &alloc).unwrap();
        assert!(m.t2.iter().any(|op| matches!(op, MirOp::Pack { dst, .. } if *dst == plan.buf[1])));
    }

    #[test]
    fn the_hand_offs_land_on_their_cores() {
        let k = Kernel::parse(ELEMENTWISE).unwrap();
        let plan = place(&k).unwrap();
        let alloc = allocate(&k).unwrap();
        let m = lower(&k, &plan, &alloc).unwrap();
        assert!(m.t0.iter().any(|op| matches!(op, MirOp::SemPost { sem: 0 })), "unpack publishes");
        assert!(m.t1.iter().any(|op| matches!(op, MirOp::SemWait { sem: 0, .. })), "math waits");
        assert!(m.t1.iter().any(|op| matches!(op, MirOp::SemPost { sem: 1 })), "math publishes");
        assert!(m.t2.iter().any(|op| matches!(op, MirOp::SemWait { sem: 1, .. })), "pack waits");
    }
}
