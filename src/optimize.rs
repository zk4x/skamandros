// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The passes over the machine IR: fusion and verification.
//!
//! Allocation (`Dst` slots, `Src` banks, lane registers, the cross-core
//! semaphores) happens in [`crate::alloc`], before lowering, because it is an SSA
//! property. What is left to do after the machine IR exists is the row body: the
//! lowering emits one load/op/store triple per `Dst` row, and fusion collapses a
//! run of them into one body plus a repeat, which is what the hardware's replay
//! expander and row-address modifier execute.

use crate::alloc::DST_SLOTS;
use crate::ir::Error;
use crate::lower::DST_ROWS;
use crate::mir::{Machine, MirOp};

/// Collapse every run of identical consecutive row triples into one body plus a
/// [`MirOp::Repeat`].
///
/// A triple is fused with the next when the instructions are identical and both
/// `Dst` rows advance by exactly one, so the body is the row-0 form and the
/// repeat walks the rest.
pub fn fuse(m: &mut Machine) {
    for core in [&mut m.t0, &mut m.t1, &mut m.t2] {
        let ops = std::mem::take(core);
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
        *core = out;
    }
}

/// The three ops at `at` if they are a lane-register row body.
fn triple(ops: &[MirOp], at: usize) -> Option<[MirOp; 3]> {
    let [a, b, c] = [*ops.get(at)?, *ops.get(at + 1)?, *ops.get(at + 2)?];
    match (a, b, c) {
        (MirOp::Sfpload { .. }, MirOp::Sfp { .. }, MirOp::Sfpstore { .. }) => Some([a, b, c]),
        _ => None,
    }
}

/// Is `following` `body` with both `Dst` rows advanced by one, everything else
/// identical?
fn same_shifted(body: &[MirOp; 3], following: &[MirOp; 3]) -> bool {
    match (body, following) {
        (
            [
                MirOp::Sfpload {
                    lreg: la,
                    row: ra,
                    dtype: ia,
                },
                MirOp::Sfp {
                    op: oa,
                    a: aa,
                    b: ba,
                    d: da,
                },
                MirOp::Sfpstore {
                    lreg: sa,
                    row: sa_row,
                    dtype: isa,
                },
            ],
            [
                MirOp::Sfpload {
                    lreg: lb,
                    row: rb,
                    dtype: ib,
                },
                MirOp::Sfp {
                    op: ob,
                    a: ab,
                    b: bb,
                    d: db,
                },
                MirOp::Sfpstore {
                    lreg: sb,
                    row: sb_row,
                    dtype: isb,
                },
            ],
        ) => {
            la == lb
                && sa == sb
                && ia == ib
                && isa == isb
                && oa == ob
                && aa == ab
                && ba == bb
                && da == db
                && *rb == ra + 2
                && *sb_row == *sa_row + 2
        }
        _ => false,
    }
}

/// Check every machine-IR invariant. A kernel that fails this must not be
/// encoded.
pub fn verify(m: &Machine) -> Result<(), Error> {
    for (name, ops) in [("t0", &m.t0), ("t1", &m.t1), ("t2", &m.t2)] {
        let mut at = 0usize;
        let mut depth = 0usize;
        while at < ops.len() {
            match ops[at] {
                MirOp::Sfpload { lreg, row, .. } => {
                    if lreg >= crate::alloc::LREGS {
                        let _ = (name, at, lreg);
                        return Err(Error::TT_ERROR_LANE_REGISTER);
                    }
                    if row >= u32::from(DST_SLOTS) * DST_ROWS {
                        let _ = (name, at, row);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                }
                MirOp::Sfpstore { lreg, row, .. } => {
                    if lreg >= crate::alloc::LREGS {
                        let _ = (name, at, lreg);
                        return Err(Error::TT_ERROR_LANE_REGISTER);
                    }
                    if row >= u32::from(DST_SLOTS) * DST_ROWS {
                        let _ = (name, at, row);
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
                    if lreg >= crate::alloc::LREGS {
                        let _ = (name, at, lreg);
                        return Err(Error::TT_ERROR_LANE_REGISTER);
                    }
                }
                MirOp::Sfp { a, b, d, .. } => {
                    for lreg in [Some(a), b, Some(d)].into_iter().flatten() {
                        if lreg >= crate::alloc::LREGS {
                            let _ = (name, at, lreg);
                            return Err(Error::TT_ERROR_LANE_REGISTER);
                        }
                    }
                }
                MirOp::Cast { a, d, .. } => {
                    for lreg in [a, d] {
                        if lreg >= crate::alloc::LREGS {
                            let _ = (name, at, lreg);
                            return Err(Error::TT_ERROR_LANE_REGISTER);
                        }
                    }
                }
                MirOp::Setdst { row } => {
                    if row >= u32::from(DST_SLOTS) * DST_ROWS {
                        let _ = (name, at, row);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                }
                MirOp::SemPost { sem }
                | MirOp::SemWait { sem, .. }
                | MirOp::SemWaitFree { sem }
                | MirOp::SemWaitData { sem }
                | MirOp::SemGet { sem } => {
                    if sem >= tt_isa::sync::Semaphore::COUNT {
                        let _ = (name, at, sem);
                        return Err(Error::TT_ERROR_SEMAPHORE);
                    }
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
                        if lreg >= crate::alloc::LREGS {
                            let _ = (name, at, lreg);
                            return Err(Error::TT_ERROR_LANE_REGISTER);
                        }
                    }
                }
                MirOp::Select { cond, t, f, d } => {
                    for lreg in [cond, t, f, d] {
                        if lreg >= crate::alloc::LREGS {
                            let _ = (name, at, lreg);
                            return Err(Error::TT_ERROR_LANE_REGISTER);
                        }
                    }
                }
                MirOp::Max { a, b, d } => {
                    for lreg in [a, b, d] {
                        if lreg >= crate::alloc::LREGS {
                            let _ = (name, at, lreg);
                            return Err(Error::TT_ERROR_LANE_REGISTER);
                        }
                    }
                }
                MirOp::Unpacr { cb, .. }
                | MirOp::Pack { cb, .. }
                | MirOp::CbWait { cb }
                | MirOp::CbPop { cb }
                | MirOp::CbReserve { cb }
                | MirOp::CbPush { cb } => {
                    let _ = cb;
                }
                MirOp::Setadczw { .. } | MirOp::Setadcx { .. } => {}
                MirOp::Mvmul { row, addr_mod, .. } => {
                    if row >= u32::from(DST_SLOTS) * DST_ROWS {
                        let _ = (name, at, row);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                    if addr_mod > 7 {
                        let _ = (name, at, addr_mod);
                        return Err(Error::TT_ERROR_ADDR_MOD);
                    }
                }
                MirOp::Zeroacc { block } => {
                    // `CLR_16` names one of 64 sixteen-row blocks of `Dst`.
                    if block >= 64 {
                        let _ = (name, at, block);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                }
                MirOp::Gapool {
                    dst_row, addr_mod, ..
                }
                | MirOp::Gmpool {
                    dst_row, addr_mod, ..
                } => {
                    if dst_row >= u32::from(DST_SLOTS) * DST_ROWS {
                        let _ = (name, at, dst_row);
                        return Err(Error::TT_ERROR_DST_ROW);
                    }
                    if addr_mod > 7 {
                        let _ = (name, at, addr_mod);
                        return Err(Error::TT_ERROR_ADDR_MOD);
                    }
                }
                MirOp::Movd2b {
                    src_row,
                    dst_row,
                    addr_mod,
                    ..
                }
                | MirOp::Movb2d {
                    src_row,
                    dst_row,
                    addr_mod,
                    ..
                } => {
                    // `SrcB` is 64 rows of 16 datums.
                    if src_row >= 64 {
                        let _ = (name, at, src_row);
                        return Err(Error::TT_ERROR_SRC_ROW);
                    }
                    if dst_row >= u32::from(DST_SLOTS) * DST_ROWS {
                        let _ = (name, at, dst_row);
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
                    dst_row,
                    addr_mod,
                    ..
                } => {
                    if src_row >= 64 {
                        let _ = (name, at, src_row);
                        return Err(Error::TT_ERROR_SRC_ROW);
                    }
                    if dst_row >= u32::from(DST_SLOTS) * DST_ROWS {
                        let _ = (name, at, dst_row);
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
    use crate::alloc::allocate;
    use crate::ir::Kernel;
    use crate::lower::lower;
    use crate::planner::place;

    const NEG: &str = "\
version 7
grid 1,1
%in = param f32 4096
%out = param f32 4096
%b0 = buffer f32 32,32
copy_to %b0 from %in [0]
%x = load %b0 [0]
%y = neg %x
store_to %b0 from %y [0]
copy_to %out from %b0 [0]
";

    fn machine() -> Machine {
        let k = Kernel::parse(NEG).unwrap();
        let types = k.verify().unwrap();
        let plan = place(&k, &types).unwrap();
        let alloc = allocate(&k, &types).unwrap();
        lower(&k, &types, &plan, &alloc).unwrap()
    }

    #[test]
    fn sixty_four_row_triples_become_one_body() {
        let mut m = machine();
        let before = m.t1.len();
        fuse(&mut m);
        // One repeat marker plus the body, around the two hand-offs and the
        // packer drain wait.
        assert_eq!(m.t1.len(), 7, "got {:?}", m.t1);
        assert!(matches!(m.t1[1], MirOp::Repeat { len: 3, count: 32 }));
        assert!(matches!(
            m.t1[2],
            MirOp::Sfpload {
                lreg: 0,
                row: 0,
                dtype: crate::ir::DType::F32
            }
        ));
        assert_eq!(m.t1.len(), before - (32 - 1) * 3 + 1);
    }

    #[test]
    fn a_hand_off_breaks_a_run() {
        let mut m = machine();
        // Cut the body: a semaphore in the middle must stop fusion across it.
        m.t1.insert(3 * 16, MirOp::SemPost { sem: 7 });
        fuse(&mut m);
        let repeats =
            m.t1.iter()
                .filter(|op| matches!(op, MirOp::Repeat { .. }))
                .count();
        assert_eq!(repeats, 2, "one repeat on each side of the hand-off");
    }

    #[test]
    fn a_different_op_is_not_fused() {
        let mut m = machine();
        // Row 1 negated with a different lane register: not the same body.
        if let MirOp::Sfp { d, .. } = &mut m.t1[5] {
            *d = 5;
        }
        fuse(&mut m);
        let repeats: Vec<u16> =
            m.t1.iter()
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
        m.t1.push(MirOp::Sfpload {
            lreg: 9,
            row: 0,
            dtype: crate::ir::DType::F32,
        });
        assert!(verify(&m).is_err());
    }

    #[test]
    fn a_dst_row_past_the_last_tile_is_refused() {
        let mut m = machine();
        m.t1.push(MirOp::Sfpstore {
            lreg: 0,
            row: 8 * DST_ROWS,
            dtype: crate::ir::DType::F32,
        });
        assert!(verify(&m).is_err());
    }

    #[test]
    fn an_odd_sfpstore_row_is_refused() {
        // `SFPSTORE`'s address names a row pair; the hardware refuses an odd
        // one, so the machine IR must too, before it reaches the encoder.
        let mut m = machine();
        m.t1.push(MirOp::Sfpstore {
            lreg: 0,
            row: 1,
            dtype: crate::ir::DType::F32,
        });
        assert_eq!(
            verify(&m).expect_err("an odd row"),
            Error::TT_ERROR_DST_ROW_ODD
        );
    }

    #[test]
    fn a_repeat_that_runs_past_its_program_is_refused() {
        let mut m = machine();
        m.t1.push(MirOp::Repeat { len: 99, count: 2 });
        assert!(verify(&m).is_err());
    }
}
