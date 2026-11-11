// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Automatic allocation: SSA values to `Dst` slots, `SrcA`/`SrcB` banks, lane
//! registers, and the semaphores between the three cores.
//!
//! The caller never names a physical resource. A value's *home* follows from how
//! it is used: a matrix operand lives in `SrcA` or `SrcB`, everything else in a
//! `Dst` slot. A value needed in both is refused rather than copied behind the
//! caller's back. `Dst` slots and `Src` banks are assigned by liveness, and lane
//! registers are the temporaries inside one SFPU body.
//!
//! The cross-core hand-off is the same for every elementwise kernel: the
//! unpacker publishes the tile, the math core waits for it and publishes its
//! result, the packer waits for that. Each hand-off is one semaphore.

use crate::ir::{Error, Kernel, Op, ValueId};
use tt_isa::sync::Semaphore;

/// Which of the three Tensix cores runs an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Unpack,
    Math,
    Pack,
}

/// Where a value lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    Dst(u8),
    SrcA(u8),
    SrcB(u8),
}

/// `Dst` slots the hardware has (32-bit tiles).
pub const DST_SLOTS: u8 = 8;
/// Banks per `Src` side.
pub const SRC_BANKS: u8 = 2;
/// Writable lane registers (`SFPLOAD`/`SFPSTORE` address `L0..L7`).
pub const LREGS: u8 = 8;

/// One cross-core hand-off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sync {
    pub role: Role,
    pub sem: Semaphore,
    pub kind: SyncKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncKind {
    /// `SEMPOST` once this core's unit has drained.
    Post,
    /// `SEMWAIT` holding the given `Before` mask.
    Wait { before: u8 },
}

/// The allocation of one kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alloc {
    /// Home of each SSA value.
    pub place: Vec<Place>,
    /// Lane registers each value's own SFPU body uses: `(input, output)`.
    pub lreg: Vec<(u8, u8)>,
    /// Hand-offs in program order, one group per producer/consumer seam.
    pub sync: Vec<Sync>,
    /// Peak `Dst` slots in use, for the report and for the limits test.
    pub dst_peak: u8,
}

impl Alloc {
    /// The home of `value`.
    pub fn place_of(&self, value: ValueId) -> Place {
        self.place[value as usize]
    }
}

/// Semaphores used by the elementwise path, in the order they are taken.
const SEM_TILE: Semaphore = match Semaphore::new(0) {
    Some(s) => s,
    None => panic!("semaphore 0 exists"),
};
const SEM_RESULT: Semaphore = match Semaphore::new(1) {
    Some(s) => s,
    None => panic!("semaphore 1 exists"),
};

/// Allocate homes and hand-offs for `k`.
pub fn allocate(k: &Kernel) -> Result<Alloc, Error> {
    let mut place: Vec<Option<Place>> = vec![None; k.ops.len()];
    let mut lreg: Vec<(u8, u8)> = vec![(0, 0); k.ops.len()];
    let mut dst_free: Vec<bool> = vec![false; DST_SLOTS as usize];
    let mut bank_free: [[bool; SRC_BANKS as usize]; 2] = [[false; SRC_BANKS as usize]; 2];
    let mut last_use: Vec<Option<usize>> = vec![None; k.ops.len()];
    let mut freed: Vec<bool> = vec![false; k.ops.len()];
    let mut live_dst = 0usize;
    let mut peak_dst = 0usize;

    // Pass 1: where each value is consumed, and when it dies.
    let mut want_src: Vec<Option<u8>> = vec![None; k.ops.len()];
    for (at, op) in k.ops.iter().enumerate() {
        for v in op.used() {
            last_use[v as usize] = Some(at);
        }
        if let Op::Matmul { a, b, .. } = *op {
            for (v, side) in [(a, 0u8), (b, 1u8)] {
                match want_src[v as usize] {
                    None => want_src[v as usize] = Some(side),
                    Some(seen) if seen != side => {
                        return Err(Error(format!(
                            "value %{v} is used as both SrcA and SrcB, which the hardware cannot do"
                        )));
                    }
                    Some(_) => {}
                }
            }
        }
    }
    for (at, op) in k.ops.iter().enumerate() {
        // Free the homes of values that died before this instruction, once.
        for id in 0..place.len() {
            let Some(home) = place[id] else { continue };
            if !freed[id] && last_use[id].is_some_and(|u| u <= at) {
                match home {
                    Place::Dst(slot) => {
                        dst_free[slot as usize] = false;
                        live_dst = live_dst.saturating_sub(1);
                    }
                    Place::SrcA(bank) => bank_free[0][bank as usize] = false,
                    Place::SrcB(bank) => bank_free[1][bank as usize] = false,
                }
                freed[id] = true;
            }
        }
        let Some(value) = op.defined() else { continue };
        if matches!(k.ops[at], Op::Sfpu { .. }) {
            // The lane registers are this value's body temporaries.
            lreg[value as usize] = (0, 1);
        }
        let home = match want_src[value as usize] {
            Some(side) => {
                let Some(bank) = bank_free[side as usize].iter().position(|free| !free) else {
                    return Err(Error(format!(
                        "value %{value} needs Src{} but both banks are live",
                        if side == 0 { 'A' } else { 'B' }
                    )));
                };
                bank_free[side as usize][bank] = true;
                if side == 0 { Place::SrcA(bank as u8) } else { Place::SrcB(bank as u8) }
            }
            None => {
                let Some(slot) = dst_free.iter().position(|free| !free) else {
                    return Err(Error(format!(
                        "value %{value} needs a Dst slot but all {DST_SLOTS} are live"
                    )));
                };
                dst_free[slot] = true;
                Place::Dst(slot as u8)
            }
        };
        place[value as usize] = Some(home);
        if matches!(home, Place::Dst(_)) {
            live_dst += 1;
            peak_dst = peak_dst.max(live_dst);
        }
    }
    let values = k.ops.iter().filter(|op| op.defined().is_some()).count();
    let mut place: Vec<Place> = (0..values)
        .map(|v| place[v].ok_or_else(|| Error("value was never placed".into())))
        .collect::<Result<_, _>>()?;
    let _ = &mut place;
    let dst_peak = peak_dst as u8;

    // Hand-offs: unpack -> math -> pack, plus the free hand-off back to unpack.
    let mut sync = Vec::new();
    if k.ops.iter().any(|op| matches!(op, Op::Unpack { .. })) {
        sync.push(Sync { role: Role::Unpack, sem: SEM_TILE, kind: SyncKind::Post });
        sync.push(Sync { role: Role::Math, sem: SEM_TILE, kind: SyncKind::Wait { before: 0 } });
    }
    if k.ops.iter().any(|op| matches!(op, Op::Sfpu { .. } | Op::Matmul { .. })) {
        sync.push(Sync { role: Role::Math, sem: SEM_RESULT, kind: SyncKind::Post });
        sync.push(Sync { role: Role::Pack, sem: SEM_RESULT, kind: SyncKind::Wait { before: 0 } });
    }
    Ok(Alloc { place, lreg, sync, dst_peak })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Kernel;

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
    fn an_elementwise_value_lands_in_dst_zero() {
        let k = Kernel::parse(ELEMENTWISE).unwrap();
        let a = allocate(&k).unwrap();
        assert_eq!(a.place_of(0), Place::Dst(0));
        assert_eq!(a.place_of(1), Place::Dst(0), "the unpack's value is consumed by the SFPU");
        assert_eq!(a.dst_peak, 1);
    }

    #[test]
    fn the_three_core_hand_offs_are_inserted() {
        let k = Kernel::parse(ELEMENTWISE).unwrap();
        let a = allocate(&k).unwrap();
        let kinds: Vec<(Role, SyncKind)> = a.sync.iter().map(|s| (s.role, s.kind)).collect();
        assert!(kinds.contains(&(Role::Unpack, SyncKind::Post)));
        assert!(kinds.contains(&(Role::Math, SyncKind::Wait { before: 0 })));
        assert!(kinds.contains(&(Role::Math, SyncKind::Post)));
        assert!(kinds.contains(&(Role::Pack, SyncKind::Wait { before: 0 })));
    }

    #[test]
    fn a_matrix_operand_lands_in_src() {
        // %0 and %1 feed the multiply: A side and B side.
        let text = "\
version 1
grid 1 1
param in f32 8192
buf 0 4096 plain rm
buf 1 4096 plain rm
buf 2 4096 plain rm
copy_in param 0 offset 0 len 4096 buf 0
copy_in param 0 offset 4096 len 4096 buf 1
%0 = unpack buf 0
%1 = unpack buf 1
%2 = matmul %0 %1
pack %2 buf 2
";
        let k = Kernel::parse(text).unwrap();
        let a = allocate(&k).unwrap();
        assert_eq!(a.place_of(0), Place::SrcA(0));
        assert_eq!(a.place_of(1), Place::SrcB(0));
        assert!(matches!(a.place_of(2), Place::Dst(_)));
    }

    #[test]
    fn nine_live_dst_values_are_refused() {
        let mut text = String::from("version 1\ngrid 1 1\nparam in f32 4096\nparam out f32 4096\n");
        for b in 0..9 {
            text.push_str(&format!("buf {b} 4096 plain rm\n"));
        }
        text.push_str("buf 9 4096 plain rm\n");
        for b in 0..9 {
            text.push_str(&format!("copy_in param 0 offset 0 len 4096 buf {b}\n"));
        }
        for b in 0..9 {
            text.push_str(&format!("%{b} = unpack buf {b}\n"));
        }
        // Consume them only at the end, so all nine are live at once.
        for b in 0..9 {
            text.push_str(&format!("%{} = sfpu neg %{b}\n", 9 + b));
        }
        for b in 0..9 {
            text.push_str(&format!("pack %{} buf 9\n", 9 + b));
        }
        let k = Kernel::parse(&text).unwrap();
        let err = allocate(&k).expect_err("nine live values");
        assert!(err.to_string().contains("Dst slot"), "{err}");
    }

    #[test]
    fn a_value_needed_on_both_src_sides_is_refused() {
        let text = "\
version 1
grid 1 1
param in f32 4096
buf 0 4096 plain rm
buf 1 4096 plain rm
copy_in param 0 offset 0 len 4096 buf 0
%0 = unpack buf 0
%1 = matmul %0 %0
pack %1 buf 1
";
        let k = Kernel::parse(text).unwrap();
        let err = allocate(&k).expect_err("both sides");
        assert!(err.to_string().contains("SrcA and SrcB"), "{err}");
    }
}
