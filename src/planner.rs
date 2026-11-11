// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! L1 placement: the caller names buffers, the driver gives them addresses.
//!
//! A DRAM read needs the two addresses congruent mod 64 and a write mod 16
//! (`tt_isa::noc::niu::Command`'s alignment check). The DRAM side of a copy is
//! the parameter's range base (a driver allocation, 64-byte aligned) plus the
//! copy's `offset`; so the L1 buffer's address carries `offset mod 64`, and a
//! buffer fed by two copies with different residues is refused rather than
//! silently misaligned.

use crate::ir::{BufKind, Error, Kernel, Layout, Op};

/// Where each declared buffer lives in L1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// Byte address in L1 per buffer id.
    pub buf: Vec<u32>,
}

const ALIGN: u32 = 64;

/// Place every buffer in `tt_isa::l1::DATA`.
pub fn place(k: &Kernel) -> Result<Plan, Error> {
    let mut read64: Vec<Option<u32>> = vec![None; k.bufs.len()];
    let mut write16: Vec<Option<u32>> = vec![None; k.bufs.len()];
    for (at, op) in k.ops.iter().enumerate() {
        match *op {
            Op::CopyIn { offset, buf, .. } => {
                let want = offset % ALIGN;
                match read64[buf as usize] {
                    None => read64[buf as usize] = Some(want),
                    Some(seen) if seen != want => {
                        return Err(Error(format!(
                            "op {at}: buffer {buf} is read at residues {seen} and {want} mod {ALIGN}"
                        )));
                    }
                    Some(_) => {}
                }
            }
            Op::CopyOut { offset, buf, .. } => {
                let want = offset % 16;
                match write16[buf as usize] {
                    None => write16[buf as usize] = Some(want),
                    Some(seen) if seen != want => {
                        return Err(Error(format!(
                            "op {at}: buffer {buf} is written at residues {seen} and {want} mod 16"
                        )));
                    }
                    Some(_) => {}
                }
            }
            _ => {}
        }
    }
    for (id, b) in k.bufs.iter().enumerate() {
        if let (Some(r), Some(w)) = (read64[id], write16[id]) {
            if r % 16 != w {
                return Err(Error(format!(
                    "buffer {id}: read residue {r} mod {ALIGN} and write residue {w} mod 16 disagree"
                )));
            }
        }
        if matches!(b.kind, BufKind::Circular { .. }) && read64[id].is_some_and(|r| r != 0) {
            return Err(Error(format!(
                "circular buffer {id}: a page move needs a {ALIGN}-byte read offset"
            )));
        }
        if let Layout::TileMajor = b.layout {
            // A tile-major buffer holds the datum blocks of whole tiles, and a
            // tile image's datums start 16 bytes into its DRAM page: every copy
            // that touches it must name that offset, so the two sides stay
            // congruent.
            for (at, op) in k.ops.iter().enumerate() {
                let offset = match *op {
                    Op::CopyIn { buf, offset, .. } | Op::CopyOut { buf, offset, .. } if buf == id as u32 => offset,
                    _ => continue,
                };
                if offset % ALIGN != tt_isa::dm::TILE_DATA as u32 % ALIGN {
                    return Err(Error(format!(
                        "op {at}: a tile-major buffer is copied from a datum block (offset ≡ {TILE_DATA} mod {ALIGN}), not {offset}",
                        TILE_DATA = tt_isa::dm::TILE_DATA,
                    )));
                }
            }
        }
    }
    let mut at = tt_isa::l1::DATA.base as u32;
    let mut buf = Vec::with_capacity(k.bufs.len());
    for (id, b) in k.bufs.iter().enumerate() {
        let want = match b.layout {
            Layout::TileMajor => tt_isa::dm::TILE_DATA as u32 % ALIGN,
            Layout::RowMajor => read64[id].or(write16[id]).unwrap_or(0),
        };
        let base = at.div_ceil(ALIGN) * ALIGN + want;
        let end = base
            .checked_add(b.bytes)
            .ok_or_else(|| Error(format!("buffer {id} overflows the L1 address space")))?;
        if u64::from(end) > tt_isa::l1::DATA.end {
            return Err(Error(format!(
                "buffer {id} ({end:#x}) runs past the L1 data arena ({:#x})",
                tt_isa::l1::DATA.end
            )));
        }
        buf.push(base);
        at = end;
    }
    Ok(Plan { buf })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_read_offset_sets_the_residue() {
        let text = "\
version 1
grid 1 1
param in f32 8192
buf 0 4096 plain rm
buf 1 4096 plain rm
copy_in param 0 offset 64 len 4096 buf 0
copy_in param 0 offset 96 len 4096 buf 1
";
        let k = Kernel::parse(text).unwrap();
        let plan = place(&k).unwrap();
        assert_eq!(plan.buf[0] % 64, 0);
        assert_eq!(plan.buf[1] % 64, 32);
        assert_ne!(plan.buf[0], plan.buf[1]);
    }

    #[test]
    fn two_residues_for_one_buffer_are_refused() {
        let text = "\
version 1
grid 1 1
param in f32 8192
buf 0 4096 plain rm
copy_in param 0 offset 0 len 4096 buf 0
copy_in param 0 offset 64 len 4096 buf 0
";
        let k = Kernel::parse(text).unwrap();
        // Both are multiples of 64 at offset 0 and 64: same residue 0.
        assert!(place(&k).is_ok());
        let text = "\
version 1
grid 1 1
param in f32 8192
buf 0 4096 plain rm
copy_in param 0 offset 0 len 4096 buf 0
copy_in param 0 offset 32 len 4096 buf 0
";
        let k = Kernel::parse(text).unwrap();
        assert!(place(&k).is_err());
    }

    #[test]
    fn a_tile_major_buffer_is_placed_at_the_datum_offset() {
        let text = "\
version 1
grid 1 1
param in f32 8192
buf 0 4096 plain tile
copy_in param 0 offset 16 len 4096 buf 0
";
        let k = Kernel::parse(text).unwrap();
        let plan = place(&k).unwrap();
        assert_eq!(plan.buf[0] % 64, tt_isa::dm::TILE_DATA as u32 % 64);
    }

    #[test]
    fn a_tile_major_buffer_copied_from_a_tile_header_is_refused() {
        let text = "\
version 1
grid 1 1
param in f32 8192
buf 0 4096 plain tile
copy_in param 0 offset 0 len 4096 buf 0
";
        let k = Kernel::parse(text).unwrap();
        let err = place(&k).expect_err("header offset");
        assert!(err.to_string().contains("datum block"), "{err}");
    }

    #[test]
    fn a_tile_major_buffer_of_partial_tiles_is_refused() {
        let text = "\
version 1
grid 1 1
param in f32 8192
buf 0 4000 plain tile
copy_in param 0 offset 16 len 4000 buf 0
";
        let err = Kernel::parse(text).expect_err("partial tile");
        assert!(err.to_string().contains("whole 4096-byte tiles"), "{err}");
    }

    #[test]
    fn the_arena_end_is_enforced() {
        let mut text = String::from("version 1\ngrid 1 1\nparam in f32 100000000\n");
        text.push_str("buf 0 100000000 plain rm\ncopy_in param 0 offset 0 len 4096 buf 0\n");
        let k = Kernel::parse(&text).unwrap();
        let err = place(&k).expect_err("does not fit");
        assert!(err.to_string().contains("data arena"), "{err}");
    }
}
