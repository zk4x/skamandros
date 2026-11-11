// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! L1 placement: the caller names buffers, the driver gives them addresses.
//!
//! Every buffer is 64-byte aligned, and every copy uses the same datum offset on
//! its DRAM side and its L1 side. The NoC requires the two addresses of a
//! request to be congruent, so equal alignment on both sides makes every copy
//! congruent by construction, for a literal offset and for a computed one alike.
//! That is why no residue has to be planned per buffer here.
//!
//! How a buffer's datums are laid out in L1 is not a property of the program.
//! The compiler stores a unit in the order the unpacker and packer use (four
//! 16x16 faces for a tile, contiguous datums for a vector), and both ends of the
//! datapath are configured from that.

use crate::ir::{Error, Kernel, Op, Types, unit_of};

/// Where each declared buffer lives in L1, by value id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub buf: Vec<Option<u32>>,
}

const ALIGN: u32 = 64;

/// Place every buffer in `tt_isa::l1::DATA`, in declaration order.
pub fn place(k: &Kernel, types: &Types) -> Result<Plan, Error> {
    let mut buf = vec![None; types.values.len()];
    let mut at = tt_isa::l1::DATA.base as u32;
    for (op_at, op) in k.ops.iter().enumerate() {
        let Op::Buffer {
            value,
            dtype,
            shape,
            ..
        } = *op
        else {
            continue;
        };
        let (_, _, bytes) = unit_of(shape, dtype)?;
        let base = at.div_ceil(ALIGN) * ALIGN;
        let end = base.checked_add(bytes).ok_or_else(|| {
            Error(format!(
                "op {op_at}: buffer {value} overflows the L1 address space"
            ))
        })?;
        if u64::from(end) > tt_isa::l1::DATA.end {
            return Err(Error(format!(
                "op {op_at}: buffer {value} ({base:#x}..{end:#x}) runs past the L1 data arena ({:#x})",
                tt_isa::l1::DATA.end
            )));
        }
        buf[value as usize] = Some(base);
        at = end;
    }
    Ok(Plan { buf })
}
