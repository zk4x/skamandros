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

/// One circular buffer's L1 ring: where it starts, how big a page is, and how
/// many pages fit. tt-metal's `CB` is described the same three ways; the ring
/// depth is the whole declared buffer, so a second copy before a pop has a page
/// to land in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cb {
    /// Byte address of the first page.
    pub base: u32,
    /// Bytes in one page: one `load`/`store_to` unit.
    pub page: u32,
    /// Pages in the ring: `size / page`.
    pub pages: u32,
}

/// The compiler's 1.0 scaler page: a reduce's `max` needs a `SrcB` row of
/// exactly 1.0, which no caller page provides, so the compiler owns one 32x32
/// page of ones and packs it once per kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnesPage {
    /// The CB id of the page's one-page ring.
    pub cb: u32,
    /// The ring itself.
    pub ring: Cb,
    /// The datum format the page was sized and packed in: the operand's.
    pub dtype: crate::ir::DType,
}

/// Where each declared buffer lives in L1, by value id, and the CB interface
/// table the host writes before the launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub buf: Vec<Option<u32>>,
    /// The CB id of each buffer value, by value id.
    pub cb: Vec<Option<u32>>,
    /// The rings, in cb-id order.
    pub cbs: Vec<Cb>,
    /// Byte address of the `LocalCBInterface` table (`riscv::cb::COUNT` entries,
    /// 32 bytes each).
    pub iface: u32,
    /// The compiler's 1.0 page, when a reduce needs one.
    pub ones: Option<OnesPage>,
}

const ALIGN: u32 = 64;

/// Place every buffer in `tt_isa::l1::DATA`, in declaration order, and the CB
/// interface table after them.
pub fn place(k: &Kernel, types: &Types) -> Result<Plan, Error> {
    let mut buf = vec![None; types.values.len()];
    let mut cb = vec![None; types.values.len()];
    let mut cbs: Vec<Cb> = Vec::new();
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
        let (unit, units, bytes) = unit_of(shape, dtype)?;
        let base = at.div_ceil(ALIGN) * ALIGN;
        let end = base.checked_add(bytes).ok_or_else(|| {
            let _ = (op_at, value);
            Error::TT_ERROR_L1_OVERFLOW
        })?;
        if u64::from(end) > tt_isa::l1::DATA.end {
            let _ = (op_at, value, base, end);
            return Err(Error::TT_ERROR_L1_ARENA);
        }
        let page = unit
            .datums()
            .checked_mul(dtype.bytes())
            .ok_or(Error::TT_ERROR_L1_OVERFLOW)?;
        buf[value as usize] = Some(base);
        cb[value as usize] = Some(cbs.len() as u32);
        cbs.push(Cb {
            base,
            page,
            pages: units,
        });
        at = end;
    }
    // A `max` reduce's `SrcB` must be a row of exactly 1.0, and the caller
    // names no scaler for it: the compiler owns one 32x32 page of ones, in the
    // datum format of the operand.
    let ones_dtype = k.ops.iter().find_map(|op| match *op {
        Op::Max { x, .. } => Some(types.of(x).dtype),
        _ => None,
    });
    let mut ones = None;
    if let Some(dtype) = ones_dtype {
        let bytes = 1024u32
            .checked_mul(dtype.bytes())
            .ok_or(Error::TT_ERROR_L1_OVERFLOW)?;
        let base = at.div_ceil(ALIGN) * ALIGN;
        let end = base.checked_add(bytes).ok_or(Error::TT_ERROR_L1_OVERFLOW)?;
        if u64::from(end) > tt_isa::l1::DATA.end {
            let _ = (base, end);
            return Err(Error::TT_ERROR_L1_ARENA);
        }
        let cb = cbs.len() as u32;
        let ring = Cb {
            base,
            page: bytes,
            pages: 1,
        };
        ones = Some(OnesPage { cb, ring, dtype });
        cbs.push(ring);
        at = end;
    }
    // The interface table follows the buffers: one `LocalCBInterface` per CB, as
    // many entries as tt-metal's table has, and only `cbs.len()` of them used.
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
    Ok(Plan {
        buf,
        cb,
        cbs,
        iface,
        ones,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Kernel;

    const TWO: &str = "\
version 7
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,64
%bo = buffer f32 32,32
copy_to %bi from %in [0]
copy_to %out from %bo [0]
grid 1,1
";

    /// A buffer is a ring of whole units: `32,64` is two 4 KiB pages, `32,32`
    /// one; the interface table follows the last buffer, 64-byte aligned, and
    /// every buffer value has a cb id.
    #[test]
    fn buffers_become_rings_and_the_table_follows_them() {
        let k = Kernel::parse(TWO).expect("the program parses");
        let types = k.verify().expect("it verifies");
        let plan = place(&k, &types).expect("it places");
        assert_eq!(plan.cbs.len(), 2);
        assert_eq!(plan.cbs[0].page, 4096);
        assert_eq!(plan.cbs[0].pages, 2);
        assert_eq!(plan.cbs[1].page, 4096);
        assert_eq!(plan.cbs[1].pages, 1);
        assert_eq!(plan.cbs[0].base + 2 * 4096, plan.cbs[1].base);
        assert_eq!(plan.iface, plan.cbs[1].base + 4096);
        assert_eq!(plan.iface % 64, 0);
        // `%bi` is value 2, `%bo` value 3 in declaration order.
        assert_eq!(plan.cb[2], Some(0));
        assert_eq!(plan.cb[3], Some(1));
    }
}
