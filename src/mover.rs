// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The boundary programs: DRAM <-> L1 over the NoC.
//!
//! Each copy in the IR becomes one NOC request: the ten initiator registers are
//! written in `tt_isa`'s order, `CMD_CTRL` triggers it, and the program waits on
//! that word and then on the transaction ID's outstanding counter before the
//! next copy. The parameter's address is its launch argument plus the copy's
//! offset; the local address is the planned L1 buffer.
//!
//! `TARG_ADDR` is the request's source and `RET_ADDR` its destination, so a
//! read names the parameter as target and L1 as return, and a write the other
//! way round. Both programs are core-agnostic RISC-V; the runtime stages them on
//! RISCV B through the `exec_b` trampoline, the reader first (the program's own
//! order for a kernel with no compute).

use crate::ir::{Error, Kernel, Op};
use crate::planner::Plan;
use crate::riscv::{self as rv, Asm, reg};
use tt_isa::noc::niu::{CMD_RD, CMD_WR, Niu, RESP_MARKED, STATIC_VC_1, TxnId, initiator, reqs_outstanding};

/// Words of launch arguments before the per-parameter pairs: grid x, grid y,
/// reserved, this tile's NOC high word.
pub const ARG_HEAD: usize = 4;

/// Transaction ID every request uses. The programs issue in order and drain
/// after each copy, so one ID is enough.
const TXN: TxnId = match TxnId::new(1) {
    Some(t) => t,
    None => panic!("transaction 1 is valid"),
};

/// The launch argument words: one `(low offset, endpoint high word)` pair per
/// parameter, after [`ARG_HEAD`].
pub const fn param_arg(param: u32) -> (i32, i32) {
    let lo = (ARG_HEAD + 2 * param as usize) * 4;
    (lo as i32, lo as i32 + 4)
}

/// Emit the reader program: every `CopyIn`, in IR order.
pub fn reader(k: &Kernel, plan: &Plan) -> Result<Vec<u32>, Error> {
    let mut a = Asm::new();
    prologue(&mut a);
    let mut any = false;
    for (at, op) in k.ops.iter().enumerate() {
        if let Op::CopyIn { param, offset, buf, bytes } = *op {
            emit_copy(&mut a, at, param, offset, plan.buf[buf as usize], bytes, true)?;
            any = true;
        }
    }
    if !any {
        return Err(Error("the kernel has no copy into L1".into()));
    }
    a.word(rv::ret());
    a.finish().map_err(Error)
}

/// Emit the writer program: every `CopyOut`, in IR order.
pub fn writer(k: &Kernel, plan: &Plan) -> Result<Vec<u32>, Error> {
    let mut a = Asm::new();
    prologue(&mut a);
    let mut any = false;
    for (at, op) in k.ops.iter().enumerate() {
        if let Op::CopyOut { buf, param, offset, bytes } = *op {
            emit_copy(&mut a, at, param, offset, plan.buf[buf as usize], bytes, false)?;
            any = true;
        }
    }
    if !any {
        return Err(Error("the kernel has no copy out of L1".into()));
    }
    a.word(rv::ret());
    a.finish().map_err(Error)
}

/// `t5` = this NIU's base, `t3` = this tile's NOC high word (the launch's third
/// argument pair, read from `a0`).
fn prologue(a: &mut Asm) {
    a.li(reg::T5, Niu::Noc0.base() as u32);
    a.word(rv::lw(reg::T3, reg::A0, 12));
}

fn emit_copy(a: &mut Asm, at: usize, param: u32, offset: u32, l1: u32, bytes: u32, read: bool) -> Result<(), Error> {
    if bytes == 0 || bytes > tt_isa::noc::niu::MAX_REQUEST_BYTES {
        return Err(Error(format!(
            "op {at}: {bytes} bytes is not one NOC request (1..={})",
            tt_isa::noc::niu::MAX_REQUEST_BYTES
        )));
    }
    let (lo_arg, hi_arg) = param_arg(param);
    // Target: the parameter's range base plus the copy's offset.
    a.word(rv::lw(reg::T1, reg::A0, lo_arg));
    if offset != 0 {
        a.li(reg::T4, offset);
        a.word(rv::add(reg::T1, reg::T1, reg::T4));
    }
    a.word(rv::lw(reg::T2, reg::A0, hi_arg));
    // `TARG_ADDR` is the request's source and `RET_ADDR` its destination: a
    // read has the parameter as source and L1 as destination, a write the
    // other way round.
    a.li(reg::T4, l1);
    if read {
        store_reg(a, initiator::TARG_ADDR_LO, reg::T1);
        store_zero(a, initiator::TARG_ADDR_MID);
        store_reg(a, initiator::TARG_ADDR_HI, reg::T2);
        store_reg(a, initiator::RET_ADDR_LO, reg::T4);
        store_zero(a, initiator::RET_ADDR_MID);
        store_reg(a, initiator::RET_ADDR_HI, reg::T3);
    } else {
        store_reg(a, initiator::TARG_ADDR_LO, reg::T4);
        store_zero(a, initiator::TARG_ADDR_MID);
        store_reg(a, initiator::TARG_ADDR_HI, reg::T3);
        store_reg(a, initiator::RET_ADDR_LO, reg::T1);
        store_zero(a, initiator::RET_ADDR_MID);
        store_reg(a, initiator::RET_ADDR_HI, reg::T2);
    }
    a.li(reg::T4, (TXN.index() as u32) << 10);
    store_reg(a, initiator::PACKET_TAG, reg::T4);
    let ctrl = if read { CMD_RD } else { CMD_WR } | RESP_MARKED | STATIC_VC_1;
    a.li(reg::T4, ctrl);
    store_reg(a, initiator::CTRL, reg::T4);
    a.li(reg::T4, bytes);
    store_reg(a, initiator::AT_LEN_BE, reg::T4);
    store_zero(a, initiator::AT_DATA);
    // Trigger, then wait until the hardware takes the request.
    let one = a.word(rv::addi(reg::T4, reg::ZERO, 1));
    store_reg(a, initiator::CMD_CTRL, reg::T4);
    let _ = one;
    let poll = format!("poll{at}");
    a.label(&poll);
    a.word(rv::lw(reg::T6, reg::T5, initiator::CMD_CTRL as i32));
    a.bne_label(reg::T6, reg::ZERO, poll.clone());
    // Then wait for the transaction to complete: read the trigger word back
    // first (`Counters.md`), then poll the ID's outstanding counter.
    a.word(rv::lw(reg::T6, reg::T5, initiator::CMD_CTRL as i32));
    let drain = format!("drain{at}");
    a.label(&drain);
    a.word(rv::lw(reg::T6, reg::T5, reqs_outstanding(TXN) as i32));
    a.bne_label(reg::T6, reg::ZERO, drain);
    Ok(())
}

fn store_reg(a: &mut Asm, at: u64, src: u8) {
    a.word(rv::sw(src, reg::T5, at as i32));
}

fn store_zero(a: &mut Asm, at: u64) {
    a.word(rv::sw(reg::ZERO, reg::T5, at as i32));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Kernel;

    const COPY: &str = "\
version 1
grid 1 1
param in f32 8192
param out f32 8192
buf 0 4096 plain rm
buf 1 4096 plain rm
copy_in param 0 offset 0 len 4096 buf 0
copy_out buf 1 param 1 offset 64 len 4096
";

    #[test]
    fn the_reader_program_ends_in_ret() {
        let k = Kernel::parse(COPY).unwrap();
        let plan = crate::planner::place(&k).unwrap();
        let words = reader(&k, &plan).unwrap();
        assert_eq!(*words.last().unwrap(), rv::ret());
    }

    #[test]
    fn the_writer_program_ends_in_ret() {
        let k = Kernel::parse(COPY).unwrap();
        let plan = crate::planner::place(&k).unwrap();
        let words = writer(&k, &plan).unwrap();
        assert_eq!(*words.last().unwrap(), rv::ret());
    }

    #[test]
    fn the_reader_issues_a_read_and_drains() {
        let k = Kernel::parse(COPY).unwrap();
        let plan = crate::planner::place(&k).unwrap();
        let words = reader(&k, &plan).unwrap();
        // The NIU base, the control word and the drain counter are all present
        // as constants formed by `lui`/`addi`.
        let ctrl = CMD_RD | RESP_MARKED | STATIC_VC_1;
        assert!(has_const(&words, ctrl), "the control word is issued");
        assert!(has_const(&words, 4096), "the request length is 4096");
        assert!(
            has_lw_imm(&words, reqs_outstanding(TXN) as i32),
            "the drain polls the counter"
        );
        assert!(has_const(&words, Niu::Noc0.base() as u32), "the NIU base is materialised");
        // Two backward branches: the trigger poll and the drain.
        let branches = words.iter().filter(|w| (**w & 0x7f) == 0x63).count();
        assert_eq!(branches, 2, "one poll for the trigger and one for the drain");
    }

    #[test]
    fn the_writer_issues_a_write() {
        let k = Kernel::parse(COPY).unwrap();
        let plan = crate::planner::place(&k).unwrap();
        let words = writer(&k, &plan).unwrap();
        let ctrl = CMD_WR | RESP_MARKED | STATIC_VC_1;
        assert!(has_const(&words, ctrl), "the write control word is issued");
    }

    /// Is there an `lw` with this (signed) immediate?
    fn has_lw_imm(words: &[u32], imm: i32) -> bool {
        words.iter().any(|w| (w & 0x7f) == 0x03 && ((w >> 12) & 7) == 2 && ((*w as i32) >> 20) == imm)
    }

    /// Is `value` formed by an adjacent `lui`/`addi` pair (what `li` emits)?
    fn has_const(words: &[u32], value: u32) -> bool {
        for i in 0..words.len().saturating_sub(1) {
            let (w, n) = (words[i], words[i + 1]);
            let is_lui = (w & 0x7f) == 0x37;
            let is_addi = (n & 0x7f) == 0x13 && ((n >> 12) & 7) == 0;
            if is_lui && is_addi && ((n >> 15) & 0x1f) == ((w >> 7) & 0x1f) {
                let total = (w & 0xffff_f000).wrapping_add(((n as i32) >> 20) as u32);
                if total == value {
                    return true;
                }
            }
        }
        false
    }
}
