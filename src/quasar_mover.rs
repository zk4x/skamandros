// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! A Quasar data mover's NoC program: the words a DM hart runs to move bytes.
//!
//! On Blackhole a mover pushes Tensix instructions and programs the NIU through
//! its memory-mapped register block. On Quasar a mover is an RV64 hart with no
//! Tensix queue of its own: it programs one of the overlay's command buffers
//! through RoCC instructions and fires the transaction with another
//! (`tt_isa::arch::quasar::issue`). The sequence transcribed here is tt-metal's
//! `ncrisc_noc_fast_read` / `ncrisc_noc_fast_write`
//! (`hw/inc/internal/tt-2xx/quasar/noc_nonblocking_api_v2.h:32-125`), and the
//! completion test is `ncrisc_noc_reads_flushed` / `..._nonposted_writes_sent`
//! (`noc_cmd_buf_common.h:487-509`).
//!
//! Two details are the hardware's, not a choice:
//!
//! * The **read** command buffer's return coordinate is this tile, so a read
//!   lands in local L1; the **write** buffer's source coordinate is this tile.
//!   A tile names itself as `x | y << 6` (`noc_parameters.h`'s
//!   `NOC_XY_ENCODING`).
//! * A transfer's address is `coordinate << 36 | local offset`: the low 36 bits
//!   are the address inside the target and everything above names the tile
//!   (`NOC_ADDR_COORD_SHIFT`, `noc_cmd_buf_common.h:41-46`). A DRAM channel is
//!   such a tile, which is why a DRAM read is a read from the channel's
//!   endpoint at the page's offset.

use crate::riscv::{addi, bne, lui, lw, reg, sw};
use crate::status::ttStatus;
use tt_isa::arch::quasar::issue;
use tt_isa::arch::quasar::noc::Endpoint;

/// Command-buffer register indices, `TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_*_
/// REG_OFFSET / 8` (`tt-2xx/tt-2.0.1/meta/registers/tt_rocc_accel_reg.h`).
pub mod register {
    /// `SRC_ADDR` and `SRC_COORD`: where the data comes from.
    pub const SRC_ADDR: u8 = 0x20 / 8;
    pub const SRC_COORD: u8 = 0x38 / 8;
    /// `DEST_ADDR` and `DEST_COORD`: where it goes.
    pub const DEST_ADDR: u8 = 0x40 / 8;
    pub const DEST_COORD: u8 = 0x58 / 8;
    /// `LEN_BYTES`: the transfer's size.
    pub const LEN_BYTES: u8 = 0x60 / 8;
    /// The request and response virtual channels.
    pub const REQ_VC: u8 = 0x68 / 8;
    pub const RESP_VC: u8 = 0x80 / 8;
    /// `MISC`: the transaction's kind.
    pub const MISC: u8 = (0x140 / 8) as u8;
}

/// The command buffers tt-metal keeps for each kind of transaction
/// (`OVERLAY_WR_CMD_BUF`, `OVERLAY_RD_CMD_BUF`).
pub const WRITE_BUFFER: u8 = 0;
pub const READ_BUFFER: u8 = 1;

/// `CMD_BUF_MISC_WRITE_TRANS` (`noc_cmd_buf_common.h:84`): a write
/// transaction. A read leaves `MISC` at its zero default.
pub const MISC_WRITE: u32 = 1 << 9;

/// Virtual channels (`noc_cmd_buf_common.h:53-67`).
pub const REQ_VC: u32 = 1;
pub const RESP_VC_READ: u32 = 12;
pub const RESP_VC_WRITE: u32 = 13;

/// Where a mover's program reads and writes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Target {
    /// Local L1, at a byte address.
    L1(u32),
    /// Another tile's address space, or a DRAM channel's.
    Remote(Endpoint),
}

impl Target {
    fn address(self) -> u32 {
        match self {
            Target::L1(at) => at,
            Target::Remote(endpoint) => endpoint.addr as u32,
        }
    }

    fn coordinate(self, local: u32) -> u32 {
        match self {
            Target::L1(_) => local,
            Target::Remote(endpoint) => endpoint.hi(),
        }
    }
}

/// A Quasar mover program under construction.
///
/// The scratch registers are the caller-saved pair the program owns; a mover
/// program is a leaf, so nothing else lives in them.
#[derive(Debug, Clone)]
pub struct Mover {
    /// The tile whose L1 the program reads and writes, `(x, y)`, and whose
    /// coordinate a transaction names as its local end.
    tile: (u8, u8),
    words: Vec<u32>,
}

/// The register a staged value passes through.
const SCRATCH: u8 = reg::T0;

impl Mover {
    pub fn new(tile: (u8, u8)) -> Self {
        Mover {
            tile,
            words: Vec::new(),
        }
    }

    /// Stage one command-buffer word: `li t0, value; cmdbuf_wr_reg buf, reg, t0`.
    fn stage(&mut self, buffer: u8, register: u8, value: u32) {
        self.load_immediate(value);
        self.words
            .push(issue::cmdbuf_wr_reg(buffer, register, SCRATCH));
    }

    /// `li rd, value`, in the two forms the ISA needs: a `lui`/`addi` pair for a
    /// value that does not fit `addi`'s sign-extended twelve bits, and a single
    /// `addi` from `zero` for one that does.
    fn load_immediate_into(&mut self, rd: u8, value: u32) {
        let high = value.wrapping_add(0x800) >> 12;
        let low = value as i32 - ((high as i32) << 12);
        if high == 0 {
            self.words.push(addi(rd, reg::ZERO, low));
        } else {
            self.words.push(lui(rd, high));
            self.words.push(addi(rd, rd, low));
        }
    }

    /// `li t0, value`, the scratch register's form.
    fn load_immediate(&mut self, value: u32) {
        self.load_immediate_into(SCRATCH, value);
    }

    /// Read `bytes` from `from` into local L1 at `l1`.
    ///
    /// This is `ncrisc_noc_fast_read`: the source address and its coordinate,
    /// the local destination, the length, then the issue.
    pub fn read(&mut self, from: Target, l1: u32, bytes: u32) -> Result<(), ttStatus> {
        let (coordinate, address) = match from {
            Target::Remote(endpoint) => (endpoint.hi(), endpoint.addr as u32),
            Target::L1(at) => {
                return Err(ttStatus::TT_ERROR_MOVER_PROGRAM);
            }
        };
        self.check_length(bytes)?;
        self.stage(READ_BUFFER, register::MISC, 0);
        self.stage(READ_BUFFER, register::REQ_VC, REQ_VC);
        self.stage(READ_BUFFER, register::RESP_VC, RESP_VC_READ);
        self.stage(READ_BUFFER, register::SRC_ADDR, address);
        self.stage(READ_BUFFER, register::SRC_COORD, coordinate);
        self.stage(READ_BUFFER, register::DEST_ADDR, l1);
        self.stage(READ_BUFFER, register::DEST_COORD, self.coordinate());
        self.stage(READ_BUFFER, register::LEN_BYTES, bytes);
        self.words.push(issue::cmdbuf_issue(READ_BUFFER));
        Ok(())
    }

    /// Write `bytes` from local L1 at `l1` to `to`.
    ///
    /// This is `ncrisc_noc_fast_write` with no multicast and no transaction id:
    /// `MISC` is rebuilt per transaction, the source is local, and the
    /// destination carries its coordinate.
    pub fn write(&mut self, l1: u32, to: Target, bytes: u32) -> Result<(), ttStatus> {
        let coordinate = match to {
            Target::Remote(endpoint) => endpoint.hi(),
            Target::L1(_) => return Err(ttStatus::TT_ERROR_MOVER_PROGRAM),
        };
        self.check_length(bytes)?;
        self.stage(WRITE_BUFFER, register::MISC, MISC_WRITE);
        self.stage(WRITE_BUFFER, register::REQ_VC, REQ_VC);
        self.stage(WRITE_BUFFER, register::RESP_VC, RESP_VC_WRITE);
        self.stage(WRITE_BUFFER, register::SRC_ADDR, l1);
        self.stage(WRITE_BUFFER, register::SRC_COORD, self.coordinate());
        self.stage(WRITE_BUFFER, register::DEST_ADDR, to.address());
        self.stage(WRITE_BUFFER, register::DEST_COORD, coordinate);
        self.stage(WRITE_BUFFER, register::LEN_BYTES, bytes);
        self.words.push(issue::cmdbuf_issue(WRITE_BUFFER));
        Ok(())
    }

    /// The largest transfer one issue moves:
    /// `NOC_OVERLAY_MAX_BYTES_IN_PACKET` (`noc_cmd_buf_common.h:74`).
    pub const MAX_BYTES: u32 = tt_isa::arch::quasar::noc::MAX_REQUEST_BYTES;

    fn check_length(&self, bytes: u32) -> Result<(), ttStatus> {
        if bytes == 0 || bytes > Self::MAX_BYTES || bytes % 4 != 0 {
            return Err(ttStatus::TT_ERROR_MOVER_PROGRAM);
        }
        Ok(())
    }

    /// Spin until every transaction this tile issued has been acknowledged.
    ///
    /// This is `ncrisc_noc_full_sync`'s read half plus the write half:
    /// `scmdbuf_wr_sent` counts writes not yet read out of L1 and `scmdbuf_tr_ack`
    /// counts transactions not yet acknowledged; both reaching zero is the
    /// flush (`noc_cmd_buf_common.h:487-509`).
    pub fn await_quiet(&mut self) {
        self.poll(issue::scmdbuf_wr_sent);
        self.poll(issue::scmdbuf_tr_ack);
    }

    /// Spin while `status` leaves its count in `T1`.
    fn poll(&mut self, status: fn(u8) -> u32) {
        let at = self.words.len();
        self.words.push(status(reg::T1));
        let distance = -(((self.words.len() - at) as i32) * 4);
        self.words.push(bne(reg::T1, reg::ZERO, distance));
    }

    /// Write one word of local L1.
    pub fn store(&mut self, l1: u32, value: u32) {
        self.load_immediate_into(reg::T1, value);
        self.load_immediate_into(SCRATCH, l1);
        self.words.push(sw(reg::T1, SCRATCH, 0));
    }

    /// Publish the launch generation: copy the word at `generation_at` to
    /// `ack_at`, which is how a Quasar hart tells the host it has finished
    /// ([`crate::device::Device::quasar_await`]).
    pub fn acknowledge(&mut self, generation_at: u32, ack_at: u32) {
        self.load_immediate_into(SCRATCH, generation_at);
        self.words.push(lw(reg::T1, SCRATCH, 0));
        self.load_immediate_into(SCRATCH, ack_at);
        self.words.push(sw(reg::T1, SCRATCH, 0));
    }

    /// Loop forever: the hart stays live for the host to inspect.
    pub fn spin(&mut self) {
        self.words.push(crate::riscv::jal(reg::ZERO, 0));
    }

    /// The program, as RV64 words.
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    pub fn into_words(self) -> Vec<u32> {
        self.words
    }

    /// This tile's packed coordinate, for the local half of a transaction.
    fn coordinate(&self) -> u32 {
        u32::from(self.tile.0) | (u32::from(self.tile.1) << 6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_read_stages_its_registers_then_issues() {
        let mut mover = Mover::new((2, 2));
        let dram = Endpoint {
            x: 2,
            y: 7,
            addr: 0x100,
        };
        mover.read(Target::Remote(dram), 0x2000, 64).unwrap();
        let words = mover.words();
        // Every staged register leaves a `li` (one or two words) and a RoCC
        // write; the transaction ends with the issue.
        assert_eq!(
            *words.last().unwrap(),
            issue::cmdbuf_issue(READ_BUFFER),
            "the transfer ends with its issue"
        );
        assert!(
            words
                .windows(1)
                .any(|w| w[0] == issue::cmdbuf_wr_reg(READ_BUFFER, register::SRC_COORD, SCRATCH))
        );
        assert!(
            words
                .windows(1)
                .any(|w| w[0] == issue::cmdbuf_wr_reg(READ_BUFFER, register::DEST_COORD, SCRATCH))
        );
    }

    #[test]
    fn a_transfer_refuses_an_empty_or_oversized_length() {
        let mut mover = Mover::new((2, 2));
        let endpoint = Endpoint {
            x: 2,
            y: 7,
            addr: 0,
        };
        assert_eq!(
            mover.read(Target::Remote(endpoint), 0x1000, 0),
            Err(ttStatus::TT_ERROR_MOVER_PROGRAM)
        );
        assert_eq!(
            mover.read(Target::Remote(endpoint), 0x1000, Mover::MAX_BYTES + 4),
            Err(ttStatus::TT_ERROR_MOVER_PROGRAM)
        );
        assert_eq!(
            mover.read(Target::Remote(endpoint), 0x1000, 6),
            Err(ttStatus::TT_ERROR_MOVER_PROGRAM)
        );
    }
}
