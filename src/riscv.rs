// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! RV32IM encoders and a tiny assembler.
//!
//! The mover programs are real RISC-V: loops are backward branches, addresses
//! are computed in registers. Values that fit `u32` are materialised with
//! `lui`/`addi`; the assembler patches branch immediates at `finish`.

/// Register numbers of the caller-saved set the programs use. A program is a
/// leaf: no `sp`, `gp`, `tp`, `ra`, `s0..s11`.
pub mod reg {
    pub const ZERO: u8 = 0;
    pub const A0: u8 = 10;
    pub const A1: u8 = 11;
    pub const A2: u8 = 12;
    pub const A3: u8 = 13;
    pub const A4: u8 = 14;
    pub const A5: u8 = 15;
    pub const A6: u8 = 16;
    pub const T0: u8 = 5;
    pub const T1: u8 = 6;
    pub const T2: u8 = 7;
    pub const T3: u8 = 28;
    pub const T4: u8 = 29;
    pub const T5: u8 = 30;
    pub const T6: u8 = 31;
}

fn r_type(funct7: u32, rs2: u8, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
    (funct7 << 25)
        | (u32::from(rs2) << 20)
        | (u32::from(rs1) << 15)
        | (funct3 << 12)
        | (u32::from(rd) << 7)
        | opcode
}

fn i_type(imm: i32, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
    let imm = (imm as u32) & 0xfff;
    (imm << 20) | (u32::from(rs1) << 15) | (funct3 << 12) | (u32::from(rd) << 7) | opcode
}

fn s_type(imm: i32, rs2: u8, rs1: u8, funct3: u32, opcode: u32) -> u32 {
    let imm = imm as u32;
    ((imm >> 5) & 0x7f) << 25
        | (u32::from(rs2) << 20)
        | (u32::from(rs1) << 15)
        | (funct3 << 12)
        | ((imm & 0x1f) << 7)
        | opcode
}

fn b_type(imm: i32, rs2: u8, rs1: u8, funct3: u32, opcode: u32) -> u32 {
    let imm = imm as u32;
    (((imm >> 12) & 1) << 31)
        | (((imm >> 5) & 0x3f) << 25)
        | (u32::from(rs2) << 20)
        | (u32::from(rs1) << 15)
        | (funct3 << 12)
        | (((imm >> 1) & 0xf) << 8)
        | (((imm >> 11) & 1) << 7)
        | opcode
}

fn u_type(imm20: u32, rd: u8, opcode: u32) -> u32 {
    (imm20 << 12) | (u32::from(rd) << 7) | opcode
}

pub fn lui(rd: u8, imm20: u32) -> u32 {
    u_type(imm20, rd, 0x37)
}

pub fn addi(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(imm, rs1, 0, rd, 0x13)
}

pub fn ori(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(imm, rs1, 6, rd, 0x13)
}

pub fn slli(rd: u8, rs1: u8, shamt: u32) -> u32 {
    r_type(0, shamt as u8, rs1, 1, rd, 0x13)
}

pub fn srli(rd: u8, rs1: u8, shamt: u32) -> u32 {
    r_type(0, shamt as u8, rs1, 5, rd, 0x13)
}

pub fn add(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0, rs2, rs1, 0, rd, 0x33)
}

pub fn sub(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x20, rs2, rs1, 0, rd, 0x33)
}

pub fn and(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0, rs2, rs1, 7, rd, 0x33)
}

pub fn or(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0, rs2, rs1, 6, rd, 0x33)
}

pub fn mul(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(1, rs2, rs1, 0, rd, 0x33)
}

pub fn div(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(1, rs2, rs1, 4, rd, 0x33)
}

pub fn rem(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(1, rs2, rs1, 6, rd, 0x33)
}

pub fn xor(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0, rs2, rs1, 4, rd, 0x33)
}

pub fn sll(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0, rs2, rs1, 1, rd, 0x33)
}

pub fn sra(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x20, rs2, rs1, 5, rd, 0x33)
}

pub fn lw(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(imm, rs1, 2, rd, 0x03)
}

pub fn lhu(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(imm, rs1, 5, rd, 0x03)
}

pub fn sw(rs2: u8, rs1: u8, imm: i32) -> u32 {
    s_type(imm, rs2, rs1, 2, 0x23)
}

pub fn sh(rs2: u8, rs1: u8, imm: i32) -> u32 {
    s_type(imm, rs2, rs1, 1, 0x23)
}

pub fn beq(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(imm, rs2, rs1, 0, 0x63)
}

pub fn bne(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(imm, rs2, rs1, 1, 0x63)
}

/// `ret`: `jalr x0, 0(x1)`.
pub fn ret() -> u32 {
    0x0000_8067
}

/// Block this core until the Tensix coprocessor has finished every instruction
/// the thread has pushed, so that what it wrote to L1 is visible to this core
/// and to the NoC (`ManualTTSync.md`).
///
/// A pushed `STALLWAIT` does not do this for a core store: it holds back later
/// *pushed* instructions, and the core runs straight on to its next `sw`/`sh`.
/// The circular-buffer push after a pack is such a store, and the consumer on
/// another core polls that counter, so the push has to wait here.
///
/// The sequence is the firmware's: start the manual sync with a store, then the
/// load that blocks until it completes, consumed by an `and` so it cannot be
/// dropped (`ManualTTSync.md`; no other load in the PC buffer range may start
/// until this one finishes).
pub fn coprocessor_done(a: &mut Asm) {
    a.li(reg::T0, tt_isa::tensix::COPROCESSOR_DONE_CHECK as u32);
    a.word(sw(reg::ZERO, reg::T0, 0));
    a.word(lw(reg::T1, reg::T0, 0));
    a.word(and(reg::T1, reg::T1, reg::ZERO));
}

/// Inline circular-buffer sync: the four tt-metal `dataflow_api.h` operations as
/// straight-line RISC-V over a [`LocalCBInterface`] table in L1.
///
/// The layout is tt-metal's `LocalCBInterface` (`circular_buffer_interface.h`),
/// 8 words: `fifo_size`, `fifo_limit`, `fifo_page_size`, `fifo_num_pages`,
/// `fifo_rd_ptr`, `fifo_wr_ptr`, `tiles_acked_received` (acked low 16,
/// received high 16), `fifo_wr_tile_ptr`. `reserve_back` and `wait_front` poll
/// and do not advance a pointer; `push_back`/`pop_front` advance the pointer and
/// its counter, wrapping only at `fifo_limit`.
///
/// Unlike tt-metal's `cb_addr_shift = 4`, the interface here holds **bytes**:
/// the host writes it and only these programs read it, so the 16-byte-unit
/// encoding buys nothing. The unpack/pack base conversion shifts at its use site.
pub mod cb {
    use super::*;

    /// Byte offsets of the `LocalCBInterface` fields.
    pub const SIZE: i32 = 0;
    pub const LIMIT: i32 = 4;
    pub const PAGE_SIZE: i32 = 8;
    pub const NUM_PAGES: i32 = 12;
    pub const RD_PTR: i32 = 16;
    pub const WR_PTR: i32 = 20;
    /// `tiles_acked` (low 16) and `tiles_received` (high 16) share this word.
    pub const COUNTERS: i32 = 24;
    /// Bytes per interface entry.
    pub const STRIDE: u32 = 32;
    /// Entries in the table: tt-metal's `NUM_CIRCULAR_BUFFERS`.
    pub const COUNT: u32 = 32;

    fn base_of(a: &mut Asm, base: u8, cb: u32, out: u8) {
        assert!(cb < COUNT, "cb id {cb} out of range");
        a.word(addi(out, base, (cb * STRIDE) as i32));
    }

    /// `cb_reserve_back(cb, n)`: poll until `num_pages - (received - acked) >= n`.
    pub fn reserve_back(a: &mut Asm, base: u8, cb: u32, n: u32, s: [u8; 3]) {
        let [p, t, u] = s;
        base_of(a, base, cb, p);
        let top = a.fresh_label();
        a.label(top.clone());
        a.word(lhu(u, p, COUNTERS + 2));
        a.word(lhu(t, p, COUNTERS));
        a.word(sub(u, u, t));
        a.word(lw(t, p, NUM_PAGES));
        a.word(sub(t, t, u));
        a.li(u, n);
        a.bltu_label(t, u, top);
    }

    /// `cb_wait_front(cb, n)`: poll until `received - acked >= n`.
    pub fn wait_front(a: &mut Asm, base: u8, cb: u32, n: u32, s: [u8; 3]) {
        let [p, t, u] = s;
        base_of(a, base, cb, p);
        let top = a.fresh_label();
        a.label(top.clone());
        a.word(lhu(u, p, COUNTERS + 2));
        a.word(lhu(t, p, COUNTERS));
        a.word(sub(u, u, t));
        a.li(t, n);
        a.bltu_label(u, t, top);
    }

    /// `cb_push_back(cb, n)`: `received += n`, advance the write pointer.
    pub fn push_back(a: &mut Asm, base: u8, cb: u32, n: u32, s: [u8; 4]) {
        let [p, t, u, v] = s;
        base_of(a, base, cb, p);
        a.li(t, n);
        a.word(lhu(u, p, COUNTERS + 2));
        a.word(add(u, u, t));
        a.word(sh(u, p, COUNTERS + 2));
        advance(a, p, WR_PTR, t, v);
    }

    /// `cb_pop_front(cb, n)`: `acked += n`, advance the read pointer.
    pub fn pop_front(a: &mut Asm, base: u8, cb: u32, n: u32, s: [u8; 4]) {
        let [p, t, u, v] = s;
        base_of(a, base, cb, p);
        a.li(t, n);
        a.word(lhu(u, p, COUNTERS));
        a.word(add(u, u, t));
        a.word(sh(u, p, COUNTERS));
        advance(a, p, RD_PTR, t, v);
    }

    /// `*ptr += n * page_size`, wrapping at `fifo_limit` (tt-metal: the pointer
    /// never passes the limit; reaching it resets by `fifo_size`).
    fn advance(a: &mut Asm, p: u8, ptr: i32, n: u8, scratch: u8) {
        a.word(lw(scratch, p, PAGE_SIZE));
        a.word(mul(scratch, scratch, n));
        a.word(lw(n, p, ptr));
        a.word(add(n, n, scratch));
        a.word(sw(n, p, ptr));
        a.word(lw(scratch, p, LIMIT));
        let done = a.fresh_label();
        a.bne_label(n, scratch, done.clone());
        a.word(lw(scratch, p, SIZE));
        a.word(sub(n, n, scratch));
        a.word(sw(n, p, ptr));
        a.label(done);
    }

    /// The interface words, ready to be written into L1.
    pub fn interface(fifo_addr: u32, size: u32, page_size: u32, num_pages: u32) -> [u32; 8] {
        assert!(
            size > 0 && page_size > 0 && size % page_size == 0,
            "cb interface: size {size} / page {page_size}"
        );
        [
            size,
            fifo_addr + size,
            page_size,
            num_pages,
            fifo_addr,
            fifo_addr,
            0,
            0,
        ]
    }
}

/// A program under construction: words, labels, and the branch fixups that
/// name them.
#[derive(Default)]
pub struct Asm {
    words: Vec<u32>,
    labels: Vec<(String, usize)>,
    fixups: Vec<(usize, String, Fixup)>,
}

#[derive(Clone, Copy)]
enum Fixup {
    Branch { funct3: u32, rs1: u8, rs2: u8 },
}

impl Asm {
    /// Append one encoded word.
    pub fn word(&mut self, w: u32) -> usize {
        self.words.push(w);
        self.words.len() - 1
    }

    /// Materialise `imm` in `rd` (`lui`+`addi`, sign-correct).
    pub fn li(&mut self, rd: u8, imm: u32) {
        let imm = imm as i32;
        let hi = (imm.wrapping_add(0x800) >> 12) as u32;
        let lo = imm - (hi as i32) * 4096;
        self.word(lui(rd, hi & 0xfffff));
        self.word(addi(rd, rd, lo));
    }

    /// Record `name` at the current word.
    pub fn label(&mut self, name: impl Into<String>) {
        self.labels.push((name.into(), self.words.len()));
    }

    /// A label name no earlier one uses.
    pub fn fresh_label(&mut self) -> String {
        format!(".L{}", self.labels.len())
    }

    pub fn bne_label(&mut self, rs1: u8, rs2: u8, label: impl Into<String>) {
        let at = self.word(0);
        self.fixups.push((
            at,
            label.into(),
            Fixup::Branch {
                funct3: 1,
                rs1,
                rs2,
            },
        ));
    }

    pub fn bltu_label(&mut self, rs1: u8, rs2: u8, label: impl Into<String>) {
        let at = self.word(0);
        self.fixups.push((
            at,
            label.into(),
            Fixup::Branch {
                funct3: 6,
                rs1,
                rs2,
            },
        ));
    }

    /// Patch every fixup and return the words. An unknown label is an error,
    /// never a silently zero branch.
    pub fn finish(mut self) -> Result<Vec<u32>, String> {
        for (at, label, kind) in std::mem::take(&mut self.fixups) {
            let Some(&(_, target)) = self.labels.iter().find(|(name, _)| *name == label) else {
                return Err(format!("no label `{label}`"));
            };
            let delta = (target as i64 - at as i64) * 4;
            let delta =
                i32::try_from(delta).map_err(|_| format!("branch to `{label}` is out of range"))?;
            self.words[at] = match kind {
                Fixup::Branch { funct3, rs1, rs2 } => b_type(delta, rs2, rs1, funct3, 0x63),
            };
        }
        Ok(self.words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_encodings() {
        // `addi x1, x0, 1`
        assert_eq!(addi(1, 0, 1), 0x0010_0093);
        // `ret`
        assert_eq!(ret(), 0x0000_8067);
        // `sw x28, 0x40(x0)`: s-type split, imm[11:5] = 2.
        assert_eq!(sw(28, 0, 0x40), 0x05c0_2023);
        // `bne x1, x0, -4`
        assert_eq!(bne(1, 0, -4), 0xfe00_9ee3);
    }

    #[test]
    fn li_is_sign_correct() {
        // 0x800 is the carry case: `lui` must step to 1 and `addi` subtract.
        assert_eq!(lui(5, 1), 0x0000_12b7);
        assert_eq!(addi(5, 5, -2048), 0x8002_8293);
    }

    #[test]
    fn a_backward_branch_patches() {
        let mut a = Asm::default();
        a.label("top");
        a.word(addi(reg::T0, reg::T0, -1));
        a.bne_label(reg::T0, reg::ZERO, "top");
        let words = a.finish().expect("patched");
        assert_eq!(words.len(), 2);
        // The branch targets the word before it: -4 bytes.
        assert_eq!(words[1], bne(reg::T0, reg::ZERO, -4));
    }

    #[test]
    fn an_unknown_label_is_an_error() {
        let mut a = Asm::default();
        a.bne_label(reg::T0, reg::ZERO, "nowhere");
        assert!(a.finish().is_err());
    }

    /// `lhu`/`sh` are the loads and stores the CB counters need; the tt-metal
    /// counter word is two `u16`s, so the sync cannot widen them to 32 bits.
    #[test]
    fn the_cb_counter_access_is_halfword() {
        assert_eq!(lhu(reg::T0, reg::A6, 26), 0x01a8_5283);
        assert_eq!(sh(reg::T0, reg::A6, 26), 0x0058_1d23);
    }

    /// The interface table has tt-metal's `LocalCBInterface` shape: 8 words,
    /// 32 apart, with the ring's two pointers starting at the FIFO base.
    #[test]
    fn the_interface_is_tt_metals_local_cb_interface() {
        let words = cb::interface(0x1000, 0x1000, 0x400, 4);
        assert_eq!(words, [0x1000, 0x2000, 0x400, 4, 0x1000, 0x1000, 0, 0]);
        assert_eq!(cb::STRIDE, 32);
        assert_eq!(cb::RD_PTR, 16);
        assert_eq!(cb::WR_PTR, 20);
        assert_eq!(cb::COUNTERS, 24);
    }

    /// The four operations assemble, poll with a backward branch and keep every
    /// address inside the 32-entry table.
    #[test]
    fn the_four_cb_syncs_assemble() {
        let mut a = Asm::default();
        cb::reserve_back(&mut a, reg::A6, 3, 1, [reg::T0, reg::T1, reg::T2]);
        cb::push_back(&mut a, reg::A6, 3, 1, [reg::T0, reg::T1, reg::T2, reg::T6]);
        cb::wait_front(&mut a, reg::A6, 3, 1, [reg::T0, reg::T1, reg::T2]);
        cb::pop_front(&mut a, reg::A6, 3, 1, [reg::T0, reg::T1, reg::T2, reg::T6]);
        let words = a.finish().expect("the syncs assemble");
        assert!(words.len() > 16);
        // The first instruction addresses entry 3 of the table: 3 * 32 = 96.
        assert_eq!(words[0], addi(reg::T0, reg::A6, 96));
    }
}
