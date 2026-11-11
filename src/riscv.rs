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
    pub const RA: u8 = 1;
    pub const A0: u8 = 10;
    pub const A1: u8 = 11;
    pub const A2: u8 = 12;
    pub const A3: u8 = 13;
    pub const A4: u8 = 14;
    pub const A5: u8 = 15;
    pub const T0: u8 = 5;
    pub const T1: u8 = 6;
    pub const T2: u8 = 7;
    pub const T3: u8 = 28;
    pub const T4: u8 = 29;
    pub const T5: u8 = 30;
    pub const T6: u8 = 31;
}

fn r_type(funct7: u32, rs2: u8, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
    (funct7 << 25) | (u32::from(rs2) << 20) | (u32::from(rs1) << 15) | (funct3 << 12) | (u32::from(rd) << 7) | opcode
}

fn i_type(imm: i32, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
    let imm = (imm as u32) & 0xfff;
    (imm << 20) | (u32::from(rs1) << 15) | (funct3 << 12) | (u32::from(rd) << 7) | opcode
}

fn s_type(imm: i32, rs2: u8, rs1: u8, funct3: u32, opcode: u32) -> u32 {
    let imm = imm as u32;
    ((imm >> 5) & 0x7f) << 25 | (u32::from(rs2) << 20) | (u32::from(rs1) << 15) | (funct3 << 12) | ((imm & 0x1f) << 7) | opcode
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

fn j_type(imm: i32, rd: u8, opcode: u32) -> u32 {
    let imm = imm as u32;
    (((imm >> 20) & 1) << 31)
        | (((imm >> 1) & 0x3ff) << 21)
        | (((imm >> 11) & 1) << 20)
        | (((imm >> 12) & 0xff) << 12)
        | (u32::from(rd) << 7)
        | opcode
}

pub fn lui(rd: u8, imm20: u32) -> u32 {
    u_type(imm20, rd, 0x37)
}

pub fn auipc(rd: u8, imm20: u32) -> u32 {
    u_type(imm20, rd, 0x17)
}

pub fn addi(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(imm, rs1, 0, rd, 0x13)
}

pub fn andi(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(imm, rs1, 7, rd, 0x13)
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

pub fn lw(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(imm, rs1, 2, rd, 0x03)
}

pub fn sw(rs2: u8, rs1: u8, imm: i32) -> u32 {
    s_type(imm, rs2, rs1, 2, 0x23)
}

pub fn beq(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(imm, rs2, rs1, 0, 0x63)
}

pub fn bne(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(imm, rs2, rs1, 1, 0x63)
}

pub fn bltu(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(imm, rs2, rs1, 6, 0x63)
}

pub fn bgeu(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(imm, rs2, rs1, 7, 0x63)
}

pub fn jal(rd: u8, imm: i32) -> u32 {
    j_type(imm, rd, 0x6f)
}

pub fn jalr(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(imm, rs1, 0, rd, 0x67)
}

pub fn fence() -> u32 {
    0x0000_000f
}

/// `ret`: `jalr x0, 0(x1)`.
pub fn ret() -> u32 {
    0x0000_8067
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
    Jal { rd: u8 },
}

impl Asm {
    pub fn new() -> Self {
        Asm::default()
    }

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

    pub fn bne_label(&mut self, rs1: u8, rs2: u8, label: impl Into<String>) {
        let at = self.word(0);
        self.fixups.push((at, label.into(), Fixup::Branch { funct3: 1, rs1, rs2 }));
    }

    pub fn beq_label(&mut self, rs1: u8, rs2: u8, label: impl Into<String>) {
        let at = self.word(0);
        self.fixups.push((at, label.into(), Fixup::Branch { funct3: 0, rs1, rs2 }));
    }

    pub fn j_label(&mut self, label: impl Into<String>) {
        let at = self.word(0);
        self.fixups.push((at, label.into(), Fixup::Jal { rd: 0 }));
    }

    /// Patch every fixup and return the words. An unknown label is an error,
    /// never a silently zero branch.
    pub fn finish(mut self) -> Result<Vec<u32>, String> {
        for (at, label, kind) in std::mem::take(&mut self.fixups) {
            let Some(&(_, target)) = self.labels.iter().find(|(name, _)| *name == label) else {
                return Err(format!("no label `{label}`"));
            };
            let delta = (target as i64 - at as i64) * 4;
            let delta = i32::try_from(delta).map_err(|_| format!("branch to `{label}` is out of range"))?;
            self.words[at] = match kind {
                Fixup::Branch { funct3, rs1, rs2 } => b_type(delta, rs2, rs1, funct3, 0x63),
                Fixup::Jal { rd } => j_type(delta, rd, 0x6f),
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
        let mut a = Asm::new();
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
        let mut a = Asm::new();
        a.bne_label(reg::T0, reg::ZERO, "nowhere");
        assert!(a.finish().is_err());
    }
}
