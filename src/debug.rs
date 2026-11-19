// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! Debug rendering: the machine kernel's MIR, and the words each role encodes
//! to.
//!
//! The MIR is the compiler's own form ([`crate::mir::MachineKernel`]'s
//! `Display`), one line per op. Below it, an encoded role is RISC-V with Tensix
//! words inline, so the listing decodes both: a Tensix entry prints its
//! mnemonic and encoded fields, a RISC-V word prints its mnemonic and operands
//! (with a branch's target as the word index it lands on). That is what makes
//! a program whose *encoding* is wrong readable — a bad `lw` is a line, not a
//! word to decode by hand.
//!
//! `SKAM_IR=1` prints both for every launch.

use core::fmt;

use crate::encode::{Elem, Stream};
use crate::mir::MachineKernel;

/// Print the machine kernel when `SKAM_IR` is set.
pub(crate) fn machine(m: &MachineKernel) {
    if std::env::var_os("SKAM_IR").is_some() {
        eprintln!("--- machine kernel ---\n{m}");
    }
}

/// Print every role's encoded program when `SKAM_IR` is set.
pub(crate) fn programs(w: &crate::encode::Words) {
    if std::env::var_os("SKAM_IR").is_none() {
        return;
    }
    eprintln!("--- encoded programs ---");
    for (name, program) in w.programs() {
        eprintln!("{name} ({} words):\n{program}", program.rendered().len());
    }
}

impl fmt::Display for Stream {
    /// One line per word: the Tensix instructions and the RISC-V around them,
    /// in the order the core executes them.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.elems.is_empty() {
            return Ok(());
        }
        for (at, elem) in self.rendered().iter().enumerate() {
            write!(f, "{at:4}: ")?;
            match elem {
                Elem::Tensix(instruction) => {
                    let def = instruction.def();
                    write!(f, "ttinsn {:<16}", def.mnemonic())?;
                    for field in def.fields() {
                        write!(f, " {}={}", field.name(), field.extract(instruction.word()))?;
                    }
                    write!(f, "  ; {:#010x}", instruction.word())?;
                }
                Elem::Riscv(word) => {
                    write!(f, "{}", riscv(*word, at))?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// The RISC-V registers a role program may name, by number.
const REGS: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4",
    "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4",
    "t5", "t6",
];

fn reg(n: u32) -> &'static str {
    REGS[n as usize & 31]
}

/// A two's complement immediate, sign-extended from `bits`.
fn sext(value: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((value << shift) as i32) >> shift
}

/// One RISC-V word of a role program, at word index `at` (a branch target is
/// printed as the index it lands on, which is what the listing's first column
/// is).
pub fn riscv(word: u32, at: usize) -> String {
    let opcode = word & 0x7f;
    let rd = (word >> 7) & 31;
    let funct3 = (word >> 12) & 7;
    let rs1 = (word >> 15) & 31;
    let rs2 = (word >> 20) & 31;
    let funct7 = word >> 25;
    let i_imm = sext(word >> 20, 12);
    let s_imm = sext(((word >> 25) << 5) | ((word >> 7) & 31), 12);
    let b_imm = sext(
        (((word >> 31) & 1) << 12)
            | (((word >> 7) & 1) << 11)
            | (((word >> 25) & 0x3f) << 5)
            | (((word >> 8) & 0xf) << 1),
        13,
    );
    let j_imm = sext(
        (((word >> 31) & 1) << 20)
            | (((word >> 12) & 0xff) << 12)
            | (((word >> 20) & 1) << 11)
            | (((word >> 21) & 0x3ff) << 1),
        21,
    );
    let target = |bytes: i32| (at as i64 + i64::from(bytes) / 4) as i64;
    match opcode {
        0x37 => format!("lui    {}, {:#x}", reg(rd), word & 0xffff_f000),
        0x17 => format!("auipc  {}, {:#x}", reg(rd), word >> 12),
        0x6f => format!("jal    {}, {}", reg(rd), target(j_imm)),
        0x67 => format!("jalr   {}, {i_imm}({})", reg(rd), reg(rs1)),
        0x63 => {
            let name = match funct3 {
                0 => "beq",
                1 => "bne",
                4 => "blt",
                5 => "bge",
                6 => "bltu",
                7 => "bgeu",
                _ => return format!(".word  {word:#010x}"),
            };
            format!(
                "{name:<6} {}, {}, {target}",
                reg(rs1),
                reg(rs2),
                target = target(b_imm)
            )
        }
        0x03 => {
            let name = match funct3 {
                0 => "lb",
                1 => "lh",
                2 => "lw",
                4 => "lbu",
                5 => "lhu",
                _ => return format!(".word  {word:#010x}"),
            };
            format!("{name:<6} {}, {i_imm}({})", reg(rd), reg(rs1))
        }
        0x23 => {
            let name = match funct3 {
                0 => "sb",
                1 => "sh",
                2 => "sw",
                _ => return format!(".word  {word:#010x}"),
            };
            format!("{name:<6} {}, {s_imm}({})", reg(rs2), reg(rs1))
        }
        0x13 => {
            let mut out = String::new();
            let name = match funct3 {
                0 => "addi",
                1 => {
                    out = format!("slli   {}, {}, {rs2}", reg(rd), reg(rs1));
                    ""
                }
                2 => "slti",
                3 => "sltiu",
                4 => "xori",
                5 => {
                    out = format!(
                        "{} {}, {}, {rs2}",
                        if funct7 == 0x20 { "srai" } else { "srli" },
                        reg(rd),
                        reg(rs1)
                    );
                    ""
                }
                6 => "ori",
                7 => "andi",
                _ => return format!(".word  {word:#010x}"),
            };
            if out.is_empty() {
                format!("{name:<6} {}, {}, {i_imm}", reg(rd), reg(rs1))
            } else {
                out
            }
        }
        0x33 => {
            let name = match (funct7, funct3) {
                (0x00, 0) => "add",
                (0x20, 0) => "sub",
                (0x00, 1) => "sll",
                (0x00, 2) => "slt",
                (0x00, 3) => "sltu",
                (0x00, 4) => "xor",
                (0x00, 5) => "srl",
                (0x20, 5) => "sra",
                (0x00, 6) => "or",
                (0x00, 7) => "and",
                (0x01, 0) => "mul",
                (0x01, 1) => "mulh",
                (0x01, 2) => "mulhsu",
                (0x01, 3) => "mulhu",
                (0x01, 4) => "div",
                (0x01, 5) => "divu",
                (0x01, 6) => "rem",
                (0x01, 7) => "remu",
                _ => return format!(".word  {word:#010x}"),
            };
            format!("{name:<6} {}, {}, {}", reg(rd), reg(rs1), reg(rs2))
        }
        0x0f => "fence".to_string(),
        0x73 => match (funct3, word >> 20) {
            (0, 0) => "ecall".to_string(),
            (0, 1) => "ebreak".to_string(),
            (1, 0) => "fence.i".to_string(),
            _ => format!(".word  {word:#010x}"),
        },
        _ => format!(".word  {word:#010x}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::riscv::{Asm, reg};

    #[test]
    fn a_load_names_its_base_and_offset() {
        // The CB counters are halfwords, the page counters full words.
        assert_eq!(riscv(0x01a2_d383, 0), "lhu    t2, 26(t0)");
        assert_eq!(riscv(0x00c2_a303, 0), "lw     t1, 12(t0)");
    }

    #[test]
    fn a_branch_names_the_word_it_lands_on() {
        // `bne a2, zero, +8` at word 8 lands on word 10.
        assert_eq!(riscv(0x0006_1463, 8), "bne    a2, zero, 10");
    }

    #[test]
    fn the_multiply_of_a_page_index_is_an_m_instruction() {
        assert_eq!(riscv(0x0277_87b3, 0), "mul    a5, a5, t2");
    }

    #[test]
    fn an_assembled_store_round_trips() {
        let mut a = Asm::default();
        a.word(crate::riscv::sw(reg::T0, reg::T5, 64));
        assert_eq!(
            riscv(a.finish().expect("finish")[0], 0),
            "sw     t0, 64(t5)"
        );
    }
}
