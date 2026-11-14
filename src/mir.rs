// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The machine IR: what each of the five cores executes, and the hardware it
//! names.
//!
//! [`MachineKernel`] is one interleaved op list plus the allocations the ops
//! name: the circular buffers ([`Cb`]), the `Dst` registers ([`Dst`]) and the
//! lane registers ([`LReg`]). Lowering creates them *unconstrained* — one `Dst`
//! per value, however many that is — and [`MachineKernel::alloc`] maps them
//! into what the hardware has. No op holds a buffer, a register or a layout:
//! it holds the id of one of the three slabs, so a pass can rename what an op
//! touches without touching the op.

use core::fmt;

use crate::ir::{DType, Dim};
use crate::slab::{Slab, SlabId};
use crate::status::ttStatus;

/// One unary SFPU instruction: a lane register into another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnarySfpOp {
    Neg,
    Abs,
    BitNot,
}

/// One binary SFPU instruction: two lane registers into a third.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinarySfpOp {
    Add,
    Sub,
    Mul,
    BitAnd,
    BitOr,
    BitXor,
    /// `d = a << b`, the logical left shift by the amount in lane `b`.
    Shl,
    /// `d = a >> b`, the logical right shift by the amount in lane `b`. The
    /// encoding negates the amount in `b`'s own lane, so `b` is scratch.
    Shr,
    /// `d = a + b` over two's complement integers.
    IAdd,
    /// `d = a - b` over two's complement integers.
    ISub,
    /// `d = a` with `d`'s exponent field replaced by `b & 0xff`.
    Setexp,
}

/// The FPU's elementwise binary set: `ELWADD`, `ELWSUB`, `ELWMUL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElwOp {
    Add,
    Sub,
    Mul,
}

/// Which part of `SrcB` the FPU broadcasts to every row of `SrcA`'s face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bcast {
    /// No broadcast: `SrcB` is read row for row, the plain `ELWADD` form.
    None,
    Row,
    Col,
    Scalar,
}

/// An integer operation the roles compute with RISC-V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Sll,
    Srl,
    /// The signed maximum (Zbb `max`, one instruction).
    Max,
    /// The signed minimum (Zbb `min`, one instruction).
    Min,
    /// Comparisons, producing `1`/`0` in the destination: the scalar form of
    /// `bool` (a datapath lane is `-1`/`0` instead).
    Slt,
    Eq,
    Ne,
}

/// This core's `core_info`: the logical grid coordinate the host wrote before
/// the launch. `x` is at this address, `y` four bytes after it.
pub const CORE_INFO: u32 = tt_isa::mailbox::MAILBOX_BASE as u32 + 0x100;

/// The edge mask a pack applies: which datums of each 32x32 tile survive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackMask {
    /// Every datum of the tile.
    None,
    /// Only row 0: one value per column, the rest zero.
    Col,
    /// Only column 0: one value per row, the rest zero.
    Row,
}

/// A writable lane register (`SFPLOAD`/`SFPSTORE` address `L0..L7`). An id
/// into [`MachineKernel::lregs`]; `lower` mints one per temporary and `alloc`
/// maps them onto the eight the hardware has.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LReg(pub u16);

/// A `Dst` register: one 32x32 tile region of the accumulator file. An id into
/// [`MachineKernel::dsts`]; `lower` mints one per value, `alloc` maps them onto
/// the eight (32-bit mode) or sixteen (16-bit mode) the hardware has.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Dst(pub u16);

impl Dst {
    /// `Dst` rows one register holds in 32-bit mode: a 32x32 tile of 16-lane
    /// rows, two rows per tile row.
    pub const ROWS: u32 = 64;
}

/// A circular buffer id: an entry of the `LocalCBInterface` table. An id into
/// [`MachineKernel::cbs`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cb(pub u16);

macro_rules! slab_id {
    ($ty:ty, $zero:expr) => {
        impl From<usize> for $ty {
            fn from(value: usize) -> Self {
                Self(value as _)
            }
        }

        impl From<$ty> for usize {
            fn from(value: $ty) -> Self {
                value.0 as usize
            }
        }

        impl SlabId for $ty {
            const ZERO: Self = Self($zero);

            fn inc(&mut self) {
                self.0 += 1;
            }
        }
    };
}

slab_id!(LReg, 0);
slab_id!(Dst, 0);
slab_id!(Cb, 0);

impl From<LReg> for u32 {
    fn from(reg: LReg) -> u32 {
        u32::from(reg.0)
    }
}

impl From<Dst> for u32 {
    fn from(reg: Dst) -> u32 {
        u32::from(reg.0)
    }
}

impl From<Cb> for u32 {
    fn from(cb: Cb) -> u32 {
        u32::from(cb.0)
    }
}

/// The core a [`MirOp`] runs on.
///
/// The two movers are RISC-V programs of their own (`Core::Reader` and
/// `Core::Writer`), and the three compute cores are T0/T1/T2. The order is the
/// order `render` lays the five programs out in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Core {
    Reader,
    Unpack,
    Math,
    Pack,
    Writer,
}

/// What a `Dst` register holds, and how big it is. The rows a value occupies
/// are on the op ([`MirOp::Sfpload`]'s `at` and so on); this is the register's
/// own datum and shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DstInfo {
    pub dtype: DType,
    /// The element shape: `32,32` for a tile register.
    pub shape: [Dim; 2],
}

/// Everything the machine side knows about one circular buffer. The id an op
/// names is the slab position; this is what the ring *is*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CbInfo {
    pub dtype: DType,
    /// One page's shape.
    pub page: [Dim; 2],
    /// Pages in the ring.
    pub len: Dim,
    /// The ring's first page in L1, assigned by [`MachineKernel::plan_cbs`].
    pub base: u32,
    /// Bytes in one page.
    pub page_bytes: u32,
    /// The `LocalCBInterface` entry that describes this ring, assigned by
    /// [`MachineKernel::plan_cbs`].
    pub iface_entry: u32,
}

impl CbInfo {
    /// A ring whose placement has not been decided yet.
    pub fn unplaced(dtype: DType, page: [Dim; 2], len: Dim) -> Self {
        Self {
            dtype,
            page,
            len,
            base: 0,
            page_bytes: page[0] * page[1] * dtype.bytes(),
            iface_entry: 0,
        }
    }
}

/// One instruction of a core's program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirOp {
    /// `SFPLOAD`: `Dst` row into a lane register.
    Sfpload {
        lreg: LReg,
        reg: Dst,
        at: u32,
        dtype: DType,
    },
    /// `SFPSTORE`: lane register into a `Dst` row.
    Sfpstore {
        lreg: LReg,
        reg: Dst,
        at: u32,
        dtype: DType,
    },
    /// `SFPLOADI` twice: the two halves of `bits` into every lane of `lreg`.
    LaneConst { lreg: LReg, bits: u32 },
    /// One unary SFPU instruction; the operand is a lane register.
    UnarySfp { op: UnarySfpOp, a: LReg, d: LReg },
    /// One binary SFPU instruction; both operands are lane registers.
    BinarySfp {
        op: BinarySfpOp,
        a: LReg,
        b: LReg,
        d: LReg,
    },
    /// `SFPMOV`: one lane into another, unchanged.
    SfpMov { a: LReg, d: LReg },
    /// `SFPABS`: the magnitude of `a`. `mod1` 0 negates a two's complement
    /// word (and leaves `INT_MIN` alone), 1 clears the sign bit of an FP32.
    SfpAbs { mod1: u32, a: LReg, d: LReg },
    /// `SFPCAST`: one conversion mode, lanewise. `mod1` 0 is
    /// `INT32_TO_FP32_RNE`, 2 `SM32_TO_INT32`, 3 `INT32_TO_SM32`.
    SfpCast { mod1: u32, a: LReg, d: LReg },
    /// `SFPEXEXP`: `a`'s exponent field, as an integer. `mod1` is the
    /// instruction's mode word: bit 3 is the "no debias" bit and bits 1:0 pick
    /// the condition-code source (2 is "sign/complement").
    SfpExexp { mod1: u32, a: LReg, d: LReg },
    /// `SFPEXMAN`: `a`'s mantissa field, as an integer. `mod1` 1 (`PAD9`) is
    /// the fraction alone; 0 puts the implicit one in bit 23.
    SfpExman { mod1: u32, a: LReg, d: LReg },
    /// `SFPIADD`: `d = a + imm` (or `a + VC` when `mod1`'s argument bit is
    /// clear), with `a` in the instruction's `VC` slot. Modifier 4 is
    /// "leave the lane flags alone", 2 the two's-complement `VC - VD` form.
    SfpIadd {
        imm: u32,
        a: LReg,
        d: LReg,
        mod1: u32,
    },
    /// `SFPSHFT`: `d = a` shifted by the amount in `amount`'s lane. `mod1` 2
    /// is the arithmetic (sign-extending) right shift; without it the shift is
    /// logical.
    SfpShft {
        mod1: u32,
        amount: LReg,
        a: LReg,
        d: LReg,
    },
    /// `SFPSETCC`: set the lane condition mask from `a`. It writes no lane, so
    /// the op has no destination: `VD` is only the backdoor-load gate, and the
    /// encoder passes 0 there.
    SfpSetcc { mod1: u32, a: LReg },
    /// `SFPENCC`: enable, disable or reset the lane condition mask. Bit 0 of
    /// `first` sets it, bit 1 resets every lane's flag.
    SfpEncc { first: u32, second: u32, mod1: u32 },
    /// `SFPARECIP`: one quadratic estimate of `1/a` (or of `e^a`).
    SfpArecip { mod1: u32, a: LReg, d: LReg },
    /// `SFPMUL24`: the low (or, with `upper`, the high) 23 bits of the product
    /// of two 23-bit integers.
    SfpMul24 {
        upper: bool,
        a: LReg,
        b: LReg,
        d: LReg,
    },
    /// `SFPMAD`: `d = a * b + c` in one rounding step. `mod1`'s bit 1 is
    /// "negate the addend" (`a * b - c`), the form the hardware uses for a
    /// bare multiply; a negated *product* is a `SfpMov` in the complement-sign
    /// form, since the instruction has no mode for it.
    SfpMad {
        mod1: u32,
        a: LReg,
        b: LReg,
        c: LReg,
        d: LReg,
    },
    /// `SFPSTOCHRND`: `a` to a small integer, keeping `mod1`'s range, with
    /// `round`'s rounding.
    SfpStochRnd { mod1: u32, a: LReg, d: LReg },
    /// `SFPGT`: compare the lane `vd` against `vc`, writing `-1`/`0` into
    /// `vd` (`mod1` bit 3, `SET_VD`) or setting the lane flags (`mod1` bit 0,
    /// `SET_CC`). `vd` is both an operand and the destination.
    SfpGt { mod1: u32, vc: LReg, vd: LReg },
    /// `SFPLE`: `vd <= vc`, the same two ways.
    SfpLe { mod1: u32, vc: LReg, vd: LReg },
    /// The unpacker's `Config` bank for `unit`.
    UnpackConfig {
        unit: u8,
        to_src: bool,
        dtype: DType,
        haloize: bool,
    },
    /// `UNPACR`: the page [`MirOp::SetUnpackBase`] names into `unit`'s
    /// configured destination.
    Unpacr { unit: u8, count: u32, flip: bool },
    /// `SETADCZW`: which face the unpacker reads next.
    Setadczw { unit: u8, face: u32 },
    /// `SETADCX`: the datum run the unpacker walks.
    Setadcx { unit: u8, first: u32, count: u32 },
    /// `MVMUL`: `Dst` rows starting at `at` accumulate `SrcA * SrcB`.
    Mvmul {
        reg: Dst,
        at: u32,
        addr_mod: u32,
        flip_a: bool,
        flip_b: bool,
    },
    /// `ELWADD`/`ELWSUB`/`ELWMUL` over eight `Dst` rows.
    Elw {
        op: ElwOp,
        bcast: Bcast,
        addr_mod: u32,
        reg: Dst,
        at: u32,
    },
    /// `GAPOOL`: fold one 16x16 face over its rows into the destination row.
    Gapool {
        flip_src_b: bool,
        flip_src_a: bool,
        addr_mod: u32,
        reg: Dst,
        at: u32,
    },
    /// `GMPOOL`: the max fold over one 16x16 face's rows.
    Gmpool {
        flip_src_b: bool,
        flip_src_a: bool,
        addr_mod: u32,
        arg_max: u32,
        reg: Dst,
        at: u32,
    },
    /// `MOVD2B`: `Dst` into `SrcB`.
    Movd2b {
        use_dst32b_lo: bool,
        src_row: u32,
        addr_mod: u32,
        move4_rows: bool,
        reg: Dst,
        at: u32,
    },
    /// `MOVB2D`: `SrcB` into `Dst`.
    Movb2d {
        use_dst32b_lo: bool,
        src_row: u32,
        addr_mod: u32,
        move4_rows: bool,
        reg: Dst,
        at: u32,
    },
    /// `MOVB2A`: `SrcB` rows into `SrcA` rows.
    Movb2a {
        src_a_row: u32,
        addr_mod: u32,
        move4_rows: bool,
        src_b_row: u32,
    },
    /// `MOVA2D`: `SrcA` rows into `Dst` rows.
    Mova2d {
        use_dst32b_lo: bool,
        src_row: u32,
        addr_mod: u32,
        move8_rows: bool,
        reg: Dst,
        at: u32,
    },
    /// `TRNSPSRCB`: transpose the 16x16 block in `SrcB` rows 16..31 in place.
    Trnspsrcb,
    /// `ZEROSRC`: write zeros (or negative infinity) over the source banks.
    /// `zero_val` is the ISA's `NegativeInfSrcA`, `write_mode` its
    /// `SingleBankMatrixUnit`, `bank_mask` its `BothBanks`, and the last two
    /// are `src_mask`'s two bits. tt-metal's `TTI_ZEROSRC(0, 1, 0, 1)` clears
    /// the current `SrcA` bank.
    Zerosrc {
        zero_val: bool,
        write_mode: bool,
        bank_mask: bool,
        clear_src_b: bool,
        clear_src_a: bool,
    },
    /// `ALU_ACC_CTRL`: the `Dst` width the Matrix Unit and the moves address.
    AluAccCtrl { fp32: bool, zero_flag_src: bool },
    /// `ALU_FORMAT_SPEC_REG0_SrcA`, through the override bit.
    SrcAFormat { dtype: DType },
    /// `SETRWC`: the `SrcA`/`SrcB`/`Dst` row counters and the fidelity phase.
    Setrwc {
        src_a: u32,
        src_b: u32,
        dst: u32,
        flip_src_a: bool,
        flip_src_b: bool,
        dst_cr: bool,
    },
    /// `INCRWC`: advance the `SrcA`/`SrcB`/`Dst` row counters. tt-metal reaches
    /// the same state with `SETRWC(CR_B, +8)`, which this IR cannot spell (the
    /// counter values are four-bit, and 16 does not fit); the ISA has the
    /// dedicated increment, and `SrcBVal` is only four bits wide.
    Incrwc { src_a: u32, src_b: u32, dst: u32 },
    /// One whole `SETC16` `ThreadConfig` word (address and value).
    ThreadWord { addr: u16, value: u16 },
    /// Wait until the Matrix Unit has drained.
    WaitMatrix,
    /// The base `Dst` register an unpack writes to.
    Setdst { reg: Dst },
    /// The packer's L1 destination: the page [`MirOp::CbReserve`] reserved.
    SetPackBase { cb: Cb },
    /// The packer's `Config` bank: the fields a pack's formats and edge mask
    /// decide.
    SetPackConfig { dtype: DType, mask: PackMask },
    /// `DEST_TARGET_REG_CFG_PACK_SEC0_Offset`: the `Dst` register and row the
    /// next pack reads its first row from.
    SetPackOffset { reg: Dst, at: u32 },
    /// `PACR` per four `Dst` rows.
    PackRows { rows: u32, last: bool },
    /// The unpacker's L1 source: the page [`MirOp::CbWait`] waited for.
    SetUnpackBase { cb: Cb, unit: u8 },
    /// `cb_wait_front`: wait for `n` pages to be published on this ring.
    CbWait { cb: Cb, n: u32 },
    /// `cb_pop_front`: release the `n` pages [`MirOp::CbWait`] waited for.
    CbPop { cb: Cb, n: u32 },
    /// `cb_reserve_back`: wait for `n` free pages on this ring.
    CbReserve { cb: Cb, n: u32 },
    /// `cb_push_back`: publish the `n` pages [`MirOp::CbReserve`] reserved.
    CbPush { cb: Cb, n: u32 },
    /// `SEMINIT`: the semaphore's initial `Value` and its `Max`.
    Semset { sem: u8, value: u8 },
    /// `SEMPOST` on this core.
    SemPost { sem: u8 },
    /// `SEMWAIT` on this core, holding `before`'s blocks.
    SemWait { sem: u8, before: u8 },
    /// `SEMWAIT` C1: hold this thread's blocks until `sem`'s value is below
    /// its maximum.
    SemWaitFree { sem: u8 },
    /// The next `len` ops form one row body run `count` times.
    Repeat { len: u16, count: u16 },
    /// Start of a loop: the ops up to the matching [`MirOp::EndLoop`] run
    /// `count` times.
    Loop { count: u32 },
    /// Start of a loop whose trip count is already in `reg`.
    LoopReg { reg: u8 },
    /// `reg = value`.
    Li { reg: u8, value: u32 },
    /// `reg = *(a0 + arg)`: a launch argument word.
    LoadArg { reg: u8, arg: i32 },
    /// `reg = *(base + at)`: a word read out of L1. The movers read a ring's
    /// page pointer out of the CB table and a program value out of a page.
    Lw { reg: u8, base: u8, at: i32 },
    /// `*(base + at) = value`: a word stored to L1 — one of the NoC initiator
    /// registers, or a value the program asked for.
    Sw { value: u8, base: u8, at: i32 },
    /// Wait until the NoC word at `field(t5)` reads zero: `lw` then `bne` back.
    /// A spin has no trip count, so it is not a [`MirOp::Loop`].
    NiuPoll { reg: u8, field: i32 },
    /// `reg = core_info[axis]`: this core's logical grid coordinate.
    LoadCoord { reg: u8, axis: u8 },
    /// `d = a op b`, in RISC-V.
    IArith { op: IOp, d: u8, a: u8, b: u8 },
    /// `d = cond != 0 ? a : b`, in RISC-V. The ISA has no conditional move
    /// (no Zicond), so this is the branch form:
    /// `bne cond, zero, +12; addi d, b, 0; jal zero, +8; addi d, a, 0` — four
    /// words, which also keeps working when `d` aliases `a` or `b`.
    Cmov { cond: u8, a: u8, b: u8, d: u8 },
    /// The branch back to the matching [`MirOp::Loop`] or [`MirOp::LoopReg`].
    EndLoop,
}

/// How an op uses one lane register. A single instruction reads its operands
/// before it writes its destination, so a destination may share a register with
/// a dying operand; a `ReadWrite` lane is both, like `SFPGT`'s `VD`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneUse {
    Read,
    Write,
    ReadWrite,
}

impl LaneUse {
    /// Whether the lane is read.
    pub fn reads(self) -> bool {
        matches!(self, LaneUse::Read | LaneUse::ReadWrite)
    }

    /// Whether the lane is written.
    pub fn writes(self) -> bool {
        matches!(self, LaneUse::Write | LaneUse::ReadWrite)
    }
}

impl MirOp {
    /// Every lane register this op touches, in operand order.
    ///
    /// The one place the lane operands are listed: the allocator's liveness and
    /// renaming, and `optimize::verify`'s lane bounds, all read this. An op
    /// that gains a lane operand and is not added here is caught by
    /// [`MirOp::map_lregs`]'s test, which walks the same table.
    pub fn lreg_params(&self) -> Vec<(LReg, LaneUse)> {
        use LaneUse::*;
        match *self {
            MirOp::Sfpload { lreg, .. } | MirOp::LaneConst { lreg, .. } => vec![(lreg, Write)],
            MirOp::Sfpstore { lreg, .. } => vec![(lreg, Read)],
            MirOp::UnarySfp { a, d, .. } => vec![(a, Read), (d, Write)],
            MirOp::BinarySfp { a, b, d, .. } => vec![(a, Read), (b, Read), (d, Write)],
            MirOp::SfpMov { a, d }
            | MirOp::SfpAbs { a, d, .. }
            | MirOp::SfpCast { a, d, .. }
            | MirOp::SfpExexp { a, d, .. }
            | MirOp::SfpExman { a, d, .. }
            | MirOp::SfpIadd { a, d, .. }
            | MirOp::SfpArecip { a, d, .. }
            | MirOp::SfpStochRnd { a, d, .. } => vec![(a, Read), (d, Write)],
            MirOp::SfpShft { amount, a, d, .. } => {
                vec![(amount, Read), (a, Read), (d, Write)]
            }
            MirOp::SfpSetcc { a, .. } => vec![(a, Read)],
            MirOp::SfpMul24 { a, b, d, .. } => vec![(a, Read), (b, Read), (d, Write)],
            MirOp::SfpMad { a, b, c, d, .. } => {
                vec![(a, Read), (b, Read), (c, Read), (d, Write)]
            }
            // The comparison reads its right-hand lane; with `SET_VD` it writes
            // the result back into it, and with `SET_CC` it only sets the flags.
            MirOp::SfpGt { vc, vd, mod1 } | MirOp::SfpLe { vc, vd, mod1 } => {
                let use_ = if mod1 & 8 != 0 { ReadWrite } else { Read };
                vec![(vc, Read), (vd, use_)]
            }
            _ => Vec::new(),
        }
    }

    /// Rewrite every lane operand through `f`. The mutable counterpart of
    /// [`MirOp::lreg_params`]: the two list the same lanes, and the test below
    /// holds them to it.
    pub fn map_lregs(&mut self, mut f: impl FnMut(&mut LReg)) {
        match self {
            MirOp::Sfpload { lreg, .. }
            | MirOp::Sfpstore { lreg, .. }
            | MirOp::LaneConst { lreg, .. } => f(lreg),
            MirOp::UnarySfp { a, d, .. }
            | MirOp::SfpAbs { a, d, .. }
            | MirOp::SfpCast { a, d, .. }
            | MirOp::SfpExexp { a, d, .. }
            | MirOp::SfpExman { a, d, .. }
            | MirOp::SfpIadd { a, d, .. }
            | MirOp::SfpArecip { a, d, .. }
            | MirOp::SfpStochRnd { a, d, .. }
            | MirOp::SfpMov { a, d } => {
                f(a);
                f(d);
            }
            MirOp::BinarySfp { a, b, d, .. } | MirOp::SfpMul24 { a, b, d, .. } => {
                f(a);
                f(b);
                f(d);
            }
            MirOp::SfpMad { a, b, c, d, .. } => {
                f(a);
                f(b);
                f(c);
                f(d);
            }
            MirOp::SfpShft { amount, a, d, .. } => {
                f(amount);
                f(a);
                f(d);
            }
            MirOp::SfpSetcc { a, .. } => f(a),
            MirOp::SfpGt { vc, vd, .. } | MirOp::SfpLe { vc, vd, .. } => {
                f(vc);
                f(vd);
            }
            _ => {}
        }
    }
}

/// One kernel's machine program: the ops, in one total order, and what they
/// name.
///
/// The slabs are created by `lower` without any hardware constraint — one `Dst`
/// per value, one lane register per temporary, one ring per buffer — and
/// [`MachineKernel::alloc`] is what brings them inside the hardware's limits.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MachineKernel {
    /// Every op, in the order the lowering met it, with the core it runs on.
    pub ops: Vec<(Core, MirOp)>,
    /// The circular buffers the ops name.
    pub cbs: Slab<Cb, CbInfo>,
    /// The `Dst` registers the ops name.
    pub dsts: Slab<Dst, DstInfo>,
    /// The lane registers the ops name.
    pub lregs: Slab<LReg, DType>,
    /// Is the `Dst` bank in 16-bit mode? One mode for the whole kernel.
    pub dst_is_16_bit: bool,
    /// The semaphore table, in index order: `(index, initial value)`.
    pub sems: Vec<(u8, u8)>,
    /// The launch grid, taken from the program's `grid` op.
    pub grid: [crate::module::GridTerm; 2],
    /// One launch argument per parameter, in program order.
    pub params: Vec<crate::module::ParamKind>,
}

impl MachineKernel {
    /// Append `op` to `core`'s program.
    pub fn push(&mut self, core: Core, op: MirOp) {
        self.ops.push((core, op));
    }

    /// One core's program, in program order. The semaphore inits are not
    /// instructions and have no program of their own, so they are not here.
    pub fn stream(&self, core: Core) -> impl Iterator<Item = MirOp> + '_ {
        self.ops
            .iter()
            .filter(move |(c, op)| *c == core && !matches!(op, MirOp::Semset { .. }))
            .map(|(_, op)| *op)
    }

    /// The `Dst` registers the hardware has: eight in 32-bit mode, sixteen in
    /// 16-bit mode (a 16-bit register is half a 32-bit one).
    pub fn dst_slots(&self) -> u16 {
        if self.dst_is_16_bit { 16 } else { 8 }
    }

    /// Bring every allocation inside the hardware's limits, in place.
    pub fn alloc(&mut self) -> Result<(), ttStatus> {
        crate::alloc::alloc(self)
    }

    /// Place the rings in L1 and build their interface entries.
    pub fn plan_cbs(&mut self) -> Result<(), ttStatus> {
        crate::alloc::plan_cbs(self)
    }

    /// Render the module a launch runs.
    pub fn render(&self) -> Result<crate::module::Module, ttStatus> {
        crate::compile::render(self)
    }

    /// Check that every op names a live slab entry, and that an op's rows fit
    /// the register it names.
    pub fn verify(&self) -> Result<(), ttStatus> {
        crate::optimize::verify(self)
    }
}

// ---------------------------------------------------------------------------
// The printed form
// ---------------------------------------------------------------------------

impl fmt::Display for MachineKernel {
    /// The machine kernel as assembly: the allocations first, then every op in
    /// the one order, tagged with the core it runs on.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (id, info) in self.cbs.iter() {
            writeln!(
                f,
                "cb{} = ring {} {},{} x{}",
                id.0,
                info.dtype.name(),
                info.page[0],
                info.page[1],
                info.len
            )?;
        }
        for (id, info) in self.dsts.iter() {
            writeln!(
                f,
                "dst{} = {} {},{}",
                id.0,
                info.dtype.name(),
                info.shape[0],
                info.shape[1]
            )?;
        }
        for (id, dtype) in self.lregs.iter() {
            writeln!(f, "lreg{} = lane {}", id.0, dtype.name())?;
        }
        writeln!(
            f,
            "mode = {}",
            if self.dst_is_16_bit {
                "16-bit"
            } else {
                "32-bit"
            }
        )?;
        for (sem, value) in &self.sems {
            writeln!(f, "sem{sem} = {value}")?;
        }
        for (core, op) in self.ops.iter() {
            write!(f, "{}: ", core.name())?;
            write_op(f, op)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Core {
    /// The name a program's lines are tagged with.
    pub fn name(self) -> &'static str {
        match self {
            Core::Reader => "reader",
            Core::Unpack => "unpack",
            Core::Math => "math",
            Core::Pack => "pack",
            Core::Writer => "writer",
        }
    }
}

fn lane(reg: LReg) -> String {
    format!("lreg{}", reg.0)
}

fn dst_addr(reg: Dst, at: u32) -> String {
    format!("dst{}[{}]", reg.0, at)
}

fn cb_name(cb: Cb) -> String {
    format!("cb{}", cb.0)
}

fn rx(reg: u8) -> String {
    format!("x{reg}")
}

/// One instruction, in the assembly form: the mnemonic and its operands, the
/// way the encoder's comments name them.
fn write_op(f: &mut dyn fmt::Write, op: &MirOp) -> fmt::Result {
    match *op {
        MirOp::Sfpload {
            lreg,
            reg,
            at,
            dtype,
        } => write!(
            f,
            "sfpload {} <- {}, {}",
            lane(lreg),
            dst_addr(reg, at),
            dtype.name()
        ),
        MirOp::Sfpstore {
            lreg,
            reg,
            at,
            dtype,
        } => write!(
            f,
            "sfpstore {}, {} <- {}",
            dst_addr(reg, at),
            dtype.name(),
            lane(lreg)
        ),
        MirOp::LaneConst { lreg, bits } => write!(f, "laneconst {} = 0x{bits:08x}", lane(lreg)),
        MirOp::UnarySfp { op, a, d } => {
            write!(f, "{} {} -> {}", unary_name(op), lane(a), lane(d))
        }
        MirOp::BinarySfp { op, a, b, d } => write!(
            f,
            "{} {}, {} -> {}",
            binary_name(op),
            lane(a),
            lane(b),
            lane(d)
        ),
        MirOp::SfpMov { a, d } => write!(f, "sfpmov {} -> {}", lane(a), lane(d)),
        MirOp::SfpAbs { mod1, a, d } => write!(f, "sfpabs {mod1} {} -> {}", lane(a), lane(d)),
        MirOp::SfpCast { mod1, a, d } => {
            write!(f, "sfpcast {mod1} {} -> {}", lane(a), lane(d))
        }
        MirOp::SfpExexp { mod1, a, d } => {
            write!(f, "sfpexexp {mod1} {} -> {}", lane(a), lane(d))
        }
        MirOp::SfpExman { mod1, a, d } => {
            write!(f, "sfpexman {mod1} {} -> {}", lane(a), lane(d))
        }
        MirOp::SfpIadd { imm, a, d, mod1 } => {
            write!(f, "sfpiadd {imm:#x}, {} -> {} ({mod1})", lane(a), lane(d))
        }
        MirOp::SfpShft { mod1, amount, a, d } => write!(
            f,
            "sfpshft {mod1}, amount {}, {} -> {}",
            lane(amount),
            lane(a),
            lane(d)
        ),
        MirOp::SfpSetcc { mod1, a } => write!(f, "sfpsetcc {mod1} {}", lane(a)),
        MirOp::SfpEncc {
            first,
            second,
            mod1,
        } => write!(f, "sfpencc {first:#x}, {second} ({mod1})"),
        MirOp::SfpArecip { mod1, a, d } => {
            write!(f, "sfparecip {mod1} {} -> {}", lane(a), lane(d))
        }
        MirOp::SfpMul24 { upper, a, b, d } => write!(
            f,
            "sfpmul24 {} {}, {} -> {}",
            if upper { "high" } else { "low" },
            lane(a),
            lane(b),
            lane(d)
        ),
        MirOp::SfpStochRnd { mod1, a, d } => {
            write!(f, "sfpstochrnd {mod1} {} -> {}", lane(a), lane(d))
        }
        MirOp::SfpMad { mod1, a, b, c, d } => write!(
            f,
            "sfpmad {mod1}, {} * {} + {} -> {}",
            lane(a),
            lane(b),
            lane(c),
            lane(d)
        ),
        MirOp::SfpGt { mod1, vc, vd } => {
            write!(
                f,
                "sfpgt {mod1}, {} > {} -> {}",
                lane(vd),
                lane(vc),
                lane(vd)
            )
        }
        MirOp::SfpLe { mod1, vc, vd } => {
            write!(
                f,
                "sfple {mod1}, {} <= {} -> {}",
                lane(vd),
                lane(vc),
                lane(vd)
            )
        }
        MirOp::UnpackConfig {
            unit,
            to_src,
            dtype,
            haloize,
        } => write!(
            f,
            "unpack_config u{unit} {}{}{}",
            if to_src { "src" } else { "dst" },
            if haloize { " haloize" } else { "" },
            format_args!(" {}", dtype.name())
        ),
        MirOp::Unpacr { unit, count, flip } => write!(
            f,
            "unpacr u{unit} {count}{}",
            if flip { " flip" } else { "" }
        ),
        MirOp::Setadczw { unit, face } => write!(f, "setadczw u{unit} face {face}"),
        MirOp::Setadcx { unit, first, count } => {
            write!(f, "setadcx u{unit} {first}..{}", first + count)
        }
        MirOp::Mvmul {
            reg,
            at,
            addr_mod,
            flip_a,
            flip_b,
        } => write!(
            f,
            "mvmul {}, am{addr_mod}{}{}",
            dst_addr(reg, at),
            if flip_a { " flip_a" } else { "" },
            if flip_b { " flip_b" } else { "" }
        ),
        MirOp::Elw {
            op,
            bcast,
            addr_mod,
            reg,
            at,
        } => write!(
            f,
            "elw{} {} {}, am{addr_mod}",
            elw_name(op),
            dst_addr(reg, at),
            bcast_name(bcast)
        ),
        MirOp::Gapool {
            flip_src_b,
            flip_src_a,
            addr_mod,
            reg,
            at,
        } => write!(
            f,
            "gapool {}, am{addr_mod}{}{}",
            dst_addr(reg, at),
            if flip_src_a { " flip_a" } else { "" },
            if flip_src_b { " flip_b" } else { "" }
        ),
        MirOp::Gmpool {
            flip_src_b,
            flip_src_a,
            addr_mod,
            arg_max,
            reg,
            at,
        } => write!(
            f,
            "gmpool {}, am{addr_mod} argmax{arg_max}{}{}",
            dst_addr(reg, at),
            if flip_src_a { " flip_a" } else { "" },
            if flip_src_b { " flip_b" } else { "" }
        ),
        MirOp::Movd2b {
            use_dst32b_lo,
            src_row,
            addr_mod,
            move4_rows,
            reg,
            at,
        } => write!(
            f,
            "movd2b {} -> srcb[{src_row}], am{addr_mod}{}{}",
            dst_addr(reg, at),
            if move4_rows { " 4rows" } else { "" },
            if use_dst32b_lo { " lo" } else { "" }
        ),
        MirOp::Movb2d {
            use_dst32b_lo,
            src_row,
            addr_mod,
            move4_rows,
            reg,
            at,
        } => write!(
            f,
            "movb2d srcb[{src_row}] -> {}, am{addr_mod}{}{}",
            dst_addr(reg, at),
            if move4_rows { " 4rows" } else { "" },
            if use_dst32b_lo { " lo" } else { "" }
        ),
        MirOp::Movb2a {
            src_a_row,
            addr_mod,
            move4_rows,
            src_b_row,
        } => write!(
            f,
            "movb2a srcb[{src_b_row}] -> srca[{src_a_row}], am{addr_mod}{}",
            if move4_rows { " 4rows" } else { "" }
        ),
        MirOp::Mova2d {
            use_dst32b_lo,
            src_row,
            addr_mod,
            move8_rows,
            reg,
            at,
        } => write!(
            f,
            "mova2d srca[{src_row}] -> {}, am{addr_mod}{}{}",
            dst_addr(reg, at),
            if move8_rows { " 8rows" } else { "" },
            if use_dst32b_lo { " lo" } else { "" }
        ),
        MirOp::Trnspsrcb => write!(f, "trnspsrcb"),
        MirOp::Zerosrc {
            zero_val,
            write_mode,
            bank_mask,
            clear_src_b,
            clear_src_a,
        } => write!(
            f,
            "zerosrc{}{}{}{}{}",
            if zero_val { " -inf" } else { "" },
            if write_mode { " single_bank" } else { "" },
            if bank_mask { " both_banks" } else { "" },
            if clear_src_b { " clr_b" } else { "" },
            if clear_src_a { " clr_a" } else { "" }
        ),
        MirOp::AluAccCtrl {
            fp32,
            zero_flag_src,
        } => write!(
            f,
            "alu_acc_ctrl {}{}",
            if fp32 { "fp32" } else { "16b" },
            if zero_flag_src { " keep_denorm" } else { "" }
        ),
        MirOp::SrcAFormat { dtype } => write!(f, "srca_format {}", dtype.name()),
        MirOp::Setrwc {
            src_a,
            src_b,
            dst,
            flip_src_a,
            flip_src_b,
            dst_cr,
        } => write!(
            f,
            "setrwc a{src_a} b{src_b} d{dst}{}{}{}",
            if flip_src_a { " flip_a" } else { "" },
            if flip_src_b { " flip_b" } else { "" },
            if dst_cr { " dst_cr" } else { "" }
        ),
        MirOp::ThreadWord { addr, value } => write!(f, "threadword 0x{addr:04x} = {value}"),
        MirOp::Incrwc { src_a, src_b, dst } => {
            write!(f, "incrwc a+{src_a} b+{src_b} d+{dst}")
        }
        MirOp::WaitMatrix => write!(f, "wait_matrix"),
        MirOp::Setdst { reg } => write!(f, "setdst dst{}", reg.0),
        MirOp::SetPackBase { cb } => write!(f, "setpackbase {}", cb_name(cb)),
        MirOp::SetPackConfig { dtype, mask } => {
            write!(f, "setpackconfig {} {}", dtype.name(), mask_name(mask))
        }
        MirOp::SetPackOffset { reg, at } => write!(f, "setpackoffset {}", dst_addr(reg, at)),
        MirOp::PackRows { rows, last } => {
            write!(f, "packrows {rows}{}", if last { " last" } else { "" })
        }
        MirOp::SetUnpackBase { cb, unit } => {
            write!(f, "setunpackbase {} u{unit}", cb_name(cb))
        }
        MirOp::CbWait { cb, n } => write!(f, "cbwait {} {}", cb_name(cb), n),
        MirOp::CbPop { cb, n } => write!(f, "cbpop {} {}", cb_name(cb), n),
        MirOp::CbReserve { cb, n } => write!(f, "cbreserve {} {}", cb_name(cb), n),
        MirOp::CbPush { cb, n } => write!(f, "cbpush {} {}", cb_name(cb), n),
        MirOp::Semset { sem, value } => write!(f, "semset {sem} = {value}"),
        MirOp::SemPost { sem } => write!(f, "sempost {sem}"),
        MirOp::SemWait { sem, before } => write!(f, "semwait {sem} before {before}"),
        MirOp::SemWaitFree { sem } => write!(f, "semwaitfree {sem}"),
        MirOp::Repeat { len, count } => write!(f, "repeat {count} x {len}"),
        MirOp::Loop { count } => write!(f, "loop {count}"),
        MirOp::LoopReg { reg } => write!(f, "loop {}", rx(reg)),
        MirOp::Li { reg, value } => write!(f, "li {}, {value}", rx(reg)),
        MirOp::LoadArg { reg, arg } => write!(f, "lw {}, {arg}(a0)", rx(reg)),
        MirOp::Lw { reg, base, at } => write!(f, "lw {}, {at}({})", rx(reg), rx(base)),
        MirOp::Sw { value, base, at } => write!(f, "sw {}, {at}({})", rx(value), rx(base)),
        MirOp::NiuPoll { reg, field } => {
            write!(
                f,
                "poll {}, {field}({})",
                rx(reg),
                rx(crate::riscv::reg::T5)
            )
        }
        MirOp::LoadCoord { reg, axis } => write!(f, "lw {}, core_info+{}(0)", rx(reg), axis * 4),
        MirOp::IArith { op, d, a, b } => {
            write!(f, "{} {}, {}, {}", iop_name(op), rx(d), rx(a), rx(b))
        }
        MirOp::Cmov { cond, a, b, d } => {
            write!(f, "cmov {}, {}, {}, {}", rx(d), rx(a), rx(b), rx(cond))
        }
        MirOp::EndLoop => write!(f, "endloop"),
    }
}

fn unary_name(op: UnarySfpOp) -> &'static str {
    match op {
        UnarySfpOp::Neg => "neg",
        UnarySfpOp::Abs => "abs",
        UnarySfpOp::BitNot => "bitnot",
    }
}

fn binary_name(op: BinarySfpOp) -> &'static str {
    match op {
        BinarySfpOp::Add => "fadd",
        BinarySfpOp::Sub => "fsub",
        BinarySfpOp::Mul => "fmul",
        BinarySfpOp::BitAnd => "and",
        BinarySfpOp::BitOr => "or",
        BinarySfpOp::BitXor => "xor",
        BinarySfpOp::Shl => "shl",
        BinarySfpOp::Shr => "shr",
        BinarySfpOp::IAdd => "iadd",
        BinarySfpOp::ISub => "isub",
        BinarySfpOp::Setexp => "setexp",
    }
}

fn elw_name(op: ElwOp) -> &'static str {
    match op {
        ElwOp::Add => "add",
        ElwOp::Sub => "sub",
        ElwOp::Mul => "mul",
    }
}

fn bcast_name(bcast: Bcast) -> &'static str {
    match bcast {
        Bcast::None => "none",
        Bcast::Row => "row",
        Bcast::Col => "col",
        Bcast::Scalar => "scalar",
    }
}

fn mask_name(mask: PackMask) -> &'static str {
    match mask {
        PackMask::None => "none",
        PackMask::Col => "col",
        PackMask::Row => "row",
    }
}

fn iop_name(op: IOp) -> &'static str {
    match op {
        IOp::Add => "add",
        IOp::Sub => "sub",
        IOp::Mul => "mul",
        IOp::Div => "div",
        IOp::Rem => "rem",
        IOp::And => "and",
        IOp::Or => "or",
        IOp::Xor => "xor",
        IOp::Sll => "sll",
        IOp::Srl => "srl",
        IOp::Max => "max",
        IOp::Min => "min",
        IOp::Slt => "slt",
        IOp::Eq => "eq",
        IOp::Ne => "ne",
    }
}
