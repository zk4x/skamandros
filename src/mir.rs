// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The machine IR: what each of the three compute cores executes.

use crate::ir::DType;

/// One SFPU instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfpOp {
    Neg,
    Abs,
    Add,
    Sub,
    Mul,
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    /// `d = a << b`, the logical left shift by the amount in lane `b`
    /// (`SFPSHFT`; its amount is signed, so `b` is non-negative here and
    /// [`SfpOp::Shr`] is the one that negates it).
    Shl,
    /// `d = a >> b`, the logical right shift by the amount in lane `b`. The
    /// encoding negates the amount in `b`'s own lane (`SFPSHFT` shifts left for
    /// a positive count and right for a negative one), so `b` is scratch: a
    /// caller that wants the amount to survive must re-load it.
    Shr,
    /// `d = a + b` over two's complement integers (`SFPIADD`'s add form). The
    /// SFPU's `SfpOp::Add` is `SFPMAD`'s float add, so a lane holding integer
    /// bits needs this one.
    IAdd,
    /// `d = a - b` over two's complement integers (`SFPIADD`'s two's
    /// complement form). `SfpOp::Sub` is the float subtract.
    ISub,
    /// `d = the exponent field of a`, as a two's complement integer and
    /// without the 127 bias (`SFPEXEXP`, tt-metal's
    /// `ckernel_sfpu_exp2.h`'s `exexp(·, NoDebias)`).
    Exexp,
    /// `d = a` with `d`'s exponent field replaced by `b & 0xff`: the sign and
    /// mantissa of `a`, the exponent of `b` (`SFPSETEXP`'s integer form,
    /// tt-metal's `setexp`). `d` must not be `a` or `b`.
    Setexp,
}

/// The FPU's elementwise binary set: `ELWADD`, `ELWSUB`, `ELWMUL`. The operand
/// order is `SrcA op SrcB`, and the result overwrites the `Dst` rows it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElwOp {
    Add,
    Sub,
    Mul,
}

/// Which part of `SrcB` the FPU broadcasts to every row of `SrcA`'s face:
/// nothing (the whole face), its row 0, its column 0, or one datum. These are
/// tt-metal's `BroadcastType` and the instruction's two `BroadcastSrcB` bits
/// (`p_elwise::SRCB_BCAST_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bcast {
    Row,
    Col,
    Scalar,
}

/// A comparison a [`MirOp::Compare`] performs; the result is `-1` for true
/// and `0` for false, lanewise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Gt,
    Ge,
}

/// An integer operation a role computes with RISC-V, for a loop bound that is
/// not a literal. The set is what an index expression may contain.
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
    Sra,
}

/// This core's `core_info`: the logical grid coordinate the host wrote before
/// the launch. `x` is at this address, `y` four bytes after it.
pub const CORE_INFO: u32 = tt_isa::mailbox::MAILBOX_BASE as u32 + 0x100;

/// The edge mask a pack applies: which datums of each 32x32 tile survive, the
/// rest leaving as zeros. tt-metal's `_llk_pack_reduce_mask_config_`: the
/// packer's two `PCK_EDGE_OFFSET` masks and `TILE_ROW_SET_MAPPING_1` select
/// the surviving rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackMask {
    /// Every datum of the tile.
    None,
    /// Only row 0: one value per column, the rest zero. This is a reduce's
    /// `col` result.
    Col,
    /// Only column 0: one value per row, the rest zero. This is a reduce's
    /// `row` result.
    Row,
}

/// One instruction of a compute core's program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirOp {
    /// `SFPLOAD`: `Dst` row into a lane register. `dtype` picks the lane's
    /// format: `INT32` for a `bool` or `i32` value, `BF16` for a 16-bit `Dst`
    /// datum, `FP32` otherwise.
    Sfpload { lreg: u8, row: u32, dtype: DType },
    /// `SFPSTORE`: lane register into a `Dst` row.
    Sfpstore { lreg: u8, row: u32, dtype: DType },
    /// `SFPLOADI` twice: the two halves of `bits` into every lane of `lreg`.
    /// The only way a compile-time constant reaches the `Dst`, so a fill of an
    /// accumulator can name its value.
    LaneConst { lreg: u8, bits: u32 },
    /// One SFPU arithmetic instruction; `b` is `None` for a unary op.
    Sfp {
        op: SfpOp,
        a: u8,
        b: Option<u8>,
        d: u8,
    },
    /// A lanewise comparison, `-1`/`0` into `d`.
    Compare { op: CmpOp, a: u8, b: u8, d: u8 },
    /// A lanewise conversion from `from` to `to`, in the SFPU's 32-bit lane,
    /// with the rounding the C type has: `(int)f` truncates, `i32` to `f32`
    /// rounds to nearest, `(bool)x` is `x != 0`.
    Cast {
        from: DType,
        to: DType,
        a: u8,
        d: u8,
    },
    /// `cond ? t : f`, lanewise, through the SFPU lane flags.
    Select { cond: u8, t: u8, f: u8, d: u8 },
    /// The lanewise maximum, through the SFPU lane flags.
    Max { a: u8, b: u8, d: u8 },
    /// The unpacker's `Config` bank for `unit`: the L1 format, which path the
    /// page takes (`Dst` at [`MirOp::Setdst`]'s row, or `SrcA`/`SrcB` at
    /// [`MirOp::Setadczw`]'s face), and — on the `Src` path — the haloize
    /// transpose the unpacker applies to each face. Written when it changes; a
    /// `Dst` config also takes the datum run from the [`MirOp::Setadcx`] in
    /// effect, so the caller sets that first. `dtype` picks the L1 format:
    /// `Bf16` for a 16-bit datum, `Fp32` otherwise (an `i32` datum is opaque 32
    /// bits).
    UnpackConfig {
        unit: u8,
        to_src: bool,
        dtype: DType,
        haloize: bool,
    },
    /// `UNPACR`: the page at the head of `cb`'s ring into `unit`'s configured
    /// destination (see [`MirOp::UnpackConfig`]), the face [`MirOp::Setadczw`]
    /// chose and the run its `count` datums name. `flip` hands the `Src` bank to
    /// the Matrix Unit. The base is read from the ring's read pointer at run
    /// time.
    Unpacr {
        cb: u32,
        unit: u8,
        count: u32,
        flip: bool,
    },
    /// `SETADCZW`: which face the unpacker reads next.
    Setadczw { unit: u8, face: u32 },
    /// `SETADCX`: the datum run the unpacker walks.
    Setadcx { unit: u8, first: u32, count: u32 },
    /// `MVMUL`: `Dst` rows starting at `row` accumulate `SrcA * SrcB` for one
    /// fidelity phase; `addr_mod` is the address-modifier entry the Matrix Unit
    /// applies afterwards, and the flips hand `SrcA`/`SrcB` back to the
    /// unpacker.
    Mvmul {
        row: u32,
        addr_mod: u32,
        flip_a: bool,
        flip_b: bool,
    },
    /// `ELWADD`/`ELWSUB`/`ELWMUL`: `SrcA op SrcB` over eight `Dst` rows, which
    /// is one half-face for a 32x32 tile. `addr_mod` steps the operands for the
    /// next instruction, `bcast` picks which part of `SrcB` the FPU broadcasts,
    /// and `dst_row` is the first row the result overwrites.
    Elw {
        op: ElwOp,
        bcast: Bcast,
        addr_mod: u32,
        dst_row: u32,
    },
    /// `GAPOOL`: the Matrix Unit folds one 16x16 face over its rows into the
    /// destination row, `SrcB`'s row being the weight — tt-metal's `SUM` with
    /// `REDUCE_COL` (and, after the transposes, the other dims).
    Gapool {
        flip_src_b: bool,
        flip_src_a: bool,
        addr_mod: u32,
        dst_row: u32,
    },
    /// `GMPOOL`: the max fold over one 16x16 face's rows into the destination
    /// row. `arg_max` is always `INDEX_DIS` (0): ttsim refuses the index bit
    /// and both cards publish no index bits.
    Gmpool {
        flip_src_b: bool,
        flip_src_a: bool,
        addr_mod: u32,
        arg_max: u32,
        dst_row: u32,
    },
    /// `MOVD2B`: `Dst` row `dst_row` into `SrcB` row `src_row`. `move4_rows`
    /// moves four rows instead of one, and `use_dst32b_lo` reads the low half
    /// of a 32-bit `Dst` datum.
    Movd2b {
        use_dst32b_lo: bool,
        src_row: u32,
        addr_mod: u32,
        move4_rows: bool,
        dst_row: u32,
    },
    /// `MOVB2D`: `SrcB` rows from `src_row` into `Dst` rows from `dst_row`, the
    /// inverse of [`MirOp::Movd2b`]. `move4_rows` moves four rows, and
    /// `use_dst32b_lo` writes the low half of a 32-bit `Dst` datum.
    Movb2d {
        use_dst32b_lo: bool,
        src_row: u32,
        addr_mod: u32,
        move4_rows: bool,
        dst_row: u32,
    },
    /// `MOVB2A`: `SrcB` rows from `src_b_row` into `SrcA` rows from `src_a_row`.
    /// A four-row move; the way the transposed low halves reach `SrcA` for a
    /// `MOVA2D`.
    Movb2a {
        src_a_row: u32,
        addr_mod: u32,
        move4_rows: bool,
        src_b_row: u32,
    },
    /// `MOVA2D`: `SrcA` rows from `src_row` into `Dst` rows from `dst_row`.
    /// `move8_rows` moves eight rows, and `use_dst32b_lo` writes the low half of
    /// a 32-bit `Dst` datum.
    Mova2d {
        use_dst32b_lo: bool,
        src_row: u32,
        addr_mod: u32,
        move8_rows: bool,
        dst_row: u32,
    },
    /// `TRNSPSRCB`: transpose the 16x16 block in `SrcB` rows 16..31 in place.
    Trnspsrcb,
    /// `ALU_ACC_CTRL`: the `Dst` width the Matrix Unit and the moves address.
    /// tt-metal clears `Fp32_enabled` around the high-half `MOVD2B`/`MOVB2D`
    /// transposes on Blackhole (issue #449) and sets it back; the transposes
    /// also keep source denormals (`Zero_Flag_disabled_src`), which the moves
    /// would otherwise flush.
    AluAccCtrl { fp32: bool, zero_flag_src: bool },
    /// `ALU_FORMAT_SPEC_REG0_SrcA`, through the override bit: the format the
    /// moves see while implied-format inference is off (see
    /// `DISABLE_IMPLIED_SRCA_FMT`). tt-metal's transpose sets it to `Float32`
    /// for the low-16-bit pass and `Tf32` for the high one.
    SrcAFormat { dtype: DType },
    /// `SETRWC`: the `SrcA`/`SrcB`/`Dst` row counters and the fidelity phase,
    /// which is how the Matrix Unit is told where its operands and accumulator
    /// start. The SFPU addresses `Dst` relative to the same counter, so the
    /// matmul restores it after its groups. `flip_src_a`/`flip_src_b` release
    /// the Src banks after a consumer (tt-metal's `CLR_A`/`CLR_B`), and
    /// `dst_cr` resets the `Dst` counter instead of incrementing it.
    Setrwc {
        src_a: u32,
        src_b: u32,
        dst: u32,
        flip_src_a: bool,
        flip_src_b: bool,
        dst_cr: bool,
    },
    /// One whole `SETC16` `ThreadConfig` word (address and value), which is how
    /// the Matrix Unit's address modifiers and fidelity phase are set.
    ThreadWord { addr: u16, value: u16 },
    /// Wait until the Matrix Unit has drained. Its writes to `Dst` are
    /// asynchronous, so a value the Matrix Unit produced is not readable — by
    /// the packer or by the next kernel — until this returns.
    WaitMatrix,
    /// The base `Dst` row an unpack writes to.
    Setdst { row: u32 },
    /// The packer drains into the page at the tail of `cb`'s ring. The base is
    /// read from the ring's write pointer at run time; `row` is the `Dst` row
    /// the first packed row comes from, the packer's `W` counter. `dtype` picks
    /// the read path: `Bf16` reads a 16-bit `Dst`, everything else 32-bit.
    Pack {
        cb: u32,
        rows: u32,
        row: u32,
        last: bool,
        dtype: DType,
        mask: PackMask,
    },
    /// `cb_wait_front`: wait for a page to be published on this ring.
    CbWait { cb: u32 },
    /// `cb_pop_front`: release the page [`MirOp::CbWait`] waited for, after the
    /// unpacker has drained it.
    CbPop { cb: u32 },
    /// `cb_reserve_back`: wait for a free page on this ring.
    CbReserve { cb: u32 },
    /// `cb_push_back`: publish the page [`MirOp::CbReserve`] reserved, after the
    /// packer has drained `Dst`.
    CbPush { cb: u32 },
    /// `SEMPOST` on this core.
    SemPost { sem: u8 },
    /// `SEMWAIT` on this core, holding `before`'s blocks.
    SemWait { sem: u8, before: u8 },
    /// `SEMWAIT` C0 only: wait until `sem` is non-zero, without taking it. The
    /// consumer's claim; the matching [`MirOp::SemGet`] comes after the work.
    SemWaitData { sem: u8 },
    /// `SEMGET`: release what [`MirOp::SemWaitData`] claimed.
    SemGet { sem: u8 },
    /// `SEMWAIT` C1: hold this thread's blocks until `sem`'s value is below its
    /// maximum. With the pack hand-off's maximum at one, that is "the packer
    /// has drained the last `Dst` section", which is what the math thread needs
    /// before it writes `Dst` again (`tt_llk`'s `wait_math_semaphores`).
    SemWaitFree { sem: u8 },
    /// The next `len` ops form one row body run `count` times, stepping the
    /// `Dst` row by one per iteration. Placed by [`crate::optimize::fuse`].
    Repeat { len: u16, count: u16 },
    /// Start of a loop: the ops up to the matching [`MirOp::EndLoop`] run
    /// `count` times, as a RISC-V branch around the body. An IR loop that was
    /// constant-bound is this, per role.
    Loop { count: u32 },
    /// Start of a loop whose trip count is already in `reg`: the dynamic bound
    /// was computed by the [`MirOp::Li`]/[`MirOp::LoadArg`]/[`MirOp::IArith`]
    /// ops just before it.
    LoopReg { reg: u8 },
    /// `reg = value`.
    Li { reg: u8, value: u32 },
    /// `reg = *(a0 + arg)`: a launch argument word, which for a scalar
    /// parameter is its value.
    LoadArg { reg: u8, arg: i32 },
    /// `reg = core_info[axis]`: this core's logical grid coordinate.
    LoadCoord { reg: u8, axis: u8 },
    /// `d = a op b`, in RISC-V.
    IArith { op: IOp, d: u8, a: u8, b: u8 },
    /// The branch back to the matching [`MirOp::Loop`] or [`MirOp::LoopReg`].
    EndLoop,
}

/// The core a [`MirOp`] runs on: the unpacker (T0), the math core (T1) or the
/// packer (T2). The movers' RISC-V programs are not in this IR (`crate::mover`
/// emits them), so three roles are the whole of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    Unpack,
    Math,
    Pack,
}

/// The three compute programs, as **one** IR: every op the lowering emits is
/// appended in program order with the core it runs on, so a pass can see a
/// hand-off and the ops on both sides of it in one sequence. The per-core
/// streams are a filter over this list ([`Machine::stream`]), which is how
/// `encode` splits them again.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Machine {
    /// Every op, in the order the lowering met it.
    pub ops: Vec<(Role, MirOp)>,
    /// Is the `Dst` bank in 16-bit mode? The width is one mode for the whole
    /// kernel, as tt-metal's `DST_ACCUM_MODE` (`fp32_dest_acc_en`) is: every
    /// value that enters `Dst` is bf16 (or a `bool` lane, which is 16 bits or
    /// less) in 16-bit mode, and a 32-bit datum forces 32-bit mode. The unpacker,
    /// the packer and the SFPU's load/store format are picked from it.
    pub dst_is_16_bit: bool,
}

impl Machine {
    /// Append `op` to `role`'s program.
    pub fn push(&mut self, role: Role, op: MirOp) {
        self.ops.push((role, op));
    }

    /// One core's program, in program order.
    pub fn stream(&self, role: Role) -> Vec<MirOp> {
        self.ops
            .iter()
            .filter(|(r, _)| *r == role)
            .map(|(_, op)| *op)
            .collect()
    }
}
