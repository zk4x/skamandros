// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

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

use crate::ir::{DType, Dim, Error};
use crate::slab::{Slab, SlabId};

/// One unary SFPU instruction: a lane register into another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnarySfpOp {
    Neg,
    Abs,
    BitNot,
    /// `d = the exponent field of a`, as a two's complement integer and
    /// without the 127 bias (`SFPEXEXP`).
    Exexp,
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
    Row,
    Col,
    Scalar,
}

/// A comparison a [`MirOp::Compare`] performs; the result is `-1` for true and
/// `0` for false, lanewise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Gt,
    Ge,
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
    Sra,
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
            const NULL: Self = Self(u16::MAX);

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

/// Identifier of an op in [`MachineKernel::ops`]: the machine side's own id
/// space, one total order over all five cores.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MId(pub u32);

impl From<usize> for MId {
    fn from(value: usize) -> Self {
        Self(value as u32)
    }
}

impl From<MId> for usize {
    fn from(value: MId) -> Self {
        value.0 as usize
    }
}

impl SlabId for MId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
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
    /// A lanewise comparison, `-1`/`0` into `d`.
    Compare {
        op: CmpOp,
        a: LReg,
        b: LReg,
        d: LReg,
    },
    /// A lanewise conversion from `from` to `to`, in the SFPU's 32-bit lane.
    Cast {
        from: DType,
        to: DType,
        a: LReg,
        d: LReg,
    },
    /// `cond ? t : f`, lanewise, through the SFPU lane flags.
    Select {
        cond: LReg,
        t: LReg,
        f: LReg,
        d: LReg,
    },
    /// The lanewise maximum, through the SFPU lane flags.
    Max { a: LReg, b: LReg, d: LReg },
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
    /// `cb_wait_front`: wait for a page to be published on this ring.
    CbWait { cb: Cb },
    /// `cb_pop_front`: release the page [`MirOp::CbWait`] waited for.
    CbPop { cb: Cb },
    /// `cb_reserve_back`: wait for a free page on this ring.
    CbReserve { cb: Cb },
    /// `cb_push_back`: publish the page [`MirOp::CbReserve`] reserved.
    CbPush { cb: Cb },
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
    /// `reg = core_info[axis]`: this core's logical grid coordinate.
    LoadCoord { reg: u8, axis: u8 },
    /// `d = a op b`, in RISC-V.
    IArith { op: IOp, d: u8, a: u8, b: u8 },
    /// The branch back to the matching [`MirOp::Loop`] or [`MirOp::LoopReg`].
    EndLoop,
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
    pub ops: Slab<MId, (Core, MirOp)>,
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
    pub fn push(&mut self, core: Core, op: MirOp) -> MId {
        self.ops.push((core, op))
    }

    /// One core's program, in program order.
    pub fn stream(&self, core: Core) -> impl Iterator<Item = MirOp> + '_ {
        self.ops
            .iter()
            .filter(move |(_, (c, _))| *c == core)
            .map(|(_, (_, op))| *op)
    }

    /// The `Dst` registers the hardware has: eight in 32-bit mode, sixteen in
    /// 16-bit mode (a 16-bit register is half a 32-bit one).
    pub fn dst_slots(&self) -> u16 {
        if self.dst_is_16_bit { 16 } else { 8 }
    }

    /// Bring every allocation inside the hardware's limits, in place.
    pub fn alloc(&mut self) -> Result<(), Error> {
        crate::alloc::alloc(self)
    }

    /// Place the rings in L1 and build their interface entries.
    pub fn plan_cbs(&mut self) -> Result<(), Error> {
        crate::alloc::plan_cbs(self)
    }

    /// Render the module a launch runs.
    pub fn render(&self) -> Result<crate::module::Module, Error> {
        crate::compile::render(self)
    }

    /// Check that every op names a live slab entry, and that an op's rows fit
    /// the register it names.
    pub fn verify(&self) -> Result<(), Error> {
        crate::optimize::verify(self)
    }
}
