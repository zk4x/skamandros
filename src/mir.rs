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

use core::fmt;

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
    /// One NoC request the movers issue: `bytes` from the parameter at launch
    /// argument `param`, whose byte offset is in `offset` (a RISC-V register the
    /// surrounding ops filled), into (`inbound`) or out of the page `cb`'s ring
    /// pointer names.
    Copy {
        cb: Cb,
        param: u32,
        bytes: u32,
        offset: u8,
        inbound: bool,
    },
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
    pub fn push(&mut self, core: Core, op: MirOp) {
        self.ops.push((core, op));
    }

    /// One core's program, in program order. The semaphore inits are not
    /// instructions and have no program of their own, so they are not here.
    pub fn stream(&self, core: Core) -> impl Iterator<Item = MirOp> + '_ {
        self.ops
            .iter()
            .filter(move |(_, (c, op))| *c == core && !matches!(op, MirOp::Semset { .. }))
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
        for (_, (core, op)) in self.ops.iter() {
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
        MirOp::Compare { op, a, b, d } => write!(
            f,
            "cmp{} {}, {} -> {}",
            cmp_name(op),
            lane(a),
            lane(b),
            lane(d)
        ),
        MirOp::Cast { from, to, a, d } => write!(
            f,
            "cast {} -> {} ({} -> {})",
            lane(a),
            lane(d),
            from.name(),
            to.name()
        ),
        MirOp::Select { cond, t, f: no, d } => write!(
            f,
            "select {}, {}, {} -> {}",
            lane(cond),
            lane(t),
            lane(no),
            lane(d)
        ),
        MirOp::Max { a, b, d } => write!(f, "max {}, {} -> {}", lane(a), lane(b), lane(d)),
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
        MirOp::CbWait { cb } => write!(f, "cbwait {}", cb_name(cb)),
        MirOp::CbPop { cb } => write!(f, "cbpop {}", cb_name(cb)),
        MirOp::CbReserve { cb } => write!(f, "cbreserve {}", cb_name(cb)),
        MirOp::CbPush { cb } => write!(f, "cbpush {}", cb_name(cb)),
        MirOp::Semset { sem, value } => write!(f, "semset {sem} = {value}"),
        MirOp::SemPost { sem } => write!(f, "sempost {sem}"),
        MirOp::SemWait { sem, before } => write!(f, "semwait {sem} before {before}"),
        MirOp::SemWaitFree { sem } => write!(f, "semwaitfree {sem}"),
        MirOp::Repeat { len, count } => write!(f, "repeat {count} x {len}"),
        MirOp::Loop { count } => write!(f, "loop {count}"),
        MirOp::LoopReg { reg } => write!(f, "loop {}", rx(reg)),
        MirOp::Li { reg, value } => write!(f, "li {}, {value}", rx(reg)),
        MirOp::LoadArg { reg, arg } => write!(f, "lw {}, {arg}(a0)", rx(reg)),
        MirOp::LoadCoord { reg, axis } => write!(f, "lw {}, core_info+{}(0)", rx(reg), axis * 4),
        MirOp::IArith { op, d, a, b } => {
            write!(f, "{} {}, {}, {}", iop_name(op), rx(d), rx(a), rx(b))
        }
        MirOp::EndLoop => write!(f, "endloop"),
        MirOp::Copy {
            cb,
            param,
            bytes,
            offset,
            inbound,
        } => write!(
            f,
            "copy {} {} param{param}[{}] {bytes} bytes",
            if inbound { "<-" } else { "->" },
            cb_name(cb),
            rx(offset)
        ),
    }
}

fn unary_name(op: UnarySfpOp) -> &'static str {
    match op {
        UnarySfpOp::Neg => "neg",
        UnarySfpOp::Abs => "abs",
        UnarySfpOp::BitNot => "bitnot",
        UnarySfpOp::Exexp => "exexp",
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

fn cmp_name(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Lt => "lt",
        CmpOp::Gt => "gt",
        CmpOp::Ge => "ge",
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
        IOp::Sra => "sra",
    }
}
