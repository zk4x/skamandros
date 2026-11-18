// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! Unified program to machine IR: the only step from the SSA program to the
//! machine.
//!
//! `load` configures and fills the unpacker, the register ops become
//! load/op/store bodies on the math core, `store_to` configures and drains the
//! packer, and the `wait`/`reserve`/`push`/`pop` the mover pass inserted become
//! the ring hand-offs. The cores run as a ring paced by semaphores: the packer
//! hands a `Dst` register back, the unpacker publishes each loaded unit, and
//! the math core publishes each stored one.
//!
//! Lowering runs the simplest possible allocation: one `Dst` per value, one
//! lane-register pair per value, one ring per buffer, and no hardware limit.
//! [`crate::alloc::alloc`] is the pass that brings those inside the limits.

use std::collections::BTreeMap;

use crate::ir::{
    BOp, DType, Dim, Kernel, Op, OpId, RegShape, UOp, ValueInfo, access_shape, unit_of,
};
use crate::mir::{
    Bcast, BinarySfpOp, Cb, Core, Dst, ElwOp, IOp, LReg, MachineKernel, MirOp, PackMask, UnarySfpOp,
};
use crate::riscv::reg;
use crate::status::ttStatus;
use tt_isa::cfg::generated::thread;

/// The `Dst` lock the math thread and the packer share: tt-metal's one
/// `MATH_PACK` semaphore, starting at zero (the math thread's turn).
pub const SEM_MATH_PACK: u8 = 6;
pub const LOCK_SEMS: [(u8, u8); 1] = [(SEM_MATH_PACK, 0)];

/// `Dst` rows one 32x32 F32 tile occupies (`tt_isa`'s SFPU row is 16 datums).
pub const DST_ROWS: u32 = 64;
/// Datums in one 32x32 tile.
pub const TILE_DATUMS: u32 = 1024;
/// Datums in one 16x16 face: one `Src` unpack's run.
pub const FACE_DATUMS: u32 = TILE_DATUMS / 4;

/// The RISC-V registers a loop counts in, by nesting depth: the index counts up
/// from zero and the body may use it as a value. `encode` opens the loop with
/// the same pair, so `eval` and the branch agree on the index register.
///
/// The compute roles have two levels (`t4`/`t5`, then `a7`/`a1`); the movers
/// have (`a1`/`a2`, then `a3`/`a4`) because `t5` is their NIU base.
const COMPUTE_PAIRS: [(u8, u8); 2] = [(reg::T4, reg::T5), (reg::A7, reg::A1)];
const MOVER_PAIRS: [(u8, u8); 2] = [(reg::A1, reg::A2), (reg::A3, reg::A4)];

/// The index registers `eval` reads an induction value from, by depth.
const COMPUTE_INDEX: [u8; 2] = [COMPUTE_PAIRS[0].0, COMPUTE_PAIRS[1].0];
const MOVER_INDEX: [u8; 2] = [MOVER_PAIRS[0].0, MOVER_PAIRS[1].0];

/// Registers a role owns for evaluating a bound expression. They are disjoint
/// from `a0` (the launch arguments), `a6` (the CB interface), the loop counters
/// (`a1` is the second level's index) and the sync scratch `encode` uses.
const SCRATCH: [u8; 5] = [reg::A2, reg::A3, reg::A4, reg::A5, reg::T3];

/// A mover program's temporaries: everything it does not keep across ops. `t5`
/// is its NIU base, `t3` this tile's NOC word, `a0`/`a6` its arguments and CB
/// table, and `a1..a4` its loop counters, so those are not in the pool. A copy's
/// whole NoC request — its addresses, its page pointer, its config words and its
/// spin — allocates from here too.
const MOVER_POOL: [u8; 6] = [reg::T6, reg::T4, reg::T0, reg::T1, reg::T2, reg::A5];

/// The registers a role's index expressions may use, and the simplest allocator
/// there is: take one before evaluating an operand, give it back the moment the
/// op that consumes it is emitted. Its depth is the expression's nesting, not
/// its size, so a right-nested sum over nine values costs three registers rather
/// than nine. An empty pool is a compile error, never a silent reuse: there is
/// no spill slot, because a role's program has no frame.
struct Pool {
    free: Vec<u8>,
}

impl Pool {
    fn new(regs: &[u8]) -> Pool {
        Pool {
            free: regs.to_vec(),
        }
    }

    fn take(&mut self) -> Result<u8, ttStatus> {
        self.free
            .pop()
            .ok_or(ttStatus::TT_ERROR_REGISTERS_EXHAUSTED)
    }

    fn give(&mut self, reg: u8) {
        self.free.push(reg);
    }
}

/// What the program means, recomputed from the op that defines each value.
/// Lowering reads `dtype`/`shape` through here and never builds a table.
struct Types<'a> {
    k: &'a Kernel,
}

impl Types<'_> {
    fn of(&self, value: OpId) -> ValueInfo {
        self.k.info(value)
    }
}

/// The simplest allocation: a `Dst` register per value and a lane-register
/// pair per value, minted in program order with no reuse.
struct Alloc {
    place: BTreeMap<OpId, Dst>,
    lreg: BTreeMap<OpId, (LReg, LReg)>,
}

impl Alloc {
    fn place_of(&self, value: OpId) -> Option<Dst> {
        self.place.get(&value).copied()
    }
}

/// The compiler's own rings: the 1.0 page a `max` needs, and a staging page for
/// a computed value the Matrix Unit reads. Placement is `plan_cbs`'s.
struct Plan {
    /// The ring of every buffer the program declares, by value.
    cb: BTreeMap<OpId, Cb>,
    /// The 1.0 page, when a reduce needs one.
    ones: Option<OnesPage>,
    /// The staging page of each value the FPU reads that is not a `load`.
    staged: BTreeMap<OpId, Cb>,
    /// The byte offset of each parameter's launch-argument block, in program
    /// order. See [`crate::mover::arg_blocks`].
    arg_block: Vec<i32>,
}

#[derive(Clone, Copy)]
struct OnesPage {
    cb: Cb,
    dtype: DType,
}

/// Mint the registers and rings the lowering names: one `Dst` per value, one
/// lane-register pair per value, one ring per buffer, and the compiler's own
/// rings (a reduce's 1.0 page, a staging page per computed `Src` operand).
///
/// Nothing here is limited by the hardware: the point of lowering with no reuse
/// is that a missing register is a `plan`-time error, not a silent overlay.
fn prepare(k: &Kernel) -> Result<(Plan, MachineKernel, Alloc), ttStatus> {
    use crate::mir::{CbInfo, DstInfo};
    let mut m = MachineKernel::default();
    let mut place: BTreeMap<OpId, Dst> = BTreeMap::new();
    let mut lreg: BTreeMap<OpId, (LReg, LReg)> = BTreeMap::new();
    for (id, op) in k.ops() {
        let homes = matches!(
            *op,
            Op::Acc { .. }
                | Op::Load { .. }
                | Op::Cast { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Select { .. }
                | Op::Transpose { .. }
        );
        if !homes {
            continue;
        }
        // A scalar is index arithmetic (it names a shape term), not a datapath
        // value: the movers evaluate it in RISC-V and it is packed nowhere. An
        // `acc` is the exception: a `1,1` acc still folds in a whole `Dst` tile
        // region and is packed from it.
        if k.info(id).shape == RegShape::Scalar && !matches!(*op, Op::Acc { .. }) {
            continue;
        }
        let dtype = k.info(id).dtype;
        let reg = m.dsts.push(DstInfo {
            dtype,
            shape: [32, 32],
        });
        place.insert(id, reg);
        // A body's lanes are the hardware's own: lane 0 is the input and lane 1
        // the output of every row body, with 2..7 as the scratch the bodies
        // name (an `exp2` or a reciprocal needs all six).
        while usize::from(m.lregs.len()) < usize::from(crate::alloc::LREGS) {
            m.lregs.push(dtype);
        }
        lreg.insert(id, (LReg(0), LReg(1)));
    }
    // One datum width for the whole bank: a value that needs more than 16 bits
    // forces 32-bit mode, which tt-metal calls `DST_ACCUM_MODE`.
    let wide = place
        .keys()
        .any(|v| matches!(k.info(*v).dtype, DType::F32 | DType::I32));
    m.dst_is_16_bit = !wide;

    let mut plan = Plan {
        cb: BTreeMap::new(),
        ones: None,
        staged: BTreeMap::new(),
        arg_block: Vec::new(),
    };
    for (id, op) in k.ops() {
        if let Op::Cb { dtype, page, len } = *op {
            let cb = m.cbs.push(CbInfo::unplaced(dtype, page, len));
            plan.cb.insert(id, cb);
        }
    }
    // A `max` reduce's `SrcB` must be a row of exactly 1.0, and the caller names
    // no scaler for it: the compiler owns one 32x32 page of ones.
    if let Some(dtype) = k.ops().find_map(|(_, op)| match *op {
        Op::Max { x, .. } => Some(k.dtype(x)),
        _ => None,
    }) {
        let cb = m.cbs.push(CbInfo::unplaced(dtype, [32, 32], 1));
        plan.ones = Some(OnesPage { cb, dtype });
    }
    // A staging page per value the FPU reads that is not a `load`: the Matrix
    // Unit's `Src` banks are filled from L1, so a computed value is packed into
    // a page of its own.
    for (_, op) in k.ops() {
        for operand in page_operands(k, op) {
            let value = owner(k, operand);
            if plan.staged.contains_key(&value) || matches!(k.at(value), Op::Load { .. }) {
                continue;
            }
            let dtype = k.dtype(value);
            let cb = m.cbs.push(CbInfo::unplaced(dtype, [32, 32], 1));
            plan.staged.insert(value, cb);
        }
    }
    m.grid = crate::compile::grid_terms(k)?;
    m.params = crate::compile::parameter_kinds(k);
    plan.arg_block = crate::mover::arg_blocks(&m.params);
    // The rings are placed *before* the walk: a mover that names a page's L1
    // address (a copy whose index is read out of a page) needs it here, and the
    // ring set is already final — the program's rings, the ones page and the
    // staging pages are all minted above.
    m.plan_cbs()?;
    Ok((plan, m, Alloc { place, lreg }))
}

/// The values one op reads from an L1 page because its `Src` banks are filled
/// by the unpacker: a `matmul`'s two operands, a reduce's operand and scaler,
/// and the two operands of an FPU broadcast.
fn page_operands(k: &Kernel, op: &Op) -> Vec<OpId> {
    if let Some((tile, narrow)) = crate::ir::broadcast_pair(k, op) {
        return vec![tile, narrow];
    }
    match *op {
        Op::Matmul { a, b, .. } => vec![a, b],
        Op::Sum { x, s, .. } => vec![x, s],
        Op::Max { x, .. } => vec![x],
        Op::Transpose { src } => vec![src],
        _ => Vec::new(),
    }
}

/// The value a bitcast chain ends at: a bitcast has no home of its own.
fn owner(k: &Kernel, value: OpId) -> OpId {
    let mut at = value;
    while let Op::Bitcast { src, .. } = *k.at(at) {
        at = src;
    }
    at
}

impl Kernel {
    /// Lower the verified SSA program to the machine IR, with one register per
    /// value and no hardware limit applied.
    pub fn lower(&self) -> Result<MachineKernel, ttStatus> {
        let k = self;
        lower_with(k)
    }
}

/// Lower `k` to the five programs.
fn lower_with(k: &Kernel) -> Result<MachineKernel, ttStatus> {
    let types = Types { k };
    let types = &types;
    let (plan, mut m, alloc) = prepare(k)?;
    let alloc = &alloc;

    // The dtype a value occupies `Dst` with. 16-bit mode stores every value as
    // a bf16 datum, the `bool` lane's `-1` pattern included; 32-bit mode holds
    // bf16 as the f32 it widens to, so the packer is the only thing that rounds
    // it, at the bf16 buffer it lands in.
    let wide = !m.dst_is_16_bit;
    let storage = |dtype: crate::ir::DType| {
        if !wide {
            crate::ir::DType::Bf16
        } else if dtype == crate::ir::DType::Bf16 {
            crate::ir::DType::F32
        } else {
            dtype
        }
    };
    // The launch argument index of every parameter, in program order: a scalar
    // parameter's value is loaded from its pair in the arguments array.
    let mut param: BTreeMap<crate::ir::OpId, u32> = BTreeMap::new();
    let mut next = 0u32;
    for (id, op) in k.ops() {
        match *op {
            Op::Param { .. } | Op::Variable { .. } => {
                param.insert(id, next);
                next += 1;
            }
            _ => {}
        }
    }
    // The induction value of every loop being lowered, innermost last: a bound
    // may name an outer loop's index, whose value is its counter register.
    let mut loops: Vec<crate::ir::OpId> = Vec::new();
    // The edge mask a pack of each value carries follows the value's shape: a
    // reduce's result keeps only its folded positions, and every value with the
    // same narrow shape keeps the same part — a `1,32` row 0, a `32,1` column 0.
    let mut acc_mask: BTreeMap<crate::ir::OpId, PackMask> = k
        .ops()
        .map(|(id, _)| {
            let mask = match types.of(id).shape {
                RegShape::Row => PackMask::Col,
                RegShape::Column => PackMask::Row,
                _ => PackMask::None,
            };
            (id, mask)
        })
        .collect();
    // Who reads each value, and where it last does: a `matmul` reads its
    // operands' CB pages straight into `SrcA`/`SrcB`, so a page stays reserved
    // until the last of its readers, and only then is popped.
    let mut last_use: BTreeMap<crate::ir::OpId, crate::ir::OpId> = BTreeMap::new();
    let mut matmul_use: std::collections::BTreeSet<crate::ir::OpId> =
        std::collections::BTreeSet::new();
    let mut non_matmul_use: std::collections::BTreeSet<crate::ir::OpId> =
        std::collections::BTreeSet::new();
    // Whether anything but a copy's index reads a value: a copy's ends are its
    // operands, every other read is an index or a datapath operand.
    let mut other_use: std::collections::BTreeSet<crate::ir::OpId> =
        std::collections::BTreeSet::new();
    for (at, op) in k.ops() {
        for v in op.parameters() {
            last_use.insert(v, at);
            if matches!(*op, Op::Copy { dst, src, .. } if v != dst && v != src) {
                continue;
            }
            other_use.insert(v);
            if let Some((tile, narrow)) = crate::ir::broadcast_pair(k, op)
                && (v == tile || v == narrow)
            {
                // A narrow operand that is a scalar *constant* is the binary
                // lane op's immediate (see `scalar_const`), not an FPU page, so
                // the tile still enters `Dst` and must be unpacked.
                let immediate =
                    k.info(narrow).shape == RegShape::Scalar && scalar_const(k, narrow).is_some();
                if !immediate {
                    // The FPU's broadcast reads both operands from `Src` pages.
                    matmul_use.insert(v);
                    continue;
                }
            }
            match *op {
                Op::Matmul { a, b, .. } if v == a || v == b => {
                    matmul_use.insert(v);
                }
                // A reduce's operands are Src operands too: the fold runs on
                // the Src path from their pages, exactly as `matmul`'s do.
                Op::Sum { x, s, .. } if v == x || v == s => {
                    matmul_use.insert(v);
                }
                Op::Max { x, .. } if v == x => {
                    matmul_use.insert(v);
                }
                _ => {
                    non_matmul_use.insert(v);
                }
            }
        }
    }
    // The page of a load that a `matmul` also reads is released at the last
    // reader, not at the load: the ring's read pointer must not move while the
    // Matrix Unit's unpack still points at it.
    // Which segment the walk is in: the reader's copies come first, then the
    // compute ring, then the writer's.
    let mut mover: Option<Core> = Some(Core::Reader);
    for (at, op) in k.ops() {
        // An op's id *is* the value it defines, so the arms name it `value`
        // without a field.
        let value = at;
        // A reduce's axis is its accumulator's shape (`ReduceDim::of`); the
        // verifier has already refused a `32,32` acc there, and a `matmul`
        // needs that same `32,32`.
        let reduce_dim = match *op {
            Op::Sum { acc, .. } | Op::Max { acc, .. } => {
                let Some(dim) = crate::ir::ReduceDim::of(types.of(acc).shape) else {
                    return Err(crate::status::internal_bug(
                        "lower: a reduce accumulator has no fold",
                    ));
                };
                Some(dim)
            }
            _ => None,
        };
        match *op {
            // A scalar parameter is one launch-argument word: the movers read
            // it and the datapath never does.
            Op::Variable { .. } => {}
            Op::Const { .. }
            | Op::Param { .. }
            | Op::Cb { .. }
            | Op::Acc { .. }
            | Op::Grid { .. } => {}
            // A scalar is index arithmetic: it names a `Dim`, and the mover
            // evaluates it in RISC-V. The datapath has no scalar values, so
            // nothing here has to produce one.
            Op::GridCoord { .. } => {}
            Op::Cast { .. }
            | Op::Bitcast { .. }
            | Op::Unary { .. }
            | Op::Binary { .. }
            | Op::Select { .. }
                if types.of(value).shape == RegShape::Scalar => {}
            Op::Loop { bound } => {
                let depth = loops.len();
                if let Some(core) = mover {
                    // A mover segment's loop is that program's own: its own
                    // index/limit pair, its own temporaries, so `t5` (the NIU
                    // base) never holds an index.
                    let (_, limit) = *MOVER_PAIRS.get(depth).ok_or(ttStatus::TT_TODO_LOOP_DEPTH)?;
                    match bound_const(k, bound) {
                        Some(count) => m.push(core, MirOp::Loop { count }),
                        None => {
                            let mut ops = Vec::new();
                            let mut scratch = Pool::new(&MOVER_POOL);
                            scalar_with(
                                k,
                                types,
                                &mut ops,
                                &param,
                                &loops,
                                bound,
                                limit,
                                &mut scratch,
                                &MOVER_INDEX,
                                &plan,
                                &m.cbs,
                            )?;
                            ops.push(MirOp::LoopReg { reg: limit });
                            for op in ops {
                                m.push(core, op);
                            }
                        }
                    }
                    loops.push(value);
                    continue;
                }
                // Two levels is what the compute roles have counter pairs for;
                // a third would need another pair, which is not reserved.
                let (_, limit) = *COMPUTE_PAIRS
                    .get(depth)
                    .ok_or(ttStatus::TT_TODO_LOOP_DEPTH)?;
                match bound_const(k, bound) {
                    Some(count) => {
                        m.push(Core::Unpack, MirOp::Loop { count });
                        m.push(Core::Math, MirOp::Loop { count });
                        m.push(Core::Pack, MirOp::Loop { count });
                    }
                    None => {
                        // A symbolic bound is RISC-V arithmetic in the role,
                        // like the C loop of a tt-metal kernel; the limit ends
                        // in the pair's limit register and the branch reads it.
                        for role in [Core::Unpack, Core::Math, Core::Pack] {
                            let mut ops = Vec::new();
                            let mut scratch = Pool::new(&SCRATCH);
                            scalar_with(
                                k,
                                types,
                                &mut ops,
                                &param,
                                &loops,
                                bound,
                                limit,
                                &mut scratch,
                                &COMPUTE_INDEX,
                                &plan,
                                &m.cbs,
                            )?;
                            ops.push(MirOp::LoopReg { reg: limit });
                            for op in ops {
                                m.push(role, op);
                            }
                        }
                    }
                }
                loops.push(value);
            }
            Op::EndLoop => {
                loops.pop();
                if let Some(core) = mover {
                    m.push(core, MirOp::EndLoop);
                    continue;
                }
                m.push(Core::Unpack, MirOp::EndLoop);
                m.push(Core::Math, MirOp::EndLoop);
                m.push(Core::Pack, MirOp::EndLoop);
            }
            // The markers split the one list into the reader's segment, the
            // compute ring and the writer's.
            Op::EndReader => {
                mover = None;
                continue;
            }
            // tt-metal's `tile_regs_acquire`: wait until `MATH_PACK` is below
            // its maximum, without taking it. The packer's release takes it.
            Op::MathLock => {
                m.push(Core::Math, MirOp::SemWaitFree { sem: SEM_MATH_PACK });
            }
            // `tile_regs_commit`: math has filled its `Dst` section.
            Op::MathUnlock => {
                m.push(Core::Math, MirOp::SemPost { sem: SEM_MATH_PACK });
            }
            // `tile_regs_wait`: wait for math, but the get is the release.
            Op::PackLock => {
                m.push(
                    Core::Pack,
                    MirOp::SemWaitNoGet {
                        sem: SEM_MATH_PACK,
                        before: 0,
                    },
                );
            }
            // `tile_regs_release`: hand `Dst` back to math.
            Op::PackUnlock => {
                m.push(Core::Pack, MirOp::SemGet { sem: SEM_MATH_PACK });
            }
            Op::EndCompute => {
                mover = Some(Core::Writer);
                continue;
            }
            Op::Shape(_) => continue,
            // The ring hand-offs `add_movers` placed: a consumer waits for a
            // page and releases it, a producer reserves one and publishes it.
            // The core is the one that does it — the unpacker and the packer in
            // the compute ring, the movers in their own segments.
            Op::Wait { cb, n } | Op::Pop { cb, n } => {
                let ring = cb_id(
                    *plan
                        .cb
                        .get(&cb)
                        .ok_or(ttStatus::TT_ERROR_BUFFER_NOT_PLACED)?,
                );
                let op = if matches!(*op, Op::Wait { .. }) {
                    MirOp::CbWait { cb: ring, n }
                } else {
                    MirOp::CbPop { cb: ring, n }
                };
                let core = match mover {
                    Some(Core::Writer) => Core::Writer,
                    _ => Core::Unpack,
                };
                m.push(core, op);
                continue;
            }
            Op::Reserve { cb, n } | Op::Push { cb, n } => {
                let ring = cb_id(
                    *plan
                        .cb
                        .get(&cb)
                        .ok_or(ttStatus::TT_ERROR_BUFFER_NOT_PLACED)?,
                );
                let op = if matches!(*op, Op::Reserve { .. }) {
                    MirOp::CbReserve { cb: ring, n }
                } else {
                    MirOp::CbPush { cb: ring, n }
                };
                let core = match mover {
                    Some(Core::Reader) => Core::Reader,
                    _ => Core::Pack,
                };
                m.push(core, op);
                continue;
            }
            Op::Copy {
                dst,
                src,
                dst_offset,
                src_offset,
                shape: access,
            } => {
                if let Some(core) = mover {
                    copy_to(
                        k, types, &plan, &param, &loops, &mut m, core, dst, src, dst_offset,
                        src_offset, access,
                    )?;
                }
                continue;
            }
            Op::Load {
                buffer,
                offset,
                shape: access,
            } => {
                // A scalar load is an index read: the mover evaluates it from
                // the page it copied, in RISC-V (see `mover::eval_value`).
                if types.of(value).shape == RegShape::Scalar {
                    continue;
                }
                // A vector load no datapath op reads is the mover's lane bases:
                // it reads one datum per lane out of the page it just copied, so
                // no `Dst` slot ever holds it. (The movers clone the load into
                // their own segment to name that page, which is why a copy's
                // index is a second such load.)
                if !other_use.contains(&value) {
                    continue;
                }
                let info = types.of(buffer);
                let dtype = match k.at(buffer) {
                    Op::Cb { dtype, .. } => *dtype,
                    _ => {
                        let _ = buffer;
                        return Err(ttStatus::TT_ERROR_NOT_A_BUFFER);
                    }
                };
                let unit_shape = access_shape(access)?;
                let reg = dst_reg(alloc, owner(k, value))?;
                // A value only an `Src` reader consumes never enters `Dst`: the
                // unpack goes straight into `SrcA`/`SrcB` at the op that reads
                // it, and this load only reserves the page.
                let lands_in_dst = non_matmul_use.contains(&value) || !matmul_use.contains(&value);
                // Only a tile (one whole page) is lowered into the datapath
                // today; a row, a column or a datum needs the narrowed unpack,
                // which is not written. A narrow access that only an `Src`
                // reader consumes needs no unpack at all: that op reads the
                // page's own layout.
                if unit_shape != RegShape::Tile && lands_in_dst {
                    let _ = (at, access);
                    return Err(ttStatus::TT_TODO_ACCESS_SHAPE);
                }
                let rows = rows_of(unit_shape, at)?;
                let cb = cb_id(plan.cb.get(&buffer).copied().ok_or_else(|| {
                    let _ = buffer;
                    ttStatus::TT_ERROR_BUFFER_NOT_PLACED
                })?);
                if lands_in_dst {
                    // A bf16 page cannot enter a 32-bit `Dst` through the
                    // unpacker: `BF16 -> BF16` is its only Dst-path conversion
                    // for that datum, and the `Dst` is one width for the whole
                    // kernel. It goes through `SrcA` instead, which is what
                    // tt-lang's `copy_tile` does for `typecast bf16 -> f32`
                    // (`llk_math_eltwise_unary_sfpu_typecast` has "no SFPU
                    // kernel needed" for that pair): the unpacker writes the
                    // page into `SrcA` as bf16, and a 32-bit `Dst` reads it back
                    // through `MOVA2D`, which treats the `SrcA` lane as TF32 and
                    // writes the full 32-bit datum. A bf16 datum widens exactly
                    // on that move (7 mantissa bits of 10, the exponent and
                    // sign unchanged), so the `cast` that follows is a rename.
                    if wide && dtype == crate::ir::DType::Bf16 {
                        // The whole page at once: one `Z` plane per face, the
                        // descriptor's four faces, 64 `SrcA` rows.
                        m.push(
                            Core::Unpack,
                            MirOp::Setadczw {
                                u0: true,
                                u1: false,
                                z0: Some(0),
                                z1: None,
                            },
                        );
                        m.push(
                            Core::Unpack,
                            MirOp::UnpackConfig {
                                unit: 0,
                                to_src: true,
                                dtype,
                                haloize: false,
                                add_dest_cntr: false,
                                ch1_z_stride: 0,
                            },
                        );
                        m.push(Core::Unpack, MirOp::SetUnpackBase { cb, unit: 0 });
                        m.push(
                            Core::Unpack,
                            MirOp::Unpacr {
                                unit: 0,
                                count: unit_shape.datums(),
                                // `UNPACR`'s bit 6: `Set_Data_Valid` on
                                // Blackhole (ttsim's schema and tt-metal's LLK
                                // argument), the bank handover the math's
                                // `MOVA2D` waits for. Without it the Matrix
                                // Unit stalls forever on an invalid bank.
                                flip: true,
                            },
                        );
                        // The `Src` row counter starts at zero, as the unpack's
                        // own destination does.
                        m.push(
                            Core::Math,
                            MirOp::Setrwc {
                                src_a: 0,
                                src_b: 0,
                                dst: 0,
                                flip_src_a: false,
                                flip_src_b: false,
                                dst_cr: false,
                            },
                        );
                        // `MOVA2D` moves eight rows per instruction, and a
                        // 32-bit `Dst` holds a tile's 1024 datums in 64 rows.
                        for band in 0..(DST_ROWS / 8) {
                            m.push(
                                Core::Math,
                                MirOp::Mova2d {
                                    use_dst32b_lo: false,
                                    src_row: 8 * band,
                                    addr_mod: 0,
                                    move8_rows: true,
                                    reg,
                                    at: 8 * band,
                                },
                            );
                        }
                        // `SETRWC(CLR_A)`: the `SrcA` bank goes back to the
                        // unpacker now that the Matrix Unit has read it.
                        m.push(
                            Core::Math,
                            MirOp::Setrwc {
                                src_a: 0,
                                src_b: 0,
                                dst: 0,
                                flip_src_a: true,
                                flip_src_b: false,
                                dst_cr: false,
                            },
                        );
                    } else {
                        // The page's wait and release are the `wait`/`pop` ops
                        // `add_movers` placed around this load.
                        m.push(Core::Unpack, MirOp::Setdst { reg });
                        m.push(
                            Core::Unpack,
                            MirOp::Setadczw {
                                u0: true,
                                u1: false,
                                z0: Some(0),
                                z1: None,
                            },
                        );
                        m.push(
                            Core::Unpack,
                            MirOp::Setadcx {
                                unit: 0,
                                first: 0,
                                count: unit_shape.datums(),
                            },
                        );
                        m.push(
                            Core::Unpack,
                            MirOp::UnpackConfig {
                                unit: 0,
                                to_src: false,
                                dtype,
                                haloize: false,
                                add_dest_cntr: false,
                                ch1_z_stride: 0,
                            },
                        );
                        m.push(Core::Unpack, MirOp::SetUnpackBase { cb, unit: 0 });
                        m.push(
                            Core::Unpack,
                            MirOp::Unpacr {
                                unit: 0,
                                count: unit_shape.datums(),
                                flip: false,
                            },
                        );
                    }
                }
                // A `matmul` operand's page is waited for and released by its
                // own `wait`/`pop` ops.
                let _ = info;
                let _ = rows;
                let _ = offset;
            }
            Op::Store {
                src,
                buffer,
                offset,
                shape: access,
            } => {
                // An accumulator is written in place in `Dst`, not packed: the
                // only form the verifier admits is a fill from a constant of
                // the accumulator's own dtype, and the SFPU writes it row by
                // row.
                if types.of(buffer).kind == crate::ir::ValueKind::Acc {
                    let (bits, from) = match *k.at(src) {
                        Op::Const { bits, dtype, .. } => (bits, dtype),
                        _ => {
                            let _ = src;
                            return Err(ttStatus::TT_TODO_ACC_STORE);
                        }
                    };
                    let acc_dtype = types.of(buffer).dtype;
                    if from != acc_dtype {
                        let _ = (from, acc_dtype);
                        return Err(ttStatus::TT_TODO_ACC_STORE);
                    }
                    // The lane is one 32-bit image: a bf16 datum widens to it.
                    let bits = lane_image(from, bits);
                    let (unit_shape, _, _) = unit_of([32, 32], acc_dtype)?;
                    let rows = rows_of(unit_shape, at)?;
                    let reg = dst_reg(alloc, buffer)?;
                    // The fill writes `Dst` rows; `reg_sync` places the wait
                    // for a packer that still reads them.
                    m.push(
                        Core::Math,
                        MirOp::LaneConst {
                            lreg: LReg(0),
                            bits,
                        },
                    );
                    // An SFPU store covers a row pair (`SFPSTORE` refuses an odd
                    // row and writes two), the same step the datapath uses.
                    for row in (0..rows).step_by(2) {
                        m.push(
                            Core::Math,
                            MirOp::Sfpstore {
                                lreg: LReg(0),
                                reg,
                                at: row,
                                dtype: acc_dtype,
                            },
                        );
                    }
                    let _ = (offset, src);
                    // An accumulator store never reads a CB page, but a page
                    // this instruction was the last reader of is released here
                    // exactly like the packed path's.
                    continue;
                }
                let dtype = match k.at(buffer) {
                    Op::Cb { dtype, .. } => *dtype,
                    _ => {
                        let _ = buffer;
                        return Err(ttStatus::TT_ERROR_NOT_A_BUFFER);
                    }
                };
                let unit_shape = access_shape(access)?;
                // Only a tile is lowered today; a narrower pack needs the
                // packer's `SETADCX` narrowing.
                if unit_shape != RegShape::Tile {
                    let _ = (at, access);
                    return Err(ttStatus::TT_TODO_ACCESS_SHAPE);
                }
                let rows = rows_of(unit_shape, at)?;
                let cb = cb_id(plan.cb.get(&buffer).copied().ok_or_else(|| {
                    let _ = buffer;
                    ttStatus::TT_ERROR_BUFFER_NOT_PLACED
                })?);
                // The ring's `reserve`/`push` are the ops `add_movers` placed
                // around this store. The `Dst` hand-off is `reg_sync`'s: the
                // math thread has published the section and the packer reads it
                // here.
                m.push(Core::Pack, MirOp::SetPackBase { cb });
                m.push(
                    Core::Pack,
                    MirOp::SetPackConfig {
                        dtype,
                        // A pack of a reduced accumulator carries the reduce's
                        // edge mask; every other pack keeps the whole tile.
                        mask: acc_mask[&owner(k, src)],
                    },
                );
                m.push(
                    Core::Pack,
                    MirOp::SetPackOffset {
                        reg: dst_reg(alloc, owner(k, src))?,
                        at: 0,
                    },
                );
                m.push(Core::Pack, MirOp::PackRows { rows, last: true });
                let _ = offset;
            }
            Op::Cast { src, dtype } => {
                let from = types.of(src).dtype;
                let shape = types.of(src).shape;
                let rows = sfp_rows(shape, at)?;
                let s = dst_reg(alloc, owner(k, src))?;
                let out = dst_reg(alloc, owner(k, value))?;
                let (la, ld) = alloc.lreg[&owner(k, value)];
                // The conversion happens in the lane. 16-bit mode stores every
                // value as a bf16 datum, so a `bool` lane *is* `-1.0`/`0.0` there
                // and an `f32` one is already bf16 precision; 32-bit mode holds
                // bf16 as the f32 it widens to. Either way the recipe is the
                // 32-bit lane's, so `cast_lanes` never sees `Bf16`.
                let lane = |dtype: crate::ir::DType| {
                    if dtype == crate::ir::DType::Bf16 {
                        crate::ir::DType::F32
                    } else {
                        dtype
                    }
                };
                for row in rows {
                    m.push(
                        Core::Math,
                        MirOp::Sfpload {
                            lreg: la,
                            reg: s,
                            at: row,
                            dtype: storage(from),
                        },
                    );
                    for op in cast_lanes(lane(from), lane(dtype), la, ld, LReg(2))? {
                        m.push(Core::Math, op);
                    }
                    m.push(
                        Core::Math,
                        MirOp::Sfpstore {
                            lreg: ld,
                            reg: out,
                            at: row,
                            dtype: storage(dtype),
                        },
                    );
                }
            }
            // A bitcast is resolved, not lowered: every reader reads the source.
            Op::Bitcast { .. } => {}
            Op::Unary { op, a } => {
                let shape = types.of(a).shape;
                let rows = sfp_rows(shape, at)?;
                let src = dst_reg(alloc, owner(k, a))?;
                let out = dst_reg(alloc, owner(k, value))?;
                let (la, ld) = alloc.lreg[&owner(k, value)];
                let dtype = storage(types.of(a).dtype);
                // `exp2` is the one transcendental no IR op spells: its
                // exponent assembly is `SFPEXEXP`/`SFPSETEXP`, so it is a body
                // here rather than a line in `src/unfold.rs`.
                if op == UOp::Exp2 {
                    let mut body = Vec::new();
                    exp2_body(&mut body, src, out, &rows, la, dtype);
                    for op in body {
                        m.push(Core::Math, op);
                    }
                    continue;
                }
                if op == UOp::Rsqrt {
                    let mut body = Vec::new();
                    rsqrt_body(&mut body, src, out, &rows, la, dtype)?;
                    for op in body {
                        m.push(Core::Math, op);
                    }
                    continue;
                }
                if op == UOp::Log2 {
                    let mut body = Vec::new();
                    log2_body(&mut body, src, out, &rows, la, dtype);
                    for op in body {
                        m.push(Core::Math, op);
                    }
                    continue;
                }
                if op == UOp::Sin || op == UOp::Cos {
                    let mut body = Vec::new();
                    sin_cos_body(&mut body, src, out, &rows, la, op == UOp::Cos, dtype)?;
                    for op in body {
                        m.push(Core::Math, op);
                    }
                    continue;
                }
                if op == UOp::Reciprocal {
                    // One Newton step per datum width: a bf16 value carries
                    // seven mantissa bits, so the quadratic estimate needs one.
                    let steps = usize::from(types.of(a).dtype != crate::ir::DType::Bf16) + 1;
                    let mut body = Vec::new();
                    reciprocal_body(&mut body, src, out, &rows, la, steps, dtype);
                    for op in body {
                        m.push(Core::Math, op);
                    }
                    continue;
                }
                let sfp = unary(op, at)?;
                for row in rows {
                    m.push(
                        Core::Math,
                        MirOp::Sfpload {
                            lreg: la,
                            reg: src,
                            at: row,
                            dtype,
                        },
                    );
                    m.push(
                        Core::Math,
                        MirOp::UnarySfp {
                            op: sfp,
                            a: la,
                            d: ld,
                        },
                    );
                    m.push(
                        Core::Math,
                        MirOp::Sfpstore {
                            lreg: ld,
                            reg: out,
                            at: row,
                            dtype,
                        },
                    );
                }
            }
            Op::Binary { op, a, b } => {
                let left = types.of(a);
                let right = types.of(b);
                // A scalar constant operand is a lane immediate, not a tile:
                // every coefficient and mask `src/unfold.rs` writes is a
                // literal, and a lane load is two instructions where a filled
                // tile would be a `Dst` slot and a sweep. Any other scalar
                // (an index expression) still has no lane and is refused.
                let ca = scalar_const(k, a);
                let cb = scalar_const(k, b);
                if left.shape != right.shape
                    && !(left.shape == RegShape::Scalar && ca.is_some())
                    && !(right.shape == RegShape::Scalar && cb.is_some())
                {
                    // A broadcast is an FPU op, and the FPU's `Src` path takes
                    // `f32` and `bf16` only. Anything else (`i32`, `bool`) needs
                    // the SFPU route, which is not lowered: say so rather than
                    // report the FPU form as merely pending.
                    if !matches!(left.dtype, DType::F32 | DType::Bf16)
                        || !matches!(right.dtype, DType::F32 | DType::Bf16)
                    {
                        let _ = (left.dtype, right.dtype);
                        return Err(ttStatus::TT_TODO_BROADCAST_SFPU);
                    }
                    // The FPU's `ELWADD`/`ELWSUB`/`ELWMUL` broadcast `SrcB`:
                    // one operand is the tile, the other its row, column or
                    // scalar. Both come from pages, so each is a `load`'s page
                    // or the compiler's staged one.
                    if !matches!(op, BOp::Add | BOp::Sub | BOp::Mul) {
                        let _ = (op, left.shape, right.shape);
                        return Err(ttStatus::TT_TODO_BROADCAST);
                    }
                    let (tile, narrow) = match (left.shape, right.shape) {
                        (RegShape::Tile, _) => (a, b),
                        (_, RegShape::Tile) => (b, a),
                        _ => {
                            let _ = (left.shape, right.shape);
                            return Err(ttStatus::TT_TODO_BROADCAST);
                        }
                    };
                    // The instruction is `SrcA op SrcB` and the tile is the
                    // one that can be `SrcA`, so a difference with the narrow
                    // value on the left has no lowering.
                    if op == BOp::Sub && tile != a {
                        let _ = (a, b);
                        return Err(ttStatus::TT_TODO_BROADCAST);
                    }
                    let bcast = match types.of(narrow).shape {
                        RegShape::Row => Bcast::Row,
                        RegShape::Column => Bcast::Col,
                        RegShape::Scalar => Bcast::Scalar,
                        RegShape::Tile => {
                            return Err(crate::status::internal_bug(
                                "lower: a broadcast with two tile operands",
                            ));
                        }
                    };
                    fpu_broadcast(
                        &mut m, k, types, &plan, alloc, &acc_mask, value, op, tile, narrow, bcast,
                        at,
                    )?;
                    // A page whose last reader this broadcast was is released
                    // here; `continue` skips the common block below.
                    continue;
                }
                let shape = if left.shape == RegShape::Scalar {
                    right.shape
                } else {
                    left.shape
                };
                let dtype = match left.dtype {
                    // bf16 runs the same lane ops as f32: in 16-bit mode the
                    // SFPU widens on load and the store truncates, so each op
                    // lands at bf16 precision; in 32-bit mode the value is the
                    // f32 it widens to and the packer rounds at the buffer.
                    crate::ir::DType::F32 | crate::ir::DType::Bf16 => storage(left.dtype),
                    // The bit and shift ops are raw 32-bit words, and the
                    // comparisons order sign-magnitude, which an `i32` lane is
                    // only where its datum is non-negative (the integer-
                    // comparison gap zyx2's `TODO.md` records). The unfold's
                    // `!= 0` tests a masked 0/1, so that is exact.
                    crate::ir::DType::I32
                        if matches!(
                            op,
                            BOp::Add
                                | BOp::Sub
                                | BOp::BitAnd
                                | BOp::BitOr
                                | BOp::BitXor
                                | BOp::BitShiftLeft
                                | BOp::BitShiftRight
                                | BOp::Mod
                                | BOp::Cmplt
                                | BOp::Cmpgt
                                | BOp::Cmpge
                                | BOp::Eq
                                | BOp::NotEq
                        ) =>
                    {
                        crate::ir::DType::I32
                    }
                    // A bool lane is `-1`/`0`, so the bitwise and/or *is* the
                    // logical one (this is what zyx's `tt_sfp_integer` does for
                    // `BOp::And`/`Or` too).
                    crate::ir::DType::Bool if matches!(op, BOp::And | BOp::Or) => {
                        crate::ir::DType::Bool
                    }
                    other => {
                        let _ = (op, other);
                        return Err(ttStatus::TT_TODO_BINARY_DTYPE);
                    }
                };
                let rows = sfp_rows(shape, at)?;
                let out = dst_reg(alloc, owner(k, value))?;
                let (la, ld) = alloc.lreg[&owner(k, value)];
                let lb = LReg(2);
                // A comparison writes `-1`/`0` lanes, so its result is stored in
                // the integer format; everything else keeps the operands'.
                let out_dtype = if matches!(
                    op,
                    BOp::Cmplt | BOp::Cmpgt | BOp::Cmpge | BOp::Eq | BOp::NotEq
                ) {
                    storage(crate::ir::DType::Bool)
                } else {
                    dtype
                };
                // `mod` on `i32` is an exact integer remainder, not a float
                // shortcut: it is tt-metal's `fmod_int32`, whose body `mod_body`
                // spells out instruction by instruction. It keeps its operands
                // in lanes 0 and 1 and leaves the result in lane 3, so the
                // constant hoisting below does not apply: both operands are
                // loaded per row.
                if op == BOp::Mod {
                    for &row in &rows {
                        match ca {
                            Some(bits) => m.push(
                                Core::Math,
                                MirOp::LaneConst {
                                    lreg: LReg(0),
                                    bits,
                                },
                            ),
                            None => m.push(
                                Core::Math,
                                MirOp::Sfpload {
                                    lreg: LReg(0),
                                    reg: dst_reg(alloc, owner(k, a))?,
                                    at: row,
                                    dtype,
                                },
                            ),
                        }
                        match cb {
                            Some(bits) => m.push(
                                Core::Math,
                                MirOp::LaneConst {
                                    lreg: LReg(1),
                                    bits,
                                },
                            ),
                            None => m.push(
                                Core::Math,
                                MirOp::Sfpload {
                                    lreg: LReg(1),
                                    reg: dst_reg(alloc, owner(k, b))?,
                                    at: row,
                                    dtype,
                                },
                            ),
                        }
                        let mut body = Vec::new();
                        mod_body(&mut body);
                        for op in body {
                            m.push(Core::Math, op);
                        }
                        m.push(
                            Core::Math,
                            MirOp::Sfpstore {
                                lreg: LReg(3),
                                reg: out,
                                at: row,
                                dtype: out_dtype,
                            },
                        );
                    }
                    continue;
                }
                // The constant lanes live for the whole sweep, so they are set
                // once, before the first row. The one exception is a right
                // shift's amount: its encoding negates the amount in its own
                // lane (there is no other way to ask `SFPSHFT` for the other
                // direction), so it is re-loaded per row.
                let reload_amount = matches!(op, BOp::BitShiftRight);
                if let Some(bits) = ca {
                    m.push(Core::Math, MirOp::LaneConst { lreg: la, bits });
                }
                if let Some(bits) = cb
                    && !reload_amount
                {
                    m.push(Core::Math, MirOp::LaneConst { lreg: lb, bits });
                }
                for row in rows {
                    if ca.is_none() {
                        m.push(
                            Core::Math,
                            MirOp::Sfpload {
                                lreg: la,
                                reg: dst_reg(alloc, owner(k, a))?,
                                at: row,
                                dtype,
                            },
                        );
                    }
                    if cb.is_none() {
                        m.push(
                            Core::Math,
                            MirOp::Sfpload {
                                lreg: lb,
                                reg: dst_reg(alloc, owner(k, b))?,
                                at: row,
                                dtype,
                            },
                        );
                    } else if reload_amount {
                        m.push(
                            Core::Math,
                            MirOp::LaneConst {
                                lreg: lb,
                                bits: match cb {
                                    Some(bits) => bits,
                                    None => {
                                        return Err(crate::status::internal_bug(
                                            "lower: a re-loaded shift amount is not a constant",
                                        ));
                                    }
                                },
                            },
                        );
                    }
                    match op {
                        BOp::Cmplt | BOp::Cmpgt | BOp::Cmpge => {
                            let mut cmp = Vec::new();
                            match op {
                                BOp::Cmplt => sfp_lt(&mut cmp, la, lb, ld),
                                BOp::Cmpgt => sfp_gt(&mut cmp, la, lb, ld),
                                _ => sfp_ge(&mut cmp, la, lb, ld),
                            }
                            for op in cmp {
                                m.push(Core::Math, op);
                            }
                        }
                        // `a == b` is `!(a > b || a < b)`, two of the ordered
                        // tests this compiler already has, combined as
                        // sign-magnitude `-1`/`0` lanes. This is zyx's
                        // `tt_sfp_compare` shape without its NaN and `-0`
                        // specials, which the ordered tests do not carry either.
                        BOp::Eq | BOp::NotEq => {
                            let mut cmp = Vec::new();
                            sfp_gt(&mut cmp, la, lb, LReg(3));
                            sfp_lt(&mut cmp, la, lb, LReg(4));
                            for op in cmp {
                                m.push(Core::Math, op);
                            }
                            m.push(
                                Core::Math,
                                MirOp::BinarySfp {
                                    op: BinarySfpOp::BitOr,
                                    a: LReg(3),
                                    b: LReg(4),
                                    d: ld,
                                },
                            );
                            if matches!(op, BOp::Eq) {
                                m.push(
                                    Core::Math,
                                    MirOp::UnarySfp {
                                        op: UnarySfpOp::BitNot,
                                        a: ld,
                                        d: ld,
                                    },
                                );
                            }
                        }
                        BOp::Max | BOp::Min => {
                            let mut m_ops = Vec::new();
                            if matches!(op, BOp::Max) {
                                sfp_max(&mut m_ops, la, lb, ld);
                            } else {
                                sfp_min(&mut m_ops, la, lb, ld);
                            }
                            for op in m_ops {
                                m.push(Core::Math, op);
                            }
                        }
                        _ => m.push(
                            Core::Math,
                            MirOp::BinarySfp {
                                // An `i32` lane is a raw 32-bit word, so its add and
                                // subtract are the integer instruction rather than
                                // `SFPMAD`'s float ones.
                                op: if left.dtype == crate::ir::DType::I32 {
                                    match op {
                                        BOp::Add => BinarySfpOp::IAdd,
                                        BOp::Sub => BinarySfpOp::ISub,
                                        other => binary(other, at)?,
                                    }
                                } else {
                                    binary(op, at)?
                                },
                                a: la,
                                b: lb,
                                d: ld,
                            },
                        ),
                    }
                    m.push(
                        Core::Math,
                        MirOp::Sfpstore {
                            lreg: ld,
                            reg: out,
                            at: row,
                            dtype: out_dtype,
                        },
                    );
                }
            }
            // `%y = transpose %x`: tt-metal's `transpose_dest_32b<true>`. The
            // tile is first moved into the destination `Dst` slot (the transpose
            // works in place), then the math thread runs tt-metal's hi/lo
            // half sequence: each 16-bit half is moved from `Dst` into `SrcB`,
            // transposed there by `TRNSPSRCB`, and written back into its
            // transposed face. The SrcA format/FP32 switches are the ones
            // tt-metal uses to keep the two halves apart (Blackhole issue #449).
            Op::Transpose { src } => {
                let shape = types.of(src).shape;
                if shape != RegShape::Tile {
                    let _ = (at, shape);
                    return Err(ttStatus::TT_ERROR_TRANSPOSE_OPERAND);
                }
                let dtype = types.of(src).dtype;
                let out = dst_reg(alloc, owner(k, value))?;
                // The page the tile comes from: a `load`'s own, or the
                // compiler's staged copy of a computed value.
                let (cb, staged) =
                    fpu_page(k, &plan, src, at, ttStatus::TT_ERROR_TRANSPOSE_OPERAND)?;
                if staged {
                    stage(&mut m, k, &types, alloc, &acc_mask, src, cb)?;
                    m.push(Core::Unpack, MirOp::CbWait { cb, n: 1 });
                }
                // tt-metal's `transpose_tile` has two paths. A 32-bit `Dst`
                // unpacks the page face-transposed into `Dst` and runs
                // `transpose_dest_32b<false>` on the math thread; a 16-bit
                // source (`bf16` here, which skamandros's unpacker cannot
                // convert to a 32-bit `Dst`) does the whole transpose on the
                // unpacker with `transpose_of_faces` and
                // `within_face_16x16_transpose`, then datacopies the result.
                if dtype != crate::ir::DType::Bf16 {
                    // tt-metal's 32-bit arm is `unpack_to_dest` with
                    // `transpose_of_faces`, then `transpose_dest_32b<false>`.
                    // ttsim refuses that sequence at its first `MOVD2B`
                    // (`tf32 use_dst32b=1 dest_32b_lo=1`, the Blackhole erratum
                    // TEN-4245), so the fp32 path is not lowered yet.
                    let _ = (at, out);
                    return Err(ttStatus::TT_TODO_TRANSPOSE);
                }
                {
                    // A bf16 page cannot enter a 32-bit `Dst` through the
                    // unpacker; the load arm's `SrcA` widening is the way in.
                    // `transpose_of_faces` + `within_face_16x16_transpose`:
                    // source faces 0,2,1,3 land in SrcA faces 0..3.
                    for (i, (src_face, dst_face)) in [(0u32, 0u32), (2, 1), (1, 2), (3, 3)]
                        .into_iter()
                        .enumerate()
                    {
                        m.push(
                            Core::Unpack,
                            MirOp::Setadczw {
                                u0: true,
                                u1: false,
                                z0: Some(src_face),
                                z1: Some(dst_face),
                            },
                        );
                        m.push(
                            Core::Unpack,
                            MirOp::UnpackConfig {
                                unit: 0,
                                to_src: true,
                                dtype,
                                haloize: true,
                                add_dest_cntr: true,
                                ch1_z_stride: dtype.bytes() * 256,
                            },
                        );
                        m.push(Core::Unpack, MirOp::SetUnpackBase { cb, unit: 0 });
                        m.push(
                            Core::Unpack,
                            MirOp::Unpacr {
                                unit: 0,
                                count: 256,
                                flip: i == 3,
                            },
                        );
                    }
                    m.push(
                        Core::Math,
                        MirOp::Setrwc {
                            src_a: 0,
                            src_b: 0,
                            dst: 0,
                            flip_src_a: false,
                            flip_src_b: false,
                            dst_cr: false,
                        },
                    );
                    for band in 0..(DST_ROWS / 8) {
                        m.push(
                            Core::Math,
                            MirOp::Mova2d {
                                use_dst32b_lo: false,
                                src_row: 8 * band,
                                addr_mod: 0,
                                move8_rows: true,
                                reg: out,
                                at: 8 * band,
                            },
                        );
                    }
                    m.push(
                        Core::Math,
                        MirOp::Setrwc {
                            src_a: 0,
                            src_b: 0,
                            dst: 0,
                            flip_src_a: true,
                            flip_src_b: false,
                            dst_cr: false,
                        },
                    );
                }
                if staged {
                    m.push(Core::Unpack, MirOp::CbPop { cb, n: 1 });
                }
            }
            Op::Select {
                cond,
                if_true,
                if_false,
            } => {
                let shape = types.of(if_true).shape;
                let rows = sfp_rows(shape, at)?;
                let sc = dst_reg(alloc, owner(k, cond))?;
                let st = dst_reg(alloc, owner(k, if_true))?;
                let sf = dst_reg(alloc, owner(k, if_false))?;
                let out = dst_reg(alloc, owner(k, value))?;
                let ld = alloc.lreg[&owner(k, value)].1;
                let lc = LReg(3);
                let lt = LReg(0);
                let lf = LReg(2);
                if types.of(cond).dtype != crate::ir::DType::Bool {
                    let _ = at;
                    return Err(ttStatus::TT_ERROR_CONDITION_NOT_BOOL);
                }
                for row in rows {
                    m.push(
                        Core::Math,
                        MirOp::Sfpload {
                            lreg: lc,
                            reg: sc,
                            at: row,
                            dtype: storage(crate::ir::DType::Bool),
                        },
                    );
                    m.push(
                        Core::Math,
                        MirOp::Sfpload {
                            lreg: lt,
                            reg: st,
                            at: row,
                            dtype: storage(types.of(if_true).dtype),
                        },
                    );
                    m.push(
                        Core::Math,
                        MirOp::Sfpload {
                            lreg: lf,
                            reg: sf,
                            at: row,
                            dtype: storage(types.of(if_false).dtype),
                        },
                    );
                    let mut pick = Vec::new();
                    sfp_select(&mut pick, lc, lt, lf, ld);
                    for op in pick {
                        m.push(Core::Math, op);
                    }
                    m.push(
                        Core::Math,
                        MirOp::Sfpstore {
                            lreg: ld,
                            reg: out,
                            at: row,
                            dtype: storage(types.of(value).dtype),
                        },
                    );
                }
            }
            // `sum row` is the Matrix Unit's `x @ scaler`: tt-metal's
            // `ReduceDim::ROW` with `PoolType::SUM` is a matmul against the
            // scaler's column of ones, not a pool instruction, so it shares
            // this sequence.
            Op::Matmul { .. } | Op::Sum { .. }
                if matches!(*op, Op::Matmul { .. })
                    || reduce_dim == Some(crate::ir::ReduceDim::Row) =>
            {
                let (acc, a, b) = match *op {
                    Op::Matmul { acc, a, b } => (acc, a, b),
                    Op::Sum { acc, x, s } => (acc, x, s),
                    _ => {
                        return Err(crate::status::internal_bug(
                            "lower: the matmul/sum-row arm matched another op",
                        ));
                    }
                };
                if types.of(acc).kind != crate::ir::ValueKind::Acc {
                    let _ = at;
                    return Err(ttStatus::TT_ERROR_MATMUL_ACC);
                }
                let dt_a = types.of(a).dtype;
                let dt_b = types.of(b).dtype;
                let dtype = types.of(acc).dtype;
                // The operands share a dtype; the accumulator may be wider
                // (bf16 operands accumulate in f32), never narrower.
                if !dt_a.is_float()
                    || dt_a != dt_b
                    || (dt_a == crate::ir::DType::F32 && dtype == crate::ir::DType::Bf16)
                {
                    let _ = (at, dt_a, dt_b, dtype);
                    return Err(ttStatus::TT_ERROR_MATMUL_DTYPE);
                }
                // The accumulator is the `Dst` width; a `bf16` accumulator in a
                // module whose bank is 32-bit would accumulate in FP32 instead.
                if dtype == crate::ir::DType::Bf16 && wide {
                    let _ = at;
                    return Err(ttStatus::TT_ERROR_MATMUL_ACCUM_WIDE);
                }
                let (cb_a, stage_a) = fpu_page(k, &plan, a, at, ttStatus::TT_ERROR_MATMUL_OPERAND)?;
                let (cb_b, stage_b) = fpu_page(k, &plan, b, at, ttStatus::TT_ERROR_MATMUL_OPERAND)?;
                // A computed operand has no page: the compiler packs it into its
                // own and the Matrix Unit reads that.
                if stage_a {
                    stage(&mut m, k, &types, alloc, &acc_mask, a, cb_a)?;
                }
                if stage_b {
                    stage(&mut m, k, &types, alloc, &acc_mask, b, cb_b)?;
                }
                let base = dst_reg(alloc, acc)?;
                // Scratch for `mvmul_group`, which emits a run of math ops: the
                // run is appended to the one IR as it is produced.
                let mut math_scratch: Vec<MirOp> = Vec::new();
                // The Matrix Unit writes the accumulator's rows; `reg_sync`
                // places the wait for a packer that still reads them.
                // The Matrix Unit's address modifiers, fidelity phase and row
                // counters are this thread's `ThreadConfig` and RWC state. Every
                // word goes out as a whole `SETC16`, so an entry carrying several
                // fields of one word is composed here.
                for entry in [
                    vec![(thread::ADDR_MOD_AB_SEC0_SrcAIncr, 0u16)],
                    vec![(thread::ADDR_MOD_DST_SEC0_DestIncr, 0)],
                    vec![(thread::ADDR_MOD_AB_SEC1_SrcAIncr, 0)],
                    vec![(thread::ADDR_MOD_DST_SEC1_FidelityIncr, 1)],
                    // End of a group's first `SrcB` half: `SrcB` on eight rows,
                    // phase back to 0.
                    vec![
                        (thread::ADDR_MOD_AB_SEC2_SrcBIncr, 8),
                        (thread::ADDR_MOD_AB_SEC2_SrcBClear, 0),
                    ],
                    vec![
                        (thread::ADDR_MOD_DST_SEC2_DestIncr, 0),
                        (thread::ADDR_MOD_DST_SEC2_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC2_FidelityClear, 1),
                    ],
                    // End of a group that is not the `fi`'s last: phase and
                    // `SrcB` back to 0.
                    vec![
                        (thread::ADDR_MOD_AB_SEC3_SrcBIncr, 0),
                        (thread::ADDR_MOD_AB_SEC3_SrcBClear, 1),
                    ],
                    vec![
                        (thread::ADDR_MOD_DST_SEC3_DestIncr, 0),
                        (thread::ADDR_MOD_DST_SEC3_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC3_FidelityClear, 1),
                    ],
                    // End of an `fi`: as 3, and the `Dst` base on 32 rows for
                    // the next `fi`.
                    vec![
                        (thread::ADDR_MOD_AB_SEC4_SrcBIncr, 0),
                        (thread::ADDR_MOD_AB_SEC4_SrcBClear, 1),
                    ],
                    vec![
                        (thread::ADDR_MOD_DST_SEC4_DestIncr, 32),
                        (thread::ADDR_MOD_DST_SEC4_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC4_FidelityClear, 1),
                    ],
                    vec![(thread::FIDELITY_BASE_Phase, 0)],
                ] {
                    m.push(Core::Math, thread_word(&entry)?);
                }
                // Src rows and the `Dst` counter start at 0.
                m.push(
                    Core::Math,
                    MirOp::Setrwc {
                        src_a: 0,
                        src_b: 0,
                        dst: 0,
                        flip_src_a: false,
                        flip_src_b: false,
                        dst_cr: false,
                    },
                );
                // The accumulator's rows are added to, never cleared: an `acc`
                // holds whatever the program last put there, and zeroing it is a
                // `store_to %acc from %zero` the program writes itself.
                // One `(fi, k)` group: A's face `(fi, k)` into `SrcB`, B's two
                // faces `(k, 0)` and `(k, 1)` into `SrcA`. The unpacker and the
                // Matrix Unit alternate through the bank handover, so the two
                // roles interleave by ownership, not by instruction position.
                // A staged operand's page is published by the packer (T2) before
                // the unpacker reads it, and released after the last face.
                for (cb, staged) in [(cb_a, stage_a), (cb_b, stage_b)] {
                    if staged {
                        m.push(Core::Unpack, MirOp::CbWait { cb, n: 1 });
                    }
                }
                for fi in 0..2u32 {
                    for kk in 0..2u32 {
                        m.push(
                            Core::Unpack,
                            MirOp::Setadczw {
                                u0: false,
                                u1: true,
                                z0: Some(2 * fi + kk),
                                z1: None,
                            },
                        );
                        m.push(
                            Core::Unpack,
                            MirOp::UnpackConfig {
                                unit: 1,
                                to_src: true,
                                dtype: dt_a,
                                haloize: false,
                                add_dest_cntr: false,
                                ch1_z_stride: 0,
                            },
                        );
                        m.push(Core::Unpack, MirOp::SetUnpackBase { cb: cb_a, unit: 1 });
                        m.push(
                            Core::Unpack,
                            MirOp::Unpacr {
                                unit: 1,
                                count: FACE_DATUMS,
                                flip: true,
                            },
                        );
                        m.push(
                            Core::Unpack,
                            MirOp::Setadczw {
                                u0: true,
                                u1: false,
                                z0: Some(2 * kk),
                                z1: None,
                            },
                        );
                        m.push(
                            Core::Unpack,
                            MirOp::UnpackConfig {
                                unit: 0,
                                to_src: true,
                                dtype: dt_b,
                                haloize: false,
                                add_dest_cntr: false,
                                ch1_z_stride: 0,
                            },
                        );
                        m.push(Core::Unpack, MirOp::SetUnpackBase { cb: cb_b, unit: 0 });
                        m.push(
                            Core::Unpack,
                            MirOp::Unpacr {
                                unit: 0,
                                count: FACE_DATUMS,
                                flip: true,
                            },
                        );
                        mvmul_group(&mut math_scratch, base, 0);
                        for op in math_scratch.drain(..) {
                            m.push(Core::Math, op);
                        }
                        m.push(
                            Core::Math,
                            MirOp::Mvmul {
                                reg: base,
                                at: 8,
                                addr_mod: MATMUL_AM_GROUP,
                                flip_a: true,
                                flip_b: false,
                            },
                        );
                        m.push(
                            Core::Unpack,
                            MirOp::Setadczw {
                                u0: true,
                                u1: false,
                                z0: Some(2 * kk + 1),
                                z1: None,
                            },
                        );
                        m.push(Core::Unpack, MirOp::SetUnpackBase { cb: cb_b, unit: 0 });
                        m.push(
                            Core::Unpack,
                            MirOp::Unpacr {
                                unit: 0,
                                count: FACE_DATUMS,
                                flip: true,
                            },
                        );
                        mvmul_group(&mut math_scratch, base, 16);
                        for op in math_scratch.drain(..) {
                            m.push(Core::Math, op);
                        }
                        let end = if kk == 0 {
                            MATMUL_AM_GROUP
                        } else {
                            MATMUL_AM_FI
                        };
                        m.push(
                            Core::Math,
                            MirOp::Mvmul {
                                reg: base,
                                at: 24,
                                addr_mod: end,
                                flip_a: true,
                                flip_b: true,
                            },
                        );
                    }
                }
                // The Matrix Unit advanced the thread's `Dst` counter as it
                // walked the accumulator, and an `SFPLOAD`/`SFPSTORE` address is
                // relative to it, so the counter goes back to 0. Its writes are
                // still in flight, so the accumulator is not readable (the pack
                // follows on T2) until the unit drains.
                m.push(Core::Math, MirOp::WaitMatrix);
                for (cb, staged) in [(cb_a, stage_a), (cb_b, stage_b)] {
                    if staged {
                        m.push(Core::Unpack, MirOp::CbPop { cb, n: 1 });
                    }
                }
                m.push(
                    Core::Math,
                    MirOp::Setrwc {
                        src_a: 0,
                        src_b: 0,
                        dst: 0,
                        flip_src_a: false,
                        flip_src_b: false,
                        dst_cr: false,
                    },
                );
            }
            // `sum col`/`max col`: `GAPOOL`/`GMPOOL` fold each 16x16 face over
            // its rows into the destination row, one value per column, and the
            // pack's edge mask drops everything below row 0 — tt-metal's
            // `_llk_math_reduce_` for `REDUCE_COL`.
            Op::Sum { .. } | Op::Max { .. } => {
                let (acc, x, scaler, max) = match *op {
                    Op::Sum { acc, x, s } => (acc, x, Some(s), false),
                    Op::Max { acc, x } => (acc, x, None, true),
                    _ => {
                        return Err(crate::status::internal_bug(
                            "lower: the reduce arm matched another op",
                        ));
                    }
                };
                let Some(dim) = reduce_dim else {
                    return Err(crate::status::internal_bug("lower: a reduce has no fold"));
                };
                if dim == crate::ir::ReduceDim::Scalar {
                    let _ = dim;
                    return Err(ttStatus::TT_TODO_REDUCE);
                }
                // `max` needs a `SrcB` row of exactly 1.0 (`GMPOOL` scales by
                // the exponent). It names no scaler, so the compiler's own
                // page (`plan.ones`) is the operand.
                let dt_x = types.of(x).dtype;
                let dtype = types.of(acc).dtype;
                if dtype == crate::ir::DType::Bf16 && wide {
                    let _ = at;
                    return Err(ttStatus::TT_ERROR_MATMUL_ACCUM_WIDE);
                }
                let (cb_x, stage_x) = fpu_page(k, &plan, x, at, ttStatus::TT_ERROR_REDUCE_OPERAND)?;
                let (cb_b, stage_b, dt_b) = match scaler {
                    Some(scaler) => {
                        let dt_s = types.of(scaler).dtype;
                        if dt_s != dt_x {
                            let _ = (dt_s, dt_x);
                            return Err(ttStatus::TT_ERROR_REDUCE_DTYPE);
                        }
                        let (cb, staged) =
                            fpu_page(k, &plan, scaler, at, ttStatus::TT_ERROR_REDUCE_OPERAND)?;
                        (cb, staged, dt_s)
                    }
                    None => {
                        let ones = plan.ones.ok_or_else(|| {
                            let _ = at;
                            ttStatus::TT_TODO_REDUCE
                        })?;
                        if ones.dtype != dt_x {
                            let _ = (ones.dtype, dt_x);
                            return Err(ttStatus::TT_ERROR_REDUCE_DTYPE);
                        }
                        (cb_id(ones.cb), false, ones.dtype)
                    }
                };
                // The 1.0 page is pushed once at kernel start; a `max` waits
                // for it before the first face and never pops it, so every
                // `max` in the program re-reads the same page.
                if scaler.is_none() {
                    m.push(Core::Unpack, MirOp::CbWait { cb: cb_b, n: 1 });
                }
                // A computed operand has no page: the compiler stages it, and
                // the unpacker waits for that page before the first face.
                if stage_x {
                    stage(&mut m, k, &types, alloc, &acc_mask, x, cb_x)?;
                    m.push(Core::Unpack, MirOp::CbWait { cb: cb_x, n: 1 });
                }
                if stage_b {
                    if let Some(scaler) = scaler {
                        stage(&mut m, k, &types, alloc, &acc_mask, scaler, cb_b)?;
                    }
                    m.push(Core::Unpack, MirOp::CbWait { cb: cb_b, n: 1 });
                }
                if types.of(acc).kind != crate::ir::ValueKind::Acc {
                    let _ = at;
                    return Err(ttStatus::TT_ERROR_REDUCE_ACC);
                }
                let base = dst_reg(alloc, acc)?;
                // The accumulator's rows are written; `reg_sync` places the
                // wait for the packer to give the section back.
                // `reduce_configure_addrmod`: entry 0 clears the phase, 1 and 2
                // step `SrcB`/`Dst`, and 3 steps only the phase.
                for entry in [
                    // `ADDR_MOD_0`: no increments, and the phase is cleared.
                    vec![
                        (thread::ADDR_MOD_AB_SEC0_SrcAIncr, 0u16),
                        (thread::ADDR_MOD_AB_SEC0_SrcBIncr, 0),
                        (thread::ADDR_MOD_AB_SEC0_SrcBClear, 0),
                    ],
                    vec![
                        (thread::ADDR_MOD_DST_SEC0_DestIncr, 0),
                        (thread::ADDR_MOD_DST_SEC0_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC0_FidelityClear, 1),
                    ],
                    // `ADDR_MOD_1`: one `SrcB` row and one `Dst` row.
                    vec![(thread::ADDR_MOD_AB_SEC1_SrcBIncr, 1)],
                    vec![(thread::ADDR_MOD_DST_SEC1_DestIncr, 1)],
                    // `ADDR_MOD_2`: eight of each.
                    vec![(thread::ADDR_MOD_AB_SEC2_SrcBIncr, 8)],
                    vec![(thread::ADDR_MOD_DST_SEC2_DestIncr, 8)],
                    // `ADDR_MOD_3`: no increments, the next phase.
                    vec![(thread::ADDR_MOD_AB_SEC3_SrcBIncr, 0)],
                    vec![
                        (thread::ADDR_MOD_DST_SEC3_DestIncr, 0),
                        (thread::ADDR_MOD_DST_SEC3_FidelityIncr, 1),
                    ],
                    vec![(thread::FIDELITY_BASE_Phase, 0)],
                ] {
                    m.push(Core::Math, thread_word(&entry)?);
                }
                m.push(
                    Core::Math,
                    MirOp::Setrwc {
                        src_a: 0,
                        src_b: 0,
                        dst: 0,
                        flip_src_a: false,
                        flip_src_b: false,
                        dst_cr: false,
                    },
                );
                if dim == crate::ir::ReduceDim::Row {
                    // tt-metal's `_llk_math_reduce_` for `REDUCE_ROW`: each
                    // face pair folds into one row of the accumulator (the
                    // haloized unpack presents the face transposed, so the fold
                    // runs along the tile's original rows), and
                    // `reduce_row_perform_transpose` turns that row into the
                    // result's column 0. Only `max` reaches this arm: `sum row`
                    // is the Matrix Unit's matmul above.
                    m.push(
                        Core::Unpack,
                        MirOp::UnpackConfig {
                            unit: 0,
                            to_src: true,
                            dtype: dt_x,
                            haloize: true,
                            add_dest_cntr: false,
                            ch1_z_stride: 0,
                        },
                    );
                    m.push(
                        Core::Unpack,
                        MirOp::UnpackConfig {
                            unit: 1,
                            to_src: true,
                            dtype: dt_b,
                            haloize: false,
                            add_dest_cntr: false,
                            ch1_z_stride: 0,
                        },
                    );
                    for group in 0..2u32 {
                        let dst_off = 32 * group;
                        for half in 0..2u32 {
                            let face = 2 * group + half;
                            // One face into `SrcA` and one row of the scaler
                            // into `SrcB` (16 datums): tt-metal's AB-reduce
                            // unpack.
                            m.push(
                                Core::Unpack,
                                MirOp::Setadczw {
                                    u0: true,
                                    u1: false,
                                    z0: Some(face),
                                    z1: None,
                                },
                            );
                            m.push(Core::Unpack, MirOp::SetUnpackBase { cb: cb_x, unit: 0 });
                            m.push(
                                Core::Unpack,
                                MirOp::Unpacr {
                                    unit: 0,
                                    count: FACE_DATUMS,
                                    flip: true,
                                },
                            );
                            m.push(
                                Core::Unpack,
                                MirOp::Setadczw {
                                    u0: false,
                                    u1: true,
                                    z0: Some(face),
                                    z1: None,
                                },
                            );
                            m.push(Core::Unpack, MirOp::SetUnpackBase { cb: cb_b, unit: 1 });
                            m.push(
                                Core::Unpack,
                                MirOp::Unpacr {
                                    unit: 1,
                                    count: 16,
                                    flip: true,
                                },
                            );
                            // The first face of a pair hands both banks back
                            // (`CLR_AB`); the second leaves them, because
                            // `TRNSPSRCB` reads the `SrcB` bank the fold just
                            // consumed (tt-metal's `CLR_AB` then `CLR_NONE`).
                            m.push(
                                Core::Math,
                                MirOp::Gmpool {
                                    flip_src_b: half == 0,
                                    flip_src_a: half == 0,
                                    addr_mod: 0,
                                    arg_max: 0,
                                    reg: base,
                                    at: dst_off,
                                },
                            );
                        }
                        // `reduce_row_perform_transpose<true>`, op for op: the
                        // high 16 bits of the fold's row go to `SrcB` rows
                        // 16..31, are transposed there and come back to the
                        // `Dst`; then the low 16 bits make the same trip. The
                        // passes are separate because `DEST_NORM` and
                        // `DEST_32B_LOW` are the two halves of one 32-bit `Dst`
                        // datum, and each pass rewrites the half it moved.
                        //
                        // `Fp32_enabled` is cleared around the moves on
                        // Blackhole (tt-metal issue #449) and restored after;
                        // denormals stay enabled in `Src` so the low halves
                        // survive the moves. The sequence never touches `SrcA`,
                        // so there is no format switch here either.
                        if wide {
                            m.push(
                                Core::Math,
                                MirOp::AluAccCtrl {
                                    fp32: true,
                                    zero_flag_src: true,
                                },
                            );
                            // The high pass, measured correct on the card: the
                            // fold's row goes to `SrcB` rows 16..31, is
                            // transposed there, and comes back as `Dst` column
                            // 0. This silicon's `MOVD2B`/`MOVB2D` with the
                            // 32-bit low operand replaces the whole datum
                            // instead of merging into its low half, so the
                            // low-half pass is not emitted at all: the values
                            // arrive rounded to their high 16 bits.
                            m.push(
                                Core::Math,
                                MirOp::Movd2b {
                                    use_dst32b_lo: false,
                                    src_row: 16,
                                    addr_mod: 0,
                                    move4_rows: false,
                                    reg: base,
                                    at: dst_off,
                                },
                            );
                            m.push(Core::Math, MirOp::Trnspsrcb);
                            m.push(
                                Core::Math,
                                MirOp::Movd2b {
                                    use_dst32b_lo: false,
                                    src_row: 16,
                                    addr_mod: 0,
                                    move4_rows: false,
                                    reg: base,
                                    at: dst_off,
                                },
                            );
                            for chunk in 0..4u32 {
                                m.push(
                                    Core::Math,
                                    MirOp::Movb2d {
                                        use_dst32b_lo: false,
                                        src_row: 16 + 4 * chunk,
                                        addr_mod: 0,
                                        move4_rows: true,
                                        reg: base,
                                        at: dst_off + 4 * chunk,
                                    },
                                );
                            }
                            m.push(
                                Core::Math,
                                MirOp::AluAccCtrl {
                                    fp32: true,
                                    zero_flag_src: false,
                                },
                            );
                        } else {
                            // `reduce_row_perform_transpose<false>`, the arm
                            // tt-lang takes on Blackhole (full-fp32 row reduce
                            // is unavailable there, tt-metal #47311): a 16-bit
                            // `Dst` datum is one `SrcB` datum, so the fold's
                            // row is moved whole, transposed, and added back
                            // over a zeroed `SrcA` — no half split and no
                            // format switch.
                            m.push(
                                Core::Math,
                                MirOp::Movd2b {
                                    use_dst32b_lo: false,
                                    src_row: 16,
                                    addr_mod: 0,
                                    move4_rows: false,
                                    reg: base,
                                    at: dst_off,
                                },
                            );
                            m.push(Core::Math, MirOp::Trnspsrcb);
                            // tt-metal moves the row again after the transpose
                            // (its multi-tile accumulation). Row 16's column 0
                            // is the same datum before and after the move, and
                            // the pack keeps column 0, so the fold's values
                            // survive.
                            m.push(
                                Core::Math,
                                MirOp::Movd2b {
                                    use_dst32b_lo: false,
                                    src_row: 16,
                                    addr_mod: 0,
                                    move4_rows: false,
                                    reg: base,
                                    at: dst_off,
                                },
                            );
                            // The two `SETRWC(CR_B, +8)`s: the transposed block
                            // is `SrcB` rows 16..31, which is where the adds
                            // read from. A counter value is four bits, so the
                            // increment instruction is what reaches 16.
                            for _ in 0..2 {
                                m.push(
                                    Core::Math,
                                    MirOp::Incrwc {
                                        src_a: 0,
                                        src_b: 8,
                                        dst: 0,
                                    },
                                );
                            }
                            m.push(
                                Core::Math,
                                MirOp::Zerosrc {
                                    zero_val: false,
                                    write_mode: true,
                                    bank_mask: false,
                                    clear_src_b: false,
                                    clear_src_a: true,
                                },
                            );
                            // Two eight-row adds over the zeroed `SrcA` copy the
                            // transposed face; `ADDR_MOD_2` steps `SrcB` and
                            // `Dst` by eight rows.
                            for _ in 0..2 {
                                m.push(
                                    Core::Math,
                                    MirOp::Elw {
                                        op: ElwOp::Add,
                                        bcast: Bcast::None,
                                        addr_mod: 2,
                                        reg: base,
                                        at: dst_off,
                                    },
                                );
                            }
                        }
                        // `SETRWC(CLR_AB, 0, 0, 0, 0, SET_BD)`: hand both banks
                        // back before the next pair unpacks.
                        m.push(
                            Core::Math,
                            MirOp::Setrwc {
                                src_a: 0,
                                src_b: 0,
                                dst: 0,
                                flip_src_a: true,
                                flip_src_b: true,
                                dst_cr: false,
                            },
                        );
                    }
                    m.push(Core::Math, MirOp::WaitMatrix);
                    // The pack's edge mask keeps column 0 of every row, and the
                    // fold's column lands at rows 0..15 and 32..47; the rows
                    // between (16..31 and 48..63) would pack stale datums, so
                    // they are written as the zeros the result defines.
                    m.push(
                        Core::Math,
                        MirOp::LaneConst {
                            lreg: LReg(0),
                            bits: 0.0f32.to_bits(),
                        },
                    );
                    for row in (16..32).chain(48..DST_ROWS).step_by(2) {
                        m.push(
                            Core::Math,
                            MirOp::Sfpstore {
                                lreg: LReg(0),
                                reg: base,
                                at: row,
                                dtype: storage(crate::ir::DType::F32),
                            },
                        );
                    }
                    m.push(
                        Core::Math,
                        MirOp::Setrwc {
                            src_a: 0,
                            src_b: 0,
                            dst: 0,
                            flip_src_a: false,
                            flip_src_b: false,
                            dst_cr: false,
                        },
                    );
                    acc_mask.insert(acc, PackMask::Row);
                } else {
                    // Four faces, `SrcA` (256 datums) and the scaler's row (16
                    // datums) each: tt-metal's AB-reduce unpack, then the fold.
                    // The scaler carries the factor in row 0 of each face
                    // (`prepare_reduce_scaler`'s row-0 fill).
                    for face in 0..4u32 {
                        // One face into `SrcA` and one row of the scaler into
                        // `SrcB` (16 datums): tt-metal's AB-reduce unpack.
                        m.push(
                            Core::Unpack,
                            MirOp::Setadczw {
                                u0: true,
                                u1: false,
                                z0: Some(face),
                                z1: None,
                            },
                        );
                        m.push(
                            Core::Unpack,
                            MirOp::UnpackConfig {
                                unit: 0,
                                to_src: true,
                                dtype: dt_x,
                                haloize: false,
                                add_dest_cntr: false,
                                ch1_z_stride: 0,
                            },
                        );
                        m.push(Core::Unpack, MirOp::SetUnpackBase { cb: cb_x, unit: 0 });
                        m.push(
                            Core::Unpack,
                            MirOp::Unpacr {
                                unit: 0,
                                count: FACE_DATUMS,
                                flip: true,
                            },
                        );
                        m.push(
                            Core::Unpack,
                            MirOp::Setadczw {
                                u0: false,
                                u1: true,
                                z0: Some(face),
                                z1: None,
                            },
                        );
                        m.push(
                            Core::Unpack,
                            MirOp::UnpackConfig {
                                unit: 1,
                                to_src: true,
                                dtype: dt_b,
                                haloize: false,
                                add_dest_cntr: false,
                                ch1_z_stride: 0,
                            },
                        );
                        m.push(Core::Unpack, MirOp::SetUnpackBase { cb: cb_b, unit: 1 });
                        m.push(
                            Core::Unpack,
                            MirOp::Unpacr {
                                unit: 1,
                                count: 16,
                                flip: true,
                            },
                        );
                        // Faces 0 and 2 fold into the first half of row 0, faces 1
                        // and 3 into the second (tt-metal's `SETRWC(CR_D, 8)`
                        // twice).
                        let dst_off = if face % 2 == 0 { 0u32 } else { 16 };
                        if max {
                            m.push(
                                Core::Math,
                                MirOp::Gmpool {
                                    flip_src_b: false,
                                    flip_src_a: false,
                                    addr_mod: 0,
                                    arg_max: 0,
                                    reg: base,
                                    at: dst_off,
                                },
                            );
                        } else {
                            // Four fidelity phases, the last clearing the phase
                            // (tt-metal's `reduce_configure_mop`).
                            for addr_mod in [3u32, 3, 3, 0] {
                                m.push(
                                    Core::Math,
                                    MirOp::Gapool {
                                        flip_src_b: false,
                                        flip_src_a: false,
                                        addr_mod,
                                        reg: base,
                                        at: dst_off,
                                    },
                                );
                            }
                        }
                        // `SETRWC(CLR_AB, 0, 0, 0, 0, SET_AD)`: hand both banks back
                        // and leave the counters where the face started.
                        m.push(
                            Core::Math,
                            MirOp::Setrwc {
                                src_a: 0,
                                src_b: 0,
                                dst: 0,
                                flip_src_a: true,
                                flip_src_b: true,
                                dst_cr: false,
                            },
                        );
                    }
                    m.push(Core::Math, MirOp::WaitMatrix);
                    // The pack's edge mask keeps the first row of *every* face,
                    // and the fold only writes faces 0 and 1; the packer cannot
                    // select faces, so faces 2 and 3 (the lower half of the tile)
                    // are written as the zeros the result defines.
                    m.push(
                        Core::Math,
                        MirOp::LaneConst {
                            lreg: LReg(0),
                            bits: 0.0f32.to_bits(),
                        },
                    );
                    for row in (32..DST_ROWS).step_by(2) {
                        m.push(
                            Core::Math,
                            MirOp::Sfpstore {
                                lreg: LReg(0),
                                reg: base,
                                at: row,
                                dtype: storage(crate::ir::DType::F32),
                            },
                        );
                    }
                    m.push(
                        Core::Math,
                        MirOp::Setrwc {
                            src_a: 0,
                            src_b: 0,
                            dst: 0,
                            flip_src_a: false,
                            flip_src_b: false,
                            dst_cr: false,
                        },
                    );
                    acc_mask.insert(acc, PackMask::Col);
                }
                // A staged operand's page is the compiler's: release it once
                // the fold has drained it.
                for (cb, staged) in [(cb_x, stage_x), (cb_b, stage_b)] {
                    if staged {
                        m.push(Core::Unpack, MirOp::CbPop { cb, n: 1 });
                    }
                }
            }
            // The arms above cover every `Op`; the matmul/sum-row one is
            // guarded, so the compiler cannot see it does, and a new variant
            // lands here loudly rather than in a silent default.
            _ => {
                return Err(crate::status::internal_bug(
                    "lower: an op reached the catch-all arm",
                ));
            }
        }
    }
    // The compiler's 1.0 page: written once, at the start of the kernel, into
    // the last `Dst` slot (the allocator keeps it free), and packed into the
    // page a `max` reduce unpacks. The fills cover the face rows the packer
    // copies into each L1 face's row 0; the rest of the packed tile is the
    // scratch slot's old contents, which nothing reads.
    if let Some(ones) = plan.ones {
        let reg = m.dsts.push(crate::mir::DstInfo {
            dtype: ones.dtype,
            shape: [32, 32],
        });
        let ones_cb = cb_id(ones.cb);
        let mut prologue: Vec<(Core, MirOp)> = vec![(
            Core::Math,
            MirOp::LaneConst {
                lreg: LReg(0),
                bits: 1.0f32.to_bits(),
            },
        )];
        for row_off in (0..DST_ROWS).step_by(2) {
            prologue.push((
                Core::Math,
                MirOp::Sfpstore {
                    lreg: LReg(0),
                    reg,
                    at: row_off,
                    dtype: storage(ones.dtype),
                },
            ));
        }
        for op in [
            MirOp::CbReserve { cb: ones_cb, n: 1 },
            MirOp::SetPackBase { cb: ones_cb },
            MirOp::SetPackConfig {
                dtype: ones.dtype,
                mask: PackMask::None,
            },
            MirOp::SetPackOffset { reg, at: 0 },
            MirOp::PackRows {
                rows: DST_ROWS,
                last: true,
            },
            MirOp::CbPush { cb: ones_cb, n: 1 },
        ] {
            prologue.push((Core::Pack, op));
        }
        // The ones page goes before every op the lowering emitted.
        m.ops.splice(0..0, prologue);
    }
    Ok(m)
}

/// `MVMUL`'s `ADDR_MOD` entries, one per column-group boundary: the next
/// fidelity phase, the end of a first `SrcB` half, the end of a group, and the
/// end of an `fi`.
const MATMUL_AM_PHASE: u32 = 1;
const MATMUL_AM_HALF: u32 = 2;
const MATMUL_AM_GROUP: u32 = 3;
const MATMUL_AM_FI: u32 = 4;

/// One `MVMUL` group into `Dst` rows `base..base + 16`: both `SrcB` halves and
/// every fidelity phase, all but the group's last multiply, which the caller
/// issues with its flips.
fn mvmul_group(role: &mut Vec<MirOp>, reg: Dst, at: u32) {
    for half in 0..2u32 {
        let phases = if half == 0 { 4 } else { 3 };
        for phase in 0..phases {
            let addr_mod = if half == 0 && phase + 1 == 4 {
                MATMUL_AM_HALF
            } else {
                MATMUL_AM_PHASE
            };
            role.push(MirOp::Mvmul {
                reg,
                at: at + 8 * half,
                addr_mod,
                flip_a: false,
                flip_b: false,
            });
        }
    }
}

/// tt-metal's `transpose_dest_32b<false>`: transpose `reg`'s four 16x16 faces
/// within themselves, in place. The face arrangement is the unpacker's job
/// (`transpose_of_faces`), which is why `transpose_tile` calls this arm for a
/// 32-bit `Dst`.
///
/// Not called yet: the fp32 `Op::Transpose` arm is `TT_TODO_TRANSPOSE` because
/// ttsim refuses the sequence's first `MOVD2B` (`tf32 use_dst32b=1
/// dest_32b_lo=1`, Blackhole erratum TEN-4245), so it has not been run here.
/// The transcription is kept for the card-side arm.
#[allow(dead_code)]
fn transpose_dst(m: &mut MachineKernel, reg: Dst) -> Result<(), ttStatus> {
    // `ADDR_MOD_1`: nothing moves. Both `Config` words go out whole, because
    // `Config` outlives the program that wrote it.
    m.push(
        Core::Math,
        MirOp::ThreadWord {
            addr: thread::ADDR_MOD_AB_SEC0_SrcAIncr.addr32() + 1,
            value: 0,
        },
    );
    m.push(
        Core::Math,
        MirOp::ThreadWord {
            addr: thread::ADDR_MOD_DST_SEC0_DestIncr.addr32() + 1,
            value: 0,
        },
    );
    // `DISABLE_IMPLIED_SRCA_FMT`: the manual `SrcA` format switches below only
    // take effect with the inferred format off.
    m.push(
        Core::Math,
        MirOp::ThreadWord {
            addr: thread::DISABLE_IMPLIED_SRCA_FMT_Base.addr32(),
            value: 1,
        },
    );
    // PRESERVE source denormals: the halves are raw 16-bit images, not numbers.
    m.push(
        Core::Math,
        MirOp::AluAccCtrl {
            fp32: true,
            zero_flag_src: true,
        },
    );
    // The counters this thread inherits are whatever the previous op left;
    // every row below is absolute.
    m.push(
        Core::Math,
        MirOp::Setrwc {
            src_a: 0,
            src_b: 0,
            dst: 0,
            flip_src_a: false,
            flip_src_b: false,
            dst_cr: false,
        },
    );
    let hold = 1u32;
    // The low 16 bits of every face go to `SrcA`, transposed. tt-metal's
    // `ckernel_template::run()` walks the four faces; the `srca` operand steps
    // by sixteen per face.
    m.push(Core::Math, MirOp::SrcAFormat { format: 5 });
    for face in 0..4u32 {
        for (i, src) in [16u32, 20, 24, 28].into_iter().enumerate() {
            m.push(
                Core::Math,
                MirOp::Movd2b {
                    use_dst32b_lo: true,
                    src_row: src,
                    addr_mod: hold,
                    move4_rows: true,
                    reg,
                    at: face * 16 + 4 * i as u32,
                },
            );
        }
        m.push(Core::Math, MirOp::Trnspsrcb);
        for (i, src) in [16u32, 20, 24, 28].into_iter().enumerate() {
            m.push(
                Core::Math,
                MirOp::Movb2a {
                    src_a_row: face * 16 + 4 * i as u32,
                    addr_mod: hold,
                    move4_rows: true,
                    src_b_row: src,
                },
            );
        }
    }
    // tt-metal's `!transpose_of_faces` arm: the unpacker has already swapped
    // the faces, so each face is only transposed within itself. `SrcAFormat`
    // is Tf32 for the half moves, as in `transpose_dest_32b<false>`.
    m.push(Core::Math, MirOp::SrcAFormat { format: 4 });
    for face in 0..4u32 {
        for (i, src) in [16u32, 20, 24, 28].into_iter().enumerate() {
            m.push(
                Core::Math,
                MirOp::Movd2b {
                    use_dst32b_lo: true,
                    src_row: src,
                    addr_mod: hold,
                    move4_rows: true,
                    reg,
                    at: face * 16 + 4 * i as u32,
                },
            );
        }
        m.push(Core::Math, MirOp::Trnspsrcb);
        for (i, src) in [16u32, 20, 24].into_iter().enumerate() {
            m.push(
                Core::Math,
                MirOp::Movb2a {
                    src_a_row: 4 * i as u32,
                    addr_mod: hold,
                    move4_rows: true,
                    src_b_row: src,
                },
            );
        }
        m.push(
            Core::Math,
            MirOp::Movb2d {
                use_dst32b_lo: false,
                src_row: 28,
                addr_mod: hold,
                move4_rows: true,
                reg,
                at: face * 16 + 12,
            },
        );
    }
    // All the high halves are back; write the low halves from `SrcA`.
    m.push(
        Core::Math,
        MirOp::AluAccCtrl {
            fp32: false,
            zero_flag_src: true,
        },
    );
    m.push(Core::Math, MirOp::SrcAFormat { format: 0 });
    for band in 0..8u32 {
        m.push(
            Core::Math,
            MirOp::Mova2d {
                use_dst32b_lo: true,
                src_row: 8 * band,
                addr_mod: hold,
                move8_rows: true,
                reg,
                at: 8 * band,
            },
        );
    }
    // Restore the state the rest of the program expects.
    m.push(
        Core::Math,
        MirOp::ThreadWord {
            addr: thread::DISABLE_IMPLIED_SRCA_FMT_Base.addr32(),
            value: 0,
        },
    );
    m.push(
        Core::Math,
        MirOp::AluAccCtrl {
            fp32: true,
            zero_flag_src: false,
        },
    );
    // `SETRWC(CLR_AB, SET_ABD)`: hand both source banks back.
    m.push(
        Core::Math,
        MirOp::Setrwc {
            src_a: 0,
            src_b: 0,
            dst: 0,
            flip_src_a: true,
            flip_src_b: true,
            dst_cr: false,
        },
    );
    Ok(())
}

/// One whole `SETC16` `ThreadConfig` word composed from `fields`.
fn thread_word(fields: &[(tt_isa::cfg::ThreadConfigField, u16)]) -> Result<MirOp, ttStatus> {
    let Some(first) = fields.first() else {
        return Err(crate::status::internal_bug(
            "lower: a thread config has no fields",
        ));
    };
    let mut entry = tt_isa::backend::ThreadConfigEntry::zeroed(first.0.addr32());
    for &(field, value) in fields {
        entry = match entry.set(field, value) {
            Ok(entry) => entry,
            Err(_) => {
                return Err(crate::status::internal_bug(format_args!(
                    "lower: thread config value {value} does not fit {field:?}"
                )));
            }
        };
    }
    Ok(MirOp::ThreadWord {
        addr: entry.addr32(),
        value: entry.value(),
    })
}

/// The CB an `Src`-read operand takes, and whether it had to be staged: a
/// `load` already has a page, while a computed value is packed into the
/// compiler's page for it (see [`stage`]). `err` is the operand rule the
/// calling op reports when the value has neither.
fn fpu_page(
    k: &Kernel,
    plan: &Plan,
    value: crate::ir::OpId,
    at: OpId,
    err: ttStatus,
) -> Result<(Cb, bool), ttStatus> {
    let id = owner(k, value);
    if k.op(id).is_none() {
        let _ = (at, value);
        return Err(err);
    }
    if let Some(Op::Load { buffer, .. }) = k.op(id) {
        return Ok((
            cb_id(plan.cb.get(buffer).copied().ok_or_else(|| {
                let _ = buffer;
                ttStatus::TT_ERROR_BUFFER_NOT_PLACED
            })?),
            false,
        ));
    }
    // Not a `load`: the value lives in `Dst` and the `Src` path reads L1, so
    // the compiler packs it into a page it owns. The verifier admits only
    // f32/bf16 operands, which is also what a staging `Pack` takes.
    let _ = at;
    let cb = cb_id(plan.staged.get(&id).copied().ok_or_else(|| {
        let _ = (id, value);
        err
    })?);
    Ok((cb, true))
}

/// `SrcA op SrcB -> Dst` with `SrcB` broadcast: tt-metal's
/// `eltwise_binary_configure_addrmod` and its `ELWADD`/`ELWSUB`/`ELWMUL` MOP.
/// The tile is `SrcA` and the narrow value `SrcB`; `ADDR_MOD_0` steps `SrcA` and
/// `Dst` eight rows per instruction and, except for a row or scalar broadcast,
/// steps `SrcB` too, and the end op hands both banks back.
///
/// Both operands are unpacked from L1 pages: a `load`'s page, or the compiler's
/// staged page for a computed value (or for an `acc`, which an FPU op never
/// reads from `Dst`).
#[allow(clippy::too_many_arguments)]
fn fpu_broadcast(
    m: &mut MachineKernel,
    k: &Kernel,
    types: &Types,
    plan: &Plan,
    alloc: &Alloc,
    acc_mask: &BTreeMap<OpId, PackMask>,
    value: crate::ir::OpId,
    op: BOp,
    tile: crate::ir::OpId,
    narrow: crate::ir::OpId,
    bcast: Bcast,
    at: OpId,
) -> Result<(), ttStatus> {
    let (cb_a, stage_a) = fpu_page(k, &plan, tile, at, ttStatus::TT_TODO_BROADCAST)?;
    let (cb_b, stage_b) = fpu_page(k, &plan, narrow, at, ttStatus::TT_TODO_BROADCAST)?;
    if stage_a {
        stage(m, k, types, alloc, acc_mask, tile, cb_a)?;
    }
    if stage_b {
        stage(m, k, types, alloc, acc_mask, narrow, cb_b)?;
    }
    let dt_a = types.of(tile).dtype;
    let dt_b = types.of(narrow).dtype;
    let out = dst_reg(alloc, owner(k, value))?;
    // The `Src` banks are filled from the pages the unpacker owns.
    for (cb, staged) in [(cb_a, stage_a), (cb_b, stage_b)] {
        if staged {
            m.push(Core::Unpack, MirOp::CbWait { cb, n: 1 });
        }
    }
    // The counters start at 0 and `ADDR_MOD_0` steps `SrcA` and `Dst` eight
    // rows per instruction, and `SrcB` too unless the broadcast holds it still.
    m.push(
        Core::Math,
        MirOp::Setrwc {
            src_a: 0,
            src_b: 0,
            dst: 0,
            flip_src_a: false,
            flip_src_b: false,
            dst_cr: false,
        },
    );
    m.push(
        Core::Math,
        thread_word(&[
            (thread::ADDR_MOD_AB_SEC0_SrcAIncr, 8),
            (
                thread::ADDR_MOD_AB_SEC0_SrcBIncr,
                match bcast {
                    Bcast::Col => 8u16,
                    Bcast::Row | Bcast::Scalar => 0,
                    Bcast::None => {
                        return Err(crate::status::internal_bug(
                            "lower: the FPU broadcast has no broadcast mode",
                        ));
                    }
                },
            ),
            (thread::ADDR_MOD_AB_SEC0_SrcBClear, 0),
            (thread::ADDR_MOD_AB_SEC0_SrcAClear, 0),
        ])?,
    );
    m.push(
        Core::Math,
        thread_word(&[
            (thread::ADDR_MOD_DST_SEC0_DestIncr, 0),
            (thread::ADDR_MOD_DST_SEC0_DestCR, 0),
            // The last phase of a multiply hands the phase counter back to 0.
            (thread::ADDR_MOD_DST_SEC0_FidelityClear, 1),
        ])?,
    );
    // `ADDR_MOD_1`: the next fidelity phase of the same eight rows — no
    // counter moves, only the phase, so the instruction re-multiplies the same
    // operands' next mantissa slice into the same `Dst` rows.
    m.push(
        Core::Math,
        thread_word(&[
            (thread::ADDR_MOD_AB_SEC1_SrcAIncr, 0),
            (thread::ADDR_MOD_AB_SEC1_SrcBIncr, 0),
            (thread::ADDR_MOD_AB_SEC1_SrcBClear, 0),
            (thread::ADDR_MOD_AB_SEC1_SrcAClear, 0),
        ])?,
    );
    m.push(
        Core::Math,
        thread_word(&[
            (thread::ADDR_MOD_DST_SEC1_DestIncr, 0),
            (thread::ADDR_MOD_DST_SEC1_DestCR, 0),
            (thread::ADDR_MOD_DST_SEC1_FidelityIncr, 1),
        ])?,
    );
    // The section is written, never accumulated into; `reg_sync` places the wait
    // for a packer that still reads it.
    // `ELWMUL`'s word carries bit 21 set, so the hardware accumulates into the
    // destination (tt-isa's measured encoding; `ELWADD`/`ELWSUB` overwrite).
    // That is also how the four fidelity phases add up, so the result section is
    // zeroed first and the phases accumulate onto nothing.
    if matches!(op, BOp::Mul) {
        m.push(
            Core::Math,
            MirOp::LaneConst {
                lreg: LReg(0),
                bits: 0.0f32.to_bits(),
            },
        );
        for row in (0..DST_ROWS).step_by(2) {
            m.push(
                Core::Math,
                MirOp::Sfpstore {
                    lreg: LReg(0),
                    reg: out,
                    at: row,
                    dtype: if m.dst_is_16_bit {
                        crate::ir::DType::Bf16
                    } else {
                        crate::ir::DType::F32
                    },
                },
            );
        }
    }
    for face in 0..4u32 {
        m.push(
            Core::Unpack,
            MirOp::Setadczw {
                u0: true,
                u1: false,
                z0: Some(face),
                z1: None,
            },
        );
        m.push(
            Core::Unpack,
            MirOp::UnpackConfig {
                unit: 0,
                to_src: true,
                dtype: dt_a,
                haloize: false,
                add_dest_cntr: false,
                ch1_z_stride: 0,
            },
        );
        m.push(Core::Unpack, MirOp::SetUnpackBase { cb: cb_a, unit: 0 });
        m.push(
            Core::Unpack,
            MirOp::Unpacr {
                unit: 0,
                count: FACE_DATUMS,
                flip: true,
            },
        );
        // `SrcB`: the page face the FPU reads for `SrcA`'s face. A narrow
        // value's page keeps only its own part, so both halves of the tile read
        // the same face: a row's faces 0 and 1 (row 0) serve the tile's four
        // faces two at a time, and a column's faces 0 and 2 (column 0) serve
        // each pair of face rows.
        let b_face = match bcast {
            Bcast::Row => face & 1,
            Bcast::Col => face & !1,
            Bcast::Scalar => 0,
            Bcast::None => {
                return Err(crate::status::internal_bug(
                    "lower: the FPU broadcast has no broadcast mode",
                ));
            }
        };
        m.push(
            Core::Unpack,
            MirOp::Setadczw {
                u0: false,
                u1: true,
                z0: Some(b_face),
                z1: None,
            },
        );
        m.push(
            Core::Unpack,
            MirOp::UnpackConfig {
                unit: 1,
                to_src: true,
                dtype: dt_b,
                haloize: false,
                add_dest_cntr: false,
                ch1_z_stride: 0,
            },
        );
        m.push(Core::Unpack, MirOp::SetUnpackBase { cb: cb_b, unit: 1 });
        m.push(
            Core::Unpack,
            MirOp::Unpacr {
                unit: 1,
                count: FACE_DATUMS,
                flip: true,
            },
        );
        // One face is two instructions of eight rows. `ELWMUL` is the Matrix
        // Unit's *sliced* multiply: one pass takes the top four mantissa bits of
        // each operand and each further phase adds the next slice into the same
        // `Dst` rows, so a single pass is a five-bit product. tt-metal's
        // `MathFidelity` counts those passes and its softmax configures
        // `HiFi4`; `ELWADD`/`ELWSUB` are one pass at any fidelity, which is why
        // only the multiply repeats. The last pass clears the phase and steps
        // the counters on to the next eight rows.
        let phases: u32 = if matches!(op, BOp::Mul) { 4 } else { 1 };
        for half in 0..2u32 {
            for phase in 0..phases {
                m.push(
                    Core::Math,
                    MirOp::Elw {
                        op: match op {
                            BOp::Add => ElwOp::Add,
                            BOp::Sub => ElwOp::Sub,
                            BOp::Mul => ElwOp::Mul,
                            other => {
                                return Err(crate::status::internal_bug(format_args!(
                                    "lower: an SFPU broadcast of {other:?}"
                                )));
                            }
                        },
                        bcast,
                        addr_mod: if phase + 1 == phases { 0 } else { 1 },
                        reg: out,
                        at: 16 * face + 8 * half,
                    },
                );
            }
        }
        // `SETRWC(CLR_AB, .., SET_AB)`: the banks go back before the next
        // face's unpack, which is the only way the unpacker may refill them
        // (tt-metal's reduce clears after every face for the same reason). The
        // unpacker refills each face from the bank's start, so the counters stay
        // at 0 and the `SFPU` afterwards addresses `Dst` relative to 0.
        m.push(
            Core::Math,
            MirOp::Setrwc {
                src_a: 0,
                src_b: 0,
                dst: 0,
                flip_src_a: true,
                flip_src_b: true,
                dst_cr: false,
            },
        );
    }
    // The counters are back to 0 for the `SFPU`, which addresses `Dst` relative
    // to them.
    m.push(
        Core::Math,
        MirOp::Setrwc {
            src_a: 0,
            src_b: 0,
            dst: 0,
            flip_src_a: false,
            flip_src_b: false,
            dst_cr: false,
        },
    );
    for (cb, staged) in [(cb_a, stage_a), (cb_b, stage_b)] {
        if staged {
            m.push(Core::Unpack, MirOp::CbPop { cb, n: 1 });
        }
    }
    Ok(())
}

/// Pack a computed value's `Dst` rows into the compiler's page for it, so the
/// FPU can read it from `Src`. This is the store path's hand-off without a
/// caller buffer: the math core publishes the section, the packer drains it
/// into the page and publishes the page, and the unpacker waits for that page.
fn stage(
    m: &mut MachineKernel,
    k: &Kernel,
    types: &Types,
    alloc: &Alloc,
    acc_mask: &BTreeMap<OpId, PackMask>,
    value: crate::ir::OpId,
    cb: Cb,
) -> Result<(), ttStatus> {
    let id = owner(k, value);
    let dtype = types.of(id).dtype;
    m.push(Core::Pack, MirOp::CbReserve { cb, n: 1 });
    m.push(Core::Pack, MirOp::SetPackBase { cb });
    m.push(
        Core::Pack,
        MirOp::SetPackConfig {
            dtype,
            mask: acc_mask[&id],
        },
    );
    m.push(
        Core::Pack,
        MirOp::SetPackOffset {
            reg: dst_reg(alloc, id)?,
            at: 0,
        },
    );
    m.push(
        Core::Pack,
        MirOp::PackRows {
            rows: DST_ROWS,
            last: true,
        },
    );
    m.push(Core::Pack, MirOp::CbPush { cb, n: 1 });
    Ok(())
}

/// The literal trip count of a bound, if it has one.
fn bound_const(k: &Kernel, bound: OpId) -> Option<u32> {
    match k.const_i32(bound) {
        Some(n) if n >= 0 => Some(n as u32),
        _ => None,
    }
}

/// Emit `dim` into `dest` as role RISC-V integer ops: a literal, or an index
/// expression over literals, scalar parameters, grid coordinates, enclosing
/// loop indices and ordinary arithmetic.
#[allow(clippy::too_many_arguments)]
fn scalar_with(
    k: &Kernel,
    types: &Types,
    role: &mut Vec<MirOp>,
    param: &BTreeMap<OpId, u32>,
    loops: &[crate::ir::OpId],
    dim: OpId,
    dest: u8,
    scratch: &mut Pool,
    counters: &[u8],
    plan: &Plan,
    cbs: &crate::slab::Slab<Cb, crate::mir::CbInfo>,
) -> Result<(), ttStatus> {
    eval(
        k, types, role, param, loops, dim, dest, scratch, counters, plan, cbs,
    )
}

/// Emit the value `value` into `dest`; `scratch` supplies the temporaries a
/// right-hand operand needs.
#[allow(clippy::too_many_arguments)]
fn eval(
    k: &Kernel,
    types: &Types,
    role: &mut Vec<MirOp>,
    param: &BTreeMap<OpId, u32>,
    loops: &[crate::ir::OpId],
    value: crate::ir::OpId,
    dest: u8,
    scratch: &mut Pool,
    counters: &[u8],
    plan: &Plan,
    cbs: &crate::slab::Slab<Cb, crate::mir::CbInfo>,
) -> Result<(), ttStatus> {
    // An enclosing loop's induction value is in that loop's counter register.
    if let Some(depth) = loops.iter().rposition(|v| *v == value) {
        let a = *counters.get(depth).ok_or(ttStatus::TT_TODO_LOOP_DEPTH)?;
        role.push(MirOp::IArith {
            op: IOp::Add,
            d: dest,
            a,
            b: reg::ZERO,
        });
        return Ok(());
    }
    let op = k.op(value).ok_or(ttStatus::TT_TODO_LOOP_ADDRESS)?;
    match *op {
        Op::Const {
            dtype: crate::ir::DType::I32,
            bits,
            ..
        } => {
            role.push(MirOp::Li {
                reg: dest,
                value: bits,
            });
            Ok(())
        }
        Op::Variable { .. } => {
            let index = *param.get(&value).ok_or(ttStatus::TT_TODO_LOOP_ADDRESS)?;
            let arg = *plan
                .arg_block
                .get(index as usize)
                .ok_or(ttStatus::TT_TODO_LOOP_ADDRESS)?;
            role.push(MirOp::LoadArg { reg: dest, arg });
            Ok(())
        }
        Op::GridCoord { axis, .. } => {
            role.push(MirOp::LoadCoord { reg: dest, axis });
            Ok(())
        }
        // A comparison is index arithmetic too: it produces a `bool` scalar,
        // which is `1`/`0` in a GPR (a datapath lane is `-1`/`0`), so the
        // cast to an index is a rename and the branch is a plain `beq`.
        Op::Binary { op, a, b, .. } if op.returns_bool() => {
            eval(
                k, types, role, param, loops, a, dest, scratch, counters, plan, cbs,
            )?;
            let right = scratch.take()?;
            eval(
                k, types, role, param, loops, b, right, scratch, counters, plan, cbs,
            )?;
            match op {
                BOp::Cmplt => role.push(MirOp::IArith {
                    op: IOp::Slt,
                    d: dest,
                    a: dest,
                    b: right,
                }),
                BOp::Cmpgt => role.push(MirOp::IArith {
                    op: IOp::Slt,
                    d: dest,
                    a: right,
                    b: dest,
                }),
                // `a >= b` is `!(a < b)`.
                BOp::Cmpge => {
                    role.push(MirOp::IArith {
                        op: IOp::Slt,
                        d: dest,
                        a: dest,
                        b: right,
                    });
                    role.push(MirOp::IArith {
                        op: IOp::Eq,
                        d: dest,
                        a: dest,
                        b: reg::ZERO,
                    });
                }
                BOp::Eq => role.push(MirOp::IArith {
                    op: IOp::Eq,
                    d: dest,
                    a: dest,
                    b: right,
                }),
                BOp::NotEq => role.push(MirOp::IArith {
                    op: IOp::Ne,
                    d: dest,
                    a: dest,
                    b: right,
                }),
                other => {
                    let _ = other;
                    return Err(ttStatus::TT_TODO_LOOP_ADDRESS);
                }
            }
            scratch.give(right);
            Ok(())
        }
        Op::Binary { op, a, b, .. } => {
            let op = int_op(op).ok_or(ttStatus::TT_TODO_LOOP_ADDRESS)?;
            eval(
                k, types, role, param, loops, a, dest, scratch, counters, plan, cbs,
            )?;
            let right = scratch.take()?;
            eval(
                k, types, role, param, loops, b, right, scratch, counters, plan, cbs,
            )?;
            role.push(MirOp::IArith {
                op,
                d: dest,
                a: dest,
                b: right,
            });
            scratch.give(right);
            Ok(())
        }
        Op::Cast { src, dtype } => {
            eval(
                k, types, role, param, loops, src, dest, scratch, counters, plan, cbs,
            )?;
            match (k.info(src).dtype, dtype) {
                // A bool scalar is already `1`/`0`.
                (DType::Bool, DType::Bool) | (DType::Bool, DType::I32) => {}
                // `(bool)x` is `x != 0`, exactly the datapath's cast.
                (DType::I32, DType::Bool) => role.push(MirOp::IArith {
                    op: IOp::Ne,
                    d: dest,
                    a: dest,
                    b: reg::ZERO,
                }),
                (DType::I32, DType::I32) => {}
                (from, to) => {
                    let _ = (from, to);
                    return Err(ttStatus::TT_TODO_LOOP_ADDRESS);
                }
            }
            Ok(())
        }
        Op::Select {
            cond,
            if_true,
            if_false,
        } => {
            let c = scratch.take()?;
            eval(
                k, types, role, param, loops, cond, c, scratch, counters, plan, cbs,
            )?;
            let f = scratch.take()?;
            eval(
                k, types, role, param, loops, if_false, f, scratch, counters, plan, cbs,
            )?;
            eval(
                k, types, role, param, loops, if_true, dest, scratch, counters, plan, cbs,
            )?;
            role.push(MirOp::Cmov {
                cond: c,
                a: dest,
                b: f,
                d: dest,
            });
            scratch.give(f);
            scratch.give(c);
            Ok(())
        }
        // A value the program put in a page: the movers read it out of L1, one
        // word. The page's address is known because the rings are placed before
        // the walk; a ring deeper than one page moves at run time, so its page
        // is refused rather than read from the wrong one.
        Op::Load { buffer, offset, .. } => {
            let (dtype, len) = match *k.at(buffer) {
                Op::Cb { dtype, len, .. } => (dtype, len),
                _ => {
                    let _ = buffer;
                    return Err(ttStatus::TT_ERROR_NOT_A_BUFFER);
                }
            };
            if len != 1 {
                let _ = (at_len(len), buffer);
                return Err(ttStatus::TT_TODO_LOOP_ADDRESS);
            }
            let cb = *plan
                .cb
                .get(&buffer)
                .ok_or(ttStatus::TT_ERROR_BUFFER_NOT_PLACED)?;
            let ring = cbs.get(cb).ok_or(ttStatus::TT_ERROR_BUFFER_NOT_PLACED)?;
            let el = dtype.bytes();
            match k.const_i32(offset) {
                Some(n) if n >= 0 => {
                    let bytes = u32::try_from(n)
                        .ok()
                        .and_then(|n| n.checked_mul(el))
                        .ok_or(ttStatus::TT_ERROR_PARAM_OFFSET_OVERFLOW)?;
                    role.push(MirOp::Li {
                        reg: dest,
                        value: ring
                            .base
                            .checked_add(bytes)
                            .ok_or(ttStatus::TT_ERROR_PARAM_OFFSET_OVERFLOW)?,
                    });
                }
                _ => {
                    let off = scratch.take()?;
                    role.push(MirOp::Li {
                        reg: dest,
                        value: ring.base,
                    });
                    eval(
                        k, types, role, param, loops, offset, off, scratch, counters, plan, cbs,
                    )?;
                    if el != 1 {
                        let scale = scratch.take()?;
                        role.push(MirOp::Li {
                            reg: scale,
                            value: el,
                        });
                        role.push(MirOp::IArith {
                            op: IOp::Mul,
                            d: off,
                            a: off,
                            b: scale,
                        });
                        scratch.give(scale);
                    }
                    role.push(MirOp::IArith {
                        op: IOp::Add,
                        d: dest,
                        a: dest,
                        b: off,
                    });
                    scratch.give(off);
                }
            }
            role.push(MirOp::Lw {
                reg: dest,
                base: dest,
                at: 0,
            });
            Ok(())
        }
        _ => Err(ttStatus::TT_TODO_LOOP_ADDRESS),
    }
}

/// The ring's page count as a number the error messages can print.
fn at_len(len: Dim) -> Dim {
    len
}

/// `SFPGT`'s `SET_VD`: write `-1`/`0` into the `VD` lane instead of the flags.
const SFPSET_VD: u32 = 8;
/// `SFPGT`'s `SET_CC`: set the lane flags.
const SFPSET_CC: u32 = 1;
/// `SFPENCC` takes the mask from `VW`: `first` 3 enables it (bit 0), 2 resets
/// every lane's flag (bit 1), and `mod1` 10 is the form the SFPU bodies use.
const SFPENCC_ON: MirOp = MirOp::SfpEncc {
    first: 3,
    second: 0,
    mod1: 10,
};
const SFPENCC_OFF: MirOp = MirOp::SfpEncc {
    first: 2,
    second: 0,
    mod1: 10,
};

/// `d = a < b` as `-1`/`0` lanes. `SFPGT` compares its `VD` lane against `VC`
/// and writes the result into `VD`, so the right-hand value moves into the
/// destination first.
fn sfp_lt(role: &mut Vec<MirOp>, a: LReg, b: LReg, d: LReg) {
    role.push(MirOp::SfpMov { a: b, d });
    role.push(MirOp::SfpGt {
        mod1: SFPSET_VD,
        vc: a,
        vd: d,
    });
}

/// `d = a > b`, in the same shape.
fn sfp_gt(role: &mut Vec<MirOp>, a: LReg, b: LReg, d: LReg) {
    role.push(MirOp::SfpMov { a, d });
    role.push(MirOp::SfpGt {
        mod1: SFPSET_VD,
        vc: b,
        vd: d,
    });
}

/// `d = a >= b`: the inverse of `<`, which `SFPLE` has as one instruction.
fn sfp_ge(role: &mut Vec<MirOp>, a: LReg, b: LReg, d: LReg) {
    role.push(MirOp::SfpMov { a: b, d });
    role.push(MirOp::SfpLe {
        mod1: SFPSET_VD,
        vc: a,
        vd: d,
    });
}

/// `d = cond != 0 ? t : f`, lanewise. `d = f` first, then the lane flags from
/// the condition (a lane bool is `-1`/`0`, and `SFPGT` against the zero lane
/// turns `-1` into a set flag), then `d = t` only where the flag is set.
fn sfp_select(role: &mut Vec<MirOp>, cond: LReg, t: LReg, f: LReg, d: LReg) {
    role.push(MirOp::SfpMov { a: f, d });
    role.push(SFPENCC_ON);
    role.push(MirOp::SfpGt {
        mod1: SFPSET_CC,
        vc: cond,
        vd: LReg(tt_isa::sfpu::LREG_ZERO as u16),
    });
    role.push(MirOp::SfpMov { a: t, d });
    role.push(SFPENCC_OFF);
}

/// `d = a > b ? a : b`. With the flags set to `a > b`, the destination takes
/// `b` everywhere and `a` where the flag holds.
fn sfp_max(role: &mut Vec<MirOp>, a: LReg, b: LReg, d: LReg) {
    role.push(MirOp::SfpMov { a: b, d });
    role.push(SFPENCC_ON);
    role.push(MirOp::SfpGt {
        mod1: SFPSET_CC,
        vc: b,
        vd: a,
    });
    role.push(MirOp::SfpMov { a, d });
    role.push(SFPENCC_OFF);
}

/// `d = a < b ? a : b`, the mirror: the destination takes `b` everywhere and
/// `a` where `b > a`, i.e. where `a` is the smaller.
fn sfp_min(role: &mut Vec<MirOp>, a: LReg, b: LReg, d: LReg) {
    role.push(MirOp::SfpMov { a: b, d });
    role.push(SFPENCC_ON);
    role.push(MirOp::SfpGt {
        mod1: SFPSET_CC,
        vc: a,
        vd: b,
    });
    role.push(MirOp::SfpMov { a, d });
    role.push(SFPENCC_OFF);
}

/// The RISC-V form of an index arithmetic op; a comparison or `pow` is not
/// one.
fn int_op(op: BOp) -> Option<IOp> {
    Some(match op {
        BOp::Add => IOp::Add,
        BOp::Sub => IOp::Sub,
        BOp::Mul => IOp::Mul,
        BOp::Div => IOp::Div,
        BOp::Mod => IOp::Rem,
        BOp::BitAnd => IOp::And,
        BOp::BitOr => IOp::Or,
        BOp::BitXor => IOp::Xor,
        BOp::BitShiftLeft => IOp::Sll,
        BOp::BitShiftRight => IOp::Srl,
        BOp::Max => IOp::Max,
        BOp::Min => IOp::Min,
        _ => return None,
    })
}

/// The bits a scalar constant operand carries, if `v` is one: `lower` puts
/// them in a lane instead of reading a `Dst` tile, which is what lets the
/// unfold write its coefficients and masks as ordinary `const` values.
/// The lane image of a constant's bits: a `bf16` constant is its sixteen-bit
/// datum (zyx's `Constant::BF16([u8; 2])`), and a lane holds the f32 it widens
/// to, the high half of that word.
fn lane_image(dtype: crate::ir::DType, bits: u32) -> u32 {
    match dtype {
        crate::ir::DType::Bf16 => (bits & 0xffff) << 16,
        _ => bits,
    }
}

/// `bits` is `from`'s lane image; the same value as `to`'s, when a lane
/// immediate can carry the conversion.
fn convert_image(from: crate::ir::DType, to: crate::ir::DType, bits: u32) -> Option<u32> {
    match (from, to) {
        (a, b) if a == b => Some(bits),
        // A bf16 datum is already the f32 image (it widened to reach the lane).
        (crate::ir::DType::Bf16, crate::ir::DType::F32) => Some(bits),
        // zyx's `bf16::from_f32`: `(x.to_bits() >> 16) as u16`, widened back.
        (crate::ir::DType::F32, crate::ir::DType::Bf16) => Some((bits >> 16) << 16),
        _ => None,
    }
}

fn scalar_const(k: &Kernel, v: crate::ir::OpId) -> Option<u32> {
    match *k.op(v)? {
        Op::Const { dtype, bits } => Some(lane_image(dtype, bits)),
        // A `cast` of a constant is the conversion it names, so both spellings
        // reach the same immediate: `const bf16 0.3`, and `const f32 0.3`
        // followed by `cast ... to bf16`.
        Op::Cast { src, dtype } => {
            let from = k.info(src).dtype;
            scalar_const(k, src).and_then(|bits| convert_image(from, dtype, bits))
        }
        _ => None,
    }
}

/// `2^x` for one tile of `Dst` rows, in lane registers.
///
/// tt-metal's `ckernel_sfpu_exp2.h` with zyx's polynomial: `x = n + f` by the
/// `1.5 * 2^23` rounding trick, `2^f` by the `exp(f ln 2)` Horner chain, and
/// the exponent assembled by `SFPEXEXP`/`SFPSETEXP` — `n` is added to the
/// exponent field, which is the step no IR op spells.
///
/// Lanes: `la` holds the input, 2 the rounding constant, 3 a temporary, 4 the
/// fractional part and then the polynomial, 5 the product (and the result), 6
/// `-bits(1.5 * 2^23)` for the integer subtraction that produces `n`, 7 the
/// coefficient of the moment. Constants are re-loaded per row, which is what
/// the allocator pass will fold.
fn exp2_body(
    role: &mut Vec<MirOp>,
    src: Dst,
    out: Dst,
    rows: &[u32],
    la: LReg,
    dtype: crate::ir::DType,
) {
    const C231: u32 = 0x4B40_0000; // 1.5 * 2^23, exact
    const NEG_C231: u32 = 0xB4C0_0000; // the same bits, two's complement negated
    // `SFPENCC`'s mask-on mode and the two `SFPSETCC` tests the body uses.
    const MASK_ON: u32 = 10;
    const SETCC_LT0: u32 = 0;
    const SETCC_NE0: u32 = 2;
    const NEG_ONE: u32 = 0xbf80_0000;
    debug_assert!(la.0 < 2, "the body's scratch lanes start at 2");
    // `n = bits(x + c231) - bits(c231)`: the integer subtraction tt-metal's
    // `_sfpu_round_to_nearest_int32_` does, and the exponent it adds.
    role.push(MirOp::LaneConst {
        lreg: LReg(2),
        bits: C231,
    });
    role.push(MirOp::LaneConst {
        lreg: LReg(6),
        bits: NEG_C231,
    });
    for &row in rows {
        role.push(MirOp::Sfpload {
            lreg: la,
            reg: src,
            at: row,
            dtype,
        });
        // tmp = x + c231, nf = tmp - c231, f = x - nf.
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: la,
            b: LReg(2),
            d: LReg(3),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Sub,
            a: LReg(3),
            b: LReg(2),
            d: LReg(4),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Sub,
            a: la,
            b: LReg(4),
            d: LReg(5),
        });
        // n = bits(tmp) - bits(c231), as a two's complement integer.
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::IAdd,
            a: LReg(3),
            b: LReg(6),
            d: la,
        });
        // The underflow predicate, carried in lane 6 (free once `n` is formed)
        // until the result is ready: `nf + 126 < 0`, i.e. the result's true
        // exponent is below the smallest normal. The exponent lane itself is an
        // integer, so the test is made on the integer part as a float.
        role.push(MirOp::LaneConst {
            lreg: LReg(6),
            bits: 0,
        });
        role.push(MirOp::LaneConst {
            lreg: LReg(7),
            bits: 126.0f32.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: LReg(4),
            b: LReg(7),
            d: LReg(7),
        });
        role.push(MirOp::SfpEncc {
            first: 3,
            second: 0,
            mod1: MASK_ON,
        });
        role.push(MirOp::SfpSetcc {
            mod1: SETCC_LT0,
            a: LReg(7),
        });
        role.push(MirOp::LaneConst {
            lreg: LReg(6),
            bits: NEG_ONE,
        });
        role.push(MirOp::SfpEncc {
            first: 2,
            second: 0,
            mod1: MASK_ON,
        });
        // s = f ln 2, then the Horner chain into lane 4.
        role.push(MirOp::LaneConst {
            lreg: LReg(7),
            bits: std::f32::consts::LN_2.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: LReg(5),
            b: LReg(7),
            d: LReg(3),
        });
        role.push(MirOp::LaneConst {
            lreg: LReg(4),
            bits: (1.0f32 / 5040.0).to_bits(),
        });
        for c in [
            1.0f32 / 720.0,
            1.0 / 120.0,
            1.0 / 24.0,
            1.0 / 6.0,
            0.5,
            1.0,
            1.0,
        ] {
            role.push(MirOp::LaneConst {
                lreg: LReg(7),
                bits: c.to_bits(),
            });
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::Mul,
                a: LReg(4),
                b: LReg(3),
                d: LReg(5),
            });
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::Add,
                a: LReg(5),
                b: LReg(7),
                d: LReg(4),
            });
        }
        // The result's exponent is the polynomial's plus n.
        role.push(MirOp::SfpExexp {
            mod1: 1,
            a: LReg(4),
            d: LReg(3),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::IAdd,
            a: LReg(3),
            b: la,
            d: LReg(5),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Setexp,
            a: LReg(4),
            b: LReg(5),
            d: LReg(5),
        });
        // Underflow: `2^k` with `k` below the smallest normal is zero. tt-metal's
        // exp tests its clamped exponent the same way ("if k == 0, exp(x) = 0
        // (fixes -inf case)"); without it `Setexp` wraps the exponent, and a
        // masked score of `-1000` comes back around `1e34` instead of zero.
        role.push(MirOp::SfpEncc {
            first: 3,
            second: 0,
            mod1: MASK_ON,
        });
        role.push(MirOp::SfpSetcc {
            mod1: SETCC_NE0,
            a: LReg(6),
        });
        role.push(MirOp::LaneConst {
            lreg: LReg(5),
            bits: 0,
        });
        role.push(MirOp::SfpEncc {
            first: 2,
            second: 0,
            mod1: MASK_ON,
        });
        role.push(MirOp::Sfpstore {
            lreg: LReg(5),
            reg: out,
            at: row,
            dtype,
        });
    }
}

/// `1/sqrt(x)` in the lane: tt-metal's `_calculate_sqrt_body_` with its
/// `RECIPROCAL` flag set, which is the accurate `rsqrt`
/// (`ckernel_sfpu_rsqrt.h` dispatches to
/// `_calculate_sqrt_internal_<…, RECIPROCAL = true, …>` unless the caller asks
/// for the `legacy_compat` form). The program constants `_init_sqrt_` leaves in
/// `vConstIntPrgm0`/`vConstFloatPrgm1`/`vConstFloatPrgm2` are lanes here: this
/// compiler has no program-constant register, and the values are all the
/// sequence reads.
///
/// The seed is the exponent-halving bit form `magic - bits(x)/2`; then one
/// corrected Newton trio `y = y (P1 + c (P2 + c))`, `c = -y·xy`; then the
/// reciprocal step `y = (1 - xy²)·y/2 + y`, with tt-metal's `addexp(y, -1)`
/// spelled as `y · 0.5` (exact for every normal `y`). The guard keeps that root
/// where `bits(x)` is neither zero nor `+inf` and writes
/// `reinterpret(inf_bits - bits(x))` elsewhere — `+inf` for `x = 0`, `0` for
/// `x = +inf` — and a negative `x` (but not `-0`) becomes NaN.
///
/// `la` is the input lane and lane 1 is the result; the other six are scratch,
/// so the sequence fits the same per-row loop as every other SFPU body.
fn rsqrt_body(
    role: &mut Vec<MirOp>,
    src: Dst,
    out: Dst,
    rows: &[u32],
    la: LReg,
    dtype: crate::ir::DType,
) -> Result<(), ttStatus> {
    // `_init_sqrt_`'s accurate-mode constants.
    const MAGIC: u32 = 0x5f11_10a0;
    const P1: f32 = 2.2825186;
    const P2: f32 = 2.2533049;
    const INF_BITS: u32 = 0x7f80_0000;
    const NAN_BITS: u32 = 0x7fc0_0000;
    debug_assert!(la.0 < 2, "the body's scratch lanes start at 2");
    for &row in rows {
        role.push(MirOp::Sfpload {
            lreg: la,
            reg: src,
            at: row,
            dtype,
        });
        // `x` lives in lane 2 for the rest of the row: the cast is a plain move
        // here, and lane 2 is the body's first scratch lane.
        role.push(MirOp::SfpMov { a: la, d: LReg(2) });
        // `i = bits(x) >> 1`, logically: the amount lane holds the integer 1 and
        // `SFPSHFT` negates it in its own lane.
        role.push(MirOp::LaneConst {
            lreg: LReg(7),
            bits: 1,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Shr,
            a: LReg(2),
            b: LReg(7),
            d: LReg(3),
        });
        // `y = reinterpret(MAGIC - i)`.
        role.push(MirOp::LaneConst {
            lreg: LReg(0),
            bits: MAGIC,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::ISub,
            a: LReg(0),
            b: LReg(3),
            d: LReg(3),
        });
        // `xy = x y`, `c = (-y) xy`.
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: LReg(2),
            b: LReg(3),
            d: LReg(4),
        });
        role.push(MirOp::UnarySfp {
            op: UnarySfpOp::Neg,
            a: LReg(3),
            d: LReg(5),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: LReg(5),
            b: LReg(4),
            d: LReg(4),
        });
        // `y = y (P1 + c (P2 + c))`.
        role.push(MirOp::LaneConst {
            lreg: LReg(6),
            bits: P2.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: LReg(6),
            b: LReg(4),
            d: LReg(6),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: LReg(6),
            b: LReg(4),
            d: LReg(6),
        });
        role.push(MirOp::LaneConst {
            lreg: LReg(7),
            bits: P1.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: LReg(6),
            b: LReg(7),
            d: LReg(6),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: LReg(3),
            b: LReg(6),
            d: LReg(3),
        });
        // `xy = x y`, `one_minus_xyy = 1 + (-y) xy`.
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: LReg(2),
            b: LReg(3),
            d: LReg(4),
        });
        role.push(MirOp::UnarySfp {
            op: UnarySfpOp::Neg,
            a: LReg(3),
            d: LReg(5),
        });
        role.push(MirOp::LaneConst {
            lreg: LReg(6),
            bits: 1.0f32.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: LReg(5),
            b: LReg(4),
            d: LReg(5),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: LReg(5),
            b: LReg(6),
            d: LReg(5),
        });
        // `y = one_minus_xyy (y/2) + y`.
        role.push(MirOp::LaneConst {
            lreg: LReg(6),
            bits: 0.5f32.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: LReg(3),
            b: LReg(6),
            d: LReg(6),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: LReg(5),
            b: LReg(6),
            d: LReg(5),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: LReg(5),
            b: LReg(3),
            d: LReg(3),
        });
        // The guard: `bits(x) != 0 && bits(x) != +inf`. The integer difference
        // `inf_bits - bits(x)` is the other branch's value, reinterpreted. The
        // two "not zero" tests go through `i32 -> f32` and two ordered compares
        // each, because a nonzero integer is a nonzero float at every magnitude
        // (the conversion saturates at `±2^31`, which is still nonzero) and the
        // SFPU's own integer predicate, `SFPSETCC`'s `NE0`, has no IR op
        // expressing it.
        role.push(MirOp::LaneConst {
            lreg: LReg(4),
            bits: INF_BITS,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::ISub,
            a: LReg(4),
            b: LReg(2),
            d: LReg(5),
        });
        for op in cast_lanes(DType::I32, DType::F32, LReg(5), LReg(6), LReg(0))? {
            role.push(op);
        }
        role.push(MirOp::LaneConst {
            lreg: LReg(7),
            bits: 0.0f32.to_bits(),
        });
        sfp_gt(role, LReg(6), LReg(7), LReg(6));
        sfp_lt(role, LReg(6), LReg(7), LReg(7));
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitOr,
            a: LReg(6),
            b: LReg(7),
            d: LReg(6),
        });
        for op in cast_lanes(DType::I32, DType::F32, LReg(2), LReg(7), LReg(0))? {
            role.push(op);
        }
        role.push(MirOp::LaneConst {
            lreg: LReg(4),
            bits: 0.0f32.to_bits(),
        });
        sfp_gt(role, LReg(7), LReg(4), LReg(7));
        sfp_lt(role, LReg(7), LReg(4), LReg(4));
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitOr,
            a: LReg(7),
            b: LReg(4),
            d: LReg(7),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitAnd,
            a: LReg(6),
            b: LReg(7),
            d: LReg(6),
        });
        sfp_select(role, LReg(6), LReg(3), LReg(5), LReg(1));
        // A negative input (but not `-0`) is NaN: tt-metal's `!FAST_APPROX`
        // tail.
        role.push(MirOp::LaneConst {
            lreg: LReg(6),
            bits: 0.0f32.to_bits(),
        });
        sfp_lt(role, LReg(2), LReg(6), LReg(7));
        role.push(MirOp::LaneConst {
            lreg: LReg(6),
            bits: NAN_BITS,
        });
        sfp_select(role, LReg(7), LReg(6), LReg(1), LReg(3));
        role.push(MirOp::Sfpstore {
            lreg: LReg(3),
            reg: out,
            at: row,
            dtype,
        });
    }
    Ok(())
}

/// `1/y` in the lane: `src/unfold.rs`'s `reciprocal` step for step. Its
/// `bitcast` to `i32` is the lane word itself here, so the body is one
/// sequence for every dtype — a bf16 datum is widened by `SFPLOAD` and the
/// result narrowed by `SFPSTORE`'s format.
///
/// `la` is the input and lane 1 the result; the other six lanes are scratch.
fn reciprocal_body(
    role: &mut Vec<MirOp>,
    src: Dst,
    out: Dst,
    rows: &[u32],
    la: LReg,
    steps: usize,
    dtype: crate::ir::DType,
) {
    const MANT: u32 = 0x007f_ffff;
    const NEG_ONE: u32 = 0xbf80_0000;
    const EMASK: u32 = 0xff80_0000;
    const K0: f32 = 0.323_232_501_745_224;
    const K1: f32 = 1.454_545_974_731_445_3;
    const K2: f32 = 2.121_212_482_452_392_6;
    const ABS: u32 = 0x7fff_ffff;
    const SIGN: u32 = 0x8000_0000;
    let (neg, tmp, est, scale, t) = (LReg(2), LReg(3), LReg(4), LReg(5), LReg(6));
    debug_assert!(la.0 < 2, "the body's scratch lanes start at 2");
    for &row in rows {
        role.push(MirOp::Sfpload {
            lreg: la,
            reg: src,
            at: row,
            dtype,
        });
        // `setman(-1.0, y)`: the sign and exponent of `-1.0` with the input's
        // mantissa, which is negative and scaled to `[1, 2)`.
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: MANT,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitAnd,
            a: la,
            b: tmp,
            d: neg,
        });
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: NEG_ONE,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitOr,
            a: neg,
            b: tmp,
            d: neg,
        });
        // `scale = ~bits(y)` with the mantissa cleared, halved.
        role.push(MirOp::UnarySfp {
            op: UnarySfpOp::BitNot,
            a: la,
            d: scale,
        });
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: EMASK,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitAnd,
            a: scale,
            b: tmp,
            d: scale,
        });
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: 0.5f32.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: scale,
            b: tmp,
            d: scale,
        });
        // The quadratic estimate, `k2 + (k1 + k0 neg) neg`.
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: K0.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: tmp,
            b: neg,
            d: t,
        });
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: K1.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: tmp,
            b: t,
            d: est,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: est,
            b: neg,
            d: t,
        });
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: K2.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: tmp,
            b: t,
            d: est,
        });
        // `t = 1 + neg est; est += est t`, one step per datum width.
        for _ in 0..steps.max(1) {
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::Mul,
                a: neg,
                b: est,
                d: t,
            });
            role.push(MirOp::LaneConst {
                lreg: tmp,
                bits: 1.0f32.to_bits(),
            });
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::Add,
                a: tmp,
                b: t,
                d: t,
            });
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::Mul,
                a: est,
                b: t,
                d: t,
            });
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::Add,
                a: est,
                b: t,
                d: est,
            });
        }
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: est,
            b: scale,
            d: est,
        });
        // `copysgn(est, y)`.
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: ABS,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitAnd,
            a: est,
            b: tmp,
            d: est,
        });
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: SIGN,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitAnd,
            a: la,
            b: tmp,
            d: t,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitOr,
            a: est,
            b: t,
            d: est,
        });
        role.push(MirOp::Sfpstore {
            lreg: est,
            reg: out,
            at: row,
            dtype,
        });
    }
}

/// `log2 x` in the lane: `src/unfold.rs`'s body (Juffa's reduction, the LLK's
/// degree-9 `log1p` chain, one `log2 e` multiply). The exponent/mantissa split
/// is the lane word's own fields, so no `bitcast` to `i32` has to name a width.
fn log2_body(
    role: &mut Vec<MirOp>,
    src: Dst,
    out: Dst,
    rows: &[u32],
    la: LReg,
    dtype: crate::ir::DType,
) {
    const C75: u32 = 0x3f40_0000;
    const EMASK: u32 = 0xff80_0000;
    const TWO_M23: u32 = 0x3400_0000;
    const COEFFS: [u32; 8] = [
        0x3ddc_2000,
        0xbe06_2000,
        0x3e13_a000,
        0xbe2a_8000,
        0x3e4c_c000,
        0xbe80_000d,
        0x3eaa_aab9,
        0xbf00_0000,
    ];
    let (e, m, s, r, p, cc) = (LReg(2), LReg(3), LReg(4), LReg(5), LReg(6), LReg(7));
    debug_assert!(la.0 < 2, "the body's scratch lanes start at 2");
    for &row in rows {
        role.push(MirOp::Sfpload {
            lreg: la,
            reg: src,
            at: row,
            dtype,
        });
        // `e = (bits(x) - bits(0.75)) & 0xff80_0000` is `k << 23`, and
        // `m = bits(x) - e` is the mantissa as a float in `[0.75, 1.5)`.
        role.push(MirOp::LaneConst {
            lreg: cc,
            bits: C75,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::ISub,
            a: la,
            b: cc,
            d: e,
        });
        role.push(MirOp::LaneConst {
            lreg: cc,
            bits: EMASK,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitAnd,
            a: e,
            b: cc,
            d: e,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::ISub,
            a: la,
            b: e,
            d: m,
        });
        role.push(MirOp::LaneConst {
            lreg: cc,
            bits: 1.0f32.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Sub,
            a: m,
            b: cc,
            d: m,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: m,
            b: m,
            d: s,
        });
        role.push(MirOp::LaneConst {
            lreg: r,
            bits: 0xbd49_6000,
        });
        for c in COEFFS {
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::Mul,
                a: r,
                b: m,
                d: p,
            });
            role.push(MirOp::LaneConst { lreg: cc, bits: c });
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::Add,
                a: p,
                b: cc,
                d: r,
            });
        }
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: r,
            b: s,
            d: r,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: r,
            b: m,
            d: r,
        });
        // `e * 2^-23` is exactly `k`.
        for op in
            cast_lanes(crate::ir::DType::I32, crate::ir::DType::F32, e, cc, p).unwrap_or_default()
        {
            role.push(op);
        }
        role.push(MirOp::LaneConst {
            lreg: p,
            bits: TWO_M23,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: cc,
            b: p,
            d: cc,
        });
        role.push(MirOp::LaneConst {
            lreg: p,
            bits: std::f32::consts::LOG2_E.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: r,
            b: p,
            d: r,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: cc,
            b: r,
            d: cc,
        });
        role.push(MirOp::Sfpstore {
            lreg: cc,
            reg: out,
            at: row,
            dtype,
        });
    }
}

/// `sin x`/`cos x` in the lane: `src/unfold.rs`'s reduction and Taylor cores,
/// with the quadrant taken from the integer `n`. The `i32` casts it spells are
/// `SFPCAST`s on the lane word, so the body is one sequence for every dtype.
///
/// `la` is the input and lane 1 the result; the other six lanes are scratch.
fn sin_cos_body(
    role: &mut Vec<MirOp>,
    src: Dst,
    out: Dst,
    rows: &[u32],
    la: LReg,
    cos: bool,
    dtype: crate::ir::DType,
) -> Result<(), ttStatus> {
    const MAGIC: f32 = 12_582_912.0;
    const SIN: [f32; 5] = [
        -1.0 / 6.0,
        1.0 / 120.0,
        -1.0 / 5040.0,
        1.0 / 362_880.0,
        -1.0 / 39_916_800.0,
    ];
    const COS: [f32; 5] = [
        -0.5,
        1.0 / 24.0,
        -1.0 / 720.0,
        1.0 / 40_320.0,
        -1.0 / 3_628_800.0,
    ];
    let pio2_hi = f32::from_bits((std::f64::consts::FRAC_PI_2 as f32).to_bits() & !0xfff);
    let pio2_lo = (std::f64::consts::FRAC_PI_2 - f64::from(pio2_hi)) as f32;
    let zero = LReg(tt_isa::sfpu::LREG_ZERO as u16);
    let (n, nf, s, r, rest, tmp) = (LReg(2), LReg(3), LReg(4), LReg(5), LReg(6), LReg(7));
    debug_assert!(la.0 < 2, "the body's scratch lanes start at 2");
    for &row in rows {
        role.push(MirOp::Sfpload {
            lreg: la,
            reg: src,
            at: row,
            dtype,
        });
        // `t = x 2/pi + magic`, `nf = t - magic`: `nf` is the rounded count.
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: std::f32::consts::FRAC_2_PI.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: la,
            b: tmp,
            d: nf,
        });
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: MAGIC.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: nf,
            b: tmp,
            d: nf,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Sub,
            a: nf,
            b: tmp,
            d: nf,
        });
        // `n = (i32)nf`, kept for the quadrant.
        for op in cast_lanes(crate::ir::DType::F32, crate::ir::DType::I32, nf, n, rest)? {
            role.push(op);
        }
        // `r = x - nf hi - nf lo`, the two-limb `pi/2` so `|r| <= pi/4`.
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: pio2_hi.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: nf,
            b: tmp,
            d: rest,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Sub,
            a: la,
            b: rest,
            d: r,
        });
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: pio2_lo.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: nf,
            b: tmp,
            d: rest,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Sub,
            a: r,
            b: rest,
            d: r,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: r,
            b: r,
            d: s,
        });
        // `sin_core` (`rest`), then `sin_r = r sin_core` back into `rest`.
        role.push(MirOp::LaneConst {
            lreg: rest,
            bits: SIN[4].to_bits(),
        });
        for coeff in SIN[..4].iter().rev() {
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::Mul,
                a: rest,
                b: s,
                d: tmp,
            });
            role.push(MirOp::LaneConst {
                lreg: nf,
                bits: coeff.to_bits(),
            });
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::Add,
                a: tmp,
                b: nf,
                d: rest,
            });
        }
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: rest,
            b: s,
            d: tmp,
        });
        role.push(MirOp::LaneConst {
            lreg: nf,
            bits: 1.0f32.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: tmp,
            b: nf,
            d: rest,
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: r,
            b: rest,
            d: rest,
        });
        // `cos_core` (`tmp`, with `la` as its scratch): `n` is still live here.
        role.push(MirOp::LaneConst {
            lreg: tmp,
            bits: COS[4].to_bits(),
        });
        for coeff in COS[..4].iter().rev() {
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::Mul,
                a: tmp,
                b: s,
                d: la,
            });
            role.push(MirOp::LaneConst {
                lreg: nf,
                bits: coeff.to_bits(),
            });
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::Add,
                a: la,
                b: nf,
                d: tmp,
            });
        }
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: tmp,
            b: s,
            d: la,
        });
        role.push(MirOp::LaneConst {
            lreg: nf,
            bits: 1.0f32.to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Add,
            a: la,
            b: nf,
            d: tmp,
        });
        // The quadrant: `q = n & 3` in `r`, `low = q & 1` in `n`, `odd` in `nf`.
        role.push(MirOp::LaneConst { lreg: nf, bits: 3 });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitAnd,
            a: n,
            b: nf,
            d: r,
        });
        role.push(MirOp::LaneConst { lreg: nf, bits: 1 });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitAnd,
            a: r,
            b: nf,
            d: n,
        });
        sfp_gt(role, n, zero, nf);
        // `neg = ((q + cos) >> 1) & 1 != 0`, each `1` fresh: `SFPSHFT` consumes
        // its amount lane, and `SFPIADD` moves one operand into its
        // destination, so `q` is copied into `n` first rather than added in
        // place.
        if cos {
            role.push(MirOp::LaneConst { lreg: la, bits: 1 });
            role.push(MirOp::BinarySfp {
                op: BinarySfpOp::IAdd,
                a: r,
                b: la,
                d: n,
            });
        } else {
            role.push(MirOp::SfpMov { a: r, d: n });
        }
        role.push(MirOp::LaneConst { lreg: la, bits: 1 });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Shr,
            a: n,
            b: la,
            d: n,
        });
        role.push(MirOp::LaneConst { lreg: la, bits: 1 });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::BitAnd,
            a: n,
            b: la,
            d: n,
        });
        sfp_gt(role, n, zero, n);
        // `base = odd ? odd_core : even_core`, then `out = neg ? -base : base`.
        let (odd_core, even_core) = if cos { (rest, tmp) } else { (tmp, rest) };
        sfp_select(role, nf, odd_core, even_core, r);
        role.push(MirOp::LaneConst {
            lreg: la,
            bits: (-1.0f32).to_bits(),
        });
        role.push(MirOp::BinarySfp {
            op: BinarySfpOp::Mul,
            a: r,
            b: la,
            d: s,
        });
        sfp_select(role, n, s, r, r);
        role.push(MirOp::Sfpstore {
            lreg: r,
            reg: out,
            at: row,
            dtype,
        });
    }
    Ok(())
}

/// `i32` `mod` on one row: tt-metal's `fmod_int32`, in lanes 0..8.
///
/// The body is `compute_unsigned_remainder_int32` of
/// `ckernel_sfpu_binary_remainder.h` with the sign fix of
/// `ckernel_sfpu_binary_fmod.h`. Both operands are taken to magnitudes in two's
/// complement and widened to FP32; the divisor is reciprocated with one Newton
/// step, the quotient's mantissa is multiplied back out as a 23-bit integer, and
/// the error the reciprocal leaves is measured in FP32 and removed with a second
/// `SFPMUL24` pair. The last adjustment clamps the remainder into `[0, b)` and
/// the negation at the end gives `fmod`'s sign of the dividend.
///
/// Nothing here is a float shortcut: every step is an integer or a bounded
/// integer conversion, so the result is the exact remainder for all `i32`.
///
/// The dividend is read from lane 0 and the divisor from lane 1; the result is
/// left in lane 3. Lanes 2..7 are scratch, and the hardware's own `1.0` is read
/// where a constant is needed (`LREG_ONE`); zero, 2^31, 2^33 and the shift
/// amounts are `SFPLOADI`s, since a `Compare` reads two work lanes.
fn mod_body(role: &mut Vec<MirOp>) {
    // Mode words from the vendored `sfpi_constants.h` (Blackhole).
    const IADD_ARG_IMM: u32 = 1;
    const IADD_CC_NONE: u32 = 4;
    const SHFT_ARITHMETIC: u32 = 2;
    const CAST_SM32_TO_FP32_RNE: u32 = 0;
    /// `MantissaMode::FractionOnly` (`SFPEXMAN_MOD1_PAD9`).
    const EXMAN_FRACTION_ONLY: u32 = 1;
    const STOCHRND_FP32_TO_UINT16: u32 = 6;
    let one = LReg(tt_isa::sfpu::LREG_ONE as u16);
    // The lanes the body names. `a` stays untouched to the end: the sign of
    // the dividend is the sign of the result.
    let a = LReg(0);
    let b = LReg(1);
    let p = LReg(2);
    let x = LReg(3);
    let y = LReg(4);
    let s = LReg(5);
    let t = LReg(6);
    let u = LReg(7);

    // |a| and |b| in two's complement. `INT_MIN` keeps its `0x8000_0000`,
    // which the sign-magnitude cast below reads as `-0.0`.
    role.push(MirOp::SfpAbs { mod1: 0, a, d: p });
    role.push(MirOp::SfpAbs {
        mod1: 0,
        a: b,
        d: b,
    });
    // The magnitudes as FP32. The cast reads the bits as a sign-magnitude
    // integer, so a magnitude of 2^31 becomes `-0.0`; the compare is
    // sign-magnitude too (`-0.0 < +0.0`), so that lane is replaced by 2^31.
    role.push(MirOp::SfpCast {
        mod1: CAST_SM32_TO_FP32_RNE,
        a: p,
        d: x,
    });
    role.push(MirOp::SfpCast {
        mod1: CAST_SM32_TO_FP32_RNE,
        a: b,
        d: y,
    });
    role.push(MirOp::LaneConst {
        lreg: s,
        bits: 2147483648.0f32.to_bits(),
    });
    role.push(MirOp::LaneConst { lreg: u, bits: 0 });
    sfp_lt(role, x, u, t);
    sfp_select(role, t, s, x, x);
    sfp_lt(role, y, u, t);
    sfp_select(role, t, s, y, y);
    // `inv = approx_recip(b)`, then one Newton step `inv *= (2 - b * inv)`,
    // which is `e = 1 - b * inv` and `inv = e * inv + inv`. The negated product
    // is a complement-sign move into a lane of its own: `SFPMAD` has no mode
    // that negates the product.
    role.push(MirOp::SfpArecip {
        mod1: 0,
        a: y,
        d: s,
    });
    role.push(MirOp::UnarySfp {
        op: UnarySfpOp::Neg,
        a: s,
        d: t,
    });
    role.push(MirOp::SfpMad {
        mod1: 0,
        a: t,
        b: y,
        c: one,
        d: t,
    });
    role.push(MirOp::SfpMad {
        mod1: 0,
        a: t,
        b: s,
        c: s,
        d: s,
    });
    // `q = a * inv + 2^33`: the alignment factor puts the quotient in the
    // mantissa's top bits, which `SFPEXMAN` then extracts as an integer.
    role.push(MirOp::LaneConst {
        lreg: t,
        bits: 8589934592.0f32.to_bits(),
    });
    role.push(MirOp::SfpMad {
        mod1: 0,
        a: x,
        b: s,
        c: t,
        d: u,
    });
    role.push(MirOp::SfpExman {
        mod1: EXMAN_FRACTION_ONLY,
        a: u,
        d: u,
    });
    // `qb = (q << 10) * b` in the low 23 bits of the product, then the
    // approximate remainder `r = a - qb`.
    role.push(MirOp::SfpMul24 {
        upper: false,
        a: u,
        b,
        d: t,
    });
    role.push(MirOp::LaneConst { lreg: x, bits: 10 });
    role.push(MirOp::SfpShft {
        mod1: SHFT_ARITHMETIC,
        amount: x,
        a: t,
        d: t,
    });
    role.push(MirOp::BinarySfp {
        op: BinarySfpOp::ISub,
        a: p,
        b: t,
        d: t,
    });
    // The correction the reciprocal's error left: `|r| / b` rounded to a
    // 16-bit integer, multiplied back out as a 32-bit product assembled from
    // two 23-bit halves and `b >> 23`.
    role.push(MirOp::SfpAbs {
        mod1: 0,
        a: t,
        d: x,
    });
    role.push(MirOp::SfpCast {
        mod1: CAST_SM32_TO_FP32_RNE,
        a: x,
        d: x,
    });
    role.push(MirOp::BinarySfp {
        op: BinarySfpOp::Mul,
        a: x,
        b: s,
        d: x,
    });
    role.push(MirOp::SfpStochRnd {
        mod1: STOCHRND_FP32_TO_UINT16,
        a: x,
        d: y,
    });
    role.push(MirOp::SfpMul24 {
        upper: false,
        a: y,
        b,
        d: p,
    });
    role.push(MirOp::SfpMul24 {
        upper: true,
        a: y,
        b,
        d: x,
    });
    role.push(MirOp::LaneConst {
        lreg: s,
        bits: (-23i32) as u32,
    });
    role.push(MirOp::SfpShft {
        mod1: SHFT_ARITHMETIC,
        amount: s,
        a: b,
        d: u,
    });
    role.push(MirOp::SfpMul24 {
        upper: false,
        a: y,
        b: u,
        d: u,
    });
    role.push(MirOp::BinarySfp {
        op: BinarySfpOp::IAdd,
        a: x,
        b: u,
        d: u,
    });
    role.push(MirOp::LaneConst { lreg: s, bits: 23 });
    role.push(MirOp::SfpShft {
        mod1: SHFT_ARITHMETIC,
        amount: s,
        a: u,
        d: u,
    });
    role.push(MirOp::BinarySfp {
        op: BinarySfpOp::IAdd,
        a: p,
        b: u,
        d: u,
    });
    // `signed_tmp = (tmp ^ r_sign) - r_sign`, with the arithmetic shift's
    // `0`/`-1` mask, then `r -= signed_tmp`.
    role.push(MirOp::LaneConst {
        lreg: s,
        bits: (-31i32) as u32,
    });
    role.push(MirOp::SfpShft {
        mod1: SHFT_ARITHMETIC,
        amount: s,
        a: t,
        d: x,
    });
    role.push(MirOp::BinarySfp {
        op: BinarySfpOp::BitXor,
        a: u,
        b: x,
        d: u,
    });
    role.push(MirOp::BinarySfp {
        op: BinarySfpOp::ISub,
        a: u,
        b: x,
        d: x,
    });
    role.push(MirOp::BinarySfp {
        op: BinarySfpOp::ISub,
        a: t,
        b: x,
        d: x,
    });
    // Clamp into `[0, b)`: `if (r < 0 && (r - 1) < 0) r += b; else if (r >= b)
    // r -= b;`. The second test sees the `r` the first branch may have
    // updated, exactly as `v_elseif` does.
    role.push(MirOp::LaneConst { lreg: u, bits: 0 });
    sfp_lt(role, x, u, y);
    role.push(MirOp::SfpIadd {
        imm: (-1i32 & 0xfff) as u32,
        a: x,
        d: s,
        mod1: IADD_ARG_IMM | IADD_CC_NONE,
    });
    sfp_lt(role, s, u, u);
    role.push(MirOp::BinarySfp {
        op: BinarySfpOp::BitAnd,
        a: y,
        b: u,
        d: y,
    });
    role.push(MirOp::BinarySfp {
        op: BinarySfpOp::IAdd,
        a: x,
        b,
        d: s,
    });
    sfp_select(role, y, s, x, x);
    sfp_ge(role, x, b, y);
    role.push(MirOp::BinarySfp {
        op: BinarySfpOp::ISub,
        a: x,
        b,
        d: s,
    });
    sfp_select(role, y, s, x, x);
    // `fmod`'s sign: the result has the dividend's sign (`checked_rem`'s,
    // RISC-V's `rem`).
    role.push(MirOp::LaneConst { lreg: s, bits: 0 });
    sfp_lt(role, a, s, y);
    role.push(MirOp::LaneConst { lreg: t, bits: 0 });
    role.push(MirOp::BinarySfp {
        op: BinarySfpOp::ISub,
        a: t,
        b: x,
        d: s,
    });
    sfp_select(role, y, s, x, x);
}

/// The `Dst` rows an SFPU body walks. An instruction covers 32 lanes, two
/// 16-datum rows, so a tile is its 64 rows two at a time. A narrow value keeps
/// the acc's narrow part (a `1,32` in rows 0 and 16, a `32,1` in column 0), and
/// its lane layout in the lane — sixteen meaningful lanes of each — is not
/// lowered: a value that reaches the datapath with that shape is refused rather
/// than walked with a tile's addressing.
fn sfp_rows(shape: RegShape, at: OpId) -> Result<Vec<u32>, ttStatus> {
    Ok(match shape {
        RegShape::Tile => (0..DST_ROWS).step_by(2).collect(),
        RegShape::Row | RegShape::Column => {
            let _ = (at, shape);
            return Err(ttStatus::TT_TODO_ACCESS_SHAPE);
        }
        RegShape::Scalar => {
            let _ = at;
            return Err(ttStatus::TT_ERROR_SCALAR_NO_ROWS);
        }
    })
}

fn rows_of(shape: RegShape, at: OpId) -> Result<u32, ttStatus> {
    match shape {
        RegShape::Scalar => {
            let _ = at;
            Err(ttStatus::TT_ERROR_SCALAR_NO_ROWS)
        }
        RegShape::Row | RegShape::Column => Ok(2),
        RegShape::Tile => Ok(DST_ROWS),
    }
}

/// The `Dst` register a value lives in. A bitcast has no home of its own, so
/// the caller passes the owning definition ([`owner`]).
fn dst_reg(alloc: &Alloc, value: OpId) -> Result<Dst, ttStatus> {
    match alloc.place_of(value) {
        Some(reg) => Ok(reg),
        None => Err(crate::status::internal_bug(format_args!(
            "lower: value {value} has no `Dst`"
        ))),
    }
}

/// The machine IR's id of a ring. The ring ids were minted by `prepare`, which
/// also bounds their count by the interface table's.
fn cb_id(cb: Cb) -> Cb {
    cb
}

/// One copy of a mover segment: the parameter-side byte offset evaluated into a
/// RISC-V register, then the NoC request itself — one request, or one per lane
/// when the index names a base per datum.
#[allow(clippy::too_many_arguments)]
fn copy_to(
    k: &Kernel,
    types: &Types,
    plan: &Plan,
    param: &BTreeMap<OpId, u32>,
    loops: &[OpId],
    m: &mut MachineKernel,
    core: Core,
    dst: OpId,
    src: OpId,
    dst_offset: OpId,
    src_offset: OpId,
    access: [Dim; 2],
) -> Result<(), ttStatus> {
    let inbound = core == Core::Reader;
    let (buffer, argument, offset) = if inbound {
        (dst, src, src_offset)
    } else {
        (src, dst, dst_offset)
    };
    let dtype = match *k.at(buffer) {
        Op::Cb { dtype, .. } => dtype,
        _ => return Err(ttStatus::TT_ERROR_NOT_A_BUFFER),
    };
    let cb = *plan
        .cb
        .get(&buffer)
        .ok_or(ttStatus::TT_ERROR_BUFFER_NOT_PLACED)?;
    let index = *param.get(&argument).ok_or(ttStatus::TT_ERROR_COPY_ENDS)?;
    // A striped tensor's pages live on different channels, so the request's
    // address is resolved on the core from the page index rather than added to
    // one base. See [`emit_request`].
    let interleaved = matches!(
        *k.at(argument),
        Op::Param {
            interleaved: true,
            ..
        }
    );
    // The index is in access units, so one step is the unit's bytes.
    let bytes = (access[0] * access[1])
        .checked_mul(dtype.bytes())
        .ok_or(ttStatus::TT_ERROR_PARAM_OFFSET_OVERFLOW)?;
    // A vector index is one base per datum: one request per lane, each base read
    // out of the page the index load copied.
    let lanes = match types.of(offset).shape {
        RegShape::Scalar => 1,
        RegShape::Row | RegShape::Column => 32,
        RegShape::Tile => 1024,
    };
    // A mover's program has its own counters: `t5` is its NIU base, so the
    // compute roles' counters and its own `a1..a4` are not interchangeable.
    let mut ops = Vec::new();
    let mut pool = Pool::new(&MOVER_POOL);
    let value = pool.take()?;
    if lanes == 1 {
        scalar_with(
            k,
            types,
            &mut ops,
            param,
            loops,
            offset,
            value,
            &mut pool,
            &MOVER_INDEX,
            plan,
            &m.cbs,
        )?;
        let scale = pool.take()?;
        ops.push(MirOp::Li {
            reg: scale,
            value: bytes,
        });
        ops.push(MirOp::IArith {
            op: IOp::Mul,
            d: value,
            a: value,
            b: scale,
        });
        pool.give(scale);
        emit_request(
            &mut ops,
            &mut pool,
            inbound,
            cb,
            *plan
                .arg_block
                .get(index as usize)
                .ok_or(ttStatus::TT_ERROR_COPY_ENDS)?,
            bytes,
            value,
            None,
            interleaved,
        )?;
    } else {
        let (page, el) = index_page(k, plan, &m.cbs, offset)?;
        // The lane loop is one more nesting level, so it takes the next counter
        // pair: a lane copy deep inside two program loops does not fit.
        let (counter, _) = *MOVER_PAIRS
            .get(loops.len())
            .ok_or(ttStatus::TT_TODO_LOOP_DEPTH)?;
        ops.push(MirOp::Loop { count: lanes });
        // This lane's base: the index page's datum, in units of the access.
        let base = pool.take()?;
        let scale = pool.take()?;
        ops.push(MirOp::Li {
            reg: base,
            value: page,
        });
        ops.push(MirOp::Li {
            reg: scale,
            value: el,
        });
        ops.push(MirOp::IArith {
            op: IOp::Mul,
            d: scale,
            a: counter,
            b: scale,
        });
        ops.push(MirOp::IArith {
            op: IOp::Add,
            d: base,
            a: base,
            b: scale,
        });
        ops.push(MirOp::Lw {
            reg: value,
            base,
            at: 0,
        });
        ops.push(MirOp::Li {
            reg: scale,
            value: bytes,
        });
        ops.push(MirOp::IArith {
            op: IOp::Mul,
            d: value,
            a: value,
            b: scale,
        });
        pool.give(scale);
        pool.give(base);
        emit_request(
            &mut ops,
            &mut pool,
            inbound,
            cb,
            *plan
                .arg_block
                .get(index as usize)
                .ok_or(ttStatus::TT_ERROR_COPY_ENDS)?,
            bytes,
            value,
            Some(counter),
            interleaved,
        )?;
        ops.push(MirOp::EndLoop);
    }
    pool.give(value);
    for op in ops {
        m.push(core, op);
    }
    Ok(())
}

/// The address of one request to a striped parameter, resolved on the core.
///
/// The block's page size and channel count are read from the launch arguments,
/// then the byte offset `value` is split the way tt-metal's interleaved address
/// generator splits it (`InterleavedAddrGen::get_noc_addr`): the page index
/// picks the channel (`page % channels`), the page's stripe number picks the
/// offset inside it (`page / channels`), and the byte inside the page is added
/// unchanged. The `(offset, endpoint)` pair comes from the channel's slot in the
/// block, which is `get_noc_addr_from_bank_id` reading its bank table.
fn interleaved_address(
    ops: &mut Vec<MirOp>,
    pool: &mut Pool,
    block: i32,
    value: u8,
) -> Result<(u8, u8), ttStatus> {
    let page = pool.take()?;
    let banks = pool.take()?;
    ops.push(MirOp::Lw {
        reg: page,
        base: reg::A0,
        at: block + crate::mover::interleaved_page(),
    });
    ops.push(MirOp::Lw {
        reg: banks,
        base: reg::A0,
        at: block + crate::mover::interleaved_banks(),
    });
    let index = pool.take()?;
    ops.push(MirOp::IArith {
        op: IOp::Div,
        d: index,
        a: value,
        b: page,
    });
    let bank = pool.take()?;
    ops.push(MirOp::IArith {
        op: IOp::Rem,
        d: bank,
        a: index,
        b: banks,
    });
    let off = pool.take()?;
    ops.push(MirOp::IArith {
        op: IOp::Mul,
        d: off,
        a: index,
        b: page,
    });
    ops.push(MirOp::IArith {
        op: IOp::Sub,
        d: off,
        a: value,
        b: off,
    });
    ops.push(MirOp::IArith {
        op: IOp::Div,
        d: index,
        a: index,
        b: banks,
    });
    ops.push(MirOp::IArith {
        op: IOp::Mul,
        d: index,
        a: index,
        b: page,
    });
    ops.push(MirOp::IArith {
        op: IOp::Add,
        d: off,
        a: off,
        b: index,
    });
    // `index`, `page` and `banks` are spent: the pool holds exactly the
    // registers the rest of the request needs.
    pool.give(index);
    pool.give(banks);
    pool.give(page);
    let index = pool.take()?;
    ops.push(MirOp::Li {
        reg: index,
        value: 8,
    });
    ops.push(MirOp::IArith {
        op: IOp::Mul,
        d: bank,
        a: bank,
        b: index,
    });
    ops.push(MirOp::IArith {
        op: IOp::Add,
        d: bank,
        a: reg::A0,
        b: bank,
    });
    pool.give(index);
    let lo = pool.take()?;
    let hi = pool.take()?;
    ops.push(MirOp::Lw {
        reg: lo,
        base: bank,
        at: block,
    });
    ops.push(MirOp::Lw {
        reg: hi,
        base: bank,
        at: block + 4,
    });
    ops.push(MirOp::IArith {
        op: IOp::Add,
        d: lo,
        a: lo,
        b: off,
    });
    pool.give(bank);
    pool.give(off);
    Ok((lo, hi))
}

/// One NoC request as the mover's own program: the parameter's range base plus
/// the byte offset in `value`, the ring's page pointer, the ten initiator
/// registers, the trigger, and the two spins that wait for the hardware to take
/// the request and for the transaction to complete.
///
/// A lane's request is one span of the page past the one before it, so `lane`
/// (the counter) adds `bytes` to the page pointer.
#[allow(clippy::too_many_arguments)]
fn emit_request(
    ops: &mut Vec<MirOp>,
    pool: &mut Pool,
    inbound: bool,
    cb: Cb,
    block: i32,
    bytes: u32,
    value: u8,
    lane: Option<u8>,
    interleaved: bool,
) -> Result<(), ttStatus> {
    use crate::riscv::cb::{RD_PTR, STRIDE, WR_PTR};
    use tt_isa::noc::niu::{CMD_RD, CMD_WR, RESP_MARKED, STATIC_VC_1, initiator, reqs_outstanding};
    if bytes == 0 || bytes > tt_isa::noc::niu::MAX_REQUEST_BYTES {
        return Err(ttStatus::TT_ERROR_NOC_REQUEST_BYTES);
    }
    let (lo_arg, hi_arg) = (block, block + 4);
    let (lo, hi) = if interleaved {
        interleaved_address(ops, pool, block, value)?
    } else {
        let lo = pool.take()?;
        let hi = pool.take()?;
        ops.push(MirOp::LoadArg {
            reg: lo,
            arg: lo_arg,
        });
        ops.push(MirOp::IArith {
            op: IOp::Add,
            d: lo,
            a: lo,
            b: value,
        });
        ops.push(MirOp::LoadArg {
            reg: hi,
            arg: hi_arg,
        });
        (lo, hi)
    };
    let page = pool.take()?;
    // A read from DRAM publishes into the ring (its write pointer); a write to
    // DRAM drains the page the ring's read pointer names.
    let field = if inbound { WR_PTR } else { RD_PTR };
    ops.push(MirOp::Lw {
        reg: page,
        base: reg::A6,
        at: (u32::from(cb.0) * STRIDE) as i32 + field,
    });
    if let Some(counter) = lane {
        let span = pool.take()?;
        let off = pool.take()?;
        ops.push(MirOp::Li {
            reg: span,
            value: bytes,
        });
        ops.push(MirOp::IArith {
            op: IOp::Mul,
            d: off,
            a: counter,
            b: span,
        });
        ops.push(MirOp::IArith {
            op: IOp::Add,
            d: page,
            a: page,
            b: off,
        });
        pool.give(off);
        pool.give(span);
    }
    // `TARG_ADDR` is the request's source and `RET_ADDR` its destination: a
    // read has the parameter as source and L1 as destination, a write the other
    // way round. The high word is this tile's NOC word, staged by the prologue.
    let stores = if inbound {
        [
            (initiator::TARG_ADDR_LO, lo),
            (initiator::TARG_ADDR_MID, reg::ZERO),
            (initiator::TARG_ADDR_HI, hi),
            (initiator::RET_ADDR_LO, page),
            (initiator::RET_ADDR_MID, reg::ZERO),
            (initiator::RET_ADDR_HI, reg::T3),
        ]
    } else {
        [
            (initiator::TARG_ADDR_LO, page),
            (initiator::TARG_ADDR_MID, reg::ZERO),
            (initiator::TARG_ADDR_HI, reg::T3),
            (initiator::RET_ADDR_LO, lo),
            (initiator::RET_ADDR_MID, reg::ZERO),
            (initiator::RET_ADDR_HI, hi),
        ]
    };
    for (at, value) in stores {
        ops.push(MirOp::Sw {
            value,
            base: reg::T5,
            at: at as i32,
        });
    }
    pool.give(lo);
    pool.give(hi);
    let ctrl = if inbound { CMD_RD } else { CMD_WR } | RESP_MARKED | STATIC_VC_1;
    let words = [
        (
            (crate::mover::TXN.index() as u32) << 10,
            initiator::PACKET_TAG,
        ),
        (ctrl, initiator::CTRL),
        (bytes, initiator::AT_LEN_BE),
    ];
    for (word, at) in words {
        ops.push(MirOp::Li {
            reg: page,
            value: word,
        });
        ops.push(MirOp::Sw {
            value: page,
            base: reg::T5,
            at: at as i32,
        });
    }
    ops.push(MirOp::Sw {
        value: reg::ZERO,
        base: reg::T5,
        at: initiator::AT_DATA as i32,
    });
    ops.push(MirOp::Li {
        reg: page,
        value: 1,
    });
    ops.push(MirOp::Sw {
        value: page,
        base: reg::T5,
        at: initiator::CMD_CTRL as i32,
    });
    // The hardware takes the request, then the transaction completes.
    ops.push(MirOp::NiuPoll {
        reg: page,
        field: initiator::CMD_CTRL as i32,
    });
    ops.push(MirOp::Lw {
        reg: page,
        base: reg::T5,
        at: initiator::CMD_CTRL as i32,
    });
    ops.push(MirOp::NiuPoll {
        reg: page,
        field: reqs_outstanding(crate::mover::TXN) as i32,
    });
    pool.give(page);
    Ok(())
}

/// The L1 address of the page a lane index reads its bases out of, and the width
/// of one datum there: the page the `load` names, on a one-page ring. A deeper
/// ring's pointer moves at run time, so its page is refused rather than read
/// from one that only happens to be there.
fn index_page(
    k: &Kernel,
    plan: &Plan,
    cbs: &crate::slab::Slab<Cb, crate::mir::CbInfo>,
    value: OpId,
) -> Result<(u32, u32), ttStatus> {
    let (buffer, offset) = match *k.at(value) {
        Op::Load { buffer, offset, .. } => (buffer, offset),
        _ => return Err(ttStatus::TT_TODO_MULTI_LANE),
    };
    let (dtype, len) = match *k.at(buffer) {
        Op::Cb { dtype, len, .. } => (dtype, len),
        _ => return Err(ttStatus::TT_ERROR_NOT_A_BUFFER),
    };
    if len != 1 {
        let _ = (value, len);
        return Err(ttStatus::TT_TODO_LOOP_ADDRESS);
    }
    let cb = *plan
        .cb
        .get(&buffer)
        .ok_or(ttStatus::TT_ERROR_BUFFER_NOT_PLACED)?;
    let ring = cbs.get(cb).ok_or(ttStatus::TT_ERROR_BUFFER_NOT_PLACED)?;
    let el = dtype.bytes();
    let units = k
        .const_i32(offset)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or(ttStatus::TT_TODO_LOOP_ADDRESS)?;
    let bytes = units
        .checked_mul(el)
        .ok_or(ttStatus::TT_ERROR_PARAM_OFFSET_OVERFLOW)?;
    let base = ring
        .base
        .checked_add(bytes)
        .ok_or(ttStatus::TT_ERROR_PARAM_OFFSET_OVERFLOW)?;
    Ok((base, el))
}

/// Follow a bitcast to the value that owns the `Dst` slot: a bitcast has no
/// home of its own. The definition is the value itself, so this is a walk of
/// the chain, not a table.
fn unary(op: UOp, at: OpId) -> Result<UnarySfpOp, ttStatus> {
    match op {
        UOp::Neg => Ok(UnarySfpOp::Neg),
        UOp::Abs => Ok(UnarySfpOp::Abs),
        UOp::BitNot => Ok(UnarySfpOp::BitNot),
        other => {
            let _ = (at, other);
            Err(ttStatus::TT_TODO_UNARY_SFPU)
        }
    }
}

fn binary(op: BOp, at: OpId) -> Result<BinarySfpOp, ttStatus> {
    match op {
        BOp::Add => Ok(BinarySfpOp::Add),
        BOp::Sub => Ok(BinarySfpOp::Sub),
        BOp::Mul => Ok(BinarySfpOp::Mul),
        BOp::And | BOp::BitAnd => Ok(BinarySfpOp::BitAnd),
        BOp::Or | BOp::BitOr => Ok(BinarySfpOp::BitOr),
        BOp::BitXor => Ok(BinarySfpOp::BitXor),
        BOp::BitShiftLeft => Ok(BinarySfpOp::Shl),
        BOp::BitShiftRight => Ok(BinarySfpOp::Shr),
        other => {
            let _ = (at, other);
            Err(ttStatus::TT_TODO_BINARY_SFPU)
        }
    }
}

/// The lane sequence that converts `a` into `d`. A conversion is not one
/// instruction, so the machine IR carries the instructions it is made of; `t`
/// is the one scratch lane the sequence needs. Every lane the sequence names
/// is an operand of some op, which is what lets the allocation passes see and
/// reuse them.
fn cast_lanes(from: DType, to: DType, a: LReg, d: LReg, t: LReg) -> Result<Vec<MirOp>, ttStatus> {
    // Mode words from the vendored `sfpi_constants.h` (Blackhole).
    const CAST_INT32_TO_SM32: u32 = 3;
    const CAST_INT32_TO_FP32_RNE: u32 = 0;
    const EXEXP_CC_SGN_COMP: u32 = 2 | 8;
    const IADD_ARG_IMM: u32 = 1;
    const IADD_2SCOMP_LREG_DST: u32 = 2;
    const IADD_CC_NONE: u32 = 4;
    const SETCC_NE0: u32 = 2;
    const ENCC_ALL: u32 = 0;
    const MASK_ON: u32 = 10;
    let zero = LReg(tt_isa::sfpu::LREG_ZERO as u16);
    let mut out: Vec<MirOp> = Vec::new();
    match (from, to) {
        (DType::F32, DType::F32) | (DType::I32, DType::I32) | (DType::Bool, DType::Bool) => {
            out.push(MirOp::SfpMov { a, d });
        }
        (DType::I32, DType::F32) => {
            // Two's complement to sign-magnitude, then sign-magnitude to FP32
            // (round to nearest even) -- the pair tt-metal's
            // `_calculate_typecast_int32_to_fp32_` splits into abs/cast/setsgn.
            out.push(MirOp::SfpCast {
                mod1: CAST_INT32_TO_SM32,
                a,
                d: t,
            });
            out.push(MirOp::SfpCast {
                mod1: CAST_INT32_TO_FP32_RNE,
                a: t,
                d,
            });
        }
        (DType::F32, DType::I32) => {
            // The exponent's lane flags decide which lanes saturate, so the
            // condition mask is on for the whole sequence.
            out.push(MirOp::SfpEncc {
                first: 3,
                second: 0,
                mod1: MASK_ON,
            });
            out.push(MirOp::LaneConst { lreg: d, bits: 0 });
            out.push(MirOp::SfpExexp {
                mod1: EXEXP_CC_SGN_COMP,
                a,
                d: t,
            });
            // Saturated lanes keep `INT_MIN` (`0x8000_0000`).
            out.push(MirOp::LaneConst {
                lreg: d,
                bits: 0x8000_0000,
            });
            out.push(MirOp::SfpIadd {
                imm: (-31i32 & 0xfff) as u32,
                a: t,
                d: t,
                mod1: IADD_ARG_IMM,
            });
            out.push(MirOp::SfpIadd {
                imm: 8,
                a: t,
                d: t,
                mod1: IADD_ARG_IMM | IADD_CC_NONE,
            });
            out.push(MirOp::SfpExman { mod1: 0, a, d });
            out.push(MirOp::SfpShft {
                mod1: 0,
                amount: t,
                a: d,
                d,
            });
            out.push(MirOp::SfpEncc {
                first: 0,
                second: 0,
                mod1: ENCC_ALL,
            });
            out.push(MirOp::SfpSetcc { mod1: 0, a });
            out.push(MirOp::SfpIadd {
                imm: 0,
                a: zero,
                d,
                mod1: IADD_2SCOMP_LREG_DST | IADD_CC_NONE,
            });
            out.push(MirOp::SfpEncc {
                first: 2,
                second: 0,
                mod1: MASK_ON,
            });
        }
        (DType::Bool, DType::I32) | (DType::Bool, DType::F32) => {
            // A comparison's true has the sign bit set and its false is zero,
            // so a logical shift by 31 is the C value: one or zero. A negative
            // amount in the lane is a right shift.
            out.push(MirOp::LaneConst {
                lreg: t,
                bits: (-31i32) as u32,
            });
            out.push(MirOp::SfpShft {
                mod1: 0,
                amount: t,
                a,
                d,
            });
            if to == DType::F32 {
                out.push(MirOp::SfpCast {
                    mod1: CAST_INT32_TO_FP32_RNE,
                    a: d,
                    d,
                });
            }
        }
        (_, DType::Bool) => {
            // `x != 0`, with `-1.0f` as the true a comparison writes.
            out.push(MirOp::SfpEncc {
                first: 3,
                second: 0,
                mod1: MASK_ON,
            });
            let cond = if from == DType::F32 {
                // `|x|` first: `-0.0` is zero and a NaN is not.
                out.push(MirOp::UnarySfp {
                    op: UnarySfpOp::Abs,
                    a,
                    d: t,
                });
                t
            } else {
                a
            };
            out.push(MirOp::SfpMov { a: zero, d });
            out.push(MirOp::SfpSetcc {
                mod1: SETCC_NE0,
                a: cond,
            });
            out.push(MirOp::LaneConst {
                lreg: d,
                bits: 0xbf80_0000,
            });
            out.push(MirOp::SfpEncc {
                first: 2,
                second: 0,
                mod1: MASK_ON,
            });
        }
        (DType::Bf16, _) | (_, DType::Bf16) => {
            let _ = (a, d, t);
            return Err(ttStatus::TT_TODO_CAST);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `matmul`, the fill and the SFPU all write `Dst` rows the packer reads.
    /// Once a `store_to` has offered its section, the next `Dst` write must wait
    /// for the packer to release it — otherwise the write lands while the pack is
    /// still reading. `reg_sync` derives that seam from the accesses: the packer
    /// posts after its pack, the math thread takes before its next write, and
    /// the post is guarded by the back-pressure wait.
    #[test]
    fn a_dst_write_after_a_post_waits_for_the_packer() {
        const PROGRAM: &str = "\
version 9
grid 1,1
%pa = param f32 1024
%pb = param f32 1024
%out = param f32 2048
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %ba [0] = %pa [0]
copy 32,32 %bb [0] = %pb [0]
%m = acc f32 32,32
%zero = const f32 0
store 32,32 %m = %zero
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
matmul %m %a %b
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
store 32,32 %m = %zero
matmul %m %a %b
store 32,32 %bo [0] = %m
copy 32,32 %out [1] = %bo [0]
";
        let k = Kernel::parse(PROGRAM).expect("it parses");
        k.verify().expect("it verifies");
        let k = crate::mover::add_movers(&k).expect("the movers are added");
        let mut m = k.lower().expect("it lowers");
        crate::optimize::reg_sync(&mut m);
        let ops: Vec<(Core, MirOp)> = m.ops.clone();
        let pack = ops
            .iter()
            .position(|(role, op)| *role == Core::Pack && matches!(op, MirOp::SetPackOffset { .. }))
            .expect("the first store packs") as usize;
        let release = ops[pack..]
            .iter()
            .position(|(_, op)| matches!(op, MirOp::SemPost { .. }))
            .map(|i| i + pack)
            .expect("the packer releases the slot it read");
        let sem = match ops[release].1 {
            MirOp::SemPost { sem } => sem,
            _ => unreachable!("an index found by the SemPost match"),
        };
        // The counter is four bits, so the producer waits below its maximum
        // before it posts.
        assert!(
            matches!(ops[release - 1].1, MirOp::SemWaitFree { sem: s } if s == sem),
            "no back-pressure wait before the post: {:?}",
            ops[release - 1]
        );
        // The next epoch's first write to that slot takes the same edge first.
        let next_write = ops[release..]
            .iter()
            .position(|(role, op)| {
                *role == Core::Math && matches!(op, MirOp::Mvmul { .. } | MirOp::Sfpstore { .. })
            })
            .map(|i| i + release)
            .expect("the second matmul writes Dst again");
        assert!(
            ops[release..next_write].iter().any(|(role, op)| {
                *role == Core::Math && matches!(op, MirOp::SemWait { sem: s, .. } if *s == sem)
            }),
            "a Dst write follows the packer's release with no take"
        );
    }

    /// A broadcast is an FPU op, and the FPU's `Src` path takes `f32` and
    /// `bf16` only: an integer broadcast has to go through the SFPU, which is
    /// not lowered. It must be reported as that route's TODO, never routed
    /// through the FPU.
    #[test]
    fn an_integer_broadcast_is_refused_as_an_sfpu_todo() {
        const PROGRAM: &str = "\
version 9
grid 1,1
%pi = param i32 1024
%po = param i32 1024
%bi = buffer i32 32,32
%bo = buffer i32 32,32
copy 32,32 %bi [0] = %pi [0]
load 32,32 %t = %bi [0]
load 1,1 %s = %bi [0]
%d = add %t %s
store 32,32 %bo [0] = %d
copy 32,32 %po [0] = %bo [0]
";
        let k = Kernel::parse(PROGRAM).expect("it parses");
        k.verify().expect("it verifies");
        let k = crate::mover::add_movers(&k).expect("the movers are added");
        assert_eq!(
            k.lower().expect_err("an integer broadcast has no path"),
            ttStatus::TT_TODO_BROADCAST_SFPU
        );
    }
}
