// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! Rendering a machine kernel into the module a launch runs.

use crate::encode;
use crate::ir::Kernel;
use crate::mir::{CbInfo, MachineKernel};
use crate::status::ttStatus;

pub use crate::module::{GridTerm, Module, ParamKind};

/// Compile one program: the virtual passes, their check, the lowering, the
/// machine passes, the allocation and the render — in that order, and nothing
/// else.
///
/// The kernel config block is the reserved minimum
/// ([`tt_isa::arch::blackhole::l1::KERNEL_CONFIG_MIN`]) unless the module's
/// programs, arguments and fan-out lists do not fit it, in which case it is
/// raised to what they need and the whole machine program is lowered again: the
/// rings' addresses move with the block's top, so the programs change too. This
/// is tt-metal's `default_l1_kernel_config_size` grown when a program does not
/// fit (`bh_hal_tensix.cpp:44,57,71`).
///
/// Every refusal is an error, never a partial module.
pub fn compile(k: &Kernel) -> Result<Module, ttStatus> {
    // A `loop ... unroll` body is written out first: everything after it sees
    // one straight-line copy per trip, with the induction folded, exactly as if
    // the program had spelled the trips itself.
    let k = crate::unroll::unroll(k)?;
    let k = crate::unfold::unfold(&k)?;
    let k = &k;
    let mut k = crate::mover::add_movers(k)?;
    crate::sync_cbs::sync_cbs(&mut k)?;
    crate::lock_dst::lock_dst(&mut k)?;
    let k = &k;
    // The one check of the virtual program: everything above is required to
    // leave a program this accepts, so a pass cannot hand a broken kernel to
    // the lowering.
    k.verify()?;
    if std::env::var_os("SKAM_IR").is_some() {
        eprintln!("--- virtual kernel ---\n{k}");
    }
    let base = tt_isa::arch::blackhole::l1::KERNEL_CONFIG.base;
    let mut block = u32::try_from(tt_isa::arch::blackhole::l1::KERNEL_CONFIG_MIN)
        .map_err(|_| ttStatus::TT_ERROR_L1_ARENA)?;
    loop {
        let data_base = u32::try_from(base + u64::from(block))
            .map_err(|_| ttStatus::TT_ERROR_L1_ARENA)?;
        let mut machine = k.lower_at(data_base)?;
        // The row bodies are fused before the hand-offs are derived: a value that
        // stays in a lane register has no `Dst` for a seam to carry.
        crate::alloc::fuse_bodies(&mut machine)?;
        // The hand-offs are derived from the accesses, before fusion replaces the
        // cross-core order with the per-core streams.
        crate::optimize::reg_sync(&mut machine)?;
        // The IR locks carry tt-metal's `Dst` section hand-off (a whole section at
        // a time, which the per-slot seams cannot express across a loop). The
        // seams above still order the unpacker's pages and pace `Src`/`Dst` reuse.
        machine.sems.extend(crate::lower::LOCK_SEMS);
        crate::optimize::fuse(&mut machine);
        // The allocator is what brings the lane registers and `Dst` ids inside the
        // hardware's limits, so it runs before the check that enforces them.
        machine.alloc()?;
        machine.verify()?;
        let module = machine.render(data_base)?;
        // Both encodings of the compute programs count: the card stages
        // `silicon_cores` and the simulator `cores`, and the block must hold
        // whichever the target runs.
        let needed = crate::device::program_block_bytes(&[
            module.reader.len(),
            module.writer.len(),
            module.setup.len(),
            module.cores[0].len().max(module.silicon_cores[0].len()),
            module.cores[1].len().max(module.silicon_cores[1].len()),
            module.cores[2].len().max(module.silicon_cores[2].len()),
        ])?;
        if needed <= block {
            if std::env::var_os("SKAM_WORDS").is_some() {
                let end = tt_isa::arch::blackhole::l1::DATA.end;
                eprintln!(
                    "program block: {} bytes ({:#x}), data arena: {} bytes ({:#x})",
                    block,
                    u64::from(block),
                    end - u64::from(data_base),
                    end - u64::from(data_base)
                );
            }
            return Ok(module);
        }
        // Grow the reserved block to what the module needs, keeping the arena
        // 64-byte aligned (`l1::DATA`'s own alignment), and lower again: the
        // rings sit above the block, so every address the programs name moves.
        block = needed.next_multiple_of(64);
    }
}

/// Render one machine kernel: the five programs as words, the setup program
/// thread 0 runs, and the launch metadata the lowering carried over.
///
/// `program_end` is the top of the block the module was lowered into: the
/// first byte of its data arena, which the runtime needs to find the argument
/// arrays.
///
/// `render` is read-only: everything that had to be decided (the rings' places,
/// the registers, the grid and the parameters) was decided before it.
pub fn render(m: &MachineKernel, program_end: u32) -> Result<Module, ttStatus> {
    crate::debug::machine(m);
    let sim = encode::encode(m, false)?;
    // The card's variant: the unpacker's tile-descriptor tail written. Only
    // the unpack program (T0) can differ; the other two are encoded anyway and
    // discarded so the two sets are produced by one function.
    let card = encode::encode(m, true)?;
    let words = sim.words()?;
    crate::debug::programs(&sim);
    // The raw words, for comparing this core's encoding with another's; the
    // listing above is what to read.
    let card_words = card.words()?;
    if std::env::var_os("SKAM_WORDS").is_some() {
        let silicon = &card_words;
        for (i, w) in words.iter().enumerate() {
            eprintln!("role {i}: {} words", w.len());
        }
        for core in 0..3 {
            eprintln!("core {core} sim:  {:08x?}", words[core + 1]);
            eprintln!("core {core} card: {:08x?}", silicon[core + 1]);
        }
        eprintln!("reader sim:  {:08x?}", words[0]);
        eprintln!("writer sim:  {:08x?}", words[4]);
    }
    let setup = encode::setup(&m.sems)?;
    let mut cbs: Vec<CbInfo> = m.cbs.iter().map(|(_, info)| *info).collect();
    cbs.sort_by_key(|info| info.base);
    let iface = m
        .cbs
        .iter()
        .map(|(_, info)| info.iface_entry)
        .min()
        .unwrap_or(0);
    let mut iface_words = Vec::with_capacity(cbs.len() * 8);
    for ring in &cbs {
        iface_words.extend_from_slice(&crate::riscv::cb::interface(
            ring.base,
            ring.page_bytes.checked_mul(ring.len).unwrap_or(0),
            ring.page_bytes,
            ring.len,
        )?);
    }
    let args_words = crate::mover::ARG_HEAD as u32 + module_params_words(&m.params);
    let mcast_base = m.mcast.first().map_or(0, |site| site.base);
    let mut module = Module {
        grid: m.grid.clone(),
        params: m.params.clone(),
        mcast_base,
        mcast_sites: m.mcast.len() as u32,
        coords: m.coords,
        reader: words[0].clone(),
        writer: words[4].clone(),
        cores: [words[1].clone(), words[2].clone(), words[3].clone()],
        silicon_cores: [
            card_words[1].clone(),
            card_words[2].clone(),
            card_words[3].clone(),
        ],
        setup,
        program_area: u32::try_from(tt_isa::arch::blackhole::l1::KERNEL_CONFIG.base)
            .map_err(|_| ttStatus::TT_ERROR_PROGRAM_AREA_SIZE)?,
        program_end,
        iface,
        cbs,
        iface_words,
        args_words,
        hash: 0,
    };
    module.hash = module_hash(&module);
    if module.reader.is_empty()
        && module.writer.is_empty()
        && module.cores.iter().all(Vec::is_empty)
    {
        return Err(ttStatus::TT_ERROR_EMPTY_MODULE);
    }
    Ok(module)
}

/// Words one argument block holds for `params` (`Module::args_words`).
fn module_params_words(params: &[ParamKind]) -> u32 {
    // A tensor names every channel's page base, then the count and the page:
    // the shape `args_for` writes.
    params
        .iter()
        .map(|kind| match kind {
            ParamKind::Tensor { .. } => crate::mover::INTERLEAVED_WORDS as u32,
            ParamKind::Scalar => 2,
        })
        .sum()
}

/// A hash of everything a launch stages on a tile, apart from the resolved
/// grid (`Module::hash`).
fn module_hash(module: &Module) -> u64 {
    let mut words: Vec<u32> = vec![
        module.program_end,
        module.iface,
        module.params.len() as u32,
        module.mcast_base,
        module.mcast_sites,
        module.coords,
    ];
    for program in [&module.reader, &module.writer, &module.setup] {
        words.push(program.len() as u32);
        words.extend_from_slice(program);
    }
    for cores in [&module.cores, &module.silicon_cores] {
        words.push(cores.len() as u32);
        for program in cores {
            words.push(program.len() as u32);
            words.extend_from_slice(program);
        }
    }
    for ring in &module.cbs {
        words.extend_from_slice(&[ring.base, ring.page_bytes, ring.len]);
    }
    crate::device::hash_words(&words)
}

/// The launch's parameter kinds, in program order.
pub fn parameter_kinds(k: &Kernel) -> Vec<ParamKind> {
    let mut kinds = Vec::new();
    for (_, op) in k.ops() {
        match *op {
            // A tensor's page is one 32x32 tile of its dtype: the unit every
            // mover access names, and the size a contiguous buffer is
            // described with.
            crate::ir::Op::Param { dtype, .. } => kinds.push(ParamKind::Tensor {
                page: crate::lower::TILE_DATUMS * u32::from(dtype.bytes()),
            }),
            crate::ir::Op::Variable { .. } => kinds.push(ParamKind::Scalar),
            _ => {}
        }
    }
    kinds
}

/// The launch's grid: the two terms of the program's `grid` op.
pub fn grid_terms(k: &Kernel) -> Result<[GridTerm; 2], ttStatus> {
    let mut grid = None;
    for (_, op) in k.ops() {
        if let crate::ir::Op::Grid { x, y } = *op {
            grid = Some((x, y));
        }
    }
    let Some((x, y)) = grid else {
        return Err(ttStatus::TT_ERROR_GRID);
    };
    Ok([term(k, x)?, term(k, y)?])
}

/// One grid term: a foldable expression is a constant, a scalar parameter is
/// the launch's argument, and anything else is refused.
fn term(k: &Kernel, value: crate::ir::OpId) -> Result<GridTerm, ttStatus> {
    if let Some(n) = k.const_i32(value) {
        return u32::try_from(n)
            .map(GridTerm::Const)
            .map_err(|_| ttStatus::TT_ERROR_GRID);
    }
    if matches!(*k.at(value), crate::ir::Op::Variable { .. }) {
        let ordinal = k
            .ops()
            .filter(|(_, op)| {
                matches!(
                    **op,
                    crate::ir::Op::Param { .. } | crate::ir::Op::Variable { .. }
                )
            })
            .position(|(id, _)| id == value)
            .ok_or(ttStatus::TT_ERROR_GRID)?;
        return Ok(GridTerm::Param(ordinal as u32));
    }
    Err(ttStatus::TT_ERROR_GRID)
}
