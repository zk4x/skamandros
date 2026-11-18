// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The compiled program: what a launch runs.

use crate::ir::BOp;

/// What a launch argument for one parameter is: a value or an address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// A scalar parameter: the argument is its value.
    Scalar,
    /// A tensor parameter: the argument is its DRAM address.
    Tensor,
    /// A tensor parameter whose pages are striped across the DRAM channels.
    /// Its argument block is the per-channel page bases, the channel count and
    /// the page size, so the program can resolve a page's channel itself — the
    /// device-side half of tt-metal's interleaved address generator
    /// (`InterleavedAddrGen` / `get_noc_addr_from_bank_id`).
    Interleaved,
}

/// One grid extent, resolved at launch from the scalar arguments. A `Dim` is a
/// literal or an `i32` value; this is the value's expression, so the runtime can
/// evaluate it once the launch arguments are known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GridTerm {
    Const(u32),
    /// The value of parameter `i` (program order), which must be a scalar.
    Param(u32),
    /// Arithmetic over the two above, evaluated at launch. `grid_eval` reads it;
    /// nothing builds one yet, because `compile::term` refuses anything but a
    /// constant or a scalar parameter.
    #[allow(dead_code)]
    Binary(BOp, Box<GridTerm>, Box<GridTerm>),
}

impl Default for GridTerm {
    fn default() -> Self {
        GridTerm::Const(1)
    }
}

/// One compiled kernel: the mover programs, the three compute programs, the
/// grid they run over, and the parameters the launch supplies.
#[derive(Debug, Clone, Default)]
pub struct Module {
    /// The grid extents, in tiles along x and y.
    pub grid: [GridTerm; 2],
    /// One launch argument per parameter, in program order.
    pub params: Vec<ParamKind>,
    /// The inbound mover program (DRAM into L1), words.
    pub reader: Vec<u32>,
    /// The outbound mover program (L1 into DRAM), words.
    pub writer: Vec<u32>,
    /// The three compute programs, T0/T1/T2, words.
    pub cores: [Vec<u32>; 3],
    /// The same three programs as the card needs them: the unpacker's
    /// tile-descriptor tail (words 2 and 3, both zero for an uncompressed
    /// tile) is written, which `ttsim` refuses to model. tt-metal writes the
    /// whole four-word descriptor every configure (`unpack_tile_descriptor_u`,
    /// `tt_llk_blackhole/common/inc/cunpack_common.h:812-828`), and `Config`
    /// outlives the program that wrote it, so the tail must not be left at
    /// whatever an earlier kernel put there. The launch picks this set on the
    /// card and [`Module::cores`] on the simulator.
    pub silicon_cores: [Vec<u32>; 3],
    /// The setup program thread 0 runs before the three roles are released: the
    /// semaphore initialisation, sized to the `Dst` slots this kernel uses.
    pub setup: Vec<u32>,
    /// The CB interface table the host writes before the launch.
    pub iface: u32,
    /// The rings the table describes, in base-address order.
    pub cbs: Vec<crate::mir::CbInfo>,
    /// The CB interface words every tile's B installs from its own L1
    /// (`riscv::cb::interface` of `cbs`), computed once because they are the
    /// same for every tile of every launch of this module.
    pub iface_words: Vec<u32>,
    /// Words in one argument block: the four the launch always passes, then two
    /// per parameter (eighteen for a striped tensor, which carries its per-bank
    /// page bases).
    pub args_words: u32,
    /// A hash of everything a launch stages on a tile -- the programs, the
    /// rings, the interface, the parameters -- without the resolved grid, which
    /// a launch adds. Two launches with the same fingerprint can reuse the
    /// installation on every tile.
    pub hash: u64,
}
