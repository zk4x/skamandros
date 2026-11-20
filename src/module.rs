// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The compiled program: what a launch runs.

use crate::ir::BOp;

/// What a launch argument for one parameter is: a value or an address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// A scalar parameter: the argument is its value.
    Scalar,
    /// A tensor parameter. Its argument block is the per-channel page bases,
    /// the channel count and the page size, so the program resolves a page's
    /// channel itself — the device-side half of tt-metal's address generator
    /// (`InterleavedAddrGen` / `get_noc_addr_from_bank_id`). Every DRAM tensor
    /// is allocated striped, so the block always has one entry per channel.
    Tensor {
        /// One page's byte size, from the parameter's dtype (a 32x32 tile).
        page: u32,
    },
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

/// The mover programs of a Quasar launch.
///
/// A Quasar tile has no Blackhole-style firmware between the host and the
/// harts: the reader and the writer are two of its movers, run from images the
/// host stages. Their programs are the same machine kernel as Blackhole's with
/// the NoC request staged in a RoCC command buffer instead of the initiator's
/// register block, so they are lowered separately (the movers' address
/// arithmetic is identical, the request is not).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QuasarProgram {
    /// The inbound mover's program (DRAM into L1), words.
    pub reader: Vec<u32>,
    /// The outbound mover's program (L1 into DRAM), words.
    pub writer: Vec<u32>,
    /// The unpack hart's program (TRISC0 of engine 0), or empty when the kernel
    /// has no datapath work.
    pub unpack: Vec<u32>,
    /// The math hart's program (TRISC1).
    pub math: Vec<u32>,
    /// The pack hart's program (TRISC2).
    pub pack: Vec<u32>,
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
    /// The movers' programs for a Quasar launch, when this kernel has no
    /// datapath work (a Quasar launch runs those two harts and nothing else).
    pub quasar: Option<QuasarProgram>,
    /// Why this module has no Quasar program, when the lowering got that far.
    /// A launch on a Quasar reports it instead of guessing: the module is still
    /// valid, and a Blackhole launch runs it exactly as before.
    pub quasar_refusal: Option<crate::status::ttStatus>,
    /// First byte of the reserved kernel config block
    /// (`tt_isa::arch::blackhole::l1::KERNEL_CONFIG`), where the programs are
    /// placed.
    pub program_area: u32,
    /// First byte past the block, where this module's data arena begins. The
    /// block runs from [`Module::program_area`] to here and holds the six
    /// programs, the reader's and writer's argument arrays and the fan-out
    /// lists; it is the reserved minimum unless the programs needed more, in
    /// which case `compile` raised it.
    pub program_end: u32,
    /// The CB interface table the host writes before the launch.
    pub iface: u32,
    /// Byte address of the first multicast site's control block, zero when the
    /// module has none. Each site is four words ([`crate::mir::McastSite`]).
    pub mcast_base: u32,
    /// How many multicast sites follow [`Module::mcast_base`].
    pub mcast_sites: u32,
    /// Byte address of the coordinate table every tile holds: one packed NoC
    /// coordinate per logical `(x, y)` ([`crate::mir::COORDS_WORDS`] words),
    /// zero when the program has no collective.
    pub coords: u32,
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
