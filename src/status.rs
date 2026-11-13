// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

#![allow(non_camel_case_types)]

//! The one error enum: what every fallible call returns.
//!
//! It is `#[repr(C)]` and integer-valued, because the C ABI is the only stable
//! surface and a caller sees only the integer. There is no second error type and
//! no string in the enum: information a caller needs is a variant, so every
//! distinct failure has its own.

/// What every fallible call returns.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ttStatus {
    /// Success. `*err` is null.
    TT_OK = 0,
    /// A caller mistake: a null required pointer, a zero length, a wrong
    /// parameter count. Nothing was read or written.
    TT_ERROR_ARGUMENT = 1,
    /// The text or bytes were not a program this compiler accepts.
    TT_ERROR_PROGRAM = 2,
    /// The handle is not one this library handed out, or was already unloaded.
    TT_ERROR_MODULE = 3,
    /// The program compiled, but this build cannot run it yet.
    TT_ERROR_UNSUPPORTED = 4,
    /// The device, a transfer or the launch failed.
    TT_ERROR_DEVICE = 5,
    /// A buffer has more than one producer or more than one consumer. The
    /// one-producer/one-consumer circular buffer is all that is implemented;
    /// the multi-producer/multi-consumer form is a concrete future item.
    TT_TODO_MULTI_PRODUCER_OR_CONSUMER = 6,
    /// A buffer's roles are not one of the implemented pairs (reader/unpacker,
    /// packer/writer, reader/writer). Any other pair, or a missing end, is a
    /// concrete future item.
    TT_TODO_CIRCULAR_BUFFER_ROLE = 7,
    /// The text is not a program this parser accepts (a token, a keyword, a
    /// bracket, a value name).
    TT_ERROR_PARSE = 8,
    /// The version line, or a byte program's version, is not the one this build
    /// reads.
    TT_ERROR_VERSION = 9,
    /// The bytes are not a `TTIR` program.
    TT_ERROR_NOT_TTIR = 10,
    /// The input continues after a complete program.
    TT_ERROR_TRAILING_INPUT = 11,
    /// A buffer's dimension is neither `1` nor a multiple of 32.
    TT_ERROR_BUFFER_SHAPE = 12,
    /// A buffer shape or a parameter dimension is zero.
    TT_ERROR_ZERO_DIMENSION = 13,
    /// A value is used before its definition.
    TT_ERROR_VALUE_UNDEFINED = 14,
    /// A value is used outside the loop it was defined in.
    TT_ERROR_VALUE_OUT_OF_SCOPE = 15,
    /// The same value name is defined twice.
    TT_ERROR_NAME_DUPLICATE = 16,
    /// The program has no grid, or more than one.
    TT_ERROR_GRID = 17,
    /// An operand's dtype or kind does not match; there is no implicit cast.
    TT_ERROR_NO_IMPLICIT_CAST = 18,
    /// A bracket offset runs past its value.
    TT_ERROR_OFFSET = 19,
    /// `endloop` without a matching `loop`.
    TT_ERROR_ENDLOOP_WITHOUT_LOOP = 20,
    /// A loop reaches lowering. Loops become RISC-V branches; that is not
    /// implemented yet.
    TT_TODO_LOOP = 21,
    /// A `sel` (or its source) condition is not a `bool`.
    TT_ERROR_CONDITION_NOT_BOOL = 22,

    // -- Shapes and definitions ------------------------------------------------
    /// A buffer's element count overflows a `u32`.
    TT_ERROR_BUFFER_SIZE_OVERFLOW = 23,
    /// A buffer's byte count overflows a `u32`.
    TT_ERROR_BUFFER_BYTES_OVERFLOW = 24,
    /// An op defines a value whose id is not the next one.
    TT_ERROR_VALUE_ORDER = 25,
    /// A defined value has no name.
    TT_ERROR_VALUE_UNNAMED = 26,
    /// A bf16 constant is not exact; bf16 is written as f32 and cast.
    TT_ERROR_CONST_BF16 = 27,
    /// A bool constant is not a datum; a bool is produced by a comparison.
    TT_ERROR_CONST_BOOL = 28,
    /// A parameter dimension is not an i32 scalar.
    TT_ERROR_DIMENSION_NOT_INDEX = 29,
    /// A loop's bound is not an i32 scalar.
    TT_ERROR_LOOP_BOUND_NOT_INDEX = 30,
    /// A grid dimension is not an i32 scalar.
    TT_ERROR_GRID_DIMENSION_NOT_INDEX = 31,
    /// A copy, load or store offset is not an i32 scalar.
    TT_ERROR_OFFSET_NOT_INDEX = 32,
    /// A copy's two ends are not one parameter and one buffer.
    TT_ERROR_COPY_ENDS = 33,
    /// A copy names a scalar parameter, which cannot hold a buffer unit.
    TT_ERROR_COPY_SCALAR_PARAM = 34,
    /// A copy's buffer and parameter dtypes differ.
    TT_ERROR_COPY_CAST = 35,
    /// An op that needs a buffer was given a value of another kind.
    TT_ERROR_NOT_A_BUFFER = 36,
    /// A direct load reads a parameter that is not a tile.
    TT_ERROR_LOAD_PARAM_NOT_TILE = 37,
    /// A direct store writes a parameter that is not a tile.
    TT_ERROR_STORE_PARAM_NOT_TILE = 38,
    /// A store's value is not a register value.
    TT_ERROR_STORE_VALUE_NOT_REGISTER = 39,
    /// A store to a parameter writes a value of another dtype.
    TT_ERROR_STORE_PARAM_CAST = 40,
    /// A store to a parameter writes a value that is not a tile.
    TT_ERROR_STORE_PARAM_VALUE_NOT_TILE = 41,
    /// A store to a buffer writes a value of another dtype.
    TT_ERROR_STORE_BUFFER_CAST = 42,
    /// A store to a buffer writes a value of another shape.
    TT_ERROR_STORE_BUFFER_SHAPE = 43,
    /// A cast's source is not a register value.
    TT_ERROR_CAST_SOURCE = 44,
    /// A unary op's operand is not a register value.
    TT_ERROR_UNARY_OPERAND = 45,
    /// `not` was given a value that is not a bool.
    TT_ERROR_UNARY_NOT_BOOL = 46,
    /// `bitnot` was given a value that is not i32.
    TT_ERROR_UNARY_BITNOT_NOT_I32 = 47,
    /// A floating-point unary op was given a non-float value.
    TT_ERROR_UNARY_NOT_FLOAT = 48,
    /// A binary op's operand is not a register value.
    TT_ERROR_BINARY_OPERANDS = 49,
    /// A logical op (`and`/`or`) was given a non-bool value.
    TT_ERROR_BINARY_LOGICAL_NOT_BOOL = 50,
    /// An integer-only op was given a non-i32 value.
    TT_ERROR_BINARY_NOT_INTEGER = 51,
    /// Arithmetic was given a bool value.
    TT_ERROR_BINARY_BOOL = 52,
    /// A binary op's two shapes cannot broadcast to one.
    TT_ERROR_BINARY_BROADCAST = 53,
    /// `sel`'s operands are not three register values.
    TT_ERROR_SELECT_OPERANDS = 54,
    /// `sel`'s two values have different dtypes.
    TT_ERROR_SELECT_CAST = 55,
    /// `sel`'s two values cannot broadcast to one shape.
    TT_ERROR_SELECT_BROADCAST = 56,
    /// `sel`'s condition cannot broadcast with the values' shape.
    TT_ERROR_SELECT_CONDITION_BROADCAST = 57,
    /// `matmul`'s operands are not two register values.
    TT_ERROR_MATMUL_OPERANDS = 58,
    /// `matmul`'s operands are not two tiles.
    TT_ERROR_MATMUL_NOT_TILES = 59,
    /// `matmul`'s operands have different dtypes.
    TT_ERROR_MATMUL_CAST = 60,
    /// The program ends inside a loop.
    TT_ERROR_LOOP_UNTERMINATED = 61,
    /// The number of defined values is not the number of names.
    TT_ERROR_VALUE_COUNT = 62,

    // -- Byte form -------------------------------------------------------------
    /// The byte program ends in the middle of a record.
    TT_ERROR_TRUNCATED = 63,
    /// A dtype tag is not one this build knows.
    TT_ERROR_DTYPE_TAG = 64,
    /// A name's bytes are not UTF-8.
    TT_ERROR_NAME_UTF8 = 65,
    /// A shape-term tag is neither a constant nor a value.
    TT_ERROR_SHAPE_TERM_TAG = 66,
    /// A unary op tag is not one this build knows.
    TT_ERROR_UNARY_TAG = 67,
    /// A binary op tag is not one this build knows.
    TT_ERROR_BINARY_TAG = 68,
    /// An instruction tag is not one this build knows.
    TT_ERROR_INSTRUCTION_TAG = 69,

    // -- Lowering --------------------------------------------------------------
    /// A cast reaches lowering.
    TT_TODO_CAST = 70,
    /// An operand shape reaches lowering with a broadcast still to do.
    TT_TODO_BROADCAST = 71,
    /// An op on a dtype the register datapath does not do reaches lowering.
    TT_TODO_BINARY_DTYPE = 72,
    /// `matmul` reaches lowering.
    TT_TODO_MATMUL = 73,
    /// A scalar register value has no row body to load or store.
    TT_ERROR_SCALAR_NO_ROWS = 74,
    /// A symbolic offset reaches lowering; only a literal offset is lowered.
    TT_TODO_SYMBOLIC_OFFSET = 75,
    /// A unary op has no SFPU encoding yet.
    TT_TODO_UNARY_SFPU = 76,
    /// A binary op has no SFPU encoding yet.
    TT_TODO_BINARY_SFPU = 77,
    /// The register datapath is FP32 or I32, not this dtype.
    TT_ERROR_DATAPATH_DTYPE = 78,
    /// A buffer the plan has no address for is accessed.
    TT_ERROR_BUFFER_NOT_PLACED = 79,

    // -- Machine IR verification ----------------------------------------------
    /// A lane register an op names does not exist.
    TT_ERROR_LANE_REGISTER = 80,
    /// A `Dst` row an op names is past the last tile.
    TT_ERROR_DST_ROW = 81,
    /// A semaphore an op names does not exist.
    TT_ERROR_SEMAPHORE = 82,
    /// A repeat with a zero length or a zero count.
    TT_ERROR_REPEAT_EMPTY = 83,
    /// A repeat's body runs past the program.
    TT_ERROR_REPEAT_BODY = 84,

    // -- Unrolling -------------------------------------------------------------
    /// A folded index is negative.
    TT_ERROR_NEGATIVE_INDEX = 85,
    /// A value is used where it is not in scope while unrolling.
    TT_ERROR_VALUE_NOT_IN_SCOPE = 86,
    /// A loop has no matching `endloop` while unrolling.
    TT_ERROR_LOOP_UNMATCHED = 87,

    // -- Planning --------------------------------------------------------------
    /// A buffer's end overflows the L1 address space.
    TT_ERROR_L1_OVERFLOW = 88,
    /// A buffer runs past the L1 data arena.
    TT_ERROR_L1_ARENA = 89,

    // -- Movers ----------------------------------------------------------------
    /// A direct access to a non-tile parameter needs the 64-byte span read,
    /// which the movers do not do yet.
    TT_TODO_SCALAR_PARAMETER = 90,
    /// A copy's parameter byte offset overflows.
    TT_ERROR_PARAM_OFFSET_OVERFLOW = 91,
    /// A copy's buffer byte offset overflows.
    TT_ERROR_BUFFER_OFFSET_OVERFLOW = 92,
    /// A copy's buffer address overflows.
    TT_ERROR_BUFFER_ADDRESS_OVERFLOW = 93,
    /// A copy's size is not one NoC request.
    TT_ERROR_NOC_REQUEST_BYTES = 94,
    /// The mover assembler could not resolve its program.
    TT_ERROR_MOVER_PROGRAM = 95,

    // -- Allocation ------------------------------------------------------------
    /// A value needs a `Dst` slot and every slot is live.
    TT_ERROR_DST_SLOTS_EXHAUSTED = 96,

    // -- Encoding --------------------------------------------------------------
    /// An instruction has no inline `.ttinsn` form.
    TT_ERROR_INSTRUCTION_NOT_INLINE = 97,
    /// A semaphore could not be initialised.
    TT_ERROR_SEMINIT = 98,
    /// The unpack program cannot encode this machine op.
    TT_ERROR_UNPACK_OP = 99,
    /// `STALLWAIT` could not be encoded.
    TT_ERROR_STALLWAIT = 100,
    /// `SETADCXX` could not be encoded.
    TT_ERROR_SETADCXX = 101,
    /// `SETADCZW` could not be encoded.
    TT_ERROR_SETADCZW = 102,
    /// `UNPACR` could not be encoded.
    TT_ERROR_UNPACR = 103,
    /// A math repeat's body runs past the program.
    TT_ERROR_MATH_REPEAT_BODY = 104,
    /// A math instruction could not be encoded.
    TT_ERROR_ENCODE = 105,
    /// A math op has no encoding yet.
    TT_ERROR_NO_ENCODING = 106,
    /// An op in the math program is not a math instruction.
    TT_ERROR_NOT_MATH = 107,
    /// The pack program cannot encode this machine op.
    TT_ERROR_PACK_OP = 108,
    /// `SETADCXX` in the packer could not be encoded.
    TT_ERROR_SETADCXX_PACK = 109,
    /// `SETADCXY` in the packer could not be encoded.
    TT_ERROR_SETADCXY_PACK = 110,
    /// `PACR` could not be encoded.
    TT_ERROR_PACR = 111,

    // -- Device ----------------------------------------------------------------
    /// The simulator library could not be loaded.
    TT_ERROR_DLOPEN = 112,
    /// The simulator library lacks an entry point.
    TT_ERROR_DLSYM = 113,
    /// No chip answers configuration space.
    TT_ERROR_NO_DEVICE = 114,
    /// The device is not a Blackhole.
    TT_ERROR_NOT_BLACKHOLE = 115,
    /// The chip has no Tensix tile.
    TT_ERROR_NO_TENSIX_TILE = 116,
    /// A zero-byte DRAM allocation.
    TT_ERROR_ALLOC_ZERO = 117,
    /// A DRAM allocation overflows the address space.
    TT_ERROR_DRAM_ADDRESS_OVERFLOW = 118,
    /// A DRAM allocation does not fit the channel.
    TT_ERROR_DRAM_FULL = 119,
    /// A DRAM transfer does not fit the channel.
    TT_ERROR_DRAM_RANGE = 120,
    /// A core cannot be loaded at the requested L1 offset.
    TT_ERROR_RESET_PC = 121,
    /// The firmware panicked.
    TT_ERROR_FIRMWARE_PANIC = 122,
    /// The firmware did not answer within the cycle budget.
    TT_ERROR_FIRMWARE_TIMEOUT = 123,
    /// An empty program cannot be run.
    TT_ERROR_EMPTY_PROGRAM = 124,
    /// A program is longer than its limit.
    TT_ERROR_PROGRAM_TOO_LONG = 125,
    /// A program would run into its arguments.
    TT_ERROR_PROGRAM_OVERLAPS_ARGS = 126,
    /// The executor refused the program.
    TT_ERROR_PROGRAM_REFUSED = 127,
    /// An L1 address and length overflow.
    TT_ERROR_L1_ADDRESS_OVERFLOW = 128,
    /// An L1 access runs past L1.
    TT_ERROR_L1_ACCESS = 129,
    /// A zero-length BAR0 access.
    TT_ERROR_BAR0_ZERO_LENGTH = 130,
    /// A BAR0 access overflows.
    TT_ERROR_BAR0_OVERFLOW = 131,
    /// A BAR0 access is not decoded.
    TT_ERROR_BAR0_NOT_DECODED = 132,
    /// A window is not aligned.
    TT_ERROR_WINDOW_ALIGNMENT = 133,

    // -- Runtime ---------------------------------------------------------------
    /// A parameter address does not fit in a word.
    TT_ERROR_PARAMETER_ADDRESS = 134,
    /// The staged-program area does not fit in a word.
    TT_ERROR_PROGRAM_AREA = 135,
    /// A mover program does not fit the staged-program area.
    TT_ERROR_PROGRAM_AREA_SIZE = 136,
    /// A role program is longer than its mailbox slot holds.
    TT_ERROR_ROLE_PROGRAM_SIZE = 137,
    /// The device mutex is poisoned.
    TT_ERROR_DEVICE_LOCK = 138,
    /// A symbolic grid reaches lowering.
    TT_TODO_SYMBOLIC_GRID = 139,
    /// The program has more parameters than a launch can pass.
    TT_ERROR_TOO_MANY_PARAMETERS = 140,
    /// The program does no data movement and no compute.
    TT_ERROR_EMPTY_MODULE = 141,
    /// An unpack's X range is empty.
    TT_ERROR_UNPACK_ZERO_RANGE = 142,
    /// An unpack of zero datums.
    TT_ERROR_UNPACK_ZERO_DATUMS = 143,
    /// A pack of zero rows.
    TT_ERROR_PACK_ZERO_ROWS = 144,
    /// A loop bound is not a constant, so it cannot become a RISC-V trip count.
    TT_ERROR_LOOP_BOUND_SYMBOLIC = 145,
    /// A loop bigger than the caller-saved registers allow nested branches for.
    TT_TODO_LOOP_DEPTH = 146,
    /// A loop-carried address expression is beyond what the movers can lower to
    /// RISC-V (a tree that needs more temporaries than the loop leaves free).
    TT_TODO_LOOP_ADDRESS = 147,
    /// A machine program's `endloop` has no matching `loop`.
    TT_ERROR_ENDLOOP_MACHINE = 148,
    /// A grid coordinate names an axis other than x or y.
    TT_ERROR_GRID_AXIS = 149,
    /// A grid term depends on this core's coordinate, which does not exist
    /// before the grid is dispatched.
    TT_ERROR_GRID_COORD = 150,
    /// A bitcast names a dtype of a different width: the bits would not all be
    /// the value.
    TT_ERROR_BITCAST_WIDTH = 151,
    /// The ARC's telemetry did not name this chip's Tensix columns usably: no
    /// table pointer, a header that does not parse, no `ENABLED_TENSIX_COL`
    /// entry, or a mask that contradicts the specification (holes in the middle,
    /// or more columns than the part has).
    TT_ERROR_TELEMETRY = 152,
    /// A bf16 value has to cross the `Dst` boundary into or out of a 32-bit
    /// datum. The unpacker cannot widen bf16 into a 32-bit `Dst`, and a 16-bit
    /// `Dst` cannot pack `f32`/`i32` (`Unpackers/FormatConversion.md`,
    /// `tt_llk_*/common/inc/cunpack_common.h` and `cpack_common.h`), so the
    /// conversion needs `SrcA`/`SrcB` and the Matrix Unit, which this compiler
    /// does not place.
    TT_ERROR_BF16_WIDEN = 153,
    /// `matmul`'s operands or its accumulator's dtype are not a pair the Matrix
    /// Unit's `Src` path takes: both operands must be `f32` or `bf16` and equal,
    /// the accumulator must be `f32` or `bf16`, and `f32` operands cannot
    /// accumulate into a 16-bit `Dst` (the `Src` format would be TF32 and the
    /// bank 16-bit).
    TT_ERROR_MATMUL_DTYPE = 154,
    /// An accumulator is `bf16` while some other value in the module forces the
    /// 32-bit `Dst`: the bank is one width, and the Matrix Unit would then
    /// accumulate in FP32, not at the `bf16` the accumulator names.
    TT_ERROR_MATMUL_ACCUM_WIDE = 155,
    /// A `matmul` operand is not a `load` value. The Matrix Unit reads
    /// `SrcA`/`SrcB`, which the unpacker fills straight from a CB page; a value
    /// the datapath computed lives in `Dst` and has no page to unpack.
    TT_ERROR_MATMUL_OPERAND = 156,
    /// A machine `MVMUL` names an address-modifier entry the Matrix Unit does
    /// not have (three bits on Blackhole).
    TT_ERROR_ADDR_MOD = 158,
    /// A `SETRWC` counter value does not fit its four-bit field.
    TT_ERROR_SETRWC = 159,
    /// A `matmul`'s destination is not an `acc` value. The Matrix Unit writes a
    /// `Dst` tile that is named once and accumulated into; an immutable value or
    /// a buffer has no accumulator to write.
    TT_ERROR_MATMUL_ACC = 160,
    /// An `acc` is declared with a shape other than `1,32`, `32,1`, `1,1` or
    /// `32,32`, or a `store_to` fills one with an access that is not a tile.
    TT_ERROR_ACC_SHAPE = 161,
    /// An `acc`'s dtype is neither `f32` nor `bf16`.
    TT_ERROR_ACC_DTYPE = 162,
    /// A `store_to` names an `acc` with an offset. An `acc` is one tile; there
    /// is nothing to index.
    TT_ERROR_ACC_OFFSET = 163,
    /// A `store_to` into an `acc` names a value that is not a scalar constant.
    /// Only a fill from a constant is implemented; a tile value moving into an
    /// accumulator needs the SFPU to walk both tiles.
    TT_TODO_ACC_STORE = 164,
    /// A machine `SFPSTORE` names an odd `Dst` row. The store's
    /// `dest_reg_addr` names a row pair and the hardware refuses an odd one.
    TT_ERROR_DST_ROW_ODD = 165,
    /// A reduce's destination is not an `acc` value. A reduce folds into a
    /// named `Dst` tile, which is accumulated into and packed out.
    TT_ERROR_REDUCE_ACC = 166,
    /// A reduce's operand is not a `load` page. The folding happens on the Src
    /// path, from an L1 page, never from a computed register value.
    TT_ERROR_REDUCE_OPERAND = 167,
    /// A reduce's dtypes are outside what the Src path takes: `f32` and `bf16`
    /// operands of one dtype, an `f32` or `bf16` accumulator, and no `f32`
    /// operand accumulating into a `bf16` accumulator.
    TT_ERROR_REDUCE_DTYPE = 168,
    /// A reduce form that is not lowered yet.
    TT_TODO_REDUCE = 169,
    /// A machine `MOVD2B` names a `SrcB` row the bank does not have (64 rows of
    /// 16 datums).
    TT_ERROR_SRC_ROW = 170,
    /// An access's shape is not one of `1,1`, `1,32`, `32,1`, `32,32`.
    TT_ERROR_ACCESS_SHAPE = 171,
    /// A `copy` unit is not 64-byte congruent: a datum or a column cannot be a
    /// NoC request.
    TT_ERROR_COPY_UNIT = 172,
    /// A shaped index on the L1 side of a copy (one page per base) is not
    /// lowered yet.
    TT_TODO_MULTI_LANE = 173,
    /// A `load`/`store` unit that is not a tile (a row, a column, one datum) is
    /// not lowered yet.
    TT_TODO_ACCESS_SHAPE = 174,
    /// A copy's lanes do not fill one L1 page: every lane moves one unit, and
    /// `lanes` units have to be exactly the page's datums.
    TT_ERROR_COPY_LANES = 175,
    /// A broadcast whose dtype the FPU's `Src` path cannot take (`i32`), so it
    /// needs the SFPU route, which is not lowered yet. The FPU takes `f32` and
    /// `bf16`; anything else must not be routed through it.
    TT_TODO_BROADCAST_SFPU = 176,
    /// A reduce names a `32,32` accumulator. An `acc`'s shape is the fold, and
    /// `32,32` has none: it is the `matmul` accumulator.
    TT_ERROR_REDUCE_ACC_SHAPE = 177,
    /// A `tf32` dtype is not implemented yet: the value is a real format
    /// (tt-metal's `DataFormat::Tf32`, a 19-bit `Src` layout), but no program
    /// can name it until the storage, the Src format selection and the
    /// explicit casts are wired.
    TT_TODO_TF32 = 178,
}

impl ttStatus {
    /// The variant's name, for the ABI's human-readable `err` string. Not state:
    /// the enum itself carries no string.
    pub const fn name(self) -> &'static str {
        match self {
            ttStatus::TT_OK => "TT_OK",
            ttStatus::TT_ERROR_ARGUMENT => "TT_ERROR_ARGUMENT",
            ttStatus::TT_ERROR_PROGRAM => "TT_ERROR_PROGRAM",
            ttStatus::TT_ERROR_MODULE => "TT_ERROR_MODULE",
            ttStatus::TT_ERROR_UNSUPPORTED => "TT_ERROR_UNSUPPORTED",
            ttStatus::TT_ERROR_DEVICE => "TT_ERROR_DEVICE",
            ttStatus::TT_TODO_MULTI_PRODUCER_OR_CONSUMER => "TT_TODO_MULTI_PRODUCER_OR_CONSUMER",
            ttStatus::TT_TODO_CIRCULAR_BUFFER_ROLE => "TT_TODO_CIRCULAR_BUFFER_ROLE",
            ttStatus::TT_ERROR_PARSE => "TT_ERROR_PARSE",
            ttStatus::TT_ERROR_VERSION => "TT_ERROR_VERSION",
            ttStatus::TT_ERROR_NOT_TTIR => "TT_ERROR_NOT_TTIR",
            ttStatus::TT_ERROR_TRAILING_INPUT => "TT_ERROR_TRAILING_INPUT",
            ttStatus::TT_ERROR_BUFFER_SHAPE => "TT_ERROR_BUFFER_SHAPE",
            ttStatus::TT_ERROR_ZERO_DIMENSION => "TT_ERROR_ZERO_DIMENSION",
            ttStatus::TT_ERROR_VALUE_UNDEFINED => "TT_ERROR_VALUE_UNDEFINED",
            ttStatus::TT_ERROR_VALUE_OUT_OF_SCOPE => "TT_ERROR_VALUE_OUT_OF_SCOPE",
            ttStatus::TT_ERROR_NAME_DUPLICATE => "TT_ERROR_NAME_DUPLICATE",
            ttStatus::TT_ERROR_GRID => "TT_ERROR_GRID",
            ttStatus::TT_ERROR_NO_IMPLICIT_CAST => "TT_ERROR_NO_IMPLICIT_CAST",
            ttStatus::TT_ERROR_OFFSET => "TT_ERROR_OFFSET",
            ttStatus::TT_ERROR_ENDLOOP_WITHOUT_LOOP => "TT_ERROR_ENDLOOP_WITHOUT_LOOP",
            ttStatus::TT_TODO_LOOP => "TT_TODO_LOOP",
            ttStatus::TT_ERROR_CONDITION_NOT_BOOL => "TT_ERROR_CONDITION_NOT_BOOL",
            ttStatus::TT_ERROR_BUFFER_SIZE_OVERFLOW => "TT_ERROR_BUFFER_SIZE_OVERFLOW",
            ttStatus::TT_ERROR_BUFFER_BYTES_OVERFLOW => "TT_ERROR_BUFFER_BYTES_OVERFLOW",
            ttStatus::TT_ERROR_VALUE_ORDER => "TT_ERROR_VALUE_ORDER",
            ttStatus::TT_ERROR_VALUE_UNNAMED => "TT_ERROR_VALUE_UNNAMED",
            ttStatus::TT_ERROR_CONST_BF16 => "TT_ERROR_CONST_BF16",
            ttStatus::TT_ERROR_CONST_BOOL => "TT_ERROR_CONST_BOOL",
            ttStatus::TT_ERROR_DIMENSION_NOT_INDEX => "TT_ERROR_DIMENSION_NOT_INDEX",
            ttStatus::TT_ERROR_LOOP_BOUND_NOT_INDEX => "TT_ERROR_LOOP_BOUND_NOT_INDEX",
            ttStatus::TT_ERROR_GRID_DIMENSION_NOT_INDEX => "TT_ERROR_GRID_DIMENSION_NOT_INDEX",
            ttStatus::TT_ERROR_OFFSET_NOT_INDEX => "TT_ERROR_OFFSET_NOT_INDEX",
            ttStatus::TT_ERROR_COPY_ENDS => "TT_ERROR_COPY_ENDS",
            ttStatus::TT_ERROR_COPY_SCALAR_PARAM => "TT_ERROR_COPY_SCALAR_PARAM",
            ttStatus::TT_ERROR_COPY_CAST => "TT_ERROR_COPY_CAST",
            ttStatus::TT_ERROR_NOT_A_BUFFER => "TT_ERROR_NOT_A_BUFFER",
            ttStatus::TT_ERROR_LOAD_PARAM_NOT_TILE => "TT_ERROR_LOAD_PARAM_NOT_TILE",
            ttStatus::TT_ERROR_STORE_PARAM_NOT_TILE => "TT_ERROR_STORE_PARAM_NOT_TILE",
            ttStatus::TT_ERROR_STORE_VALUE_NOT_REGISTER => "TT_ERROR_STORE_VALUE_NOT_REGISTER",
            ttStatus::TT_ERROR_STORE_PARAM_CAST => "TT_ERROR_STORE_PARAM_CAST",
            ttStatus::TT_ERROR_STORE_PARAM_VALUE_NOT_TILE => "TT_ERROR_STORE_PARAM_VALUE_NOT_TILE",
            ttStatus::TT_ERROR_STORE_BUFFER_CAST => "TT_ERROR_STORE_BUFFER_CAST",
            ttStatus::TT_ERROR_STORE_BUFFER_SHAPE => "TT_ERROR_STORE_BUFFER_SHAPE",
            ttStatus::TT_ERROR_CAST_SOURCE => "TT_ERROR_CAST_SOURCE",
            ttStatus::TT_ERROR_UNARY_OPERAND => "TT_ERROR_UNARY_OPERAND",
            ttStatus::TT_ERROR_UNARY_NOT_BOOL => "TT_ERROR_UNARY_NOT_BOOL",
            ttStatus::TT_ERROR_UNARY_BITNOT_NOT_I32 => "TT_ERROR_UNARY_BITNOT_NOT_I32",
            ttStatus::TT_ERROR_UNARY_NOT_FLOAT => "TT_ERROR_UNARY_NOT_FLOAT",
            ttStatus::TT_ERROR_BINARY_OPERANDS => "TT_ERROR_BINARY_OPERANDS",
            ttStatus::TT_ERROR_BINARY_LOGICAL_NOT_BOOL => "TT_ERROR_BINARY_LOGICAL_NOT_BOOL",
            ttStatus::TT_ERROR_BINARY_NOT_INTEGER => "TT_ERROR_BINARY_NOT_INTEGER",
            ttStatus::TT_ERROR_BINARY_BOOL => "TT_ERROR_BINARY_BOOL",
            ttStatus::TT_ERROR_BINARY_BROADCAST => "TT_ERROR_BINARY_BROADCAST",
            ttStatus::TT_ERROR_SELECT_OPERANDS => "TT_ERROR_SELECT_OPERANDS",
            ttStatus::TT_ERROR_SELECT_CAST => "TT_ERROR_SELECT_CAST",
            ttStatus::TT_ERROR_SELECT_BROADCAST => "TT_ERROR_SELECT_BROADCAST",
            ttStatus::TT_ERROR_SELECT_CONDITION_BROADCAST => "TT_ERROR_SELECT_CONDITION_BROADCAST",
            ttStatus::TT_ERROR_MATMUL_OPERANDS => "TT_ERROR_MATMUL_OPERANDS",
            ttStatus::TT_ERROR_MATMUL_NOT_TILES => "TT_ERROR_MATMUL_NOT_TILES",
            ttStatus::TT_ERROR_MATMUL_CAST => "TT_ERROR_MATMUL_CAST",
            ttStatus::TT_ERROR_LOOP_UNTERMINATED => "TT_ERROR_LOOP_UNTERMINATED",
            ttStatus::TT_ERROR_VALUE_COUNT => "TT_ERROR_VALUE_COUNT",
            ttStatus::TT_ERROR_TRUNCATED => "TT_ERROR_TRUNCATED",
            ttStatus::TT_ERROR_DTYPE_TAG => "TT_ERROR_DTYPE_TAG",
            ttStatus::TT_ERROR_NAME_UTF8 => "TT_ERROR_NAME_UTF8",
            ttStatus::TT_ERROR_SHAPE_TERM_TAG => "TT_ERROR_SHAPE_TERM_TAG",
            ttStatus::TT_ERROR_UNARY_TAG => "TT_ERROR_UNARY_TAG",
            ttStatus::TT_ERROR_BINARY_TAG => "TT_ERROR_BINARY_TAG",
            ttStatus::TT_ERROR_INSTRUCTION_TAG => "TT_ERROR_INSTRUCTION_TAG",
            ttStatus::TT_TODO_CAST => "TT_TODO_CAST",
            ttStatus::TT_TODO_BROADCAST => "TT_TODO_BROADCAST",
            ttStatus::TT_TODO_BINARY_DTYPE => "TT_TODO_BINARY_DTYPE",
            ttStatus::TT_TODO_MATMUL => "TT_TODO_MATMUL",
            ttStatus::TT_ERROR_SCALAR_NO_ROWS => "TT_ERROR_SCALAR_NO_ROWS",
            ttStatus::TT_TODO_SYMBOLIC_OFFSET => "TT_TODO_SYMBOLIC_OFFSET",
            ttStatus::TT_TODO_UNARY_SFPU => "TT_TODO_UNARY_SFPU",
            ttStatus::TT_TODO_BINARY_SFPU => "TT_TODO_BINARY_SFPU",
            ttStatus::TT_ERROR_DATAPATH_DTYPE => "TT_ERROR_DATAPATH_DTYPE",
            ttStatus::TT_ERROR_BUFFER_NOT_PLACED => "TT_ERROR_BUFFER_NOT_PLACED",
            ttStatus::TT_ERROR_LANE_REGISTER => "TT_ERROR_LANE_REGISTER",
            ttStatus::TT_ERROR_DST_ROW => "TT_ERROR_DST_ROW",
            ttStatus::TT_ERROR_SEMAPHORE => "TT_ERROR_SEMAPHORE",
            ttStatus::TT_ERROR_REPEAT_EMPTY => "TT_ERROR_REPEAT_EMPTY",
            ttStatus::TT_ERROR_REPEAT_BODY => "TT_ERROR_REPEAT_BODY",
            ttStatus::TT_ERROR_NEGATIVE_INDEX => "TT_ERROR_NEGATIVE_INDEX",
            ttStatus::TT_ERROR_VALUE_NOT_IN_SCOPE => "TT_ERROR_VALUE_NOT_IN_SCOPE",
            ttStatus::TT_ERROR_LOOP_UNMATCHED => "TT_ERROR_LOOP_UNMATCHED",
            ttStatus::TT_ERROR_L1_OVERFLOW => "TT_ERROR_L1_OVERFLOW",
            ttStatus::TT_ERROR_L1_ARENA => "TT_ERROR_L1_ARENA",
            ttStatus::TT_TODO_SCALAR_PARAMETER => "TT_TODO_SCALAR_PARAMETER",
            ttStatus::TT_ERROR_PARAM_OFFSET_OVERFLOW => "TT_ERROR_PARAM_OFFSET_OVERFLOW",
            ttStatus::TT_ERROR_BUFFER_OFFSET_OVERFLOW => "TT_ERROR_BUFFER_OFFSET_OVERFLOW",
            ttStatus::TT_ERROR_BUFFER_ADDRESS_OVERFLOW => "TT_ERROR_BUFFER_ADDRESS_OVERFLOW",
            ttStatus::TT_ERROR_NOC_REQUEST_BYTES => "TT_ERROR_NOC_REQUEST_BYTES",
            ttStatus::TT_ERROR_MOVER_PROGRAM => "TT_ERROR_MOVER_PROGRAM",
            ttStatus::TT_ERROR_DST_SLOTS_EXHAUSTED => "TT_ERROR_DST_SLOTS_EXHAUSTED",
            ttStatus::TT_ERROR_INSTRUCTION_NOT_INLINE => "TT_ERROR_INSTRUCTION_NOT_INLINE",
            ttStatus::TT_ERROR_SEMINIT => "TT_ERROR_SEMINIT",
            ttStatus::TT_ERROR_UNPACK_OP => "TT_ERROR_UNPACK_OP",
            ttStatus::TT_ERROR_STALLWAIT => "TT_ERROR_STALLWAIT",
            ttStatus::TT_ERROR_SETADCXX => "TT_ERROR_SETADCXX",
            ttStatus::TT_ERROR_SETADCZW => "TT_ERROR_SETADCZW",
            ttStatus::TT_ERROR_UNPACR => "TT_ERROR_UNPACR",
            ttStatus::TT_ERROR_MATH_REPEAT_BODY => "TT_ERROR_MATH_REPEAT_BODY",
            ttStatus::TT_ERROR_ENCODE => "TT_ERROR_ENCODE",
            ttStatus::TT_ERROR_NO_ENCODING => "TT_ERROR_NO_ENCODING",
            ttStatus::TT_ERROR_NOT_MATH => "TT_ERROR_NOT_MATH",
            ttStatus::TT_ERROR_PACK_OP => "TT_ERROR_PACK_OP",
            ttStatus::TT_ERROR_SETADCXX_PACK => "TT_ERROR_SETADCXX_PACK",
            ttStatus::TT_ERROR_SETADCXY_PACK => "TT_ERROR_SETADCXY_PACK",
            ttStatus::TT_ERROR_PACR => "TT_ERROR_PACR",
            ttStatus::TT_ERROR_DLOPEN => "TT_ERROR_DLOPEN",
            ttStatus::TT_ERROR_DLSYM => "TT_ERROR_DLSYM",
            ttStatus::TT_ERROR_NO_DEVICE => "TT_ERROR_NO_DEVICE",
            ttStatus::TT_ERROR_NOT_BLACKHOLE => "TT_ERROR_NOT_BLACKHOLE",
            ttStatus::TT_ERROR_NO_TENSIX_TILE => "TT_ERROR_NO_TENSIX_TILE",
            ttStatus::TT_ERROR_ALLOC_ZERO => "TT_ERROR_ALLOC_ZERO",
            ttStatus::TT_ERROR_DRAM_ADDRESS_OVERFLOW => "TT_ERROR_DRAM_ADDRESS_OVERFLOW",
            ttStatus::TT_ERROR_DRAM_FULL => "TT_ERROR_DRAM_FULL",
            ttStatus::TT_ERROR_DRAM_RANGE => "TT_ERROR_DRAM_RANGE",
            ttStatus::TT_ERROR_RESET_PC => "TT_ERROR_RESET_PC",
            ttStatus::TT_ERROR_FIRMWARE_PANIC => "TT_ERROR_FIRMWARE_PANIC",
            ttStatus::TT_ERROR_FIRMWARE_TIMEOUT => "TT_ERROR_FIRMWARE_TIMEOUT",
            ttStatus::TT_ERROR_EMPTY_PROGRAM => "TT_ERROR_EMPTY_PROGRAM",
            ttStatus::TT_ERROR_PROGRAM_TOO_LONG => "TT_ERROR_PROGRAM_TOO_LONG",
            ttStatus::TT_ERROR_PROGRAM_OVERLAPS_ARGS => "TT_ERROR_PROGRAM_OVERLAPS_ARGS",
            ttStatus::TT_ERROR_PROGRAM_REFUSED => "TT_ERROR_PROGRAM_REFUSED",
            ttStatus::TT_ERROR_L1_ADDRESS_OVERFLOW => "TT_ERROR_L1_ADDRESS_OVERFLOW",
            ttStatus::TT_ERROR_L1_ACCESS => "TT_ERROR_L1_ACCESS",
            ttStatus::TT_ERROR_BAR0_ZERO_LENGTH => "TT_ERROR_BAR0_ZERO_LENGTH",
            ttStatus::TT_ERROR_BAR0_OVERFLOW => "TT_ERROR_BAR0_OVERFLOW",
            ttStatus::TT_ERROR_BAR0_NOT_DECODED => "TT_ERROR_BAR0_NOT_DECODED",
            ttStatus::TT_ERROR_WINDOW_ALIGNMENT => "TT_ERROR_WINDOW_ALIGNMENT",
            ttStatus::TT_ERROR_PARAMETER_ADDRESS => "TT_ERROR_PARAMETER_ADDRESS",
            ttStatus::TT_ERROR_PROGRAM_AREA => "TT_ERROR_PROGRAM_AREA",
            ttStatus::TT_ERROR_PROGRAM_AREA_SIZE => "TT_ERROR_PROGRAM_AREA_SIZE",
            ttStatus::TT_ERROR_ROLE_PROGRAM_SIZE => "TT_ERROR_ROLE_PROGRAM_SIZE",
            ttStatus::TT_ERROR_DEVICE_LOCK => "TT_ERROR_DEVICE_LOCK",
            ttStatus::TT_TODO_SYMBOLIC_GRID => "TT_TODO_SYMBOLIC_GRID",
            ttStatus::TT_ERROR_TOO_MANY_PARAMETERS => "TT_ERROR_TOO_MANY_PARAMETERS",
            ttStatus::TT_ERROR_EMPTY_MODULE => "TT_ERROR_EMPTY_MODULE",
            ttStatus::TT_ERROR_UNPACK_ZERO_RANGE => "TT_ERROR_UNPACK_ZERO_RANGE",
            ttStatus::TT_ERROR_UNPACK_ZERO_DATUMS => "TT_ERROR_UNPACK_ZERO_DATUMS",
            ttStatus::TT_ERROR_PACK_ZERO_ROWS => "TT_ERROR_PACK_ZERO_ROWS",
            ttStatus::TT_ERROR_LOOP_BOUND_SYMBOLIC => "TT_ERROR_LOOP_BOUND_SYMBOLIC",
            ttStatus::TT_TODO_LOOP_DEPTH => "TT_TODO_LOOP_DEPTH",
            ttStatus::TT_TODO_LOOP_ADDRESS => "TT_TODO_LOOP_ADDRESS",
            ttStatus::TT_ERROR_ENDLOOP_MACHINE => "TT_ERROR_ENDLOOP_MACHINE",
            ttStatus::TT_ERROR_GRID_AXIS => "TT_ERROR_GRID_AXIS",
            ttStatus::TT_ERROR_GRID_COORD => "TT_ERROR_GRID_COORD",
            ttStatus::TT_ERROR_BITCAST_WIDTH => "TT_ERROR_BITCAST_WIDTH",
            ttStatus::TT_ERROR_TELEMETRY => "TT_ERROR_TELEMETRY",
            ttStatus::TT_ERROR_BF16_WIDEN => "TT_ERROR_BF16_WIDEN",
            ttStatus::TT_ERROR_MATMUL_DTYPE => "TT_ERROR_MATMUL_DTYPE",
            ttStatus::TT_ERROR_MATMUL_ACCUM_WIDE => "TT_ERROR_MATMUL_ACCUM_WIDE",
            ttStatus::TT_ERROR_MATMUL_OPERAND => "TT_ERROR_MATMUL_OPERAND",
            ttStatus::TT_ERROR_ADDR_MOD => "TT_ERROR_ADDR_MOD",
            ttStatus::TT_ERROR_SETRWC => "TT_ERROR_SETRWC",
            ttStatus::TT_ERROR_MATMUL_ACC => "TT_ERROR_MATMUL_ACC",
            ttStatus::TT_ERROR_ACC_SHAPE => "TT_ERROR_ACC_SHAPE",
            ttStatus::TT_ERROR_ACC_DTYPE => "TT_ERROR_ACC_DTYPE",
            ttStatus::TT_ERROR_ACC_OFFSET => "TT_ERROR_ACC_OFFSET",
            ttStatus::TT_TODO_ACC_STORE => "TT_TODO_ACC_STORE",
            ttStatus::TT_ERROR_REDUCE_ACC => "TT_ERROR_REDUCE_ACC",
            ttStatus::TT_ERROR_REDUCE_OPERAND => "TT_ERROR_REDUCE_OPERAND",
            ttStatus::TT_ERROR_REDUCE_DTYPE => "TT_ERROR_REDUCE_DTYPE",
            ttStatus::TT_TODO_REDUCE => "TT_TODO_REDUCE",
            ttStatus::TT_ERROR_SRC_ROW => "TT_ERROR_SRC_ROW",
            ttStatus::TT_ERROR_ACCESS_SHAPE => "TT_ERROR_ACCESS_SHAPE",
            ttStatus::TT_ERROR_COPY_UNIT => "TT_ERROR_COPY_UNIT",
            ttStatus::TT_TODO_MULTI_LANE => "TT_TODO_MULTI_LANE",
            ttStatus::TT_TODO_ACCESS_SHAPE => "TT_TODO_ACCESS_SHAPE",
            ttStatus::TT_ERROR_COPY_LANES => "TT_ERROR_COPY_LANES",
            ttStatus::TT_TODO_BROADCAST_SFPU => "TT_TODO_BROADCAST_SFPU",
            ttStatus::TT_ERROR_REDUCE_ACC_SHAPE => "TT_ERROR_REDUCE_ACC_SHAPE",
            ttStatus::TT_TODO_TF32 => "TT_TODO_TF32",
            ttStatus::TT_ERROR_DST_ROW_ODD => "TT_ERROR_DST_ROW_ODD",
        }
    }
}

impl core::fmt::Display for ttStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}
