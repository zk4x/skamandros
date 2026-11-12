# skamandros — libTT

A kernel compiler and driver for Tenstorrent, shipped as `libTT.so`.

Skamandros (Σκάμανδρος) is the river god of the river beside Troy, called Xanthos
by the gods. A *torrent* is a river: this is the god of the torrent.

The name of the crate is not the name of the artifact. What people link against
is `libTT.so` and its C functions (`tt*`); the program format is `TTIR`; the crate
name only appears in `Cargo.toml`, the docs and the repo.

## What it is

One semantic program in, a set of machine programs out.

The caller writes one SSA program over DRAM parameters, L1 buffers and register
values. It names no processor, no `Dst` slot, no `Src` bank, no lane register,
no semaphore, no unpacker, no packer and no Tensix word:

```
version 6
grid 1,1
%in  = param f32 1024,128
%out = param f32 1024,128
%b0  = buffer f32 32,64          # two 32x32 units, double buffered
%n   = const i32 2
%i   = loop %n
%o   = mul %i %n
copy_to %b0 from %in [%o]
%x = load %b0 [%o]
%y = neg %x
store_to %b0 from %y [%o]
copy_to %out from %b0 [%o]
endloop
```

Parameters and buffers are mutable, not SSA. A shape is a comma-separated list of
terms, and a term is a literal or an `i32` value; a product is built by an op
first. A parameter may be arbitrary-dimensional and symbolic, a scalar parameter
is the shape `0`, and a buffer is two constant dimensions, each `1` or a multiple
of 32. There is no implicit dtype conversion: `cast %x to f32` converts a
value with the C type's rounding, and `bitcast %x to i32` renames the same bits
between dtypes of equal width. A bf16 datum is two bytes in L1, and bf16 is the
compute path the `Dst` is sized for: a kernel whose values are all bf16 runs in
the 16-bit bank, with bf16 arithmetic and comparisons; a `bool` lane lives there
too, as the `-1.0`/`0.0` pattern the bf16 store keeps.

The op set is the low-level one and nothing more: zyx's `UOp`
(`neg`, `not`, `bitnot`, `exp2`, `log2`, `reciprocal`, `sqrt`, `rsqrt`, `sin`,
`cos`, `floor`, `trunc`, `abs`), zyx's `BOp` (`add`, `sub`, `mul`, `div`, `pow`,
`mod`, `cmplt`, `cmpgt`, `cmpge`, `max`, `or`, `and`, `bitxor`, `bitor`, `bitand`,
`bitshiftleft`, `bitshiftright`, `noteq`, `eq`), `sel` (`cond ? if_true :
if_false`, the branchless `v_if`), and the matrix multiply. A comparison produces
a `bool`, and `sel` takes one. There is no `relu`/`elu`/`sigmoid` op.

The crate decides which machine programs each instruction belongs to, where each
register value lives, which lane registers its body uses, how a buffer is read
into a register and written back, and which semaphores pace the three compute
cores. Both forms are accepted and versioned: the text form above, and a byte
form (`TTIR`).

## Design

- **One program, machine programs out.** There is no reader/compute/writer split
  in the IR: an instruction is classified by its operands, and the compiler emits
  only the programs that end up non-empty — the inbound mover, the outbound
  mover, and T0/T1/T2.
- **Unpack and pack are internal.** The IR has `copy_to` between DRAM and L1,
  `load` from L1 into a register and `store_to` back. The unpacker descriptor,
  the packer address generator, the tile order, the base compensation for the
  header the hardware assumes, and the ADCs are all the compiler's.
- **A parameter can be named directly.** `load %in [%o]` and `store_to %out from
  %x [%o]` are the CUDA-like form: `mover::add_movers` (zyx's `tt_add_movers`)
  stages the access through a one-unit L1 buffer and inserts the inbound or
  outbound copy. `copy_to` is still accepted, but it is not required and names no
  mover.
- **A copy has a destination offset and a source offset.** `copy_to %b from %in
  [d] [s]` — both normal `i32` scalars or literals, destination first; one
  bracket is shorthand for both. The staged copy above puts the access index on
  the parameter side and `0` on the buffer side, because a buffer is a ring page
  while the parameter names the access.
- **Every buffer has one producer and one consumer**: the reader publishes and
  the unpacker consumes an input buffer; the packer publishes and the writer
  consumes an output buffer; a compute-free DRAM-to-DRAM transfer is one buffer
  the reader publishes and the writer drains. A buffer used in both directions
  is a loud error, as in tt-metal's circular buffers.
- **Register values get resources automatically.** A value's `Dst` slot follows
  from liveness, lane registers are each body's temporaries, and the cross-core
  hand-offs are one semaphore per producer/consumer seam.
- **L1 is declared, not managed.** The program declares every buffer and its
  shape; the compiler places it and may not change its size. The shape is the
  footprint, so a `32,64` buffer is two pages and the program chooses it.
- **The firmware ships inside the library.** `exec_b`, the three role executors
  (`exec_t0`/`exec_t1`/`exec_t2`), the NC mover and the other images are built
  from `firmware/tt-firmware` and embedded, so a consumer needs neither the
  RISC-V target nor `llvm-objcopy`. This is what makes the `.so` self-contained.
- **A role program is RISC-V code, as in tt-metal.** T0/T1/T2 and the movers are
  programs the firmware calls with the launch arguments in `a0`; the Tensix
  words sit inline as the `.ttinsn` form, and a program returns with `ret`. There
  is no staged word list pushed into `INSTRN_BUF`.
- **A program format, not a Rust API.** C ABI only: pointers, lengths, scalars.
  Everything else in the crate is free to move.

## API

```c
typedef struct ttModule *ttModule;

typedef enum {
    TT_OK                = 0,
    TT_ERROR_ARGUMENT    = 1,
    TT_ERROR_PROGRAM     = 2,
    TT_ERROR_MODULE      = 3,
    TT_ERROR_UNSUPPORTED = 4,
    TT_ERROR_DEVICE      = 5,
} ttStatus;

uint32_t ttAbiVersion(void);

ttStatus ttCompile    (const char *text, size_t len, ttModule *out, char **err);
ttStatus ttModuleLoad (const uint8_t *bytes, size_t len, ttModule *out, char **err);
ttStatus ttModuleUnload(ttModule module, char **err);

ttStatus ttInit(char **err);
ttStatus ttRelease(char **err);

ttStatus ttMemAlloc(size_t bytes, uint64_t *addr, char **err);
ttStatus ttMemcpyH2D(uint64_t dst, const void *src, size_t len, char **err);
ttStatus ttMemcpyD2H(void *dst, uint64_t src, size_t len, char **err);

ttStatus ttLaunch(ttModule module, const uint64_t *params, size_t nparams, char **err);

void ttFree(void *ptr);
```

`ttCompile` takes the program's text form and `ttModuleLoad` its byte form
(`TTIR`); each refuses the other form rather than guessing. Both return an
opaque module handle, stored inside the library: the caller never sees the
programs, only the pointer it passes to `ttLaunch` and gives back to
`ttModuleUnload`.

A buffer address is opaque too: what `ttMemAlloc` hands back is what
`ttMemcpy*` and `ttLaunch` take. A launch is synchronous — every program has
finished when it returns. A launch argument is the DRAM address of a tensor
parameter, and the value itself of a scalar one (`%n = param i32 0`), as CUDA's
kernel parameter buffer carries pointers and values. `ttRelease` closes the
device; the simulator stays initialized, so a later `ttInit` opens a fresh
device over it, and module handles are unaffected.

Every code has one meaning:

| code | meaning |
|---|---|
| `TT_OK` | success; `*err` is null |
| `TT_ERROR_ARGUMENT` | a null required pointer, a zero length, or a parameter count that is not the module's; nothing was read or written |
| `TT_ERROR_PROGRAM` | the text or bytes are not a program this compiler accepts; nothing was done |
| `TT_ERROR_MODULE` | the handle is not one this library handed out, or was already unloaded; nothing was done |
| `TT_ERROR_UNSUPPORTED` | the program compiled, but this build cannot run it; nothing was done |
| `TT_ERROR_DEVICE` | the device, a transfer or the launch failed; the only state that can be left unknown |
| `TT_ERROR_PARSE` … | the concrete failure, one variant each: parse, version, not-`TTIR`, trailing input, buffer shape/zero dimension, undefined/out-of-scope/duplicate name, grid, implicit cast, offset, `endloop` without `loop`, `sel` condition, and the `TT_TODO_*` variants for the unimplemented lowering features |

The enum in `src/status.rs` is the only error type: `ir::Error` is an alias for
it, the variants are integers, and the ABI returns the variant. The
`char **err` out-parameter carries the variant's name for a human; it is not the
error channel.

An error message is a NUL-terminated string the caller frees with `ttFree`; a
null pointer there is a no-op.

The device is `libttsim` through `dlopen`: `$TT_METAL_SIMULATOR` (tt-metal's
own name, `TTSIM.md`), or `libttsim_bh.so` in the current directory. Nothing
links it.

## Building and testing

```bash
cargo build          # target/debug/libTT.so
cargo test           # unit tests and the end-to-end suite
```

The end-to-end suite loads `target/<profile>/libTT.so`, so build the library
before running it (`cargo build && cargo test`); `cargo test` alone does not
rebuild the `cdylib` if only its Rust sources changed.

The end-to-end suite also needs Tenstorrent's functional simulator, `ttsim`,
which is not part of this repository: download `libttsim_bh.so` from the
[ttsim releases](https://github.com/tenstorrent/ttsim/releases) and either name
it in `$TT_METAL_SIMULATOR` (tt-metal's own variable, `TTSIM.md`) or drop it in
`vendor/` next to the crate (`vendor/` is git-ignored):

```bash
mkdir -p vendor && curl -L -o vendor/libttsim_bh.so \
  https://github.com/tenstorrent/ttsim/releases/latest/download/libttsim_bh.so
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml vendor/soc_descriptor.yaml
```

`1.13.0` or newer: that is the first release that runs a bf16 `UnpackToDst`
into a 16-bit `Dst`. `1.16.0` or newer is what this tree is measured on: it
refuses accesses to a harvested tile's registers, which is why the driver takes
the live Tensix grid from the ARC's `ENABLED_TENSIX_COL` telemetry instead of
assuming a full part.

The measure is the end-to-end suite (`tests/e2e.rs`): it `dlopen`s the built
`libTT.so` with `libloading` and drives the C ABI exactly as a consumer would.
The test crate is linked to the library (it is the same package), and uses it
only to build the byte-form fixture; every call under test goes through the
loaded `.so`. Unit tests exist under the modules; they are not the measure.

## Layout

| module | what it owns |
|---|---|
| `ir` | the SSA program: types, text form, byte form, verifier |
| `planner` | L1 placement by the congruence each buffer's copies need |
| `alloc` | SSA value → `Dst`/`SrcA`/`SrcB`, lane registers, core hand-offs |
| `lower` | SSA → machine IR, one program per compute core |
| `optimize` | row-body fusion and the machine-IR verifier |
| `encode` | machine IR → Tensix instructions |
| `mover` | the reader/writer RISC-V programs |
| `riscv` | RV32IM encoders and the branch assembler |
| `compile` | the pipeline, and the module it produces |
| `device` | the simulator, the TLB window, DRAM/L1 access, core start |
| `runtime` | the open device, allocations, transfers and the launch |
| `abi` | the `extern "C"` surface |
| `images` | the firmware images the build produced |

## Status

- The unified IR, the verifier, L1 placement, allocation, lowering, fusion, the
  unpack and pack encoders and the mover programs: in the tree.
- `libTT.so` compiles a program, allocates DRAM, moves host bytes in and out,
  and runs a module on the simulator. A copy kernel — DRAM into L1 and back —
  runs end to end through the C ABI, and the bytes come back equal.
- An elementwise kernel runs the whole datapath through the C ABI: the inbound
  mover on RISCV B, the three compute cores (unpack on T0, math on T1, pack on
  T2) released together as RISC-V programs, and the outbound mover. A 32x32 FP32
  tile of inputs comes back negated, every datum.
- A `gate_proj`-shaped GEMM runs over the whole grid: tiled bf16 operands, an
  f32 accumulator, one K loop inside one output-tile loop, the tile count per
  core derived from the grid at launch (`%per = div %ntiles %nsites`). 240 output
  tiles over the simulator's 12 x 10 tiles, every datum checked against a host
  GEMM (`tests/e2e.rs::gate_proj_runs_over_a_full_grid`).
- The three compute programs are paced as a ring by three semaphores: the
  packer hands a `Dst` slot back, the unpacker publishes each loaded unit, and
  the math core publishes each stored one. Thread 0 runs the setup program
  (`SEMINIT` each semaphore with max 1, as tt-metal's boot sequence does) to
  completion before the three roles are released, so no role depends on another
  role's initialisation having already run. The pack hand-off is one section
  deep, as tt_llk's is: the math core waits until the packer has drained the
  previous section before it writes `Dst` again, and the packer only releases
  the section after the pack has retired.
- A tensor parameter is **tiled**: its bytes are the tile layout below, not a
  row-major matrix, exactly as tt-metal's DRAM tensors are in `TILE_LAYOUT`.
  `copy_to` moves one page of `page` bytes contiguously, so a kernel never
  gathers a tile out of row-major data; the host does that when it fills the
  parameter.
- A launch runs the module's whole grid: every tile of it gets the module's
  programs and arguments, and all of them run concurrently. A symbolic grid
  (`grid %nx,%ny`) is resolved at launch from the scalar arguments before any
  core starts. `%x = grid.x` and `%y = grid.y` are this tile's logical
  coordinate, host-written into the tile's `core_info` like tt-metal's
  `absolute_logical_x/y`; the logical `(x, y)` maps to the physical worker tile
  the way tt-metal's worker grid does.
- The register datapath lowered today is FP32 (and `i32` for the bitwise ops):
  `neg`/`abs`, `add`/`sub`/`mul`/`max` on equal shapes, `cmplt`/`cmpgt`/`cmpge`
  producing a `bool`, `sel` consuming one, and
  `bitand`/`bitor`/`bitxor`/`bitnot` on `i32`, plus logical `or`/`and` on
  `bool`. All of them run end to end through the C ABI.
- A loop is a RISC-V branch, in the movers and in every compute role: a
  constant bound is a literal, a symbolic one is loaded from the launch
  arguments (or computed from them and `grid.x`/`grid.y` in RISC-V) and
  branched on. There is no unroll. Offsets may be any index expression: a
  literal, a loop index, a scalar parameter, a grid coordinate, or arithmetic
  over them.
- `cast` converts in the lane with the C type's rounding (`(int)f` truncates
  toward zero, `(float)i` rounds to nearest even, `(bool)x` is `x != 0`), and
  `bitcast` is the same bits under another name. Both run end to end through
  the C ABI; `bitcast` lowers to no instruction at all. A conversion *between
  bf16 and a 32-bit dtype* is a `Dst` boundary, not a lane instruction: the
  packer narrows into a bf16 buffer, and widening out of one is refused
  (`TT_ERROR_BF16_WIDEN`) because the hardware has no such path through `Dst`.
- `matmul` is the Matrix Unit's: `matmul %acc %a %b` multiplies two 32x32 tiles
  and adds the product into `%acc`, a named accumulator declared as
  `%acc = acc f32 32,32` (or `bf16`). It adds rather than replaces, so a K loop
  is one `matmul` under one `loop` — the accumulator is written every iteration
  and packed once after it. `acc` starts unspecified: a program that needs zeros
  writes them itself (`store_to %acc from %zero`, a fill from an `f32` constant).
  Both operands must be `load` values — the unpacker fills `SrcA`/`SrcB` straight
  from their CB pages, in the `Src` format their dtype implies (`TF32` for `f32`,
  `BF16` for bf16) — and they are the same dtype: `bf16`/`bf16`, `bf16`/`f32`
  and `f32`/`f32` run; an `f32` operand with a `bf16` accumulator is refused. A
  `bf16` accumulator needs the whole module to be 16-bit, since the bank is one
  width. A `load` that also feeds a datapath op keeps its `Dst` unpack and is
  unpacked once more into `Src` at the matmul; its page is released after the
  Matrix Unit has read it.
- Broadcast, `eq`/`noteq`, the remaining unary and binary ops, the `bool`
  datapath in a parameter or buffer, and the reduce op are accepted by the
  format and verifier and refused loudly by lowering, not guessed at.
- bf16 is a real L1 format and the native compute path. A bf16 buffer is two
  bytes per datum, and the `Dst` width is one mode for the whole module, as
  tt-metal's `DST_ACCUM_MODE` (`fp32_dest_acc_en`) is: if every value that
  enters `Dst` is bf16 or a `bool` lane, the bank is 16-bit, the unpacker and
  packer are set to `Bf16`, and the SFPU widens on load and truncates on store
  — so bf16 arithmetic (`neg`, `abs`, `add`, `sub`, `mul`, `max`, comparisons,
  `sel`) runs in the lane at bf16 precision with no rounding glue, at half the
  `Dst` rows. A 32-bit value (`f32`, `i32`) makes the module 32-bit instead, and
  a bf16 buffer written from that bank is the packer's narrowing conversion.
- A bf16 value cannot *widen* into a 32-bit `Dst`: the unpacker has no
  `BF16 -> FP32` Dst path and a 16-bit `Dst` cannot pack `f32`/`i32`, so
  `cast %x to f32` on a bf16 value is `TT_ERROR_BF16_WIDEN` (the conversion
  exists only on the `SrcA`/`SrcB` path, which this compiler drives for
  `matmul` operands and not for a `cast`).
- Transcendentals are the caller's to unfold. zyx rewrites `Exp2`/`Log2`/`Sin`/
  `Cos` (and `Pow` as `exp2(y * log2 x)`) into primitive arithmetic in
  `tt_unfold`, before the backend sees them, and skamandros does the same by
  refusing them: the compiler takes the low-level arithmetic, not the
  approximation tables.
