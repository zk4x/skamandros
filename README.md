# skamandros

[![crates.io](https://img.shields.io/crates/v/skamandros.svg)](https://crates.io/crates/skamandros)

A kernel compiler and driver for Tenstorrent Blackhole, shipped as
`libskamandros.so` and `skamandros.h`.

    "since he was engaged on the left, by the banks of Skamandros,
     where the death-toll was highest and the war-cries rose"

## Why?

Tenstorrent has amazing and powerful hardware, but the software is a bit unwieldy. This library grew
over ~5 months of working on the zyx Tenstorrent backend and my failures to utilize tt-metal.

## How?

A PTX-like assembly gets compiled and dispatched to all the Tenstorrent baby cores. Firmware and compiler are both
inside one `libskamandros.so`, about 1 MB on the disk.

A simple matmul, distributed over a grid of Tensix cores, with symbolic dims:

```
version 9
%nx = param i32 0
%ny = param i32 0
%ntiles = param i32 0
%ktiles = param i32 0
%ncols = param i32 0
%pa = param bf16 16384
%pw = param bf16 245760
%pc = param f32 245760

%ba = buffer bf16 32,32
%bb = buffer bf16 32,32
%bc = buffer f32 32,32
%acc = acc f32 32,32
%zero = const f32 0
%tile = const i32 1
%x = grid.x
%y = grid.y
%nsites = mul %nx %ny
%sitey = mul %y %nx
%site = add %sitey %x
%per = div %ntiles %nsites
%i = loop %per
  %step = mul %i %nsites
  %t = add %site %step
  %m = div %t %ncols
  %n = mod %t %ncols
  %arow = mul %m %ktiles
  %brow = mul %n %ktiles
  store 32,32 %acc = %zero
  %j = loop %ktiles
    %ka = add %arow %j
    %kb = add %brow %j
    %oa = mul %ka %tile
    %ob = mul %kb %tile
    copy 32,32 %ba [0] = %pa [%oa]   # L1 <- DRAM
    copy 32,32 %bb [0] = %pw [%ob]   # L1 <- DRAM
    load 32,32 %a = %ba [0]          # dst tile <- L1
    load 32,32 %b = %bb [0]          # src tile <- L1
    matmul %acc %a %b                # Dst accumulator += a x b
  endloop                            # the K walk
  store 32,32 %bc [0] = %acc
  %coff = mul %t %tile
  copy 32,32 %pc [%coff] = %bc [0]
endloop                              # this core's output tiles

grid %nx,%ny
```

The caller writes SSA over DRAM parameters, L1 buffers and register values, and names no processor, `Dst` slot, `Src` bank,
lane register, semaphore, unpacker, packer or Tensix word. A shape is a comma-separated list of terms, each a literal
or an `i32` value; a scalar parameter is the shape `0`, and a buffer is two constant dimensions, each `1` or a multiple of 32.
Every `load`, `store` and `copy` names its unit — `32,32` a tile, `1,32` a row, `32,1` a column, `1,1` one
datum — and its index counts those units. There is no implicit dtype conversion: `cast` converts with the C type's rounding,
`bitcast` renames the same bits, and the dtypes are `f32`, `bf16`, `i32` and `bool`. The ops are [zyx's](https://github.com/zk4x/zyx) `UOp` and
`BOp` plus `matmul`, `sum` and `max`; there is no `relu`/`elu`/`sigmoid` - these can be written manually using other
primitive ops and *skamandros automatically optimizes memory allocation around them* (that is automatic **fusion**, no llks,
[you are welcome](https://github.com/geohot/tt-tiny)).

### No implicit casts

There are no implicit conversions, anywhere. Every binary op, `sel`, comparison and `copy` needs both sides to have the same dtype,
and the verifier refuses a mixed pair with `TT_ERROR_NO_IMPLICIT_CAST`; there is no promotion, so an `f32` tile and an `i32` literal
in one `add` is an error, not a widening. A conversion is always written out: `cast` (C rounding) or `bitcast` (the same
bits under another name).

A `bool` is not an index. A comparison produces a `bool`, and an `i32` context — a loop bound, an index, a shape term,
an offset — refuses a `bool` operand with `TT_ERROR_DIMENSION_NOT_INDEX` (or `TT_ERROR_OFFSET_NOT_INDEX`): write
`cast %c to i32`, which is `1` for true and `0` for false. The two forms of a `bool` are:

- a scalar `bool`, used in index arithmetic and evaluated in RISC-V, is `1`/`0`;
- a `bool` lane in a tile, where the comparison runs on the SFPU, is `-1`/`0`, and `cast` to `i32` normalizes it to `1`/`0`.

## Let the compiler do the job

The compiler decides which machine programs each instruction belongs to, where each value lives, which lane registers its body uses,
how a buffer is read, and which core runs what: a reader on RISCV B, the unpacker on T0, the math on T1, the packer on T2, a writer on NC.
Two IRs carry it: the virtual `Op` (one id space — an `OpId` *is* the value) and the machine `MirOp` (one ISA instruction, or a structural pair).

So it's similar to PTX or LLVM (though no phi nodes) in that the compiler handles scheduling, allocation and some small simple
optimizations. This is a fully deterministic compiler. Other ML compilers (e.g. zyx) can then search over different configs using skamandros
as their last step of the lowering.

## Install

The build compiles the device firmware into the library, so two toolchain pieces come first:

```bash
rustup target add riscv32im-unknown-none-elf   # the firmware's cores
rustup component add llvm-tools                # objcopy/objdump, for the images
cargo build --release
```

This writes `libskamandros.so` and `skamandros.h` into `target/release`; copy both to your path.
For usage, look at examples and tests.

## Requirements

- Linux, with the card reached through `tt-kmd` (`/dev/tenstorrent/0`).
- A Tenstorrent Blackhole p100a, or Tenstorrent's functional simulator (`ttsim` 1.16.0 or newer) named in
  `$TT_METAL_SIMULATOR`.
- Rust 1.85 or newer to build, with the `riscv32im-unknown-none-elf` target and `llvm-tools` (see Install).

## The C ABI

The build generates `skamandros.h` next to `libskamandros.so`, from the code that declares it:

- `ttAbiVersion` — the ABI version this library implements, `1`. A consumer compares it against the header it was built with.
- `ttCompile` — compiles a program's text form into an opaque `ttModule` handle; `*out` is the handle, or null with the code saying why.
- `ttMemAlloc` — reserves a byte range of DRAM and returns its address, 64-byte aligned. The address is opaque: hand it back to the copies and the launch.
- `ttMemcpyH2D` — copies host bytes into DRAM at an address `ttMemAlloc` returned. A null pointer or a zero length is refused, nothing read.
- `ttMemcpyD2H` — copies DRAM bytes back into host memory, the other direction of the same transfer. Same argument rules as `ttMemcpyH2D`.
- `ttLaunch` — runs a module over its grid, synchronously: every program has finished when it returns. `params` holds one word per kernel parameter in program order, a DRAM address for a tensor and the value itself for a scalar.
- `ttModuleUnload` — drops a module handle. A second unload of the same handle is `TT_ERROR_MODULE`, never a double free; a compiled module is reusable until then.
- `ttInit` — opens the device, idempotently. The target comes from the environment: `$TT_METAL_SIMULATOR` for the simulator, `TT_HARDWARE=1` for the card, and with neither set it refuses rather than guessing.
- `ttRelease` — closes the device and returns the chip to its idle operating point; a later `ttInit` opens a fresh one. Module handles are unaffected.
- `ttFree` — frees an error message this library returned. A null pointer is a no-op, and module handles are not freed here.

Every call but `ttAbiVersion` and `ttFree` returns a `ttStatus`, and on failure writes a message the caller frees with
`ttFree`. The text form is the only program form `ttCompile` reads.

## Building and testing

```bash
cargo build          # target/debug/libskamandros.so
cargo test           # unit tests, then the end-to-end suite (TT_HARDWARE=1 for the card)
```

The end-to-end suite (`tests/e2e.rs`) `dlopen`s the built library and drives the C ABI exactly as a consumer would;
it builds the `cdylib` itself first. It sets no environment of its own: the transport comes from the environment, as `ttInit`
reads it. A simulator run needs Tenstorrent's functional simulator, which is not part of this repository,
named in `$TT_METAL_SIMULATOR` (a copy in `vendor/`, git-ignored, works well), with the Blackhole descriptor beside it:

```bash
mkdir -p vendor && curl -L -o vendor/libttsim_bh.so \
  https://github.com/tenstorrent/ttsim/releases/latest/download/libttsim_bh.so
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml vendor/soc_descriptor.yaml
TT_METAL_SIMULATOR=$PWD/vendor/libttsim_bh.so cargo test
```

A card run exports `TT_HARDWARE=1` and leaves `$TT_METAL_SIMULATOR` unset. With neither set, `ttInit` refuses
instead of guessing which target was meant.

`ttsim` 1.16.0 or newer is what this tree is measured on. Unit tests are not the measure; the end-to-end suite is.

## Status

Working end to end through the C ABI:

- [x] copies,
- [x] elementwise arithmetic,
- [x] comparisons,
- [x] `sel`,
- [x] casts and bitcasts,
- [x] bf16 — constants, a bf16 page widening into a 32-bit `Dst` through `SrcA` and `MOVA2D`, bf16 reduces, and the
  transcendentals `log2`, `sin`, `cos`, `reciprocal`, `sqrt`, `div` as lane bodies on the 32-bit lane, so a bf16 kernel
  stays 16-bit,
- [x] both reduces,
- [x] `matmul`,
- [ ] `rms_norm` over two tokens and the real hidden size — it is red on the card, green on sim
- [x] a symbolic loop bound,
- [x] a symbolic grid,
- [x] a gather,
- [x] a `gate_proj`-shaped GEMM over the whole grid (240 tiles, every datum checked),
- [x] `Dst` reuse by liveness, so a kernel's *values* may outnumber the bank's slots: `softmax` needs eleven live values
  in eight slots,
- [ ] mcast and multi device are not working yet.

## Examples

`examples/` holds three minimal C consumers that `dlopen` the library and drive the C ABI: a DRAM loopback,
an elementwise kernel and a matmul. They are not tests — each one compiles its program, uploads the inputs,
launches and exits.

Build the library, then put both the library and its generated header on the search paths:

```bash
cargo build --release
export LD_LIBRARY_PATH="$PWD/target/release:$LD_LIBRARY_PATH"   # libskamandros.so, for dlopen
export CPATH="$PWD/target/release:$CPATH"                        # skamandros.h, for gcc
cd examples
gcc dram_loopback.c -o dram_loopback && ./dram_loopback
```

The same for `eltwise.c` and `matmul.c`. `-ldl` is only needed on a glibc older than 2.34, and
`target/debug` works exactly the same way for a debug build. The device still comes from the
environment: `TT_HARDWARE=1` for the card, or `$TT_METAL_SIMULATOR` for the simulator.

## License

Apache-2.0. Copyright (C) 2026 zk4x.

Skamandros was enabled by the amazing work by Knight-Ops https://github.com/Knight-Ops/metalium-rs without
whom I wouldn't have dreamed about doing the whole driver myself.
