# skamandros

A kernel compiler and driver for Tenstorrent Blackhole, shipped as
`libskamandros.so`.

Skamandros (Σκάμανδρος) is the river god of the river beside Troy, called Xanthos
by the gods. A *torrent* is a river: this is the god of the torrent.

## What it is

One semantic program in, a set of machine programs out.

```
version 9
grid 1,1
%in  = param f32 1024,128
%out = param f32 1024,128
%b0  = buffer f32 32,64          # two 32x32 units, double buffered
%n   = const i32 2
%i   = loop %n
%o   = mul %i %n
copy 32,32 %b0 [%o] = %in [%o]
load 32,32 %x = %b0 [%o]
%y = neg %x
store 32,32 %b0 [%o] = %y
copy 32,32 %out [%o] = %b0 [%o]
endloop
```

The caller writes SSA over DRAM parameters, L1 buffers and register values, and
names no processor, `Dst` slot, `Src` bank, lane register, semaphore, unpacker,
packer or Tensix word. A shape is a comma-separated list of terms, each a literal
or an `i32` value; a scalar parameter is the shape `0`, and a buffer is two
constant dimensions, each `1` or a multiple of 32. Every `load`, `store` and
`copy` names its unit — `32,32` a tile, `1,32` a row, `32,1` a column, `1,1` one
datum — and its index counts those units. There is no implicit dtype conversion:
`cast` converts with the C type's rounding, `bitcast` renames the same bits, and
the dtypes are `f32`, `bf16`, `i32` and `bool`. The ops are zyx's `UOp` and
`BOp` plus `sel`, `matmul`, `sum`, `max` and `min`; there is no
`relu`/`elu`/`sigmoid`.

### No implicit casts

There are no implicit conversions, anywhere. Every binary op, `sel`, comparison
and `copy` needs both sides to have the same dtype, and the verifier refuses a
mixed pair with `TT_ERROR_NO_IMPLICIT_CAST`; there is no promotion, so an `f32`
tile and an `i32` literal in one `add` is an error, not a widening. A
conversion is always written out: `cast` (C rounding) or `bitcast` (the same
bits under another name).

A `bool` is not an index. A comparison produces a `bool`, and an `i32` context —
a loop bound, an index, a shape term, an offset — refuses a `bool` operand with
`TT_ERROR_DIMENSION_NOT_INDEX` (or `TT_ERROR_OFFSET_NOT_INDEX`): write
`cast %c to i32`, which is `1` for true and `0` for false. The two forms of a
`bool` are:

- a scalar `bool`, used in index arithmetic and evaluated in RISC-V, is `1`/`0`;
- a `bool` lane in a tile, where the comparison runs on the SFPU, is `-1`/`0`,
  and `cast` to `i32` normalizes it to `1`/`0`.

A conditional value is `sel %c %a %b` with a `bool` `%c`, never arithmetic on
the `bool` itself; on the scalar side it is a RISC-V conditional move (`bne`
over two moves — the ISA has no `cmov`).

The compiler decides which machine programs each instruction belongs to, where
each value lives, which lane registers its body uses, how a buffer is read, and
which core runs what: a reader on RISCV B, the unpacker on T0, the math on T1,
the packer on T2, a writer on NC. Two IRs carry it: the virtual `Op` (one id
space — an `OpId` *is* the value) and the machine `MirOp` (one ISA instruction,
or a structural pair).

## Building and testing

```bash
cargo build          # target/debug/libskamandros.so
cargo test           # unit tests, then the end-to-end suite
```

The end-to-end suite (`tests/e2e.rs`) `dlopen`s the built library and drives the
C ABI exactly as a consumer would; it builds the `cdylib` itself first. It needs
Tenstorrent's functional simulator, which is not part of this repository: put
`libttsim_bh.so` in `vendor/` (git-ignored) or name it in `$TT_METAL_SIMULATOR`,
with the Blackhole descriptor beside it:

```bash
mkdir -p vendor && curl -L -o vendor/libttsim_bh.so \
  https://github.com/tenstorrent/ttsim/releases/latest/download/libttsim_bh.so
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml vendor/soc_descriptor.yaml
```

`ttsim` 1.16.0 or newer is what this tree is measured on. Unit tests are not the
measure; the end-to-end suite is.

## The C ABI

The build generates `skamandros.h` next to `libskamandros.so`, from the code
that declares it: `ttCompile`, `ttMemAlloc`, `ttMemcpyH2D`, `ttMemcpyD2H`,
`ttLaunch`, `ttModuleUnload`, `ttInit`, `ttRelease` and `ttFree`, each returning
a `ttStatus` and writing a message the caller frees with `ttFree`. `ttAbiVersion`
is 1. The text form is the only program form `ttCompile` reads.

## Status

Working end to end through the C ABI: copies, elementwise arithmetic,
comparisons, `sel`, casts and bitcasts, bf16 (constants, a bf16 page widening
into a 32-bit `Dst` through `SrcA` and `MOVA2D`, bf16 reduces, and the
transcendentals — `log2`, `sin`, `cos`, `reciprocal`, `sqrt`, `div` — as lane
bodies on the 32-bit lane, so a bf16 kernel stays 16-bit), both
reduces, `matmul`, `rms_norm` over two tokens and the real hidden size, a
symbolic loop bound, a symbolic grid, a gather, and a
`gate_proj`-shaped GEMM over the whole grid (240 tiles, every datum checked).
The allocator reuses `Dst` by liveness, so a kernel's *values* may outnumber
the bank's slots: `softmax` needs eleven live values in eight slots.

Open, in the order they block a release:

- **The 16-bit `Dst` accumulates at bf16; the 32-bit paths are exact.** The
  acc's declared dtype is the fold's dtype (the verifier now takes a wider acc),
  so a `bf16` accumulator over 96 tiles of `x²` drifts +13.2 %
  (`tests/e2e.rs::bf16_sum_row_over_many_tiles`, ignored) — the per-tile steps
  are quantized to the `Dst`'s bf16 ulp, so it is the accumulator, not the fold.
  The all-f32 fold is exact at every tile count, loaded or computed.
  `acc f32` over bf16 tiles compiles, but a wide module stages the computed
  operand through the packer's `0x105` conversion, which ttsim does not model,
  so `bf16_rms_norm_runs_on_the_simulator` needs silicon with `acc f32`.
- `ttsim` does not model the packer's `f32 -> bf16` conversion
  (`pack_fmt_conv_mode=0x105`), so that path — and every bf16 page written out
  of a 32-bit `Dst` — compiles but cannot be run here. It needs silicon.
- `trunc`/`floor` still round through `i32` in the IR, so a bf16 kernel that
  uses them runs 32-bit and its bf16 output hits that same packer path.
- The hardware link is written but has never run on silicon: `Device::open_hardware`
  opens the card through tt-kmd (`src/kmd.rs`), maps BAR0, takes the session
  lock and a driver-allocated 2 MiB window, and the same `Device` surface runs on
  both. Nothing chooses between the two constructors yet — `ttInit` still opens
  the simulator — so this is unticked until a kernel runs through both.

## License

Apache-2.0. Copyright (C) 2026 zk4x.
