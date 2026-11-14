# skamandros

A kernel compiler and driver for Tenstorrent Blackhole, shipped as
`libskamandros.so`.

Skamandros (Σκάμανδρος) is the river god of the river beside Troy, called Xanthos
by the gods. A *torrent* is a river: this is the god of the torrent.

The crate's name is not the artifact's: a consumer links `libskamandros.so` and
calls its C functions (`tt*`), and the program format is `TTIR`.

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
`BOp` plus `sel`, `matmul`, `sum` and `max`; there is no `relu`/`elu`/`sigmoid`.

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

`include/skamandros.h` declares it: `ttCompile`, `ttMemAlloc`, `ttMemcpyH2D`,
`ttMemcpyD2H`, `ttLaunch`, `ttModuleUnload`, `ttInit`, `ttRelease` and `ttFree`,
each returning a `ttStatus` and writing a message the caller frees with
`ttFree`. `ttAbiVersion` is 2.

## Status

Working end to end through the C ABI: copies, elementwise arithmetic,
comparisons, `sel`, casts and bitcasts, bf16, both reduces, `matmul`, a symbolic
loop bound, a symbolic grid, a gather, `rms_norm`, and a `gate_proj`-shaped GEMM
over the whole grid (240 tiles, every datum checked).

Open, in the order they block a release:

- A hand-off race: two staged pages in one loop body and a one-page ring let a
  page be refilled while the unpacker still reads it (rope's two looped tests).
- `softmax` needs nine live `Dst`; the bank has eight.
- `Kernel::from_bytes` is a `todo!()`, so `ttModuleLoad` panics across the ABI;
  `param bool` and `buffer bool` do the same in the verifier.
- The hardware link is not written: this runs on `ttsim` only.

## License

Apache-2.0. Copyright (C) 2026 zk4x.
