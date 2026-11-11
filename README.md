# skamandros — libTT

A kernel compiler and driver for Tenstorrent, shipped as `libTT.so`.

Skamandros (Σκάμανδρος) is the river god of the river beside Troy, called Xanthos
by the gods. A *torrent* is a river: this is the god of the torrent.

The name of the crate is not the name of the artifact. What people link against
is `libTT.so` and its C functions (`tt*`); the program format is `TTIR`; the crate
name only appears in `Cargo.toml`, the docs and the repo.

## What it is

One semantic program in, a set of machine programs out.

The caller writes SSA tiles and the L1 traffic it wants, and names no processor,
no `Dst` slot, no `Src` bank, no lane register, no semaphore, and no Tensix word:

```
version 1
grid 1 1
param in f32 4096
param out f32 4096
buf 0 4096 plain tile          # a 32x32 tile image
buf 1 4096 plain rm            # a contiguous datum run
copy_in param 0 offset 16 len 4096 buf 0
%0 = unpack buf 0
%1 = sfpu neg %0
pack %1 buf 1
copy_out buf 1 param 1 offset 0 len 4096
```

The crate decides which of the five programs each instruction belongs to, where
each value lives, which lane registers its body uses, and which semaphores pace
the three compute cores. Both forms are accepted and versioned: the text form
above, and a byte form (`TTIR`).

## Design

- **Semantic in, machine out.** The device is not one program: the movers run
  RISC-V, and the three compute cores run Tensix instructions. One semantic
  program becomes `reader`, `writer`, `T0`, `T1`, `T2`, handled internally.
- **SSA values get resources automatically.** A value consumed by a matrix op
  lives in `SrcA`/`SrcB`, everything else in a `Dst` slot; slots and banks are
  assigned by liveness, lane registers are each body's temporaries, and the
  cross-core hand-offs are one semaphore per producer/consumer seam. A value
  needed on both `Src` sides is refused rather than copied.
- **L1 stays manual.** Which buffer, how many bytes, which offset, and every
  `reserve`/`push`/`wait`/`pop` is the caller's program. The driver places the
  buffers in L1 and nothing else. A buffer declares its layout — `rm` (contiguous
  datums) or `tile` (4096-byte tile images, four faces) — because the unpacker's
  descriptor and the packer's address generator are configured from it.
- **L1 holds datum bytes only.** A `tile` buffer is the 4096-byte tile body, no
  16-byte DRAM header; the unpack/pack base compensates the fixed header step.
- **The firmware ships inside the library.** `exec_b`, the NC mover and the three
  role images are built from `firmware/tt-firmware` and embedded, so a consumer
  needs neither the RISC-V target nor `llvm-objcopy`. This is what makes the
  `.so` self-contained.
- **A program format, not a Rust API.** C ABI only: pointers, lengths, scalars.
  Everything else in the crate is free to move.

## API

```c
uint32_t tt_abi_version(void);

int tt_compile(const uint8_t *ir, size_t ir_len,
               uint8_t **out, size_t *out_len,
               char **err, size_t *err_len);

void tt_free(uint8_t *ptr, size_t len);
```

`ir` is the text form, or the byte form if it starts with `TTIR`. On success
`out` is a module: magic `TTMD`, version, grid, parameter count, then the
reader, writer and T0/T1/T2 programs, each a word count and its words. On
failure the code is non-zero, `out` is null, and `err` carries a message.

The runtime:

```c
int ttInit(char **err, size_t *err_len);
int ttRelease(void);

int ttMemAlloc(size_t bytes, uint64_t *addr, char **err, size_t *err_len);
int ttMemcpyH2D(uint64_t dst, const void *src, size_t len, char **err, size_t *err_len);
int ttMemcpyD2H(void *dst, uint64_t src, size_t len, char **err, size_t *err_len);

int ttLaunch(const uint8_t *module, size_t len,
             const uint64_t *params, size_t nparams,
             char **err, size_t *err_len);
int ttSynchronize(void);
```

A buffer address is opaque: what `ttMemAlloc` hands back is what `ttMemcpy*`
and `ttLaunch` take. A launch is synchronous — every program has finished when
it returns — so `ttSynchronize` has nothing to wait for and only reports that a
device is open. `ttRelease` closes the device; the simulator stays initialized,
so a later `ttInit` opens a fresh device over it.

The device is `libttsim` through `dlopen`: `$TT_TTSIM_LIB`, or
`libttsim_bh.so` in the current directory. Nothing links it.

## Building and testing

```bash
cargo build          # target/debug/libTT.so
cargo test           # unit tests and the end-to-end suite
```

The end-to-end suite loads `target/<profile>/libTT.so`, so build the library
before running it (`cargo build && cargo test`); `cargo test` alone does not
rebuild the `cdylib` if only its Rust sources changed.

The measure is the end-to-end suite (`tests/e2e.rs`): it `dlopen`s the built
`libTT.so` with `libloading` — nothing links the Rust library — and drives the C
ABI exactly as a consumer would. Unit tests exist under the modules; they are not
the measure.

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
| `compile` | the pipeline, and the module bytes |
| `device` | the simulator, the TLB window, DRAM/L1 access, core start |
| `runtime` | the open device, allocations, transfers and the launch |
| `abi` | the `extern "C"` surface |
| `images` | the firmware images the build produced |

## Status

- Program format, verifier, L1 placement, allocation, lowering, fusion, the math
  encoder and the boundary programs: in the tree.
- `libTT.so` compiles a program, allocates DRAM, moves host bytes in and out,
  and runs a module on the simulator. A copy kernel — DRAM into L1 and back —
  runs end to end through the C ABI, and the bytes come back equal.
- A launch runs the reader and then the writer on RISCV B through the embedded
  `exec_b` trampoline, which is the program's own order for a kernel with no
  compute. A module with compute programs is refused rather than run half-way.
- A launch is one Tensix tile: a module whose grid is larger than 1x1 is
  refused.
- The unpack and pack encoders are refused with a message, not guessed. Until
  they exist no module has a compute program, so nothing is lost by the
  refusal; the three compute cores get their launch after them.
