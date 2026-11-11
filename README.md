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
version 3
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
of 32. There is no implicit dtype conversion: `cast %x to bf16` is the only one.

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
- **Register values get resources automatically.** A value's `Dst` slot follows
  from liveness, lane registers are each body's temporaries, and the cross-core
  hand-offs are one semaphore per producer/consumer seam.
- **L1 is declared, not managed.** The program declares every buffer and its
  shape; the compiler places it and may not change its size. The shape is the
  footprint, so a `32,64` buffer is two pages and the program chooses it.
- **The firmware ships inside the library.** `exec_b`, the NC mover and the three
  role images are built from `firmware/tt-firmware` and embedded, so a consumer
  needs neither the RISC-V target nor `llvm-objcopy`. This is what makes the
  `.so` self-contained.
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
finished when it returns. `ttRelease` closes the device; the simulator stays
initialized, so a later `ttInit` opens a fresh device over it, and module
handles are unaffected.

Every code has one meaning:

| code | meaning |
|---|---|
| `TT_OK` | success; `*err` is null |
| `TT_ERROR_ARGUMENT` | a null required pointer, a zero length, or a parameter count that is not the module's; nothing was read or written |
| `TT_ERROR_PROGRAM` | the text or bytes are not a program this compiler accepts; nothing was done |
| `TT_ERROR_MODULE` | the handle is not one this library handed out, or was already unloaded; nothing was done |
| `TT_ERROR_UNSUPPORTED` | the program compiled, but this build cannot run it; nothing was done |
| `TT_ERROR_DEVICE` | the device, a transfer or the launch failed; the only state that can be left unknown |

An error message is a NUL-terminated string the caller frees with `ttFree`; a
null pointer there is a no-op.

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
  T2) released together, and the outbound mover. A 32x32 FP32 tile of inputs
  comes back negated, every datum.
- The three compute programs are paced as a ring by three semaphores: the
  packer hands a `Dst` slot back, the unpacker publishes each loaded unit, and
  the math core publishes each stored one. Thread 0 runs the setup program
  (`SEMINIT` each semaphore with max 1, as tt-metal's boot sequence does) to
  completion before the three roles are released, so no role depends on another
  role's initialisation having already run.
- A launch is one Tensix tile: a module whose grid is larger than 1x1 is
  refused. A symbolic grid parses and verifies but is not lowered.
- The register datapath lowered today is FP32 (and `i32` for the bitwise ops),
  straight-line, with literal offsets: `neg`/`abs`, `add`/`sub`/`mul`/`max` on
  equal shapes, `cmplt`/`cmpgt`/`cmpge` producing a `bool`, `sel` consuming one,
  and `bitand`/`bitor`/`bitxor`/`bitnot` on `i32`. All of them run end to end
  through the C ABI. A loop with a constant bound is unrolled before lowering,
  and constant index arithmetic (including the offset of each iteration) folds
  to a literal.
- A loop with a symbolic bound, symbolic offsets, `cast`, broadcast, `matmul`,
  `eq`/`noteq`, the remaining unary and binary ops, the `bool` datapath in a
  parameter or buffer, and the reduce op are accepted by the format and verifier
  and refused loudly by lowering, not guessed at.
- Transcendentals are the caller's to unfold. zyx rewrites `Exp2`/`Log2`/`Sin`/
  `Cos` (and `Pow` as `exp2(y * log2 x)`) into primitive arithmetic in
  `tt_unfold`, before the backend sees them, and skamandros does the same by
  refusing them: the compiler takes the low-level arithmetic, not the
  approximation tables.
