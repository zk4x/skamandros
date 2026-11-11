# skamandros — libTT

A kernel compiler and driver for Tenstorrent, with a program format in and a
library out.

- **One semantic program in, machine programs out.** The caller writes SSA
  tiles: L1 buffers, the manual DRAM↔L1 copies, the explicit circular-buffer
  syncs (`reserve`/`push`/`wait`/`pop`), and values produced by unpack, SFPU and
  matrix ops. The crate decides lane registers, `Dst` slots, `SrcA`/`SrcB` banks,
  the semaphores between the unpack/math/pack cores, and emits the reader,
  writer and role programs.
- **L1 stays manual.** Which buffer, which offset, which bytes, and every
  circular-buffer sync is the caller's program; the crate places them in L1 and
  nothing else.
- **A program format, not a Rust API.** Both forms are accepted and versioned:
  the text form (`version 1`, one directive per line) and the byte form
  (`TTIR`). The C ABI is the stable surface; the Rust items are free to move.

## Layout

| module | what it owns |
|---|---|
| `ir` | the SSA program: types, text form, byte form, verifier |
| `planner` | L1 buffer placement |
| `alloc` | SSA value → `Dst` / `SrcA` / `SrcB`, lane registers, cross-core sync |
| `emit` | reader/writer RISC-V programs, T0/T1/T2 Tensix programs |
| `runtime` | device (simulator now, hardware later), staging, dispatch |
| `abi` | `extern "C"` surface, `libTT.so` |

## Status

The IR, both forms and the verifier are in place (`tests/ir.rs`). The planner,
allocator, emitter and runtime follow.
