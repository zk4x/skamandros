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

What a consumer sees is `libTT.so` and its C ABI, tested end to end by loading
the built library with `dlopen` and calling it (`tests/e2e.rs`). Unit tests exist
under the modules, but they are not the measure: the measure is the `.so`.

```c
uint32_t tt_abi_version(void);
int tt_compile(const uint8_t *ir, size_t ir_len,
               uint8_t **out, size_t *out_len,
               char **err, size_t *err_len);
void tt_free(uint8_t *ptr, size_t len);
```

`ir` is the text form (or the byte form if it starts with `TTIR`); `out` is a
module (magic `TTMD`, the grid, then the reader, writer and T0/T1/T2 programs).
The runtime calls that take a module and run it are not there yet.
