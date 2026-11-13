//! Running a launch's RISC-V program on a baby core: invalidating the
//! instruction cache after the host wrote code to L1, and calling it with the
//! launch's arguments.
//!
//! tt-metal's firmware calls a kernel the same way (`brisc.cc`: the launch
//! message names the text, the core invalidates the cache for the last written
//! word and calls it). There is no runtime translator and no instruction-push
//! path here: a compiler emits the branches and addresses itself.

use crate::{l1_write32, publish};
use tt_isa::cfg::{generated::global::RISCV_IC_INVALIDATE_InvalidateAll, ConfigBank};

const ZERO: u32 = 0;
const A0: u32 = 10;
/// `jalr x0, 0(ra)`.
const RET: u32 = 0x0000_8067;

/// `addi rd, rs1, imm`.
const fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
    (((imm as u32) & 0xfff) << 20) | (rs1 << 15) | (rd << 7) | 0x13
}

/// Invalidate `mask`'s instruction caches for the code written up to
/// `last_written`.
///
/// The write to the invalidate register must be processed after the writes to
/// L1, or the flush can happen first and the prefetcher or branch predictor
/// refill the cache with the old code (`InstructionCache.md`, "Cache
/// invalidation"). The sequence the page gives -- a read back of the last
/// write, consumed, then the mask store -- is followed by the 24 instructions it
/// says may be needed for the pipeline to drain.
///
/// # Safety
///
/// Only this core's own cache (`mask` = 4 for T1) has been exercised; the
/// Tensix backend must be out of reset.
pub unsafe fn invalidate_icache(last_written: u64, mask: u32) {
    publish();
    let register = RISCV_IC_INVALIDATE_InvalidateAll.riscv_address(ConfigBank::Bank0);
    // SAFETY: the documented register, a whole-word store (`sw`: RISC-V cannot
    // write `Config` any narrower); the word holds only this field.
    unsafe {
        core::arch::asm!(
            "lw   {t}, 0({last})",
            "andi {t}, {t}, 0",
            "sw   {mask}, 0({register})",
            last = in(reg) last_written as u32,
            mask = in(reg) mask,
            register = in(reg) register as u32,
            t = out(reg) _,
            options(nostack),
        );
    }
    for _ in 0..24 {
        // SAFETY: a no-op; not `spin_loop`, whose `pause` Blackhole lacks.
        unsafe { core::arch::asm!("nop") };
    }
}

/// Call the code at `buffer` (a function taking nothing, returning `a0`).
///
/// # Safety
///
/// `buffer` holds valid RV32IM code ending in a return, and its cache lines are
/// current (`invalidate_icache`).
pub unsafe fn call(buffer: u64) -> u32 {
    // SAFETY: the caller's contract.
    let f: extern "C" fn() -> u32 = unsafe { core::mem::transmute(buffer as usize) };
    f()
}

/// Call the code at `buffer` with `a0 = arg` (a function taking one pointer,
/// returning `a0`).
///
/// # Safety
///
/// `buffer` holds valid RV32IM code ending in a return, and its cache lines are
/// current (`invalidate_icache`).
pub unsafe fn call_with(buffer: u64, arg: u32) -> u32 {
    // The C ABI puts the first argument in `a0`; `ret` returns `a0`.
    let f: extern "C" fn(*const u32) -> u32 = unsafe { core::mem::transmute(buffer as usize) };
    f(arg as *const u32)
}

/// `a0 = value; ret`: the smallest code with an observable result, for the
/// invalidation self-test.
///
/// # Safety
///
/// `buffer` is L1 the caller owns, for two words.
pub unsafe fn write_return_value(buffer: u64, value: i32) {
    // SAFETY: the caller owns `buffer`.
    unsafe {
        l1_write32(buffer, addi(A0, ZERO, value));
        l1_write32(buffer + 4, RET);
    }
}
