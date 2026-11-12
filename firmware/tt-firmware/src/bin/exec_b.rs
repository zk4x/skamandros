//! Execute a launched RISC-V program on RISCV B.
//!
//! tt-metal's shape (`brisc.cc`, `kernel_main()`): resident firmware, a launch
//! message naming the kernel text, the kernel called as a function, the core
//! reporting completion. Here the launch message is the single-core mailbox:
//! `PROGRAM_ADDR` (an L1 address in the program cache), `PROGRAM_LEN`, `ARGS`,
//! and a `GENERATION` that starts the run; the core publishes `RESULT`, `DONE`
//! and `ACK`. There is no staged-program slot and no separate `SEQ`/`DONE`
//! control block.
//!
//! The program is plain RV32IM ending in `ret`. It may use the caller-saved
//! registers only (`t0`-`t6`, `a0`-`a7`); this image makes no promises beyond
//! that, as tt-metal's firmware does not either.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::mailbox::launch::Mailbox;

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_exec(Mailbox::b(), 0x1)
}
