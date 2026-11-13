//! The unpack role's executor: T0 / Tensix thread 0.
//!
//! Linked at the role slot (`build.rs`). The host writes a launch message
//! naming the program's address in the program cache, its length and the
//! arguments base, then bumps the generation; [`tt_firmware::corpus::run_exec`]
//! calls it and acknowledges. The program is RISC-V with the Tensix words
//! inline and ends in `ret`.
//!
//! This replaces the staged-program pusher (`role_t0`) for a compiler that
//! emits the RISC-V itself, as zyx's NOTES describes and tt-metal does.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::mailbox::launch::Mailbox;

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_exec(Mailbox::of(0), 1 << 1)
}
