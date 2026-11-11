//! The unpack role's executor: T0 / Tensix thread 0.
//!
//! Linked at the role slot (`build.rs`), so the same launch path runs it: the
//! host stages a program and its arguments, writes the role descriptor and
//! releases the core. The program is RISC-V with the Tensix words inline and
//! ends in `ret`; [`tt_firmware::corpus::run_exec`] calls it.
//!
//! This replaces the staged-program pusher (`role_t0`) for a compiler that
//! emits the RISC-V itself, as zyx's NOTES describes and tt-metal does.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::tensix::Thread0;

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_exec::<Thread0>()
}
