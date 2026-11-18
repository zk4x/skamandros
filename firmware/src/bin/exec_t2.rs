//! The pack role's executor: T2 / Tensix thread 2.
//!
//! See `exec_t0` for the shape.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::arch::blackhole::mailbox::launch::Mailbox;

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_exec(Mailbox::of(2), 1 << 3)
}
