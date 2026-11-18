//! The math role's executor: T1 / Tensix thread 1.
//!
//! See `exec_t0` for the shape. T1 is the core ttsim lets read `Dst`.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::arch::blackhole::mailbox::launch::Mailbox;

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_exec(Mailbox::of(1), 1 << 2)
}
