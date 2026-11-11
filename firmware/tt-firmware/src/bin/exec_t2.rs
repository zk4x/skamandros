//! The pack role's executor: T2 / Tensix thread 2.
//!
//! See `exec_t0` for the shape.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::tensix::Thread2;

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_exec::<Thread2>()
}
