//! The math role's executor: T1 / Tensix thread 1.
//!
//! See `exec_t0` for the shape. T1 is the core ttsim lets read `Dst`.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::tensix::Thread1;

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_exec::<Thread1>()
}
