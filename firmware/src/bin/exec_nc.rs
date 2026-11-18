//! The writer's executor: RISCV NC.
//!
//! NC cannot invalidate its own instruction cache, so it passes no mask:
//! tt-metal's BRISC invalidates NC's cache along with the TRISCs' before every
//! launch (`brisc.cc`), and ours does the same from `exec_b`. The program itself
//! is called the same way: a resident generation loop around the text the launch
//! message names ([`tt_firmware::corpus::run_exec`]).
//!
//! Loaded at NC's image address with a two-instruction stub at its reset PC
//! (`tt_isa::arch::blackhole::dm::nc::stub`), because ttsim refuses NC's reset-PC override.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::arch::blackhole::mailbox::launch::Mailbox;

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_exec(Mailbox::nc(), 0)
}
