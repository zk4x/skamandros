//! The Blackhole addresses the firmware names, as `tt-isa` names them now.
//!
//! The architecture split moved every Blackhole-only constant out of the crate
//! root: a debug register is an associated const of `Arch`, and the mechanisms
//! that are Blackhole's alone are fields of `BlackholeSpecials`. The names here
//! are the ones the images read, so a call site cites a fact rather than a
//! variant match.

use tt_isa::arch::Arch;
use tt_isa::arch::blackhole::Blackhole;
use tt_isa::arch::{BlackholeSpecials, Specials};

/// Blackhole's own mechanisms, unwrapped at compile time.
///
/// `Arch::SPECIALS` is the one place the crate names them, so this is where an
/// image gets them: the `Quasar` arm is unreachable and a compile-time panic, not
/// a run-time one.
pub const SPECIALS: BlackholeSpecials = match Blackhole::SPECIALS {
    Specials::Blackhole(specials) => specials,
    Specials::Quasar => panic!("skamandros builds Blackhole images"),
};

/// `RISCV_DEBUG_REG_WALL_CLOCK_L`: the counter's low half (`Arch::WALL_CLOCK_L`).
pub const WALL_CLOCK_L: u64 = Blackhole::WALL_CLOCK_L;

/// The semaphore window T0/T1/T2 reach (`BlackholeSpecials::semaphore_access`).
pub const SEMAPHORE_ACCESS: u64 = SPECIALS.semaphore_access;

/// Loading here blocks until this thread's instructions have retired.
pub const COPROCESSOR_DONE_CHECK: u64 = SPECIALS.coprocessor_done_check;

/// Loading here blocks until the MOP expander has finished.
pub const MOP_EXPANDER_DONE_CHECK: u64 = SPECIALS.mop_expander_done_check;

/// The MOP expander's configuration window (`Arch::MOP_CFG_BASE`).
pub const MOP_CFG_BASE: u64 = Blackhole::MOP_CFG_BASE;

/// `RISCV_DEBUG_REG_TIMESTAMP`: a store appends an event.
pub const TIMESTAMP: u64 = SPECIALS.timestamp;

/// The header of a 128-bit timestamper event carrying `token`.
///
/// The low three bits of the header choose the event's size; zero is the 128-bit
/// form, which needs no write-accumulation buffer, so events from several cores
/// cannot interleave inside one another (`DebugTimestamper.md`). The crate keeps
/// the record's size ([`tt_isa::tensix::TIMESTAMP_EVENT_BYTES`]) but not this
/// header, because the encoding is the same on both architectures.
pub const fn event_128(token: u32) -> u32 {
    token << 3
}
