//! The RISCV B data mover: GDDR <-> this tile's L1, one descriptor at a time.
//!
//! Protocol and layout are `tt_isa::arch::blackhole::dm`, which the host shares. Resident: loaded
//! once and left running, so a kernel's operands and results move without the
//! host touching either. Keeps the heartbeat, so the host can tell a stuck
//! mover from a dead core. The mover itself is `src/mover.rs`, shared with
//! RISCV NC's (`dm_nc`).

#![no_std]
#![no_main]

/// B's mailbox, list ring and scratch.
const M: tt_isa::arch::blackhole::dm::Mover = tt_isa::arch::blackhole::dm::Mover::B;

include!("../mover.rs");
