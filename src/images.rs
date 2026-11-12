// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The firmware images, built from `firmware/tt-firmware` by `build.rs`.
//!
//! Each image is linked for one core and loaded at that core's reset PC (`link.x`,
//! or its own slot for the three role images). `build.rs` also disassembles every
//! image and refuses any instruction Blackhole does not implement, so an image
//! here is known to be executable before anything runs it.
//!
//! This is what makes `libTT.so` self-contained: the consumer's build needs
//! neither the RISC-V target nor llvm-objcopy, only this crate's build did.

use tt_isa::tensix::Core;

/// One image: the core it runs on, its bytes, and where it is loaded in L1.
pub type Image = (Core, &'static [u8], u64);

/// Where `link.x` places an image: RISCV T0's default reset PC, which every image
/// but the role images and the per-core probes uses.
pub const LOAD_ADDRESS: u64 = Core::T0.default_reset_pc() as u64;

/// The three role images -- unpack on T0, math on T1, pack on T2 -- each linked
/// at its own slot (`tt_isa::l1::roles`), so all three fit in one tile's L1 at
/// once and each starts through its reset-PC override.
pub const ROLES: [Image; 3] = [
    (
        Core::T0,
        include_bytes!(env!("FIRMWARE_ROLE_T0")),
        tt_isa::l1::roles::BASE[0],
    ),
    (
        Core::T1,
        include_bytes!(env!("FIRMWARE_ROLE_T1")),
        tt_isa::l1::roles::BASE[1],
    ),
    (
        Core::T2,
        include_bytes!(env!("FIRMWARE_ROLE_T2")),
        tt_isa::l1::roles::BASE[2],
    ),
];

/// The program executor: reads a staged program's address, length and arguments
/// from the mailbox, invalidates the I-cache and calls it.
pub const EXEC_B: &[u8] = include_bytes!(env!("FIRMWARE_EXEC_B"));

/// The three role executors -- unpack on T0, math on T1, pack on T2 -- each
/// linked at its role slot. They read their own role mailbox and call the
/// program with `a0` at its arguments, the way tt-metal's firmware calls a
/// kernel; nothing is pushed.
pub const EXEC_ROLES: [Image; 3] = [
    (
        Core::T0,
        include_bytes!(env!("FIRMWARE_EXEC_T0")),
        tt_isa::l1::roles::BASE[0],
    ),
    (
        Core::T1,
        include_bytes!(env!("FIRMWARE_EXEC_T1")),
        tt_isa::l1::roles::BASE[1],
    ),
    (
        Core::T2,
        include_bytes!(env!("FIRMWARE_EXEC_T2")),
        tt_isa::l1::roles::BASE[2],
    ),
];

/// RISCV NC's executor: the writer's program, called the way the TRISCs call
/// theirs. NC cannot invalidate its own cache, so `exec_b` invalidates it before
/// each launch; the image runs from `dm_nc`'s slot behind a jump at NC's reset
/// PC.
pub const EXEC_NC: &[u8] = include_bytes!(env!("FIRMWARE_EXEC_NC"));

/// The four subordinate executors, loaded held and released together after
/// RISCV B has invalidated every instruction cache.
pub const EXEC_SUBORDINATES: [Image; 4] = [
    EXEC_ROLES[0],
    EXEC_ROLES[1],
    EXEC_ROLES[2],
    (Core::NC, EXEC_NC, tt_isa::dm::nc::IMAGE_BASE),
];

/// RISCV NC's mover image (`tt_isa::dm::nc`), reached through the stub at NC's
/// reset PC.
pub const DM_NC: &[u8] = include_bytes!(env!("FIRMWARE_DM_NC"));

/// RISCV B's data mover (`tt_isa::dm`), loaded at L1 offset 0.
pub const DM_B: &[u8] = include_bytes!(env!("FIRMWARE_DM_B"));

/// Every image this build produced, by name, with its load address.
///
/// The list exists so a test can name every image the build gate judged, rather
/// than trusting that the one it embeds was among them.
pub fn all() -> Vec<(&'static str, &'static [u8])> {
    vec![
        ("exec_b", EXEC_B),
        ("exec_t0", EXEC_ROLES[0].1),
        ("exec_t1", EXEC_ROLES[1].1),
        ("exec_t2", EXEC_ROLES[2].1),
        ("exec_nc", EXEC_NC),
        ("dm_nc", DM_NC),
        ("dm_b", DM_B),
        ("role_t0", ROLES[0].1),
        ("role_t1", ROLES[1].1),
        ("role_t2", ROLES[2].1),
    ]
}
