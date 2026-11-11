// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The firmware images the driver ships: built from `firmware/tt-firmware` by
//! `build.rs`, judged instruction by instruction there, and embedded here.

use TT::images;

#[test]
fn every_image_is_present_and_linked_where_it_is_loaded() {
    for (name, bytes) in images::all() {
        assert!(!bytes.is_empty(), "{name} is empty");
    }
    let [t0, t1, t2] = images::ROLES;
    assert_eq!(t0.0, tt_isa::tensix::Core::T0);
    assert_eq!(t1.0, tt_isa::tensix::Core::T1);
    assert_eq!(t2.0, tt_isa::tensix::Core::T2);
    for (i, (_, bytes, at)) in images::ROLES.iter().enumerate() {
        assert_eq!(*at, tt_isa::l1::roles::BASE[i], "role {i} is loaded at its slot");
        assert!(
            bytes.len() as u64 <= tt_isa::l1::roles::SLOT,
            "role {i} is {} bytes for a {} byte slot",
            bytes.len(),
            tt_isa::l1::roles::SLOT
        );
    }
}

#[test]
fn the_executor_is_the_image_the_mailbox_starts() {
    // The executor is linked at T0's reset PC, which `link.x` also uses for the
    // images loaded through a plain reset.
    assert_eq!(images::LOAD_ADDRESS, u64::from(tt_isa::tensix::Core::T0.default_reset_pc()));
    assert!(!images::EXEC_B.is_empty());
}
