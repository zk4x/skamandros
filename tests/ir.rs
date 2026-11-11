// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The IR as a program format: the text form is the contract, the byte form is
//! the shipping encoding, and the two are checked against each other.

use TT::ir::Kernel;

const ELEMENTWISE: &str = "\
version 1
grid 1 1
param in f32 4096
param out f32 4096
buf 0 4096 plain rm
buf 1 4096 plain rm
copy_in param 0 offset 0 len 4096 buf 0
%0 = unpack buf 0
%1 = sfpu neg %0
pack %1 buf 1
copy_out buf 1 param 1 offset 0 len 4096
";

#[test]
fn text_round_trips() {
    let k = Kernel::parse(ELEMENTWISE).expect("the elementwise program parses");
    assert_eq!(k.to_text(), ELEMENTWISE, "the text form is canonical");
    let again = Kernel::parse(&k.to_text()).expect("canonical text parses");
    assert_eq!(k, again);
}

#[test]
fn byte_round_trips() {
    let k = Kernel::parse(ELEMENTWISE).expect("parses");
    let bytes = k.to_bytes();
    assert_eq!(&bytes[..4], b"TTIR");
    let back = Kernel::from_bytes(&bytes).expect("the byte program reads back");
    assert_eq!(k, back, "byte form is the same program");
}

#[test]
fn circular_buffers_and_sync_round_trip() {
    let text = "\
version 1
grid 2 3
param in f32 8192
param out f32 8192
buf 0 4096 circ 2 rm
copy_in param 0 offset 0 len 4096 buf 0
reserve 0 1
push 0 1
wait 0 1
pop 0 1
";
    let k = Kernel::parse(text).expect("parses");
    assert_eq!(k.to_text(), text);
    assert_eq!(Kernel::from_bytes(&k.to_bytes()).unwrap(), k);
    assert_eq!(k.grid, [2, 3]);
}

#[test]
fn a_use_before_its_definition_is_refused() {
    let text = "version 1\ngrid 1 1\nparam out f32 4096\nbuf 0 4096 plain rm\n%0 = sfpu neg %1\n";
    let err = Kernel::parse(text).expect_err("undefined value");
    assert!(err.to_string().contains("%1"), "{err}");
}

#[test]
fn an_unknown_version_is_refused() {
    let text = "version 2\ngrid 1 1\n";
    let err = Kernel::parse(text).expect_err("unknown version");
    assert!(err.to_string().contains("version 2"), "{err}");
}

#[test]
fn a_wrong_magic_is_refused() {
    let mut bytes = Kernel::parse(ELEMENTWISE).unwrap().to_bytes();
    bytes[0] = b'X';
    let err = Kernel::from_bytes(&bytes).expect_err("bad magic");
    assert!(err.to_string().contains("TTIR"), "{err}");
}

#[test]
fn trailing_bytes_are_refused() {
    let mut bytes = Kernel::parse(ELEMENTWISE).unwrap().to_bytes();
    bytes.push(0);
    let err = Kernel::from_bytes(&bytes).expect_err("trailing byte");
    assert!(err.to_string().contains("trailing"), "{err}");
}

#[test]
fn a_copy_past_its_parameter_is_refused() {
    let text = "\
version 1
grid 1 1
param in f32 4096
buf 0 4096 plain rm
copy_in param 0 offset 4096 len 4096 buf 0
";
    let err = Kernel::parse(text).expect_err("copy past param");
    assert!(err.to_string().contains("past param 0"), "{err}");
}

#[test]
fn sync_on_a_plain_buffer_is_refused() {
    let text = "version 1\ngrid 1 1\nparam in f32 4096\nbuf 0 4096 plain rm\nreserve 0 1\n";
    let err = Kernel::parse(text).expect_err("reserve on plain");
    assert!(err.to_string().contains("not a circular"), "{err}");
}

#[test]
fn more_pages_than_the_buffer_holds_is_refused() {
    let text = "version 1\ngrid 1 1\nparam in f32 8192\nbuf 0 8192 circ 2 rm\npush 0 3\n";
    let err = Kernel::parse(text).expect_err("overfill");
    assert!(err.to_string().contains("3 pages"), "{err}");
}
