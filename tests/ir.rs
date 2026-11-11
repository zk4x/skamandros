// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The IR as a program format: the text form is the contract, the byte form is
//! the shipping encoding, and the two are checked against each other.

use TT::ir::Kernel;

/// The unified program: no reader/compute/writer segments, no unpack/pack, no
/// buffer layouts, no circular-buffer operations. Parameters and buffers are
/// mutable names; values are SSA; a shape term is a literal or a value; the
/// grid is as symbolic as a shape; index math has the full basic op set; and
/// `store_to`/`copy_to` put the destination first.
#[test]
fn the_unified_program_round_trips() {
    const PROGRAM: &str = "\
version 2
grid 4,6
%p0 = param bf16 1024,132,453
%p1 = param bf16 1024,232
%b0 = buffer bf16 32,64
%len = const i32 32
%two = const i32 2
%half = const f32 0.5
%i = loop %len
%i0 = mul %i %two
%i1 = add %i0 %two
%i2 = sub %i1 %two
%i3 = div %i2 %two
%i4 = rem %i3 %two
%i5 = shr %i4 %two
%i6 = shl %i5 %two
copy_to %b0 from %p0 [%i6]
%x = load %b0 [0]
%xf = cast %x to f32
%y = neg %xf
%z = mul %y %half
%w = abs %z
%c = cast %w to bf16
store_to %b0 from %c [0]
copy_to %p1 from %b0 [%i6]
endloop
%a = load %b0 [1024]
%b = load %b0 [0]
%m = matmul %a %b
";

    let k = Kernel::parse(PROGRAM).unwrap_or_else(|e| panic!("the unified program parses: {e}"));
    assert_eq!(k.to_text(), PROGRAM, "the text form is canonical");
    assert_eq!(Kernel::parse(&k.to_text()).unwrap(), k);
    assert_eq!(Kernel::from_bytes(&k.to_bytes()).unwrap(), k);

    // Symbolic grid and symbolic parameter dims, with the grid after the values
    // it reads: a scalar parameter is a rank-0 shape.
    const SYMBOLIC: &str = "\
version 2
%d = param i32 0
%x = param bf16 1024,%d
grid %d,8
";

    let k = Kernel::parse(SYMBOLIC).unwrap_or_else(|e| panic!("the symbolic program parses: {e}"));
    assert_eq!(k.to_text(), SYMBOLIC);
    assert_eq!(Kernel::from_bytes(&k.to_bytes()).unwrap(), k);
}

#[test]
fn a_buffer_shape_that_is_not_one_or_a_multiple_of_32_is_refused() {
    for shape in ["33,32", "32,33", "16,32"] {
        let text = format!("version 2\n%x = buffer f32 {shape}\ngrid 1,1\n");
        let err = Kernel::parse(&text).expect_err(shape);
        assert!(err.to_string().contains("multiple of 32"), "{err}");
    }
    // A buffer is always two-dimensional and always constant.
    for shape in ["32", "32,%d"] {
        let text = format!("version 2\n%d = const i32 4\n%x = buffer f32 {shape}\ngrid 1,1\n");
        assert!(
            Kernel::parse(&text).is_err(),
            "{shape} is not a buffer shape"
        );
    }
}

#[test]
fn a_zero_dimension_is_not_the_way_to_write_a_tensor() {
    let text = "version 2\n%p = param f32 0,32\ngrid 1,1\n";
    let err = Kernel::parse(text).expect_err("a zero in a dimension list");
    assert!(err.to_string().contains("zero"), "{err}");
}

#[test]
fn an_implicit_dtype_conversion_is_refused() {
    let text = "\
version 2
%a = const f32 1.0
%b = const i32 1
%c = add %a %b
grid 1,1
";
    let err = Kernel::parse(text).expect_err("mixed dtypes");
    assert!(err.to_string().contains("no implicit cast"), "{err}");
}

#[test]
fn a_use_before_its_definition_is_refused() {
    let text = "version 2\n%x = add %a %b\n%a = const i32 1\n%b = const i32 2\ngrid 1,1\n";
    let err = Kernel::parse(text).expect_err("undefined value");
    assert!(err.to_string().contains("unknown value"), "{err}");
}

#[test]
fn a_value_does_not_escape_its_loop() {
    let text = "\
version 2
%n = const i32 4
%i = loop %n
%x = add %i %n
endloop
%y = add %x %n
grid 1,1
";
    let err = Kernel::parse(text).expect_err("out of scope");
    assert!(err.to_string().contains("out of scope"), "{err}");
}

#[test]
fn a_program_needs_exactly_one_grid() {
    let err = Kernel::parse("version 2\n%x = const i32 1\n").expect_err("no grid");
    assert!(err.to_string().contains("no grid"), "{err}");
    let err = Kernel::parse("version 2\ngrid 1,1\ngrid 2,2\n").expect_err("two grids");
    assert!(err.to_string().contains("more than one grid"), "{err}");
}

#[test]
fn an_unknown_version_is_refused_in_both_forms() {
    let err = Kernel::parse("version 1\ngrid 1,1\n").expect_err("version 1");
    assert!(err.to_string().contains("version 1"), "{err}");

    let mut bytes = Kernel::parse("version 2\ngrid 1,1\n%x = const i32 1\n")
        .unwrap()
        .to_bytes();
    assert_eq!(&bytes[..4], b"TTIR");
    bytes[4] = 1;
    let err = Kernel::from_bytes(&bytes).expect_err("byte version 1");
    assert!(err.to_string().contains("version"), "{err}");
    bytes[4] = 2;
    bytes[0] = b'X';
    assert!(Kernel::from_bytes(&bytes).is_err(), "a wrong magic");
    let mut bytes = Kernel::parse("version 2\ngrid 1,1\n%x = const i32 1\n")
        .unwrap()
        .to_bytes();
    bytes.push(0);
    let err = Kernel::from_bytes(&bytes).expect_err("trailing byte");
    assert!(err.to_string().contains("trailing"), "{err}");
}

/// A copy kernel over a two-page buffer: the compiler derives the mover
/// programs from the operands, and no compute program exists.
#[test]
fn a_copy_kernel_compiles_to_mover_programs_only() {
    const COPY: &str = "\
version 2
%in = param f32 8192
%out = param f32 8192
%b0 = buffer f32 32,64
copy_to %b0 from %in [0]
copy_to %b0 from %in [1024]
copy_to %out from %b0 [0]
copy_to %out from %b0 [1024]
grid 1,1
";
    let k = Kernel::parse(COPY).expect("the copy kernel parses");
    let m = TT::compile::compile(&k).expect("the copy kernel compiles");
    assert_eq!(m.params, 2);
    assert!(!m.reader.is_empty(), "two inbound copies");
    assert!(!m.writer.is_empty(), "two outbound copies");
    assert!(m.cores.iter().all(Vec::is_empty), "no compute");
}

/// Load, compute, store: the same unified program produces the three compute
/// programs as well as the two mover ones.
#[test]
fn an_elementwise_kernel_compiles_to_all_five_programs() {
    const NEG: &str = "\
version 2
%in = param f32 4096
%out = param f32 4096
%b0 = buffer f32 32,32
copy_to %b0 from %in [0]
%x = load %b0 [0]
%y = neg %x
store_to %b0 from %y [0]
copy_to %out from %b0 [0]
grid 1,1
";
    let k = Kernel::parse(NEG).expect("the elementwise kernel parses");
    let m = TT::compile::compile(&k).expect("the elementwise kernel compiles");
    assert!(!m.reader.is_empty());
    assert!(!m.writer.is_empty());
    assert!(
        m.cores.iter().all(|c| !c.is_empty()),
        "unpack, math and pack"
    );
}

/// A loop with a constant bound is unrolled before anything reads the program,
/// and `%o = mul %i %step` folds to the literal offset of each iteration.
#[test]
fn a_constant_loop_unrolls_into_literal_offsets() {
    const LOOP_TWO: &str = "\
version 2
%in = param f32 2048
%out = param f32 2048
%b0 = buffer f32 32,64
%two = const i32 2
%step = const i32 1024
%i = loop %two
%o = mul %i %step
copy_to %b0 from %in [%o]
%x = load %b0 [%o]
%y = neg %x
store_to %b0 from %y [%o]
copy_to %out from %b0 [%o]
endloop
grid 1,1
";
    let k = Kernel::parse(LOOP_TWO).expect("the loop kernel parses");
    let u = TT::unroll::constant_loops(&k).expect("the loop unrolls");
    assert!(
        u.ops
            .iter()
            .all(|op| !matches!(op, TT::ir::Op::Loop { .. } | TT::ir::Op::EndLoop)),
        "no loop survives: {:?}",
        u.ops
    );
    let loads: Vec<TT::ir::Dim> = u
        .ops
        .iter()
        .filter_map(|op| match *op {
            TT::ir::Op::Load { offset, .. } => Some(offset),
            _ => None,
        })
        .collect();
    assert_eq!(loads, vec![TT::ir::Dim::Const(0), TT::ir::Dim::Const(1024)]);
    u.verify().expect("the unrolled program verifies");

    let m = TT::compile::compile(&k).expect("the loop kernel compiles");
    assert!(!m.reader.is_empty() && !m.writer.is_empty());
    assert!(m.cores.iter().all(|c| !c.is_empty()));
}

/// A loop with a symbolic bound is left alone, and lowering still refuses it
/// loudly rather than unrolling a guess.
#[test]
fn a_dynamic_loop_is_still_refused() {
    const DYNAMIC: &str = "\
version 2
%n = param i32 0
%in = param f32 2048
%out = param f32 2048
%b0 = buffer f32 32,32
%i = loop %n
%x = load %b0 [0]
%y = neg %x
store_to %b0 from %y [0]
endloop
grid 1,1
";
    let k = Kernel::parse(DYNAMIC).expect("the dynamic loop parses");
    let err = TT::compile::compile(&k).expect_err("a dynamic loop is not lowered");
    assert!(err.to_string().contains("loops are not lowered"), "{err}");
}
