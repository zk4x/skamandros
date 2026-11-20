// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! The IR as a program format: the text form is the contract, and it is the
//! only form.

use crate::compile;
use crate::compile::ParamKind;
use crate::ir::{DType, Kernel, Op, RegShape};
use crate::status::ttStatus;

/// The unified program: no reader/compute/writer segments, no unpack/pack, no
/// buffer layouts, no circular-buffer operations. Parameters and buffers are
/// mutable names; values are SSA; a shape term is a literal or a value; the
/// grid is as symbolic as a shape; index math has the full basic op set; and
/// `store_to`/`copy_to` put the destination first.
#[test]
fn the_unified_program_round_trips() {
    const PROGRAM: &str = "\
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
%i4 = mod %i3 %two
%i5 = bitshiftright %i4 %two
%i6 = bitshiftleft %i5 %two
%ic = cmplt %i6 %two
%is = sel %ic %i6 %two
copy 32,32 %b0 [0] = %p0 [%i6]
load 32,32 %x = %b0 [0]
%xf = cast %x to f32
%y = neg %xf
%z = mul %y %half
%w = abs %z
%c = cast %w to bf16
store 32,32 %b0 [0] = %c
copy 32,32 %p1 [%i6] = %b0 [0]
endloop
load 32,32 %a = %b0 [1]
load 32,32 %b = %b0 [0]
%m = acc f32 32,32
matmul %m %a %b
";

    let k = Kernel::parse(PROGRAM).unwrap_or_else(|e| panic!("the unified program parses: {e}"));
    // The literal dims, the shapes and the interned indices are ops of their
    // own now, so the count is the program's plus what the text form implies.
    assert_eq!(k.len(), 46, "every op of the unified program");

    // Symbolic grid and symbolic parameter dims, with the grid after the values
    // it reads: a scalar parameter is a rank-0 shape.
    const SYMBOLIC: &str = "\
%d = param i32 0
%x = param bf16 1024,%d
grid %d,8
";

    let k = Kernel::parse(SYMBOLIC).unwrap_or_else(|e| panic!("the symbolic program parses: {e}"));
    // The literal dim, the shape and the grid's two terms are ops of their own.
    assert_eq!(
        k.len(),
        6,
        "the scalar param, the tensor param and the grid"
    );
}

#[test]
fn a_buffer_shape_that_is_not_one_or_a_multiple_of_32_is_refused() {
    for shape in ["33,32", "32,33", "16,32"] {
        let text = format!("%x = buffer f32 {shape}\ngrid 1,1\n");
        let err = Kernel::parse(&text).expect_err(shape);
        assert_eq!(err, ttStatus::TT_ERROR_BUFFER_SHAPE, "{shape}");
    }
    // A buffer is always two-dimensional and always constant.
    for shape in ["32", "32,%d"] {
        let text = format!("%d = const i32 4\n%x = buffer f32 {shape}\ngrid 1,1\n");
        assert!(
            Kernel::parse(&text).is_err(),
            "{shape} is not a buffer shape"
        );
    }
}

#[test]
fn a_zero_dimension_is_not_the_way_to_write_a_tensor() {
    let text = "%p = param f32 0,32\ngrid 1,1\n";
    let err = Kernel::parse(text).expect_err("a zero in a dimension list");
    assert_eq!(err, ttStatus::TT_ERROR_ZERO_DIMENSION);
}

#[test]
fn a_bf16_constant_is_the_high_half_of_its_literal() {
    // zyx's `bf16::from_f32`: `(x.to_bits() >> 16) as u16`. The datum is the
    // sixteen bits, and the text form is the f32 image they widen to, so the
    // printer and the parser round trip.
    for (literal, bits) in [
        ("0.5", 0x3f00u32),
        ("-1.5", 0xbfc0),
        // Not bf16-exact: the low sixteen bits of the f32 are dropped.
        ("0.3", 0x3e99),
        ("1.1", 0x3f8c),
        ("0.1", 0x3dcc),
    ] {
        let text = format!("%c = const bf16 {literal}\ngrid 1,1\n");
        let k = Kernel::parse(&text).expect("a bf16 constant parses");
        let (_, op) = k.ops().next().expect("the constant");
        assert_eq!(
            *op,
            Op::Const {
                dtype: DType::Bf16,
                bits
            },
            "{literal}"
        );
        // The printed form is the datum's f32 image; parsing it truncates back
        // to the same sixteen bits.
        let image = f32::from_bits(bits << 16);
        let decimal = if image.fract() == 0.0 {
            format!("{image:.1}")
        } else {
            format!("{image}")
        };
        let printed = k.to_string();
        assert_eq!(
            printed.lines().next(),
            Some(format!("%c = const bf16 {decimal}").as_str()),
            "the printed form of {literal}: {printed}"
        );
        let again = Kernel::parse(&format!("%c = const bf16 {decimal}\ngrid 1,1\n"))
            .expect("the printed constant parses back");
        assert_eq!(
            *again.ops().next().expect("the constant").1,
            Op::Const {
                dtype: DType::Bf16,
                bits
            }
        );
    }
}

#[test]
fn a_bf16_constant_is_not_its_f32_literal() {
    // `0.1` is neither a bf16 nor an f32 datum: the constant keeps the high
    // half, which is a different number.
    let text = "%c = const bf16 0.1\ngrid 1,1\n";
    let k = Kernel::parse(text).expect("it parses");
    let (_, op) = k.ops().next().expect("the constant");
    let Op::Const { bits, .. } = *op else {
        unreachable!("the constant")
    };
    assert_eq!(bits, (0.1f32.to_bits() >> 16) & 0xffff);
    assert_ne!(f32::from_bits(bits << 16), 0.1);
}

#[test]
fn an_implicit_dtype_conversion_is_refused() {
    let text = "\
%a = const f32 1.0
%b = const i32 1
%c = add %a %b
grid 1,1
";
    let err = Kernel::parse(text).expect_err("mixed dtypes");
    assert_eq!(err, ttStatus::TT_ERROR_NO_IMPLICIT_CAST);
}

#[test]
fn a_use_before_its_definition_is_refused() {
    let text = "%x = add %a %b\n%a = const i32 1\n%b = const i32 2\ngrid 1,1\n";
    let err = Kernel::parse(text).expect_err("undefined value");
    assert_eq!(err, ttStatus::TT_ERROR_VALUE_UNDEFINED);
}

#[test]
fn a_value_does_not_escape_its_loop() {
    let text = "\
%n = const i32 4
%i = loop %n
%x = add %i %n
endloop
%y = add %x %n
grid 1,1
";
    let err = Kernel::parse(text).expect_err("out of scope");
    assert_eq!(err, ttStatus::TT_ERROR_VALUE_OUT_OF_SCOPE);
}

#[test]
fn a_program_needs_exactly_one_grid() {
    let err = Kernel::parse("%x = const i32 1\n").expect_err("no grid");
    assert_eq!(err, ttStatus::TT_ERROR_GRID);
    let err = Kernel::parse("grid 1,1\ngrid 2,2\n").expect_err("two grids");
    assert_eq!(err, ttStatus::TT_ERROR_GRID);
}

/// A copy kernel over a two-page buffer: the compiler derives the mover
/// programs from the operands, and no compute program exists.
#[test]
fn a_copy_kernel_compiles_to_mover_programs_only() {
    const COPY: &str = "\
%in = param f32 8192
%out = param f32 8192
%b0 = buffer f32 32,64
copy 32,32 %b0 [0] = %in [0]
copy 32,32 %b0 [1] = %in [1]
copy 32,32 %out [0] = %b0 [0]
copy 32,32 %out [1] = %b0 [1]
grid 1,1
";
    let k = Kernel::parse(COPY).expect("the copy kernel parses");
    let m = compile::compile(&k).expect("the copy kernel compiles");
    assert_eq!(
        m.params,
        vec![
            ParamKind::Tensor { page: 4096 },
            ParamKind::Tensor { page: 4096 }
        ]
    );
    assert!(!m.reader.is_empty(), "two inbound copies");
    assert!(!m.writer.is_empty(), "two outbound copies");
    assert!(m.cores.iter().all(Vec::is_empty), "no compute");
}

/// A tensor parameter names pages and nothing else: the launch hands the
/// module the DRAM bank table, so the compiled parameter is one kind whatever
/// the buffer's allocation is, and its page is one tile of its dtype.
#[test]
fn a_tensor_parameter_is_one_kind_whatever_its_allocation() {
    const PROGRAM: &str = "\
%in = param f32 1024
%w = param f32 4096
%out = param f32 1024
%b0 = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
copy 32,32 %out [0] = %b0 [0]
grid 1,1
";
    let k = Kernel::parse(PROGRAM).expect("a parameter parses");
    let text = k.to_string();
    assert!(
        text.contains("%w = param f32 4096\n"),
        "the declaration is the dtype and the count: {text}"
    );
    let m = compile::compile(&k).expect("it compiles");
    assert_eq!(
        m.params,
        vec![
            ParamKind::Tensor { page: 4096 },
            ParamKind::Tensor { page: 4096 },
            ParamKind::Tensor { page: 4096 }
        ]
    );
}

/// Load, compute, store: the same unified program produces the three compute
/// programs as well as the two mover ones.
#[test]
fn an_elementwise_kernel_compiles_to_all_five_programs() {
    const NEG: &str = "\
%in = param f32 4096
%out = param f32 4096
%bi = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = neg %x
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
grid 1,1
";
    let k = Kernel::parse(NEG).expect("the elementwise kernel parses");
    let m = compile::compile(&k).expect("the elementwise kernel compiles");
    assert!(!m.reader.is_empty());
    assert!(!m.writer.is_empty());
    assert!(
        m.cores.iter().all(|c| !c.is_empty()),
        "unpack, math and pack"
    );
}

/// A direct parameter access is the `tt_add_movers` boundary: the caller names
/// `load %in` / `store_to %out`, and the compiler stages it through a buffer and
/// emits the two copies.
#[test]
fn a_direct_parameter_access_gets_its_movers() {
    const DIRECT: &str = "\
%in = param f32 4096
%out = param f32 4096
load 32,32 %x = %in [0]
%y = neg %x
store 32,32 %out [0] = %y
grid 1,1
";
    let k = Kernel::parse(DIRECT).expect("the direct kernel parses");
    k.verify().expect("the direct form verifies");
    let staged = crate::mover::add_movers(&k).expect("the access is staged");
    let buffers = staged
        .ops()
        .filter(|(_, op)| matches!(op, Op::Cb { .. }))
        .count();
    let copies = staged
        .ops()
        .filter(|(_, op)| matches!(op, Op::Copy { .. }))
        .count();
    assert_eq!(buffers, 2, "one staged buffer per direct access");
    assert_eq!(copies, 2, "one inbound and one outbound copy");
    assert!(
        staged
            .ops()
            .filter(|(_, op)| match *op {
                Op::Load { offset, .. } => staged.const_i32(*offset) == Some(0),
                _ => false,
            })
            .count()
            == 1,
        "the staged load reads its buffer at zero"
    );
    // The same program still compiles to all five programs.
    let m = compile::compile(&k).expect("the direct kernel compiles");
    assert!(!m.reader.is_empty(), "the staged inbound copy");
    assert!(!m.writer.is_empty(), "the staged outbound copy");
    assert!(
        m.cores.iter().all(|c| !c.is_empty()),
        "unpack, math and pack"
    );
}

/// A vector buffer's unit is 32 datums, but a `1,1` access selects one datum:
/// that is how a mover reads an index out of a row it copied. A `1,32` access
/// still reads the whole vector.
#[test]
fn a_symbolic_offset_into_a_row_buffer_reads_one_datum() {
    const ROW: &str = "\
grid 1,1
%ids = param i32 1024
%ib = buffer i32 1,32
copy 1,32 %ib [0] = %ids [0]
%i = const i32 0
load 1,1 %x = %ib [%i]
load 1,32 %y = %ib [0]
";
    let k = Kernel::parse(ROW).expect("the row program parses");
    k.verify().expect("the row program verifies");
    let value = |name: &str| {
        *k.names
            .iter()
            .find(|(_, n)| n.as_str() == name)
            .map(|(id, _)| id)
            .unwrap_or_else(|| panic!("`{name}` is defined"))
    };
    assert_eq!(
        k.info(value("x")).shape,
        RegShape::Scalar,
        "a `1,1` access is one datum"
    );
    assert_eq!(
        k.info(value("y")).shape,
        RegShape::Row,
        "a `1,32` access is the whole row"
    );
}

/// One offset, one meaning: it counts units of the access shape, so the same
/// index names a page for a `32,32` tile, a row for a `1,32`, a column for a
/// `32,1` and a datum for a `1,1`. Nothing else tells a narrow access which part
/// of the page it means.
#[test]
fn the_offset_counts_units_of_the_access_shape() {
    const PROGRAM: &str = "\
grid 1,1
%p = param f32 2048
%b = buffer f32 32,64
copy 32,32 %b [0] = %p [0]
copy 32,32 %b [1] = %p [1]
load 32,32 %t = %b [1]
load 1,32 %r = %b [5]
load 32,1 %c = %b [3]
load 1,1 %d = %b [7]
";
    let k = Kernel::parse(PROGRAM).expect("the units parse");
    let units: Vec<([u32; 2], Option<i32>)> = k
        .ops()
        .filter_map(|(_, op)| match *op {
            Op::Load { offset, shape, .. } => Some((shape, k.const_i32(offset))),
            _ => None,
        })
        .collect();
    assert_eq!(
        units,
        vec![
            ([32, 32], Some(1)),
            ([1, 32], Some(5)),
            ([32, 1], Some(3)),
            ([1, 1], Some(7)),
        ],
        "each access counts its own unit"
    );
    // The text form prints the one index back unchanged: the shape carries the
    // unit, so there is no second number to print.
    let printed = k.to_string();
    for line in [
        "load 32,32 %t = %b [1]",
        "load 1,32 %r = %b [5]",
        "load 1,1 %d = %b [7]",
    ] {
        assert!(printed.contains(line), "{line} in\n{printed}");
    }
}

/// A narrow access against a parameter counts units of its shape, and the mover
/// stages each unit as its own page: the ring's page is the unit (`1,32`), the
/// copy moves that unit at the unit's own index, and the access against the ring
/// names page zero, because the value is the page.
#[test]
fn a_narrow_parameter_access_is_staged_as_its_own_page() {
    const PROGRAM: &str = "\
grid 1,1
%p = param f32 1024
%o = param f32 1024
%i = const i32 37
load 1,32 %x = %p [%i]
store 1,32 %o [%i] = %x
";
    let k = Kernel::parse(PROGRAM).expect("a narrow parameter access parses");
    let staged = crate::mover::add_movers(&k).expect("the access is staged");
    let pages: Vec<[crate::ir::Dim; 2]> = staged
        .ops()
        .filter_map(|(_, op)| match *op {
            Op::Cb { page, .. } => Some(page),
            _ => None,
        })
        .collect();
    assert!(
        pages.contains(&[1, 32]),
        "a 1,32 access stages a 1,32 page: {pages:?}"
    );
    // The index is the unit's own: no page/row division anywhere.
    for (at, op) in staged.ops() {
        if let Op::Binary { op, b, .. } = *op {
            if matches!(op, crate::ir::BOp::Div | crate::ir::BOp::Mod)
                && staged.const_i32(b) == Some(32)
            {
                panic!("a narrow access splits its index at {at:?}");
            }
        }
    }
    // The access against the ring names page zero.
    let mut named = 0;
    for (_, op) in staged.ops() {
        match *op {
            Op::Load { offset, .. } | Op::Store { offset, .. } => {
                named += usize::from(staged.const_i32(offset) == Some(0));
            }
            _ => {}
        }
    }
    assert!(named >= 2, "the ring's own page is named by each access");
    // The staged program is what verification sees, and it holds.
    staged.verify().expect("the staged program verifies");
}

/// The access shape is the unit: a copy's unit has to be one NoC request, a
/// shape between the four is refused, and the lanes of a vector index fill one
/// `L1` page exactly.
#[test]
fn an_access_shape_is_the_unit_a_copy_moves() {
    use ttStatus as S;
    let refused = |text: &str| Kernel::parse(text).expect_err("the program is refused");

    assert_eq!(
        refused(
            "\
grid 1,1
%p = param f32 1024
%b = buffer f32 1,1
copy 1,1 %b [0] = %p [0]
"
        ),
        S::TT_ERROR_COPY_UNIT,
        "one datum is not a NoC request"
    );
    assert_eq!(
        refused(
            "\
grid 1,1
%p = param f32 1024
%b = buffer f32 32,32
copy 16,16 %b [0] = %p [0]
"
        ),
        S::TT_ERROR_ACCESS_SHAPE,
        "`16,16` is not one of the four units"
    );
    // One base per datum of a vector index, one request per lane: the lanes
    // fill the page exactly, so 32 rows of a tile-sized buffer gather in 32
    // requests.
    const GATHER: &str = "\
grid 1,1
%ids = param i32 32
%w = param f32 32768
%out = param f32 1024
%ib = buffer i32 1,32
%wb = buffer f32 32,32
copy 1,32 %ib [0] = %ids [0]
load 1,32 %rows = %ib [0]
copy 1,32 %wb [0] = %w [%rows]
copy 32,32 %out [0] = %wb [0]
";
    // The gather is one request per lane, and the mover reads the lane bases out
    // of the page it copied: a compile-time L1 address, which `plan_cbs` places
    // before the lowering so the mover can name it.
    let k = Kernel::parse(GATHER).expect("the vector index is 32 lanes");
    let m = compile::compile(&k).expect("the gather is data movement");
    assert!(!m.reader.is_empty(), "the gather is data movement");

    // A tile per lane would need 32 pages, not one.
    assert_eq!(
        refused(
            "\
grid 1,1
%pid = param i32 32
%w = param f32 32768
%b = buffer f32 32,32
%idxb = buffer i32 1,32
copy 1,32 %idxb [0] = %pid [0]
load 1,32 %i = %idxb [0]
copy 32,32 %b [0] = %w [%i]
"
        ),
        S::TT_ERROR_COPY_LANES,
        "32 tiles are not the page's 32 rows"
    );
    // The L1 side is the ring's page, so it names one base, not a vector.
    assert_eq!(
        refused(
            "\
grid 1,1
%pid = param i32 32
%w = param f32 32768
%b = buffer f32 32,32
%idxb = buffer i32 1,32
copy 1,32 %idxb [0] = %pid [0]
load 1,32 %i = %idxb [0]
copy 1,32 %b [%i] = %w [%i]
"
        ),
        S::TT_TODO_MULTI_LANE,
        "a vector names 32 destination pages"
    );
}

/// A narrow `load`/`store` names the page that holds its unit: a `1,32` page is
/// the value, so the run is sixteen datums to a `Dst` row. A tile-holding page
/// read as a narrow value is loud `TT_TODO_ACCESS_SHAPE` when the value lands in
/// `Dst` — the run would name another tile row's datums — but an `Src` reader
/// takes it as it is, because the FPU's broadcast reads row 0 of each face and
/// that is what a tile page holds. A column is refused everywhere it would need
/// a lane layout. The one narrow load that runs without a datapath is the
/// mover's lane bases (`load 1,32` whose only reader is a copy's index), which
/// `an_access_shape_is_the_unit_a_copy_moves` gates.
#[test]
fn a_narrow_access_names_a_page_of_its_own_unit() {
    use ttStatus as S;
    let case = |what: &str, text: &str, want: S| {
        let err = match Kernel::parse(text) {
            Ok(k) => compile::compile(&k).expect_err(what),
            Err(e) => e,
        };
        assert_eq!(err, want, "{what}");
    };
    // A tile-holding page cannot be read into `Dst` as a row or a column: the
    // value is not the page, so the run would name another tile row's datums.
    for shape in ["1,32", "32,1"] {
        let text = format!(
            "\
grid 1,1
%p = param f32 1024
%b = buffer f32 32,32
copy 32,32 %b [0] = %p [0]
load {shape} %x = %b [0]
%y = neg %x
"
        );
        case(&format!("a {shape} load"), &text, S::TT_TODO_ACCESS_SHAPE);
    }
    // The FPU's broadcast is the exception: it reads the row out of the tile's
    // face, so a row taken from a tile page feeds it without an unpack.
    Kernel::parse(
        "\
grid 1,1
%p = param f32 1024
%o = param f32 1024
%b = buffer f32 32,32
%c = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %b [0] = %p [0]
copy 32,32 %c [0] = %p [0]
load 32,32 %t = %b [0]
load 1,32 %r = %c [0]
%y = mul %t %r
store 32,32 %bo [0] = %y
copy 32,32 %o [0] = %bo [0]
",
    )
    .and_then(|k| compile::compile(&k))
    .expect("a row of a tile page feeds the FPU's broadcast");
    // A row page holds the value: the load runs, the body walks row pair zero,
    // and the pack writes the two rows back.
    Kernel::parse(
        "\
grid 1,1
%p = param f32 1024
%o = param f32 1024
%v = buffer f32 1,32
copy 1,32 %v [0] = %p [0]
load 1,32 %x = %v [0]
%y = neg %x
store 1,32 %v [0] = %y
copy 1,32 %o [0] = %v [0]
",
    )
    .and_then(|k| compile::compile(&k))
    .expect("a 1,32 row through a 1,32 page compiles");
    // A row of a bf16 parameter is refused: its half is 32 bytes, and a NoC read
    // needs its two addresses congruent mod 64, which no fixed L1 placement of
    // an odd row can satisfy (`lower::narrow_pair`).
    case(
        "a bf16 row of a parameter",
        "\
grid 1,1
%p = param bf16 1024
%v = buffer bf16 1,32
copy 1,32 %v [0] = %p [0]
",
        S::TT_TODO_ACCESS_SHAPE,
    );

    // A column page is a page of its own too, but a column has no lane layout
    // yet: a column copy is not even a unit the verifier admits, so such a page
    // cannot be filled from a parameter at all.
    case(
        "a 32,1 copy from a parameter",
        "\
grid 1,1
%p = param f32 1024
%v = buffer f32 32,1
copy 32,1 %v [0] = %p [0]
load 32,1 %x = %v [0]
%y = neg %x
",
        S::TT_ERROR_COPY_UNIT,
    );
    // A store of an accumulator is the one narrow store the verifier admits
    // (the region is packed whole and the mask trims it); the pack of a narrow
    // access is the whole masked page, so it has no lowering either.
    for shape in ["1,32", "32,1", "1,1"] {
        let text = format!(
            "\
grid 1,1
%out = param f32 1024
%bo = buffer f32 32,32
%acc = acc f32 {shape}
%zero = const f32 0
store 32,32 %acc = %zero
store {shape} %bo [0] = %acc
copy 32,32 %out [0] = %bo [0]
"
        );
        case(
            &format!("a {shape} acc store"),
            &text,
            S::TT_TODO_ACCESS_SHAPE,
        );
    }
}

/// A constant-bound loop stays a loop: the movers and the compute roles get a
/// RISC-V branch, not a copy of the body per iteration.
#[test]
fn a_constant_loop_becomes_a_branch() {
    const LOOP_TWO: &str = "\
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,64
%bo = buffer f32 32,64
%two = const i32 2
%step = const i32 1
%i = loop %two
%o = mul %i %step
copy 32,32 %bi [0] = %in [%o]
load 32,32 %x = %bi [0]
%y = neg %x
store 32,32 %bo [0] = %y
copy 32,32 %out [%o] = %bo [0]
endloop
grid 1,1
";
    let k = Kernel::parse(LOOP_TWO).expect("the loop kernel parses");
    let m = compile::compile(&k).expect("the loop kernel compiles");
    assert!(!m.reader.is_empty() && !m.writer.is_empty());
    assert!(m.cores.iter().all(|c| !c.is_empty()));
    // Both movers branch backwards: one body, run twice.
    assert!(has_backward_jump(&m.reader), "the reader loops");
    assert!(has_backward_jump(&m.writer), "the writer loops");
    for (role, words) in ["t0", "t1", "t2"].into_iter().zip(&m.cores) {
        assert!(has_backward_jump(words), "{role} loops");
    }
}

/// A RISC-V `JAL` (opcode `0x6f`) whose J-type immediate is negative: the
/// back edge of a loop body. The edge is a `bgeu` over a long `jal` because a
/// body can be larger than the +-4 KiB a B-type branch reaches.
fn has_backward_jump(words: &[u32]) -> bool {
    words.iter().any(|word| {
        word & 0x7f == 0x6f && {
            // imm[20] is the sign bit, in the word's bit 31.
            let imm20 = (word >> 31) & 1;
            imm20 == 1
        }
    })
}

/// A loop whose bound is a launch-bound scalar becomes the same RISC-V branch:
/// the count is loaded from the arguments in each role, and the movers branch.
#[test]
fn a_dynamic_loop_becomes_a_branch() {
    const DYNAMIC: &str = "\
%n = param i32 0
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,32
%bo = buffer f32 32,32
%i = loop %n
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = neg %x
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
endloop
grid 1,1
";
    let k = Kernel::parse(DYNAMIC).expect("the dynamic loop parses");
    let m = compile::compile(&k).expect("a dynamic loop is lowered");
    assert!(has_backward_jump(&m.reader), "the reader loops");
    assert!(has_backward_jump(&m.writer), "the writer loops");
    for (role, words) in ["t0", "t1", "t2"].into_iter().zip(&m.cores) {
        assert!(has_backward_jump(words), "{role} loops");
    }
}

/// `grid.x`/`grid.y` name this core's logical coordinate, and a grid extent may
/// be a launch-bound scalar.
#[test]
fn the_grid_is_symbolic_and_the_coordinate_is_a_value() {
    const SYMBOLIC: &str = "\
%nx = param i32 0
%ny = param i32 0
%in = param f32 2048
%out = param f32 2048
%x = grid.x
%y = grid.y
%o = mul %x %ny
%o2 = add %o %y
%bi = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bi [0] = %in [%o2]
load 32,32 %v = %bi [0]
%n = neg %v
store 32,32 %bo [0] = %n
copy 32,32 %out [%o2] = %bo [0]
grid %nx,%ny
";
    let k = Kernel::parse(SYMBOLIC).expect("the symbolic grid parses");
    let m = compile::compile(&k).expect("the symbolic grid compiles");
    assert_eq!(
        m.params,
        vec![
            ParamKind::Scalar,
            ParamKind::Scalar,
            ParamKind::Tensor { page: 4096 },
            ParamKind::Tensor { page: 4096 }
        ]
    );
}

/// Comparisons produce a `bool`, `sel` takes one, and `relu` is not an op:
/// the set is zyx's `BOp`/`UOp` plus `sel` and `matmul`.
#[test]
fn comparisons_make_a_bool_and_sel_takes_one() {
    const SELECT: &str = "\
%x = const i32 3
%y = const i32 5
%lt = cmplt %x %y
%m = sel %lt %x %y
grid 1,1
";
    let k = Kernel::parse(SELECT).expect("the select program parses");
    k.verify().expect("it verifies");
    let of = |name: &str| {
        *k.names
            .iter()
            .find(|(_, n)| n.as_str() == name)
            .map(|(id, _)| id)
            .expect("the program names it")
    };
    assert_eq!(k.dtype(of("lt")), DType::Bool, "cmplt is bool");
    assert_eq!(k.dtype(of("m")), DType::I32, "sel is i32");

    // A bool condition is required, and a value with no bool is not one.
    let bad = "%x = const i32 3\n%m = sel %x %x %x\ngrid 1,1\n";
    let err = Kernel::parse(bad).expect_err("an i32 condition is refused");
    assert_eq!(err, ttStatus::TT_ERROR_CONDITION_NOT_BOOL);

    let err = Kernel::parse("%x = const i32 3\n%y = relu %x\ngrid 1,1\n")
        .expect_err("relu is not an op");
    assert_eq!(err, ttStatus::TT_ERROR_PARSE);
}

/// A bitcast is a reinterpretation, not a conversion: the bits are the value, so
/// the two dtypes must be the same width. It lowers to nothing at all.
#[test]
fn a_bitcast_is_the_same_bits_under_another_name() {
    const SAME: &str = "\
%in = param f32 2048
%out = param i32 2048
%bi = buffer f32 32,32
%bo = buffer i32 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = bitcast %x to i32
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
grid 1,1
";
    let k = Kernel::parse(SAME).expect("a bitcast between 32-bit dtypes");
    let m = compile::compile(&k).expect("a bitcast lowers to nothing");
    // The unpacker copies the page into `Dst` and the packer reads it back out;
    // a bitcast issues no instruction of its own, so no math program exists —
    // and the runtime does not start a role it has no words for.
    assert!(
        !m.cores[0].is_empty() && !m.cores[2].is_empty(),
        "unpack and pack"
    );
    assert!(
        m.cores[1].is_empty(),
        "a bitcast is a copy, not a math program"
    );

    // 32 bits do not fit a 16-bit dtype, and a bitcast may not convert.
    const NARROW: &str = "\
%b = buffer f32 32,32
load 32,32 %x = %b [0]
%y = bitcast %x to bf16
grid 1,1
";
    let err = Kernel::parse(NARROW).expect_err("a narrower dtype cannot hold the bits");
    assert_eq!(err, ttStatus::TT_ERROR_BITCAST_WIDTH);
}

/// `a x b` accumulating into an `acc` over two `f32` tiles: the operands must
/// be `load`s, because the Matrix Unit's `SrcA`/`SrcB` are filled by the
/// unpacker from a CB page.
const MATMUL: &str = "\
grid 1,1
%pa = param f32 1024
%pb = param f32 1024
%out = param f32 1024
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %ba [0] = %pa [0]
copy 32,32 %bb [0] = %pb [0]
%m = acc f32 32,32
%zero = const f32 0
store 32,32 %m = %zero
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
matmul %m %a %b
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
";

#[test]
fn a_matmul_compiles_to_the_matrix_unit() {
    let k = Kernel::parse(MATMUL).expect("it parses");
    let m = compile::compile(&k).unwrap_or_else(|e| panic!("it compiles: {e}"));
    assert_eq!(m.params.len(), 3);
}

#[test]
fn a_matmul_is_refused_for_what_the_matrix_unit_cannot_take() {
    use ttStatus as S;
    let case = |what: &str, text: &str, want: S| {
        let err = match Kernel::parse(text) {
            Ok(k) => compile::compile(&k).expect_err(what),
            Err(e) => e,
        };
        assert_eq!(err, want, "{what}");
    };

    // An operand the datapath computed has no page of its own; the compiler
    // stages it: it owns a page and packs the value into it, so the Matrix Unit
    // still reads `Src` from L1. Three caller buffers plus that page.
    const NEGATED: &str = "\
grid 1,1
%pa = param f32 1024
%out = param f32 1024
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %ba [0] = %pa [0]
copy 32,32 %bb [0] = %pa [0]
%m = acc f32 32,32
%zero = const f32 0
store 32,32 %m = %zero
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
%n = neg %a
matmul %m %n %b
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
";
    let k = Kernel::parse(NEGATED).expect("a negated operand parses");
    let m = compile::compile(&k).expect("a negated operand is staged");
    assert_eq!(m.cbs.len(), 4, "three buffers and the staging page");

    // The destination is not an accumulator. Only an `acc` is written in place
    // by the Matrix Unit; a buffer is L1 and a value is immutable.
    case(
        "a buffer where the accumulator goes",
        "\
grid 1,1
%pa = param f32 1024
%out = param f32 1024
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %ba [0] = %pa [0]
copy 32,32 %bb [0] = %pa [0]
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
matmul %bo %a %b
copy 32,32 %out [0] = %bo [0]
",
        S::TT_ERROR_MATMUL_ACC,
    );

    // An accumulator whose shape is a reduce's (a vector, not a tile): the
    // Matrix Unit writes a whole 32x32 tile and has no fold to accumulate into.
    case(
        "a 1,32 accumulator",
        "\
grid 1,1
%pa = param f32 1024
%out = param f32 1024
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %ba [0] = %pa [0]
copy 32,32 %bb [0] = %pa [0]
%m = acc f32 1,32
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
matmul %m %a %b
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
",
        S::TT_ERROR_MATMUL_ACC,
    );

    // A bf16 accumulator in a module whose `Dst` is 32-bit: the Matrix Unit
    // would accumulate in FP32, not at the bf16 the accumulator names.
    case(
        "bf16 accumulation beside an f32 value",
        "\
grid 1,1
%pa = param bf16 1024
%pf = param f32 1024
%out = param bf16 1024
%of = param f32 1024
%ba = buffer bf16 32,32
%bb = buffer bf16 32,32
%bf = buffer f32 32,32
%bg = buffer f32 32,32
%bo = buffer bf16 32,32
copy 32,32 %ba [0] = %pa [0]
copy 32,32 %bb [0] = %pa [0]
copy 32,32 %bf [0] = %pf [0]
%m = acc bf16 32,32
%zero = const bf16 0
store 32,32 %m = %zero
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
load 32,32 %f = %bf [0]
%g = neg %f
store 32,32 %bg [0] = %g
copy 32,32 %of [0] = %bg [0]
matmul %m %a %b
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
",
        S::TT_ERROR_MATMUL_ACCUM_WIDE,
    );

    // `f32` operands are TF32 in `Src` and cannot accumulate into a 16-bit
    // `Dst`.
    case(
        "f32 operands, bf16 accumulator",
        "\
grid 1,1
%pa = param f32 1024
%out = param bf16 1024
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer bf16 32,32
copy 32,32 %ba [0] = %pa [0]
copy 32,32 %bb [0] = %pa [0]
%m = acc bf16 32,32
%zero = const bf16 0
store 32,32 %m = %zero
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
matmul %m %a %b
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
",
        S::TT_ERROR_MATMUL_DTYPE,
    );

    // An `i32` operand: integer matmul is `Int8`/`Int32` `Src` formats and a
    // different set of phases.
    case(
        "i32 operands",
        "\
grid 1,1
%pa = param i32 1024
%out = param i32 1024
%ba = buffer i32 32,32
%bo = buffer i32 32,32
copy 32,32 %ba [0] = %pa [0]
%m = acc f32 32,32
%zero = const f32 0
store 32,32 %m = %zero
load 32,32 %a = %ba [0]
matmul %m %a %a
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
",
        S::TT_ERROR_MATMUL_DTYPE,
    );
}

#[test]
fn an_acc_is_one_32x32_tile_of_an_accumulating_dtype() {
    use ttStatus as S;
    let case = |what: &str, text: &str, want: S| {
        let err = Kernel::parse(text).expect_err(what);
        assert_eq!(err, want, "{what}");
    };
    case(
        "a 16x16 accumulator",
        "%m = acc f32 16,16\ngrid 1,1\n",
        S::TT_ERROR_ACC_SHAPE,
    );
    case(
        "an i32 accumulator",
        "%m = acc i32 32,32\ngrid 1,1\n",
        S::TT_ERROR_ACC_DTYPE,
    );
    // The four shapes an `acc` may have. `32,32` is `matmul`'s; the others are
    // a reduce's, and the shape picks the fold.
    for shape in ["1,32", "32,1", "1,1", "32,32"] {
        let text = format!("%m = acc f32 {shape}\ngrid 1,1\n");
        let k = Kernel::parse(&text).unwrap_or_else(|e| panic!("acc f32 {shape}: {e}"));
        assert_eq!(k.len(), 4, "acc f32 {shape}");
    }
    // `tf32` is a real format with no path yet: named, never guessed.
    case(
        "a tf32 parameter",
        "%p = param tf32 1024\ngrid 1,1\n",
        S::TT_TODO_TF32,
    );
    // A store into an accumulator names no offset, and only a constant fills
    // it.
    case(
        "an offset on an accumulator store",
        "\
grid 1,1
%m = acc f32 32,32
%zero = const f32 0
store 32,32 %m [1024] = %zero
",
        S::TT_ERROR_ACC_OFFSET,
    );
    case(
        "a tile stored into an accumulator",
        "\
grid 1,1
%p = param f32 1024
%b = buffer f32 32,32
copy 32,32 %b [0] = %p [0]
%m = acc f32 32,32
load 32,32 %x = %b [0]
store 32,32 %m = %x
",
        S::TT_TODO_ACC_STORE,
    );
}

/// One tile in and one tile out, with `body` as the reduce: the program a
/// reduce gate is written as. `acc` is the accumulator's shape, which is also
/// the fold; `with_scaler` adds the second operand, which `max` does not name.
fn reduce_program(acc: &str, body: &str, with_scaler: bool) -> String {
    let mut s = String::from("grid 1,1\n%px = param f32 1024\n");
    if with_scaler {
        s.push_str("%ps = param f32 1024\n");
    }
    s.push_str("%out = param f32 1024\n%bx = buffer f32 32,32\n");
    if with_scaler {
        s.push_str("%bs = buffer f32 32,32\n");
    }
    s.push_str("%bo = buffer f32 32,32\ncopy 32,32 %bx [0] = %px [0]\n");
    if with_scaler {
        s.push_str("copy 32,32 %bs [0] = %ps [0]\n");
    }
    s.push_str(&format!(
        "%m = acc f32 {acc}\n%zero = const f32 0\nstore 32,32 %m = %zero\nload 32,32 %x = %bx [0]\n"
    ));
    if with_scaler {
        s.push_str("load 32,32 %s = %bs [0]\n");
    }
    s.push_str(body);
    s.push_str("\nstore 32,32 %bo [0] = %m\ncopy 32,32 %out [0] = %bo [0]\n");
    s
}

/// The three accumulator shapes and both folds parse; the shape is the axis.
/// `sum` names the scaler; `max` does not, because `GMPOOL` applies only the
/// exponent of a non-unity scaler.
#[test]
fn the_reduce_forms_parse() {
    for (acc, body, with_scaler) in [
        ("32,1", "sum %m %x %s", true),
        ("1,32", "sum %m %x %s", true),
        ("1,1", "sum %m %x %s", true),
        ("32,1", "max %m %x", false),
        ("1,32", "max %m %x", false),
        ("1,1", "max %m %x", false),
    ] {
        let text = reduce_program(acc, body, with_scaler);
        let k = Kernel::parse(&text).unwrap_or_else(|e| panic!("{acc} {body} parses: {e}"));
        assert!(!k.ops.is_empty(), "{acc} {body}");
    }
    // The direction word is the shape's job now, so it no longer parses.
    for body in ["sum row %m %x %s", "max col %m %x"] {
        let text = reduce_program("1,32", body, body.starts_with("sum"));
        assert!(
            Kernel::parse(&text).is_err(),
            "`{body}` still spells a direction"
        );
    }
}

/// The accumulator's declared dtype is the fold's: a bf16 tile folded into an
/// f32 `Dst` is a legal program (the fold accumulates at the acc's width).
#[test]
fn a_wider_acc_parses_and_compiles() {
    const PROGRAM: &str = "\
grid 1,1
%px = param bf16 1024
%ps = param bf16 1024
%out = param f32 1024
%bx = buffer bf16 32,32
%bs = buffer bf16 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bs [0] = %ps [0]
%m = acc f32 32,1
%zero = const f32 0
store 32,32 %m = %zero
load 32,32 %x = %bx [0]
load 32,32 %s = %bs [0]
sum %m %x %s
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
";
    let k = Kernel::parse(PROGRAM).expect("the bf16 tile into an f32 acc parses");
    let m = compile::compile(&k).unwrap_or_else(|e| panic!("it compiles: {e}"));
    assert_eq!(m.params.len(), 3);
}

#[test]
fn a_sum_row_compiles_to_the_matrix_unit() {
    let k = Kernel::parse(&reduce_program("32,1", "sum %m %x %s", true)).expect("it parses");
    let m = compile::compile(&k).unwrap_or_else(|e| panic!("it compiles: {e}"));
    assert_eq!(m.params.len(), 3);
}

#[test]
fn a_sum_col_compiles_to_the_gapool() {
    let k = Kernel::parse(&reduce_program("1,32", "sum %m %x %s", true)).expect("it parses");
    let m = compile::compile(&k).unwrap_or_else(|e| panic!("it compiles: {e}"));
    assert_eq!(m.params.len(), 3);
}

#[test]
fn a_max_col_compiles_to_the_gmpool() {
    let k = Kernel::parse(&reduce_program("1,32", "max %m %x", false)).expect("it parses");
    let m = compile::compile(&k).unwrap_or_else(|e| panic!("it compiles: {e}"));
    // The compiler owns one extra page for `max`'s 1.0 `SrcB`.
    assert_eq!(m.params.len(), 2);
}

#[test]
fn a_max_row_compiles_to_the_gmpool_and_the_transposes() {
    let k = Kernel::parse(&reduce_program("32,1", "max %m %x", false)).expect("it parses");
    let m = compile::compile(&k).unwrap_or_else(|e| panic!("it compiles: {e}"));
    assert_eq!(m.params.len(), 2);
}

#[test]
fn a_reduce_is_refused_for_what_the_src_path_cannot_take() {
    use ttStatus as S;
    let case = |what: &str, text: &str, want: S| {
        let err = match Kernel::parse(text) {
            Ok(k) => compile::compile(&k).expect_err(what),
            Err(e) => e,
        };
        assert_eq!(err, want, "{what}");
    };

    // An operand the datapath computed has no page of its own, so the compiler
    // stages it: the fold reads a page the compiler owns.
    let k = Kernel::parse(&reduce_program("32,1", "%n = neg %x\nsum %m %n %s", true))
        .expect("a negated operand parses");
    compile::compile(&k).expect("a negated operand is staged");
    // The destination is not an accumulator.
    case(
        "a buffer where the accumulator goes",
        &reduce_program("32,1", "sum %bo %x %s", true),
        S::TT_ERROR_REDUCE_ACC,
    );
    // `32,32` has no fold: it is the `matmul` accumulator.
    case(
        "a 32,32 accumulator",
        &reduce_program("32,32", "sum %m %x %s", true),
        S::TT_ERROR_REDUCE_ACC_SHAPE,
    );
    // The scaler must be the operand's dtype: `SrcB` takes the same format.
    case(
        "an f32 tile with a bf16 scaler",
        "\
grid 1,1
%px = param f32 1024
%ps = param bf16 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bs = buffer bf16 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bs [0] = %ps [0]
%m = acc f32 1,32
%zero = const f32 0
store 32,32 %m = %zero
load 32,32 %x = %bx [0]
load 32,32 %s = %bs [0]
sum %m %x %s
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
",
        S::TT_ERROR_REDUCE_DTYPE,
    );
    // 32-bit operands cannot accumulate into a 16-bit `Dst`.
    case(
        "f32 operands into a bf16 accumulator",
        "\
grid 1,1
%px = param f32 1024
%ps = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bs = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bs [0] = %ps [0]
%m = acc bf16 1,32
%zero = const bf16 0
store 32,32 %m = %zero
load 32,32 %x = %bx [0]
load 32,32 %s = %bs [0]
sum %m %x %s
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
",
        S::TT_ERROR_REDUCE_DTYPE,
    );
    // The `1,1` fold's machine path is not built: refused loudly, never
    // lowered as something else.
    case(
        "the scalar fold",
        &reduce_program("1,1", "sum %m %x %s", true),
        S::TT_TODO_REDUCE,
    );
}

/// `transpose` is one 32x32 register tile. It compiles for a bf16 value (the
/// 16-bit arm `transpose_tile` uses), and the text form round-trips.
#[test]
fn a_transpose_is_one_tile() {
    const PROGRAM: &str = "\
grid 1,1
%pin = param bf16 1024
%pout = param f32 1024
%bin = buffer bf16 32,32
%bout = buffer f32 32,32
copy 32,32 %bin [0] = %pin [0]
load 32,32 %x = %bin [0]
%y = transpose %x
%z = cast %y to f32
store 32,32 %bout [0] = %z
copy 32,32 %pout [0] = %bout [0]
";
    let k = Kernel::parse(PROGRAM).expect("a bf16 transpose parses");
    let text = k.to_string();
    assert!(text.contains("transpose %x"), "the op round-trips: {text}");
    compile::compile(&k).expect("the bf16 transpose lowers");

    // The 32-bit arm is not lowered: ttsim refuses tt-metal's first half move.
    const F32: &str = "\
grid 1,1
%pin = param f32 1024
%pout = param f32 1024
%bin = buffer f32 32,32
%bout = buffer f32 32,32
copy 32,32 %bin [0] = %pin [0]
load 32,32 %x = %bin [0]
%y = transpose %x
store 32,32 %bout [0] = %y
copy 32,32 %pout [0] = %bout [0]
";
    let k = Kernel::parse(F32).expect("an f32 transpose parses");
    assert_eq!(
        compile::compile(&k).expect_err("the f32 arm is a TODO"),
        ttStatus::TT_TODO_TRANSPOSE
    );

    // A row is not a square: the op names one whole tile.
    const ROW: &str = "\
grid 1,1
%pin = param bf16 32
%pout = param f32 32
%bin = buffer bf16 1,32
%bout = buffer f32 1,32
copy 1,32 %bin [0] = %pin [0]
load 1,32 %x = %bin [0]
%y = transpose %x
%z = cast %y to f32
store 1,32 %bout [0] = %z
copy 1,32 %pout [0] = %bout [0]
";
    let err = Kernel::parse(ROW).expect_err("a row has no transpose");
    assert_eq!(err, ttStatus::TT_ERROR_TRANSPOSE_OPERAND);
}

/// A ring written under one symbolic bound and read under another: nothing
/// proves the two loops trip the same number of times, so some launch leaves
/// one role waiting for a page that is never published. The refusal is the
/// compile-time guarantee `sync_cbs::check_rings_balanced` gives.
#[test]
fn a_ring_moved_by_two_different_bounds_does_not_compile() {
    const PROGRAM: &str = "\
grid 1,1
%n = param i32 0
%m = param i32 0
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,64
%bo = buffer f32 32,64
%i = loop %n
copy 32,32 %bi [0] = %in [0]
endloop
%j = loop %m
load 32,32 %x = %bi [0]
%y = neg %x
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
endloop
";
    let k = Kernel::parse(PROGRAM).expect("the program parses");
    assert_eq!(
        compile::compile(&k).expect_err("the two bounds are not comparable"),
        ttStatus::TT_ERROR_RING_UNBALANCED
    );
}

/// The ring hand-offs are the compiler's. `sync_cbs` places them after the
/// program is split into its roles, so a text program that names one has
/// stepped into the machine representation: a caller error, not an internal
/// bug, and refused at the boundary the caller crossed.
#[test]
fn a_program_may_not_write_its_own_ring_hand_offs() {
    for word in ["wait", "reserve", "push", "pop"] {
        let text = format!(
            "\
grid 1,1
%in = param f32 1024
%out = param f32 1024
%b = buffer f32 32,32
%one = const i32 1
copy 32,32 %b [0] = %in [0]
%i = loop %one
{word} %b 1
endloop
copy 32,32 %out [0] = %b [0]
"
        );
        assert_eq!(
            Kernel::parse(&text).err(),
            Some(ttStatus::TT_ERROR_PROGRAM),
            "`{word}` must be refused as a program error"
        );
    }
}

/// A collective names the cores that take part, not a remote address: its two
/// spans are the whole of its surface. The pick is a cell or the receiver's own
/// per axis, and a roll's shift is the sender's delta.
#[test]
fn a_collective_names_its_cores() {
    use crate::ir::Extent;
    const PROGRAM: &str = "\
%a = buffer bf16 32,32
%five = const i32 5
%seven = const i32 7
%six = const i32 6
%two = const i32 2
%one = const i32 1
%neg = const i32 -2
%gx = const i32 8
%gy = const i32 8
%zero = const i32 0
expand %a [%one] [%five..%seven, ..] from [%six, ..]
roll %a [.., %two..%six] by [%one, %neg]
grid %gx,%gy
";
    let k = Kernel::parse(PROGRAM).expect("the collective parses");
    assert_eq!(k.to_string(), PROGRAM, "the text form round-trips");

    let mut expands = 0;
    let mut rolls = 0;
    for (_, op) in k.ops() {
        match *op {
            Op::Expand { span, from, .. } => {
                expands += 1;
                assert!(matches!(span.x, Extent::Range(..)), "five to seven");
                assert!(matches!(span.y, Extent::Whole), "and every row");
                assert!(matches!(from.x, Extent::At(_)), "from one column");
                assert!(matches!(from.y, Extent::Whole), "at the same row");
            }
            Op::Roll { span, by, .. } => {
                rolls += 1;
                assert!(matches!(span.x, Extent::Whole), "every column");
                assert!(matches!(span.y, Extent::Range(..)), "rows two to six");
                assert_eq!(k.const_i32(by[0]), Some(1));
                assert_eq!(k.const_i32(by[1]), Some(-2), "the sign survives");
            }
            _ => {}
        }
    }
    assert_eq!((expands, rolls), (1, 1), "both ops are in the program");
}

/// A collective that moves nothing, or that names an operand which is not a
/// buffer, is refused when the program is read.
#[test]
fn a_collective_that_moves_nothing_is_refused() {
    use ttStatus as S;
    let refused = |text: &str| Kernel::parse(text).expect_err("the program is refused");
    let program = |body: &str| {
        format!("%a = buffer bf16 32,32\n%p = param bf16 1024\n{body}\ngrid 8,8\n")
    };

    // `..` on every axis is the receiver's own page: the pick is not narrower
    // than the span, so no axis fans out.
    assert_eq!(
        refused(&program("expand %a [1] [5..7, ..] from [.., ..]")),
        S::TT_ERROR_EXPAND_PICK,
        "a pick that is `..` everywhere is not a leader set"
    );
    // A range in the pick has no meaning: a pick axis is one cell or `..`.
    assert_eq!(
        refused(&program("expand %a [1] [5..7, ..] from [5..7, ..]")),
        S::TT_ERROR_EXPAND_PICK,
        "a pick is a cell or the receiver's own, never a run"
    );
    // A pinned axis names one cell, so both sides must name the same one.
    assert_eq!(
        refused(&program("expand %a [1] [5..7, 6] from [6, 8]")),
        S::TT_ERROR_EXPAND_PICK,
        "the pick must lie in the span"
    );
    // A shift of zero is not a movement.
    assert_eq!(
        refused(&program("roll %a [.., ..] by [0, 0]")),
        S::TT_ERROR_ROLL_SHIFT,
        "a roll of nothing is refused"
    );
    // Both collectives move a buffer: a parameter has no cores to fan out from.
    assert_eq!(
        refused(&program("expand %p [1] [.., ..] from [0, 0]")),
        S::TT_ERROR_NOT_A_BUFFER,
        "a parameter is not an L1 buffer"
    );
    assert_eq!(
        refused(&program("roll %p [.., ..] by [1, 0]")),
        S::TT_ERROR_NOT_A_BUFFER,
        "a parameter is not an L1 buffer"
    );
}

/// A copy's span names the cores whose L1 holds the ring page, and it
/// round-trips: the whole-grid form is implicit, every narrower one is printed.
#[test]
fn a_copy_span_round_trips() {
    const PROGRAM: &str = "\
%in = param f32 1024
%b0 = buffer f32 32,32
%x = grid.x
%y = grid.y
copy 32,32 %b0 [0] cores [0, ..] = %in [%x]
copy 32,32 %b0 [0] cores [1..3, %y] = %in [%x]
copy 32,32 %b0 [0] = %in [%x]
grid 4,2
";
    let k = Kernel::parse(PROGRAM).expect("a spanned copy parses");
    let text = k.to_string();
    assert!(
        text.contains("cores [0, ..]") && text.contains("cores [1..3, %"),
        "a narrowed span round-trips: {text}"
    );
    // The whole-grid span is every core, so it is left implicit: exactly two
    // of the three copies print a `cores` clause.
    assert_eq!(text.matches(" cores [").count(), 2, "{text}");
    k.verify().expect("a spanned copy verifies");
}

#[test]
fn an_accumulator_is_a_tile_or_a_block_of_tiles() {
    // A `Dst` region is one tile; a block of tiles is a block of regions, which
    // is what `out_subblock_h x out_subblock_w` and tt-lang's `ttl.block` are.
    for shape in ["32,32", "1,32", "32,1", "1,1", "64,64", "64,32", "32,64", "96,64"] {
        let text = format!("%a = acc f32 {shape}\ngrid 1,1\n");
        Kernel::parse(&text).unwrap_or_else(|e| panic!("{shape} is an accumulator: {e}"));
    }
    // Not tiles: not a multiple of 32, or a shape that is not tile-aligned.
    for shape in ["16,16", "33,32", "32,33"] {
        let text = format!("%a = acc f32 {shape}\ngrid 1,1\n");
        let err = Kernel::parse(&text).expect_err(shape);
        assert_eq!(err, ttStatus::TT_ERROR_ACC_SHAPE, "{shape}");
    }
    // More tiles than the 32-bit `Dst` file holds: a band, spilled by the
    // program into `interm`, not one accumulator.
    for shape in ["128,128", "64,160"] {
        let text = format!("%a = acc f32 {shape}\ngrid 1,1\n");
        let err = Kernel::parse(&text).expect_err(shape);
        assert_eq!(err, ttStatus::TT_ERROR_DST_SLOTS_EXHAUSTED, "{shape}");
    }
}
