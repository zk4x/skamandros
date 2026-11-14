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
version 9
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
version 9
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
        let text = format!("version 9\n%x = buffer f32 {shape}\ngrid 1,1\n");
        let err = Kernel::parse(&text).expect_err(shape);
        assert_eq!(err, ttStatus::TT_ERROR_BUFFER_SHAPE, "{shape}");
    }
    // A buffer is always two-dimensional and always constant.
    for shape in ["32", "32,%d"] {
        let text = format!("version 9\n%d = const i32 4\n%x = buffer f32 {shape}\ngrid 1,1\n");
        assert!(
            Kernel::parse(&text).is_err(),
            "{shape} is not a buffer shape"
        );
    }
}

#[test]
fn a_zero_dimension_is_not_the_way_to_write_a_tensor() {
    let text = "version 9\n%p = param f32 0,32\ngrid 1,1\n";
    let err = Kernel::parse(text).expect_err("a zero in a dimension list");
    assert_eq!(err, ttStatus::TT_ERROR_ZERO_DIMENSION);
}

#[test]
fn an_implicit_dtype_conversion_is_refused() {
    let text = "\
version 9
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
    let text = "version 9\n%x = add %a %b\n%a = const i32 1\n%b = const i32 2\ngrid 1,1\n";
    let err = Kernel::parse(text).expect_err("undefined value");
    assert_eq!(err, ttStatus::TT_ERROR_VALUE_UNDEFINED);
}

#[test]
fn a_value_does_not_escape_its_loop() {
    let text = "\
version 9
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
    let err = Kernel::parse("version 9\n%x = const i32 1\n").expect_err("no grid");
    assert_eq!(err, ttStatus::TT_ERROR_GRID);
    let err = Kernel::parse("version 9\ngrid 1,1\ngrid 2,2\n").expect_err("two grids");
    assert_eq!(err, ttStatus::TT_ERROR_GRID);
}

#[test]
fn an_unknown_version_is_refused() {
    let err = Kernel::parse("version 1\ngrid 1,1\n").expect_err("version 1");
    assert_eq!(err, ttStatus::TT_ERROR_VERSION);
}

/// A copy kernel over a two-page buffer: the compiler derives the mover
/// programs from the operands, and no compute program exists.
#[test]
fn a_copy_kernel_compiles_to_mover_programs_only() {
    const COPY: &str = "\
version 9
%in = param f32 8192
%out = param f32 8192
%b0 = buffer f32 32,64
copy 32,32 %b0 [0] = %in [0]
copy 32,32 %b0 [1] = %in [1]
copy 32,32 %out [0] = %b0 [0]
copy 32,32 %out [1] = %b0 [0]
grid 1,1
";
    let k = Kernel::parse(COPY).expect("the copy kernel parses");
    let m = compile::compile(&k).expect("the copy kernel compiles");
    assert_eq!(m.params, vec![ParamKind::Tensor, ParamKind::Tensor]);
    assert!(!m.reader.is_empty(), "two inbound copies");
    assert!(!m.writer.is_empty(), "two outbound copies");
    assert!(m.cores.iter().all(Vec::is_empty), "no compute");
}

/// Load, compute, store: the same unified program produces the three compute
/// programs as well as the two mover ones.
#[test]
fn an_elementwise_kernel_compiles_to_all_five_programs() {
    const NEG: &str = "\
version 9
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
version 9
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
        .filter(|(_, op)| matches!(op, Op::CopyTo { .. }))
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
version 9
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
version 9
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
version 9
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
version 9
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
version 9
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
version 9
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

/// A narrow `load`/`store` names a shape the format has, but no lowering yet:
/// every form is a loud `TT_TODO_ACCESS_SHAPE`, never a whole-tile move. The
/// one narrow load that does run is the mover's lane bases (`load 1,32` whose
/// only reader is a copy's index), which `an_access_shape_is_the_unit_a_copy_moves`
/// gates.
#[test]
fn a_narrow_access_is_refused_until_its_lowering_lands() {
    use ttStatus as S;
    let case = |what: &str, text: &str, want: S| {
        let err = match Kernel::parse(text) {
            Ok(k) => compile::compile(&k).expect_err(what),
            Err(e) => e,
        };
        assert_eq!(err, want, "{what}");
    };
    // A `1,32` row and a `32,1` column have no machine path yet: a row is one
    // face-row run per half, and a column has no unpack form at all (tt-metal
    // materialises one only through a transpose). A `1,1` access is not here:
    // it is the mover's scalar read (`a_symbolic_offset_into_a_row_buffer_reads_one_datum`).
    for shape in ["1,32", "32,1"] {
        let text = format!(
            "\
version 9
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
    // A page with no faces (`1,32`/`32,1` buffer) is refused the same way: its
    // unit is 32 datums, not a tile.
    case(
        "a 1,32 load from a row page",
        "\
version 9
grid 1,1
%p = param f32 1024
%v = buffer f32 1,32
copy 1,32 %v [0] = %p [0]
load 1,32 %x = %v [0]
%y = neg %x
",
        S::TT_TODO_ACCESS_SHAPE,
    );
    // A store of an accumulator is the one narrow store the verifier admits
    // (the region is packed whole and the mask trims it); the pack of a narrow
    // access is the whole masked page, so it has no lowering either.
    for shape in ["1,32", "32,1", "1,1"] {
        let text = format!(
            "\
version 9
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
version 9
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,64
%bo = buffer f32 32,64
%two = const i32 2
%step = const i32 1
%i = loop %two
%o = mul %i %step
copy 32,32 %bi [%o] = %in [%o]
load 32,32 %x = %bi [%o]
%y = neg %x
store 32,32 %bo [%o] = %y
copy 32,32 %out [%o] = %bo [%o]
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
version 9
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
version 9
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
            ParamKind::Tensor,
            ParamKind::Tensor
        ]
    );
}

/// Comparisons produce a `bool`, `sel` takes one, and `relu` is not an op:
/// the set is zyx's `BOp`/`UOp` plus `sel` and `matmul`.
#[test]
fn comparisons_make_a_bool_and_sel_takes_one() {
    const SELECT: &str = "\
version 9
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
    let bad = "version 9\n%x = const i32 3\n%m = sel %x %x %x\ngrid 1,1\n";
    let err = Kernel::parse(bad).expect_err("an i32 condition is refused");
    assert_eq!(err, ttStatus::TT_ERROR_CONDITION_NOT_BOOL);

    let err = Kernel::parse("version 9\n%x = const i32 3\n%y = relu %x\ngrid 1,1\n")
        .expect_err("relu is not an op");
    assert_eq!(err, ttStatus::TT_ERROR_PARSE);
}

/// A bitcast is a reinterpretation, not a conversion: the bits are the value, so
/// the two dtypes must be the same width. It lowers to nothing at all.
#[test]
fn a_bitcast_is_the_same_bits_under_another_name() {
    const SAME: &str = "\
version 9
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
version 9
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
version 9
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
version 9
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
version 9
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
version 9
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
version 9
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
%zero = const f32 0
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
version 9
grid 1,1
%pa = param f32 1024
%out = param bf16 1024
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer bf16 32,32
copy 32,32 %ba [0] = %pa [0]
copy 32,32 %bb [0] = %pa [0]
%m = acc bf16 32,32
%zero = const f32 0
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
version 9
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
        "version 9\n%m = acc f32 16,16\ngrid 1,1\n",
        S::TT_ERROR_ACC_SHAPE,
    );
    case(
        "an i32 accumulator",
        "version 9\n%m = acc i32 32,32\ngrid 1,1\n",
        S::TT_ERROR_ACC_DTYPE,
    );
    // The four shapes an `acc` may have. `32,32` is `matmul`'s; the others are
    // a reduce's, and the shape picks the fold.
    for shape in ["1,32", "32,1", "1,1", "32,32"] {
        let text = format!("version 9\n%m = acc f32 {shape}\ngrid 1,1\n");
        let k = Kernel::parse(&text).unwrap_or_else(|e| panic!("acc f32 {shape}: {e}"));
        assert_eq!(k.len(), 4, "acc f32 {shape}");
    }
    // `tf32` is a real format with no path yet: named, never guessed.
    case(
        "a tf32 parameter",
        "version 9\n%p = param tf32 1024\ngrid 1,1\n",
        S::TT_TODO_TF32,
    );
    // A store into an accumulator names no offset, and only a constant fills
    // it.
    case(
        "an offset on an accumulator store",
        "\
version 9
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
version 9
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
    let mut s = String::from("version 9\ngrid 1,1\n%px = param f32 1024\n");
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
version 9
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
version 9
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
    // The `1,1` fold's machine path is not built: refused loudly, never
    // lowered as something else.
    case(
        "the scalar fold",
        &reduce_program("1,1", "sum %m %x %s", true),
        S::TT_TODO_REDUCE,
    );
}
