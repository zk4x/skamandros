// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The IR as a program format: the text form is the contract, the byte form is
//! the shipping encoding, and the two are checked against each other.

use crate::compile;
use crate::compile::ParamKind;
use crate::ir::{DType, Dim, Kernel, Op};
use crate::status::ttStatus;

/// The unified program: no reader/compute/writer segments, no unpack/pack, no
/// buffer layouts, no circular-buffer operations. Parameters and buffers are
/// mutable names; values are SSA; a shape term is a literal or a value; the
/// grid is as symbolic as a shape; index math has the full basic op set; and
/// `store_to`/`copy_to` put the destination first.
#[test]
fn the_unified_program_round_trips() {
    const PROGRAM: &str = "\
version 7
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
%m = acc f32 32,32
matmul %m %a %b
";

    let k = Kernel::parse(PROGRAM).unwrap_or_else(|e| panic!("the unified program parses: {e}"));
    assert_eq!(k.ops.len(), 31, "every op of the unified program");

    // Symbolic grid and symbolic parameter dims, with the grid after the values
    // it reads: a scalar parameter is a rank-0 shape.
    const SYMBOLIC: &str = "\
version 7
%d = param i32 0
%x = param bf16 1024,%d
grid %d,8
";

    let k = Kernel::parse(SYMBOLIC).unwrap_or_else(|e| panic!("the symbolic program parses: {e}"));
    assert_eq!(
        k.ops.len(),
        3,
        "the scalar param, the tensor param and the grid"
    );
}

#[test]
fn a_buffer_shape_that_is_not_one_or_a_multiple_of_32_is_refused() {
    for shape in ["33,32", "32,33", "16,32"] {
        let text = format!("version 7\n%x = buffer f32 {shape}\ngrid 1,1\n");
        let err = Kernel::parse(&text).expect_err(shape);
        assert_eq!(err, ttStatus::TT_ERROR_BUFFER_SHAPE, "{shape}");
    }
    // A buffer is always two-dimensional and always constant.
    for shape in ["32", "32,%d"] {
        let text = format!("version 7\n%d = const i32 4\n%x = buffer f32 {shape}\ngrid 1,1\n");
        assert!(
            Kernel::parse(&text).is_err(),
            "{shape} is not a buffer shape"
        );
    }
}

#[test]
fn a_zero_dimension_is_not_the_way_to_write_a_tensor() {
    let text = "version 7\n%p = param f32 0,32\ngrid 1,1\n";
    let err = Kernel::parse(text).expect_err("a zero in a dimension list");
    assert_eq!(err, ttStatus::TT_ERROR_ZERO_DIMENSION);
}

#[test]
fn an_implicit_dtype_conversion_is_refused() {
    let text = "\
version 7
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
    let text = "version 7\n%x = add %a %b\n%a = const i32 1\n%b = const i32 2\ngrid 1,1\n";
    let err = Kernel::parse(text).expect_err("undefined value");
    assert_eq!(err, ttStatus::TT_ERROR_PARSE);
}

#[test]
fn a_value_does_not_escape_its_loop() {
    let text = "\
version 7
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
    let err = Kernel::parse("version 7\n%x = const i32 1\n").expect_err("no grid");
    assert_eq!(err, ttStatus::TT_ERROR_GRID);
    let err = Kernel::parse("version 7\ngrid 1,1\ngrid 2,2\n").expect_err("two grids");
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
version 7
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
version 7
%in = param f32 4096
%out = param f32 4096
%bi = buffer f32 32,32
%bo = buffer f32 32,32
copy_to %bi from %in [0]
%x = load %bi [0]
%y = neg %x
store_to %bo from %y [0]
copy_to %out from %bo [0]
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
version 7
%in = param f32 4096
%out = param f32 4096
%x = load %in [0]
%y = neg %x
store_to %out from %y [0]
grid 1,1
";
    let k = Kernel::parse(DIRECT).expect("the direct kernel parses");
    let types = k.verify().expect("the direct form verifies");
    let staged = crate::mover::add_movers(&k, &types).expect("the access is staged");
    let buffers = staged
        .ops
        .iter()
        .filter(|op| matches!(op, Op::Buffer { .. }))
        .count();
    let copies = staged
        .ops
        .iter()
        .filter(|op| matches!(op, Op::CopyTo { .. }))
        .count();
    assert_eq!(buffers, 2, "one staged buffer per direct access");
    assert_eq!(copies, 2, "one inbound and one outbound copy");
    assert!(
        staged
            .ops
            .iter()
            .filter(|op| matches!(
                op,
                Op::Load {
                    offset: Dim::Const(0),
                    ..
                }
            ))
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

/// A constant-bound loop stays a loop: the movers and the compute roles get a
/// RISC-V branch, not a copy of the body per iteration.
#[test]
fn a_constant_loop_becomes_a_branch() {
    const LOOP_TWO: &str = "\
version 7
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,64
%bo = buffer f32 32,64
%two = const i32 2
%step = const i32 1024
%i = loop %two
%o = mul %i %step
copy_to %bi from %in [%o]
%x = load %bi [%o]
%y = neg %x
store_to %bo from %y [%o]
copy_to %out from %bo [%o]
endloop
grid 1,1
";
    let k = Kernel::parse(LOOP_TWO).expect("the loop kernel parses");
    let m = compile::compile(&k).expect("the loop kernel compiles");
    assert!(!m.reader.is_empty() && !m.writer.is_empty());
    assert!(m.cores.iter().all(|c| !c.is_empty()));
    // Both movers branch backwards: one body, run twice.
    assert!(has_backward_branch(&m.reader), "the reader loops");
    assert!(has_backward_branch(&m.writer), "the writer loops");
    for (role, words) in ["t0", "t1", "t2"].into_iter().zip(&m.cores) {
        assert!(has_backward_branch(words), "{role} loops");
    }
}

/// A RISC-V `BRANCH` (opcode `0x63`) whose B-type immediate is negative, i.e.
/// a branch back to an earlier word.
fn has_backward_branch(words: &[u32]) -> bool {
    words.iter().any(|word| {
        word & 0x7f == 0x63 && {
            // imm[12] is the sign bit.
            let imm12 = (word >> 31) & 1;
            imm12 == 1
        }
    })
}

/// A loop whose bound is a launch-bound scalar becomes the same RISC-V branch:
/// the count is loaded from the arguments in each role, and the movers branch.
#[test]
fn a_dynamic_loop_becomes_a_branch() {
    const DYNAMIC: &str = "\
version 7
%n = param i32 0
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,32
%bo = buffer f32 32,32
%i = loop %n
copy_to %bi from %in [0]
%x = load %bi [0]
%y = neg %x
store_to %bo from %y [0]
copy_to %out from %bo [0]
endloop
grid 1,1
";
    let k = Kernel::parse(DYNAMIC).expect("the dynamic loop parses");
    let m = compile::compile(&k).expect("a dynamic loop is lowered");
    assert!(has_backward_branch(&m.reader), "the reader loops");
    assert!(has_backward_branch(&m.writer), "the writer loops");
    for (role, words) in ["t0", "t1", "t2"].into_iter().zip(&m.cores) {
        assert!(has_backward_branch(words), "{role} loops");
    }
}

/// `grid.x`/`grid.y` name this core's logical coordinate, and a grid extent may
/// be a launch-bound scalar.
#[test]
fn the_grid_is_symbolic_and_the_coordinate_is_a_value() {
    const SYMBOLIC: &str = "\
version 7
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
copy_to %bi from %in [%o2]
%v = load %bi [0]
%n = neg %v
store_to %bo from %n [0]
copy_to %out from %bo [%o2]
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
version 7
%x = const i32 3
%y = const i32 5
%lt = cmplt %x %y
%m = sel %lt %x %y
grid 1,1
";
    let k = Kernel::parse(SELECT).expect("the select program parses");
    let types = k.verify().expect("it verifies");
    assert_eq!(types.values[2].dtype, DType::Bool, "cmplt is bool");
    assert_eq!(types.values[3].dtype, DType::I32, "sel is i32");

    // A bool condition is required, and a value with no bool is not one.
    let bad = "version 7\n%x = const i32 3\n%m = sel %x %x %x\ngrid 1,1\n";
    let err = Kernel::parse(bad).expect_err("an i32 condition is refused");
    assert_eq!(err, ttStatus::TT_ERROR_CONDITION_NOT_BOOL);

    let err = Kernel::parse("version 7\n%x = const i32 3\n%y = relu %x\ngrid 1,1\n")
        .expect_err("relu is not an op");
    assert_eq!(err, ttStatus::TT_ERROR_PARSE);
}

/// A bitcast is a reinterpretation, not a conversion: the bits are the value, so
/// the two dtypes must be the same width. It lowers to nothing at all.
#[test]
fn a_bitcast_is_the_same_bits_under_another_name() {
    const SAME: &str = "\
version 7
%in = param f32 2048
%out = param i32 2048
%bi = buffer f32 32,32
%bo = buffer i32 32,32
copy_to %bi from %in [0]
%x = load %bi [0]
%y = bitcast %x to i32
store_to %bo from %y [0]
copy_to %out from %bo [0]
grid 1,1
";
    let k = Kernel::parse(SAME).expect("a bitcast between 32-bit dtypes");
    let m = compile::compile(&k).expect("a bitcast lowers to nothing");
    assert!(
        m.cores.iter().all(|c| !c.is_empty()),
        "unpack, math and pack"
    );

    // 32 bits do not fit a 16-bit dtype, and a bitcast may not convert.
    const NARROW: &str = "\
version 7
%b = buffer f32 32,32
%x = load %b [0]
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
version 7
grid 1,1
%pa = param f32 1024
%pb = param f32 1024
%out = param f32 1024
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer f32 32,32
copy_to %ba from %pa [0]
copy_to %bb from %pb [0]
%m = acc f32 32,32
%zero = const f32 0
store_to %m from %zero
%a = load %ba [0]
%b = load %bb [0]
matmul %m %a %b
store_to %bo from %m [0]
copy_to %out from %bo [0]
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

    // An operand the datapath computed has no page to unpack into `Src`.
    case(
        "a negated operand",
        "\
version 7
grid 1,1
%pa = param f32 1024
%out = param f32 1024
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer f32 32,32
copy_to %ba from %pa [0]
copy_to %bb from %pa [0]
%m = acc f32 32,32
%zero = const f32 0
store_to %m from %zero
%a = load %ba [0]
%b = load %bb [0]
%n = neg %a
matmul %m %n %b
store_to %bo from %m [0]
copy_to %out from %bo [0]
",
        S::TT_ERROR_MATMUL_OPERAND,
    );

    // The destination is not an accumulator. Only an `acc` is written in place
    // by the Matrix Unit; a buffer is L1 and a value is immutable.
    case(
        "a buffer where the accumulator goes",
        "\
version 7
grid 1,1
%pa = param f32 1024
%out = param f32 1024
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer f32 32,32
copy_to %ba from %pa [0]
copy_to %bb from %pa [0]
%a = load %ba [0]
%b = load %bb [0]
matmul %bo %a %b
copy_to %out from %bo [0]
",
        S::TT_ERROR_MATMUL_ACC,
    );

    // A bf16 accumulator in a module whose `Dst` is 32-bit: the Matrix Unit
    // would accumulate in FP32, not at the bf16 the accumulator names.
    case(
        "bf16 accumulation beside an f32 value",
        "\
version 7
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
copy_to %ba from %pa [0]
copy_to %bb from %pa [0]
copy_to %bf from %pf [0]
%m = acc bf16 32,32
%zero = const f32 0
store_to %m from %zero
%a = load %ba [0]
%b = load %bb [0]
%f = load %bf [0]
%g = neg %f
store_to %bg from %g [0]
copy_to %of from %bg [0]
matmul %m %a %b
store_to %bo from %m [0]
copy_to %out from %bo [0]
",
        S::TT_ERROR_MATMUL_ACCUM_WIDE,
    );

    // `f32` operands are TF32 in `Src` and cannot accumulate into a 16-bit
    // `Dst`.
    case(
        "f32 operands, bf16 accumulator",
        "\
version 7
grid 1,1
%pa = param f32 1024
%out = param bf16 1024
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer bf16 32,32
copy_to %ba from %pa [0]
copy_to %bb from %pa [0]
%m = acc bf16 32,32
%zero = const f32 0
store_to %m from %zero
%a = load %ba [0]
%b = load %bb [0]
matmul %m %a %b
store_to %bo from %m [0]
copy_to %out from %bo [0]
",
        S::TT_ERROR_MATMUL_DTYPE,
    );

    // An `i32` operand: integer matmul is `Int8`/`Int32` `Src` formats and a
    // different set of phases.
    case(
        "i32 operands",
        "\
version 7
grid 1,1
%pa = param i32 1024
%out = param i32 1024
%ba = buffer i32 32,32
%bo = buffer i32 32,32
copy_to %ba from %pa [0]
%m = acc f32 32,32
%zero = const f32 0
store_to %m from %zero
%a = load %ba [0]
matmul %m %a %a
store_to %bo from %m [0]
copy_to %out from %bo [0]
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
        "version 7\n%m = acc f32 16,16\ngrid 1,1\n",
        S::TT_ERROR_ACC_SHAPE,
    );
    case(
        "an i32 accumulator",
        "version 7\n%m = acc i32 32,32\ngrid 1,1\n",
        S::TT_ERROR_ACC_DTYPE,
    );
    // A store into an accumulator names no offset, and only a constant fills
    // it.
    case(
        "an offset on an accumulator store",
        "\
version 7
grid 1,1
%m = acc f32 32,32
%zero = const f32 0
store_to %m from %zero [1024]
",
        S::TT_ERROR_ACC_OFFSET,
    );
    case(
        "a tile stored into an accumulator",
        "\
version 7
grid 1,1
%p = param f32 1024
%b = buffer f32 32,32
copy_to %b from %p [0]
%m = acc f32 32,32
%x = load %b [0]
store_to %m from %x
",
        S::TT_TODO_ACC_STORE,
    );
}

/// One tile in and one tile out, with `body` as the reduce: the program a
/// reduce gate is written as. `with_scaler` adds the second operand, which
/// `max` does not name.
fn reduce_program(body: &str, with_scaler: bool) -> String {
    let mut s = String::from("version 7\ngrid 1,1\n%px = param f32 1024\n");
    if with_scaler {
        s.push_str("%ps = param f32 1024\n");
    }
    s.push_str("%out = param f32 1024\n%bx = buffer f32 32,32\n");
    if with_scaler {
        s.push_str("%bs = buffer f32 32,32\n");
    }
    s.push_str("%bo = buffer f32 32,32\ncopy_to %bx from %px [0]\n");
    if with_scaler {
        s.push_str("copy_to %bs from %ps [0]\n");
    }
    s.push_str(
        "%m = acc f32 32,32\n%zero = const f32 0\nstore_to %m from %zero\n%x = load %bx [0]\n",
    );
    if with_scaler {
        s.push_str("%s = load %bs [0]\n");
    }
    s.push_str(body);
    s.push_str("\nstore_to %bo from %m [0]\ncopy_to %out from %bo [0]\n");
    s
}

/// The three dimensions and both folds parse. `sum` names the scaler; `max` does
/// not, because `GMPOOL` applies only the exponent of a non-unity scaler.
#[test]
fn the_reduce_forms_parse() {
    for (body, with_scaler) in [
        ("sum row %m %x %s", true),
        ("sum col %m %x %s", true),
        ("sum scalar %m %x %s", true),
        ("max row %m %x", false),
        ("max col %m %x", false),
        ("max scalar %m %x", false),
    ] {
        let text = reduce_program(body, with_scaler);
        let k = Kernel::parse(&text).unwrap_or_else(|e| panic!("{body} parses: {e}"));
        assert!(!k.ops.is_empty(), "{body}");
    }
}

#[test]
fn a_sum_row_compiles_to_the_matrix_unit() {
    let k = Kernel::parse(&reduce_program("sum row %m %x %s", true)).expect("it parses");
    let m = compile::compile(&k).unwrap_or_else(|e| panic!("it compiles: {e}"));
    assert_eq!(m.params.len(), 3);
}

#[test]
fn a_sum_col_compiles_to_the_gapool() {
    let k = Kernel::parse(&reduce_program("sum col %m %x %s", true)).expect("it parses");
    let m = compile::compile(&k).unwrap_or_else(|e| panic!("it compiles: {e}"));
    assert_eq!(m.params.len(), 3);
}

#[test]
fn a_max_col_compiles_to_the_gmpool() {
    let k = Kernel::parse(&reduce_program("max col %m %x", false)).expect("it parses");
    let m = compile::compile(&k).unwrap_or_else(|e| panic!("it compiles: {e}"));
    // The compiler owns one extra page for `max`'s 1.0 `SrcB`.
    assert_eq!(m.params.len(), 2);
}

#[test]
fn a_max_row_compiles_to_the_gmpool_and_the_transposes() {
    let k = Kernel::parse(&reduce_program("max row %m %x", false)).expect("it parses");
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

    // An operand the datapath computed has no page to unpack into `Src`.
    case(
        "a negated operand",
        &reduce_program("%n = neg %x\nsum row %m %n %s", true),
        S::TT_ERROR_REDUCE_OPERAND,
    );
    // The destination is not an accumulator.
    case(
        "a buffer where the accumulator goes",
        &reduce_program("sum row %bo %x %s", true),
        S::TT_ERROR_REDUCE_ACC,
    );
    // The scaler must be the operand's dtype: `SrcB` takes the same format.
    case(
        "an f32 tile with a bf16 scaler",
        "\
version 7
grid 1,1
%px = param f32 1024
%ps = param bf16 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bs = buffer bf16 32,32
%bo = buffer f32 32,32
copy_to %bx from %px [0]
copy_to %bs from %ps [0]
%m = acc f32 32,32
%zero = const f32 0
store_to %m from %zero
%x = load %bx [0]
%s = load %bs [0]
sum row %m %x %s
store_to %bo from %m [0]
copy_to %out from %bo [0]
",
        S::TT_ERROR_REDUCE_DTYPE,
    );
    // 32-bit operands cannot accumulate into a 16-bit `Dst`.
    case(
        "f32 operands into a bf16 accumulator",
        "\
version 7
grid 1,1
%px = param f32 1024
%ps = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bs = buffer f32 32,32
%bo = buffer f32 32,32
copy_to %bx from %px [0]
copy_to %bs from %ps [0]
%m = acc bf16 32,32
%zero = const f32 0
store_to %m from %zero
%x = load %bx [0]
%s = load %bs [0]
sum row %m %x %s
store_to %bo from %m [0]
copy_to %out from %bo [0]
",
        S::TT_ERROR_REDUCE_DTYPE,
    );
    // The forms whose lowering lands next are refused loudly, never lowered
    // as something else.
    case(
        "sum scalar",
        &reduce_program("sum scalar %m %x %s", true),
        S::TT_TODO_REDUCE,
    );
}
