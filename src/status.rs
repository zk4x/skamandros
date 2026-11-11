// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

#![allow(non_camel_case_types)]

//! The one error enum: what every fallible call returns.
//!
//! It is `#[repr(C)]` and integer-valued, because the C ABI is the only stable
//! surface and a caller sees only the integer. There is no second error type and
//! no string in the enum: information a caller needs is a variant.
//!
//! The migration from the string errors is in progress (`TODO.md`, "The error
//! channel"): a new failure gets its own variant here as it is written, and the
//! older sites move to variants gradually.

/// What every fallible call returns.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ttStatus {
    /// Success. `*err` is null.
    TT_OK = 0,
    /// A caller mistake: a null required pointer, a zero length, a wrong
    /// parameter count. Nothing was read or written.
    TT_ERROR_ARGUMENT = 1,
    /// The text or bytes were not a program this compiler accepts.
    TT_ERROR_PROGRAM = 2,
    /// The handle is not one this library handed out, or was already unloaded.
    TT_ERROR_MODULE = 3,
    /// The program compiled, but this build cannot run it yet.
    TT_ERROR_UNSUPPORTED = 4,
    /// The device, a transfer or the launch failed.
    TT_ERROR_DEVICE = 5,
    /// A buffer has more than one producer or more than one consumer. The
    /// one-producer/one-consumer circular buffer is all that is implemented;
    /// the multi-producer/multi-consumer form is a concrete future item.
    TT_TODO_MULTI_PRODUCER_OR_CONSUMER = 6,
    /// A buffer's roles are not one of the implemented pairs (reader/unpacker,
    /// packer/writer, reader/writer). Any other pair, or a missing end, is a
    /// concrete future item.
    TT_TODO_CIRCULAR_BUFFER_ROLE = 7,
    /// The text is not a program this parser accepts (a token, a keyword, a
    /// bracket, a value name).
    TT_ERROR_PARSE = 8,
    /// The version line, or a byte program's version, is not the one this build
    /// reads.
    TT_ERROR_VERSION = 9,
    /// The bytes are not a `TTIR` program.
    TT_ERROR_NOT_TTIR = 10,
    /// The input continues after a complete program.
    TT_ERROR_TRAILING_INPUT = 11,
    /// A buffer's dimension is neither `1` nor a multiple of 32.
    TT_ERROR_BUFFER_SHAPE = 12,
    /// A buffer shape or a parameter dimension is zero.
    TT_ERROR_ZERO_DIMENSION = 13,
    /// A value is used before its definition.
    TT_ERROR_VALUE_UNDEFINED = 14,
    /// A value is used outside the loop it was defined in.
    TT_ERROR_VALUE_OUT_OF_SCOPE = 15,
    /// The same value name is defined twice.
    TT_ERROR_NAME_DUPLICATE = 16,
    /// The program has no grid, or more than one.
    TT_ERROR_GRID = 17,
    /// An operand's dtype or kind does not match; there is no implicit cast.
    TT_ERROR_NO_IMPLICIT_CAST = 18,
    /// A bracket offset runs past its value, or a term is not an index.
    TT_ERROR_OFFSET = 19,
    /// `endloop` without a matching `loop`.
    TT_ERROR_ENDLOOP_WITHOUT_LOOP = 20,
    /// A loop reaches lowering. Loops become RISC-V branches; that is not
    /// implemented yet.
    TT_TODO_LOOP = 21,
    /// A `sel` (or its source) condition is not a `bool`.
    TT_ERROR_CONDITION_NOT_BOOL = 22,
}

impl ttStatus {
    /// The variant's name, for the ABI's human-readable `err` string. Not state:
    /// the enum itself carries no string.
    pub const fn name(self) -> &'static str {
        match self {
            ttStatus::TT_OK => "TT_OK",
            ttStatus::TT_ERROR_ARGUMENT => "TT_ERROR_ARGUMENT",
            ttStatus::TT_ERROR_PROGRAM => "TT_ERROR_PROGRAM",
            ttStatus::TT_ERROR_MODULE => "TT_ERROR_MODULE",
            ttStatus::TT_ERROR_UNSUPPORTED => "TT_ERROR_UNSUPPORTED",
            ttStatus::TT_ERROR_DEVICE => "TT_ERROR_DEVICE",
            ttStatus::TT_TODO_MULTI_PRODUCER_OR_CONSUMER => "TT_TODO_MULTI_PRODUCER_OR_CONSUMER",
            ttStatus::TT_TODO_CIRCULAR_BUFFER_ROLE => "TT_TODO_CIRCULAR_BUFFER_ROLE",
            ttStatus::TT_ERROR_PARSE => "TT_ERROR_PARSE",
            ttStatus::TT_ERROR_VERSION => "TT_ERROR_VERSION",
            ttStatus::TT_ERROR_NOT_TTIR => "TT_ERROR_NOT_TTIR",
            ttStatus::TT_ERROR_TRAILING_INPUT => "TT_ERROR_TRAILING_INPUT",
            ttStatus::TT_ERROR_BUFFER_SHAPE => "TT_ERROR_BUFFER_SHAPE",
            ttStatus::TT_ERROR_ZERO_DIMENSION => "TT_ERROR_ZERO_DIMENSION",
            ttStatus::TT_ERROR_VALUE_UNDEFINED => "TT_ERROR_VALUE_UNDEFINED",
            ttStatus::TT_ERROR_VALUE_OUT_OF_SCOPE => "TT_ERROR_VALUE_OUT_OF_SCOPE",
            ttStatus::TT_ERROR_NAME_DUPLICATE => "TT_ERROR_NAME_DUPLICATE",
            ttStatus::TT_ERROR_GRID => "TT_ERROR_GRID",
            ttStatus::TT_ERROR_NO_IMPLICIT_CAST => "TT_ERROR_NO_IMPLICIT_CAST",
            ttStatus::TT_ERROR_OFFSET => "TT_ERROR_OFFSET",
            ttStatus::TT_ERROR_ENDLOOP_WITHOUT_LOOP => "TT_ERROR_ENDLOOP_WITHOUT_LOOP",
            ttStatus::TT_TODO_LOOP => "TT_TODO_LOOP",
            ttStatus::TT_ERROR_CONDITION_NOT_BOOL => "TT_ERROR_CONDITION_NOT_BOOL",
        }
    }
}

impl core::fmt::Display for ttStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}
