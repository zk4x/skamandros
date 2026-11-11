// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The C ABI: the only stable surface `libTT.so` exports.
//!
//! Everything crosses as bytes plus scalars. A call returns `0` on success and a
//! non-zero code on failure, with a message in the out parameters; the caller
//! frees any buffer with [`tt_free`]. No Rust type, no allocation the caller did
//! not ask for.

use std::ffi::{c_char, c_int};

use crate::compile;
use crate::ir::Kernel;

/// Success.
pub const TT_OK: c_int = 0;
/// The program was refused; the error buffer says why.
pub const TT_INVALID: c_int = 1;
/// A null pointer or a zero length where one is required.
pub const TT_ARGUMENT: c_int = 2;

/// The ABI version this library implements.
pub const ABI_VERSION: u32 = 1;

/// The ABI version. A consumer compares this against the header it was built
/// with.
#[unsafe(no_mangle)]
pub extern "C" fn tt_abi_version() -> u32 {
    ABI_VERSION
}

/// Compile a program.
///
/// `ir`/`ir_len` is the text form (or, if it starts with `TTIR`, the byte form).
/// On success `*out` is a module (see `Module::to_bytes`) and `*out_len` its
/// length. On failure the code is non-zero, `*out` is null, and `*err`/`*err_len`
/// carry a message.
///
/// # Safety
///
/// Every pointer must be valid for its stated length; `out`, `out_len`, `err`
/// and `err_len` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_compile(
    ir: *const u8,
    ir_len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
    err: *mut *mut c_char,
    err_len: *mut usize,
) -> c_int {
    if out.is_null() || out_len.is_null() || err.is_null() || err_len.is_null() {
        return TT_ARGUMENT;
    }
    unsafe {
        *out = std::ptr::null_mut();
        *out_len = 0;
        *err = std::ptr::null_mut();
        *err_len = 0;
    }
    if ir.is_null() || ir_len == 0 {
        return fail(TT_ARGUMENT, "empty program", err, err_len);
    }
    let bytes = unsafe { std::slice::from_raw_parts(ir, ir_len) };
    let parsed = if bytes.starts_with(b"TTIR") {
        Kernel::from_bytes(bytes)
    } else {
        match std::str::from_utf8(bytes) {
            Ok(text) => Kernel::parse(text),
            Err(e) => {
                return fail(TT_INVALID, &format!("the program is not UTF-8: {e}"), err, err_len);
            }
        }
    };
    let kernel = match parsed {
        Ok(k) => k,
        Err(e) => return fail(TT_INVALID, &e.to_string(), err, err_len),
    };
    let module = match compile::compile(&kernel) {
        Ok(m) => m,
        Err(e) => return fail(TT_INVALID, &e.to_string(), err, err_len),
    };
    let mut bytes = module.to_bytes();
    let len = bytes.len();
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    unsafe {
        *out = ptr;
        *out_len = len;
    }
    TT_OK
}

/// Free a buffer the library returned.
///
/// # Safety
///
/// `ptr`/`len` must be a pair the library returned, freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// The grid a module launches over.
///
/// # Safety
///
/// `module`/`len` must be a module the library returned, and the two out
/// pointers writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_module_grid(
    module: *const u8,
    len: usize,
    x: *mut u32,
    y: *mut u32,
) -> c_int {
    if module.is_null() || len == 0 || x.is_null() || y.is_null() {
        return TT_ARGUMENT;
    }
    let bytes = unsafe { std::slice::from_raw_parts(module, len) };
    match compile::Module::from_bytes(bytes) {
        Ok(m) => {
            unsafe {
                *x = m.grid[0];
                *y = m.grid[1];
            }
            TT_OK
        }
        Err(_) => TT_INVALID,
    }
}

/// Copy one of a module's five programs into the caller's buffer.
///
/// `core` is `0` reader, `1` writer, `2` T0, `3` T1, `4` T2. The words are copied
/// into `out` (at most `cap` of them) and the program's true length is written to
/// `out_len`; a shorter buffer is an error, never a truncation.
///
/// # Safety
///
/// `module`/`len` must be a module the library returned; `out` must be writable
/// for `cap` words; `out_len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_module_program(
    module: *const u8,
    len: usize,
    core: u32,
    out: *mut u32,
    cap: usize,
    out_len: *mut usize,
) -> c_int {
    if module.is_null() || len == 0 || out_len.is_null() {
        return TT_ARGUMENT;
    }
    let bytes = unsafe { std::slice::from_raw_parts(module, len) };
    let m = match compile::Module::from_bytes(bytes) {
        Ok(m) => m,
        Err(_) => return TT_INVALID,
    };
    let words: &[u32] = match core {
        0 => &m.reader,
        1 => &m.writer,
        2..=4 => &m.cores[core as usize - 2],
        _ => return TT_ARGUMENT,
    };
    unsafe { *out_len = words.len() };
    // A null `out` is a length query.
    if out.is_null() {
        return TT_OK;
    }
    if words.len() > cap {
        return TT_ARGUMENT;
    }
    if !words.is_empty() {
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), out, words.len()) };
    }
    TT_OK
}

fn fail(code: c_int, message: &str, err: *mut *mut c_char, err_len: *mut usize) -> c_int {
    let mut bytes = message.as_bytes().to_vec();
    bytes.push(0);
    let len = bytes.len();
    let ptr = bytes.as_mut_ptr() as *mut c_char;
    std::mem::forget(bytes);
    unsafe {
        *err = ptr;
        *err_len = len;
    }
    code
}
