// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The C ABI: the only stable surface `libTT.so` exports.
//!
//! Everything crosses as pointers, lengths and scalars. A call returns
//! [`ttStatus`], and on failure writes a NUL-terminated message to the `err` out
//! pointer, which the caller frees with [`ttFree`]. A compiled module is an
//! opaque handle ([`ttModule`]): the caller never sees its bytes, only the
//! pointer [`ttCompile`]/[`ttModuleLoad`] hands back and [`ttLaunch`] consumes.
//!
//! # Errors
//!
//! One code, one meaning:
//!
//! * [`ttStatus::TT_ERROR_ARGUMENT`] -- a required pointer is null, a length is
//!   zero, the parameter count is not the module's, or a null buffer was given a
//!   nonzero length. Nothing was read or written.
//! * [`ttStatus::TT_ERROR_PROGRAM`] -- the text or bytes are not a program this
//!   compiler accepts. Nothing was done.
//! * [`ttStatus::TT_ERROR_MODULE`] -- the handle is not one this library handed
//!   out, or it was already unloaded. Nothing was done.
//! * [`ttStatus::TT_ERROR_UNSUPPORTED`] -- the program is valid and compiled, but
//!   this build cannot run it. Nothing was done.
//! * [`ttStatus::TT_ERROR_DEVICE`] -- the device, a transfer or a launch failed.
//!   This is the only code that can leave the device in an unknown state.

#![allow(non_camel_case_types)]

use std::collections::BTreeMap;
use std::ffi::{CString, c_char, c_void};
use std::sync::{Mutex, MutexGuard};

use crate::compile::{self, Module};
use crate::ir::Kernel;
use crate::runtime;
pub use crate::status::ttStatus;

/// The ABI version this library implements.
pub const ABI_VERSION: u32 = 2;

/// A compiled module. Opaque: the pointer is the handle, and its contents are
/// the library's.
#[repr(C)]
pub struct ttModule {
    _private: [u8; 0],
}

/// Every module this library has handed out. The map owns each module, so a
/// handle stays valid until [`ttModuleUnload`], and a pointer that is not a key
/// names nothing.
static MODULES: Mutex<BTreeMap<usize, Box<Module>>> = Mutex::new(BTreeMap::new());

fn modules() -> MutexGuard<'static, BTreeMap<usize, Box<Module>>> {
    // Poisoning needs a panic while the map is held, which cannot leave it
    // inconsistent; recover rather than fail every later compile.
    MODULES.lock().unwrap_or_else(|p| p.into_inner())
}

fn add_module(module: Module) -> *mut ttModule {
    let mut modules = modules();
    let boxed = Box::new(module);
    let ptr = (&*boxed) as *const Module as *mut ttModule;
    modules.insert(ptr as usize, boxed);
    ptr
}

fn lookup_module(modules: &BTreeMap<usize, Box<Module>>, handle: *mut ttModule) -> Option<&Module> {
    modules.get(&(handle as usize)).map(|m| &**m)
}

/// The ABI version. A consumer compares this against the header it was built
/// with.
#[unsafe(no_mangle)]
pub extern "C" fn ttAbiVersion() -> u32 {
    ABI_VERSION
}

/// Compile a program from its text form.
///
/// On success `*out` is a module handle and `*err` is null. On failure `*out` is
/// null and the code says why.
///
/// # Safety
///
/// `text` must be readable for `len` bytes; `out` and `err` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ttCompile(
    text: *const c_char,
    len: usize,
    out: *mut *mut ttModule,
    err: *mut *mut c_char,
) -> ttStatus {
    clear(err);
    if out.is_null() {
        return fail(ttStatus::TT_ERROR_ARGUMENT, err);
    }
    unsafe { *out = std::ptr::null_mut() };
    if text.is_null() || len == 0 {
        return fail(ttStatus::TT_ERROR_ARGUMENT, err);
    }
    let bytes = unsafe { std::slice::from_raw_parts(text as *const u8, len) };
    let Ok(source) = std::str::from_utf8(bytes) else {
        return fail(ttStatus::TT_ERROR_PROGRAM, err);
    };
    let kernel = match Kernel::parse(source) {
        Ok(k) => k,
        Err(e) => return fail(e, err),
    };
    let module = match compile::compile(&kernel) {
        Ok(m) => m,
        Err(e) => return fail(e, err),
    };
    unsafe { *out = add_module(module) };
    ttStatus::TT_OK
}

/// Compile a program from its byte form (`TTIR`).
///
/// Same contract as [`ttCompile`]; the input must be the byte form, not text.
///
/// # Safety
///
/// `bytes` must be readable for `len` bytes; `out` and `err` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ttModuleLoad(
    bytes: *const u8,
    len: usize,
    out: *mut *mut ttModule,
    err: *mut *mut c_char,
) -> ttStatus {
    clear(err);
    if out.is_null() {
        return fail(ttStatus::TT_ERROR_ARGUMENT, err);
    }
    unsafe { *out = std::ptr::null_mut() };
    if bytes.is_null() || len == 0 {
        return fail(ttStatus::TT_ERROR_ARGUMENT, err);
    }
    let bytes = unsafe { std::slice::from_raw_parts(bytes, len) };
    let kernel = match Kernel::from_bytes(bytes) {
        Ok(k) => k,
        Err(e) => return fail(e, err),
    };
    let module = match compile::compile(&kernel) {
        Ok(m) => m,
        Err(e) => return fail(e, err),
    };
    unsafe { *out = add_module(module) };
    ttStatus::TT_OK
}

/// Drop a module this library handed out. A second unload of the same handle is
/// [`ttStatus::TT_ERROR_MODULE`], never a double free.
///
/// # Safety
///
/// `module` must be a handle [`ttCompile`]/[`ttModuleLoad`] returned, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ttModuleUnload(module: *mut ttModule, err: *mut *mut c_char) -> ttStatus {
    clear(err);
    if module.is_null() {
        return fail(ttStatus::TT_ERROR_ARGUMENT, err);
    }
    match modules().remove(&(module as usize)) {
        Some(_) => ttStatus::TT_OK,
        None => fail(ttStatus::TT_ERROR_MODULE, err),
    }
}

/// Open the device.
///
/// Idempotent. The simulator is initialized once per process, from
/// `$TT_TTSIM_LIB` or `libttsim_bh.so` in the current directory.
///
/// # Safety
///
/// `err` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ttInit(err: *mut *mut c_char) -> ttStatus {
    clear(err);
    match runtime::init() {
        Ok(()) => ttStatus::TT_OK,
        Err(e) => fail(e, err),
    }
}

/// Close the device. DRAM allocations from it are gone; a later [`ttInit`]
/// opens a fresh device over the same simulator. Module handles are unaffected.
///
/// # Safety
///
/// `err` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ttRelease(err: *mut *mut c_char) -> ttStatus {
    clear(err);
    match runtime::release() {
        Ok(()) => ttStatus::TT_OK,
        Err(e) => fail(e, err),
    }
}

/// Reserve `bytes` of DRAM and write its address to `*addr`.
///
/// The address is opaque to the caller: pass it back to `ttMemcpyH2D`,
/// `ttMemcpyD2H` and `ttLaunch`. Allocations are 64-byte aligned.
///
/// # Safety
///
/// `addr` and `err` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ttMemAlloc(
    bytes: usize,
    addr: *mut u64,
    err: *mut *mut c_char,
) -> ttStatus {
    clear(err);
    if addr.is_null() {
        return fail(ttStatus::TT_ERROR_ARGUMENT, err);
    }
    unsafe { *addr = 0 };
    if bytes == 0 {
        return fail(ttStatus::TT_ERROR_ARGUMENT, err);
    }
    match runtime::alloc(bytes) {
        Ok(at) => {
            unsafe { *addr = at };
            ttStatus::TT_OK
        }
        Err(e) => fail(e, err),
    }
}

/// Copy `len` host bytes into DRAM at `dst`.
///
/// # Safety
///
/// `src` must be readable for `len` bytes; `err` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ttMemcpyH2D(
    dst: u64,
    src: *const u8,
    len: usize,
    err: *mut *mut c_char,
) -> ttStatus {
    clear(err);
    if src.is_null() || len == 0 {
        return fail(ttStatus::TT_ERROR_ARGUMENT, err);
    }
    let bytes = unsafe { std::slice::from_raw_parts(src, len) };
    match runtime::h2d(dst, bytes) {
        Ok(()) => ttStatus::TT_OK,
        Err(e) => fail(e, err),
    }
}

/// Copy `len` DRAM bytes at `src` into host memory at `dst`.
///
/// # Safety
///
/// `dst` must be writable for `len` bytes; `err` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ttMemcpyD2H(
    dst: *mut u8,
    src: u64,
    len: usize,
    err: *mut *mut c_char,
) -> ttStatus {
    clear(err);
    if dst.is_null() || len == 0 {
        return fail(ttStatus::TT_ERROR_ARGUMENT, err);
    }
    let out = unsafe { std::slice::from_raw_parts_mut(dst, len) };
    match runtime::d2h(src, out) {
        Ok(()) => ttStatus::TT_OK,
        Err(e) => fail(e, err),
    }
}

/// Run a module over its grid. `params` holds one DRAM address per kernel
/// parameter, in program order; `nparams` must equal the module's count.
///
/// The launch is synchronous: every program has finished when this returns.
///
/// # Safety
///
/// `module` must be a live handle; `params` readable for `nparams` addresses;
/// `err` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ttLaunch(
    module: *mut ttModule,
    params: *const u64,
    nparams: usize,
    err: *mut *mut c_char,
) -> ttStatus {
    clear(err);
    if module.is_null() {
        return fail(ttStatus::TT_ERROR_ARGUMENT, err);
    }
    if params.is_null() && nparams != 0 {
        return fail(ttStatus::TT_ERROR_ARGUMENT, err);
    }
    let addresses = if nparams == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(params, nparams) }
    };
    let modules = modules();
    let Some(m) = lookup_module(&modules, module) else {
        return fail(ttStatus::TT_ERROR_MODULE, err);
    };
    // The count is the caller's contract, not the device's business: refuse it
    // before anything is touched.
    if m.params as usize != addresses.len() {
        return fail(ttStatus::TT_ERROR_ARGUMENT, err);
    }
    match runtime::launch(m, addresses) {
        Ok(()) => ttStatus::TT_OK,
        // The failure is the status itself: a watchdog timeout reaches the
        // caller as `TT_ERROR_FIRMWARE_TIMEOUT`, not collapsed into a generic
        // device code.
        Err(e) => fail(e, err),
    }
}

/// Free a string this library returned. A null pointer is a no-op.
///
/// Module handles are not freed here; use [`ttModuleUnload`].
///
/// # Safety
///
/// `ptr` must be null or a pointer [`fail`]'s callers wrote, freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ttFree(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    unsafe { drop(CString::from_raw(ptr as *mut c_char)) };
}

/// Clear the caller's error out-parameter, if there is one.
fn clear(err: *mut *mut c_char) {
    if !err.is_null() {
        unsafe { *err = std::ptr::null_mut() };
    }
}

/// Leave the variant's name for the caller and return its integer. A null
/// `err` is tolerated: the status is the answer, its name is the spelling.
///
/// The name is the only text on the channel; a detail that matters is a
/// variant, not a message (`status.rs`).
fn fail(status: ttStatus, err: *mut *mut c_char) -> ttStatus {
    if !err.is_null() {
        let name = status.name().replace('\0', " ");
        let msg = CString::new(name).expect("a variant name has no NUL");
        unsafe { *err = msg.into_raw() };
    }
    status
}
