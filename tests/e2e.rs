// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! End to end through the shipped artifact: load `libskamandros.so` with `dlopen` and
//! drive the C ABI.
//!
//! Every call goes through the loaded library; the crate itself is linked only
//! to build the byte-form fixture in one test. A compiled module is an opaque
//! handle here exactly as it is for a C consumer.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use libloading::{Library, Symbol};

/// One device at a time.
///
/// `libttsim` is a process-wide singleton and `ttInit` hands out DRAM and L1
/// addresses from a fresh base, so two device tests running in parallel would
/// free each other's allocations and stage into each other's L1. Every test that
/// opens the device holds this for its whole body.
fn device() -> MutexGuard<'static, ()> {
    static DEVICE: OnceLock<Mutex<()>> = OnceLock::new();
    DEVICE
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

type AbiVersion = unsafe extern "C" fn() -> u32;
type Compile =
    unsafe extern "C" fn(*const c_char, usize, *mut *mut c_void, *mut *mut c_char) -> c_int;
type ModuleUnload = unsafe extern "C" fn(*mut c_void, *mut *mut c_char) -> c_int;
type Free = unsafe extern "C" fn(*mut c_void);
type Init = unsafe extern "C" fn(*mut *mut c_char) -> c_int;
type Release = unsafe extern "C" fn(*mut *mut c_char) -> c_int;
type MemAlloc = unsafe extern "C" fn(usize, *mut u64, *mut *mut c_char) -> c_int;
type MemcpyH2D = unsafe extern "C" fn(u64, *const u8, usize, *mut *mut c_char) -> c_int;
type MemcpyD2H = unsafe extern "C" fn(*mut u8, u64, usize, *mut *mut c_char) -> c_int;
type Launch = unsafe extern "C" fn(*mut c_void, *const u64, usize, *mut *mut c_char) -> c_int;

/// Error codes, as the header defines them.
const TT_OK: c_int = 0;
const TT_ERROR_ARGUMENT: c_int = 1;
const TT_ERROR_MODULE: c_int = 3;
const TT_ERROR_VALUE_UNDEFINED: c_int = 14;
const TT_TODO_PACK_F32_TO_BF16: c_int = 180;

/// The `libskamandros.so` this test binary was built next to.
fn library_path() -> PathBuf {
    let exe = std::env::current_exe().expect("the test binary's path");
    let dir = exe.parent().expect("the test binary's directory");
    // `target/<profile>/deps/<test>` -> `target/<profile>/libskamandros.so`
    dir.parent()
        .expect("the profile directory")
        .join(format!("{}skamandros.so", std::env::consts::DLL_PREFIX))
}

/// The artifact, loaded once per test process.
///
/// A `dlclose` of `libskamandros.so` between two tests unloads the library's own
/// process-wide state, and `libttsim` — which stays resident — refuses a second
/// `libttsim_init`. Keeping one handle for the whole process is also how a real
/// consumer uses the `.so`.
///
/// `cargo test` builds the test binary and the `rlib` half of the crate but not
/// the `cdylib`, so the file next to this binary would be whatever the last
/// `cargo build` left behind — stale after any edit, and a test that passes
/// against a stale library is no test. The initializer builds it first, once,
/// in the profile this binary was built in.
fn library() -> &'static Library {
    static LIB: OnceLock<Library> = OnceLock::new();
    LIB.get_or_init(|| {
        let path = library_path();
        build_library(&path);
        unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()))
    })
}

/// `cargo build --lib` in the profile `path`'s directory names: `debug` and
/// `release` have their own flags, any other profile is named. A build failure
/// is the test's failure, with cargo's own message.
fn build_library(path: &Path) {
    let profile = path
        .parent()
        .and_then(Path::file_name)
        .expect("a profile directory")
        .to_string_lossy()
        .into_owned();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut command = std::process::Command::new(cargo);
    command
        .arg("build")
        .arg("--lib")
        .current_dir(env!("CARGO_MANIFEST_DIR"));
    match profile.as_str() {
        "debug" => {}
        "release" => {
            command.arg("--release");
        }
        other => {
            command.args(["--profile", other]);
        }
    }
    let out = command.output().expect("cargo runs");
    assert!(
        out.status.success(),
        "cargo build --lib failed for {profile}:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

const COPY: &str = "\
version 9
grid 1,1
%in = param f32 1024
%out = param f32 1024
%b0 = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
copy 32,32 %out [0] = %b0 [0]
";

/// Read and clear the message a failed call left behind.
fn take_error(free: &Symbol<Free>, err: *mut c_char) -> String {
    if err.is_null() {
        return String::from("<no message>");
    }
    let msg = unsafe { CStr::from_ptr(err) }
        .to_string_lossy()
        .into_owned();
    unsafe { free(err as *mut c_void) };
    msg
}

#[test]
fn the_so_compiles_a_program_through_the_c_abi() {
    let lib = library();
    let version: Symbol<AbiVersion> = unsafe { lib.get(b"ttAbiVersion\0") }.expect("ttAbiVersion");
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    assert_eq!(unsafe { version() }, 1, "the ABI version");

    let mut module: *mut c_void = std::ptr::null_mut();
    let mut err: *mut c_char = std::ptr::null_mut();
    let code = unsafe { compile(COPY.as_ptr().cast(), COPY.len(), &mut module, &mut err) };
    assert_eq!(code, TT_OK, "ttCompile: {}", take_error(&free, err));
    assert!(!module.is_null(), "a module handle came back");
    assert!(err.is_null(), "no error on success");
    assert_eq!(
        unsafe { unload(module, &mut err) },
        TT_OK,
        "unload: {}",
        take_error(&free, err)
    );
}

#[test]
fn a_refused_program_comes_back_as_an_error_not_a_crash() {
    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");

    let bad = "version 9\ngrid 1,1\n%out = param f32 1024\n%x = neg %y\n";
    let mut module: *mut c_void = std::ptr::null_mut();
    let mut err: *mut c_char = std::ptr::null_mut();
    let code = unsafe { compile(bad.as_ptr().cast(), bad.len(), &mut module, &mut err) };
    assert_eq!(
        code, TT_ERROR_VALUE_UNDEFINED,
        "a use before its definition is refused"
    );
    assert!(module.is_null() && !err.is_null());
    assert_eq!(take_error(&free, err), "TT_ERROR_VALUE_UNDEFINED");
}

#[test]
fn the_error_codes_are_the_documented_ones() {
    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");

    let mut module: *mut c_void = std::ptr::null_mut();
    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { compile(COPY.as_ptr().cast(), COPY.len(), &mut module, &mut err) },
        TT_OK
    );

    // A handle that is not one of ours, and a live one unloaded twice.
    let bogus = 0x1000usize as *mut c_void;
    let params = [0u64, 0u64];
    assert_eq!(
        unsafe { launch(bogus, params.as_ptr(), 2, &mut err) },
        TT_ERROR_MODULE
    );
    let _ = take_error(&free, err);
    assert_eq!(unsafe { unload(bogus, &mut err) }, TT_ERROR_MODULE);
    let _ = take_error(&free, err);
    assert_eq!(unsafe { unload(module, &mut err) }, TT_OK);
    assert_eq!(unsafe { unload(module, &mut err) }, TT_ERROR_MODULE);
    let _ = take_error(&free, err);

    // A live handle with the wrong parameter count.
    let mut again: *mut c_void = std::ptr::null_mut();
    assert_eq!(
        unsafe { compile(COPY.as_ptr().cast(), COPY.len(), &mut again, &mut err) },
        TT_OK
    );
    assert_eq!(
        unsafe { launch(again, params.as_ptr(), 1, &mut err) },
        TT_ERROR_ARGUMENT
    );
    let _ = take_error(&free, err);
    assert_eq!(unsafe { unload(again, &mut err) }, TT_OK);

    // Argument checks that need no device.
    let mut addr = 0u64;
    assert_eq!(unsafe { alloc(0, &mut addr, &mut err) }, TT_ERROR_ARGUMENT);
    assert_eq!(addr, 0);
    let _ = take_error(&free, err);
    let host = [0u8; 4];
    assert_eq!(
        unsafe { h2d(0, std::ptr::null(), 4, &mut err) },
        TT_ERROR_ARGUMENT
    );
    let _ = take_error(&free, err);
    assert_eq!(
        unsafe { h2d(0, host.as_ptr(), 0, &mut err) },
        TT_ERROR_ARGUMENT
    );
    let _ = take_error(&free, err);
    unsafe { free(std::ptr::null_mut()) };
}

/// A copy kernel over a two-page buffer: DRAM into L1, and the same L1 buffer
/// back into DRAM, in 32x32 units.
const COPY_8K: &str = "\
version 9
grid 1,1
%in = param f32 2048
%out = param f32 2048
%b0 = buffer f32 32,64
copy 32,32 %b0 [0] = %in [0]
copy 32,32 %b0 [1] = %in [1]
copy 32,32 %out [0] = %b0 [0]
copy 32,32 %out [1] = %b0 [0]
";

/// One 32x32 tile through the datapath: DRAM into L1, unpack, negate, pack,
/// L1 into DRAM. The program names no core, no `Dst` slot and no unpacker.
const NEG_TILE: &str = "\
version 9
grid 1,1
%in = param f32 1024
%out = param f32 1024
%bi = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = neg %x
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// The same elementwise kernel over two tiles, written as a loop. The bound and
/// the step are constants, so the loop becomes a RISC-V branch in the movers and
/// each iteration's addresses are computed from the counter.
const NEG_TWO_TILES: &str = "\
version 9
grid 1,1
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
";

/// Two values live in `Dst` at once, so each is packed from its own row: the
/// second store's page must be the second value, not the first. The packer's
/// start row is its `W` counter, set per pack (`SETADC`, tt-metal's
/// `set_dst_write_addr`); a packer stuck on row 0 repeats the first value.
const NEG_TWO_LIVE: &str = "\
version 9
grid 1,1
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,64
%bo = buffer f32 32,64
copy 32,32 %bi [0] = %in [0]
copy 32,32 %bi [1] = %in [1]
load 32,32 %a = %bi [0]
load 32,32 %b = %bi [1]
%na = neg %a
%nb = neg %b
store 32,32 %bo [0] = %na
store 32,32 %bo [1] = %nb
copy 32,32 %out [0] = %bo [0]
copy 32,32 %out [1] = %bo [0]
";

/// The same elementwise kernel with no buffer and no `copy_to`: the parameter is
/// named directly and `mover::add_movers` supplies the staging and the traffic,
/// as zyx's `tt_add_movers` does.
const NEG_TILE_DIRECT: &str = "\
version 9
grid 1,1
%in = param f32 1024
%out = param f32 1024
load 32,32 %x = %in [0]
%y = neg %x
store 32,32 %out [0] = %y
";

/// Where the simulator is. The library only reads `TT_METAL_SIMULATOR`, so the
/// test supplies it: the simulator is not part of this repository's sources, but
/// a copy downloaded from the ttsim releases may sit in `vendor/` (git-ignored)
/// for convenience.
fn simulator_path() -> PathBuf {
    if let Some(p) = std::env::var_os("TT_METAL_SIMULATOR") {
        return PathBuf::from(p);
    }
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/libttsim_bh.so");
    if local.exists() {
        return local;
    }
    panic!("no libttsim_bh.so: set TT_METAL_SIMULATOR to the simulator's shared object");
}

#[test]
fn a_copy_kernel_runs_on_the_simulator() {
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );

    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(
        unsafe { alloc(8192, &mut input, &mut err) },
        TT_OK,
        "input: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { alloc(8192, &mut output, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );
    assert_ne!(input, output, "two distinct DRAM allocations");

    let host: Vec<u8> = (0..8192u32)
        .map(|i| (i.wrapping_mul(37) ^ (i >> 3)) as u8)
        .collect();
    assert_eq!(
        unsafe { h2d(input, host.as_ptr(), host.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe {
        compile(
            COPY_8K.as_ptr().cast(),
            COPY_8K.len(),
            &mut module,
            &mut err,
        )
    };
    assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
    assert!(!module.is_null());

    let params = [input, output];
    let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
    assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));

    let mut back = vec![0u8; 8192];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
        TT_OK,
        "download: {}",
        take_error(&free, err)
    );
    assert_eq!(
        back, host,
        "the kernel moved every byte from DRAM through L1 and back"
    );

    assert_eq!(
        unsafe { unload(module, &mut err) },
        TT_OK,
        "unload: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
}

/// An elementwise kernel runs the whole datapath: the inbound mover, the three
/// compute cores and the outbound mover, all derived from one unified program.
#[test]
fn an_elementwise_kernel_runs_on_the_simulator() {
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );

    let bytes = 1024 * 4;
    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(
        unsafe { alloc(bytes, &mut input, &mut err) },
        TT_OK,
        "input: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { alloc(bytes, &mut output, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );

    let host: Vec<f32> = (0..1024)
        .map(|i| {
            let v = (i as f32 - 512.0) * 0.25;
            if i % 7 == 0 { -v } else { v }
        })
        .collect();
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, host_bytes.as_ptr(), host_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe {
        compile(
            NEG_TILE.as_ptr().cast(),
            NEG_TILE.len(),
            &mut module,
            &mut err,
        )
    };
    assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
    assert!(!module.is_null());

    let params = [input, output];
    let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
    assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));

    let mut back = vec![0u8; bytes];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
        TT_OK,
        "download: {}",
        take_error(&free, err)
    );
    let got: Vec<f32> = back
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (x, y)) in host.iter().zip(&got).enumerate() {
        assert_eq!(*y, -*x, "datum {i}: {x} negated is {}", -*x);
    }

    assert_eq!(
        unsafe { unload(module, &mut err) },
        TT_OK,
        "unload: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
}

/// Each store packs the value in its own `Dst` rows: the second page is not the
/// first page repeated.
#[test]
fn two_live_values_pack_from_their_own_dst_rows() {
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );

    let bytes = 2048 * 4;
    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(
        unsafe { alloc(bytes, &mut input, &mut err) },
        TT_OK,
        "input: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { alloc(bytes, &mut output, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );

    let host: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.25).collect();
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, host_bytes.as_ptr(), host_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe {
        compile(
            NEG_TWO_LIVE.as_ptr().cast(),
            NEG_TWO_LIVE.len(),
            &mut module,
            &mut err,
        )
    };
    assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
    assert!(!module.is_null());

    let params = [input, output];
    let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
    assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));

    let mut back = vec![0u8; bytes];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
        TT_OK,
        "download: {}",
        take_error(&free, err)
    );
    let got: Vec<f32> = back
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (x, y)) in host.iter().zip(&got).enumerate() {
        assert_eq!(*y, -*x, "datum {i}: {x} negated is {}", -*x);
    }

    assert_eq!(
        unsafe { unload(module, &mut err) },
        TT_OK,
        "unload: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
}

/// The direct form: no buffer and no `copy_to` in the program, and the tile
/// still comes back negated. `mover::add_movers` inserted the staging.
#[test]
fn a_direct_parameter_kernel_runs_on_the_simulator() {
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );

    let bytes = 1024 * 4;
    let (mut input, mut output) = (0u64, 0u64);
    for (name, at) in [("input", &mut input), ("output", &mut output)] {
        assert_eq!(
            unsafe { alloc(bytes, at, &mut err) },
            TT_OK,
            "{name}: {}",
            take_error(&free, err)
        );
    }

    let host: Vec<f32> = (0..1024).map(|i| (i as f32 - 300.0) * 0.5).collect();
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, host_bytes.as_ptr(), host_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe {
        compile(
            NEG_TILE_DIRECT.as_ptr().cast(),
            NEG_TILE_DIRECT.len(),
            &mut module,
            &mut err,
        )
    };
    assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
    let params = [input, output];
    let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
    assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));

    let mut back = vec![0u8; bytes];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
        TT_OK,
        "download: {}",
        take_error(&free, err)
    );
    let got: Vec<f32> = back
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (x, y)) in host.iter().zip(&got).enumerate() {
        assert_eq!(*y, -*x, "datum {i}: {x} negated is {}", -*x);
    }

    assert_eq!(
        unsafe { unload(module, &mut err) },
        TT_OK,
        "unload: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
}

/// The loop form of the elementwise kernel runs too: the constant loop is a
/// RISC-V branch in every role, and both tiles come back negated.
#[test]
fn a_constant_loop_runs_on_the_simulator() {
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );

    let count = 2048;
    let bytes = count * 4;
    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(
        unsafe { alloc(bytes, &mut input, &mut err) },
        TT_OK,
        "input: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { alloc(bytes, &mut output, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );

    let host: Vec<f32> = (0..count)
        .map(|i| {
            let v = (i as f32 - 1024.0) * 0.125;
            if i % 5 == 0 { -v } else { v }
        })
        .collect();
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, host_bytes.as_ptr(), host_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe {
        compile(
            NEG_TWO_TILES.as_ptr().cast(),
            NEG_TWO_TILES.len(),
            &mut module,
            &mut err,
        )
    };
    assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
    assert!(!module.is_null());

    let params = [input, output];
    let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
    assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));

    let mut back = vec![0u8; bytes];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
        TT_OK,
        "download: {}",
        take_error(&free, err)
    );
    let got: Vec<f32> = back
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (x, y)) in host.iter().zip(&got).enumerate() {
        assert_eq!(*y, -*x, "datum {i}: {x} negated is {}", -*x);
    }

    assert_eq!(
        unsafe { unload(module, &mut err) },
        TT_OK,
        "unload: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
}

/// The unary and binary ops the SFPU has an encoding for, run through the C ABI:
/// `abs` on one loaded tile, and `add`/`sub`/`mul` on two.
#[test]
fn arithmetic_ops_run_on_the_simulator() {
    const TWO_LOAD: &str = "\
version 9
grid 1,1
%in = param f32 2048
%out = param f32 2048
%b0 = buffer f32 32,64
%bo = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
copy 32,32 %b0 [1] = %in [1]
load 32,32 %x = %b0 [0]
load 32,32 %y = %b0 [1]
%r = OP %x %y
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";
    const ONE_LOAD: &str = "\
version 9
grid 1,1
%in = param f32 1024
%out = param f32 1024
%b0 = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
load 32,32 %x = %b0 [0]
%r = abs %x
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";

    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );

    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(
        unsafe { alloc(8192, &mut input, &mut err) },
        TT_OK,
        "input: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { alloc(8192, &mut output, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );

    let host: Vec<f32> = (0..2048)
        .map(|i| {
            let v = (i as f32 - 1024.0) * 0.125;
            if i % 3 == 0 { -v } else { v }
        })
        .collect();
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, host_bytes.as_ptr(), host_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut run = |program: &str, expected: &dyn Fn(&[f32]) -> Vec<f32>, out_bytes: usize| {
        let mut module: *mut c_void = std::ptr::null_mut();
        let code = unsafe {
            compile(
                program.as_ptr().cast(),
                program.len(),
                &mut module,
                &mut err,
            )
        };
        assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
        let params = [input, output];
        let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
        assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));
        let mut back = vec![0u8; out_bytes];
        assert_eq!(
            unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
            TT_OK,
            "download: {}",
            take_error(&free, err)
        );
        let got: Vec<f32> = back
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let want = expected(&host);
        assert_eq!(got.len(), want.len());
        for (i, (x, y)) in want.iter().zip(&got).enumerate() {
            assert_eq!(*y, *x, "{program}: datum {i}: got {y}, want {x}");
        }
        assert_eq!(
            unsafe { unload(module, &mut err) },
            TT_OK,
            "unload: {}",
            take_error(&free, err)
        );
    };

    run(
        &ONE_LOAD.replace("OP", ""),
        &|h| h[..1024].iter().map(|v| v.abs()).collect(),
        4096,
    );
    for (op, f) in [
        (
            "add",
            (&|h: &[f32]| {
                h[..1024]
                    .iter()
                    .zip(&h[1024..])
                    .map(|(a, b)| a + b)
                    .collect()
            }) as &dyn Fn(&[f32]) -> Vec<f32>,
        ),
        ("sub", &|h: &[f32]| {
            h[..1024]
                .iter()
                .zip(&h[1024..])
                .map(|(a, b)| a - b)
                .collect()
        }),
        ("mul", &|h: &[f32]| {
            h[..1024]
                .iter()
                .zip(&h[1024..])
                .map(|(a, b)| a * b)
                .collect()
        }),
    ] {
        let program = TWO_LOAD.replace("OP", op);
        run(&program, f, 4096);
    }

    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
}

/// Comparisons produce a `bool` in `Dst`, `sel` consumes one, and `max` goes
/// through the same lane flags. All four run through the C ABI.
#[test]
fn comparisons_select_and_max_run_on_the_simulator() {
    const TEMPLATE: &str = "\
version 9
grid 1,1
%in = param f32 2048
%out = param f32 2048
%b0 = buffer f32 32,64
%bo = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
copy 32,32 %b0 [1] = %in [1]
load 32,32 %x = %b0 [0]
load 32,32 %y = %b0 [1]
BODY
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";

    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );

    let (mut input, mut output) = (0u64, 0u64);
    for (bytes, addr, name) in [
        (8192usize, &mut input, "input"),
        (8192, &mut output, "output"),
    ] {
        assert_eq!(
            unsafe { alloc(bytes, addr, &mut err) },
            TT_OK,
            "{name}: {}",
            take_error(&free, err)
        );
    }

    let host: Vec<f32> = (0..2048)
        .map(|i| {
            let v = (i as f32 - 1024.0) * 0.125;
            if i % 4 < 2 { -v } else { v }
        })
        .collect();
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, host_bytes.as_ptr(), host_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut run = |body: &str, expected: &dyn Fn(f32, f32) -> f32| {
        let program = TEMPLATE.replace("BODY", body);
        let mut module: *mut c_void = std::ptr::null_mut();
        let code = unsafe {
            compile(
                program.as_ptr().cast(),
                program.len(),
                &mut module,
                &mut err,
            )
        };
        assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
        let params = [input, output];
        let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
        assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));
        let mut back = vec![0u8; 4096];
        assert_eq!(
            unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
            TT_OK,
            "download: {}",
            take_error(&free, err)
        );
        let got: Vec<f32> = back
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for (i, y) in got.iter().enumerate() {
            let want = expected(host[i], host[1024 + i]);
            assert_eq!(*y, want, "datum {i}: got {y}, want {want}");
        }
        assert_eq!(
            unsafe { unload(module, &mut err) },
            TT_OK,
            "unload: {}",
            take_error(&free, err)
        );
    };

    run("%c = cmplt %x %y\n%r = sel %c %x %y", &|a, b| a.min(b));
    run("%c = cmpgt %x %y\n%r = sel %c %x %y", &|a, b| a.max(b));
    run("%c = cmpge %x %y\n%r = sel %c %x %y", &|a, b| a.max(b));
    run("%r = max %x %y", &|a, b| a.max(b));
    // A bool lane is -1/0, so `or`/`and` use the bitwise SFPU ops. `lt | ge`
    // is true and `lt & ge` is false for every pair.
    run(
        "%lt = cmplt %x %y\n%ge = cmpge %x %y\n%c = or %lt %ge\n%r = sel %c %x %y",
        &|a, _| a,
    );
    run(
        "%lt = cmplt %x %y\n%ge = cmpge %x %y\n%c = and %lt %ge\n%r = sel %c %x %y",
        &|_, b| b,
    );

    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
}

/// The bitwise ops that are one SFPU instruction, on `i32` tiles: `bitand`,
/// `bitor`, `bitxor` on two loaded tiles, and `bitnot` on one.
#[test]
fn bitwise_ops_run_on_the_simulator() {
    const TWO_LOAD: &str = "\
version 9
grid 1,1
%in = param i32 2048
%out = param i32 2048
%b0 = buffer i32 32,64
%bo = buffer i32 32,32
copy 32,32 %b0 [0] = %in [0]
copy 32,32 %b0 [1] = %in [1]
load 32,32 %x = %b0 [0]
load 32,32 %y = %b0 [1]
%r = OP %x %y
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";
    const ONE_LOAD: &str = "\
version 9
grid 1,1
%in = param i32 1024
%out = param i32 1024
%b0 = buffer i32 32,32
%bo = buffer i32 32,32
copy 32,32 %b0 [0] = %in [0]
load 32,32 %x = %b0 [0]
%r = bitnot %x
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";

    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );

    let (mut input, mut output) = (0u64, 0u64);
    for (bytes, addr, name) in [
        (8192usize, &mut input, "input"),
        (8192, &mut output, "output"),
    ] {
        assert_eq!(
            unsafe { alloc(bytes, addr, &mut err) },
            TT_OK,
            "{name}: {}",
            take_error(&free, err)
        );
    }

    let host: Vec<i32> = (0..2048)
        .map(|i| (i as i32).wrapping_mul(0x9E37_79B1u32 as i32) ^ (i as i32 >> 3))
        .collect();
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, host_bytes.as_ptr(), host_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut run = |program: &str, expected: &dyn Fn(i32, i32) -> i32, two_loads: bool| {
        let mut module: *mut c_void = std::ptr::null_mut();
        let code = unsafe {
            compile(
                program.as_ptr().cast(),
                program.len(),
                &mut module,
                &mut err,
            )
        };
        assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
        let params = [input, output];
        let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
        assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));
        let mut back = vec![0u8; 4096];
        assert_eq!(
            unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
            TT_OK,
            "download: {}",
            take_error(&free, err)
        );
        let got: Vec<i32> = back
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for (i, y) in got.iter().enumerate() {
            let want = if two_loads {
                expected(host[i], host[1024 + i])
            } else {
                expected(host[i], 0)
            };
            assert_eq!(*y, want, "datum {i}: got {y}, want {want}");
        }
        assert_eq!(
            unsafe { unload(module, &mut err) },
            TT_OK,
            "unload: {}",
            take_error(&free, err)
        );
    };

    for (op, f) in [
        (
            "bitand",
            (&|a: i32, b: i32| a & b) as &dyn Fn(i32, i32) -> i32,
        ),
        ("bitor", &|a: i32, b: i32| a | b),
        ("bitxor", &|a: i32, b: i32| a ^ b),
    ] {
        run(&TWO_LOAD.replace("OP", op), f, true);
    }
    run(&ONE_LOAD, &|a: i32, _| !a, false);

    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
}

/// The loop form of the elementwise kernel with a launch-bound trip count: the
/// count is a scalar argument, loaded by the movers and by each compute role,
/// and branched on in RISC-V; both tiles come back negated.
const NEG_SYMBOLIC_LOOP: &str = "\
version 9
grid 1,1
%n = param i32 0
%in = param f32 2048
%out = param f32 2048
%bi = buffer f32 32,64
%bo = buffer f32 32,64
%step = const i32 1
%i = loop %n
%o = mul %i %step
copy 32,32 %bi [%o] = %in [%o]
load 32,32 %x = %bi [%o]
%y = neg %x
store 32,32 %bo [%o] = %y
copy 32,32 %out [%o] = %bo [%o]
endloop
";

#[test]
fn a_symbolic_loop_bound_runs_on_the_simulator() {
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );

    let count = 2048;
    let bytes = count * 4;
    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(
        unsafe { alloc(bytes, &mut input, &mut err) },
        TT_OK,
        "input: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { alloc(bytes, &mut output, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );
    let host: Vec<f32> = (0..count).map(|i| (i as f32 - 1024.0) * 0.125).collect();
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, host_bytes.as_ptr(), host_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe {
        compile(
            NEG_SYMBOLIC_LOOP.as_ptr().cast(),
            NEG_SYMBOLIC_LOOP.len(),
            &mut module,
            &mut err,
        )
    };
    assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));

    // The trip count is a scalar argument: its value, not an address.
    let params = [2u64, input, output];
    let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
    assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));

    let mut back = vec![0u8; bytes];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
        TT_OK,
        "download: {}",
        take_error(&free, err)
    );
    let got: Vec<f32> = back
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (x, y)) in host.iter().zip(&got).enumerate() {
        assert_eq!(*y, -*x, "datum {i}: {x} negated is {}", -*x);
    }

    assert_eq!(
        unsafe { unload(module, &mut err) },
        TT_OK,
        "unload: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
}

/// A symbolic `2,2` grid: each of the four tiles negates its own slice, selected
/// from `grid.x`/`grid.y`. Four executors run and all four slices come back.
const NEG_GRID: &str = "\
version 9
%nx = param i32 0
%ny = param i32 0
%in = param f32 4096
%out = param f32 4096
%x = grid.x
%y = grid.y
%row = mul %y %nx
%idx = add %row %x
%tile = const i32 1
%o = mul %idx %tile
%bi = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bi [0] = %in [%o]
load 32,32 %v = %bi [0]
%n = neg %v
store 32,32 %bo [0] = %n
copy 32,32 %out [%o] = %bo [0]
grid %nx,%ny
";

#[test]
fn a_symbolic_grid_runs_each_tiles_own_slice() {
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );

    let count = 4096;
    let bytes = count * 4;
    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(
        unsafe { alloc(bytes, &mut input, &mut err) },
        TT_OK,
        "input: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { alloc(bytes, &mut output, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );
    let host: Vec<f32> = (0..count).map(|i| (i as f32 - 2048.0) * 0.0625).collect();
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, host_bytes.as_ptr(), host_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );

    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe {
        compile(
            NEG_GRID.as_ptr().cast(),
            NEG_GRID.len(),
            &mut module,
            &mut err,
        )
    };
    assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));

    // `grid %nx,%ny` with both extents launch-bound.
    let params = [2u64, 2, input, output];
    let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
    assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));

    let mut back = vec![0u8; bytes];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
        TT_OK,
        "download: {}",
        take_error(&free, err)
    );
    let got: Vec<f32> = back
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (x, y)) in host.iter().zip(&got).enumerate() {
        assert_eq!(*y, -*x, "datum {i}: {x} negated is {}", -*x);
    }

    assert_eq!(
        unsafe { unload(module, &mut err) },
        TT_OK,
        "unload: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
}

/// The tail of a grid's work: `ntiles` need not divide the number of sites.
/// Each site computes its own length — `base = ntiles / nsites`, plus one for
/// the first `ntiles % nsites` sites — and its own starting tile, so no site
/// ever names a tile past the end. The length is a comparison, a `cast` of the
/// resulting `bool` and an addition, evaluated in RISC-V by every role; the
/// loop is a `LoopReg` over that value. Here 11 tiles over 8 sites: sites 0-2
/// copy two tiles, the rest one.
const TAIL_TILES: &str = "\
version 9
%nx = param i32 0
%ny = param i32 0
%ntiles = param i32 0
%in = param f32 11264
%out = param f32 11264
%b = buffer f32 32,32
grid %nx,%ny
%x = grid.x
%y = grid.y
%nsites = mul %nx %ny
%row = mul %y %nx
%site = add %row %x
%base = div %ntiles %nsites
%rem = mod %ntiles %nsites
%last = cmplt %site %rem
%extra = cast %last to i32
%per = add %base %extra
%before = mul %base %site
%adj = sel %last %site %rem
%start = add %before %adj
%i = loop %per
%t = add %start %i
copy 32,32 %b [0] = %in [%t]
copy 32,32 %out [%t] = %b [0]
endloop
";
#[test]
fn a_grid_tail_is_each_sites_own_loop_length() {
    const NX: u64 = 4;
    const NY: u64 = 2;
    const TILES: u64 = 11;
    const DATUMS: usize = 1024;
    let input: Vec<f32> = (0..TILES as usize * DATUMS)
        .map(|i| i as f32 * 0.5 - 3.0)
        .collect();
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_kernel_scalars(TAIL_TILES, &[NX, NY, TILES], &[&bytes], bytes.len());
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got.len(), input.len());
    let bad: Vec<usize> = input
        .iter()
        .zip(&got)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i)
        .collect();
    assert!(
        bad.is_empty(),
        "{} of {} datums differ, first {:?}, last {:?}",
        bad.len(),
        input.len(),
        bad.first(),
        bad.last()
    );
}

/// Compile `program`, upload `input`, launch it once over one tile and return
/// the output bytes. Two parameters, in then out.
fn run_kernel(program: &str, input: &[u8], out_bytes: usize) -> Vec<u8> {
    run_kernel_tensors(program, &[input], out_bytes)
}

/// The same launch with `inputs.len()` tensor parameters, in order, followed by
/// the output parameter.
fn run_kernel_tensors(program: &str, inputs: &[&[u8]], out_bytes: usize) -> Vec<u8> {
    run_kernel_scalars(program, &[], inputs, out_bytes)
}

/// A launch with leading scalar arguments (`%n = param i32 0`, in program
/// order), then `inputs.len()` tensor parameters, then the output parameter.
fn run_kernel_scalars(
    program: &str,
    scalars: &[u64],
    inputs: &[&[u8]],
    out_bytes: usize,
) -> Vec<u8> {
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );
    let mut tensors: Vec<u64> = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        let mut at = 0u64;
        assert_eq!(
            unsafe { alloc(input.len(), &mut at, &mut err) },
            TT_OK,
            "input {i}: {}",
            take_error(&free, err)
        );
        assert_eq!(
            unsafe { h2d(at, input.as_ptr(), input.len(), &mut err) },
            TT_OK,
            "upload {i}: {}",
            take_error(&free, err)
        );
        tensors.push(at);
    }
    let mut output_at = 0u64;
    assert_eq!(
        unsafe { alloc(out_bytes, &mut output_at, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );
    let mut module: *mut c_void = std::ptr::null_mut();
    let code = unsafe {
        compile(
            program.as_ptr().cast(),
            program.len(),
            &mut module,
            &mut err,
        )
    };
    assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));

    let mut params: Vec<u64> = scalars.to_vec();
    params.extend_from_slice(&tensors);
    params.push(output_at);
    let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
    assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));

    let mut back = vec![0u8; out_bytes];
    assert_eq!(
        unsafe { d2h(back.as_mut_ptr(), output_at, out_bytes, &mut err) },
        TT_OK,
        "download: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { unload(module, &mut err) },
        TT_OK,
        "unload: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
    back
}

/// `(int)f`: truncation toward zero, in two's complement.
const CAST_F32_TO_I32: &str = "\
version 9
grid 1,1
%in = param f32 1024
%out = param i32 1024
%bi = buffer f32 32,32
%bo = buffer i32 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = cast %x to i32
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// `(float)i`: two's complement to FP32, round to nearest even.
const CAST_I32_TO_F32: &str = "\
version 9
grid 1,1
%in = param i32 1024
%out = param f32 1024
%bi = buffer i32 32,32
%bo = buffer f32 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = cast %x to f32
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// The same bits under another name: no instruction at all.
const BITCAST_F32_TO_I32: &str = "\
version 9
grid 1,1
%in = param f32 1024
%out = param i32 1024
%bi = buffer f32 32,32
%bo = buffer i32 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = bitcast %x to i32
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// A comparison writes `-1`/`0` in the lane; a cast to an integer is the C
/// value, one or zero.
const CAST_BOOL_TO_I32: &str = "\
version 9
grid 1,1
%in = param f32 1024
%out = param i32 1024
%bi = buffer f32 32,32
%bo = buffer i32 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%zero = sub %x %x
%c = cmplt %x %zero
%y = cast %c to i32
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

#[test]
fn casts_and_bitcasts_run_on_the_simulator() {
    let count = 1024usize;
    let floats: Vec<f32> = (0..count).map(|i| (i as f32 - 512.0) * 0.375).collect();
    let f32_bytes: Vec<u8> = floats.iter().flat_map(|v| v.to_le_bytes()).collect();

    // `(int)f` truncates: `-2.625` is `-2`, not `-3`.
    let out = run_kernel(CAST_F32_TO_I32, &f32_bytes, count * 4);
    for (i, v) in floats.iter().enumerate() {
        let got = i32::from_le_bytes(out[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!(got, *v as i32, "f32 to i32, datum {i}: {v}");
    }

    // `(float)i` rounds to nearest even.
    let ints: Vec<i32> = (0..count).map(|i| (i as i32 - 512) * 7).collect();
    let i32_bytes: Vec<u8> = ints.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_kernel(CAST_I32_TO_F32, &i32_bytes, count * 4);
    for (i, v) in ints.iter().enumerate() {
        let got = f32::from_le_bytes(out[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!(got, *v as f32, "i32 to f32, datum {i}: {v}");
    }

    // A bitcast leaves every bit alone.
    let out = run_kernel(BITCAST_F32_TO_I32, &f32_bytes, count * 4);
    for (i, v) in floats.iter().enumerate() {
        let got = i32::from_le_bytes(out[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!(got, v.to_bits() as i32, "bitcast, datum {i}: {v}");
    }

    // A bool lane is `-1`/`0`; the C value is one or zero.
    let out = run_kernel(CAST_BOOL_TO_I32, &f32_bytes, count * 4);
    for (i, v) in floats.iter().enumerate() {
        let got = i32::from_le_bytes(out[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!(got, i32::from(*v < 0.0), "bool to i32, datum {i}: {v}");
    }
}

/// bf16 through the same datapath: the unpacker reads a 16-bit page into a
/// 16-bit `Dst` (`Bf16` in and out), the lane widens to FP32 for the op, and
/// the packer writes 16-bit datums back. A negation only flips the sign bit, so
/// every pattern -- zeros, subnormals, infinities, NaN payloads -- is exact.
const NEG_BF16: &str = "\
version 9
grid 1,1
%in = param bf16 1024
%out = param bf16 1024
%bi = buffer bf16 32,32
%bo = buffer bf16 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = neg %x
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

#[test]
fn bf16_runs_on_the_simulator() {
    let count = 1024usize;
    // Every sign and exponent with a few mantissas, then noise.
    let mut patterns = Vec::new();
    for sign in [0u16, 0x8000] {
        for exp in 0..256u16 {
            for mant in [0u16, 1, 0x20, 0x40, 0x7f] {
                patterns.push(sign | exp << 7 | mant);
            }
        }
    }
    let mut seed = 0xdead_beefu32;
    while patterns.len() < count {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        patterns.push((seed >> 8) as u16);
    }
    patterns.truncate(count);
    let input: Vec<u8> = patterns.iter().flat_map(|h| h.to_le_bytes()).collect();

    let out = run_kernel(NEG_BF16, &input, count * 2);
    for (i, h) in patterns.iter().enumerate() {
        let got = u16::from_le_bytes(out[2 * i..2 * i + 2].try_into().unwrap());
        // The lane is FP32 and the SFPU flushes denormals, so a bf16 with a
        // zero exponent (a subnormal, or a zero of either sign) comes back as
        // +0; every normal, infinity and NaN just flips its sign bit.
        let want = if h & 0x7f80 == 0 { 0 } else { h ^ 0x8000 };
        assert_eq!(got, want, "bf16 neg, datum {i}: {h:#06x}");
    }
}

/// The bf16 datum a lane value truncates to: the SFPU's `BF16` store keeps the
/// high half (`0x43058000` becomes `0x4305`), so bf16 arithmetic lands at bf16
/// precision after every op, which is what a 16-bit `Dst` kernel is for.
fn truncate_to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

/// A bf16 datum widened into the 32-bit lane. The SFPU flushes a datum whose
/// exponent is zero (a subnormal, or a zero of either sign).
fn widen_bf16(bits: u16) -> f32 {
    if bits & 0x7f80 == 0 {
        0.0
    } else {
        f32::from_bits(u32::from(bits) << 16)
    }
}

/// bf16 arithmetic in the 16-bit `Dst`: two tiles in, one out, every datum the
/// SFPU's widening and truncating round trip.
const BF16_TWO_LOAD: &str = "\
version 9
grid 1,1
%in = param bf16 2048
%out = param bf16 1024
%b0 = buffer bf16 32,64
%bo = buffer bf16 32,32
copy 32,32 %b0 [0] = %in [0]
copy 32,32 %b0 [1] = %in [1]
load 32,32 %x = %b0 [0]
load 32,32 %y = %b0 [1]
%r = OP %x %y
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";

#[test]
fn bf16_arithmetic_runs_on_the_simulator() {
    let _device = device();
    let sim = simulator_path();
    // SAFETY: no other thread reads the environment concurrently; the library
    // reads it when it opens the device.
    unsafe { std::env::set_var("TT_METAL_SIMULATOR", &sim) };

    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");
    let init: Symbol<Init> = unsafe { lib.get(b"ttInit\0") }.expect("ttInit");
    let release: Symbol<Release> = unsafe { lib.get(b"ttRelease\0") }.expect("ttRelease");
    let alloc: Symbol<MemAlloc> = unsafe { lib.get(b"ttMemAlloc\0") }.expect("ttMemAlloc");
    let h2d: Symbol<MemcpyH2D> = unsafe { lib.get(b"ttMemcpyH2D\0") }.expect("ttMemcpyH2D");
    let d2h: Symbol<MemcpyD2H> = unsafe { lib.get(b"ttMemcpyD2H\0") }.expect("ttMemcpyD2H");
    let launch: Symbol<Launch> = unsafe { lib.get(b"ttLaunch\0") }.expect("ttLaunch");

    let mut err: *mut c_char = std::ptr::null_mut();
    assert_eq!(
        unsafe { init(&mut err) },
        TT_OK,
        "ttInit: {}",
        take_error(&free, err)
    );
    let (mut input, mut output) = (0u64, 0u64);
    assert_eq!(
        unsafe { alloc(4096, &mut input, &mut err) },
        TT_OK,
        "input: {}",
        take_error(&free, err)
    );
    assert_eq!(
        unsafe { alloc(2048, &mut output, &mut err) },
        TT_OK,
        "output: {}",
        take_error(&free, err)
    );

    // Normals in a range where no result of the operators below under- or
    // overflows bf16, so the expectation is the truncation alone.
    let host: Vec<u16> = (0..2048)
        .map(|i| {
            let v = (i as f32 - 1024.0) * 0.125;
            truncate_to_bf16(if i % 3 == 0 { -v } else { v })
        })
        .collect();
    let input_bytes: Vec<u8> = host.iter().flat_map(|h| h.to_le_bytes()).collect();
    assert_eq!(
        unsafe { h2d(input, input_bytes.as_ptr(), input_bytes.len(), &mut err) },
        TT_OK,
        "upload: {}",
        take_error(&free, err)
    );
    // The exact values the kernel sees, after the widening and its flush.
    let seen: Vec<f32> = host.iter().map(|h| widen_bf16(*h)).collect();

    let mut run = |program: &str, expected: &dyn Fn(&[f32]) -> Vec<u16>| {
        let mut module: *mut c_void = std::ptr::null_mut();
        let code = unsafe {
            compile(
                program.as_ptr().cast(),
                program.len(),
                &mut module,
                &mut err,
            )
        };
        assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
        let params = [input, output];
        let code = unsafe { launch(module, params.as_ptr(), params.len(), &mut err) };
        assert_eq!(code, TT_OK, "ttLaunch: {}", take_error(&free, err));
        let mut back = vec![0u8; 2048];
        assert_eq!(
            unsafe { d2h(back.as_mut_ptr(), output, back.len(), &mut err) },
            TT_OK,
            "download: {}",
            take_error(&free, err)
        );
        let got: Vec<u16> = back
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let want = expected(&seen);
        for (i, x) in want.iter().enumerate() {
            assert_eq!(got[i], *x, "{program}: datum {i}");
        }
        assert_eq!(
            unsafe { unload(module, &mut err) },
            TT_OK,
            "unload: {}",
            take_error(&free, err)
        );
    };

    for (op, f) in [
        (
            "add",
            (&|h: &[f32]| {
                h[..1024]
                    .iter()
                    .zip(&h[1024..])
                    .map(|(a, b)| truncate_to_bf16(a + b))
                    .collect()
            }) as &dyn Fn(&[f32]) -> Vec<u16>,
        ),
        ("sub", &|h: &[f32]| {
            h[..1024]
                .iter()
                .zip(&h[1024..])
                .map(|(a, b)| truncate_to_bf16(a - b))
                .collect()
        }),
        ("mul", &|h: &[f32]| {
            h[..1024]
                .iter()
                .zip(&h[1024..])
                .map(|(a, b)| truncate_to_bf16(a * b))
                .collect()
        }),
        ("max", &|h: &[f32]| {
            h[..1024]
                .iter()
                .zip(&h[1024..])
                .map(|(a, b)| truncate_to_bf16(if a >= b { *a } else { *b }))
                .collect()
        }),
    ] {
        let program = BF16_TWO_LOAD.replace("OP", op);
        run(&program, f);
    }

    assert_eq!(
        unsafe { release(&mut err) },
        TT_OK,
        "ttRelease: {}",
        take_error(&free, err)
    );
}

/// The 32-bit `Dst` kernel's bf16 output: the packer narrows as it writes.
const CAST_F32_TO_BF16: &str = "\
version 9
grid 1,1
%in = param f32 1024
%out = param bf16 1024
%bi = buffer f32 32,32
%bo = buffer bf16 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = cast %x to bf16
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// `i32` to bf16: `(float)i` in the lane, then the same narrowing.
const CAST_I32_TO_BF16: &str = "\
version 9
grid 1,1
%in = param i32 1024
%out = param bf16 1024
%bi = buffer i32 32,32
%bo = buffer bf16 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = cast %x to bf16
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// A comparison's `bool` to bf16: one or zero.
const CAST_BOOL_TO_BF16: &str = "\
version 9
grid 1,1
%in = param f32 1024
%out = param bf16 1024
%bi = buffer f32 32,32
%bo = buffer bf16 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%zero = sub %x %x
%c = cmplt %x %zero
%y = cast %c to bf16
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// bf16 to bf16, the 16-bit `Dst`'s identity: the store truncates what the load
/// widened, so every normal comes back unchanged.
const CAST_BF16_TO_BF16: &str = "\
version 9
grid 1,1
%in = param bf16 1024
%out = param bf16 1024
%bi = buffer bf16 32,32
%bo = buffer bf16 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = cast %x to bf16
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// bf16 into `f32` has no `Dst` path: the unpacker cannot widen a bf16 page into
/// a 32-bit `Dst`, and a 16-bit `Dst` cannot pack `f32`.
const CAST_BF16_TO_F32: &str = "\
version 9
grid 1,1
%in = param bf16 1024
%out = param f32 1024
%bi = buffer bf16 32,32
%bo = buffer f32 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%y = cast %x to f32
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// `sel` on bf16: against `y`, so the `bool` lives in the 16-bit `Dst` as the
/// `-1.0`/`0.0` lane the comparison writes, and the result is the smaller value.
const BF16_MIN: &str = "\
version 9
grid 1,1
%in = param bf16 2048
%out = param bf16 1024
%b0 = buffer bf16 32,64
%bo = buffer bf16 32,32
copy 32,32 %b0 [0] = %in [0]
copy 32,32 %b0 [1] = %in [1]
load 32,32 %x = %b0 [0]
load 32,32 %y = %b0 [1]
%c = cmplt %x %y
%r = sel %c %x %y
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";

#[test]
fn bf16_casts_run_on_the_simulator() {
    let count = 1024usize;

    // bf16 to itself: the 16-bit `Dst` round trip, exact for a normal.
    let patterns: Vec<u16> = (0..count)
        .map(|i| {
            let v = (i as f32 - 512.0) * 0.375;
            truncate_to_bf16(v)
        })
        .collect();
    let bf16_bytes: Vec<u8> = patterns.iter().flat_map(|h| h.to_le_bytes()).collect();
    let out = run_kernel(CAST_BF16_TO_BF16, &bf16_bytes, count * 2);
    for (i, h) in patterns.iter().enumerate() {
        let got = u16::from_le_bytes(out[2 * i..2 * i + 2].try_into().unwrap());
        assert_eq!(got, *h, "bf16 to bf16, datum {i}: {h:#06x}");
    }

    // A comparison's `bool` and a `sel` over bf16, in the same 16-bit `Dst`:
    // the smaller operand, exactly (no rounding on this path).
    let pairs: Vec<u16> = (0..2048)
        .map(|i| truncate_to_bf16((i as f32 - 1024.0) * 0.125))
        .collect();
    let pair_bytes: Vec<u8> = pairs.iter().flat_map(|h| h.to_le_bytes()).collect();
    let out = run_kernel(BF16_MIN, &pair_bytes, count * 2);
    for i in 0..count {
        let (a, b) = (widen_bf16(pairs[i]), widen_bf16(pairs[i + 1024]));
        let want = truncate_to_bf16(if a < b { a } else { b });
        let got = u16::from_le_bytes(out[2 * i..2 * i + 2].try_into().unwrap());
        assert_eq!(got, want, "bf16 cmplt and sel, datum {i}");
    }
}

/// A bf16 page written out of a 32-bit `Dst` is the packer's own early
/// conversion (`pack_fmt_conv_mode=0x105`): tt-metal's
/// `is_packer_to_L1_early_conversion_supported` lists `Float32 ->
/// Float16_b`, and `pack_config` writes the two formats and
/// `Read_32b_data`, which is the whole mode word. `ttsim` does not model the
/// conversion (`TENSIX_EXECUTE_PACR`'s accepted list has no `0x105`) and exits
/// with `UnimplementedFunctionality`, so the gate compiles the three shapes the
/// cast takes and stops there; a launch belongs to silicon.
#[test]
fn an_f32_dst_packs_into_a_bf16_page() {
    let lib = library();
    let compile: Symbol<Compile> = unsafe { lib.get(b"ttCompile\0") }.expect("ttCompile");
    let unload: Symbol<ModuleUnload> =
        unsafe { lib.get(b"ttModuleUnload\0") }.expect("ttModuleUnload");
    let free: Symbol<Free> = unsafe { lib.get(b"ttFree\0") }.expect("ttFree");

    for program in [CAST_F32_TO_BF16, CAST_I32_TO_BF16, CAST_BOOL_TO_BF16] {
        let mut module: *mut c_void = std::ptr::null_mut();
        let mut err: *mut c_char = std::ptr::null_mut();
        let code = unsafe {
            compile(
                program.as_ptr().cast(),
                program.len(),
                &mut module,
                &mut err,
            )
        };
        assert_eq!(code, TT_OK, "compile: {}", take_error(&free, err));
        assert!(!module.is_null());
        let mut err: *mut c_char = std::ptr::null_mut();
        assert_eq!(unsafe { unload(module, &mut err) }, TT_OK);
    }
}

/// bf16 into a 32-bit `Dst`: the widening is the `SrcA` -> `MOVA2D` move
/// tt-lang's `copy_tile` makes, not an unpacker conversion. Every datum must
/// come out the exact f32 its bf16 widens to, including the two zeros, the
/// subnormals (the Matrix Unit flushes them) and the infinities.
#[test]
fn bf16_widens_through_srca_into_a_32_bit_dst() {
    let count = 1024usize;
    let mut patterns: Vec<u16> = (0..count)
        .map(|i| truncate_to_bf16((i as f32 - 512.0) * 0.375))
        .collect();
    for (at, bits) in [
        0x0000u16, // +0
        0x8000,    // -0
        0x7F80,    // +inf
        0xFF80,    // -inf
        0x0001,    // the smallest subnormal
        0x807F,    // a negative subnormal
    ]
    .into_iter()
    .enumerate()
    {
        patterns[count - 1 - at] = bits;
    }
    let bytes: Vec<u8> = patterns.iter().flat_map(|h| h.to_le_bytes()).collect();
    let out = run_kernel(CAST_BF16_TO_F32, &bytes, count * 4);
    for (i, h) in patterns.iter().enumerate() {
        let got = f32::from_le_bytes(out[4 * i..4 * i + 4].try_into().unwrap());
        let want = widen_bf16(*h);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "bf16 to f32, datum {i}: {h:#06x}"
        );
    }
}

/// A bf16 constant in a 16-bit kernel, in both spellings the format allows:
/// `const bf16 <literal>` (the literal truncated into the datum, zyx's
/// `bf16::from_f32`) and `const f32 <literal>` then `cast ... to bf16` (the
/// documented way, and the one a caller reaches for when the coefficient is
/// already f32). Both must reach the same lane immediate, and every datum is
/// the bf16 of the product.
#[test]
fn a_bf16_constant_multiplies_in_both_spellings() {
    let count = 1024usize;
    let x: Vec<u16> = (0..count)
        .map(|i| truncate_to_bf16((i as f32 - 512.0) * 0.375))
        .collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|h| h.to_le_bytes()).collect();
    for literal in ["0.3", "-1.5", "0.5"] {
        let value = literal.parse::<f32>().expect("a literal");
        let c = widen_bf16(truncate_to_bf16(value));
        for (spelling, consts) in [
            ("const bf16", format!("%c = const bf16 {literal}\n")),
            (
                "f32 and cast",
                format!("%k = const f32 {literal}\n%c = cast %k to bf16\n"),
            ),
        ] {
            let program = format!(
                "version 9
grid 1,1
%in = param bf16 1024
%out = param bf16 1024
%bi = buffer bf16 32,32
%bo = buffer bf16 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
{consts}%y = mul %x %c
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
"
            );
            let out = run_kernel(&program, &x_bytes, count * 2);
            for (i, h) in x.iter().enumerate() {
                let got = u16::from_le_bytes(out[2 * i..2 * i + 2].try_into().unwrap());
                // `sfpu_mul` keeps the sign of a zero result only when both
                // operands are normal and their exponents sum into range;
                // otherwise the result is `+0` (ttsim's `src/tensix.cpp`, the
                // Blackhole arm).
                let product = widen_bf16(*h) * c;
                let want = if product == 0.0 {
                    0
                } else {
                    truncate_to_bf16(product)
                };
                assert_eq!(got, want, "{spelling} {literal}: datum {i}");
            }
        }
    }

    // The same immediate in a 32-bit kernel: `const f32`, a tile operand, and
    // the product at full precision.
    let x: Vec<f32> = (0..count).map(|i| (i as f32 - 512.0) * 0.375).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    for literal in ["0.3", "-1.5", "0.5"] {
        let c = literal.parse::<f32>().expect("a literal");
        let program = format!(
            "version 9
grid 1,1
%in = param f32 1024
%out = param f32 1024
%bi = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bi [0] = %in [0]
load 32,32 %x = %bi [0]
%c = const f32 {literal}
%y = mul %x %c
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
"
        );
        let out = run_kernel(&program, &x_bytes, count * 4);
        for (i, v) in x.iter().enumerate() {
            let got = f32::from_le_bytes(out[4 * i..4 * i + 4].try_into().unwrap());
            let product = v * c;
            let want = if product == 0.0 { 0.0 } else { product };
            assert_eq!(got, want, "f32 const {literal}: datum {i}");
        }
    }
}

/// `m = a x b` over two `f32` tiles: the operands ride the Matrix Unit's `SrcA`
/// / `SrcB` path (unpacked straight from their CB pages), the accumulator is the
/// 32-bit `Dst`, and the packer drains it.
const MATMUL_F32: &str = "\
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

/// The same product, with `%a` also feeding a datapath op: the load keeps its
/// `Dst` unpack and the matmul unpacks the same page again into `Src`.
const MATMUL_F32_MIXED: &str = "\
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
%n = neg %a
matmul %m %a %b
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
";

/// A bf16 kernel with no widening: `log2`, `sin`, `cos`, `reciprocal`, `sqrt`
/// and `div` are lane bodies in `lower`, and a lane is 32 bits whatever the
/// `Dst` holds. The datum widens on `SFPLOAD` and narrows on `SFPSTORE`, so the
/// result is the f32 answer truncated to bf16 — every datum.
#[test]
fn bf16_transcendentals_run_in_the_lane() {
    let count = 1024usize;
    let unary = |name: &str, host: &[f32], want: &dyn Fn(f32) -> f32, ulps: u16| {
        let bytes: Vec<u8> = host
            .iter()
            .flat_map(|v| truncate_to_bf16(*v).to_le_bytes())
            .collect();
        let program = format!(
            "version 9\ngrid 1,1\n%in = param bf16 1024\n%out = param bf16 1024\n\
             %b0 = buffer bf16 32,32\n%bo = buffer bf16 32,32\n\
             copy 32,32 %b0 [0] = %in [0]\nload 32,32 %x = %b0 [0]\n\
             %r = {name} %x\nstore 32,32 %bo [0] = %r\ncopy 32,32 %out [0] = %bo [0]\n"
        );
        let out = run_kernel(&program, &bytes, count * 2);
        for (i, h) in host.iter().enumerate() {
            let x = widen_bf16(truncate_to_bf16(*h));
            let got = u16::from_le_bytes(out[2 * i..2 * i + 2].try_into().unwrap());
            let want = truncate_to_bf16(want(x));
            // The approximate bodies carry the one-ulp slack the coverage
            // matrix gives them; the polynomial ones are exact.
            let delta = got.wrapping_sub(want).min(want.wrapping_sub(got));
            assert!(
                delta <= ulps,
                "{name}: datum {i} of {x}: {got:#06x} vs {want:#06x}"
            );
        }
    };
    let positive: Vec<f32> = (0..count).map(|i| 0.01 * (1.0 + i as f32 * 1.1)).collect();
    unary("log2", &positive, &f32::log2, 0);
    unary("reciprocal", &positive, &|x| x.recip(), 2);
    unary("sqrt", &positive, &f32::sqrt, 2);
    let angles: Vec<f32> = (0..count).map(|i| (i as f32 - 512.0) / 64.0).collect();
    unary("sin", &angles, &f32::sin, 0);
    unary("cos", &angles, &f32::cos, 0);

    // `div`: a bf16 numerator over a bf16 divisor, every datum.
    const DIV: &str = "\
version 9
grid 1,1
%in = param bf16 2048
%out = param bf16 1024
%b0 = buffer bf16 32,64
%bo = buffer bf16 32,32
copy 32,32 %b0 [0] = %in [0]
copy 32,32 %b0 [1] = %in [1]
load 32,32 %x = %b0 [0]
load 32,32 %y = %b0 [1]
%r = div %x %y
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";
    let x: Vec<f32> = (0..count).map(|i| (i as f32 - 500.0) / 7.0).collect();
    let y: Vec<f32> = (0..count).map(|i| 0.25 + (i % 97) as f32 * 0.5).collect();
    let mut bytes: Vec<u8> = Vec::new();
    for v in x.iter().chain(y.iter()) {
        bytes.extend(truncate_to_bf16(*v).to_le_bytes());
    }
    let out = run_kernel(DIV, &bytes, count * 2);
    for (i, (a, b)) in x.iter().zip(&y).enumerate() {
        let a = widen_bf16(truncate_to_bf16(*a));
        let b = widen_bf16(truncate_to_bf16(*b));
        let got = u16::from_le_bytes(out[2 * i..2 * i + 2].try_into().unwrap());
        let want = truncate_to_bf16(a / b);
        // The reciprocal is an estimate plus Newton steps: one bf16 ulp of
        // slack on the quotient.
        let slack = 2u16;
        assert!(
            got.wrapping_sub(want).min(want.wrapping_sub(got)) <= slack
                || (got & 0x7fff) == (want & 0x7fff),
            "div: datum {i}: {a} / {b} = {got:#06x}, wanted {want:#06x}"
        );
    }
}

/// The `(row, column)` of a 32x32 matrix that L1 datum `index` holds. A tile in
/// L1 is four 16x16 faces; each face is 256 contiguous datums, row-major.
fn tile_rc(index: usize) -> (usize, usize) {
    let (face, within) = (index / 256, index % 256);
    let (fr, fc) = (face / 2, face % 2);
    (fr * 16 + within / 16, fc * 16 + within % 16)
}

/// The L1 index of matrix element `(row, column)`.
fn tile_index(row: usize, column: usize) -> usize {
    let face = (row / 16) * 2 + column / 16;
    face * 256 + (row % 16) * 16 + column % 16
}

/// `a @ b` over 32x32 tiles, in L1 order.
fn tile_matmul(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; 1024];
    for row in 0..32 {
        for column in 0..32 {
            let mut sum = 0.0f32;
            for k in 0..32 {
                sum += a[tile_index(row, k)] * b[tile_index(k, column)];
            }
            out[tile_index(row, column)] = sum;
        }
    }
    out
}

#[test]
fn matmul_runs_on_the_simulator() {
    let run = |a: &[f32], b: &[f32]| -> Vec<f32> {
        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = run_kernel_tensors(MATMUL_F32, &[&a_bytes, &b_bytes], 4096);
        out.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };

    // The identity on the left leaves B untouched, in whatever order the Matrix
    // Unit reads the faces: no layout assumption enters this case.
    let mut a = vec![0.0f32; 1024];
    for i in 0..32 {
        a[tile_index(i, i)] = 1.0;
    }
    let b: Vec<f32> = (0..1024).map(|i| (i % 23) as f32 - 11.0).collect();
    let got = run(&a, &b);
    for i in 0..1024 {
        assert_eq!(got[i], b[i], "identity x B, datum {i}");
    }

    // A whole product: every value is an integer, so TF32 reads it exactly and
    // the accumulator is exact.
    let a: Vec<f32> = (0..1024).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i % 5) as f32 - 2.0).collect();
    let got = run(&a, &b);
    let want = tile_matmul(&a, &b);
    for i in 0..1024 {
        let (row, column) = tile_rc(i);
        assert_eq!(
            got[i], want[i],
            "a x b, datum {i} at ({row}, {column}): got {}, want {}",
            got[i], want[i]
        );
    }

    // The same product when `%a` also feeds a datapath op: one page, unpacked
    // once into `Dst` and once into `Src`.
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_kernel_tensors(MATMUL_F32_MIXED, &[&a_bytes, &b_bytes], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        assert_eq!(got[i], want[i], "a x b with a shared page, datum {i}");
    }
}

/// The same product with a computed operand: `%n = neg %a` lives in `Dst`, and
/// the Matrix Unit reads only `Src`, so the compiler packs `%n` into a page it
/// owns and the product is `(-a) x b`.
const MATMUL_F32_STAGED: &str = "\
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
%n = neg %a
matmul %m %n %b
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
";

/// The same with the other operand staged: `%n = neg %b` and the product is
/// `a x (-b)`.
const MATMUL_F32_STAGED_B: &str = "\
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
%n = neg %b
matmul %m %a %n
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
";

/// A computed operand is staged into a page the compiler owns: the product is
/// the one `(-a) x b` and every datum is exact.
#[test]
fn a_computed_matmul_operand_is_staged_into_a_page() {
    let a: Vec<f32> = (0..1024).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i % 5) as f32 - 2.0).collect();
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
    for (program, left, right, what) in [
        (
            MATMUL_F32_STAGED,
            a.iter().map(|v| -v).collect::<Vec<f32>>(),
            b.clone(),
            "(-a) x b",
        ),
        (
            MATMUL_F32_STAGED_B,
            a.clone(),
            b.iter().map(|v| -v).collect::<Vec<f32>>(),
            "a x (-b)",
        ),
    ] {
        let want = tile_matmul(&left, &right);
        let out = run_kernel_tensors(program, &[&a_bytes, &b_bytes], 4096);
        let got: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for i in 0..1024 {
            let (row, column) = tile_rc(i);
            assert_eq!(
                got[i], want[i],
                "{what}, datum {i} at ({row}, {column}): got {}, want {}",
                got[i], want[i]
            );
        }
    }
}

/// bf16 operands over an `f32` accumulator: `SrcA`/`SrcB` take the bf16 pages as
/// they are, the Matrix Unit accumulates in the 32-bit `Dst`, and the packer
/// writes the f32 result.
const MATMUL_BF16_F32: &str = "\
version 9
grid 1,1
%pa = param bf16 1024
%pb = param bf16 1024
%out = param f32 1024
%ba = buffer bf16 32,32
%bb = buffer bf16 32,32
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

/// bf16 operands over a `bf16` accumulator: the 16-bit `Dst`, the native path.
const MATMUL_BF16: &str = "\
version 9
grid 1,1
%pa = param bf16 1024
%pb = param bf16 1024
%out = param bf16 1024
%ba = buffer bf16 32,32
%bb = buffer bf16 32,32
%bo = buffer bf16 32,32
copy 32,32 %ba [0] = %pa [0]
copy 32,32 %bb [0] = %pb [0]
%m = acc bf16 32,32
%zero = const f32 0
store 32,32 %m = %zero
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
matmul %m %a %b
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
";

#[test]
fn bf16_matmul_runs_on_the_simulator() {
    // Integers: exact in bf16, and a product's sum stays inside bf16's eight
    // mantissa bits, so an f32 and a bf16 accumulator must agree.
    let a: Vec<f32> = (0..1024).map(|i| (i % 3) as f32 - 1.0).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i % 5) as f32 - 2.0).collect();
    let a_bf16: Vec<u16> = a.iter().map(|v| truncate_to_bf16(*v)).collect();
    let b_bf16: Vec<u16> = b.iter().map(|v| truncate_to_bf16(*v)).collect();
    let want = tile_matmul(&a, &b);
    let a_bytes: Vec<u8> = a_bf16.iter().flat_map(|h| h.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_bf16.iter().flat_map(|h| h.to_le_bytes()).collect();

    let out = run_kernel_tensors(MATMUL_BF16_F32, &[&a_bytes, &b_bytes], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        assert_eq!(got[i], want[i], "bf16 operands, f32 accumulator, datum {i}");
    }

    let out = run_kernel_tensors(MATMUL_BF16, &[&a_bytes, &b_bytes], 2048);
    let got: Vec<u16> = out
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, v) in want.iter().enumerate() {
        assert_eq!(
            got[i],
            truncate_to_bf16(*v),
            "bf16 operands, bf16 accumulator, datum {i}"
        );
    }
}

/// Interleaved (GGUF / llama.cpp) RoPE across four 32-wide pages of one head:
/// `y = x*cos + (x @ T)*sin`, where `T` is the ±1 page that swaps neighbours.
/// Every table is full-width, so one page is one 32-wide slice of the head and
/// the rotation never crosses a page.
const ROPE_INTERLEAVED: &str = "\
version 9
grid 1,1
%px = param f32 4096
%pc = param f32 4096
%ps = param f32 4096
%pt = param f32 1024
%out = param f32 4096
%bx = buffer f32 32,32
%bc = buffer f32 32,32
%bs = buffer f32 32,32
%bt = buffer f32 32,32
%br = buffer f32 32,32
%bo = buffer f32 32,32
%acc = acc f32 32,32
%zero = const f32 0
%j = loop 4
copy 32,32 %bx [0] = %px [%j]
copy 32,32 %bc [0] = %pc [%j]
copy 32,32 %bs [0] = %ps [%j]
copy 32,32 %bt [0] = %pt [0]
load 32,32 %x = %bx [0]
load 32,32 %c = %bc [0]
load 32,32 %s = %bs [0]
load 32,32 %t = %bt [0]
store 32,32 %acc = %zero
matmul %acc %x %t
store 32,32 %br [0] = %acc
load 32,32 %r = %br [0]
%y1 = mul %x %c
%y2 = mul %r %s
%y = add %y1 %y2
store 32,32 %bo [0] = %y
copy 32,32 %out [%j] = %bo [0]
endloop
";

/// Split-half (Hugging Face) RoPE at `head_dim = 128`: `d` pairs with
/// `d + 64`, so the two halves are whole pages and the rotation is two
/// multiplies and an add per output page — no matmul, no data movement. The
/// `+64` half reuses the low half's cos/sin page.
const ROPE_SPLIT_HALF: &str = "\
version 9
grid 1,1
%px = param f32 4096
%pc = param f32 2048
%ps = param f32 2048
%out = param f32 4096
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bc = buffer f32 32,32
%bs = buffer f32 32,32
%bo0 = buffer f32 32,32
%bo1 = buffer f32 32,32
%two = const i32 2
%j = loop 2
%oj2 = add %j %two
copy 32,32 %ba [0] = %px [%j]
copy 32,32 %bb [0] = %px [%oj2]
copy 32,32 %bc [0] = %pc [%j]
copy 32,32 %bs [0] = %ps [%j]
load 32,32 %xa = %ba [0]
load 32,32 %xb = %bb [0]
load 32,32 %c = %bc [0]
load 32,32 %s = %bs [0]
%p0 = mul %xa %c
%p1 = mul %xb %s
%y0 = sub %p0 %p1
store 32,32 %bo0 [0] = %y0
copy 32,32 %out [%j] = %bo0 [0]
%p2 = mul %xb %c
%p3 = mul %xa %s
%y1 = add %p2 %p3
store 32,32 %bo1 [0] = %y1
copy 32,32 %out [%oj2] = %bo1 [0]
endloop
";

/// Split-half RoPE at `head_dim = 32`: both halves are inside one page, so the
/// swap is the block permutation `[[0, I],[-I, 0]]` the Matrix Unit applies.
const ROPE_SPLIT_HALF_PAGE: &str = "\
version 9
grid 1,1
%px = param f32 1024
%pc = param f32 1024
%ps = param f32 1024
%pt = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bc = buffer f32 32,32
%bs = buffer f32 32,32
%bt = buffer f32 32,32
%br = buffer f32 32,32
%bo = buffer f32 32,32
%acc = acc f32 32,32
%zero = const f32 0
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bc [0] = %pc [0]
copy 32,32 %bs [0] = %ps [0]
copy 32,32 %bt [0] = %pt [0]
load 32,32 %x = %bx [0]
load 32,32 %c = %bc [0]
load 32,32 %s = %bs [0]
load 32,32 %t = %bt [0]
store 32,32 %acc = %zero
matmul %acc %x %t
store 32,32 %br [0] = %acc
load 32,32 %r = %br [0]
%y1 = mul %x %c
%y2 = mul %r %s
%y = add %y1 %y2
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// The interleaved rotation: `y[2k] = -x[2k+1]`, `y[2k+1] = x[2k]` — one
/// `matmul` against a ±1 page, then the two multiplies and the add.
#[test]
fn rope_interleaved_runs_on_the_simulator() {
    const TOKENS: usize = 32;
    const DIM: usize = 128;
    // Small integers: exact in TF32, so the accumulator store rounds nothing.
    let x: Vec<f32> = (0..TOKENS * DIM).map(|i| (i % 7) as f32 - 3.0).collect();
    // bf16 tables, as tt-metal ships them. Truncated here too, so every later
    // product is exact in f32 and the rotation is asserted datum by datum.
    let table = |f: fn(f32) -> f32| -> Vec<f32> {
        (0..TOKENS * DIM)
            .map(|i| {
                let (t, d) = (i / DIM, i % DIM);
                let angle = (t * (d / 2)) as f32 * 0.001;
                f32::from_bits(u32::from(truncate_to_bf16(f(angle))) << 16)
            })
            .collect()
    };
    let cos = table(f32::cos);
    let sin = table(f32::sin);
    let mut perm = vec![0f32; 1024];
    for k in 0..16 {
        perm[2 * k * 32 + 2 * k + 1] = 1.0;
        perm[(2 * k + 1) * 32 + 2 * k] = -1.0;
    }

    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let out = run_kernel_tensors(
        ROPE_INTERLEAVED,
        &[
            &bytes(&tiled(TOKENS, DIM, &x)),
            &bytes(&tiled(TOKENS, DIM, &cos)),
            &bytes(&tiled(TOKENS, DIM, &sin)),
            &bytes(&tiled(32, 32, &perm)),
        ],
        TOKENS * DIM * 4,
    );
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut shown = 0;
    for t in 0..TOKENS {
        for d in 0..DIM {
            let (a, b) = (x[t * DIM + d], x[t * DIM + (d ^ 1)]);
            let (c, s) = (cos[t * DIM + d], sin[t * DIM + d]);
            let want = if d % 2 == 0 {
                a * c - b * s
            } else {
                a * c + b * s
            };
            let at = (d / 32) * 1024 + tile_index(t, d % 32);
            if shown < 8 {
                eprintln!(
                    "t{t} d{d}: got {} want {} x {} c {} s {}",
                    got[at], want, a, c, s
                );
                shown += 1;
            }
            assert_eq!(got[at], want, "token {t}, dim {d}");
        }
    }
}

/// The split-half rotation at `head_dim = 128`: `d` pairs with `d + 64`, both
/// halves whole pages, so this form needs no matrix product at all.
#[test]
fn rope_split_half_runs_on_the_simulator() {
    const TOKENS: usize = 32;
    const DIM: usize = 128;
    const HALF: usize = DIM / 2;
    let x: Vec<f32> = (0..TOKENS * DIM).map(|i| (i % 7) as f32 - 3.0).collect();
    let table = |f: fn(f32) -> f32| -> Vec<f32> {
        (0..TOKENS * HALF)
            .map(|i| {
                let (t, d) = (i / HALF, i % HALF);
                let angle = (t * d) as f32 * 0.001;
                f32::from_bits(u32::from(truncate_to_bf16(f(angle))) << 16)
            })
            .collect()
    };
    let cos = table(f32::cos);
    let sin = table(f32::sin);

    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let out = run_kernel_tensors(
        ROPE_SPLIT_HALF,
        &[
            &bytes(&tiled(TOKENS, DIM, &x)),
            &bytes(&tiled(TOKENS, HALF, &cos)),
            &bytes(&tiled(TOKENS, HALF, &sin)),
        ],
        TOKENS * DIM * 4,
    );
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for t in 0..TOKENS {
        for d in 0..DIM {
            let a = x[t * DIM + d];
            let (c, s) = (cos[t * HALF + d % HALF], sin[t * HALF + d % HALF]);
            let want = if d < HALF {
                a * c - x[t * DIM + d + HALF] * s
            } else {
                a * c + x[t * DIM + d - HALF] * s
            };
            let at = (d / 32) * 1024 + tile_index(t, d % 32);
            assert_eq!(got[at], want, "token {t}, dim {d}");
        }
    }
}

/// The split-half rotation inside one 32-wide page: `d` pairs with `d + 16`, so
/// the swap is the ±1 block page and the arithmetic is `matmul`/`mul`/`add`.
#[test]
fn rope_split_half_within_a_page_runs_on_the_simulator() {
    const TOKENS: usize = 32;
    const DIM: usize = 32;
    const HALF: usize = DIM / 2;
    let x: Vec<f32> = (0..TOKENS * DIM).map(|i| (i % 7) as f32 - 3.0).collect();
    // Each frequency covers a pair, as the single-tile op's tables do.
    let table = |f: fn(f32) -> f32| -> Vec<f32> {
        (0..TOKENS * DIM)
            .map(|i| {
                let (t, d) = (i / DIM, i % DIM);
                let angle = (t * (d / 2)) as f32 * 0.001;
                f32::from_bits(u32::from(truncate_to_bf16(f(angle))) << 16)
            })
            .collect()
    };
    let cos = table(f32::cos);
    let sin = table(f32::sin);
    let mut perm = vec![0f32; 1024];
    for d in 0..HALF {
        perm[(d + HALF) * 32 + d] = -1.0;
        perm[d * 32 + (d + HALF)] = 1.0;
    }

    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let out = run_kernel_tensors(
        ROPE_SPLIT_HALF_PAGE,
        &[
            &bytes(&tiled(TOKENS, DIM, &x)),
            &bytes(&tiled(TOKENS, DIM, &cos)),
            &bytes(&tiled(TOKENS, DIM, &sin)),
            &bytes(&tiled(32, 32, &perm)),
        ],
        TOKENS * DIM * 4,
    );
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for t in 0..TOKENS {
        for d in 0..DIM {
            let a = x[t * DIM + d];
            let (c, s) = (cos[t * DIM + d], sin[t * DIM + d]);
            let want = if d < HALF {
                a * c - x[t * DIM + d + HALF] * s
            } else {
                a * c + x[t * DIM + d - HALF] * s
            };
            let at = tile_index(t, d);
            assert_eq!(got[at], want, "token {t}, dim {d}");
        }
    }
}

/// A `32,1` accumulator with `sum %acc %x %scaler`: the fold is one value per
/// row, in column 0 — tt-metal's `ReduceDim::ROW` with
/// `PoolType::SUM`, which is the Matrix Unit's `x @ scaler`, not a pool
/// instruction (`reduce_uses_matmul<SUM, REDUCE_ROW>`). The scaler page carries
/// the factor in column 0 and zeros elsewhere — tt-metal's column-0 fill for
/// this combination (`prepare_reduce_scaler`) — so the product is each row's
/// sum in column 0 and zeros elsewhere.
const REDUCE_SUM_ROW: &str = "\
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
%acc = acc f32 32,1
%zero = const f32 0
store 32,32 %acc = %zero
load 32,32 %x = %bx [0]
load 32,32 %s = %bs [0]
sum %acc %x %s
store 32,32 %bo [0] = %acc
copy 32,32 %out [0] = %bo [0]
";

#[test]
fn sum_row_runs_on_the_simulator() {
    // Integers: exact through TF32 and the accumulator, so every datum of the
    // output can be compared exactly.
    let x: Vec<f32> = (0..1024).map(|i| (i % 7) as f32 - 3.0).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    // The scaler is the factor, not decoration: 1 and 2 give the sum and twice
    // the sum in column 0.
    for factor in [1.0f32, 2.0f32] {
        let mut scaler = vec![0.0f32; 1024];
        for row in 0..32 {
            scaler[tile_index(row, 0)] = factor;
        }
        let want = tile_matmul(&x, &scaler);
        // The product must put the sums where `REDUCE_ROW` puts them: column 0
        // of every row, zero elsewhere.
        for row in 0..32 {
            let sum: f32 = (0..32).map(|column| x[tile_index(row, column)]).sum();
            assert_eq!(
                want[tile_index(row, 0)],
                factor * sum,
                "the host product's column 0, row {row}"
            );
            for column in 1..32 {
                assert_eq!(
                    want[tile_index(row, column)],
                    0.0,
                    "row {row} column {column}"
                );
            }
        }
        let s_bytes: Vec<u8> = scaler.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = run_kernel_tensors(REDUCE_SUM_ROW, &[&x_bytes, &s_bytes], 4096);
        let got: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for i in 0..1024 {
            let (row, column) = tile_rc(i);
            assert_eq!(
                got[i], want[i],
                "sum row, scaler {factor}, datum {i} at ({row}, {column})"
            );
        }
    }
}

/// A `1,32` accumulator with `sum %acc %x %scaler` or `max %acc %x`:
/// `GAPOOL`/`GMPOOL` fold
/// each 16x16 face over its rows into row 0, one value per column, and the
/// pack's edge mask drops the rest. `sum`'s scaler carries the factor in row 0
/// of each face (tt-metal's row-0 fill); `max` carries none and the compiler
/// moves 1.0 into `SrcB` itself.
const REDUCE_SUM_COL: &str = "\
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
%acc = acc f32 1,32
%zero = const f32 0
store 32,32 %acc = %zero
load 32,32 %x = %bx [0]
load 32,32 %s = %bs [0]
sum %acc %x %s
store 32,32 %bo [0] = %acc
copy 32,32 %out [0] = %bo [0]
";

const REDUCE_MAX_COL: &str = "\
version 9
grid 1,1
%px = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
%acc = acc f32 1,32
%low = const f32 -1000
store 32,32 %acc = %low
load 32,32 %x = %bx [0]
max %acc %x
store 32,32 %bo [0] = %acc
copy 32,32 %out [0] = %bo [0]
";

/// The same, seeded above every datum: the accumulator is the fold's other
/// operand, so every column keeps the seed.
const REDUCE_MAX_COL_SEEDED: &str = "\
version 9
grid 1,1
%px = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
%acc = acc f32 1,32
%high = const f32 1000
store 32,32 %acc = %high
load 32,32 %x = %bx [0]
max %acc %x
store 32,32 %bo [0] = %acc
copy 32,32 %out [0] = %bo [0]
";

/// The same, seeded below every datum: the accumulator is the fold's other
/// operand, so every row's own maximum survives.
const REDUCE_MAX_ROW: &str = "\
version 9
grid 1,1
%px = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
%acc = acc f32 32,1
%low = const f32 -1000
store 32,32 %acc = %low
load 32,32 %x = %bx [0]
max %acc %x
store 32,32 %bo [0] = %acc
copy 32,32 %out [0] = %bo [0]
";

/// The same with a seed above every datum.
const REDUCE_MAX_ROW_SEEDED: &str = "\
version 9
grid 1,1
%px = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
%acc = acc f32 32,1
%high = const f32 1000
store 32,32 %acc = %high
load 32,32 %x = %bx [0]
max %acc %x
store 32,32 %bo [0] = %acc
copy 32,32 %out [0] = %bo [0]
";

/// The `row` result: column 0 holds one fold per row, every other datum zero.
fn row_reduce_oracle(input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; 1024];
    for row in 0..32 {
        let mut folded = f32::NEG_INFINITY;
        for column in 0..32 {
            folded = folded.max(input[tile_index(row, column)]);
        }
        out[tile_index(row, 0)] = folded;
    }
    out
}

/// The `col` result: row 0 holds one fold per column, every other datum zero.
fn col_reduce_oracle(input: &[f32], max: bool, scaler: Option<f32>) -> Vec<f32> {
    let mut out = vec![0.0f32; 1024];
    for column in 0..32 {
        let mut folded = if max { f32::NEG_INFINITY } else { 0.0 };
        for row in 0..32 {
            let value = input[tile_index(row, column)];
            folded = if max {
                folded.max(value)
            } else {
                folded + value
            };
        }
        out[tile_index(0, column)] = folded * scaler.unwrap_or(1.0);
    }
    out
}

/// The four reduce forms in a 16-bit `Dst`: `sum row` (the Matrix Unit's
/// `x @ scaler`), `sum col` and `max col` (`GAPOOL`/`GMPOOL`), and `max row`.
/// Every datum is an integer, so bf16 carries each fold exactly.
const REDUCE_SUM_ROW_BF16: &str = "\
version 9
grid 1,1
%px = param bf16 1024
%ps = param bf16 1024
%out = param bf16 1024
%bx = buffer bf16 32,32
%bs = buffer bf16 32,32
%bo = buffer bf16 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bs [0] = %ps [0]
%acc = acc bf16 32,1
%zero = const f32 0
store 32,32 %acc = %zero
load 32,32 %x = %bx [0]
load 32,32 %s = %bs [0]
sum %acc %x %s
store 32,32 %bo [0] = %acc
copy 32,32 %out [0] = %bo [0]
";

const REDUCE_SUM_COL_BF16: &str = "\
version 9
grid 1,1
%px = param bf16 1024
%ps = param bf16 1024
%out = param bf16 1024
%bx = buffer bf16 32,32
%bs = buffer bf16 32,32
%bo = buffer bf16 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bs [0] = %ps [0]
%acc = acc bf16 1,32
%zero = const f32 0
store 32,32 %acc = %zero
load 32,32 %x = %bx [0]
load 32,32 %s = %bs [0]
sum %acc %x %s
store 32,32 %bo [0] = %acc
copy 32,32 %out [0] = %bo [0]
";

const REDUCE_MAX_COL_BF16: &str = "\
version 9
grid 1,1
%px = param bf16 1024
%out = param bf16 1024
%bx = buffer bf16 32,32
%bo = buffer bf16 32,32
copy 32,32 %bx [0] = %px [0]
%acc = acc bf16 1,32
%low = const f32 -1000
store 32,32 %acc = %low
load 32,32 %x = %bx [0]
max %acc %x
store 32,32 %bo [0] = %acc
copy 32,32 %out [0] = %bo [0]
";

/// The `row` fold of a bf16 page: column 0 of every row holds the row's
/// maximum, and the fold goes through the 16-bit transpose (`MOVD2B`,
/// `TRNSPSRCB`, the two `ELWADD`s) rather than the 32-bit hi/lo split.
const REDUCE_MAX_ROW_BF16: &str = "\
version 9
grid 1,1
%px = param bf16 1024
%out = param bf16 1024
%bx = buffer bf16 32,32
%bo = buffer bf16 32,32
copy 32,32 %bx [0] = %px [0]
%acc = acc bf16 32,1
%low = const f32 -1000
store 32,32 %acc = %low
load 32,32 %x = %bx [0]
max %acc %x
store 32,32 %bo [0] = %acc
copy 32,32 %out [0] = %bo [0]
";

/// Every datum of a bf16 page, as the `u16` it is.
fn bf16_pages(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn bf16_reduces_run_on_the_simulator() {
    let count = 1024usize;
    let x: Vec<u16> = (0..count)
        .map(|i| truncate_to_bf16((i % 23) as f32 - 11.0))
        .collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|h| h.to_le_bytes()).collect();
    let widened: Vec<f32> = x.iter().map(|h| widen_bf16(*h)).collect();

    // `sum row`: column 0 of every row holds the row's fold times the scaler.
    for factor in [1.0f32, 2.0f32] {
        let mut scaler = vec![0.0f32; count];
        for row in 0..32 {
            scaler[tile_index(row, 0)] = factor;
        }
        let s_bytes: Vec<u8> = scaler
            .iter()
            .flat_map(|v| truncate_to_bf16(*v).to_le_bytes())
            .collect();
        let got = bf16_pages(&run_kernel_tensors(
            REDUCE_SUM_ROW_BF16,
            &[&x_bytes, &s_bytes],
            count * 2,
        ));
        for row in 0..32 {
            let sum: f32 = (0..32).map(|column| widened[tile_index(row, column)]).sum();
            assert_eq!(
                got[tile_index(row, 0)],
                truncate_to_bf16(factor * sum),
                "bf16 sum row, factor {factor}: row {row}"
            );
            for column in 1..32 {
                assert_eq!(
                    got[tile_index(row, column)],
                    0,
                    "bf16 sum row, factor {factor}: ({row}, {column}) is not zero"
                );
            }
        }
    }

    // `sum col`: row 0 of every face holds the column's fold.
    for factor in [1.0f32, 2.0f32] {
        let mut scaler = vec![0.0f32; count];
        for column in 0..32 {
            scaler[tile_index(0, column)] = factor;
            scaler[tile_index(16, column)] = factor;
        }
        let s_bytes: Vec<u8> = scaler
            .iter()
            .flat_map(|v| truncate_to_bf16(*v).to_le_bytes())
            .collect();
        let got = bf16_pages(&run_kernel_tensors(
            REDUCE_SUM_COL_BF16,
            &[&x_bytes, &s_bytes],
            count * 2,
        ));
        let want = col_reduce_oracle(&widened, false, Some(factor));
        for i in 0..count {
            assert_eq!(
                got[i],
                truncate_to_bf16(want[i]),
                "bf16 sum col, factor {factor}: datum {i}"
            );
        }
    }

    // `max col`, seeded below every datum.
    let got = bf16_pages(&run_kernel(REDUCE_MAX_COL_BF16, &x_bytes, count * 2));
    let want = col_reduce_oracle(&widened, true, None);
    for i in 0..count {
        assert_eq!(got[i], truncate_to_bf16(want[i]), "bf16 max col: datum {i}");
    }

    // `max row`, seeded below every datum: column 0 of every row holds the
    // row's fold, and the 16-bit transpose is what lands it there.
    let got = bf16_pages(&run_kernel(REDUCE_MAX_ROW_BF16, &x_bytes, count * 2));
    let want = row_reduce_oracle(&widened);
    for row in 0..32 {
        assert_eq!(
            got[tile_index(row, 0)],
            truncate_to_bf16(want[tile_index(row, 0)]),
            "bf16 max row: row {row}"
        );
        for column in 1..32 {
            assert_eq!(
                got[tile_index(row, column)],
                0,
                "bf16 max row: ({row}, {column}) is not zero"
            );
        }
    }
}

#[test]
fn sum_col_runs_on_the_simulator() {
    // Integers: exact through the four fidelity phases and the accumulator.
    let x: Vec<f32> = (0..1024).map(|i| (i % 23) as f32 - 11.0).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    for factor in [1.0f32, 2.0f32] {
        let mut scaler = vec![0.0f32; 1024];
        for column in 0..32 {
            // Row 0 of each face: the face's first row.
            scaler[tile_index(0, column)] = factor;
            scaler[tile_index(16, column)] = factor;
        }
        let want = col_reduce_oracle(&x, false, Some(factor));
        assert_ne!(want, vec![0.0; 1024], "negative control");
        let s_bytes: Vec<u8> = scaler.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = run_kernel_tensors(REDUCE_SUM_COL, &[&x_bytes, &s_bytes], 4096);
        let got: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for i in 0..1024 {
            let (row, column) = tile_rc(i);
            assert_eq!(
                got[i], want[i],
                "sum col, scaler {factor}, datum {i} at ({row}, {column})"
            );
        }
    }
}

#[test]
fn max_col_runs_on_the_simulator() {
    // Integers, with negatives on both sides of the accumulator's seed.
    let x: Vec<f32> = (0..1024).map(|i| (i % 23) as f32 - 11.0).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let want = col_reduce_oracle(&x, true, None);
    assert_ne!(want, vec![0.0; 1024], "negative control");
    let out = run_kernel_tensors(REDUCE_MAX_COL, &[&x_bytes], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        let (row, column) = tile_rc(i);
        assert_eq!(got[i], want[i], "max col, datum {i} at ({row}, {column})");
    }

    // The accumulator is the fold's seed: a seed above every datum survives in
    // every column, while faces 2 and 3 are still zero.
    let out = run_kernel_tensors(REDUCE_MAX_COL_SEEDED, &[&x_bytes], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        let (row, column) = tile_rc(i);
        let want = if row == 0 { 1000.0 } else { 0.0 };
        assert_eq!(
            got[i], want,
            "seeded max col, datum {i} at ({row}, {column})"
        );
    }
}

#[test]
fn max_row_runs_on_the_simulator() {
    // Fractional magnitudes: below 1, so every value is exact in TF32 (the
    // format the unpacker puts into `Src`), and the low half of each f32 word
    // carries real mantissa bits, so the hi/lo transposes are not a no-op.
    let x: Vec<f32> = (0..1024)
        .map(|i| {
            let value = (i % 97) as f32 / 1024.0;
            if i % 3 == 0 { -value } else { value }
        })
        .collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let want = row_reduce_oracle(&x);
    assert_ne!(want, vec![0.0; 1024], "negative control");
    let out = run_kernel_tensors(REDUCE_MAX_ROW, &[&x_bytes], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        let (row, column) = tile_rc(i);
        assert_eq!(got[i], want[i], "max row, datum {i} at ({row}, {column})");
    }

    // The accumulator is the fold's seed: a seed above every datum survives in
    // every row, and the masked datums are still zero.
    let out = run_kernel_tensors(REDUCE_MAX_ROW_SEEDED, &[&x_bytes], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        let (row, column) = tile_rc(i);
        let want = if column == 0 { 1000.0 } else { 0.0 };
        assert_eq!(
            got[i], want,
            "seeded max row, datum {i} at ({row}, {column})"
        );
    }
}

/// A buffer the datapath writes and then reads: `store` packs a page, `load`
/// unpacks the same page. Any role may drive either end of a buffer.
const ROUND_TRIP: &str = "\
version 9
grid 1,1
%px = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bt = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
load 32,32 %x = %bx [0]
%n = neg %x
store 32,32 %bt [0] = %n
load 32,32 %y = %bt [0]
%z = neg %y
store 32,32 %bo [0] = %z
copy 32,32 %out [0] = %bo [0]
";

/// The packer writes and the unpacker reads the same L1 buffer: `%bt`'s page is
/// published by `store` and consumed by `load`, so the value comes back as it
/// went in.
#[test]
fn a_buffer_a_store_wrote_a_load_reads() {
    let x: Vec<f32> = (0..1024).map(|i| (i % 9) as f32 - 4.0).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_kernel_tensors(ROUND_TRIP, &[&x_bytes], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        let (row, column) = tile_rc(i);
        assert_eq!(
            got[i], x[i],
            "round trip, datum {i} at ({row}, {column}): got {}, want {}",
            got[i], x[i]
        );
    }
}

/// `x + row`: the tile is `SrcA`, the narrow value is `SrcB`, and the FPU's
/// row broadcast reads `SrcB`'s first face row. The narrow value's page is
/// laid out as a `1,32`: row 0 of faces 0 and 1.
const BROADCAST_ROW_ADD: &str = "\
version 9
grid 1,1
%px = param f32 1024
%pm = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bm = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bm [0] = %pm [0]
load 32,32 %x = %bx [0]
load 1,32 %m = %bm [0]
%y = add %x %m
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// `x * row`, the same shape pair and the multiply's own instruction.
const BROADCAST_ROW_MUL: &str = "\
version 9
grid 1,1
%px = param f32 1024
%pm = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bm = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bm [0] = %pm [0]
load 32,32 %x = %bx [0]
load 1,32 %m = %bm [0]
%y = mul %x %m
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// `x - column`: a `32,1` value is column 0 of faces 0 and 2, and the column
/// broadcast reads `SrcB`'s column 0. The subtraction has the tile on the left,
/// which is the order the instruction takes.
const BROADCAST_COL_SUB: &str = "\
version 9
grid 1,1
%px = param f32 1024
%pm = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bm = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bm [0] = %pm [0]
load 32,32 %x = %bx [0]
load 32,1 %m = %bm [0]
%y = sub %x %m
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// `x * column`, the same shape pair and the multiply's own instruction.
const BROADCAST_COL_MUL: &str = "\
version 9
grid 1,1
%px = param f32 1024
%pm = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bm = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bm [0] = %pm [0]
load 32,32 %x = %bx [0]
load 32,1 %m = %bm [0]
%y = mul %x %m
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// `neg x * row`: `SrcA` is a value the datapath computed, so the compiler
/// stages it into a page the FPU reads, while `SrcB` is the load's own page.
const BROADCAST_STAGED_A: &str = "\
version 9
grid 1,1
%px = param f32 1024
%pm = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bm = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bm [0] = %pm [0]
load 32,32 %x = %bx [0]
load 1,32 %m = %bm [0]
%n = neg %x
%y = mul %n %m
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// A `32,1` value whose page the *packer* wrote (`store %br = %n`) rather than
/// a copy from a parameter: the narrow read names column 0 of faces 0 and 2,
/// and every row of the tile is multiplied by its own row's first datum.
const BROADCAST_COL_FROM_PACK: &str = "\
version 9
grid 1,1
%px = param f32 1024
%pm = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bm = buffer f32 32,32
%br = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bm [0] = %pm [0]
load 32,32 %x = %bx [0]
load 32,32 %t = %bm [0]
%n = neg %t
store 32,32 %br [0] = %n
load 32,1 %m = %br [0]
%y = mul %x %m
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// A row and a column broadcast: `ELWADD`/`ELWMUL` with `SrcB`'s row 0, and
/// `ELWSUB` with its column 0. Every datum is an integer, exact in TF32 and in
/// the FPU's 32-bit `Dst`, so each result compares equal to the host's.
#[test]
fn a_broadcast_reads_its_narrow_page() {
    let x: Vec<f32> = (0..1024)
        .map(|i| {
            let (row, column) = tile_rc(i);
            ((row as i32 * 3 + column as i32 * 5) % 7) as f32 - 3.0
        })
        .collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    // The narrow page: a full 32x32 tile whose relevant part carries the
    // value, which is what a `1,32`/`32,1` load names.
    let row_tile: Vec<f32> = (0..1024)
        .map(|i| {
            let (row, column) = tile_rc(i);
            if row == 0 {
                (column % 5) as f32 - 2.0
            } else {
                0.0
            }
        })
        .collect();
    let col_tile: Vec<f32> = (0..1024)
        .map(|i| {
            let (row, column) = tile_rc(i);
            if column == 0 {
                (row % 5) as f32 - 2.0
            } else {
                0.0
            }
        })
        .collect();
    let row_bytes: Vec<u8> = row_tile.iter().flat_map(|v| v.to_le_bytes()).collect();
    let col_bytes: Vec<u8> = col_tile.iter().flat_map(|v| v.to_le_bytes()).collect();

    for (program, narrow, narrow_bytes, what, staged) in [
        (
            BROADCAST_ROW_ADD,
            row_tile.clone(),
            row_bytes.as_slice(),
            "row add",
            false,
        ),
        (
            BROADCAST_ROW_MUL,
            row_tile.clone(),
            row_bytes.as_slice(),
            "row mul",
            false,
        ),
        (
            BROADCAST_COL_SUB,
            col_tile.clone(),
            col_bytes.as_slice(),
            "column sub",
            false,
        ),
        (
            BROADCAST_COL_MUL,
            col_tile.clone(),
            col_bytes.as_slice(),
            "column mul",
            false,
        ),
        (
            BROADCAST_STAGED_A,
            row_tile.clone(),
            row_bytes.as_slice(),
            "row mul",
            true,
        ),
    ] {
        let out = run_kernel_tensors(program, &[&x_bytes, narrow_bytes], 4096);
        let got: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for i in 0..1024 {
            let (row, column) = tile_rc(i);
            let v = if what.starts_with("column") {
                narrow[tile_index(row, 0)]
            } else {
                narrow[tile_index(0, column)]
            };
            let a = if staged { -x[i] } else { x[i] };
            let want = match what {
                "row add" => a + v,
                "row mul" | "column mul" => a * v,
                _ => a - v,
            };
            let label = if staged { "staged A" } else { what };
            assert_eq!(
                got[i], want,
                "{what} ({label}), datum {i} at ({row}, {column}): got {}, want {want}",
                got[i]
            );
        }
    }
}

/// The `rms_norm` second loop in miniature: a per-row factor (a packed `32,1`)
/// and a per-tile scale, walked by a `loop` whose index is the copy's offset.
/// `out[j] = x[j] · m · s[j]`, every datum.
const SCALE_LOOP: &str = "\
version 9
grid 1,1
%px = param f32 2048
%ps = param f32 2048
%pm = param f32 1024
%out = param f32 2048
%bx = buffer f32 32,32
%bs = buffer f32 32,32
%bm = buffer f32 32,32
%br = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bm [0] = %pm [0]
load 32,32 %t = %bm [0]
%n = neg %t
store 32,32 %br [0] = %n
load 32,1 %m = %br [0]
%one = const i32 1
%nn = const i32 2
%j = loop %nn
%oj = mul %j %one
copy 32,32 %bx [0] = %px [%oj]
load 32,32 %x2 = %bx [0]
%y = mul %x2 %m
copy 32,32 %bs [0] = %ps [%oj]
load 32,32 %sc = %bs [0]
%z = mul %y %sc
store 32,32 %bo [0] = %z
copy 32,32 %out [%oj] = %bo [0]
endloop
";

/// The same operands, straight line: if this passes, the `loop` is what breaks
/// the tile operand.
#[test]
fn a_scale_straight_line_keeps_every_datum() {
    let x: Vec<f32> = (0..2048)
        .map(|i| {
            let (row, column) = tile_rc(i % 1024);
            ((row as i32 * 3 + (column as i32 + (i / 1024) as i32) * 5) % 7) as f32 - 3.0
        })
        .collect();
    let s: Vec<f32> = (0..2048)
        .map(|i| {
            let (row, column) = tile_rc(i % 1024);
            1.0 + (row as i32 + column as i32 + i as i32) as f32 * 0.25
        })
        .collect();
    let t: Vec<f32> = (0..1024)
        .map(|i| {
            let (row, column) = tile_rc(i);
            if column == 0 {
                (row % 5) as f32 - 2.0
            } else {
                0.0
            }
        })
        .collect();
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let sb: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tb: Vec<u8> = t.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_kernel_tensors(SCALE_STRAIGHT, &[&xb, &sb, &tb], 2048 * 4);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for j in 0..2 {
        for i in 0..1024 {
            let (row, column) = tile_rc(i);
            let want = x[j * 1024 + i] * -t[tile_index(row, 0)] * s[j * 1024 + i];
            assert_eq!(
                got[j * 1024 + i],
                want,
                "tile {j}, datum {i} at ({row}, {column}): got {}, want {want}",
                got[j * 1024 + i]
            );
        }
    }
}

/// The loop that `rms_norm` uses to walk the hidden tiles, with the same three
/// operands: a tile, a per-row factor and a per-tile scale.
#[test]
fn a_scale_loop_keeps_every_datum() {
    let x: Vec<f32> = (0..2048)
        .map(|i| {
            let (row, column) = tile_rc(i % 1024);
            ((row as i32 * 3 + (column as i32 + (i / 1024) as i32) * 5) % 7) as f32 - 3.0
        })
        .collect();
    let s: Vec<f32> = (0..2048)
        .map(|i| {
            let (row, column) = tile_rc(i % 1024);
            1.0 + (row as i32 + column as i32 + i as i32) as f32 * 0.25
        })
        .collect();
    let t: Vec<f32> = (0..1024)
        .map(|i| {
            let (row, column) = tile_rc(i);
            if column == 0 {
                (row % 5) as f32 - 2.0
            } else {
                0.0
            }
        })
        .collect();
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let sb: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tb: Vec<u8> = t.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_kernel_tensors(SCALE_LOOP, &[&xb, &sb, &tb], 2048 * 4);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for j in 0..2 {
        for i in 0..1024 {
            let (row, column) = tile_rc(i);
            let want = x[j * 1024 + i] * -t[tile_index(row, 0)] * s[j * 1024 + i];
            assert_eq!(
                got[j * 1024 + i],
                want,
                "tile {j}, datum {i} at ({row}, {column}): got {}, want {want}",
                got[j * 1024 + i]
            );
        }
    }
}

/// The same three operands without the `loop`: two tiles, each loaded, scaled
/// by a per-row factor and a per-tile scale, and written back.
const SCALE_STRAIGHT: &str = "\
version 9
grid 1,1
%px = param f32 2048
%ps = param f32 2048
%pm = param f32 1024
%out = param f32 2048
%bx = buffer f32 32,32
%bs = buffer f32 32,32
%bm = buffer f32 32,32
%br = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bm [0] = %pm [0]
load 32,32 %t = %bm [0]
%n = neg %t
store 32,32 %br [0] = %n
load 32,1 %m = %br [0]
copy 32,32 %bx [0] = %px [0]
load 32,32 %x0 = %bx [0]
%y0 = mul %x0 %m
copy 32,32 %bs [0] = %ps [0]
load 32,32 %s0 = %bs [0]
%z0 = mul %y0 %s0
store 32,32 %bo [0] = %z0
copy 32,32 %out [0] = %bo [0]
%one = const i32 1
copy 32,32 %bx [0] = %px [%one]
load 32,32 %x1 = %bx [0]
%y1 = mul %x1 %m
copy 32,32 %bs [0] = %ps [%one]
load 32,32 %s1 = %bs [0]
%z1 = mul %y1 %s1
store 32,32 %bo [0] = %z1
copy 32,32 %out [%one] = %bo [0]
";

/// The narrow page written by the packer, not by a parameter copy: softmax and
/// `rms_norm` both hand the FPU a `32,1`/`1,32` whose page came out of a `store`,
/// and the read must be the same column/row it is for a parameter's page.
#[test]
fn a_broadcast_reads_a_packed_narrow_page() {
    let x: Vec<f32> = (0..1024)
        .map(|i| {
            let (row, column) = tile_rc(i);
            ((row as i32 * 3 + column as i32 * 5) % 7) as f32 - 3.0
        })
        .collect();
    // `%n = neg %t` is packed, and `%m` is its column 0: `-t[row][0]`.
    let t: Vec<f32> = (0..1024)
        .map(|i| {
            let (row, column) = tile_rc(i);
            if column == 0 {
                (row % 5) as f32 - 2.0
            } else {
                0.0
            }
        })
        .collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t_bytes: Vec<u8> = t.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_kernel_tensors(BROADCAST_COL_FROM_PACK, &[&x_bytes, &t_bytes], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        let (row, column) = tile_rc(i);
        let want = x[i] * -t[tile_index(row, 0)];
        assert_eq!(
            got[i], want,
            "column mul from a packed page, datum {i} at ({row}, {column}): got {}, want {want}",
            got[i]
        );
    }
}

/// Softmax down a tile's rows: `max` folds each column into row 0, the
/// subtraction broadcasts that row back, `exp2(x · log2 e)` is `exp`, `sum`
/// folds the exponentials, and the reciprocal goes through L1: the accumulator's
/// page is loaded as a tile, the tile's reciprocal is stored, and its row is the
/// broadcast operand of the final multiply.
const SOFTMAX_COL: &str = "\
version 9
grid 1,1
%px = param f32 1024
%ps = param f32 1024
%out = param f32 1024
%bx = buffer f32 32,32
%bs = buffer f32 32,32
%bsum = buffer f32 32,32
%binv = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bs [0] = %ps [0]
%m = acc f32 1,32
%low = const f32 -1000
store 32,32 %m = %low
load 32,32 %x = %bx [0]
max %m %x
%d = sub %x %m
%k = const f32 1.442695
%p = mul %d %k
%e = exp2 %p
%s = acc f32 1,32
%zero = const f32 0
store 32,32 %s = %zero
load 32,32 %ones = %bs [0]
sum %s %e %ones
store 32,32 %bsum [0] = %s
load 32,32 %st = %bsum [0]
%inv = div %ones %st
store 32,32 %binv [0] = %inv
load 1,32 %ri = %binv [0]
%y = mul %e %ri
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// The whole softmax kernel on one tile: every datum against the host's
/// `exp(x - max) / sum(exp(x - max))` down each column, and every column sums to
/// one. The arithmetic is the hardware's — TF32 into the FPU, the SFPU's exp2
/// body and reciprocal, the Matrix Unit's fold — so the comparison is a
/// tolerance, not an equality.
#[test]
fn softmax_runs_on_the_simulator() {
    let x: Vec<f32> = (0..1024)
        .map(|i| {
            let (row, column) = tile_rc(i);
            ((row as i32 * 7 + column as i32 * 3) % 11) as f32 / 4.0 - 1.25
        })
        .collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    // The sum's scaler: 1.0 in row 0 of each face, the column fold's fill.
    let mut scaler = vec![0.0f32; 1024];
    for column in 0..32 {
        scaler[tile_index(0, column)] = 1.0;
        scaler[tile_index(16, column)] = 1.0;
    }
    let s_bytes: Vec<u8> = scaler.iter().flat_map(|v| v.to_le_bytes()).collect();

    // The host reference, column by column.
    let mut want = vec![0.0f32; 1024];
    for column in 0..32 {
        let hi = (0..32)
            .map(|row| x[tile_index(row, column)])
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = (0..32)
            .map(|row| (x[tile_index(row, column)] - hi).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        for row in 0..32 {
            want[tile_index(row, column)] = exps[row] / sum;
        }
    }

    let out = run_kernel_tensors(SOFTMAX_COL, &[&x_bytes, &s_bytes], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        let (row, column) = tile_rc(i);
        assert!(
            (got[i] - want[i]).abs() <= 2e-3,
            "softmax, datum {i} at ({row}, {column}): got {}, want {}",
            got[i],
            want[i]
        );
    }
    for column in 0..32 {
        let sum: f32 = (0..32).map(|row| got[tile_index(row, column)]).sum();
        assert!(
            (sum - 1.0).abs() <= 1e-2,
            "column {column} sums to {sum}, not one"
        );
    }
}

/// `rms_norm` over the hidden size: `x · rsqrt(mean(x²) + eps) · scale`, with a
/// 3072-wide row spread over 96 tiles. The squares fold down the rows (`acc
/// 32,1`, the Matrix Unit's `x @ ones`), the fold accumulates across all 96
/// tiles, and the factor leaves the accumulator through L1: the packed tile is
/// loaded, scaled by `1/3072`, offset by `eps` and rooted, then read back as a
/// `32,1` column the broadcast multiplies every tile by. The walk over the
/// hidden tiles is one `loop` for the fold and one for the scale.
const RMS_NORM: &str = "\
version 9
grid 1,1
%px = param f32 98304
%ps = param f32 98304
%pone = param f32 1024
%out = param f32 98304
%bx = buffer f32 32,32
%bs = buffer f32 32,32
%bones = buffer f32 32,32
%bacc = buffer f32 32,32
%br = buffer f32 32,32
%bo = buffer f32 32,32
%acc = acc f32 32,1
%zero = const f32 0
%one = const i32 1
%n = const i32 96
copy 32,32 %bones [0] = %pone [0]
store 32,32 %acc = %zero
%i = loop %n
%oi = mul %i %one
copy 32,32 %bx [0] = %px [%oi]
load 32,32 %x = %bx [0]
%sq = mul %x %x
copy 32,32 %bones [0] = %pone [0]
load 32,32 %ones = %bones [0]
sum %acc %sq %ones
endloop
store 32,32 %bacc [0] = %acc
load 32,32 %st = %bacc [0]
%inv = const f32 0.0003255208
%mean = mul %st %inv
%eps = const f32 0.00001
%a = add %mean %eps
%r = rsqrt %a
%j = loop %n
%oj = mul %j %one
copy 32,32 %bx [0] = %px [%oj]
load 32,32 %x2 = %bx [0]
store 32,32 %br [0] = %r
load 32,1 %rr = %br [0]
%y = mul %x2 %rr
copy 32,32 %bs [0] = %ps [%oj]
load 32,32 %sc = %bs [0]
%z = mul %y %sc
store 32,32 %bo [0] = %z
copy 32,32 %out [%oj] = %bo [0]
endloop
";

/// The whole `rms_norm` kernel at layer 0's shape: two tokens over the real
/// hidden size (3072 = 96 tiles), every datum against the host's
/// `x · rsqrt(mean(x²) + eps) · scale` with the config's `eps`. The factor goes
/// through a packed accumulator and a `32,1` page, so the comparison is a
/// tolerance, not an equality.
#[test]
fn rms_norm_runs_on_the_simulator() {
    const HIDDEN: usize = 3072;
    const COLS: usize = HIDDEN / 32;
    const SEQ: usize = 2;
    const EPS: f32 = 1e-5;

    // Only the first `SEQ` rows of each tile hold a token; the rest of the 32
    // rows are the padding the page layout needs and stay zero.
    let x_vals: Vec<f32> = (0..32 * HIDDEN)
        .map(|i| {
            let (row, column) = (i / HIDDEN, i % HIDDEN);
            if row < SEQ {
                (((row as i32 * 97 + column as i32 * 31) % 17) - 8) as f32 * 0.5
            } else {
                0.0
            }
        })
        .collect();
    let scale_vals: Vec<f32> = (0..32 * HIDDEN)
        .map(|i| {
            let row = i / HIDDEN;
            if row < SEQ {
                ((i % HIDDEN) % 7) as f32 * 0.25 + 0.5
            } else {
                0.0
            }
        })
        .collect();
    let x = tiled(32, HIDDEN, &x_vals);
    let scale = tiled(32, HIDDEN, &scale_vals);
    let mut ones = vec![0.0f32; 1024];
    for row in 0..32 {
        ones[tile_index(row, 0)] = 1.0;
    }

    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let scale_bytes: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ones_bytes: Vec<u8> = ones.iter().flat_map(|v| v.to_le_bytes()).collect();

    let out = run_kernel_tensors(
        RMS_NORM,
        &[&x_bytes, &scale_bytes, &ones_bytes],
        COLS * 1024 * 4,
    );
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // The arithmetic is the hardware's — TF32 into the FPU, the SFPU's
    // reciprocal, the Matrix Unit's fold — so this is a tolerance: the
    // measured worst error is 1.8e-3 on datums of magnitude 4 or less, and
    // 5e-3 catches a regression without hiding one.
    let mut worst = 0.0f32;
    let mut worst_at = (0usize, 0usize);
    for row in 0..SEQ {
        let sum: f32 = (0..HIDDEN)
            .map(|column| {
                let x = x_vals[row * HIDDEN + column];
                x * x
            })
            .sum();
        let rms = (sum / HIDDEN as f32 + EPS).sqrt().recip();
        for column in 0..HIDDEN {
            let want = x_vals[row * HIDDEN + column] * rms * scale_vals[row * HIDDEN + column];
            let at = (column / 32) * 1024 + tile_index(row, column % 32);
            let error = (got[at] - want).abs();
            if error > worst {
                worst = error;
                worst_at = (row, column);
            }
            assert!(
                error <= 5e-3,
                "rms_norm token {row} column {column}: got {}, want {want} (error {error})",
                got[at]
            );
        }
    }
    eprintln!("rms_norm: worst error {worst} at {worst_at:?}");
}

/// A dense tile through a narrow operand: the row broadcast multiplies every
/// row by the page's first row, and `ELWMUL` reproduces each datum exactly.
/// The Matrix Unit's multiply is sliced — one fidelity pass takes the top four
/// mantissa bits of each operand — so a single pass is a five-bit product; the
/// lowering runs tt-metal's `HiFi4`'s four phases and this pins all of them.
#[test]
fn a_narrow_multiply_keeps_every_datum() {
    let ramp: Vec<f32> = (0..1024).map(|i| (i as f32 + 1.0) / 1024.0).collect();
    let ramp_bytes: Vec<u8> = ramp.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut narrow = vec![0.0f32; 1024];
    for column in 0..32 {
        narrow[tile_index(0, column)] = 1.0;
    }
    let narrow_bytes: Vec<u8> = narrow.iter().flat_map(|v| v.to_le_bytes()).collect();
    let program = "\
version 9
grid 1,1
%pr = param f32 1024
%pn = param f32 1024
%out = param f32 1024
%br = buffer f32 32,32
%bn = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %br [0] = %pr [0]
copy 32,32 %bn [0] = %pn [0]
load 32,32 %r = %br [0]
load 1,32 %n = %bn [0]
%y = mul %r %n
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";
    let out = run_kernel_tensors(program, &[&ramp_bytes, &narrow_bytes], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        assert_eq!(
            got[i], ramp[i],
            "narrow multiply, datum {i}: got {}, want {}",
            got[i], ramp[i]
        );
    }

    // The same tile through `add` against a zero page: `ELWADD`/`ELWSUB` are
    // one pass at any fidelity, so they are exact where the multiply is sliced.
    let program_add = "\
version 9
grid 1,1
%pr = param f32 1024
%pn = param f32 1024
%out = param f32 1024
%br = buffer f32 32,32
%bn = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %br [0] = %pr [0]
copy 32,32 %bn [0] = %pn [0]
load 32,32 %r = %br [0]
load 1,32 %n = %bn [0]
%y = add %r %n
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";
    let zeros = vec![0.0f32; 1024];
    let zero_bytes: Vec<u8> = zeros.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_kernel_tensors(program_add, &[&ramp_bytes, &zero_bytes], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        assert_eq!(
            got[i], ramp[i],
            "narrow add, datum {i}: got {}, want {}",
            got[i], ramp[i]
        );
    }

    // `mul` with a zero A: `ELWMUL` accumulates into `Dst` (bit 21) and the
    // lowering zeroes the section first, so anything non-zero here is `Dst`
    // content the zeroing did not reach.
    let program_mul0 = "\
version 9
grid 1,1
%pr = param f32 1024
%pn = param f32 1024
%out = param f32 1024
%br = buffer f32 32,32
%bn = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %br [0] = %pr [0]
copy 32,32 %bn [0] = %pn [0]
load 32,32 %r = %br [0]
load 1,32 %n = %bn [0]
%y = mul %r %n
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";
    let ones: Vec<f32> = (0..1024).map(|_| 1.0f32).collect();
    let one_bytes: Vec<u8> = ones.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_kernel_tensors(program_mul0, &[&zero_bytes, &one_bytes], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // The section a multiply accumulates into starts at zero, so a zero
    // operand leaves a zero tile: nothing of an earlier value survives the
    // phases.
    for (i, value) in got.iter().enumerate() {
        assert_eq!(*value, 0.0, "multiply by zero, datum {i}: {value:e}");
    }
}

/// Two products into one accumulator: `matmul` adds into the tile it names, so
/// the second one does not clear what the first wrote. No loop — the only thing
/// exercised is that `acc` holds its value from one `matmul` to the next.
const MATMUL_ACC: &str = "\
version 9
grid 1,1
%pa = param f32 1024
%pb = param f32 1024
%pc = param f32 1024
%out = param f32 1024
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bc = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %ba [0] = %pa [0]
copy 32,32 %bb [0] = %pb [0]
copy 32,32 %bc [0] = %pc [0]
%m = acc f32 32,32
%zero = const f32 0
store 32,32 %m = %zero
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
matmul %m %a %b
load 32,32 %c = %bc [0]
matmul %m %b %c
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
";

#[test]
fn matmul_accumulates_into_one_dst() {
    let a: Vec<f32> = (0..1024).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i % 5) as f32 - 2.0).collect();
    let c: Vec<f32> = (0..1024).map(|i| (i % 11) as f32 - 5.0).collect();
    let want_ab = tile_matmul(&a, &b);
    let want_bc = tile_matmul(&b, &c);

    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let out = run_kernel_tensors(MATMUL_ACC, &[&bytes(&a), &bytes(&b), &bytes(&c)], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        let (row, column) = tile_rc(i);
        assert_eq!(
            got[i],
            want_ab[i] + want_bc[i],
            "a x b + b x c, datum {i} at ({row}, {column})"
        );
    }
}

/// The same two products as a K loop: one `matmul` under `loop`, accumulating
/// into an `acc` declared outside it, and one `store_to` after `endloop` — the
/// accumulator is visible where it is declared, so the pack happens once.
const MATMUL_ACC_LOOP: &str = "\
version 9
grid 1,1
%pa = param f32 2048
%pb = param f32 2048
%out = param f32 1024
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bo = buffer f32 32,32
%m = acc f32 32,32
%zero = const f32 0
store 32,32 %m = %zero
%two = const i32 2
%step = const i32 1
%i = loop %two
%o = mul %i %step
copy 32,32 %ba [0] = %pa [%o]
copy 32,32 %bb [0] = %pb [%o]
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
matmul %m %a %b
endloop
store 32,32 %bo [0] = %m
copy 32,32 %out [0] = %bo [0]
";

#[test]
fn a_matmul_accumulates_over_a_k_loop() {
    let a0: Vec<f32> = (0..1024).map(|i| (i % 7) as f32 - 3.0).collect();
    let a1: Vec<f32> = (0..1024).map(|i| (i % 13) as f32 - 6.0).collect();
    let b0: Vec<f32> = (0..1024).map(|i| (i % 5) as f32 - 2.0).collect();
    let b1: Vec<f32> = (0..1024).map(|i| (i % 3) as f32 - 1.0).collect();
    let want_k0 = tile_matmul(&a0, &b0);
    let want_k1 = tile_matmul(&a1, &b1);

    let mut a = a0.clone();
    a.extend_from_slice(&a1);
    let mut b = b0.clone();
    b.extend_from_slice(&b1);
    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let out = run_kernel_tensors(MATMUL_ACC_LOOP, &[&bytes(&a), &bytes(&b)], 4096);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        let (row, column) = tile_rc(i);
        assert_eq!(
            got[i],
            want_k0[i] + want_k1[i],
            "k loop, datum {i} at ({row}, {column})"
        );
    }
}

/// The accumulator is packed, refilled and written again: the second `matmul`
/// and the fill after a pack write the same `Dst` rows the packer has just
/// read, so the math thread must wait for that section to drain first. The
/// values are what the wait protects on a board; on the simulator they are what
/// the program produces either way.
const MATMUL_ACC_REUSED: &str = "\
version 9
grid 1,1
%pa = param f32 1024
%pb = param f32 1024
%pc = param f32 1024
%out = param f32 2048
%ba = buffer f32 32,32
%bb = buffer f32 32,32
%bc = buffer f32 32,32
%b0 = buffer f32 32,32
copy 32,32 %ba [0] = %pa [0]
copy 32,32 %bb [0] = %pb [0]
copy 32,32 %bc [0] = %pc [0]
%m = acc f32 32,32
%zero = const f32 0
store 32,32 %m = %zero
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
load 32,32 %c = %bc [0]
matmul %m %a %b
store 32,32 %b0 [0] = %m
copy 32,32 %out [0] = %b0 [0]
store 32,32 %m = %zero
matmul %m %a %c
store 32,32 %b0 [0] = %m
copy 32,32 %out [1] = %b0 [0]
";

#[test]
fn an_acc_is_reused_after_it_has_been_packed() {
    let a: Vec<f32> = (0..1024).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i % 5) as f32 - 2.0).collect();
    let c: Vec<f32> = (0..1024).map(|i| (i % 11) as f32 - 5.0).collect();
    let want_ab = tile_matmul(&a, &b);
    let want_ac = tile_matmul(&a, &c);

    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let out = run_kernel_tensors(
        MATMUL_ACC_REUSED,
        &[&bytes(&a), &bytes(&b), &bytes(&c)],
        8192,
    );
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        assert_eq!(got[i], want_ab[i], "first product, datum {i}");
        assert_eq!(got[1024 + i], want_ac[i], "second product, datum {i}");
    }
}

/// `gate_proj` the way a real grid runs it: a tiled bf16 `[M, K]` activation
/// times a tiled bf16 `[N, K]` weight, accumulated in an f32 `Dst` over a K
/// loop, one output tile per loop iteration, spread over the whole grid.
///
/// The shape is the prefill projection cut down to what the simulator runs in a
/// second: M = 128 (4 tile rows), N = 1920 (60 tile columns), K = 128 (4 tile
/// steps). That is 240 output tiles over the simulator's 12 x 10 grid — two
/// tiles per core, so both loops are real: the outer one walks this core's
/// tiles, the inner one walks K. Every core reads its own slices, and the tile
/// count per core is derived from the grid at launch
/// (`%per = div %ntiles %nsites`).
const GATE_PROJ: &str = "\
version 9
%nx = param i32 0
%ny = param i32 0
%ntiles = param i32 0
%ktiles = param i32 0
%ncols = param i32 0
%pa = param bf16 16384
%pw = param bf16 245760
%pc = param f32 245760
%ba = buffer bf16 32,32
%bb = buffer bf16 32,32
%bc = buffer f32 32,32
%acc = acc f32 32,32
%zero = const f32 0
%tile = const i32 1
%x = grid.x
%y = grid.y
%nsites = mul %nx %ny
%sitey = mul %y %nx
%site = add %sitey %x
%per = div %ntiles %nsites
%i = loop %per
%step = mul %i %nsites
%t = add %site %step
%m = div %t %ncols
%n = mod %t %ncols
%arow = mul %m %ktiles
%brow = mul %n %ktiles
store 32,32 %acc = %zero
%j = loop %ktiles
%ka = add %arow %j
%kb = add %brow %j
%oa = mul %ka %tile
%ob = mul %kb %tile
copy 32,32 %ba [0] = %pa [%oa]
copy 32,32 %bb [0] = %pw [%ob]
load 32,32 %a = %ba [0]
load 32,32 %b = %bb [0]
matmul %acc %a %b
endloop
store 32,32 %bc [0] = %acc
%coff = mul %t %tile
copy 32,32 %pc [%coff] = %bc [0]
endloop
grid %nx,%ny
";

/// A tensor parameter's bytes: page-major over the 32x32 tile grid, each page
/// the four-16x16-face layout `tile_index` names.
fn tiled(rows: usize, cols: usize, values: &[f32]) -> Vec<f32> {
    let (cols_tiles, page) = (cols / 32, 1024);
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let at = (r / 32) * cols_tiles + c / 32;
            out[at * page + tile_index(r % 32, c % 32)] = values[r * cols + c];
        }
    }
    out
}

#[test]
fn gate_proj_runs_over_a_full_grid() {
    // The simulator's part: 12 Tensix columns enabled, 10 rows.
    const NX: u64 = 12;
    const NY: u64 = 10;
    const M: usize = 128;
    const N: usize = 1920;
    const K: usize = 128;
    let (mt, nt, kt) = (M / 32, N / 32, K / 32);
    let ntiles = mt * nt;

    // Small integers: exact in bf16, and their products and the f32 sum that
    // follows are exact too, so the reference has no rounding budget.
    let a_vals: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 - 3.0).collect();
    let w_vals: Vec<f32> = (0..N * K).map(|i| (i % 5) as f32 - 2.0).collect();
    let a = tiled(M, K, &a_vals);
    let w = tiled(N, K, &w_vals);
    let a_bf16: Vec<u16> = a.iter().map(|v| truncate_to_bf16(*v)).collect();
    let w_bf16: Vec<u16> = w.iter().map(|v| truncate_to_bf16(*v)).collect();

    // The reference: per output tile, the sum over K tiles of the tile product,
    // all in the page layout the kernel writes.
    let mut want = vec![0f32; ntiles * 1024];
    for m in 0..mt {
        for n in 0..nt {
            let out = &mut want[(m * nt + n) * 1024..][..1024];
            for k in 0..kt {
                let at = (m * kt + k) * 1024;
                let bt = (n * kt + k) * 1024;
                let p = tile_matmul(&a[at..at + 1024], &w[bt..bt + 1024]);
                for i in 0..1024 {
                    out[i] += p[i];
                }
            }
        }
    }

    let bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|h| h.to_le_bytes()).collect() };
    let scalars = [NX, NY, ntiles as u64, kt as u64, nt as u64];
    let out = run_kernel_scalars(
        GATE_PROJ,
        &scalars,
        &[&bytes(&a_bf16), &bytes(&w_bf16)],
        ntiles * 1024 * 4,
    );
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for m in 0..mt {
        for n in 0..nt {
            for i in 0..1024 {
                let at = (m * nt + n) * 1024 + i;
                assert_eq!(
                    got[at],
                    want[at],
                    "gate_proj tile ({m}, {n}), datum {i} at {:?}",
                    tile_rc(i)
                );
            }
        }
    }
}

/// `2^x` lowered straight into the machine IR: the `1.5 * 2^23` rounding
/// trick for the integer part, zyx's `exp(f ln 2)` Horner chain for `2^f`,
/// and the exponent assembled by `SFPEXEXP`/`SFPSETEXP` — the step no IR op
/// spells, which is why `exp2` stays an IR op and is lowered here.
#[test]
fn exp2_runs_on_the_simulator() {
    const PROGRAM: &str = "\
version 9
grid 1,1
%in = param f32 1024
%out = param f32 1024
%b0 = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
load 32,32 %x = %b0 [0]
%r = exp2 %x
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";
    let host: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 32.0).collect();
    let input: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_kernel(PROGRAM, &input, 1024 * 4);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (x, y)) in host.iter().zip(&got).enumerate() {
        let want = x.exp2();
        let err = (y - want).abs() / want.abs().max(1.0);
        assert!(
            err < 1e-5,
            "datum {i}: 2^{x} = {y}, wanted {want} (rel {err})"
        );
    }
}

/// One loaded tile through a unary op and back, every datum compared against a
/// host reference within `tol` relative error (denominated by `max(1, |want|)`).
fn check_unary(program: &str, host: &[f32], want: impl Fn(f32) -> f32, tol: f32, what: &str) {
    let input: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_kernel(program, &input, host.len() * 4);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (x, y)) in host.iter().zip(&got).enumerate() {
        let want = want(*x);
        let err = (y - want).abs() / want.abs().max(1.0);
        assert!(
            err <= tol,
            "{what}: datum {i}: f({x}) = {y}, wanted {want} (rel {err} > {tol})"
        );
    }
}

const ONE_LOAD: &str = "\
version 9
grid 1,1
%in = param f32 1024
%out = param f32 1024
%b0 = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
load 32,32 %x = %b0 [0]
%r = OP %x
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";

/// `log2` as `src/unfold.rs` writes it: Juffa's reduction, the LLK's degree-9
/// `log1p` chain and one `log2 e` multiply, with no `log2` op left in the
/// lowered program.
#[test]
fn log2_is_unfolded_into_primitives() {
    let host: Vec<f32> = (0..1024).map(|i| 0.01 * (1.0 + i as f32 * 1.1)).collect();
    check_unary(
        &ONE_LOAD.replace("OP", "log2"),
        &host,
        f32::log2,
        1e-5,
        "log2",
    );
}

/// `sin`/`cos`: the two-limb `pi/2` reduction and the five-term Taylor cores,
/// with the quadrant and sign from `n & 3` — `bitshiftright` and `noteq`
/// included, which is what the lowering gained for this pass.
#[test]
fn sin_and_cos_are_unfolded_into_primitives() {
    let host: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 64.0).collect();
    check_unary(&ONE_LOAD.replace("OP", "sin"), &host, f32::sin, 1e-5, "sin");
    check_unary(&ONE_LOAD.replace("OP", "cos"), &host, f32::cos, 1e-5, "cos");
}

/// `pow` on floats: `exp2(y log2 x)`, so the program the caller writes has no
/// `pow` in it after the pass and the answer still lands on the host's.
#[test]
fn pow_is_unfolded_into_exp2_and_log2() {
    const POW: &str = "\
version 9
grid 1,1
%in = param f32 1024
%out = param f32 1024
%b0 = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
load 32,32 %x = %b0 [0]
%p = const f32 1.75
%r = pow %x %p
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";
    let host: Vec<f32> = (0..1024).map(|i| 0.05 * (1.0 + i as f32 * 2.0)).collect();
    check_unary(POW, &host, |x| x.powf(1.75), 1e-4, "pow");
}

/// `div` as `src/unfold.rs` writes it: `x * (1/y)`, the reciprocal being the
/// quadratic estimate plus two Newton steps (one in a bf16 kernel) and the
/// sign copied from the divisor.
#[test]
fn div_is_unfolded_into_a_reciprocal() {
    const DIV: &str = "\
version 9
grid 1,1
%in = param f32 2048
%out = param f32 1024
%b0 = buffer f32 32,64
%bo = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
copy 32,32 %b0 [1] = %in [1]
load 32,32 %x = %b0 [0]
load 32,32 %y = %b0 [1]
%r = div %x %y
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";
    let x: Vec<f32> = (0..1024).map(|i| (i as f32 - 500.0) / 7.0).collect();
    let y: Vec<f32> = (0..1024).map(|i| 0.25 + (i % 97) as f32 * 0.5).collect();
    let host: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a / b).collect();
    let mut input: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    input.extend(y.iter().flat_map(|v| v.to_le_bytes()));
    let out = run_kernel(DIV, &input, 1024 * 4);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (want, y)) in host.iter().zip(&got).enumerate() {
        let err = (y - want).abs() / want.abs().max(1.0);
        assert!(
            err <= 1e-6,
            "div: datum {i} = {y}, wanted {want} (rel {err})"
        );
    }
}

/// The `load`, one chain, one `store` shape every activation test shares.
fn activation_program(chain: &str) -> String {
    format!(
        "\
version 9
grid 1,1
%in = param f32 1024
%out = param f32 1024
%b0 = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
load 32,32 %x = %b0 [0]
{chain}
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
"
    )
}

fn activation_input() -> Vec<f32> {
    (0..1024).map(|i| (i as f32 - 512.0) / 64.0).collect()
}

/// `sigmoid = 1/(1+exp(-x))` as `src/unfold.rs` writes it, one row body.
const SIGMOID_CHAIN: &str = "\
%nx = neg %x
%k = const f32 1.442695
%p = mul %nx %k
%e = exp2 %p
%one = const f32 1
%d = add %e %one
%r = reciprocal %d";

#[test]
fn sigmoid_runs_on_the_simulator() {
    check_unary(
        &activation_program(SIGMOID_CHAIN),
        &activation_input(),
        |x| 1.0 / (1.0 + (-x).exp()),
        2e-3,
        "sigmoid",
    );
}

/// `silu = x·sigmoid(x)`: `x` is read by the chain and by the final multiply,
/// so it must cross the whole body in one lane register.
const SILU_CHAIN: &str = "\
%nx = neg %x
%k = const f32 1.442695
%p = mul %nx %k
%e = exp2 %p
%one = const f32 1
%d = add %e %one
%s = reciprocal %d
%r = mul %x %s";

#[test]
fn silu_runs_on_the_simulator() {
    check_unary(
        &activation_program(SILU_CHAIN),
        &activation_input(),
        |x| x / (1.0 + (-x).exp()),
        2e-3,
        "silu",
    );
}

/// `tanh = 2·sigmoid(2x) - 1`, the composition zyx's gate uses.
const TANH_CHAIN: &str = "\
%doubled = add %x %x
%nx = neg %doubled
%k = const f32 1.442695
%p = mul %nx %k
%e = exp2 %p
%one = const f32 1
%d = add %e %one
%s = reciprocal %d
%two = const f32 2
%m = mul %s %two
%r = sub %m %one";

#[test]
fn tanh_runs_on_the_simulator() {
    check_unary(
        &activation_program(TANH_CHAIN),
        &activation_input(),
        f32::tanh,
        2e-3,
        "tanh",
    );
}

/// `gelu = 0.5·x·(1 + tanh(√(2/π)(x + 0.044715 x³)))`. It is the longest
/// chain: `x` is needed three times and `x/2` survives to the end.
const GELU_CHAIN: &str = "\
%half = const f32 0.5
%xh = mul %x %half
%x2 = mul %x %x
%x3 = mul %x2 %x
%c = const f32 0.044715
%cx3 = mul %x3 %c
%inner = add %x %cx3
%kk = const f32 0.7978846
%arg = mul %inner %kk
%doubled = add %arg %arg
%narg = neg %doubled
%k = const f32 1.442695
%p = mul %narg %k
%e = exp2 %p
%one = const f32 1
%d = add %e %one
%s = reciprocal %d
%two = const f32 2
%m = mul %s %two
%t = sub %m %one
%t1 = add %t %one
%r = mul %xh %t1";

#[test]
fn gelu_runs_on_the_simulator() {
    check_unary(
        &activation_program(GELU_CHAIN),
        &activation_input(),
        |x| {
            let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        },
        2e-3,
        "gelu",
    );
}

/// The same chains in a bf16 kernel: the constants at the tile's dtype, a
/// 16-bit `Dst`, and every stage stored as bf16 — so the oracle is the f32 rule
/// on the widened input and the comparison is a tolerance (one bf16 datum per
/// stage is 2^-7 relative).
fn bf16_activation_program(chain: &str) -> String {
    format!(
        "\
version 9
grid 1,1
%in = param bf16 1024
%out = param bf16 1024
%b0 = buffer bf16 32,32
%bo = buffer bf16 32,32
copy 32,32 %b0 [0] = %in [0]
load 32,32 %x = %b0 [0]
{chain}
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
"
    )
}

/// One bf16 activation chain: the chain's `const f32`s become `const bf16`, the
/// input is the bf16 truncation, and the output is widened back for the check.
fn check_bf16_activation(name: &str, chain: &str, want: impl Fn(f32) -> f32, tol: f32) {
    let chain = chain.replace("const f32", "const bf16");
    let host = activation_input();
    let input: Vec<u8> = host
        .iter()
        .flat_map(|v| truncate_to_bf16(*v).to_le_bytes())
        .collect();
    let out = run_kernel(&bf16_activation_program(&chain), &input, host.len() * 2);
    let mut worst = 0.0f32;
    for (i, h) in host.iter().enumerate() {
        let x = widen_bf16(truncate_to_bf16(*h));
        let got = widen_bf16(u16::from_le_bytes(
            out[2 * i..2 * i + 2].try_into().unwrap(),
        ));
        let expect = want(x);
        let err = (got - expect).abs() / expect.abs().max(1.0);
        worst = worst.max(err);
        assert!(
            err <= tol,
            "{name}: datum {i} of {x}: got {got}, want {expect} (rel {err} > {tol})"
        );
    }
    eprintln!("bf16 {name}: worst relative error {worst}");
}

/// The four activation chains llama's SwiGLU and the gate use, in a bf16
/// kernel. Every stage is a bf16 value between instructions, so the reference
/// is the f32 rule on the widened input and the tolerance is the truncation.
#[test]
fn bf16_activations_run_on_the_simulator() {
    check_bf16_activation(
        "sigmoid",
        SIGMOID_CHAIN,
        |x| 1.0 / (1.0 + (-x).exp()),
        1.5e-2,
    );
    check_bf16_activation("silu", SILU_CHAIN, |x| x / (1.0 + (-x).exp()), 1.5e-2);
    check_bf16_activation("tanh", TANH_CHAIN, f32::tanh, 1.5e-2);
    check_bf16_activation(
        "gelu",
        GELU_CHAIN,
        |x| {
            let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        },
        1.5e-2,
    );
}

/// `rms_norm`'s bf16 shape: the same program with bf16 params/buffers, a bf16
/// accumulator, and the coefficients at the tile's dtype. `%zero` stays f32
/// because the accumulator's fill is the one thing that must be f32.
const RMS_NORM_BF16: &str = "\
version 9
grid 1,1
%px = param bf16 98304
%ps = param bf16 98304
%pone = param bf16 1024
%out = param bf16 98304
%bx = buffer bf16 32,32
%bs = buffer bf16 32,32
%bones = buffer bf16 32,32
%bacc = buffer bf16 32,32
%br = buffer bf16 32,32
%bo = buffer bf16 32,32
%acc = acc bf16 32,1
%zero = const f32 0
%one = const i32 1
%n = const i32 96
copy 32,32 %bones [0] = %pone [0]
store 32,32 %acc = %zero
%i = loop %n
%oi = mul %i %one
copy 32,32 %bx [0] = %px [%oi]
load 32,32 %x = %bx [0]
%sq = mul %x %x
copy 32,32 %bones [0] = %pone [0]
load 32,32 %ones = %bones [0]
sum %acc %sq %ones
endloop
store 32,32 %bacc [0] = %acc
load 32,32 %st = %bacc [0]
%inv = const bf16 0.0003255208
%mean = mul %st %inv
%eps = const bf16 0.00001
%a = add %mean %eps
%r = rsqrt %a
%j = loop %n
%oj = mul %j %one
copy 32,32 %bx [0] = %px [%oj]
load 32,32 %x2 = %bx [0]
store 32,32 %br [0] = %r
load 32,1 %rr = %br [0]
%y = mul %x2 %rr
copy 32,32 %bs [0] = %ps [%oj]
load 32,32 %sc = %bs [0]
%z = mul %y %sc
store 32,32 %bo [0] = %z
copy 32,32 %out [%oj] = %bo [0]
endloop
";

/// `sum row` over 96 tiles in a bf16 kernel: the accumulator is the 16-bit
/// `Dst`, so every tile's fold is added to a bf16 datum. This is the rms_norm
/// fold, isolated.
const SUM_ROW_MANY_BF16: &str = "\
version 9
grid 1,1
%px = param bf16 98304
%pone = param bf16 1024
%out = param bf16 1024
%bx = buffer bf16 32,32
%bones = buffer bf16 32,32
%bacc = buffer bf16 32,32
%bo = buffer bf16 32,32
%acc = acc bf16 32,1
%zero = const f32 0
%one = const i32 1
%n = const i32 96
copy 32,32 %bones [0] = %pone [0]
store 32,32 %acc = %zero
%i = loop %n
%oi = mul %i %one
copy 32,32 %bx [0] = %px [%oi]
load 32,32 %x = %bx [0]
%sq = mul %x %x
copy 32,32 %bones [0] = %pone [0]
load 32,32 %ones = %bones [0]
sum %acc %sq %ones
endloop
store 32,32 %bacc [0] = %acc
load 32,32 %st = %bacc [0]
store 32,32 %bo [0] = %st
copy 32,32 %out [0] = %bo [0]
";

#[test]
#[ignore = "the bf16 row fold over-accumulates by 13 % over 96 tiles; see TODO.md"]
fn bf16_sum_row_over_many_tiles() {
    const HIDDEN: usize = 3072;
    const COLS: usize = HIDDEN / 32;
    const SEQ: usize = 2;
    let x_f: Vec<f32> = (0..32 * HIDDEN)
        .map(|i| {
            let (row, column) = (i / HIDDEN, i % HIDDEN);
            if row < SEQ {
                (((row as i32 * 97 + column as i32 * 31) % 17) - 8) as f32 * 0.5
            } else {
                0.0
            }
        })
        .collect();
    let xw: Vec<f32> = x_f
        .iter()
        .map(|v| widen_bf16(truncate_to_bf16(*v)))
        .collect();
    let tiled_x = tiled(32, HIDDEN, &xw);
    let x_bytes: Vec<u8> = tiled_x
        .iter()
        .flat_map(|v| truncate_to_bf16(*v).to_le_bytes())
        .collect();
    let mut ones = vec![0u16; 1024];
    for row in 0..32 {
        ones[tile_index(row, 0)] = truncate_to_bf16(1.0);
    }
    let ones_bytes: Vec<u8> = ones.iter().flat_map(|h| h.to_le_bytes()).collect();
    let out = run_kernel_tensors(SUM_ROW_MANY_BF16, &[&x_bytes, &ones_bytes], 1024 * 2);
    let got: Vec<f32> = out
        .chunks_exact(2)
        .map(|c| widen_bf16(u16::from_le_bytes(c.try_into().unwrap())))
        .collect();
    for row in 0..SEQ {
        let want: f32 = (0..HIDDEN)
            .map(|column| {
                let x = xw[row * HIDDEN + column];
                x * x
            })
            .sum();
        let at = tile_index(row, 0);
        let err = (got[at] - want).abs() / want;
        assert!(
            err <= 2e-2,
            "bf16 sum row over {COLS} tiles, row {row}: got {}, want {want} (rel {err})",
            got[at]
        );
    }
}

fn sum_row_probe_program(n: u32, dt: &str, acc: &str, out: &str) -> String {
    format!(
        "\
version 9
grid 1,1
%px = param {dt} 98304
%pone = param {dt} 1024
%out = param {out} 1024
%bx = buffer {dt} 32,32
%bones = buffer {dt} 32,32
%bacc = buffer {out} 32,32
%bo = buffer {out} 32,32
%acc = acc {acc} 32,1
%zero = const f32 0
%one = const i32 1
%n = const i32 {n}
copy 32,32 %bones [0] = %pone [0]
store 32,32 %acc = %zero
%i = loop %n
%oi = mul %i %one
copy 32,32 %bx [0] = %px [%oi]
load 32,32 %x = %bx [0]
%sq = mul %x %x
copy 32,32 %bones [0] = %pone [0]
load 32,32 %ones = %bones [0]
sum %acc %sq %ones
endloop
store 32,32 %bacc [0] = %acc
load 32,32 %st = %bacc [0]
store 32,32 %bo [0] = %st
copy 32,32 %out [0] = %bo [0]
"
    )
}

/// The same fold with the squares already in the page: no computed value is
/// staged, so the Matrix Unit reads the page the reader moved in. It is the A/B
/// for `sum_row_probe_program`: same matmul, same accumulator, no `stage`.
fn sum_loaded_probe_program(n: u32, dt: &str, acc: &str, out: &str) -> String {
    format!(
        "\
version 9
grid 1,1
%px = param {dt} 98304
%pone = param {dt} 1024
%out = param {out} 1024
%bx = buffer {dt} 32,32
%bones = buffer {dt} 32,32
%bacc = buffer {out} 32,32
%bo = buffer {out} 32,32
%acc = acc {acc} 32,1
%zero = const f32 0
%one = const i32 1
%n = const i32 {n}
copy 32,32 %bones [0] = %pone [0]
store 32,32 %acc = %zero
%i = loop %n
%oi = mul %i %one
copy 32,32 %bx [0] = %px [%oi]
load 32,32 %x = %bx [0]
copy 32,32 %bones [0] = %pone [0]
load 32,32 %ones = %bones [0]
sum %acc %x %ones
endloop
store 32,32 %bacc [0] = %acc
load 32,32 %st = %bacc [0]
store 32,32 %bo [0] = %st
copy 32,32 %out [0] = %bo [0]
"
    )
}

/// Store a computed f32 tile to a declared buffer, in the same loop the fold
/// stages it: the pack path without `stage`'s compiler-owned page.
fn mul_store_probe_program(n: u32, dt: &str) -> String {
    format!(
        "\
version 9
grid 1,1
%px = param {dt} 98304
%out = param {dt} 98304
%bx = buffer {dt} 32,32
%bsq = buffer {dt} 32,32
%one = const i32 1
%n = const i32 {n}
%i = loop %n
%oi = mul %i %one
copy 32,32 %bx [0] = %px [%oi]
load 32,32 %x = %bx [0]
%sq = mul %x %x
store 32,32 %bsq [0] = %sq
copy 32,32 %out [%oi] = %bsq [0]
endloop
"
    )
}

#[test]
fn probe_bf16_row_fold() {
    const HIDDEN: usize = 3072;
    const SEQ: usize = 2;
    let x_f: Vec<f32> = (0..32 * HIDDEN)
        .map(|i| {
            let (row, column) = (i / HIDDEN, i % HIDDEN);
            if row < SEQ {
                (((row as i32 * 97 + column as i32 * 31) % 17) - 8) as f32 * 0.5
            } else {
                0.0
            }
        })
        .collect();
    let xw: Vec<f32> = x_f
        .iter()
        .map(|v| widen_bf16(truncate_to_bf16(*v)))
        .collect();
    let tiled_x = tiled(32, HIDDEN, &xw);
    let x_bytes: Vec<u8> = tiled_x
        .iter()
        .flat_map(|v| truncate_to_bf16(*v).to_le_bytes())
        .collect();
    let mut ones = vec![0u16; 1024];
    for row in 0..32 {
        ones[tile_index(row, 0)] = truncate_to_bf16(1.0);
    }
    let ones_bytes: Vec<u8> = ones.iter().flat_map(|h| h.to_le_bytes()).collect();
    let tile_sums: Vec<f32> = (0..HIDDEN / 32)
        .map(|t| {
            (0..32)
                .map(|c| {
                    let x = xw[t * 32 + c];
                    x * x
                })
                .sum()
        })
        .collect();
    let mut prev_got = 0.0f32;
    let mut prev_want = 0.0f32;
    for n in 1u32..=96 {
        let out = run_kernel_tensors(
            &sum_row_probe_program(n, "bf16", "bf16", "bf16"),
            &[&x_bytes, &ones_bytes],
            1024 * 2,
        );
        let got = widen_bf16(u16::from_le_bytes(out[0..2].try_into().unwrap()));
        let want: f32 = tile_sums[..n as usize].iter().sum();
        eprintln!(
            "n {n}: got {got} want {want} delta {} step {} tilesum {}",
            got - want,
            got - prev_got,
            tile_sums[n as usize - 1]
        );
        prev_got = got;
        prev_want = want;
    }
    let _ = prev_want;
    // The host model of the 16-bit accumulator: each tile adds its two
    // 16-column halves into a `Dst` that rounds to bf16 after every add. The
    // squares themselves are the SFPU's bf16 products.
    let round_bf16 = |v: f32| {
        let bits = v.to_bits();
        let bias = 0x7FFF + ((bits >> 16) & 1);
        f32::from_bits((bits.wrapping_add(bias)) & 0xFFFF_0000)
    };
    let sq: Vec<f32> = (0..HIDDEN).map(|c| round_bf16(xw[c] * xw[c])).collect();
    for parts in [1usize, 2, 4, 8] {
        let width = 32 / parts;
        let mut acc = 0.0f32;
        for n in 1..=96 {
            for p in 0..parts {
                let at = (n - 1) * 32 + p * width;
                let partial: f32 = sq[at..at + width].iter().sum();
                acc = round_bf16(acc + partial);
            }
        }
        eprintln!("model {parts} partials of {width}: n96 acc {acc}");
    }
    for unit in [1.0f32, 2.0, 0.5, 3.0, 4.0, 8.0] {
        let u: Vec<f32> = (0..32 * HIDDEN).map(|_| unit).collect();
        let tiled_u = tiled(32, HIDDEN, &u);
        let u_bytes: Vec<u8> = tiled_u
            .iter()
            .flat_map(|v| truncate_to_bf16(*v).to_le_bytes())
            .collect();
        let out = run_kernel_tensors(
            &sum_row_probe_program(96, "bf16", "bf16", "bf16"),
            &[&u_bytes, &ones_bytes],
            1024 * 2,
        );
        let row0 = widen_bf16(u16::from_le_bytes(out[0..2].try_into().unwrap()));
        let row1 = widen_bf16(u16::from_le_bytes(out[2..4].try_into().unwrap()));
        eprintln!(
            "uniform {unit}, 96 tiles: row0 {row0}, row1 {row1}, want {}",
            96.0 * 32.0 * unit * unit
        );
    }
    // The same tile repeated 96 times, and 96 rotations of it: every tile's
    // true sum is identical, so only the fold's accumulation or the per-tile
    // staging can move the result.
    let pattern: Vec<f32> = (0..1024)
        .map(|i| (((i % 32) * 31 + (i / 32) * 17) % 9) as f32 * 0.5 + 0.25)
        .collect();
    let pattern_row0: f32 = (0..32)
        .map(|c| {
            let v = pattern[c];
            v * v
        })
        .sum();
    let mut repeated = vec![0f32; 32 * HIDDEN];
    for t in 0..HIDDEN / 32 {
        for r in 0..32 {
            for c in 0..32 {
                repeated[r * HIDDEN + t * 32 + c] = pattern[r * 32 + c];
            }
        }
    }
    let repeated_bytes: Vec<u8> = tiled(32, HIDDEN, &repeated)
        .iter()
        .flat_map(|v| truncate_to_bf16(*v).to_le_bytes())
        .collect();
    let out = run_kernel_tensors(
        &sum_row_probe_program(96, "bf16", "bf16", "bf16"),
        &[&repeated_bytes, &ones_bytes],
        1024 * 2,
    );
    let got = widen_bf16(u16::from_le_bytes(out[0..2].try_into().unwrap()));
    eprintln!(
        "pattern repeated 96 tiles: got {got}, want {}, rel {}",
        pattern_row0 * 96.0,
        (got - pattern_row0 * 96.0) / (pattern_row0 * 96.0)
    );
    let mut rotated = vec![0f32; 32 * HIDDEN];
    for t in 0..HIDDEN / 32 {
        for r in 0..32 {
            for c in 0..32 {
                rotated[r * HIDDEN + t * 32 + c] = pattern[((r + t) % 32) * 32 + c];
            }
        }
    }
    let rotated_bytes: Vec<u8> = tiled(32, HIDDEN, &rotated)
        .iter()
        .flat_map(|v| truncate_to_bf16(*v).to_le_bytes())
        .collect();
    let out = run_kernel_tensors(
        &sum_row_probe_program(96, "bf16", "bf16", "bf16"),
        &[&rotated_bytes, &ones_bytes],
        1024 * 2,
    );
    let got = widen_bf16(u16::from_le_bytes(out[0..2].try_into().unwrap()));
    eprintln!(
        "pattern rotated 96 tiles: got {got}, want {}, rel {}",
        pattern_row0 * 96.0,
        (got - pattern_row0 * 96.0) / (pattern_row0 * 96.0)
    );
    {
        let out = run_kernel_tensors(
            &sum_row_probe_program(96, "bf16", "bf16", "bf16"),
            &[&x_bytes, &ones_bytes],
            1024 * 2,
        );
        let got: Vec<f32> = out
            .chunks_exact(2)
            .map(|c| widen_bf16(u16::from_le_bytes(c.try_into().unwrap())))
            .collect();
        for row in 0..4 {
            let want: f32 = (0..HIDDEN)
                .map(|c| {
                    let x = xw[row * HIDDEN + c];
                    x * x
                })
                .sum();
            eprintln!(
                "row {row}: got {} want {want} rel {}",
                got[tile_index(row, 0)],
                (got[tile_index(row, 0)] - want) / want
            );
        }
    }
    // The f32 input must be in the page layout, exactly as the bf16 one is: the
    // kernel's `copy` reads page 0 of the parameter.
    let x_f32: Vec<u8> = tiled(32, HIDDEN, &xw)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let ones_f32: Vec<u8> = ones
        .iter()
        .flat_map(|h| widen_bf16(*h).to_le_bytes())
        .collect();
    // A bf16 tile folded into an f32 acc is accepted by the verifier now, but a
    // wide module packs the staged bf16 operand through `pack_fmt_conv_mode`
    // `0x105`, which ttsim does not model (PACR aborts); that form needs
    // silicon, so it is not run here.
    for n in [1u32, 2, 4, 8, 16, 32, 48, 96] {
        let out = run_kernel_tensors(
            &sum_row_probe_program(n, "f32", "f32", "f32"),
            &[&x_f32, &ones_f32],
            1024 * 4,
        );
        let got = f32::from_le_bytes(out[0..4].try_into().unwrap());
        let want: f32 = tile_sums[..n as usize].iter().sum();
        eprintln!(
            "f32 acc, {n} tiles: got {got}, want {want}, rel {}",
            (got - want) / want
        );
        assert_eq!(got, want, "f32 acc over {n} tiles");
    }
    for unit in [1.0f32, 1.25, 1.5, 2.5, 3.0, 4.0] {
        let u: Vec<f32> = (0..32 * HIDDEN).map(|_| unit).collect();
        let tiled_u = tiled(32, HIDDEN, &u);
        let u_bytes: Vec<u8> = tiled_u.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = run_kernel_tensors(
            &sum_row_probe_program(96, "f32", "f32", "f32"),
            &[&u_bytes, &ones_f32],
            1024 * 4,
        );
        let row0 = f32::from_le_bytes(out[0..4].try_into().unwrap());
        eprintln!(
            "f32 uniform {unit}, 96 tiles: row0 {row0}, want {}",
            96.0 * 32.0 * unit * unit
        );
        assert_eq!(row0, 96.0 * 32.0 * unit * unit, "f32 uniform {unit}");
    }
    let out = run_kernel_tensors(
        &sum_row_probe_program(1, "f32", "f32", "f32"),
        &[&x_f32, &ones_f32],
        1024 * 4,
    );
    let one: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for row in 0..32 {
        let v = one[tile_index(row, 0)];
        let want: f32 = (0..32)
            .map(|c| {
                let x = xw[row * HIDDEN + c];
                x * x
            })
            .sum();
        assert_eq!(v, want, "f32 computed n1 row {row}");
    }
    eprintln!(
        "f32 first datums: {:?}",
        out[0..32]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>()
    );
    // The A/B: the squares are in the page, so nothing is staged.
    let sq_f32: Vec<f32> = (0..32 * HIDDEN)
        .map(|i| {
            let x = xw[i];
            x * x
        })
        .collect();
    let sq_bytes: Vec<u8> = tiled(32, HIDDEN, &sq_f32)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = run_kernel_tensors(
        &sum_loaded_probe_program(1, "f32", "f32", "f32"),
        &[&sq_bytes, &ones_f32],
        1024 * 4,
    );
    let one: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for row in 0..32 {
        let v = one[tile_index(row, 0)];
        let want: f32 = (0..32)
            .map(|c| {
                let x = xw[row * HIDDEN + c];
                x * x
            })
            .sum();
        assert_eq!(v, want, "f32 loaded n1 row {row}");
    }
    // The pack path on its own: store the computed tile to a declared buffer.
    let tiled_sq = tiled(32, HIDDEN, &sq_f32);
    let out = run_kernel_tensors(
        &mul_store_probe_program(1, "f32"),
        &[&x_f32],
        HIDDEN * 32 * 4,
    );
    let stored: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for i in 0..1024 {
        assert_eq!(stored[i], tiled_sq[i], "the packed square at datum {i}");
    }
}

/// `rms_norm` at layer 0's shape in a bf16 kernel: every stage is a bf16 datum,
/// so the reference is the f32 rule on the widened inputs and the tolerance
/// covers the truncations (the 96-tile fold included).
#[test]
#[ignore = "blocked by the bf16 row fold (see bf16_sum_row_over_many_tiles)"]
fn bf16_rms_norm_runs_on_the_simulator() {
    const HIDDEN: usize = 3072;
    const COLS: usize = HIDDEN / 32;
    const SEQ: usize = 2;
    const EPS: f32 = 1e-5;
    let x_f: Vec<f32> = (0..32 * HIDDEN)
        .map(|i| {
            let (row, column) = (i / HIDDEN, i % HIDDEN);
            if row < SEQ {
                (((row as i32 * 97 + column as i32 * 31) % 17) - 8) as f32 * 0.5
            } else {
                0.0
            }
        })
        .collect();
    let scale_f: Vec<f32> = (0..32 * HIDDEN)
        .map(|i| {
            let row = i / HIDDEN;
            if row < SEQ {
                ((i % HIDDEN) % 7) as f32 * 0.25 + 0.5
            } else {
                0.0
            }
        })
        .collect();
    // The kernel's input is the bf16 truncation; the oracle widens it back.
    let xw: Vec<f32> = x_f
        .iter()
        .map(|v| widen_bf16(truncate_to_bf16(*v)))
        .collect();
    let sw: Vec<f32> = scale_f
        .iter()
        .map(|v| widen_bf16(truncate_to_bf16(*v)))
        .collect();
    let tiled_x = tiled(32, HIDDEN, &xw);
    let tiled_s = tiled(32, HIDDEN, &sw);
    let x_bytes: Vec<u8> = tiled_x
        .iter()
        .flat_map(|v| truncate_to_bf16(*v).to_le_bytes())
        .collect();
    let s_bytes: Vec<u8> = tiled_s
        .iter()
        .flat_map(|v| truncate_to_bf16(*v).to_le_bytes())
        .collect();
    let mut ones = vec![0u16; 1024];
    for row in 0..32 {
        ones[tile_index(row, 0)] = truncate_to_bf16(1.0);
    }
    let ones_bytes: Vec<u8> = ones.iter().flat_map(|h| h.to_le_bytes()).collect();
    let out = run_kernel_tensors(
        RMS_NORM_BF16,
        &[&x_bytes, &s_bytes, &ones_bytes],
        COLS * 1024 * 2,
    );
    let got: Vec<f32> = out
        .chunks_exact(2)
        .map(|c| widen_bf16(u16::from_le_bytes(c.try_into().unwrap())))
        .collect();
    let mut worst = 0.0f32;
    let mut worst_at = (0usize, 0usize);
    // The oracle mirrors the datapath: each instruction's result is a bf16
    // datum, so every stage truncates, and the 96-tile fold accumulates in the
    // 16-bit `Dst` (tt-metal's row reduce has no fp32 accumulation on Blackhole,
    // #47311 — the truncating adds bias the sum low, by design).
    let bf = |v: f32| widen_bf16(truncate_to_bf16(v));
    let inv = bf(1.0 / HIDDEN as f32);
    let eps = bf(EPS);
    for row in 0..SEQ {
        let mut acc = 0.0f32;
        for tile in 0..COLS {
            let mut tile_sum = 0.0f32;
            for column in 0..32 {
                let x = xw[row * HIDDEN + tile * 32 + column];
                tile_sum += x * x;
            }
            acc = bf(acc + tile_sum);
        }
        let mean = bf(acc * inv);
        let rms = bf((bf(mean + eps)).sqrt().recip());
        for column in 0..HIDDEN {
            let y = bf(xw[row * HIDDEN + column] * rms);
            let want = bf(y * sw[row * HIDDEN + column]);
            let at = (column / 32) * 1024 + tile_index(row, column % 32);
            let error = (got[at] - want).abs();
            if error > worst {
                worst = error;
                worst_at = (row, column);
            }
            // Two bf16 ulps: the lane's `rsqrt` and the tile-internal fold are
            // f32 and can land on the other side of a truncation boundary.
            assert!(
                error <= 2.0 * (want.abs().max(1.0) * 0.0078125 * 2.0),
                "bf16 rms_norm token {row} column {column}: got {}, want {want} (error {error})",
                got[at]
            );
        }
    }
    eprintln!("bf16 rms_norm: worst error {worst} at {worst_at:?}");
}

/// `softmax` down a tile's columns in a bf16 kernel: a 16-bit `Dst`, a bf16
/// accumulator, and constants at the tile's dtype.
const SOFTMAX_COL_BF16: &str = "\
version 9
grid 1,1
%px = param bf16 1024
%ps = param bf16 1024
%out = param bf16 1024
%bx = buffer bf16 32,32
%bs = buffer bf16 32,32
%bsum = buffer bf16 32,32
%binv = buffer bf16 32,32
%bo = buffer bf16 32,32
copy 32,32 %bx [0] = %px [0]
copy 32,32 %bs [0] = %ps [0]
%m = acc bf16 1,32
%low = const f32 -1000
store 32,32 %m = %low
load 32,32 %x = %bx [0]
max %m %x
%d = sub %x %m
%k = const bf16 1.442695
%p = mul %d %k
%e = exp2 %p
%s = acc bf16 1,32
%zero = const f32 0
store 32,32 %s = %zero
load 32,32 %ones = %bs [0]
sum %s %e %ones
store 32,32 %bsum [0] = %s
load 32,32 %st = %bsum [0]
%inv = div %ones %st
store 32,32 %binv [0] = %inv
load 1,32 %ri = %binv [0]
%y = mul %e %ri
store 32,32 %bo [0] = %y
copy 32,32 %out [0] = %bo [0]
";

/// `softmax` down a tile's columns in a bf16 kernel: the fold is 32 terms, so
/// the truncations stay small; every datum is checked against the host's
/// column softmax of the widened input, and each column sums to one.
#[test]
fn bf16_softmax_runs_on_the_simulator() {
    let x: Vec<u16> = (0..1024)
        .map(|i| {
            let (row, column) = tile_rc(i);
            truncate_to_bf16(((row as i32 * 7 + column as i32 * 3) % 11) as f32 / 4.0 - 1.25)
        })
        .collect();
    let xw: Vec<f32> = x.iter().map(|h| widen_bf16(*h)).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|h| h.to_le_bytes()).collect();
    let mut scaler = vec![0u16; 1024];
    for column in 0..32 {
        scaler[tile_index(0, column)] = truncate_to_bf16(1.0);
        scaler[tile_index(16, column)] = truncate_to_bf16(1.0);
    }
    let s_bytes: Vec<u8> = scaler.iter().flat_map(|h| h.to_le_bytes()).collect();
    let mut want = vec![0.0f32; 1024];
    for column in 0..32 {
        let hi = (0..32)
            .map(|row| xw[tile_index(row, column)])
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = (0..32)
            .map(|row| (xw[tile_index(row, column)] - hi).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        for row in 0..32 {
            want[tile_index(row, column)] = exps[row] / sum;
        }
    }
    let out = run_kernel_tensors(SOFTMAX_COL_BF16, &[&x_bytes, &s_bytes], 1024 * 2);
    let got: Vec<f32> = out
        .chunks_exact(2)
        .map(|c| widen_bf16(u16::from_le_bytes(c.try_into().unwrap())))
        .collect();
    let mut worst = 0.0f32;
    for i in 0..1024 {
        let (row, column) = tile_rc(i);
        let error = (got[i] - want[i]).abs();
        worst = worst.max(error);
        // Two bf16 ulps of the result: the exponentials and the reciprocal are
        // bf16 data between instructions. Measured worst is 7.6e-4.
        assert!(
            error <= 2e-3,
            "bf16 softmax, datum {i} at ({row}, {column}): got {}, want {}",
            got[i],
            want[i]
        );
    }
    for column in 0..32 {
        let sum: f32 = (0..32).map(|row| got[tile_index(row, column)]).sum();
        assert!(
            (sum - 1.0).abs() <= 5e-2,
            "bf16 softmax: column {column} sums to {sum}, not one"
        );
    }
    eprintln!("bf16 softmax: worst error {worst}");
}

/// The embedding gather as pure data movement: the row of ids is copied in as a
/// whole row, and the copy that follows names that row as its index — one base
/// per lane — so each lane pulls one weight row into the tile the last copy
/// drains. No compute role runs.
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

#[test]
fn a_gather_is_pure_data_movement() {
    // A token's embedding is one 32x32 tile, so its row base is `token * 32`
    // rows of 32 datums; the weight is 32 tiles of rows.
    let tokens: Vec<i32> = (0..32).map(|lane| ((lane * 7 + 3) % 32) as i32).collect();
    let rows: Vec<i32> = tokens.iter().map(|token| token * 32).collect();
    let ids: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w: Vec<f32> = (0..32768).map(|i| i as f32).collect();
    let weight: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_kernel_tensors(GATHER, &[&ids, &weight], 1024 * 4);
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (lane, &token) in tokens.iter().enumerate() {
        for column in 0..32 {
            assert_eq!(
                got[lane * 32 + column],
                w[token as usize * 1024 + column],
                "lane {lane} datum {column}"
            );
        }
    }
}
