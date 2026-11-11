// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The runtime: a device, DRAM buffers, host transfers, and a launch.
//!
//! A launch is synchronous. The three compute programs are not started yet, so
//! every module that can be compiled today is a boundary program or a pair of
//! them; such a module runs on RISCV B through [`Device::run_program`], the
//! reader first and then the writer, which is the program's own order. A module
//! with non-empty compute programs is refused rather than run half-way.

use std::sync::Mutex;

use crate::compile::Module;
use crate::device::Device;
use crate::ir::Error;
use tt_isa::noc::{Noc0, NocCoord};

/// The one open device, if [`init`] has been called.
static DEVICE: Mutex<Option<Device>> = Mutex::new(None);

/// Bytes of the staged-program area a program may use before its arguments.
const PROGRAM_BYTES: u32 = 0x2000;

/// Open the device. Idempotent; the simulator is initialized once per process.
pub fn init() -> Result<(), Error> {
    let mut device = lock()?;
    if device.is_none() {
        *device = Some(Device::open()?);
    }
    Ok(())
}

/// Close the device. DRAM allocations from this device are gone; a later
/// [`init`] opens a fresh one over the same simulator.
pub fn release() -> Result<(), Error> {
    *lock()? = None;
    Ok(())
}

/// Reserve `bytes` of DRAM and return its address.
pub fn alloc(bytes: usize) -> Result<u64, Error> {
    with_device(|d| d.alloc_dram(bytes))
}

/// Copy host bytes into DRAM at `address`.
pub fn h2d(address: u64, bytes: &[u8]) -> Result<(), Error> {
    with_device(|d| d.dram_write(address, bytes))
}

/// Copy DRAM at `address` into host bytes.
pub fn d2h(address: u64, out: &mut [u8]) -> Result<(), Error> {
    with_device(|d| d.dram_read(address, out))
}

/// Why a launch did not happen, or did not finish.
#[derive(Debug)]
pub enum LaunchError {
    /// The caller's parameter count is not the module's.
    Argument(String),
    /// The module is valid, but this build cannot run it.
    Unsupported(String),
    /// The device failed.
    Device(Error),
}

/// Run a module over its grid with one DRAM address per parameter.
///
/// The launch arguments are built here, not by the caller: the grid, this
/// tile's high word, and each parameter's `(low, high)` pair. A count that
/// disagrees with the module is refused.
pub fn launch(module: &Module, params: &[u64]) -> Result<(), LaunchError> {
    if module.params as usize != params.len() {
        return Err(LaunchError::Argument(format!(
            "the module takes {} parameters, {n} were given",
            module.params,
            n = params.len()
        )));
    }
    if module.grid != [1, 1] {
        return Err(LaunchError::Unsupported(format!(
            "a launch over a {}x{} grid is not implemented",
            module.grid[0], module.grid[1]
        )));
    }
    if module.cores.iter().any(|program| !program.is_empty()) {
        return Err(LaunchError::Unsupported(
            "this module has compute programs; the three compute cores are not launched yet".into(),
        ));
    }
    if module.reader.is_empty() && module.writer.is_empty() {
        return Err(LaunchError::Unsupported("the module has no program to run".into()));
    }
    with_device(|device| run(device, module, params)).map_err(LaunchError::Device)
}

fn run(device: &mut Device, module: &Module, params: &[u64]) -> Result<(), Error> {
    let tile: NocCoord<Noc0> = device.tile();
    let mut args: Vec<u32> = vec![module.grid[0], module.grid[1], 0, me_hi(tile)];
    let hi = device.dram_hi();
    for &address in params {
        let low = u32::try_from(address).map_err(|_| {
            Error(format!(
                "parameter address {address:#x} does not fit in a word"
            ))
        })?;
        args.push(low);
        args.push(hi);
    }

    let program_at = u32::try_from(crate::device::PROGRAM_AREA)
        .map_err(|_| Error("the staged-program area does not fit in a word".into()))?;
    let args_at = program_at + PROGRAM_BYTES;
    for (name, words) in [("reader", &module.reader), ("writer", &module.writer)] {
        if words.is_empty() {
            continue;
        }
        if 4 * words.len() as u32 > PROGRAM_BYTES {
            return Err(Error(format!(
                "the {name} program ({} words) does not fit in {PROGRAM_BYTES:#x} bytes",
                words.len()
            )));
        }
    }

    device.start_executor()?;
    for (name, words) in [("reader", &module.reader), ("writer", &module.writer)] {
        if words.is_empty() {
            continue;
        }
        device
            .run_program(program_at, args_at, words, &args)
            .map_err(|e| Error(format!("the {name} program: {e}")))?;
    }
    Ok(())
}

/// This tile's NOC address-high word: the coordinate packed as `x | y << 6`.
fn me_hi(tile: NocCoord<Noc0>) -> u32 {
    (tile.x() as u32 & 0x3F) | ((tile.y() as u32 & 0x3F) << 6)
}

fn lock() -> Result<std::sync::MutexGuard<'static, Option<Device>>, Error> {
    DEVICE
        .lock()
        .map_err(|_| Error("the device lock is poisoned".into()))
}

fn with_device<T>(f: impl FnOnce(&mut Device) -> Result<T, Error>) -> Result<T, Error> {
    let mut device = lock()?;
    let device = device
        .as_mut()
        .ok_or_else(|| Error("no device is open: call ttInit first".into()))?;
    f(device)
}
