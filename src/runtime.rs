// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The runtime: a device, DRAM buffers, host transfers, and a launch.
//!
//! A launch is synchronous: the inbound mover runs on RISCV B to completion, then
//! the three compute cores are staged and released together, then the outbound
//! mover runs on RISCV B. A module without compute programs skips the middle
//! step, and a module with neither mover runs only the compute cores.

use std::sync::Mutex;

use crate::compile::Module;
use crate::device::Device;
use crate::ir::Error;
use tt_isa::mailbox::{self, status};
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
    if module.reader.is_empty()
        && module.writer.is_empty()
        && module.cores.iter().all(Vec::is_empty)
    {
        return Err(LaunchError::Unsupported(
            "the module has no program to run".into(),
        ));
    }
    with_device(|device| run(device, module, params)).map_err(LaunchError::Device)
}

fn run(device: &mut Device, module: &Module, params: &[u64]) -> Result<(), Error> {
    let tile: NocCoord<Noc0> = device.tile();
    let mut args: Vec<u32> = vec![module.grid[0], module.grid[1], module.iface, me_hi(tile)];
    let hi = device.dram_hi();
    for &address in params {
        let low = u32::try_from(address).map_err(|_| {
            let _ = address;
            Error::TT_ERROR_PARAMETER_ADDRESS
        })?;
        args.push(low);
        args.push(hi);
    }

    let program_at =
        u32::try_from(crate::device::PROGRAM_AREA).map_err(|_| Error::TT_ERROR_PROGRAM_AREA)?;
    let args_at = program_at + PROGRAM_BYTES;
    for (name, words) in [("reader", &module.reader), ("writer", &module.writer)] {
        if words.is_empty() {
            continue;
        }
        if 4 * words.len() as u32 > PROGRAM_BYTES {
            let _ = (name, words.len());
            return Err(Error::TT_ERROR_PROGRAM_AREA_SIZE);
        }
    }

    device.start_executor()?;
    // The CB interface is host state: tt-metal's `setup_local_cb_read_write_interfaces`
    // writes each `LocalCBInterface` before the roles start, and every mover and
    // compute role reads its ring pointers from there. Bytes, not tt-metal's
    // 16-byte units (`riscv::cb`).
    if !module.cbs.is_empty() {
        let mut words = Vec::with_capacity(module.cbs.len() * 8);
        for ring in &module.cbs {
            words.extend_from_slice(&crate::riscv::cb::interface(
                ring.base,
                ring.page * ring.pages,
                ring.page,
                ring.pages,
            ));
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        device.l1_write(u64::from(module.iface), &bytes)?;
    }
    // Every core of the launch reads the same arguments array; write it before
    // the first core starts, not only as a side effect of a mover run.
    let arg_bytes: Vec<u8> = args.iter().flat_map(|w| w.to_le_bytes()).collect();
    device.l1_write(u64::from(args_at), &arg_bytes)?;
    if !module.reader.is_empty() {
        device.run_program(program_at, args_at, &module.reader, &args)?;
    }
    if module.cores.iter().any(|words| !words.is_empty()) {
        run_compute(device, module, args_at)?;
    }
    if !module.writer.is_empty() {
        device.run_program(program_at, args_at, &module.writer, &args)?;
    }
    Ok(())
}

/// Stage the three compute programs in their role slots, release T0/T1/T2
/// together, and wait for every started role to finish.
///
/// The backend is released first, while the cores are still held: no Tensix
/// instruction may be pushed into a held backend. Then thread 0 runs the setup
/// program (the semaphore initialisation) to completion, and only after that are
/// the three roles released together — the order tt-metal uses, where BRISC
/// initialises the semaphores before it releases the TRISCs.
fn run_compute(device: &mut Device, module: &Module, args_at: u32) -> Result<(), Error> {
    device.release_tensix_backend()?;
    stage_role(device, 0, &module.setup, args_at)?;
    let (core, image, at) = crate::images::EXEC_ROLES[0];
    device.load_and_start(core, image, at)?;
    device.wait_for(mailbox::role::Mailbox::of(0).status(), |s| {
        s == status::DONE
    })?;

    let mut images = Vec::new();
    for (thread, words) in module.cores.iter().enumerate() {
        if words.is_empty() {
            continue;
        }
        stage_role(device, thread, words, args_at)?;
        images.push(crate::images::EXEC_ROLES[thread]);
    }
    device.load_and_start_together(&images)?;
    for (thread, words) in module.cores.iter().enumerate() {
        if words.is_empty() {
            continue;
        }
        let mb = mailbox::role::Mailbox::of(thread as u32);
        device.wait_for(mb.status(), |s| s == status::DONE)?;
    }
    Ok(())
}

/// Stage `words` in `thread`'s role slot and write its descriptor.
fn stage_role(
    device: &mut Device,
    thread: usize,
    words: &[u32],
    args_at: u32,
) -> Result<(), Error> {
    if words.len() as u32 > mailbox::PROGRAM_MAX {
        let _ = (thread, words.len());
        return Err(Error::TT_ERROR_ROLE_PROGRAM_SIZE);
    }
    let mb = mailbox::role::Mailbox::of(thread as u32);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    device.l1_write(mb.program(), &code)?;
    device.write32(mb.status(), 0)?;
    let descriptor = mailbox::Descriptor {
        thread_index: thread as u32,
        dst_access_fmt: 0,
        program_len: words.len() as u32,
        // An executor calls the program; there is nothing to push and no push
        // window. The program lives in the fixed slot (`program_addr` 0).
        push_window: 0,
        args: args_at,
        ..Default::default()
    };
    for (at, value) in descriptor.writes(mb) {
        device.write32(at, value)?;
    }
    Ok(())
}

/// This tile's NOC address-high word: the coordinate packed as `x | y << 6`.
fn me_hi(tile: NocCoord<Noc0>) -> u32 {
    (tile.x() as u32 & 0x3F) | ((tile.y() as u32 & 0x3F) << 6)
}

fn lock() -> Result<std::sync::MutexGuard<'static, Option<Device>>, Error> {
    DEVICE.lock().map_err(|_| Error::TT_ERROR_DEVICE_LOCK)
}

fn with_device<T>(f: impl FnOnce(&mut Device) -> Result<T, Error>) -> Result<T, Error> {
    let mut device = lock()?;
    let device = device.as_mut().ok_or(Error::TT_ERROR_NO_DEVICE)?;
    f(device)
}
