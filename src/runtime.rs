// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The runtime: a device, DRAM buffers, host transfers, and a launch.
//!
//! A launch is synchronous: the inbound mover runs on RISCV B to completion, then
//! the three compute cores are staged and released together, then the outbound
//! mover runs on RISCV B. A module without compute programs skips the middle
//! step, and a module with neither mover runs only the compute cores.
//!
//! Every core has a launch message (`mailbox::launch::Mailbox`): the program's
//! address in `l1::PROGRAM_CACHE`, its length, the arguments base and a
//! generation. A program is placed once and later launches only name it; the
//! resident executor on the core runs the generation it sees
//! (tt-metal's launch message and kernel-text offset).

use std::sync::Mutex;

use crate::compile::Module;
use crate::device::Device;
use crate::ir::Error;
use tt_isa::mailbox;
use tt_isa::noc::{Noc0, NocCoord};
use tt_isa::tensix::Core;

/// The one open device, if [`init`] has been called.
static DEVICE: Mutex<Option<Device>> = Mutex::new(None);

/// First byte of `l1::PROGRAM_CACHE`, where every program is placed once.
const PROGRAM_AREA: u32 = crate::device::PROGRAM_AREA as u32;
/// Bytes in one program-cache slot: `mailbox::MAX_WORDS` words.
const SLOT_BYTES: u32 = 0x8000;
/// Slot 0 and 1: RISCV B's reader and writer. 3..6: T0/T1/T2. The arguments
/// follow the last slot.
const SLOT_READER: u32 = 0;
const SLOT_WRITER: u32 = 1;
const ROLE_SLOT: [u32; 3] = [3, 4, 5];
const ARGS_AT: u32 = PROGRAM_AREA + 6 * SLOT_BYTES;

/// `index`'s slot in the program cache.
const fn slot(index: u32) -> u32 {
    PROGRAM_AREA + index * SLOT_BYTES
}

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

/// Run a module over its grid with one DRAM address per parameter.
///
/// The launch arguments are built here, not by the caller: the grid, this
/// tile's high word, and each parameter's `(low, high)` pair. A count that
/// disagrees with the module is refused. Every failure is a [`ttStatus`]
/// variant; there is no second error channel and no string error.
pub fn launch(module: &Module, params: &[u64]) -> Result<(), Error> {
    if module.params as usize != params.len() {
        return Err(Error::TT_ERROR_ARGUMENT);
    }
    if module.grid != [1, 1] {
        return Err(Error::TT_ERROR_UNSUPPORTED);
    }
    if module.reader.is_empty()
        && module.writer.is_empty()
        && module.cores.iter().all(Vec::is_empty)
    {
        return Err(Error::TT_ERROR_UNSUPPORTED);
    }
    with_device(|device| run(device, module, params))
}

fn run(device: &mut Device, module: &Module, params: &[u64]) -> Result<(), Error> {
    let result = launch_steps(device, module, params);
    if result.is_err() {
        // A failed launch is a status, not a wedged tile: stop the cores this
        // launch started and clear the mailboxes it wrote before returning, so
        // the next launch starts from a known state (user requirement). No
        // `ttReset` exists; this is the driver's own cleanup.
        device.quiesce()?;
    }
    result
}

fn launch_steps(device: &mut Device, module: &Module, params: &[u64]) -> Result<(), Error> {
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

    if (ARGS_AT as u64) + 4 * args.len() as u64 > tt_isa::l1::PROGRAM_CACHE.end {
        let _ = args.len();
        return Err(Error::TT_ERROR_PROGRAM_AREA_SIZE);
    }
    for (name, words) in [("reader", &module.reader), ("writer", &module.writer)] {
        if words.is_empty() {
            continue;
        }
        if 4 * words.len() as u32 > SLOT_BYTES {
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
    device.l1_write(u64::from(ARGS_AT), &arg_bytes)?;
    if !module.reader.is_empty() {
        device.run_program(slot(SLOT_READER), ARGS_AT, &module.reader, &args)?;
    }
    if module.cores.iter().any(|words| !words.is_empty()) {
        run_compute(device, module, ARGS_AT)?;
    }
    if !module.writer.is_empty() {
        device.run_program(slot(SLOT_WRITER), ARGS_AT, &module.writer, &args)?;
    }
    Ok(())
}

/// Stage the three compute programs in the program cache, release T0/T1/T2
/// together, and wait for every started role to acknowledge.
///
/// The backend is released first, while the cores are still held: no Tensix
/// instruction may be pushed into a held backend. Then thread 0 runs the setup
/// program (the semaphore initialisation) to completion, and only after that are
/// the three roles released together -- the order tt-metal uses, where BRISC
/// initialises the semaphores before it releases the TRISCs.
fn run_compute(device: &mut Device, module: &Module, args_at: u32) -> Result<(), Error> {
    device.release_tensix_backend()?;
    let setup = stage_role(device, 0, &module.setup, args_at)?;
    let (core, image, at) = crate::images::EXEC_ROLES[0];
    device.load_and_start(core, image, at)?;
    device.await_launch(mailbox::launch::Mailbox::of(0), setup)?;
    // The setup executor is resident and would pick up the real program as soon
    // as its generation lands. Hold T0; `load_and_start_together` reloads it
    // with the other two.
    device.hold(Core::T0)?;

    let mut images = Vec::new();
    let mut launched = Vec::new();
    for (thread, words) in module.cores.iter().enumerate() {
        if words.is_empty() {
            continue;
        }
        let generation = stage_role(device, thread, words, args_at)?;
        images.push(crate::images::EXEC_ROLES[thread]);
        launched.push((thread, generation));
    }
    device.load_and_start_together(&images)?;
    for (thread, generation) in launched {
        device.await_launch(mailbox::launch::Mailbox::of(thread as u32), generation)?;
    }
    Ok(())
}

/// Place `words` at `thread`'s fixed cache slot and write its launch message
/// (thread index, text address, length, arguments), returning the generation
/// that starts it.
fn stage_role(
    device: &mut Device,
    thread: usize,
    words: &[u32],
    args_at: u32,
) -> Result<u32, Error> {
    if words.is_empty() || words.len() as u32 > mailbox::MAX_WORDS {
        let _ = (thread, words.len());
        return Err(Error::TT_ERROR_ROLE_PROGRAM_SIZE);
    }
    if 4 * words.len() as u32 > SLOT_BYTES {
        let _ = (thread, words.len());
        return Err(Error::TT_ERROR_PROGRAM_AREA_SIZE);
    }
    let mb = mailbox::launch::Mailbox::of(thread as u32);
    let text = slot(ROLE_SLOT[thread]);
    device.place(text, words)?;
    device.write32(mb.thread_index(), thread as u32)?;
    device.write32(mb.dst_access_fmt(), 0)?;
    device.write32(mb.dump_row_first(), 0)?;
    device.write32(mb.dump_row_count(), 0)?;
    device.write32(mb.trace(), 0)?;
    device.write32(mb.push_window(), 0)?;
    device.write32(mb.unwedge(), 0)?;
    device.write32(mb.mop_cfg_valid(), 0)?;
    device.write32(mb.status(), 0)?;
    device.launch(mb, text, words.len() as u32, args_at)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `jal x0, 0`: the program never returns.
    const HANG: u32 = 0x0000_006f;
    /// `jalr x0, 0(ra)`: the program returns at once.
    const RET: u32 = 0x0000_8067;

    fn module(reader: Vec<u32>) -> Module {
        Module {
            grid: [1, 1],
            params: 0,
            reader,
            writer: Vec::new(),
            cores: Default::default(),
            setup: Vec::new(),
            iface: 0,
            cbs: Vec::new(),
        }
    }

    /// A launch that hangs is a status, not a wedged tile: the watchdog's
    /// timeout reaches the caller and the next launch still runs.
    ///
    /// Ignored because it opens the simulator, which needs `TT_TTSIM_LIB`.
    /// Run with `TT_TTSIM_LIB=... cargo test --lib -- --ignored`.
    #[test]
    #[ignore = "needs the simulator: set TT_TTSIM_LIB"]
    fn a_timed_out_launch_returns_timeout_and_the_next_launch_runs() {
        init().expect("init");
        match launch(&module(vec![HANG]), &[]) {
            Err(Error::TT_ERROR_FIRMWARE_TIMEOUT) => {}
            other => panic!("expected a timeout, got {other:?}"),
        }
        // The failed launch held B and cleared its mailboxes, so this one
        // stages the executor again instead of waiting on a stale `DONE`.
        launch(&module(vec![RET]), &[]).expect("a launch after the timeout");
        release().expect("release");
    }
}
