// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! `/dev/tenstorrent/N`: the card through `tt-kmd`.
//!
//! The silicon counterpart to [`crate::device::Device::open`], which reaches the
//! simulator through `dlopen`. Everything above this module is unchanged: one
//! 2 MiB TLB window, its 96-bit configuration written through BAR0, L1 and DRAM
//! through the window, and the launch mailboxes in L1.
//!
//! The ABI here is transcribed from the header of the module this host actually
//! binds — `/usr/src/tenstorrent-2.11.0/ioctl.h`, the DKMS source of
//! `tenstorrent` 2.11.0 (`modinfo tenstorrent`) — and the driver reports its
//! ioctl API version at open, which is checked against [`API_VERSION`]. The
//! structs carry the same field order and widths as the header, so the layouts
//! are the header's; `KMd` never passes an address the driver has not reported.
//!
//! What is deliberately absent: `TENSTORRENT_IOCTL_RESET_DEVICE`. Resets are the
//! owner's, always, and a reset here would be an unsupervised ASIC power cycle —
//! the same event that has tripped a host PSU.
//!
//! What the driver does for us that the simulator does not need:
//!
//! * **TLB windows are arbitrated.** [`Kmd::open`] asks the driver for a 2 MiB
//!   window ([`abi::ALLOCATE_TLB`]) instead of assuming index 0, so another
//!   process's window is not retargeted underneath it
//!   (tt-metal: `PCIDevice::allocate_tlb`, `pci_device.cpp:846`).
//! * **The card is claimed.** [`Kmd::open`] takes the session lock — the last of
//!   the driver's 64, `TENSTORRENT_RESOURCE_LOCK_COUNT` — so a second
//!   `libskamandros` refuses rather than driving the same tiles. The lock is
//!   advisory: `tt-smi` does not take it and must keep working, which it does,
//!   because the reset ioctl is gated on the driver's own `reset_rwsem` and
//!   never on a resource lock (`chardev.c:749-756`).
//! * **A crash leaves the cores held.** [`Kmd::hold_on_close`] registers a NoC
//!   write the *driver* performs when this file descriptor closes, however it
//!   closes — the only cleanup that survives a segfault or the OOM killer.
//!
//! Time runs on its own on silicon: [`crate::device::Device::clock`] is a no-op
//! through this link, and the wait budget becomes wall clock (see
//! `device::SILICON_MIN_WAIT`).

use std::fs::File;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd};
use std::path::Path;

use crate::status::ttStatus;

/// Where the device nodes live (`tt-kmd` `chardev.c`, `DEV_DIR`).
pub const DEV_DIR: &str = "/dev/tenstorrent";

/// `TENSTORRENT_DRIVER_VERSION` (`ioctl.h:10`) — the ioctl API version this
/// module's structs describe, not the module's release number.
pub const API_VERSION: u32 = 2;

/// The compute session's resource lock (`TENSTORRENT_RESOURCE_LOCK_COUNT - 1`).
///
/// Indices 0-15 are the Ethernet cores' by convention (`ioctl.h:217-232`); the
/// driver ties a lock to the open file description and releases it only when it
/// closes, so the claim lasts as long as this [`Kmd`] and survives a chip reset.
pub const SESSION_LOCK: u8 = 63;

/// BAR0's aperture: every 2 MiB window and the configuration array live here
/// (Blackhole `HostToDeviceTLBs.md:3,14`; the length `/sys/bus/pci/devices` and
/// `query_mappings` both report).
pub const BAR0_BYTES: u64 = 512 * 1024 * 1024;

/// Window indices 0..=201 are the 2 MiB windows in BAR0
/// (`blackhole::TLB_COUNT_2M`); 202..=209 are the 4 GiB windows in BAR4, whose
/// apertures this module's BAR0 mapping does not cover.
pub const NUM_2MIB_WINDOWS: u32 = 202;

/// Tenstorrent's PCI vendor id, and a Blackhole's device id (`blackhole.c:59`
/// carries the ARC coordinates beside them; the driver's `pci_device_id` table
/// is the authority).
pub const VENDOR_ID_TENSTORRENT: u16 = 0x1E52;
pub const DEVICE_ID_BLACKHOLE: u16 = 0xB140;

/// The tt-kmd userspace ABI, transcribed from `/usr/src/tenstorrent-2.11.0/ioctl.h`.
pub mod abi {
    /// `TENSTORRENT_IOCTL_MAGIC` (`ioctl.h:12`).
    const MAGIC: u64 = 0xFA;

    /// `_IO(type, nr)`: every tt-kmd ioctl is declared with the bare `_IO` macro
    /// rather than `_IOR`/`_IOW`, so the request is just `(type << 8) | nr`.
    const fn io(nr: u64) -> u64 {
        (MAGIC << 8) | nr
    }

    // `ioctl.h:14-31`, in numeric order.
    pub const GET_DEVICE_INFO: u64 = io(0);
    pub const QUERY_MAPPINGS: u64 = io(2);
    pub const GET_DRIVER_INFO: u64 = io(5);
    pub const LOCK_CTL: u64 = io(8);
    pub const ALLOCATE_TLB: u64 = io(11);
    pub const FREE_TLB: u64 = io(12);
    pub const SET_NOC_CLEANUP: u64 = io(14);

    /// `tenstorrent_mapping.mapping_id` (`ioctl.h:35-40`). Resource numbers are
    /// PCI BARs 0, 2 and 4 — not array indices — and each BAR has an uncached
    /// and a write-combining mapping. Only BAR0's uncached view is mapped: it
    /// carries every 2 MiB window and the configuration array, and it is the view
    /// the configuration registers must go through (a write-combining store may be
    /// merged or reordered).
    pub mod mapping_id {
        pub const RESOURCE0_UC: u32 = 1;
        pub const RESOURCE0_WC: u32 = 2;
    }

    /// `TENSTORRENT_LOCK_CTL_*` at the value this module uses (`ioctl.h:212-215`).
    /// `RELEASE` (1) is deliberately not declared: tt-kmd frees a descriptor's
    /// resource locks itself when it closes (`chardev.c:1103`), after the
    /// close-time cleanup write, which is the order this driver wants.
    pub mod lock {
        pub const ACQUIRE: u32 = 0;
    }

    /// `tenstorrent_get_device_info` (`ioctl.h:47-66`).
    #[repr(C)]
    #[derive(Default)]
    pub struct GetDeviceInfo {
        pub in_output_size_bytes: u32,
        pub out_output_size_bytes: u32,
        pub vendor_id: u16,
        pub device_id: u16,
        pub subsystem_vendor_id: u16,
        pub subsystem_id: u16,
        /// `[0:2]` function, `[3:7]` device, `[8:15]` bus (`ioctl.h:57`).
        pub bus_dev_fn: u16,
        pub max_dma_buf_size_log2: u16,
        pub pci_domain: u16,
        pub reserved: u16,
    }

    /// `tenstorrent_get_driver_info` (`ioctl.h:125-141`).
    #[repr(C)]
    #[derive(Default)]
    pub struct GetDriverInfo {
        pub in_output_size_bytes: u32,
        pub out_output_size_bytes: u32,
        /// The ioctl API version this module speaks — `TENSTORRENT_DRIVER_VERSION`.
        pub driver_version: u32,
        pub driver_version_major: u8,
        pub driver_version_minor: u8,
        pub driver_version_patch: u8,
        pub reserved0: u8,
    }

    /// `tenstorrent_mapping` (`ioctl.h:73-78`).
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct Mapping {
        pub mapping_id: u32,
        pub reserved: u32,
        pub mapping_base: u64,
        pub mapping_size: u64,
    }

    /// `tenstorrent_query_mappings_in` (`ioctl.h:68-71`).
    #[repr(C)]
    #[derive(Default)]
    pub struct QueryMappingsIn {
        pub output_mapping_count: u32,
        pub reserved: u32,
    }

    /// `tenstorrent_allocate_tlb` (`ioctl.h:271-287`): the index, and the mmap
    /// keys for the uncached and write-combining views of the window.
    #[repr(C)]
    #[derive(Default)]
    pub struct AllocateTlb {
        pub in_size: u64,
        pub in_reserved: u64,
        pub out_id: u32,
        pub out_reserved0: u32,
        pub out_mmap_offset_uc: u64,
        pub out_mmap_offset_wc: u64,
        pub out_reserved1: u64,
    }

    /// `tenstorrent_free_tlb` (`ioctl.h:289-299`).
    #[repr(C)]
    #[derive(Default)]
    pub struct FreeTlb {
        pub in_id: u32,
    }

    /// `tenstorrent_lock_ctl` (`ioctl.h:235-250`).
    #[repr(C)]
    #[derive(Default)]
    pub struct LockCtl {
        pub in_output_size_bytes: u32,
        pub in_flags: u32,
        pub in_index: u8,
        pub in_reserved: [u8; 3],
        pub out_value: u8,
        pub out_reserved: [u8; 3],
    }

    /// `tenstorrent_set_noc_cleanup` (`ioctl.h:350-364`): a NoC write the driver
    /// performs when the file descriptor closes.
    #[repr(C)]
    #[derive(Default)]
    pub struct SetNocCleanup {
        pub argsz: u32,
        pub flags: u32,
        pub enabled: u8,
        pub x: u8,
        pub y: u8,
        pub noc: u8,
        pub reserved0: u32,
        pub addr: u64,
        pub data: u64,
    }
}

/// `ioctl`'s request argument, as libc declares it: `c_ulong` in glibc,
/// `c_int` in musl. Every request here is `_IO(0xFA, n)`, which fits either.
#[cfg(target_env = "musl")]
type Request = libc::c_int;
#[cfg(not(target_env = "musl"))]
type Request = libc::c_ulong;

/// One syscall failure, with the call named, as [`ttStatus::TT_ERROR_DEVICE`].
///
/// The ABI's status codes are the caller's (they are numbered and public), and
/// none of them separates "the ioctl was refused" from "the card's window did
/// not answer", so the detail goes to stderr where a bring-up needs it.
fn failed(what: &str, error: std::io::Error) -> ttStatus {
    eprintln!("libskamandros: {what}: {error}");
    ttStatus::TT_ERROR_DEVICE
}

/// Issue an ioctl whose argument is a single `#[repr(C)]` struct.
fn call<T>(fd: BorrowedFd<'_>, request: u64, arg: &mut T) -> std::io::Result<()> {
    // SAFETY: `arg` is a live, correctly-sized, correctly-aligned instance of the
    // struct this request number expects, and the driver only reads or writes
    // through it for the duration of the call.
    let rc = unsafe { libc::ioctl(fd.as_raw_fd(), request as Request, arg as *mut T) };
    if rc < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// One mmap'd PCI BAR, unmapped on drop.
///
/// The only `unsafe` in this module, together with [`call`]: an MMIO aperture has
/// no Rust-level owner, so every access goes through a volatile pointer and the
/// mapping's length is checked before it is used.
pub struct Bar {
    ptr: *mut u8,
    len: usize,
}

// An aperture is device memory, not shared heap state; moving one between threads
// is sound. Deliberately not `Sync`: a window another thread can retarget is the
// race `Device`'s `&mut self` exists to prevent.
unsafe impl Send for Bar {}

impl Bar {
    /// Map `len` bytes of `fd` at `offset` — a `mapping_base` the driver reported,
    /// a key it decodes, not a byte offset into anything.
    fn new(fd: BorrowedFd<'_>, offset: u64, len: usize) -> std::io::Result<Bar> {
        // SAFETY: a fresh mapping of a device file; the kernel validates `offset`
        // against its own table of mappable entities and fails rather than
        // handing back a bad pointer.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                offset as libc::off_t,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Bar {
            ptr: ptr.cast(),
            len,
        })
    }

    /// Copy `dst.len()` bytes out of the aperture at `offset`.
    ///
    /// Volatile and, where alignment allows, dword-at-a-time. Neither is an
    /// optimisation: these are device registers, so the compiler may not elide,
    /// reorder or merge the accesses, and several regions behind this aperture
    /// (the TLB configuration array among them) are dword-only. A `memcpy` would
    /// be free to split a dword into bytes.
    fn read(&self, offset: usize, dst: &mut [u8]) {
        assert!(offset + dst.len() <= self.len);
        let mut i = 0;
        while i < dst.len() {
            let at = offset + i;
            if at % 4 == 0 && dst.len() - i >= 4 {
                // SAFETY: bounds checked above; `at` is 4-aligned within the map.
                let v = unsafe { std::ptr::read_volatile(self.ptr.add(at).cast::<u32>()) };
                dst[i..i + 4].copy_from_slice(&v.to_le_bytes());
                i += 4;
            } else {
                // SAFETY: bounds checked above.
                dst[i] = unsafe { std::ptr::read_volatile(self.ptr.add(at)) };
                i += 1;
            }
        }
    }

    /// Copy `src` into the aperture at `offset`. See [`Bar::read`].
    fn write(&self, offset: usize, src: &[u8]) {
        assert!(offset + src.len() <= self.len);
        let mut i = 0;
        while i < src.len() {
            let at = offset + i;
            if at % 4 == 0 && src.len() - i >= 4 {
                let v = u32::from_le_bytes(src[i..i + 4].try_into().unwrap());
                // SAFETY: bounds checked above; `at` is 4-aligned within the map.
                unsafe { std::ptr::write_volatile(self.ptr.add(at).cast::<u32>(), v) };
                i += 4;
            } else {
                // SAFETY: bounds checked above.
                unsafe { std::ptr::write_volatile(self.ptr.add(at), src[i]) };
                i += 1;
            }
        }
    }
}

impl Drop for Bar {
    fn drop(&mut self) {
        // SAFETY: the region was mapped by `Bar::new` with this base and length,
        // and nothing borrows it after `drop` begins.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

/// One Blackhole reached through one `/dev/tenstorrent/N`.
///
/// Holds the file descriptor the driver ties the session lock and the cleanup
/// write to, BAR0 in both of the driver's views, and the 2 MiB window the driver
/// allocated for this session. tt-kmd exposes no configuration-space ioctl —
/// `GetDeviceInfo` carries identity, not the BARs — and nothing here needs
/// configuration space: the BARs come from `QUERY_MAPPINGS`.
pub struct Kmd {
    // Declared before `fd` so the apertures are unmapped before the descriptor
    // they were mapped through is closed.
    /// BAR0 uncached: the TLB configuration array, the PCIe NIU's configuration
    /// word and the ARC's APB doorbell. Stores here must not be merged.
    bar0: Bar,
    /// BAR0 write-combined: the TLB windows themselves, where bulk L1 and DRAM
    /// bytes go. tt-metal takes its windows this way (`TlbManager::configure_tlb`
    /// passes `TlbMapping::WC`, `tlb_manager.cpp:44`) and its registers through
    /// the uncached view (`pci_device.cpp:531-536`).
    bar0_wc: Bar,
    fd: File,
    window: u32,
}

impl Kmd {
    /// Open device `index`, check the driver's ioctl API, map BAR0, claim the
    /// session and allocate the window.
    ///
    /// The open is deliberately *without* `O_APPEND`, which tt-kmd reads as a
    /// power policy (`chardev.c:950-998`): a legacy open asks the driver for the
    /// busy operating point immediately and gates the array again after the last
    /// close. That is what tt-metal's own open does today (`open_pci_device`,
    /// `pci_device.cpp:400-404`, "O_APPEND is temporarily disabled") and what
    /// `tt-smi` does (`chip_reset.py:62-64`), so it is the configuration this
    /// card's other tools expect.
    pub fn open(index: u16) -> Result<Kmd, ttStatus> {
        let path = Path::new(DEV_DIR).join(index.to_string());
        let fd = File::options()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| failed("opening the device", e))?;

        // A matching header is no evidence about the module that is loaded, so
        // ask the module itself before any struct below is handed to it.
        let mut info = abi::GetDriverInfo {
            in_output_size_bytes: (std::mem::size_of::<abi::GetDriverInfo>() - 4) as u32,
            ..Default::default()
        };
        call(fd.as_fd(), abi::GET_DRIVER_INFO, &mut info)
            .map_err(|e| failed("asking the driver its ioctl version", e))?;
        if info.driver_version != API_VERSION {
            eprintln!(
                "libskamandros: {} speaks tt-kmd ioctl API {}, but this build is written \
                 against {}",
                path.display(),
                info.driver_version,
                API_VERSION
            );
            return Err(ttStatus::TT_ERROR_VERSION);
        }

        let mut dev = abi::GetDeviceInfo {
            in_output_size_bytes: (std::mem::size_of::<abi::GetDeviceInfo>() - 4) as u32,
            ..Default::default()
        };
        call(fd.as_fd(), abi::GET_DEVICE_INFO, &mut dev)
            .map_err(|e| failed("reading the device's identity", e))?;
        if dev.vendor_id != VENDOR_ID_TENSTORRENT || dev.device_id != DEVICE_ID_BLACKHOLE {
            eprintln!(
                "libskamandros: {path:?} is {:#06x}:{:#06x}, not a Blackhole",
                dev.vendor_id, dev.device_id
            );
            return Err(ttStatus::TT_ERROR_NOT_BLACKHOLE);
        }
        // BAR0 in both of the driver's views. The configuration array, the NIU's
        // configuration word and the ARC's doorbell registers are dword-only and
        // must not be merged or reordered, so they go through the uncached view;
        // the TLB windows carry bulk bytes, which is what write-combining is for
        // (tt-metal: `PCIDevice` maps BAR0 UC, `pci_device.cpp:531-536`, and
        // `TlbManager::configure_tlb` asks for `TlbMapping::WC` windows,
        // `tlb_manager.cpp:44`).
        let mappings =
            query_mappings(fd.as_fd(), 8).map_err(|e| failed("querying the BAR mappings", e))?;
        let bar0 = Bar::new(
            fd.as_fd(),
            bar0_mapping(&mappings, abi::mapping_id::RESOURCE0_UC, &path, "uncached")?,
            BAR0_BYTES as usize,
        )
        .map_err(|e| failed("mapping BAR0 uncached", e))?;
        let bar0_wc = Bar::new(
            fd.as_fd(),
            bar0_mapping(
                &mappings,
                abi::mapping_id::RESOURCE0_WC,
                &path,
                "write-combined",
            )?,
            BAR0_BYTES as usize,
        )
        .map_err(|e| failed("mapping BAR0 write-combined", e))?;

        // Claim the card. The lock is advisory — it cannot stop `tt-smi` — but it
        // does stop a second `libskamandros`, which is the case a session can
        // control. Held until this descriptor closes, which is when tt-kmd frees
        // it, after the close-time cleanup write (`chardev.c:1101-1104`).
        let locked = lock(fd.as_fd(), SESSION_LOCK, abi::lock::ACQUIRE)
            .map_err(|e| failed("taking the session lock", e))?;
        if !locked {
            eprintln!(
                "libskamandros: {path:?} is claimed by another process (session lock {SESSION_LOCK})"
            );
            return Err(ttStatus::TT_ERROR_DEVICE);
        }

        // One 2 MiB window, from the driver rather than assumed. Asking is what
        // keeps a second process's window from being retargeted underneath it
        // (tt-metal `PCIDevice::allocate_tlb`, `pci_device.cpp:846-869`).
        let mut tlb = abi::AllocateTlb {
            in_size: 2 * 1024 * 1024,
            ..Default::default()
        };
        call(fd.as_fd(), abi::ALLOCATE_TLB, &mut tlb)
            .map_err(|e| failed("allocating a TLB window", e))?;
        if tlb.out_id >= NUM_2MIB_WINDOWS {
            eprintln!(
                "libskamandros: the driver handed out window {}, which is a 4 GiB window in \
                 BAR4; this driver maps BAR0 only",
                tlb.out_id
            );
            return Err(ttStatus::TT_ERROR_DEVICE);
        }

        Ok(Kmd {
            bar0,
            bar0_wc,
            fd,
            window: tlb.out_id,
        })
    }

    /// The window index the driver allocated.
    pub fn window(&self) -> u32 {
        self.window
    }

    /// Write `bytes` into BAR0 at `offset`, uncached.
    pub fn bar0_write(&self, offset: u64, bytes: &[u8]) -> Result<(), ttStatus> {
        let end = offset
            .checked_add(bytes.len() as u64)
            .ok_or(ttStatus::TT_ERROR_BAR0_OVERFLOW)?;
        if end > BAR0_BYTES {
            return Err(ttStatus::TT_ERROR_BAR0_NOT_DECODED);
        }
        self.bar0.write(offset as usize, bytes);
        Ok(())
    }

    /// Read `out.len()` bytes of BAR0 at `offset`, uncached.
    pub fn bar0_read(&self, offset: u64, out: &mut [u8]) -> Result<(), ttStatus> {
        let end = offset
            .checked_add(out.len() as u64)
            .ok_or(ttStatus::TT_ERROR_BAR0_OVERFLOW)?;
        if end > BAR0_BYTES {
            return Err(ttStatus::TT_ERROR_BAR0_NOT_DECODED);
        }
        self.bar0.read(offset as usize, out);
        Ok(())
    }

    /// Write `bytes` into the window aperture at `offset`, write-combined, and
    /// push them out.
    ///
    /// This is the bulk path: L1 images, mailbox words, DRAM tensors. The fence
    /// is what makes the write visible before anything the host does next — a
    /// retarget, a configuration register `bar0_write` puts in a different
    /// mapping, the generation that starts a core — because a write-combining
    /// buffer is not obligated to drain on its own.
    pub fn window_write(&self, offset: u64, bytes: &[u8]) -> Result<(), ttStatus> {
        let end = offset
            .checked_add(bytes.len() as u64)
            .ok_or(ttStatus::TT_ERROR_BAR0_OVERFLOW)?;
        if end > BAR0_BYTES {
            return Err(ttStatus::TT_ERROR_BAR0_NOT_DECODED);
        }
        self.bar0_wc.write(offset as usize, bytes);
        flush_writes();
        Ok(())
    }

    /// Read `out.len()` bytes of the window aperture at `offset`. Reads of
    /// write-combined memory are not cached, so they see the card.
    pub fn window_read(&self, offset: u64, out: &mut [u8]) -> Result<(), ttStatus> {
        let end = offset
            .checked_add(out.len() as u64)
            .ok_or(ttStatus::TT_ERROR_BAR0_OVERFLOW)?;
        if end > BAR0_BYTES {
            return Err(ttStatus::TT_ERROR_BAR0_NOT_DECODED);
        }
        self.bar0_wc.read(offset as usize, out);
        Ok(())
    }

    /// Register the NoC write the driver performs when this descriptor closes.
    ///
    /// Holding every baby RISC-V on `tile` in reset is the one cleanup that
    /// survives this process dying without running a destructor: the driver
    /// writes it (`tt_cdev_release_noc_cleanup`, `chardev.c:1030-1038`). One
    /// write covers one tile, which is the card's whole protocol, so a launch
    /// across a grid is covered on the tile named here only.
    pub fn hold_on_close(&self, x: u8, y: u8, address: u64, value: u32) -> Result<(), ttStatus> {
        let mut arg = abi::SetNocCleanup {
            argsz: std::mem::size_of::<abi::SetNocCleanup>() as u32,
            enabled: 1,
            x,
            y,
            noc: 0,
            addr: address,
            data: u64::from(value),
            ..Default::default()
        };
        call(self.fd.as_fd(), abi::SET_NOC_CLEANUP, &mut arg)
            .map_err(|e| failed("registering the close-time cleanup write", e))
    }
}

impl Drop for Kmd {
    /// Give the window back, so a later session need not restart the process to
    /// get it.
    ///
    /// The session lock is deliberately *not* released here. tt-kmd frees a
    /// descriptor's resource locks itself, after the close-time cleanup write
    /// (`tt_cdev_release`: `tt_cdev_release_noc_cleanup` then
    /// `tt_cdev_release_resource_locks`, `chardev.c:1101-1104`). Releasing it
    /// early would let a second `libskamandros` claim the card in the window
    /// between this release and the descriptor closing, and the cleanup write
    /// this fd registered would then hold *that* process's cores.
    fn drop(&mut self) {
        let mut arg = abi::FreeTlb { in_id: self.window };
        if let Err(e) = call(self.fd.as_fd(), abi::FREE_TLB, &mut arg) {
            eprintln!("libskamandros: freeing the TLB window: {e}");
        }
    }
}

/// Push this core's write-combining buffers to the device.
///
/// Stores through the write-combined BAR0 view are buffered, and a later store
/// through the uncached view — a window retarget, a configuration register, a
/// mask — can leave before them. tt-metal fences for the same reason around its
/// barriers (`tt_driver_atomics::sfence`, `local_chip.cpp:474-500`).
fn flush_writes() {
    // `SFENCE` is the instruction x86 defines as flushing the write-combining
    // buffers; elsewhere a sequentially consistent fence is the equivalent.
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `sfence` takes no operands and touches no memory; it only orders
    // this core's stores.
    unsafe {
        core::arch::x86_64::_mm_sfence();
    }
    #[cfg(not(target_arch = "x86_64"))]
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

/// BAR0's mmap key for one of the driver's views, after checking the driver
/// reports the whole BAR through it.
fn bar0_mapping(
    mappings: &[abi::Mapping],
    id: u32,
    path: &Path,
    what: &str,
) -> Result<u64, ttStatus> {
    let Some(mapping) = mappings.iter().find(|m| m.mapping_id == id).copied() else {
        eprintln!("libskamandros: {path:?} reports no {what} BAR0 mapping");
        return Err(ttStatus::TT_ERROR_DEVICE);
    };
    if mapping.mapping_size != BAR0_BYTES {
        eprintln!(
            "libskamandros: {path:?} reports BAR0 {what} as {} bytes, not {BAR0_BYTES}",
            mapping.mapping_size
        );
        return Err(ttStatus::TT_ERROR_DEVICE);
    }
    Ok(mapping.mapping_base)
}

/// `QUERY_MAPPINGS` — the BAR apertures and the mmap keys that reach them.
///
/// The argument is a flexible-array struct: a count, then that many entries the
/// driver writes back (`memory.c`). Built as one byte buffer because Rust has no
/// flexible array member.
fn query_mappings(fd: BorrowedFd<'_>, count: u32) -> std::io::Result<Vec<abi::Mapping>> {
    let head = std::mem::size_of::<abi::QueryMappingsIn>();
    let entry = std::mem::size_of::<abi::Mapping>();
    let mut buf = vec![0u8; head + entry * count as usize];
    buf[0..4].copy_from_slice(&count.to_le_bytes());

    // SAFETY: the buffer is at least as large as the driver writes, and `Vec<u8>`
    // of this size is 8-aligned.
    let rc = unsafe {
        libc::ioctl(
            fd.as_raw_fd(),
            abi::QUERY_MAPPINGS as Request,
            buf.as_mut_ptr(),
        )
    };
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }

    let mut out = Vec::new();
    for i in 0..count as usize {
        let at = head + entry * i;
        let field = |o: usize, n: usize| &buf[at + o..at + o + n];
        let id = u32::from_le_bytes(field(0, 4).try_into().unwrap());
        if id == 0 {
            continue;
        }
        out.push(abi::Mapping {
            mapping_id: id,
            reserved: 0,
            mapping_base: u64::from_le_bytes(field(8, 8).try_into().unwrap()),
            mapping_size: u64::from_le_bytes(field(16, 8).try_into().unwrap()),
        });
    }
    Ok(out)
}

/// Take or release one of the driver's 64 arbitrated locks.
///
/// `ACQUIRE` reports 0 when another process holds it rather than failing, so the
/// return value is the answer, not an error.
fn lock(fd: BorrowedFd<'_>, index: u8, flags: u32) -> std::io::Result<bool> {
    // `output_size_bytes` is the size of the `out` half alone: the driver clears
    // that many bytes from `out` on (`clear_user(&arg->out, ..)`), so the whole
    // struct's size would write past the argument, onto the caller's stack.
    let mut arg = abi::LockCtl {
        in_output_size_bytes: (std::mem::size_of::<abi::LockCtl>()
            - std::mem::offset_of!(abi::LockCtl, out_value)) as u32,
        in_flags: flags,
        in_index: index,
        ..Default::default()
    };
    call(fd, abi::LOCK_CTL, &mut arg)?;
    Ok(arg.out_value == 1)
}
