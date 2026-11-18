// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! `libskamandros.so`: a Tenstorrent kernel compiler and driver for Blackhole.
//!
//! The only public module is `abi`, the C ABI. Everything else is private: a
//! consumer links `libTT.so` and calls `ttCompile`, `ttLaunch` and the transfers
//! beside them, exactly as the header describes. Nothing in this crate is a Rust
//! library, and nothing outside it should need to be.
#![allow(non_snake_case)] // the C ABI's names are the caller's, not Rust's.

pub mod abi;

mod alloc;
mod compile;
mod device;
mod encode;
mod ir;
mod kmd;
mod lock_dst;
mod lower;
mod mir;
mod module;
mod mover;
mod optimize;
mod quasar_mover;
mod riscv;
mod runtime;
mod slab;
mod status;
mod sync_cbs;
mod unfold;
mod verify;

/// The hash set zyx's `Slab` keeps its free list in. zyx uses the crate's own
/// `FHasher`; with no equivalent here the standard hasher stands in, which
/// changes nothing the slab observes.
type Set<T> = std::collections::HashSet<
    T,
    std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>,
>;

#[cfg(test)]
mod ir_tests;
