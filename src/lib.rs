// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `libTT.so`: a Tenstorrent kernel compiler and driver for Blackhole.
//!
//! The only public module is `abi`, the C ABI. Everything else is private: a
//! consumer links `libTT.so` and calls `ttCompile`, `ttLaunch` and the transfers
//! beside them, exactly as the header describes. Nothing in this crate is a Rust
//! library, and nothing outside it should need to be.

pub mod abi;

mod alloc;
mod compile;
mod device;
mod encode;
mod ir;
mod lower;
mod mir;
mod module;
mod mover;
mod optimize;
mod planner;
mod riscv;
mod runtime;
mod status;
mod unfold;

#[cfg(test)]
mod ir_tests;
