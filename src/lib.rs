// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

// The artifact is `libTT.so`, so the crate is `TT`.
#![allow(non_snake_case)]

//! The Tenstorrent kernel compiler and driver.
//!
//! One semantic SSA program in ([`ir`]), a set of role programs and a launch
//! out: the reader/writer RISC-V programs and the T0/T1/T2 Tensix programs are
//! produced by `emit` from the allocation `alloc` makes, and `runtime` loads and
//! runs them. The C ABI in `abi` is the only stable surface; every other item is
//! free to move.
//!
//! The IR is a *program format*, not a Rust API: it is accepted as text or as
//! bytes, and versioned ([`ir::VERSION`]).

pub mod abi;
pub mod alloc;
pub mod compile;
pub mod device;
pub mod encode;
pub mod images;
pub mod ir;
pub mod lower;
pub mod mir;
pub mod mover;
pub mod optimize;
pub mod planner;
pub mod riscv;
pub mod runtime;

pub use ir::{DType, Dim, Kernel, Op, RegShape, Types, ValueId};
