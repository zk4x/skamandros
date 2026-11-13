// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The transcendentals become primitives.
//!
//! An op the SFPU has no instruction for (`exp`, `log`, `sin`, `cos`, ...) is
//! rewritten here into the primitives it is built from — `exp2` and a multiply,
//! `log2` and a multiply, a polynomial for the trigonometric pair — so that
//! everything below this pass sees only ops with an instruction. This is a
//! virtual-IR pass: it runs before anything is placed, and every value it
//! creates is an ordinary value of the program.
//!
//! TODO(the IR re-model): the body is the next port.

use crate::ir::{Error, Kernel};

/// Rewrite every op the SFPU has no instruction for.
pub fn unfold(k: &Kernel) -> Result<Kernel, Error> {
    let _ = k;
    todo!("unfold is being re-pointed at the new op set")
}
