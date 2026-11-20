// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! `unfold`: the transcendentals written as the primitives the IR has.
//!
//! zyx does this in `tt_unfold`, on the way into its backend, so that the
//! native lowering only ever sees `add`/`sub`/`mul`, the bit ops, `Bitcast`,
//! `Cast`, comparisons and `Select`. skamandros does it itself: a caller writes
//! `log2`, `sin`, `cos`, `pow`, `div`, `reciprocal`, `sqrt`, `floor` or `trunc`
//! and this pass turns each into that primitive form, in the same arithmetic
//! zyx's bodies use.
//!
//! `exp2` is the one transcendental that stays an IR op. Its exponent assembly
//! is `SFPEXEXP`/`SFPSETEXP`, which no IR op spells, so `lower` emits it as a
//! machine body instead of it being unfolded here. `rsqrt` is the other: its
//! body is `lower`'s, and `sqrt` is the reciprocal of it.
//!
//! The pass runs on the unified IR, before `lower`: every value it creates is
//! an ordinary value, and the allocator is what turns a chain into one row
//! body.
//!
//! Nothing here is a new op: `log2`/`sin`/`cos` are `UOp`s, `pow`/`div` are
//! `BOp`s, and both are part of the program format already. A `pow` whose
//! operands are not floats is left alone, and `lower` refuses it loudly.

use std::collections::BTreeMap;

use crate::ir::{BOp, DType, Kernel, Op, OpId, UOp};
use crate::status::ttStatus;

/// Rewrite every transcendental `k` has into its primitive form.
pub fn unfold(k: &Kernel) -> Result<Kernel, ttStatus> {
    let mut b = Build {
        out: Kernel::default(),
        old: k,
        map: BTreeMap::new(),
        seq: 0,
    };
    for (id, op) in k.ops() {
        match *op {
            Op::Unary {
                op: uop @ (UOp::Floor | UOp::Trunc | UOp::Sqrt),
                a,
            } => {
                let x = b.map_of(a);
                let dtype = k.dtype(a);
                let out = match uop {
                    UOp::Floor => b.floor(x, dtype),
                    UOp::Trunc => b.trunc(x, dtype),
                    // `sqrt x = 1/rsqrt x`: at `x = 0` the reciprocal square
                    // root is `+inf` and the reciprocal of that is `0`, at
                    // `x = +inf` it is `0` and the reciprocal is `+inf`, and a
                    // negative `x` is a `NaN` through both. Both are lane
                    // bodies in `lower`.
                    _ => {
                        let r = b.un(UOp::Rsqrt, x, "sqrt_rsqrt");
                        b.un(UOp::Reciprocal, r, "sqrt_recip")
                    }
                };
                b.name(id, out);
                b.map.insert(id, out);
            }
            Op::Binary {
                op: bop @ (BOp::Pow | BOp::Div),
                a,
                b: rhs,
            } => {
                let x = b.map_of(a);
                let y = b.map_of(rhs);
                let float = k.dtype(a) != DType::I32 && k.dtype(rhs) != DType::I32;
                let out = match (bop, float) {
                    // `x^y = exp2(y log2 x)`. zyx's `unfold_pows` also folds an
                    // integer exponent by squaring and `x^0` to one; this keeps
                    // the float path only, and an integer `pow` is left for
                    // `lower` to refuse rather than silently mis-lowered here.
                    (BOp::Pow, true) => {
                        let l = b.un(UOp::Log2, x, "pow_log2");
                        let p = b.bin(BOp::Mul, y, l, "pow_mul");
                        b.un(UOp::Exp2, p, "pow_exp2")
                    }
                    // `x / y = x * (1/y)`, the reciprocal being the lane body
                    // in `lower` (tt-metal's `_sfpu_reciprocal_`: a quadratic
                    // estimate and one Newton step per datum width).
                    (BOp::Div, true) => {
                        let r = b.un(UOp::Reciprocal, y, "div_recip");
                        b.bin(BOp::Mul, x, r, "div_mul")
                    }
                    _ => {
                        b.copy(id, op.clone())?;
                        continue;
                    }
                };
                b.name(id, out);
                b.map.insert(id, out);
            }
            _ => b.copy(id, op.clone())?,
        }
    }
    Ok(b.out)
}

/// The kernel being rewritten, the one being built, and the value map between
/// them. Definitions are visited in program order, so a use always finds its
/// definition already mapped (`Kernel::verify` guarantees the dominance).
struct Build<'a> {
    out: Kernel,
    old: &'a Kernel,
    map: BTreeMap<OpId, OpId>,
    /// A counter for the names of the values this pass creates, so that
    /// unfolding the same shape twice does not emit the same name twice.
    seq: u32,
}

impl Build<'_> {
    fn map_of(&self, v: OpId) -> OpId {
        self.map[&v]
    }

    /// Give the value about to be pushed the name `tag{seq}`.
    fn value(&mut self, tag: &str) {
        let id = self.out.ops.max_id();
        self.out.names.insert(id, format!("{tag}{}", self.seq));
        self.seq += 1;
    }

    /// Carry the old value's name onto `new`, the value this pass produced.
    fn name(&mut self, old: OpId, new: OpId) {
        if let Some(name) = self.old.names.get(&old) {
            self.out.names.insert(new, name.clone());
        }
    }

    fn kf(&mut self, x: f32, tag: &str) -> OpId {
        self.value(tag);
        self.out.push(Op::Const {
            dtype: DType::F32,
            bits: x.to_bits(),
        })
    }

    fn bin(&mut self, op: BOp, a: OpId, b: OpId, tag: &str) -> OpId {
        self.value(tag);
        self.out.push(Op::Binary { op, a, b })
    }

    fn un(&mut self, op: UOp, a: OpId, tag: &str) -> OpId {
        self.value(tag);
        self.out.push(Op::Unary { op, a })
    }

    fn cast(&mut self, src: OpId, dtype: DType, tag: &str) -> OpId {
        self.value(tag);
        self.out.push(Op::Cast { dtype, src })
    }

    fn select(&mut self, cond: OpId, t: OpId, f: OpId, tag: &str) -> OpId {
        self.value(tag);
        self.out.push(Op::Select {
            cond,
            if_true: t,
            if_false: f,
        })
    }

    /// `trunc x`: toward zero. The IR's cast to `i32` truncates (the SFPU's
    /// `SFPCAST`), so `(f32)(i32)x` is the answer wherever the integer cast is
    /// exact. Outside that range the value is already integral, and a `NaN` is
    /// not a value to round: both pass through unchanged.
    /// `dtype` is the value's own: the constants and the `i32` round trip come
    /// back at it, so a bf16 tile only meets bf16 operands (`select` and the
    /// comparisons take one dtype), and a constant is the bf16 the lane reads.
    fn trunc(&mut self, x: OpId, dtype: DType) -> OpId {
        let i = self.cast(x, DType::I32, "trunc_i");
        let t = self.cast(i, dtype, "trunc_f");
        let abs = self.un(UOp::Abs, x, "trunc_abs");
        let big = self.kf(2_147_483_648.0, "trunc_big");
        let big = self.at(big, dtype, "trunc_big");
        let over = self.bin(BOp::Cmpge, abs, big, "trunc_over");
        let nan = self.bin(BOp::NotEq, x, x, "trunc_nan");
        let pass = self.bin(BOp::Or, over, nan, "trunc_pass");
        self.select(pass, x, t, "trunc")
    }

    /// `floor x = trunc x` less one when `x` is negative and not integral.
    /// `trunc` already let the non-finite values through, so `x - trunc x` is
    /// zero there and the adjustment never fires.
    fn floor(&mut self, x: OpId, dtype: DType) -> OpId {
        let t = self.trunc(x, dtype);
        let frac = self.bin(BOp::Sub, x, t, "floor_frac");
        let zero = self.kf(0.0, "floor_zero");
        let zero = self.at(zero, dtype, "floor_zero");
        let down = self.bin(BOp::Cmplt, frac, zero, "floor_down");
        let one = self.kf(1.0, "floor_one");
        let one = self.at(one, dtype, "floor_one");
        let less = self.bin(BOp::Sub, t, one, "floor_less");
        self.select(down, less, t, "floor")
    }

    /// A constant at `dtype`: a `cast` of a constant is the conversion it
    /// names, and the lane reads the result's image (a bf16 constant is the
    /// truncation of the literal, zyx's `bf16::from_f32`).
    fn at(&mut self, constant: OpId, dtype: DType, tag: &str) -> OpId {
        if dtype == DType::F32 {
            constant
        } else {
            self.cast(constant, dtype, tag)
        }
    }

    /// Copy an op unchanged, with its operands remapped. A value *is* its
    /// defining op, so a copied value takes the id of the copy.
    fn copy(&mut self, old: OpId, op: Op) -> Result<(), ttStatus> {
        let mut op = op;
        crate::ir::renumber(&mut op, &self.map)?;
        let id = self.out.push(op);
        self.map.insert(old, id);
        if let Some(name) = self.old.names.get(&old) {
            self.out.names.insert(id, name.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG2: &str = "\
grid 1,1
%in = param f32 1024
%out = param f32 1024
%b0 = buffer f32 32,32
%bo = buffer f32 32,32
copy 32,32 %b0 [0] = %in [0]
load 32,32 %x = %b0 [0]
%r = log2 %x
store 32,32 %bo [0] = %r
copy 32,32 %out [0] = %bo [0]
";

    #[test]
    fn the_transcendentals_stay_ops_for_the_lane_bodies() {
        // `log2`, `sin`, `cos` and `reciprocal` are lane bodies in `lower`
        // now; `pow`/`div`/`sqrt` are the compositions the pass still writes.
        let k = Kernel::parse(LOG2).expect("the program parses");
        let out = unfold(&k).expect("the unfold runs");
        out.verify().expect("the unfolded program verifies");
        assert!(
            out.ops()
                .any(|(_, op)| matches!(op, Op::Unary { op: UOp::Log2, .. })),
            "log2 must survive the unfold as the lane body's op"
        );
        let sqrt = LOG2.replace("%r = log2 %x", "%r = sqrt %x");
        let k = Kernel::parse(&sqrt).expect("the sqrt program parses");
        let out = unfold(&k).expect("the unfold runs");
        out.verify().expect("the unfolded sqrt verifies");
        assert!(
            !out.ops()
                .any(|(_, op)| matches!(op, Op::Unary { op: UOp::Sqrt, .. })),
            "sqrt is `rsqrt` then `reciprocal`"
        );
    }
}
