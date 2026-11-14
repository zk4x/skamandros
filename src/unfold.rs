// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! `unfold`: the transcendentals written as the primitives the IR has.
//!
//! zyx does this in `tt_unfold`, on the way into its backend, so that the
//! native lowering only ever sees `add`/`sub`/`mul`, the bit ops, `Bitcast`,
//! `Cast`, comparisons and `Select`. skamandros does it itself: a caller writes
//! `log2`, `sin`, `cos`, `pow` or `div` and this pass turns each into that
//! primitive form, in the same arithmetic zyx's bodies use.
//!
//! `exp2` is the one transcendental that stays an IR op. Its exponent assembly
//! is `SFPEXEXP`/`SFPSETEXP`, which no IR op spells, so `lower` emits it as a
//! machine body instead of it being unfolded here.
//!
//! The pass runs on the unified IR, before `lower`: every value it creates is
//! an ordinary value, and the allocator is what turns a chain into one row
//! body.
//!
//! Nothing here is a new op: `log2`/`sin`/`cos` are `UOp`s, `pow`/`div` are
//! `BOp`s, and both are part of the program format already. A `pow` whose
//! operands are not floats is left alone, and `lower` refuses it loudly.

use std::collections::BTreeMap;

use crate::ir::{BOp, DType, Error, Kernel, Op, OpId, UOp};

/// Rewrite every transcendental `k` has into its primitive form.
pub fn unfold(k: &Kernel) -> Result<Kernel, Error> {
    let mut b = Build {
        out: Kernel::default(),
        old: k,
        map: BTreeMap::new(),
        seq: 0,
    };
    for (id, op) in k.ops() {
        match *op {
            Op::Unary {
                op: uop @ (UOp::Log2 | UOp::Sin | UOp::Cos),
                a,
            } => {
                let x = b.map_of(a);
                let out = match uop {
                    UOp::Log2 => b.log2(x),
                    UOp::Sin => b.sin_cos(x, false),
                    _ => b.sin_cos(x, true),
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
                        let l = b.log2(x);
                        let p = b.bin(BOp::Mul, y, l, "pow_mul");
                        b.un(UOp::Exp2, p, "pow_exp2")
                    }
                    // `x / y = x * (1/y)`, the reciprocal being tt-metal's
                    // `_sfpu_reciprocal_`: a quadratic estimate and then one
                    // Newton step per datum width, as `ckernel_sfpu_recip.h`
                    // documents.
                    (BOp::Div, true) => {
                        let steps = if k.dtype(a) == DType::Bf16 { 1 } else { 2 };
                        let r = b.reciprocal(y, steps);
                        b.bin(BOp::Mul, x, r, "div_mul")
                    }
                    _ => {
                        b.copy(id, op.clone());
                        continue;
                    }
                };
                b.name(id, out);
                b.map.insert(id, out);
            }
            _ => b.copy(id, op.clone()),
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

    fn ki(&mut self, x: u32, tag: &str) -> OpId {
        self.value(tag);
        self.out.push(Op::Const {
            dtype: DType::I32,
            bits: x,
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

    fn bitcast(&mut self, src: OpId, dtype: DType, tag: &str) -> OpId {
        self.value(tag);
        self.out.push(Op::Bitcast { dtype, src })
    }

    fn select(&mut self, cond: OpId, t: OpId, f: OpId, tag: &str) -> OpId {
        self.value(tag);
        self.out.push(Op::Select {
            cond,
            if_true: t,
            if_false: f,
        })
    }

    /// `1 + s (c1 + s (c2 + ...))` over `coeffs`, the non-constant terms in
    /// increasing power of `s` (zyx's `tt_horner`).
    fn horner(&mut self, s: OpId, coeffs: &[f32], tag: &str) -> OpId {
        let mut p = self.kf(coeffs[coeffs.len() - 1], tag);
        for c in coeffs[..coeffs.len() - 1].iter().rev() {
            let cc = self.kf(*c, tag);
            let q = self.bin(BOp::Mul, p, s, tag);
            p = self.bin(BOp::Add, q, cc, tag);
        }
        let one = self.kf(1.0, tag);
        let q = self.bin(BOp::Mul, p, s, tag);
        self.bin(BOp::Add, q, one, tag)
    }

    /// `log2 x` by tt-metal's `calculate_log_body` (Norbert Juffa's `logf`),
    /// zyx's body: the exponent falls out of `bits(x) - bits(0.75)` with the
    /// mantissa masked off, leaving `x = m 2^k` with `m` in `[0.75, 1.5)`;
    /// `log1p(m - 1)` is the LLK's shared degree-9 polynomial, and
    /// `log2 x = k + log2(m)` folds the `ln 2`/`log2 e` round trip into one
    /// multiply. The specials (`-inf`/`NaN` inputs) are dropped, the same ML
    /// approximation `tt_exp2` takes.
    fn log2(&mut self, x: OpId) -> OpId {
        let bits = self.bitcast(x, DType::I32, "log2_bits");
        let c75 = self.ki(0x3f40_0000, "log2_c75");
        let e = self.bin(BOp::Sub, bits, c75, "log2_e");
        let emask = self.ki(0xff80_0000, "log2_emask");
        let e = self.bin(BOp::BitAnd, e, emask, "log2_e");
        let m = self.bin(BOp::Sub, bits, e, "log2_m");
        let m = self.bitcast(m, DType::F32, "log2_m");
        let one = self.kf(1.0, "log2_one");
        let m = self.bin(BOp::Sub, m, one, "log2_m");
        let s = self.bin(BOp::Mul, m, m, "log2_s");
        let mut r = self.kf(f32::from_bits(0xbd49_6000), "log2_r");
        for c in [
            0x3ddc_2000u32,
            0xbe06_2000,
            0x3e13_a000,
            0xbe2a_8000,
            0x3e4c_c000,
            0xbe80_000d,
            0x3eaa_aab9,
            0xbf00_0000,
        ] {
            let cc = self.kf(f32::from_bits(c), "log2_c");
            let p = self.bin(BOp::Mul, r, m, "log2_p");
            r = self.bin(BOp::Add, p, cc, "log2_r");
        }
        let r = self.bin(BOp::Mul, r, s, "log2_r");
        let r = self.bin(BOp::Add, r, m, "log2_r");
        // `e` is `k << 23`, so `e * 2^-23` is exactly `k`.
        let ef = self.cast(e, DType::F32, "log2_k");
        let two_m23 = self.kf(f32::from_bits(0x3400_0000), "log2_two_m23");
        let kf = self.bin(BOp::Mul, ef, two_m23, "log2_k");
        let log2e = self.kf(std::f32::consts::LOG2_E, "log2_log2e");
        let rl = self.bin(BOp::Mul, r, log2e, "log2_rl");
        self.bin(BOp::Add, kf, rl, "log2_out")
    }

    /// `sin x` / `cos x`: the classical float reduction (`n = round(x 2/pi)`
    /// by the magic constant, a two-limb `pi/2` so `|r| <= pi/4`) and a
    /// five-term Taylor core, with the quadrant from `q = n & 3` — its low bit
    /// picks `sin` or `cos`, `q >> 1` (for cosine `((q + 1) >> 1) & 1`) flips
    /// the sign. zyx's `tt_sin_cos`.
    fn sin_cos(&mut self, x: OpId, cos: bool) -> OpId {
        let pio2_hi = f32::from_bits((std::f64::consts::FRAC_PI_2 as f32).to_bits() & !0xfff);
        let pio2_lo = (std::f64::consts::FRAC_PI_2 - f64::from(pio2_hi)) as f32;
        let magic = self.kf(12_582_912.0, "trig_magic");
        let inv_pio2 = self.kf(std::f32::consts::FRAC_2_PI, "trig_inv_pio2");
        let hi = self.kf(pio2_hi, "trig_pio2_hi");
        let lo = self.kf(pio2_lo, "trig_pio2_lo");
        let t = self.bin(BOp::Mul, x, inv_pio2, "trig_t");
        let t = self.bin(BOp::Add, t, magic, "trig_t");
        let nf = self.bin(BOp::Sub, t, magic, "trig_nf");
        let n = self.cast(nf, DType::I32, "trig_n");
        let a = self.bin(BOp::Mul, nf, hi, "trig_a");
        let r = self.bin(BOp::Sub, x, a, "trig_r");
        let b = self.bin(BOp::Mul, nf, lo, "trig_b");
        let r = self.bin(BOp::Sub, r, b, "trig_r");
        let s = self.bin(BOp::Mul, r, r, "trig_s");
        let sin_core = self.horner(
            s,
            &[
                -1.0 / 6.0,
                1.0 / 120.0,
                -1.0 / 5040.0,
                1.0 / 362_880.0,
                -1.0 / 39_916_800.0,
            ],
            "trig_sin",
        );
        let cos_core = self.horner(
            s,
            &[
                -0.5,
                1.0 / 24.0,
                -1.0 / 720.0,
                1.0 / 40_320.0,
                -1.0 / 3_628_800.0,
            ],
            "trig_cos",
        );
        let sin_r = self.bin(BOp::Mul, r, sin_core, "trig_sin_r");
        // `q`'s low bit picks the core, `q >> 1` the sign.
        let three = self.ki(3, "trig_three");
        let one_i = self.ki(1, "trig_one_i");
        let zero_i = self.ki(0, "trig_zero_i");
        let q = self.bin(BOp::BitAnd, n, three, "trig_q");
        let low = self.bin(BOp::BitAnd, q, one_i, "trig_low");
        let odd = self.bin(BOp::NotEq, low, zero_i, "trig_odd");
        let neg = if cos {
            let q1 = self.bin(BOp::Add, q, one_i, "trig_q1");
            let half = self.bin(BOp::BitShiftRight, q1, one_i, "trig_half");
            let bit = self.bin(BOp::BitAnd, half, one_i, "trig_bit");
            self.bin(BOp::NotEq, bit, zero_i, "trig_neg")
        } else {
            let half = self.bin(BOp::BitShiftRight, q, one_i, "trig_half");
            let bit = self.bin(BOp::BitAnd, half, one_i, "trig_bit");
            self.bin(BOp::NotEq, bit, zero_i, "trig_neg")
        };
        let (odd_core, even_core) = if cos {
            (sin_r, cos_core)
        } else {
            (cos_core, sin_r)
        };
        let base = self.select(odd, odd_core, even_core, "trig_base");
        let minus_one = self.kf(-1.0, "trig_minus_one");
        let flipped = self.bin(BOp::Mul, base, minus_one, "trig_flipped");
        self.select(neg, flipped, base, "trig_out")
    }

    /// `1/y` by tt-metal's `_sfpu_reciprocal_` (`ckernel_sfpu_recip.h`): a
    /// quadratic estimate over the mantissa scaled to `[1, 2)`, `steps` Newton
    /// iterations (`t = 1 - x y; y = y + y t`), then the exponent restored with
    /// `~bits(y)`'s field and the sign copied from the input.
    fn reciprocal(&mut self, y: OpId, steps: usize) -> OpId {
        let bits = self.bitcast(y, DType::I32, "recip_bits");
        // `setman(-1.0, y)`: the sign and exponent of `-1.0`, the mantissa of
        // the input, which makes the value negative and scales it to `[1, 2)`.
        let mant = self.ki(0x007f_ffff, "recip_mant");
        let m = self.bin(BOp::BitAnd, bits, mant, "recip_mant");
        let neg_one = self.ki(0xbf80_0000, "recip_neg_one");
        let neg_bits = self.bin(BOp::BitOr, m, neg_one, "recip_neg_bits");
        let neg = self.bitcast(neg_bits, DType::F32, "recip_neg");
        // The quadratic estimate `k2 - k1 x + k0 x^2`.
        let k0 = self.kf(0.323_232_501_745_224, "recip_k0");
        let k1 = self.kf(1.454_545_974_731_445_3, "recip_k1");
        let k2 = self.kf(2.121_212_482_452_392_6, "recip_k2");
        let p = self.bin(BOp::Mul, k0, neg, "recip_p");
        let mut est = self.bin(BOp::Add, k1, p, "recip_est");
        let p = self.bin(BOp::Mul, est, neg, "recip_p");
        est = self.bin(BOp::Add, k2, p, "recip_est");
        // `scale = ~bits(y)` with the mantissa cleared, halved.
        let not = self.un(UOp::BitNot, bits, "recip_not");
        let emask = self.ki(0xff80_0000, "recip_emask");
        let sbits = self.bin(BOp::BitAnd, not, emask, "recip_scale");
        let scale = self.bitcast(sbits, DType::F32, "recip_scale");
        let half = self.kf(0.5, "recip_half");
        let scale = self.bin(BOp::Mul, scale, half, "recip_scale");
        let one = self.kf(1.0, "recip_one");
        // `t = 1 + neg y` is `1 - x y` with the sign folded into `neg`.
        for _ in 0..steps.max(1) {
            let ny = self.bin(BOp::Mul, neg, est, "recip_ny");
            let t = self.bin(BOp::Add, one, ny, "recip_t");
            let yt = self.bin(BOp::Mul, est, t, "recip_yt");
            est = self.bin(BOp::Add, est, yt, "recip_est");
        }
        let est = self.bin(BOp::Mul, est, scale, "recip_scaled");
        // `copysgn(est, y)`: the sign comes from the input, the rest from the
        // estimate.
        let ebits = self.bitcast(est, DType::I32, "recip_est_bits");
        let abs_mask = self.ki(0x7fff_ffff, "recip_abs_mask");
        let mag = self.bin(BOp::BitAnd, ebits, abs_mask, "recip_mag");
        let sign_mask = self.ki(0x8000_0000, "recip_sign_mask");
        let sign = self.bin(BOp::BitAnd, bits, sign_mask, "recip_sign");
        let signed = self.bin(BOp::BitOr, mag, sign, "recip_signed");
        self.bitcast(signed, DType::F32, "recip_out")
    }

    /// Copy an op unchanged, with its operands remapped. A value *is* its
    /// defining op, so a copied value takes the id of the copy.
    fn copy(&mut self, old: OpId, op: Op) {
        let mut op = op;
        crate::ir::renumber(&mut op, &self.map).expect("a use follows its definition");
        let id = self.out.push(op);
        self.map.insert(old, id);
        if let Some(name) = self.old.names.get(&old) {
            self.out.names.insert(id, name.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG2: &str = "\
version 9
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
    fn log2_leaves_no_transcendental_behind() {
        let k = Kernel::parse(LOG2).expect("the program parses");
        let out = unfold(&k).expect("the unfold runs");
        out.verify().expect("the unfolded program verifies");
        assert!(
            !out.ops().any(|(_, op)| matches!(
                op,
                Op::Unary {
                    op: UOp::Log2 | UOp::Sin | UOp::Cos,
                    ..
                }
            )),
            "the unfolded program still has a transcendental"
        );
    }
}
