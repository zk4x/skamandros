// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Constant-bound loops, unrolled.
//!
//! A loop whose bound is a literal (or folds to one) is a shorthand for its body
//! written out that many times, with the induction value replaced by the
//! iteration's literal. Unrolling it here, before anything reads the program,
//! makes every offset inside the body a literal — which is what the movers and
//! the unpack/pack encoders take today — and lets the rest of the pipeline keep
//! refusing a loop it cannot yet run.
//!
//! Constant index arithmetic is folded with it, because an offset written as
//! `%o = mul %i %step` is the normal way to address iteration `%i`; leaving the
//! `mul` in place would unroll the loop and still have no literal offset. Every
//! fold is exact two's-complement `i32`, and anything that cannot be folded
//! exactly (division by zero, a shift outside `0..32`) is an error, never a
//! guess.
//!
//! Values keep their names on the first copy; a later copy gets `#n` appended,
//! because two definitions of one name are refused by the verifier.

use std::collections::BTreeMap;

use crate::ir::{BinOp, DType, Dim, Error, Kernel, Op, ValueId};

/// Unroll every loop whose bound is constant, and fold the index arithmetic
/// inside the copies. A loop with a symbolic bound is left in place.
pub fn constant_loops(k: &Kernel) -> Result<Kernel, Error> {
    let mut u = Unroller {
        k,
        out: Kernel {
            ops: Vec::new(),
            names: Vec::new(),
        },
        folds: BTreeMap::new(),
        unrolled_body: 0,
    };
    let mut subs = BTreeMap::new();
    u.emit(0, k.ops.len(), &mut subs)?;
    Ok(u.out)
}

struct Unroller<'a> {
    k: &'a Kernel,
    out: Kernel,
    /// The literal value of every `i32` scalar definition, in the *output* id
    /// space. This is what makes a bound or an offset constant.
    folds: BTreeMap<ValueId, i32>,
    /// Non-zero while emitting a copy of an unrolled loop body, where index
    /// arithmetic is folded into literals.
    unrolled_body: u32,
}

impl Unroller<'_> {
    fn emit(
        &mut self,
        mut at: usize,
        end: usize,
        subs: &mut BTreeMap<ValueId, ValueId>,
    ) -> Result<(), Error> {
        while at < end {
            if let Op::Loop { value, bound } = self.k.ops[at] {
                let bound = self.dim(bound, subs)?;
                if let Dim::Const(n) = bound {
                    let close = self.matching_end(at, end)?;
                    for iteration in 0..n {
                        let mut inner = subs.clone();
                        let name = self
                            .fresh_name(&format!("{}~{iteration}", self.k.names[value as usize]));
                        let const_id = self.define(name)?;
                        self.out.ops.push(Op::Const {
                            value: const_id,
                            dtype: DType::I32,
                            bits: iteration,
                        });
                        self.folds.insert(const_id, iteration as i32);
                        inner.insert(value, const_id);
                        self.unrolled_body += 1;
                        self.emit(at + 1, close, &mut inner)?;
                        self.unrolled_body -= 1;
                    }
                    at = close + 1;
                    continue;
                }
            }
            self.copy(at, subs)?;
            at += 1;
        }
        Ok(())
    }

    /// Copy one instruction, resolving its operands through `subs`.
    fn copy(&mut self, at: usize, subs: &mut BTreeMap<ValueId, ValueId>) -> Result<(), Error> {
        let new = match self.k.ops[at] {
            Op::Const { value, dtype, bits } => {
                let id = self.redefine(value)?;
                if dtype == DType::I32 {
                    self.folds.insert(id, bits as i32);
                }
                Op::Const {
                    value: id,
                    dtype,
                    bits,
                }
            }
            Op::Param {
                value,
                dtype,
                ref dims,
            } => {
                let id = self.redefine(value)?;
                let dims = dims
                    .iter()
                    .map(|d| self.dim(*d, subs))
                    .collect::<Result<Vec<_>, _>>()?;
                Op::Param {
                    value: id,
                    dtype,
                    dims,
                }
            }
            Op::Buffer {
                value,
                dtype,
                shape,
            } => {
                let id = self.redefine(value)?;
                Op::Buffer {
                    value: id,
                    dtype,
                    shape,
                }
            }
            Op::Grid { columns, rows } => Op::Grid {
                columns: self.dim(columns, subs)?,
                rows: self.dim(rows, subs)?,
            },
            Op::Loop { value, bound } => {
                let id = self.redefine(value)?;
                let bound = self.dim(bound, subs)?;
                subs.insert(value, id);
                Op::Loop { value: id, bound }
            }
            Op::EndLoop => Op::EndLoop,
            Op::CopyTo { dst, src, offset } => Op::CopyTo {
                dst: self.map(dst, subs)?,
                src: self.map(src, subs)?,
                offset: self.dim(offset, subs)?,
            },
            Op::Load {
                value,
                buffer,
                offset,
            } => {
                let id = self.redefine(value)?;
                Op::Load {
                    value: id,
                    buffer: self.map(buffer, subs)?,
                    offset: self.dim(offset, subs)?,
                }
            }
            Op::StoreTo {
                value,
                buffer,
                offset,
            } => Op::StoreTo {
                value: self.map(value, subs)?,
                buffer: self.map(buffer, subs)?,
                offset: self.dim(offset, subs)?,
            },
            Op::Cast { value, src, dtype } => {
                let id = self.redefine(value)?;
                Op::Cast {
                    value: id,
                    src: self.map(src, subs)?,
                    dtype,
                }
            }
            Op::Unary { value, op, a } => {
                let id = self.redefine(value)?;
                Op::Unary {
                    value: id,
                    op,
                    a: self.map(a, subs)?,
                }
            }
            Op::Binary {
                value,
                op,
                a,
                b: rhs,
            } => {
                let a = self.map(a, subs)?;
                let rhs = self.map(rhs, subs)?;
                let folded = if self.unrolled_body > 0 {
                    self.fold(op, a, rhs)
                } else {
                    None
                };
                match folded {
                    Some(n) => {
                        let id = self.redefine(value)?;
                        self.folds.insert(id, n);
                        Op::Const {
                            value: id,
                            dtype: DType::I32,
                            bits: n as u32,
                        }
                    }
                    None => {
                        let id = self.redefine(value)?;
                        Op::Binary {
                            value: id,
                            op,
                            a,
                            b: rhs,
                        }
                    }
                }
            }
            Op::Select {
                value,
                cond,
                if_true,
                if_false,
            } => {
                let id = self.redefine(value)?;
                Op::Select {
                    value: id,
                    cond: self.map(cond, subs)?,
                    if_true: self.map(if_true, subs)?,
                    if_false: self.map(if_false, subs)?,
                }
            }
            Op::Matmul { value, a, b: rhs } => {
                let id = self.redefine(value)?;
                Op::Matmul {
                    value: id,
                    a: self.map(a, subs)?,
                    b: self.map(rhs, subs)?,
                }
            }
        };
        if let (Some(old), Some(id)) = (self.k.ops[at].defined(), new.defined()) {
            subs.insert(old, id);
        }
        self.out.ops.push(new);
        Ok(())
    }

    /// The literal `a op b`, when both are known `i32` and the result is exact.
    fn fold(&self, op: BinOp, a: ValueId, b: ValueId) -> Option<i32> {
        let a = *self.folds.get(&a)?;
        let b = *self.folds.get(&b)?;
        let value = match op {
            BinOp::Add => a.wrapping_add(b),
            BinOp::Sub => a.wrapping_sub(b),
            BinOp::Mul => a.wrapping_mul(b),
            BinOp::Div => a.checked_div(b)?,
            BinOp::Mod => a.checked_rem(b)?,
            BinOp::Pow => a.checked_pow(u32::try_from(b).ok()?)?,
            BinOp::Max => a.max(b),
            BinOp::BitAnd => a & b,
            BinOp::BitOr => a | b,
            BinOp::BitXor => a ^ b,
            BinOp::BitShiftLeft => a.checked_shl(u32::try_from(b).ok()?)?,
            BinOp::BitShiftRight => a.checked_shr(u32::try_from(b).ok()?)?,
            // A comparison makes a bool and a logical op takes one; neither is
            // an `i32` constant to fold into an index.
            BinOp::Cmplt
            | BinOp::Cmpgt
            | BinOp::Cmpge
            | BinOp::Eq
            | BinOp::NotEq
            | BinOp::Or
            | BinOp::And => return None,
        };
        Some(value)
    }

    fn dim(&self, dim: Dim, subs: &BTreeMap<ValueId, ValueId>) -> Result<Dim, Error> {
        match dim {
            Dim::Const(n) => Ok(Dim::Const(n)),
            Dim::Value(v) => {
                let v = self.map(v, subs)?;
                match self.folds.get(&v) {
                    Some(n) if *n < 0 => Err(Error(format!(
                        "the value %{v} is {n}; an index cannot be negative"
                    ))),
                    Some(n) => Ok(Dim::Const(*n as u32)),
                    None => Ok(Dim::Value(v)),
                }
            }
        }
    }

    fn map(&self, v: ValueId, subs: &BTreeMap<ValueId, ValueId>) -> Result<ValueId, Error> {
        subs.get(&v).copied().ok_or_else(|| {
            Error(format!(
                "%{v} is used where it is not in scope; the program did not verify"
            ))
        })
    }

    /// The output id for a copied definition, with its name recorded.
    fn redefine(&mut self, old: ValueId) -> Result<ValueId, Error> {
        let name = self.fresh_name(&self.k.names[old as usize]);
        self.define(name)
    }

    fn define(&mut self, name: String) -> Result<ValueId, Error> {
        if self.out.names.iter().any(|n| *n == name) {
            return Err(Error(format!("the unrolled program defines %{name} twice")));
        }
        let id = self.out.names.len() as ValueId;
        self.out.names.push(name);
        Ok(id)
    }

    fn fresh_name(&self, base: &str) -> String {
        if !self.out.names.iter().any(|n| n == base) {
            return base.to_string();
        }
        for k in 1.. {
            let candidate = format!("{base}#{k}");
            if !self.out.names.iter().any(|n| *n == candidate) {
                return candidate;
            }
        }
        unreachable!("a free name always exists")
    }

    /// The `endloop` matching the `loop` at `at`.
    fn matching_end(&self, at: usize, end: usize) -> Result<usize, Error> {
        let mut depth = 0usize;
        for (index, op) in self.k.ops.iter().enumerate().take(end).skip(at) {
            match op {
                Op::Loop { .. } => depth += 1,
                Op::EndLoop => {
                    depth = depth
                        .checked_sub(1)
                        .ok_or_else(|| Error(format!("op {index}: endloop without a loop")))?;
                    if depth == 0 {
                        return Ok(index);
                    }
                }
                _ => {}
            }
        }
        Err(Error(format!(
            "the loop at op {at} has no matching endloop"
        )))
    }
}
