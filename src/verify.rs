// Copyright (C) 2026 zk4x
// SPDX-License-Identifier: Apache-2.0

//! [`Kernel::verify`]: the compiler's own invariant check, run on the program
//! the virtual passes produced, immediately before the lowering.
//!
//! A text program is checked where it is read ([`Kernel::parse`] and
//! [`Kernel::check`]); by the time the passes have run, every refusal a caller
//! could have caused is already behind us, so a program this rejects is this
//! compiler's own bug. Every failure names its reason on stderr and answers
//! [`ttStatus::TT_INTERNAL_BUG`] — there is no caller error left to spell with
//! its own code.
//!
//! The second half is the ring FIFO: it simulates each ring's
//! reserve/push/wait/pop traffic and proves the hand-offs `sync_cbs` placed
//! cannot deadlock on the board. A loop is analysed by its body, not by
//! multiplying a trip count: the body is walked once from the state the loop is
//! entered with, and
//!
//! - a literal trip count `n >= 2` also walks the body from the state of its
//!   last iteration (`base + (n - 1) * delta`), which is exact because every
//!   FIFO test is linear in the iteration index, so its extremes are the two
//!   ends;
//! - a trip count that is not a literal walks the body once and nothing more.
//!   The sections' loops run concurrently and the ring's own blocking (a
//!   `reserve` on a full ring, a `wait` on an empty one) absorbs any difference
//!   between them, so the traffic is proved at the end of the program: every
//!   ring may still hold pages — a producer one page ahead of its consumer is a
//!   normal pipeline — but no more than its own depth, and a consumer may never
//!   pop a page that was not pushed. A symbolic bound is therefore accepted,
//!   never refused (`tt-lang`'s `ttl-verify-dfb-lifecycle` rule).

use std::collections::BTreeMap;

use crate::ir::{Kernel, Op, OpId};
use crate::status::{internal_bug, ttStatus};

impl Kernel {
    /// Check the program the passes produced: the parse-level invariants still
    /// hold, and the ring hand-offs can complete.
    pub fn verify(&self) -> Result<(), ttStatus> {
        if let Err(status) = self.check() {
            return Err(internal_bug(format_args!(
                "verify: a pass left a program its own parse would refuse: {status}"
            )));
        }
        self.check_rings()
    }

    /// Simulate every ring's traffic and refuse a program whose producer or
    /// consumer would wait for a page that never comes.
    fn check_rings(&self) -> Result<(), ttStatus> {
        let ids: Vec<OpId> = self.ops().map(|(id, _)| id).collect();
        let mut ends = vec![usize::MAX; ids.len()];
        let mut open: Vec<usize> = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            match *self.at(*id) {
                Op::Loop { .. } => open.push(i),
                Op::EndLoop => {
                    let Some(start) = open.pop() else {
                        return Err(internal_bug("verify: an end-of-loop has no open loop"));
                    };
                    ends[start] = i;
                }
                _ => {}
            }
        }
        let mut state = Fifo::default();
        if let Err(what) =
            walk(self, &ids, 0, ids.len(), &ends, &mut state).and_then(|()| state.section_end())
        {
            return Err(internal_bug(format_args!("verify: {what}")));
        }
        // The tt-lang rule, not exact balance: a ring may end holding pages —
        // a producer that runs one page ahead of its consumer leaves the last
        // page published and unread, which the hardware neither notices nor
        // cares about — but the leftover may not exceed the ring's depth, and
        // the consumer may never pop more pages than were pushed.
        for (cb, pages) in state.rings.iter() {
            let depth = match ring_depth(self, *cb) {
                Ok(depth) => depth,
                Err(what) => return Err(internal_bug(format_args!("verify: {what}"))),
            };
            let outstanding = pages.received - pages.acked;
            if outstanding < 0 || outstanding > depth {
                return Err(internal_bug(format_args!(
                    "verify: ring {cb} ends holding {outstanding} page(s), depth {depth}"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The ring FIFO
// ---------------------------------------------------------------------------

/// One ring's counters, as the hand-offs move them.
///
/// `reserved` is pages a producer reserved and has not published; `waited` is
/// pages a consumer waited for and has not released. Both are open hand-offs
/// and must be zero wherever the scope that opened them ends. `received` and
/// `acked` are the ring's two hardware counters: the pages pushed, and the
/// pages popped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Pages {
    reserved: i64,
    received: i64,
    acked: i64,
    waited: i64,
}

impl Pages {
    /// Pages pushed to a consumer that it has not released or already taken.
    fn available(self) -> i64 {
        self.received - self.acked - self.waited
    }
}

/// The counters of every ring the program names.
#[derive(Clone, Default)]
struct Fifo {
    rings: BTreeMap<OpId, Pages>,
}

impl Fifo {
    /// The counters of `cb`, zero on first sight.
    fn at(&self, cb: OpId) -> Pages {
        self.rings.get(&cb).copied().unwrap_or_default()
    }

    fn at_mut(&mut self, cb: OpId) -> &mut Pages {
        self.rings.entry(cb).or_default()
    }

    /// What one op does to the ring it names. Every other op leaves the rings
    /// alone.
    fn step(&mut self, k: &Kernel, op: &Op) -> Result<(), &'static str> {
        match *op {
            // `cb_reserve_back`: poll until a page is free, then take it. The
            // pages a consumer has already popped are free; pages it has not
            // popped yet are what the program-wide check at the end accounts
            // for. What this catches is a reservation no push can ever
            // satisfy: more pages reserved at once than the ring holds.
            Op::Reserve { cb, n } => {
                let depth = ring_depth(k, cb)?;
                let asked = i64::from(n);
                if self.at(cb).reserved + asked > depth {
                    return Err("a reserve asks for more pages than the ring holds");
                }
                self.at_mut(cb).reserved += asked;
            }
            // `cb_push_back`: publish the page just filled.
            Op::Push { cb, n } => {
                let pushed = i64::from(n);
                if self.at(cb).reserved < pushed {
                    return Err("a push publishes a page that was never reserved");
                }
                let pages = self.at_mut(cb);
                pages.reserved -= pushed;
                pages.received += pushed;
            }
            // `cb_wait_front`: poll until the page is published.
            Op::Wait { cb, n } => {
                let asked = i64::from(n);
                if self.at(cb).available() < asked {
                    return Err("a wait asks for a page that was never pushed");
                }
                self.at_mut(cb).waited += asked;
            }
            // `cb_pop_front`: release the page just consumed.
            Op::Pop { cb, n } => {
                let released = i64::from(n);
                if self.at(cb).waited < released {
                    return Err("a pop releases a page that was never waited for");
                }
                let pages = self.at_mut(cb);
                pages.waited -= released;
                pages.acked += released;
            }
            _ => {}
        }
        Ok(())
    }

    /// A section ended: the core that owns an open hand-off is in another
    /// section, so nothing would ever complete it.
    fn section_end(&self) -> Result<(), &'static str> {
        for pages in self.rings.values() {
            if pages.reserved != 0 || pages.waited != 0 {
                return Err("a section ends with a reservation or a wait still open");
            }
        }
        Ok(())
    }

    /// The change `body` made to every ring, as the difference from `base`.
    fn delta(&self, base: &Fifo) -> BTreeMap<OpId, Pages> {
        let mut out: BTreeMap<OpId, Pages> = BTreeMap::new();
        for (cb, after) in self.rings.iter() {
            let before = base.at(*cb);
            out.insert(
                *cb,
                Pages {
                    reserved: after.reserved - before.reserved,
                    received: after.received - before.received,
                    acked: after.acked - before.acked,
                    waited: after.waited - before.waited,
                },
            );
        }
        out
    }

    /// `self`, with `delta` applied `times` times.
    fn advance(&self, delta: &BTreeMap<OpId, Pages>, times: i64) -> Fifo {
        let mut out = self.clone();
        for (cb, d) in delta {
            let pages = out.at_mut(*cb);
            pages.reserved += d.reserved * times;
            pages.received += d.received * times;
            pages.acked += d.acked * times;
            pages.waited += d.waited * times;
        }
        out
    }
}

/// Walk `ids[from..to]`, applying every hand-off and following every loop.
///
/// `ends[i]` is the matching `endloop` of the loop at `i`. A literal trip count
/// is exact at the two ends of the iteration range; any other count walks the
/// body once and lets the program-wide check do the proving.
fn walk(
    k: &Kernel,
    ids: &[OpId],
    from: usize,
    to: usize,
    ends: &[usize],
    state: &mut Fifo,
) -> Result<(), &'static str> {
    let mut i = from;
    while i < to {
        if let Op::Loop { bound } = *k.at(ids[i]) {
            let body_end = ends[i];
            if body_end == usize::MAX || body_end >= to {
                return Err("a loop's end is not inside the walked range");
            }
            match k.const_i32(bound) {
                // Nothing runs: no traffic.
                Some(0) => {}
                Some(n) if n > 0 => {
                    let trips = i64::from(n);
                    if trips == 1 {
                        walk(k, ids, i + 1, body_end, ends, state)?;
                    } else {
                        let base = state.clone();
                        // The first iteration, from the state the loop is
                        // entered with.
                        walk(k, ids, i + 1, body_end, ends, state)?;
                        let delta = state.delta(&base);
                        // The last one: every FIFO test is linear in the
                        // iteration index, so between the two ends lies
                        // nothing worse. Walking the body from the start of
                        // iteration `trips - 1` leaves the state after
                        // iteration `trips - 1`, which is the loop's own.
                        let mut last = base.advance(&delta, trips - 1);
                        walk(k, ids, i + 1, body_end, ends, &mut last)?;
                        *state = last;
                    }
                }
                _ => {
                    // A count this build cannot read as a literal — a symbolic
                    // bound, or a negative datum the lowering also treats
                    // symbolically: walk the body once. Every iteration starts
                    // from the state the body leaves behind, and two loops of
                    // one ring only have to agree on their per-iteration
                    // traffic — the ring's own blocking (reserve on full, wait
                    // on empty) makes up the difference. The program-wide page
                    // check at the end is what proves the traffic matches, so
                    // nothing is refused merely for having a symbolic bound.
                    walk(k, ids, i + 1, body_end, ends, state)?;
                }
            }
            i = body_end + 1;
            continue;
        }
        let op = k.at(ids[i]);
        state.step(k, op)?;
        if matches!(*op, Op::EndReader | Op::EndCompute) {
            state.section_end()?;
        }
        i += 1;
    }
    Ok(())
}

/// The pages a ring holds, from its declaration.
fn ring_depth(k: &Kernel, cb: OpId) -> Result<i64, &'static str> {
    match *k.at(cb) {
        Op::Cb { len, .. } => Ok(i64::from(len)),
        _ => Err("a hand-off names a value that is not a ring"),
    }
}
