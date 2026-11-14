//! Two-lane SIMD for a chunk's scalar function (#2786): the lane-parallel arithmetic of a
//! dual's tangent lanes, as `f64x2` operations.
//!
//! A chunk's [`Func`] is scalar — one instruction per lane of every dual operation — and a
//! slot's lanes are contiguous in the buffer. So the same operation on two neighbouring lanes,
//! reading neighbouring lanes (or one shared scalar: a dual's value lane), is one vector
//! operation. [`packs`] finds them bottom-up from the slots' loads, the way an SLP vectorizer
//! does: a pair of loads of lanes `k`, `k + 1` of a slot is a pack; a pair of instructions of
//! the same operation whose operands are, position by position, a pack or one shared value is
//! a pack; a pair of stores of a pack's lanes to neighbouring lanes is one vector store. Each
//! lane computes exactly its scalar instruction — the same IEEE operation on the same operands,
//! nothing reassociated — so the vector code gives the scalar tier's bits.
//!
//! A pack is emitted where its later member was, so the nodes are rescheduled: a topological
//! order of the packs and the remaining scalars ([`Packs::order`]) that keeps every memory
//! access and host call in program order relative to the ones it could alias.
use super::ir::{self, Inst, Shim1, V};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// One node of the vectorized function: a scalar instruction or a pack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Node {
    Scalar(V),
    Pack(u32),
}

/// A function's packs and the order its nodes are emitted in.
pub(super) struct Packs {
    /// Per instruction: its pack and lane, when packed.
    pub(super) of: Vec<Option<(u32, u8)>>,
    /// Each pack's members, lane 0 then lane 1.
    pub(super) members: Vec<[V; 2]>,
    /// Every node, in an order that respects each node's operands and memory order; the
    /// nodes of a reverse's forward prefix all come before its seed loop's.
    pub(super) order: Vec<Node>,
}

/// An operation that can be a pack (beyond loads and stores).
fn vector_op(i: &Inst) -> bool {
    matches!(
        i,
        Inst::Add(..)
            | Inst::Sub(..)
            | Inst::Mul(..)
            | Inst::Div(..)
            | Inst::Neg(..)
            | Inst::Abs(..)
            | Inst::Call1(Shim1::Sqrt, _)
    )
}

/// Operand `j` of `i` ([`operands`]' order).
pub(super) fn operand(i: &Inst, j: usize) -> V {
    operands(i).0[j]
}

/// The value operands of `i` (a store's stored value included).
pub(super) fn operands(i: &Inst) -> ([V; 3], usize) {
    match *i {
        Inst::Add(a, b)
        | Inst::Sub(a, b)
        | Inst::Mul(a, b)
        | Inst::Div(a, b)
        | Inst::Powf(a, b)
        | Inst::Cmp(_, a, b)
        | Inst::And(a, b)
        | Inst::Or(a, b)
        | Inst::Gamma { a, x: b, .. } => ([a, b, 0], 2),
        Inst::Neg(a)
        | Inst::Abs(a)
        | Inst::Floor(a)
        | Inst::Ceil(a)
        | Inst::Call1(_, a)
        | Inst::StopGrad(a)
        | Inst::HostArg(a)
        | Inst::Store { v: a, .. } => ([a, 0, 0], 1),
        Inst::Select(c, a, b) => ([c, a, b], 3),
        Inst::Load { .. } | Inst::Const(_) | Inst::Host(_) | Inst::HostOut(_) => ([0; 3], 0),
    }
}

/// The two instructions are the same operation (operands aside).
fn same_op(a: &Inst, b: &Inst) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
        && match (a, b) {
            (Inst::Call1(x, _), Inst::Call1(y, _)) => x == y,
            _ => true,
        }
}

/// Which buffer a memory access reads or writes: the slots (`0`), the adjoints (`1`), or
/// none (the read-only literals).
fn mem_class(arg: u8) -> Option<u8> {
    match arg {
        0 | 3 => Some(0),
        4 | 7 => Some(1),
        _ => None,
    }
}

/// The packs of `f` (`lanes` lanes per slot; with `rev`, its first `prefix` instructions are
/// the reverse's forward, the rest its seed loop's body). `None` when nothing packs.
pub(super) fn packs(f: &ir::Func, lanes: usize, rev: bool, prefix: usize) -> Option<Packs> {
    let insts = &f.insts;
    let n = insts.len();
    // A host call runs on a queue its instructions fill in program order: left scalar.
    if lanes < 3
        || insts
            .iter()
            .any(|i| matches!(i, Inst::Host(_) | Inst::HostArg(_) | Inst::HostOut(_)))
    {
        return None;
    }
    let region = |i: usize| (rev && i >= prefix) as u8;
    let mut users: Vec<Vec<V>> = vec![Vec::new(); n];
    for (at, i) in insts.iter().enumerate() {
        let (ops, k) = operands(i);
        for &o in &ops[..k] {
            users[o as usize].push(at as V);
        }
    }
    // The last store into each buffer before an instruction, to keep a pair of loads from
    // straddling one.
    let mut last_store: Vec<[Option<u32>; 2]> = Vec::with_capacity(n);
    {
        let mut cur = [None, None];
        for (at, i) in insts.iter().enumerate() {
            last_store.push(cur);
            if let Inst::Store { arg, .. } = *i {
                if let Some(c) = mem_class(arg) {
                    cur[c as usize] = Some(at as u32);
                }
            }
        }
    }
    let mut of: Vec<Option<(u32, u8)>> = vec![None; n];
    let mut members: Vec<[V; 2]> = Vec::new();
    let mut work: Vec<u32> = Vec::new();
    fn add(
        of: &mut [Option<(u32, u8)>],
        members: &mut Vec<[V; 2]>,
        m: [V; 2],
        work: &mut Vec<u32>,
    ) {
        let p = members.len() as u32;
        of[m[0] as usize] = Some((p, 0));
        of[m[1] as usize] = Some((p, 1));
        members.push(m);
        work.push(p);
    }
    // Seeds: the loads of neighbouring lanes of a slot, paired from lane 1 (lane 0, the value,
    // is computed apart from its tangents).
    let mut loads: HashMap<(u8, u32), V> = HashMap::new();
    for (at, i) in insts.iter().enumerate() {
        if let Inst::Load { arg, off } = *i {
            if mem_class(arg).is_some() {
                loads.insert((arg, off), at as V);
            }
        }
    }
    let mut keys: Vec<(u8, u32)> = loads.keys().copied().collect();
    keys.sort_unstable();
    for (arg, off) in keys {
        let lane = off as usize % lanes;
        if lane % 2 != 1 || lane + 1 >= lanes {
            continue;
        }
        let (Some(&a), Some(&b)) = (loads.get(&(arg, off)), loads.get(&(arg, off + 1))) else {
            continue;
        };
        let c = mem_class(arg).expect("a buffer") as usize;
        if region(a as usize) != region(b as usize)
            || last_store[a as usize][c] != last_store[b as usize][c]
        {
            continue;
        }
        add(&mut of, &mut members, [a, b], &mut work);
    }
    // Whether `to` depends on `from` (`from < to`): a backward walk from `to`, bounded.
    let depends = |to: V, from: V| -> bool {
        let mut stack = vec![to];
        let mut seen = 0usize;
        while let Some(x) = stack.pop() {
            if x == from {
                return true;
            }
            seen += 1;
            if seen > 256 {
                return true;
            }
            let (ops, k) = operands(&insts[x as usize]);
            for &o in &ops[..k] {
                if o >= from {
                    stack.push(o);
                }
            }
        }
        false
    };
    // Grow: the users of a pack's lanes that pair up.
    while let Some(p) = work.pop() {
        let [p0, p1] = members[p as usize];
        for &u0 in &users[p0 as usize] {
            if of[u0 as usize].is_some() || !vector_op(&insts[u0 as usize]) {
                continue;
            }
            let (ops0, k) = operands(&insts[u0 as usize]);
            let found = users[p1 as usize].iter().copied().find(|&u1| {
                if u1 == u0
                    || of[u1 as usize].is_some()
                    || !same_op(&insts[u0 as usize], &insts[u1 as usize])
                    || region(u0 as usize) != region(u1 as usize)
                {
                    return false;
                }
                let (ops1, _) = operands(&insts[u1 as usize]);
                let fits = (0..k).all(|j| {
                    let (x0, x1) = (ops0[j], ops1[j]);
                    // A shared operand is a splat — unless it is one of the new pair.
                    (x0 == x1 && x0 != u0 && x0 != u1)
                        || of[x0 as usize]
                            .is_some_and(|(q, l)| l == 0 && members[q as usize][1] == x1)
                });
                let (lo, hi) = (u0.min(u1), u0.max(u1));
                fits && !depends(hi, lo)
            });
            if let Some(u1) = found {
                add(&mut of, &mut members, [u0, u1], &mut work);
            }
        }
    }
    // Stores of a pack's lanes to neighbouring lanes of a slot.
    let mut stores: HashMap<(u8, u32), V> = HashMap::new();
    for (at, i) in insts.iter().enumerate() {
        if let Inst::Store { arg, off, .. } = *i {
            if (arg == 3 && !rev) || (arg == 4 && rev) {
                stores.insert((arg, off), at as V);
            }
        }
    }
    let mut keys: Vec<(u8, u32)> = stores.keys().copied().collect();
    keys.sort_unstable();
    for (arg, off) in keys {
        if (off as usize % lanes) + 1 >= lanes {
            continue;
        }
        let (Some(&a), Some(&b)) = (stores.get(&(arg, off)), stores.get(&(arg, off + 1))) else {
            continue;
        };
        if of[a as usize].is_some() || of[b as usize].is_some() {
            continue;
        }
        let (Inst::Store { v: va, .. }, Inst::Store { v: vb, .. }) =
            (insts[a as usize], insts[b as usize])
        else {
            continue;
        };
        let paired = of[va as usize].is_some_and(|(q, l)| l == 0 && members[q as usize][1] == vb);
        if !paired || region(a as usize) != region(b as usize) {
            continue;
        }
        // Nothing of the same buffer between them (an adjoint store reads back its slot).
        let (lo, hi) = (a.min(b) as usize, a.max(b) as usize);
        let c = if arg == 3 { 0 } else { 1 };
        if insts[lo + 1..hi].iter().any(|i| match *i {
            Inst::Load { arg, .. } | Inst::Store { arg, .. } => mem_class(arg) == Some(c),
            _ => false,
        }) {
            continue;
        }
        let p = members.len() as u32;
        of[a as usize] = Some((p, 0));
        of[b as usize] = Some((p, 1));
        members.push([a, b]);
    }
    if members.is_empty() {
        return None;
    }
    let order = schedule(insts, &of, &members, rev, prefix)?;
    Some(Packs { of, members, order })
}

/// A topological order of the nodes, each as early as its operands allow and in program order
/// otherwise; `None` on a cycle (a pack that would have to come before itself).
fn schedule(
    insts: &[Inst],
    of: &[Option<(u32, u8)>],
    members: &[[V; 2]],
    rev: bool,
    prefix: usize,
) -> Option<Vec<Node>> {
    let n = insts.len();
    let np = members.len();
    // Node ids: instructions `0..n` (a packed one stands for its pack), packs `n..n + np`.
    let node = |i: usize| match of[i] {
        Some((p, _)) => n + p as usize,
        None => i,
    };
    let total = n + np;
    // A node's anchor: where it was (a pack: its later member), its rank among the ready.
    let mut anchor = vec![0usize; total];
    let mut live = vec![false; total];
    for i in 0..n {
        if of[i].is_none() {
            live[i] = true;
            anchor[i] = i;
        }
    }
    for (p, m) in members.iter().enumerate() {
        live[n + p] = true;
        anchor[n + p] = m[0].max(m[1]) as usize;
    }
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); total];
    let mut last_store: [Option<usize>; 2] = [None, None];
    let mut loads_since: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for (at, i) in insts.iter().enumerate() {
        let d = node(at);
        let (ops, k) = operands(i);
        for &o in &ops[..k] {
            let s = node(o as usize);
            if s != d {
                preds[d].push(s);
            }
        }
        match *i {
            Inst::Load { arg, .. } => {
                if let Some(c) = mem_class(arg) {
                    if let Some(s) = last_store[c as usize] {
                        if s != d {
                            preds[d].push(s);
                        }
                    }
                    loads_since[c as usize].push(d);
                }
            }
            Inst::Store { arg, .. } => {
                if let Some(c) = mem_class(arg) {
                    let c = c as usize;
                    for &l in &loads_since[c] {
                        if l != d {
                            preds[d].push(l);
                        }
                    }
                    loads_since[c].clear();
                    if let Some(s) = last_store[c] {
                        if s != d {
                            preds[d].push(s);
                        }
                    }
                    last_store[c] = Some(d);
                }
            }
            _ => {}
        }
    }
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); total];
    let mut indeg = vec![0usize; total];
    for d in 0..total {
        if !live[d] {
            continue;
        }
        preds[d].sort_unstable();
        preds[d].dedup();
        for &s in &preds[d] {
            succs[s].push(d);
            indeg[d] += 1;
        }
    }
    let region = |d: usize| (rev && anchor[d] >= prefix) as usize;
    let mut order = Vec::with_capacity(total);
    let mut ready: [BinaryHeap<Reverse<(usize, usize)>>; 2] =
        [BinaryHeap::new(), BinaryHeap::new()];
    for d in 0..total {
        if live[d] && indeg[d] == 0 {
            ready[region(d)].push(Reverse((anchor[d], d)));
        }
    }
    for r in 0..2 {
        while let Some(Reverse((_, d))) = ready[r].pop() {
            order.push(if d >= n {
                Node::Pack((d - n) as u32)
            } else {
                Node::Scalar(d as V)
            });
            for &s in &succs[d] {
                indeg[s] -= 1;
                if indeg[s] == 0 {
                    let rs = region(s);
                    // A forward-prefix node cannot wait on a seed-loop one.
                    if rs < r {
                        return None;
                    }
                    ready[rs].push(Reverse((anchor[s], s)));
                }
            }
        }
    }
    let want = live.iter().filter(|&&x| x).count();
    (order.len() == want).then_some(order)
}
