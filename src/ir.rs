// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The semantic kernel IR: one SSA program, machine programs out.
//!
//! What the caller writes is what the kernel *means*: L1 buffers, the manual
//! DRAM <-> L1 copies, the circular-buffer sync instructions, and SSA values
//! produced by unpack / SFPU / matrix ops and consumed by pack or another
//! compute op. Nothing here names a lane register, a `Dst` slot, a `SrcA`/`SrcB`
//! bank, a Tensix word or a core: [`crate::alloc`] decides those, and
//! [`crate::emit`] turns the result into the role programs.
//!
//! Both forms are canonical: the text form ([`Kernel::parse`],
//! [`Kernel::to_text`]) and the byte form ([`Kernel::from_bytes`],
//! [`Kernel::to_bytes`]). They are checked against each other by a round-trip
//! test, as `llvm-as`/`llvm-dis` are.

use core::fmt;

/// Identifier of an SSA value.
pub type ValueId = u32;

/// Identifier of an L1 buffer.
pub type BufId = u32;

/// Element type of a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
}

impl DType {
    /// Bytes of one datum.
    pub const fn bytes(self) -> u32 {
        match self {
            DType::F32 => 4,
        }
    }

    fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
        }
    }
}

/// A kernel argument: read-only or written by the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    In,
    Out,
}

impl ParamKind {
    fn name(self) -> &'static str {
        match self {
            ParamKind::In => "in",
            ParamKind::Out => "out",
        }
    }
}

/// One kernel argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Param {
    pub kind: ParamKind,
    pub dtype: DType,
    /// Bytes the argument occupies.
    pub bytes: u32,
}

/// How an L1 buffer is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufKind {
    /// A plain span of L1: written by a copy or a pack, read by an unpack.
    Plain,
    /// A circular buffer: `pages` pages of `bytes / pages` bytes, driven by the
    /// explicit reserve / push / wait / pop instructions.
    Circular { pages: u8 },
}

/// One declared L1 buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Buf {
    pub bytes: u32,
    pub kind: BufKind,
}

/// The elementwise operations the IR names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfpuKind {
    Neg,
    Abs,
    Relu,
}

impl SfpuKind {
    fn name(self) -> &'static str {
        match self {
            SfpuKind::Neg => "neg",
            SfpuKind::Abs => "abs",
            SfpuKind::Relu => "relu",
        }
    }
}

/// One instruction of the semantic program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// Move `bytes` from `param` at byte `offset` into L1 `buf`. Runs on the
    /// reader core; L1 is the caller's choice.
    CopyIn { param: u32, offset: u32, buf: BufId, bytes: u32 },
    /// Move L1 `buf` into `param` at byte `offset`. Runs on the writer core.
    CopyOut { buf: BufId, param: u32, offset: u32, bytes: u32 },
    /// `reserve_back`: open `pages` back pages of a circular buffer.
    Reserve { buf: BufId, pages: u8 },
    /// `push_back`: publish `pages` written pages.
    Push { buf: BufId, pages: u8 },
    /// `wait_front`: block until `pages` front pages hold data.
    Wait { buf: BufId, pages: u8 },
    /// `pop_front`: release `pages` consumed pages.
    Pop { buf: BufId, pages: u8 },
    /// L1 `buf` becomes an SSA value: the unpacker moves the tile into `Dst`
    /// or a `Src` bank, chosen by [`crate::alloc`] from the value's uses.
    Unpack { value: ValueId, buf: BufId },
    /// Elementwise SFPU: `b` is `None` for a unary op. Result is an SSA value.
    Sfpu { value: ValueId, kind: SfpuKind, a: ValueId, b: Option<ValueId> },
    /// Matrix multiply: `a x b` accumulates into a `Dst` value.
    Matmul { value: ValueId, a: ValueId, b: ValueId },
    /// The value's tile is packed into L1 `buf`.
    Pack { value: ValueId, buf: BufId },
}

impl Op {
    /// The value this op defines, if any.
    pub fn defined(&self) -> Option<ValueId> {
        match *self {
            Op::Unpack { value, .. } | Op::Sfpu { value, .. } | Op::Matmul { value, .. } => Some(value),
            _ => None,
        }
    }

    /// Values this op reads.
    pub fn used(&self) -> Vec<ValueId> {
        match *self {
            Op::Sfpu { a, b, .. } => b.map_or_else(|| vec![a], |b| vec![a, b]),
            Op::Matmul { a, b, .. } => vec![a, b],
            Op::Pack { value, .. } => vec![value],
            _ => Vec::new(),
        }
    }
}

/// A parsed, verified-by-construction kernel. Call [`Kernel::verify`] before
/// handing it to the allocator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    /// Launch grid: (columns, rows) of points, each a Tensix tile.
    pub grid: [u32; 2],
    pub params: Vec<Param>,
    pub bufs: Vec<Buf>,
    pub ops: Vec<Op>,
}

/// What can be wrong with a kernel. Every check reports the line (text form) or
/// the op index (byte form) that failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl Error {
    fn new(msg: impl Into<String>) -> Self {
        Error(msg.into())
    }
}

impl Kernel {
    /// Check the SSA form and every reference. A kernel that fails this must
    /// never reach the allocator.
    pub fn verify(&self) -> Result<(), Error> {
        if self.grid[0] == 0 || self.grid[1] == 0 {
            return Err(Error::new("grid must be at least 1x1"));
        }
        let mut defined: Vec<bool> = Vec::new();
        let mut next_value: ValueId = 0;
        for (at, op) in self.ops.iter().enumerate() {
            for v in op.used() {
                match defined.get(v as usize) {
                    Some(true) => {}
                    _ => {
                        return Err(Error::new(format!(
                            "op {at} uses value %{v}, which is not defined before it"
                        )));
                    }
                }
            }
            if let Some(v) = op.defined() {
                if v != next_value {
                    return Err(Error::new(format!(
                        "op {at} defines %{v} but the next SSA value is %{next_value}"
                    )));
                }
                defined.push(true);
                next_value += 1;
            }
            match *op {
                Op::CopyIn { param, offset, buf, bytes } => {
                    self.param_bytes(param, at)?;
                    self.buf_bytes(buf, at)?;
                    if bytes == 0 {
                        return Err(Error::new(format!("op {at}: copy length is zero")));
                    }
                    if u64::from(offset) + u64::from(bytes) > u64::from(self.params[param as usize].bytes) {
                        return Err(Error::new(format!("op {at}: copy runs past param {param}")));
                    }
                    if bytes > self.bufs[buf as usize].bytes {
                        return Err(Error::new(format!("op {at}: copy is larger than buffer {buf}")));
                    }
                }
                Op::CopyOut { buf, param, offset, bytes } => {
                    self.param_bytes(param, at)?;
                    self.buf_bytes(buf, at)?;
                    if bytes == 0 {
                        return Err(Error::new(format!("op {at}: copy length is zero")));
                    }
                    if u64::from(offset) + u64::from(bytes) > u64::from(self.params[param as usize].bytes) {
                        return Err(Error::new(format!("op {at}: copy runs past param {param}")));
                    }
                    if bytes > self.bufs[buf as usize].bytes {
                        return Err(Error::new(format!("op {at}: copy is larger than buffer {buf}")));
                    }
                }
                Op::Reserve { buf, pages } | Op::Push { buf, pages } | Op::Wait { buf, pages } | Op::Pop { buf, pages } => {
                    let b = self.bufs.get(buf as usize).ok_or_else(|| Error::new(format!("op {at}: no buffer {buf}")))?;
                    let BufKind::Circular { pages: capacity } = b.kind else {
                        return Err(Error::new(format!("op {at}: buffer {buf} is not a circular buffer")));
                    };
                    if pages == 0 {
                        return Err(Error::new(format!("op {at}: zero pages")));
                    }
                    if pages > capacity {
                        return Err(Error::new(format!(
                            "op {at}: {pages} pages of a {capacity}-page circular buffer"
                        )));
                    }
                }
                Op::Unpack { buf, .. } => {
                    self.buf_bytes(buf, at)?;
                }
                Op::Pack { buf, .. } => {
                    self.buf_bytes(buf, at)?;
                }
                Op::Sfpu { .. } | Op::Matmul { .. } => {}
            }
        }
        Ok(())
    }

    fn param_bytes(&self, param: u32, at: usize) -> Result<(), Error> {
        if self.params.get(param as usize).is_none() {
            return Err(Error::new(format!("op {at}: no param {param}")));
        }
        Ok(())
    }

    fn buf_bytes(&self, buf: BufId, at: usize) -> Result<(), Error> {
        if self.bufs.get(buf as usize).is_none() {
            return Err(Error::new(format!("op {at}: no buffer {buf}")));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Byte form
// ---------------------------------------------------------------------------

const MAGIC: [u8; 4] = *b"TTIR";

impl Kernel {
    /// The byte form: the same program as [`Kernel::to_text`], length-prefixed
    /// and little-endian. This is the form a driver ships; the text form is the
    /// form it reads.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        put_u32(&mut b, VERSION);
        put_u32(&mut b, self.grid[0]);
        put_u32(&mut b, self.grid[1]);
        put_u32(&mut b, self.params.len() as u32);
        for p in &self.params {
            b.push(match p.kind {
                ParamKind::In => 0,
                ParamKind::Out => 1,
            });
            b.push(match p.dtype {
                DType::F32 => 0,
            });
            put_u32(&mut b, p.bytes);
        }
        put_u32(&mut b, self.bufs.len() as u32);
        for buf in &self.bufs {
            put_u32(&mut b, buf.bytes);
            match buf.kind {
                BufKind::Plain => {
                    b.push(0);
                    b.push(0);
                }
                BufKind::Circular { pages } => {
                    b.push(1);
                    b.push(pages);
                }
            }
        }
        put_u32(&mut b, self.ops.len() as u32);
        for op in &self.ops {
            match *op {
                Op::CopyIn { param, offset, buf, bytes } => {
                    b.push(0);
                    put_u32(&mut b, param);
                    put_u32(&mut b, offset);
                    put_u32(&mut b, buf);
                    put_u32(&mut b, bytes);
                }
                Op::CopyOut { buf, param, offset, bytes } => {
                    b.push(1);
                    put_u32(&mut b, buf);
                    put_u32(&mut b, param);
                    put_u32(&mut b, offset);
                    put_u32(&mut b, bytes);
                }
                Op::Reserve { buf, pages } => {
                    b.push(2);
                    put_u32(&mut b, buf);
                    b.push(pages);
                }
                Op::Push { buf, pages } => {
                    b.push(3);
                    put_u32(&mut b, buf);
                    b.push(pages);
                }
                Op::Wait { buf, pages } => {
                    b.push(4);
                    put_u32(&mut b, buf);
                    b.push(pages);
                }
                Op::Pop { buf, pages } => {
                    b.push(5);
                    put_u32(&mut b, buf);
                    b.push(pages);
                }
                Op::Unpack { value, buf } => {
                    b.push(6);
                    put_u32(&mut b, value);
                    put_u32(&mut b, buf);
                }
                Op::Sfpu { value, kind, a, b: second } => {
                    b.push(7);
                    put_u32(&mut b, value);
                    b.push(match kind {
                        SfpuKind::Neg => 0,
                        SfpuKind::Abs => 1,
                        SfpuKind::Relu => 2,
                    });
                    put_u32(&mut b, a);
                    match second {
                        Some(v) => {
                            b.push(1);
                            put_u32(&mut b, v);
                        }
                        None => b.push(0),
                    }
                }
                Op::Matmul { value, a, b: second } => {
                    b.push(8);
                    put_u32(&mut b, value);
                    put_u32(&mut b, a);
                    put_u32(&mut b, second);
                }
                Op::Pack { value, buf } => {
                    b.push(9);
                    put_u32(&mut b, value);
                    put_u32(&mut b, buf);
                }
            }
        }
        b
    }

    /// Read the byte form. A wrong magic or an unknown version is refused, not
    /// guessed at.
    pub fn from_bytes(bytes: &[u8]) -> Result<Kernel, Error> {
        let mut r = Cursor { at: 0, bytes };
        if r.take(4)? != MAGIC {
            return Err(Error::new("not a TTIR byte program"));
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(Error::new(format!("byte program version {version} is not {VERSION}")));
        }
        let mut k = Kernel { grid: [r.u32()?, r.u32()?], params: Vec::new(), bufs: Vec::new(), ops: Vec::new() };
        for _ in 0..r.u32()? {
            let kind = match r.u8()? {
                0 => ParamKind::In,
                1 => ParamKind::Out,
                other => return Err(Error::new(format!("param kind {other}"))),
            };
            let dtype = match r.u8()? {
                0 => DType::F32,
                other => return Err(Error::new(format!("dtype {other}"))),
            };
            k.params.push(Param { kind, dtype, bytes: r.u32()? });
        }
        for _ in 0..r.u32()? {
            let bytes = r.u32()?;
            let tag = r.u8()?;
            let pages = r.u8()?;
            let kind = match tag {
                0 => BufKind::Plain,
                1 => BufKind::Circular { pages },
                other => return Err(Error::new(format!("buffer kind {other}"))),
            };
            k.bufs.push(Buf { bytes, kind });
        }
        for _ in 0..r.u32()? {
            let op = match r.u8()? {
                0 => Op::CopyIn { param: r.u32()?, offset: r.u32()?, buf: r.u32()?, bytes: r.u32()? },
                1 => Op::CopyOut { buf: r.u32()?, param: r.u32()?, offset: r.u32()?, bytes: r.u32()? },
                2 => Op::Reserve { buf: r.u32()?, pages: r.u8()? },
                3 => Op::Push { buf: r.u32()?, pages: r.u8()? },
                4 => Op::Wait { buf: r.u32()?, pages: r.u8()? },
                5 => Op::Pop { buf: r.u32()?, pages: r.u8()? },
                6 => Op::Unpack { value: r.u32()?, buf: r.u32()? },
                7 => {
                    let value = r.u32()?;
                    let kind = match r.u8()? {
                        0 => SfpuKind::Neg,
                        1 => SfpuKind::Abs,
                        2 => SfpuKind::Relu,
                        other => return Err(Error::new(format!("sfpu kind {other}"))),
                    };
                    let a = r.u32()?;
                    let b = match r.u8()? {
                        0 => None,
                        1 => Some(r.u32()?),
                        other => return Err(Error::new(format!("sfpu operand flag {other}"))),
                    };
                    Op::Sfpu { value, kind, a, b }
                }
                8 => Op::Matmul { value: r.u32()?, a: r.u32()?, b: r.u32()? },
                9 => Op::Pack { value: r.u32()?, buf: r.u32()? },
                other => return Err(Error::new(format!("op tag {other}"))),
            };
            k.ops.push(op);
        }
        if r.at != r.bytes.len() {
            return Err(Error::new(format!("{} trailing bytes", r.bytes.len() - r.at)));
        }
        k.verify()?;
        Ok(k)
    }
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

struct Cursor<'a> {
    at: usize,
    bytes: &'a [u8],
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], Error> {
        let end = self.at + n;
        if end > self.bytes.len() {
            return Err(Error::new("byte program ends mid-record"));
        }
        let s = &self.bytes[self.at..end];
        self.at = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}


// ---------------------------------------------------------------------------
// Text form
// ---------------------------------------------------------------------------

/// The IR's version, shared by both forms. A kernel whose version this build
/// does not know is refused, never read best-effort.
pub const VERSION: u32 = 1;

impl Kernel {
    /// Parse the text form. Line-oriented, `#` starts a comment; every value is
    /// `%N`, and a value-producing op is written `%N = ...`.
    pub fn parse(text: &str) -> Result<Kernel, Error> {
        let mut k = Kernel { grid: [1, 1], params: Vec::new(), bufs: Vec::new(), ops: Vec::new() };
        let mut version: Option<u32> = None;
        let mut next_value: ValueId = 0;
        for (line_no, raw) in text.lines().enumerate() {
            let at = line_no + 1;
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut it = line.split_whitespace().peekable();
            let head = it.next().unwrap();
            // `%N = <op> ...`
            if let Some(name) = head.strip_prefix('%') {
                let value: ValueId = name.parse().map_err(|_| Error::new(format!("line {at}: bad value `{head}`")))?;
                if it.next() != Some("=") {
                    return Err(Error::new(format!("line {at}: expected `=` after {head}")));
                }
                let op = it.next().ok_or_else(|| Error::new(format!("line {at}: missing op after `=`")))?;
                if value != next_value {
                    return Err(Error::new(format!("line {at}: defines %{value}, expected %{next_value}")));
                }
                let parsed = match op {
                    "unpack" => {
                        expect(&mut it, "buf", at)?;
                        Op::Unpack { value, buf: num(&mut it, at)? }
                    }
                    "sfpu" => {
                        let kind = match it.next().ok_or_else(|| Error::new(format!("line {at}: missing sfpu kind")))? {
                            "neg" => SfpuKind::Neg,
                            "abs" => SfpuKind::Abs,
                            "relu" => SfpuKind::Relu,
                            other => return Err(Error::new(format!("line {at}: unknown sfpu kind `{other}`"))),
                        };
                        let a = value_ref(it.next(), at)?;
                        let b = it.next().map(|s| value_ref(Some(s), at)).transpose()?;
                        Op::Sfpu { value, kind, a, b }
                    }
                    "matmul" => {
                        let a = value_ref(it.next(), at)?;
                        let b = value_ref(it.next(), at)?;
                        Op::Matmul { value, a, b }
                    }
                    other => return Err(Error::new(format!("line {at}: unknown value op `{other}`"))),
                };
                if it.next().is_some() {
                    return Err(Error::new(format!("line {at}: trailing input")));
                }
                k.ops.push(parsed);
                next_value += 1;
                continue;
            }
            match head {
                "version" => {
                    let v: u32 = num(&mut it, at)?;
                    if v != VERSION {
                        return Err(Error::new(format!("line {at}: IR version {v} is not {VERSION}")));
                    }
                    version = Some(v);
                }
                "grid" => {
                    k.grid = [num(&mut it, at)?, num(&mut it, at)?];
                }
                "param" => {
                    let kind = match it.next().ok_or_else(|| Error::new(format!("line {at}: missing kind")))? {
                        "in" => ParamKind::In,
                        "out" => ParamKind::Out,
                        other => return Err(Error::new(format!("line {at}: unknown param kind `{other}`"))),
                    };
                    let dtype = match it.next().ok_or_else(|| Error::new(format!("line {at}: missing dtype")))? {
                        "f32" => DType::F32,
                        other => return Err(Error::new(format!("line {at}: unknown dtype `{other}`"))),
                    };
                    let bytes: u32 = num(&mut it, at)?;
                    k.params.push(Param { kind, dtype, bytes });
                }
                "buf" => {
                    let id: u32 = num(&mut it, at)?;
                    if id as usize != k.bufs.len() {
                        return Err(Error::new(format!("line {at}: buffer {id} out of order")));
                    }
                    let bytes: u32 = num(&mut it, at)?;
                    let kind = match it.next().ok_or_else(|| Error::new(format!("line {at}: missing buffer kind")))? {
                        "plain" => BufKind::Plain,
                        "circ" => BufKind::Circular { pages: num(&mut it, at)? },
                        other => return Err(Error::new(format!("line {at}: unknown buffer kind `{other}`"))),
                    };
                    k.bufs.push(Buf { bytes, kind });
                }
                "copy_in" => {
                    expect(&mut it, "param", at)?;
                    let param: u32 = num(&mut it, at)?;
                    expect(&mut it, "offset", at)?;
                    let offset: u32 = num(&mut it, at)?;
                    expect(&mut it, "len", at)?;
                    let bytes: u32 = num(&mut it, at)?;
                    expect(&mut it, "buf", at)?;
                    let buf: BufId = num(&mut it, at)?;
                    k.ops.push(Op::CopyIn { param, offset, buf, bytes });
                }
                "copy_out" => {
                    expect(&mut it, "buf", at)?;
                    let buf: BufId = num(&mut it, at)?;
                    expect(&mut it, "param", at)?;
                    let param: u32 = num(&mut it, at)?;
                    expect(&mut it, "offset", at)?;
                    let offset: u32 = num(&mut it, at)?;
                    expect(&mut it, "len", at)?;
                    let bytes: u32 = num(&mut it, at)?;
                    k.ops.push(Op::CopyOut { buf, param, offset, bytes });
                }
                "pack" => {
                    let value = value_ref(it.next(), at)?;
                    expect(&mut it, "buf", at)?;
                    let buf: BufId = num(&mut it, at)?;
                    k.ops.push(Op::Pack { value, buf });
                }
                "reserve" | "push" | "wait" | "pop" => {
                    let buf: BufId = num(&mut it, at)?;
                    let pages: u8 = num(&mut it, at)?;
                    k.ops.push(match head {
                        "reserve" => Op::Reserve { buf, pages },
                        "push" => Op::Push { buf, pages },
                        "wait" => Op::Wait { buf, pages },
                        _ => Op::Pop { buf, pages },
                    });
                }
                other => return Err(Error::new(format!("line {at}: unknown directive `{other}`"))),
            }
            if it.next().is_some() {
                return Err(Error::new(format!("line {at}: trailing input")));
            }
        }
        if version.is_none() {
            return Err(Error::new("missing `version`"));
        }
        k.verify()?;
        Ok(k)
    }

    /// Print the text form. `parse(to_text(k)) == k` for every verified kernel.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("version {VERSION}\n"));
        s.push_str(&format!("grid {} {}\n", self.grid[0], self.grid[1]));
        for p in &self.params {
            s.push_str(&format!("param {} {} {}\n", p.kind.name(), p.dtype.name(), p.bytes));
        }
        for (id, b) in self.bufs.iter().enumerate() {
            match b.kind {
                BufKind::Plain => s.push_str(&format!("buf {id} {} plain\n", b.bytes)),
                BufKind::Circular { pages } => s.push_str(&format!("buf {id} {} circ {pages}\n", b.bytes)),
            }
        }
        for op in &self.ops {
            match *op {
                Op::CopyIn { param, offset, buf, bytes } => {
                    s.push_str(&format!("copy_in param {param} offset {offset} len {bytes} buf {buf}\n"));
                }
                Op::CopyOut { buf, param, offset, bytes } => {
                    s.push_str(&format!("copy_out buf {buf} param {param} offset {offset} len {bytes}\n"));
                }
                Op::Reserve { buf, pages } => s.push_str(&format!("reserve {buf} {pages}\n")),
                Op::Push { buf, pages } => s.push_str(&format!("push {buf} {pages}\n")),
                Op::Wait { buf, pages } => s.push_str(&format!("wait {buf} {pages}\n")),
                Op::Pop { buf, pages } => s.push_str(&format!("pop {buf} {pages}\n")),
                Op::Unpack { value, buf } => s.push_str(&format!("%{value} = unpack buf {buf}\n")),
                Op::Sfpu { value, kind, a, b } => match b {
                    Some(b) => s.push_str(&format!("%{value} = sfpu {} %{a} %{b}\n", kind.name())),
                    None => s.push_str(&format!("%{value} = sfpu {} %{a}\n", kind.name())),
                },
                Op::Matmul { value, a, b } => {
                    s.push_str(&format!("%{value} = matmul %{a} %{b}\n"));
                }
                Op::Pack { value, buf } => s.push_str(&format!("pack %{value} buf {buf}\n")),
            }
        }
        s
    }
}

fn num<'a, T: core::str::FromStr>(it: &mut impl Iterator<Item = &'a str>, at: usize) -> Result<T, Error> {
    let s = it.next().ok_or_else(|| Error::new(format!("line {at}: missing number")))?;
    s.parse().map_err(|_| Error::new(format!("line {at}: bad number `{s}`")))
}

fn expect<'a>(it: &mut impl Iterator<Item = &'a str>, word: &str, at: usize) -> Result<(), Error> {
    match it.next() {
        Some(got) if got == word => Ok(()),
        Some(got) => Err(Error::new(format!("line {at}: expected `{word}`, got `{got}`"))),
        None => Err(Error::new(format!("line {at}: expected `{word}`"))),
    }
}

fn value_ref(token: Option<&str>, at: usize) -> Result<ValueId, Error> {
    let token = token.ok_or_else(|| Error::new(format!("line {at}: missing value")))?;
    let digits = token
        .strip_prefix('%')
        .ok_or_else(|| Error::new(format!("line {at}: `{token}` is not a value")))?;
    digits.parse().map_err(|_| Error::new(format!("line {at}: bad value `{token}`")))
}
