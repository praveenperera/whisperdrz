//! Fixed-length token bitsets, the CPU form of the `[V]` bool masks in `DecoderState`

use crate::text::Token;

/// One bit per vocab entry; a set bit means the token is blocked (or, for the
/// `*_enabled` and `*_mask` tables named so in python, selected)
#[derive(Clone, PartialEq, Eq)]
pub struct TokenMask {
    words: Vec<u64>,
    len: usize,
}

impl std::fmt::Debug for TokenMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TokenMask(len={}, set={})", self.len, self.count())
    }
}

impl TokenMask {
    /// All clear
    pub fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    /// All set
    pub fn full(len: usize) -> Self {
        let mut m = Self::new(len);
        m.fill(true);
        m
    }

    /// The ids below `len` set, as python's `mask[t[t < V]] = True`
    pub fn from_ids(len: usize, ids: impl IntoIterator<Item = Token>) -> Self {
        let mut m = Self::new(len);
        for id in ids {
            if (id as usize) < len {
                m.set(id);
            }
        }

        m
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, id: Token) -> bool {
        let i = id as usize;
        i < self.len && self.words[i / 64] >> (i % 64) & 1 == 1
    }

    /// Sets one bit; ids past the end are ignored, as in python's `t < V` filters
    pub fn set(&mut self, id: Token) {
        let i = id as usize;
        if i < self.len {
            self.words[i / 64] |= 1 << (i % 64);
        }
    }

    pub fn clear(&mut self, id: Token) {
        let i = id as usize;
        if i < self.len {
            self.words[i / 64] &= !(1 << (i % 64));
        }
    }

    /// `mask[:] = value`
    pub fn fill(&mut self, value: bool) {
        self.words.fill(if value { u64::MAX } else { 0 });
        self.trim();
    }

    /// `mask |= other`
    pub fn or(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= b;
        }
    }

    /// `mask &= ~other`
    pub fn and_not(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= !b;
        }
    }

    /// `mask |= ~other`
    pub fn or_not(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= !b;
        }

        self.trim();
    }

    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Set ids in ascending order
    pub fn ones(&self) -> impl Iterator<Item = Token> + '_ {
        self.words.iter().enumerate().flat_map(|(wi, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    return None;
                }

                let bit = w.trailing_zeros();
                w &= w - 1;
                Some((wi * 64) as Token + bit)
            })
        })
    }

    fn trim(&mut self) {
        let rem = self.len % 64;
        if rem != 0
            && let Some(last) = self.words.last_mut()
        {
            *last &= (1u64 << rem) - 1;
        }
    }
}
