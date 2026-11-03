//! CPython's `random` module: MT19937 seeded through `init_by_array`, with
//! `random()`, `uniform()`, `getrandbits()` and `randint()` drawing words exactly as
//! `_randommodule.c` and `random.py` (3.12) do

use eyre::{Result, bail};

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// A `random.Random` instance
#[derive(Clone)]
pub struct PyRandom {
    mt: [u32; N],
    index: usize,
}

impl std::fmt::Debug for PyRandom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyRandom").field("index", &self.index).finish_non_exhaustive()
    }
}

impl PyRandom {
    /// `random.seed(n)` for a non-negative int: the seed's 32-bit words, least
    /// significant first, go through `init_by_array` (0 is the one-word key `[0]`)
    pub fn new(seed: u64) -> Self {
        let mut key = vec![seed as u32];
        if seed >> 32 != 0 {
            key.push((seed >> 32) as u32);
        }

        let mut r = Self { mt: [0; N], index: N + 1 };
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1_812_433_253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }

        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }

            if j >= key.len() {
                j = 0;
            }
        }

        for _ in 0..N - 1 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
        }

        self.mt[0] = 0x8000_0000;
    }

    /// `genrand_uint32`
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            for k in 0..N {
                let y = (self.mt[k] & UPPER_MASK) | (self.mt[(k + 1) % N] & LOWER_MASK);
                let mag = if y & 1 == 1 { MATRIX_A } else { 0 };
                self.mt[k] = self.mt[(k + M) % N] ^ (y >> 1) ^ mag;
            }

            self.index = 0;
        }

        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random()`: 53 random bits from two words
    pub fn random(&mut self) -> f64 {
        let a = self.next_u32() >> 5;
        let b = self.next_u32() >> 6;
        (a as f64 * 67_108_864.0 + b as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `uniform(a, b)`: `a + (b - a) * random()`
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }

    /// `getrandbits(k)` for `k <= 64`: whole words least significant first, the last
    /// one shifted down to the bits still needed
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        match k {
            0 => 0,
            1..=32 => (self.next_u32() >> (32 - k)) as u64,
            _ => {
                let lo = self.next_u32() as u64;
                let hi = (self.next_u32() >> (64 - k.min(64))) as u64;
                lo | (hi << 32)
            }
        }
    }

    /// `_randbelow_with_getrandbits(n)`: rejection sampling on `n.bit_length()` bits
    fn randbelow(&mut self, n: u64) -> u64 {
        let k = u64::BITS - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `randint(a, b)`: `randrange(a, b + 1)`, an int in `[a, b]`
    pub fn randint(&mut self, a: i64, b: i64) -> Result<i64> {
        let width = b as i128 + 1 - a as i128;
        if width <= 0 {
            bail!("empty range in randrange({a}, {})", b as i128 + 1);
        }

        Ok((a as i128 + self.randbelow(width as u64) as i128) as i64)
    }
}

/// The two draws the line crop makes, so a caller can observe them
pub trait CropRng {
    /// `random.uniform(a, b)`
    fn uniform(&mut self, a: f64, b: f64) -> f64;
    /// `random.randint(a, b)`
    fn randint(&mut self, a: i64, b: i64) -> Result<i64>;
}

impl CropRng for PyRandom {
    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        PyRandom::uniform(self, a, b)
    }

    fn randint(&mut self, a: i64, b: i64) -> Result<i64> {
        PyRandom::randint(self, a, b)
    }
}
