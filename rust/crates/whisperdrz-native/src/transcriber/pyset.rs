//! CPython 3.12's `set` hash table for small non-negative ints, so iteration order
//! matches the reference
//!
//! `_run_batched_inference` keeps its free slots in a `set` and refills from
//! `list(empty_slots)[:refill_count]`, so which slots get the next chunks follows the
//! set's table layout. That layout depends on the table size and the whole
//! insert/discard history (dummies, the last-dummy reuse rule, resizes), not on
//! the values alone. This is a transcription of `Objects/setobject.c` (3.12):
//! `set_add_entry`, `set_discard_entry`, `set_table_resize` and `set_insert_clean`.
//! An int's hash is the int itself for the values used here

/// `LINEAR_PROBES`
const LINEAR_PROBES: usize = 9;
/// `PERTURB_SHIFT`
const PERTURB_SHIFT: u32 = 5;
/// `PySet_MINSIZE`
const MIN_SIZE: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Entry {
    Unused,
    Dummy,
    Active(usize),
}

/// A CPython `set` of small non-negative ints
#[derive(Debug, Clone)]
pub struct PySet {
    table: Vec<Entry>,
    /// Active plus dummy entries (`so->fill`)
    fill: usize,
    /// Active entries (`so->used`)
    used: usize,
}

impl Default for PySet {
    fn default() -> Self {
        Self::new()
    }
}

impl PySet {
    /// `set()`
    pub fn new() -> Self {
        Self {
            table: vec![Entry::Unused; MIN_SIZE],
            fill: 0,
            used: 0,
        }
    }

    /// A set built by adding `items` in order, as a set comprehension does
    pub fn from_items(items: impl IntoIterator<Item = usize>) -> Self {
        let mut set = Self::new();
        for i in items {
            set.add(i);
        }

        set
    }

    fn mask(&self) -> usize {
        self.table.len() - 1
    }

    pub fn len(&self) -> usize {
        self.used
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    pub fn contains(&self, key: usize) -> bool {
        matches!(self.table[self.lookup(key)], Entry::Active(k) if k == key)
    }

    /// `set_lookkey`: the slot holding `key`, or the unused slot that ends its probe
    /// sequence
    fn lookup(&self, key: usize) -> usize {
        let mask = self.mask();
        let mut perturb = key;
        let mut i = key & mask;
        loop {
            let probes = if i + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            for j in i..=i + probes {
                match self.table[j] {
                    Entry::Unused => return j,
                    Entry::Active(k) if k == key => return j,
                    _ => {}
                }
            }

            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }

    /// `set.add`: probes past dummies to an unused slot, then fills the last dummy
    /// seen on the way if there was one (`freeslot`); otherwise takes the unused slot
    /// and resizes once `fill * 5 >= mask * 3`
    pub fn add(&mut self, key: usize) {
        let mask = self.mask();
        let mut perturb = key;
        let mut i = key & mask;
        let mut freeslot = None;
        let unused = 'probe: loop {
            let probes = if i + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            for j in i..=i + probes {
                match self.table[j] {
                    Entry::Unused => break 'probe j,
                    Entry::Active(k) if k == key => return,
                    Entry::Active(_) => {}
                    Entry::Dummy => freeslot = Some(j),
                }
            }

            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        };

        self.used += 1;
        if let Some(j) = freeslot {
            self.table[j] = Entry::Active(key);
            return;
        }

        self.fill += 1;
        self.table[unused] = Entry::Active(key);
        if self.fill * 5 >= mask * 3 {
            let minused = if self.used > 50_000 {
                self.used * 2
            } else {
                self.used * 4
            };
            self.resize(minused);
        }
    }

    /// `set.discard`: the entry becomes a dummy, `fill` is unchanged
    pub fn discard(&mut self, key: usize) {
        let j = self.lookup(key);
        if let Entry::Active(_) = self.table[j] {
            self.table[j] = Entry::Dummy;
            self.used -= 1;
        }
    }

    /// `set_table_resize`: the smallest power of two above `minused` (at least 8),
    /// active entries reinserted in old-table order, dummies dropped
    fn resize(&mut self, minused: usize) {
        let mut size = MIN_SIZE;
        while size <= minused {
            size <<= 1;
        }

        // a small table with no dummies is left alone
        if size == MIN_SIZE && self.table.len() == MIN_SIZE && self.fill == self.used {
            return;
        }

        let old = std::mem::replace(&mut self.table, vec![Entry::Unused; size]);
        self.fill = self.used;
        let mask = size - 1;
        for e in old {
            if let Entry::Active(key) = e {
                self.insert_clean(mask, key);
            }
        }
    }

    /// `set_insert_clean`: the first unused slot on the probe sequence
    fn insert_clean(&mut self, mask: usize, key: usize) {
        let mut perturb = key;
        let mut i = key & mask;
        loop {
            if self.table[i] == Entry::Unused {
                self.table[i] = Entry::Active(key);
                return;
            }

            if i + LINEAR_PROBES <= mask {
                for j in i + 1..=i + LINEAR_PROBES {
                    if self.table[j] == Entry::Unused {
                        self.table[j] = Entry::Active(key);
                        return;
                    }
                }
            }

            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }

    /// Iteration order (`list(s)`): table order, skipping unused and dummy slots
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.table.iter().filter_map(|e| match e {
            Entry::Active(k) => Some(*k),
            _ => None,
        })
    }

    /// `list(s)`
    pub fn to_vec(&self) -> Vec<usize> {
        self.iter().collect()
    }
}
