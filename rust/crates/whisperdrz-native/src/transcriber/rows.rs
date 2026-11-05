//! The second half of `decode_batch` over a step's logits as they land: each row's
//! argmax and corrections ([`Decoder::finish`]) after [`Decoder::prepare`] planned it
//!
//! The calling thread finishes rows in order, waiting only for the logits chunk a row
//! needs, so it starts on the first rows while the rest still copy. A large batch also
//! gets rayon helpers, spawned once the first chunk lands: a parked worker takes about
//! 0.1 ms to wake, longer than a small batch's whole remaining work. The calling thread
//! publishes rows to the helpers as they land and spins, rather than sleeps, until the
//! last row is done, so no wake sits at the end of the step either

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use eyre::{Result, eyre};
use half::f16;

use super::model::StepLogits;
use crate::decode::{Decision, Decoder, Prepared, SlotState};

/// Rows that read logits, below which the calling thread finishes a step alone
const HELPER_MIN_ROWS: usize = 16;
/// Rows that read logits per rayon helper
const ROWS_PER_HELPER: usize = 6;

/// A row waiting for [`Decoder::finish`]
pub(crate) struct Pending<'s> {
    pub state: &'s mut SlotState,
    pub prepared: Prepared,
}

/// Rayon helpers for `rows` rows that read logits
fn helpers_for(rows: usize) -> usize {
    if rows < HELPER_MIN_ROWS {
        return 0;
    }

    (rows / ROWS_PER_HELPER).min(rayon::current_num_threads().saturating_sub(1))
}

/// [`Decoder::finish`] on every row, row `i` reading row `i` of `logits`; returns the
/// decisions in row order. Each row's result depends only on its own logits and state,
/// so which thread finishes it changes nothing
pub(crate) fn finish(
    decoder: &Decoder,
    rows: Vec<Pending<'_>>,
    logits: &dyn StepLogits,
) -> Result<Vec<Decision>> {
    let wanted = rows
        .iter()
        .filter(|r| Decoder::needs_logits(&r.prepared))
        .count();
    finish_with(decoder, rows, logits, helpers_for(wanted))
}

/// [`finish`] with `helpers` rayon helpers, none meaning the calling thread alone
fn finish_with(
    decoder: &Decoder,
    rows: Vec<Pending<'_>>,
    logits: &dyn StepLogits,
    helpers: usize,
) -> Result<Vec<Decision>> {
    let n = rows.len();
    eyre::ensure!(
        logits.rows() == n,
        "{} logits rows for {n} rows",
        logits.rows()
    );
    let needs: Vec<bool> = rows
        .iter()
        .map(|r| Decoder::needs_logits(&r.prepared))
        .collect();
    if helpers == 0 || n == 0 {
        return rows
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                let row = if needs[i] { Some(logits.row(i)?) } else { None };
                decoder.finish(r.prepared, row, r.state)
            })
            .collect();
    }

    let shared = Shared {
        decoder,
        rows: rows.into_iter().map(|r| Mutex::new(Some(r))).collect(),
        needs,
        landed: (0..n).map(|_| OnceLock::new()).collect(),
        results: (0..n).map(|_| OnceLock::new()).collect(),
        next: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
        abort: AtomicBool::new(false),
    };
    let mut publisher = Publisher {
        shared: &shared,
        logits,
        published: 0,
        error: None,
    };
    rayon::in_place_scope(|scope| {
        // a panic on this thread must still stop helpers waiting for unpublished rows,
        // or the scope waits for them forever
        let _abort = AbortOnExit(&shared.abort);
        let mut spawned = false;
        while let Some(i) = shared.take() {
            if shared.needs[i] && !publisher.publish_through(i) {
                shared.fail(i);
                break;
            }

            if !spawned && publisher.published > 0 {
                // the first rows have landed, so the helpers' wake overlaps the rest
                for _ in 0..helpers {
                    scope.spawn(|_| shared.work());
                }

                spawned = true;
            }

            shared.finish_row(i);
            publisher.publish_landed();
        }

        // helpers may still wait for rows the queue handed them
        if publisher.error.is_none() {
            publisher.publish_through(n - 1);
        }

        if publisher.error.is_some() {
            shared.abort.store(true, Ordering::Release);
        }

        while shared.done.load(Ordering::Acquire) < shared.taken() {
            std::hint::spin_loop();
        }
    });

    if let Some(e) = publisher.error {
        return Err(e);
    }

    shared
        .results
        .into_iter()
        .enumerate()
        .map(|(i, r)| {
            r.into_inner()
                .unwrap_or_else(|| Err(eyre!("row {i} never finished")))
        })
        .collect()
}

/// The rows and their progress, shared by the calling thread and the helpers
struct Shared<'d, 's, 'l> {
    decoder: &'d Decoder<'d>,
    rows: Vec<Mutex<Option<Pending<'s>>>>,
    needs: Vec<bool>,
    /// Row `i`'s logits once the calling thread has seen them land
    landed: Vec<OnceLock<&'l [f16]>>,
    results: Vec<OnceLock<Result<Decision>>>,
    /// The next row to hand out
    next: AtomicUsize,
    /// Rows finished, with or without an error
    done: AtomicUsize,
    /// The logits failed to arrive: helpers stop waiting for them
    abort: AtomicBool,
}

impl Shared<'_, '_, '_> {
    fn take(&self) -> Option<usize> {
        let i = self.next.fetch_add(1, Ordering::Relaxed);
        (i < self.rows.len()).then_some(i)
    }

    /// Rows handed out so far
    fn taken(&self) -> usize {
        self.next.load(Ordering::Relaxed).min(self.rows.len())
    }

    /// A helper: takes rows until none are left, waiting for each one's logits
    fn work(&self) {
        while let Some(i) = self.take() {
            if self.needs[i] {
                while self.landed[i].get().is_none() {
                    if self.abort.load(Ordering::Acquire) {
                        self.fail(i);
                        return;
                    }

                    std::hint::spin_loop();
                }
            }

            self.finish_row(i);
        }
    }

    /// Finishes row `i`, whose logits (if it reads them) have been published
    fn finish_row(&self, i: usize) {
        // counted even if the row panics, so the calling thread's spin still ends and
        // the scope can rethrow the panic
        let _done = Done(&self.done);
        let taken = self.rows[i]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let result = match taken {
            Some(r) => {
                let row = self.landed[i].get().copied();
                self.decoder.finish(r.prepared, row, r.state)
            }
            None => Err(eyre!("row {i} handed out twice")),
        };
        let _ = self.results[i].set(result);
    }

    /// Row `i` ends without its logits
    fn fail(&self, i: usize) {
        let _ = self.results[i].set(Err(eyre!("step logits never arrived")));
        self.done.fetch_add(1, Ordering::Release);
    }
}

/// Tells the helpers to stop waiting for logits when dropped
struct AbortOnExit<'a>(&'a AtomicBool);

impl Drop for AbortOnExit<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Counts one finished row when dropped
struct Done<'a>(&'a AtomicUsize);

impl Drop for Done<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Release);
    }
}

/// The calling thread's view of the logits: publishes rows to the helpers as they land
struct Publisher<'a, 's, 'l> {
    shared: &'a Shared<'a, 's, 'l>,
    logits: &'l dyn StepLogits,
    /// Rows published so far, a prefix
    published: usize,
    error: Option<eyre::Report>,
}

impl Publisher<'_, '_, '_> {
    /// Publishes every row up to and including `i`, waiting for them; false on error
    fn publish_through(&mut self, i: usize) -> bool {
        while self.published <= i {
            match self.logits.row(self.published) {
                Ok(row) => {
                    let _ = self.shared.landed[self.published].set(row);
                    self.published += 1;
                }
                Err(e) => {
                    self.error = Some(e);
                    return false;
                }
            }
        }

        true
    }

    /// Publishes the rows that have landed without waiting
    fn publish_landed(&mut self) {
        match self.logits.landed() {
            Ok(upto) if upto > self.published => {
                self.publish_through(upto - 1);
            }
            Ok(_) => {}
            Err(e) => self.error = Some(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use eyre::Result;
    use half::f16;

    use super::{Pending, finish_with};
    use crate::decode::{DecodeConfig, Decoder, DecoderTables, SlotState, Status};
    use crate::text::Tokenizer;
    use crate::transcriber::{Slot, StepLogits};

    /// Logits that land one row per `landed` query, as a chunked download would
    struct Trickle {
        data: Vec<f16>,
        vocab: usize,
        rows: usize,
        landed: Cell<usize>,
    }

    impl StepLogits for Trickle {
        fn rows(&self) -> usize {
            self.rows
        }

        fn row(&self, r: usize) -> Result<&[f16]> {
            eyre::ensure!(r < self.rows, "row {r}");
            self.landed.set(self.landed.get().max(r + 1));
            Ok(&self.data[r * self.vocab..(r + 1) * self.vocab])
        }

        fn landed(&self) -> Result<usize> {
            let n = (self.landed.get() + 1).min(self.rows);
            self.landed.set(n);
            Ok(n)
        }

        fn complete(&self) -> Result<()> {
            Ok(())
        }
    }

    /// [`Trickle`] whose row `panic_at` panics when read, like a model whose logits
    /// are shorter than its row count
    struct PanicAt(Trickle, usize);

    impl StepLogits for PanicAt {
        fn rows(&self) -> usize {
            self.0.rows()
        }

        fn row(&self, r: usize) -> Result<&[f16]> {
            assert_ne!(r, self.1, "row {r} has no logits");
            self.0.row(r)
        }

        fn landed(&self) -> Result<usize> {
            self.0.landed()
        }

        fn complete(&self) -> Result<()> {
            self.0.complete()
        }
    }

    fn noise(n: usize, vocab: usize) -> Vec<f16> {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        (0..n * vocab)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                f16::from_f32((seed % 4000) as f32 / 100.0 - 20.0)
            })
            .collect()
    }

    /// States whose plans mix forced tokens (not started, the transcribe token) with
    /// sampled language rows
    fn states(n: usize) -> Vec<SlotState> {
        (0..n)
            .map(|i| {
                let mut s = Slot::new(i, "").state;
                s.status = if i % 7 == 3 {
                    Status::Ready
                } else {
                    Status::Started
                };
                s.token_count = if i % 5 == 4 { 2 } else { 1 };
                s.last_token_id = 50258;
                s
            })
            .collect()
    }

    #[test]
    fn helpers_decide_as_the_calling_thread_alone() -> Result<()> {
        let tables = DecoderTables::new(Tokenizer::multilingual()?)?;
        let decoder = Decoder::new(&tables, DecodeConfig::default())?;
        let (n, vocab) = (40, tables.n_vocab());
        let data = noise(n, vocab);

        let mut runs = Vec::new();
        for helpers in [0, 4, 9] {
            let mut states = states(n);
            let rows = states
                .iter_mut()
                .map(|state| {
                    let prepared = decoder.prepare(state, None)?;
                    Ok(Pending { state, prepared })
                })
                .collect::<Result<Vec<_>>>()?;
            let logits = Trickle {
                data: data.clone(),
                vocab,
                rows: n,
                landed: Cell::new(0),
            };
            let decisions = finish_with(&decoder, rows, &logits, helpers)?;
            assert_eq!(decisions.len(), n);
            runs.push((decisions, states));
        }

        let (want, want_states) = &runs[0];
        // the language rows sampled, the rest were forced
        assert!(want.iter().any(|d| !d.samples.is_empty()));
        assert!(want.iter().any(|d| d.samples.is_empty()));
        for (got, got_states) in &runs[1..] {
            assert_eq!(got, want);
            assert_eq!(got_states, want_states);
        }

        Ok(())
    }

    /// A panic while the calling thread publishes rows must reach the caller, not leave
    /// the helpers waiting for rows that will never land
    #[test]
    fn publisher_panic_stops_the_helpers() -> Result<()> {
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let run = || -> Result<()> {
                let tables = DecoderTables::new(Tokenizer::multilingual()?)?;
                let decoder = Decoder::new(&tables, DecodeConfig::default())?;
                let (n, vocab) = (40, tables.n_vocab());
                let mut states = states(n);
                let rows = states
                    .iter_mut()
                    .map(|state| {
                        let prepared = decoder.prepare(state, None)?;
                        Ok(Pending { state, prepared })
                    })
                    .collect::<Result<Vec<_>>>()?;
                // the second row that reads logits, so the helpers have started
                let panic_at = rows
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| Decoder::needs_logits(&r.prepared))
                    .nth(1)
                    .map(|(i, _)| i)
                    .ok_or_else(|| eyre::eyre!("fewer than two sampled rows"))?;
                let logits = PanicAt(
                    Trickle {
                        data: noise(n, vocab),
                        vocab,
                        rows: n,
                        landed: Cell::new(0),
                    },
                    panic_at,
                );
                let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    finish_with(&decoder, rows, &logits, 4)
                }));
                eyre::ensure!(caught.is_err(), "the row panic was swallowed");
                Ok(())
            };
            let _ = done_tx.send(run());
        });

        done_rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .map_err(|_| eyre::eyre!("finish hung after a publisher panic"))?
    }
}
