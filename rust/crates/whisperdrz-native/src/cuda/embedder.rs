//! ECAPA2 on a helper thread, so speaker embeddings run alongside decoding
//!
//! The thread owns the only [`Ecapa2`], on a [`Gpu`] of its own: its own stream at
//! the lowest priority, memory pool, cuBLAS/cuBLASLt handles and workspaces, all on
//! the device's primary context. The decoder's stream runs at the highest priority, so
//! ECAPA2 kernels take SMs when the decoder leaves them idle, such as during the
//! scheduler's host phases between steps.
//!
//! Every embedding, early or on demand, runs the same kernels and GEMM algorithms on
//! that stream, and an embedding is a function of its audio alone, so a crop
//! embedded early is the one the remap would have computed. Results are cached by
//! `(start, len)` for the current file; [`Embedder::reset`] starts the next file and
//! makes the thread skip any crop of an earlier one still queued

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use eyre::{Result, WrapErr, ensure, eyre};

use super::ecapa2::{self, Ecapa2, Precision};
use super::{Blas, Gpu, Priority, nvtx};
use crate::speaker::EMBED_DIM;

/// A crop's place in the file's audio
type Key = (usize, usize);

/// What the thread hands back: the embedding, or the error as text
type Embedding = std::result::Result<[f32; EMBED_DIM], String>;

/// How often a waiting caller checks that the thread is still alive
const LIVENESS_POLL: Duration = Duration::from_millis(100);

enum Job {
    /// A crop of the file of `generation`, cached by `(start, len)`
    Crop {
        generation: u64,
        audio: Arc<[f32]>,
        start: usize,
        len: usize,
    },
    /// Audio outside any file, answered on `reply` and not cached
    Direct {
        audio: Vec<f32>,
        reply: SyncSender<Embedding>,
    },
}

/// What the thread and its callers share
struct Shared {
    /// Results of the current generation; the generation changes under this lock
    results: Mutex<HashMap<Key, Embedding>>,
    ready: Condvar,
    generation: AtomicU64,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, HashMap<Key, Embedding>> {
        // a panic on the thread cannot leave the map half written, so take it anyway
        self.results
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// The helper thread and the current file's crop cache
pub struct Embedder {
    jobs: Option<Sender<Job>>,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    /// Crops of the current file sent to the thread
    requested: HashSet<Key>,
}

impl Embedder {
    /// Starts the thread, which loads the exported ECAPA2 weights at `path` onto a
    /// background stream of GPU `ordinal`; returns once they are on the device
    pub fn spawn(ordinal: usize, path: &Path, precision: Precision) -> Result<Self> {
        let shared = Arc::new(Shared {
            results: Mutex::default(),
            ready: Condvar::new(),
            generation: AtomicU64::new(0),
        });
        let (jobs, rx) = mpsc::channel();
        let (loaded_tx, loaded) = mpsc::sync_channel(1);
        let path: PathBuf = path.to_owned();
        let thread_shared = shared.clone();
        let thread = std::thread::Builder::new()
            .name("ecapa2".to_owned())
            .spawn(move || {
                let worker = Worker::load(ordinal, &path, precision);
                let worker = match worker {
                    Ok(w) => {
                        let _ = loaded_tx.send(Ok(()));
                        w
                    }
                    Err(e) => {
                        let _ = loaded_tx.send(Err(e));
                        return;
                    }
                };
                worker.run(&rx, &thread_shared);
            })
            .wrap_err("start the ECAPA2 thread")?;
        loaded
            .recv()
            .map_err(|_| eyre!("the ECAPA2 thread exited while loading"))??;

        Ok(Self {
            jobs: Some(jobs),
            shared,
            thread: Some(thread),
            requested: HashSet::new(),
        })
    }

    fn send(&self, job: Job) -> Result<()> {
        self.jobs
            .as_ref()
            .ok_or_else(|| eyre!("the ECAPA2 thread is stopped"))?
            .send(job)
            .map_err(|_| eyre!("the ECAPA2 thread exited"))
    }

    fn alive(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }

    /// Starts a new file: drops the cached crops and makes the thread skip queued ones
    pub fn reset(&mut self) {
        let mut results = self.shared.lock();
        self.shared.generation.fetch_add(1, Ordering::SeqCst);
        results.clear();
        drop(results);
        self.requested.clear();
    }

    /// Queues `audio[start..start + len]` of the current file, once per crop; never
    /// waits for the GPU
    pub fn prefetch(&mut self, audio: &Arc<[f32]>, start: usize, len: usize) -> Result<()> {
        ensure!(
            start.checked_add(len).is_some_and(|end| end <= audio.len()),
            "crop {start}+{len} of {} samples",
            audio.len()
        );
        if !self.requested.insert((start, len)) {
            return Ok(());
        }

        self.send(Job::Crop {
            generation: self.shared.generation.load(Ordering::SeqCst),
            audio: audio.clone(),
            start,
            len,
        })
    }

    /// The embedding of `audio[start..start + len]` of the current file, queued first if
    /// no prefetch named it
    pub fn embed_crop(
        &mut self,
        audio: &Arc<[f32]>,
        start: usize,
        len: usize,
    ) -> Result<[f32; EMBED_DIM]> {
        self.prefetch(audio, start, len)?;
        let mut results = self.shared.lock();
        loop {
            if let Some(r) = results.get(&(start, len)) {
                return r.clone().map_err(|e| eyre!(e));
            }

            ensure!(self.alive(), "the ECAPA2 thread exited");
            results = self
                .shared
                .ready
                .wait_timeout(results, LIVENESS_POLL)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
    }

    /// The embedding of `audio`, outside the crop cache
    pub fn embed(&self, audio: &[f32]) -> Result<[f32; EMBED_DIM]> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.send(Job::Direct {
            audio: audio.to_vec(),
            reply,
        })?;
        answer
            .recv()
            .map_err(|_| eyre!("the ECAPA2 thread exited"))?
            .map_err(|e| eyre!(e))
    }
}

impl Drop for Embedder {
    fn drop(&mut self) {
        // closing the channel ends the thread's loop once its current crop is done
        self.shared.generation.fetch_add(1, Ordering::SeqCst);
        self.jobs = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The thread's GPU state
struct Worker {
    gpu: Gpu,
    blas: Blas,
    ecapa: Ecapa2,
}

impl Worker {
    fn load(ordinal: usize, path: &Path, precision: Precision) -> Result<Self> {
        let gpu = Gpu::with_priority(ordinal, Priority::Background)?;
        let blas = Blas::new(&gpu)?;
        let ecapa = Ecapa2::load(&gpu, path, precision)?;
        Ok(Self { gpu, blas, ecapa })
    }

    /// The L2-normalized embedding, written into a NaN-filled array so a short or
    /// missing result can't pass for a real one
    fn embed(&self, audio: &[f32]) -> Result<[f32; EMBED_DIM]> {
        let _range = nvtx::Range::new("ecapa2_bg");
        let e = self.ecapa.embed(&self.gpu, &self.blas, audio)?;
        let n = ecapa2::normalize(&e);
        ensure!(
            n.len() == EMBED_DIM,
            "ECAPA2 embedding of {} values, want {EMBED_DIM}",
            n.len()
        );
        let mut out = [f32::NAN; EMBED_DIM];
        out.copy_from_slice(&n);
        ensure!(
            out.iter().all(|v| v.is_finite()),
            "non-finite ECAPA2 embedding"
        );
        Ok(out)
    }

    fn run(self, jobs: &Receiver<Job>, shared: &Shared) {
        for job in jobs {
            match job {
                Job::Crop {
                    generation,
                    audio,
                    start,
                    len,
                } => {
                    if shared.generation.load(Ordering::SeqCst) != generation {
                        continue;
                    }

                    let r = self
                        .embed(&audio[start..start + len])
                        .map_err(|e| format!("{e:?}"));
                    let mut results = shared.lock();
                    // a reset since the crop started means it belongs to an earlier file
                    if shared.generation.load(Ordering::SeqCst) == generation {
                        results.insert((start, len), r);
                    }

                    drop(results);
                    shared.ready.notify_all();
                }
                Job::Direct { audio, reply } => {
                    let _ = reply.send(self.embed(&audio).map_err(|e| format!("{e:?}")));
                }
            }
        }
    }
}
