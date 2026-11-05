//! Self-speculative decoding: each decode step feeds every row its last token plus
//! drafted tokens and verifies them in one forward ([`Model::launch_verify`])
//!
//! Acceptance is exact greedy. A row's positions run through the constrained decoder in
//! order on the row's own state, exactly as the one-token path runs them a step at a
//! time: [`Decoder::prepare`] and [`Decoder::finish`] on the logits of that position,
//! `update_state_after_decode`, the token write, and the main loop's safety stops before
//! the next position. A draft is kept while it equals the token that processing emits,
//! not the raw argmax, since Phase 4 rewrites picks. The first mismatch, or any stop,
//! ends the row's chain; the token emitted there comes from the same logits row the
//! one-token step would have read. So every window decodes the tokens it would one
//! token a step, given logits that depend only on each position's token prefix
//!
//! Drafts need no new weights:
//! - tokens the FSM forces (`Plan::Fixed`), found by planning on a copy of the state;
//!   they are always accepted, and a row whose window ends inside its forced chain
//!   needs no forward at all
//! - with [`SpecConfig::carry`], the word Phase 4 replaced with a bare space so a
//!   timestamp could go first: after that timestamp, the word without its space
//! - with [`SpecConfig::ngram`], at other sampled positions, what followed the last
//!   [`NGRAM`] text tokens where they last occurred: earlier in the window, else in a
//!   window decoded before ([`NgramLog`])
//!
//! Drafts beyond a sampled position are guesses on a copy whose later state may differ
//! from the real one; they only change how many tokens a step keeps, never which

use std::collections::{HashMap, VecDeque};

use eyre::{Result, ensure, eyre};

use super::Transcriber;
use super::graphs::{VerifyRow, VerifyStep, get_bucket};
use super::model::{Model, StepLogits};
use super::schedule::{RowState, safety_stop};
use super::trace;
use crate::decode::{Decision, DecodeConfig, Decoder, Prepared, Stage, Status};
use crate::text::Token;

/// Most drafts a row feeds in one step
pub const MAX_DRAFTS: usize = 7;

/// The bare space Phase 4 puts in place of a word so a timestamp can precede it
const BARE_SPACE: Token = 220;

/// Text tokens an n-gram draft matches on
pub const NGRAM: usize = 3;

/// Speculative decoding settings ([`super::Config::spec`])
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecConfig {
    /// Most drafts a row feeds per step, at most [`MAX_DRAFTS`]
    pub k: usize,
    /// Steps over more rows than this run one token a row
    pub max_rows: usize,
    /// Draft the word Phase 4 deferred behind a forced timestamp
    pub carry: bool,
    /// Most n-gram drafts a row feeds per step, guesses of what followed the last text
    /// tokens where they occurred before; 0 turns the source off
    pub ngram: usize,
}

impl Default for SpecConfig {
    /// Forced tokens only, up to 4 a row, at up to 16 rows
    fn default() -> Self {
        Self {
            k: 4,
            max_rows: 16,
            carry: false,
            ngram: 0,
        }
    }
}

impl SpecConfig {
    /// The setting the CLI runs unless `WHISPERDRZ_SPEC` says otherwise: forced, carried
    /// and one n-gram draft a row, up to 4 a row, at up to 16 rows. Wider caps and longer
    /// drafts ran faster on vox24 but moved its speaker-turn (cp) or word edits past what
    /// the reference's own reruns show, so this is the fastest setting within that spread
    pub const SHIPPED: Self = Self {
        k: 4,
        max_rows: 16,
        carry: true,
        ngram: 1,
    };

    /// Greedy decoding only: verification compares drafts with the greedy pick, which
    /// sampling would have to replace with rejection sampling
    pub fn validate(&self, decode: &DecodeConfig) -> Result<()> {
        ensure!(
            decode.temperature == 0.0,
            "speculative decoding verifies greedy picks; temperature {} is not greedy",
            decode.temperature
        );
        ensure!(
            (1..=MAX_DRAFTS).contains(&self.k),
            "speculative k {} outside 1..={MAX_DRAFTS}",
            self.k
        );
        ensure!(self.max_rows >= 1, "speculative max_rows must be positive");
        Ok(())
    }

    /// Parses `k=4,rows=16,carry`, the format of the CLI's `WHISPERDRZ_SPEC`; unnamed
    /// settings keep [`SpecConfig::default`]
    pub fn parse(s: &str) -> Result<Self> {
        let mut out = Self::default();
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            match part.split_once('=') {
                Some(("k", v)) => out.k = v.parse()?,
                Some(("rows", v)) => out.max_rows = v.parse()?,
                None if part == "carry" => out.carry = true,
                Some(("ngram", v)) => out.ngram = v.parse()?,
                None if part == "ngram" => out.ngram = MAX_DRAFTS,
                None if part == "on" => {}
                _ => eyre::bail!("unknown speculative setting {part:?}"),
            }
        }

        Ok(out)
    }
}

/// Speculative decoding counters
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecStats {
    /// Steps that ran a verify forward
    pub verify_steps: usize,
    /// Steps with no drafts, run one token a row
    pub plain_steps: usize,
    /// Of those, steps over more rows than [`SpecConfig::max_rows`], and their rows
    pub wide_steps: usize,
    pub wide_rows: usize,
    /// Of those, steps whose drafts would overfill the model's verify workspace
    pub full_steps: usize,
    /// Steps whose rows all ended inside forced chains, run without the GPU
    pub host_steps: usize,
    /// Rows of verify steps that fed the GPU
    pub verify_rows: usize,
    /// Rows that ended inside a forced chain without a forward
    pub host_rows: usize,
    /// Tokens the verify rows emitted
    pub verify_emitted: usize,
    /// Drafts fed, and kept, by source
    pub forced_fed: usize,
    pub forced_kept: usize,
    pub carry_fed: usize,
    pub carry_kept: usize,
    pub ngram_fed: usize,
    pub ngram_kept: usize,
    /// Real tokens fed to verify forwards, padding excluded
    pub verify_tokens: usize,
}

impl SpecStats {
    /// One line for logs
    pub fn summary(&self) -> String {
        let per = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f64 / b as f64 };
        format!(
            "spec: {} verify steps ({} rows, {:.3} tokens a row, {:.3} fed a row), {} plain \
             ({} over the row cap, {} rows, {} over capacity), {} host-only steps ({} rows); forced {}/{} kept, \
             carry {}/{} kept, n-gram {}/{} kept",
            self.verify_steps,
            self.verify_rows,
            per(self.verify_emitted, self.verify_rows),
            per(self.verify_tokens, self.verify_rows),
            self.plain_steps,
            self.wide_steps,
            self.wide_rows,
            self.full_steps,
            self.host_steps,
            self.host_rows,
            self.forced_kept,
            self.forced_fed,
            self.carry_kept,
            self.carry_fed,
            self.ngram_kept,
            self.ngram_fed,
        )
    }
}

/// What drafting remembers about a slot between steps
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowMemory {
    /// The window position of a bare space Phase 4 put in place of the model's pick,
    /// and that pick
    carry: Option<(usize, Token)>,
    /// The window's text tokens as of the last draft, logged when the slot's next window
    /// starts
    seen: Vec<Token>,
}

/// Text tokens of finished windows, each followed by a separator, and where each
/// [`NGRAM`] of them last occurred
#[derive(Debug, Default)]
pub struct NgramLog {
    log: Vec<Token>,
    /// The position after the latest occurrence of each n-gram
    after: HashMap<[Token; NGRAM], usize>,
}

/// Ends each window in the log, so no continuation runs into the next window
const SEPARATOR: Token = Token::MAX;

impl NgramLog {
    /// Appends a finished window's text tokens
    pub fn push_window(&mut self, text: &[Token]) {
        if text.len() <= NGRAM {
            return;
        }

        let start = self.log.len();
        self.log.extend_from_slice(text);
        self.log.push(SEPARATOR);
        for i in start..start + text.len() - NGRAM + 1 {
            let mut key = [0; NGRAM];
            key.copy_from_slice(&self.log[i..i + NGRAM]);
            self.after.insert(key, i + NGRAM);
        }
    }

    /// What followed `key` at its latest occurrence, up to the end of that window
    fn continuation(&self, key: &[Token]) -> &[Token] {
        let Ok(key) = <[Token; NGRAM]>::try_from(key) else {
            return &[];
        };
        let Some(&at) = self.after.get(&key) else {
            return &[];
        };
        let end = self.log[at..]
            .iter()
            .position(|&t| t == SEPARATOR)
            .map_or(self.log.len(), |e| at + e);
        &self.log[at..end]
    }
}

/// What followed the last [`NGRAM`] tokens of `text` at their latest earlier occurrence
/// in `text`, up to its end
fn in_window_continuation(text: &[Token]) -> &[Token] {
    let n = text.len();
    if n <= NGRAM {
        return &[];
    }

    let key = &text[n - NGRAM..];
    (0..n - NGRAM)
        .rev()
        .find(|&i| &text[i..i + NGRAM] == key)
        .map_or(&[], |i| &text[i + NGRAM..])
}

/// Where a draft came from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Forced,
    Carry,
    Ngram,
}

/// One row's drafts for a step
#[derive(Debug, Clone, Default)]
struct Draft {
    tokens: Vec<Token>,
    sources: Vec<Source>,
    /// Every draft is forced and the window ends at the last one, so the row needs no
    /// logits and no forward: the window's KV is not read again
    host_only: bool,
}

impl Draft {
    /// Leading drafts the FSM forces on a state known for certain: the first draft is
    /// planned on the row's real state, and each later one on the state its forced
    /// predecessors give. The logits before them are never read
    fn certain_forced(&self) -> usize {
        self.sources
            .iter()
            .take_while(|&&s| s == Source::Forced)
            .count()
    }
}

impl<M: Model> Transcriber<M> {
    /// One speculative decode step over `decode_slots`: drafts, then a verify forward
    /// over the rows that need logits, then each row's chain in order
    pub(crate) fn spec_step(&mut self, decode_slots: &[usize], spec: SpecConfig) -> Result<()> {
        let tables = self.tables.clone();
        let decoder = Decoder::new(&tables, self.config.decode)?;
        let draft_range = trace::Range::new("draft");
        let drafts: Vec<Draft> = if decode_slots.len() <= spec.max_rows {
            if spec.ngram > 0 {
                for &s in decode_slots {
                    self.note_window_text(s);
                }
            }

            decode_slots
                .iter()
                .map(|&s| self.draft(&decoder, s, spec))
                .collect::<Result<_>>()?
        } else {
            self.spec_stats.wide_steps += 1;
            self.spec_stats.wide_rows += decode_slots.len();
            vec![Draft::default(); decode_slots.len()]
        };
        drop(draft_range);
        if drafts.iter().all(|d| d.tokens.is_empty()) {
            self.spec_stats.plain_steps += 1;
            return self.one_token_step(decode_slots);
        }

        // rows whose window ends inside a forced chain feed nothing
        let gpu: Vec<usize> = (0..decode_slots.len())
            .filter(|&i| !drafts[i].host_only)
            .collect();
        let fed: Vec<Vec<Token>> = gpu
            .iter()
            .map(|&i| {
                let s = decode_slots[i];
                let seq_len = self.seq_lens[s];
                ensure!(seq_len > 0, "slot {s} has no tokens");
                let mut f = vec![self.all_tokens[s][seq_len - 1]];
                f.extend_from_slice(&drafts[i].tokens);
                ensure!(
                    self.kv_lens[s] + f.len() <= self.config.max_tokens,
                    "slot {s} feeds {} tokens at KV length {}, past {}",
                    f.len(),
                    self.kv_lens[s],
                    self.config.max_tokens
                );
                Ok(f)
            })
            .collect::<Result<_>>()?;
        // valid settings can draft more than the model's verify workspace holds (52 rows
        // of 8 at k=7); such a step runs one token a row instead of failing. The LM head's
        // rows are a subset of the packed tokens, so their bucket is never larger
        let packed = fed.iter().map(Vec::len).sum::<usize>();
        if get_bucket(packed, &self.buckets) > self.model.verify_capacity() {
            self.spec_stats.plain_steps += 1;
            self.spec_stats.full_steps += 1;
            return self.one_token_step(decode_slots);
        }

        let host = trace::Range::new("host_decode");
        // a queued alignment's texts go in before any row's chain can clear that slot's
        // text (the non-English exit): the chains run while the logits hold the model, so
        // the join comes first, as if before this step's updates
        if decode_slots
            .iter()
            .any(|&s| self.align_pending_on(s) && self.may_stop_non_english(s))
        {
            self.resolve_align()?;
        }

        let mut prepared: Vec<Option<Prepared>> = Vec::new();
        let mut emitted = vec![(0, 0); decode_slots.len()];
        if gpu.is_empty() {
            self.spec_stats.host_steps += 1;
            let mut rs = RowState {
                slots: &mut self.slots,
                all_tokens: &mut self.all_tokens,
                seq_lens: &mut self.seq_lens,
                probe: &mut self.probe,
                config: &self.config,
            };
            for (i, &s) in decode_slots.iter().enumerate() {
                emitted[i] = run_chain(
                    &mut rs,
                    &mut self.spec_rows,
                    &decoder,
                    s,
                    None,
                    &drafts[i],
                    None,
                )?;
            }
        } else {
            let rows: Vec<VerifyRow<'_>> = gpu
                .iter()
                .zip(&fed)
                .map(|(&i, f)| {
                    let s = decode_slots[i];
                    VerifyRow {
                        slot: s,
                        kv_len: self.kv_lens[s],
                        fed: f,
                        forced: drafts[i].certain_forced(),
                    }
                })
                .collect();
            let trash = self.slots.len();
            let step = VerifyStep::new(&rows, &self.buckets, trash);
            self.spec_stats.verify_steps += 1;
            self.spec_stats.verify_rows += gpu.len();
            self.spec_stats.verify_tokens += step.real_tokens();
            self.kv_lens[trash] = 0;

            let all_tokens = &self.all_tokens;
            let slots = &mut self.slots;
            let logits = self.model.launch_verify(&step, &mut || {
                let _range = trace::Range::new("prepare");
                prepared = gpu
                    .iter()
                    .map(|&i| {
                        let s = decode_slots[i];
                        let state = &mut slots[s].state;
                        let window = Decoder::repetition_window(state.token_count)
                            .map(|range| &all_tokens[s][range]);
                        decoder.prepare(state, window).map(Some)
                    })
                    .collect::<Result<_>>()?;
                Ok(())
            })?;
            ensure!(
                logits.rows() == step.need.len(),
                "verify returned {} logits rows for {} needed",
                logits.rows(),
                step.need.len()
            );
            let mut rs = RowState {
                slots: &mut self.slots,
                all_tokens: &mut self.all_tokens,
                seq_lens: &mut self.seq_lens,
                probe: &mut self.probe,
                config: &self.config,
            };
            let memory = &mut self.spec_rows[..];
            let kv_lens = &mut self.kv_lens;
            let mut g = 0;
            for (i, &s) in decode_slots.iter().enumerate() {
                emitted[i] = if drafts[i].host_only {
                    run_chain(&mut rs, memory, &decoder, s, None, &drafts[i], None)?
                } else {
                    let first = prepared
                        .get_mut(g)
                        .and_then(Option::take)
                        .ok_or_else(|| eyre!("verify row {g} was never prepared"))?;
                    let n = run_chain(
                        &mut rs,
                        memory,
                        &decoder,
                        s,
                        Some(first),
                        &drafts[i],
                        Some((&*logits, &step, g)),
                    )?;
                    // the chain has committed its tokens, so the KV length moves with them:
                    // if a later row or the logits fail, a retry feeds each row's last
                    // token at its own position. The last token and each kept draft hold
                    // the window's tokens; the rest are past the length and get overwritten
                    kv_lens[s] += n.0;
                    g += 1;
                    n
                };
            }

            // the step's GPU errors surface before the scheduler moves on
            logits.complete()?;
            drop(logits);
        }

        drop(host);
        // the last alignment's texts go in once its thread is done, without a wait, as in
        // the one-token step
        self.resolve_align_if_done()?;
        for (d, &(emitted, kept)) in drafts.iter().zip(&emitted) {
            for (j, src) in d.sources.iter().enumerate() {
                let hit = j < kept;
                match src {
                    Source::Forced => {
                        self.spec_stats.forced_fed += 1;
                        self.spec_stats.forced_kept += usize::from(hit);
                    }
                    Source::Carry => {
                        self.spec_stats.carry_fed += 1;
                        self.spec_stats.carry_kept += usize::from(hit);
                    }
                    Source::Ngram => {
                        self.spec_stats.ngram_fed += 1;
                        self.spec_stats.ngram_kept += usize::from(hit);
                    }
                }
            }

            if d.host_only {
                self.spec_stats.host_rows += 1;
            } else {
                self.spec_stats.verify_emitted += emitted;
            }
        }

        Ok(())
    }

    /// Drafts for `slot_idx`: plans each position on a copy of its state, taking forced
    /// tokens and, at sampled positions, guesses from the enabled sources, until a
    /// position has no draft, the copy's window stops, or the budget runs out
    fn draft(&self, decoder: &Decoder, slot_idx: usize, spec: SpecConfig) -> Result<Draft> {
        let max_seq_len = self.config.max_tokens;
        let kv_len = self.kv_lens[slot_idx];
        // every fed position must fit the KV: the last token at kv_len, drafts after it
        let budget = spec.k.min(max_seq_len.saturating_sub(kv_len + 1));
        let mut out = Draft::default();
        if budget == 0 {
            return Ok(out);
        }

        let tables = decoder.tables();
        let mut st = self.slots[slot_idx].state.clone();
        let mut toks = self.all_tokens[slot_idx].clone();
        let mut certain = true;
        let mut pending: VecDeque<(Token, Source)> = VecDeque::new();
        // each rejected guess still costs its fed token and logits row
        let mut ngram_left = spec.ngram;
        while out.tokens.len() < budget {
            if !out.tokens.is_empty() {
                safety_stop(&mut st, max_seq_len);
                if st.status != Status::Started {
                    break;
                }
            }

            let window =
                Decoder::repetition_window(st.token_count).and_then(|range| toks.get(range));
            // a guessed path may reach a state the real one never does; drafting just stops
            let Ok(forced) = decoder.lookahead(&mut st, window) else {
                break;
            };
            let (text, tok, src) = match forced {
                Some((text, tok)) => {
                    if pending.front().is_some_and(|p| p.0 == tok) {
                        pending.pop_front();
                    } else {
                        pending.clear();
                    }

                    (text, tok, Source::Forced)
                }
                None => {
                    certain = false;
                    if pending.is_empty() && spec.carry {
                        pending.extend(self.carry_guess(slot_idx, &st, &toks)?);
                    }
                    if pending.is_empty() && ngram_left > 0 {
                        let most = ngram_left.min(budget - out.tokens.len());
                        pending.extend(self.ngram_guess(&st, &toks, most));
                        ngram_left -= pending.len();
                    }

                    match pending.pop_front() {
                        Some((tok, src)) => (tables.vocab.text(tok).to_owned(), tok, src),
                        None => break,
                    }
                }
            };

            if st.update_after_decode(&text, tok, false).is_err() {
                break;
            }

            if st.token_count <= max_seq_len {
                toks[st.token_count - 1] = tok;
            }

            out.tokens.push(tok);
            out.sources.push(src);
            if st.status != Status::Started {
                out.host_only = certain;
                break;
            }
        }

        Ok(out)
    }

    /// Records the slot's window text so far, for the log when its next window starts
    fn note_window_text(&mut self, slot_idx: usize) {
        let init = self.init_tokens.len();
        let eot = self.tables.tok.eot();
        let n = self.seq_lens[slot_idx];
        let seen = &mut self.spec_rows[slot_idx].seen;
        seen.clear();
        seen.extend(
            self.all_tokens[slot_idx][init.min(n)..n]
                .iter()
                .filter(|&&t| t < eot),
        );
    }

    /// Ends a slot's window for drafting: its text goes to the n-gram log and its
    /// memory resets, as its KV does
    pub(crate) fn spec_new_window(&mut self, slot_idx: usize) {
        let memory = std::mem::take(&mut self.spec_rows[slot_idx]);
        self.spec_log.push_window(&memory.seen);
    }

    /// The n-gram draft at a sampled position: up to `most` text tokens that followed the
    /// window's last [`NGRAM`] text tokens, earlier in the window, else in the log. Only
    /// right after a text token, so a timestamp the model may pick next isn't guessed
    /// over
    fn ngram_guess(
        &self,
        st: &crate::decode::SlotState,
        toks: &[Token],
        most: usize,
    ) -> Vec<(Token, Source)> {
        let eot = self.tables.tok.eot();
        let n = st.token_count.min(toks.len());
        let init = self.init_tokens.len().min(n);
        if n == 0 || toks[n - 1] >= eot {
            return Vec::new();
        }

        let text: Vec<Token> = toks[init..n].iter().copied().filter(|&t| t < eot).collect();
        let mut next = in_window_continuation(&text);
        if next.is_empty() && text.len() >= NGRAM {
            next = self.spec_log.continuation(&text[text.len() - NGRAM..]);
        }

        next.iter()
            .take(most)
            .map(|&t| (t, Source::Ngram))
            .collect()
    }

    /// The carry draft at a sampled position: right after `[bare space, timestamp]`,
    /// where the space replaced the model's ` X`, propose `X` without its space
    fn carry_guess(
        &self,
        slot_idx: usize,
        st: &crate::decode::SlotState,
        toks: &[Token],
    ) -> Result<Vec<(Token, Source)>> {
        let n = st.token_count;
        let Some((at, pick)) = self.spec_rows[slot_idx].carry else {
            return Ok(Vec::new());
        };
        if n < 2 || at + 2 != n || toks.get(at) != Some(&BARE_SPACE) {
            return Ok(Vec::new());
        }

        let tables = &self.tables;
        if !tables.is_timing(toks[n - 1]) {
            return Ok(Vec::new());
        }

        let word = tables.vocab.text(pick);
        let Some(rest) = word.strip_prefix(' ').filter(|r| !r.is_empty()) else {
            return Ok(Vec::new());
        };
        Ok(tables
            .tok
            .encode(rest)?
            .into_iter()
            .map(|t| (t, Source::Carry))
            .collect())
    }
}

/// Runs one row's chain: position `j` reads the logits of the row's fed token `j` when
/// it samples, emits its token and updates the row as the one-token step would; the
/// chain goes on while the token equals draft `j` and the window keeps decoding. Returns
/// the tokens emitted and the drafts kept. `logits` (the logits, their step, the row's
/// index in it) is `None` for a host-only row, which must end inside its drafts
#[allow(clippy::too_many_arguments)]
fn run_chain(
    rs: &mut RowState<'_>,
    memory: &mut [RowMemory],
    decoder: &Decoder,
    slot_idx: usize,
    mut first: Option<Prepared>,
    draft: &Draft,
    logits: Option<(&dyn StepLogits, &VerifyStep, usize)>,
) -> Result<(usize, usize)> {
    let max_seq_len = rs.config.max_tokens;
    let m = draft.tokens.len();
    if let Some((_, step, g)) = logits {
        ensure!(
            step.slots[g] == slot_idx && step.row_len(g) == m + 1,
            "verify row {g} (slot {}) fed {} tokens; slot {slot_idx} has {m} drafts",
            step.slots[g],
            step.row_len(g)
        );
    }

    let mut emitted = 0;
    let mut kept = 0;
    loop {
        let prepared = match first.take() {
            Some(p) => p,
            None => {
                let state = &mut rs.slots[slot_idx].state;
                let window = Decoder::repetition_window(state.token_count)
                    .map(|range| &rs.all_tokens[slot_idx][range]);
                decoder.prepare(state, window)?
            }
        };
        let row = if Decoder::needs_logits(&prepared) {
            let (l, step, g) = logits.ok_or_else(|| {
                eyre!(
                    "slot {slot_idx}: host-only row samples at position {emitted} of its \
                     forced chain {:?}",
                    draft.tokens
                )
            })?;
            let r = step.logit_row(step.starts[g] + emitted).ok_or_else(|| {
                eyre!(
                    "slot {slot_idx} samples at position {emitted} of its drafts {:?}, \
                     which were to be forced",
                    draft.tokens
                )
            })?;
            Some(l.row(r)?)
        } else {
            None
        };

        let state = &mut rs.slots[slot_idx].state;
        let index = state.token_count;
        let d = decoder.finish(prepared, row, state)?;
        remember_carry(&mut memory[slot_idx], index, &d, decoder);
        rs.apply_decision(slot_idx, &d, true)?;
        emitted += 1;
        // the next position follows only on the draft the step fed there
        let on_draft = emitted <= m && d.token == draft.tokens[emitted - 1];
        kept += usize::from(on_draft);
        if rs.slots[slot_idx].state.status != Status::Started || !on_draft {
            break;
        }

        // the main loop's safety stops, which the one-token path runs before each step
        safety_stop(&mut rs.slots[slot_idx].state, max_seq_len);
        if rs.slots[slot_idx].state.status != Status::Started {
            break;
        }
    }

    if draft.host_only {
        ensure!(
            emitted == m && rs.slots[slot_idx].state.status != Status::Started,
            "slot {slot_idx}: host-only chain {:?} emitted {emitted} tokens and left the \
             window {:?}",
            draft.tokens,
            rs.slots[slot_idx].state.status
        );
    }

    Ok((emitted, kept))
}

/// Records the model's pick when Phase 4 replaced it with a bare space, for
/// [`Transcriber::carry_guess`]; `index` is the position the token takes
fn remember_carry(memory: &mut RowMemory, index: usize, d: &Decision, decoder: &Decoder) {
    let Some(main) = d.samples.first().filter(|s| s.stage == Stage::Main) else {
        return;
    };
    let pick = main.argmax.token;
    if d.token != BARE_SPACE || pick == BARE_SPACE {
        return;
    }

    let tables = decoder.tables();
    let text = tables.vocab.text(pick);
    if pick < tables.tok.eot() && text.starts_with(' ') && text.len() > 1 {
        memory.carry = Some((index, pick));
    }
}
