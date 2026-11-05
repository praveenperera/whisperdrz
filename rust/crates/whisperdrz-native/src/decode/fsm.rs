//! The constrained-decoding state machine, `decode_batch` (`decoding.py:971-2049`) one
//! row at a time
//!
//! Python classifies a batch into constraint groups and applies each group's mask
//! with one tensor op. Every op touches only its own rows, apart from the `][` block
//! that applies to all of them, so running one row through the same op sequence gives
//! the same mask. The ops below run in python's order, which matters where a step
//! overwrites a whole row (`batch_mask[rows] = True`)

use std::sync::Arc;

use eyre::{Result, ensure, eyre};
use half::f16;

use super::logits::{self, Argmax};
use super::mask::TokenMask;
use super::state::{SlotState, Status};
use super::tables::{BRACKET_CLOSE_OPEN, DecoderTables, TIMING_STEP, parse_timing};
use super::text_rules::{
    self, TIMINGS_FORMAT, Unclosed, after_speaker, asterisk_state, count_words, get_line,
    has_timed_content, has_timed_word, is_word_char, last_timing_in, line_has_content, tag_closes,
    tag_opens, text_has_word,
};
use crate::text::vocab::starts_with_timing;
use crate::text::{Token, py};

/// `MAX_TOKENS`: a window stops at this many tokens
pub const MAX_TOKENS: usize = 448;
const MIN_REPEATS: usize = 5;
/// Tokens the repetition check reads, `MIN_REPEATS * 20`
pub const REPETITION_WINDOW: usize = MIN_REPEATS * 20;
const MAX_WORDS_WITHOUT_TIMING: usize = 5;
const BRACKET_MAX_LEN: usize = 20;

// fixed token ids decode_batch blocks by number
const BRACKET_OPEN: Token = 58;
const BRACKET_CLOSE: Token = 60;
const SPACE_BRACKET: Token = 542;
const CUT_TOKEN: Token = 6672;
const BARE_SPACE: Token = 220;

const EOT_TEXT: &str = "<|endoftext|>";

/// The `decode_batch` keyword settings
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeConfig {
    pub temperature: f64,
    pub allow_tags: bool,
    pub allow_asterisks: bool,
    pub speaker_tags: bool,
}

impl Default for DecodeConfig {
    /// What the pipeline runs: greedy, no `<tags>`, no `*emphasis*`, speaker tags on
    fn default() -> Self {
        Self {
            temperature: 0.0,
            allow_tags: false,
            allow_asterisks: false,
            speaker_tags: true,
        }
    }
}

impl DecodeConfig {
    /// Only the pipeline's settings are ported; the rest are rejected
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature == 0.0,
            "temperature {} not supported: only greedy decoding is ported",
            self.temperature
        );
        ensure!(!self.allow_tags, "allow_tags=True is not ported");
        ensure!(!self.allow_asterisks, "allow_asterisks=True is not ported");
        ensure!(self.speaker_tags, "speaker_tags=False is not ported");
        Ok(())
    }
}

/// Which argmax picked a token
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The batched sample over the constraint mask (Phase 3)
    Main,
    /// `_sample_masked` after an EOT came too early (Phase 4, `-inf` fill)
    EotResample,
    /// The batched Phase 4 resample (timing or sparse-timestamp rejection)
    Correction,
}

/// One argmax on the way to a token, kept so parity tests can show how close calls were
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub stage: Stage,
    pub argmax: Argmax,
    /// The repetition penalty touched this row and a true division (torch's CPU kernel)
    /// would have picked another token
    pub division_sensitive: bool,
}

/// The token a row emits
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub text: String,
    pub token: Token,
    /// Argmaxes that ran for this row, in order (empty for forced tokens)
    pub samples: Vec<Sample>,
}

impl Decision {
    fn set(&mut self, text: &str, token: Token) {
        self.text = text.to_owned();
        self.token = token;
    }
}

/// The decoder over a fixed set of tables and settings
pub struct Decoder<'t> {
    tables: &'t DecoderTables,
    config: DecodeConfig,
}

/// A row [`Decoder::prepare`] planned and masked, waiting for its logits
pub struct Prepared {
    plan: Plan,
    /// `None` for a forced token, which reads no logits
    masks: Option<Masks>,
}

/// Phase 2's output for a sampled row
struct Masks {
    /// The language's base mask, which Phase 4 starts its resample masks from
    base: Arc<TokenMask>,
    mask: TokenMask,
    /// The mask is the valid timing range, so no repetition penalty applies
    special_timing: bool,
}

/// What Phase 1 decided for a row
enum Plan {
    Fixed(String, Token),
    /// Language token at the first step
    Language,
    /// Speaker digit after `[` on a new line
    SpeakerDigit,
    /// Timing-only sample: forced timestamp or line-start timing
    Timing {
        cur: String,
        max_timing: Option<f64>,
        min_timing: Option<f64>,
    },
    Normal(Box<Normal>),
}

/// Context of a normally decoded row (python's `ctx` dict plus its group memberships)
struct Normal {
    line: String,
    cur: String,
    last_timing: Option<f64>,
    last_token_str: String,
    even_timings: bool,
    last_was_timing: bool,
    last_was_space: bool,
    words_since_timing: usize,
    max_timing: Option<f64>,
    duration: f64,
    offset: f64,
    /// `force_timing_indices` or `close_bracket_even_timing_indices`: the mask is
    /// replaced by the valid timing range
    special_timing: bool,
    after_timing: AfterTiming,
    last_was_space_group: bool,
    space_even_timing: bool,
    /// `partial` after the open `[`, when inside one
    in_bracket: Option<String>,
    in_angle: bool,
    in_asterisk: bool,
    in_paren: bool,
    even_not_after_timing: bool,
    punct_or_bracket: bool,
    block_open_bracket: bool,
    block_punct_nonalnum: bool,
    block_quote: bool,
}

/// The `last_was_timing_*` groups
#[derive(Clone, Copy, PartialEq, Eq)]
enum AfterTiming {
    No,
    /// Even count, a word already timed: only space-starting tokens and newline
    Even,
    /// Even count, nothing timed yet: only a bare space
    EvenFirstWord,
    /// Even count, a timed tag but no word: bare space or newline
    EvenTimedTag,
    /// Odd count: no space-starting tokens
    Odd,
}

impl<'t> Decoder<'t> {
    pub fn new(tables: &'t DecoderTables, config: DecodeConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { tables, config })
    }

    pub fn tables(&self) -> &'t DecoderTables {
        self.tables
    }

    /// The `all_tokens` slice the repetition check reads at this token count:
    /// `max(0, n - 100)..n` when `n >= 25` and `n % 10 == 0`
    pub fn repetition_window(token_count: usize) -> Option<std::ops::Range<usize>> {
        (token_count >= MIN_REPEATS * 5 && token_count.is_multiple_of(10))
            .then(|| token_count.saturating_sub(REPETITION_WINDOW)..token_count)
    }

    /// One row of `decode_batch`: picks the next token for `state` from its fp16
    /// `logits` row and updates the fields `decode_batch` mutates. `window` is
    /// `all_tokens[slot, range]` for [`Self::repetition_window`] and is only read when
    /// that is `Some`. The caller then applies [`SlotState::update_after_decode`]
    pub fn decode_row(
        &self,
        logits: &[f16],
        state: &mut SlotState,
        window: Option<&[Token]>,
    ) -> Result<Decision> {
        self.check_row(logits)?;
        let prepared = self.prepare(state, window)?;
        self.finish(prepared, Some(logits), state)
    }

    /// The half of [`Self::decode_row`] that reads no logits: Phase 1's plan and
    /// Phase 2's mask. It may change `state` as the plan does, so call it once per row
    /// and step, then [`Self::finish`] with the row's logits. Splitting the row lets
    /// this half run while the GPU computes the step
    pub fn prepare(&self, state: &mut SlotState, window: Option<&[Token]>) -> Result<Prepared> {
        let ds = self.tables;
        let plan = self.plan(state, window)?;
        if matches!(plan, Plan::Fixed(..)) {
            return Ok(Prepared { plan, masks: None });
        }

        let base = ds.base_blocked(&state.language);
        let mut mask = (*base).clone();
        let special_timing = match &plan {
            Plan::Language => {
                // all blocked, then only language tokens enabled
                mask.fill(true);
                mask.and_not(&ds.masks.lang_enabled);
                false
            }
            Plan::SpeakerDigit => {
                mask.fill(true);
                mask.and_not(&ds.masks.speaker_id_mask);
                false
            }
            Plan::Timing {
                cur,
                max_timing,
                min_timing,
            } => {
                self.timing_only(&mut mask, cur, *max_timing, *min_timing);
                true
            }
            Plan::Normal(n) => {
                self.normal_mask(&mut mask, &base, n, state.last_token_id);
                if n.special_timing {
                    self.timing_only(&mut mask, &n.cur, n.max_timing, n.last_timing);
                }

                n.special_timing
            }
            Plan::Fixed(..) => unreachable!(),
        };

        Ok(Prepared {
            plan,
            masks: Some(Masks {
                base,
                mask,
                special_timing,
            }),
        })
    }

    /// Whether [`Self::finish`] reads the row's logits; a forced token doesn't
    pub fn needs_logits(prepared: &Prepared) -> bool {
        prepared.masks.is_some()
    }

    /// Phase 1 alone, for drafting: the token the next position is forced to, or `None`
    /// when it samples. Changes `state` as [`Self::prepare`] would, so run it on a copy
    pub fn lookahead(
        &self,
        state: &mut SlotState,
        window: Option<&[Token]>,
    ) -> Result<Option<(String, Token)>> {
        Ok(match self.plan(state, window)? {
            Plan::Fixed(text, token) => Some((text, token)),
            _ => None,
        })
    }

    /// The rest of [`Self::decode_row`] for a row [`Self::prepare`] planned, on the
    /// same `state`: Phase 3's argmax and Phase 4's corrections. `logits` may be `None`
    /// only when [`Self::needs_logits`] is false
    pub fn finish(
        &self,
        prepared: Prepared,
        logits: Option<&[f16]>,
        state: &mut SlotState,
    ) -> Result<Decision> {
        let ds = self.tables;
        let Prepared { plan, masks } = prepared;
        let Some(Masks {
            base,
            mask,
            special_timing,
        }) = masks
        else {
            let Plan::Fixed(text, token) = plan else {
                unreachable!("only a fixed plan has no mask")
            };
            return Ok(Decision {
                text,
                token,
                samples: Vec::new(),
            });
        };
        let logits = logits.ok_or_else(|| eyre!("a sampled row needs its logits"))?;
        self.check_row(logits)?;
        let normal = match &plan {
            Plan::Normal(n) => Some(n.as_ref()),
            _ => None,
        };

        // Phase 3: additive mask, repetition penalty on non-timing rows, argmax
        let mut division_sensitive = false;
        let main = if !special_timing && state.token_count > 0 {
            let mut recent = state.recent_tokens.clone();
            recent.sort_unstable();
            recent.dedup();
            recent.retain(|&t| (t as usize) < logits.len());

            let masked = logits::MaskedRow::new(logits, &mask, 0.0, &recent);
            let before: Vec<f16> = recent.iter().map(|&t| masked.value(t)).collect();
            let penalized: Vec<f16> = before.iter().map(|&x| logits::penalize(x)).collect();
            let picked = masked.argmax(&penalized);
            // the true-division row differs from the penalized one only at `recent`, and
            // only matters when some entry there rounds differently
            let alt_at: Vec<f16> = before
                .iter()
                .map(|&x| logits::penalize_true_division(x))
                .collect();
            if alt_at.iter().zip(&penalized).any(|(a, p)| a != p) {
                division_sensitive = masked.argmax(&alt_at).token != picked.token;
            }

            picked
        } else {
            logits::MaskedRow::new(logits, &mask, 0.0, &[]).argmax(&[])
        };
        let mut decision = Decision {
            text: ds.vocab.text(main.token).to_owned(),
            token: main.token,
            samples: vec![Sample {
                stage: Stage::Main,
                argmax: main,
                division_sensitive,
            }],
        };

        if matches!(plan, Plan::Language) {
            state.language = decision.text.replace("<|", "").replace("|>", "");
        }

        // Phase 4: corrections for normal rows
        if let Some(n) = normal
            && let Some(resample) = self.correct(logits, state, n, &base, &mut decision)?
        {
            let a = logits::MaskedRow::new(logits, &resample, -0.0, &[]).argmax(&[]);
            decision.samples.push(Sample {
                stage: Stage::Correction,
                argmax: a,
                division_sensitive: false,
            });
            decision.set(ds.vocab.text(a.token), a.token);
        }

        Ok(decision)
    }

    fn check_row(&self, logits: &[f16]) -> Result<()> {
        ensure!(
            logits.len() == self.tables.n_vocab(),
            "logits row has {} entries, vocab is {}",
            logits.len(),
            self.tables.n_vocab()
        );
        Ok(())
    }

    /// Phase 1 for one row
    fn plan(&self, state: &mut SlotState, window: Option<&[Token]>) -> Result<Plan> {
        let ds = self.tables;
        let eot = ds.tok.eot();
        let token_count = state.token_count;
        let eot_plan = || Plan::Fixed(EOT_TEXT.to_owned(), eot);

        if state.status != Status::Started || token_count >= MAX_TOKENS {
            state.stop_reason = Some(if state.status != Status::Started {
                "not_started".to_owned()
            } else {
                format!("max_tokens:{token_count}>={MAX_TOKENS}")
            });
            state.status = Status::Process;
            return Ok(eot_plan());
        }

        if state.last_token_id == eot {
            state.status = Status::Process;
            state.stop_reason = Some("eot_predicted".to_owned());
            return Ok(eot_plan());
        }

        if let Some(range) = Self::repetition_window(token_count) {
            let window = window.ok_or_else(|| {
                eyre!("repetition check at {token_count} tokens needs all_tokens[{range:?}]")
            })?;
            ensure!(
                window.len() == range.len(),
                "repetition window has {} tokens, want {}",
                window.len(),
                range.len()
            );
            if let Some(pattern) = self.repeated_pattern(window) {
                let repeated = ds.tok.decode(&pattern)?;
                state.status = Status::Process;
                state.stop_reason =
                    Some(format!("repetition:{}x{MIN_REPEATS}", py::repr(&repeated)));
                return Ok(eot_plan());
            }
        }

        if token_count == 1 {
            state.forced_timing_before_newline = false;
            return Ok(Plan::Language);
        }

        if token_count == 2 {
            return Ok(Plan::Fixed(String::new(), ds.tok.transcribe()));
        }

        // a forced timestamp fires before predict_next so it precedes a deferred bracket
        if state.force_timestamp_next {
            state.force_timestamp_next = false;
            let line = get_line(&state.cur);
            let last_timing = match TIMINGS_FORMAT.captures_iter(line).last() {
                Some(c) => {
                    Some(py::float(&c[1]).ok_or_else(|| eyre!("ValueError: float({:?})", &c[1]))?)
                }
                None => None,
            };
            return Ok(Plan::Timing {
                cur: state.cur.clone(),
                max_timing: self.max_timing(state),
                min_timing: last_timing,
            });
        }

        if let Some(text) = state.predict_next.take() {
            if text == "\n" && !line_has_content(&state.cur) {
                state.status = Status::Process;
                state.stop_reason = Some("empty_line_eot".to_owned());
                return Ok(eot_plan());
            }

            if text == EOT_TEXT {
                state.status = Status::Process;
                state.stop_reason = Some("eot_deferred".to_owned());
                return Ok(Plan::Fixed(text, eot));
            }

            let id = *ds
                .tok
                .encode(&text)?
                .first()
                .ok_or_else(|| eyre!("IndexError: predict_next {text:?} encodes to nothing"))?;
            return Ok(Plan::Fixed(text, id));
        }

        let cur = state.cur.clone();
        let line = get_line(&cur).to_owned();
        let line_len = py::len(&line);

        // speaker tag at line start: "[", digit, "]", " ", then a timing
        if self.config.speaker_tags && line_len < 5 {
            let ids = ds.ids;
            return Ok(match line_len {
                0 => Plan::Fixed("[".to_owned(), ids.open_bracket),
                1 if py::strip(&cur) == "[" => Plan::Fixed("0".to_owned(), ids.zero),
                1 => Plan::SpeakerDigit,
                2 => Plan::Fixed("]".to_owned(), ids.close_bracket),
                3 => Plan::Fixed(" ".to_owned(), ids.space),
                _ => {
                    // timestamps only have to rise within the same speaker; speakers can overlap
                    let speaker = line.chars().nth(1);
                    let prev_lines: Vec<&str> = py::rstrip(&cur).split('\n').collect();
                    let mut min_timing = None;
                    if prev_lines.len() >= 2
                        && let Some(speaker) = speaker
                        && prev_lines[prev_lines.len() - 2].starts_with(&format!("[{speaker}]"))
                    {
                        min_timing = last_timing_in(&cur);
                    }

                    Plan::Timing {
                        cur,
                        max_timing: self.max_timing(state),
                        min_timing,
                    }
                }
            });
        }

        // the no-speaker-tags line start is rejected by DecodeConfig::validate
        if line != state.line_cache {
            let matches: Vec<_> = TIMINGS_FORMAT.find_iter(&line).collect();
            state.timing_count = matches.len();
            state.last_timing = matches.last().and_then(|m| parse_timing(m.as_str()));
            state.line_cache = line.clone();
        }

        let timing_count = state.timing_count;
        let even_timings = timing_count.is_multiple_of(2);
        let last_timing = state.last_timing;
        let last_token_str = ds.vocab.text(state.last_token_id).to_owned();
        let last_was_timing = last_token_str.starts_with("<|") && last_token_str.ends_with("|>");
        let last_char = py::last_char(&line);
        let last_was_space = last_char == Some(' ');

        let (duration, offset) = (state.duration, state.offset);
        let time_remaining = match last_timing {
            Some(t) if state.has_duration() => Some(duration - offset - t),
            _ => None,
        };
        let near_end = time_remaining.is_some_and(|t| t < 3.0);
        let approaching_end = time_remaining.is_some_and(|t| t < 5.0);

        let mut words_since_timing = 0;
        if timing_count > 0
            && let Some(end) = line.rfind("|>")
        {
            words_since_timing = count_words(&line[end + 2..]);
        }

        let in_bracket = line.rfind('[') > line.rfind(']');
        let opens = tag_opens(&line);
        let closes = tag_closes(&line);
        let in_angle = opens.last() > closes.last();
        let in_paren = line.rfind('(') > line.rfind(')');
        let (in_asterisk, asterisk_close) = asterisk_state(&line);

        // force a timing near the end, or before the first word when only tags are timed
        let force_timing = last_was_space
            && (near_end
                || (approaching_end && words_since_timing >= 2)
                || (even_timings && !has_timed_word(&line)));

        let mut after_timing = AfterTiming::No;
        let mut last_was_space_group = false;
        let mut space_even_timing = false;
        if force_timing {
        } else if last_was_timing {
            after_timing = if !even_timings {
                AfterTiming::Odd
            } else if has_timed_word(&line) {
                AfterTiming::Even
            } else if has_timed_content(&line) {
                AfterTiming::EvenTimedTag
            } else {
                AfterTiming::EvenFirstWord
            };
        } else if last_was_space {
            last_was_space_group = true;
            // timing before an inline [tag]: word<|t|> <|t|>[tag]
            space_even_timing = even_timings && has_timed_word(&line);
        }

        let mut bracket_partial = None;
        if in_bracket {
            let partial = line.rfind('[').map_or("", |i| &line[i + 1..]);
            if (!partial.is_empty() && !ds.bracket_prefixes.contains(partial))
                || py::len(partial) > BRACKET_MAX_LEN
            {
                return Ok(Plan::Fixed("]".to_owned(), ds.ids.close_bracket));
            }

            bracket_partial = Some(partial.to_owned());
        }

        // an open `*...` whose body went bad (space, newline, too long) is closed now
        if in_asterisk && asterisk_close.is_some() {
            return Ok(Plan::Fixed("*".to_owned(), ds.ids.asterisk));
        }

        let punct = |c: Option<char>| c.is_some_and(|c| ".,!?-'".contains(c));
        Ok(Plan::Normal(Box::new(Normal {
            max_timing: self.max_timing(state),
            special_timing: force_timing || last_token_str == "]",
            after_timing,
            last_was_space_group,
            space_even_timing,
            in_bracket: bracket_partial,
            in_angle,
            in_asterisk,
            in_paren,
            even_not_after_timing: even_timings && !last_was_space && !force_timing,
            punct_or_bracket: punct(last_char) || last_token_str == "]",
            block_open_bracket: is_word_char(last_char) || punct(last_char),
            block_punct_nonalnum: !is_word_char(last_char),
            block_quote: last_token_str.contains('"'),
            line,
            cur,
            last_timing,
            last_token_str,
            even_timings,
            last_was_timing,
            last_was_space,
            words_since_timing,
            duration,
            offset,
        })))
    }

    /// `(duration - offset) if duration else None`
    fn max_timing(&self, state: &SlotState) -> Option<f64> {
        state
            .has_duration()
            .then_some(state.duration - state.offset)
    }

    /// The repeated pattern (timing tokens dropped) that makes up the window's tail
    /// `MIN_REPEATS` times over, shortest first
    fn repeated_pattern(&self, window: &[Token]) -> Option<Vec<Token>> {
        let tokens: Vec<Token> = window
            .iter()
            .copied()
            .filter(|&t| !self.tables.is_timing(t))
            .collect();
        if tokens.len() < MIN_REPEATS {
            return None;
        }

        for len in 1..=tokens.len() / MIN_REPEATS {
            let tail = &tokens[tokens.len() - MIN_REPEATS * len..];
            let pattern = &tail[..len];
            if tail.chunks(len).all(|c| c == pattern) {
                return Some(pattern.to_vec());
            }
        }

        None
    }

    /// Block everything but the valid timing range (`special_timing_indices`)
    fn timing_only(
        &self,
        mask: &mut TokenMask,
        cur: &str,
        max_timing: Option<f64>,
        min_timing: Option<f64>,
    ) {
        let line = get_line(cur);
        let (s, e) = text_rules::timing_range(cur, line, max_timing, min_timing);
        mask.fill(true);
        if s < e {
            for &t in self.tables.timing_ids(s, e) {
                mask.clear(t);
            }
        }
    }

    /// Phase 1.5 and Phase 2 for a normal row, in python's op order
    fn normal_mask(&self, mask: &mut TokenMask, base: &TokenMask, n: &Normal, last_token: Token) {
        let ds = self.tables;
        let m = &ds.masks;
        let ids = ds.ids;

        // eot before any content while audio remains
        if n.last_timing.is_none() && n.duration != 0.0 && n.duration - n.offset > 2.0 {
            mask.set(ds.tok.eot());
        }

        // newline only once the line has a word or tag and its timings are paired
        let without_timestamps = TIMINGS_FORMAT.replace_all(after_speaker(&n.line), "");
        let has_line_content =
            text_has_word(&without_timestamps) || without_timestamps.contains('[');
        if !has_line_content || !n.even_timings {
            mask.set(ids.newline);
        }

        if last_token == BRACKET_OPEN || last_token == SPACE_BRACKET {
            mask.set(CUT_TOKEN);
        }

        mask.set(BRACKET_CLOSE_OPEN);
        if last_token == BRACKET_CLOSE {
            mask.set(BRACKET_OPEN);
            mask.set(SPACE_BRACKET);
        }

        if n.after_timing != AfterTiming::No {
            mask.or(&m.after_timing_blocked);
        }

        match n.after_timing {
            AfterTiming::Even => {
                mask.fill(true);
                mask.and_not(&m.space_starting_enabled);
                mask.or(base);
                mask.set(SPACE_BRACKET);
            }
            AfterTiming::EvenFirstWord => {
                mask.fill(true);
                mask.clear(BARE_SPACE);
            }
            AfterTiming::EvenTimedTag => {
                mask.fill(true);
                mask.clear(BARE_SPACE);
                mask.clear(ids.newline);
            }
            AfterTiming::Odd => mask.or(&m.block_space_starting),
            AfterTiming::No => {}
        }

        if n.last_was_space_group {
            mask.or(&m.block_punct);
            mask.or(&m.block_space_starting);
        }

        if n.block_punct_nonalnum {
            mask.or(&m.block_punct);
        }

        if n.block_quote {
            mask.or(&m.block_quote);
        }

        match &n.in_bracket {
            Some(partial) => {
                // only tokens that keep the tag a valid prefix; a bare `]` if none do
                let allowed = ds
                    .bracket_allowed
                    .get(partial.as_str())
                    .map_or(&[][..], Vec::as_slice);
                let allow = if allowed.is_empty() {
                    TokenMask::from_ids(ds.n_vocab(), [ids.close_bracket])
                } else {
                    TokenMask::from_ids(ds.n_vocab(), allowed.iter().copied())
                };
                mask.or_not(&allow);
            }
            None => mask.or(&m.block_contains_close_bracket),
        }

        if n.in_angle {
            mask.or(&m.in_angle_blocked);
            mask.or(&m.block_asterisk);
            mask.or(&m.space_asterisk_token);
            mask.or(&m.block_open_bracket);
        } else {
            mask.or(&m.block_greater_than);
        }

        if n.in_asterisk {
            mask.or(&m.block_open_bracket);
            mask.or(&m.block_less_than);
        }

        if n.in_paren {
            mask.or(&m.in_paren_blocked);
        } else {
            mask.or(&m.block_close_paren);
        }

        if n.even_not_after_timing {
            mask.or(&m.block_timings);
        }

        if n.punct_or_bracket {
            mask.or(&m.block_punct);
        }

        if n.block_open_bracket {
            mask.or(&m.block_open_bracket);
        }

        // no timestamps at or before the line's last one
        if let Some(t) = n.last_timing
            && t > 0.0
        {
            let end = py::round(t / TIMING_STEP) + 1;
            if end > 0 {
                for &id in ds.timing_ids(0, end as usize) {
                    mask.set(id);
                }
            }
        }
    }

    /// Phase 4 for a normal row: may replace the token outright, defer it through
    /// `predict_next`, or return the mask for the batched resample
    fn correct(
        &self,
        logits: &[f16],
        state: &mut SlotState,
        n: &Normal,
        base: &TokenMask,
        d: &mut Decision,
    ) -> Result<Option<TokenMask>> {
        let ds = self.tables;
        let ids = ds.ids;
        let m = &ds.masks;
        let line = n.line.as_str();
        let last_char = py::last_char(line);

        let last_non_space_was_timing =
            n.last_was_timing || (n.last_was_space && py::rstrip(line).ends_with("|>"));
        let mut is_eot = d.text == EOT_TEXT;

        // an eot before any content, with audio left, is resampled without eot
        if is_eot && n.last_timing.is_none() && n.duration - n.offset > 2.0 {
            let mut blocked = base.clone();
            blocked.set(ds.tok.eot());
            if !is_word_char(last_char) {
                blocked.or(&m.block_punct);
            }

            let row: Vec<f16> = logits
                .iter()
                .zip(0..)
                .map(|(&x, i)| if blocked.get(i) { f16::NEG_INFINITY } else { x })
                .collect();
            let a = logits::argmax(&row);
            d.samples.push(Sample {
                stage: Stage::EotResample,
                argmax: a,
                division_sensitive: false,
            });
            d.set(ds.vocab.text(a.token), a.token);
            is_eot = false;
        }

        let newtext = d.text.clone();
        let next_is_space = newtext.starts_with(' ');
        let next_is_timing = starts_with_timing(&newtext);
        let mut correction = false;
        let mut force_timing_sample = false;

        // a timing before newline or eot, but open brackets close first
        if (newtext == "\n" || is_eot) && !last_non_space_was_timing {
            let u = Unclosed::of(line);
            if n.last_token_str != "]" && u.bracket {
                d.set("]", ids.close_bracket);
            } else if n.last_token_str != ">" && u.angle {
                d.set(">", ids.greater);
            } else if n.last_token_str != ")" && u.paren {
                d.set(")", ids.close_paren);
            } else {
                // the asterisk close needs allow_asterisks, which validate() rejects
                state.predict_next = Some(newtext.clone());
                force_timing_sample = true;
            }

            correction = true;
        }

        // a timing before "("
        if !correction && (newtext == " (" || (newtext == "(" && !n.last_was_timing)) {
            state.predict_next = Some(newtext.clone());
            force_timing_sample = true;
            correction = true;
        }

        // a timing before a [tag]
        if !correction
            && !n.last_was_timing
            && (newtext == "[" || (newtext.starts_with(" [") && !newtext.starts_with(" []")))
        {
            if newtext.starts_with(" [") {
                // the space goes first so the timing doesn't jam against punctuation
                d.set(" ", ids.space);
                state.predict_next = Some(newtext[1..].to_owned());
                state.force_timestamp_next = true;
            } else {
                state.predict_next = Some(newtext.clone());
                force_timing_sample = true;
            }

            correction = true;
        }

        // open brackets close before a space, and an odd timing pairs up first
        if !correction && next_is_space {
            let u = Unclosed::of(line);
            if n.last_token_str != "]" && u.bracket {
                d.set("]", ids.close_bracket);
                correction = true;
            } else if n.last_token_str != ">" && u.angle {
                d.set(">", ids.greater);
                correction = true;
            } else if n.last_token_str != ")" && u.paren {
                d.set(")", ids.close_paren);
                correction = true;
            } else if !n.even_timings {
                force_timing_sample = true;
                correction = true;
            }
        }

        let (time_remaining, near_end, approaching_end) = timing_status(n);

        // near the end, a bare space so the next step times the word
        if !correction
            && next_is_space
            && newtext != " "
            && (near_end || (approaching_end && n.words_since_timing >= 2))
        {
            d.set(" ", ids.space);
            correction = true;
        }

        // too many untimed words: a space now, a timestamp next
        if !correction
            && next_is_space
            && !state.force_timing_every_word
            && n.words_since_timing >= MAX_WORDS_WITHOUT_TIMING
            && n.even_timings
        {
            d.set(" ", ids.space);
            state.force_timestamp_next = true;
            correction = true;
        }

        // sparse timing: an unwanted opening timestamp is resampled without timings
        if !correction && !state.force_timing_every_word && !n.space_even_timing {
            let is_opening = newtext.starts_with("<|")
                && newtext.ends_with("|>")
                && n.even_timings
                && n.last_was_space;
            let periodic = n.words_since_timing >= 3;
            let forced = near_end || (approaching_end && n.words_since_timing >= 2);
            if is_opening && !periodic && !forced {
                let mut blocked = base.clone();
                blocked.or(&m.block_timings);
                if n.last_was_space {
                    blocked.or(&m.block_space_starting);
                }

                if time_remaining > 2.0 {
                    blocked.set(ds.tok.eot());
                }

                if !is_word_char(last_char) {
                    blocked.or(&m.block_punct);
                }

                return Ok(Some(blocked));
            }
        }

        // no timestamp earlier than the line's last one
        if !correction
            && next_is_timing
            && let Some(last) = n.last_timing
            && newtext.starts_with("<|")
            && newtext.ends_with("|>")
        {
            let predicted =
                parse_timing(&newtext).ok_or_else(|| eyre!("ValueError: float({newtext:?})"))?;
            if predicted < last {
                force_timing_sample = true;
                correction = true;
            }
        }

        // a timing not right after a timing is resampled from the valid range
        if !correction && next_is_timing && !n.last_was_timing {
            force_timing_sample = true;
        }

        if force_timing_sample {
            let (s, e) = text_rules::timing_range(&n.cur, line, n.max_timing, n.last_timing);
            let mut blocked = TokenMask::full(ds.n_vocab());
            if s < e {
                for &t in ds.timing_ids(s, e) {
                    blocked.clear(t);
                }
            }

            return Ok(Some(blocked));
        }

        Ok(None)
    }
}

/// `_get_timing_status(ctx)`: (time remaining, near end, approaching end)
fn timing_status(n: &Normal) -> (f64, bool, bool) {
    let time_remaining = n.duration - n.offset - n.last_timing.unwrap_or(0.0);
    let has_timing = n.last_timing.is_some();
    (
        time_remaining,
        has_timing && time_remaining < 3.0,
        has_timing && time_remaining < 5.0,
    )
}
