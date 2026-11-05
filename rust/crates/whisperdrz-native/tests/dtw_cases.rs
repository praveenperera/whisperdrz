//! DTW port against values the Python reference produced on the CPU
//! (`tools/gen_dtw_cases.py`): the numba kernel, `_dtw_from_neg_matrix`,
//! `_apply_anchor_bands`, parsing, chunking, the text rebuild and whole
//! `fill_word_timings_batch` runs with the GPU forward replaced by synthetic matrices

// each test crate uses part of the shared helpers
#[allow(dead_code)]
#[path = "support/dtw.rs"]
mod support;

use std::cell::Cell;

use serde_json::Value;
use support::{
    TableTokenizer, bits, decode_floats, decode_matrix, opt_f64, opt_usize, read_json, strings,
    token_groups,
};
use whisperdrz_native::dtw::{
    CharTokens, DtwOptions, Matrices, PredWord, Speaker, TagAnchor, WordAnchor, apply_anchor_bands,
    chunk_speaker_lines, dtw_from_neg_matrix, dtw_path, fill_word_timings_batch, is_py_space,
    parse_predicted_lines, rebuild_with_timings, render_chunk,
};

fn cases(section: &str) -> Vec<Value> {
    let all = read_json("cases.json");
    all[section].as_array().unwrap().clone()
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn speaker(v: &Value) -> Speaker {
    match v.get("n") {
        Some(n) => Speaker::Num(n.as_u64().unwrap().to_string()),
        None => Speaker::Crosstalk,
    }
}

#[test]
fn whitespace_matches_python_isspace() {
    let all = read_json("cases.json");
    let py: std::collections::HashSet<u32> = all["py_space"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_u64().unwrap() as u32)
        .collect();
    for c in (0..0x11_0000u32).filter_map(char::from_u32) {
        assert_eq!(
            is_py_space(c),
            py.contains(&(c as u32)),
            "U+{:04X}",
            c as u32
        );
    }
}

#[test]
fn kernel_paths_match_numba() {
    let cases = cases("kernel");
    for (k, case) in cases.iter().enumerate() {
        let x = decode_matrix(&case["x"]);
        let (text_idx, time_idx) = dtw_path(&x);
        assert_eq!(
            text_idx,
            ints(&case["text_idx"]),
            "case {k} ({}x{}) text_idx",
            x.rows,
            x.cols
        );
        assert_eq!(
            time_idx,
            ints(&case["time_idx"]),
            "case {k} ({}x{}) time_idx",
            x.rows,
            x.cols
        );
    }

    assert_eq!(cases.len(), 32);
}

#[test]
fn neg_matrix_dtw_matches_python() {
    for (k, case) in cases("neg_matrix").iter().enumerate() {
        let x = decode_matrix(&case["x"]);
        let words = strings(&case["words"]);
        let got = dtw_from_neg_matrix(&x, words.len(), &token_groups(&case["word_tokens"]));
        let want = case["ret"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "case {k} length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(words[i], w[0].as_str().unwrap(), "case {k} word {i}");
            assert_eq!(
                (g.start, g.end),
                (w[1].as_f64().unwrap(), w[2].as_f64().unwrap()),
                "case {k} word {i}"
            );
        }
    }
}

#[test]
fn anchor_bands_match_python() {
    for (k, case) in cases("bands").iter().enumerate() {
        let mut x = decode_matrix(&case["x"]);
        let words = strings(&case["words"]);
        let word_anchors: Vec<WordAnchor> = case["word_anchors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| WordAnchor {
                word: a[0].as_u64().unwrap() as usize,
                start: a[1].as_f64().unwrap(),
                end: a[2].as_f64().unwrap(),
            })
            .collect();
        let tag_anchors: Vec<TagAnchor> = case["tag_anchors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| TagAnchor {
                start: a[0].as_f64().unwrap(),
                end: a[1].as_f64().unwrap(),
                prev: opt_usize(&a[2]),
                next: opt_usize(&a[3]),
            })
            .collect();
        apply_anchor_bands(
            &mut x,
            words.len(),
            &token_groups(&case["word_tokens"]),
            case["audio_cap"].as_u64().unwrap() as usize,
            &word_anchors,
            &tag_anchors,
            case["slack"].as_i64().unwrap(),
        )
        .unwrap();
        let (_, after) = decode_floats(&case["after"]);
        assert_eq!(bits(&x.data), bits(&after), "case {k}");
    }
}

#[test]
fn parse_matches_python() {
    for case in cases("parse") {
        let text = case["text"].as_str().unwrap();
        let p = parse_predicted_lines(text).unwrap();

        let flat: Vec<PredWord> = case["flat"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| PredWord {
                text: w[0].as_str().unwrap().to_owned(),
                start: opt_f64(&w[1]),
                end: opt_f64(&w[2]),
            })
            .collect();
        assert_eq!(p.words, flat, "{text:?} words");

        let lines: Vec<(Speaker, Vec<usize>)> = case["lines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| {
                (
                    speaker(&l[0]),
                    l[1].as_array()
                        .unwrap()
                        .iter()
                        .map(|w| w[0].as_u64().unwrap() as usize)
                        .collect(),
                )
            })
            .collect();
        let got: Vec<_> = p
            .lines
            .iter()
            .map(|l| (l.speaker.clone(), l.words.clone()))
            .collect();
        assert_eq!(got, lines, "{text:?} lines");

        let tags: Vec<TagAnchor> = case["tag_anchors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| TagAnchor {
                start: a[0].as_f64().unwrap(),
                end: a[1].as_f64().unwrap(),
                prev: opt_usize(&a[2]),
                next: opt_usize(&a[3]),
            })
            .collect();
        assert_eq!(p.tag_anchors, tags, "{text:?} tag anchors");
    }
}

#[test]
fn chunking_matches_python() {
    let tok = TableTokenizer::load();
    let mut multi = 0;
    for case in cases("chunk") {
        let text = case["text"].as_str().unwrap();
        let max_tokens = case["max_tokens"].as_u64().unwrap() as usize;
        let p = parse_predicted_lines(text).unwrap();
        let mut ct = CharTokens::new(&tok);
        let chunks = chunk_speaker_lines(&p.lines, &p.words, &mut ct, max_tokens);

        let got: Vec<Vec<(Speaker, Vec<usize>)>> = chunks
            .iter()
            .map(|c| {
                c.iter()
                    .map(|l| (l.speaker.clone(), l.words.to_vec()))
                    .collect()
            })
            .collect();
        let want: Vec<Vec<(Speaker, Vec<usize>)>> = case["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                c.as_array()
                    .unwrap()
                    .iter()
                    .map(|l| {
                        (
                            speaker(&l[0]),
                            l[1].as_array()
                                .unwrap()
                                .iter()
                                .map(|w| w.as_u64().unwrap() as usize)
                                .collect(),
                        )
                    })
                    .collect()
            })
            .collect();
        assert_eq!(got, want, "{text:?} at {max_tokens}");
        multi += usize::from(chunks.len() > 1);

        for (ci, chunk) in chunks.iter().enumerate() {
            let (rendered, _) = render_chunk(chunk, &p.words);
            assert_eq!(
                rendered,
                case["rendered"][ci].as_str().unwrap(),
                "{text:?} chunk {ci}"
            );
            let ids: Vec<u32> = case["ids"][ci]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(
                ct.tokens_chars(&rendered).ids,
                ids,
                "{text:?} chunk {ci} ids"
            );
        }
    }

    assert!(multi > 5, "too few multi-chunk cases ({multi})");
}

#[test]
fn rebuild_matches_python() {
    for case in cases("rebuild") {
        let text = case["text"].as_str().unwrap();
        let p = parse_predicted_lines(text).unwrap();
        let timings: Vec<Option<(f64, f64)>> = case["timings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (!t.is_null()).then(|| (t[0].as_f64().unwrap(), t[1].as_f64().unwrap())))
            .collect();
        let got = rebuild_with_timings(text, &p.words, &timings).unwrap();
        assert_eq!(got, case["ret"].as_str().unwrap(), "{text:?}");
    }
}

/// Rust's `{:.2}` against Python's `f"{x:.2f}"`, the rebuild's only float formatting
#[test]
fn two_decimal_format_matches_python() {
    for case in cases("format") {
        let v = case[0].as_f64().unwrap();
        let q = (v * 50.0).round_ties_even() / 50.0;
        assert_eq!(
            format!("{:.2}", q.max(0.0)),
            case[1].as_str().unwrap(),
            "quantised {v}"
        );
        assert_eq!(format!("{v:.2}"), case[2].as_str().unwrap(), "raw {v}");
    }
}

#[test]
fn batch_matches_python() {
    let tok = TableTokenizer::load();
    for (k, case) in cases("batch").iter().enumerate() {
        let slots: Vec<(String, i64)> = case["slots"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| (s[0].as_str().unwrap().to_owned(), s[1].as_i64().unwrap()))
            .collect();
        let opts = DtwOptions {
            max_tokens: case["max_tokens"].as_u64().unwrap() as usize,
            n_audio_ctx: case["n_audio_ctx"].as_u64().unwrap() as usize,
        };
        let prep = &case["prep"];
        let called = Cell::new(false);
        let got = fill_word_timings_batch(&slots, &tok, &opts, |batch| {
            called.set(true);
            assert!(!prep.is_null(), "case {k}: Python never ran the forward");
            let lens: Vec<usize> = prep["actual_lens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let caps: Vec<usize> = prep["audio_caps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(batch.actual_lens, lens, "case {k} actual_lens");
            assert_eq!(batch.audio_caps, caps, "case {k} audio_caps");
            for (bi, row) in case["full_tokens"].as_array().unwrap().iter().enumerate() {
                let want: Vec<u32> = row
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|t| t.as_u64().unwrap() as u32)
                    .collect();
                assert_eq!(batch.row(bi), &want[..], "case {k} full_tokens row {bi}");
            }

            let (shape, data) = decode_floats(&prep["matrices"]);
            Ok(Matrices {
                shape: [shape[0], shape[1], shape[2]],
                data,
            })
        })
        .unwrap();
        assert_eq!(
            called.get(),
            !prep.is_null(),
            "case {k}: forward ran in only one of Python and Rust"
        );
        assert_eq!(got, strings(&case["ret"]), "case {k}");
    }
}
