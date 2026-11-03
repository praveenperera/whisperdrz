//! Shared helpers for the DTW tests: the char-token table the Python tokenizer wrote
//! and decoders for the committed matrices

use std::collections::HashMap;
use std::path::PathBuf;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde_json::Value;
use whisperdrz_native::dtw::{DtwTokenizer, NegMatrix};

pub fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/dtw")
}

pub fn read_json(name: &str) -> Value {
    let path = data_dir().join(name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

/// `Tokenizer.encode(ch)` for every char the tests use, from `tools/gen_dtw_cases.py`
pub struct TableTokenizer {
    sot: Vec<u32>,
    no_ts: u32,
    eot: u32,
    chars: HashMap<String, Vec<u32>>,
}

impl TableTokenizer {
    pub fn load() -> Self {
        let v = read_json("tokenizer.json");
        let ids = |v: &Value| {
            v.as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        };
        let chars = v["chars"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, t)| (k.clone(), ids(t)))
            .collect();
        Self {
            sot: ids(&v["sot_sequence"]),
            no_ts: v["no_timestamps"].as_u64().unwrap() as u32,
            eot: v["eot"].as_u64().unwrap() as u32,
            chars,
        }
    }
}

impl DtwTokenizer for TableTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.chars
            .get(text)
            .unwrap_or_else(|| {
                panic!("{text:?} is not in tokenizer.json; rerun tools/gen_dtw_cases.py")
            })
            .clone()
    }

    fn eot(&self) -> u32 {
        self.eot
    }

    fn no_timestamps(&self) -> u32 {
        self.no_ts
    }

    fn sot_sequence(&self) -> &[u32] {
        &self.sot
    }
}

fn shape(v: &Value) -> Vec<usize> {
    v["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap() as usize)
        .collect()
}

/// A `{"shape", "b64"}` little-endian f32 array, or `{"shape", "f16"}` widened to f32
pub fn decode_floats(v: &Value) -> (Vec<usize>, Vec<f32>) {
    if let Some(b) = v.get("b64") {
        let bytes = STANDARD.decode(b.as_str().unwrap()).unwrap();
        return (
            shape(v),
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect(),
        );
    }

    let bytes = STANDARD.decode(v["f16"].as_str().unwrap()).unwrap();
    (
        shape(v),
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| half::f16::from_le_bytes(c).to_f32())
            .collect(),
    )
}

pub fn decode_matrix(v: &Value) -> NegMatrix {
    let (shape, data) = decode_floats(v);
    NegMatrix::new(shape[0], shape[1], data).unwrap()
}

/// Bit patterns, so `inf` and `-0.0` compare exactly
pub fn bits(data: &[f32]) -> Vec<u32> {
    data.iter().map(|v| v.to_bits()).collect()
}

pub fn opt_f64(v: &Value) -> Option<f64> {
    if v.is_null() {
        None
    } else {
        Some(v.as_f64().unwrap())
    }
}

pub fn opt_usize(v: &Value) -> Option<usize> {
    if v.is_null() {
        None
    } else {
        Some(v.as_u64().unwrap() as usize)
    }
}

pub fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_owned())
        .collect()
}

pub fn token_groups(v: &Value) -> Vec<Vec<u32>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|g| {
            g.as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_u64().unwrap() as u32)
                .collect()
        })
        .collect()
}
