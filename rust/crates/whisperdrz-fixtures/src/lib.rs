//! Readers for the reference traces written by `rust/tools/trace_reference.py`
//!
//! A trace set (`short3`, `e21c600`) holds one directory per audio file with
//! `events.jsonl` (one recorded call per line, ordered by `seq`) and `result.json`
//! (final text and segments), plus a shared `blobs/` store of `.npy` arrays keyed by
//! content hash. Values are JSON with a few tagged objects for what JSON lacks:
//!
//! - `{"__a": dtype, "shape": [...], "data": [...]}` an inline array, or with
//!   `"blob": sha20` an array in `blobs/<sha20>.npy` (`"saved": true` when written)
//! - `{"__audio": [start, len]}` a slice of the file's own 16 kHz audio
//! - `{"__tuple": [...]}`, `{"__set": [...]}` (CPython iteration order) and
//!   `{"__items": [[k, v], ...]}` for dicts with non-string keys
//! - `{"__f": "nan" | "inf" | "-inf"}` a non-finite float
//! - `{"__state": {...}}` an `InferenceState` snapshot (plain fields only)
//! - `{"__type": name}` an object the tracer doesn't serialize (models, tokenizers)
//!
//! Tests find the traces through `WHISPERDRZ_FIXTURES`, defaulting to
//! `~/.cache/whisperdrz/fixtures/ref-a783291`, and skip with a message when they are
//! missing so `cargo test` passes on a machine without them

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use eyre::{Result, WrapErr, bail, ensure, eyre};
use serde_json::{Map, Value};

/// Root of the trace sets
pub fn root() -> PathBuf {
    if let Some(dir) = std::env::var_os("WHISPERDRZ_FIXTURES") {
        return PathBuf::from(dir);
    }

    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    home.join(".cache/whisperdrz/fixtures/ref-a783291")
}

/// One trace set, such as `short3`
pub struct TraceSet(PathBuf);

impl TraceSet {
    /// Opens a set under [`root`], or `None` (with a note on stderr) when it isn't there
    pub fn open(name: &str) -> Option<Self> {
        let dir = root().join(name);
        if dir.join("summary.json").exists() {
            return Some(Self(dir));
        }

        eprintln!("skipping: no trace set at {} (set WHISPERDRZ_FIXTURES)", dir.display());
        None
    }

    pub fn dir(&self) -> &Path {
        &self.0
    }

    /// File stems in trace order
    pub fn stems(&self) -> Result<Vec<String>> {
        let summary = read_json(&self.0.join("summary.json"))?;
        let files = summary["files"].as_array().ok_or_else(|| eyre!("summary.json has no files"))?;
        files
            .iter()
            .map(|f| f["stem"].as_str().map(str::to_owned).ok_or_else(|| eyre!("file without stem")))
            .collect()
    }

    /// Every recorded call for one file, in `seq` order
    pub fn events(&self, stem: &str) -> Result<Vec<Event>> {
        let path = self.0.join(stem).join("events.jsonl");
        let reader = BufReader::new(File::open(&path).wrap_err_with(|| format!("open {}", path.display()))?);
        let mut out = Vec::new();
        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            let value: Value =
                serde_json::from_str(&line).wrap_err_with(|| format!("{}:{}", path.display(), i + 1))?;
            out.push(Event::from_value(value)?);
        }

        out.sort_by_key(|e| e.seq);
        Ok(out)
    }

    /// Calls recorded under one label, such as `decode_batch` or `clean.tags.remove_orphan_paired_tags`
    pub fn calls(&self, stem: &str, label: &str) -> Result<Vec<Event>> {
        Ok(self.events(stem)?.into_iter().filter(|e| e.ev == label).collect())
    }

    /// `result.json`: final `text`, `segments`, `sha256`, `samples` and `raw_chunks`
    pub fn result(&self, stem: &str) -> Result<Value> {
        read_json(&self.0.join(stem).join("result.json"))
    }

    /// The array a tagged `{"__a": ...}` value holds, inline or from the blob store
    pub fn array(&self, value: &Value) -> Result<Array> {
        let obj = value.as_object().ok_or_else(|| eyre!("not an array value: {value}"))?;
        let dtype = obj.get("__a").and_then(Value::as_str).ok_or_else(|| eyre!("not an array value: {value}"))?;
        let shape = obj["shape"]
            .as_array()
            .ok_or_else(|| eyre!("array without shape"))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize).ok_or_else(|| eyre!("bad dim")))
            .collect::<Result<Vec<_>>>()?;

        if let Some(data) = obj.get("data") {
            let data = data.as_array().ok_or_else(|| eyre!("array data is not a list"))?;
            return Ok(Array { dtype: dtype.to_owned(), shape, data: ArrayData::from_json(dtype, data)? });
        }

        let blob = obj.get("blob").and_then(Value::as_str).ok_or_else(|| eyre!("array without data or blob"))?;
        let path = self.blob_path(blob);
        ensure!(path.exists(), "blob {blob} was not saved (trace run without --save-blobs?)");
        let array = read_npy(&path)?;
        ensure!(array.shape == shape, "blob {blob} shape {:?} != recorded {:?}", array.shape, shape);
        Ok(array)
    }

    pub fn blob_path(&self, sha: &str) -> PathBuf {
        self.0.parent().unwrap_or(&self.0).join("blobs").join(format!("{sha}.npy"))
    }
}

/// One recorded call
#[derive(Debug, Clone)]
pub struct Event {
    pub seq: u64,
    /// Label: the function name, `module.function` for module-level text helpers
    pub ev: String,
    /// Nesting depth among recorded calls on the same thread
    pub depth: u64,
    /// Arguments by parameter name
    pub args: Map<String, Value>,
    /// Mutable arguments (states, lists, dicts, arrays) re-encoded after the call
    pub after: Option<Map<String, Value>>,
    pub ret: Option<Value>,
    /// Set when the call raised, as `Type: message`
    pub raise: Option<String>,
}

impl Event {
    fn from_value(value: Value) -> Result<Self> {
        let Value::Object(mut obj) = value else { bail!("event is not an object") };
        let seq = obj.get("seq").and_then(Value::as_u64).ok_or_else(|| eyre!("event without seq"))?;
        let ev = obj.get("ev").and_then(Value::as_str).ok_or_else(|| eyre!("event without ev"))?.to_owned();
        let depth = obj.get("depth").and_then(Value::as_u64).unwrap_or(0);
        let args = match obj.remove("args") {
            Some(Value::Object(m)) => m,
            // random.* events log positional args as a list
            Some(Value::Array(a)) => Map::from_iter([("args".to_owned(), Value::Array(a))]),
            _ => Map::new(),
        };
        let after = match obj.remove("after") {
            Some(Value::Object(m)) => Some(m),
            _ => None,
        };
        let raise = obj.get("raise").and_then(Value::as_str).map(str::to_owned);
        Ok(Self { seq, ev, depth, args, after, ret: obj.remove("ret"), raise })
    }

    pub fn arg(&self, name: &str) -> Result<&Value> {
        self.args.get(name).ok_or_else(|| eyre!("{} #{} has no arg {name}", self.ev, self.seq))
    }

    pub fn after_arg(&self, name: &str) -> Result<&Value> {
        self.after
            .as_ref()
            .and_then(|a| a.get(name))
            .ok_or_else(|| eyre!("{} #{} has no after-value for {name}", self.ev, self.seq))
    }

    pub fn ret(&self) -> Result<&Value> {
        self.ret.as_ref().ok_or_else(|| eyre!("{} #{} has no return value", self.ev, self.seq))
    }
}

/// Strips the tracer's tags that plain values don't need: tuples and sets become
/// arrays, non-finite floats become their repr string. Arrays, states and audio refs
/// keep their tags
pub fn plain(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(plain).collect()),
        Value::Object(obj) => {
            for tag in ["__tuple", "__set"] {
                if let Some(Value::Array(items)) = obj.get(tag) {
                    return Value::Array(items.iter().map(plain).collect());
                }
            }

            if let Some(f) = obj.get("__f") {
                return f.clone();
            }

            Value::Object(obj.iter().map(|(k, v)| (k.clone(), plain(v))).collect())
        }
        other => other.clone(),
    }
}

/// A float from a JSON number or a tagged non-finite float
pub fn as_f64(value: &Value) -> Option<f64> {
    if let Some(f) = value.as_f64() {
        return Some(f);
    }

    match value.get("__f")?.as_str()? {
        "nan" => Some(f64::NAN),
        "inf" => Some(f64::INFINITY),
        "-inf" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

/// An array with its numpy dtype string (`float16`, `float32`, `int64`, `bool`, ...)
#[derive(Debug, Clone)]
pub struct Array {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: ArrayData,
}

#[derive(Debug, Clone)]
pub enum ArrayData {
    F16(Vec<half::f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

impl ArrayData {
    fn from_json(dtype: &str, data: &[Value]) -> Result<Self> {
        let f = |v: &Value| as_f64(v).ok_or_else(|| eyre!("bad float {v}"));
        let i = |v: &Value| v.as_i64().ok_or_else(|| eyre!("bad int {v}"));
        Ok(match dtype {
            "float16" => Self::F16(data.iter().map(|v| f(v).map(half::f16::from_f64)).collect::<Result<_>>()?),
            "float32" => Self::F32(data.iter().map(|v| f(v).map(|x| x as f32)).collect::<Result<_>>()?),
            "float64" => Self::F64(data.iter().map(f).collect::<Result<_>>()?),
            "int32" => Self::I32(data.iter().map(|v| i(v).map(|x| x as i32)).collect::<Result<_>>()?),
            "int64" => Self::I64(data.iter().map(i).collect::<Result<_>>()?),
            "bool" => Self::Bool(data.iter().map(|v| v.as_bool().ok_or_else(|| eyre!("bad bool"))).collect::<Result<_>>()?),
            other => bail!("unsupported dtype {other}"),
        })
    }
}

impl Array {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Values widened to f32 (f16, f32 and f64 arrays only)
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        Ok(match &self.data {
            ArrayData::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
            ArrayData::F32(v) => v.clone(),
            ArrayData::F64(v) => v.iter().map(|&x| x as f32).collect(),
            _ => bail!("{} is not a float array", self.dtype),
        })
    }

    /// Values as i64 (integer and bool arrays only)
    pub fn to_i64(&self) -> Result<Vec<i64>> {
        Ok(match &self.data {
            ArrayData::I32(v) => v.iter().map(|&x| x as i64).collect(),
            ArrayData::I64(v) => v.clone(),
            ArrayData::Bool(v) => v.iter().map(|&x| x as i64).collect(),
            _ => bail!("{} is not an integer array", self.dtype),
        })
    }
}

/// Reads a little-endian, C-order `.npy` file (format 1.0 or 2.0)
pub fn read_npy(path: &Path) -> Result<Array> {
    let mut bytes = Vec::new();
    File::open(path).wrap_err_with(|| format!("open {}", path.display()))?.read_to_end(&mut bytes)?;
    ensure!(bytes.len() > 10 && &bytes[..6] == b"\x93NUMPY", "{} is not .npy", path.display());

    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        v => bail!("npy version {v}"),
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len])?;
    ensure!(header.contains("'fortran_order': False"), "fortran-order npy");

    let descr = field(header, "'descr':")?.trim_matches(|c| c == '\'' || c == ' ');
    let shape_src = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| eyre!("npy header without shape"))?;
    let shape = shape_src
        .trim_start_matches([' ', '('])
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| eyre!("bad dim {s}: {e}")))
        .collect::<Result<Vec<_>>>()?;

    let body = &bytes[start + header_len..];
    let n: usize = shape.iter().product();
    let take = |width: usize| -> Result<&[u8]> {
        ensure!(body.len() == n * width, "npy body {} bytes, want {}", body.len(), n * width);
        Ok(body)
    };
    let (dtype, data) = match descr {
        "<f2" => ("float16", ArrayData::F16(take(2)?.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]])).collect())),
        "<f4" => ("float32", ArrayData::F32(take(4)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())),
        "<f8" => ("float64", ArrayData::F64(take(8)?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())),
        "<i4" => ("int32", ArrayData::I32(take(4)?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())),
        "<i8" => ("int64", ArrayData::I64(take(8)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())),
        "|b1" => ("bool", ArrayData::Bool(take(1)?.iter().map(|&b| b != 0).collect())),
        other => bail!("unsupported npy dtype {other}"),
    };
    Ok(Array { dtype: dtype.to_owned(), shape, data })
}

fn field<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    header
        .split(key)
        .nth(1)
        .and_then(|s| s.split(',').next())
        .ok_or_else(|| eyre!("npy header without {key}"))
}

fn read_json(path: &Path) -> Result<Value> {
    let file = File::open(path).wrap_err_with(|| format!("open {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file)).wrap_err_with(|| format!("parse {}", path.display()))
}
