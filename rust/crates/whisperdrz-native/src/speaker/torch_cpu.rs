//! The f32 arithmetic of the torch 2.8 CPU kernels `cluster_speakers` runs, in
//! their summation order, for the AVX2 build torch dispatches to on the reference
//! machine (Core Ultra 7 270K Plus: no AVX-512, so `Vectorized<float>` has 8 lanes)
//!
//! Matched bit for bit against torch on that CPU (`tests/data/torch_cpu.json`):
//! `F.normalize` (the `norm` last-dim fast path), `pow(2).sum(-1)` (cascade sum) and
//! `cdist` up to 25 rows (scalar loop). `cdist` past 25 rows goes through MKL
//! `sgemm`, whose blocking depends on the matrix size and its thread split; the
//! model here matches most elements and is off by an ulp or two on the rest, so
//! callers log how close each decision came (see [`super::cluster`])

/// AVX2 `Vectorized<float>` width
const LANES: usize = 8;

/// One `Vectorized<float>`
type Vec8 = [f32; LANES];

fn vadd(acc: &mut Vec8, v: &Vec8) {
    for (a, b) in acc.iter_mut().zip(v) {
        *a += b;
    }
}

/// `x.norm(2, dim=-1)` for one contiguous row: 8 lane accumulators of `v * v` (no
/// FMA), folded left to right, then the scalar tail, then `sqrt`
pub fn norm2(x: &[f32]) -> f32 {
    let (vecs, tail) = x.as_chunks::<LANES>();
    let mut acc = [0f32; LANES];
    for v in vecs {
        for (a, &e) in acc.iter_mut().zip(v) {
            *a += e * e;
        }
    }

    let mut total = acc[0];
    for &a in &acc[1..] {
        total += a;
    }

    for &e in tail {
        total += e * e;
    }

    total.sqrt()
}

/// `F.normalize(x, p=2, dim=-1)` for one row, in place: `x / max(norm, 1e-12)`
pub fn normalize(x: &mut [f32]) {
    let denom = norm2(x).max(1e-12);
    for v in x {
        *v /= denom;
    }
}

/// `x.sum(-1)` for one contiguous row: `cascade_sum`'s inner reduction. Four
/// interleaved 8-lane accumulators (`multi_row_sum` with `ilp_factor` 4) take the
/// vectors in groups of four, folding up a cascade level every 16 groups; leftover
/// vectors go to the first accumulator, the four are folded in order, and the
/// scalar tail is summed before the lanes are added left to right
pub fn sum(x: &[f32]) -> f32 {
    const ILP: usize = 4;
    const LEVELS: usize = 4;
    // `level_power` is 4 for any row under 2^16 groups
    const LEVEL_STEP: usize = 16;

    type Quad = [Vec8; ILP];
    let add_quad = |acc: &mut Quad, q: &Quad| {
        for (a, v) in acc.iter_mut().zip(q) {
            vadd(a, v);
        }
    };

    let (vecs, tail) = x.as_chunks::<LANES>();
    let (quads, rest) = vecs.as_chunks::<ILP>();
    let mut acc = [[[0f32; LANES]; ILP]; LEVELS];
    let mut i = 0;
    while i + LEVEL_STEP <= quads.len() {
        for q in &quads[i..i + LEVEL_STEP] {
            add_quad(&mut acc[0], q);
        }

        i += LEVEL_STEP;
        for j in 1..LEVELS {
            let lower = std::mem::take(&mut acc[j - 1]);
            add_quad(&mut acc[j], &lower);
            if i & (0xF << (j * 4)) != 0 {
                break;
            }
        }
    }

    for q in &quads[i..] {
        add_quad(&mut acc[0], q);
    }

    for j in 1..LEVELS {
        let upper = acc[j];
        add_quad(&mut acc[0], &upper);
    }

    let [mut p0, p1, p2, p3] = acc[0];
    for v in rest {
        vadd(&mut p0, v);
    }

    for p in [p1, p2, p3] {
        vadd(&mut p0, &p);
    }

    let mut total = 0f32;
    for &e in tail {
        total += e;
    }

    for p in p0 {
        total += p;
    }

    total
}

/// `torch.cdist(a, b)` (p = 2) between row sets of width `dim`, row-major `[ra, rb]`
pub fn cdist(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let (ra, rb) = (a.len() / dim, b.len() / dim);
    if ra > 25 || rb > 25 {
        euclidean_dist_mm(a, b, dim)
    } else {
        cdist_direct(a, b, dim)
    }
}

/// `_cdist_forward` on the CPU: per pair, a scalar sum of `|x - y|^2` (no FMA), then
/// `sqrt`
fn cdist_direct(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len() / dim * (b.len() / dim));
    for x in a.chunks_exact(dim) {
        for y in b.chunks_exact(dim) {
            let mut agg = 0f32;
            for (&p, &q) in x.iter().zip(y) {
                let d = (p - q).abs();
                agg += d * d;
            }

            out.push(agg.sqrt());
        }
    }

    out
}

/// `_euclidean_dist`: `[-2x, |x|^2, 1] @ [y, 1, |y|^2]^T`, clamped at 0, `sqrt`.
/// The product models MKL `sgemm` as FMA chains over k: one chain up to 29 rows,
/// two interleaved chains (even and odd k) summed above that. Exact for 28, 31 and
/// 33 rows in the probe; elsewhere a few elements per matrix differ by an ulp or two
fn euclidean_dist_mm(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let pack = |rows: &[f32], scale: f32, norm_last: bool| -> Vec<Vec<f32>> {
        rows.chunks_exact(dim)
            .map(|r| {
                let sq: Vec<f32> = r.iter().map(|v| v * v).collect();
                let norm = sum(&sq);
                let mut out: Vec<f32> = r.iter().map(|v| v * scale).collect();
                if norm_last {
                    out.extend([1.0, norm]);
                } else {
                    out.extend([norm, 1.0]);
                }
                out
            })
            .collect()
    };

    let xa = pack(a, -2.0, false);
    let yb = pack(b, 1.0, true);
    let chains = if xa.len().max(yb.len()) <= 29 { 1 } else { 2 };
    let mut out = Vec::with_capacity(xa.len() * yb.len());
    for x in &xa {
        for y in &yb {
            let mut acc = [0f32; 2];
            for (k, (&p, &q)) in x.iter().zip(y).enumerate() {
                let c = k % chains;
                acc[c] = p.mul_add(q, acc[c]);
            }

            let dot = if chains == 1 { acc[0] } else { acc[0] + acc[1] };
            out.push(dot.max(0.0).sqrt());
        }
    }

    out
}
