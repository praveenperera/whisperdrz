//! `speaker_clustering.cluster_speakers`: centroid-linkage agglomerative clustering
//! of L2-normalized embeddings, in torch's f32 CPU arithmetic

use eyre::{Result, ensure};

use super::torch_cpu::{
    CdistPath, cdist, cdist_direct_pair, column_sum, mm_lhs, mm_pair, mm_rhs, normalize,
};

/// How close one merge decision came to going another way
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MergeStep {
    /// Active clusters when the distances were taken
    pub active: usize,
    /// Smallest distance (the merged pair, or the one that stopped clustering)
    pub min: f32,
    /// Smallest distance over every other pair, `inf` with two clusters left
    pub runner_up: f32,
    /// Whether `cdist` took the MKL matmul path (more than 25 active rows), where
    /// the model isn't bit-exact
    pub matmul: bool,
}

/// The labels plus the decision margins behind them
#[derive(Debug, Clone, PartialEq)]
pub struct Clustering {
    /// Global speaker id of each input embedding, in input order
    pub labels: Vec<i64>,
    pub steps: Vec<MergeStep>,
}

impl Clustering {
    /// Smallest gap between a step's winning distance and its runner-up or the
    /// threshold, over the steps that took the inexact matmul path
    pub fn matmul_margin(&self, threshold: f64) -> Option<f32> {
        let t = threshold as f32;
        self.steps
            .iter()
            .filter(|s| s.matmul)
            .map(|s| (s.runner_up - s.min).min((s.min - t).abs()))
            .min_by(f32::total_cmp)
    }
}

/// The merge loop: while more than one cluster is active, merge the closest pair of
/// centroids unless they are further apart than `threshold`.
///
/// Each step's `cdist` over the active centroids takes its arithmetic from the active
/// count alone ([`CdistPath`]), and each element from its two centroids, so the
/// distances stay in an `n x n` table: a merge recomputes the merged cluster's row
/// and column, and a change of path recomputes the whole table. Every element, and
/// so every decision, is the one a full `cdist` per step gives, in O(n^2 dim) total
/// instead of O(n^3 dim)
fn merge(
    clusters: &mut [usize],
    centroids: &mut [Vec<f32>],
    sizes: &mut [f32],
    active: &mut [bool],
    threshold: f32,
) -> Vec<MergeStep> {
    let n = centroids.len();
    let mut steps = Vec::new();
    let mut lhs: Vec<Vec<f32>> = centroids.iter().map(|c| mm_lhs(c)).collect();
    let mut rhs: Vec<Vec<f32>> = centroids.iter().map(|c| mm_rhs(c)).collect();
    let mut dists = vec![f32::INFINITY; n * n];
    let pair = |lhs: &[Vec<f32>], rhs: &[Vec<f32>], cs: &[Vec<f32>], a: usize, b: usize, path| {
        match path {
            CdistPath::Direct => cdist_direct_pair(&cs[a], &cs[b]),
            _ => mm_pair(&lhs[a], &rhs[b], path),
        }
    };
    let mut path = None;
    let mut dirty: Option<usize> = None;

    loop {
        let idx: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
        let r = idx.len();
        if r <= 1 {
            break;
        }

        let p = CdistPath::for_rows(r);
        if path != Some(p) {
            for &a in &idx {
                for &b in &idx {
                    dists[a * n + b] = pair(&lhs, &rhs, centroids, a, b, p);
                }
            }
            path = Some(p);
        } else if let Some(m) = dirty {
            for &b in &idx {
                dists[m * n + b] = pair(&lhs, &rhs, centroids, m, b, p);
                dists[b * n + m] = pair(&lhs, &rhs, centroids, b, m, p);
            }
        }

        // the `[r, r]` matrix over active clusters in index order, diagonal at inf
        let at = |pa: usize, pb: usize| {
            if pa == pb {
                f32::INFINITY
            } else {
                dists[idx[pa] * n + idx[pb]]
            }
        };
        // `dists.min()`, then the first row-major position holding it
        let mut min_val = f32::INFINITY;
        for pa in 0..r {
            for pb in 0..r {
                min_val = min_val.min(at(pa, pb));
            }
        }
        let pos = (0..r * r)
            .position(|q| at(q / r, q % r) == min_val)
            .unwrap_or(0);
        let (pr, pc) = (pos / r, pos % r);
        let merged_pair = (pr.min(pc), pr.max(pc));
        let mut runner_up = f32::INFINITY;
        for pa in 0..r {
            for pb in 0..r {
                if pa != pb && (pa.min(pb), pa.max(pb)) != merged_pair {
                    runner_up = runner_up.min(at(pa, pb));
                }
            }
        }
        steps.push(MergeStep {
            active: r,
            min: min_val,
            runner_up,
            matmul: r > 25,
        });

        if min_val > threshold {
            break;
        }

        let (mut i, mut j) = (idx[pr], idx[pc]);
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }

        let (size_i, size_j) = (sizes[i], sizes[j]);
        let total = size_i + size_j;
        let mut merged: Vec<f32> = centroids[i]
            .iter()
            .zip(&centroids[j])
            .map(|(&a, &b)| (a * size_i + b * size_j) / total)
            .collect();
        normalize(&mut merged);
        lhs[i] = mm_lhs(&merged);
        rhs[i] = mm_rhs(&merged);
        centroids[i] = merged;
        sizes[i] = total;
        dirty = Some(i);

        for c in clusters.iter_mut() {
            if *c == j {
                *c = i;
            }
        }

        active[j] = false;
    }

    steps
}

/// Renumbers labels to 0, 1, 2... in ascending order of the old label
/// (`unique()` sorts)
fn renumber(labels: &mut [usize]) {
    let mut uniq = labels.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    for l in labels.iter_mut() {
        *l = uniq.binary_search(l).unwrap_or(0);
    }
}

/// Row mean over `rows`, as `t[mask].mean(0)`: a sum down the rows, then a divide
fn mean_rows(emb: &[Vec<f32>], rows: &[usize]) -> Vec<f32> {
    let dim = emb[0].len();
    let mut out = vec![0f32; dim];
    for (d, o) in out.iter_mut().enumerate() {
        let col: Vec<f32> = rows.iter().map(|&r| emb[r][d]).collect();
        *o = column_sum(&col) / rows.len() as f32;
    }

    out
}

/// `cluster_speakers(embeddings, threshold, min_cluster_size)`, returning the label
/// of each embedding in input order. The Python returns a dict keyed by
/// `(segment, local_id)`; with distinct keys (as the remap passes) that dict is these
/// labels in input order
pub fn cluster_speakers(
    embeddings: &[Vec<f32>],
    threshold: f64,
    min_cluster_size: usize,
) -> Result<Clustering> {
    let n = embeddings.len();
    if n <= 1 {
        return Ok(Clustering {
            labels: vec![0; n],
            steps: Vec::new(),
        });
    }

    let dim = embeddings[0].len();
    ensure!(
        embeddings.iter().all(|e| e.len() == dim),
        "embeddings of different widths"
    );

    let mut emb: Vec<Vec<f32>> = embeddings.to_vec();
    for e in &mut emb {
        normalize(e);
    }

    let mut clusters: Vec<usize> = (0..n).collect();
    let mut centroids = emb.clone();
    let mut sizes = vec![1f32; n];
    let mut active = vec![true; n];
    // `min_val > threshold` compares an f32 tensor with a Python float, in f32
    let threshold_f32 = threshold as f32;
    let steps = merge(
        &mut clusters,
        &mut centroids,
        &mut sizes,
        &mut active,
        threshold_f32,
    );

    renumber(&mut clusters);

    // `min(min_cluster_size, max(1, round(0.1 * n)))`; at the default of 1 every
    // cluster is large enough and this whole block is skipped
    let min_size = min_cluster_size.min((0.1 * n as f64).round_ties_even().max(1.0) as usize);
    let n_labels = clusters.iter().max().map_or(0, |m| m + 1);
    let counts: Vec<usize> = (0..n_labels)
        .map(|l| clusters.iter().filter(|&&c| c == l).count())
        .collect();
    let large: Vec<usize> = (0..n_labels).filter(|&l| counts[l] >= min_size).collect();
    let small: Vec<usize> = (0..n_labels).filter(|&l| counts[l] < min_size).collect();

    if large.is_empty() {
        clusters.iter_mut().for_each(|c| *c = 0);
    } else if !small.is_empty() {
        let members =
            |cl: &[usize], l: usize| -> Vec<usize> { (0..n).filter(|&i| cl[i] == l).collect() };
        let mut large_centroids = Vec::new();
        for &l in &large {
            let mut c = mean_rows(&emb, &members(&clusters, l));
            normalize(&mut c);
            large_centroids.extend(c);
        }

        for &s in &small {
            let mut c = mean_rows(&emb, &members(&clusters, s));
            normalize(&mut c);
            let d = cdist(&c, &large_centroids, dim);
            let nearest = d
                .iter()
                .enumerate()
                .fold(
                    (0, f32::INFINITY),
                    |b, (k, &v)| if v < b.1 { (k, v) } else { b },
                )
                .0;
            let target = large[nearest];
            for c in clusters.iter_mut() {
                if *c == s {
                    *c = target;
                }
            }
        }

        renumber(&mut clusters);
    }

    Ok(Clustering {
        labels: clusters.into_iter().map(|c| c as i64).collect(),
        steps,
    })
}

#[cfg(test)]
mod tests {
    use super::super::pyrandom::PyRandom;
    use super::super::torch_cpu::{cdist, normalize};
    use super::{MergeStep, merge};

    /// The merge loop as it was, with a full `cdist` per step: the oracle the
    /// incremental table must match bit for bit
    fn merge_full(
        clusters: &mut [usize],
        centroids: &mut [Vec<f32>],
        sizes: &mut [f32],
        active: &mut [bool],
        threshold: f32,
    ) -> Vec<MergeStep> {
        let n = centroids.len();
        let dim = centroids[0].len();
        let mut steps = Vec::new();
        while active.iter().filter(|&&a| a).count() > 1 {
            let idx: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
            let r = idx.len();
            let flat: Vec<f32> = idx
                .iter()
                .flat_map(|&i| centroids[i].iter().copied())
                .collect();
            let mut dists = cdist(&flat, &flat, dim);
            for i in 0..r {
                dists[i * r + i] = f32::INFINITY;
            }

            let min_val = dists.iter().copied().fold(f32::INFINITY, f32::min);
            let pos = dists.iter().position(|&d| d == min_val).unwrap_or(0);
            let (pr, pc) = (pos / r, pos % r);
            let runner_up = (0..r * r)
                .filter(|&p| {
                    let (a, b) = (p / r, p % r);
                    a != b && (a.min(b), a.max(b)) != (pr.min(pc), pr.max(pc))
                })
                .map(|p| dists[p])
                .fold(f32::INFINITY, f32::min);
            steps.push(MergeStep {
                active: r,
                min: min_val,
                runner_up,
                matmul: r > 25,
            });
            if min_val > threshold {
                break;
            }

            let (mut i, mut j) = (idx[pr], idx[pc]);
            if i > j {
                std::mem::swap(&mut i, &mut j);
            }

            let (size_i, size_j) = (sizes[i], sizes[j]);
            let total = size_i + size_j;
            let mut merged: Vec<f32> = centroids[i]
                .iter()
                .zip(&centroids[j])
                .map(|(&a, &b)| (a * size_i + b * size_j) / total)
                .collect();
            normalize(&mut merged);
            centroids[i] = merged;
            sizes[i] = total;
            for c in clusters.iter_mut() {
                if *c == j {
                    *c = i;
                }
            }

            active[j] = false;
        }

        steps
    }

    /// [`merge`] and [`merge_full`]: clusters, centroids, sizes, active, threshold
    type MergeFn =
        fn(&mut [usize], &mut [Vec<f32>], &mut [f32], &mut [bool], f32) -> Vec<MergeStep>;

    /// Unit vectors around `speakers` random centers, ECAPA2's width
    fn embeddings(rng: &mut PyRandom, n: usize, speakers: usize, spread: f64) -> Vec<Vec<f32>> {
        let centers: Vec<Vec<f64>> = (0..speakers)
            .map(|_| (0..192).map(|_| rng.uniform(-1.0, 1.0)).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centers[i * 7 % speakers];
                let mut v: Vec<f32> = c
                    .iter()
                    .map(|&x| (x + rng.uniform(-spread, spread)) as f32)
                    .collect();
                normalize(&mut v);
                v
            })
            .collect()
    }

    /// Same labels, same steps (bit for bit) and same final centroids as the full
    /// recompute, across every path change (25/26, 29/30, 64/65 active clusters) and
    /// thresholds that stop early or merge everything
    #[test]
    fn incremental_merge_matches_full_cdist() {
        let mut rng = PyRandom::new(7);
        let mut cases = 0;
        for n in [2, 3, 25, 26, 27, 30, 31, 64, 65, 66, 90, 140, 253] {
            for (speakers, spread) in [(1, 0.05), (4, 0.4), (9, 1.0), (n, 0.0)] {
                let emb = embeddings(&mut rng, n, speakers.max(1), spread);
                for threshold in [0.3f32, 1.1, 2.5] {
                    let run = |f: MergeFn| {
                        let mut clusters: Vec<usize> = (0..n).collect();
                        let mut centroids = emb.clone();
                        let mut sizes = vec![1f32; n];
                        let mut active = vec![true; n];
                        let steps = f(
                            &mut clusters,
                            &mut centroids,
                            &mut sizes,
                            &mut active,
                            threshold,
                        );
                        let bits: Vec<Vec<u32>> = centroids
                            .iter()
                            .map(|c| c.iter().map(|v| v.to_bits()).collect())
                            .collect();
                        let step_bits: Vec<(usize, u32, u32, bool)> = steps
                            .iter()
                            .map(|s| (s.active, s.min.to_bits(), s.runner_up.to_bits(), s.matmul))
                            .collect();
                        (clusters, step_bits, bits, active)
                    };
                    let full = run(merge_full);
                    let fast = run(merge);
                    assert!(
                        full == fast,
                        "n {n} speakers {speakers} spread {spread} threshold {threshold}: incremental merge differs"
                    );
                    assert!(!full.1.is_empty());
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 13 * 4 * 3);
    }
}
