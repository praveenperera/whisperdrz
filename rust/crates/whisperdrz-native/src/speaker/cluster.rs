//! `speaker_clustering.cluster_speakers`: centroid-linkage agglomerative clustering
//! of L2-normalized embeddings, in torch's f32 CPU arithmetic

use eyre::{Result, ensure};

use super::torch_cpu::{cdist, column_sum, normalize};

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
pub fn cluster_speakers(embeddings: &[Vec<f32>], threshold: f64, min_cluster_size: usize) -> Result<Clustering> {
    let n = embeddings.len();
    if n <= 1 {
        return Ok(Clustering { labels: vec![0; n], steps: Vec::new() });
    }

    let dim = embeddings[0].len();
    ensure!(embeddings.iter().all(|e| e.len() == dim), "embeddings of different widths");

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
    let mut steps = Vec::new();

    while active.iter().filter(|&&a| a).count() > 1 {
        let idx: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
        let r = idx.len();
        let flat: Vec<f32> = idx.iter().flat_map(|&i| centroids[i].iter().copied()).collect();
        let mut dists = cdist(&flat, &flat, dim);
        for i in 0..r {
            dists[i * r + i] = f32::INFINITY;
        }

        // `dists.min()`, then the first row-major position holding it
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
        steps.push(MergeStep { active: r, min: min_val, runner_up, matmul: r > 25 });

        if min_val > threshold_f32 {
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

    renumber(&mut clusters);

    // `min(min_cluster_size, max(1, round(0.1 * n)))`; at the default of 1 every
    // cluster is large enough and this whole block is skipped
    let min_size = min_cluster_size.min((0.1 * n as f64).round_ties_even().max(1.0) as usize);
    let n_labels = clusters.iter().max().map_or(0, |m| m + 1);
    let counts: Vec<usize> = (0..n_labels).map(|l| clusters.iter().filter(|&&c| c == l).count()).collect();
    let large: Vec<usize> = (0..n_labels).filter(|&l| counts[l] >= min_size).collect();
    let small: Vec<usize> = (0..n_labels).filter(|&l| counts[l] < min_size).collect();

    if large.is_empty() {
        clusters.iter_mut().for_each(|c| *c = 0);
    } else if !small.is_empty() {
        let members = |cl: &[usize], l: usize| -> Vec<usize> { (0..n).filter(|&i| cl[i] == l).collect() };
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
            let nearest = d.iter().enumerate().fold((0, f32::INFINITY), |b, (k, &v)| if v < b.1 { (k, v) } else { b }).0;
            let target = large[nearest];
            for c in clusters.iter_mut() {
                if *c == s {
                    *c = target;
                }
            }
        }

        renumber(&mut clusters);
    }

    Ok(Clustering { labels: clusters.into_iter().map(|c| c as i64).collect(), steps })
}
