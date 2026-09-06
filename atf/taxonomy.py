"""Domain taxonomy + heuristic bucketing of FFN expert chunks.

IMPORTANT CAVEAT: without labeled activation data (running the original
model over a calibration corpus and recording which neurons fire for which
kind of input) there is no way to know what an expert chunk is "actually"
specialized in. What this module does instead is a *statistical* k-means
bucketing over simple weight statistics, purely so each expert gets a
stable, distinct (domain_id, sub_domain_id) pair for the manifest / future
knowledge index. Treat DOMAIN_NAMES as storage slots, not semantic labels —
if real semantic routing matters later, this needs to be replaced with
calibration-data-driven clustering (e.g. cluster on activation patterns
over a real text corpus, not on the raw weights).
"""
from __future__ import annotations

import numpy as np

DOMAIN_NAMES = [
    "general", "language", "structure", "numeric",
    "long-range", "rare-token", "syntax", "domain-7",
]
SUB_DOMAIN_NAMES = ["core", "peripheral"]


def chunk_stats(w: np.ndarray) -> np.ndarray:
    """Cheap per-chunk feature vector used only to bucket chunks (no
    semantic meaning implied)."""
    row_norm = np.linalg.norm(w, axis=1)
    return np.array([
        float(np.mean(np.abs(w))),
        float(np.std(w)),
        float(np.mean(row_norm)),
        float(np.std(row_norm)),
        float(np.mean(w > 0)),
    ], dtype=np.float64)


def kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, min(k, n))
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for i in range(iters):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = d.argmin(axis=1)
        if i > 0 and np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(axis=0)
    return labels


def classify_from_stats(stats: list[np.ndarray], n_domains: int | None = None) -> list[tuple[int, int]]:
    """Like classify_experts but takes precomputed chunk_stats vectors
    (so the caller never has to keep the weight chunks alive)."""
    if n_domains is None:
        n_domains = len(DOMAIN_NAMES)
    feats = np.stack(stats)
    std = feats.std(0)
    std = np.where(std == 0, 1.0, std)
    feats = (feats - feats.mean(0)) / std
    domain_ids = kmeans(feats, n_domains)
    sub_ids = np.zeros_like(domain_ids)
    for d in np.unique(domain_ids):
        idx = np.where(domain_ids == d)[0]
        if len(idx) < 2:
            continue
        vals = np.array([stats[i][2] for i in idx])
        median = np.median(vals)
        for i, v in zip(idx, vals):
            sub_ids[i] = 0 if v >= median else 1
    return list(zip(domain_ids.tolist(), sub_ids.tolist()))


def classify_experts(chunks: list[np.ndarray], n_domains: int | None = None) -> list[tuple[int, int]]:
    """Returns [(domain_id, sub_domain_id), ...] aligned with `chunks`."""
    if n_domains is None:
        n_domains = len(DOMAIN_NAMES)
    feats = np.stack([chunk_stats(c) for c in chunks])
    std = feats.std(0)
    std = np.where(std == 0, 1.0, std)
    feats = (feats - feats.mean(0)) / std
    domain_ids = kmeans(feats, n_domains)
    sub_ids = np.zeros_like(domain_ids)
    for d in np.unique(domain_ids):
        idx = np.where(domain_ids == d)[0]
        if len(idx) < 2:
            continue
        vals = np.array([chunk_stats(chunks[i])[2] for i in idx])
        median = np.median(vals)
        for i, v in zip(idx, vals):
            sub_ids[i] = 0 if v >= median else 1
    return list(zip(domain_ids.tolist(), sub_ids.tolist()))
