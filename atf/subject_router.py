"""Subject-Relevance LOD (SR-LOD) — Phase 1: subject scoring + domain ranking.

See docs/SUBJECT_RELEVANCE_LOD_DESIGN.md for the full design, the honest
risk assessment (taxonomy.py's domain buckets are k-means over raw weight
statistics, NOT semantically trained — see §4.3), and why this is Phase 1
of a multi-phase plan rather than something wired into the engine yet.

NOT WIRED INTO engine.py. Pure, unit-testable functions only — same shape
as atf/router.py's score_prompt/tier_for, which this intentionally mirrors.

Naming: this is SR-LOD, distinct from atf/router.py's reasoning-tier LOD
and atf/format.py's LodLevel storage-precision tiers. Do not rename these
to `lod_level` / `ATF_LOD_*` anywhere — keep `srlod_*` / `ATF_SRLOD_*` to
avoid colliding with the other two meanings of "LOD" already in this repo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# SRLOD_0 is the pinned/always-resident tier (dense weights + "general"
# domain). Domains 1..N_SRLOD_TIERS are ranked by relevance each turn.
# See docs/SUBJECT_RELEVANCE_LOD_DESIGN.md §2 for why 13 requires
# reconverting existing .atf files (taxonomy.py currently emits 8 domains,
# not 14) -- this module works with however many domains the loaded model
# actually has; it does not assume 14 until that reconversion happens.
SRLOD_GENERAL_DOMAIN_ID = 0


def embed_subject(recent_token_ids: list[int], token_embedding: np.ndarray) -> np.ndarray:
    """Option A from the design doc §4.2: mean-pool the model's own token
    embedding vectors over the recent conversation as a cheap subject
    vector. No extra model, no extra forward pass.

    Args:
        recent_token_ids: token ids from the recent conversation (caller
            decides the window -- e.g. last N tokens of the current prompt,
            or the whole running conversation; not decided by this function).
        token_embedding: the model's dense token_embd weight, shape
            [vocab_size, hidden_dim]. Caller supplies it (always resident
            per the design doc §3.1 -- this module never loads weights).

    Returns:
        A single [hidden_dim] vector (mean over the selected token rows).

    Known weakness (documented in the design doc, not hidden here): raw
    embedding mean-pooling has no attention/context mixing and will drift
    on long or topically-mixed conversations. This is a deliberate first
    cut, not a claim that it's a strong topic signal.
    """
    if not recent_token_ids:
        return np.zeros(token_embedding.shape[1], dtype=np.float32)
    rows = token_embedding[np.asarray(recent_token_ids, dtype=np.int64)]
    return rows.mean(axis=0).astype(np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rank_domains(
    subject_vec: np.ndarray,
    expert_centroids: dict[int, np.ndarray],
    exclude_domain_ids: frozenset[int] = frozenset({SRLOD_GENERAL_DOMAIN_ID}),
) -> list[tuple[int, float]]:
    """Rank domain ids by cosine similarity of their centroid to the
    current subject vector, most relevant first.

    Args:
        subject_vec: output of embed_subject (or any equivalent [D] vector).
        expert_centroids: domain_id -> centroid vector [D]. This is meant
            to come from KnowledgeIndex.expert_centroids (atf/format.py) --
            see docs/SUBJECT_RELEVANCE_LOD_DESIGN.md §4.1: whether convert.py
            actually populates this today is UNVERIFIED (Phase 0). If it
            comes back empty, this function has nothing to rank and callers
            must fall back to "everything resident" (the safe default),
            not silently rank against garbage.
        exclude_domain_ids: domains never ranked because they're pinned
            resident (SRLOD_0 / general) regardless of relevance.

    Returns:
        List of (domain_id, similarity) sorted descending by similarity.
    """
    scored = [
        (domain_id, _cosine_sim(subject_vec, centroid))
        for domain_id, centroid in expert_centroids.items()
        if domain_id not in exclude_domain_ids
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


@dataclass
class SrLodPlan:
    """Which domains should be resident for the upcoming turn.

    Produced once per turn (see design doc §5 -- never per-token), consumed
    by the not-yet-built ExpertPager (Phase 2).
    """
    resident_domain_ids: set[int]
    evicted_domain_ids: set[int]
    subject_vec: np.ndarray
    ranking: list[tuple[int, float]]  # full ranking, for logging/debugging


def plan_for_turn(
    subject_vec: np.ndarray,
    expert_centroids: dict[int, np.ndarray],
    all_domain_ids: set[int],
    k_resident: int,
    pinned_domain_ids: frozenset[int] = frozenset({SRLOD_GENERAL_DOMAIN_ID}),
) -> SrLodPlan:
    """Decide the resident set for one turn: pinned domains + the top
    k_resident ranked domains. Default k_resident should start high (see
    design doc §5 -- "most tiers resident" is the safe starting default,
    tightened only after §6 Phase 4 benchmarking shows it's safe).
    """
    ranking = rank_domains(subject_vec, expert_centroids, exclude_domain_ids=pinned_domain_ids)
    top = {domain_id for domain_id, _ in ranking[:k_resident]}
    resident = set(pinned_domain_ids) | top
    evicted = all_domain_ids - resident
    return SrLodPlan(
        resident_domain_ids=resident,
        evicted_domain_ids=evicted,
        subject_vec=subject_vec,
        ranking=ranking,
    )
