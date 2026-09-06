"""Speculative-decoding accept/reject primitives (gamma/v3.1, Phase 1).

Per SPEC_DECODING_PROPOSAL.md Phase 1 ("minimal correctness test,
single-engine"): pure functions only, no Engine changes, no mx.array,
no I/O. These implement the standard speculative-decoding accept /
residual-sample algorithm (Leviathan et al. 2023, "Fast Inference from
Transformers via Speculative Decoding", section 3.2) that
SPEC_DECODING_PROPOSAL.md Phase 2 (Engine._spec_step / shallow
self-speculative drafting) will call once the engine-side plumbing
lands. See that file's §3/§3a for the full decode-loop design this
module is one piece of, and §5 for why Phase 1 is pure-function-only:
every later phase is gated on this one having a green, deterministic
test suite first.

Do NOT add Engine, mx.array, or GDN/KV-cache state here -- that is
explicitly Phase 2+ scope.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "accept_token", "residual_sample", "spec_decode_step",
    "SPEC_PADDING_VOCAB_START", "clamp_padding",
]


# ── Padding-row guard (Phase 0 finding) ────────────────────────────────
# The .atf LM head has vocab_size=248320 (a multiple of 64 for hardware
# alignment) but the Qwen/Qwen3.5-9B tokenizer only knows 248070 entries.
# Rows [248070, 248319] are zero-initialized padding that the tokenizer
# panics on if you try to decode them. The drafter (shallow self-spec)
# is the most likely path to put non-negligible mass on a padding id
# because shallow-layer logits are noisier than full-stack ones.
#
# Constants -- these mirror docs/spec_vocab_check_findings.md. The
# single source of truth is the .atf header's vocab_size and the
# Qwen/Qwen3.5-9B tokenizer's vocab_size; both are fixed for this
# model family so we hard-code them here rather than threading
# through the Engine (which would force a Phase 2 API change for
# one bit of defensive code).
SPEC_PADDING_VOCAB_START = 248070    # first padding-row id
SPEC_VALID_VOCAB_SIZE = 248070       # the only ids the tokenizer knows


def clamp_padding(logits) -> np.ndarray:
    """Mask the LM-head padding rows to -inf so the sampler can never
    pick them. Pure function: takes any array-like of shape
    [..., vocab] (where vocab >= SPEC_PADDING_VOCAB_START), returns a
    new float64 array with entries in [SPEC_PADDING_VOCAB_START, ...]
    set to -inf. Safe to call on logits from a drafter forward that
    has no business producing padding ids.

    If the input vocab is smaller than SPEC_PADDING_VOCAB_START, the
    array is returned unchanged (defensive -- smaller vocabs are
    technically possible on Qwen2-derivative models and would just
    mean no padding to mask).
    """
    arr = np.asarray(logits, dtype=np.float64)
    if arr.shape[-1] > SPEC_PADDING_VOCAB_START:
        # Build a mask once, broadcast over leading dims
        flat = arr.reshape(-1, arr.shape[-1])
        flat[:, SPEC_PADDING_VOCAB_START:] = -np.inf
        return flat.reshape(arr.shape)
    return arr


def _softmax(logits) -> np.ndarray:
    """Numerically-stable softmax over the last axis, float64 internally
    (accept/reject ratios are sensitive to precision near p_draft -> 0;
    keeping this module's own float64 avoids inheriting whatever
    dtype the caller's logits happen to be in, e.g. mx.array bf16)."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def accept_token(
    draft_token: int,
    draft_logits,
    target_logits,
    rand: float,
) -> bool:
    """Standard speculative-decoding accept test.

    Accept the drafted token iff ``rand < min(1, p_target(tok) /
    p_draft(tok))``. ``rand`` is an injected uniform-[0,1) draw rather
    than sampled internally, so callers (and tests) are deterministic
    and reproducible -- the same discipline
    tests/test_gdn_scan_parity.py-style tests in this codebase use for
    numeric parity.

    Args:
        draft_token: the token id the draft model sampled.
        draft_logits: 1D array-like of draft-model logits, shape [vocab].
        target_logits: 1D array-like of target-model logits at the same
            position, shape [vocab].
        rand: a uniform-[0,1) draw, injected by the caller.

    Returns:
        True if the draft token should be accepted as-is.
    """
    p_draft = _softmax(draft_logits)[draft_token]
    p_target = _softmax(target_logits)[draft_token]
    if p_draft <= 0.0:
        # Only reachable via float underflow (draft logits assigning
        # ~0 probability to a token it nonetheless sampled, e.g. from
        # a non-greedy / temperature-adjusted draft sampler upstream).
        # Treat as reject rather than divide-by-zero.
        return False
    accept_prob = min(1.0, float(p_target / p_draft))
    return rand < accept_prob


def residual_sample(
    draft_logits,
    target_logits,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample the rejection "correction" token from the residual
    distribution norm(max(0, p_target - p_draft)) (Leviathan et al.
    2023 section 3.2, the distribution guaranteeing the combined
    accept+residual process samples exactly from p_target).

    Args:
        draft_logits: 1D array-like, shape [vocab].
        target_logits: 1D array-like, shape [vocab].
        rng: injected numpy Generator for deterministic tests. If
            omitted, a fresh (non-deterministic) default_rng() is used.

    Returns:
        A sampled token id.
    """
    p_draft = _softmax(draft_logits)
    p_target = _softmax(target_logits)
    residual = np.maximum(0.0, p_target - p_draft)
    total = residual.sum()
    if total <= 0.0:
        # Degenerate case: p_draft dominates p_target everywhere (the
        # draft was "too confident" at every token). No positive mass
        # to correct into -- fall back to sampling target directly,
        # which is what the combined process converges to anyway.
        residual = p_target
        total = residual.sum()
    residual = residual / total
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(len(residual), p=residual))


def spec_decode_step(
    draft_tokens,
    draft_logits,
    target_logits,
    rand_values,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], int]:
    """Run the accept/reject loop over K drafted tokens (one draft-verify
    cycle), matching SPEC_DECODING_PROPOSAL.md section 3's pseudocode.

    Args:
        draft_tokens: list of K drafted token ids.
        draft_logits: array-like, shape [K, vocab] -- the draft model's
            logits at each drafted position.
        target_logits: array-like, shape [K, vocab] -- the target
            model's logits from ONE batched verification forward pass
            over the K drafted positions.
        rand_values: list of K uniform-[0,1) draws, one per position,
            injected for determinism.
        rng: injected numpy Generator, used only if a residual sample
            is needed (i.e. on the first rejection). Omit for
            non-deterministic residual sampling.

    Returns:
        (accepted_tokens, num_drafts_accepted):
          - accepted_tokens: the committed token ids for this cycle.
            Length == K if every draft was accepted (no residual token
            appended -- matches the proposal's pseudocode, which does
            not include the "bonus token" from position K; that is
            Phase 2 engine-wiring scope, since it needs an extra
            target-logits row past the last draft).
            Length == num_drafts_accepted + 1 if a draft was rejected
            partway through: the accepted prefix plus one
            residual-sampled correction token.
          - num_drafts_accepted: how many of the K drafts were
            accepted before the first rejection (0..K).
    """
    draft_logits = np.asarray(draft_logits, dtype=np.float64)
    target_logits = np.asarray(target_logits, dtype=np.float64)
    k = len(draft_tokens)
    assert draft_logits.shape[0] == k, (
        f"draft_logits has {draft_logits.shape[0]} rows, expected {k} "
        f"(one per drafted token)"
    )
    assert target_logits.shape[0] == k, (
        f"target_logits has {target_logits.shape[0]} rows, expected {k} "
        f"(one verification row per drafted position)"
    )
    assert len(rand_values) == k, (
        f"rand_values has {len(rand_values)} entries, expected {k}"
    )

    accepted: list[int] = []
    for i in range(k):
        if accept_token(draft_tokens[i], draft_logits[i], target_logits[i],
                         rand_values[i]):
            accepted.append(draft_tokens[i])
            continue
        correction = residual_sample(draft_logits[i], target_logits[i], rng=rng)
        accepted.append(correction)
        return accepted, i

    return accepted, k
