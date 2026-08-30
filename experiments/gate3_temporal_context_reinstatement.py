"""Gate 3: content-addressed temporal-context reinstatement.

Synthetic structural analogue of hippocampal-guided cortical reinstatement.

At E1 every episode contains:
    context_before, cue, context_after

The context vectors are random and have no learnable population-level mapping
from cue -> context.  At E2 the same cue is re-encountered with noise.

A slow cue-only decoder therefore cannot reconstruct the old temporal
neighbourhood on unseen episodes.

A fast episodic key-value bank can:
    key   = E1 cue representation
    value = normalized average of E1-1 / E1+1 context

At E2, attention over episodic keys retrieves/reinstates the old context.

This is deliberately ordinary content-addressed memory.  The experiment asks
what architectural role it has, not whether it is novel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


N_EPISODES = 1_000
CUE_DIM = 64
CONTEXT_DIM = 32
CUE_NOISE = 0.20
BETA = 30.0
RIDGE = 1.0
SEEDS = 10


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run_seed(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    cues = normalize_rows(rng.normal(size=(N_EPISODES, CUE_DIM)))
    context_before = normalize_rows(rng.normal(size=(N_EPISODES, CONTEXT_DIM)))
    context_after = normalize_rows(rng.normal(size=(N_EPISODES, CONTEXT_DIM)))

    temporal_context = normalize_rows(context_before + context_after)

    # E2: same event re-encountered with representational noise.
    e2_cues = normalize_rows(
        cues + CUE_NOISE * rng.normal(size=cues.shape)
    )

    train = np.arange(N_EPISODES // 2)
    test = np.arange(N_EPISODES // 2, N_EPISODES)

    # Slow cortical-style cue-only attacker:
    # learn a population mapping from cue -> surrounding context on other
    # episodes, then test on unseen episodes.
    x_train = np.c_[cues[train], np.ones(len(train))]
    gram = x_train.T @ x_train + RIDGE * np.eye(x_train.shape[1])
    w = np.linalg.solve(gram, x_train.T @ temporal_context[train])

    cue_only = np.c_[e2_cues[test], np.ones(len(test))] @ w
    cue_only = normalize_rows(cue_only)
    cue_only_cos = np.sum(cue_only * temporal_context[test], axis=1)

    # Fast episodic index.  This is mathematically just attention over an
    # explicit episodic key/value bank.
    similarity = e2_cues[test] @ cues.T
    attention = softmax_rows(BETA * similarity)

    reinstated = normalize_rows(attention @ temporal_context)
    episodic_cos = np.sum(
        reinstated * temporal_context[test], axis=1
    )

    retrieved_id = np.argmax(similarity, axis=1)
    top1 = float(np.mean(retrieved_id == test))

    # A scalar measure of how strongly the correct episodic index was engaged.
    correct_index_weight = attention[np.arange(len(test)), test]
    index_reinstatement_corr = float(
        np.corrcoef(correct_index_weight, episodic_cos)[0, 1]
    )

    # Kill the temporal association while preserving the same keys, queries,
    # and attention weights.
    shuffled_values = temporal_context[rng.permutation(N_EPISODES)]
    shuffled = normalize_rows(attention @ shuffled_values)
    shuffled_cos = np.sum(
        shuffled * temporal_context[test], axis=1
    )

    # Kill cue/index correspondence itself.
    random_queries = normalize_rows(
        rng.normal(size=(len(test), CUE_DIM))
    )
    random_attention = softmax_rows(BETA * (random_queries @ cues.T))
    random_reinstated = normalize_rows(
        random_attention @ temporal_context
    )
    random_index_cos = np.sum(
        random_reinstated * temporal_context[test], axis=1
    )

    return {
        "cue_only_cosine": float(cue_only_cos.mean()),
        "episodic_reinstatement_cosine": float(episodic_cos.mean()),
        "shuffled_temporal_link_cosine": float(shuffled_cos.mean()),
        "random_index_cosine": float(random_index_cos.mean()),
        "episodic_top1_identity": top1,
        "correct_index_weight": float(correct_index_weight.mean()),
        "index_weight_vs_reinstatement_corr": index_reinstatement_corr,
    }


def aggregate(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for key in per_seed[0]:
        values = np.array([row[key] for row in per_seed])
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return out


def main() -> None:
    per_seed = [run_seed(seed) for seed in range(SEEDS)]
    summary = aggregate(per_seed)

    print("Gate 3 — temporal context reinstatement")
    for key, value in summary.items():
        print(
            f"{key:38s} "
            f"{value['mean']:.4f} ± {value['std']:.4f}"
        )

    receipt = {
        "gate": 3,
        "n_episodes": N_EPISODES,
        "cue_dim": CUE_DIM,
        "context_dim": CONTEXT_DIM,
        "cue_noise": CUE_NOISE,
        "beta": BETA,
        "ridge": RIDGE,
        "seeds": SEEDS,
        "summary": summary,
        "per_seed": per_seed,
    }

    out = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "gate3_temporal_context_reinstatement.json"
    )
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
