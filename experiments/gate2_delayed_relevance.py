"""Gate 2: delayed relevance needs fast trace before slow state.

This gate is motivated by a precise limitation of explicit-state runtimes:
an observation may become important only after the moment when a compact
persistent state had to decide what to retain.

Each case arrives as 16 unrelated scalar fields.  The field that will matter
is not revealed until 20-80 steps later.  Much later (220-420 steps after the
case began) the same fact is queried again.

Compared systems:
- full_history: keep every raw field forever;
- early_state: one persistent scalar chosen before relevance is known;
- oracle_state: one persistent scalar chosen with future knowledge;
- fast_only: bounded raw episodic buffer, no consolidation;
- hybrid: bounded raw buffer, then consolidate exactly the requested scalar.

This is an information/memory gate, not a biological hippocampus model.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np


N_CASES = 5_000
N_FIELDS = 16
FIRST_DELAY = (20, 80)
REPEAT_DELAY = (220, 420)
BUFFER_HORIZON = 100
SEEDS = 10


def run_seed(seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)

    records: dict[int, tuple[int, np.ndarray, int, int, int]] = {}
    queries: dict[int, list[tuple[int, int, int, str]]] = defaultdict(list)

    for case_id in range(N_CASES):
        t0 = case_id
        values = rng.integers(0, 2**31 - 1, size=N_FIELDS, dtype=np.int64)
        relevant_field = int(rng.integers(N_FIELDS))
        first_t = t0 + int(rng.integers(FIRST_DELAY[0], FIRST_DELAY[1] + 1))
        repeat_t = t0 + int(rng.integers(REPEAT_DELAY[0], REPEAT_DELAY[1] + 1))

        records[case_id] = (t0, values, relevant_field, first_t, repeat_t)
        target = int(values[relevant_field])
        queries[first_t].append((case_id, relevant_field, target, "first"))
        queries[repeat_t].append((case_id, relevant_field, target, "repeat"))

    names = ("full_history", "early_state", "oracle_state", "fast_only", "hybrid")
    score = {
        name: {"first": 0, "repeat": 0, "first_n": 0, "repeat_n": 0}
        for name in names
    }
    peak_scalars = {name: 0 for name in names}

    full_history: dict[int, np.ndarray] = {}
    early_state: dict[int, tuple[int, int]] = {}
    oracle_state: dict[int, tuple[int, int]] = {}
    fast_only: dict[int, tuple[int, np.ndarray]] = {}
    hybrid_buffer: dict[int, tuple[int, np.ndarray]] = {}
    hybrid_slow: dict[int, tuple[int, int]] = {}

    max_t = N_CASES + REPEAT_DELAY[1] + 5

    def mark(name: str, query_type: str, answer: int | None, target: int) -> None:
        score[name][query_type] += int(answer == target)
        score[name][query_type + "_n"] += 1

    for t in range(max_t):
        # Bounded fast traces forget raw episodes after BUFFER_HORIZON.
        for buffer in (fast_only, hybrid_buffer):
            expired = [
                case_id
                for case_id, (t0, _) in buffer.items()
                if t - t0 > BUFFER_HORIZON
            ]
            for case_id in expired:
                del buffer[case_id]

        # One new case per step.
        if t < N_CASES:
            case_id = t
            t0, values, relevant_field, _, _ = records[case_id]

            full_history[case_id] = values.copy()

            # Compact state gets exactly one scalar per case.  This arm must
            # choose before it knows which field future consequence will need.
            chosen_field = int(rng.integers(N_FIELDS))
            early_state[case_id] = (chosen_field, int(values[chosen_field]))

            # Positive control: future relevance known at write time.
            oracle_state[case_id] = (
                relevant_field,
                int(values[relevant_field]),
            )

            fast_only[case_id] = (t0, values.copy())
            hybrid_buffer[case_id] = (t0, values.copy())

        for case_id, field, target, query_type in queries.get(t, []):
            mark(
                "full_history",
                query_type,
                int(full_history[case_id][field]),
                target,
            )

            stored = early_state.get(case_id)
            early_answer = stored[1] if stored is not None and stored[0] == field else None
            mark("early_state", query_type, early_answer, target)

            stored = oracle_state.get(case_id)
            oracle_answer = stored[1] if stored is not None and stored[0] == field else None
            mark("oracle_state", query_type, oracle_answer, target)

            episode = fast_only.get(case_id)
            fast_answer = int(episode[1][field]) if episode is not None else None
            mark("fast_only", query_type, fast_answer, target)

            episode = hybrid_buffer.get(case_id)
            if episode is not None:
                hybrid_answer = int(episode[1][field])
                # Relevance has now been revealed.  Only this scalar is
                # consolidated into persistent state.
                hybrid_slow[case_id] = (field, hybrid_answer)
            else:
                stored = hybrid_slow.get(case_id)
                hybrid_answer = (
                    stored[1]
                    if stored is not None and stored[0] == field
                    else None
                )
            mark("hybrid", query_type, hybrid_answer, target)

        peak_scalars["full_history"] = max(
            peak_scalars["full_history"], len(full_history) * N_FIELDS
        )
        peak_scalars["early_state"] = max(
            peak_scalars["early_state"], len(early_state)
        )
        peak_scalars["oracle_state"] = max(
            peak_scalars["oracle_state"], len(oracle_state)
        )
        peak_scalars["fast_only"] = max(
            peak_scalars["fast_only"], len(fast_only) * N_FIELDS
        )
        peak_scalars["hybrid"] = max(
            peak_scalars["hybrid"],
            len(hybrid_buffer) * N_FIELDS + len(hybrid_slow),
        )

    result: dict[str, dict[str, float]] = {}
    for name in names:
        result[name] = {
            "first_accuracy": score[name]["first"] / score[name]["first_n"],
            "repeat_accuracy": score[name]["repeat"] / score[name]["repeat_n"],
            "peak_stored_scalars": float(peak_scalars[name]),
        }
    return result


def aggregate(per_seed: list[dict[str, dict[str, float]]]) -> dict:
    names = per_seed[0].keys()
    out = {}
    for name in names:
        out[name] = {}
        for metric in per_seed[0][name].keys():
            values = np.array([seed[name][metric] for seed in per_seed], dtype=float)
            out[name][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
    return out


def main() -> None:
    per_seed = [run_seed(seed) for seed in range(SEEDS)]
    summary = aggregate(per_seed)

    receipt = {
        "gate": 2,
        "n_cases": N_CASES,
        "n_fields": N_FIELDS,
        "first_delay": list(FIRST_DELAY),
        "repeat_delay": list(REPEAT_DELAY),
        "buffer_horizon": BUFFER_HORIZON,
        "seeds": SEEDS,
        "summary": summary,
        "per_seed": per_seed,
    }

    print("Gate 2 — delayed relevance")
    for name, values in summary.items():
        f = values["first_accuracy"]
        r = values["repeat_accuracy"]
        m = values["peak_stored_scalars"]
        print(
            f"{name:14s} "
            f"first {f['mean']:.4f} ± {f['std']:.4f}  "
            f"repeat {r['mean']:.4f} ± {r['std']:.4f}  "
            f"peak scalars {m['mean']:.1f}"
        )

    out = Path(__file__).resolve().parents[1] / "results" / "gate2_delayed_relevance.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
