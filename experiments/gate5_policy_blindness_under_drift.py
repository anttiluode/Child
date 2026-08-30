"""Gate 5: a successful memory policy can become blind under drift.

Gate 4 learned a slow-memory write value from the delayed relevance outcomes
of every candidate event.  That is a useful full-feedback upper bound, but it
does not model a harder case: after an event is discarded, the system may no
longer have an address/feature trace with which to interpret later feedback.

This gate makes feedback selective.  Five observable event groups occur in
equal numbers.  Slow memory can retain one group's worth of events (20/100).
Before an unannounced change, group 0 is most useful.  In one changed world,
group 0 becomes less useful while previously poor group 1 becomes best.  In a
harder second world, group 0's outcome distribution remains exactly unchanged
while group 1 improves off-policy.  A purely greedy gate can therefore remain
satisfied with group 0 forever and never observe evidence from group 1.

Compared policies:

- frozen: retain according to the old value model;
- greedy_selected: adapt, but only from retained-event feedback;
- fixed_reserve: spend four of twenty writes on random exploration forever;
- surprise_burst: normally greedy, but temporarily spend half the write budget
  on exploration when retained outcomes are much worse than predicted;
- fast_trace: keep discarded event identities for one feedback delay, allowing
  the value model to learn from all outcomes at the cost of 80 extra temporary
  trace slots;
- random, oracle, and full-memory controls.

The estimators are deliberately transparent exponentially weighted group means.
The result is an observability/resource-allocation receipt, not a novel
continual-learning algorithm or a model of biological memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


N_GROUPS = 5
EVENTS_PER_GROUP = 20
BLOCK_SIZE = N_GROUPS * EVENTS_PER_GROUP
MEMORY_BUDGET = 20

WARMUP_BLOCKS = 40
PRE_SHIFT_BLOCKS = 160
POST_SHIFT_BLOCKS = 240
POST_EARLY_BLOCKS = 25
POST_LATE_BLOCKS = 80
FEEDBACK_DELAY_BLOCKS = 1

OLD_PROBABILITY = np.array([0.75, 0.12, 0.10, 0.08, 0.06])
NEW_PROBABILITY_VISIBLE = np.array([0.45, 0.90, 0.10, 0.08, 0.06])
NEW_PROBABILITY_HIDDEN = np.array([0.75, 0.90, 0.10, 0.08, 0.06])
SHIFT_SCENARIOS = {
    "visible_drop": NEW_PROBABILITY_VISIBLE,
    "strictly_hidden": NEW_PROBABILITY_HIDDEN,
}

VALUE_LEARNING_RATE = 0.05
FIXED_RESERVE_SLOTS = 4
BURST_RESERVE_SLOTS = 12
BURST_LENGTH = 20
SURPRISE_GAP = 0.15
SURPRISE_PATIENCE = 4

RECOVERY_WINDOW = 8
RECOVERY_NEW_GROUP_SHARE = 0.75
SEEDS = 40

POLICIES = (
    "frozen",
    "greedy_selected",
    "fixed_reserve",
    "surprise_burst",
    "fast_trace",
    "random",
    "oracle",
    "full",
)


def update_values(
    values: np.ndarray,
    groups: np.ndarray,
    outcomes: np.ndarray,
) -> None:
    """Update group values with a per-observation exponential step.

    Combining n observations uses 1 - (1-lr)^n, so a group observed twenty
    times in a block receives more evidence than a group observed once.
    """

    for group in range(N_GROUPS):
        mask = groups == group
        count = int(mask.sum())
        if count == 0:
            continue
        step = 1.0 - (1.0 - VALUE_LEARNING_RATE) ** count
        target = float(outcomes[mask].mean())
        values[group] += step * (target - values[group])


def top_k(
    scores: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return k high-score indices with seeded random tie-breaking."""

    jitter = rng.uniform(0.0, 1e-9, size=len(scores))
    return np.argsort(scores + jitter)[-k:]


def select_with_reserve(
    scores: np.ndarray,
    reserve_slots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Select exploit slots plus random reserve slots under one fixed budget."""

    exploit_slots = MEMORY_BUDGET - reserve_slots
    exploit = top_k(scores, exploit_slots, rng)
    available = np.setdiff1d(
        np.arange(BLOCK_SIZE), exploit, assume_unique=False
    )
    reserve = (
        rng.choice(available, reserve_slots, replace=False)
        if reserve_slots
        else np.empty(0, dtype=int)
    )
    selected = np.concatenate((exploit, reserve))
    return selected, exploit


@dataclass
class PendingFeedback:
    groups: np.ndarray
    outcomes: np.ndarray
    selected: np.ndarray
    exploit: np.ndarray
    phase: str


def make_world(
    seed: int,
    new_probability: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Create one shared non-stationary stream for every compared policy."""

    rng = np.random.default_rng(seed + 20_000)
    groups_per_block: list[np.ndarray] = []
    outcomes_per_block: list[np.ndarray] = []

    for block in range(PRE_SHIFT_BLOCKS + POST_SHIFT_BLOCKS):
        groups = np.repeat(np.arange(N_GROUPS), EVENTS_PER_GROUP)
        rng.shuffle(groups)
        probabilities = (
            OLD_PROBABILITY if block < PRE_SHIFT_BLOCKS else new_probability
        )
        outcomes = rng.random(BLOCK_SIZE) < probabilities[groups]
        groups_per_block.append(groups)
        outcomes_per_block.append(outcomes)

    return groups_per_block, outcomes_per_block


def warmup_values(seed: int) -> np.ndarray:
    """Learn the old relevance law with full feedback before memory is scarce."""

    rng = np.random.default_rng(seed + 10_000)
    values = np.full(N_GROUPS, 0.5, dtype=np.float64)
    for _ in range(WARMUP_BLOCKS):
        groups = np.repeat(np.arange(N_GROUPS), EVENTS_PER_GROUP)
        rng.shuffle(groups)
        outcomes = rng.random(BLOCK_SIZE) < OLD_PROBABILITY[groups]
        update_values(values, groups, outcomes)
    return values


def recovery_time(new_group_share: list[float]) -> int | None:
    """First post-shift window allocating at least 75% to the new group."""

    post = np.asarray(new_group_share[PRE_SHIFT_BLOCKS:], dtype=float)
    if len(post) < RECOVERY_WINDOW:
        return None
    kernel = np.ones(RECOVERY_WINDOW) / RECOVERY_WINDOW
    rolling = np.convolve(post, kernel, mode="valid")
    hits = np.flatnonzero(rolling >= RECOVERY_NEW_GROUP_SHARE)
    # Report the end of the first qualifying window as a one-based number of
    # post-shift blocks consumed, not the zero-based window start.
    return int(hits[0] + RECOVERY_WINDOW) if len(hits) else None


def run_policy(
    seed: int,
    policy: str,
    initial_values: np.ndarray,
    groups_per_block: list[np.ndarray],
    outcomes_per_block: list[np.ndarray],
    new_probability: np.ndarray,
) -> dict[str, float | int | None]:
    rng = np.random.default_rng(seed * 1_009 + POLICIES.index(policy) * 97 + 7)
    values = initial_values.copy()
    detection_reference = initial_values.copy()
    pending: PendingFeedback | None = None
    burst_remaining = 0
    surprise_streak = 0
    false_bursts = 0

    recall: list[float] = []
    new_group_share: list[float] = []
    exploration_writes: list[int] = []

    for block, (groups, outcomes) in enumerate(
        zip(groups_per_block, outcomes_per_block)
    ):
        # Feedback from the previous block is now available.  Selected-memory
        # policies have addresses only for retained events.  fast_trace paid to
        # keep the other 80 identities alive for this interval.
        if pending is not None:
            if policy == "surprise_burst" and len(pending.exploit):
                expected = float(
                    detection_reference[pending.groups[pending.exploit]].mean()
                )
                observed = float(pending.outcomes[pending.exploit].mean())
                surprise_streak = (
                    surprise_streak + 1
                    if expected - observed >= SURPRISE_GAP
                    else 0
                )
                if (
                    surprise_streak >= SURPRISE_PATIENCE
                    and burst_remaining == 0
                ):
                    burst_remaining = BURST_LENGTH
                    false_bursts += int(pending.phase == "old")
                    surprise_streak = 0
                    detection_reference = values.copy()

            if policy == "fast_trace":
                update_values(values, pending.groups, pending.outcomes)
            elif policy in (
                "greedy_selected",
                "fixed_reserve",
                "surprise_burst",
            ):
                update_values(
                    values,
                    pending.groups[pending.selected],
                    pending.outcomes[pending.selected],
                )

        probabilities = (
            OLD_PROBABILITY if block < PRE_SHIFT_BLOCKS else new_probability
        )
        predicted = values[groups]

        if policy == "full":
            selected = np.arange(BLOCK_SIZE)
            exploit = selected
            reserve_slots = 0
        elif policy == "random":
            selected = rng.choice(
                BLOCK_SIZE, MEMORY_BUDGET, replace=False
            )
            exploit = np.empty(0, dtype=int)
            reserve_slots = MEMORY_BUDGET
        elif policy == "oracle":
            selected = top_k(probabilities[groups], MEMORY_BUDGET, rng)
            exploit = selected
            reserve_slots = 0
        elif policy == "fixed_reserve":
            reserve_slots = FIXED_RESERVE_SLOTS
            selected, exploit = select_with_reserve(
                predicted, reserve_slots, rng
            )
        elif policy == "surprise_burst":
            reserve_slots = BURST_RESERVE_SLOTS if burst_remaining > 0 else 0
            selected, exploit = select_with_reserve(
                predicted, reserve_slots, rng
            )
            if burst_remaining > 0:
                burst_remaining -= 1
        else:
            reserve_slots = 0
            selected, exploit = select_with_reserve(predicted, 0, rng)

        total_relevant = int(outcomes.sum())
        retained_relevant = int(outcomes[selected].sum())
        recall.append(retained_relevant / max(total_relevant, 1))
        new_group_share.append(float(np.mean(groups[selected] == 1)))
        exploration_writes.append(reserve_slots)

        phase = "old" if block < PRE_SHIFT_BLOCKS else "new"
        pending = PendingFeedback(
            groups=groups,
            outcomes=outcomes,
            selected=selected,
            exploit=exploit,
            phase=phase,
        )

    recovery = recovery_time(new_group_share)
    post_start = PRE_SHIFT_BLOCKS
    post_early_end = post_start + POST_EARLY_BLOCKS

    return {
        "pre_shift_recall": float(np.mean(recall[:post_start])),
        "post_early_recall": float(
            np.mean(recall[post_start:post_early_end])
        ),
        "post_late_recall": float(np.mean(recall[-POST_LATE_BLOCKS:])),
        "post_all_recall": float(np.mean(recall[post_start:])),
        "post_late_new_group_share": float(
            np.mean(new_group_share[-POST_LATE_BLOCKS:])
        ),
        "recovery_blocks": recovery,
        "recovered": int(recovery is not None),
        "exploration_writes_per_block": float(np.mean(exploration_writes)),
        "false_pre_shift_bursts": false_bursts,
        "persistent_slots": BLOCK_SIZE if policy == "full" else MEMORY_BUDGET,
        "extra_temporary_trace_slots": 80 if policy == "fast_trace" else 0,
    }


def run_seed(
    seed: int,
    scenario: str = "visible_drop",
) -> dict[str, dict[str, float | int | None]]:
    if scenario not in SHIFT_SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    new_probability = SHIFT_SCENARIOS[scenario]
    groups, outcomes = make_world(seed, new_probability)
    initial = warmup_values(seed)
    return {
        policy: run_policy(
            seed,
            policy,
            initial,
            groups,
            outcomes,
            new_probability,
        )
        for policy in POLICIES
    }


def aggregate(
    rows: list[dict[str, dict[str, float | int | None]]]
) -> dict[str, dict[str, dict[str, float | None]]]:
    summary: dict[str, dict[str, dict[str, float | None]]] = {}
    for policy in POLICIES:
        policy_summary: dict[str, dict[str, float | None]] = {}
        keys = rows[0][policy].keys()
        for key in keys:
            values = [row[policy][key] for row in rows]
            present = np.asarray([v for v in values if v is not None], dtype=float)
            if key == "recovery_blocks":
                policy_summary[key] = {
                    "mean_successful": (
                        float(present.mean()) if len(present) else None
                    ),
                    "std_successful": (
                        float(present.std()) if len(present) else None
                    ),
                }
            else:
                policy_summary[key] = {
                    "mean": float(present.mean()),
                    "std": float(present.std()),
                }
        summary[policy] = policy_summary
    return summary


def main() -> None:
    summary = {
        scenario: aggregate(
            [run_seed(seed, scenario) for seed in range(SEEDS)]
        )
        for scenario in SHIFT_SCENARIOS
    }

    print("Gate 5 — policy blindness under drift")
    for scenario in SHIFT_SCENARIOS:
        print(f"\n{scenario}")
        for policy in POLICIES:
            pre = summary[scenario][policy]["pre_shift_recall"]
            early = summary[scenario][policy]["post_early_recall"]
            late = summary[scenario][policy]["post_late_recall"]
            recovered = summary[scenario][policy]["recovered"]
            recovery = summary[scenario][policy]["recovery_blocks"]
            recovery_mean = recovery["mean_successful"]
            recovery_text = (
                "-" if recovery_mean is None else f"{recovery_mean:.1f}"
            )
            print(
                f"{policy:18s} "
                f"pre {pre['mean']:.4f}  "
                f"post-early {early['mean']:.4f}  "
                f"post-late {late['mean']:.4f}  "
                f"recovered {recovered['mean']:.2f}  "
                f"blocks {recovery_text}"
            )

    receipt = {
        "gate": 5,
        "question": "Can selective memory observe that its own policy became wrong, including when change occurs only off-policy?",
        "groups": N_GROUPS,
        "events_per_group": EVENTS_PER_GROUP,
        "block_size": BLOCK_SIZE,
        "memory_budget": MEMORY_BUDGET,
        "warmup_blocks": WARMUP_BLOCKS,
        "pre_shift_blocks": PRE_SHIFT_BLOCKS,
        "post_shift_blocks": POST_SHIFT_BLOCKS,
        "feedback_delay_blocks": FEEDBACK_DELAY_BLOCKS,
        "old_relevance_probability": OLD_PROBABILITY.tolist(),
        "new_relevance_probability": {
            name: value.tolist()
            for name, value in SHIFT_SCENARIOS.items()
        },
        "value_learning_rate": VALUE_LEARNING_RATE,
        "fixed_reserve_slots": FIXED_RESERVE_SLOTS,
        "burst_reserve_slots": BURST_RESERVE_SLOTS,
        "burst_length": BURST_LENGTH,
        "surprise_gap": SURPRISE_GAP,
        "surprise_patience": SURPRISE_PATIENCE,
        "recovery_window": RECOVERY_WINDOW,
        "recovery_new_group_share": RECOVERY_NEW_GROUP_SHARE,
        "seeds": SEEDS,
        "summary": summary,
    }

    out = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "gate5_policy_blindness_under_drift.json"
    )
    out.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
