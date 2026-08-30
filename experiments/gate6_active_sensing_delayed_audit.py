"""Gate 6: learn when to sense under delayed external audit.

Prediction failure can mean that the model is wrong, that the present
observation is insufficient, or that an action should change the world.  This
gate stops treating those as one operation.

Each trial has a hidden binary target and one of two visible contexts.  A free
binary cue has context-dependent reliability.  A second probe is more reliable
but costs utility.  The two contexts exchange reliabilities without warning.
After acting, correctness is revealed only after a delay.

An online controller learns cue reliabilities and buys the probe only when its
estimated value exceeds the observation cost.  To associate a delayed audit
with the context, cues, probe decision, and action that produced it, the
controller must keep a short trial trace.  A no-trace learner cannot condition
the delayed audit and remains at its conservative always-probe initialization.

This is a transparent contextual value-of-information toy.  It is not a
general active-perception algorithm and learning itself is still automatic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


PRE_SHIFT_TRIALS = 6_000
POST_SHIFT_TRIALS = 6_000
TOTAL_TRIALS = PRE_SHIFT_TRIALS + POST_SHIFT_TRIALS
AUDIT_DELAY = 12
PROBE_COST = 0.08
PROBE_RELIABILITY = 0.90
BASE_RELIABILITY_OLD = np.array([0.90, 0.55])
BASE_RELIABILITY_NEW = np.array([0.55, 0.90])
INITIAL_BASE_RELIABILITY = 0.50
INITIAL_PROBE_RELIABILITY = 0.75
RELIABILITY_LEARNING_RATE = 0.04
LATE_WINDOW = 2_000
EARLY_POST_WINDOW = 500
RECOVERY_WINDOW = 200
RECOVERY_HIGH_SENSE = 0.80
RECOVERY_LOW_SENSE = 0.20
TRACE_SCALARS_PER_TRIAL = 5
SEEDS = 40

POLICIES = (
    "no_sense",
    "always_sense",
    "oracle",
    "learned_no_trace",
    "learned_trace",
    "zero_delay_learner",
)


@dataclass(frozen=True)
class World:
    context: np.ndarray
    target: np.ndarray
    base_cue: np.ndarray
    probe_cue: np.ndarray


@dataclass(frozen=True)
class TrialTrace:
    context: int
    base_cue: int
    probed: bool
    probe_cue: int
    action: int


class ReliabilityLearner:
    """Contextual cue calibration updated from an externally scored action."""

    def __init__(self) -> None:
        self.base = np.full(2, INITIAL_BASE_RELIABILITY, dtype=float)
        self.probe = INITIAL_PROBE_RELIABILITY

    def should_probe(self, context: int) -> bool:
        value = max(self.base[context], self.probe) - self.base[context]
        return bool(value > PROBE_COST)

    def choose(self, context: int, base_cue: int, probe_cue: int) -> int:
        if self.probe >= self.base[context]:
            return probe_cue
        return base_cue

    def update(self, trace: TrialTrace, action_correct: bool) -> None:
        # In this binary task, action plus an external correctness audit reveals
        # the target without the learner receiving a privileged target label.
        target = trace.action if action_correct else 1 - trace.action
        base_correct = float(trace.base_cue == target)
        c = trace.context
        self.base[c] += RELIABILITY_LEARNING_RATE * (
            base_correct - self.base[c]
        )
        if trace.probed:
            probe_correct = float(trace.probe_cue == target)
            self.probe += RELIABILITY_LEARNING_RATE * (
                probe_correct - self.probe
            )


def make_world(seed: int) -> World:
    """Create one shared stream, including unused counterfactual probe cues."""

    rng = np.random.default_rng(seed + 60_000)
    context = rng.integers(0, 2, size=TOTAL_TRIALS)
    target = rng.integers(0, 2, size=TOTAL_TRIALS)
    reliability = np.where(
        np.arange(TOTAL_TRIALS)[:, None] < PRE_SHIFT_TRIALS,
        BASE_RELIABILITY_OLD,
        BASE_RELIABILITY_NEW,
    )
    p_base = reliability[np.arange(TOTAL_TRIALS), context]
    base_correct = rng.random(TOTAL_TRIALS) < p_base
    probe_correct = rng.random(TOTAL_TRIALS) < PROBE_RELIABILITY
    base_cue = np.where(base_correct, target, 1 - target)
    probe_cue = np.where(probe_correct, target, 1 - target)
    return World(context, target, base_cue, probe_cue)


def recovery_time(context: np.ndarray, sensed: np.ndarray) -> int | None:
    """First completed post-shift window with the new sensing allocation."""

    post_context = context[PRE_SHIFT_TRIALS:]
    post_sensed = sensed[PRE_SHIFT_TRIALS:]
    for end in range(RECOVERY_WINDOW, len(post_context) + 1):
        start = end - RECOVERY_WINDOW
        c = post_context[start:end]
        s = post_sensed[start:end]
        rate_0 = float(s[c == 0].mean())
        rate_1 = float(s[c == 1].mean())
        if rate_0 >= RECOVERY_HIGH_SENSE and rate_1 <= RECOVERY_LOW_SENSE:
            return end
    return None


def mean_where(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean())


def run_policy(world: World, policy: str) -> dict[str, float | int | None]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")

    learner = (
        ReliabilityLearner()
        if policy.startswith("learned") or policy == "zero_delay_learner"
        else None
    )
    pending: deque[TrialTrace] = deque()
    correct = np.zeros(TOTAL_TRIALS, dtype=float)
    sensed = np.zeros(TOTAL_TRIALS, dtype=bool)
    peak_trace_records = 0

    for trial in range(TOTAL_TRIALS):
        if policy == "learned_trace" and len(pending) == AUDIT_DELAY:
            old = pending.popleft()
            old_trial = trial - AUDIT_DELAY
            audit = bool(old.action == world.target[old_trial])
            assert learner is not None
            learner.update(old, audit)

        context = int(world.context[trial])
        base_cue = int(world.base_cue[trial])
        probe_cue = int(world.probe_cue[trial])

        if policy == "no_sense":
            probe = False
        elif policy == "always_sense":
            probe = True
        elif policy == "oracle":
            true_base = (
                BASE_RELIABILITY_OLD
                if trial < PRE_SHIFT_TRIALS
                else BASE_RELIABILITY_NEW
            )
            probe = bool(PROBE_RELIABILITY - true_base[context] > PROBE_COST)
        else:
            assert learner is not None
            probe = learner.should_probe(context)

        if not probe:
            action = base_cue
        elif learner is not None:
            action = learner.choose(context, base_cue, probe_cue)
        else:
            action = probe_cue

        trace = TrialTrace(context, base_cue, probe, probe_cue, action)
        correct[trial] = float(action == world.target[trial])
        sensed[trial] = probe

        if policy == "learned_trace":
            pending.append(trace)
            peak_trace_records = max(peak_trace_records, len(pending))
        elif policy == "zero_delay_learner":
            assert learner is not None
            learner.update(trace, bool(correct[trial]))
        # learned_no_trace receives the delayed scalar audit but has no context,
        # cue, probe, or action address with which to interpret it.

    utility = correct - PROBE_COST * sensed
    pre = slice(PRE_SHIFT_TRIALS - LATE_WINDOW, PRE_SHIFT_TRIALS)
    early = slice(PRE_SHIFT_TRIALS, PRE_SHIFT_TRIALS + EARLY_POST_WINDOW)
    late = slice(TOTAL_TRIALS - LATE_WINDOW, TOTAL_TRIALS)
    late_context = world.context[late]
    late_sensed = sensed[late]
    recovery = recovery_time(world.context, sensed)

    return {
        "pre_late_accuracy": float(correct[pre].mean()),
        "pre_late_sense_rate": float(sensed[pre].mean()),
        "pre_late_utility": float(utility[pre].mean()),
        "post_early_utility": float(utility[early].mean()),
        "post_late_accuracy": float(correct[late].mean()),
        "post_late_sense_rate": float(sensed[late].mean()),
        "post_late_utility": float(utility[late].mean()),
        "post_late_sense_context_0": mean_where(
            late_sensed, late_context == 0
        ),
        "post_late_sense_context_1": mean_where(
            late_sensed, late_context == 1
        ),
        "recovery_trials": recovery,
        "recovered": int(recovery is not None),
        "peak_trace_records": peak_trace_records,
        "peak_trace_scalars": peak_trace_records * TRACE_SCALARS_PER_TRIAL,
        "final_base_reliability_0": (
            float(learner.base[0]) if learner is not None else None
        ),
        "final_base_reliability_1": (
            float(learner.base[1]) if learner is not None else None
        ),
        "final_probe_reliability": (
            float(learner.probe) if learner is not None else None
        ),
    }


def run_seed(seed: int) -> dict[str, dict[str, float | int | None]]:
    world = make_world(seed)
    return {policy: run_policy(world, policy) for policy in POLICIES}


def aggregate(
    rows: list[dict[str, dict[str, float | int | None]]]
) -> dict[str, dict[str, dict[str, float | None]]]:
    summary: dict[str, dict[str, dict[str, float | None]]] = {}
    for policy in POLICIES:
        policy_summary: dict[str, dict[str, float | None]] = {}
        for key in rows[0][policy]:
            values = [row[policy][key] for row in rows]
            present = np.asarray([v for v in values if v is not None], dtype=float)
            if key == "recovery_trials":
                policy_summary[key] = {
                    "mean_successful": (
                        float(present.mean()) if len(present) else None
                    ),
                    "std_successful": (
                        float(present.std()) if len(present) else None
                    ),
                }
            elif len(present):
                policy_summary[key] = {
                    "mean": float(present.mean()),
                    "std": float(present.std()),
                }
            else:
                policy_summary[key] = {"mean": None, "std": None}
        summary[policy] = policy_summary
    return summary


def main() -> None:
    summary = aggregate([run_seed(seed) for seed in range(SEEDS)])
    print("Gate 6 — active sensing under delayed audit")
    for policy in POLICIES:
        row = summary[policy]
        recovery = row["recovery_trials"]["mean_successful"]
        recovery_text = "-" if recovery is None else f"{recovery:.1f}"
        print(
            f"{policy:20s} "
            f"pre utility {row['pre_late_utility']['mean']:.4f}  "
            f"post-early {row['post_early_utility']['mean']:.4f}  "
            f"post-late {row['post_late_utility']['mean']:.4f}  "
            f"sense {row['post_late_sense_rate']['mean']:.4f}  "
            f"recovery {recovery_text}"
        )

    receipt = {
        "gate": 6,
        "question": (
            "Can a controller learn when to buy another observation when cue "
            "reliability changes and correctness is delayed?"
        ),
        "pre_shift_trials": PRE_SHIFT_TRIALS,
        "post_shift_trials": POST_SHIFT_TRIALS,
        "audit_delay_trials": AUDIT_DELAY,
        "probe_cost": PROBE_COST,
        "probe_reliability": PROBE_RELIABILITY,
        "base_reliability_old": BASE_RELIABILITY_OLD.tolist(),
        "base_reliability_new": BASE_RELIABILITY_NEW.tolist(),
        "reliability_learning_rate": RELIABILITY_LEARNING_RATE,
        "trace_scalars_per_trial": TRACE_SCALARS_PER_TRIAL,
        "seeds": SEEDS,
        "summary": summary,
    }
    out = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "gate6_active_sensing_delayed_audit.json"
    )
    out.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
