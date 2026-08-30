"""Gate 8: constitutional competition between executable laws.

This is the first integration gate in Child.  It asks whether a bounded
learner can discover a missing structural term without allowing the current
incumbent model to control all of the evidence on which it is judged.

The incumbent law is a known linear function

    f_0(x) = beta dot x.

The world contains one initially unavailable interaction

    f_*(x) = f_0(x) + s * gamma * x[a] * x[b].

An incumbent-controlled experiment policy chooses one-axis inputs.  Every
pair interaction is zero on those inputs, so the incumbent appears correct
even though it is wrong on dense inputs.  A separate coverage channel spends
one in four trials on theory-independent dense interventions.  Aggregate
residual energy can then say that the model family is wrong.

At the birth boundary, a mutation operator admits every signed pair
interaction as a new executable law.  Before that boundary the old linear
model had no reason to retain pair-specific statistics.  A bounded raw audit
ledger can replay representative pre-birth evidence against the newborn laws;
an early-compression control has to buy new experiments.

The roles are deliberately separated:

* laws predict and compete;
* the judge scores immutable delayed outcomes;
* the coverage channel cannot be disabled by the incumbent;
* the ledger preserves causal query/outcome addresses;
* the birth operator may expand the law family but cannot rewrite old scores;
* observationally equivalent winners return NOT_IDENTIFIABLE.

This is a finite quadratic toy with a supplied mutation grammar, known noise,
and an exact base law.  It is not an autonomous scientist or a general program
synthesis system.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np


DIM = 7
GAMMA = 0.80
NOISE_STD = 0.30
PRE_TRIALS = 400
AUDIT_DELAY = 12
COVERAGE_PERIOD = 4
TRACE_CAPACITY = 16
TRACE_CAPACITY_SWEEP = (0, 4, 6, 8, 10, 12, 16)
POSTERIOR_THRESHOLD = 0.95
MAX_POST_PROBES = 80
MISMATCH_MSE_THRESHOLD = 0.30
SEEDS = 40
SCALARS_PER_RAW_RECORD = DIM + 1

POLICIES = (
    "captured_incumbent",
    "fixed_catalog_coverage",
    "no_replay_random",
    "no_replay_active",
    "shuffled_address_trace",
    "constitutional_trace",
    "full_audit_ledger",
    "full_history",
)


@dataclass(frozen=True)
class Candidate:
    pair: tuple[int, int]
    sign: int

    @property
    def name(self) -> str:
        a, b = self.pair
        prefix = "+" if self.sign > 0 else "-"
        return f"{prefix}x{a}*x{b}"


@dataclass(frozen=True)
class World:
    beta: np.ndarray
    true_pair: tuple[int, int]
    true_sign: int
    alias_constraint: bool


@dataclass(frozen=True)
class Evidence:
    x: np.ndarray
    y: float
    coverage: bool


def candidates() -> list[Candidate]:
    """All one-step mutations available at the birth boundary."""

    return [
        Candidate(pair, sign)
        for pair in combinations(range(DIM), 2)
        for sign in (-1, 1)
    ]


def make_world(seed: int, alias_constraint: bool = False) -> World:
    rng = np.random.default_rng(seed + 80_000)
    beta = rng.normal(0.0, 0.35, size=DIM)
    if alias_constraint:
        # x[1] == x[2] makes x[0]x[1] and x[0]x[2] exactly equivalent.
        pair = (0, 1)
    else:
        pair = tuple(
            sorted(rng.choice(DIM, size=2, replace=False).tolist())
        )
    sign = int(rng.choice((-1, 1)))
    return World(beta, pair, sign, alias_constraint)


def obey_constraint(x: np.ndarray, alias_constraint: bool) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    if alias_constraint:
        x[2] = x[1]
    return x


def base_prediction(world: World, x: np.ndarray) -> float:
    return float(world.beta @ x)


def interaction_value(pair: tuple[int, int], x: np.ndarray) -> float:
    return float(x[pair[0]] * x[pair[1]])


def noiseless_world_prediction(world: World, x: np.ndarray) -> float:
    interaction = interaction_value(world.true_pair, x)
    return base_prediction(world, x) + world.true_sign * GAMMA * interaction


def observe(world: World, x: np.ndarray, rng: np.random.Generator) -> float:
    return float(noiseless_world_prediction(world, x) + rng.normal(0, NOISE_STD))


def easy_query(
    rng: np.random.Generator, alias_constraint: bool
) -> np.ndarray:
    """An incumbent-friendly input on which every interaction is zero."""

    x = np.zeros(DIM, dtype=float)
    # Avoid axes 1 and 2 in the alias world: enforcing x2=x1 would otherwise
    # turn a one-axis query into a two-axis query.
    choices = [0, 3, 4, 5, 6] if alias_constraint else list(range(DIM))
    x[int(rng.choice(choices))] = float(rng.choice((-1, 1)))
    return obey_constraint(x, alias_constraint)


def coverage_query(
    rng: np.random.Generator, alias_constraint: bool
) -> np.ndarray:
    """A theory-independent intervention with broad interaction support."""

    x = rng.choice((-1.0, 1.0), size=DIM)
    return obey_constraint(x, alias_constraint)


def collect_delayed_evidence(
    world: World,
    seed: int,
    mode: str,
) -> tuple[list[Evidence], int]:
    """Issue experiments, then deliver outcomes through an addressed delay."""

    if mode not in {"captured", "mixed", "coverage"}:
        raise ValueError(f"unknown evidence mode: {mode}")

    query_rng = np.random.default_rng(seed + 81_000)
    noise_rng = np.random.default_rng(seed + 82_000 + 97 * len(mode))
    pending: deque[tuple[int, Evidence]] = deque()
    delivered: list[Evidence] = []
    peak_pending = 0

    for trial in range(PRE_TRIALS + AUDIT_DELAY):
        while pending and pending[0][0] == trial:
            _, evidence = pending.popleft()
            delivered.append(evidence)

        if trial >= PRE_TRIALS:
            continue

        use_coverage = mode == "coverage" or (
            mode == "mixed" and trial % COVERAGE_PERIOD == 0
        )
        x = (
            coverage_query(query_rng, world.alias_constraint)
            if use_coverage
            else easy_query(query_rng, world.alias_constraint)
        )
        y = observe(world, x, noise_rng)
        pending.append(
            (trial + AUDIT_DELAY, Evidence(x=x, y=y, coverage=use_coverage))
        )
        peak_pending = max(peak_pending, len(pending))

    assert len(delivered) == PRE_TRIALS
    return delivered, peak_pending


def residual(world: World, evidence: Evidence) -> float:
    return float(evidence.y - base_prediction(world, evidence.x))


def mismatch_mse(world: World, evidence: list[Evidence]) -> float:
    coverage = [row for row in evidence if row.coverage]
    selected = coverage if coverage else evidence
    values = np.asarray([residual(world, row) for row in selected])
    return float(np.mean(values**2))


def bounded_audit_trace(
    evidence: list[Evidence], seed: int, capacity: int = TRACE_CAPACITY
) -> list[Evidence]:
    """Keep an online theory-independent reservoir of the audit channel."""

    if capacity < 0:
        raise ValueError("trace capacity must be nonnegative")
    rng = np.random.default_rng(seed + 83_000)
    trace: list[Evidence] = []
    coverage_seen = 0
    for row in evidence:
        if not row.coverage:
            continue
        coverage_seen += 1
        if len(trace) < capacity:
            trace.append(row)
            continue
        replacement = int(rng.integers(coverage_seen))
        if replacement < capacity:
            trace[replacement] = row
    return trace


def shuffle_causal_addresses(
    world: World, evidence: list[Evidence], seed: int
) -> list[Evidence]:
    """Keep residual values but destroy which query produced each residual."""

    rng = np.random.default_rng(seed + 84_000)
    order = rng.permutation(len(evidence))
    old_residuals = [residual(world, row) for row in evidence]
    return [
        Evidence(
            x=row.x.copy(),
            y=base_prediction(world, row.x) + old_residuals[int(other)],
            coverage=row.coverage,
        )
        for row, other in zip(evidence, order)
    ]


def candidate_predictions(
    laws: list[Candidate], xs: np.ndarray
) -> np.ndarray:
    result = np.empty((len(xs), len(laws)), dtype=float)
    for column, law in enumerate(laws):
        a, b = law.pair
        result[:, column] = law.sign * GAMMA * xs[:, a] * xs[:, b]
    return result


def evidence_log_likelihoods(
    world: World,
    laws: list[Candidate],
    evidence: list[Evidence],
) -> np.ndarray:
    if not evidence:
        return np.zeros(len(laws), dtype=float)
    xs = np.stack([row.x for row in evidence])
    observed = np.asarray([residual(world, row) for row in evidence])
    predicted = candidate_predictions(laws, xs)
    errors = observed[:, None] - predicted
    return -0.5 * np.sum(errors**2, axis=0) / (NOISE_STD**2)


def posterior(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    return weights / weights.sum()


@lru_cache(maxsize=2)
def exact_domain(alias_constraint: bool) -> np.ndarray:
    rows = []
    for values in product((-1.0, 0.0, 1.0), repeat=DIM):
        x = np.asarray(values)
        if np.all(x == 0):
            continue
        if alias_constraint and x[2] != x[1]:
            continue
        rows.append(x)
    return np.stack(rows)


@lru_cache(maxsize=2)
def dense_domain(alias_constraint: bool) -> np.ndarray:
    rows = []
    for values in product((-1.0, 1.0), repeat=DIM):
        x = np.asarray(values)
        if alias_constraint and x[2] != x[1]:
            continue
        rows.append(x)
    return np.stack(rows)


def equivalence_classes(
    laws: list[Candidate], alias_constraint: bool
) -> list[list[int]]:
    """Partition laws by predictions on every legally observable input."""

    if laws == candidates():
        return [
            list(group)
            for group in canonical_equivalence_classes(alias_constraint)
        ]

    domain = exact_domain(alias_constraint)
    predictions = candidate_predictions(laws, domain)
    groups: dict[bytes, list[int]] = {}
    for index in range(len(laws)):
        # Values are exact multiples of GAMMA; sign patterns avoid float
        # tolerance becoming part of the identifiability definition.
        signature = np.sign(predictions[:, index]).astype(np.int8).tobytes()
        groups.setdefault(signature, []).append(index)
    return list(groups.values())


@lru_cache(maxsize=2)
def canonical_equivalence_classes(
    alias_constraint: bool,
) -> tuple[tuple[int, ...], ...]:
    laws = candidates()
    domain = exact_domain(alias_constraint)
    predictions = candidate_predictions(laws, domain)
    groups: dict[bytes, list[int]] = {}
    for index in range(len(laws)):
        signature = np.sign(predictions[:, index]).astype(np.int8).tobytes()
        groups.setdefault(signature, []).append(index)
    return tuple(tuple(group) for group in groups.values())


def decision(
    weights: np.ndarray, classes: list[list[int]]
) -> tuple[str, list[int], float]:
    masses = np.asarray([weights[group].sum() for group in classes])
    best = int(np.argmax(masses))
    winner = classes[best]
    mass = float(masses[best])
    if mass < POSTERIOR_THRESHOLD:
        return "UNRESOLVED", winner, mass
    if len(winner) == 1:
        return "IDENTIFIED", winner, mass
    return "NOT_IDENTIFIABLE", winner, mass


def true_candidate_index(world: World, laws: list[Candidate]) -> int:
    return laws.index(Candidate(world.true_pair, world.true_sign))


@lru_cache(maxsize=None)
def probe_pool(alias_constraint: bool, seed: int) -> np.ndarray:
    """A shared legal action set containing sparse and dense interventions."""

    rng = np.random.default_rng(seed + 85_000)
    rows = [
        easy_query(rng, alias_constraint)
        for _ in range(4 * DIM)
    ]
    for _ in range(256):
        x = rng.choice((-1.0, 0.0, 1.0), size=DIM)
        if np.all(x == 0):
            x[0] = 1.0
        rows.append(obey_constraint(x, alias_constraint))
    rows.extend(dense_domain(alias_constraint))
    return np.unique(np.stack(rows), axis=0)


def choose_probe(
    mode: str,
    pool: np.ndarray,
    prediction_matrix: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "random":
        return pool[int(rng.integers(len(pool)))].copy()
    if mode == "captured":
        # A contestant-owned judge asks a question on which all contestants
        # agree, preserving apparent certainty rather than testing it.
        variance = (
            prediction_matrix**2 @ weights
            - (prediction_matrix @ weights) ** 2
        )
        return pool[int(np.argmin(variance))].copy()
    if mode != "active":
        raise ValueError(f"unknown probe mode: {mode}")
    variance = (
        prediction_matrix**2 @ weights
        - (prediction_matrix @ weights) ** 2
    )
    best = np.flatnonzero(np.isclose(variance, variance.max()))
    return pool[int(rng.choice(best))].copy()


def law_rmse(
    world: World,
    law: Candidate | None,
) -> float:
    domain = dense_domain(world.alias_constraint)
    truth = np.asarray(
        [noiseless_world_prediction(world, x) for x in domain]
    )
    base = domain @ world.beta
    if law is None:
        predicted = base
    else:
        a, b = law.pair
        predicted = base + law.sign * GAMMA * domain[:, a] * domain[:, b]
    return float(np.sqrt(np.mean((truth - predicted) ** 2)))


def run_competition(
    world: World,
    seed: int,
    initial_evidence: list[Evidence],
    probe_mode: str,
) -> dict[str, float | int | str]:
    laws = candidates()
    classes = equivalence_classes(laws, world.alias_constraint)
    truth_index = true_candidate_index(world, laws)
    log_weights = evidence_log_likelihoods(world, laws, initial_evidence)
    weights = posterior(log_weights)
    status, winner, mass = decision(weights, classes)

    pool = probe_pool(world.alias_constraint, seed)
    predictions = candidate_predictions(laws, pool)
    query_rng = np.random.default_rng(seed + 86_000 + len(probe_mode) * 101)
    noise_rng = np.random.default_rng(seed + 87_000 + len(probe_mode) * 103)
    post_probes = 0

    while status == "UNRESOLVED" and post_probes < MAX_POST_PROBES:
        x = choose_probe(probe_mode, pool, predictions, weights, query_rng)
        y = observe(world, x, noise_rng)
        row = Evidence(x=x, y=y, coverage=True)
        log_weights += evidence_log_likelihoods(world, laws, [row])
        weights = posterior(log_weights)
        status, winner, mass = decision(weights, classes)
        post_probes += 1

    representative = laws[winner[0]]
    return {
        "status": status,
        "truth_covered": int(truth_index in winner and status != "UNRESOLVED"),
        "winner_class_size": len(winner),
        "top_class_mass": mass,
        "truth_posterior": float(weights[truth_index]),
        "post_birth_probes": post_probes,
        "noiseless_dense_rmse": law_rmse(world, representative),
    }


def policy_result(
    world: World,
    seed: int,
    policy: str,
    captured: list[Evidence],
    mixed: list[Evidence],
    peak_pending: int,
) -> dict[str, float | int | str]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")

    source = captured if policy == "captured_incumbent" else mixed
    pre_mse = mismatch_mse(world, source)
    mismatch = int(pre_mse > MISMATCH_MSE_THRESHOLD)
    dense_pre_probes = int(sum(row.coverage for row in source))
    pending_scalars = peak_pending * SCALARS_PER_RAW_RECORD

    if policy == "captured_incumbent":
        return {
            "status": "INCUMBENT_CONFIRMED",
            "mismatch_detected": mismatch,
            "hypothesis_birth": 0,
            "truth_covered": 0,
            "winner_class_size": 0,
            "top_class_mass": 0.0,
            "truth_posterior": 0.0,
            "post_birth_probes": 0,
            "dense_pre_probes": dense_pre_probes,
            "pre_audit_residual_mse": pre_mse,
            "noiseless_dense_rmse": law_rmse(world, None),
            "persistent_raw_records": 0,
            "peak_raw_scalars": pending_scalars,
        }

    if policy == "fixed_catalog_coverage":
        return {
            "status": "MODEL_MISMATCH" if mismatch else "INCUMBENT_CONFIRMED",
            "mismatch_detected": mismatch,
            "hypothesis_birth": 0,
            "truth_covered": 0,
            "winner_class_size": 0,
            "top_class_mass": 0.0,
            "truth_posterior": 0.0,
            "post_birth_probes": 0,
            "dense_pre_probes": dense_pre_probes,
            "pre_audit_residual_mse": pre_mse,
            "noiseless_dense_rmse": law_rmse(world, None),
            "persistent_raw_records": 0,
            "peak_raw_scalars": pending_scalars,
        }

    trace = bounded_audit_trace(mixed, seed)
    if policy == "no_replay_random":
        initial: list[Evidence] = []
        probe_mode = "random"
    elif policy == "no_replay_active":
        initial = []
        probe_mode = "active"
    elif policy == "shuffled_address_trace":
        initial = shuffle_causal_addresses(world, trace, seed)
        probe_mode = "active"
    elif policy == "constitutional_trace":
        initial = trace
        probe_mode = "active"
    elif policy == "full_audit_ledger":
        initial = [row for row in mixed if row.coverage]
        probe_mode = "active"
    elif policy == "full_history":
        initial = mixed
        probe_mode = "active"
    else:  # pragma: no cover - guarded above
        raise AssertionError(policy)

    result = run_competition(world, seed, initial, probe_mode)
    raw_records = len(initial)
    result.update(
        {
            "mismatch_detected": mismatch,
            "hypothesis_birth": mismatch,
            "dense_pre_probes": dense_pre_probes,
            "pre_audit_residual_mse": pre_mse,
            "persistent_raw_records": raw_records,
            "peak_raw_scalars": (
                raw_records * SCALARS_PER_RAW_RECORD + pending_scalars
            ),
        }
    )
    return result


def run_seed(
    seed: int, alias_constraint: bool = False
) -> dict[str, dict[str, float | int | str]]:
    world = make_world(seed, alias_constraint)
    captured, captured_pending = collect_delayed_evidence(
        world, seed, "captured"
    )
    mixed, mixed_pending = collect_delayed_evidence(world, seed, "mixed")
    assert captured_pending == mixed_pending == AUDIT_DELAY
    return {
        policy: policy_result(
            world, seed, policy, captured, mixed, mixed_pending
        )
        for policy in POLICIES
    }


def aggregate(
    rows: list[dict[str, dict[str, float | int | str]]]
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for policy in POLICIES:
        statuses = Counter(str(row[policy]["status"]) for row in rows)
        item: dict[str, object] = {"status_counts": dict(statuses)}
        for key, value in rows[0][policy].items():
            if key == "status" or not isinstance(value, (int, float, bool)):
                continue
            values = np.asarray([float(row[policy][key]) for row in rows])
            item[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        summary[policy] = item
    return summary


def capacity_sweep(seeds: int) -> dict[str, dict[str, float]]:
    """Birth-time recovery before the controller may buy another probe."""

    laws = candidates()
    classes = equivalence_classes(laws, False)
    rows = {
        capacity: {"identified": [], "truth": [], "mass": []}
        for capacity in TRACE_CAPACITY_SWEEP
    }
    for seed in range(seeds):
        world = make_world(seed, False)
        mixed, _ = collect_delayed_evidence(world, seed, "mixed")
        truth_index = true_candidate_index(world, laws)
        for capacity in TRACE_CAPACITY_SWEEP:
            trace = bounded_audit_trace(mixed, seed, capacity)
            weights = posterior(
                evidence_log_likelihoods(world, laws, trace)
            )
            status, winner, mass = decision(weights, classes)
            rows[capacity]["identified"].append(status == "IDENTIFIED")
            rows[capacity]["truth"].append(
                status == "IDENTIFIED" and truth_index in winner
            )
            rows[capacity]["mass"].append(mass)

    return {
        str(capacity): {
            "identified_fraction": float(
                np.mean(rows[capacity]["identified"])
            ),
            "truth_recovery_fraction": float(
                np.mean(rows[capacity]["truth"])
            ),
            "mean_top_class_mass": float(
                np.mean(rows[capacity]["mass"])
            ),
        }
        for capacity in TRACE_CAPACITY_SWEEP
    }


def run_all(seeds: int = SEEDS) -> dict[str, object]:
    identifiable_rows = [run_seed(seed, False) for seed in range(seeds)]
    alias_rows = [run_seed(seed, True) for seed in range(seeds)]
    return {
        "config": {
            "dimensions": DIM,
            "candidate_laws": len(candidates()),
            "interaction_magnitude": GAMMA,
            "noise_std": NOISE_STD,
            "pre_birth_trials": PRE_TRIALS,
            "coverage_period": COVERAGE_PERIOD,
            "audit_delay": AUDIT_DELAY,
            "trace_capacity": TRACE_CAPACITY,
            "posterior_threshold": POSTERIOR_THRESHOLD,
            "max_post_birth_probes": MAX_POST_PROBES,
            "seeds": seeds,
        },
        "identifiable": aggregate(identifiable_rows),
        "aliased": aggregate(alias_rows),
        "birth_time_capacity_sweep": capacity_sweep(seeds),
        "per_seed_identifiable": identifiable_rows,
        "per_seed_aliased": alias_rows,
    }


def mean(summary: dict[str, object], policy: str, key: str) -> float:
    policy_row = summary[policy]
    metric = policy_row[key]
    assert isinstance(metric, dict)
    return float(metric["mean"])


def main() -> None:
    receipt = run_all()
    result_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "gate8_constitutional_scientist.json"
    )
    result_path.write_text(json.dumps(receipt, indent=2) + "\n")

    identified = receipt["identifiable"]
    aliased = receipt["aliased"]
    assert isinstance(identified, dict)
    assert isinstance(aliased, dict)

    print("Gate 8 — constitutional executable-law competition")
    print("\nIdentifiable world")
    print(
        f"{'policy':28s} {'truth':>8s} {'post probes':>12s} "
        f"{'dense pre':>10s} {'raw scalars':>12s} {'RMSE':>8s}"
    )
    for policy in POLICIES:
        print(
            f"{policy:28s} "
            f"{mean(identified, policy, 'truth_covered'):8.3f} "
            f"{mean(identified, policy, 'post_birth_probes'):12.2f} "
            f"{mean(identified, policy, 'dense_pre_probes'):10.1f} "
            f"{mean(identified, policy, 'peak_raw_scalars'):12.1f} "
            f"{mean(identified, policy, 'noiseless_dense_rmse'):8.3f}"
        )

    print("\nAliased world — constitutional trace")
    alias_status = aliased["constitutional_trace"]["status_counts"]
    print(f"status counts: {alias_status}")
    print(
        "true equivalence-class coverage: "
        f"{mean(aliased, 'constitutional_trace', 'truth_covered'):.3f}"
    )

    print("\nBirth-time bounded-ledger capacity")
    capacity = receipt["birth_time_capacity_sweep"]
    assert isinstance(capacity, dict)
    for records, row in capacity.items():
        assert isinstance(row, dict)
        print(
            f"{records:>2s} records  "
            f"identified={float(row['identified_fraction']):.3f}  "
            f"truth={float(row['truth_recovery_fraction']):.3f}"
        )
    print(f"\nWrote {result_path}")


if __name__ == "__main__":
    main()
