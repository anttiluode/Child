"""Gate 7: a moving address over an equal-budget temporal body.

Gate 3 used dense content attention to recover the index of an old encounter.
This gate starts *after* that anchor has been recovered.  A later request asks
for context at a signed temporal offset from the anchor.  A one-hot read state
must move through a sparse persistent graph until it reaches that represented
time.

The graph is the slow body S.  At every hop a fast conductance rule opens one
available edge: the edge whose destination is locally closest to the requested
address.  All graph arms have exactly the same number of persistent directed
edges.

This is a routing / skip-graph toy, not a hippocampus model and not a claim that
graph traversal beats RAM.  Direct indexed access is the boring software
attacker.  The experiment asks the narrower physical-style question: which
sparse bodies make a temporal address locally navigable per edge?
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np


N_NODES = 256
OUT_DEGREE = 12
HOP_BUDGET = 8
GRAPH_SEEDS = 20
QUERIES_PER_WORKLOAD = 4_000
SMALL_WORLD_REWIRE = 0.25


def circular_delta(target: np.ndarray | int, source: np.ndarray | int):
    """Signed shortest displacement on an even-sized ring."""

    return (np.asarray(target) - np.asarray(source) + N_NODES // 2) % N_NODES - N_NODES // 2


def circular_distance(target: np.ndarray | int, source: np.ndarray | int):
    return np.abs(circular_delta(target, source))


def circulant_body(positive_offsets: list[int]) -> np.ndarray:
    """Build a degree-matched directed body from symmetric ring offsets."""

    if len(positive_offsets) * 2 != OUT_DEGREE:
        raise ValueError("positive offsets must supply half the out-degree")
    if len(set(positive_offsets)) != len(positive_offsets):
        raise ValueError("offsets must be unique")

    offsets = np.array(
        [value for offset in positive_offsets for value in (offset, -offset)],
        dtype=np.int64,
    )
    nodes = np.arange(N_NODES, dtype=np.int64)[:, None]
    return (nodes + offsets[None, :]) % N_NODES


def _undirected_edges(body: np.ndarray) -> set[tuple[int, int]]:
    return {
        (min(node, int(neighbour)), max(node, int(neighbour)))
        for node, neighbours in enumerate(body)
        for neighbour in neighbours
    }


def _body_from_undirected_edges(edges: set[tuple[int, int]]) -> np.ndarray:
    neighbours = [set() for _ in range(N_NODES)]
    for left, right in edges:
        neighbours[left].add(right)
        neighbours[right].add(left)
    body = np.array([sorted(row) for row in neighbours], dtype=np.int64)
    validate_body(body)
    return body


def _degree_preserving_swaps(
    body: np.ndarray,
    rng: np.random.Generator,
    requested_swaps: int,
    protected: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Randomize reciprocal support without changing any node's degree."""

    protected = protected or set()
    edge_set = _undirected_edges(body)
    edges = list(edge_set)
    completed = 0
    attempts = 0
    max_attempts = max(1_000, requested_swaps * 100)

    while completed < requested_swaps and attempts < max_attempts:
        attempts += 1
        first_index, second_index = rng.choice(len(edges), 2, replace=False)
        first = edges[int(first_index)]
        second = edges[int(second_index)]
        if first in protected or second in protected:
            continue

        a, b = first
        c, d = second
        if rng.random() < 0.5:
            a, b = b, a
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue

        proposal_a = (min(a, d), max(a, d))
        proposal_b = (min(c, b), max(c, b))
        if proposal_a == proposal_b:
            continue
        if proposal_a in edge_set or proposal_b in edge_set:
            continue

        edge_set.remove(first)
        edge_set.remove(second)
        edge_set.add(proposal_a)
        edge_set.add(proposal_b)
        edges[int(first_index)] = proposal_a
        edges[int(second_index)] = proposal_b
        completed += 1

    if completed != requested_swaps:
        raise RuntimeError(
            f"only completed {completed}/{requested_swaps} degree-preserving swaps"
        )
    return _body_from_undirected_edges(edge_set)


def random_sparse_body(rng: np.random.Generator) -> np.ndarray:
    """A reciprocal 12-regular graph obtained by thoroughly mixing a ring."""

    local = circulant_body([1, 2, 3, 4, 5, 6])
    undirected_edge_count = N_NODES * OUT_DEGREE // 2
    return _degree_preserving_swaps(
        local,
        rng,
        requested_swaps=10 * undirected_edge_count,
    )


def small_world_body(rng: np.random.Generator) -> np.ndarray:
    """Reciprocal small-world body, preserving degree and +/-1 ring edges."""

    local = circulant_body([1, 2, 3, 4, 5, 6])
    protected = {
        (min(node, (node + 1) % N_NODES), max(node, (node + 1) % N_NODES))
        for node in range(N_NODES)
    }
    undirected_edge_count = N_NODES * OUT_DEGREE // 2
    # Each double-edge swap replaces two supports, so p*E/2 swaps rewires
    # approximately fraction p while keeping every node at degree 12.
    requested = int(SMALL_WORLD_REWIRE * undirected_edge_count / 2)
    return _degree_preserving_swaps(
        local,
        rng,
        requested_swaps=requested,
        protected=protected,
    )


def validate_body(body: np.ndarray) -> None:
    if body.shape != (N_NODES, OUT_DEGREE):
        raise ValueError(f"wrong body shape: {body.shape}")
    for node, neighbours in enumerate(body):
        if node in neighbours:
            raise ValueError("self edge found")
        if len(set(neighbours.tolist())) != OUT_DEGREE:
            raise ValueError("duplicate outgoing edge found")
        for neighbour in neighbours:
            if node not in body[int(neighbour)]:
                raise ValueError("non-reciprocal support found")


def greedy_offset_hops(distance: int, positive_offsets: list[int]) -> int:
    """Number of locally greedy hops for a homogeneous circulant body."""

    remaining = int(distance)
    steps = np.array(
        sorted({value for offset in positive_offsets for value in (offset, -offset)}),
        dtype=np.int64,
    )
    for hop in range(1, N_NODES + 1):
        candidates = circular_delta(remaining, steps)
        choice = int(np.argmin(np.abs(candidates)))
        if abs(int(candidates[choice])) >= abs(remaining):
            return N_NODES + 1
        remaining = int(candidates[choice])
        if remaining == 0:
            return hop
    return N_NODES + 1


def mixed_distance_weights() -> np.ndarray:
    """Training workload: half local, one quarter multiscale, one quarter broad."""

    weights = np.full(N_NODES // 2, 0.25 / (N_NODES // 2), dtype=float)
    weights[:8] += 0.50 / 8
    for distance in (1, 2, 4, 8, 16, 32, 64, 128):
        weights[distance - 1] += 0.25 / 8
    return weights / weights.sum()


def learn_lag_offsets() -> list[int]:
    """Greedily learn six shared offsets from the mixed training workload."""

    distances = np.arange(1, N_NODES // 2 + 1)
    weights = mixed_distance_weights()
    selected = [1]  # guarantees exact reach rather than a disconnected coin set

    while len(selected) < OUT_DEGREE // 2:
        best: tuple[float, int] | None = None
        for candidate in range(2, N_NODES // 2):
            if candidate in selected:
                continue
            offsets = selected + [candidate]
            hops = np.array(
                [greedy_offset_hops(int(distance), offsets) for distance in distances],
                dtype=float,
            )
            score = float(weights @ hops)
            key = (score, candidate)
            if best is None or key < best:
                best = key
        assert best is not None
        selected.append(best[1])
        selected.sort()

    return selected


def sample_queries(
    rng: np.random.Generator,
    workload: str,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors = rng.integers(0, N_NODES, size=count)

    if workload == "local":
        magnitudes = rng.integers(1, 9, size=count)
    elif workload == "uniform":
        magnitudes = rng.integers(1, N_NODES // 2 + 1, size=count)
    elif workload == "mixed":
        arm = rng.choice(3, size=count, p=[0.50, 0.25, 0.25])
        magnitudes = rng.integers(1, N_NODES // 2 + 1, size=count)
        local = arm == 0
        multiscale = arm == 1
        magnitudes[local] = rng.integers(1, 9, size=int(local.sum()))
        powers = np.array([1, 2, 4, 8, 16, 32, 64, 128])
        magnitudes[multiscale] = rng.choice(powers, size=int(multiscale.sum()))
    else:
        raise ValueError(f"unknown workload: {workload}")

    signs = rng.choice(np.array([-1, 1]), size=count)
    targets = (anchors + signs * magnitudes) % N_NODES
    return anchors, targets, magnitudes


def greedy_route(
    body: np.ndarray,
    anchor: int,
    target: int,
) -> tuple[bool, int, bool]:
    """Route using only current neighbours and their temporal addresses."""

    current = int(anchor)
    if current == target:
        return True, 0, False

    for hop in range(1, HOP_BUDGET + 1):
        neighbours = body[current]
        neighbour_distance = circular_distance(target, neighbours)
        next_node = int(neighbours[int(np.argmin(neighbour_distance))])

        if circular_distance(target, next_node) >= circular_distance(target, current):
            return False, hop - 1, True

        current = next_node
        if current == target:
            return True, hop, False

    return False, HOP_BUDGET, False


def all_pairs_shortest_hops(body: np.ndarray) -> np.ndarray:
    """Oracle route table: measures reach in the support, not local usability."""

    unreachable = N_NODES + 1
    result = np.full((N_NODES, N_NODES), unreachable, dtype=np.int16)

    for source in range(N_NODES):
        result[source, source] = 0
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            next_distance = int(result[source, node]) + 1
            for neighbour in body[node]:
                neighbour = int(neighbour)
                if result[source, neighbour] == unreachable:
                    result[source, neighbour] = next_distance
                    queue.append(neighbour)
    return result


def evaluate_body(
    body: np.ndarray,
    queries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, dict[str, float]]:
    validate_body(body)
    shortest = all_pairs_shortest_hops(body)
    output: dict[str, dict[str, float]] = {}

    for workload, (anchors, targets, _magnitudes) in queries.items():
        success = []
        hops = []
        stuck = []
        for anchor, target in zip(anchors, targets, strict=True):
            did_reach, used_hops, did_stick = greedy_route(
                body,
                int(anchor),
                int(target),
            )
            success.append(did_reach)
            hops.append(used_hops)
            stuck.append(did_stick)

        success_array = np.asarray(success, dtype=bool)
        hops_array = np.asarray(hops, dtype=float)
        oracle_hops = shortest[anchors, targets].astype(float)
        capped = np.where(success_array, hops_array, HOP_BUDGET + 1)

        output[workload] = {
            "greedy_success": float(success_array.mean()),
            "greedy_success_hops": float(
                hops_array[success_array].mean() if success_array.any() else HOP_BUDGET + 1
            ),
            "greedy_capped_hops": float(capped.mean()),
            "greedy_stuck": float(np.mean(stuck)),
            "greedy_neighbour_inspections": float(
                (np.maximum(hops_array, 1.0) * OUT_DEGREE).mean()
            ),
            "oracle_shortest_success": float(np.mean(oracle_hops <= HOP_BUDGET)),
            "oracle_shortest_hops": float(oracle_hops.mean()),
        }

    return output


def build_bodies(seed: int, learned_offsets: list[int]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(10_000 + seed)
    return {
        "local": circulant_body([1, 2, 3, 4, 5, 6]),
        "random_sparse": random_sparse_body(rng),
        "small_world": small_world_body(rng),
        "dyadic_multiscale": circulant_body([1, 2, 4, 8, 16, 32]),
        "learned_lag_support": circulant_body(learned_offsets),
    }


def run_seed(
    seed: int,
    query_count: int = QUERIES_PER_WORKLOAD,
    learned_offsets: list[int] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    if learned_offsets is None:
        learned_offsets = learn_lag_offsets()
    rng = np.random.default_rng(seed)
    queries = {
        workload: sample_queries(rng, workload, query_count)
        for workload in ("local", "mixed", "uniform")
    }
    return {
        name: evaluate_body(body, queries)
        for name, body in build_bodies(seed, learned_offsets).items()
    }


def aggregate(
    per_seed: list[dict[str, dict[str, dict[str, float]]]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    output = {}
    for body in per_seed[0]:
        output[body] = {}
        for workload in per_seed[0][body]:
            output[body][workload] = {}
            for metric in per_seed[0][body][workload]:
                values = np.array(
                    [seed[body][workload][metric] for seed in per_seed],
                    dtype=float,
                )
                output[body][workload][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                }
    return output


def attacker_receipt() -> dict[str, dict[str, float | str]]:
    weights = mixed_distance_weights()
    distances = np.arange(1, N_NODES // 2 + 1)
    return {
        "direct_index": {
            "success": 1.0,
            "address_fetches": 1.0,
            "cost_note": "requires ordinary random-access memory",
        },
        "dense_attention_scan": {
            "success": 1.0,
            "address_comparisons": float(N_NODES),
            "cost_note": "global comparison over every trace slot",
        },
        "unit_step_bidirectional_scan": {
            "local_success": 1.0,
            "mixed_success": float(weights[distances <= HOP_BUDGET].sum()),
            "uniform_success": float(np.mean(distances <= HOP_BUDGET)),
            "cost_note": "one local temporal edge traversed per hop",
        },
    }


def main() -> None:
    learned_offsets = learn_lag_offsets()
    per_seed = [
        run_seed(seed, learned_offsets=learned_offsets)
        for seed in range(GRAPH_SEEDS)
    ]
    summary = aggregate(per_seed)

    print("Gate 7 — temporal routing body")
    print(f"learned positive lag offsets: {learned_offsets}")
    print(
        f"nodes={N_NODES} out_degree={OUT_DEGREE} "
        f"persistent_edges={N_NODES * OUT_DEGREE} hop_budget={HOP_BUDGET}"
    )
    print("\nuniform temporal-offset workload")
    print("body                    greedy success   capped hops   oracle success")
    for body, workloads in summary.items():
        row = workloads["uniform"]
        print(
            f"{body:24s} "
            f"{row['greedy_success']['mean']:.4f} ± {row['greedy_success']['std']:.4f}   "
            f"{row['greedy_capped_hops']['mean']:.3f}   "
            f"{row['oracle_shortest_success']['mean']:.4f}"
        )

    receipt = {
        "gate": 7,
        "nodes": N_NODES,
        "out_degree": OUT_DEGREE,
        "persistent_directed_edges": N_NODES * OUT_DEGREE,
        "hop_budget": HOP_BUDGET,
        "graph_seeds": GRAPH_SEEDS,
        "queries_per_workload": QUERIES_PER_WORKLOAD,
        "small_world_rewire": SMALL_WORLD_REWIRE,
        "learned_positive_offsets": learned_offsets,
        "attackers": attacker_receipt(),
        "summary": summary,
        "per_seed": per_seed,
    }

    out = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "gate7_temporal_routing_body.json"
    )
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
