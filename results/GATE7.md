# Gate 7 — temporal routing over a sparse body

## Question

After Gate 3 has retrieved an old episode's anchor, can a fast read state reach
a requested represented-time offset through a slow sparse body using only its
current neighbours?  This is a routing test, not a claim about physical time
travel or biological hippocampus.

## World and resource budget

- 256 circular represented-time addresses;
- out-degree 12 for every body;
- 3,072 persistent directed supports per body;
- eight-hop route budget;
- 20 independent graph seeds and 4,000 queries per workload;
- local router chooses the available neighbour closest to the target address.

The random and small-world arms are reciprocal, degree-preserving graphs.  An
oracle shortest-path score is reported separately so graph reach is not
mistaken for local navigability.

## Uniform temporal-offset receipt

| body | greedy success | capped hops | oracle success |
|---|---:|---:|---:|
| local ±1…±6 | `0.3739 ± 0.0075` | `7.317` | `0.3739` |
| reciprocal random | `0.1032 ± 0.0053` | `8.258` | `1.0000` |
| degree-matched small-world | `0.9694 ± 0.0087` | `4.346` | `1.0000` |
| dyadic ±1,2,4,8,16,32 | `1.0000 ± 0.0000` | `3.621` | `1.0000` |
| learned lag support 1,2,4,8,32,64 | `1.0000 ± 0.0000` | `3.151` | `1.0000` |

The reciprocal random body is the key control: paths exist, but a local
address-only policy cannot discover them reliably.  Multiscale supports make
the ring locally navigable under this fixed budget.

## Attackers

| attacker | success | cost |
|---|---:|---|
| direct index | `1.0000` | one random-access fetch |
| dense attention scan | `1.0000` | 256 address comparisons |
| unit-step bidirectional scan | `0.0625` on uniform offsets | one local edge per hop |

The gate therefore does not show a general computational advantage over RAM or
dense attention.  It shows a topology effect under a deliberately local
readout rule.

## Earned statement

> Under an equal persistent edge budget, dyadic or learned multiscale temporal
> supports make long represented-time offsets locally navigable; random graph
> reach without a local route policy does not.  The result is an auditable
> routing property, not evidence of agency, intelligence, or time travel.

## Reproduce

```bash
python experiments/gate7_temporal_routing_body.py
python -m unittest discover -s tests -v
```

Raw receipt: [gate7_temporal_routing_body.json](gate7_temporal_routing_body.json).
