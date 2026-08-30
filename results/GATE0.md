# Gate 0 — local next-state prediction

## Question

Can individually parameterized local predictors use persistent receiver history to predict an otherwise ambiguous next state, with no BPTT or global attention operation?

## World

- 24 positions on a ring.
- One active pulse.
- Long clockwise/counter-clockwise runs.
- Random direction reversal probability `1/80` each step.
- Current position alone is insufficient to infer direction.

## Candidate

Each destination cell owns a radius-2 linear next-state predictor over:

```text
5 current local activations
5 persistent local traces
1 bias
```

The trace is:

```text
q(t+1) = 0.40 q(t) + 0.60 x(t)
```

The local update is:

```text
e_j = x_j(t+1) - prediction_j(t+1)
w_j <- w_j + 0.03 e_j features_j(t)
```

No error is propagated backward through the trace dynamics.

## Frozen evaluation

Eight seeds; 12,000 continuous steps; learning frozen after 8,000.

| system | accuracy |
|---|---:|
| current only | `0.494186 ± 0.064689` |
| trace addresses scrambled | `0.492654 ± 0.063997` |
| **stateful local predictor** | **`0.987528 ± 0.001253`** |
| explicit previous-position attacker | **`0.987747 ± 0.001250`** |

The explicit attacker is the correct boring solution. Random unannounced reversals impose an expected continuation ceiling near `79/80 = 0.9875`.

## Autonomous closure

After training, reset fast state, show only two pulses defining one direction, then feed the model's predicted winning cell back as its next input.

Three trained seeds × both directions × 96 generated steps:

```text
rollout fidelity = 1.000000 ± 0.000000
```

## Ablation

Scrambling trace addresses makes the history still exist numerically but disconnects it from persistent location. Performance collapses to the current-only baseline.

This is why the gate is about **persistent addressability**, not merely adding an extra vector called memory.

## Earned statement

> A locally connected predictive population can convert receiver-carried history into correct continuation of an otherwise ambiguous present; when its own predicted next state is fed back, the one-step predictor becomes a self-running pulse generator.

## Not earned

- novelty over Markov prediction;
- transformer replacement;
- biological learning;
- general credit assignment;
- spontaneous oscillation;
- source separation;
- active inference;
- useful autonomous agency.

## Next attack

Give the predictor a limited action that changes what it will observe next. Test whether minimizing prediction error learns the world or instead suppresses the difficult dynamics.
