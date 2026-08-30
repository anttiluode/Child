# Gate 1 — the illegal predictor

## Question

If a predictor can alter the transition process that generates its future
observations, will prediction pressure improve its model or make the coupled
world easier to predict?

## Setup

The Gate-0 ring receives one extra degree of freedom at the output boundary.

At the currently active cell:

```text
TRANSMIT
    -> let the pulse move according to the external ring dynamics

HOLD
    -> keep the pulse at the same cell for one step
```

The external direction reverses with probability `0.15` per step.

The local next-state predictor is explicitly told which action was taken, so
this is not a hidden-action problem.  It receives local activity, persistent
local trace, action, and bias.

Every location also owns one stochastic output gate.  In the prediction-only
arm its local scalar reward is simply:

```text
1  if the next active cell was predicted correctly
0  otherwise
```

The gate uses a REINFORCE-style one-step local update.  No gradient passes
through the predictor or environment.

## Why HOLD is an illegal shortcut

If the gate closes, the next observation equals the present one.  Once that
transition is learned, the world becomes nearly trivial to predict.

The predictor has therefore changed the data-generating process instead of
becoming a better predictor of the original moving process.

This is not illegal mathematically.  It is "illegal" only relative to the
external task we had intended the machine to solve.

## Frozen 10-seed receipt

The predictor first receives 4,000 random-action steps so both TRANSMIT and
HOLD are known.  Gate adaptation then runs for 12,000 steps.  Predictor and
gate parameters are frozen for a final 4,000-step evaluation.

| arm | movement fraction | prediction accuracy | mean transmit probability |
|---|---:|---:|---:|
| **prediction only** | **0.0456 ± 0.0140** | **0.9544 ± 0.0140** | 0.0831 ± 0.0132 |
| prediction + local homeostasis | 0.8716 ± 0.0033 | 0.7216 ± 0.0050 | 0.8736 ± 0.0023 |
| fixed transmit probability 0.80 | 0.7991 ± 0.0048 | 0.6517 ± 0.0045 | 0.8000 |
| forced open | 1.0000 | 0.8483 ± 0.0030 | 1.0000 |

The prediction-only gate suppresses about **95.4%** of possible motion.

Relative to prediction-only, the homeostatic arm carries about **19.1x** as
much pulse traffic.

The fixed/forced controls are intentionally embarrassing: no learned
homeostatic mechanism is needed if the designer simply forbids closure.

## What the homeostatic term does

The second learned arm receives the same prediction reward plus a purely local
pressure on its expected transmission probability:

```text
delta logit_homeo =
    eta_h * (target_transmission - p_transmit)
```

with target `0.80`.

This is not a model of a real AIS.  It is the smallest executable version of a
separate constraint acting at an output boundary.

## Earned statement

> **When a predictor controls part of the transition process that produces its
> future input, prediction accuracy alone can select a low-dynamics attractor
> that makes prediction easy.  A separate viability/homeostatic constraint can
> preserve a high-flow regime, but an explicit fixed gate is still the simpler
> engineering solution in this toy.**

## More important mathematical statement

The optimization distribution is endogenous.

Ordinary supervised prediction assumes samples are drawn from an externally
specified process:

```math
x_{t+1} ~ P_world(. | x_t)
```

Here the learner also controls part of that process:

```math
a_t ~ pi_psi(. | s_t)

x_{t+1} ~ P_world(. | x_t, a_t)
```

so minimizing prediction loss jointly over predictor and policy can alter the
trajectory distribution itself:

```math
min_(theta, psi)
    E_(tau ~ P_psi) [
        ell(f_theta(s_t, a_t), x_(t+1))
    ].
```

A low-entropy self-induced trajectory can therefore be an optimum even when it
fails the external purpose.

## Not earned

- active inference novelty;
- a biological AIS learning rule;
- a theory of motivation or agency;
- proof that homeostasis is sufficient for useful intelligence;
- superiority to fixed control;
- a transformer replacement.

## Next

Gate 2 should distinguish **surprise** from **lack of observability**.

When prediction is poor, the machine should have three very different possible
responses:

```text
LEARN   -> improve internal model
SENSE   -> obtain missing information
ACT     -> alter the world
```

The experiment should force these three operators to have different costs and
different downstream consequences.
