# Child — put prediction inside the running system

**Status: Gates 0–1 built, 2026-08-30.**

`Child` starts from one intentionally naive question:

> **What happens if the useful part of next-token prediction is moved inside a continuously running network, so each unit predicts only the next state of its local surroundings while fast state and persistent relationships carry history?**

This is not a claim that brains perform language-model next-token prediction. It is not a biological neuron simulator and it is not yet an alternative to a transformer.

The seam under test is:

```text
next-state prediction
        +
receiver-carried history
        +
persistent local addressability
        +
state-dependent continuation
        +
later: active observation + delayed consequence
```

The point of Gate 0 is to make that seam executable before adding dendrites, neurotransmitters, oscillators, global reward, or another mythology layer.

## Why this exists

A transformer has a simple prediction objective, but the learned computation is not simply 'output a token'. Context changes hidden state; hidden state changes routing; routing changes what information reaches the next computation.

The older repo line approached related operations from the other side:

```text
Sunday / FunctionalArbors
    history changes persistent transfer structure

Monday / Tuesday
    temporal structure can expose hidden causes / reusable coordinates

yrotisopeRweN / T-800NNP
    present-time-identical events route differently because receiver history differs

KyberDyyni1
    fast computation can precede slow consolidation

MovingProblem
    persistent meaning can require more than the current coordinate frame

AlgoSchalgo
    when the current observation is insufficient, choose another observation
```

`Child` asks what happens if those operations are placed around **prediction itself**.

## The key inversion

A language model is roughly:

```text
stored context
      ↓
large shared model
      ↓
next-token distribution
      ↓
chosen token is appended
      ↓
run the model again
```

The first Child toy is:

```text
local activity x_j(t)
       +
persistent receiver trace q_j(t)
       ↓
cell j predicts its own next activation
       ↓
local one-step prediction error
       ↓
cell j's persistent incoming structure changes
       ↓
optional: predicted activity becomes x(t+1)
       ↓
the predictor is now a dynamical generator
```

There is no global attention matrix in Gate 0. Every cell owns a small fixed-radius parameter vector. This does **not** make it better than attention; it makes the contrast testable.

# Gate 0 — local next-state prediction needs history

`experiments/gate0_local_next_state.py`

A one-hot pulse travels around a 24-cell ring. It spends long stretches moving clockwise or counter-clockwise and reverses randomly with probability `1/80` per step.

At one instant, the current pulse position does not tell you which neighbour activates next. The missing variable is direction carried by recent history.

Each destination cell sees only five current local activations, optionally five exponentially decaying local traces, and one bias. It predicts only its own next activation.

```text
prediction_j(t+1) = w_j · features_j(t)

local_error_j = x_j(t+1) - prediction_j(t+1)

w_j <- w_j + eta * local_error_j * features_j(t)
```

There is no autograd, BPTT, reverse traversal through layers, global hidden-state error, global attention matrix, or episode reset. The next local state is still a teaching signal, so this is not a solution to general biological credit assignment.

## Receipt

Eight independent continuous streams; learning freezes after step 8,000 and evaluation continues to step 12,000:

| system | frozen next-state accuracy |
|---|---:|
| current activity only | `0.4942 ± 0.0647` |
| persistent trace, addresses scrambled | `0.4927 ± 0.0640` |
| **persistent local receiver trace** | **`0.9875 ± 0.0013`** |
| explicit previous-position Markov attacker | **`0.9877 ± 0.0013`** |

The boring attacker is the correct explanation. Random unannounced reversals impose an expected continuation ceiling near `79/80 = 0.9875`, and the stateful local predictor essentially reaches it.

So Gate 0 does not discover a new predictor. It earns this narrower statement:

> **A population of individually parameterized local next-state predictors can use receiver-carried history to turn an otherwise ambiguous present into the correct continuation. Persistent addressability matters: scrambling where the history lives destroys the effect.**

## Next token -> pulse

After training, show only two pulses establishing direction, then stop supplying the world and feed the predicted winner back as the next activity state.

```text
autonomous 96-step travelling-wave fidelity = 1.0000 ± 0.0000
```

Thus:

```text
one-step predictor
        ↓ feed prediction back
recurrent dynamical system
        ↓
self-maintaining pulse trajectory
```

This is the exact bridge that motivated the repo: **if a prediction changes the state from which the next prediction is made, repeated prediction is dynamics.**

# The important trap — prediction can cheat

The next gate attacks prediction itself.

If a predictor is allowed to alter the system it predicts, it can reduce error in two ways:

```text
A. learn the changing world better

B. make the world easier to predict
```

B can be useful control. It can also become silence, synchrony, a frozen attractor, or repetitive self-stimulation.

This is the repo's version of the earlier legal/illegal-operator thought. From the objective's perspective, changing the observation process may be perfectly legal even when it dodges the intended task.

An organism has other constraints: homeostasis, damage, metabolic cost, inherited drives, other organisms, and consequences that cannot all be satisfied by becoming silent. Child should add such constraints explicitly rather than assume prediction alone creates intelligence.

# Intended architecture, if later gates earn it

```text
external stream
      ↓
local observations
      ↓
fast receiver state q_i(t)
      ↓
local next-state model
      ↓
local prediction error
      ├─────────────→ fast correction / routing
      ↓
eligibility
      ↓ delayed consequence
slow local structure
      ↓
changes future flow

if observation cannot support the needed distinction:
      ↓
optional PULSE -> acquire another view
```

Gate 0 deliberately does not yet include biological spikes, theta, neurotransmitter fields, dendritic morphology, structural growth, active sensing, organism-level reward, slow consolidation, forgetting, drifting internal coordinates, or a transformer benchmark.

# Planned attacks

## Gate 1 — the illegal predictor — BUILT

The pulse now owns a local output gate: TRANSMIT lets the external dynamics advance; HOLD keeps the current state in place.

The predictor is action-conditioned, so HOLD is honestly modeled rather than hidden from it. The output gate then learns from prediction correctness alone.

Ten-seed frozen receipt:

| arm | movement | prediction accuracy |
|---|---:|---:|
| **prediction only** | **0.0456 ± 0.0140** | **0.9544 ± 0.0140** |
| prediction + local homeostasis | 0.8716 ± 0.0033 | 0.7216 ± 0.0050 |
| fixed gate p=0.80 | 0.7991 ± 0.0048 | 0.6517 ± 0.0045 |
| forced open | 1.0000 | 0.8483 ± 0.0030 |

Prediction-only suppresses about 95% of possible motion. It did not become a brilliant model of the moving world; it changed the data-generating process into a nearly static one.

This earns:

> **When a predictor controls part of the transition process that generates its future input, prediction loss alone can select a low-dynamics attractor that makes prediction easy. A separate viability/homeostatic constraint can keep the system in a high-flow regime.**

The fixed gate remains the boring engineering attacker. See [results/GATE1.md](results/GATE1.md) and [MATH.md](MATH.md).

## Gate 2 — pulse only when information is missing

Bring in the narrow AlgoSchalgo result: prediction error is not observability. Buy an extra observation only when the current view cannot support the required distinction.

## Gate 3 — fast state becomes slow knowledge

A temporary fast correction succeeds; consequence arrives later. Test whether correctly addressed eligibility can consolidate only the reusable part into bounded local structure.

## Gate 4 — representation drift

Let the child's own internal coordinates become plastic. Test whether persistent address, temporal identity, and path memory preserve old function without global retraining.

## Gate 5 — boring attackers

Explicit Markov/state-space models, reservoir + linear readout, GRU/RNN, small SSM, tiny transformer, and matched active-sensing controllers are all allowed to win.

# Scaling

Gate 0 has `N=24`, radius `r=2`, and 11 parameters per cell. Its persistent predictive structure is local, scaling roughly as `O(Nr)`. If useful tasks demand arbitrary personal all-to-all relationships, the structure approaches `O(N²)` and the comparison with attention changes substantially.

# Run

```bash
python -m pip install -r requirements.txt
python experiments/gate0_local_next_state.py
python experiments/gate1_illegal_predictor.py
python -m unittest discover -s tests -v
```

The executable writes `results/gate0_local_next_state.json`. `index.html` is a browser illustration, not the experimental receipt.

# Lineage

- [T-800NNP](https://github.com/anttiluode/T-800NNP) — continuously running receiver state changes routing.
- [KyberDyyni1](https://github.com/anttiluode/KyberDyyni1) — fast computation versus slow consolidation and temporal correspondence.
- [MovingProblem](https://github.com/anttiluode/MovingProblem) — identity under moving representations and path memory.
- [AlgoSchalgo](https://github.com/anttiluode/AlgoSchalgo) — observability, ambiguity, and active measurement.
- [Tuesday](https://github.com/anttiluode/Tuesday) — temporal structure as a source of identifiability.
- [GeoNeuronX](https://github.com/anttiluode/GeoNeuronX) — history materialized into receiver state before readout.

The repo name is allowed to remain embarrassing until the machine earns a better one.
