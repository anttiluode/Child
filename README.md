# Child — put prediction inside the running system

**Status: Gates 0–6 built, 2026-08-30.**

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

## Gate 2 — delayed relevance: fast trace -> slow state — BUILT

The new SKILL.state paper makes explicit mutable execution state a strong
alternative to replaying an ever-growing conversation history.  Its own
boundary is equally important: an observation may become relevant only later,
after the runtime failed to preserve it.

Gate 2 gives every case 16 fields but reveals which one matters only 20–80
steps later.  The same fact is queried again after the fast trace has expired.

| system | first query | later repeat | peak scalar storage |
|---|---:|---:|---:|
| full history | 1.0000 | 1.0000 | 80,000 |
| early one-field state | 0.0652 ± 0.0028 | 0.0652 ± 0.0028 | 5,000 |
| oracle one-field state | 1.0000 | 1.0000 | 5,000 |
| bounded fast trace only | 1.0000 | 0.0000 | 1,616 |
| **fast trace -> slow state** | **1.0000** | **1.0000** | **6,566 ± 1.9** |

The oracle remains the best answer when relevance is knowable at write time.

The hybrid buys something only when **future relevance is delayed**: retain a
bounded high-detail trace until the system discovers what mattered, then
project only that fragment into persistent state.

See [results/GATE2.md](results/GATE2.md).

## Gate 3 — content-addressed temporal-context reinstatement — BUILT

A noisy re-encountered cue must recover information about the events that
surrounded its original encounter.

The fast mechanism is intentionally ordinary episodic attention:

```math
alpha = softmax(beta q K^T)

reinstated_context = alpha V.
```

Ten-seed receipt:

| system / quantity | result |
|---|---:|
| cue-only held-out decoder | -0.0016 ± 0.0048 cosine |
| **episodic reinstatement** | **0.9299 ± 0.0059** |
| shuffled temporal links | 0.0022 ± 0.0108 |
| random index | 0.0047 ± 0.0062 |
| correct E1 top-1 identity | 0.9218 ± 0.0078 |
| correct-index weight vs reinstatement | **r = 0.8415 ± 0.0146** |

This is not a hippocampus model.  It demonstrates the architectural role of a
persistent episodic index: a cue can retrieve its old temporal neighborhood
even when no population-level cue->context law exists.

See [results/GATE3.md](results/GATE3.md).

## Gate 4 — learn what to remember — BUILT

Slow memory gets only 20 slots for every 100 experiences.  Future relevance is
unknown at write time but statistically predictable from current features.

| write policy | later-needed events retained |
|---|---:|
| random | 0.1991 ± 0.0022 |
| current salience | 0.3964 ± 0.0031 |
| **learned future-value gate** | **0.4874 ± 0.0029** |
| oracle probability ranking | 0.4887 ± 0.0027 |
| full memory | 1.0000 |

So selection quality can matter independently of raw memory size.  The learned
gate preserves about 2.45x as many later-needed events as random retention
under the same 20% capacity.

See [results/GATE4.md](results/GATE4.md).

## Gate 5 — policy blindness under drift — BUILT

Gate 4's learned filter received later relevance outcomes for every candidate,
including discarded events.  Gate 5 makes that feedback selective and changes
the relevance law without announcement.

The old best group remains moderately useful while a previously poor,
unobserved group becomes optimal.  A greedy continuously updated gate never
discovers it in any of 40 seeds.

| 20-slot policy | pre-shift recall | late post-shift recall | recovery |
|---|---:|---:|---:|
| greedy selected-feedback | 0.6831 ± 0.0066 | 0.2805 ± 0.0057 | 0/40 |
| fixed four-slot reserve | 0.5671 ± 0.0068 | 0.4815 ± 0.0056 | 40/40 |
| **surprise-triggered reserve** | **0.6821 ± 0.0089** | **0.5702 ± 0.0055** | **40/40** |
| **fast trace** | **0.6831 ± 0.0066** | **0.5702 ± 0.0055** | **40/40** |
| oracle probability ranking | 0.6831 ± 0.0066 | 0.5702 ± 0.0055 | 40/40 |

The triggered policy reaches oracle late recall with only 0.615 exploratory
writes per block on average, versus four for the fixed reserve.  The fast trace
recovers much faster but temporarily stores the 80 discarded event identities.

This earns a precise boundary: online learning cannot recover a value change in
a region from which its own policy obtains no identifying feedback.  Some
exploration, side information, or surviving trace is necessary.

In a stricter second world the selected group's outcomes do not change at all
while an ignored group becomes better.  The surprise-triggered policy then
fails in 39/40 seeds; its one recovery is an accidental false-alarm burst.
Fixed reserve and fast trace recover in 40/40.  Surprise can reveal that the
known region deteriorated; it cannot reveal an improvement it never observes.

See [results/GATE5.md](results/GATE5.md).

## Gate 6 — active sensing under delayed audit — BUILT

Prediction failure has at least three meanings:

```text
my model is wrong         -> LEARN
my observation is weak    -> SENSE
the world should change   -> ACT
```

Force these operations to have different costs and consequences.  The
AlgoSchalgo observability results and Gate-1 illegal-predictor failure become
attackers.

Gate 6 gives each trial a hidden binary target, a visible context, and a free
cue whose reliability depends on context.  A `0.90`-reliable probe costs `0.08`
utility.  After 6,000 trials the two contexts exchange free-cue reliabilities;
action correctness arrives 12 trials late.

| policy | late accuracy | late probe rate | late net utility |
|---|---:|---:|---:|
| no sense | `0.7256 ± 0.0099` | `0.0000` | `0.7256 ± 0.0099` |
| always sense | `0.9008 ± 0.0070` | `1.0000` | `0.8208 ± 0.0070` |
| learned, no delayed trace | `0.9008 ± 0.0070` | `1.0000` | `0.8208 ± 0.0070` |
| **learned + 12-trial trace** | **`0.8989 ± 0.0066`** | **`0.5415 ± 0.0225`** | **`0.8556 ± 0.0067`** |
| zero-delay learner | `0.8990 ± 0.0063` | `0.5394 ± 0.0217` | `0.8558 ± 0.0064` |
| oracle sensing threshold | `0.8989 ± 0.0065` | `0.4996 ± 0.0106` | `0.8589 ± 0.0064` |

The delayed-trace learner reallocates sensing after the shift in all 40 seeds
and nearly matches both the zero-delay learner and oracle.  Its peak bill is 12
pending records × five scalars = 60 temporary scalars.  The no-trace learner
cannot attach the delayed audit to the old context/cues/action and remains at
its conservative always-probe initialization.

See [results/GATE6.md](results/GATE6.md).

## Gate 7 — sparse body topology

Test the candidate slow-body × fast-conductance factorization under equal
persistent edge budget.  Compare local, random, small-world,
dyadic/fractal-like, and learned sparse supports.

## Gate 8 — collective operator

Use the supplied "instanton" lesson carefully: a low-dimensional apparent
agent can be a collective coordinate of distributed deterministic dynamics,
not a privileged executive particle.  Ask whether a population order parameter
can control a task more robustly than a random projection or hand-designed
executive.

## Gate 9 — representation drift

Let the child's own internal coordinates become plastic. Test whether persistent
address, temporal identity, and path memory preserve old function without
global retraining.

## Attackers

Explicit Markov/state-space models, reservoir + linear readout, GRU/RNN, small
SSM, tiny transformer, ordinary associative memory, and matched active-sensing
controllers are all allowed to win.

# Scaling

Gate 0 has `N=24`, radius `r=2`, and 11 parameters per cell. Its persistent predictive structure is local, scaling roughly as `O(Nr)`. If useful tasks demand arbitrary personal all-to-all relationships, the structure approaches `O(N²)` and the comparison with attention changes substantially.

# Run

```bash
python -m pip install -r requirements.txt
python experiments/gate0_local_next_state.py
python experiments/gate1_illegal_predictor.py
python experiments/gate2_delayed_relevance.py
python experiments/gate3_temporal_context_reinstatement.py
python experiments/gate4_learn_what_to_remember.py
python experiments/gate5_policy_blindness_under_drift.py
python experiments/gate6_active_sensing_delayed_audit.py
python -m unittest discover -s tests -v
```

The experiment scripts write JSON receipts under `results/`. `index.html` is a
cumulative browser map with Gate 0 and later interactive illustrations, not a
replacement for the canonical experimental receipts.

# Lineage

- [T-800NNP](https://github.com/anttiluode/T-800NNP) — continuously running receiver state changes routing.
- [KyberDyyni1](https://github.com/anttiluode/KyberDyyni1) — fast computation versus slow consolidation and temporal correspondence.
- [MovingProblem](https://github.com/anttiluode/MovingProblem) — identity under moving representations and path memory.
- [AlgoSchalgo](https://github.com/anttiluode/AlgoSchalgo) — observability, ambiguity, and active measurement.
- [Tuesday](https://github.com/anttiluode/Tuesday) — temporal structure as a source of identifiability.
- [GeoNeuronX](https://github.com/anttiluode/GeoNeuronX) — history materialized into receiver state before readout.

## Working architecture notes

- [MATH.md](MATH.md) — stateful operator equations and the endogenous-prediction problem.
- [BODY_ATTENTION.md](BODY_ATTENTION.md) — candidate **slow body × fast conductance** factorization of attention.
- [PAPERS.md](PAPERS.md) — neuroscience / ML references that constrain rather than validate the design.
- [INSTANTON_NOTE.md](INSTANTON_NOTE.md) — why the green “particle” is actually a collective coordinate of a nonlinear field.

The repo name is allowed to remain embarrassing until the machine earns a better one.
