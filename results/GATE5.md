# Gate 5 — policy blindness under drift

## Question

Can a selective memory system discover that its own write policy has become
wrong when it receives outcome feedback only for the events it retained?

This gate exists because Gate 4 contained a generous assumption: after choosing
20 of 100 events, its relevance learner was updated from the later outcomes of
**all 100**.  Gate 5 makes access to that feedback explicit and costly.

## The identifiability wall

Let `g` be an observable event group, `y` its later relevance, and `a` the
memory decision:

```math
a \in \{0,1\}.
```

Without a surviving address/feature trace, relevance feedback is observed only
when the event was retained:

```math
y \text{ observed only if } a=1.
```

If a deterministic policy never retains group `g=1`, then two worlds can agree
on every selected event while differing arbitrarily on:

```math
P(y=1 \mid g=1).
```

They induce the same observed feedback stream.  No better learning rate or
larger predictor can identify which world it inhabits from those observations.

The system must obtain off-policy evidence through at least one of:

```text
nonzero exploration / reserve
        OR
side information
        OR
a trace that survives until feedback arrives.
```

## World

Every block contains 100 candidates: 20 events from each of five visible
groups.  Slow memory has 20 slots.

During a 40-block full-feedback warmup, and for the next 160 constrained
blocks, future relevance probabilities are:

```text
old world: [0.75, 0.12, 0.10, 0.08, 0.06]
```

The best policy is to retain group 0.

Then, without announcement, the law becomes:

```text
visible-drop world: [0.45, 0.90, 0.10, 0.08, 0.06]
```

Group 0 remains moderately valuable.  That detail is crucial: a greedy policy
continues receiving enough success to prefer its familiar region, while the
previously poor and now optimal group 1 remains unobserved.

A second, stricter shift changes only the unselected group:

```text
strictly hidden world: [0.75, 0.90, 0.10, 0.08, 0.06]
```

For a policy that keeps selecting group 0, the old world and strictly hidden
world generate exactly the same feedback distribution.  Surprise about the
selected support therefore cannot systematically detect this change.

Feedback arrives one block after the write decision.  Value estimates are
transparent exponentially weighted group means.

## Policies

### Frozen

Keep using the old value estimates.

### Greedy selected-feedback

Continuously update, but only from retained events.  This tests whether
"online learning" alone solves the blindness.

### Fixed reserve

Use 16 slots greedily and four slots for uniform random exploration on every
block.  It keeps nonzero observation support everywhere, but permanently gives
up 20% of its exploitation capacity.

### Surprise-triggered burst

Normally use all 20 slots greedily.  If four consecutive retained-event blocks
underperform their pre-change reference by at least 0.15, spend 12 of 20 slots
on random exploration for 20 blocks, then return to exploitation.

This detector is hand-designed for abrupt negative change.  It is a resource
schedule, not a general change-point algorithm.

### Fast trace

Use all 20 slow slots greedily, but temporarily retain the other 80 event
identities for the one-block feedback delay.  The value estimator therefore
gets full feedback and can update before the next selection.

### Controls

- random 20-slot memory;
- 20-slot oracle that knows the current relevance probabilities;
- full 100-slot memory.

## Forty-seed receipt

Recall is the fraction of all later-relevant events that the policy retained.
Recovery is the first completed eight-block post-shift window with at least 75%
of slow slots allocated to the newly valuable group.

| policy | pre-shift recall | first 25 post-shift | last 80 post-shift | recovered | recovery blocks |
|---|---:|---:|---:|---:|---:|
| frozen | `0.6831 ± 0.0066` | `0.2789 ± 0.0096` | `0.2805 ± 0.0057` | 0/40 | — |
| greedy selected-feedback | `0.6831 ± 0.0066` | `0.2789 ± 0.0096` | `0.2805 ± 0.0057` | 0/40 | — |
| fixed four-slot reserve | `0.5671 ± 0.0068` | `0.3832 ± 0.0384` | `0.4815 ± 0.0056` | 40/40 | `19.3 ± 4.2` |
| **surprise-triggered reserve** | **`0.6821 ± 0.0089`** | `0.3037 ± 0.0209` | **`0.5702 ± 0.0055`** | **40/40** | `31.2 ± 4.4` |
| **fast trace** | **`0.6831 ± 0.0066`** | **`0.5564 ± 0.0107`** | **`0.5702 ± 0.0055`** | **40/40** | **`8.0 ± 0.0`** |
| random | `0.2010 ± 0.0066` | `0.2044 ± 0.0125` | `0.1981 ± 0.0071` | 0/40 | — |
| oracle 20-slot | `0.6831 ± 0.0066` | `0.5703 ± 0.0110` | `0.5702 ± 0.0055` | 40/40 | `8.0 ± 0.0` |
| full 100-slot | `1.0000` | `1.0000` | `1.0000` | n/a | n/a |

The recovery-block numbers above include the eight blocks required to certify
the recovery window.

## Strictly off-policy shift

| policy | first 25 post-shift | last 80 post-shift | recovered | recovery blocks |
|---|---:|---:|---:|---:|
| frozen | `0.3970 ± 0.0084` | `0.3961 ± 0.0045` | 0/40 | — |
| greedy selected-feedback | `0.3970 ± 0.0084` | `0.3961 ± 0.0045` | 0/40 | — |
| fixed four-slot reserve | `0.3550 ± 0.0095` | `0.4095 ± 0.0050` | 40/40 | `34.3 ± 6.5` |
| surprise-triggered reserve | `0.3953 ± 0.0146` | `0.3983 ± 0.0137` | **1/40** | `26.0` |
| **fast trace** | **`0.4674 ± 0.0088`** | **`0.4758 ± 0.0048`** | **40/40** | **`8.5 ± 0.6`** |
| random | `0.2021 ± 0.0109` | `0.1994 ± 0.0069` | 0/40 | — |
| oracle 20-slot | `0.4765 ± 0.0073` | `0.4782 ± 0.0040` | 40/40 | `8.0 ± 0.0` |
| full 100-slot | `1.0000` | `1.0000` | n/a | n/a |

The surprise policy's one recovery is a false-alarm exploration burst, not
detection of the hidden improvement.  Under this scenario its observations are
distributed exactly as they would be if no off-policy improvement had occurred.
The experiment therefore shows both sides:

```text
on-policy deterioration
    can trigger temporary exploration

strictly off-policy improvement
    cannot trigger from selected feedback alone
```

## Resource receipt

| policy | persistent slots | mean exploratory writes / block | extra temporary trace slots |
|---|---:|---:|---:|
| greedy selected-feedback | 20 | 0 | 0 |
| fixed reserve | 20 | 4.000 | 0 |
| **surprise-triggered reserve** | **20** | **0.615 ± 0.094** | **0** |
| **fast trace** | **20** | **0** | **80** |

The triggered policy uses about **6.50x fewer exploratory writes** than the
fixed reserve, retains essentially all of the pre-shift performance, and
eventually reaches the 20-slot oracle's late recall.  Its cost is slower
recovery.

The fast trace recovers almost immediately and also reaches the oracle, but it
temporarily preserves all 100 event identities—80 more than slow memory.

This is the actual trade:

```text
continuous reserve
    cheap storage, permanent opportunity cost, quick discovery

triggered reserve
    cheap storage, low average opportunity cost, slower discovery

fast trace
    high temporary storage, nearly immediate discovery
```

## Earned statement

> **A learned memory gate can become self-sealing under distribution shift:
> continuous parameter updating is insufficient when the policy receives no
> identifying feedback from the region it has learned not to retain.  Recovery
> requires observation support supplied here by persistent exploration,
> surprise-triggered exploration, or a temporary full-feedback trace.**

## Why this matters for Child

Gate 2 said temporary detail helps when relevance is delayed.

Gate 4 said a learned filter spends scarce slow memory better when relevance is
predictable.

Gate 5 now shows their tension:

```text
successful slow filter
        ↓
efficient compression
        ↓
world changes outside selected support
        ↓
compression becomes an observation mask
        ↓
reserve / trace restores evidence
        ↓
slow filter can change
```

So imperfect efficiency is not merely waste.  It can be the price of retaining
the ability to discover that the efficient policy is wrong.

## Attackers and prior territory

This toy is a transparent instance of selective-label / contextual-bandit
feedback, not a new theorem.  Selective experience replay, reservoir sampling,
change-point detection, propensity-weighted estimation, and contextual-bandit
exploration are stronger established formulations.

The point is architectural: Gate 4's full feedback had to live somewhere.
Gate 5 makes the machine pay either in slow write opportunities or temporary
trace capacity.

## Not earned

- a general distribution-shift detector;
- optimal exploration scheduling;
- a model of hippocampal novelty or dopamine;
- recovery when relevance arrives after the fast trace expires;
- a deterministic guarantee of finding strictly off-policy change with finite
  random reserve;
- high-dimensional representation learning;
- novelty over selective labels, bandits, or continual-learning replay.

The strictly hidden scenario confirms that a surprise detector receives no
information-bearing trigger when group 0 remains exactly as rewarding.  Only
persistent reserve, side information, or full-support trace can systematically
reveal that improvement.

## Next attack

Remove the hand-given five-group coordinate and the conveniently detectable
drop on the selected support.

Use a drifting high-dimensional stream in which:

- useful regions must be learned rather than named;
- some shifts are visible on-policy and some occur entirely off-policy;
- feedback delay can exceed the fast-trace horizon;
- matched contextual-bandit and change-point methods are allowed to win.

Only after that should Child claim that its fast/slow memory architecture adds
anything beyond ordinary exploration with bookkeeping.
