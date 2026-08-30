# Gate 4 — learn what to remember

## Question

Can a slow-memory write gate learn **which present experiences are likely to
matter later**, rather than storing everything or relying on one current
salience signal?

This is the computational version of the user's observation that a capable
memory system may differ partly in what it successfully selects for durable
retention.

It is not a model of intelligence, attention, or human memory capacity.

## World

Every block contains 100 candidate experiences.

Slow memory may retain only 20.

So all constrained systems receive the exact same 20% persistence budget.

At write time, each event exposes six observable features.  Future relevance
is probabilistic and depends on a hidden combination of those features.

Only later is it revealed whether that event was queried.

## Systems

### Random

Keep 20 arbitrary events.

### Current salience

Keep the 20 events with the largest value of one immediately salient feature.

That feature really is related to future relevance, so this is a meaningful
heuristic rather than a straw man.

### Learned future-value gate

Predict:

```math
p_i(t)
  = P(event i will matter later | features_i)
```

and retain the 20 highest-scoring events.

Only after the retention decision do future query outcomes arrive.  They update
the relevance model for subsequent blocks.

### Oracle

Knows the true future query **probability**, but not which stochastic queries
will actually happen.

Same 20-event budget.

### Full memory

Stores all 100 events.

## 20-seed receipt

| policy | fraction of future queries recoverable |
|---|---:|
| random 20% memory | `0.1991 ± 0.0022` |
| current salience heuristic | `0.3964 ± 0.0031` |
| **learned future-value gate** | **`0.4874 ± 0.0029`** |
| oracle probability ranking | `0.4887 ± 0.0027` |
| full memory | `1.0000` |

The learned gate nearly reaches the oracle under the same fixed memory budget.

It preserves about **2.45x** as many later-needed events as random retention
while storing exactly the same number of events.

## Earned statement

> **When future relevance has predictable structure, a bounded memory system
> can learn a write priority from delayed outcomes and use the same physical
> memory capacity substantially better than random retention or a simpler
> current-salience heuristic.**

## Attention is not memory

This gate deliberately separates:

```text
ATTENTION
    allocate processing now

MEMORY WRITE PRIORITY
    estimate expected future value of preserving this event
```

One signal may influence both, but the objectives differ.

A highly attended event can still be useless later.

A weakly salient event can become important after subsequent experience.

## Mathematical object

For event trace `e_t` and slow-memory budget `B`, define a learned value

```math
V_phi(e_t)
  =
  E[
    future utility of retaining e_t
    |
    current observable state
  ].
```

The write operation becomes a constrained selection problem:

```math
choose M_t subset E_t

such that |M_t| <= B

to maximize
    sum_(e in M_t) V_phi(e).
```

Delayed outcomes train `V_phi`.

This is ordinary value prediction / caching-style resource allocation, not a
new learning principle.

## Relation to Gate 2

Gate 2 handled the hard case where future relevance was **not knowable** at
write time:

```text
uncertain relevance
    -> preserve temporary detail
    -> wait
    -> later select
```

Gate 4 handles the complementary case:

```text
relevance is uncertain
but statistically predictable
    -> learn what tends to deserve consolidation
```

A practical memory system likely needs both.

## Relation to forgetting

Forgetting is now a positive operation rather than merely failure.

Under finite capacity, forgetting is the complement of allocation:

```text
retain high expected future value
        +
release low expected future value
        =
keep plastic capacity available
```

The next serious problem is distribution shift: a gate trained on yesterday's
relevance law may confidently throw away exactly what becomes important
tomorrow.

## Not earned

- smarter people have this exact mechanism;
- dopamine implements this value equation;
- attention equals consolidation;
- biological hippocampal selection;
- optimal continual-memory allocation;
- novelty over learned caching / value prediction.

## Next attack

Change the future-relevance law halfway through the stream.

Compare:

- frozen memory-value predictor;
- continuously adaptive predictor;
- predictor with exploration/reserve;
- temporary fast trace that rescues low-valued events until the new relevance
  law is discovered.

This joins forgetting, plasticity, and the Gate-2 fast trace in one experiment.

## Hidden assumption exposed by Gate 5

This implementation updates the learned value gate from the later outcomes of
all 100 candidate events, including the 80 it did not retain.

That is legitimate full feedback, but it means the experiment does not pay for
the event address/features needed to interpret feedback about discarded items.
Gate 5 removes that assumption.  When outcomes are visible only for retained
events, a greedy adaptive gate can become self-confirming under distribution
shift; reserve sampling or a temporary trace is needed to restore observation
support.
