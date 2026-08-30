# Gate 2 — delayed relevance: fast trace before slow state

## Why this gate exists

The new SKILL.state result makes an important systems point: long-horizon agents
do not necessarily need to replay an ever-growing textual trajectory.  If the
current structured execution state is a sufficient statistic, transient
reasoning/history can be discarded.

Its own limitation is equally important:

> an earlier observation may become relevant only later, after its relevance
> was not recognized and therefore was never committed to explicit state.

Gate 2 isolates exactly that wall.

This is an engineering experiment, not a hippocampal model.

## World

Each of 5,000 cases arrives with 16 random scalar fields.

At encoding time there is **no information** about which field will matter.

20–80 steps later, a query reveals the relevant field and asks for its value.

220–420 steps after the original case, the same fact is queried again.

The fast buffer expires after 100 steps, so it can answer the first query but
cannot survive to the repeat query.

## Systems

### Full history

Store all 16 fields forever.

This is the append-only-history upper bound.

### Early structured state

Persistent memory is restricted to one scalar per case.  It must choose that
scalar before future relevance is known.

### Oracle structured state

Same one-scalar budget, but it is told the future relevant field when the case
arrives.

This is the correct positive control: explicit state is excellent when the
sufficient statistic is knowable at write time.

### Fast only

Keep the raw 16-field episode in a bounded 100-step buffer.  It solves delayed
relevance while the episode is still present, but deliberately never
consolidates.

### Hybrid fast -> slow

Keep the raw episode only in the bounded fast buffer.

When the first query finally reveals which field matters, retrieve that field
and write **only that one scalar** into persistent slow state.

Then discard the raw episode normally.

## 10-seed receipt

| system | first delayed query | later repeat | peak stored scalars |
|---|---:|---:|---:|
| full history | **1.0000** | **1.0000** | 80,000 |
| early structured state | 0.0652 | 0.0652 | 5,000 |
| oracle structured state | **1.0000** | **1.0000** | 5,000 |
| fast trace only | **1.0000** | 0.0000 | 1,616 |
| **fast trace -> slow state** | **1.0000** | **1.0000** | **6,566** |

Chance that the early one-field state happened to retain the later-requested
field is 1/16 = 0.0625, matching the receipt.

The hybrid stores about 12.2x fewer scalar values at peak than full history in
this finite run while retaining perfect first and repeat retrieval.

As the number of completed cases grows, the bounded fast-buffer overhead stays
constant and the asymptotic storage ratio approaches the 16x raw-to-selected
field ratio of this toy.

## Earned statement

> **When future relevance is not knowable at observation time, a bounded
> high-detail fast trace can preserve experience until relevance is revealed;
> the now-relevant fragment can then be projected into a much smaller
> persistent state.**

This is the computational seam behind the hippocampal analogy:

```text
experience arrives
      ↓
fast, high-detail, temporary addressable trace
      ↓
later event reveals what mattered
      ↓
selective consolidation
      ↓
slow persistent state
```

## Crucial negative / attacker

The oracle structured state uses less memory than the hybrid and is perfect.

Therefore the hybrid is **not** intrinsically superior to explicit state.

It buys something only when:

```text
relevance is delayed
AND
the machine cannot know the sufficient statistic at write time.
```

## Relation to SKILL.state

SKILL.state is a runtime-level architecture, whereas Gate 2 is a tiny synthetic
memory calculation.

The point of contact is its sufficient-statistic assumption.

A sensible agent architecture may therefore need both:

```text
canonical current state
    for what is already known to matter

bounded episodic trace
    for recent material whose relevance is not settled yet
```

That is different from retaining all reasoning forever.

## Not earned

- biological hippocampal replay;
- cortical consolidation dynamics;
- replay during sleep;
- a new memory algorithm;
- semantic compression;
- optimal forgetting;
- a claim that 100 steps is the right buffer horizon.

## Next attack

Do not give the fast trace an exact case address.

Make retrieval content-based and interference-prone, then ask whether
temporal/rhythmic structure or pattern separation buys anything over ordinary
associative memory.
