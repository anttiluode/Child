# Gate 3 — temporal context reinstatement

## Motivation

Zou, Hutchinson & Kuhl (bioRxiv, 2025), *Hippocampal-guided reconstruction
of an event's prior temporal context*, report that when a natural scene was
re-encountered, high-level visual cortex (LOTC) reinstated information about
the visual scenes that had surrounded the original encounter.  Greater
E1-E2 pattern similarity in hippocampal CA1 and CA2/3/DG predicted stronger
LOTC temporal-context reconstruction.

The biological result is not reproduced here.

Gate 3 asks for the smallest computational analogue:

> Can a re-encountered event act as a content address that reinstates the
> temporal neighborhood of its earlier occurrence?

## World

At the original encounter E1, each episode is:

```text
random visual context A
        ↓
      CUE
        ↓
random visual context B
```

The two context vectors are independent of the cue across episodes.

At E2, the same cue is presented again with representational noise.

The required output is the normalized average of A and B: the old temporal
context.

## Systems

### Slow cue-only decoder

Fit a ridge map:

```math
cue -> temporal_context
```

on half the episodes and evaluate on unseen episodes.

Because the cue/context association is episodic and random, there is no
population law to generalize.

### Fast episodic index

Store E1 pairs:

```text
key   = cue
value = old adjacent context
```

At E2:

```math
alpha
  = softmax(beta q K^T)

reinstated_context
  = alpha V.
```

This is ordinary content-addressed attention over episodic memory.

### Shuffled temporal link

Preserve the exact same query/key matching but shuffle which temporal-context
value is attached to each key.

### Random index

Preserve the memory but destroy correspondence between the E2 cue and its E1
index.

## 10-seed receipt

| quantity | result |
|---|---:|
| cue-only held-out context cosine | `-0.0016 ± 0.0048` |
| **episodic reinstatement cosine** | **`0.9299 ± 0.0059`** |
| shuffled temporal link | `0.0022 ± 0.0108` |
| random index | `0.0047 ± 0.0062` |
| correct episodic identity, top-1 | `0.9218 ± 0.0078` |
| mean attention weight on correct E1 | `0.8256 ± 0.0081` |
| correct-index weight vs reinstatement cosine | **`r = 0.8415 ± 0.0146`** |

## Earned statement

> **When a repeated event contains enough information to recover a persistent
> episodic index, content-addressed fast memory can reinstate information from
> that event's old temporal neighborhood even when no cue-to-context mapping
> generalizes across episodes.**

The association itself matters: shuffling the temporal links destroys
reinstatement while leaving query/key similarity untouched.

## Important non-result

This is not evidence that the synthetic index is a hippocampus or that the
value bank is visual cortex.

Mathematically, the positive mechanism is simply attention / kernel retrieval
over episodic key-value memory.

That is useful here precisely because it gives a clean bridge:

```text
Transformer attention
    searches current token/context representations

episodic attention
    searches persistent encounter representations
```

The architectural question is where that memory lives and how long it
persists.

## Relation to Gate 2

Gate 2 showed:

```text
keep high-detail recent material
until later relevance becomes known
        ↓
consolidate selected fragment
```

Gate 3 removes Gate 2's exact case address.

A later cue must recover the earlier episode from content similarity.

So the fast system now has two jobs:

1. preserve unresolved recent detail;
2. provide a persistent/content address by which later activity can recover it.

## Relation to rhythmic replay

Gate 3 retrieves an old temporal neighborhood in one matrix operation.

It does **not** yet implement forward or backward replay.

The next honest attack is to replace the stored context value with locally
linked episode states:

```text
... E1-2 <-> E1-1 <-> E1 <-> E1+1 <-> E1+2 ...
```

Then a recovered E1 index must launch a trajectory through those links to
reconstruct its context.

Compare that recurrent traversal with direct key-value retrieval.

If direct retrieval is cheaper and equally capable, rhythmic replay has not
earned an engineering role.

## Not earned

- a biological hippocampal index;
- hippocampus -> cortex causality;
- cortical engrams;
- replay;
- pattern separation;
- memory consolidation;
- a new attention mechanism.

## Paper

Futing Zou, J. Benjamin Hutchinson, Brice A. Kuhl.
*Hippocampal-guided reconstruction of an event's prior temporal context.*
bioRxiv 2025.08.05.668710, version 2 (2025).
