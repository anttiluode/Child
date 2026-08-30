# Papers that constrain Child

This is a map of what the papers actually motivate.  They do not validate the
software equations.

## Aizenbud et al. — dendritic morphology and synaptic nonlinearities

Ido Aizenbud et al. (PNAS, 2026), *Dendritic morphology and synaptic
nonlinearities enhance functional complexity in human cortical neurons*.

Useful constraint:

```text
single-neuron computation is not exhausted by one scalar weighted sum

dendritic extent / branching
    -> compartmentalization and larger input surface

NMDA nonlinearities
    -> nonlinear local integration

morphology + biophysics
    -> measurable differences in I/O complexity
```

Child takes only the architectural suggestion that one unit may contain several
partly independent stateful computations before output.

It does not infer that biological dendrites minimize next-state prediction
error.

## Leterrier — axon initial segment

Christophe Leterrier (J. Neurosci., 2018), *The Axon Initial Segment: An
Updated Viewpoint*.

Useful constraint:

```text
somatodendritic computation
        ↓
distinct AIS boundary
        ↓
regulated action-potential generation
        ↓
axon / downstream targets
```

AIS composition and position affect excitability and can adapt to developmental
and physiological conditions.

Child uses this only to justify separating:

```text
internal predictive state
from
output/transmission control
```

Gate 1's stochastic transmit/hold variable is not a biological AIS model.

## Francioni et al. — vectorized instructive signals in dendrites

Valerio Francioni et al. (Nature, 2026), *Vectorized instructive signals in
cortical dendrites*.

The especially relevant result for Child is that dendritic activity in a
neurofeedback task contained reward/error-related information whose sign
depended on the causal role of individual neurons, predicted learning-related
activity changes, and whose targeted perturbation impaired learning.

That is important because it attacks an over-simple dichotomy:

```text
ANN:
    vector-valued backpropagation

brain:
    one global scalar reward
```

The biological story may include neuron-specific / compartment-specific
instructive information without literally implementing reverse-mode
backpropagation.

Child should therefore not assume that all slow learning receives only one
broadcast scalar consequence.

## Yamada & Chao — predicting what and when

Yohei Yamada and Zenas C. Chao (Communications Biology, 2026), *Joint encoding
of "what" and "when" predictions through error-modulated plasticity in
biologically plausible spiking networks*.

Their recurrent spiking model uses an error-modulated, attention-gated Hebbian
rule and learns event identity, timing, and probability in one population.

This is uncomfortably close to Child's motivating seam and should be treated as
an attacker/reference, not ignored.

Important difference to test rather than proclaim:

```text
their model:
    recurrent spiking population + local error-gated learning

Child:
    persistent local addresses
    + explicitly separated output gate
    + later active observation / slow structural memory
```

If later Child gates reduce to the Yamada-Chao mechanism, their formulation
wins.

## TTT layers

Yu Sun et al. (ICML, 2025), *Learning to (Learn at Test Time): RNNs with
Expressive Hidden States*.

TTT makes the recurrent hidden state itself a trainable model updated by
self-supervised learning during the test sequence.

This is a direct mainstream reference for the idea:

```text
hidden state
    need not be a passive vector;
    it can itself be an online learner
```

Child's potential distinction is locality, persistent physical-style
addressability, and the separation of fast state / output gate / slow
structure.  Those distinctions must earn measurable value.


## Badhe, Tiwari & Chung — SKILL.state

Sanket Badhe, Priyanka Tiwari and Jonghyun Chung (arXiv 2608.26263, accepted
at EMNLP 2026), *SKILL.state: Scalable Long-Horizon Agent Skills*.

The runtime replaces append-only conversational history with a mutable
structured execution state.  At step t the model receives only:

```math
(P, Sigma_t, O_t)
```

where P is the immutable skill specification, Sigma_t the persistent execution
state, and O_t the latest observation.  Intermediate reasoning is discarded
after a validated state update.

This is a strong engineering precedent for the Child idea that operational
state need not be reconstructed from the whole textual trajectory.

The paper also states the limitation that directly motivates Gate 2: explicit
state is lossless only when it can be made a sufficient statistic.  It can fail
when the relevance of an old observation was not recognized when that
observation arrived and therefore was never committed to state.

Child Gate 2 tests the smallest complement:

```text
canonical structured state
        +
bounded high-detail fast trace
        ↓
delayed relevance
        ↓
selective fast -> slow consolidation
```

This does not make Gate 2 a competitor to SKILL.state.  It attacks one of the
paper's declared boundary conditions.

## Fractal morphology — useful caution

Fractal geometry is relevant to dendritic arbors, but Child should not turn
"fractal" into a magic complexity scalar.

Aizenbud et al. did not establish that greater fractal dimension causes greater
functional complexity.  In their measured feature set, total dendritic area was
the strongest single predictor of FCI, while branch allocation/path extent
also mattered strongly; raw branch count was much weaker.

Separate work by Smith et al. (Scientific Reports, 2021) and Rowland et al.
(Frontiers in Network Physiology, 2023) treats limited-range fractal dimension
as an integrated measure of dendritic length, forking, weaving, space filling,
connectivity, and construction/operating cost.

The Child translation worth testing is therefore not:

```text
higher fractal dimension -> smarter
```

but:

```text
multiscale sparse support
    -> more useful reach across scales per persistent edge budget?
```

A future equal-budget topology gate should compare local, random, small-world,
dyadic/fractal-like, and learned sparse supports before attributing any benefit
to fractality.


## Zou, Hutchinson & Kuhl — hippocampal-guided temporal-context reconstruction

Futing Zou, J. Benjamin Hutchinson & Brice A. Kuhl, *Hippocampal-guided
reconstruction of an event's prior temporal context*, bioRxiv
2025.08.05.668710 (version 2, 2025).

This human 7T fMRI preprint used repeated natural scenes and inverted encoding
models.  On re-encountering a scene, lateral occipitotemporal cortex (LOTC)
contained information about visual scenes that had surrounded the original
encounter.

E1-E2 pattern similarity in hippocampal CA1 and CA2/3/dentate gyrus predicted
the strength of LOTC temporal-context reconstruction; the reverse LOTC->MTL
analysis was not significant.

Child takes only the structural question:

```text
current cue
    -> recover persistent episodic index
    -> reinstate information that was temporally adjacent before
```

Gate 3 implements this with ordinary key-value attention.  It does not claim
hippocampal biology or causal directionality.
