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

## von Hünerbein et al. — dendritic structure as learning machinery

Ben von Hünerbein et al. (arXiv:2608.23251, 2026), *Dendritic structure enables
powerful plasticity*.

This new review/preprint sharpens the reason compartments might matter.  Its
claim is not merely that dendrites expand a neuron's representational function;
compartmentalization can give synapses faster and more specific access to
locally differentiated variables from which error-like learning signals can be
constructed.

That is directly relevant to Child's candidate separation:

```text
local compartment state
    +
cell/branch-specific instruction
        ↓
more addressed slow plasticity than
one Hebbian pair times one global scalar
```

It is a review and architectural argument, not evidence that
`slow body × fast conductance` is a useful AI implementation.  A point model or
ordinary backpropagation remains the engineering attacker.

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

## Selective labels — the policy controls which outcomes become knowable

Himabindu Lakkaraju et al. (KDD, 2017), *The Selective Labels Problem:
Evaluating Algorithmic Predictions in the Presence of Unobservables*.

Their application is human/algorithmic decision evaluation, not memory.  The
relevant mathematical warning is general: outcomes may be observed only for
cases selected by the historical policy, so the labeled set does not identify
performance on the unselected population.

Gate 5 uses a small memory analogue:

```text
discard event address
        ↓
later outcome cannot update the value of that event region
        ↓
write policy shapes its own future training support
```

This is established selective-feedback territory.  Child's contribution is
only to make Gate 4 pay for the trace or exploration that its full-feedback
update had assumed.

## Isele & Cosgun — selective experience replay

David Isele and Akansel Cosgun (AAAI, 2018), *Selective Experience Replay for
Lifelong Learning*.

They compare long-term replay selection by surprise, reward, distribution
matching, and coverage.  Distribution matching was the most consistently
successful strategy in their tested domains.

This is an important attacker for Child's learned memory value:

```text
highest predicted future value only
    may collapse observation coverage

distribution / coverage reserve
    may preserve adaptation under change
```

Gate 5's four random reserve slots are deliberately cruder than established
selective replay methods.

## Sun et al. — information-theoretic online memory selection

Shengyang Sun et al. (ICLR, 2022), *Information-theoretic Online Memory
Selection for Continual Learning*.

The paper proposes surprise and learnability criteria and a stochastic
information-theoretic reservoir sampler, emphasizing both which stream items
are informative and when memory should update.

This attacks any suggestion that Gate 5 invented surprise-triggered memory or
stochastic reserve.  The remaining Child question is architectural: when
outcome relevance is delayed, which information must survive in a fast trace so
that a stronger established memory-selection rule can be updated at all?

## Bajcsy — active perception

Ruzena Bajcsy (Proceedings of the IEEE, 1988), *Active Perception*, DOI
[`10.1109/5.5968`](https://doi.org/10.1109/5.5968).

The durable constraint is that sensing is not merely a passive input channel.
An agent can control how it samples the world, so observation choice belongs
inside the decision problem and must carry a cost.

Gate 6 is a very small value-of-observation instance:

```text
free ambiguous cue
        ↓
buy another observation?
        ↓
act
        ↓
delayed external audit
```

It does not reproduce Bajcsy's geometric, robotic, or control-theoretic
framework.

## Gottlieb et al. — attention and information seeking

Jacqueline Gottlieb (Neuron, 2012), *Attention, Learning, and the Value of
Information*, DOI
[`10.1016/j.neuron.2012.09.034`](https://doi.org/10.1016/j.neuron.2012.09.034),
and Gottlieb, Oudeyer, Lopes & Baranes (Trends in Cognitive Sciences, 2013),
*Information-Seeking, Curiosity, and Attention: Computational and Neural
Mechanisms*, DOI
[`10.1016/j.tics.2013.09.001`](https://doi.org/10.1016/j.tics.2013.09.001).

These reviews connect information sampling to uncertainty reduction, reward,
learning progress, curiosity, and eye-movement control.  They are prior
territory against any claim that Child invented uncertainty-sensitive sensing.

Gate 6 tests a narrower architectural seam: delayed consequence cannot train a
context-dependent sensing decision unless the old observation/action address
survives until the audit arrives.
