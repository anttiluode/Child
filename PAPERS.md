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
