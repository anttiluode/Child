# Temporal replay: making “jumping in represented time” testable

The useful version of the instanton thought is not a fourth spatial dimension
and not physical time travel.  It is a **read state whose address is a time in
the model's representation**.  That address can advance, reverse, change
speed, or jump across a learned long-range edge while the underlying world and
the computer's clock remain causal.

## The correspondence

| idea | what it means in this project |
|---|---|
| Antti's instanton | a distributed nonlinear field whose green trajectory is a collective observable, not a particle equation |
| Child G2 | a bounded fast trace keeps unresolved events reachable until delayed relevance arrives |
| Child G3 | a cue retrieves an old episode and its temporal neighbourhood |
| Child G6 | a delayed audit trains a conditional action only because its old address survives |
| KyberDyyni1 | rhythmic population activity can provide a phase/address and can reverse orientation |
| Gate 7 | a fast read head routes over a slow sparse temporal body under a fixed edge budget |

The missing operation was between “find the old episode” and “use its old
context”: move through represented time with a bounded, inspectable state.

## A minimal state machine

After an episodic cue has returned an anchor, the mutable execution state can be
written as

```text
Σ = (anchor, phase, direction, speed branch, fast trace, confidence)
```

The requested offset selects a target address.  A local conductance rule then
updates a fast phase/read-head using only edges present in the persistent body:

```text
cue
  ↓
episodic anchor
  ↓
phase θ + signed velocity v_k
  ↓
local sparse temporal routing
  ↓
reinstated context
  ↓
validated mutable state update / write
```

Here `v_k` is a regime or speed branch, not a hidden homunculus.  A reversal is
an update of direction after new evidence; a jump is a nonlocal address update
or a large phase advance.  Both are observable in the state trace and can be
charged for in memory, hops, and failed routes.

## Gate 7 receipt

The benchmark uses a 256-slot circular represented-time address space.  Every
body has out-degree 12 (3,072 persistent directed supports) and an eight-hop
budget.  The router chooses the locally available neighbour closest to the
requested target.  Results below are 20 graph seeds × 4,000 queries per
workload; the displayed uncertainties are seed standard deviations.

| body | uniform greedy success | capped hops | oracle shortest-path success |
|---|---:|---:|---:|
| local ±1…±6 | `0.3739 ± 0.0075` | `7.317` | `0.3739` |
| reciprocal random | `0.1032 ± 0.0053` | `8.258` | `1.0000` |
| degree-matched small-world | `0.9694 ± 0.0087` | `4.346` | `1.0000` |
| dyadic ±1,2,4,8,16,32 | `1.0000 ± 0.0000` | `3.621` | `1.0000` |
| learned lag support 1,2,4,8,32,64 | `1.0000 ± 0.0000` | `3.151` | `1.0000` |

The reciprocal random graph is the useful control: almost every target is
reachable within eight hops if an oracle supplies a route table, yet the local
read rule succeeds on only 10.3% of uniform queries.  Existence of a path is
not the same as a usable local dynamical operator.

The ordinary software attackers remain stronger in their own regime:

| attacker | success | charged operation |
|---|---:|---|
| direct index | `1.0000` | one random-access fetch |
| dense attention scan | `1.0000` | 256 address comparisons |
| unit-step bidirectional scan | `0.0625` on uniform queries | one local edge per hop |

Therefore Gate 7 does not claim that sparse routing beats RAM, dense attention,
or a transformer on a general task.  It earns the narrower statement that a
multiscale body can make long represented-time offsets locally navigable under
the same persistent-edge budget, while random long-range reach is not enough.

## Relation to replay and neuroscience

Reich et al. (2026) model forward and reverse hippocampal replay with a
delay-coupled neural field.  Their analysis gives a low-dimensional phase
description and distinct stable replay-speed branches, including reverse
replay, rather than treating sequence recall as a static lookup.  See
[*Forward and reverse delay-driven hippocampal replay without symmetric
plasticity*](https://arxiv.org/abs/2608.21814), arXiv:2608.21814.

That is prior territory and a valuable constraint on Child.  Gate 7 is not a
reproduction of the neural-field equations.  It tests the adjacent engineering
question: once an anchor is known, does a bounded moving address over sparse
support buy anything measurable?

The instanton observation adds a second constraint.  In a nonlinear field, a
centre-of-mass “jump” can be caused by mass transfer between separated regions
or by a singular/poorly conditioned readout.  It should count as a temporal
jump only when the local field/state, phase, and address diagnostics agree—not
because one projected dot moves abruptly.

## What would make this more than a metaphor?

The collective-operator comparison is now deferred until after the integrated
evidence-control gates.  [Gate 8](results/GATE8.md) deliberately permits a
boring explicit controller and first tests whether competing laws, independent
coverage, delayed causal address, bounded pre-birth evidence, and honest
non-identifiability can coexist in one loop.

A later substrate gate should compare a population-derived phase against a
hand-designed executive variable and a random projection, with the same action
interface and perturbation budget.  It should also test actual forward/reverse
replay and speed switching.  The idea is weakened or killed if:

- a direct or dense attacker wins after equalizing the real resource bill;
- the apparent jump disappears when the full state rather than a centroid is
  inspected;
- reversal and speed branches must be hard-coded instead of emerging from
  delays, plasticity, or a learned policy;
- the read head cannot preserve the cue/action address until delayed evidence
  arrives.

Until then, “intelligence” is too strong a word.  The defensible result is a
small architecture for **addressable, reversible, multiscale replay** whose
state is explicit and whose costs can be audited.
