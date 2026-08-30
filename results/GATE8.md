# Gate 8 — constitutional executable-law competition

## Question

Can a bounded learner discover a structural law that is absent from its live
hypothesis set when:

- the incumbent model can make itself look correct by selecting easy evidence;
- consequences arrive after a delay;
- the new law is born only after old observations have been compressed;
- retaining the complete audit history is allowed as an attacker;
- and some worlds do not contain enough information to select one unique law?

This is the first integration gate in `Child`.  It combines the legal/illegal
operator, observation support, delayed causal address, bounded unresolved
memory, hypothesis birth, active experiment choice, and explicit refusal.

It is still a small synthetic system.

## World

The incumbent already knows a seven-variable linear law:

```text
f0(x) = beta dot x
```

The true world contains one extra signed interaction:

```text
f*(x) = beta dot x + s * 0.80 * x[a] * x[b] + Gaussian(0, 0.30)
```

The pair `(a,b)` and sign `s` are hidden.

Before the birth boundary, no pair interaction is a live hypothesis.

## How the incumbent captures its judge

The incumbent-controlled policy selects one-axis interventions:

```text
x = +/- e_i
```

Every pair product is then zero:

```text
x[a] * x[b] = 0
```

The missing interaction is therefore invisible on every selected experiment.
Across forty seeds the incumbent's observed residual MSE is:

```text
0.09045 +/- 0.00601
```

That is almost exactly the known noise variance `0.30^2 = 0.09`, so the
incumbent appears correctly specified.

On dense held-out inputs its noiseless RMSE is nevertheless:

```text
0.8000
```

The model has not learned the world.  It has selected a jurisdiction in which
its missing term cannot appear.

## Separation of powers

Gate 8 gives different operations to different components:

```text
law population
    proposes executable predictions

immutable judge
    scores delayed external outcomes

coverage channel
    spends one in four trials on theory-independent dense interventions

causal address queue
    preserves which query produced an outcome across a 12-trial delay

bounded audit ledger
    retains 16 representative coverage records without using their outcomes
    to decide which records survive

birth operator
    expands the law grammar but cannot select or rewrite old evidence

identifiability guard
    returns an observational equivalence class rather than breaking a tie
```

The coverage channel produces a residual MSE of:

```text
0.73564 +/- 0.04663
```

The fixed incumbent family therefore receives `MODEL_MISMATCH` in all forty
seeds.

## Hypothesis birth

At the birth boundary, a deliberately limited mutation grammar admits all
signed pair interactions:

```text
+/- 0.80 * x[i] * x[j]
```

There are:

```text
2 * choose(7,2) = 42
```

new executable laws.

The true law is guaranteed to be inside this supplied grammar.  This is an
operator-family expansion, not unrestricted equation invention.

Each newborn law is scored using a Gaussian likelihood.  A posterior mass of
at least `0.95` is required before making a decision.

## Policies and attackers

### Captured incumbent

The incumbent selects only experiments on which all pair interactions vanish.
It reports itself confirmed.

### Fixed catalog plus coverage

The independent coverage channel detects that the model family is wrong, but
the system is forbidden to birth another law.

### Birth without replay

The system admits the 42 new laws but has already discarded the raw pre-birth
evidence.  It must reacquire evidence using either random or
posterior-disagreement probes.

### Shuffled-address trace

This control retains exactly the same input vectors and base-law residual
values as the bounded trace, but permutes which residual belongs to which
intervention.

### Constitutional trace

Sixteen theory-independent coverage records survive.  Newborn laws may be
scored against them, but they cannot alter them.

### Full audit ledger

The stronger memory attacker stores all 100 independent coverage records.

### Full history

The literal-history attacker stores all 400 pre-birth records.

Every policy also pays for the 12 pending query addresses required by the
delayed audit.

## Forty-seed identifiable receipt

| policy | final status | truth covered | new probes after birth | dense pre-birth probes | peak raw scalars | noiseless dense RMSE |
|---|---:|---:|---:|---:|---:|---:|
| captured incumbent | confirmed 40/40 | 0/40 | 0 | 0 | 96 | 0.800 |
| fixed catalog + coverage | mismatch 40/40 | 0/40 | 0 | 100 | 96 | 0.800 |
| birth, no replay, random probes | identified 40/40 | **40/40** | `8.78 +/- 2.86` | 100 | 96 | 0.000 |
| birth, no replay, active probes | identified 40/40 | **40/40** | `5.55 +/- 0.50` | 100 | 96 | 0.000 |
| shuffled query/outcome addresses | identified 40/40 | **2/40** | `0.30 +/- 0.78` | 100 | 224 | 1.075 |
| **bounded constitutional trace** | identified 40/40 | **40/40** | **0** | 100 | **224** | **0.000** |
| complete audit ledger | identified 40/40 | **40/40** | 0 | 100 | 896 | 0.000 |
| literal full history | identified 40/40 | **40/40** | 0 | 100 | 3,296 | 0.000 |

The bounded trace matches both complete-memory attackers on this task while
using:

```text
4.0x fewer peak raw scalars than the complete audit ledger
14.7x fewer peak raw scalars than literal full history
```

The capacity is not treated as a free binary switch.  Before permitting any
new experiment, birth-time recovery changes with the online reservoir size:

| retained audit records | uniquely identified | true law recovered |
|---:|---:|---:|
| 0 | 0/40 | 0/40 |
| 4 | 3/40 | 3/40 |
| 6 | 19/40 | 19/40 |
| 8 | 34/40 | 34/40 |
| 10 | 40/40 | 40/40 |
| 12 | 40/40 | 40/40 |
| 16 | 40/40 | 40/40 |

The transition is a finite signal-to-noise result for this supplied law
family.  Sixteen is used for the headline arm because it leaves margin; it is
not asserted to be a universal memory size.

The no-replay arms are important controls.  The result is not that discovery
becomes impossible without old evidence.  They reacquire the law reliably,
but pay for new experiments.  Active disagreement saves about 3.23 probes
relative to random reacquisition.

## Causal address is not bookkeeping decoration

The shuffled-address arm keeps the same storage and the same marginal input
and residual values.  Only provenance is destroyed.

It still crosses the `0.95` posterior threshold in all forty seeds.  Its mean
top-class mass is:

```text
0.9966
```

But its returned law contains the truth in only:

```text
2/40 = 5.0%
```

This arm is not merely uncertain.  It is confidently wrong.  Delayed evidence
must remain attached to the intervention that produced it.

## Deliberate non-identifiability

The second world restricts all legal observations to:

```text
x[2] = x[1]
```

Consequently:

```text
x[0] * x[1] = x[0] * x[2]
```

for every possible legal intervention.

Those two laws are not merely difficult to distinguish.  They are the same
observable function inside this world.

Forty-seed constitutional result:

```text
NOT_IDENTIFIABLE              40/40
returned class size           2.0
true-law class coverage       40/40
post-birth probes             0
```

The system does not select one law because its posterior happens to contain
floating-point jitter.  It decodes only to the quotient supported by the legal
experiment set.

## Earned statement

> **When a live model influences which evidence will judge it, it can remain
> apparently correct by selecting a region on which its missing structure is
> silent.  A theory-independent coverage channel can expose the resulting
> misspecification.  If new executable laws are admitted later, a small
> causally addressed audit ledger can score them using pre-birth evidence and
> avoid reacquiring that evidence; when all legal interventions leave several
> laws equivalent, the correct result is the equivalence class rather than an
> invented winner.**

## What is not earned

- a general scientist;
- unrestricted program or equation synthesis;
- discovery when the true law is outside the supplied quadratic grammar;
- learning the base law, interaction magnitude, or observation-noise model;
- optimal audit allocation;
- robustness to corrupt or adversarial evidence sources;
- compound interactions or changing laws;
- a theorem that sixteen records are sufficient beyond this finite world;
- superiority to ordinary quadratic regression or Bayesian experimental
  design when those methods receive the correct full model family;
- a population-level replacement for the explicit evidence controller.

## Why this replaces the old Gate 8 order

The planned collective-operator comparison asked whether a distributed phase
readout could beat an explicit executive.  That remains a valid substrate
question, but Gate 0 through Gate 7 had not yet assembled their surviving
functions into one causal loop.

Gate 8 therefore allows the boring explicit controller to win and tests the
larger operation first:

```text
observe -> audit -> preserve -> admit -> compare -> intervene -> refuse
```

Only after that loop survives harder worlds is it useful to ask whether its
controller should be centralized, distributed, oscillatory, or geometric.

## Next attack

The clean next failure is source corruption.

Residual-priority memory can mistake a noisy or adversarial source for a new
law.  Give the system several evidence sources whose reliability changes, let
one source generate high-surprise nonsense, and ask whether provenance plus a
coverage reserve can learn which evidence deserves to found a theory.

After that, change the representation coordinates and test whether old audit
records retain the same causal meaning.
