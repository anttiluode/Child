# Gate 6 — active sensing under delayed audit

## Question

Can a controller learn when to pay for another observation when the reliability
of its first cue changes and correctness is revealed only after a delay?

The point is to stop using one vague operation for three different failures:

```text
model estimate is wrong       -> LEARN from an audit
present evidence is weak      -> SENSE at a cost
a decision is required        -> ACT on the target
```

## World

Every trial has a hidden binary target and one of two visible contexts.  A free
binary cue has context-dependent reliability.  An optional binary probe is
correct with probability `0.90` and costs `0.08` utility.  The controller then
acts; a binary correctness audit arrives 12 trials later.

Before an unannounced shift, the free-cue reliabilities are:

```text
context 0: 0.90
context 1: 0.55
```

After 6,000 trials they exchange:

```text
context 0: 0.55
context 1: 0.90
```

The world runs for another 6,000 trials.  Contexts and targets are balanced,
and every policy receives the same seeded streams.

## The value-of-observation rule

Let `r_c` be the learned reliability of the free cue in context `c`, let `s`
be probe reliability, and let `k` be probe cost.  With these binary symmetric
cues, the stronger cue wins when they disagree, so:

```math
U(\text{act now} \mid c) = r_c
```

and

```math
U(\text{sense then act} \mid c) = \max(r_c,s)-k.
```

The probe is bought only when:

```math
\max(r_c,s)-r_c > k.
```

For the `0.55` context the expected gain is `0.90 - 0.55 - 0.08 =
0.27`.  For the `0.90` context the probe adds no expected accuracy and loses
its full cost.

## Why delayed audit needs a trace

At action time, the learner temporarily retains:

```text
(context, free cue, probe flag, probe cue, action)
```

When the correctness bit arrives, action plus correctness reconstructs the
binary target.  The learner can then update the reliability attached to the
old context and cues.

Without that five-scalar address, the delayed scalar may update an aggregate
task statistic, but it cannot say which past context or cue was reliable.  The
no-trace learner therefore remains at its conservative initialization and buys
the probe on every trial.

## Policies

- **no sense:** always act from the free cue;
- **always sense:** always buy the probe and act from it;
- **oracle:** knows the current cue reliabilities and probes only the weak
  context;
- **learned, no trace:** begins conservatively but cannot condition delayed
  audits on past trials;
- **learned + delayed trace:** learns reliability with an exponential update
  and keeps 12 pending trial records;
- **zero-delay learner:** matched learning rule with immediate audit.

## Forty-seed receipt

Net utility is action accuracy minus `0.08 × probe rate`.

| policy | pre-shift late utility | first 500 post-shift | post-shift late accuracy | post-shift probe rate | post-shift late utility | recovery |
|---|---:|---:|---:|---:|---:|---:|
| no sense | `0.7231 ± 0.0106` | `0.7229 ± 0.0197` | `0.7256 ± 0.0099` | `0.0000` | `0.7256 ± 0.0099` | — |
| always sense | `0.8206 ± 0.0057` | `0.8185 ± 0.0141` | `0.9008 ± 0.0070` | `1.0000` | `0.8208 ± 0.0070` | — |
| **oracle** | **`0.8597 ± 0.0064`** | **`0.8593 ± 0.0118`** | `0.8989 ± 0.0065` | **`0.4996 ± 0.0106`** | **`0.8589 ± 0.0064`** | `200.0` trials |
| learned, no trace | `0.8206 ± 0.0057` | `0.8185 ± 0.0141` | `0.9008 ± 0.0070` | `1.0000` | `0.8208 ± 0.0070` | — |
| **learned + delayed trace** | **`0.8565 ± 0.0068`** | `0.8422 ± 0.0111` | **`0.8989 ± 0.0066`** | **`0.5415 ± 0.0225`** | **`0.8556 ± 0.0067`** | `283.9 ± 71.0` trials |
| zero-delay learner | `0.8566 ± 0.0075` | `0.8452 ± 0.0123` | `0.8990 ± 0.0063` | `0.5394 ± 0.0217` | `0.8558 ± 0.0064` | `263.8 ± 71.5` trials |

Recovery is the first completed 200-trial post-shift window in which context 0
is probed on at least 80% of its trials and context 1 on at most 20%.

## Did the learned policy actually reallocate sensing?

| policy | late probe rate, context 0 | late probe rate, context 1 |
|---|---:|---:|
| oracle | `1.0000` | `0.0000` |
| learned + delayed trace | `0.9986 ± 0.0053` | `0.0852 ± 0.0405` |
| zero-delay learner | `0.9990 ± 0.0040` | `0.0807 ± 0.0365` |
| learned, no trace | `1.0000` | `1.0000` |

The delayed-trace learner retains the always-sense policy's approximately 90%
action accuracy while eliminating nearly half of the probe purchases.  Its late
utility is within `0.0033` of the oracle and `0.0003` of the zero-delay control.

## Resource receipt

| policy | peak pending trial records | peak trace scalars | late extra observations / trial |
|---|---:|---:|---:|
| no sense | 0 | 0 | `0.0000` |
| always sense / learned no trace | 0 | 0 | `1.0000` |
| oracle | 0 | 0 | `0.4996 ± 0.0106` |
| **learned + delayed trace** | **12** | **60** | **`0.5415 ± 0.0225`** |
| zero-delay learner | 0 across trials | 0 across trials | `0.5394 ± 0.0217` |

The trace does not improve the sensor.  It preserves the address needed for a
later consequence to train the decision to use that sensor.

## Earned statement

> **When observation value depends on context, a delayed external audit plus a
> surviving action/observation trace can train an online controller to acquire
> evidence selectively.  Here it preserves the accuracy of always sensing,
> removes nearly half the observation cost, and reallocates sensing after the
> free cue's reliability changes.**

## Prior territory and attackers

This is classical active-perception / value-of-information territory.  Bajcsy's
active-perception formulation treats sensing as an agent-controlled operation;
Gottlieb's attention/value-of-information work and the Gottlieb–Oudeyer–Lopes–
Baranes review connect information sampling to uncertainty, reward, curiosity,
and eye-movement control.

The oracle threshold is the boring decision-theoretic attacker.  The
zero-delay learner is the credit-assignment control.  The always-sense policy
shows that high accuracy alone does not establish efficient observation.

## Not earned

- a general POMDP or active-inference algorithm;
- a learned representation of uncertainty;
- a decision about whether learning itself is worth its cost—updates are
  automatic here;
- actions that change future latent dynamics;
- recovery when the probe's own reliability changes after the controller stops
  sampling it;
- selective or missing external audits;
- a biological model of attention, eye movements, or neuromodulation;
- novelty over active perception, value of information, sensor scheduling, or
  delayed eligibility traces.

## Next attack

Gate 7 should test the proposed slow-body × fast-conductance factorization
against local, random, small-world, multiscale, and learned sparse supports
under one matched persistent-edge budget.
