# Child mathematics — from a matrix to a stateful operator

This file is a working mathematical vocabulary, not a novelty claim.

## 1. Transformer reference point

A transformer layer produces an input-conditioned routing operator such as

```math
A(X) = softmax(QK^T / sqrt(d)).
```

The effective routing is therefore dynamic even though the learned parameter
matrices are fixed during ordinary inference.

Child asks what changes if the operator also depends on **persistent internal
state** and **slow local structure** that survive from one physical timestep to
the next.

## 2. Local stateful cell

For cell `i`, let:

- `x_Ni(t)` be activity arriving from its persistent neighbourhood;
- `q_i(t)` be fast internal state;
- `M_i(t)` be slow persistent structure;
- `theta_i(t)` be output-gate / excitability state.

A generic local update is

```math
q_i(t+1)
  = F_i(q_i(t), x_Ni(t); M_i(t)).
```

The cell's one-step prediction is

```math
xhat_i(t+1)
  = G_i(q_i(t+1), x_Ni(t); M_i(t)).
```

The output boundary is separate:

```math
a_i(t)
  ~ pi_i(. | q_i(t), theta_i(t)).
```

The next network/world state depends on those emitted actions:

```math
x(t+1)
  = T(x(t), a(t), xi(t)).
```

This makes prediction **endogenous**: the learner helps determine the process
from which its next training sample arrives.

## 3. Fast, gate, and slow updates

A minimal local predictive update is

```math
e_i(t)
  = x_i(t+1) - xhat_i(t+1)

M_i(t+1)
  = M_i(t) + eta * e_i(t) * phi_i(t).
```

This is only one-step local error learning.

An output gate can receive a distinct scalar consequence:

```math
theta_i(t+1)
  = theta_i(t)
  + eta_a (R_t - b_i) grad log pi_i(a_i | q_i)
  + eta_h (rho_star - rho_i).
```

The final term represents a homeostatic constraint on output statistics rather
than task prediction.

Later gates may add an eligibility state `E_i(t)` so delayed consequence can
modify slow structure without replaying the full trajectory.

## 4. The effective operator is a trajectory-dependent family

For analysis we can still linearize the running system around its current
state:

```math
W_eff(t)
  = dF/dx |_(q(t), M(t), theta(t)).
```

But now this matrix is only a local tangent description.

The computational object is better written as

```math
W_eff(t)
  = W(x(t), q(t), M(t), theta(t)).
```

Two identical present inputs can therefore encounter different effective
operators because the receiver state or persistent structure differs.

This is the mathematical version of the recurring repo sentence:

> history changes what the next signal can reach.

## 5. Why this is not simply attention

Attention already gives input-conditioned routing.

Child deliberately adds three potential asymmetries:

```text
persistent local fast state
    q(t) does not have to be reconstructed from a stored token list

slow online structure
    M(t) may change during use

endogenous observation/action
    emitted activity can change which future data exist
```

Whether any of these is worth the engineering cost is an experimental
question.

## 6. The illegal-predictor degeneracy

Ordinary prediction:

```math
min_theta E_(x ~ P_world) ell(f_theta(x_t), x_(t+1)).
```

Coupled prediction/control:

```math
min_(theta, psi)
E_(tau ~ P_psi)
ell(f_theta(s_t, a_t), x_(t+1)).
```

Because `psi` changes the trajectory distribution, the learner may reduce
loss by reducing environmental entropy.

A constrained version is

```math
min_(theta, psi) E[ell]

subject to
    E[v(x_t, a_t)] >= v_min,
    E[c(a_t)] <= c_max.
```

or with a Lagrangian/homeostatic penalty,

```math
J =
E[ell]
+ lambda_v (v_star - vbar)^2
+ lambda_c E[c(a)].
```

Gate 1 is the smallest executable example of this distinction.

## 7. Selective-memory feedback is policy-dependent observation

Let event `e_t` expose features `phi_t`, let `y_t` be its later relevance, and
let the memory write decision be:

```math
a_t in {0,1}.
```

If discarding the event also discards the address/features needed to interpret
its later outcome, the value learner observes:

```math
O_t = (a_t, a_t y_t, a_t phi_t).
```

For a feature region `R` with zero write propensity,

```math
P(a_t=1 | phi_t in R) = 0,
```

the relevance law inside `R` is not identifiable from `O_t`.  Two worlds can
agree on every selected outcome and differ arbitrarily on:

```math
P(y_t=1 | phi_t in R).
```

Continuous parameter updates do not solve missing support.  At least one of
the following must make the counterfactual region observable:

```text
nonzero exploration propensity
side information / structural assumption
temporary trace surviving until feedback
```

Gate 5 is a finite categorical demonstration of this standard
selective-feedback boundary.

## 8. What could become genuinely interesting

The target is not to simulate every dendritic compartment.

The interesting question is whether a sparse local parameterization can
produce useful state-conditioned operators without paying the full cost of an
arbitrary dense relation matrix.

Possible future object:

```math
W_eff(t)
  = sum_k alpha_k(q_i(t), x_Ni(t)) B_k(M_i)
```

where each cell has a small persistent bank of local operator fragments
`B_k`, and fast state chooses their current mixture.

That is close enough to attention to compare directly, but different enough to
test whether persistent local dynamics buy something.

## 9. Active sensing under delayed audit

Let a visible context be `c_t`, a hidden binary target be `z_t`, and a free cue
be correct with context-dependent reliability:

```math
P(x_t = z_t \mid c_t=c)=r_c.
```

An optional probe `s_t` is correct with reliability `r_s` and costs `k`.  For
the symmetric binary Gate-6 world, acting from the stronger of two disagreeing
cues gives:

```math
U(\text{ACT now}\mid c)=r_c,
```

```math
U(\text{SENSE then ACT}\mid c)=\max(r_c,r_s)-k.
```

Therefore the value-of-observation gate is:

```math
\text{SENSE iff } \max(r_c,r_s)-r_c > k.
```

Correctness arrives `D` trials later.  Updating `r_c` then requires an
eligibility/address record:

```math
e_t=(c_t,x_t,\text{probed}_t,s_t,a_t).
```

The delayed audit is:

```math
u_{t+D}=1[a_t=z_t].
```

In the binary toy, `(a_t,u_{t+D})` identifies `z_t`, after which `e_t` permits
the free and probed cue reliabilities to be updated.  Without `e_t`, a scalar
audit may estimate aggregate performance but cannot identify which old context
or observation source should change.

Gate 6 instantiates this with `D=12`, `r_s=0.90`, `k=0.08`, and a swap from
`r=[0.90,0.55]` to `[0.55,0.90]`.

## 10. Compression for hypotheses that do not exist yet

Let `H_t` be the live hypothesis family and let a memory compressor retain:

```math
C_{H_t}(D_{1:t}).
```

It may be a sufficient statistic for comparing every `h in H_t`:

```math
p(C_{H_t}(D) \mid h)
```

without preserving the raw dataset `D`.

If a new law `h_new` is admitted later, that guarantee does not carry over:

```math
h_new notin H_t

C_{H_t}(D) generally not sufficient for
p(D \mid h_new) / p(D \mid h).
```

The learner then has three finite choices:

```text
retain all raw history
reacquire the distinguishing evidence
retain a bounded theory-independent audit sample
```

Gate 8 tests the third choice.

Let the incumbent also influence the experiment distribution:

```math
x_t \sim pi_h(x).
```

A misspecified incumbent can choose a support `S_h` on which its missing term
vanishes:

```math
f_*(x) = f_h(x)  for every x in S_h,
```

even though:

```math
P_mu(f_*(x) != f_h(x)) > 0
```

under an independent coverage distribution `mu`.

No amount of fitting on `pi_h` identifies the missing term.  A constitutional
mixture reserves nonzero support outside the incumbent policy:

```math
pi_const(x)
  = (1-epsilon) pi_h(x) + epsilon mu(x),
```

with `epsilon > 0`.

The raw audit ledger preserves tuples such as:

```math
r_t = (x_t, y_t, source_t, action_t, time_t).
```

Removing the address and retaining only the multisets `{x_t}` and `{y_t}`
does not preserve a causal likelihood.  Gate 8's shuffled-address control is
therefore expected to be confidently wrong rather than merely less efficient.

Finally, legal experiments induce an observational equivalence relation:

```math
h_i ~ h_j
iff
h_i(x) = h_j(x) for every legal x.
```

The strongest identifiable answer is the quotient class:

```math
[h] = {h' : h' ~ h}.
```

Returning one arbitrary member of a multi-law class adds information that the
experiment did not contain.
