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

## 7. What could become genuinely interesting

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
