# Benchmark geometry review — registered structure and limits

Read-only inspection of the benchmark, generation helper, common prompt and EXP-15
preregistration. No F1/P1 results, task instances, candidate executions or model
calls were used. A small NumPy calculation diagonalized the two analytic Rosenbrock
Hessians below; it did not evaluate the benchmark or any optimizer.

**The implementation agrees with the registered specification.** This is a
clarification of geometric diversity and terminology, not a semantic defect,
invalidation, benchmark amendment or demonstrated cause of an arm comparison.

## Exact quadratic geometry

Write a>0 for amplitude, m_i for shift, s_i for coordinate scale and w_i for weight.
[benchmark.py](../../benchmark.py) implements

`f(x) = a Σ_i (w_i / s_i²) (x_i − m_i)²`,

for both `sphere` and `quadratic`. Its Hessian is constant and diagonal:
`H = 2a diag(w_i/s_i²)`. Consequently its spectral condition number is exactly
`κ₂(H) = max_i(w_i/s_i²) / min_i(w_i/s_i²)`.

| Registered label | Parameter support | Geometry in candidate coordinates x | Condition-number support bound |
|---|---|---|---:|
| Sphere | w_i=1; independent s_i∈[0.75,1.5] | Shifted diagonal quadratic with mild anisotropy | 1≤κ≤4 |
| Quadratic | independent w_i=10^U(0,3)∈[1,1000]; same scales | Shifted diagonal quadratic with wider possible anisotropy | 1≤κ≤4000 |

Sphere is isotropic in the transformed coordinates z_i=(x_i−m_i)/s_i. It is
isotropic in x only when all coordinate scales are equal; independent continuous
scale draws ordinarily make them unequal. “Diagonally scaled Sphere” or “mildly
anisotropic diagonal quadratic” accurately describes the implemented task.
Calling it an unqualified isotropic sphere in x would obscure this distinction.

Quadratic's actual κ is not fixed at1000 or4000: independently sampled weights can
be similar. “Variable-conditioning diagonal quadratic, potentially ill-conditioned”
is more precise than asserting every instance is strongly ill-conditioned. The
limits above are support envelopes, not empirical ranges or typical values of
any frozen split. At a fixed positive objective level, ellipsoid semiaxis ratio
is √κ: at most2 for Sphere and √4000≈63.25 for Quadratic. Translation and positive
amplitude do not change κ. Dividing objectives by the positive per-task reference
normalization also leaves κ and the optimum unchanged.

All optima are x=m with value0, strictly inside [-5,5]^d because m_i∈[-2,2]. The
central-shift restriction and low dimensions2/4 are additional shared structure;
normalization removes amplitude differences from regret, not this geometric prior.

## Rosenbrock is a distinct coupled, nonconvex geometry

The third family uses y_i=1+(x_i−m_i)/s_i and
`a Σ_i [100(y_(i+1)−y_i²)² + (1−y_i)²]`. Its adjacent-coordinate nonlinear coupling
survives diagonal scaling. It is not another separable quadratic and has no single
constant positive-definite Hessian condition number over the search box. For
example y_1=0,y_2=1 gives first Hessian diagonal entry −398 before scaling, at a
feasible point, establishing nonconvexity inside the bounds.

At the optimum y=1, before amplitude/coordinate scaling, the Hessian has off-diagonal
entries−400 and diagonal entries [802,200] in2D, [802,1002,1002,200] in4D. Direct
symmetric eigendecomposition gives local κ≈2508.01 and3180.02 respectively. For the
actual Hessian `a D^−1 H D^−1`, D=diag(s_i), the conservative congruence inequality
`κ(H)/4 ≤ κ(a D^−1 H D^−1) ≤ 4κ(H)` gives local envelopes approximately
[627,10032] in2D and[795,12720] in4D. These are conservative bounds at the optimum,
not attained extrema of this scale family, global convexity claims or observed
conditioning of any experimental task.

## What was already registered, and what remains a hypothesis

[PREREG_EXP15.md](../../PREREG_EXP15.md), lines22–27, explicitly gives independent
coordinate scales, z_i=(x_i−shift_i)/scale_i, Sphere=aΣz_i² and Quadratic=aΣw_i z_i².
The same equations already appear in the initial preregistration commit
`13b88440` and were retained in the confirmatory freeze `0643691b`.
The [fresh-instance helper](../generation.py) preserves those parameter laws.
Thus two of the three family labels, or four of six equally weighted
family/dimension strata, are diagonal quadratics: **two thirds of primary weight**.
These are different parameter distributions within overlapping functional classes,
not evidence of duplicate task instances or an implementation/specification mismatch.

This structural overlap plausibly makes a small stock of familiar optimization
strategies competitive. As a mathematical illustration, a known diagonal quadratic
`Σq_i x_i² + Σb_i x_i + c` is identifiable in exact arithmetic from2d+1 suitable
axis probes: f(0),f(+h e_i),f(−h e_i). Their symmetric differences recover q_i and
b_i, hence its minimizer. That statement assumes the quadratic class is known;
it is not a tested optimizer, an anytime-AUC guarantee or a result for Rosenbrock.
The actual program receives neither family identity nor hidden parameters, and
AUC charges the identification probes as well as later evaluations.

The common generation prompt describes smooth deterministic low-dimensional
convex/curved-valley tasks; it does **not** disclose the family names or source.
Therefore “LLM priors may make independent generation competitive on this narrow
geometry” is a hypothesis, not proof of benchmark leakage, memorization, saturation
or the cause of EXP-15's inconclusive R/A2 advantage. This review neither tests
that causal explanation nor changes the benchmark to favor feedback. Broader
geometries or an explicit prior/structure intervention would require a separately
registered future experiment with fresh data and a stated estimand.
