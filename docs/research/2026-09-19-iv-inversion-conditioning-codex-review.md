# Mathematical review

**The proposed criterion is not a certificate.** It combines a local inverse approximation, an empirical discretization-error estimate, and insufficient existence tests. These are useful diagnostics, but their conjunction does not establish an IV-error bound.

The trigger exposes two distinct failures: the interpolated price lies below the observed reference-price range, and the surface oscillates in volatility. Neither is adequately represented by a filtered, pointwise vega-scaled error.

I read the complete note and the relevant implementation. I did not fetch sources or modify files. Source assessments below distinguish established mathematical content from quotations and table references I cannot independently authenticate offline.

## 1. Errors and overstatements in the note

### 1.1 The checked-out reference path is not Ultra

There is an important discrepancy with the supplied context.

[make_validate_fn](/home/kai/work/mango-option/src/option/table/adaptive_metrics.cpp) calls `solve_american_option(p)`. That calls `AmericanOptionSolver::create(params)` without a grid specification, which invokes `estimate_pde_grid(params)` with default `GridAccuracyParams`.

Those defaults use `tol = 1e-2`, whereas the explicit Ultra profile uses `tol = 5e-6` and different resolution limits. See [grid_spec_types.hpp](/home/kai/work/mango-option/src/option/grid_spec_types.hpp:42) and [grid_spec_types.cpp](/home/kai/work/mango-option/src/option/grid_spec_types.cpp:125).

Neither setting supplies a proven price-error bound. In particular, the trigger’s externally measured reference accuracy cannot automatically be assigned to this validation path.

### 1.2 “Defensible criterion”: local approximations are promoted to guarantees

Let \(V\) denote the exact model price and \(\widehat V\) the numerical reference. The note needs this distinction throughout.

**First-order IV conversion.** Where \(V\) is differentiable and \(\nu=V'(\sigma)>0\),

\[
|\Delta\sigma|\approx \frac{|\Delta V|}{\nu}
\]

is correct. Ackerer et al. support its use as an approximation to an IV loss. They do not establish a finite-error bound for an American obstacle problem.

**Curvature gate.** The displayed leading-order correction is reasonable after replacing the signed \(\delta_{\rm true}\) by its magnitude. As written, comparing a signed negative displacement with positive \(e_\sigma\) is inconsistent.

More substantially:

- A local second derivative does not bound curvature over the inversion interval.
- The \(O(\delta^3)\) expansion needs regularity that must be established near free-boundary transitions.
- \(e_\sigma\le\epsilon\) does not establish that linearization is accurate.
- The proposed curvature inequality does not guarantee 10% accuracy merely because \(\gamma=0.1\).
- A noisy three-point second difference need not detect a nearby plateau.

The Black–Scholes identity \(\mathcal V/\nu=d_1d_2/\sigma\) is correct. It does not validate the analogous numerical gate for the American reference.

**Measurability gate.**

\[
\frac{\delta_{\rm ref}}{\nu}\le\rho\tau_\sigma
\]

is a sensible *local resolution diagnostic*. Higham’s rule of thumb supports that interpretation, not the claimed necessary-and-sufficient certification condition.

A rigorous finite-error result needs control over the inverse throughout the relevant price interval. For example, a positive lower slope bound \(m\) gives an inverse Lipschitz bound \(1/m\). The derivative at one point does not.

Also, numerical pricing error is forward error for the pricing problem. It becomes input perturbation for a separately defined inverse problem; that distinction matters when invoking backward-error terminology.

**The final conjunction does not even test the claimed tolerance.** It omits \(e_\sigma\le\tau_\sigma\), the surface-side existence condition previously introduced, and the bracket monotonicity condition. Even as an engineering acceptance test, it is incomplete.

### 1.3 The three volatility solves do not measure solver error

The bump pair cannot distinguish:

- genuine curvature;
- discretization bias;
- grid-selection changes with volatility;
- numerical noise.

All three prices can share a large bias and still form a perfectly smooth stencil.

If each stencil price has a rigorous absolute error bound \(\delta\), the numerical-error contributions alone satisfy

\[
|\widehat\nu-\nu_{\rm stencil}|
\le \frac{\delta}{\epsilon},
\qquad
|\widehat{\mathcal V}-\mathcal V_{\rm stencil}|
\le \frac{4\delta}{\epsilon^2}.
\]

Derivative truncation error must be added separately.

For the trigger, \(\epsilon=0.0014\). If \(\delta=0.001\), these bounds are approximately \(0.714\) for vega and \(2041\) for vomma. Thus the note’s hypothetical price uncertainty would not establish that the measured vega \(0.2196\) is positive.

Jäckel’s smoothness discussion supports concern about attainable root accuracy. It does not make three samples an error estimator. Likewise, applying a \(\sqrt{\epsilon_{\rm machine}}\)-type numerical-differentiation heuristic directly to an absolute dollar error \(\delta_{\rm ref}\) is dimensionally and mathematically unjustified.

### 1.4 Identifiability, continuation, and existence are conflated

The proposed three conditions are neither a general characterization of identifiability nor an existence theorem.

- Positive derivative is sufficient for a differentiable local inverse under suitable regularity. It is not necessary for uniqueness: a strictly increasing function can have zero derivative at a point.
- Strict increase over the entire domain is stronger than necessary for uniqueness at one target price.
- Continuity and weak monotonicity imply existence **only when the target lies in the function’s range**.
- On a closed bounded volatility domain, a continuous nondecreasing function has level sets that are empty or closed intervals. This statement requires the domain qualification.
- \(V-I>\delta\) establishes a resolved premium above immediate exercise if \(\delta\) really bounds error. It does not establish resolved volatility sensitivity.
- For an exact quote, being below known intrinsic can be determined without uncertainty in the PDE reference.
- Rejecting \(V_{\rm surf}-I\le\delta\) through `nullopt` would discard evidence of a surface defect.

Liu et al.’s stopping-region observation is appropriate in the interior of a stopping region: price equals payoff and is locally independent of volatility. Boundary cases require care. Their training filters are numerical data-selection rules, not a theorem establishing the note’s proposed thresholds.

The quoted observation that one criterion “should cover” the other does not prove mathematical independence. Retaining both for robustness is an implementation decision.

### 1.5 The dividend and “wrong branch” argument is unsupported

Two continuation regions as a function of **spot** do not imply multiple inverse branches as a function of **volatility**. Nor does holding before a dividend and exercising afterward establish such branches.

For a monotone reference curve, multiple exact inverse solutions form a level interval, rather than disconnected branches. An oscillatory interpolant can create disconnected roots, but that is a separate defect.

Also, an observed numerical plateau above immediate intrinsic is not proof of an exact positive-volatility plateau in the continuous model. Small probabilities of alternative outcomes can create exponentially small sensitivity that a truncated-grid PDE calculation cannot resolve.

The trigger’s immediate-exercise premium is about \$0.407. Consequently, an immediate-exercise test misses the economically relevant near-flat region.

### 1.6 Richardson/GCI: sign error and excessive claims

From

\[
V_h=V+Ch^p+o(h^p),
\]

the correct identities are

\[
V_h-V_{rh}=C(1-r^p)h^p+o(h^p),
\]

and

\[
V-V_h\approx \frac{V_h-V_{rh}}{r^p-1}.
\]

Section 3.2 reverses both signs in its displayed derivation. Its absolute-magnitude estimator survives.

Other qualifications:

- GCI is an empirical uncertainty estimate, not a deterministic error enclosure.
- The conventional safety factors do not repair a failed asymptotic expansion.
- Three grids give one observed order; they do not establish its stability.
- The displayed logarithmic ratio requires compatible signs and nonzero differences. Taking absolute values would conceal oscillatory convergence rather than validate it.
- Measuring order once per profile does not establish pointwise order near every free boundary or dividend event.
- Comparing accuracy presets is not necessarily a controlled \(h,2h,4h\) experiment.
- Shared domain-truncation, boundary, penalty, or jump-interpolation bias can be invisible to the difference.

The assertion that reference error should be largest where vega is smallest is unsupported. In an immediate-exercise region, the obstacle can give essentially exact prices with exactly zero vega.

### 1.7 What the cited sources do—and do not—support

| Source / section | Assessment |
|---|---|
| **Higham; Trefethen–Bau, §§1 and 4** | The scalar condition numbers are correct where the inverse is differentiable. The relative perturbation relationship is first-order, not an exact finite-error equality. Neither source licenses the final certificate. |
| **Jäckel 2006, §1.2** | Small-volatility flatness and floating-point attainability are relevant. His normalized volatility convention must be distinguished from annualized volatility. The displayed “limit = intrinsic + correction” should be written as an asymptotic expansion. Positive-volatility European invertibility is not destroyed mathematically by very small vega. |
| **Jäckel 2015, §1.3** | Evaluation smoothness and cancellation are relevant. They do not quantify PDE discretization error or justify the bump-pair estimator. |
| **Claimed upper-bound condition formula** | As written, the claimed relative-IV lower bound proportional to \(b_{\max}/(b_{\max}-b)\) omits volatility-dependent factors. For normalized total volatility \(s\to\infty\), \(b_{\max}-b\asymp\phi(s/2)/s\), giving relative condition number approximately \(4b_{\max}/[s^2(b_{\max}-b)]\). The displayed inequality is not a general bound. The original equation needs checking before attribution. |
| **Forsyth–Vetzal, §3.1** | Distinguishing nonlinear-solver convergence from discretization order is correct. Their reported convergence behavior is evidence for their schemes and tests, not a universal order for constant-step projected/penalty American solvers. Finite Newton termination depends on the discrete problem’s assumptions. |
| **Rannacher / Pooley–Forsyth–Vetzal** | Smoothing remedies and conditional convergence are relevant. Results for uncertain-volatility PDEs do not directly establish error bounds for this American solver with cash-dividend jumps. |
| **Ekström, §2.3** | Weak monotonicity and continuity are appropriate underlying principles under the theorem’s assumptions. Extension to the implemented dividend jump and stock-boundary conventions requires justification. |
| **American put upper bound** | \(P\le K\) needs assumptions such as nonnegative rates and nonnegative stock. It is false in general with negative rates, which the note also discusses. |
| **Ackerer et al., §5** | The approximate inverse-vega loss is relevant. Restricting to OTM options does **not** avoid low vega: deep-OTM options have low vega, and European call/put vegas at the same strike are equal. |
| **Other loss-function examples, §5** | The distinction between inverse-vega conversion and vega importance weighting is sound. Examples of loss choices do not determine a unique certification objective. Subtracting intrinsic may aid approximation but does not restore identifiability. |

I cannot authenticate the exact quotation wording, equation numbering, or convergence-table entries offline. Their broadly plausible content should not be treated as independent verification.

### 1.8 The trigger calculation contains additional overclaims

Using the supplied values,

\[
e_{\rm lin}=\frac{14.12631-14.1090}{0.2196}
\approx0.07883=788.3\text{ bps}.
\]

That arithmetic is correct. Its interpretation as an actual IV error is not.

- The Newton displacement is about **56 bump widths**.
- The interpolated price is about **\$0.01589 below the observed plateau**.
- There is no inverse distance to report when the inverse does not exist.
- Calling 788 bps a lower bound on a nonexistent inverse is meaningless without defining an extended-valued loss.
- A \$0.001 reference error corresponds locally to approximately **45.5 bps**. It fails a one-third allocation of a 100-bps tolerance, contrary to the note’s unqualified “passes” claim.
- \(1.2\times10^{-3}\) is slightly outside the stated \(10^{-4}\)–\(10^{-3}\) relative-price range.

The price miss is evidence of a surface discrepancy. Its certification still requires an established reference uncertainty.

## 2. Assessment of the tentative design

### Gate 1: retain as a diagnostic, not a certificate

Replace “measured reference error” with “estimated uncertainty” unless actual enclosures are available.

Even with an enclosure, use a finite inverse bound or finite-volatility price separation. A noisy central-difference vega cannot support the proposed guarantee.

### Gate 2: replace immediate-exercise premium with an inverse-set test

Three different questions need separate answers:

1. **Existence:** is the target price in the attainable range?
2. **Uniqueness:** does its inverse set contain one point?
3. **Resolution:** is that inverse set sufficiently narrow after numerical uncertainty is included?

An exercise-boundary test addresses only one source of degeneracy.

A noise-aware bump test can establish resolved increase across the stencil:

\[
\widehat V_+-\widehat V_->\delta_++\delta_-.
\]

There is no arbitrary \(k\) here if the uncertainties are rigorous bounds. But this establishes a positive **secant**, not a positive derivative throughout the interval or uniqueness.

A certified lower envelope is stronger for detecting impossibility. Under justified volatility monotonicity, \(V(0)\) is a lower envelope. Alternatively, the value of any admissible exercise policy is a lower bound on the American value. A suitable policy bound may directly expose the trigger’s below-envelope price.

### Richardson/GCI: conditional engineering estimate

**Answer to (b):** one coarse solve is insufficient for a defensible bound in the stated setting.

A controlled experiment must account for spatial refinement, time refinement, event alignment, payoff/jump smoothing, domain truncation, and algebraic/penalty tolerances. The existing grid estimator changes the domain with volatility, which also complicates numerical differentiation.

When observed convergence is unstable:

- continue controlled refinement and separate likely error sources;
- test domain enlargement and tighter algebraic tolerances;
- label unresolved results as unresolved;
- if a rigorous certificate is required, use validated bounds—such as comparison-principle sub/supersolutions or a posteriori estimates with established stability constants.

Agreement between solvers or successive grids is useful evidence, but not automatically a bound. A larger safety factor cannot convert unexplained convergence into certification.

### Reference inversion: bracket it, and retain uncertainty

**Answer to (d):** unsafeguarded secant/Newton iteration is not the right certification mechanism.

It can leave the domain, encounter a flat region, or find a numerical artifact. Moreover,

\[
|\widehat V(\sigma_k)-V_{\rm surf}|\le\delta_{\rm ref}
\]

is a price-residual statement, not an IV-accuracy statement. A wide plateau can satisfy it.

Use interval-aware bracketing. Terminate based on a volatility enclosure, or explicitly report unresolved width. Bounds must apply at every oracle evaluation, not just at \(\sigma_{\rm ref}\).

A certified lower vega bound is sufficient:

\[
V'(\sigma)\ge m>0\quad\text{throughout the relevant interval}
\]

implies, for an existing root \(V(\sigma_*)=V_{\rm surf}\),

\[
|\sigma_*-\sigma_{\rm ref}|
\le
\frac{|V_{\rm surf}-\widehat V(\sigma_{\rm ref})|+\delta_{\rm ref}}{m}.
\]

However, establishing \(m\) is difficult. A finite collection of positive secants does not establish it, and a plateau anywhere in an unnecessarily broad bracket makes the bound vacuous.

### Outside-domain prices are not finite IV errors

For the trigger, the distance to the lower domain edge is

\[
0.14-0.10=0.04=400\text{ bps}.
\]

This is only the minimum distance to a hypothetical root below the domain. Such a root may not exist anywhere.

Reporting 400 bps as the error would let this failure pass the 2000-bps viability gate. It is therefore unsuitable as the scalar used for acceptance.

Use separate statuses:

- outside the specified domain’s price range;
- below a global attainable-price lower bound;
- unresolved because reference intervals overlap;
- nonunique inverse.

### The proposed “true IV error” is not the query-time error

The tentative design computes

\[
\left|V^{-1}\!\left(V_{\rm surf}(\sigma_0)\right)-\sigma_0\right|.
\]

The query-time product instead computes approximately

\[
\left|V_{\rm surf}^{-1}\!\left(V(\sigma_0)\right)-\sigma_0\right|.
\]

They agree only approximately under suitable small-error assumptions. An oscillating surface can behave very differently in these two experiments.

The checked-out [interpolated solver](/home/kai/work/mango-option/src/option/interpolated_iv_solver.hpp:684) also has a default 17-point multiple-root screen. Its quartile-vega precheck and finite root screen do not prove monotonicity or uniqueness.

## 3. Four principled alternatives

Costs below count PDE solves at distinct volatility/grid combinations. The current reference preparation uses **three solves per point**. Rigorous error-enclosure construction may require additional work beyond those counts.

### A. Certify price accuracy; report IV conditionally

**Principle:** certify the forward quantity the table approximates. Derive IV statements only where the inverse supports them.

Given a valid price enclosure,

\[
|\widehat V-V|\le\delta,
\qquad
|V_{\rm surf}-V|
\le |V_{\rm surf}-\widehat V|+\delta.
\]

Choose an absolute or strike-normalized price requirement. Relative-to-price error is problematic near zero; relative-to-time-value error is problematic near exercise.

**Constants:**

- Chosen: price tolerance and normalization.
- Measured/estimated: price discrepancy and reference uncertainty.
- Proven, if available: uncertainty enclosure.
- No vega threshold is required.

**Cost:** one base solve; a two-grid estimate adds one. That is two total, versus three currently. Rigorous enclosure cost is method-dependent.

**Trigger:** price discrepancy approximately \$0.01731, or \(1.52\times10^{-4}\) of strike, plus uncertainty. Separately flag the below-envelope price if established. Price acceptance depends transparently on the selected price tolerance.

**Answer to (a):** this is the cleanest primary target for a general-purpose price table. Vega-scaled price error remains a useful local diagnostic, but it should not be called an IV certificate.

### B. Test the IV tolerance directly using finite price separation

**Principle:** continuity and monotonicity can certify a finite inverse neighborhood without derivatives or linearization.

Let

\[
a=\sigma_0-\tau_\sigma,\qquad b=\sigma_0+\tau_\sigma
\]

lie within the domain. Suppose valid enclosures satisfy

\[
L(s)\le V(s)\le U(s).
\]

For target \(y=V_{\rm surf}(\sigma_0)\), the condition

\[
U(a)<y<L(b)
\]

guarantees a reference root in \((a,b)\). Monotonicity ensures every root lies within that interval. This establishes tolerance-level localization, although not necessarily exact uniqueness.

Failure of these inequalities can mean **unresolved**, rather than definitely failed. Domain-edge cases require explicit one-sided treatment.

**Constants:**

- Chosen: only \(\tau_\sigma\), domain, and the definition of the guarantee.
- No \(\rho\), vega floor, or curvature threshold.
- Reference enclosures must be valid.

**Cost:** two shifted-volatility solves beyond the base; three total, matching today’s solve count. Two-grid estimates at all three locations require six total solves, but still give empirical rather than rigorous bounds.

**Trigger:** fails a narrow IV neighborhood test. For a 100-bps tolerance, the supplied \(V(0.13)=14.12517\) already exceeds the surface price by roughly \$0.01617. Reliable intervals would resolve that failure.

**Seam:** this naturally gives pass/fail/unresolved. Cached reference brackets at multiple radii can instead produce a conservative scalar IV-radius bound. A Boolean surrogate must not masquerade as a measured IV error.

### C. Use interval inversion and report an inverse set

**Principle:** treat inversion with numerical uncertainty as a set-valued problem.

Using valid reference enclosures, construct an outer set containing every possible solution of \(V(\sigma)=y\). With monotonicity, interval bracketing can shrink that set or establish that it is empty.

Report:

- empty: no inverse in the domain;
- narrow enclosure: resolved to a stated volatility accuracy;
- wide enclosure: insufficient information;
- demonstrated plateau: nonunique.

For a nonempty enclosure \(A\), a conservative scalar is

\[
e_{\rm upper}=\sup_{\sigma\in A}|\sigma-\sigma_0|.
\]

Existence must be established separately; a nonempty outer enclosure alone does not prove a root exists.

**Constants:**

- Chosen: \(\tau_\sigma\), domain, computational budget.
- No mandatory error-budget split or vega threshold.
- GCI safety factors remain chosen conventions if empirical intervals are substituted for rigorous ones.

**Cost:** up to two domain-endpoint solves plus \(n\) adaptive oracle evaluations, with per-evaluation uncertainty work. No fixed small upper bound: flat regions can force refinement or an unresolved result.

**Trigger:** below-domain-range once the lower-endpoint uncertainty is smaller than the approximately \$0.01589 gap. A valid global lower envelope can strengthen this to global nonexistence.

This is preferable to a lower-vega approach near flat regions because it does not require differentiability or a uniformly positive derivative.

### D. Validate the actual query-time inverse

**Principle:** test the operation users consume.

At each reference point, pass \(\widehat V(\sigma_0)\) to the actual interpolated IV solver and compare its returned volatility with \(\sigma_0\). Record failure and ambiguity as outcomes.

To make a certified statement, propagate the reference price interval through the surface inverse. Establish surface monotonicity or isolate all relevant roots; the existing 17-point screen is only a screen.

**Constants:**

- Chosen: \(\tau_\sigma\), domain, solver policy.
- Reference uncertainty and inverse ambiguity remain explicit.
- Existing query thresholds must be justified separately or reported as operational policy.

**Cost:** zero additional PDE solves beyond the base reference. Extra work consists of surface evaluations and root/derivative analysis. Reference refinement may still be needed.

**Trigger:** the supplied pointwise values are insufficient to determine the result. The actual oscillating surface must be queried. It could return a displaced root or refuse because of multiple roots. Claiming a numerical outcome without that test would be speculation.

This is the most relevant validation for an IV-serving product, but it does not replace price validation.

## 4. Recommendation and adaptive-loop integration

Use **price-error validation plus explicit inverse outcomes**, with query-time round trips as the primary operational IV test. For a reference-inverse guarantee, use finite tolerance brackets or interval inversion.

Do not make local vega, immediate time value, or GCI-derived conditioning gates the definition of whether a surface defect exists.

### What the current seam permits

[ScoreErrorFn](/home/kai/work/mango-option/src/option/table/adaptive_refinement.hpp:269) has only two outcomes:

- finite nonnegative scalar;
- `nullopt`: no evidence.

In [the loop](/home/kai/work/mango-option/src/option/table/adaptive_refinement.cpp:628):

- skipped points disappear from maxima, averages, counts, and refinement bins;
- nonfinite scores disqualify candidates;
- holdout maximum drives candidate selection;
- at least one measured holdout point suffices for the measurement-count check;
- viability uses the fixed \(0.20\) bound.

Consequently, **the existing return type alone cannot honestly encode error, nonexistence, ambiguity, and reference failure.**

Keep it as a scalar projection if useful, but add an explicit companion assessment carrying status and price residual. Extend cached reference data for intervals/brackets. Avoid candidate-dependent oracle solves hidden inside the supposedly arithmetic scorer: they invalidate current caching and solve-accounting assumptions.

### How to consume noninvertible points—answer to (e)

Distinguish reference limitations from surface defects:

1. **Reference unresolved:** refine the reference, or report missing certification coverage. Do not blame the surface.
2. **Reference intrinsically nonidentifying:** exclude from scalar IV statistics, retain price validation and coverage status.
3. **Surface price outside a resolved reference range:** record a surface failure and retain its price-distance residual for refinement.
4. **Surface inverse ambiguous or failing:** record an operational IV failure.

Admission based on reference quality should be independent of the candidate surface. Otherwise a worse candidate can improve its maximum by making difficult points disappear.

For candidate ranking, use a declared ordering that includes failures and residuals before comparing scalar IV maxima. Propagate failures into refinement bins so they can be repaired.

For final acceptance there is an unavoidable choice:

- A universal IV guarantee must reject a surface with a resolved counterexample.
- A conditional-domain guarantee may return a surface with explicit unsupported regions and a query fallback/refusal.
- A statistical or coverage guarantee needs a declared failure allowance and appropriate sampling methodology.

There is no mathematically honest rule that simultaneously promises a universal maximum-error guarantee and ignores a difficult counterexample.

The \(0.20\) viability bound should apply only to quantities actually bounded in volatility units. It must not consume domain-edge distances, price residuals, or invented finite penalties as though they were IV errors.

Finally, Latin-hypercube testing establishes sampled performance. Repeatedly selecting against the holdout also makes it a selection set. Neither constitutes a uniform-domain certificate without additional analytical bounds or a specified statistical guarantee.

### Can all extra constants be removed?—answer to (f)

**A reference/surface error-budget split is not mathematically mandatory.** Validated interval methods can propagate all uncertainties together and compare the final enclosure directly with \(\tau_\sigma\). That removes \(\rho\), curvature thresholds, and vega floors.

But \(\tau_\sigma\) alone cannot determine:

- acceptable price error where IV is undefined;
- allowed unsupported coverage;
- statistical confidence;
- computational budget;
- whether the requested certificate concerns the model, the numerical oracle, or the query algorithm.

With empirical GCI estimates, safety factors remain conventions. Calling them literature-derived does not make them measured constants.

## 5. Open questions the specification must pin down

1. **Target quantity:** reference-equivalent volatility, actual surface-inverse output, or both?
2. **Reference:** exact continuous model or a designated numerical oracle?
3. **Accuracy path:** should validation explicitly select Ultra, and how will controlled grid refinement be specified?
4. **Meaning of certification:** empirical sampled validation, statistical coverage, or deterministic bounds?
5. **Domain:** distinguish outside-table-range from globally unattainable prices.
6. **Ambiguity:** require exact uniqueness, or accept an inverse set narrower than a stated tolerance?
7. **Dividend model:** establish monotonicity and lower-envelope results for the implemented stock/jump boundary convention.
8. **Price requirement:** define acceptable error where IV is unresolved or undefined.
9. **Coverage policy:** determine when unsupported points require rejection, fallback, or restricted-domain reporting.
10. **Reference uncertainty:** include domain, discretization, jump interpolation, obstacle/penalty, and algebraic errors.
11. **Loop contract:** add statuses, stable coverage accounting, failure-driven refinement, and accurate PDE-solve counts.
12. **Trigger evidence:** verify the apparent plateau and lower envelope under controlled reference refinement before labeling them exact model features.