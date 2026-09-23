# When is implied-volatility inversion well-posed, and how should a certified IV-error metric be defined?

Research note, 2026-09-19. Question: what is the mathematically sound criterion for the price→σ
inversion being well-posed, and how should a certified IV-error metric for an interpolated
American price surface be defined where vega degenerates?

All claims below are traced to the source that owns them. Sources actually opened are listed at
the bottom; where only a secondary source was reachable this is stated inline.

---

## Defensible criterion

### 0. Notation

At a validation point $p = (S, K, \tau, \sigma_{\mathrm{ref}}, r, q, \text{divs})$:

- $V_{\mathrm{ref}}(\sigma)$ — the FDM reference price as a function of volatility, all else fixed.
- $V_{\mathrm{surf}}$ — the interpolated surface's price at $p$.
- $\nu_{\mathrm{ref}} = \partial V_{\mathrm{ref}}/\partial\sigma$ at $\sigma_{\mathrm{ref}}$ (American vega; no closed form).
- $\mathcal{V}_{\mathrm{ref}} = \partial^2 V_{\mathrm{ref}}/\partial\sigma^2$ (vomma / volga).
- $\delta_{\mathrm{ref}}$ — the reference solver's own price error at $p$ (a *measured* quantity, see §3).
- $I$ — intrinsic value; $\tau_\sigma$ — the IV tolerance we wish to certify to (e.g. 100 bps).

### 1. The IV error of a surface at a point (first order)

$$
e_\sigma(p) \;=\; \frac{\bigl|V_{\mathrm{surf}} - V_{\mathrm{ref}}\bigr|}{\nu_{\mathrm{ref}}}
$$

This is the first-order (linearized) inverse image of the price discrepancy. Its justification is
the elementary Taylor argument used throughout the calibration literature and stated explicitly by
Ackerer–Tagasovska–Vatter: *"$\sigma_j - \hat\sigma_j \approx (\pi_j - \hat\pi_j)/\nu_j$ when
$\pi_j \approx \hat\pi_j$"* ([Deep Smoothing of the Implied Volatility Surface](https://arxiv.org/pdf/1906.05065), Appendix B, eq. (10) and the line following it).

**Validity condition (curvature / linearization gate).** From
$V(\sigma + \delta) = V(\sigma) + \nu\,\delta + \tfrac12\mathcal{V}\delta^2 + O(\delta^3)$, the
relative error of the first-order inversion is

$$
\frac{|\delta_{\text{true}} - e_\sigma|}{e_\sigma} \;\approx\; \tfrac12 \left|\frac{\mathcal{V}_{\mathrm{ref}}}{\nu_{\mathrm{ref}}}\right| e_\sigma .
$$

Require $\tfrac12\,|\mathcal{V}_{\mathrm{ref}}/\nu_{\mathrm{ref}}|\, e_\sigma \le \gamma$ for some small
$\gamma$ (§"free constants" below). For Black–Scholes the ratio is closed-form,
$\mathcal{V}/\nu = d_1 d_2/\sigma$, so the gate reads
$\tfrac12 |d_1 d_2|\, e_\sigma/\sigma \le \gamma$ — i.e. deep in/out of the money, where
$|d_1 d_2|$ is large, linearization fails first. For the American case $\mathcal{V}_{\mathrm{ref}}$
has no closed form, but it is the *second* difference of the same three-point bump stencil that
already yields $\nu_{\mathrm{ref}}$ (§3), so it costs nothing extra.

Note the sign of the failure in the mango-option trigger case: the price is flat in $\sigma$
*below* $\sigma_{\mathrm{ref}}$, so the secant slope over
$[\sigma_{\mathrm{surf}}, \sigma_{\mathrm{ref}}]$ is **smaller** than the tangent slope
$\nu_{\mathrm{ref}}$. Therefore $e_\sigma$ *under*-states the true inversion distance, and may
under-state an outright non-existence. The curvature gate is what catches this.

### 2. Measurability (certification) condition

A point can be certified to IV tolerance $\tau_\sigma$ only if the reference's own price error,
pushed through the same inversion, is smaller than (a fraction $\rho$ of) that tolerance:

$$
\boxed{\;\frac{\delta_{\mathrm{ref}}}{\nu_{\mathrm{ref}}} \;\le\; \rho\,\tau_\sigma
\qquad\Longleftrightarrow\qquad
\nu_{\mathrm{ref}} \;\ge\; \frac{\delta_{\mathrm{ref}}}{\rho\,\tau_\sigma}\;}
$$

This is exactly Higham's rule of thumb "forward error $\lesssim$ condition number $\times$ backward
error" (Higham 2002, §1.6, p. 9) applied to $f: V \mapsto \sigma$, whose absolute condition
number is $\hat\kappa = |f'(V)| = 1/\nu$. It is also Jäckel's criterion in a different dress: *"in
order to be able to compute implied volatility to a relative accuracy of, say, $10^{-15}$, we first
need to have a Black function that near the solution is smooth down to the same relative
accuracy"* (Jäckel, *Let's Be Rational*, §7).

The key point is that this replaces an **unanchored absolute vega floor** (`|vega| < 1e-4`) with a
**derived** one: the floor on vega is *not* a free constant, it is $\delta_{\mathrm{ref}}/(\rho\tau_\sigma)$,
and it moves with the reference's own accuracy at that point and with the tolerance being claimed.

**How $\delta_{\mathrm{ref}}$ is measured, not chosen.** Two independent estimators, both already
affordable at table-build time:

1. **Richardson / grid-convergence.** With a grid refinement ratio $r_g$ (usually 2) and observed
   order $p$,
   $$
   \delta_{\mathrm{ref}} \;\approx\; F_s\,\frac{\bigl|V_h - V_{r_g h}\bigr|}{r_g^{\,p} - 1},
   \qquad
   p_{\mathrm{obs}} = \frac{\ln\bigl[(V_{4h}-V_{2h})/(V_{2h}-V_h)\bigr]}{\ln r_g},
   $$
   i.e. Roache's generalized-Richardson error estimator and Grid Convergence Index, with safety
   factor $F_s = 1.25$ when three or more grids are used and $F_s = 3.0$ for a two-grid comparison
   (Roache 1994, Roache 1994; formulas verified against the NASA Glenn CFD verification tutorial,
   which reproduces them — see "Sources not reached"). For $p=2$, $r_g=2$ this is the familiar
   $e \approx (V_h - V_{2h})/3$.
   **Do not assume $p = 2$: measure it.** For American options under a penalty/projected method
   with *constant* timesteps, Forsyth & Vetzal measure convergence ratios of $2.8$–$3.2$, i.e.
   $p \approx 1.5$, not $4$/$p=2$ (Forsyth & Vetzal 2002, Table 8.1 and §9). Quadratic
   convergence is only restored with Rannacher smoothing *plus* their variable-timestep selector
   (ratios $4.0$–$4.5$, Forsyth & Vetzal 2002, Table 11.1). A Richardson estimator hard-wired to $p=2$ on a
   constant-timestep American solve would understate $\delta_{\mathrm{ref}}$ by a large factor.

2. **The bump pair.** The three solves $V(\sigma \pm h), V(\sigma)$ used for FD vega also expose
   the noise floor of the solver: the residual non-smoothness of
   $\sigma \mapsto V_{\mathrm{ref}}(\sigma)$ across the stencil bounds what any root-finder can
   resolve. This is precisely Jäckel's "maximum attainable accuracy" argument: *"any root-finding
   procedure cannot resolve a root $\sigma^*$ below a relative resolution of $\Delta\nu$ if the
   objective function appears to have multiple roots within $\sigma^* \pm \Delta\sigma$ with
   $\Delta\sigma = \Delta\nu\cdot\sigma^*$"* (Jäckel, *Let's Be Rational*, §7).

### 3. Identifiability condition

The point is *identifiable* — a σ exists and is locally unique — only if all three hold:

$$
\text{(i)}\;\; \nu_{\mathrm{ref}} > 0 \text{ strictly};\qquad
\text{(ii)}\;\; V_{\mathrm{ref}} - I \;>\; \delta_{\mathrm{ref}} \;(\text{and } V_{\mathrm{surf}} - I > \delta_{\mathrm{ref}});\qquad
\text{(iii)}\;\; V_{\mathrm{ref}} \text{ strictly increasing in } \sigma \text{ on the bracket.}
$$

- (i)/(iii) are the standard statement for American options: *"the derivative of the option price
  with respect to the volatility, the option's Vega, becomes zero in the stopping region for
  American call and put options … $\partial\sigma/\partial V_{am} = 1/\text{Vega} \to \infty$ …
  When we invert the American Black–Scholes pricing problem in the stopping regions, there is no
  unique solution for the implied volatility. Therefore, the definition domain of Formula (7)
  should be the continuation region"* (Liu, Leitao, Borovykh & Oosterlee 2020, §2.3.1, eq. (9)).
- Existence follows from weak monotonicity + continuity of the American price in σ
  (Ekström 2004, Corollary 2.7 and Theorem 4.5) — but Ekström's monotonicity is *weak*
  ($\le$), which is exactly why flat plateaus, and hence non-uniqueness, are admissible.
- (ii) is the no-arbitrage existence condition: below intrinsic (or, for a European, above
  $b_{\max}$) no σ exists at all. Jäckel's normalized form makes the admissible window explicit:
  $0 \le b \le b_{\max} \le 1$ with $b = V/(\delta\sqrt{FK})$ and $b_{\max}=e^{\theta x/2}$
  (Jäckel, *Let's Be Rational*, eqs. (2.1)–(2.10)).
- The $\delta_{\mathrm{ref}}$ on the right of (ii) is what makes it *anchored*: "strictly above
  intrinsic" is not decidable at a resolution finer than the solver's own error, so the test must
  be "above intrinsic by more than the solver can measure".

**Note that (ii) and (i) are not redundant.** Liu et al. enforce both and say why: *"In principle,
Criterion (20) should cover Criterion (21), but for robustness reasons, both will be enforced"*
(Liu et al. 2020, §3.2.1, eqs. (20)–(21)). That is independent literature support for the *shape* of
mango-option's current two-filter score function — the defect is only that its two constants are
unanchored.

### 4. The resulting criterion, and which constants are measured vs. chosen

| Quantity | Status | Source of the value |
|---|---|---|
| $\nu_{\mathrm{ref}}$ | **measured** | central difference of the bump pair |
| $\mathcal{V}_{\mathrm{ref}}$ | **measured** | second difference of the same bump pair |
| $\delta_{\mathrm{ref}}$ | **measured** | Richardson/GCI over two or three resolutions (Roache 1994), with $p$ observed not assumed (Forsyth & Vetzal 2002) |
| $p$ (convergence order) | **measured** | three-grid ratio; do not hard-code 2 |
| $I$ (intrinsic) | exact | closed form |
| $\tau_\sigma$ (IV tolerance certified) | **free choice** | product decision; market convention is vol points / bps |
| $\rho$ (share of $\tau_\sigma$ given to the reference) | **free choice** | no literature value; $\rho \in [1/10, 1/3]$ is the usual engineering split |
| $F_s$ (Richardson safety factor) | **free, but literature-fixed** | Roache: 1.25 (≥3 grids), 3.0 (2 grids) |
| $\gamma$ (curvature gate) | **free choice** | no literature value; $\gamma \approx 0.1$ makes the linearization good to 10% |

**Final criterion.** Report $e_\sigma(p) = |V_{\mathrm{surf}} - V_{\mathrm{ref}}|/\nu_{\mathrm{ref}}$ and
mark it *certified at tolerance $\tau_\sigma$* iff

$$
\nu_{\mathrm{ref}} \ge \frac{\delta_{\mathrm{ref}}}{\rho\,\tau_\sigma}
\;\;\wedge\;\;
V_{\mathrm{ref}} - I > \delta_{\mathrm{ref}}
\;\;\wedge\;\;
\tfrac12\left|\frac{\mathcal{V}_{\mathrm{ref}}}{\nu_{\mathrm{ref}}}\right| e_\sigma \le \gamma .
$$

If the first two fail, the point is **not measurable** and must be excluded from IV-error
statistics (and said to be excluded, with the reason). If only the third fails, the point is
measurable but the *linear* metric is not trustworthy: either bracket-and-solve the true σ, or
report a lower bound and flag it. Reporting $e_\sigma$ where the first two fail is reporting the
condition number of the inversion, not the quality of the surface.

**Worked check on the trigger case.** $K=113.72$, $S=100$, $\tau=0.272$, $\sigma=0.14$, $r=0.055$,
$\$0.50$ dividend at $0.022$y, $V_{\mathrm{ref}}=14.126$, $\nu_{\mathrm{ref}}=0.22$,
$I=13.72$, surface miss $\$0.017$.
Relative condition number of the inversion (§1 below):
$\kappa_{\mathrm{rel}} = V/(\sigma\nu) = 14.126/(0.14\times 0.22) = 459$.
Relative price error $= 0.017/14.126 = 1.2\times10^{-3}$, i.e. the surface is performing at its
stated $10^{-4}$–$10^{-3}$ relative price accuracy. Pushed through:
$\delta\sigma/\sigma = 459 \times 1.2\times10^{-3} = 0.55$, $\delta\sigma = 0.077 = 770$ bps —
reproducing the reported 788 bps to within rounding. So the 788 bps is *arithmetically correct*;
the question is only whether it is *certifiable*. With a plausible $\delta_{\mathrm{ref}} \sim 10^{-3}$
absolute, the measurement floor is $\delta_{\mathrm{ref}}/\nu = 4.5\times10^{-3} = 45$ bps, so the
point passes the measurability gate and 788 bps is a real surface defect, not a measurement
artefact. What it fails is the **curvature gate**: with the price flat below $\sigma \approx 0.12$
and $\sigma_{\mathrm{ref}}=0.14$, the implied $\sigma_{\mathrm{surf}} \approx 0.063$ lies inside
(or below) the plateau, so the linearization is invalid over that span and 788 bps is a *lower
bound* on a possibly non-existent inversion. The correct report for such a point is
"IV not identifiable below σ≈0.12; price error $\$0.017$ ($1.2\times10^{-3}$ relative)", not a
bps number.

---

## 1. Conditioning of the price→σ map

### 1.1 The condition numbers

Let $f: V \mapsto \sigma$ be the inverse of $\sigma \mapsto V(\sigma)$. Using the scalar
definitions in §4 below:

- **Absolute condition number** $\hat\kappa = |f'(V)| = 1/\nu$. Hence $\delta\sigma \approx \delta V/\nu$.
- **Relative condition number**
  $$
  \kappa_{\mathrm{rel}} = \left|\frac{V f'(V)}{f(V)}\right| = \frac{V}{\sigma\,\nu}
  = \frac{1}{\eta_\sigma}, \qquad
  \eta_\sigma := \frac{\partial V}{\partial \sigma}\frac{\sigma}{V} = \frac{\sigma\nu}{V},
  $$
  i.e. the reciprocal of the **volatility elasticity of price**. So
  $(\delta\sigma/\sigma) = (\delta V/V)\,/\,\eta_\sigma$.

Both follow directly from the textbook scalar definitions (§4) applied to $f$; there is no
finance-specific content in them beyond identifying $f' = 1/\nu$.

### 1.2 Jäckel, "By Implication" (2006)

Jäckel works with the normalized price $b := p/(\delta\sqrt{FK})$ and $x := \ln(F/K)$
(*By Implication*, eq. (2.1)), with intrinsic $\iota := h(\theta x)\cdot\theta\cdot(e^{x/2}-e^{-x/2})$
(*By Implication*, eq. (2.9)).

- He states the asymptotics $\lim_{\sigma\to 0} b = \iota + x\varphi(x/\sigma)(\sigma/x)^3$
  (*By Implication*, eq. (2.8)) and comments: *"From equation (2.8) we can see what happens as volatility
  approaches zero (for $x \ne 0$): since $\varphi(y)$ decays more rapidly than $y^{-n}$ for any
  positive integer $n$ as $y \to \infty$, the Black option price does not permit for any regular
  expansion for small volatilities. **The extremely flat functional form of $b$ for small $\sigma$
  for $x \ne 0$** … is where the trouble starts."* (§2.1). This is the European analogue of the
  American exercise-region degeneracy: the map is *not* invertible to any useful precision where
  it is flat.
- On the accuracy target: *"any implied volatility solver should be able to produce a
  comparatively, i.e. relatively, accurate figure even for parameter combinations that mean that
  $\sigma$ is a very small or moderately large number … **This clearly requires any solver
  termination criterion to be based on relative accuracy in $\sigma$, not in function value.**"*
  (§1, p. 1). Directly relevant: a price-space tolerance is not a σ-space tolerance.
- On outright non-attainability: *"the zero levels at the back of all shown diagrams for non-zero
  $x$ and very small $\sigma$ represent outright calculation failures since for those parameter
  combinations the normalised Black function value is smaller than the smallest representable
  floating point number … **those areas in the parameter plane are not attainable in practice**"*
  (footnote 5, p. 3).

### 1.3 Jäckel, "Let's Be Rational" (2015)

The paper's central accuracy claim is framed as *attainable* accuracy, i.e. a bound set by the
conditioning of the price function, not by the root-finder:

- **The smoothness-resolution bound (§7).** *"Any root-finding procedure cannot resolve a root
  $\sigma^*$ below a relative resolution of $\Delta\nu$ if the objective function appears to have
  multiple roots within $\sigma^* \pm \Delta\sigma$ with $\Delta\sigma = \Delta\nu \cdot \sigma^*$.
  This is what we alluded to earlier when we referred to the maximum attainable accuracy: **in
  order to be able to compute implied volatility to a relative accuracy of, say, $10^{-15}$, we
  first need to have a Black function that near the solution is smooth down to the same relative
  accuracy.**"* This is the single most important statement for our purposes: the achievable σ
  accuracy is capped by the *price function's own* accuracy near the root. Our $\delta_{\mathrm{ref}}$
  plays the role of his round-off noise floor.
- **Subtractive cancellation bound (§6, eqs. (6.7)–(6.8)).** With $h=x/\sigma$, $t=\sigma/2$ and
  $\epsilon$ = `DBL_EPSILON`, the relative evaluation error of the normalized Black function is
  $$
  \frac{b_{\text{numerical}}}{b_{\text{exact}}} - 1 \;\approx\; \frac{\Phi(h)\varepsilon_3 + \varphi(h)t\varepsilon_4}{\varphi(h)t}
  \;\approx\; \frac{1}{t}\cdot\frac{\Phi(h)}{\varphi(h)}\cdot\varepsilon_3 ,
  $$
  and *"as $t \to 0$, the relative error grows like the inverse of $t$, and there is nothing we can
  do about it"* — unless the subtraction is avoided altogether (which is what the rest of §6 does).
- **The explicit condition-number-times-epsilon bound (§6, eq. (6.18)).** Near the upper bound
  $\beta \lesssim b_{\max}$:
  $$
  \left|\frac{\Delta\sigma}{\sigma}\right| \;\gtrsim\; \frac{b_{\max}}{b_{\max}-\beta}\cdot\epsilon
  $$
  with *"$\epsilon$ being as before the relative hardware accuracy. **The problem here is
  unsurmountable.** It is caused by the fact that the input number $\beta$, when it is, say, within
  $10^{-m}$ (relative) of $b_{\max}$, only contains approximately $(N-m)$ decimal digits of relevant
  information, with $N := |\log_{10}(\epsilon)|$ … **This limit case, however, is in practice of no
  concern since this is the situation of volatilities and prices being so high that prices have no
  discernible vega.**"* Note the form: it is literally (input error) × (condition number), and
  Jäckel's own diagnosis of the degenerate case is "no discernible vega".
- Related rule of thumb, footnote 2, p. 2: *"if a function $f$ has relative accuracy $\epsilon$,
  then its numerical second order derivative $f''$ can only attain $\sqrt\epsilon$, i.e. half on a
  logarithmic scale."* Relevant to our FD-vomma gate: a vomma computed by second differencing a
  reference of accuracy $\delta_{\mathrm{ref}}$ is itself only good to $\sim\sqrt{\delta_{\mathrm{ref}}}$
  relative, so the curvature gate should be treated as an order-of-magnitude test, not a precise one.

Jäckel's per-point quantity is therefore the **normalized price** $b = V/(\delta\sqrt{FK})$ and its
distance from its two limits ($\iota$ below, $b_{\max}$ above); attainable relative σ accuracy
degrades as $\epsilon$ divided by that distance (in normalized units).

---

## 2. Identifiability

### 2.1 American exercise region: vega is exactly zero, σ is not identified

Liu, Leitao, Borovykh & Oosterlee state it as cleanly as anyone (Liu et al. 2020, §2.3.1):

> "An important aspect when extracting the implied volatility is that the derivative of the option
> price with respect to the volatility, the option's Vega, becomes zero in the stopping region for
> American call and put options. It is well-known that, $|\Delta| = |\partial V_{am}/\partial S| = 1$,
> $\mathrm{Vega} = \partial V_{am}/\partial\sigma = 0$. (9) In other words, the American option
> prices do not depend on the volatility in the stopping regions. … Consequently,
> $\partial\sigma/\partial V_{am} = 1/\mathrm{Vega} \to \infty$. When we invert the American
> Black–Scholes pricing problem in the stopping regions, there is no unique solution for the implied
> volatility. Therefore, the definition domain of Formula (7) should be the continuation region."

They also note (§1) that root-finders *"may fail to converge when [they] accidentally explore the
early-exercise region (where the output appears insensitive to input parameter changes, as the
gradient, Vega, equals zero in that region)"*, and (§2.3.1) that *"these solutions may have
difficulties especially with deep in-the-money options. One of the reasons is that option prices
are insensitive to the underlying volatility deep in the money"*, citing Kutner's QAM paper.

Critically for the mango-option dividend case, the same paper shows an American put with **two
disjoint continuation regions** separated by a stopping region (their Figure 2, with
$r=-0.01$, $q=-0.06$: *"The option value in the solid black line hits the payoff function twice.
The stopping region is between the two early-exercise points."*, and Figure 5: *"There are two
isolated continuation regions."*). Discrete dividends produce the same topology in the time
direction: hold through the ex-date, exercise after. A root-finder started in the wrong region
converges to the wrong branch — they add a Remark to that effect (§2.3.1).

### 2.2 Their exclusion criteria — direct prior art for a low-vega filter

In building their training set they prescribe exactly two thresholds (Liu et al. 2020, §3.2.1):

$$
|V_{am}(S_t,K,\tau,r,q,\sigma) - H(K,S_t)| > \epsilon_1 \quad (20), \qquad
\mathrm{Vega} > \epsilon_2 \quad (21)
$$

with the comment: *"As early-exercise takes place with options that are ITM, the above two criteria
only apply to ITM samples. In principle, Criterion (20) should cover Criterion (21), but for
robustness reasons, both will be enforced."* They also apply **gradient squashing** — subtracting
intrinsic value and learning the *time value* instead — because "ANNs are not accurate when
functions with steep gradients need to be approximated" (§3.2.2). That is a second, independent
reason to make the surface carry time value rather than total price.

**They do not say how to choose $\epsilon_1, \epsilon_2$.** This is the gap the criterion in the
synthesis closes by tying both to $\delta_{\mathrm{ref}}$ and $\tau_\sigma$.

### 2.3 Existence and uniqueness for American implied volatility

- **Existence.** Liu et al.: *"Existence of $\sigma^*$ can be guaranteed by the monotonicity of the
  Black–Scholes equation with respect to the volatility in the holding region"* (§2.3.1). The
  underlying monotonicity theorem for American prices is Ekström's: for a **convex** payoff and
  $|\sigma_1(s,t)| \le |\sigma_2(s,t)|$ pointwise, $P(s,0;\sigma_1) \le P(s,0;\sigma_2)$
  (Ekström 2004, Corollary 2.7, Ekström 2004, attributed there to El Karoui–Jeanblanc-Picqué–Shreve 1998
  and Hobson 1998 under somewhat different conditions). Theorem 4.2 extends this to non-convex
  payoffs satisfying $g(as) \le a g(s)$ for $a \ge 1$ (equivalently $g(s)/s$ decreasing).
  Continuity in σ is Theorem 4.5.
- **Uniqueness fails wherever the inequality is not strict.** Ekström's results are all stated with
  weak inequality ($\le$). Nothing in the theory forbids a plateau, and in the stopping region the
  plateau is exact (Liu et al. eq. (9)). So: *monotone + continuous ⇒ the solution set is a closed
  interval, possibly a single point, possibly empty.* "Implied volatility" is only a function on
  the continuation region.
- **Non-existence.** A quote strictly below intrinsic, or above the no-arbitrage upper bound
  (for an American put, $P \le K$; for the European normalized form, $b \le b_{\max} = e^{\theta x/2}$,
  Jäckel, *Let's Be Rational*, eqs. (2.7)–(2.10)), has no implied volatility at all. With a numerical
  reference this must be tested at the reference's own resolution, hence the
  $V_{\mathrm{ref}} - I > \delta_{\mathrm{ref}}$ form.

---

## 3. Anchoring the reference's own price accuracy

### 3.1 What order of convergence is actually established

**Forsyth & Vetzal (2002)**, "Quadratic convergence for valuing American options using a penalty
method", SIAM J. Sci. Comput. 23(6):2095–2122. Two distinct "quadratic" claims must not be conflated:

1. **Quadratic convergence of the *nonlinear penalty iteration*** (Newton) at each timestep —
   Theorem 6.1: the iteration converges to the unique solution, monotonically, with **finite
   termination** ("for an iterate sufficiently close to the solution, the algorithm terminates in
   one iteration"), and is globally convergent with full Newton steps. Observed cost: 1.4–1.6
   iterations/timestep (§9).
2. **Second-order convergence of the *discretization*** — this is the one that licenses Richardson,
   and it is *conditional*:
   - European put, Rannacher smoothing, Table 7.1: change ratios $4.0, 4.0, 4.0$ → clean $p=2$.
     Without smoothing: $4.3, 2.2, 2.1$ — *"converges erratically"*.
   - **American put, constant timesteps**, Table 8.1: ratios $3.2, 3.0, 2.8$ (σ=0.2) and
     $3.2, 3.1, 2.9$ (σ=0.8). §9: *"We do not observe quadratic convergence for the implicit
     handling of the American constraint. An error ratio of about 2.8 would be consistent with
     global timestepping convergence at a rate of $O[(\Delta\tau)^{3/2}]$."* The cause is the
     exercise-boundary asymptotics $V = \text{const} + O(\tau^{3/2})$ (calls, eq. (9.1)) and
     $V = \text{const} + O[(\tau\log\tau)^{3/2}]$ (puts, eq. (9.2)), giving local CN error
     $O[(\Delta\tau)^3/\tau^{3/2}]$ (eq. (9.3)) and global $O[(\Delta\tau)^{3/2}]$ (eq. (9.4)).
   - **American put, variable timesteps** from selector (10.1) enforcing
     $\max_i|V_i^{n+1}-V_i^n| \simeq d$, Table 11.1: ratios $4.3, 4.0, 4.5$ (σ=0.2) and
     $4.3, 4.3, 4.2$ (σ=0.8) → quadratic restored, consistent with the analysis
     $\Delta\tau^{n+1} = O(d\sqrt{\tau^n})$ (9.7) ⇒ local error $O(d^3)$ (9.8) ⇒ global $O(d^2)$ (9.9).
   - The authors add the caveat: *"We make no claim that the above analysis of the time truncation
     error is in any way precise, but only suggestive of an appropriate timestepping strategy."*

**Rannacher timestepping.** Forsyth & Vetzal (§7), following Rannacher (1984), *Numer. Math.*
43:309–327: *"if we take constant timesteps with a Crank–Nicolson method, then second order
convergence (in time) can be guaranteed if (i) after each non-smooth initial state, we take two
fully implicit timesteps, and then use Crank–Nicolson thereafter (payoffs with discontinuous
derivatives qualify as non-smooth); and (ii) the initial conditions are $l_2$ projected onto the
space of basis functions."* Rationale: *"Crank–Nicolson is only A-stable, not strongly A-stable.
This means that some errors are damped very slowly, resulting in oscillations."* They note that for
a plain put *"no smoothing is required provided we have a node at K"*, since the piecewise-linear
initial condition is already in the basis space.

**Pooley, Forsyth & Vetzal (2003)**, IMA J. Numer. Anal. 23:241–267, "Numerical convergence
properties of option pricing PDEs with uncertain volatility" (the uncertain-volatility companion;
the discontinuous-payoff paper is Pooley, Vetzal & Forsyth, *J. Comp. Finance* 6(4):25–40, "Convergence
remedies for non-smooth payoffs in option pricing", which I could not open). Their Table 2 shows
fully implicit converging at ratio $\approx 1.97$ (first order) while *plain Crank–Nicolson gives
ratios $1.41, 1.34, 1.51$ — "either converging to a non-viscosity solution, or has a slowly growing
instability"*. With Rannacher (2 or 4 implicit steps) *"Both approaches give (nearly) quadratic
convergence … both methods appear to converge to the correct solution, with no evidence of
instability"* (Table 4). They also demonstrate the a-posteriori use directly: *"Assuming a linear
rate of convergence, the extrapolated solution using fully implicit timestepping (Table 2) is
2.2977, in excellent agreement with the results in Table 4."*

**Takeaway for mango-option:** the order $p$ that goes into a Richardson error estimate is a
property of the *scheme plus the timestep policy plus the payoff smoothing*, and the American
literature documents it landing anywhere in $[1, 2]$. It must be measured per configuration from a
three-resolution ratio, not assumed.

### 3.2 Why a two-resolution difference is a legitimate per-point error estimate

Roache's generalized Richardson framework is the canonical justification: the GCI is *"based upon a
grid refinement error estimator derived from the theory of generalized Richardson Extrapolation"*
and is *"recommended for use whether or not Richardson Extrapolation is actually used to improve
the accuracy, and in some cases even if the conditions for the theory do not strictly hold"*
(Roache 1994, abstract). Assuming an asymptotic error expansion $V_h = V_{\text{exact}} + C h^p + o(h^p)$,
two resolutions give $V_h - V_{r_g h} = C h^p (r_g^p - 1) + o(h^p)$, hence

$$
V_{\text{exact}} - V_h \;\approx\; -\frac{V_h - V_{r_g h}}{r_g^{\,p} - 1},
$$

i.e. $e \approx (V_h - V_{2h})/(2^p-1) = (V_h - V_{2h})/3$ for $p=2$, $r_g=2$. The safety factor
$F_s$ exists precisely because the asymptotic-range assumption is the weak link; Roache prescribes
$F_s = 1.25$ when the observed order has been confirmed from three or more grids and $F_s = 3.0$
for a bare two-grid comparison.

Three caveats specific to this application:
1. **Asymptotic range.** The estimate is only meaningful if $p_{\mathrm{obs}}$ is close to the
   theoretical $p$ and stable across refinement levels. Forsyth & Vetzal's own ratio tables are
   the template for checking this.
2. **Non-smooth convergence for American options.** Because the free boundary moves discretely
   with the grid, the error is not a smooth function of $h$; Geske–Johnson-style Richardson in the
   number of exercise dates is documented to converge non-uniformly for options with discontinuous
   exercise boundaries. Treat the Richardson number as an *estimate of magnitude*, use $F_s$, and
   never extrapolate the value into the reference itself without a separate check.
3. **The exercise-boundary region is the worst case**, and that is exactly where deep-ITM,
   low-vega, dividend-adjacent validation points live. Expect $\delta_{\mathrm{ref}}$ to be
   *largest* precisely where $\nu_{\mathrm{ref}}$ is smallest — which is why the two must be
   combined into a single measurability test rather than filtered independently.

---

## 4. Standard numerical-analysis framing

**Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002.**

§1.5 (pp. 6–7) defines the terms: for $\hat y \approx y = f(x)$, *"the value of $|\Delta x|$ (or
$\min|\Delta x|$), possibly divided by $|x|$, is called the **backward error**. The absolute and
relative errors of $\hat y$ are called **forward errors**."*

§1.6 (pp. 8–9), "Conditioning": from
$\hat y - y = f(x+\Delta x) - f(x) = f'(x)\Delta x + \frac{f''(x+\theta\Delta x)}{2}(\Delta x)^2$,

> "the quantity $c(x) = \left|\dfrac{x f'(x)}{f(x)}\right|$ measures, for small $\Delta x$, the
> relative change in the output for a given relative change in the input, and it is called the
> **(relative) condition number of $f$**."

and then the rule that licenses the whole criterion:

> "When backward error, forward error, and the condition number are defined in a consistent fashion
> we have the useful rule of thumb that
> **forward error $\lesssim$ condition number $\times$ backward error**,
> with approximate equality possible. One way to interpret this rule of thumb is to say that the
> computed solution to an ill-conditioned problem can have a large forward error. For even if the
> computed solution has a small backward error, this error can be amplified by a factor as large as
> the condition number when passing to the forward error."

**Trefethen & Bau, *Numerical Linear Algebra*, SIAM 1997, Lecture 12 "Conditioning and Condition
Numbers".** The definitions (only reachable to me via a course page reproducing them — see Sources
not reached):

$$
\hat\kappa(x) = \lim_{\delta\to 0}\ \sup_{\|\delta x\|\le\delta} \frac{\|\delta f\|}{\|\delta x\|} = \|J(x)\|,
\qquad
\kappa(x) = \lim_{\delta\to 0}\ \sup_{\|\delta x\|\le\delta} \frac{\|\delta f\|/\|f(x)\|}{\|\delta x\|/\|x\|} = \frac{\|J(x)\|}{\|f(x)\|/\|x\|},
$$

reducing in the scalar case to $\hat\kappa = |f'(x)|$ and $\kappa = |f'(x)|\,|x|/|f(x)|$ — the same
quantities as Higham's $c(x)$.

**Consequence for a certified metric.** A derived quantity $\sigma = f(V)$ can only be certified to
tolerance $\tau_\sigma$ where $\hat\kappa \times (\text{input error}) \le \tau_\sigma$. With
$\hat\kappa = 1/\nu$ and input error $\delta_{\mathrm{ref}}$, this *is* the measurability condition
$\nu \ge \delta_{\mathrm{ref}}/\tau_\sigma$. Nothing finance-specific is being assumed; the finance
content is only in identifying $f' = 1/\nu$ and in knowing that $\nu \to 0$ in the exercise region.

---

## 5. Prior art: vega-weighted and condition-aware error metrics

Two distinct constructions both get called "vega weighting" and must be kept apart.

**(a) Price error divided by vega = IV error (the conversion we want).**
Ackerer, Tagasovska & Vatter, "Deep Smoothing of the Implied Volatility Surface"
(NeurIPS 2020; arXiv:1906.05065), Appendix B, calibrate stochastic-volatility models by minimizing
the **vega-weighted RMSE**

$$
\sqrt{\frac{1}{N}\sum_{j=1}^{N}\left(\frac{\pi_j - \hat\pi_j}{\nu_j}\right)^2}
\tag{10}
$$

with the stated justification: *"The loss (10) is a computationally efficient approximation for the
implied volatility surface RMSE criterion which follows by observing that
$\sigma_j - \hat\sigma_j \approx \dfrac{\pi_j-\hat\pi_j}{\nu_j}$ when $\pi_j \approx \hat\pi_j$."*
They restrict $N$ to out-of-the-money options — i.e. they implicitly avoid the deep-ITM low-vega
regime rather than dividing by a near-zero $\nu_j$. In their conclusion they note that
*"the loss function could be improved by avoiding to penalize models for predictions inside the
spread, or using vega-weighting, as is commonly done by practitioners."*

**(b) Weighting *vol*-space errors by vega (an importance weight, not a conversion).**
Bianchetti & Carlicchi, "Interest Rates After The Credit Crunch" (arXiv:1103.2567, §on SABR
calibration) define a *vega-weighted error function*
$\mathrm{err}_w = \bigl[\sum_i (\sigma^{\mathrm{mkt}}_i - \sigma^{\mathrm{SABR}}_i)^2 w_i\bigr]^{1/2}$ with
$w_i = \nu_i/\sum_j \nu_j$, justified as: *"Weighting the errors by the sensitivity of the options
to shifts of the volatility allows, during the calibration procedure, to give more importance to
the near-ATM areas of the volatility surface, with high vega sensitivities and market liquidity,
and less importance to OTM areas, with lower vega and liquidity."* This is a **liquidity/relevance**
weighting and is the opposite direction to (a) — it *down*-weights low-vega points rather than
amplifying them. Both are defensible; conflating them is not.

**Choice of loss function, and why the choice matters.**
Christoffersen & Jacobs, "The importance of the loss function in option valuation",
*J. Financial Economics* 72:291–318 (2004; CIRANO WP 2003s-52). They define
$\mathrm{IV\,MSE}(\theta) = \frac1n\sum_i(\bar\sigma_i - \sigma_i(\theta))^2$ with
$\bar\sigma_i = BS^{-1}(C_i,\dots)$ (§2.1, eqs. (3)–(4)) alongside \$MSE and %MSE, and argue that
*"the choice of loss function is key because it implicitly assumes a particular error structure"*
and that estimation and evaluation loss functions must be **aligned**. Relevant here: if the
product's acceptance criterion is stated in IV terms, the surface's *build-time* objective should
also be in IV terms (or vega-weighted price terms) — not raw price RMSE. They also note the
mirror-image pathology for relative price loss: *"short time-to-maturity out-of-the-money options
with valuations close to zero will implicitly get assigned a lot of weight and can thus create
numerical instability."*

**Fitting in price space without weights.**
Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces", *Quantitative Finance* 14(1):59–71
(2014; arXiv:1204.0646), §5.2 give their calibration recipe: *"Given mid implied volatilities …
compute mid option prices using the Black–Scholes formula. Fit the square-root SVI surface by
minimizing sum of squared distances between the fitted prices and the mid option prices … change
SVI parameters slice-by-slice so as to minimize the sum of squared distances between the fitted
prices and the mid option prices with a big penalty for crossing either the previous slice or the
next slice."* They use **unweighted price distances**, and explicitly flag the sensitivity:
*"There are obviously many possible variations on this recipe. The objective function may be
changed … Changing the objective function on the other hand will make some difference especially
for very short expirations."* So Gatheral–Jacquier is **not** a source for vega weighting; it is a
source for "fitting in price space, unweighted, is a real choice with known short-dated
consequences". (I checked the full arXiv text; the word "vega" does not appear.)

**Explicitly excluding low-vega points from the sample.** The clearest primary statement remains
Liu et al. §3.2.1 eqs. (20)–(21) (quoted in §2.2 above): a time-value threshold *and* a vega
threshold, both applied only to ITM samples, with the vega threshold retained "for robustness
reasons" even though the time-value threshold should subsume it. I did not find a source that
derives either threshold from a measured reference accuracy — that appears to be the gap.

---

## Sources reached / not reached

### Reached (opened and read the primary text)

- **Jäckel, P. (2015/2016), "Let's be rational", *Wilmott* 2015(75):40–53**; revision 1035′ of 25 Mar 2016, 12 pp. — <http://www.jaeckel.org/LetsBeRational.pdf> (note: the site's TLS cert does not match the hostname; fetched over plain HTTP). Read in full; §§2, 3, 5, 6, 7 and eqs. (2.1)–(2.10), (6.1)–(6.18) quoted above.
- **Jäckel, P. (2006/2010), "By implication", *Wilmott* magazine, Nov 2006**, 6 pp. — <http://www.jaeckel.org/ByImplication.pdf>. Read §§1, 2, 2.1, 3.
- **Forsyth, P.A. & Vetzal, K.R. (2002), "Quadratic convergence for valuing American options using a penalty method", *SIAM J. Sci. Comput.* 23(6):2095–2122** — <https://cs.uwaterloo.ca/~paforsyt/con7.pdf> (author preprint, 28 pp.). Read §§1, 4, 6, 7, 8, 9, 10, 11 and Tables 7.1, 8.1, 11.1.
- **Pooley, D.M., Forsyth, P.A. & Vetzal, K.R. (2003), "Numerical convergence properties of option pricing PDEs with uncertain volatility", *IMA J. Numer. Anal.* 23:241–267** — <https://cs.uwaterloo.ca/~paforsyt/numuncert.pdf> (preprint, 24 pp.). Read §§3.2, 4 and Tables 2–5.
- **Liu, S., Leitao, Á., Borovykh, A. & Oosterlee, C.W. (2020/2022), "On Calibration Neural Networks for extracting implied information from American options"** — arXiv:2001.11786 <https://arxiv.org/pdf/2001.11786>; published as "On a Neural Network to Extract Implied Information from American Options", *Applied Mathematical Finance* 29(3) (2022). Read §§1, 2.3.1, 3.2.1, 3.2.2 and Figures 2, 3, 5.
- **Ekström, E. (2004), "Properties of American option prices", *Stochastic Processes and their Applications* 114:265–278** — <https://www.ma.imperial.ac.uk/~ajacquie/IC_AMDP/IC_AMDP_Docs/Literature/Projects/Ekstrom.pdf>. Read §§2, 4.1, 4.3 (Corollary 2.7, Theorems 4.2, 4.3, 4.5).
- **Higham, N.J. (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM** — full-text PDF at <http://ftp.demec.ufpr.br/CFD/bibliografia/Higham_2002_Accuracy%20and%20Stability%20of%20Numerical%20Algorithms.pdf>. Read §§1.5–1.6, pp. 6–9.
- **Ackerer, D., Tagasovska, N. & Vatter, T. (2020), "Deep Smoothing of the Implied Volatility Surface", NeurIPS 2020** — arXiv:1906.05065 <https://arxiv.org/pdf/1906.05065>. Read §5 and Appendix B eq. (10).
- **Gatheral, J. & Jacquier, A. (2014), "Arbitrage-free SVI volatility surfaces", *Quantitative Finance* 14(1):59–71** — arXiv:1204.0646 <https://arxiv.org/pdf/1204.0646>. Read §5.2 in full; searched the whole text for "vega"/"weight".
- **Christoffersen, P. & Jacobs, K. (2004), "The importance of the loss function in option valuation", *J. Financial Economics* 72:291–318** — CIRANO working paper 2003s-52, <https://cirano.qc.ca/files/publications/2003s-52.pdf>. Read §§1, 2.1, 2.2 and eqs. (1)–(9).
- **Bianchetti, M. & Carlicchi, M. (2012), "Interest Rates After The Credit Crunch: Multiple-Curve Vanilla Derivatives and SABR"** — arXiv:1103.2567 <https://arxiv.org/pdf/1103.2567>. Read the SABR calibration section, eqs. [1]–[2].
- **Burkovska, O., Glau, K., Mahlstedt, M. & Wohlmuth, B., "Model reduction for calibration of American options"** — arXiv:1611.06452. Skimmed; confirms the de-Americanization framing but adds nothing on conditioning.

### Not reached (cited from the secondary source that reproduces them, or from metadata only)

- **Trefethen, L.N. & Bau, D. (1997), *Numerical Linear Algebra*, SIAM, Lecture 12 "Conditioning and Condition Numbers".** The author's page (<https://people.maths.ox.ac.uk/trefethen/text.html>) posts only Lectures 1–5. The definitions quoted in §4 were taken from a course page that reproduces Lecture 12's definitions verbatim (<https://bilman.github.io/Lecture-12.html>); they agree with Higham's §1.6, which I did read in the original.
- **Roache, P.J. (1994), "Perspective: A Method for Uniform Reporting of Grid Refinement Studies", *ASME J. Fluids Eng.* 116(3):405–413** (DOI 10.1115/1.2910291). Paywalled at ASME. Abstract read via the ASME/SciSpace listings; the GCI and observed-order formulas in §3.2 were taken from the NASA Glenn "Examining Spatial (Grid) Convergence" tutorial (<https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html>), which states them with attribution to Roache. Also relevant and not opened: Roache, *Verification and Validation in Computational Science and Engineering*, Hermosa 1998.
- **Rannacher, R. (1984), "Finite element solution of diffusion problems with irregular data", *Numer. Math.* 43:309–327.** Paywalled; its content is quoted here as restated by Forsyth & Vetzal §7 and Pooley–Forsyth–Vetzal §4, both of which I read in the original.
- **Pooley, D.M., Vetzal, K.R. & Forsyth, P.A. (2003), "Convergence remedies for non-smooth payoffs in option pricing", *J. Computational Finance* 6(4):25–40.** Paywalled at Risk.net; only the abstract was reachable (averaging initial data, shifting the grid, projection — each insufficient alone, sufficient when combined with a special timestepping method). The uncertain-volatility companion paper was read instead.
- **Kutner, G.W., "Determining the implied volatility for American options using the QAM"** — cited by Liu et al. as the source for "difficulties especially with deep in-the-money options"; not located in full text.
- **Fengler, M.R. (2005), *Semiparametric Modeling of Implied Volatility*, Springer; and Fengler (2009), "Arbitrage-free smoothing of the implied volatility surface", *Quantitative Finance* 9(4):417–428.** Both paywalled; I could not verify whether Fengler's spline weights are vega-based, so no claim is made about them here.
- **Cont, R. & da Fonseca, J. (2002), "Dynamics of implied volatility surfaces", *Quantitative Finance* 2(1):45–60.** Not opened; they work directly in IV space, so they are a precedent for the IV-space metric rather than for vega weighting, but I did not verify a weighting statement.
- **Vomma $= \nu\, d_1 d_2/\sigma$** (used in §1 of the synthesis) is standard and was cross-checked only against secondary references; it also follows in two lines from differentiating $\nu = S e^{-q\tau}\varphi(d_1)\sqrt\tau$.
