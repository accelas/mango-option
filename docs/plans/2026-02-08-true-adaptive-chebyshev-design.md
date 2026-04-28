# True Adaptive Chebyshev Design

> Design specification for hierarchical, nested Chebyshev-like grids with local
> hp-refinement and incremental PDE reuse. Replaces the current global
> fixed-degree tensor approach.

**Context:** The global tensor Chebyshev 4D interpolant (40x15x15x10 CGL nodes)
achieves 1-4 bps at T>=180d with 150 PDE solves, but cannot adapt: CGL nodes are
not nested, headroom couples degree to domain, and the refinement loop lacks causal
dimension attribution. This design fixes all three.

**Goal:** Adaptive Chebyshev 4D EEP interpolation with incremental PDE cost,
local resolution control, and bucketed convergence targets.

---

## Core Idea

Use hierarchical, nested Chebyshev-like grids + local hp refinement:

1. **p-refine** where smooth (raise polynomial degree).
2. **h-refine** where non-smooth (split domain into local elements).
3. **Reuse PDE slices** incrementally via nested nodes on expensive axes (sigma, rate).

---

## 1. Mathematical Formulation

1. Map each axis to [-1, 1] on a **fixed physical domain**.
   No n-dependent headroom during refinement.
2. Use **nested Clenshaw-Curtis levels** per axis:
   `x_{l,j} = cos(pi * j / 2^l)`, `j = 0..2^l`.
3. Represent surrogate as hierarchical sum of increments:
   `f ≈ Σ_{i in I} Δ_i f`, where `i = (ix, it, is, ir)` is a multi-index level.
4. Use **local elements** in difficult regions (short tau, exercise-boundary
   neighborhoods). Inside each element, use tensor Chebyshev interpolation.

This avoids global re-fit instability and gives local resolution control.

---

## 2. Data Structure

1. Maintain an **element tree** over (x, tau, sigma, rate); in practice start
   with splits on x, tau.
2. Each element stores:
   - bounds
   - per-axis polynomial levels
   - interpolation coefficients
   - local error indicators
3. **Global evaluator:**
   - find element(s) covering query
   - blend across overlaps with partition-of-unity weights
   - return price/vega/IV objective safely

---

## 3. Incremental PDE Reuse (critical)

1. PDE cost is driven by `N_sigma * N_rate`.
2. With nested levels, refining sigma adds only new sigma nodes:
   `new_solves = ΔN_sigma * N_rate_existing`.
3. Refining rate similarly:
   `new_solves = N_sigma_existing * ΔN_rate`.
4. **Refining x: zero new PDE solves.** Resample cached snapshot slices at new
   moneyness nodes via spline interpolation.
5. **Refining tau: two modes.**
   - **Precomputed superset (memory-heavy):** At initialization, solve all
     (sigma, rate) slices with max-level snapshot times. Tau refinement then
     resamples from existing snapshots at zero PDE cost, but requires
     `O(2^L_max)` snapshot storage per slice.
   - **Re-solve (default):** Adding new tau nodes requires re-solving all
     `N_sigma * N_rate` slices with the augmented snapshot schedule. Cost
     equals a full sigma/rate rebuild. Prefer this mode unless memory budget
     allows the superset approach.

   The cost model must distinguish these two axes. The adaptive loop (section 4)
   must account for tau-refine cost correctly when scoring ROI.

This fixes the "full rebuild every iteration" failure mode for sigma, rate, and
x axes. Tau refinement cost depends on the chosen mode.

---

## 4. True Adaptive Loop

1. Initialize coarse levels, e.g. `(lx, lt, ls, lr) = (4, 3, 3, 3)` and one element.
2. Build current surrogate from cached PDE slices.
3. Validate on stratified probes plus adversarial probes near boundaries.
4. Compute **local indicators:**
   - hierarchical surplus magnitude
   - holdout residual in price
   - IV-weighted residual: `|ΔP| / max(vega, vega_floor)`
   - monotonicity/arbitrage sanity checks
5. Generate **candidate actions:**
   - p-refine axis d in element e
   - h-split element e (usually in x or tau)
6. Score each candidate by **ROI:**
   `score = expected_error_drop / total_incremental_cost`

   **Total incremental cost** includes PDE solves, coefficient rebuild, and
   memory. For zero-PDE actions (x-refine, tau-refine with precomputed
   superset), use a floor cost equal to the rebuild/resample work to prevent
   infinite ROI from dominating selection. Specifically:
   - `cost = max(pde_cost + rebuild_cost, cost_floor)`
   - `cost_floor` calibrated so that a zero-PDE x-refine competes fairly with
     a small sigma-refine, e.g. `cost_floor = time_to_resample_one_element`.
   - Among actions at the floor, break ties by expected error drop (greedy).
7. Apply top-k actions under per-iteration PDE budget.
8. **Stop when bucketed targets met** (p95/p99 by regime), not just global max.

---

## 5. Dimension Attribution That Actually Works

Do not use one random 4D max point to pick an axis. Use **action-based
attribution:**

1. For each candidate refine action, estimate local error drop from surplus
   decay or mini cross-validation.
2. Compare against incremental solve cost.
3. Choose best ROI action.

This gives causal attribution instead of tie-break artifacts.

---

## 6. Handling Non-Smooth EEP Regions

1. Keep **local h-refinement** near short maturities and exercise boundary
   transition.
2. **Constraint model:** Positivity and monotonicity must be enforced at
   fit/element level, not deferred to evaluation-time clipping. Options:
   - **Constrained least-squares post-processing** per element: project
     Chebyshev coefficients onto the convex cone satisfying positivity and
     monotonicity at a dense check grid. This preserves C^0 continuity within
     elements.
   - **Penalty terms** in the surplus computation that down-weight actions
     producing constraint violations.
3. **Derivative continuity requirement:** The IV solve path uses Newton
   iteration, which requires continuous first derivatives (vega). Element
   interfaces must provide at least **C^1 continuity** via:
   - Overlap blending with C^1 partition-of-unity weights (e.g. smooth bump
     functions, not linear ramps).
   - Overlap width >= 2 * local mesh spacing to ensure the blended region has
     enough polynomial support.
4. Blending must not mask approximation defects. The error indicator (section 4)
   must evaluate accuracy **within** each element independently, before blending.

Global single-polynomial Chebyshev is the wrong tool for these regions.

---

## 7. Domain Policy

1. Define physical domains from user ranges.
2. Add **fixed padding once** (or no padding with clamp policy).
3. Keep domain fixed during refinement. Only levels/degrees change.

This removes the degree-vs-domain confound completely.

4. **Domain expansion action:** If probes near domain boundaries exceed error
   thresholds (boundary hit-rate > 5% of validation probes), the adaptive loop
   may trigger a domain expansion:
   - Expand the affected axis by one padding unit.
   - Invalidate and rebuild all cached slices on that axis:
     - **x:** resample only (zero PDE cost).
     - **sigma, rate, tau:** full re-solve of all `N_sigma * N_rate` slices
       (tau expansion requires new snapshot times, same cost as sigma/rate
       unless using precomputed superset mode from section 3).
   - Apply hysteresis: once expanded, do not shrink back in the same build
     session to avoid oscillation.
   - If expansion is triggered more than twice on the same axis, **hard-route
     to B-spline adaptive fallback** — the user range is likely misspecified
     or the problem regime requires a different method.

---

## 8. Build Plan

Each phase has mandatory verification gates. A phase cannot proceed to the next
until all gates pass.

### Phase A: Nested levels + PDE slice cache

Replace CGL non-nested counts with nested Clenshaw-Curtis levels on sigma/rate.
Add PDE slice cache keyed by node identity.

**Verification gates:**
- [ ] Cache correctness: interpolant built from cached slices matches
  interpolant built from fresh solves to machine epsilon (bitwise on
  coefficients after rounding to 1e-14).
- [ ] Incremental cost accounting: refining sigma from level l to l+1 solves
  exactly `ΔN_sigma * N_rate` new PDEs (instrumented counter).
- [ ] No regression: locked baseline accuracy (40x15x15x10, 150 PDE) reproduced
  within 0.5 bps of current results on the standard probe set.

### Phase B: Surplus indicators + ROI selector

Implement hierarchical surplus indicators and ROI candidate selector.

**Verification gates:**
- [ ] ROI scores are finite for all candidate actions (no division by zero;
  floor cost applied for zero-PDE actions).
- [ ] Tau-refine cost correctly reflects re-solve mode (not reported as zero).
- [ ] Convergence by bucket: p95 improves monotonically (within noise) for
  each (sigma, tau) bucket across iterations. Log per-bucket metrics.
- [ ] Adaptive build with 200 PDE budget achieves p95 <= 10 bps for T>=90d
  σ=30% (matching or beating the locked baseline).

### Phase C: Local element splits + overlap blending

Add local element splits on x/tau with overlap blending.

**Verification gates:**
- [ ] C^1 continuity at element interfaces: numerical derivative (central
  difference, h=1e-7) matches across interface to 1e-4 relative.
- [ ] Monotonicity: EEP(x) is non-increasing in x = ln(S/K) for puts (deeper
  ITM = more early exercise value), verified on a dense 1000-point x-grid
  per element.
- [ ] Solve success rate >= 95% for T>=30d probes (Brent convergence).
- [ ] Short-maturity improvement: T=60d p95 < 15 bps (currently 100+ bps).

### Phase D: Productionize with fallback

Routing fallback to B-spline adaptive when confidence fails.

**Verification gates:**
- [ ] Fallback triggers correctly when domain expansion limit exceeded.
- [ ] End-to-end IV accuracy no worse than B-spline adaptive on the full
  standard probe set (all maturities, all vols).
- [ ] Build time within 2x of B-spline adaptive for equivalent accuracy tier.
- [ ] No regression on existing `bazel test //...` suite.

---

## 9. Expected Outcome

1. Real adaptivity with incremental cost.
2. Avoid uniform over-refinement.
3. Target short-end hard regions locally.
4. Keep B-spline as universal fallback until this matures.

---

## Current Findings Supporting This Design

### Anisotropic Sweep Results (locked baseline)

Config: 40x15x15x10, frozen domains, use_tucker=false, 150 PDE solves.

| Regime | q=0 p95 | q=2% p95 |
|--------|---------|----------|
| T>=1y, σ=30% | ~2 bps | ~2 bps |
| 60d-180d, σ=30% | ~8 bps | ~8 bps |
| T>=1y, σ=15% | ~3 bps | ~3 bps |
| T<60d | 100+ bps | 100+ bps |

### Key Experimental Insights

- **3D dimensionless hits ~10-15 bps floor** at T=90d-1y because EEP depends on
  σ beyond what κ=2r/σ² captures. 4D achieves 1-4 bps in same regime.
- **Rate-axis ablation:** Domain tightening dominant at T>=1y (29→3 bps same
  degree-5). Higher degree dominant at T=90d (47→7 bps same domain).
- **Tucker HOSVD:** Zero compression. EEP is full-rank in all modes. Disabled.
- **Ns=20 worse than Ns=15** with frozen domain: high-degree global interpolation
  amplifies PDE noise near exercise boundary.
- **Global single-polynomial Chebyshev cannot handle T<60d.** Exercise boundary
  sharpness requires local h-refinement, not more global nodes.
