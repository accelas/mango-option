<!-- SPDX-License-Identifier: MIT -->
# Adaptive Grid Refinement for American Option Price Interpolation

## Problem Statement

We solve the American option pricing problem via finite difference methods (FDM) on a transformed spatial-temporal grid. To enable fast repeated queries, we precompute option prices on a sparse multidimensional grid and use interpolation to evaluate off-grid points.

**Goal**: Achieve implied volatility (IV) interpolation error below a target threshold (e.g., 5 basis points) by adaptively refining the moneyness grid in high-error regions.

## 1. Price Table Structure

We maintain a 4D lookup table:

$$V(m, \tau, \sigma, r)$$

where:
- $m \in [m_{\min}, m_{\max}]$ is **moneyness** (spot/strike ratio)
- $\tau \in [\tau_{\min}, \tau_{\max}]$ is **time to maturity**
- $\sigma \in [\sigma_{\min}, \sigma_{\max}]$ is **volatility**
- $r \in [r_{\min}, r_{\max}]$ is **risk-free rate**

Each dimension is discretized:
- Moneyness: $\{m_1, m_2, \ldots, m_{N_m}\}$ (initially log-spaced, adaptively refined)
- Maturity: $\{\tau_1, \tau_2, \ldots, \tau_{N_\tau}\}$ (fixed, typically linear)
- Volatility: $\{\sigma_1, \sigma_2, \ldots, \sigma_{N_\sigma}\}$ (fixed, typically linear)
- Rate: $\{r_1, r_2, \ldots, r_{N_r}\}$ (fixed, typically linear)

**Total grid points**: $N_{\text{total}} = N_m \times N_\tau \times N_\sigma \times N_r$

Each grid point $V_{i,j,k,\ell} = V(m_i, \tau_j, \sigma_k, r_\ell)$ is computed via FDM solution of the Black-Scholes PDE with early exercise boundary conditions.

## 2. Interpolation Strategy

For off-grid queries $(m, \tau, \sigma, r)$, we use **4D cubic interpolation**:

$$\hat{V}(m, \tau, \sigma, r) = \mathcal{I}[\{V_{i,j,k,\ell}\}](m, \tau, \sigma, r)$$

where $\mathcal{I}$ denotes the 4D cubic interpolation operator (typically tensor-product natural cubic splines).

## 3. Error Metric: Implied Volatility Error

Direct price interpolation error $|V - \hat{V}|$ is not a good metric because:
1. Prices vary by orders of magnitude across the parameter space
2. Trading decisions are made in IV space, not price space

Instead, we measure error in **implied volatility (IV) space**:

### 3.1 IV Error Calculation

For a validation point $(m, \tau, \sigma, r)$:

1. **Ground truth**: Compute exact price via FDM:
   $$V_{\text{exact}} = \text{FDM}(m, \tau, \sigma, r)$$

2. **Interpolated price**: Query the price table:
   $$\hat{V} = \mathcal{I}[\{V_{i,j,k,\ell}\}](m, \tau, \sigma, r)$$

3. **Invert both to IV**:
   - $\sigma_{\text{exact}} = \text{IV}^{-1}(V_{\text{exact}}, m, \tau, r)$
   - $\hat{\sigma} = \text{IV}^{-1}(\hat{V}, m, \tau, r)$

4. **IV error in basis points**:
   $$\epsilon_{\text{IV}} = |\sigma_{\text{exact}} - \hat{\sigma}| \times 10000 \quad \text{(bp)}$$

### 3.2 Validation Sample Generation

We generate $N_{\text{samples}}$ validation points via **stratified random sampling**:

For each dimension $(m, \tau, \sigma, r)$:
1. Select random grid cell $[x_i, x_{i+1}]$ (uniform over cells)
2. Sample uniformly within cell: $x \sim \mathcal{U}(x_i, x_{i+1})$

This ensures coverage across the entire parameter space while focusing on interpolation (not extrapolation).

### 3.3 Error Statistics

From $\{\epsilon_{\text{IV}}^{(1)}, \epsilon_{\text{IV}}^{(2)}, \ldots, \epsilon_{\text{IV}}^{(N)}\}$, we compute:

- **Mean error**: $\bar{\epsilon} = \frac{1}{N}\sum_{i=1}^N \epsilon_i$
- **Median error**: $\tilde{\epsilon} = \text{median}(\{\epsilon_i\})$
- **P95 error**: $\epsilon_{95} = \text{quantile}_{0.95}(\{\epsilon_i\})$
- **P99 error**: $\epsilon_{99} = \text{quantile}_{0.99}(\{\epsilon_i\})$
- **Coverage fraction**: $f_{\text{target}} = \frac{|\{i : \epsilon_i < \epsilon_{\text{target}}\}|}{N}$

**Convergence criterion**: $\epsilon_{95} < \epsilon_{\text{target}}$ and $f_{\text{target}} > 0.95$

## 4. Adaptive Refinement Algorithm

### 4.1 High-Level Structure

```
Input: Initial grid {m₁, ..., m_N}, target error ε_target, max iterations K
Output: Refined grid achieving ε₉₅ < ε_target

For k = 1 to K:
    1. Precompute prices V(mᵢ, τⱼ, σₖ, rₗ) via FDM for all grid points
    2. Build interpolation structures (e.g., spline coefficients)
    3. Validate: compute IV errors on N_samples test points
    4. If ε₉₅ < ε_target and f_target > 0.95: CONVERGED
    5. Identify high-error regions
    6. Generate refinement points in those regions
    7. Expand moneyness grid: {m₁, ..., m_N} ← merge_sorted({m₁, ..., m_N} ∪ {new points})
```

### 4.2 High-Error Region Identification

**Input**: Validation results with IV errors $\{\epsilon_i\}$ and locations $\{(m_i, \tau_i, \sigma_i, r_i)\}$

**High-error sample**: Any validation point with $\epsilon_i \geq \epsilon_{\text{target}}$

**Interval binning**:
- Partition moneyness axis into intervals $I_j = [m_j, m_{j+1}]$ for $j = 1, \ldots, N_m - 1$
- For each interval, count high-error samples:
  $$C_j = |\{i : m_i \in I_j \text{ and } \epsilon_i \geq \epsilon_{\text{target}}\}|$$

**Refinement threshold**: $C_{\min} = 3$ (require ≥3 high-error samples to refine an interval)

### 4.3 Refinement Point Selection

For each interval $I_j = [m_j, m_{j+1}]$ with $C_j \geq C_{\min}$:

1. Add midpoint (geometric mean for log-spaced grids):
   $$m_{\text{new}} = \sqrt{m_j \cdot m_{j+1}}$$

2. Limit: Add at most $N_m$ new points per iteration (prevents grid explosion)

**Grid merging**:
1. Merge old and new points: $\mathcal{M} = \{m_1, \ldots, m_N\} \cup \{m_{\text{new},1}, \ldots, m_{\text{new},K}\}$
2. Sort and remove duplicates (with tolerance $\delta = 10^{-10}$)
3. Update grid: $\{m_1, \ldots, m_{N'}\} \leftarrow \text{sorted\_unique}(\mathcal{M})$

### 4.4 Price Preservation During Grid Expansion

**Critical invariant**: After grid expansion, prices at original grid points must be **exactly preserved**.

**Implementation**:
1. Allocate new price array with dimension $N_{\text{total}}' = N_m' \times N_\tau \times N_\sigma \times N_r$
2. Initialize all entries to NaN
3. For each old grid point $(m_{\text{old},i}, \tau_j, \sigma_k, r_\ell)$:
   - Find new index: $i_{\text{new}} = \text{find\_index}(m_{\text{old},i}, \{m_1', \ldots, m_{N_m'}'\})$
   - Copy price: $V'_{i_{\text{new}},j,k,\ell} = V_{i,j,k,\ell}$
4. On next `precompute()` call, only compute NaN entries (new points)

This avoids recomputing old points and ensures no numerical drift.

## 5. Current Implementation Status

### 5.1 Test Configuration

**Initial grid** (AccuracyImprovement test):
- Moneyness: 10 points, log-spaced $[0.7, 1.3]$
- Maturity: 4 points $\{0.1, 0.25, 0.5, 1.0\}$ years
- Volatility: 4 points $\{0.15, 0.20, 0.25, 0.30\}$
- Rate: 2 points $\{0.02, 0.05\}$
- **Total**: $10 \times 4 \times 4 \times 2 = 320$ grid points

**Target**: $\epsilon_{95} < 5$ bp

**FDM solver configuration**:
- Spatial grid: 101 points
- Time steps: 500 (adaptive per maturity)
- TR-BDF2 scheme with implicit obstacle conditions

### 5.2 Expected vs. Observed Behavior

**Test: AccuracyImprovement (lines 41-95 in `adaptive_accuracy_test.cc`)**

#### Expected Behavior

Under the adaptive refinement algorithm, we expect:

1. **Initial iteration** (coarse grid):
   - Large interpolation errors due to sparse grid (10 points over $[0.7, 1.3]$)
   - Expected: $\epsilon_{95} \approx 100\text{-}500$ bp

2. **Subsequent iterations** (refined grid):
   - Errors should decrease **monotonically**: $\epsilon_{95}^{(k+1)} < \epsilon_{95}^{(k)}$
   - With 6-12 new points per iteration, expect error reduction:
     - **Iteration 2** (16 points): $\epsilon_{95} \approx 50\text{-}200$ bp (2-5× improvement)
     - **Iteration 3** (28 points): $\epsilon_{95} \approx 10\text{-}50$ bp (continuing improvement)
   - **Convergence**: Within 3-5 iterations, achieve target $\epsilon_{95} < 5$ bp

3. **Theoretical justification**:
   - Cubic splines: interpolation error $\sim O(h^4)$ where $h$ is grid spacing
   - Doubling grid density: $h \rightarrow h/2 \Rightarrow$ error $\rightarrow$ error/16
   - Conservative estimate with PDE discretization error: factor 4-8× improvement per refinement

#### Actual Observed Behavior

| Iteration | Grid Size $N_m$ | New Points Added | Mean $\bar{\epsilon}$ (bp) | P95 $\epsilon_{95}$ (bp) | P99 $\epsilon_{99}$ (bp) | Coverage $f_{5\text{bp}}$ | High-Error Samples |
|-----------|-----------------|------------------|----------------------------|--------------------------|--------------------------|---------------------------|--------------------|
| **1**     | 10              | —                | 583.84                     | **1134.84**              | 1431.61                  | 0.0%                      | 70/100             |
| **2**     | 16              | 6                | 585.01                     | **1134.62**              | 1431.98                  | 0.0%                      | 71/100             |
| **3**     | 28              | 12               | 585.14                     | **1134.68**              | 1431.92                  | 0.0%                      | 71/100             |

**Key Observations**:

1. ✅ **Grid expansion succeeds**: $N_m = 10 \rightarrow 16 \rightarrow 28$ (180% growth)
2. ✅ **Refinement points identified**: 6 and 12 points added in correct locations (intervals with ≥3 high-error samples)
3. ✅ **High-error sample count consistent**: ~70/100 samples fail (suggests validation samples are valid, not NaN)
4. ❌ **P95 error is CONSTANT**: $\epsilon_{95} = 1134.7 \pm 0.1$ bp across all three iterations
   - **Expected**: Monotonic decrease to ~5-50 bp
   - **Observed**: No improvement whatsoever (within 0.01% variation)
5. ❌ **Mean error also constant**: $\bar{\epsilon} = 584.7 \pm 0.7$ bp (similarly invariant)
6. ❌ **Errors are enormous**: 1134 bp = 11.34% implied volatility difference
   - For context: if true IV = 20%, interpolated IV = 31.34% or 8.66%
   - This is not "refinement needed" but "catastrophic interpolation failure"

#### Diagnostic Questions

**Why are errors EXACTLY constant?**

The error values are not just "not improving" — they are **numerically identical** to 3 decimal places across iterations. This suggests:

1. **Hypothesis**: New grid points are not being interpolated at all
   - Interpolation still uses the original 10-point grid
   - New points are computed but ignored during interpolation

2. **Hypothesis**: Interpolation workspace is stale
   - Cubic spline coefficients precomputed for 10-point grid
   - After expansion to 16/28 points, old coefficients still used
   - Result: interpolation unchanged despite grid growth

3. **Hypothesis**: Validation samples are identical
   - Random seed not updated between iterations
   - Same 100 samples tested every time
   - But samples might be positioned such that they always query the same interpolation region

4. **Hypothesis**: Grid expansion corrupts the price table
   - Despite stride formula fix (PR #65), some subtle corruption remains
   - New points have garbage values
   - Interpolation uses corrupted data → produces same wrong results

5. **Hypothesis**: Coordinate transform issue
   - Prices stored in one coordinate system (e.g., log-moneyness)
   - Interpolation queries in different system (e.g., raw moneyness)
   - Result: completely wrong interpolation behavior that is insensitive to grid refinement

**Why are errors so large?**

An 11% IV error is not typical of cubic spline interpolation, even on coarse grids. This suggests a systematic error, not just insufficient resolution.

**Possible causes**:
1. **Interpolation uses wrong dimension**: querying along maturity instead of moneyness
2. **Memory layout mismatch**: stride calculation still wrong despite fix
3. **Extrapolation instead of interpolation**: validation samples outside grid bounds
4. **Units mismatch**: prices stored in cents but interpolated as dollars (factor 100 error)
5. **NaN contamination**: some prices are NaN, cubic splines propagate to entire domain

## **UPDATE: ROOT CAUSE IDENTIFIED** ✅

Following ChatGPT's diagnostic strategy, we systematically eliminated suspects:

1. ✅ **Interpolation context bugs** (PR #66):
   - `price_table_create_ex()` never created interpolation context
   - Grid expansion didn't recreate context with new dimensions
   - **Fixed**: Both bugs corrected
   - **Verified**: Diagnostic test shows interpolation works perfectly (error ~1e-4)

2. ✅ **3-Point FDM "Ground Truth"** - **THE ACTUAL ROOT CAUSE**:
   - **Location**: `src/validation.c:199-206`
   - **Bug**: Validation solves FDM on only **3 spatial points** for "ground truth"
   - **Impact**: 3-point FDM produces ~11% IV error regardless of interpolation quality
   - **Why constant error**: Ground truth is always wrong by same systematic amount
   - **Why refinement fails**: Comparing good interpolation against garbage reference

**Evidence**:
```c
// src/validation.c:199-206
double m_grid[3] = {s->moneyness * 0.95, s->moneyness, s->moneyness * 1.05};
AmericanOptionResult fdm_result = american_option_solve(
    &option_data, m_grid, 3, ...);  // <--- ONLY 3 POINTS!

// src/american_option.c:386-392
SpatialGrid grid = {
    .n_points = n_m,  // This is 3!!!
    ...
};
```

**Comparison**:
- Table precomputation: 101 spatial points → accurate FDM
- Validation "ground truth": **3 spatial points** → 1134bp error

See `docs/validation-3point-bug.md` for detailed analysis and fix options.

**Possible explanations (OBSOLETE - kept for historical record)**:

### Hypothesis 1: Initial Grid Too Coarse
- 10 points over $[0.7, 1.3]$ may be insufficient for cubic interpolation
- With such large spacing, interpolation may be extrapolating between widely separated points
- Adding midpoints may not sufficiently densify the grid

### Hypothesis 2: Interpolation Workspace Not Updated
- Cubic spline coefficients are precomputed and stored in `interp_context`
- After grid expansion, if `interp_context` is not **reallocated/rebuilt**, interpolation may use:
  - Stale coefficients computed for old grid size
  - Wrong array dimensions
  - Out-of-bounds memory access

### Hypothesis 3: Validation Samples Outside Grid Bounds
- If validation samples fall outside $[m_1, m_{N_m}]$, interpolation returns NaN
- These get filtered out but don't contribute to error statistics
- Effective sample size may be tiny (explains identical errors across iterations)

### Hypothesis 4: Coordinate Transform Mismatch
- Moneyness grid uses `COORD_RAW` (no transform)
- FDM solver uses log-space internally
- Potential mismatch in forward/inverse transformations during:
  - Price computation (`grid_point_to_option`)
  - Interpolation queries (`price_table_interpolate_4d`)

### Hypothesis 5: Test Setup Issue
- "Before" validation (line 64-69) measures error on **empty table** (all NaN)
- This explains `p95_before = 0 bp` in test output
- Test expectation `p95_after < p95_before` is always false

## 6. Questions for Expert Review

1. **Algorithm design**: Is the geometric mean $\sqrt{m_j \cdot m_{j+1}}$ the correct refinement strategy for log-spaced moneyness grids? Should we use error-weighted point selection instead of simple midpoints?

2. **Interpolation theory**: With cubic splines on a very coarse grid (10 points), what is the expected worst-case interpolation error? Are we below the theoretical limit?

3. **Convergence theory**: For this PDE-based pricing + interpolation problem, what convergence rate should we expect as we refine the grid? Is $O(h^4)$ achievable or limited by PDE discretization error?

4. **Error propagation**: We measure error in IV space after two inversions (price → IV). How much error amplification occurs through the nonlinear IV inversion? Could this explain the large errors?

5. **Validation methodology**: Is stratified random sampling over grid cells the right approach? Should we use quasi-random sequences (Sobol/Halton) or importance sampling weighted by option trading activity?

6. **Debugging strategy**: Given that errors are **exactly constant** across iterations, what diagnostic tests would you recommend to isolate the failure mode?

## 7. Reproducibility

**Test command**:
```bash
bazel test //tests:adaptive_accuracy_test \
  --test_filter="AdaptiveAccuracyTest.AccuracyImprovement" \
  --test_output=all
```

**Expected behavior**: P95 error should decrease from ~1135bp → ~50bp → ~5bp
**Actual behavior**: P95 error constant at ~1135bp

**Code references**:
- Adaptive refinement: `src/validation.c:457-576`
- Grid expansion: `src/price_table.c:1168-1332`
- Validation: `src/validation.c:74-341`
- Test: `tests/adaptive_accuracy_test.cc:41-95`
