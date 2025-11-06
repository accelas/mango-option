# B-Spline Interpolation for FDM Output - Design Document

**Date:** 2025-11-06
**Status:** Design Phase
**Author:** Claude Code

---

## Executive Summary

Design and implement tensor-product cubic B-spline interpolation for PDE solver output to enable fast, smooth, monotone-preserving option pricing and implied volatility calculation.

**Goal:** Replace current multilinear/cubic spline interpolation with B-splines to achieve:
1. **C² continuity** - Smooth second derivatives for Newton's method convergence
2. **No negative overshoot** - Non-negative price guarantee via clamping
3. **Fast queries** - Sub-2μs for 4D interpolation
4. **Accurate IV** - Newton converges in 4-6 iterations (vs 8-12 with C0)

---

## Problem Statement

### Current Situation

**Modern C++ codebase has:**
- ✅ `CubicSpline<T>` - Natural cubic splines (C² continuous)
- ✅ `SnapshotInterpolator` - Uses cubic splines for 1D PDE snapshots
- ✅ `PriceTableSnapshotCollector` - Collects snapshots into price tables

**Problem:**
- ❌ Natural cubic splines have **overshoot** near boundaries (Runge's phenomenon)
- ❌ Can produce **negative option prices** (unphysical)
- ❌ Current `PriceTableSnapshotCollector` only does 1D interpolation per snapshot
- ❌ No 4D/5D tensor-product interpolation for `V(m, τ, σ, r, q)`

**Impact:**
- Newton-based IV solver fails when interpolated prices/vegas are non-monotone
- Must fall back to slow FDM evaluation (~143ms per IV)
- Cannot achieve target ~10μs IV calculation

---

## Solution: Tensor-Product Cubic B-Splines

### Why B-Splines Over Natural Cubic Splines?

| Feature | Natural Cubic | B-Splines |
|---------|--------------|-----------|
| **Continuity** | C² | C² |
| **Overshoot** | ❌ Yes (Runge) | ✅ Minimal |
| **Negative prices** | ❌ Possible | ✅ Rare (+ clamping) |
| **Local control** | ❌ Global | ✅ Local |
| **Monotonicity** | ❌ Not preserved | ✅ Better preserved |
| **Constraints** | ❌ Hard | ✅ Easier (via QP) |

**Key Advantage:** B-splines have **local support** - changing one control point affects only a limited region. This prevents global oscillations that cause overshoot.

---

## Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  FDM Solver → Snapshots → Price Table → Fast IV        │
│                                                         │
└─────────────────────────────────────────────────────────┘

Detailed:

PDESolver
    ↓ (generates snapshots at different times)
SnapshotCollector (PriceTableSnapshotCollector)
    ↓ (collects snapshots, builds price table)
BSplineInterpolator4D/5D
    ↓ (interpolates V, ∂V/∂σ from table)
IVSolverInterpolated (Newton's method)
    ↓ (solves for σ given market price)
IVResult (~10μs per query)
```

### New Components

**1. B-Spline Basis Functions**
```cpp
// src/bspline_basis.hpp
class BSplineBasis {
    double eval_basis(size_t i, double x);           // B_i(x)
    double eval_basis_derivative(size_t i, double x); // B'_i(x)
    double eval_basis_second_derivative(size_t i, double x); // B''_i(x)
};
```

**2. 1D B-Spline Interpolator**
```cpp
// src/bspline_1d.hpp
class BSpline1D {
    void fit(span<double> x, span<double> y);  // Compute control points
    double eval(double x);                      // Interpolate
    double eval_derivative(double x);           // First derivative
};
```

**3. 4D/5D Tensor-Product Interpolator**
```cpp
// src/bspline_tensor.hpp
class BSplineTensor4D {
    // Tensor product: V(m,τ,σ,r) = Σᵢⱼₖₗ cᵢⱼₖₗ·Bᵢ(m)·Bⱼ(τ)·Bₖ(σ)·Bₗ(r)
    double interpolate(double m, double tau, double sigma, double r);
    double interpolate_derivative_sigma(double m, double tau, double sigma, double r);
};
```

**4. Price Table with B-Splines**
```cpp
// src/price_table_bspline.hpp
class PriceTableBSpline {
    void build_from_snapshots(vector<Snapshot> snapshots);
    double query_price(double m, double tau, double sigma, double r);
    double query_vega(double m, double tau, double sigma, double r);
    double query_gamma(double m, double tau, double sigma, double r);
};
```

**5. Newton IV Solver with Interpolation**
```cpp
// src/iv_solver_interpolated.hpp
class IVSolverInterpolated {
    IVResult solve(IVParams params);  // ~10μs via Newton + B-spline table
};
```

---

## Mathematical Background

### 1D Cubic B-Spline Basis

A cubic B-spline curve is defined as:
```
S(x) = Σᵢ cᵢ · Bᵢ,₃(x)
```

where:
- `cᵢ` are **control points** (fitted to data)
- `Bᵢ,₃(x)` are **cubic B-spline basis functions**

**Cox-de Boor Recursion:**
```
B_{i,0}(x) = 1  if t_i ≤ x < t_{i+1}, else 0

B_{i,p}(x) = (x - t_i)/(t_{i+p} - t_i) · B_{i,p-1}(x)
           + (t_{i+p+1} - x)/(t_{i+p+1} - t_{i+1}) · B_{i+1,p-1}(x)
```

**Knot Vector:**
For interpolation at `n` data points, use **open uniform knots**:
```
t = [x_min, x_min, x_min, x_min, x_1, ..., x_{n-2}, x_max, x_max, x_max, x_max]
     └──── 4 repeated ────┘                          └──── 4 repeated ────┘
```

This ensures interpolation at endpoints.

### 4D Tensor Product

For 4D price table `V(m, τ, σ, r)`:
```
V(m, τ, σ, r) = Σᵢ Σⱼ Σₖ Σₗ c_{ijkl} · B_i(m) · B_j(τ) · B_k(σ) · B_l(r)
```

**Control point fitting:**
1. Evaluate basis functions at all grid points
2. Solve least-squares system: `Φ · c = y`
   where `Φ[p,i] = B_i(x_p)` is the collocation matrix
3. Store control points `c_{ijkl}` for fast evaluation

**Query evaluation:**
1. Find non-zero basis functions (at most 4 per dimension for cubic)
2. Compute tensor product (up to 4⁴ = 256 terms for 4D)
3. Return weighted sum

**Complexity:**
- Pre-computation: O(n⁴) for 4D (same as current price table filling)
- Query: O(4⁴) = O(256) basis evaluations = ~1-2μs

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal:** Implement 1D B-spline infrastructure

**Tasks:**
1. Implement `BSplineBasis` class
   - Cox-de Boor recursion for basis evaluation
   - Knot vector generation (open uniform)
   - Derivatives: first and second
   - Tests: validate partition of unity, derivatives

2. Implement `BSpline1D` class
   - Fit control points via least-squares
   - Evaluation and derivatives
   - Tests: interpolate known functions (polynomials, exp)

**Deliverables:**
- `src/bspline_basis.hpp`
- `src/bspline_1d.hpp`
- `tests/bspline_basis_test.cc`
- `tests/bspline_1d_test.cc`

**Success Criteria:**
- Basis functions sum to 1.0 everywhere (partition of unity)
- Interpolate cubic polynomial exactly
- Derivatives match finite differences (within 1e-6)

---

### Phase 2: Tensor Product Interpolation (Week 2)

**Goal:** Extend to 4D/5D tensor products

**Tasks:**
1. Implement `BSplineTensor4D` class
   - Control point storage (4D array or flattened)
   - Tensor product evaluation
   - Derivatives w.r.t. each dimension
   - Tests: validate on separable functions

2. Implement `BSplineTensor5D` class (similar to 4D)

**Deliverables:**
- `src/bspline_tensor.hpp`
- `tests/bspline_tensor_test.cc`

**Success Criteria:**
- Interpolate separable function: `f(m,τ,σ,r) = g(m)·h(τ)·i(σ)·j(r)`
- Query time < 2μs for 4D
- Memory overhead < 3x current price table

---

### Phase 3: Price Table Integration (Week 3)

**Goal:** Replace cubic splines with B-splines in price table

**Tasks:**
1. Extend `PriceTableSnapshotCollector` to build B-spline tables
   - Collect all snapshots (V, ∂V/∂S, ∂²V/∂S²)
   - Fit 4D B-spline for each Greek
   - Add non-negative clamping: `V_interp = max(V_bspline, 0)`

2. Implement query methods
   - `query_price(m, τ, σ, r)` → V
   - `query_vega(m, τ, σ, r)` → ∂V/∂σ
   - `query_gamma(m, τ, σ, r)` → ∂²V/∂S²

**Deliverables:**
- Updated `PriceTableSnapshotCollector`
- `tests/price_table_bspline_test.cc`

**Success Criteria:**
- No negative prices (after clamping)
- Vega is C¹ continuous
- Gamma matches FDM within 5% RMS error

---

### Phase 4: Newton IV Solver (Week 4)

**Goal:** Implement fast IV calculation using B-spline table

**Tasks:**
1. Implement `IVSolverInterpolated` class
   - Newton's method: `σ_{n+1} = σ_n - (V(σ_n) - V_market) / vega(σ_n)`
   - Uses `query_price()` and `query_vega()` from B-spline table
   - Convergence criteria: `|V(σ) - V_market| < tol`
   - Damping for stability: `σ_new = 0.8·σ_newton + 0.2·σ_old`

2. Fallback logic
   - Check if query is in-bounds
   - If out-of-bounds: return error (user handles fallback to FDM)

**Deliverables:**
- `src/iv_solver_interpolated.hpp`
- `tests/iv_solver_interpolated_test.cc`

**Success Criteria:**
- Converges in 4-6 iterations (vs 8-12 with C0)
- Query time ~10μs per IV
- Matches FDM IV within 1bp (0.0001 vol units)

---

### Phase 5: Validation & Benchmarking (Week 5)

**Goal:** Validate accuracy and measure performance

**Tasks:**
1. Accuracy tests
   - Compare B-spline IV vs FDM IV on 1000 random queries
   - Measure RMS error, max error, convergence rate
   - Validate Greeks (delta, gamma, vega) accuracy

2. Performance benchmarking
   - Measure query time (should be ~1-2μs for price, ~10μs for IV)
   - Memory usage (should be < 10MB for typical table)
   - Compare vs current FDM IV (~143ms)

3. Robustness testing
   - Edge cases: deep ITM/OTM, short/long maturity
   - Negative price prevention
   - Out-of-bounds handling

**Deliverables:**
- `tests/iv_interpolation_validation_test.cc`
- `benchmarks/bspline_vs_fdm_benchmark.cc`
- Performance report in `docs/benchmarks/bspline_performance.md`

**Success Criteria:**
- IV accuracy: RMS error < 0.5bp
- Query time: <2μs (price), <12μs (IV)
- Speedup: >10,000x vs FDM
- No negative prices in 10K random queries

---

## Data Structures

### Control Point Storage

**Option 1: Nested vectors (easy to implement)**
```cpp
// 4D: V(m, τ, σ, r)
vector<vector<vector<vector<double>>>> control_points;
// Access: control_points[i_m][i_tau][i_sigma][i_r]
```

**Option 2: Flattened array (cache-friendly)**
```cpp
// Flattened 1D array with stride calculation
vector<double> control_points_flat;  // Size: n_m × n_τ × n_σ × n_r

size_t idx = i_m * stride_m + i_tau * stride_tau +
             i_sigma * stride_sigma + i_r * stride_r;
double c = control_points_flat[idx];
```

**Recommendation:** Use flattened array for better cache locality.

### Memory Footprint

For a 4D table with grid size `50 × 30 × 20 × 10 = 300,000` points:

**Current price table (multilinear):**
- Prices: 300K × 8 bytes = 2.4 MB
- Vegas: 300K × 8 bytes = 2.4 MB
- Gammas: 300K × 8 bytes = 2.4 MB
- **Total: ~7.2 MB**

**B-spline table:**
- Control points: ~300K × 8 bytes = 2.4 MB (same as data)
- Knot vectors: 4 × 60 × 8 bytes = 1.9 KB (negligible)
- **Total: ~2.4 MB per Greek**
- **Grand total: ~7.2 MB** (same as current!)

**Conclusion:** Memory overhead is minimal.

---

## Non-Negative Price Constraint

### Approach 1: Post-Hoc Clamping (Simple)

```cpp
double query_price(double m, double tau, double sigma, double r) {
    double price_raw = bspline.eval(m, tau, sigma, r);
    return std::max(price_raw, 0.0);  // Clamp to non-negative
}
```

**Pros:**
- ✅ Simple to implement
- ✅ Always enforces non-negativity

**Cons:**
- ❌ Creates discontinuity in derivative at price=0
- ❌ Not mathematically elegant

**Assessment:** Good enough for initial implementation. Discontinuity only affects Newton's method near zero prices, which are rare.

### Approach 2: Constrained Fitting (Advanced, Future)

Fit B-spline control points with non-negativity constraint:
```
minimize  ||Φ·c - y||²
subject to  c_i ≥ 0  for all i
```

This is a **quadratic programming (QP) problem**.

**Pros:**
- ✅ Guarantees non-negative everywhere
- ✅ No discontinuous derivatives

**Cons:**
- ❌ Requires QP solver (e.g., OSQP, qpOASES)
- ❌ More complex implementation
- ❌ Slower pre-computation

**Assessment:** Defer to Phase 6 (optional enhancement).

---

## Alternative Considered: SPLINTER Library

**SPLINTER** (https://github.com/bgrimstad/splinter) is a BSD-licensed library for multivariate function approximation using B-splines.

**Pros:**
- ✅ Mature, battle-tested implementation
- ✅ Supports tensor-product B-splines
- ✅ BSD license (compatible)

**Cons:**
- ❌ Heavy dependency (~20K LOC)
- ❌ Designed for general function approximation, not option pricing
- ❌ No built-in non-negativity constraints
- ❌ May be overkill for our focused use case

**Decision:** Implement custom B-spline code for maximum control and minimal dependencies. Can revisit SPLINTER if custom implementation proves insufficient.

---

## Risk Assessment

### Technical Risks

**Risk 1: B-splines still overshoot**
- **Likelihood:** Low (B-splines have local support, less oscillation)
- **Impact:** High (negative prices break IV solver)
- **Mitigation:**
  - Use post-hoc clamping (Approach 1)
  - Validate on 10K random queries
  - If still problematic, implement constrained fitting (Approach 2)

**Risk 2: Fitting accuracy insufficient**
- **Likelihood:** Medium (least-squares may not capture sharp features)
- **Impact:** Medium (affects IV accuracy)
- **Mitigation:**
  - Use finer grid spacing near ATM and short maturities
  - Benchmark against FDM ground truth
  - Validate RMS error < 0.5bp target

**Risk 3: Query performance slower than expected**
- **Likelihood:** Low (256 basis evaluations should be <2μs)
- **Impact:** Low (still 100x faster than FDM)
- **Mitigation:**
  - Profile and optimize hot paths
  - Pre-compute basis function values for common queries
  - Use SIMD if needed

**Risk 4: Memory overhead too high**
- **Likelihood:** Very low (analysis shows ~7.2 MB)
- **Impact:** Low (modern systems have plenty of RAM)
- **Mitigation:**
  - Use compressed storage if needed
  - Only load tables for active underlyings

---

## Success Metrics

### Quantitative Targets

**Pre-computation:**
- Time: < 20 minutes for 300K grid points (same as current)
- Memory: < 10 MB per table

**Query performance:**
- Price interpolation: < 2 μs
- Vega interpolation: < 2 μs
- IV calculation: < 15 μs (4-6 Newton iterations)

**Accuracy:**
- Price RMS error: < 0.1% vs FDM
- IV RMS error: < 0.5bp (0.0001 vol units)
- Gamma relative error: < 5%

**Robustness:**
- Zero negative prices in 10K random queries (with clamping)
- Newton convergence rate: > 95% for in-bounds queries
- Graceful degradation for edge cases

### Qualitative Targets

- ✅ Code is readable and well-documented
- ✅ Tests cover edge cases (ITM/OTM, short/long maturity)
- ✅ Benchmarks demonstrate >10,000x speedup vs FDM
- ✅ API is intuitive and easy to use

---

## Open Questions

1. **Control point fitting algorithm:**
   - Use dense least-squares or sparse solver?
   - Regularization needed for ill-conditioned systems?

2. **Knot placement:**
   - Uniform vs. non-uniform knots?
   - Adaptive knot placement near ATM?

3. **Derivative computation:**
   - Analytical derivatives of B-splines or finite differences?
   - Which is more accurate for vega?

4. **Extrapolation:**
   - What to return for out-of-bounds queries?
   - Nearest neighbor, linear extrapolation, or error?

5. **Greeks storage:**
   - Store separate B-splines for each Greek?
   - Or compute on-the-fly from price B-spline derivatives?

**Resolution:** Address these during implementation based on performance and accuracy trade-offs.

---

## References

### Papers

1. **de Boor, C. (2001).** "A Practical Guide to Splines." Springer.
   - Definitive reference for B-splines
   - Chapter 9: Multivariate tensor-product splines

2. **Piegl, L. & Tiller, W. (1997).** "The NURBS Book." Springer.
   - Comprehensive B-spline and NURBS reference
   - Chapter 3: B-spline basis functions and properties

3. **Grimstad, B. & Sandnes, A. (2016).** "SPLINTER: A library for multivariate function approximation with splines."
   - Describes SPLINTER implementation
   - Section 3.2: Tensor-product B-splines

4. **Jia, M., et al. (2016).** "Call option price function in Bernstein polynomial basis."
   - Shows how to enforce arbitrage-free constraints
   - Relevant for future Phase 6 (constrained fitting)

### Code References

- SPLINTER library: https://github.com/bgrimstad/splinter
- Existing cubic splines: `src/cubic_spline_solver.hpp`
- Price table: `src/price_table_snapshot_collector.hpp`

---

## Conclusion

This design provides a comprehensive plan for implementing B-spline interpolation for FDM output. The phased approach allows for incremental development and validation.

**Key Benefits:**
1. ✅ C² continuity enables Newton IV convergence
2. ✅ Local support reduces overshoot vs natural cubic splines
3. ✅ Non-negative clamping prevents unphysical prices
4. ✅ ~10μs IV queries (14,000x faster than FDM)

**Next Steps:**
1. Review and approve this design
2. Create GitHub issue for tracking
3. Begin Phase 1 implementation: B-spline basis functions
