# B-Spline FDM Interpolation - Critical Design Fixes (ADDENDUM)

**Date:** 2025-11-06
**Status:** Design Revision - Addressing Critical Flaws

---

## Critical Issues Identified

Three fundamental flaws in the original design make it **unimplementable**:

1. ❌ **Dense least-squares fitting is O(n³) and infeasible** (terabytes of memory for 300k points)
2. ❌ **Post-hoc clamping breaks derivative consistency** (Newton diverges with inconsistent gradients)
3. ❌ **No data collection strategy for 4D/5D grids** (current collector only produces 2D)

This addendum **fixes these critical issues** with implementable solutions.

---

## Fix 1: Separable Tensor-Product Fitting (de Boor's Approach)

### Original (WRONG) Approach

```cpp
// Build dense collocation matrix Φ
// For 4D grid with n_m × n_τ × n_σ × n_r = 50×30×20×10 = 300k points:
// Φ is 300k × 300k DENSE matrix
// Solve: Φ · c = y
// Memory: 300k × 300k × 8 bytes = 720 GB (IMPOSSIBLE!)
// Time: O(n³) = O(300k³) ≈ years
```

**This is completely infeasible.**

### Correct Approach: Sequential 1D Fitting

Exploit the **separable structure** of tensor-product B-splines:

```
V(m,τ,σ,r) = Σᵢⱼₖₗ cᵢⱼₖₗ · Bᵢ(m)·Bⱼ(τ)·Bₖ(σ)·Bₗ(r)
```

**Key insight:** We can fit each dimension **independently** in sequence.

#### Algorithm: Alternating 1D Fits

**Step 1: Fit along m-axis (fix τ, σ, r)**

For each combination (j, k, l):
```
V_data[i,j,k,l] ≈ Σᵢ c¹ᵢⱼₖₗ · Bᵢ(mᵢ)

Solve 1D system (n_m × n_m BANDED):
Φ_m · c¹[:,j,k,l] = V_data[:,j,k,l]
```

**Step 2: Fit along τ-axis (using c¹ from step 1)**

For each (i, k, l):
```
c¹[i,j,k,l] ≈ Σⱼ c²ᵢⱼₖₗ · Bⱼ(τⱼ)

Solve 1D system (n_τ × n_τ BANDED):
Φ_τ · c²[i,:,k,l] = c¹[i,:,k,l]
```

**Step 3: Fit along σ-axis**

For each (i, j, l):
```
Φ_σ · c³[i,j,:,l] = c²[i,j,:,l]
```

**Step 4: Fit along r-axis**

For each (i, j, k):
```
Φ_r · c⁴[i,j,k,:] = c³[i,j,k,:]
```

**Final control points:** `c = c⁴`

#### Complexity Analysis

**Memory:**
- Largest matrix: 50×50 (for m-axis) = 2,500 doubles = 20 KB
- Total temp storage: O(n_max²) where n_max = max(n_m, n_τ, n_σ, n_r) = 50
- **Memory: ~20 KB (vs 720 GB for dense approach!)**

**Time:**
- Each 1D solve: O(n_dim) for banded tridiagonal (Thomas algorithm)
- Number of 1D solves: n_m·n_τ·n_σ·n_r / n_dim per dimension
- Total: O(n_total) where n_total = n_m·n_τ·n_σ·n_r = 300k
- **Time: ~1 second (vs years for dense approach!)**

#### Implementation: Separable Fitting Class

```cpp
/// Tensor-product B-spline fitter using separable 1D solves
class BSplineTensorFitter4D {
public:
    /// Fit 4D data with separable B-splines
    ///
    /// @param grid_m Moneyness grid points
    /// @param grid_tau Maturity grid points
    /// @param grid_sigma Volatility grid points
    /// @param grid_r Rate grid points
    /// @param data 4D array of values (flattened: [i_m][i_tau][i_sigma][i_r])
    /// @return Control points (same dimensions as data)
    std::vector<double> fit(
        std::span<const double> grid_m,
        std::span<const double> grid_tau,
        std::span<const double> grid_sigma,
        std::span<const double> grid_r,
        std::span<const double> data)
    {
        const size_t n_m = grid_m.size();
        const size_t n_tau = grid_tau.size();
        const size_t n_sigma = grid_sigma.size();
        const size_t n_r = grid_r.size();

        // Allocate control point storage
        std::vector<double> c_current(data.begin(), data.end());
        std::vector<double> c_next(c_current.size());

        // Step 1: Fit along m-axis
        BSplineBasis basis_m(n_m, grid_m.front(), grid_m.back());
        for (size_t j = 0; j < n_tau; ++j) {
            for (size_t k = 0; k < n_sigma; ++k) {
                for (size_t l = 0; l < n_r; ++l) {
                    // Extract 1D slice along m
                    std::vector<double> slice_data(n_m);
                    for (size_t i = 0; i < n_m; ++i) {
                        size_t idx = i*n_tau*n_sigma*n_r + j*n_sigma*n_r + k*n_r + l;
                        slice_data[i] = c_current[idx];
                    }

                    // Fit 1D B-spline
                    auto control_1d = fit_1d_bspline(basis_m, grid_m, slice_data);

                    // Store back
                    for (size_t i = 0; i < n_m; ++i) {
                        size_t idx = i*n_tau*n_sigma*n_r + j*n_sigma*n_r + k*n_r + l;
                        c_next[idx] = control_1d[i];
                    }
                }
            }
        }
        c_current = c_next;

        // Step 2: Fit along τ-axis
        BSplineBasis basis_tau(n_tau, grid_tau.front(), grid_tau.back());
        // ... similar loop over (i, k, l) ...

        // Step 3: Fit along σ-axis
        BSplineBasis basis_sigma(n_sigma, grid_sigma.front(), grid_sigma.back());
        // ... similar loop over (i, j, l) ...

        // Step 4: Fit along r-axis
        BSplineBasis basis_r(n_r, grid_r.front(), grid_r.back());
        // ... similar loop over (i, j, k) ...

        return c_current;  // Final control points
    }

private:
    /// Fit 1D B-spline using banded least-squares
    std::vector<double> fit_1d_bspline(
        const BSplineBasis& basis,
        std::span<const double> x,
        std::span<const double> y)
    {
        const size_t n = x.size();

        // Build collocation matrix Φ (banded, typically tridiagonal)
        std::vector<double> Phi_diag(n);
        std::vector<double> Phi_upper(n-1);
        std::vector<double> Phi_lower(n-1);

        for (size_t i = 0; i < n; ++i) {
            Phi_diag[i] = basis.eval_basis(i, x[i]);
            if (i < n-1) {
                Phi_upper[i] = basis.eval_basis(i, x[i+1]);
                Phi_lower[i] = basis.eval_basis(i+1, x[i]);
            }
        }

        // Solve tridiagonal system: Φ · c = y
        // (Use Thomas algorithm - already have ThomasSolver in codebase!)
        std::vector<double> control_points(n);
        ThomasWorkspace<double> workspace(n);
        solve_thomas(
            std::span{Phi_lower},
            std::span{Phi_diag},
            std::span{Phi_upper},
            std::span{y},
            std::span{control_points},
            workspace.get()
        );

        return control_points;
    }
};
```

**References:**
- de Boor (2001), Chapter 11: "Approximation by splines"
- Dierckx (1993), "Curve and Surface Fitting with Splines" - describes separable fitting

---

## Fix 2: Derivative-Consistent Clamping

### The Problem with Naive Clamping

```cpp
// WRONG: Inconsistent derivatives
double price = max(V_bspline(σ), 0);      // Clamp price
double vega = ∂V_bspline/∂σ;              // But derivative unchanged!

// Newton's method: σ_new = σ - (price - price_market) / vega
// Problem: vega doesn't match the clamped price surface!
// Result: Newton diverges or converges to wrong value
```

**Mathematical inconsistency:**
If `V = max(V_spline, 0)`, then `∂V/∂σ` should reflect the clamping:
```
∂V/∂σ = { ∂V_spline/∂σ   if V_spline > 0  (free region)
        { 0               if V_spline ≤ 0  (clamped region)
```

### Solution 1: Compute Derivatives from Clamped Surface (Simple)

**Don't store separate vega B-spline.** Instead, compute vega via finite differences on the clamped price:

```cpp
class BSplinePriceTable {
public:
    double query_price(double m, double tau, double sigma, double r) {
        double V_raw = bspline_.eval(m, tau, sigma, r);
        return std::max(V_raw, 0.0);  // Clamp
    }

    double query_vega(double m, double tau, double sigma, double r) {
        // Centered finite difference on CLAMPED surface
        const double h = 1e-5;
        double V_plus = query_price(m, tau, sigma + h, r);   // Uses clamped price!
        double V_minus = query_price(m, tau, sigma - h, r);  // Uses clamped price!
        return (V_plus - V_minus) / (2*h);
    }

    double query_gamma(double m, double tau, double sigma, double r) {
        // Second difference on CLAMPED surface
        const double h = 1e-4;
        double V_center = query_price(m, tau, sigma, r);
        double V_plus = query_price(m, tau, sigma + h, r);
        double V_minus = query_price(m, tau, sigma - h, r);
        return (V_plus - 2*V_center + V_minus) / (h*h);
    }
};
```

**Pros:**
- ✅ Mathematically consistent (derivatives match clamped surface)
- ✅ Simple implementation
- ✅ Guarantees Newton sees consistent gradients

**Cons:**
- ❌ Slower (3 B-spline evals per vega query instead of 1)
- ❌ Finite-difference error (~1e-5 for double precision with h=1e-5)

**Performance impact:**
- Price query: 1 B-spline eval (~1μs)
- Vega query: 3 B-spline evals (~3μs)
- IV solve: 5 iterations × 2 queries (price + vega) × 3 evals = ~30μs
- **Still 4,800x faster than FDM (143ms)!**

### Solution 2: Smooth Penalty Function (Advanced)

Instead of hard clamping, use a **smooth penalty** that keeps derivatives continuous:

```cpp
double smooth_clamp(double V, double epsilon = 1e-6) {
    if (V > epsilon) {
        return V;  // Normal region
    } else {
        // Smooth quadratic transition near zero
        // V_smooth(V) = ε/2 + V²/(2ε) for V ∈ [0, ε]
        return epsilon/2 + V*V/(2*epsilon);
    }
}
```

**Derivative:**
```
∂V_smooth/∂σ = { ∂V/∂σ           if V > ε
               { (V/ε)·∂V/∂σ     if V ∈ [0,ε]  (smooth transition)
```

**Pros:**
- ✅ C¹ continuous (smooth derivatives)
- ✅ Can use analytical B-spline derivatives (fast)
- ✅ No discontinuity for Newton's method

**Cons:**
- ❌ Allows slightly negative prices (down to -ε²/(2ε) = -ε/2)
- ❌ More complex implementation

**Assessment:** Defer to future optimization. Use Solution 1 (finite differences) initially.

---

## Fix 3: 4D/5D Data Collection Strategy

### Current Reality: Only 2D Tables

**What we have:**
```cpp
// Run PDE at FIXED σ=0.20, r=0.05
AmericanOptionSolver solver(/*σ=0.20, r=0.05*/);
solver.solve();

// Collect snapshots at different times
PriceTableSnapshotCollector collector(/*m_grid, tau_grid*/);
for (auto& snapshot : solver.snapshots()) {
    collector.collect(snapshot);  // Builds V(m, τ) for FIXED σ, r
}

// Result: 2D table V(m, τ) for σ=0.20, r=0.05
```

**This gives us ONE 2D slice** of the desired 4D table `V(m, τ, σ, r)`.

### What We Need: 4D Grid from Multiple PDE Runs

To build 4D B-spline table `V(m, τ, σ, r)`, we need data on a **Cartesian grid** across all dimensions:

```
Grid dimensions:
- Moneyness: m ∈ {0.7, 0.75, ..., 1.3}        (n_m = 50)
- Maturity: τ ∈ {0.027, 0.1, ..., 2.0}        (n_τ = 30)
- Volatility: σ ∈ {0.10, 0.15, ..., 0.60}     (n_σ = 20)
- Rate: r ∈ {0.0, 0.02, ..., 0.10}            (n_r = 10)

Total grid points: 50 × 30 × 20 × 10 = 300,000
```

**Data collection strategy:**

1. **Run PDE solver for each (σ, r) combination:**
   ```
   for σ in [0.10, 0.15, 0.20, ..., 0.60]:      # 20 values
       for r in [0.0, 0.02, 0.05, ..., 0.10]:    # 10 values
           # Run PDE with these parameters
           solver = AmericanOptionSolver(σ, r)
           solver.solve()

           # Collect snapshots → V(m, τ) for this (σ, r)
           collector.add_slice(σ, r, solver.snapshots())
   ```

2. **Total PDE runs:** 20 × 10 = **200 runs**

3. **Computational cost:**
   - Single PDE run: ~20ms (existing AmericanOptionSolver)
   - Total pre-computation: 200 × 20ms = **4 seconds**
   - **Acceptable for overnight batch!**

### Revised Data Collection API

```cpp
/// 4D Price table builder from multiple PDE runs
class PriceTable4DBuilder {
public:
    /// Constructor: define 4D grid
    PriceTable4DBuilder(
        std::span<const double> moneyness_grid,
        std::span<const double> maturity_grid,
        std::span<const double> volatility_grid,
        std::span<const double> rate_grid)
        : m_grid_(moneyness_grid.begin(), moneyness_grid.end())
        , tau_grid_(maturity_grid.begin(), maturity_grid.end())
        , sigma_grid_(volatility_grid.begin(), volatility_grid.end())
        , r_grid_(rate_grid.begin(), rate_grid.end())
    {
        // Allocate 4D storage
        const size_t n_total = m_grid_.size() * tau_grid_.size() *
                              sigma_grid_.size() * r_grid_.size();
        prices_.resize(n_total, 0.0);
    }

    /// Add PDE solution for specific (σ, r) slice
    void add_pde_slice(double sigma, double rate,
                      const std::vector<Snapshot>& snapshots)
    {
        // Find indices for this (σ, r)
        size_t i_sigma = find_index(sigma_grid_, sigma);
        size_t i_r = find_index(r_grid_, rate);

        // Process each snapshot (different τ)
        for (const auto& snap : snapshots) {
            size_t i_tau = snap.user_index;  // Tau index from snapshot

            // Interpolate snapshot to moneyness grid
            SnapshotInterpolator interp;
            interp.build(snap.spatial_grid, snap.solution);

            for (size_t i_m = 0; i_m < m_grid_.size(); ++i_m) {
                double m = m_grid_[i_m];
                double S = m * K_ref_;  // Convert to spot price
                double V = interp.eval(S);

                // Store in 4D array
                size_t idx = flatten_4d(i_m, i_tau, i_sigma, i_r);
                prices_[idx] = V;
            }
        }
    }

    /// Build B-spline interpolator from collected data
    BSplineTensor4D build_interpolator() {
        // Fit separable B-splines using BSplineTensorFitter4D
        BSplineTensorFitter4D fitter;
        auto control_points = fitter.fit(
            m_grid_, tau_grid_, sigma_grid_, r_grid_,
            prices_
        );

        return BSplineTensor4D(
            m_grid_, tau_grid_, sigma_grid_, r_grid_,
            control_points
        );
    }

private:
    std::vector<double> m_grid_, tau_grid_, sigma_grid_, r_grid_;
    std::vector<double> prices_;  // 4D array (flattened)
    double K_ref_ = 100.0;

    size_t flatten_4d(size_t i_m, size_t i_tau, size_t i_sigma, size_t i_r) {
        return i_m * (tau_grid_.size() * sigma_grid_.size() * r_grid_.size())
             + i_tau * (sigma_grid_.size() * r_grid_.size())
             + i_sigma * r_grid_.size()
             + i_r;
    }
};
```

### Usage Example

```cpp
// Pre-computation (overnight batch)
PriceTable4DBuilder builder(
    moneyness_grid,  // 50 points
    maturity_grid,   // 30 points
    volatility_grid, // 20 points
    rate_grid        // 10 points
);

// Run PDE for each (σ, r) combination
for (double sigma : volatility_grid) {
    for (double rate : rate_grid) {
        // Configure PDE solver
        AmericanOptionParams params{
            .strike = 100.0,
            .volatility = sigma,
            .risk_free_rate = rate,
            .time_to_maturity = 2.0,
            .option_type = OptionType::PUT
        };

        // Solve PDE
        AmericanOptionSolver solver(params, pde_grid);
        solver.solve();

        // Add this (σ,r) slice to 4D table
        builder.add_pde_slice(sigma, rate, solver.get_snapshots());
    }
}

// Build B-spline interpolator
auto interpolator = builder.build_interpolator();

// Save to disk
interpolator.save("spx_put_4d.bspline");
```

**Key points:**
1. ✅ Explicit loop over (σ, r) combinations
2. ✅ Each PDE run contributes one 2D slice V(m, τ)
3. ✅ Total cost: 200 PDE runs × 20ms = 4 seconds (acceptable)
4. ✅ Result: Complete 4D grid ready for B-spline fitting

---

## Revised Implementation Phases

### Phase 1: 1D B-Spline Infrastructure (Week 1)

**Goal:** Implement separable 1D B-spline fitting (not dense 4D!)

1. Implement `BSplineBasis` (Cox-de Boor recursion)
2. Implement `BSpline1D` with **banded least-squares** (use existing `ThomasSolver`)
3. Test: Fit polynomials, validate derivatives

**Success criteria:**
- Solve 50×50 banded system in <1ms
- Interpolate cubic polynomial with <1e-10 error

---

### Phase 2: Separable Tensor-Product Fitting (Week 2)

**Goal:** Implement `BSplineTensorFitter4D` using sequential 1D fits

1. Implement separable fitting algorithm (4 sequential 1D fits)
2. Test on separable function: `f(m,τ,σ,r) = g(m)·h(τ)·i(σ)·j(r)`
3. Benchmark: Compare sequential vs hypothetical dense (show infeasibility)

**Success criteria:**
- Fit 300k-point grid in <1 second
- Memory usage <50 MB (vs terabytes for dense)
- RMS error <1e-6 on separable test function

---

### Phase 3: 4D Data Collection (Week 3)

**Goal:** Implement `PriceTable4DBuilder` with multi-run PDE strategy

1. Implement 4D grid builder
2. Integrate with `AmericanOptionSolver`
3. Run 200 PDE solves to build 4D table
4. Store 4D data in flattened array

**Success criteria:**
- Build 4D table (50×30×20×10) in <10 seconds
- Memory: ~10 MB for prices + vegas + gammas
- Validate: Check ATM put price matches known values

---

### Phase 4: Derivative-Consistent Queries (Week 4)

**Goal:** Implement clamped queries with consistent derivatives

1. Implement `query_price()` with clamping
2. Implement `query_vega()` via finite differences on clamped surface
3. Implement `query_gamma()` similarly
4. Test: Validate derivative consistency near V≈0 region

**Success criteria:**
- Vega query time <5μs (3 B-spline evals)
- Derivatives match finite differences (within 1e-5)
- No discontinuities in derivatives (except at V=0 boundary)

---

### Phase 5: Newton IV Solver (Week 5)

**Goal:** Implement fast IV solver using B-spline table

1. Implement `IVSolverInterpolated` with Newton's method
2. Use `query_price()` and `query_vega()` from Phase 4
3. Test convergence on 1000 random queries
4. Benchmark vs existing FDM IV solver

**Success criteria:**
- Converges in 4-6 iterations for 95% of in-bounds queries
- Total IV solve time <30μs (vs 143ms FDM = 4,800x speedup)
- RMS IV error <0.5bp vs FDM ground truth

---

## Updated Success Metrics

### Pre-Computation (Acceptable for Overnight Batch)

- **Time:** <10 seconds for 200 PDE runs + B-spline fitting
- **Memory:** <50 MB for 4D table (prices + control points)

### Query Performance

- **Price interpolation:** <2μs (1 B-spline tensor-product eval)
- **Vega interpolation:** <5μs (3 B-spline evals via finite diff)
- **IV calculation:** <30μs (5 Newton iterations × 2 queries)
  - **Speedup vs FDM:** 143ms / 30μs = **4,800x**

### Accuracy

- **Price RMS error:** <0.1% vs FDM ground truth
- **IV RMS error:** <0.5bp (0.0001 vol units)
- **Derivative consistency:** Vega from finite diff matches Newton convergence

### Robustness

- **No negative prices:** Clamping ensures V ≥ 0 always
- **Derivative consistency:** Finite diff on clamped surface ensures gradients match
- **Convergence rate:** >95% for in-bounds queries

---

## Conclusion

This addendum **fixes the three critical design flaws**:

1. ✅ **Separable fitting** → O(n) work instead of O(n³), feasible for 300k points
2. ✅ **Derivative-consistent clamping** → Finite diff on clamped surface, Newton sees correct gradients
3. ✅ **Multi-run PDE data collection** → Explicit strategy for building 4D grid (200 runs × 20ms = 4 sec)

The revised design is **implementable, efficient, and mathematically consistent**.

**Next steps:**
1. Approve revised design
2. Begin Phase 1: 1D B-spline with banded least-squares
3. Integrate with existing `ThomasSolver` and `AmericanOptionSolver`
