# Unified Grid Architecture with Adaptive Refinement (P6)

**Date:** 2025-11-01
**Status:** Design Approved
**Related Issue:** #39 (P6: Adaptive grid refinement)
**Dependencies:** PR #52 (Non-uniform grids)

---

## Executive Summary

This design proposes a **unified grid architecture** where the price table's moneyness grid serves directly as the FDM solver's spatial grid, eliminating interpolation errors and enabling zero-copy memory operations. Combined with adaptive refinement and pointer-swapping optimization, this approach achieves:

- **20× faster precomputation** (300ms vs 6s for 300 table entries)
- **20,000× less memory copying** (2.4 KB vs 48 MB)
- **Zero interpolation error** between FDM solver and price table
- **Target accuracy**: <1bp IV error for 95% of validation points

---

## Problem Statement

### Current Limitations

**Inefficient FDM Grid Usage:**
- Solving PDE 300 times (20 moneyness × 15 parameter combinations)
- Each solve uses internal 101-point grid, extracts 1 value, discards 100
- **Cost:** 300 solves × 20ms = 6 seconds

**Interpolation Layer Overhead:**
- FDM solves on 101-point grid in log-moneyness space
- Interpolate solution to extract value at target moneyness
- Introduces interpolation error (~0.01-0.1%)
- Wastes computed solution values

**Excessive Memory Copying:**
- memcpy at every time step: 1000 copies × 20 points × 8 bytes = 160 KB per solve
- Total: 300 solves × 160 KB = 48 MB of redundant copying

### Requirements (from Issue #39, P6)

1. Start with coarse grid (ADAPTIVE_FAST preset: ~10 moneyness points)
2. Validate interpolation error against random samples
3. Refine grid in high-error regions
4. Iterate until 95% of points have <1bp IV error
5. Integrate into `price_table_precompute()` pipeline

---

## Background

### Key Insight #1: FDM Solution Reuse

When solving Black-Scholes PDE for American options:
- **Input:** (S, K, τ, σ, r, q)
- **FDM grid:** 101 spatial points spanning [S_min, S_max]
- **Output:** Option price for each spatial point

For price table with 20 moneyness points (m = S/K):
- Old approach: Solve PDE 20 times at different S values
- **New approach:** Solve PDE once, extract 20 values from solution

**Savings:** 20× reduction in PDE solves per parameter combination.

### Key Insight #2: Unified Grid

FDM solver operates on spatial grid in log-moneyness space:
- x = ln(S/K) where S = spot, K = strike
- Price table stores values at moneyness points m = S/K

**Realization:** Instead of solving on internal 101-point grid and interpolating, solve directly on the price table's moneyness grid:

```
Old: FDM grid (101 pts) → Interpolate → Table grid (20 pts)
New: FDM grid = Table grid (20 pts) → Direct memcpy
```

**Benefits:**
- Zero interpolation error
- Zero wasted computation
- Simpler code (no interpolation layer)

### Key Insight #3: Pointer Swapping

TR-BDF2 time-stepping requires 3 solution buffers:
- `u_current`: Current time level
- `u_next`: Next time level
- `u_stage`: Intermediate stage

Old implementation: memcpy `u_next` → `u_current` at each step (1000 times)

**New approach:** Swap pointers, one final memcpy to output:

```c
// Time-stepping loop (1000 iterations)
for (step = 0; step < n_steps; step++) {
    solve_trbdf2_stages(...);

    // Swap pointers (zero-copy)
    double *temp = u_current;
    u_current = u_next;
    u_next = temp;
}

// Single final copy to caller's buffer
memcpy(solution_out, u_current, n * sizeof(double));
```

**Reduction:** 1000 memcpy operations → 1 memcpy operation = **1000× less copying**.

### Key Insight #4: Adaptive Refinement Cost Asymmetry

**Adding moneyness point:**
- Extract one more value from existing FDM solution
- Cost: ~1 μs (memory access)

**Adding parameter point (τ, σ, r, or q):**
- Solve new PDE (1000 time steps)
- Cost: ~20 ms

**Asymmetry:** 20,000× cost difference!

**Strategy:** Prioritize moneyness refinement over parameter refinement.

---

## Proposed Solution

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Price Table                                                 │
│  ┌────────────────────┐                                     │
│  │ Moneyness Grid     │ [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3] │
│  │ (n_m = 20)         │ (shared with FDM solver)            │
│  └────────────────────┘                                     │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Prices Array (n_m × n_τ × n_σ × n_r)                  │ │
│  │ Layout: LAYOUT_M_INNER (moneyness contiguous)         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ Direct memory write
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  FDM Solver (Unified Grid)                                   │
│                                                              │
│  american_option_solve_on_grid(                             │
│      table->moneyness_grid,    ← Use table's grid directly  │
│      table->n_moneyness,                                    │
│      tau, sigma, r, q, K,                                   │
│      dt, n_steps,                                           │
│      &table->prices[offset]    ← Write directly to table    │
│  );                                                          │
│                                                              │
│  Internal: Pointer swapping (zero-copy time-stepping)       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │u_current │  │ u_next   │  │ u_stage  │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
│       ↓ ↑ swap     ↓ ↑ swap                                 │
│  (1000 swaps, 1 memcpy to output)                           │
└─────────────────────────────────────────────────────────────┘
```

### Workflow

**1. Initialization**
```c
// Create price table with coarse moneyness grid
GridConfig config = grid_preset_get(
    GRID_PRESET_ADAPTIVE_FAST,  // ~10 moneyness points
    0.7, 1.3,      // moneyness range
    0.027, 2.0,    // maturity range
    0.10, 0.80,    // volatility range
    0.0, 0.10,     // rate range
    0.0, 0.0       // no dividend
);

GeneratedGrids grids = grid_generate_all(&config);
OptionPriceTable *table = price_table_create(
    grids.moneyness, grids.n_moneyness,  // 10 points initially
    grids.maturity, grids.n_maturity,    // 15 points
    grids.volatility, grids.n_volatility, // 20 points
    grids.rate, grids.n_rate,            // 10 points
    NULL, 0,
    OPTION_PUT, EXERCISE_AMERICAN
);
```

**2. Adaptive Precomputation**
```c
AdaptiveRefinementConfig adapt_config = {
    .target_iv_error = 0.0001,      // 1bp target
    .max_iterations = 5,
    .max_total_points = 100000,
    .validation_samples = 1000,
    .refinement_threshold = 2.0,    // Refine if error > 2× target
    .verbose = true
};

int status = price_table_precompute_adaptive(
    table, 0.001, 1000, &adapt_config
);
```

**3. Refinement Loop** (internal to `price_table_precompute_adaptive`)
```c
for (iter = 0; iter < config->max_iterations; iter++) {
    // Solve all (τ, σ, r, q) combinations with current moneyness grid
    for each (i_tau, i_sigma, i_r):
        size_t offset = index_4d(0, i_tau, i_sigma, i_r, dims);

        american_option_solve_on_grid(
            table->moneyness_grid,
            table->n_moneyness,
            tau[i_tau], sigma[i_sigma], r[i_r], 0.0,
            100.0,  // strike
            dt, n_steps,
            &table->prices[offset]  // Direct write to table
        );

    // Validate interpolation error
    ValidationResult result = validate_interpolation_error(
        table, config->validation_samples, config->target_iv_error
    );

    // Check convergence
    if (result.max_iv_error < config->target_iv_error &&
        result.fraction_below_1bp > 0.95) {
        break;  // Success!
    }

    // Identify high-error regions and add refinement points
    double *new_points = identify_refinement_points(&result, &n_new);
    expand_table_grid(table, new_points, n_new);
    free(new_points);

    // Next iteration solves on expanded grid
}
```

**4. Result**
- Price table with adaptively refined moneyness grid
- Target accuracy achieved (<1bp IV error for 95% of points)
- Typical final grid size: 15-25 moneyness points (vs 50+ for uniform grid)

---

## API Design

### Core Unified Grid Solver

```c
/**
 * Solve American option PDE on provided moneyness grid (unified grid design)
 *
 * Uses pointer swapping during time-stepping to eliminate memcpy overhead,
 * with single final copy to output buffer.
 *
 * @param moneyness_grid Price table's moneyness grid (n_moneyness points)
 * @param n_moneyness Number of moneyness points
 * @param tau Time to maturity (years)
 * @param sigma Volatility
 * @param r Risk-free rate
 * @param q Dividend yield
 * @param strike Strike price (K)
 * @param dt Time step size
 * @param n_time_steps Number of backward time steps
 * @param solution_out Output buffer (n_moneyness doubles) - receives option prices
 * @return 0 on success, non-zero on error
 *
 * Grid Requirements:
 * - moneyness_grid must be sorted ascending
 * - Points represent m = S/K (spot/strike ratio)
 * - For log-space PDE, grid is converted internally: x = ln(m)
 *
 * Memory:
 * - Allocates internal workspace (12n doubles)
 * - Uses pointer swaps during time-stepping (O(1) overhead per step)
 * - Single memcpy to solution_out at completion (O(n) total)
 */
int american_option_solve_on_grid(
    const double *moneyness_grid,
    size_t n_moneyness,
    double tau, double sigma, double r, double q,
    double strike,
    double dt, size_t n_time_steps,
    double *solution_out
);
```

### Adaptive Precomputation

```c
/**
 * Adaptive refinement configuration
 */
typedef struct {
    double target_iv_error;      ///< Target IV error (e.g., 0.0001 for 1bp)
    size_t max_iterations;        ///< Maximum refinement iterations (default: 5)
    size_t max_total_points;      ///< Memory limit on total grid points
    size_t validation_samples;    ///< Number of random validation points (default: 1000)
    double refinement_threshold;  ///< Add points where error > threshold × target
    bool verbose;                 ///< Print progress information
} AdaptiveRefinementConfig;

/**
 * Default adaptive refinement configuration
 */
AdaptiveRefinementConfig adaptive_default_config(void);

/**
 * Precompute option prices with adaptive grid refinement
 *
 * Workflow:
 * 1. Start with coarse grid (from table's initial configuration)
 * 2. Compute all prices on current grid
 * 3. Validate interpolation error on random samples
 * 4. If error > target, add points in high-error regions
 * 5. Repeat until target accuracy or iteration limit
 *
 * @param table Price table to populate (must have initial grid)
 * @param dt PDE time step size
 * @param n_time_steps Number of backward time steps
 * @param config Adaptive refinement configuration
 * @return 0 on success, non-zero on error
 *
 * Memory Layout Requirement:
 * - Table must use LAYOUT_M_INNER for contiguous moneyness values
 *
 * Progress Monitoring:
 * - Use USDT probes to monitor refinement progress
 * - IVCALC_TRACE_ALGO_PROGRESS reports iteration and grid size
 */
int price_table_precompute_adaptive(
    OptionPriceTable *table,
    double dt,
    size_t n_time_steps,
    const AdaptiveRefinementConfig *config
);
```

### Validation API

```c
/**
 * Validation result for interpolation error
 */
typedef struct {
    double max_iv_error;          ///< Maximum IV error (bp)
    double mean_iv_error;         ///< Mean IV error (bp)
    double p95_iv_error;          ///< 95th percentile IV error (bp)
    double fraction_below_1bp;    ///< Fraction of points with error < 1bp
    size_t n_samples;             ///< Number of validation samples

    // High-error regions for refinement
    double *high_error_moneyness; ///< Moneyness values with high error
    size_t n_high_error;          ///< Number of high-error points
} ValidationResult;

/**
 * Validate interpolation error on random samples
 *
 * @param table Price table to validate
 * @param n_samples Number of random validation points
 * @param target_error Target IV error (bp)
 * @return Validation result (caller must free high_error_moneyness)
 */
ValidationResult validate_interpolation_error(
    const OptionPriceTable *table,
    size_t n_samples,
    double target_error
);

/**
 * Free validation result memory
 */
void validation_result_free(ValidationResult *result);
```

### Grid Expansion API (Internal)

```c
/**
 * Expand price table with additional moneyness points
 *
 * Strategy:
 * 1. Merge new points with existing grid (sorted)
 * 2. Allocate larger price array
 * 3. Copy existing prices to new positions
 * 4. Mark new points as NaN (to be computed)
 * 5. Update table structure
 *
 * @param table Price table to expand
 * @param new_m_points New moneyness points to add
 * @param n_new Number of new points
 */
void expand_table_grid(
    OptionPriceTable *table,
    const double *new_m_points,
    size_t n_new
);
```

---

## Performance Analysis

### Cost Model Comparison

**Old Architecture (Current):**
```
FDM solves:    300 solves (20 moneyness × 15 parameters)
Per solve:     1000 steps × 101 points × ~50 flops = 5M flops (~20ms)
Total FDM:     300 × 20ms = 6,000ms = 6.0 seconds
Interpolation: 300 × 10μs = 3ms
Memcpy:        300 × 160KB = 48 MB
Total:         ~6 seconds
```

**Unified Grid Architecture (Proposed):**
```
FDM solves:    15 solves (one per (τ, σ, r, q) combo)
Per solve:     1000 steps × 20 points × ~50 flops = 1M flops (~20ms)
Total FDM:     15 × 20ms = 300ms
Interpolation: 0 (direct solve on table grid)
Memcpy:        15 × 160 bytes = 2.4 KB
Total:         ~300ms
```

**Speedup:** 6000ms / 300ms = **20× faster**

**Memory reduction:** 48 MB / 2.4 KB = **20,000× less copying**

### Adaptive Refinement Overhead

**Per iteration cost:**
- Precompute: 300ms (unified grid solve)
- Validation: 1000 samples × 0.5μs interpolation = 0.5ms
- Error analysis: negligible
- Grid expansion: one-time realloc + copy (~1ms)
- **Total per iteration:** ~302ms

**Typical refinement sequence:**
```
Iteration 0: 10 moneyness points (coarse start)
Iteration 1: Add 5 points → 15 points (high-error regions)
Iteration 2: Add 3 points → 18 points (remaining gaps)
Iteration 3: Add 2 points → 20 points (fine-tuning)
Converged: 95% of points < 1bp IV error
```

**Total adaptive cost:** 4 iterations × ~300ms = **~1.2 seconds**

**Comparison to non-adaptive dense grid:**
- Dense uniform grid: 50 moneyness points × 15 parameters = 750 solves
- Cost: 750 × 20ms (on 50-point grid) = 15 seconds

**Adaptive is 12× faster than dense grid while achieving target accuracy.**

### Memory Footprint

**During precomputation:**
- Price table: 20 × 15 × 20 × 10 × 8 bytes = 480 KB
- Per-solve workspace: 12 × 20 × 8 bytes = 1.92 KB
- **Total peak:** ~482 KB

**After precomputation:**
- Price table: 480 KB (persists)
- Workspace: freed after each solve

**Comparison to dense uniform grid:**
- Dense: 50 × 30 × 20 × 10 × 8 = 2.4 MB (5× larger)

---

## Implementation Details

### Pointer Swapping Optimization

**Current TR-BDF2 implementation** (src/pde_solver.c:526):
```c
// Every time step (1000+ times):
if (status == 0) {
    memcpy(u_current, u_next, n * sizeof(double));
}
```

**Proposed refactoring:**

1. **Change workspace allocation** from fixed slices to swappable pointers:

```c
// Current: Fixed slices from workspace buffer
solver->u_current = solver->workspace + offset;
solver->u_next = solver->workspace + offset + n;
solver->u_stage = solver->workspace + offset + 2*n;

// Proposed: Separate buffers with swappable pointers
double *buffer_A = aligned_alloc(64, n * sizeof(double));
double *buffer_B = aligned_alloc(64, n * sizeof(double));
double *buffer_C = aligned_alloc(64, n * sizeof(double));

double *u_current = buffer_A;
double *u_next = buffer_B;
double *u_stage = buffer_C;
```

2. **Replace memcpy with pointer swap** in time-stepping loop:

```c
// In pde_solver_step_internal():
if (status == 0) {
    // Swap pointers instead of memcpy
    double *temp = u_current;
    u_current = u_next;
    u_next = temp;
}
```

3. **Track final buffer location** and copy once to output:

```c
// After solve completes:
memcpy(solution_out, u_current, n * sizeof(double));
```

**Performance impact:**
- Before: 1000 memcpy × 20 points × 8 bytes = 160 KB per solve
- After: 1 memcpy × 20 points × 8 bytes = 160 bytes per solve
- **Reduction: 1000× fewer bytes copied**

### Memory Layout Requirement

For zero-copy direct writes, price table must use **LAYOUT_M_INNER**:

```c
// Layout: moneyness is innermost dimension
// prices[i_tau][i_sigma][i_r][i_m] in memory
size_t index_4d(size_t i_m, size_t i_tau, size_t i_sigma, size_t i_r,
                size_t n_m, size_t n_tau, size_t n_sigma, size_t n_r) {
    return ((i_tau * n_sigma + i_sigma) * n_r + i_r) * n_m + i_m;
}

// This ensures moneyness values are contiguous:
// prices[offset + 0], prices[offset + 1], ..., prices[offset + n_m-1]
```

Direct write into table memory:
```c
for each (i_tau, i_sigma, i_r, i_q):
    size_t offset = index_4d(0, i_tau, i_sigma, i_r, dims);

    american_option_solve_on_grid(
        table->moneyness_grid,
        table->n_moneyness,
        tau, sigma, r, q, strike,
        dt, n_steps,
        &table->prices[offset]  // ← Contiguous n_moneyness-sized slice
    );
```

**Constraint validation:**
```c
// In price_table_precompute_adaptive():
if (table->layout != LAYOUT_M_INNER) {
    return PRICE_TABLE_ERROR_LAYOUT_MISMATCH;
}
```

### Grid Coordinate Transformation

FDM solver works in **log-moneyness space** (x = ln(S/K)):

```c
// Transform moneyness grid to log-space
double *x_grid = malloc(n_moneyness * sizeof(double));
for (size_t i = 0; i < n_moneyness; i++) {
    x_grid[i] = log(moneyness_grid[i]);
}

// Solve Black-Scholes PDE in x-space
solve_black_scholes_pde(x_grid, n_moneyness, tau, sigma, r, q, solution);

// Transform back: option price = solution * strike
for (size_t i = 0; i < n_moneyness; i++) {
    solution_out[i] = solution[i] * strike;
}

free(x_grid);
```

### Refinement Point Selection

**Strategy:** Add points midway between existing grid points in high-error regions:

```c
double* identify_refinement_points(
    const ValidationResult *result,
    size_t *n_new_out
) {
    // 1. Group validation samples by moneyness bins
    // 2. Compute average error per bin
    // 3. Identify bins with error > threshold
    // 4. For each high-error bin, add midpoint between grid boundaries

    size_t n_new = 0;
    double *new_points = malloc(result->n_high_error * sizeof(double));

    for (size_t i = 0; i < result->n_high_error; i++) {
        double m = result->high_error_moneyness[i];

        // Find bracketing grid points
        size_t idx_left = find_left_bracket(table->moneyness_grid,
                                            table->n_moneyness, m);
        double m_left = table->moneyness_grid[idx_left];
        double m_right = table->moneyness_grid[idx_left + 1];

        // Add midpoint (in log-space for better distribution)
        double m_mid = exp((log(m_left) + log(m_right)) / 2.0);
        new_points[n_new++] = m_mid;
    }

    // Remove duplicates
    sort_and_unique(new_points, &n_new);

    *n_new_out = n_new;
    return new_points;
}
```

---

## Greek Computation Optimization

The unified grid architecture enables a significant optimization for computing option Greeks: **extract Delta, Gamma, and Theta directly from the FDM solver at zero additional cost**.

### Background: Greeks from FDM

The FDM solver computes spatial and temporal derivatives as part of solving the PDE:

**Black-Scholes PDE in log-moneyness space (x = ln(S/K)):**
```
∂u/∂τ = L(u) = (σ²/2)·∂²u/∂x² + (r - q - σ²/2)·∂u/∂x - r·u
```

To solve this, the spatial operator evaluates:
- **First derivative:** ∂u/∂x (for advection term)
- **Second derivative:** ∂²u/∂x² (for diffusion term)

These derivatives are **directly related to option Greeks:**
- **Delta:** Δ = ∂V/∂S = (∂u/∂x) · (1/S)
- **Gamma:** Γ = ∂²V/∂S² = (∂²u/∂x²) · (1/S²)
- **Theta:** Θ = -∂V/∂τ (from time-stepping)

### Direct Greek Extraction

Instead of computing Greeks via finite differences on the price table (old approach from P4), we extract them directly during the FDM solve.

**Extended API signature:**

```c
/**
 * Solve American option PDE with Greeks computation
 *
 * Computes option prices along with Delta, Gamma, and Theta directly
 * from FDM spatial and temporal derivatives (zero additional cost).
 *
 * @param moneyness_grid Price table's moneyness grid
 * @param n_moneyness Number of moneyness points
 * @param tau Time to maturity
 * @param sigma Volatility
 * @param r Risk-free rate
 * @param q Dividend yield
 * @param strike Strike price
 * @param dt Time step size
 * @param n_time_steps Number of backward time steps
 * @param solution_out Output: Option prices (n_moneyness doubles)
 * @param delta_out Output: Delta values (n_moneyness doubles), or NULL to skip
 * @param gamma_out Output: Gamma values (n_moneyness doubles), or NULL to skip
 * @param theta_out Output: Theta values (n_moneyness doubles), or NULL to skip
 * @return 0 on success, non-zero on error
 *
 * Greeks are computed:
 * - Delta: From ∂u/∂x using central differences on spatial grid
 * - Gamma: From ∂²u/∂x² (already computed in spatial operator)
 * - Theta: From final time step difference
 *
 * Vega and Rho still require finite differences across parameter dimensions.
 */
int american_option_solve_on_grid_with_greeks(
    const double *moneyness_grid,
    size_t n_moneyness,
    double tau, double sigma, double r, double q,
    double strike,
    double dt, size_t n_time_steps,
    double *solution_out,
    double *delta_out,
    double *gamma_out,
    double *theta_out
);
```

### Implementation Details

**1. Delta Computation (∂V/∂S)**

```c
// After FDM solve completes, compute spatial derivative
const double dx = x_grid[1] - x_grid[0];  // Log-space grid spacing

// Interior points: central difference
for (size_t i = 1; i < n_moneyness - 1; i++) {
    // ∂u/∂x in log-space
    double dudx = (solution[i+1] - solution[i-1]) / (2.0 * dx);

    // Transform to delta: Δ = ∂V/∂S = (∂u/∂x) · (1/S)
    double S = moneyness_grid[i] * strike;
    delta_out[i] = dudx / S;
}

// Boundary points: one-sided differences
double dudx_left = (solution[1] - solution[0]) / dx;
delta_out[0] = dudx_left / (moneyness_grid[0] * strike);

double dudx_right = (solution[n-1] - solution[n-2]) / dx;
delta_out[n-1] = dudx_right / (moneyness_grid[n-1] * strike);
```

**2. Gamma Computation (∂²V/∂S²)**

```c
// Gamma is already computed in the spatial operator!
// Just need to extract it and transform coordinates

for (size_t i = 1; i < n_moneyness - 1; i++) {
    // ∂²u/∂x² in log-space (central difference)
    double d2udx2 = (solution[i+1] - 2.0*solution[i] + solution[i-1]) / (dx * dx);

    // Transform to gamma: Γ = ∂²V/∂S² = (∂²u/∂x² - ∂u/∂x) / S²
    double S = moneyness_grid[i] * strike;
    double dudx = (solution[i+1] - solution[i-1]) / (2.0 * dx);

    gamma_out[i] = (d2udx2 - dudx) / (S * S);
}

// Boundary points: use one-sided stencils
```

**3. Theta Computation (-∂V/∂τ)**

```c
// Save the previous time level during final time step
// (requires minimal modification to solver)

// In pde_solver_step_internal(), before final step:
if (step == n_steps - 1) {
    memcpy(u_prev_time, u_current, n * sizeof(double));
}

// After solve completes:
for (size_t i = 0; i < n_moneyness; i++) {
    // Θ = -∂V/∂τ (negative because we step backward in time)
    theta_out[i] = -(solution_out[i] - u_prev_time[i]) / dt;
}
```

**4. Vega and Rho (Finite Differences)**

Vega (∂V/∂σ) and Rho (∂V/∂r) still require solving at different parameter values:

```c
// Computed from price table after precomputation
void compute_vega_from_table(OptionPriceTable *table) {
    const double dsigma = 0.01;  // 1% volatility bump

    for each table entry (i_m, i_tau, i_sigma, i_r):
        // Use interpolation to get prices at σ ± Δσ
        double price_up = interpolate_4d(
            table, m, tau, sigma + dsigma, r
        );
        double price_down = interpolate_4d(
            table, m, tau, sigma - dsigma, r
        );

        vega = (price_up - price_down) / (2.0 * dsigma);
}

// Similar for rho
void compute_rho_from_table(OptionPriceTable *table) {
    const double dr = 0.0001;  // 1bp rate bump
    // ... similar finite difference ...
}
```

### Integration with Precomputation

**Modified precomputation workflow:**

```c
// Allocate Greek arrays in price table
table->delta = malloc(total_size * sizeof(double));
table->gamma = malloc(total_size * sizeof(double));
table->theta = malloc(total_size * sizeof(double));
table->vega = malloc(total_size * sizeof(double));
table->rho = malloc(total_size * sizeof(double));

// Precompute with Greeks
for each (i_tau, i_sigma, i_r, i_q):
    size_t offset = index_4d(0, i_tau, i_sigma, i_r, dims);

    american_option_solve_on_grid_with_greeks(
        table->moneyness_grid,
        table->n_moneyness,
        tau, sigma, r, q, strike,
        dt, n_steps,
        &table->prices[offset],   // Prices
        &table->delta[offset],    // Delta (direct from FDM)
        &table->gamma[offset],    // Gamma (direct from FDM)
        &table->theta[offset]     // Theta (direct from FDM)
    );

// Compute remaining Greeks from finite differences
compute_vega_from_table(table);
compute_rho_from_table(table);
```

### Performance Analysis

**Old approach (P4 - finite differences on price table):**

For each Greek at each table entry:
1. Interpolate price at (S + ΔS)
2. Interpolate price at (S - ΔS)
3. Compute finite difference

Cost per Greek:
- 2 interpolations × 0.5μs = 1.0μs
- Finite difference arithmetic: ~0.1μs
- **Total: ~1.1μs per Greek**

For Delta, Gamma, Theta on 300 entries:
- 3 Greeks × 300 entries × 1.1μs = **~1ms**

**New approach (direct from FDM):**

Greeks computed during FDM solve:
- Delta: Central difference on existing solution (~10 flops per point)
- Gamma: Already computed in spatial operator (~0 extra flops)
- Theta: One subtraction per point (~2 flops per point)

Cost per solve:
- Delta: 20 points × 10 flops × ~1ns = 0.2μs
- Gamma: 0μs (free - already computed)
- Theta: 20 points × 2 flops × ~1ns = 0.04μs
- **Total: ~0.24μs per solve** (15 solves = 3.6μs total)

**Speedup: 1ms / 3.6μs ≈ 280× faster for Delta/Gamma/Theta**

(Essentially free - rounding error in FDM solve time)

### Accuracy Comparison

**Finite difference Greeks (old approach):**
```
Δ_FD = (V(S + ΔS) - V(S - ΔS)) / (2ΔS)
Error = O(ΔS²) + interpolation_error
```

With interpolation error ~0.1-0.5%, total error ~0.2-1% for Greeks.

**Direct Greeks (new approach):**
```
Δ_FDM = (∂u/∂x) / S computed from FDM grid
Error = O(dx²) only
```

With dx ≈ 0.05 (20-point tanh grid), error ~0.25% (0.05²).

**No interpolation error - 2-3× more accurate Greeks.**

### Storage Requirement

**Price table with Greeks:**
```
Prices: n_m × n_tau × n_sigma × n_r × 8 bytes
Greeks: 5 × (prices size) for delta, gamma, theta, vega, rho

Total: 6 × prices size

Example (20 × 15 × 20 × 10):
- Prices: 480 KB
- Greeks: 2.4 MB
- Total: 2.88 MB
```

**Memory layout (contiguous arrays):**
```c
struct OptionPriceTable {
    double *prices;    // Base price array
    double *delta;     // Same dimensions
    double *gamma;
    double *theta;
    double *vega;
    double *rho;
    // ... other fields ...
};
```

### API Extensions

**Query Greeks along with price:**

```c
// Single-point query with all Greeks
OptionData price_table_query_with_greeks(
    const OptionPriceTable *table,
    double moneyness, double tau, double sigma, double r
) {
    OptionData result;
    result.price = interpolate_4d(table->prices, ...);
    result.delta = interpolate_4d(table->delta, ...);
    result.gamma = interpolate_4d(table->gamma, ...);
    result.theta = interpolate_4d(table->theta, ...);
    result.vega = interpolate_4d(table->vega, ...);
    result.rho = interpolate_4d(table->rho, ...);
    return result;
}
```

### Testing Strategy for Greeks

**Test 1: Greek sign correctness**
```c
void test_greek_signs() {
    // American put Greeks
    double delta[20], gamma[20], theta[20];

    solve_american_put_with_greeks(..., delta, gamma, theta);

    // Put delta: negative
    for (size_t i = 0; i < 20; i++) {
        EXPECT_LT(delta[i], 0.0);
    }

    // Gamma: positive (convexity)
    for (size_t i = 0; i < 20; i++) {
        EXPECT_GT(gamma[i], 0.0);
    }

    // Theta: typically negative (time decay)
    // (except deep ITM American options)
}
```

**Test 2: Greek accuracy vs finite differences**
```c
void test_greek_accuracy() {
    // Compute Greeks directly
    double delta_direct[20];
    solve_with_greeks(..., delta_direct, ...);

    // Compute via finite differences on fine grid
    double delta_fd = (solve(S + dS) - solve(S - dS)) / (2*dS);

    // Direct method should match FD within 1%
    EXPECT_NEAR(delta_direct[10], delta_fd, 0.01 * delta_fd);
}
```

**Test 3: Put-call parity for Greeks**
```c
void test_put_call_parity_greeks() {
    // Call and put with same parameters
    double delta_call, delta_put;
    solve_call_with_greeks(..., &delta_call, ...);
    solve_put_with_greeks(..., &delta_put, ...);

    // Put-call parity: Δ_call - Δ_put = exp(-qτ)
    double expected_diff = exp(-q * tau);
    EXPECT_NEAR(delta_call - delta_put, expected_diff, 1e-6);
}
```

---

## Testing Strategy

### Unit Tests

**Test 1: Unified grid solver correctness**
```c
void test_unified_grid_solver() {
    double m_grid[] = {0.8, 0.9, 1.0, 1.1, 1.2};
    size_t n = 5;
    double solution[5];

    int status = american_option_solve_on_grid(
        m_grid, n,
        0.25, 0.20, 0.05, 0.0,  // tau, sigma, r, q
        100.0, 0.001, 250,      // strike, dt, n_steps
        solution
    );

    EXPECT_EQ(status, 0);
    EXPECT_GT(solution[2], 0.0);  // ATM option has value

    // Verify monotonicity (put prices decrease with moneyness)
    for (size_t i = 1; i < n; i++) {
        EXPECT_LT(solution[i], solution[i-1]);
    }
}
```

**Test 2: Pointer swapping produces identical results**
```c
void test_pointer_swapping_equivalence() {
    // Solve with old memcpy approach
    double solution_memcpy[20];
    solve_with_memcpy(solution_memcpy);

    // Solve with new pointer swapping
    double solution_swap[20];
    solve_with_pointer_swap(solution_swap);

    // Results should be identical
    for (size_t i = 0; i < 20; i++) {
        EXPECT_NEAR(solution_memcpy[i], solution_swap[i], 1e-12);
    }
}
```

**Test 3: Grid expansion preserves existing data**
```c
void test_grid_expansion() {
    OptionPriceTable *table = create_test_table(10);  // 10 moneyness points

    // Precompute initial prices
    precompute_all_prices(table);

    // Save old prices
    double *old_prices = save_prices(table);

    // Expand grid
    double new_points[] = {0.85, 0.95, 1.05};
    expand_table_grid(table, new_points, 3);

    EXPECT_EQ(table->n_moneyness, 13);  // 10 + 3

    // Verify old prices preserved at correct indices
    for (size_t i = 0; i < 10; i++) {
        size_t new_idx = find_old_point_in_new_grid(i);
        EXPECT_EQ(table->prices[new_idx], old_prices[i]);
    }

    free(old_prices);
}
```

### Integration Tests

**Test 4: Adaptive refinement convergence**
```c
void test_adaptive_refinement_convergence() {
    GridConfig config = grid_preset_get(
        GRID_PRESET_ADAPTIVE_FAST,  // Start coarse
        0.7, 1.3, 0.027, 2.0, 0.10, 0.80, 0.0, 0.10, 0.0, 0.0
    );

    GeneratedGrids grids = grid_generate_all(&config);
    OptionPriceTable *table = price_table_create(
        grids.moneyness, grids.n_moneyness,
        grids.maturity, grids.n_maturity,
        grids.volatility, grids.n_volatility,
        grids.rate, grids.n_rate,
        NULL, 0,
        OPTION_PUT, EXERCISE_AMERICAN
    );

    size_t initial_n = table->n_moneyness;

    AdaptiveRefinementConfig adapt_config = adaptive_default_config();
    adapt_config.target_iv_error = 0.0001;  // 1bp

    int status = price_table_precompute_adaptive(
        table, 0.001, 1000, &adapt_config
    );

    EXPECT_EQ(status, 0);

    // Grid should have expanded
    EXPECT_GT(table->n_moneyness, initial_n);

    // Validate final accuracy
    ValidationResult result = validate_interpolation_error(
        table, 1000, 0.0001
    );

    EXPECT_LT(result.max_iv_error, 0.0001);
    EXPECT_GT(result.fraction_below_1bp, 0.95);

    validation_result_free(&result);
    price_table_destroy(table);
}
```

**Test 5: Memory layout requirement enforcement**
```c
void test_layout_enforcement() {
    OptionPriceTable *table = create_test_table_with_layout(
        LAYOUT_TAU_INNER  // Wrong layout!
    );

    AdaptiveRefinementConfig config = adaptive_default_config();

    int status = price_table_precompute_adaptive(
        table, 0.001, 1000, &config
    );

    // Should fail with layout error
    EXPECT_EQ(status, PRICE_TABLE_ERROR_LAYOUT_MISMATCH);

    price_table_destroy(table);
}
```

### Accuracy Benchmarks

**Benchmark 1: Unified vs interpolation approach**
```c
void benchmark_accuracy_unified_vs_interpolation() {
    // Reference: Very fine grid (1001 points)
    double price_ref = solve_american_fine_grid(
        100.0, 100.0, 0.25, 0.20, 0.05, 0.0, 1001
    );

    // Method 1: Old (101-point grid + interpolate)
    double price_interp = solve_and_interpolate(
        100.0, 100.0, 0.25, 0.20, 0.05, 0.0, 101
    );

    // Method 2: Unified (20-point grid, direct)
    double m_grid[20];
    generate_tanh_grid(m_grid, 20, 0.7, 1.3, 1.0, 3.0);
    double solution[20];
    american_option_solve_on_grid(
        m_grid, 20, 0.25, 0.20, 0.05, 0.0, 100.0, 0.001, 250, solution
    );
    double price_unified = solution[10];  // ATM

    double error_interp = fabs(price_interp - price_ref);
    double error_unified = fabs(price_unified - price_ref);

    // Unified should have lower error (no interpolation)
    EXPECT_LT(error_unified, error_interp);

    printf("Interpolation error: %.6f\n", error_interp);
    printf("Unified grid error: %.6f\n", error_unified);
}
```

---

## Migration Path

### Phase 1: Core Infrastructure
- Implement `american_option_solve_on_grid()` with unified grid support
- Refactor TR-BDF2 solver to use pointer swapping instead of memcpy
- Add LAYOUT_M_INNER memory layout to price table

### Phase 2: Validation Framework
- Implement `validate_interpolation_error()`
- Add IV error computation (price → IV via Brent's method)
- Implement random sampling strategy

### Phase 3: Adaptive Refinement
- Implement `expand_table_grid()` for grid expansion
- Implement `identify_refinement_points()` for error-based selection
- Integrate into `price_table_precompute_adaptive()`

### Phase 4: Testing & Benchmarking
- Unit tests for all components
- Integration tests for full adaptive workflow
- Accuracy benchmarks vs reference solutions
- Performance benchmarks vs old approach

### Phase 5: Documentation & Examples
- Update CLAUDE.md with new workflow
- Add example program: `example_adaptive_refinement.c`
- Update benchmark documentation

---

## Future Work

### P6.1: Multi-Dimensional Refinement
Current design focuses on moneyness refinement (cheapest). Future work:
- Adaptive maturity refinement (concentrate near short-term)
- Adaptive volatility refinement (concentrate near typical trading range)
- Cost-aware multi-dimensional refinement strategy

### P6.2: Parallel Adaptive Refinement
Current design solves parameter combinations sequentially. Opportunities:
- OpenMP parallelization across (τ, σ, r, q) combinations
- Batch validation for better cache locality
- Parallel error analysis on validation set

### P6.3: Hierarchical Error Estimation
Current approach uses random sampling. Alternatives:
- Richardson extrapolation for error estimation
- Hierarchical grid comparison (coarse vs fine)
- A posteriori error bounds from PDE theory

### P6.4: Reuse of Existing Solutions
When expanding grid, we recompute all prices. Potential optimization:
- Reuse prices at existing grid points (marked non-NaN)
- Only compute new points (marked NaN)
- Requires tracking which points need computation

---

## References

1. **TR-BDF2 Time-Stepping:**
   Ascher, Ruuth, Wetton (1995). "Implicit-explicit methods for time-dependent partial differential equations."

2. **Non-Uniform Grids (PR #52):**
   docs/plans/2025-11-01-non-uniform-grids-design.md

3. **Adaptive Mesh Refinement:**
   Berger & Colella (1989). "Local adaptive mesh refinement for shock hydrodynamics."

4. **Option Pricing with FDM:**
   Wilmott, Howison, Dewynne (1995). "The Mathematics of Financial Derivatives."

5. **Multilinear Interpolation:**
   Press et al. (2007). "Numerical Recipes: The Art of Scientific Computing."

---

## Appendix: Error Analysis

### Interpolation Error Sources

1. **Grid discretization error:** O(dx²) for uniform grids, O(dx⁴) for non-uniform
2. **Time discretization error:** O(dt²) for TR-BDF2
3. **Interpolation error:** O(dx²) for linear, O(dx⁴) for cubic splines
4. **IV inversion error:** O(ε_price × ∂IV/∂price) via Brent's method

**Unified grid eliminates (3), reducing total error by ~10-30%.**

### Target Accuracy Justification

**1bp IV error = 0.0001 in volatility space**

For ATM options with σ ≈ 0.20, 1bp represents:
- Relative error: 0.0001 / 0.20 = 0.05%
- Price impact: ~$0.10 on $100 option (vega ≈ $40/vol point)

**Why 1bp?**
- Market bid-ask spreads: 1-5bp for liquid options
- Trading costs: ~1-2bp
- Model risk: ~2-5bp
- **Conclusion:** <1bp is sub-market precision, appropriate for pricing library

### Validation Set Size

**1000 random samples** provides:
- Confidence interval: ±3% at 95% confidence
- Coverage: ~99% of parameter space visited (4D grid)
- Compute time: <1ms (negligible overhead)

Sufficient for adaptive refinement decisions.

---

## Sign-Off

**Design reviewed and approved for implementation.**

**Next Steps:**
1. Create worktree for P6 implementation
2. Implement Phase 1 (Core Infrastructure)
3. Write unit tests as components are built
4. Integrate and test full adaptive workflow
5. Benchmark vs current approach
6. Create PR for review

**Estimated effort:** 3-5 days of focused development.
