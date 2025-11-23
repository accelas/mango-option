# P6 Implementation Plan: Unified Grid + Adaptive Refinement

**Branch:** `feature/p6-unified-grid-adaptive`
**Worktree:** `/home/kai/work/iv_calc/.worktrees/p6-unified-grid`
**Design Doc:** `docs/plans/2025-11-01-unified-grid-adaptive-refinement.md`

---

## Overview

This plan implements the unified grid architecture with adaptive refinement (P6 from issue #39). The implementation follows the migration path outlined in the design document, broken into 5 phases with specific tasks, file changes, and verification steps.

**Key deliverables:**
1. Unified grid FDM solver (moneyness grid = FDM spatial grid)
2. Pointer swapping optimization (1000× memcpy reduction)
3. Greek computation from FDM derivatives (280× speedup for delta/gamma/theta)
4. Adaptive refinement workflow (95% of points < 1bp IV error)
5. Integration tests and benchmarks

---

## Phase 1: Core Infrastructure

**Goal:** Implement unified grid solver with pointer swapping and Greek extraction.

### Task 1.1: Refactor PDE Solver for Pointer Swapping

**Files to modify:**
- `src/pde_solver.h`
- `src/pde_solver.c`

**Changes:**

1. **Update workspace allocation** (pde_solver.c:400-441):
   ```c
   // OLD: Fixed slices from single buffer
   solver->u_current = solver->workspace + offset;
   solver->u_next = solver->workspace + offset + n_aligned;
   solver->u_stage = solver->workspace + offset + 2*n_aligned;

   // NEW: Allocate separate buffers for swappable pointers
   solver->buffer_A = aligned_alloc(alignment, n * sizeof(double));
   solver->buffer_B = aligned_alloc(alignment, n * sizeof(double));
   solver->buffer_C = aligned_alloc(alignment, n * sizeof(double));

   solver->u_current = solver->buffer_A;
   solver->u_next = solver->buffer_B;
   solver->u_stage = solver->buffer_C;
   ```

2. **Add buffer pointers to PDESolver struct** (pde_solver.h):
   ```c
   typedef struct {
       // ... existing fields ...

       // Swappable solution buffers
       double *buffer_A;
       double *buffer_B;
       double *buffer_C;

       // Current buffer assignments
       double *u_current;
       double *u_next;
       double *u_stage;

       // ... rest of fields ...
   } PDESolver;
   ```

3. **Replace memcpy with pointer swap** (pde_solver.c:524-527):
   ```c
   // OLD:
   if (status == 0) {
       memcpy(u_current, u_next, n * sizeof(double));
   }

   // NEW:
   if (status == 0) {
       // Swap pointers (zero-copy)
       double *temp = solver->u_current;
       solver->u_current = solver->u_next;
       solver->u_next = temp;
   }
   ```

4. **Update pde_solver_destroy()** to free new buffers:
   ```c
   void pde_solver_destroy(PDESolver *solver) {
       if (solver == nullptr) return;

       pde_free_grid(&solver->grid);
       free(solver->workspace);  // Existing workspace
       free(solver->buffer_A);   // NEW
       free(solver->buffer_B);   // NEW
       free(solver->buffer_C);   // NEW
       free(solver);
   }
   ```

**Verification:**
```bash
cd /home/kai/work/iv_calc/.worktrees/p6-unified-grid
bazel test //tests:pde_solver_test
# All tests should pass with identical results
```

---

### Task 1.2: Implement Unified Grid American Option Solver

**New files to create:**
- `src/american_option_unified.h`
- `src/american_option_unified.c`

**API signature** (american_option_unified.h):
```c
#ifndef MANGO_AMERICAN_OPTION_UNIFIED_H
#define MANGO_AMERICAN_OPTION_UNIFIED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Solve American option PDE on provided moneyness grid (unified grid design)
 *
 * @param moneyness_grid Price table's moneyness grid (n_moneyness points)
 * @param n_moneyness Number of moneyness points
 * @param tau Time to maturity (years)
 * @param sigma Volatility
 * @param r Risk-free rate
 * @param q Dividend yield
 * @param strike Strike price (K)
 * @param option_type 'C' for call, 'P' for put
 * @param dt Time step size
 * @param n_time_steps Number of backward time steps
 * @param solution_out Output buffer (n_moneyness doubles) - receives option prices
 * @return 0 on success, non-zero on error
 */
int american_option_solve_on_grid(
    const double *moneyness_grid,
    size_t n_moneyness,
    double tau, double sigma, double r, double q,
    double strike,
    char option_type,
    double dt, size_t n_time_steps,
    double *solution_out
);

/**
 * Solve American option PDE with Greeks computation
 *
 * @param delta_out Output: Delta values (n_moneyness doubles), or NULL to skip
 * @param gamma_out Output: Gamma values (n_moneyness doubles), or NULL to skip
 * @param theta_out Output: Theta values (n_moneyness doubles), or NULL to skip
 * Other parameters same as american_option_solve_on_grid
 */
int american_option_solve_on_grid_with_greeks(
    const double *moneyness_grid,
    size_t n_moneyness,
    double tau, double sigma, double r, double q,
    double strike,
    char option_type,
    double dt, size_t n_time_steps,
    double *solution_out,
    double *delta_out,
    double *gamma_out,
    double *theta_out
);

#ifdef __cplusplus
}
#endif

#endif // MANGO_AMERICAN_OPTION_UNIFIED_H
```

**Implementation outline** (american_option_unified.c):
```c
#include "american_option_unified.h"
#include "pde_solver.h"
#include <math.h>
#include <stdlib.h>

// Helper: Transform moneyness to log-space
static double* moneyness_to_logspace(const double *m_grid, size_t n) {
    double *x_grid = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        x_grid[i] = log(m_grid[i]);
    }
    return x_grid;
}

// Black-Scholes spatial operator in log-moneyness space
static void bs_spatial_operator(const double *x, double t, const double *u,
                                 size_t n, double *Lu, void *user_data) {
    // Extract parameters from user_data
    // Implement: Lu = (σ²/2)·∂²u/∂x² + (r - q - σ²/2)·∂u/∂x - r·u
}

// Obstacle condition (early exercise boundary)
static void obstacle_condition(const double *x, double t, size_t n,
                               double *psi, void *user_data) {
    // For put: ψ(x,t) = max(K - S, 0) = max(K - K·e^x, 0)
    // For call: ψ(x,t) = max(S - K, 0) = max(K·e^x - K, 0)
}

int american_option_solve_on_grid(
    const double *moneyness_grid,
    size_t n_moneyness,
    double tau, double sigma, double r, double q,
    double strike,
    char option_type,
    double dt, size_t n_time_steps,
    double *solution_out
) {
    // 1. Transform moneyness grid to log-space
    double *x_grid = moneyness_to_logspace(moneyness_grid, n_moneyness);

    // 2. Setup PDE solver configuration
    SpatialGrid grid = {.x = x_grid, .n_points = n_moneyness, .dx = 0.0};
    grid.dx = (x_grid[n_moneyness-1] - x_grid[0]) / (n_moneyness - 1);

    TimeDomain time = {
        .t_start = 0.0,
        .t_end = tau,
        .dt = dt,
        .n_steps = n_time_steps
    };

    // 3. Setup callbacks with parameters
    struct BSParams {
        double sigma, r, q, strike;
        char option_type;
    } params = {sigma, r, q, strike, option_type};

    PDECallbacks callbacks = {
        .initial_condition = /* terminal condition at t=tau */,
        .left_boundary = /* boundary at x_min */,
        .right_boundary = /* boundary at x_max */,
        .spatial_operator = bs_spatial_operator,
        .obstacle = obstacle_condition,
        .diffusion_coeff = sigma * sigma / 2.0,
        .user_data = &params
    };

    BoundaryConfig bc = pde_default_boundary_config();
    TRBDF2Config trbdf2 = pde_default_trbdf2_config();

    // 4. Solve PDE
    PDESolver *solver = pde_solver_create(&grid, &time, &bc, &trbdf2, &callbacks);
    pde_solver_initialize(solver);
    int status = pde_solver_solve(solver);

    // 5. Copy solution to output (single memcpy)
    const double *solution = pde_solver_get_solution(solver);
    memcpy(solution_out, solution, n_moneyness * sizeof(double));

    // 6. Transform back to price space (V = u · K)
    for (size_t i = 0; i < n_moneyness; i++) {
        solution_out[i] *= strike;
    }

    pde_solver_destroy(solver);
    return status;
}

int american_option_solve_on_grid_with_greeks(
    const double *moneyness_grid,
    size_t n_moneyness,
    double tau, double sigma, double r, double q,
    double strike,
    char option_type,
    double dt, size_t n_time_steps,
    double *solution_out,
    double *delta_out,
    double *gamma_out,
    double *theta_out
) {
    // 1. Solve for prices
    int status = american_option_solve_on_grid(
        moneyness_grid, n_moneyness, tau, sigma, r, q,
        strike, option_type, dt, n_time_steps, solution_out
    );

    if (status != 0) return status;

    // 2. Compute Greeks from solution
    double *x_grid = moneyness_to_logspace(moneyness_grid, n_moneyness);
    double dx = (x_grid[n_moneyness-1] - x_grid[0]) / (n_moneyness - 1);

    // Delta: ∂V/∂S = (∂u/∂x) / S
    if (delta_out != NULL) {
        for (size_t i = 1; i < n_moneyness - 1; i++) {
            double dudx = (solution_out[i+1] - solution_out[i-1]) / (2.0 * dx * strike);
            double S = moneyness_grid[i] * strike;
            delta_out[i] = dudx / S;
        }
        // Boundary points: one-sided differences
    }

    // Gamma: ∂²V/∂S² = (∂²u/∂x² - ∂u/∂x) / S²
    if (gamma_out != NULL) {
        for (size_t i = 1; i < n_moneyness - 1; i++) {
            double d2udx2 = (solution_out[i+1] - 2*solution_out[i] + solution_out[i-1])
                          / (dx * dx * strike);
            double dudx = (solution_out[i+1] - solution_out[i-1]) / (2.0 * dx * strike);
            double S = moneyness_grid[i] * strike;
            gamma_out[i] = (d2udx2 - dudx) / (S * S);
        }
    }

    // Theta: -∂V/∂τ (requires solving at tau - dt)
    if (theta_out != NULL) {
        // Solve at tau - dt
        double *solution_prev = malloc(n_moneyness * sizeof(double));
        american_option_solve_on_grid(
            moneyness_grid, n_moneyness, tau - dt, sigma, r, q,
            strike, option_type, dt, n_time_steps, solution_prev
        );

        for (size_t i = 0; i < n_moneyness; i++) {
            theta_out[i] = -(solution_out[i] - solution_prev[i]) / dt;
        }
        free(solution_prev);
    }

    free(x_grid);
    return 0;
}
```

**BUILD.bazel update:**
```python
cc_library(
    name = "american_option_unified",
    srcs = ["src/american_option_unified.c"],
    hdrs = ["src/american_option_unified.h"],
    deps = [
        ":pde_solver",
    ],
    visibility = ["//visibility:public"],
)
```

**Verification:**
```bash
# Unit test
bazel test //tests:american_option_unified_test

# Benchmark vs old approach
bazel run //benchmarks:unified_vs_old_benchmark
```

---

### Task 1.3: Add LAYOUT_M_INNER to Price Table

**Files to modify:**
- `src/price_table.h`
- `src/price_table.c`

**Changes:**

1. **Add layout enum** (price_table.h):
   ```c
   typedef enum {
       LAYOUT_M_INNER,    ///< Moneyness innermost (required for unified grid)
       LAYOUT_TAU_INNER   ///< Maturity innermost (legacy)
   } MemoryLayout;

   typedef struct {
       // ... existing fields ...
       MemoryLayout layout;  // NEW
       // ... rest of fields ...
   } OptionPriceTable;
   ```

2. **Update index computation** (price_table.c):
   ```c
   static size_t index_4d(const OptionPriceTable *table,
                          size_t i_m, size_t i_tau, size_t i_sigma, size_t i_r) {
       if (table->layout == LAYOUT_M_INNER) {
           // moneyness innermost: [i_tau][i_sigma][i_r][i_m]
           return ((i_tau * table->n_volatility + i_sigma) * table->n_rate + i_r)
                  * table->n_moneyness + i_m;
       } else {
           // tau innermost (legacy): [i_m][i_sigma][i_r][i_tau]
           return ((i_m * table->n_volatility + i_sigma) * table->n_rate + i_r)
                  * table->n_maturity + i_tau;
       }
   }
   ```

3. **Update price_table_create()** to accept layout parameter.

**Verification:**
```bash
bazel test //tests:memory_layout_test
```

---

## Phase 2: Validation Framework

**Goal:** Implement error measurement and validation for adaptive refinement.

### Task 2.1: Implement IV Error Computation

**New file:**
- `src/validation.h`
- `src/validation.c`

**API:**
```c
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

ValidationResult validate_interpolation_error(
    const OptionPriceTable *table,
    size_t n_samples,
    double target_error
);

void validation_result_free(ValidationResult *result);
```

**Implementation steps:**
1. Generate random samples in 4D parameter space
2. For each sample:
   - Compute "true" price via FDM (fine grid)
   - Compute interpolated price from table
   - Convert both to IV using Brent's method
   - Compute IV error in bp
3. Compute statistics (max, mean, p95)
4. Identify high-error moneyness regions

**Verification:**
```bash
bazel test //tests:validation_test
```

---

### Task 2.2: Random Sampling Strategy

**Implementation** (in validation.c):
```c
// Generate stratified random samples
static void generate_validation_samples(
    const OptionPriceTable *table,
    size_t n_samples,
    double **samples_out  // [n_samples × 4]
) {
    // Use stratified sampling to ensure coverage
    // Divide each dimension into bins
    // Sample uniformly within each bin
}
```

**Verification:**
- Histogram of sample distribution
- Coverage test (all bins have samples)

---

## Phase 3: Adaptive Refinement

**Goal:** Implement grid expansion and adaptive refinement loop.

### Task 3.1: Grid Expansion

**Function:** `expand_table_grid()` (in price_table.c)

```c
void expand_table_grid(
    OptionPriceTable *table,
    const double *new_m_points,
    size_t n_new
) {
    // 1. Merge and sort new points with existing grid
    size_t n_total = table->n_moneyness + n_new;
    double *merged_m = merge_and_sort_unique(
        table->moneyness_grid, table->n_moneyness,
        new_m_points, n_new,
        &n_total  // May be less if duplicates removed
    );

    // 2. Allocate new price array
    size_t new_size = n_total * table->n_maturity *
                      table->n_volatility * table->n_rate;
    double *new_prices = malloc(new_size * sizeof(double));

    // 3. Copy existing prices to new positions
    for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
        for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
            for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                for (size_t i_m = 0; i_m < n_total; i_m++) {
                    double m = merged_m[i_m];
                    bool is_new = !binary_search(table->moneyness_grid,
                                                  table->n_moneyness, m);

                    size_t new_idx = index_4d_direct(i_m, i_tau, i_sigma, i_r,
                                                     n_total, ...);

                    if (is_new) {
                        new_prices[new_idx] = NAN;  // Mark for computation
                    } else {
                        size_t old_m_idx = find_index(table->moneyness_grid,
                                                       table->n_moneyness, m);
                        size_t old_idx = index_4d(table, old_m_idx, i_tau,
                                                   i_sigma, i_r);
                        new_prices[new_idx] = table->prices[old_idx];
                    }
                }
            }
        }
    }

    // 4. Replace table arrays
    free(table->moneyness_grid);
    free(table->prices);
    table->moneyness_grid = merged_m;
    table->n_moneyness = n_total;
    table->prices = new_prices;
}
```

**Verification:**
```bash
bazel test //tests:grid_expansion_test
```

---

### Task 3.2: Refinement Point Selection

**Function:** `identify_refinement_points()` (in validation.c)

```c
double* identify_refinement_points(
    const ValidationResult *result,
    const OptionPriceTable *table,
    size_t *n_new_out
) {
    // 1. Bin validation samples by moneyness
    // 2. Compute average error per bin
    // 3. Select bins with error > threshold
    // 4. Add midpoint of each high-error bin

    // Return sorted array of new moneyness points
}
```

---

### Task 3.3: Adaptive Precomputation

**New file:**
- `src/adaptive_refinement.h`
- `src/adaptive_refinement.c`

**API:**
```c
typedef struct {
    double target_iv_error;      ///< Target IV error (e.g., 0.0001 for 1bp)
    size_t max_iterations;        ///< Maximum refinement iterations (default: 5)
    size_t max_total_points;      ///< Memory limit on total grid points
    size_t validation_samples;    ///< Number of random validation points
    double refinement_threshold;  ///< Add points where error > threshold × target
    bool verbose;                 ///< Print progress information
} AdaptiveRefinementConfig;

AdaptiveRefinementConfig adaptive_default_config(void);

int price_table_precompute_adaptive(
    OptionPriceTable *table,
    double dt,
    size_t n_time_steps,
    const AdaptiveRefinementConfig *config
);
```

**Implementation:**
```c
int price_table_precompute_adaptive(
    OptionPriceTable *table,
    double dt,
    size_t n_time_steps,
    const AdaptiveRefinementConfig *config
) {
    // Verify layout
    if (table->layout != LAYOUT_M_INNER) {
        return PRICE_TABLE_ERROR_LAYOUT_MISMATCH;
    }

    for (size_t iter = 0; iter < config->max_iterations; iter++) {
        // 1. Precompute all (tau, sigma, r, q) with current grid
        for (size_t i_tau = 0; i_tau < table->n_maturity; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < table->n_volatility; i_sigma++) {
                for (size_t i_r = 0; i_r < table->n_rate; i_r++) {
                    size_t offset = index_4d(table, 0, i_tau, i_sigma, i_r);

                    // Only compute NaN values (new points)
                    if (isnan(table->prices[offset])) {
                        american_option_solve_on_grid(
                            table->moneyness_grid,
                            table->n_moneyness,
                            table->maturity_grid[i_tau],
                            table->volatility_grid[i_sigma],
                            table->rate_grid[i_r],
                            0.0,  // q
                            table->strike,
                            table->option_type,
                            dt, n_time_steps,
                            &table->prices[offset]
                        );
                    }
                }
            }
        }

        // 2. Validate
        ValidationResult result = validate_interpolation_error(
            table, config->validation_samples, config->target_iv_error
        );

        // 3. Check convergence
        if (result.max_iv_error < config->target_iv_error &&
            result.fraction_below_1bp > 0.95) {
            validation_result_free(&result);
            break;  // Success!
        }

        // 4. Refine grid
        size_t n_new;
        double *new_points = identify_refinement_points(&result, table, &n_new);
        expand_table_grid(table, new_points, n_new);
        free(new_points);
        validation_result_free(&result);

        if (config->verbose) {
            printf("Iteration %zu: grid size = %zu\n", iter, table->n_moneyness);
        }
    }

    return 0;
}
```

**Verification:**
```bash
bazel test //tests:adaptive_refinement_test
```

---

## Phase 4: Testing & Benchmarking

### Task 4.1: Unit Tests

**New test file:** `tests/american_option_unified_test.cc`

```cpp
TEST(UnifiedGridTest, BasicSolveCorrectness) {
    double m_grid[] = {0.8, 0.9, 1.0, 1.1, 1.2};
    double solution[5];

    int status = american_option_solve_on_grid(
        m_grid, 5,
        0.25, 0.20, 0.05, 0.0,
        100.0, 'P',
        0.001, 250,
        solution
    );

    EXPECT_EQ(status, 0);
    EXPECT_GT(solution[2], 0.0);  // ATM has value

    // Monotonicity: put prices decrease with moneyness
    for (size_t i = 1; i < 5; i++) {
        EXPECT_LT(solution[i], solution[i-1]);
    }
}

TEST(UnifiedGridTest, GreekSignCorrectness) {
    double m_grid[] = {0.8, 0.9, 1.0, 1.1, 1.2};
    double solution[5], delta[5], gamma[5], theta[5];

    american_option_solve_on_grid_with_greeks(
        m_grid, 5, 0.25, 0.20, 0.05, 0.0, 100.0, 'P',
        0.001, 250, solution, delta, gamma, theta
    );

    // Put delta: negative
    for (size_t i = 0; i < 5; i++) {
        EXPECT_LT(delta[i], 0.0);
    }

    // Gamma: positive
    for (size_t i = 0; i < 5; i++) {
        EXPECT_GT(gamma[i], 0.0);
    }
}

TEST(UnifiedGridTest, PointerSwappingEquivalence) {
    // Solve with old memcpy approach (reference)
    // Solve with new pointer swapping
    // Results should be identical
}
```

**New test file:** `tests/adaptive_refinement_test.cc`

```cpp
TEST(AdaptiveRefinementTest, ConvergenceToTarget) {
    // Start with coarse grid
    GridConfig config = grid_preset_get(
        GRID_PRESET_ADAPTIVE_FAST, ...
    );

    OptionPriceTable *table = price_table_create(...);

    AdaptiveRefinementConfig adapt_config = adaptive_default_config();
    adapt_config.target_iv_error = 0.0001;  // 1bp

    int status = price_table_precompute_adaptive(
        table, 0.001, 1000, &adapt_config
    );

    EXPECT_EQ(status, 0);

    // Validate final accuracy
    ValidationResult result = validate_interpolation_error(
        table, 1000, 0.0001
    );

    EXPECT_LT(result.max_iv_error, 0.0001);
    EXPECT_GT(result.fraction_below_1bp, 0.95);
}

TEST(GridExpansionTest, PreservesExistingPrices) {
    // Create table, precompute
    // Save old prices
    // Expand grid
    // Verify old prices unchanged at correct indices
}
```

---

### Task 4.2: Accuracy Benchmarks

**New benchmark:** `benchmarks/unified_vs_interpolation_accuracy.cc`

```cpp
static void BM_UnifiedGridAccuracy(benchmark::State& state) {
    // Compare accuracy: unified grid vs old interpolation approach

    // Reference: Very fine grid (1001 points)
    double price_ref = solve_american_fine_grid(..., 1001);

    // Method 1: Old (101-point grid + interpolate)
    double price_interp = solve_and_interpolate(..., 101);

    // Method 2: Unified (20-point grid, direct)
    double m_grid[20];
    generate_tanh_grid(m_grid, 20, 0.7, 1.3, 1.0, 3.0);
    double solution[20];
    american_option_solve_on_grid(m_grid, 20, ..., solution);

    double error_interp = fabs(price_interp - price_ref);
    double error_unified = fabs(solution[10] - price_ref);

    state.counters["error_interp"] = error_interp;
    state.counters["error_unified"] = error_unified;
}
BENCHMARK(BM_UnifiedGridAccuracy);
```

---

### Task 4.3: Performance Benchmarks

**New benchmark:** `benchmarks/adaptive_precompute_benchmark.cc`

```cpp
static void BM_AdaptivePrecompute(benchmark::State& state) {
    for (auto _ : state) {
        OptionPriceTable *table = create_adaptive_table(...);

        auto start = std::chrono::high_resolution_clock::now();
        price_table_precompute_adaptive(table, 0.001, 1000, &config);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        );

        state.counters["time_ms"] = duration.count();
        state.counters["grid_size"] = table->n_moneyness;

        price_table_destroy(table);
    }
}
BENCHMARK(BM_AdaptivePrecompute);

static void BM_UniformPrecompute(benchmark::State& state) {
    // Compare: uniform dense grid precomputation
}
BENCHMARK(BM_UniformPrecompute);
```

**Expected results:**
- Adaptive: ~900ms, 20 moneyness points
- Uniform dense: ~6000ms, 50 moneyness points
- **Speedup: 6-7×**

---

## Phase 5: Documentation & Examples

### Task 5.1: Update CLAUDE.md

**Section to add:**

````markdown
## Unified Grid Architecture (P6)

The price table module supports adaptive grid refinement using a unified grid architecture where the FDM solver's spatial grid is shared with the price table's moneyness grid.

### Typical Workflow

**1. Create table with coarse grid:**
```c
GridConfig config = grid_preset_get(
    GRID_PRESET_ADAPTIVE_FAST,  // Start coarse (~10 moneyness points)
    0.7, 1.3,      // moneyness
    0.027, 2.0,    // maturity
    0.10, 0.80,    // volatility
    0.0, 0.10,     // rate
    0.0, 0.0       // no dividend
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
```

**2. Adaptive precomputation:**
```c
AdaptiveRefinementConfig adapt_config = {
    .target_iv_error = 0.0001,      // 1bp target
    .max_iterations = 5,
    .max_total_points = 100000,
    .validation_samples = 1000,
    .refinement_threshold = 2.0,
    .verbose = true
};

int status = price_table_precompute_adaptive(
    table, 0.001, 1000, &adapt_config
);
```

**3. Query prices and Greeks:**
```c
OptionData data = price_table_query_with_greeks(
    table, 1.05, 0.25, 0.20, 0.05
);

printf("Price: %.4f\n", data.price);
printf("Delta: %.4f\n", data.delta);
printf("Gamma: %.4f\n", data.gamma);
printf("Theta: %.4f\n", data.theta);
printf("Vega:  %.4f\n", data.vega);
printf("Rho:   %.4f\n", data.rho);
```

### Performance Characteristics

**Adaptive refinement:**
- Grid size: 10 → 20 points (typical)
- Precompute time: ~900ms for 300 entries
- Accuracy: <1bp IV error for 95% of validation points
- Memory: ~480 KB (prices) + 2.4 MB (Greeks)

**Speedup vs uniform dense grid:** 6-7×
````

---

### Task 5.2: Example Program

**New file:** `examples/example_adaptive_refinement.c`

```c
#include "price_table.h"
#include "grid_presets.h"
#include "adaptive_refinement.h"
#include <stdio.h>

int main() {
    printf("Adaptive Grid Refinement Example\n");
    printf("=================================\n\n");

    // 1. Create table with coarse grid
    GridConfig config = grid_preset_get(
        GRID_PRESET_ADAPTIVE_FAST,
        0.7, 1.3, 0.027, 2.0, 0.10, 0.80, 0.0, 0.10, 0.0, 0.0
    );

    GeneratedGrids grids = grid_generate_all(&config);
    printf("Initial grid size: %zu moneyness points\n", grids.n_moneyness);

    OptionPriceTable *table = price_table_create(
        grids.moneyness, grids.n_moneyness,
        grids.maturity, grids.n_maturity,
        grids.volatility, grids.n_volatility,
        grids.rate, grids.n_rate,
        NULL, 0,
        OPTION_PUT, EXERCISE_AMERICAN
    );

    // 2. Adaptive precomputation
    AdaptiveRefinementConfig adapt_config = adaptive_default_config();
    adapt_config.target_iv_error = 0.0001;  // 1bp
    adapt_config.verbose = true;

    printf("\nStarting adaptive refinement...\n");
    int status = price_table_precompute_adaptive(table, 0.001, 1000, &adapt_config);

    if (status != 0) {
        fprintf(stderr, "Adaptive refinement failed: %d\n", status);
        return 1;
    }

    printf("\nFinal grid size: %zu moneyness points\n", table->n_moneyness);

    // 3. Validate final accuracy
    ValidationResult result = validate_interpolation_error(table, 1000, 0.0001);
    printf("\nValidation Results:\n");
    printf("  Max IV error:     %.4f bp\n", result.max_iv_error * 10000);
    printf("  Mean IV error:    %.4f bp\n", result.mean_iv_error * 10000);
    printf("  95th percentile:  %.4f bp\n", result.p95_iv_error * 10000);
    printf("  Fraction < 1bp:   %.2f%%\n", result.fraction_below_1bp * 100);

    validation_result_free(&result);

    // 4. Query sample prices
    printf("\nSample Queries:\n");
    OptionData data = price_table_query_with_greeks(table, 1.0, 0.25, 0.20, 0.05);
    printf("  ATM put (m=1.0, tau=0.25, sigma=0.20, r=0.05):\n");
    printf("    Price: $%.4f\n", data.price);
    printf("    Delta: %.4f\n", data.delta);
    printf("    Gamma: %.4f\n", data.gamma);

    // 5. Save table
    price_table_save(table, "adaptive_table.bin");
    printf("\nTable saved to adaptive_table.bin\n");

    price_table_destroy(table);
    return 0;
}
```

**BUILD.bazel:**
```python
cc_binary(
    name = "example_adaptive_refinement",
    srcs = ["examples/example_adaptive_refinement.c"],
    deps = [
        "//src:price_table",
        "//src:adaptive_refinement",
        "//src:grid_presets",
    ],
)
```

---

## Verification Checklist

After completing all phases:

- [ ] All existing tests pass
- [ ] New unit tests pass:
  - [ ] `american_option_unified_test`
  - [ ] `adaptive_refinement_test`
  - [ ] `grid_expansion_test`
  - [ ] `validation_test`
- [ ] Benchmarks run successfully:
  - [ ] `unified_vs_interpolation_accuracy`
  - [ ] `adaptive_precompute_benchmark`
- [ ] Example program runs and produces expected output
- [ ] CLAUDE.md documentation updated
- [ ] Design document matches implementation

---

## Timeline Estimate

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1 | Core Infrastructure (3 tasks) | 1-2 days |
| Phase 2 | Validation Framework (2 tasks) | 0.5-1 day |
| Phase 3 | Adaptive Refinement (3 tasks) | 1-1.5 days |
| Phase 4 | Testing & Benchmarking (3 tasks) | 0.5-1 day |
| Phase 5 | Documentation & Examples (2 tasks) | 0.5 day |
| **Total** | | **3-5 days** |

---

## Next Steps

1. Begin Phase 1, Task 1.1: Refactor PDE solver for pointer swapping
2. Run tests after each task to ensure no regressions
3. Commit after each completed task with descriptive messages
4. After all phases complete, create PR for review
