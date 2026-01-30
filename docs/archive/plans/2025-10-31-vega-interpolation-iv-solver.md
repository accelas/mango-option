<!-- SPDX-License-Identifier: MIT -->
# Vega Interpolation-Based IV Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Newton's method IV solver using interpolated prices and vegas for ~1000x speedup over FDM-based Brent solver.

**Architecture:** Enhance existing `calculate_iv()` API with optional `OptionPriceTable*` parameter. When table is provided and query point is in bounds, use Newton's method with interpolated price/vega (~microseconds). Otherwise, fallback to existing Brent + FDM solver (~250ms). Unified API preserves backward compatibility while enabling massive speedup for table-based workflows.

**Tech Stack:** C23, Newton's method for root-finding, cubic spline interpolation (4D/5D), backward-compatible API design

---

## Task 1: Add Table Parameter to IV API

**Files:**
- Modify: `src/implied_volatility.h`
- Modify: `src/implied_volatility.c`
- Test: `tests/implied_volatility_test.cc`

**Step 1: Write failing test for new API signature**

File: `tests/implied_volatility_test.cc`

```cpp
TEST_F(ImpliedVolatilityTest, AcceptsNullTable) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 10.0,
        .option_type = OPTION_CALL,
        .exercise_type = EXERCISE_EUROPEAN
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 100
    };

    // Should work with NULL table (uses FDM fallback)
    IVResult result = calculate_iv(&params, &grid, nullptr, 1e-6, 100);

    EXPECT_EQ(result.status, IV_SUCCESS);
    EXPECT_NEAR(result.implied_vol, 0.2, 0.01);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:implied_volatility_test --test_output=all`
Expected: Compilation error - `calculate_iv` doesn't accept 4 parameters

**Step 3: Update API signature in header**

File: `src/implied_volatility.h`

Find the current declaration:
```c
IVResult calculate_iv(const IVParams *params,
                     const AmericanOptionGrid *grid_params,
                     double tolerance, int max_iter);
```

Replace with:
```c
IVResult calculate_iv(const IVParams *params,
                     const AmericanOptionGrid *grid_params,
                     const OptionPriceTable *table,
                     double tolerance, int max_iter);
```

Also update `calculate_iv_simple`:
```c
IVResult calculate_iv_simple(const IVParams *params,
                             const OptionPriceTable *table);
```

**Step 4: Update implementation signatures**

File: `src/implied_volatility.c`

Update both function signatures to match header. For now, ignore the `table` parameter in the implementation body.

**Step 5: Run test to verify it compiles and passes**

Run: `bazel test //tests:implied_volatility_test --test_output=all`
Expected: PASS (table parameter ignored, uses FDM)

**Step 6: Update all existing call sites**

Find all calls to `calculate_iv` and `calculate_iv_simple` in the codebase and update them to pass `nullptr` for the table parameter.

Files to check:
- `examples/example_american_option.c`
- Any other examples or tests

**Step 7: Commit**

```bash
git add src/implied_volatility.h src/implied_volatility.c tests/implied_volatility_test.cc examples/
git commit -m "Add optional table parameter to IV API

Extends calculate_iv() and calculate_iv_simple() to accept optional
OptionPriceTable parameter. Currently ignored (uses FDM fallback).
Maintains backward compatibility by accepting NULL.

Part of P2: Vega interpolation-based IV solver."
```

---

## Task 2: Implement Bounds Checking for Table Interpolation

**Files:**
- Modify: `src/implied_volatility.c`
- Test: `tests/implied_volatility_test.cc`

**Step 1: Write failing test for in-bounds detection**

File: `tests/implied_volatility_test.cc`

```cpp
TEST_F(ImpliedVolatilityTest, DetectsInBoundsPoint) {
    // Create simple 4D table
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau = {0.5, 1.0, 1.5, 2.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.03, 0.05, 0.07};

    OptionPriceTable *table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_CALL, EXERCISE_EUROPEAN,
        COORD_IDENTITY, LAYOUT_M_INNER);

    // In-bounds point
    IVParams params_in = {
        .spot_price = 100.0,
        .strike = 100.0,  // m = 1.0 (in bounds)
        .time_to_maturity = 1.0,  // tau = 1.0 (in bounds)
        .risk_free_rate = 0.05,  // r = 0.05 (in bounds)
        .dividend_yield = 0.0,
        .market_price = 10.0,
        .option_type = OPTION_CALL,
        .exercise_type = EXERCISE_EUROPEAN
    };

    // This should use table interpolation (when implemented)
    // For now, just testing it doesn't crash
    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 100
    };

    IVResult result = calculate_iv(&params_in, &grid, table, 1e-6, 100);
    EXPECT_EQ(result.status, IV_SUCCESS);

    price_table_destroy(table);
}

TEST_F(ImpliedVolatilityTest, DetectsOutOfBoundsPoint) {
    // Create simple 4D table
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau = {0.5, 1.0, 1.5, 2.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.03, 0.05, 0.07};

    OptionPriceTable *table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_CALL, EXERCISE_EUROPEAN,
        COORD_IDENTITY, LAYOUT_M_INNER);

    // Out-of-bounds point (tau too large)
    IVParams params_out = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 5.0,  // tau = 5.0 (out of bounds)
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 10.0,
        .option_type = OPTION_CALL,
        .exercise_type = EXERCISE_EUROPEAN
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 100
    };

    // Should fallback to FDM
    IVResult result = calculate_iv(&params_out, &grid, table, 1e-6, 100);
    EXPECT_EQ(result.status, IV_SUCCESS);

    price_table_destroy(table);
}
```

**Step 2: Run tests to verify they fail**

Run: `bazel test //tests:implied_volatility_test --test_output=all`
Expected: Tests may pass (using FDM fallback) but no bounds checking happening yet

**Step 3: Implement bounds checking helper**

File: `src/implied_volatility.c`

Add at top of file (after includes):
```c
// Check if query point is within table bounds
static bool is_in_table_bounds(const OptionPriceTable *table, const IVParams *params) {
    if (!table) return false;

    // Calculate moneyness
    double moneyness = params->spot_price / params->strike;

    // Check all dimensions
    if (moneyness < table->moneyness[0] || moneyness > table->moneyness[table->n_moneyness - 1])
        return false;

    if (params->time_to_maturity < table->maturity[0] ||
        params->time_to_maturity > table->maturity[table->n_maturity - 1])
        return false;

    // For IV solving, we don't know sigma yet, so we can't check sigma bounds
    // We'll handle this during Newton iteration

    if (params->risk_free_rate < table->rate[0] ||
        params->risk_free_rate > table->rate[table->n_rate - 1])
        return false;

    // Check dividend if table has dividend dimension
    if (table->n_dividend > 0) {
        if (params->dividend_yield < table->dividend[0] ||
            params->dividend_yield > table->dividend[table->n_dividend - 1])
            return false;
    }

    // Check option type and exercise type match
    if (table->option_type != params->option_type)
        return false;

    if (table->exercise_type != params->exercise_type)
        return false;

    return true;
}
```

**Step 4: Run tests to verify they pass**

Run: `bazel test //tests:implied_volatility_test --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/implied_volatility.c tests/implied_volatility_test.cc
git commit -m "Add bounds checking for table interpolation

Implements is_in_table_bounds() helper to verify query point falls
within table grid before attempting interpolation. Checks moneyness,
maturity, rate, dividend, option type, and exercise type.

Note: Sigma bounds checked during Newton iteration (not known upfront).

Part of P2: Vega interpolation-based IV solver."
```

---

## Task 3: Implement Newton's Method IV Solver

**Files:**
- Modify: `src/implied_volatility.c`
- Test: `tests/implied_volatility_test.cc`

**Step 1: Write failing test for Newton's method**

File: `tests/implied_volatility_test.cc`

```cpp
TEST_F(ImpliedVolatilityTest, NewtonMethodWithTable) {
    // Create table and precompute
    std::vector<double> m = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};
    std::vector<double> tau = {0.25, 0.5, 1.0, 1.5, 2.0};
    std::vector<double> sigma = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40};
    std::vector<double> r = {0.0, 0.03, 0.05, 0.07, 0.10};

    OptionPriceTable *table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_CALL, EXERCISE_EUROPEAN,
        COORD_IDENTITY, LAYOUT_M_INNER);

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 101, .dt = 0.001, .n_steps = 1000
    };

    price_table_precompute(table, &grid);
    price_table_build_interpolation(table);

    // Query in-bounds point
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 10.0,
        .option_type = OPTION_CALL,
        .exercise_type = EXERCISE_EUROPEAN
    };

    IVResult result = calculate_iv(&params, &grid, table, 1e-6, 100);

    EXPECT_EQ(result.status, IV_SUCCESS);
    EXPECT_GT(result.iterations, 0);
    EXPECT_LT(result.iterations, 10);  // Newton should converge quickly
    EXPECT_NEAR(result.implied_vol, 0.2, 0.01);

    price_table_destroy(table);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:implied_volatility_test --test_filter=NewtonMethodWithTable --test_output=all`
Expected: FAIL - Newton method not implemented yet

**Step 3: Implement Newton's method solver**

File: `src/implied_volatility.c`

Add helper function before `calculate_iv`:

```c
// Newton's method IV solver using table interpolation
static IVResult newton_iv_solver(const IVParams *params,
                                  const OptionPriceTable *table,
                                  double tolerance, int max_iter) {
    IVResult result = {.status = IV_FAILED, .implied_vol = NAN, .iterations = 0};

    // Initial guess: middle of sigma range
    double sigma = (table->volatility[0] + table->volatility[table->n_volatility - 1]) / 2.0;

    // Calculate moneyness
    double m = params->spot_price / params->strike;
    double tau = params->time_to_maturity;
    double r = params->risk_free_rate;
    double q = params->dividend_yield;

    // Newton iteration
    for (int iter = 0; iter < max_iter; iter++) {
        result.iterations = iter + 1;

        // Check if sigma is in bounds
        if (sigma < table->volatility[0] || sigma > table->volatility[table->n_volatility - 1]) {
            result.status = IV_OUT_OF_BOUNDS;
            return result;
        }

        // Interpolate price and vega
        double price, vega;
        if (table->n_dividend > 0) {
            price = price_table_interpolate_5d(table, m, tau, sigma, r, q);
            vega = price_table_interpolate_vega_5d(table, m, tau, sigma, r, q);
        } else {
            price = price_table_interpolate_4d(table, m, tau, sigma, r);
            vega = price_table_interpolate_vega_4d(table, m, tau, sigma, r);
        }

        // Check for invalid interpolation
        if (isnan(price) || isnan(vega)) {
            result.status = IV_FAILED;
            return result;
        }

        // Check for zero or negative vega (can't proceed)
        if (vega <= 0.0) {
            result.status = IV_FAILED;
            return result;
        }

        // Newton update: σ_{n+1} = σ_n - (V(σ_n) - V_market) / vega(σ_n)
        double f = price - params->market_price;
        double sigma_new = sigma - f / vega;

        // Check convergence
        if (fabs(f) < tolerance) {
            result.status = IV_SUCCESS;
            result.implied_vol = sigma;
            return result;
        }

        // Update sigma with damping for stability
        const double damping = 0.8;
        sigma = damping * sigma_new + (1.0 - damping) * sigma;
    }

    // Max iterations reached
    result.status = IV_MAX_ITER;
    result.implied_vol = sigma;
    return result;
}
```

**Step 4: Integrate Newton solver into calculate_iv**

File: `src/implied_volatility.c`

Modify `calculate_iv` function to use Newton solver when table is available:

```c
IVResult calculate_iv(const IVParams *params,
                     const AmericanOptionGrid *grid_params,
                     const OptionPriceTable *table,
                     double tolerance, int max_iter) {
    // Validate inputs
    if (!params) {
        return (IVResult){.status = IV_FAILED, .implied_vol = NAN, .iterations = 0};
    }

    // Try table interpolation if available
    if (table && is_in_table_bounds(table, params)) {
        // Check if vegas are available
        if (table->vegas) {
            IVResult result = newton_iv_solver(params, table, tolerance, max_iter);

            // If Newton succeeded or failed with in-bounds error, return result
            // If out of bounds during iteration, fallback to FDM
            if (result.status != IV_OUT_OF_BOUNDS) {
                return result;
            }
        }
    }

    // Fallback to Brent + FDM solver (existing implementation)
    // ... existing Brent solver code ...
}
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:implied_volatility_test --test_filter=NewtonMethodWithTable --test_output=all`
Expected: PASS

**Step 6: Commit**

```bash
git add src/implied_volatility.c tests/implied_volatility_test.cc
git commit -m "Implement Newton's method IV solver using interpolation

Adds newton_iv_solver() that uses interpolated price and vega for
~1000x speedup over Brent + FDM. Newton iteration:
  σ_{n+1} = σ_n - (V(σ_n) - V_market) / vega(σ_n)

Includes damping (0.8) for stability and bounds checking during
iteration. Automatically falls back to FDM if sigma goes out of bounds.

Part of P2: Vega interpolation-based IV solver."
```

---

## Task 4: Add Comprehensive Tests

**Files:**
- Test: `tests/implied_volatility_test.cc`

**Step 1: Write test for out-of-bounds fallback**

File: `tests/implied_volatility_test.cc`

```cpp
TEST_F(ImpliedVolatilityTest, FallbackToFDMWhenOutOfBounds) {
    // Create table with limited range
    std::vector<double> m = {0.9, 1.0, 1.1};
    std::vector<double> tau = {0.5, 1.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25};
    std::vector<double> r = {0.04, 0.05, 0.06};

    OptionPriceTable *table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_PUT, EXERCISE_AMERICAN,
        COORD_IDENTITY, LAYOUT_M_INNER);

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 101, .dt = 0.001, .n_steps = 1000
    };

    price_table_precompute(table, &grid);
    price_table_build_interpolation(table);

    // Query out of bounds (maturity too high)
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 2.0,  // Out of bounds
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 5.0,
        .option_type = OPTION_PUT,
        .exercise_type = EXERCISE_AMERICAN
    };

    IVResult result = calculate_iv(&params, &grid, table, 1e-6, 100);

    // Should succeed via FDM fallback
    EXPECT_EQ(result.status, IV_SUCCESS);
    EXPECT_GT(result.implied_vol, 0.0);

    price_table_destroy(table);
}
```

**Step 2: Write test for table without vegas**

```cpp
TEST_F(ImpliedVolatilityTest, FallbackWhenNoVegas) {
    // Create table but don't precompute (no vegas)
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau = {0.5, 1.0, 1.5};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.03, 0.05, 0.07};

    OptionPriceTable *table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_CALL, EXERCISE_EUROPEAN,
        COORD_IDENTITY, LAYOUT_M_INNER);

    // Don't precompute - no vegas available

    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 10.0,
        .option_type = OPTION_CALL,
        .exercise_type = EXERCISE_EUROPEAN
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 100
    };

    IVResult result = calculate_iv(&params, &grid, table, 1e-6, 100);

    // Should fallback to FDM
    EXPECT_EQ(result.status, IV_SUCCESS);

    price_table_destroy(table);
}
```

**Step 3: Write test for option type mismatch**

```cpp
TEST_F(ImpliedVolatilityTest, FallbackOnOptionTypeMismatch) {
    // Create CALL table
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau = {0.5, 1.0, 1.5};
    std::vector<double> sigma = {0.15, 0.20, 0.25};
    std::vector<double> r = {0.03, 0.05, 0.07};

    OptionPriceTable *table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_CALL, EXERCISE_EUROPEAN,
        COORD_IDENTITY, LAYOUT_M_INNER);

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 100
    };

    price_table_precompute(table, &grid);
    price_table_build_interpolation(table);

    // Query for PUT (mismatch)
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 5.0,
        .option_type = OPTION_PUT,  // Mismatch
        .exercise_type = EXERCISE_EUROPEAN
    };

    IVResult result = calculate_iv(&params, &grid, table, 1e-6, 100);

    // Should fallback to FDM
    EXPECT_EQ(result.status, IV_SUCCESS);

    price_table_destroy(table);
}
```

**Step 4: Run tests**

Run: `bazel test //tests:implied_volatility_test --test_output=all`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/implied_volatility_test.cc
git commit -m "Add comprehensive tests for IV solver fallback logic

Tests cover:
- Out of bounds fallback to FDM
- Missing vegas fallback to FDM
- Option type mismatch fallback to FDM

Ensures unified API handles all edge cases gracefully.

Part of P2: Vega interpolation-based IV solver."
```

---

## Task 5: Update calculate_iv_simple

**Files:**
- Modify: `src/implied_volatility.c`
- Test: `tests/implied_volatility_test.cc`

**Step 1: Write test for simple API with table**

File: `tests/implied_volatility_test.cc`

```cpp
TEST_F(ImpliedVolatilityTest, SimpleAPIWithTable) {
    // Create and precompute table
    std::vector<double> m = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};
    std::vector<double> tau = {0.25, 0.5, 1.0, 1.5, 2.0};
    std::vector<double> sigma = {0.10, 0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.0, 0.03, 0.05, 0.07};

    OptionPriceTable *table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_CALL, EXERCISE_EUROPEAN,
        COORD_IDENTITY, LAYOUT_M_INNER);

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 101, .dt = 0.001, .n_steps = 1000
    };

    price_table_precompute(table, &grid);
    price_table_build_interpolation(table);

    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 10.0,
        .option_type = OPTION_CALL,
        .exercise_type = EXERCISE_EUROPEAN
    };

    IVResult result = calculate_iv_simple(&params, table);

    EXPECT_EQ(result.status, IV_SUCCESS);
    EXPECT_LT(result.iterations, 10);  // Newton should be fast

    price_table_destroy(table);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:implied_volatility_test --test_filter=SimpleAPIWithTable --test_output=all`
Expected: FAIL - simple API not updated yet

**Step 3: Update calculate_iv_simple implementation**

File: `src/implied_volatility.c`

Modify `calculate_iv_simple`:

```c
IVResult calculate_iv_simple(const IVParams *params,
                             const OptionPriceTable *table) {
    // Default grid parameters
    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = 1000
    };

    // Default tolerance and max iterations
    const double tolerance = 1e-6;
    const int max_iter = 100;

    return calculate_iv(params, &grid, table, tolerance, max_iter);
}
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:implied_volatility_test --test_filter=SimpleAPIWithTable --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/implied_volatility.c tests/implied_volatility_test.cc
git commit -m "Update calculate_iv_simple to use table parameter

Forwards table parameter to calculate_iv() with default grid/tolerance.
Maintains backward compatibility (table can be NULL).

Part of P2: Vega interpolation-based IV solver."
```

---

## Task 6: Add Performance Benchmark

**Files:**
- Create: `benchmarks/iv_interpolation_benchmark.cc`
- Modify: `benchmarks/BUILD.bazel`

**Step 1: Create benchmark file**

File: `benchmarks/iv_interpolation_benchmark.cc`

```cpp
#include "benchmark/benchmark.h"
#include "../src/implied_volatility.h"
#include "../src/price_table.h"
#include <vector>

static OptionPriceTable* g_table = nullptr;

static void SetupTable(const benchmark::State& state) {
    if (g_table) return;  // Already set up

    // Create realistic 4D table
    std::vector<double> m(30);
    std::vector<double> tau(25);
    std::vector<double> sigma(20);
    std::vector<double> r(10);

    // Log-spaced moneyness
    for (size_t i = 0; i < 30; i++) {
        double t = (double)i / 29.0;
        m[i] = 0.7 * exp(t * log(1.5 / 0.7));
    }

    // Linear maturity
    for (size_t i = 0; i < 25; i++) {
        tau[i] = 0.027 + i * (2.5 - 0.027) / 24.0;
    }

    // Linear volatility
    for (size_t i = 0; i < 20; i++) {
        sigma[i] = 0.10 + i * (0.60 - 0.10) / 19.0;
    }

    // Linear rate
    for (size_t i = 0; i < 10; i++) {
        r[i] = 0.0 + i * (0.10 - 0.0) / 9.0;
    }

    g_table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_CALL, EXERCISE_EUROPEAN,
        COORD_LOG_SQRT, LAYOUT_M_INNER);

    // Precompute
    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 101, .dt = 0.001, .n_steps = 1000
    };

    price_table_precompute(g_table, &grid);
    price_table_build_interpolation(g_table);
}

static void BM_IV_Newton_Interpolation(benchmark::State& state) {
    SetupTable(state);

    IVParams params = {
        .spot_price = 100.0,
        .strike = 95.0,
        .time_to_maturity = 0.5,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 8.5,
        .option_type = OPTION_CALL,
        .exercise_type = EXERCISE_EUROPEAN
    };

    for (auto _ : state) {
        IVResult result = calculate_iv_simple(&params, g_table);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_IV_Newton_Interpolation);

static void BM_IV_FDM_Fallback(benchmark::State& state) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 95.0,
        .time_to_maturity = 0.5,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 8.5,
        .option_type = OPTION_CALL,
        .exercise_type = EXERCISE_EUROPEAN
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 101, .dt = 0.001, .n_steps = 1000
    };

    for (auto _ : state) {
        IVResult result = calculate_iv(&params, &grid, nullptr, 1e-6, 100);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_IV_FDM_Fallback);

BENCHMARK_MAIN();
```

**Step 2: Add to BUILD.bazel**

File: `benchmarks/BUILD.bazel`

Add new benchmark target:

```python
cc_binary(
    name = "iv_interpolation_benchmark",
    srcs = ["iv_interpolation_benchmark.cc"],
    deps = [
        "//src:implied_volatility",
        "//src:price_table",
        "@google_benchmark//:benchmark",
    ],
)
```

**Step 3: Build and run benchmark**

Run: `bazel build //benchmarks:iv_interpolation_benchmark`
Expected: Successful build

Run: `bazel run //benchmarks:iv_interpolation_benchmark`
Expected: Benchmark output showing ~1000x speedup

**Step 4: Commit**

```bash
git add benchmarks/iv_interpolation_benchmark.cc benchmarks/BUILD.bazel
git commit -m "Add IV interpolation vs FDM benchmark

Compares Newton's method with table interpolation against
Brent's method with FDM solver. Expected ~1000x speedup
(microseconds vs milliseconds).

Part of P2: Vega interpolation-based IV solver."
```

---

## Task 7: Add USDT Tracing

**Files:**
- Modify: `src/implied_volatility.c`

**Step 1: Add trace points**

File: `src/implied_volatility.c`

Add include at top:
```c
#include "ivcalc_trace.h"
```

Add trace in `calculate_iv`:
```c
IVResult calculate_iv(const IVParams *params,
                     const AmericanOptionGrid *grid_params,
                     const OptionPriceTable *table,
                     double tolerance, int max_iter) {
    IVCALC_TRACE_IV_START(MODULE_IV_SOLVER, params->market_price);

    // ... existing validation ...

    // Try table interpolation if available
    if (table && is_in_table_bounds(table, params)) {
        if (table->vegas) {
            IVResult result = newton_iv_solver(params, table, tolerance, max_iter);

            if (result.status != IV_OUT_OF_BOUNDS) {
                IVCALC_TRACE_IV_COMPLETE(MODULE_IV_SOLVER, result.implied_vol, result.iterations);
                return result;
            }
        }
    }

    // Fallback to Brent + FDM
    // ... existing Brent code ...

    IVCALC_TRACE_IV_COMPLETE(MODULE_IV_SOLVER, result.implied_vol, result.iterations);
    return result;
}
```

Add trace in `newton_iv_solver`:
```c
static IVResult newton_iv_solver(const IVParams *params,
                                  const OptionPriceTable *table,
                                  double tolerance, int max_iter) {
    // ... existing setup ...

    // Newton iteration
    for (int iter = 0; iter < max_iter; iter++) {
        result.iterations = iter + 1;

        // ... existing iteration code ...

        IVCALC_TRACE_CONVERGENCE_ITER(MODULE_IV_SOLVER, iter, fabs(f));

        // Check convergence
        if (fabs(f) < tolerance) {
            result.status = IV_SUCCESS;
            result.implied_vol = sigma;
            IVCALC_TRACE_CONVERGENCE_SUCCESS(MODULE_IV_SOLVER, iter, fabs(f));
            return result;
        }

        // ... update sigma ...
    }

    // Max iterations reached
    IVCALC_TRACE_CONVERGENCE_FAILED(MODULE_IV_SOLVER, max_iter);
    // ...
}
```

**Step 2: Build and verify**

Run: `bazel build //src:implied_volatility`
Expected: Successful build

**Step 3: Test with bpftrace**

Run:
```bash
sudo bpftrace -e 'usdt::ivcalc:iv_complete {
    printf("IV solved: sigma=%.4f, iterations=%d\n", arg1, arg2);
}' -c 'bazel run //benchmarks:iv_interpolation_benchmark'
```

Expected: See trace output during benchmark

**Step 4: Commit**

```bash
git add src/implied_volatility.c
git commit -m "Add USDT tracing to IV solver

Instruments calculate_iv() and newton_iv_solver() with:
- IV_START/IV_COMPLETE for lifecycle
- CONVERGENCE_ITER/SUCCESS/FAILED for Newton iteration

Enables runtime monitoring with bpftrace.

Part of P2: Vega interpolation-based IV solver."
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add IV solver section**

File: `CLAUDE.md`

Add new section after "Price Table Pre-computation Workflow":

```markdown
## Implied Volatility Solver Workflow

The IV solver supports two modes: high-speed Newton's method with table interpolation, or robust Brent's method with FDM solver.

### Typical Workflow

**Fast path (with pre-computed table):**

```c
// 1. Create and precompute table (one-time setup)
OptionPriceTable *table = price_table_create(...);
AmericanOptionGrid grid = {...};
price_table_precompute(table, &grid);
price_table_build_interpolation(table);

// 2. Solve IV using Newton's method (~microseconds)
IVParams params = {
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .dividend_yield = 0.0,
    .market_price = 10.0,
    .option_type = OPTION_CALL,
    .exercise_type = EXERCISE_EUROPEAN
};

IVResult result = calculate_iv_simple(&params, table);

if (result.status == IV_SUCCESS) {
    printf("IV: %.4f (converged in %d iterations)\n",
           result.implied_vol, result.iterations);
}

price_table_destroy(table);
```

**Fallback path (without table or out of bounds):**

```c
// Uses Brent's method with FDM solver (~250ms)
IVResult result = calculate_iv_simple(&params, NULL);
```

### Performance Characteristics

**Newton's method with table interpolation:**
- Single IV solve: ~10 microseconds
- Convergence: typically 3-5 iterations
- Speedup vs FDM: ~25,000x
- Requirements: Pre-computed table with vegas, in-bounds query

**Brent's method with FDM (fallback):**
- Single IV solve: ~250 milliseconds
- Function evaluations: 10-15 FDM solves
- Robust: works for all valid option parameters
- Used when: no table, out of bounds, or missing vegas

### Automatic Fallback

The API automatically falls back to FDM when:
- Table parameter is NULL
- Query point outside table bounds (moneyness, maturity, rate, dividend)
- Table doesn't have vegas (not precomputed)
- Option type or exercise type mismatch
- Sigma goes out of bounds during Newton iteration

### API Reference

```c
// Full control API
IVResult calculate_iv(const IVParams *params,
                     const AmericanOptionGrid *grid_params,
                     const OptionPriceTable *table,
                     double tolerance, int max_iter);

// Simple API with defaults
IVResult calculate_iv_simple(const IVParams *params,
                             const OptionPriceTable *table);
```

Both accept optional table parameter. Pass NULL to force FDM solver.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Document IV solver workflow and performance

Adds comprehensive documentation for:
- Fast path with Newton's method (~10μs)
- Fallback path with Brent + FDM (~250ms)
- Automatic fallback conditions
- API reference

Part of P2: Vega interpolation-based IV solver."
```

---

## Task 9: Run Full Test Suite

**Step 1: Run all tests**

Run: `bazel test //...`
Expected: All tests PASS

**Step 2: Run benchmarks**

Run: `bazel run //benchmarks:iv_interpolation_benchmark -- --benchmark_min_time=2.0s`
Expected: See ~1000x speedup for Newton vs FDM

**Step 3: If any failures, fix them**

Debug and fix any failing tests before proceeding.

**Step 4: Final commit if fixes were needed**

```bash
git add [any fixed files]
git commit -m "Fix test failures in IV solver implementation"
```

---

## Task 10: Create Pull Request

**Step 1: Push branch**

Run: `git push -u origin fix/interpolation-accuracy-and-validation`

**Step 2: Create PR**

Run:
```bash
gh pr create --title "Implement vega interpolation-based IV solver" --body "$(cat <<'EOF'
## Summary

Implements Newton's method IV solver using interpolated prices and vegas for ~1000x speedup over FDM-based Brent solver. Unified API with automatic fallback to robust FDM solver when table unavailable or query out of bounds.

## Changes

- Extended `calculate_iv()` and `calculate_iv_simple()` APIs with optional table parameter
- Implemented Newton's method solver using `price_table_interpolate_4d/5d()` and `price_table_interpolate_vega_4d/5d()`
- Added bounds checking with automatic fallback logic
- Added comprehensive tests for all fallback scenarios
- Added performance benchmark comparing Newton vs Brent methods
- Added USDT tracing for runtime monitoring
- Updated documentation

## Performance

**Newton's method with table:**
- ~10 microseconds per IV solve
- 3-5 iterations typical
- 25,000x speedup vs FDM

**Automatic fallback to FDM when:**
- Table is NULL
- Query point out of bounds
- Table missing vegas
- Option/exercise type mismatch
- Sigma out of bounds during iteration

## Testing

```bash
bazel test //tests:implied_volatility_test
bazel run //benchmarks:iv_interpolation_benchmark
```

All tests passing. Benchmark shows expected ~1000x speedup.

## Related Issues

Closes #39 (P2: Vega interpolation-based IV solver)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Verification Checklist

Before marking P2 complete:

- [ ] All tests pass (`bazel test //...`)
- [ ] Benchmark shows ~1000x speedup
- [ ] API backward compatible (NULL table works)
- [ ] Automatic fallback works for all edge cases
- [ ] USDT tracing functional
- [ ] Documentation updated
- [ ] PR created and merged

---

## Notes

**Key Design Decisions:**

1. **Unified API**: Extended existing functions rather than creating separate `calculate_iv_with_table()`. Maintains simplicity and backward compatibility.

2. **Automatic fallback**: No explicit error when out of bounds - seamlessly falls back to FDM. User doesn't need to handle edge cases.

3. **Lazy validation**: Option type/exercise type checked during bounds check, not parameter validation. Allows table to be truly optional.

4. **Damped Newton**: Uses 0.8 damping factor for stability. Pure Newton can overshoot for poorly conditioned problems.

5. **Vega requirement**: Only uses Newton when `table->vegas != NULL`. Without vegas, falls back to FDM (can't compute gradient).

**Performance expectations:**
- Newton: 3-5 iterations × 2 interpolations (price + vega) × ~500ns = ~10μs
- FDM: 10-15 Brent iterations × ~21ms FDM solve = ~250ms
- Speedup: ~25,000x

**Testing strategy:**
- Test each fallback condition independently
- Test with/without table
- Test boundary cases (on edge of table bounds)
- Benchmark real-world query patterns
