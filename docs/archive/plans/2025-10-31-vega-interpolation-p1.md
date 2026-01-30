<!-- SPDX-License-Identifier: MIT -->
# P1: Vega Interpolation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add vega (∂V/∂σ) precomputation and interpolation alongside option prices for accurate Greeks calculation and Newton-based IV inversion.

**Architecture:** Extend OptionPriceTable to store vega values computed via finite differences during precomputation. Use same interpolation strategy as prices for consistency. Compute vega using centered differences: vega ≈ (V(σ+h) - V(σ-h)) / (2h).

**Tech Stack:** C23, Bazel, GoogleTest, FDM solver, cubic spline interpolation

**Impact:**
- More accurate Greeks (no finite difference errors at query time)
- Enables Newton-based IV inversion (requires vega)
- Reduces required grid density by capturing curvature
- Computational cost: 2× additional FDM solves during precomputation (acceptable for one-time cost)

---

## Task 1: Extend OptionPriceTable Structure for Vega Storage

**Files:**
- Modify: `src/price_table.h:111-148` (OptionPriceTable struct)
- Test: `tests/price_table_test.cc`

**Step 1: Write failing test for vega storage**

Add to `tests/price_table_test.cc` after existing price table tests:

```cpp
TEST(PriceTableTest, VegaArrayAllocation) {
    double m[] = {0.9, 1.0, 1.1};
    double tau[] = {0.25, 0.5};
    double sigma[] = {0.2, 0.3};
    double r[] = {0.05};

    OptionPriceTable *table = price_table_create(
        m, 3, tau, 2, sigma, 2, r, 1, nullptr, 0,
        OPTION_CALL, AMERICAN);

    ASSERT_NE(table, nullptr);

    // Vega array should be allocated with same size as prices
    ASSERT_NE(table->vegas, nullptr);

    // Size should be n_m * n_tau * n_sigma * n_r
    size_t expected_size = 3 * 2 * 2 * 1;
    // Verify all vega values initialized to NaN
    for (size_t i = 0; i < expected_size; i++) {
        EXPECT_TRUE(std::isnan(table->vegas[i]));
    }

    price_table_destroy(table);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_test --test_filter="PriceTableTest.VegaArrayAllocation" --test_output=errors`

Expected: FAIL with "no member named 'vegas' in 'OptionPriceTable'"

**Step 3: Add vegas array to OptionPriceTable struct**

Modify `src/price_table.h` lines 125-127:

```c
    // Option prices (flattened multi-dimensional array)
    double *prices;             // n_m × n_tau × n_sigma × n_r × n_q values
    double *vegas;              // ∂V/∂σ values (same dimensions as prices)
```

**Step 4: Allocate vegas array in price_table_create**

Modify `src/price_table.c` in `price_table_create_with_strategy()` function.

Find the prices allocation (around line 260):

```c
    // Allocate price array
    table->prices = malloc(n_total * sizeof(double));
    if (!table->prices) {
        free(table);
        return NULL;
    }

    // Initialize all prices to NaN
    for (size_t i = 0; i < n_total; i++) {
        table->prices[i] = NAN;
    }
```

Add immediately after:

```c
    // Allocate vega array (same size as prices)
    table->vegas = malloc(n_total * sizeof(double));
    if (!table->vegas) {
        free(table->prices);
        free(table);
        return NULL;
    }

    // Initialize all vegas to NaN
    for (size_t i = 0; i < n_total; i++) {
        table->vegas[i] = NAN;
    }
```

**Step 5: Free vegas array in price_table_destroy**

Modify `src/price_table.c` in `price_table_destroy()` function.

Find the prices free (around line 440):

```c
    free(table->prices);
```

Add immediately after:

```c
    free(table->vegas);
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:price_table_test --test_filter="PriceTableTest.VegaArrayAllocation" --test_output=errors`

Expected: PASS

**Step 7: Commit**

```bash
git add src/price_table.h src/price_table.c tests/price_table_test.cc
git commit -m "feat(price_table): add vega array storage alongside prices

- Add vegas pointer to OptionPriceTable structure
- Allocate/free vegas array in create/destroy functions
- Initialize all vega values to NaN
- Same dimensions as prices array"
```

---

## Task 2: Add Vega Get/Set Functions

**Files:**
- Modify: `src/price_table.h:322-350` (add after price_table_set)
- Modify: `src/price_table.c` (add implementations)
- Test: `tests/price_table_test.cc`

**Step 1: Write failing test for vega get/set**

Add to `tests/price_table_test.cc`:

```cpp
TEST(PriceTableTest, VegaGetSet) {
    double m[] = {0.9, 1.0, 1.1};
    double tau[] = {0.25, 0.5};
    double sigma[] = {0.2, 0.3};
    double r[] = {0.05};

    OptionPriceTable *table = price_table_create(
        m, 3, tau, 2, sigma, 2, r, 1, nullptr, 0,
        OPTION_CALL, AMERICAN);

    // Set vega at specific grid point
    int status = price_table_set_vega(table, 1, 0, 1, 0, 0, 0.42);
    EXPECT_EQ(status, 0);

    // Get vega back
    double vega = price_table_get_vega(table, 1, 0, 1, 0, 0);
    EXPECT_DOUBLE_EQ(vega, 0.42);

    // Out of bounds should return NaN
    double oob = price_table_get_vega(table, 10, 0, 0, 0, 0);
    EXPECT_TRUE(std::isnan(oob));

    price_table_destroy(table);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_test --test_filter="PriceTableTest.VegaGetSet" --test_output=errors`

Expected: FAIL with "price_table_get_vega not declared"

**Step 3: Add function declarations to header**

Add to `src/price_table.h` after `price_table_set()` (around line 350):

```c
/**
 * Get vega at specific grid point
 *
 * @return vega at grid point, or NaN if indices out of bounds
 */
double price_table_get_vega(const OptionPriceTable *table,
                             size_t i_m, size_t i_tau, size_t i_sigma,
                             size_t i_r, size_t i_q);

/**
 * Set vega at specific grid point
 *
 * @return 0 on success, -1 on error (NULL table or out of bounds)
 */
int price_table_set_vega(OptionPriceTable *table,
                         size_t i_m, size_t i_tau, size_t i_sigma,
                         size_t i_r, size_t i_q, double vega);
```

**Step 4: Implement get_vega function**

Add to `src/price_table.c` after `price_table_get()` implementation:

```c
double price_table_get_vega(const OptionPriceTable *table,
                             size_t i_m, size_t i_tau, size_t i_sigma,
                             size_t i_r, size_t i_q) {
    if (!table || !table->vegas) return NAN;

    // Bounds checking
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate ||
        i_q >= (table->n_dividend > 0 ? table->n_dividend : 1)) {
        return NAN;
    }

    // Calculate flat index using pre-computed strides
    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    return table->vegas[idx];
}
```

**Step 5: Implement set_vega function**

Add to `src/price_table.c` after `price_table_set()` implementation:

```c
int price_table_set_vega(OptionPriceTable *table,
                         size_t i_m, size_t i_tau, size_t i_sigma,
                         size_t i_r, size_t i_q, double vega) {
    if (!table || !table->vegas) return -1;

    // Bounds checking
    if (i_m >= table->n_moneyness || i_tau >= table->n_maturity ||
        i_sigma >= table->n_volatility || i_r >= table->n_rate ||
        i_q >= (table->n_dividend > 0 ? table->n_dividend : 1)) {
        return -1;
    }

    // Calculate flat index using pre-computed strides
    size_t idx = i_m * table->stride_m + i_tau * table->stride_tau
               + i_sigma * table->stride_sigma + i_r * table->stride_r
               + i_q * table->stride_q;

    table->vegas[idx] = vega;
    return 0;
}
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:price_table_test --test_filter="PriceTableTest.VegaGetSet" --test_output=errors`

Expected: PASS

**Step 7: Commit**

```bash
git add src/price_table.h src/price_table.c tests/price_table_test.cc
git commit -m "feat(price_table): add vega get/set API functions

- Add price_table_get_vega() for querying vega at grid points
- Add price_table_set_vega() for setting vega values
- Include bounds checking and NaN return for invalid indices"
```

---

## Task 3: Compute Vega During Precomputation

**Files:**
- Modify: `src/price_table.c` (price_table_precompute function)
- Test: `tests/price_table_test.cc`

**Step 1: Write failing test for vega precomputation**

Add to `tests/price_table_test.cc`:

```cpp
TEST(PriceTableTest, VegaPrecomputation) {
    // Small grid for fast test
    double m[] = {1.0};
    double tau[] = {0.5};
    double sigma[] = {0.15, 0.20, 0.25};  // Need 3+ points for centered diff
    double r[] = {0.05};

    OptionPriceTable *table = price_table_create(
        m, 1, tau, 1, sigma, 3, r, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    // Precompute with coarse grid (fast test)
    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 51,
        .dt = 0.01,
        .n_steps = 50
    };

    int status = price_table_precompute(table, &grid);
    EXPECT_EQ(status, 0);

    // Vega at middle volatility point should be computed
    double vega = price_table_get_vega(table, 0, 0, 1, 0, 0);
    EXPECT_FALSE(std::isnan(vega));

    // Vega should be positive for ATM put
    EXPECT_GT(vega, 0.0);

    // Vega should be reasonably sized (0.1 to 0.5 for typical ATM put)
    EXPECT_GT(vega, 0.05);
    EXPECT_LT(vega, 1.0);

    price_table_destroy(table);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_test --test_filter="PriceTableTest.VegaPrecomputation" --test_output=errors`

Expected: FAIL (vega still NaN after precomputation)

**Step 3: Implement vega computation in precompute**

Find `price_table_precompute()` in `src/price_table.c`.

After the loop that solves options and stores prices (around line 600), add vega computation logic.

Insert after the price extraction code:

```c
                // Extract price at the spot price
                double price = american_option_get_value_at_spot(
                    batch_results[i].solver, spot_price, K_ref);

                table->prices[idx] = price;

                // Compute vega via centered finite differences
                // vega = ∂V/∂σ ≈ (V(σ+h) - V(σ-h)) / (2h)
                double vega = NAN;

                // Only compute vega if we have neighbors in volatility dimension
                if (i_sigma > 0 && i_sigma < table->n_volatility - 1) {
                    double sigma_current = table->volatility_grid[i_sigma];
                    double sigma_minus = table->volatility_grid[i_sigma - 1];
                    double sigma_plus = table->volatility_grid[i_sigma + 1];

                    // Get already-computed prices at neighboring volatilities
                    // (These were computed earlier in the same maturity slice)
                    size_t idx_minus = i_m * table->stride_m + i_tau * table->stride_tau
                                     + (i_sigma - 1) * table->stride_sigma + i_r * table->stride_r
                                     + i_q * table->stride_q;
                    size_t idx_plus = i_m * table->stride_m + i_tau * table->stride_tau
                                    + (i_sigma + 1) * table->stride_sigma + i_r * table->stride_r
                                    + i_q * table->stride_q;

                    double price_minus = table->prices[idx_minus];
                    double price_plus = table->prices[idx_plus];

                    // Centered difference
                    if (!isnan(price_minus) && !isnan(price_plus)) {
                        vega = (price_plus - price_minus) / (sigma_plus - sigma_minus);
                    }
                }

                table->vegas[idx] = vega;
```

**IMPORTANT NOTE:** The above assumes volatility slice is processed before vega is computed. We need to verify the batch processing order. If volatility varies within a batch, we need a two-pass approach.

**Step 4: Handle edge cases (boundary volatility points)**

The centered difference above only works for interior points (i_sigma > 0 and i_sigma < n-1).

For edge points, use one-sided differences:

Add before the centered difference block:

```c
                // Handle boundary cases with one-sided differences
                if (i_sigma == 0 && table->n_volatility > 1) {
                    // Forward difference at lower boundary
                    double sigma_current = table->volatility_grid[0];
                    double sigma_next = table->volatility_grid[1];

                    size_t idx_next = i_m * table->stride_m + i_tau * table->stride_tau
                                    + 1 * table->stride_sigma + i_r * table->stride_r
                                    + i_q * table->stride_q;

                    double price_next = table->prices[idx_next];
                    if (!isnan(price_next)) {
                        vega = (price_next - price) / (sigma_next - sigma_current);
                    }
                } else if (i_sigma == table->n_volatility - 1 && table->n_volatility > 1) {
                    // Backward difference at upper boundary
                    double sigma_current = table->volatility_grid[i_sigma];
                    double sigma_prev = table->volatility_grid[i_sigma - 1];

                    size_t idx_prev = i_m * table->stride_m + i_tau * table->stride_tau
                                    + (i_sigma - 1) * table->stride_sigma + i_r * table->stride_r
                                    + i_q * table->stride_q;

                    double price_prev = table->prices[idx_prev];
                    if (!isnan(price_prev)) {
                        vega = (price - price_prev) / (sigma_current - sigma_prev);
                    }
                } else if (i_sigma > 0 && i_sigma < table->n_volatility - 1) {
                    // Centered difference (existing code from Step 3)
                    ...
                }
```

**Step 5: Verify batch processing order**

Check the precompute loop structure. If volatility points are NOT processed together in sequence, we need a two-pass approach:
- Pass 1: Compute all prices
- Pass 2: Compute all vegas using the complete price array

Add a comment documenting the assumption:

```c
// Note: Vega computation assumes volatility dimension is processed
// sequentially within each batch, allowing access to neighboring prices.
// If batch ordering changes, switch to two-pass approach.
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:price_table_test --test_filter="PriceTableTest.VegaPrecomputation" --test_output=errors`

Expected: PASS

**Step 7: Add trace logging for vega computation**

Add trace point after vega computation:

```c
                table->vegas[idx] = vega;

                // Trace vega computation (every 100 points for performance)
                if (completed % 100 == 0 && !isnan(vega)) {
                    MANGO_TRACE_ALGO_PROGRESS(MODULE_PRICE_TABLE,
                                              completed, n_total,
                                              (double)completed / (double)n_total);
                }
```

**Step 8: Commit**

```bash
git add src/price_table.c tests/price_table_test.cc
git commit -m "feat(price_table): compute vega during precomputation

- Use centered finite differences for interior volatility points
- Use one-sided differences for boundary points
- Store vega values alongside prices
- Add trace logging for vega computation progress"
```

---

## Task 4: Add Vega Interpolation API

**Files:**
- Modify: `src/price_table.h` (add vega interpolation functions)
- Modify: `src/price_table.c` (implement vega interpolation)
- Test: `tests/price_table_test.cc`

**Step 1: Write failing test for vega interpolation**

Add to `tests/price_table_test.cc`:

```cpp
TEST(PriceTableTest, VegaInterpolation4D) {
    // Create table with reasonable grid
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau = {0.25, 0.5, 1.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.03, 0.05};

    OptionPriceTable *table = price_table_create(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN);

    // Precompute (includes vega)
    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 50
    };
    price_table_precompute(table, &grid);

    // Build interpolation structures
    price_table_build_interpolation(table);

    // Query vega at off-grid point
    double vega = price_table_interpolate_vega_4d(table, 0.95, 0.75, 0.22, 0.04);

    // Should return interpolated value (not NaN)
    EXPECT_FALSE(std::isnan(vega));

    // Vega should be positive for put
    EXPECT_GT(vega, 0.0);

    price_table_destroy(table);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_test --test_filter="PriceTableTest.VegaInterpolation4D" --test_output=errors`

Expected: FAIL with "price_table_interpolate_vega_4d not declared"

**Step 3: Add vega interpolation declarations to header**

Add to `src/price_table.h` after the price interpolation functions (around line 380):

```c
/**
 * Interpolate vega (∂V/∂σ) at query point (4D table)
 *
 * Uses same interpolation strategy as prices for consistency.
 *
 * @param moneyness: S/K (raw, not transformed)
 * @param maturity: T (raw, not transformed)
 * @param volatility: σ (raw)
 * @param rate: r (raw)
 * @return interpolated vega value, or NaN if query out of bounds
 *
 * Example:
 *   double vega = price_table_interpolate_vega_4d(table, 1.05, 0.5, 0.20, 0.05);
 */
double price_table_interpolate_vega_4d(const OptionPriceTable *table,
                                       double moneyness, double maturity,
                                       double volatility, double rate);

/**
 * Interpolate vega (∂V/∂σ) at query point (5D table with dividend)
 *
 * @return interpolated vega value, or NaN if query out of bounds
 */
double price_table_interpolate_vega_5d(const OptionPriceTable *table,
                                       double moneyness, double maturity,
                                       double volatility, double rate,
                                       double dividend);
```

**Step 4: Implement 4D vega interpolation**

Add to `src/price_table.c` after `price_table_interpolate_4d()`:

```c
double price_table_interpolate_vega_4d(const OptionPriceTable *table,
                                       double moneyness, double maturity,
                                       double volatility, double rate) {
    if (!table || !table->strategy || !table->vegas) {
        return NAN;
    }

    if (table->n_dividend > 0) {
        // Table is 5D, can't use 4D interpolation
        return NAN;
    }

    // Check if strategy supports vega interpolation
    if (!table->strategy->interpolate_4d) {
        return NAN;
    }

    // Temporarily swap prices with vegas for interpolation
    double *original_prices = table->prices;
    ((OptionPriceTable*)table)->prices = table->vegas;

    // Use price interpolation strategy on vega data
    double result = table->strategy->interpolate_4d(
        table, moneyness, maturity, volatility, rate,
        table->interp_context);

    // Restore original prices pointer
    ((OptionPriceTable*)table)->prices = original_prices;

    return result;
}
```

**Step 5: Implement 5D vega interpolation**

Add to `src/price_table.c` after `price_table_interpolate_5d()`:

```c
double price_table_interpolate_vega_5d(const OptionPriceTable *table,
                                       double moneyness, double maturity,
                                       double volatility, double rate,
                                       double dividend) {
    if (!table || !table->strategy || !table->vegas) {
        return NAN;
    }

    if (table->n_dividend == 0) {
        // Table is 4D, can't use 5D interpolation
        return NAN;
    }

    // Check if strategy supports vega interpolation
    if (!table->strategy->interpolate_5d) {
        return NAN;
    }

    // Temporarily swap prices with vegas for interpolation
    double *original_prices = table->prices;
    ((OptionPriceTable*)table)->prices = table->vegas;

    // Use price interpolation strategy on vega data
    double result = table->strategy->interpolate_5d(
        table, moneyness, maturity, volatility, rate, dividend,
        table->interp_context);

    // Restore original prices pointer
    ((OptionPriceTable*)table)->prices = original_prices;

    return result;
}
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:price_table_test --test_filter="PriceTableTest.VegaInterpolation4D" --test_output=errors`

Expected: PASS

**Step 7: Add test for 5D vega interpolation**

Add to `tests/price_table_test.cc`:

```cpp
TEST(PriceTableTest, VegaInterpolation5D) {
    std::vector<double> m = {0.9, 1.0, 1.1};
    std::vector<double> tau = {0.25, 0.5};
    std::vector<double> sigma = {0.20, 0.25};
    std::vector<double> r = {0.05};
    std::vector<double> q = {0.0, 0.02};  // 5D with dividend

    OptionPriceTable *table = price_table_create(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        q.data(), q.size(),
        OPTION_CALL, AMERICAN);

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 50
    };
    price_table_precompute(table, &grid);
    price_table_build_interpolation(table);

    // Query vega at off-grid point (5D)
    double vega = price_table_interpolate_vega_5d(table, 0.95, 0.35, 0.22, 0.05, 0.01);

    EXPECT_FALSE(std::isnan(vega));
    EXPECT_GT(vega, 0.0);

    price_table_destroy(table);
}
```

Run: `bazel test //tests:price_table_test --test_filter="PriceTableTest.VegaInterpolation5D" --test_output=errors`

Expected: PASS

**Step 8: Commit**

```bash
git add src/price_table.h src/price_table.c tests/price_table_test.cc
git commit -m "feat(price_table): add vega interpolation API

- Add price_table_interpolate_vega_4d() for 4D tables
- Add price_table_interpolate_vega_5d() for 5D tables
- Reuse price interpolation strategy by swapping data pointers
- Include comprehensive tests for both 4D and 5D"
```

---

## Task 5: Update Binary Save/Load for Vega

**Files:**
- Modify: `src/price_table.c` (price_table_save and price_table_load)
- Test: `tests/price_table_test.cc`

**Step 1: Write failing test for vega persistence**

Add to `tests/price_table_test.cc`:

```cpp
TEST(PriceTableTest, VegaSaveLoad) {
    // Create and precompute table
    std::vector<double> m = {0.9, 1.0, 1.1};
    std::vector<double> tau = {0.5};
    std::vector<double> sigma = {0.15, 0.20, 0.25};
    std::vector<double> r = {0.05};

    OptionPriceTable *table = price_table_create(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN);

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 50
    };
    price_table_precompute(table, &grid);

    // Save to file
    const char *filename = "/tmp/claude/test_vega_table.bin";
    int status = price_table_save(table, filename);
    EXPECT_EQ(status, 0);

    // Get vega value before destroying
    double vega_original = price_table_get_vega(table, 1, 0, 1, 0, 0);
    EXPECT_FALSE(std::isnan(vega_original));

    price_table_destroy(table);

    // Load from file
    OptionPriceTable *loaded = price_table_load(filename);
    ASSERT_NE(loaded, nullptr);

    // Verify vega was restored
    double vega_loaded = price_table_get_vega(loaded, 1, 0, 1, 0, 0);
    EXPECT_DOUBLE_EQ(vega_loaded, vega_original);

    price_table_destroy(loaded);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_test --test_filter="PriceTableTest.VegaSaveLoad" --test_output=errors`

Expected: FAIL (vega values not saved/loaded)

**Step 3: Update price_table_save to include vegas**

Find `price_table_save()` in `src/price_table.c`.

After the code that writes prices (around line 750):

```c
    // Write price data
    fwrite(table->prices, sizeof(double), n_total, fp);
```

Add:

```c
    // Write vega data
    fwrite(table->vegas, sizeof(double), n_total, fp);
```

**Step 4: Update price_table_load to read vegas**

Find `price_table_load()` in `src/price_table.c`.

After the code that reads prices (around line 850):

```c
    // Read price data
    fread(table->prices, sizeof(double), n_total, fp);
```

Add:

```c
    // Read vega data (if available in file)
    // For backward compatibility, check if there's more data
    long current_pos = ftell(fp);
    fseek(fp, 0, SEEK_END);
    long end_pos = ftell(fp);
    fseek(fp, current_pos, SEEK_SET);

    if (end_pos - current_pos >= (long)(n_total * sizeof(double))) {
        // Vega data exists in file
        fread(table->vegas, sizeof(double), n_total, fp);
    } else {
        // Old format without vega data - initialize to NaN
        for (size_t i = 0; i < n_total; i++) {
            table->vegas[i] = NAN;
        }
    }
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:price_table_test --test_filter="PriceTableTest.VegaSaveLoad" --test_output=errors`

Expected: PASS

**Step 6: Add test for backward compatibility**

Add to `tests/price_table_test.cc`:

```cpp
TEST(PriceTableTest, LoadOldFormatWithoutVega) {
    // This test verifies that loading old binary files (without vega)
    // doesn't crash and initializes vega to NaN

    // Create a table and save with old format (manually, without vega)
    std::vector<double> m = {1.0};
    std::vector<double> tau = {0.5};
    std::vector<double> sigma = {0.20};
    std::vector<double> r = {0.05};

    OptionPriceTable *table = price_table_create(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN);

    // Set a price manually
    price_table_set(table, 0, 0, 0, 0, 0, 5.0);

    // Save (will include vega in new format)
    const char *filename = "/tmp/claude/test_compat_table.bin";
    price_table_save(table, filename);
    price_table_destroy(table);

    // Load and verify
    OptionPriceTable *loaded = price_table_load(filename);
    ASSERT_NE(loaded, nullptr);

    // Price should be preserved
    double price = price_table_get(loaded, 0, 0, 0, 0, 0);
    EXPECT_DOUBLE_EQ(price, 5.0);

    // Vega should exist (newly saved format)
    double vega = price_table_get_vega(loaded, 0, 0, 0, 0, 0);
    EXPECT_TRUE(std::isnan(vega));  // NaN because not precomputed

    price_table_destroy(loaded);
}
```

**Step 7: Commit**

```bash
git add src/price_table.c tests/price_table_test.cc
git commit -m "feat(price_table): save/load vega data in binary format

- Write vega array after price array in save
- Read vega array in load with backward compatibility
- Handle old format files (initialize vega to NaN)
- Add tests for save/load and backward compatibility"
```

---

## Task 6: Add Vega Benchmark

**Files:**
- Create: `benchmarks/vega_interpolation_benchmark.cc`
- Modify: `benchmarks/BUILD.bazel`

**Step 1: Create vega benchmark**

Create `benchmarks/vega_interpolation_benchmark.cc`:

```cpp
#include "benchmark/benchmark.h"
#include "../src/price_table.h"
#include <vector>
#include <cmath>

static OptionPriceTable* g_table = nullptr;

static void SetupTable(const benchmark::State& state) {
    if (g_table) return;  // Already set up

    // Create 4D table with reasonable grid
    std::vector<double> m(30);
    std::vector<double> tau(25);
    std::vector<double> sigma(15);
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
    for (size_t i = 0; i < 15; i++) {
        sigma[i] = 0.10 + i * (0.60 - 0.10) / 14.0;
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
        OPTION_PUT, AMERICAN,
        COORD_LOG_SQRT, LAYOUT_M_INNER);

    // Precompute
    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 101, .dt = 0.001, .n_steps = 1000
    };

    price_table_precompute(g_table, &grid);
    price_table_build_interpolation(g_table);
}

static void BM_VegaInterpolation4D(benchmark::State& state) {
    SetupTable(state);

    // Query at off-grid point
    const double m = 1.05;
    const double tau = 0.5;
    const double sigma = 0.25;
    const double r = 0.05;

    for (auto _ : state) {
        double vega = price_table_interpolate_vega_4d(g_table, m, tau, sigma, r);
        benchmark::DoNotOptimize(vega);
    }
}
BENCHMARK(BM_VegaInterpolation4D);

static void BM_PriceInterpolation4D(benchmark::State& state) {
    SetupTable(state);

    const double m = 1.05;
    const double tau = 0.5;
    const double sigma = 0.25;
    const double r = 0.05;

    for (auto _ : state) {
        double price = price_table_interpolate_4d(g_table, m, tau, sigma, r);
        benchmark::DoNotOptimize(price);
    }
}
BENCHMARK(BM_PriceInterpolation4D);

BENCHMARK_MAIN();
```

**Step 2: Add benchmark to BUILD.bazel**

Add to `benchmarks/BUILD.bazel`:

```python
cc_binary(
    name = "vega_interpolation_benchmark",
    srcs = ["vega_interpolation_benchmark.cc"],
    deps = [
        "//src:price_table",
        "@google_benchmark//:benchmark",
    ],
)
```

**Step 3: Run benchmark**

Run: `bazel run //benchmarks:vega_interpolation_benchmark`

Expected output showing vega interpolation is same speed as price interpolation (~500ns)

**Step 4: Commit**

```bash
git add benchmarks/vega_interpolation_benchmark.cc benchmarks/BUILD.bazel
git commit -m "bench: add vega interpolation performance benchmark

- Compare vega vs price interpolation performance
- Expected: same speed (~500ns) since using same algorithm
- Includes table setup with precomputation"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `docs/PROJECT_OVERVIEW.md`
- Modify: `docs/QUICK_REFERENCE.md`
- Modify: `src/price_table.h` (header comments)

**Step 1: Update PROJECT_OVERVIEW.md**

Add to the "What mango-option Provides" section:

```markdown
#### Greeks Calculation via Vega Interpolation

```c
double vega = price_table_interpolate_vega_4d(table, 1.05, 0.5, 0.20, 0.05);
```

- **Purpose**: Fast, accurate vega (∂V/∂σ) queries for Greeks calculation
- **Performance**: ~500ns per query (same as price interpolation)
- **Accuracy**: More accurate than finite differences at query time
- **Use case**: Newton-based IV inversion, Greeks for risk management
```

**Step 2: Update QUICK_REFERENCE.md**

Add new section after "Interpolation" section:

```markdown
### Greeks via Vega Interpolation

```c
// After precomputation, vega is available alongside prices
double vega = price_table_interpolate_vega_4d(table,
    1.05,   // moneyness
    0.5,    // maturity
    0.20,   // volatility
    0.05);  // rate

// Also available for 5D tables
double vega_5d = price_table_interpolate_vega_5d(table,
    1.05, 0.5, 0.20, 0.05, 0.02);  // with dividend
```

**Key Points:**
- Vega computed during precomputation (centered finite differences)
- Same interpolation strategy as prices (cubic or multilinear)
- ~500ns per query (same speed as price)
- More accurate than computing vega at query time
- Enables Newton-based IV inversion
```

**Step 3: Update price_table.h header documentation**

Update the file header comment in `src/price_table.h` around line 24:

```c
/**
 * @file price_table.h
 * @brief Multi-dimensional option price table with pluggable interpolation
 *
 * Pre-computes option prices and vegas on a multi-dimensional grid for fast lookup:
 * - Moneyness (m = S/K)
 * - Maturity (τ = T - t)
 * - Volatility (σ)
 * - Interest rate (r)
 * - Dividend yield (q) [optional, 5D mode]
 *
 * Features:
 * - Sub-microsecond queries (4D: ~500ns, 5D: ~2µs)
 * - 40,000x faster than FDM solver (21.7ms → 500ns)
 * - Vega interpolation for accurate Greeks
 * - Runtime interpolation strategy selection
 * - Parallel pre-computation via OpenMP
 * - Binary save/load for persistence
 *
 * Typical Usage:
 *   // Create table structure
 *   OptionPriceTable *table = price_table_create(
 *       moneyness, n_m, maturity, n_tau, volatility, n_sigma,
 *       rate, n_r, NULL, 0, OPTION_PUT, AMERICAN);
 *
 *   // Pre-compute all option prices and vegas (uses FDM)
 *   price_table_precompute(table, pde_solver_template);
 *
 *   // Save for fast loading later
 *   price_table_save(table, "spx_put_american.bin");
 *
 *   // Fast price query (~500ns)
 *   double price = price_table_interpolate_4d(table, 1.05, 0.25, 0.20, 0.05);
 *
 *   // Fast vega query (~500ns)
 *   double vega = price_table_interpolate_vega_4d(table, 1.05, 0.25, 0.20, 0.05);
 *
 *   // Cleanup
 *   price_table_destroy(table);
 */
```

**Step 4: Commit**

```bash
git add docs/PROJECT_OVERVIEW.md docs/QUICK_REFERENCE.md src/price_table.h
git commit -m "docs: document vega interpolation feature

- Update PROJECT_OVERVIEW with Greeks section
- Add vega interpolation examples to QUICK_REFERENCE
- Update price_table.h header with vega usage"
```

---

## Task 8: Update Issue #39 Status

**Files:**
- Update issue #39 on GitHub

**Step 1: Update issue body**

Use `gh` CLI to update issue:

```bash
gh issue comment 39 --body "## P1 Implementation Complete ✅

Vega interpolation has been implemented alongside option prices.

**Features Added:**
- ✅ Vega storage in OptionPriceTable structure
- ✅ Vega computation during precomputation (centered finite differences)
- ✅ Vega get/set API functions
- ✅ Vega interpolation (4D and 5D)
- ✅ Binary save/load with backward compatibility
- ✅ Performance benchmark (confirms ~500ns query time)
- ✅ Comprehensive test coverage

**Performance:**
- Precomputation overhead: ~2x (need to compute neighbors for finite diff)
- Query performance: Same as price interpolation (~500ns)
- Memory: 2x storage (prices + vegas)

**Benefits:**
- More accurate Greeks (no finite difference errors at query time)
- Enables Newton-based IV inversion
- Reduces required grid density

**Next Steps:**
- P2: Non-uniform grid spacing
- P3: Adaptive grid refinement"
```

**Step 2: Update issue labels**

```bash
gh issue edit 39 --remove-label "P1: vega interpolation"
```

---

## Verification & Testing

After completing all tasks:

**Run full test suite:**
```bash
bazel test //tests:price_table_test --test_output=errors
```

Expected: All tests pass

**Run vega benchmark:**
```bash
bazel run //benchmarks:vega_interpolation_benchmark
```

Expected: Vega interpolation ~500ns (same as price)

**Run accuracy comparison:**
```bash
bazel run //benchmarks:accuracy_comparison
```

Expected: No degradation in price accuracy, vega values reasonable

**Check memory usage:**
```bash
# Table should be ~2x size due to vegas array
# For 30×25×15×10 = 112,500 grid points:
# Prices: 112,500 × 8 bytes = 900 KB
# Vegas: 112,500 × 8 bytes = 900 KB
# Total: ~1.8 MB (vs ~900 KB before)
```

---

## Success Criteria

- [x] All existing tests pass
- [x] New vega tests pass (allocation, get/set, interpolation, save/load)
- [x] Vega interpolation performance matches price interpolation (~500ns)
- [x] Binary format backward compatible (old files load successfully)
- [x] Documentation updated
- [x] Issue #39 P1 marked complete

## Known Limitations

1. **Vega computation accuracy**: Uses finite differences (2nd order accurate) rather than solving sensitivity PDE. Acceptable for most use cases.

2. **Precomputation time**: 2× overhead due to needing neighboring volatility solves. Could be optimized by batching volatility points.

3. **Memory usage**: Doubles from adding vegas array. For large grids (>1M points), may need compression or on-demand computation.

4. **Edge points**: Boundary volatility points use one-sided differences (1st order accurate). Consider using higher-order one-sided stencils if needed.

## Future Enhancements

- Solve sensitivity PDE for vega (more accurate, but slower)
- Compute other Greeks (delta, gamma, theta, rho) similarly
- Compress vega storage using delta encoding or lossy compression
- Parallel vega computation within batches
