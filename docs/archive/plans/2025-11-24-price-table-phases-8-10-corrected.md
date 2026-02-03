# Price Table Phases 8-10 Implementation Plan (Corrected)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the remaining phases (8-10) of the price table refactor with correct batch solving architecture: solve one PDE per (σ, r) pair and reuse surfaces across (m, τ) grid points.

**Architecture:**
- make_batch iterates only over high-cost axes (volatility × rate), creating Nσ × Nr batch entries with normalized parameters (Spot=Strike=K_ref)
- solve_batch reuses BatchAmericanOptionSolver with snapshot registration for maturity grid
- extract_tensor interpolates from recorded spatial/temporal surfaces to populate full 4D tensor
- fit_coeffs uses BSplineNDSeparable to create interpolation coefficients

**Tech Stack:** C++23, BatchAmericanOptionSolver, mdspan, cubic splines, BSplineNDSeparable

---

## Phase 8: Batch Solving with Surface Reuse

### Task 8.1: Fix make_batch to iterate (σ, r) only

**Files:**
- Modify: `src/option/price_table_builder.cpp:17-54`
- Modify: `tests/price_table_builder_test.cc:117-149`

**Step 1: Write failing test for correct batch size**

Update `tests/price_table_builder_test.cc`:

```cpp
TEST(PriceTableBuilderTest, MakeBatchIteratesVolatilityAndRateOnly) {
    // Design: make_batch should iterate axes[2] × axes[3] only (vol × rate)
    // NOT all grid points (would explode PDE count)

    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value(),
        .n_time = 1000,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0, 1.1};      // moneyness: 3 points
    axes.grids[1] = {0.1, 0.5, 1.0};      // maturity: 3 points
    axes.grids[2] = {0.15, 0.20, 0.25};   // volatility: 3 points
    axes.grids[3] = {0.02, 0.05};         // rate: 2 points

    // Should create 3 × 2 = 6 batch entries (vol × rate)
    // NOT 3 × 3 × 3 × 2 = 54 entries (all axes)
    auto batch = builder.make_batch_for_testing(axes);

    EXPECT_EQ(batch.size(), 6);  // Nσ × Nr

    // Verify all batch entries use normalized params (Spot = Strike = K_ref)
    for (const auto& params : batch) {
        EXPECT_DOUBLE_EQ(params.spot, 100.0);
        EXPECT_DOUBLE_EQ(params.strike, 100.0);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_test --test_filter=MakeBatchIteratesVolatilityAndRateOnly`
Expected: FAIL (current implementation creates 54 entries, not 6)

**Step 3: Fix make_batch implementation**

Update `src/option/price_table_builder.cpp`:

```cpp
template <size_t N>
std::vector<AmericanOptionParams>
PriceTableBuilder<N>::make_batch(const PriceTableAxes<N>& axes) const {
    if constexpr (N == 4) {
        std::vector<AmericanOptionParams> batch;

        // Iterate only over high-cost axes: axes[2] (σ) and axes[3] (r)
        // This creates Nσ × Nr batch entries, NOT Nm × Nt × Nσ × Nr
        // Each solve produces a surface over (m, τ) that gets reused
        const size_t Nσ = axes.grids[2].size();
        const size_t Nr = axes.grids[3].size();
        batch.reserve(Nσ * Nr);

        // Normalized parameters: Spot = Strike = K_ref
        // Moneyness and maturity are handled via grid interpolation in extract_tensor
        const double K_ref = config_.K_ref;

        for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
            for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
                double sigma = axes.grids[2][σ_idx];
                double r = axes.grids[3][r_idx];

                // Normalized solve: Spot = Strike = K_ref
                // Surface will be interpolated across m and τ in extract_tensor
                AmericanOptionParams params{
                    .strike = K_ref,
                    .spot = K_ref,
                    .maturity = axes.grids[1].back(),  // Max maturity for this (σ, r)
                    .volatility = sigma,
                    .rate = r,
                    .continuous_dividend_yield = config_.dividend_yield,
                    .option_type = config_.option_type,
                    .discrete_dividends = config_.discrete_dividends
                };

                batch.push_back(params);
            }
        }

        return batch;
    } else {
        // Return empty batch for N≠4
        return {};
    }
}
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:price_table_builder_test --test_filter=MakeBatchIteratesVolatilityAndRateOnly`
Expected: PASS

**Step 5: Update existing test**

Update `MakeBatch4D` test to expect correct batch size (6 instead of 54):

```cpp
TEST(PriceTableBuilderTest, MakeBatch4D) {
    // ... existing setup ...

    // Should create 1 × 1 = 1 option (vol × rate)
    // NOT 2 × 2 × 1 × 1 = 4 options
    auto batch = builder.make_batch_for_testing(axes);
    EXPECT_EQ(batch.size(), 1);  // 1 vol × 1 rate

    // ... rest of test ...
}
```

**Step 6: Commit**

```bash
git add src/option/price_table_builder.cpp tests/price_table_builder_test.cc
git commit -m "Fix make_batch to iterate (σ, r) only, not all axes

Correct batch generation to solve one PDE per (volatility, rate) pair
using normalized parameters (Spot=Strike=K_ref). This matches the
original price_table_4d_builder architecture and avoids PDE explosion.

Each solve produces a surface over (moneyness, maturity) that will be
reused via interpolation in extract_tensor."
```

---

### Task 8.2: Implement solve_batch with snapshot registration

**Files:**
- Modify: `src/option/price_table_builder.cpp:56-80`
- Create test: `tests/price_table_builder_test.cc` (new test)

**Step 1: Write failing test**

```cpp
TEST(PriceTableBuilderTest, SolveBatchRegistersMaturitySnapshots) {
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value(),
        .n_time = 1000,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0};
    axes.grids[1] = {0.1, 0.5, 1.0};  // 3 maturity points
    axes.grids[2] = {0.20};           // 1 vol
    axes.grids[3] = {0.05};           // 1 rate

    auto batch_params = builder.make_batch_for_testing(axes);
    auto batch_result = builder.solve_batch_for_testing(batch_params, axes);

    // Verify snapshots were registered (should have 3 snapshots)
    ASSERT_EQ(batch_result.results.size(), 1);
    ASSERT_TRUE(batch_result.results[0].has_value());

    auto grid = batch_result.results[0]->grid();
    EXPECT_GE(grid->num_snapshots(), axes.grids[1].size());
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_test --test_filter=SolveBatchRegistersMaturitySnapshots`
Expected: FAIL (solve_batch doesn't exist yet)

**Step 3: Implement solve_batch**

Add to `src/option/price_table_builder.hpp`:

```cpp
private:
    BatchAmericanOptionResult solve_batch(
        const std::vector<AmericanOptionParams>& batch,
        const PriceTableAxes<N>& axes) const;

public:  // For testing
    BatchAmericanOptionResult solve_batch_for_testing(
        const std::vector<AmericanOptionParams>& batch,
        const PriceTableAxes<N>& axes) const {
        return solve_batch(batch, axes);
    }
```

Implement in `src/option/price_table_builder.cpp`:

```cpp
template <size_t N>
BatchAmericanOptionResult
PriceTableBuilder<N>::solve_batch(
    const std::vector<AmericanOptionParams>& batch,
    const PriceTableAxes<N>& axes) const
{
    if constexpr (N != 4) {
        // Return empty result for N≠4
        BatchAmericanOptionResult result;
        result.failed_count = batch.size();
        return result;
    }

    // Configure solver with grid bounds and time steps from config
    BatchAmericanOptionSolver solver;
    solver.set_grid_bounds(config_.grid_estimator.x_min(),
                          config_.grid_estimator.x_max());
    solver.set_grid_size(config_.grid_estimator.n_points());
    solver.set_time_steps(config_.n_time);

    // Register maturity grid as snapshot times
    // This enables extract_tensor to access surfaces at each maturity point
    solver.set_snapshot_times(axes.grids[1]);  // axes.grids[1] = maturity axis

    // Solve batch with shared grid optimization (normalized chain solver)
    return solver.solve_batch(batch, true);  // use_shared_grid = true
}
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:price_table_builder_test --test_filter=SolveBatchRegistersMaturitySnapshots`
Expected: PASS

**Step 5: Commit**

```bash
git add src/option/price_table_builder.hpp src/option/price_table_builder.cpp tests/price_table_builder_test.cc
git commit -m "Implement solve_batch with snapshot registration

Register maturity grid (axes.grids[1]) as snapshot times so BatchAmericanOptionSolver
records surfaces at each maturity point. This enables extract_tensor to interpolate
across the full (m, τ) grid from Nσ × Nr solves."
```

---

## Phase 9: Surface Extraction with Interpolation

### Task 9.1: Implement extract_tensor with cubic spline interpolation

**Files:**
- Modify: `src/option/price_table_builder.cpp:82-120`
- Create test: `tests/price_table_builder_test.cc` (new test)
- Reference: `src/option/price_table_extraction.cpp` (existing implementation)

**Step 1: Write failing test**

```cpp
TEST(PriceTableBuilderTest, ExtractTensorInterpolatesSurfaces) {
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value(),
        .n_time = 1000,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0, 1.1};      // 3 moneyness points
    axes.grids[1] = {0.1, 0.5, 1.0};      // 3 maturity points
    axes.grids[2] = {0.20};               // 1 vol
    axes.grids[3] = {0.05};               // 1 rate

    auto batch_params = builder.make_batch_for_testing(axes);
    auto batch_result = builder.solve_batch_for_testing(batch_params, axes);
    auto tensor_result = builder.extract_tensor_for_testing(batch_result, axes);

    ASSERT_TRUE(tensor_result.has_value());
    auto tensor = tensor_result.value();

    // Tensor should have full 4D shape: 3×3×1×1 = 9 points
    EXPECT_EQ(tensor.view.extent(0), 3);  // moneyness
    EXPECT_EQ(tensor.view.extent(1), 3);  // maturity
    EXPECT_EQ(tensor.view.extent(2), 1);  // volatility
    EXPECT_EQ(tensor.view.extent(3), 1);  // rate

    // Verify prices are populated (not NaN or zero)
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double price = tensor.view[i, j, 0, 0];
            EXPECT_TRUE(std::isfinite(price));
            EXPECT_GT(price, 0.0);
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_table_builder_test --test_filter=ExtractTensorInterpolatesSurfaces`
Expected: FAIL (extract_tensor doesn't exist yet)

**Step 3: Implement extract_tensor**

Add to `src/option/price_table_builder.hpp`:

```cpp
#include "mango/math/cubic_spline_solver.hpp"

private:
    std::expected<PriceTensor<N>, std::string> extract_tensor(
        const BatchAmericanOptionResult& batch,
        const PriceTableAxes<N>& axes) const;

public:  // For testing
    std::expected<PriceTensor<N>, std::string> extract_tensor_for_testing(
        const BatchAmericanOptionResult& batch,
        const PriceTableAxes<N>& axes) const {
        return extract_tensor(batch, axes);
    }
```

Implement in `src/option/price_table_builder.cpp`:

```cpp
template <size_t N>
std::expected<PriceTensor<N>, std::string>
PriceTableBuilder<N>::extract_tensor(
    const BatchAmericanOptionResult& batch,
    const PriceTableAxes<N>& axes) const
{
    if constexpr (N != 4) {
        return std::unexpected("extract_tensor only supports N=4");
    }

    const size_t Nm = axes.grids[0].size();  // moneyness
    const size_t Nt = axes.grids[1].size();  // maturity
    const size_t Nσ = axes.grids[2].size();  // volatility
    const size_t Nr = axes.grids[3].size();  // rate

    // Verify batch size matches (σ, r) grid
    const size_t expected_batch_size = Nσ * Nr;
    if (batch.results.size() != expected_batch_size) {
        MANGO_TRACE_RUNTIME_ERROR(MODULE_PRICE_TABLE,
            batch.results.size(), expected_batch_size);
        return std::unexpected(
            "Batch size mismatch: expected " + std::to_string(expected_batch_size) +
            " results (Nσ × Nr), got " + std::to_string(batch.results.size()));
    }

    // Create tensor
    const size_t total_points = Nm * Nt * Nσ * Nr;
    const size_t tensor_bytes = total_points * sizeof(double);
    const size_t arena_bytes = tensor_bytes + 64;  // 64-byte alignment padding

    auto arena = memory::AlignedArena::create(arena_bytes);
    if (!arena.has_value()) {
        return std::unexpected("Failed to create arena: " + arena.error());
    }

    std::array<size_t, N> shape = {Nm, Nt, Nσ, Nr};
    auto tensor_result = PriceTensor<N>::create(shape, arena.value());
    if (!tensor_result.has_value()) {
        return std::unexpected("Failed to create tensor: " + tensor_result.error());
    }

    auto tensor = tensor_result.value();

    // Precompute log-moneyness for interpolation
    std::vector<double> log_moneyness(Nm);
    for (size_t i = 0; i < Nm; ++i) {
        log_moneyness[i] = std::log(axes.grids[0][i]);
    }

    // Extract prices from each (σ, r) surface
    for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
        for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
            size_t batch_idx = σ_idx * Nr + r_idx;
            const auto& result_expected = batch.results[batch_idx];

            if (!result_expected.has_value()) {
                // Fill with NaN for failed solves
                for (size_t i = 0; i < Nm; ++i) {
                    for (size_t j = 0; j < Nt; ++j) {
                        tensor.view[i, j, σ_idx, r_idx] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
                continue;
            }

            const auto& result = result_expected.value();
            auto grid = result.grid();
            auto x_grid = grid->x();  // Spatial grid (log-moneyness)

            // For each maturity snapshot
            for (size_t j = 0; j < Nt; ++j) {
                // Get spatial solution at this maturity
                std::span<const double> spatial_solution = result.at_time(j);

                // Interpolate across moneyness using cubic spline
                // This resamples the PDE solution onto our moneyness grid
                CubicSplineSolver spline;
                auto spline_result = spline.solve(
                    std::vector<double>(x_grid.begin(), x_grid.end()),
                    std::vector<double>(spatial_solution.begin(), spatial_solution.end()));

                if (!spline_result.has_value()) {
                    // Spline fitting failed, fill with NaN
                    for (size_t i = 0; i < Nm; ++i) {
                        tensor.view[i, j, σ_idx, r_idx] = std::numeric_limits<double>::quiet_NaN();
                    }
                    continue;
                }

                // Evaluate spline at each moneyness point
                for (size_t i = 0; i < Nm; ++i) {
                    double price = spline_result->eval(log_moneyness[i]);
                    tensor.view[i, j, σ_idx, r_idx] = price;
                }
            }
        }
    }

    return tensor;
}
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:price_table_builder_test --test_filter=ExtractTensorInterpolatesSurfaces`
Expected: PASS

**Step 5: Add BUILD dependency**

Update `src/option/BUILD.bazel`:

```python
cc_library(
    name = "price_table_builder",
    srcs = ["price_table_builder.cpp"],
    hdrs = ["price_table_builder.hpp"],
    deps = [
        ":price_table_axes",
        ":price_table_config",
        ":price_table_metadata",
        ":price_table_surface",
        ":price_table_tensor",
        ":price_tensor",
        ":american_option_batch",
        "//src/math:cubic_spline_solver",  # ADD THIS
        "//src/support:aligned_arena",
        "//common:ivcalc_trace_hdr",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 6: Commit**

```bash
git add src/option/price_table_builder.cpp src/option/price_table_builder.hpp src/option/BUILD.bazel tests/price_table_builder_test.cc
git commit -m "Implement extract_tensor with cubic spline interpolation

Extract prices from Nσ × Nr batch results by interpolating spatial surfaces
across moneyness grid at each maturity snapshot. This reuses each PDE solve
to populate the full (m, τ) slice, avoiding the need to solve at every grid point.

Matches architecture from price_table_extraction.cpp."
```

---

## Phase 10: B-Spline Fitting and Pipeline Integration

### Task 10.1: Implement fit_coeffs

**Files:**
- Modify: `src/option/price_table_builder.cpp:122-160`
- Test already exists from Phase 9.1 (previous implementation)

**Step 1: Verify existing test still makes sense**

The existing `FitCoeffsReturnsCorrectSize` test should still be valid - just verify the test expectations match the corrected architecture.

**Step 2: Implement fit_coeffs (unchanged from previous)**

The `fit_coeffs` implementation doesn't need changes - it operates on the tensor regardless of how it was populated.

```cpp
template <size_t N>
std::expected<std::vector<double>, std::string>
PriceTableBuilder<N>::fit_coeffs(
    const PriceTensor<N>& tensor,
    const PriceTableAxes<N>& axes) const
{
    if constexpr (N != 4) {
        MANGO_TRACE_RUNTIME_ERROR(MODULE_PRICE_TABLE, N, 0);
        return std::unexpected(
            "fit_coeffs only supports N=4 dimensions. Requested N=" +
            std::to_string(N));
    }

    // Extract grids for BSplineNDSeparable
    std::array<std::vector<double>, N> grids;
    for (size_t i = 0; i < N; ++i) {
        grids[i] = axes.grids[i];
    }

    // Create fitter
    auto fitter_result = BSplineNDSeparable<double, N>::create(std::move(grids));
    if (!fitter_result.has_value()) {
        return std::unexpected("Failed to create fitter: " + fitter_result.error());
    }

    // Extract values from tensor (convert mdspan to vector)
    size_t total_points = axes.total_points();
    std::vector<double> values;
    values.reserve(total_points);

    // Extract in row-major order using for_each_axis_index
    if constexpr (N == 4) {
        for_each_axis_index<0>(axes, [&](const std::array<size_t, N>& indices) {
            values.push_back(tensor.view[indices[0], indices[1], indices[2], indices[3]]);
        });
    }

    // Fit B-spline coefficients
    auto fit_result = fitter_result->fit(values);
    if (!fit_result.has_value()) {
        return std::unexpected("B-spline fitting failed: " + fit_result.error());
    }

    return std::move(fit_result.value());
}
```

**Step 3: Run tests**

Run: `bazel test //tests:price_table_builder_test --test_filter=FitCoeffs*`
Expected: PASS

**Step 4: Commit (if any changes)**

```bash
git add src/option/price_table_builder.cpp
git commit -m "Verify fit_coeffs works with corrected tensor extraction"
```

---

### Task 10.2: Complete build() pipeline

**Files:**
- Modify: `src/option/price_table_builder.cpp:10-40`
- Test already exists: `BuildCreatesValidSurface`

**Step 1: Update build() to use corrected pipeline**

```cpp
template <size_t N>
std::expected<std::shared_ptr<const PriceTableSurface<N>>, std::string>
PriceTableBuilder<N>::build(const PriceTableAxes<N>& axes) {
    if constexpr (N != 4) {
        MANGO_TRACE_RUNTIME_ERROR(MODULE_PRICE_TABLE, N, 0);
        return std::unexpected("build() only supports N=4");
    }

    // Step 1: Validate axes
    auto axes_valid = axes.validate();
    if (!axes_valid.has_value()) {
        return std::unexpected("Invalid axes: " + axes_valid.error());
    }

    // Step 2: Generate batch (Nσ × Nr entries)
    auto batch_params = make_batch(axes);
    if (batch_params.empty()) {
        return std::unexpected("make_batch returned empty batch");
    }

    // Step 3: Solve batch with snapshot registration
    auto batch_result = solve_batch(batch_params, axes);
    if (batch_result.failed_count > 0) {
        return std::unexpected(
            "solve_batch had " + std::to_string(batch_result.failed_count) +
            " failures out of " + std::to_string(batch_result.results.size()));
    }

    // Step 4: Extract tensor via interpolation
    auto tensor_result = extract_tensor(batch_result, axes);
    if (!tensor_result.has_value()) {
        return std::unexpected("extract_tensor failed: " + tensor_result.error());
    }

    // Step 5: Fit B-spline coefficients
    auto coeffs_result = fit_coeffs(tensor_result.value(), axes);
    if (!coeffs_result.has_value()) {
        return std::unexpected("fit_coeffs failed: " + coeffs_result.error());
    }

    // Step 6: Create metadata
    PriceTableMetadata metadata{
        .K_ref = config_.K_ref,
        .dividend_yield = config_.dividend_yield,
        .discrete_dividends = config_.discrete_dividends
    };

    // Step 7: Build immutable surface
    return PriceTableSurface<N>::build(axes, std::move(coeffs_result.value()), metadata);
}
```

**Step 2: Run existing integration tests**

Run: `bazel test //tests:price_table_builder_test --test_filter=Build*`
Expected: PASS

**Step 3: Commit**

```bash
git add src/option/price_table_builder.cpp
git commit -m "Complete build() pipeline with corrected architecture

Wire together corrected pipeline:
1. make_batch: Nσ × Nr batch entries (not all grid points)
2. solve_batch: Register maturity snapshots
3. extract_tensor: Interpolate from surfaces
4. fit_coeffs: B-spline fitting
5. PriceTableSurface::build()

This matches the original price_table_4d_builder design and avoids
the PDE explosion bug."
```

---

## Verification

### Task 11: Run full test suite and verify performance

**Step 1: Run all price table builder tests**

```bash
bazel test //tests:price_table_builder_test
```

Expected: All tests pass

**Step 2: Verify batch size is correct**

Check that a 50×30×20×10 grid generates 20×10 = 200 batch entries, not 300,000.

**Step 3: Commit verification**

```bash
git add .
git commit -m "Verify corrected implementation

Tests confirm:
- make_batch generates Nσ × Nr batch entries
- solve_batch registers maturity snapshots
- extract_tensor interpolates from surfaces
- build() pipeline completes successfully

Architecture now matches original price_table_4d_builder design."
```

---

## Summary

This plan corrects the three critical architectural errors:

1. **make_batch** now iterates only (σ, r) axes → Nσ × Nr solves, not Nm × Nt × Nσ × Nr
2. **extract_tensor** interpolates from spatial/temporal surfaces → full tensor from batch results
3. **PriceTableConfig** retains grid_estimator and n_time → solver configuration preserved

The corrected implementation matches the original price_table_4d_builder architecture and enables efficient price table construction.
