# Discrete Dividend Interpolated IV — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add discrete dividend support to the interpolated IV path via maturity segmentation, backward chaining, and multi-K_ref strike interpolation.

**Architecture:** Build N+1 price table segments per K_ref at dividend boundaries, chain backward, wrap in layered surface types (`SegmentedPriceSurface` → `SegmentedMultiKRefSurface`), and template `IVSolverInterpolated` on a `PriceSurface` concept so the user sees a single factory.

**Tech Stack:** C++23 (concepts, `std::expected`, designated initializers, `std::span`), Bazel, GoogleTest, B-spline interpolation, TR-BDF2 PDE solver.

**Design doc:** `docs/plans/2026-01-31-discrete-dividend-interpolated-iv-design.md`

---

## Task 1: AmericanPriceSurface — Accept RawPrice content and add bounds accessors

Modify `AmericanPriceSurface` to handle `SurfaceContent::RawPrice` and expose bounds for the `PriceSurface` concept.

**Files:**
- Modify: `src/option/table/american_price_surface.hpp`
- Modify: `src/option/table/american_price_surface.cpp`
- Test: `tests/american_price_surface_test.cc` (create or extend existing)
- Modify: `tests/BUILD.bazel` (add test target if new)

**Step 1: Write failing tests for RawPrice create() and bounds accessors**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/american_price_surface.hpp"
#include "mango/option/table/price_table_surface.hpp"

using namespace mango;

// Helper: build a minimal 4D PriceTableSurface with given metadata
static std::shared_ptr<const PriceTableSurface<4>> make_test_surface(
    SurfaceContent content, double K_ref = 100.0) {
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};  // moneyness
    axes.grids[1] = {0.1, 0.25, 0.5, 1.0};       // maturity
    axes.grids[2] = {0.10, 0.20, 0.30, 0.40};    // vol
    axes.grids[3] = {0.02, 0.05};                 // rate
    // Fill coefficients (need enough for B-spline fitting)
    size_t n = axes.grids[0].size() * axes.grids[1].size()
             * axes.grids[2].size() * axes.grids[3].size();
    std::vector<double> coeffs(n, 0.01);  // small positive values
    PriceTableMetadata meta{
        .K_ref = K_ref, .dividend_yield = 0.0,
        .m_min = 0.8, .m_max = 1.2,
        .content = content, .discrete_dividends = {}
    };
    return PriceTableSurface<4>::build(std::move(axes), std::move(coeffs), meta)
        .value();
}

TEST(AmericanPriceSurfaceTest, CreateAcceptsRawPrice) {
    auto surface = make_test_surface(SurfaceContent::RawPrice);
    auto result = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(result.has_value()) << "RawPrice should be accepted by create()";
}

TEST(AmericanPriceSurfaceTest, CreateStillAcceptsEEP) {
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium);
    auto result = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_TRUE(result.has_value()) << "EEP should still be accepted";
}

TEST(AmericanPriceSurfaceTest, BoundsAccessors) {
    auto surface = make_test_surface(SurfaceContent::EarlyExercisePremium);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT).value();
    EXPECT_DOUBLE_EQ(aps.m_min(), 0.8);
    EXPECT_DOUBLE_EQ(aps.m_max(), 1.2);
    EXPECT_DOUBLE_EQ(aps.tau_min(), 0.1);
    EXPECT_DOUBLE_EQ(aps.tau_max(), 1.0);
    EXPECT_DOUBLE_EQ(aps.sigma_min(), 0.10);
    EXPECT_DOUBLE_EQ(aps.sigma_max(), 0.40);
    EXPECT_DOUBLE_EQ(aps.rate_min(), 0.02);
    EXPECT_DOUBLE_EQ(aps.rate_max(), 0.05);
}

TEST(AmericanPriceSurfaceTest, RawPriceVegaReturnsNaN) {
    auto surface = make_test_surface(SurfaceContent::RawPrice);
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT).value();
    double v = aps.vega(100.0, 100.0, 0.5, 0.2, 0.05);
    EXPECT_TRUE(std::isnan(v));
}

TEST(AmericanPriceSurfaceTest, CreateRejectsRawPriceWithDiscreteDividends) {
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};
    axes.grids[1] = {0.1, 0.25, 0.5, 1.0};
    axes.grids[2] = {0.10, 0.20, 0.30, 0.40};
    axes.grids[3] = {0.02, 0.05};
    size_t n = 5 * 4 * 4 * 2;
    std::vector<double> coeffs(n, 0.01);
    PriceTableMetadata meta{
        .K_ref = 100.0, .dividend_yield = 0.0,
        .m_min = 0.8, .m_max = 1.2,
        .content = SurfaceContent::RawPrice,
        .discrete_dividends = {{0.25, 1.0}}  // non-empty
    };
    auto surface = PriceTableSurface<4>::build(std::move(axes), std::move(coeffs), meta).value();
    auto result = AmericanPriceSurface::create(surface, OptionType::PUT);
    ASSERT_FALSE(result.has_value()) << "Should reject RawPrice with discrete dividends";
}
```

**Step 2: Run tests to verify they fail**

Run: `bazel test //tests:american_price_surface_test --test_output=all`
Expected: Compilation errors or test failures (create() rejects RawPrice, bounds accessors don't exist)

**Step 3: Implement changes**

In `src/option/table/american_price_surface.hpp`, add bounds accessors:
```cpp
[[nodiscard]] double m_min() const noexcept;
[[nodiscard]] double m_max() const noexcept;
[[nodiscard]] double tau_min() const noexcept;
[[nodiscard]] double tau_max() const noexcept;
[[nodiscard]] double sigma_min() const noexcept;
[[nodiscard]] double sigma_max() const noexcept;
[[nodiscard]] double rate_min() const noexcept;
[[nodiscard]] double rate_max() const noexcept;
```

In `src/option/table/american_price_surface.cpp`:
- `create()`: Remove the check `if (meta.content != SurfaceContent::EarlyExercisePremium)`. Keep the `discrete_dividends` rejection for both content types.
- `price()`: Branch on `surface_->metadata().content`:
  - `EarlyExercisePremium`: existing logic (EEP * K/K_ref + European)
  - `RawPrice`: return `surface_->value({spot / K_ref_, tau, sigma, rate})` (raw spline value, no scaling, no European addback). Assert `strike == K_ref_` via `MANGO_ASSERT` or equivalent debug check.
- `vega()`: For `RawPrice`, return `std::numeric_limits<double>::quiet_NaN()`.
- Bounds accessors: read from `surface_->axes()` and `surface_->metadata()`.

**Step 4: Run tests to verify they pass**

Run: `bazel test //tests:american_price_surface_test --test_output=all`
Expected: All PASS

**Step 5: Run full test suite to check for regressions**

Run: `bazel test //...`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add src/option/table/american_price_surface.hpp src/option/table/american_price_surface.cpp tests/american_price_surface_test.cc tests/BUILD.bazel
git commit -m "Add RawPrice support and bounds accessors to AmericanPriceSurface"
```

---

## Task 2: PriceSurface concept

Define the concept that `AmericanPriceSurface` satisfies. This is a header-only addition.

**Files:**
- Create: `src/option/table/price_surface_concept.hpp`
- Modify: `src/option/table/BUILD.bazel` (add header-only target)
- Test: `tests/price_surface_concept_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing test — static_assert that AmericanPriceSurface satisfies the concept**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/price_surface_concept.hpp"
#include "mango/option/table/american_price_surface.hpp"

using namespace mango;

static_assert(PriceSurface<AmericanPriceSurface>,
    "AmericanPriceSurface must satisfy PriceSurface concept");

TEST(PriceSurfaceConceptTest, AmericanPriceSurfaceSatisfiesConcept) {
    // Compile-time check above is the real test.
    // This test exists so the test binary runs.
    SUCCEED();
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:price_surface_concept_test --test_output=all`
Expected: Compilation error (concept doesn't exist yet)

**Step 3: Implement the concept**

Create `src/option/table/price_surface_concept.hpp`:
```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include <concepts>

namespace mango {

template <typename S>
concept PriceSurface = requires(const S& s, double spot, double strike,
                                double tau, double sigma, double rate) {
    { s.price(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { s.vega(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { s.m_min() } -> std::convertible_to<double>;
    { s.m_max() } -> std::convertible_to<double>;
    { s.tau_min() } -> std::convertible_to<double>;
    { s.tau_max() } -> std::convertible_to<double>;
    { s.sigma_min() } -> std::convertible_to<double>;
    { s.sigma_max() } -> std::convertible_to<double>;
    { s.rate_min() } -> std::convertible_to<double>;
    { s.rate_max() } -> std::convertible_to<double>;
};

}  // namespace mango
```

Add BUILD target in `src/option/table/BUILD.bazel`:
```bazel
cc_library(
    name = "price_surface_concept",
    hdrs = ["price_surface_concept.hpp"],
    visibility = ["//visibility:public"],
)
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:price_surface_concept_test --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/option/table/price_surface_concept.hpp src/option/table/BUILD.bazel tests/price_surface_concept_test.cc tests/BUILD.bazel
git commit -m "Add PriceSurface concept for surface type abstraction"
```

---

## Task 3: AmericanOptionSolver — Custom initial condition support

Add optional IC override so the solver can accept a custom initial condition instead of the static payoff.

**Files:**
- Modify: `src/option/american_option.hpp`
- Modify: `src/option/american_option.cpp`
- Test: `tests/american_option_custom_ic_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing test**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/american_option.hpp"

using namespace mango;

TEST(AmericanOptionSolverTest, CustomInitialCondition) {
    // Use a custom IC that's slightly different from standard payoff
    // Verify the solver runs and produces a result
    PricingParams params{
        .spot = 100.0, .strike = 100.0, .maturity = 0.5,
        .rate = 0.05, .dividend_yield = 0.02, .type = OptionType::PUT,
        .volatility = 0.20
    };
    auto [grid_spec, time_domain] = estimate_grid_for_option(params);
    std::pmr::synchronized_pool_resource pool;
    auto workspace = PDEWorkspace::create(grid_spec, &pool).value();

    // Custom IC: payoff + small constant (simulates chained terminal condition)
    auto custom_ic = [](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(1.0 - std::exp(x[i]), 0.0) + 0.01;
        }
    };

    AmericanOptionSolver solver(params, workspace);
    solver.set_initial_condition(custom_ic);
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->price(), 0.0);
}

TEST(AmericanOptionSolverTest, DefaultPayoffStillWorks) {
    // Regression: ensure default behavior (no custom IC) is unchanged
    PricingParams params{
        .spot = 100.0, .strike = 100.0, .maturity = 0.5,
        .rate = 0.05, .dividend_yield = 0.02, .type = OptionType::PUT,
        .volatility = 0.20
    };
    auto [grid_spec, time_domain] = estimate_grid_for_option(params);
    std::pmr::synchronized_pool_resource pool;
    auto workspace = PDEWorkspace::create(grid_spec, &pool).value();

    AmericanOptionSolver solver(params, workspace);
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->price(), 0.0);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:american_option_custom_ic_test --test_output=all`
Expected: Compilation error (`set_initial_condition` doesn't exist)

**Step 3: Implement**

In `src/option/american_option.hpp`, add:
```cpp
using InitialCondition = std::function<void(std::span<const double>, std::span<double>)>;

void set_initial_condition(InitialCondition ic) { custom_ic_ = std::move(ic); }
```

Add member: `std::optional<InitialCondition> custom_ic_;`

In `src/option/american_option.cpp`, in `solve()` at lines 109-119, change:
```cpp
if (params_.type == OptionType::PUT) {
    AmericanPutSolver pde_solver(params_, grid, workspace_);
    if (custom_ic_) {
        pde_solver.initialize(*custom_ic_);
    } else {
        pde_solver.initialize(AmericanPutSolver::payoff);
    }
    // ... rest unchanged
}
// Same for CALL branch
```

**Step 4: Run tests to verify they pass**

Run: `bazel test //tests:american_option_custom_ic_test --test_output=all`
Expected: All PASS

**Step 5: Run full test suite**

Run: `bazel test //...`
Expected: No regressions

**Step 6: Commit**

```bash
git add src/option/american_option.hpp src/option/american_option.cpp tests/american_option_custom_ic_test.cc tests/BUILD.bazel
git commit -m "Add custom initial condition support to AmericanOptionSolver"
```

---

## Task 4: PriceTableBuilder — Custom IC, RawPrice mode, τ=0 support

Three additive changes to the existing builder.

**Files:**
- Modify: `src/option/table/price_table_config.hpp`
- Modify: `src/option/table/price_table_builder.hpp`
- Modify: `src/option/table/price_table_builder.cpp`
- Test: `tests/price_table_builder_raw_price_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing tests**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/american_price_surface.hpp"

using namespace mango;

TEST(PriceTableBuilderTest, BuildWithRawPriceMode) {
    // Build a surface in RawPrice mode (no EEP subtraction)
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau_grid = {0.1, 0.25, 0.5};
    std::vector<double> vol_grid = {0.15, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.03, 0.05};

    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .dividend_yield = 0.0,
        .surface_content = SurfaceContent::RawPrice,  // new field
    };

    auto setup = PriceTableBuilder<4>::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, config.K_ref,
        GridAccuracyParams{}, config.option_type);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;

    // Override config on builder
    builder.set_surface_content(SurfaceContent::RawPrice);
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());

    // Verify metadata says RawPrice
    EXPECT_EQ(result->surface->metadata().content, SurfaceContent::RawPrice);
}

TEST(PriceTableBuilderTest, BuildWithCustomIC) {
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau_grid = {0.0, 0.1, 0.25, 0.5};  // includes τ=0
    std::vector<double> vol_grid = {0.15, 0.20, 0.30, 0.40};
    std::vector<double> rate_grid = {0.03, 0.05};

    auto custom_ic = [](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(1.0 - std::exp(x[i]), 0.0) + 0.005;
        }
    };

    // Build with custom IC and τ=0 allowed
    // (Exact API may vary — this tests the concept)
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .allow_tau_zero = true,
    };

    auto setup = PriceTableBuilder<4>::from_vectors(
        m_grid, tau_grid, vol_grid, rate_grid, config.K_ref,
        GridAccuracyParams{}, config.option_type);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;

    builder.set_initial_condition(custom_ic);
    builder.set_allow_tau_zero(true);
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
}
```

**Step 2: Run tests to verify they fail**

Run: `bazel test //tests:price_table_builder_raw_price_test --test_output=all`
Expected: Compilation errors (new API doesn't exist)

**Step 3: Implement**

In `src/option/table/price_table_config.hpp`, add to `PriceTableConfig`:
```cpp
SurfaceContent surface_content = SurfaceContent::EarlyExercisePremium;
bool allow_tau_zero = false;
```

In `src/option/table/price_table_builder.hpp`, add to the class:
```cpp
using InitialCondition = std::function<void(std::span<const double>, std::span<double>)>;

void set_initial_condition(InitialCondition ic) { custom_ic_ = std::move(ic); }
void set_surface_content(SurfaceContent content) { config_.surface_content = content; }
void set_allow_tau_zero(bool allow) { config_.allow_tau_zero = allow; }
```

Add member: `std::optional<InitialCondition> custom_ic_;`

In `src/option/table/price_table_builder.cpp`:
- `build()`: In axes validation, change the `τ > 0` check (around line 55-67) to:
  ```cpp
  if (!config_.allow_tau_zero && axes.grids[1].front() <= 0.0) {
      return std::unexpected(...);
  }
  ```
- `solve_batch()`: Thread `custom_ic_` to `AmericanOptionSolver` via the SetupCallback or by modifying the solve path to pass IC to each solver.
- `extract_tensor()`: When `config_.surface_content == SurfaceContent::RawPrice`, skip the EEP subtraction (lines ~443-474). Store `K_ref * normalized_price` directly instead of subtracting European price.
- Set `metadata.content = config_.surface_content` when building the result.

**Step 4: Run tests to verify they pass**

Run: `bazel test //tests:price_table_builder_raw_price_test --test_output=all`
Expected: All PASS

**Step 5: Run full test suite**

Run: `bazel test //...`
Expected: No regressions

**Step 6: Commit**

```bash
git add src/option/table/price_table_config.hpp src/option/table/price_table_builder.hpp src/option/table/price_table_builder.cpp tests/price_table_builder_raw_price_test.cc tests/BUILD.bazel
git commit -m "Add custom IC, RawPrice mode, and τ=0 support to PriceTableBuilder"
```

---

## Task 5: SegmentedPriceSurface

Internal component that owns N+1 segments for a single K_ref and handles segment selection, spot adjustment, and FD vega.

**Files:**
- Create: `src/option/table/segmented_price_surface.hpp`
- Create: `src/option/table/segmented_price_surface.cpp`
- Modify: `src/option/table/BUILD.bazel`
- Test: `tests/segmented_price_surface_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing tests**

Test segment selection, spot adjustment, and the worked example from the design doc.

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/segmented_price_surface.hpp"

using namespace mango;

// Use mock/stub AmericanPriceSurface segments that return known values.
// For unit testing, build minimal real surfaces via PriceTableBuilder
// or use a test helper that creates surfaces with known spline values.

TEST(SegmentedPriceSurfaceTest, FindsCorrectSegment) {
    // Two segments: seg1 τ∈[0, 0.5], seg0 τ∈(0.5, 1.0]
    // Query τ=0.8 → segment 0
    // Query τ=0.3 → segment 1
    // (Build real segments via PriceTableBuilder or mock)
    // Verify correct segment is selected
}

TEST(SegmentedPriceSurfaceTest, SpotAdjustmentForDividend) {
    // One dividend D=2.0 at t=0.5, T=1.0, K_ref=100
    // Query τ=0.8 (t_query=0.2): dividend at t=0.5 is in (0.2, 0.5] → subtract
    // S_adj = 100 - 2 = 98
}

TEST(SegmentedPriceSurfaceTest, NoDividendAdjustmentAfterExDiv) {
    // Query τ=0.3 (t_query=0.7): dividend at t=0.5 is NOT in (0.7, 1.0] → no adjustment
    // S_adj = 100
}

TEST(SegmentedPriceSurfaceTest, LocalTimeConversion) {
    // Query τ=0.8, segment τ_start=0.5 → τ_local = 0.3
    // Query τ=0.3, segment τ_start=0.0 → τ_local = 0.3
}

TEST(SegmentedPriceSurfaceTest, FDVegaForRawPriceSegment) {
    // RawPrice segment: vega computed via FD
    // Verify vega is finite and reasonable
}

TEST(SegmentedPriceSurfaceTest, AnalyticVegaForEEPSegment) {
    // Last segment (EEP): vega uses AmericanPriceSurface::vega() directly
}

TEST(SegmentedPriceSurfaceTest, SpotClampWhenSAdjNegative) {
    // Large dividend: D > S → S_adj clamped to ε
}

TEST(SegmentedPriceSurfaceTest, BoundsSpanFullMaturityRange) {
    // tau_min from last segment, tau_max from first segment
}
```

**Step 2: Run tests to verify they fail**

Run: `bazel test //tests:segmented_price_surface_test --test_output=all`
Expected: Compilation error (class doesn't exist)

**Step 3: Implement**

Create `src/option/table/segmented_price_surface.hpp`:
```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <utility>
#include "mango/option/table/american_price_surface.hpp"

namespace mango {

class SegmentedPriceSurface {
public:
    struct Segment {
        AmericanPriceSurface surface;
        double tau_start;  // global τ start
        double tau_end;    // global τ end
    };

    struct Config {
        std::vector<Segment> segments;  // ordered: last segment first (lowest τ)
        std::vector<std::pair<double, double>> dividends;  // (calendar_time, amount)
        double K_ref;
        double T;  // expiry in calendar time
    };

    static std::expected<SegmentedPriceSurface, ValidationError> create(Config config);

    [[nodiscard]] double price(double spot, double strike,
                               double tau, double sigma, double rate) const;
    [[nodiscard]] double vega(double spot, double strike,
                              double tau, double sigma, double rate) const;

    [[nodiscard]] double m_min() const noexcept;
    [[nodiscard]] double m_max() const noexcept;
    [[nodiscard]] double tau_min() const noexcept;
    [[nodiscard]] double tau_max() const noexcept;
    [[nodiscard]] double sigma_min() const noexcept;
    [[nodiscard]] double sigma_max() const noexcept;
    [[nodiscard]] double rate_min() const noexcept;
    [[nodiscard]] double rate_max() const noexcept;
    [[nodiscard]] double K_ref() const noexcept { return K_ref_; }

private:
    SegmentedPriceSurface() = default;

    struct DividendEntry {
        double calendar_time;
        double amount;
    };

    std::vector<Segment> segments_;
    std::vector<DividendEntry> dividends_;
    double K_ref_;
    double T_;
};

}  // namespace mango
```

Implement `src/option/table/segmented_price_surface.cpp` following the query algorithm from the design doc (calendar-time spot adjustment, segment selection, local τ conversion, FD vega for RawPrice segments with `ε_σ = max(1e-4, 1e-4 * σ)`).

**Step 4: Run tests to verify they pass**

Run: `bazel test //tests:segmented_price_surface_test --test_output=all`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/option/table/segmented_price_surface.hpp src/option/table/segmented_price_surface.cpp src/option/table/BUILD.bazel tests/segmented_price_surface_test.cc tests/BUILD.bazel
git commit -m "Add SegmentedPriceSurface for single-K_ref segmented pricing"
```

---

## Task 6: SegmentedPriceTableBuilder

Orchestrates backward-chained construction for a single K_ref.

**Files:**
- Create: `src/option/table/segmented_price_table_builder.hpp`
- Create: `src/option/table/segmented_price_table_builder.cpp`
- Modify: `src/option/table/BUILD.bazel`
- Test: `tests/segmented_price_table_builder_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing test**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/segmented_price_table_builder.hpp"

using namespace mango;

TEST(SegmentedPriceTableBuilderTest, BuildWithOneDividend) {
    // One dividend D=2.0 at t=0.5 (calendar), T=1.0
    // Should produce 2 segments:
    //   Segment 1: τ ∈ [0, 0.5], EEP, payoff IC
    //   Segment 0: τ ∈ (0.5, 1.0], RawPrice, chained IC
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.0,
        .dividends = {{0.5, 2.0}},
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity = 1.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05},
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    // Verify segment count
    // Verify last segment is EEP
    // Verify earlier segment is RawPrice
    // Verify price is reasonable for ATM put
    double price = result->price(100.0, 100.0, 0.8, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
}

TEST(SegmentedPriceTableBuilderTest, BuildWithNoDividends) {
    // No dividends → single EEP segment
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .dividends = {},
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity = 1.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05},
    };

    auto result = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(result.has_value());
}

TEST(SegmentedPriceTableBuilderTest, MoneyGridExpansion) {
    // Large dividend relative to K_ref → grid should expand
    // Verify m_min stays > 0
}

TEST(SegmentedPriceTableBuilderTest, DividendEdgeCases) {
    // Dividend at expiry → ignored
    // Dividend at t=0 → ignored
    // Two dividends same date → merged
}
```

**Step 2: Run tests to verify they fail**

Run: `bazel test //tests:segmented_price_table_builder_test --test_output=all`
Expected: Compilation error

**Step 3: Implement**

The builder:
1. Filters/merges dividends (edge cases from design doc)
2. Sorts dividends, computes segment boundaries in τ
3. Expands moneyness grid by `max(D_i / K_ref)`
4. Builds last segment via `PriceTableBuilder` (standard payoff, EEP mode)
5. For each earlier segment (backward):
   - Creates custom IC lambda from previous segment's surface at τ=0
   - Builds via `PriceTableBuilder` (custom IC, RawPrice mode, τ=0 allowed)
6. Assembles into `SegmentedPriceSurface`

**Step 4: Run tests to verify they pass**

Run: `bazel test //tests:segmented_price_table_builder_test --test_output=all`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/option/table/segmented_price_table_builder.hpp src/option/table/segmented_price_table_builder.cpp src/option/table/BUILD.bazel tests/segmented_price_table_builder_test.cc tests/BUILD.bazel
git commit -m "Add SegmentedPriceTableBuilder for backward-chained construction"
```

---

## Task 7: SegmentedMultiKRefSurface

Wraps multiple `SegmentedPriceSurface` instances with strike interpolation. Satisfies `PriceSurface`.

**Files:**
- Create: `src/option/table/segmented_multi_kref_surface.hpp`
- Create: `src/option/table/segmented_multi_kref_surface.cpp`
- Modify: `src/option/table/BUILD.bazel`
- Test: `tests/segmented_multi_kref_surface_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing tests**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/segmented_multi_kref_surface.hpp"
#include "mango/option/table/price_surface_concept.hpp"

using namespace mango;

static_assert(PriceSurface<SegmentedMultiKRefSurface>,
    "SegmentedMultiKRefSurface must satisfy PriceSurface concept");

TEST(SegmentedMultiKRefSurfaceTest, StrikeInterpolation) {
    // Build with K_refs = {80, 100, 120}
    // Query at strike=90 → interpolate between K_ref=80 and K_ref=100
    // Verify price is between the two endpoint prices
}

TEST(SegmentedMultiKRefSurfaceTest, StrikeClampOutsideRange) {
    // Query at strike=60 (below all K_refs) → clamp to lowest
    // Query at strike=150 (above all K_refs) → clamp to highest
}

TEST(SegmentedMultiKRefSurfaceTest, BoundsIntersection) {
    // Bounds should be intersection across all entries
    // m_min = max of per-entry m_min, m_max = min of per-entry m_max
}

TEST(SegmentedMultiKRefSurfaceTest, VegaInterpolation) {
    // Vega should be interpolated same way as price
}

TEST(SegmentedMultiKRefSurfaceTest, CreateRejectsEmptyEntries) {
    // No entries → validation error
}

TEST(SegmentedMultiKRefSurfaceTest, CreateRejectsDegenerateBoundsIntersection) {
    // Entries whose bounds don't overlap → validation error
}
```

**Step 2: Run tests to verify they fail**

Run: `bazel test //tests:segmented_multi_kref_surface_test --test_output=all`
Expected: Compilation error

**Step 3: Implement**

Create the class with `create()` factory, `price()`, `vega()`, bounds accessors, and bounds intersection logic. Log warning via USDT probe when intersection shrinks domain by >10%.

**Step 4: Run tests to verify they pass**

Run: `bazel test //tests:segmented_multi_kref_surface_test --test_output=all`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/option/table/segmented_multi_kref_surface.hpp src/option/table/segmented_multi_kref_surface.cpp src/option/table/BUILD.bazel tests/segmented_multi_kref_surface_test.cc tests/BUILD.bazel
git commit -m "Add SegmentedMultiKRefSurface with strike interpolation"
```

---

## Task 8: SegmentedMultiKRefBuilder

Top-level builder that chooses K_ref values and assembles the multi-K_ref surface.

**Files:**
- Create: `src/option/table/segmented_multi_kref_builder.hpp`
- Create: `src/option/table/segmented_multi_kref_builder.cpp`
- Modify: `src/option/table/BUILD.bazel`
- Test: `tests/segmented_multi_kref_builder_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing tests**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/segmented_multi_kref_builder.hpp"

using namespace mango;

TEST(SegmentedMultiKRefBuilderTest, BuildWithExplicitKRefs) {
    MultiKRefConfig kref_config{
        .K_refs = {80.0, 100.0, 120.0},
    };
    SegmentedMultiKRefBuilder::Config config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .dividends = {{0.25, 1.50}, {0.50, 1.50}},
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity = 1.0,
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05},
        .kref_config = kref_config,
    };

    auto result = SegmentedMultiKRefBuilder::build(config);
    ASSERT_TRUE(result.has_value());

    // Verify pricing works at various strikes
    double price = result->price(100.0, 95.0, 0.5, 0.20, 0.05);
    EXPECT_GT(price, 0.0);
}

TEST(SegmentedMultiKRefBuilderTest, AutoKRefSelection) {
    MultiKRefConfig kref_config{
        .K_ref_count = 3,
        .K_ref_span = 0.2,
    };
    // ... build and verify 3 entries are created
}
```

**Step 2: Run tests to verify they fail**

Run: `bazel test //tests:segmented_multi_kref_builder_test --test_output=all`
Expected: Compilation error

**Step 3: Implement**

Builder logic: auto K_ref selection (log-spaced around spot), loop over K_refs calling `SegmentedPriceTableBuilder::build()`, assemble into `SegmentedMultiKRefSurface`.

**Step 4: Run tests to verify they pass**

Run: `bazel test //tests:segmented_multi_kref_builder_test --test_output=all`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/option/table/segmented_multi_kref_builder.hpp src/option/table/segmented_multi_kref_builder.cpp src/option/table/BUILD.bazel tests/segmented_multi_kref_builder_test.cc tests/BUILD.bazel
git commit -m "Add SegmentedMultiKRefBuilder for multi-strike discrete dividend tables"
```

---

## Task 9: IVSolverInterpolated — Templatize on PriceSurface concept

Template the solver and replace concrete `AmericanPriceSurface` references with concept-constrained `Surface` parameter.

**Files:**
- Modify: `src/option/iv_solver_interpolated.hpp`
- Modify: `src/option/iv_solver_interpolated.cpp` (may become `.ipp` or inline in header for templates)
- Modify: `src/option/BUILD.bazel`
- Test: `tests/iv_solver_interpolated_test.cc` (extend existing)
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing tests**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/iv_solver_interpolated.hpp"
#include "mango/option/table/segmented_multi_kref_surface.hpp"
#include "mango/option/table/price_surface_concept.hpp"

using namespace mango;

// Verify the solver compiles with both surface types
static_assert(PriceSurface<AmericanPriceSurface>);
static_assert(PriceSurface<SegmentedMultiKRefSurface>);

TEST(IVSolverInterpolatedTest, WorksWithAmericanPriceSurface) {
    // Build standard AmericanPriceSurface (existing path)
    // Create IVSolverInterpolated<AmericanPriceSurface>
    // Solve an IV query
    // Verify result is reasonable
}

TEST(IVSolverInterpolatedTest, WorksWithSegmentedMultiKRefSurface) {
    // Build SegmentedMultiKRefSurface (with discrete dividends)
    // Create IVSolverInterpolated<SegmentedMultiKRefSurface>
    // Solve an IV query
    // Verify result is reasonable
}

TEST(IVSolverInterpolatedTest, BoundsFromConcept) {
    // Verify solver reads bounds from concept accessors
    // Not from eep_surface().axes() directly
}
```

**Step 2: Run tests to verify they fail**

Run: `bazel test //tests:iv_solver_interpolated_test --test_output=all`
Expected: Compilation errors (solver isn't templated yet)

**Step 3: Implement**

Convert `IVSolverInterpolated` to a class template:
```cpp
template <PriceSurface Surface>
class IVSolverInterpolated { ... };
```

Key changes:
- Replace `AmericanPriceSurface american_surface_` with `Surface surface_`
- Replace `eep_surface().axes()` / `metadata()` lookups with `surface_.m_min()`, etc.
- Remove `K_ref_` member (no longer needed)
- Move implementation to header (template) or `.ipp` include
- Add type alias: `using IVSolverInterpolatedStandard = IVSolverInterpolated<AmericanPriceSurface>;`

**Step 4: Run tests to verify they pass**

Run: `bazel test //tests:iv_solver_interpolated_test --test_output=all`
Expected: All PASS

**Step 5: Run full test suite — critical regression check**

Run: `bazel test //...`
Expected: All tests pass. Any test using `IVSolverInterpolated` by name must still compile (type alias covers this).

**Step 6: Commit**

```bash
git add src/option/iv_solver_interpolated.hpp src/option/iv_solver_interpolated.cpp src/option/BUILD.bazel tests/iv_solver_interpolated_test.cc tests/BUILD.bazel
git commit -m "Templatize IVSolverInterpolated on PriceSurface concept"
```

---

## Task 10: User-facing factory — `make_iv_solver`

Single factory function that hides the two paths.

**Files:**
- Create: `src/option/iv_solver_factory.hpp`
- Create: `src/option/iv_solver_factory.cpp`
- Modify: `src/option/BUILD.bazel`
- Test: `tests/iv_solver_factory_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing tests**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/iv_solver_factory.hpp"

using namespace mango;

TEST(IVSolverFactoryTest, NoDividendsUsesStandardPath) {
    IVSolverConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .dividend_yield = 0.02,
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity_grid = {0.1, 0.25, 0.5, 1.0},
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05},
    };

    auto solver = make_iv_solver(config);
    ASSERT_TRUE(solver.has_value());

    IVQuery query{/* ... ATM put ... */};
    auto result = solver->solve(query);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->implied_vol, 0.0);
}

TEST(IVSolverFactoryTest, DiscreteDividendsUsesSegmentedPath) {
    IVSolverConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .dividend_yield = 0.0,
        .discrete_dividends = {{0.25, 1.50}, {0.50, 1.50}},
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity_grid = {0.1, 0.25, 0.5, 1.0},
        .vol_grid = {0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.03, 0.05},
    };

    auto solver = make_iv_solver(config);
    ASSERT_TRUE(solver.has_value());

    IVQuery query{/* ... ATM put ... */};
    auto result = solver->solve(query);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->implied_vol, 0.0);
}
```

**Step 2: Run tests to verify they fail**

Run: `bazel test //tests:iv_solver_factory_test --test_output=all`
Expected: Compilation error

**Step 3: Implement**

The factory returns a type-erased solver (e.g., `std::variant` of the two specializations wrapped in a common solve interface, or a `std::function`-based wrapper). The exact type erasure mechanism depends on what's simplest — a `std::variant<IVSolverInterpolated<AmericanPriceSurface>, IVSolverInterpolated<SegmentedMultiKRefSurface>>` with a `visit` dispatcher is likely cleanest.

**Step 4: Run tests to verify they pass**

Run: `bazel test //tests:iv_solver_factory_test --test_output=all`
Expected: All PASS

**Step 5: Run full test suite**

Run: `bazel test //...`
Expected: All pass

**Step 6: Commit**

```bash
git add src/option/iv_solver_factory.hpp src/option/iv_solver_factory.cpp src/option/BUILD.bazel tests/iv_solver_factory_test.cc tests/BUILD.bazel
git commit -m "Add make_iv_solver factory hiding discrete dividend path"
```

---

## Task 11: Integration test — End-to-end discrete dividend IV

Full integration test exercising the complete pipeline: factory → segmented build → IV solve.

**Files:**
- Test: `tests/discrete_dividend_iv_integration_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write integration test**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/iv_solver_factory.hpp"
#include "mango/option/iv_solver_fdm.hpp"

using namespace mango;

TEST(DiscreteDividendIVIntegrationTest, IVMatchesFDMWithinTolerance) {
    // Build interpolated IV solver with discrete dividends
    IVSolverConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .discrete_dividends = {{0.25, 1.50}},
        .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
        .maturity_grid = {0.1, 0.25, 0.5, 1.0},
        .vol_grid = {0.10, 0.15, 0.20, 0.30, 0.40},
        .rate_grid = {0.02, 0.05},
    };
    auto interp_solver = make_iv_solver(config);
    ASSERT_TRUE(interp_solver.has_value());

    // Compare against FDM IV solver for several queries
    IVSolverFDM fdm_solver;

    std::vector<IVQuery> queries = {
        // ATM, various maturities
        {.spot = 100, .strike = 100, .maturity = 0.5, .rate = 0.05,
         .dividend_yield = 0.0, .type = OptionType::PUT,
         .market_price = 7.0,  // approximate
         .discrete_dividends = {{0.25, 1.50}}},
        // OTM
        {.spot = 100, .strike = 90, .maturity = 0.5, /* ... */},
        // ITM
        {.spot = 100, .strike = 110, .maturity = 0.5, /* ... */},
    };

    for (const auto& query : queries) {
        auto fdm_result = fdm_solver.solve_impl(query);
        auto interp_result = interp_solver->solve(query);

        if (fdm_result.has_value() && interp_result.has_value()) {
            EXPECT_NEAR(interp_result->implied_vol, fdm_result->implied_vol, 0.005)
                << "IV mismatch for strike=" << query.strike
                << " maturity=" << query.maturity;
        }
    }
}

TEST(DiscreteDividendIVIntegrationTest, MultipleQuarterlyDividends) {
    // 4 quarterly dividends, 1-year expiry
    IVSolverConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .discrete_dividends = {
            {0.25, 0.50}, {0.50, 0.50}, {0.75, 0.50}, {1.00, 0.50}
        },
        // ... grids ...
    };
    auto solver = make_iv_solver(config);
    ASSERT_TRUE(solver.has_value());
    // Verify pricing works across the full maturity range
}

TEST(DiscreteDividendIVIntegrationTest, NoDividendPathMatchesExisting) {
    // Regression: factory with no dividends should match existing IVSolverInterpolated
    // Build both ways and compare
}
```

**Step 2: Run test**

Run: `bazel test //tests:discrete_dividend_iv_integration_test --test_output=all`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/discrete_dividend_iv_integration_test.cc tests/BUILD.bazel
git commit -m "Add end-to-end integration test for discrete dividend IV"
```

---

## Task 12: Documentation update

Update API guide and architecture docs to cover discrete dividend support.

**Files:**
- Modify: `docs/API_GUIDE.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `CLAUDE.md` (if patterns section needs updating)

**Step 1: Update API_GUIDE.md**

Add a new section "Discrete Dividend IV" showing the `make_iv_solver` factory usage with dividends.

**Step 2: Update ARCHITECTURE.md**

Add the component hierarchy diagram and mention the segmentation approach.

**Step 3: Commit**

```bash
git add docs/API_GUIDE.md docs/ARCHITECTURE.md
git commit -m "Document discrete dividend support in API guide and architecture"
```

---

## Dependency Graph

```
Task 1 (AmericanPriceSurface RawPrice + bounds)
Task 2 (PriceSurface concept)
Task 3 (AmericanOptionSolver custom IC)
  │
  └──→ Task 4 (PriceTableBuilder IC + RawPrice + τ=0)
         │
         └──→ Task 5 (SegmentedPriceSurface) ←── Task 1
                │
                └──→ Task 6 (SegmentedPriceTableBuilder) ←── Task 4
                       │
                       └──→ Task 7 (SegmentedMultiKRefSurface) ←── Task 2
                              │
                              └──→ Task 8 (SegmentedMultiKRefBuilder)
                                     │
Task 9 (IVSolverInterpolated template) ←── Task 2, Task 7
  │
  └──→ Task 10 (make_iv_solver factory) ←── Task 8, Task 9
         │
         └──→ Task 11 (Integration test) ←── Task 10
                │
                └──→ Task 12 (Documentation)
```

Tasks 1, 2, and 3 can be done in parallel. Tasks 5 and 9 can overlap once their dependencies are met.
