# EEP Decomposition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decompose American option prices into European + early exercise premium (EEP) for B-spline interpolation, reducing accuracy degradation from the free boundary singularity.

**Architecture:** Store EEP in the price tensor instead of raw American prices. A new `AmericanPriceSurface` wrapper reconstructs full prices at query time using closed-form Black-Scholes. The generic `PriceTableSurface<N>` is unchanged.

**Tech Stack:** C++23, Bazel, GoogleTest, existing `PriceTableBuilder`/`PriceTableSurface` framework, `norm_cdf`/`bs_d1` from `black_scholes_analytics.hpp`.

**Design document:** `docs/plans/2026-01-31-eep-decomposition-design.md`

**Worktree:** `/home/kai/work/mango-option-eep` on branch `feature/eep-decomposition`

---

### Task 1: OptionResult and OptionSolver concepts

**Files:**
- Create: `src/option/option_concepts.hpp`
- Create: `tests/option_concepts_test.cc`
- Modify: `src/option/BUILD.bazel:155` (add new target after `option_spec`)
- Modify: `tests/BUILD.bazel` (add new test target)

**Step 1: Write the concept header**

Create `src/option/option_concepts.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "src/option/option_spec.hpp"
#include <concepts>

namespace mango {

/// An option pricing result that provides value and Greeks
template <typename R>
concept OptionResult = requires(const R& r, double spot_price) {
    { r.value() } -> std::convertible_to<double>;
    { r.value_at(spot_price) } -> std::convertible_to<double>;
    { r.delta() } -> std::convertible_to<double>;
    { r.gamma() } -> std::convertible_to<double>;
    { r.theta() } -> std::convertible_to<double>;
    { r.spot() } -> std::convertible_to<double>;
    { r.strike() } -> std::convertible_to<double>;
    { r.maturity() } -> std::convertible_to<double>;
    { r.volatility() } -> std::convertible_to<double>;
    { r.option_type() } -> std::same_as<OptionType>;
};

/// Result that also provides vega
template <typename R>
concept OptionResultWithVega = OptionResult<R> && requires(const R& r) {
    { r.vega() } -> std::convertible_to<double>;
};

/// A solver whose solve() produces an OptionResult
template <typename S>
concept OptionSolver = requires(const S& solver) {
    { solver.solve() } -> OptionResult;
};

}  // namespace mango
```

**Step 2: Add BUILD target**

Add to `src/option/BUILD.bazel` after the `option_spec` target (after line 164):

```python
cc_library(
    name = "option_concepts",
    hdrs = ["option_concepts.hpp"],
    deps = [
        ":option_spec",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 3: Write test**

Create `tests/option_concepts_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/option_concepts.hpp"
#include "src/option/american_option_result.hpp"

namespace mango {
namespace {

// AmericanOptionResult should satisfy OptionResult
static_assert(OptionResult<AmericanOptionResult>,
    "AmericanOptionResult must satisfy OptionResult concept");

// AmericanOptionResult does NOT satisfy OptionResultWithVega (no vega() yet)
static_assert(!OptionResultWithVega<AmericanOptionResult>,
    "AmericanOptionResult should not satisfy OptionResultWithVega");

TEST(OptionConceptsTest, StaticAssertionsCompile) {
    // If this test compiles, the static_asserts above passed.
    SUCCEED();
}

}  // namespace
}  // namespace mango
```

**Step 4: Add test BUILD target**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "option_concepts_test",
    size = "small",
    srcs = ["option_concepts_test.cc"],
    deps = [
        "//src/option:option_concepts",
        "//src/option:american_option_result",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Build and test**

Run: `bazel test //tests:option_concepts_test --test_output=all`
Expected: PASS (static_asserts compile, trivial test passes)

**Step 6: Commit**

```bash
git add src/option/option_concepts.hpp src/option/BUILD.bazel \
        tests/option_concepts_test.cc tests/BUILD.bazel
git commit -m "Add OptionResult and OptionSolver concepts"
```

---

### Task 2: EuropeanOptionSolver and EuropeanOptionResult

**Files:**
- Create: `src/option/european_option.hpp`
- Create: `src/option/european_option.cpp`
- Create: `tests/european_option_test.cc`
- Modify: `src/option/BUILD.bazel` (add `european_option` target)
- Modify: `tests/BUILD.bazel` (add test target)

**Step 1: Write the failing test**

Create `tests/european_option_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/european_option.hpp"
#include "src/option/option_concepts.hpp"
#include <cmath>

namespace mango {
namespace {

// EuropeanOptionResult must satisfy both concepts
static_assert(OptionResult<EuropeanOptionResult>,
    "EuropeanOptionResult must satisfy OptionResult");
static_assert(OptionResultWithVega<EuropeanOptionResult>,
    "EuropeanOptionResult must satisfy OptionResultWithVega");

// Known value: ATM put, S=K=100, τ=1, σ=0.20, r=0.05, q=0.02
// Reference: Haug "The Complete Guide to Option Pricing Formulas"
// d1 = (ln(1) + (0.05 - 0.02 + 0.02)*1) / 0.20 = 0.25
// d2 = 0.25 - 0.20 = 0.05
// P = 100*exp(-0.05)*N(-0.05) - 100*exp(-0.02)*N(-0.25)
//   = 95.1229*0.4801 - 98.0199*0.4013 = 45.6740 - 39.3354 = 6.3386
TEST(EuropeanOptionTest, ATMPutKnownValue) {
    PricingParams params{100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 0.20};
    auto solver = EuropeanOptionSolver::create(params);
    ASSERT_TRUE(solver.has_value());
    auto result = solver->solve();
    EXPECT_NEAR(result.value(), 6.3386, 0.01);
}

// Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
TEST(EuropeanOptionTest, PutCallParity) {
    PricingParams put_params{100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 0.30};
    PricingParams call_params{100.0, 100.0, 1.0, 0.05, 0.02, OptionType::CALL, 0.30};

    auto put = EuropeanOptionSolver::create(put_params)->solve();
    auto call = EuropeanOptionSolver::create(call_params)->solve();

    double S = 100.0, K = 100.0, r = 0.05, q = 0.02, T = 1.0;
    double parity = S * std::exp(-q * T) - K * std::exp(-r * T);
    EXPECT_NEAR(call.value() - put.value(), parity, 1e-10);
}

// Delta bounds: put delta in [-1, 0], call delta in [0, 1]
TEST(EuropeanOptionTest, DeltaBounds) {
    PricingParams put_params{100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20};
    PricingParams call_params{100.0, 100.0, 1.0, 0.05, 0.0, OptionType::CALL, 0.20};

    auto put = EuropeanOptionSolver::create(put_params)->solve();
    auto call = EuropeanOptionSolver::create(call_params)->solve();

    EXPECT_GE(put.delta(), -1.0);
    EXPECT_LE(put.delta(), 0.0);
    EXPECT_GE(call.delta(), 0.0);
    EXPECT_LE(call.delta(), 1.0);
}

// Gamma is non-negative
TEST(EuropeanOptionTest, GammaNonNegative) {
    PricingParams params{100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20};
    auto result = EuropeanOptionSolver::create(params)->solve();
    EXPECT_GE(result.gamma(), 0.0);
}

// Vega is non-negative
TEST(EuropeanOptionTest, VegaNonNegative) {
    PricingParams params{100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20};
    auto result = EuropeanOptionSolver::create(params)->solve();
    EXPECT_GE(result.vega(), 0.0);
}

// value_at matches value at spot
TEST(EuropeanOptionTest, ValueAtSpot) {
    PricingParams params{100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20};
    auto result = EuropeanOptionSolver::create(params)->solve();
    EXPECT_NEAR(result.value(), result.value_at(100.0), 1e-12);
}

// Greeks vs finite differences
TEST(EuropeanOptionTest, DeltaMatchesFiniteDiff) {
    double S = 100.0, K = 100.0, tau = 1.0, r = 0.05, q = 0.02, sigma = 0.20;
    double eps = 0.01;

    auto p_up = EuropeanOptionSolver::create(
        PricingParams{S + eps, K, tau, r, q, OptionType::PUT, sigma})->solve();
    auto p_dn = EuropeanOptionSolver::create(
        PricingParams{S - eps, K, tau, r, q, OptionType::PUT, sigma})->solve();
    auto p = EuropeanOptionSolver::create(
        PricingParams{S, K, tau, r, q, OptionType::PUT, sigma})->solve();

    double fd_delta = (p_up.value() - p_dn.value()) / (2.0 * eps);
    EXPECT_NEAR(p.delta(), fd_delta, 1e-6);
}

// Accessor consistency
TEST(EuropeanOptionTest, Accessors) {
    PricingParams params{100.0, 95.0, 0.5, 0.04, 0.01, OptionType::CALL, 0.25};
    auto result = EuropeanOptionSolver::create(params)->solve();
    EXPECT_EQ(result.spot(), 100.0);
    EXPECT_EQ(result.strike(), 95.0);
    EXPECT_EQ(result.maturity(), 0.5);
    EXPECT_EQ(result.volatility(), 0.25);
    EXPECT_EQ(result.option_type(), OptionType::CALL);
}

// Edge case: deep ITM put ≈ K*exp(-rT) - S*exp(-qT)
TEST(EuropeanOptionTest, DeepITMPut) {
    PricingParams params{50.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20};
    auto result = EuropeanOptionSolver::create(params)->solve();
    double intrinsic_pv = 100.0 * std::exp(-0.05) - 50.0;
    EXPECT_NEAR(result.value(), intrinsic_pv, 0.5);
}

// Edge case: deep OTM put ≈ 0
TEST(EuropeanOptionTest, DeepOTMPut) {
    PricingParams params{200.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20};
    auto result = EuropeanOptionSolver::create(params)->solve();
    EXPECT_NEAR(result.value(), 0.0, 0.01);
}

// Validation: negative spot rejected
TEST(EuropeanOptionTest, NegativeSpotRejected) {
    PricingParams params{-100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20};
    auto solver = EuropeanOptionSolver::create(params);
    EXPECT_FALSE(solver.has_value());
}

}  // namespace
}  // namespace mango
```

**Step 2: Add test BUILD target**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "european_option_test",
    size = "small",
    srcs = ["european_option_test.cc"],
    deps = [
        "//src/option:european_option",
        "//src/option:option_concepts",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3: Run test to verify it fails**

Run: `bazel test //tests:european_option_test --test_output=all`
Expected: BUILD FAILURE (european_option.hpp does not exist)

**Step 4: Write implementation**

Create `src/option/european_option.hpp` with `EuropeanOptionResult` and `EuropeanOptionSolver`. Uses `norm_cdf`, `bs_d1`, `norm_pdf` from `black_scholes_analytics.hpp`.

Key formulas:
- Put: `P = K·e^(-rτ)·N(-d2) - S·e^(-qτ)·N(-d1)`
- Call: `C = S·e^(-qτ)·N(d1) - K·e^(-rτ)·N(d2)`
- Delta put: `-e^(-qτ)·N(-d1)`, call: `e^(-qτ)·N(d1)`
- Gamma: `e^(-qτ)·φ(d1) / (S·σ·√τ)`
- Vega: `S·e^(-qτ)·√τ·φ(d1)` (same as existing `bs_vega`)
- Theta put: `-S·e^(-qτ)·φ(d1)·σ/(2√τ) + r·K·e^(-rτ)·N(-d2) - q·S·e^(-qτ)·N(-d1)`
- Theta call: `-S·e^(-qτ)·φ(d1)·σ/(2√τ) - r·K·e^(-rτ)·N(d2) + q·S·e^(-qτ)·N(d1)`
- Rho put: `-K·τ·e^(-rτ)·N(-d2)`, call: `K·τ·e^(-rτ)·N(d2)`

`value_at(S_new)` recomputes with `S_new` in place of `spot`.

`EuropeanOptionSolver::create()` delegates to `validate_pricing_params()`.

Create `src/option/european_option.cpp` for the implementation.

**Step 5: Add BUILD target**

Add to `src/option/BUILD.bazel`:

```python
cc_library(
    name = "european_option",
    srcs = ["european_option.cpp"],
    hdrs = ["european_option.hpp"],
    deps = [
        ":option_spec",
        "//src/math:black_scholes_analytics",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 6: Run tests**

Run: `bazel test //tests:european_option_test --test_output=all`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/option/european_option.hpp src/option/european_option.cpp \
        src/option/BUILD.bazel tests/european_option_test.cc tests/BUILD.bazel
git commit -m "Add EuropeanOptionSolver with closed-form Black-Scholes"
```

---

### Task 3: SurfaceContent metadata and PriceTableConfig flag

**Files:**
- Modify: `src/option/table/price_table_metadata.hpp:17-23`
- Modify: `src/option/table/price_table_config.hpp:24-31`
- Modify: `tests/price_table_metadata_test.cc` (add new tests)
- Modify: `tests/price_table_config_test.cc` (add new tests)

**Step 1: Write failing test for metadata**

Add to `tests/price_table_metadata_test.cc`:

```cpp
// SurfaceContent enum defaults to RawPrice
TEST(PriceTableMetadataTest, DefaultContentIsRawPrice) {
    PriceTableMetadata meta;
    EXPECT_EQ(meta.content, SurfaceContent::RawPrice);
}

TEST(PriceTableMetadataTest, CanSetToEEP) {
    PriceTableMetadata meta;
    meta.content = SurfaceContent::EarlyExercisePremium;
    EXPECT_EQ(meta.content, SurfaceContent::EarlyExercisePremium);
}
```

**Step 2: Run to verify failure**

Run: `bazel test //tests:price_table_metadata_test --test_output=all`
Expected: BUILD FAILURE (SurfaceContent not defined)

**Step 3: Add SurfaceContent to metadata**

Modify `src/option/table/price_table_metadata.hpp`. Add before the struct:

```cpp
/// What the surface tensor contains
enum class SurfaceContent : uint8_t {
    RawPrice = 0,              ///< Raw American option prices
    EarlyExercisePremium = 1   ///< P_Am - P_Eu (requires reconstruction)
};
```

Add field to `PriceTableMetadata` struct (after `m_max`):

```cpp
    SurfaceContent content = SurfaceContent::RawPrice;  ///< What tensor stores
```

**Step 4: Add store_eep to PriceTableConfig**

Modify `src/option/table/price_table_config.hpp`. Add field to `PriceTableConfig` struct (after `max_failure_rate`):

```cpp
    bool store_eep = false;  ///< Store early exercise premium instead of raw prices
```

**Step 5: Write config test**

Add to `tests/price_table_config_test.cc`:

```cpp
TEST(PriceTableConfigTest, DefaultStoreEepIsFalse) {
    PriceTableConfig config;
    EXPECT_FALSE(config.store_eep);
}
```

**Step 6: Run tests**

Run: `bazel test //tests:price_table_metadata_test //tests:price_table_config_test --test_output=all`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/option/table/price_table_metadata.hpp src/option/table/price_table_config.hpp \
        tests/price_table_metadata_test.cc tests/price_table_config_test.cc
git commit -m "Add SurfaceContent enum and store_eep config flag"
```

---

### Task 4: EEP subtraction in PriceTableBuilder::extract_tensor

**Files:**
- Modify: `src/option/table/price_table_builder.cpp:436-444` (extract_tensor loop)
- Modify: `src/option/table/price_table_builder.hpp` (add european_option include)
- Modify: `src/option/table/BUILD.bazel:73` (add european_option dep)
- Modify: `tests/price_table_builder_test.cc` (add EEP test)

**Step 1: Write failing test**

Add to `tests/price_table_builder_test.cc`:

```cpp
// EEP mode: tensor values should be non-negative (American >= European)
// and smaller than raw American prices
TEST(PriceTableBuilderTest, EEPModeProducesNonNegativeValues) {
    // Build a small table in EEP mode
    std::vector<double> moneyness = {0.9, 1.0, 1.1};
    std::vector<double> maturity = {0.5, 1.0};
    std::vector<double> volatility = {0.15, 0.25};
    std::vector<double> rate = {0.03, 0.05};

    // Build with store_eep = true
    auto setup_eep = PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, volatility, rate, 100.0,
        GridAccuracyParams{}, OptionType::PUT, 0.0, 0.0);
    ASSERT_TRUE(setup_eep.has_value());
    auto& [builder_eep, axes_eep] = setup_eep.value();
    // TODO: Need way to set store_eep before build — may need config access

    auto result_eep = builder_eep.build(axes_eep);
    ASSERT_TRUE(result_eep.has_value());
    EXPECT_EQ(result_eep->surface->metadata().content,
              SurfaceContent::EarlyExercisePremium);
}
```

Note: The exact test will depend on how `store_eep` is plumbed through to the builder. The `from_vectors` factory creates the config internally. Either add a `store_eep` parameter to factory methods, or set it on the config before calling `build()`. Check `PriceTableConfig` — the builder stores `config_` privately. The factory methods construct the config and pass it to the constructor.

**Step 2: Modify extract_tensor in price_table_builder.cpp**

At line 441-442, the current code is:

```cpp
double normalized_price = spline.eval(log_moneyness[i]);
tensor.view[i, j, σ_idx, r_idx] = K_ref * normalized_price;
```

Add EEP subtraction after line 442, guarded by `config_.store_eep`:

```cpp
double normalized_price = spline.eval(log_moneyness[i]);
double american_price = K_ref * normalized_price;

if (config_.store_eep) {
    // Compute European price for this grid point
    double m = moneyness[i];
    double tau = axes.grids[1][j];
    double sigma = axes.grids[2][σ_idx];
    double rate = axes.grids[3][r_idx];
    double spot = m * K_ref;

    auto eu = EuropeanOptionSolver(PricingParams{
        spot, K_ref, tau, rate, config_.dividend_yield,
        config_.option_type, sigma}).solve();

    double eep_raw = american_price - eu.value();

    // Softplus floor: smooth non-negativity
    constexpr double kSharpness = 100.0;
    tensor.view[i, j, σ_idx, r_idx] =
        std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
} else {
    tensor.view[i, j, σ_idx, r_idx] = american_price;
}
```

Also set metadata content in the `build()` method where metadata is constructed:

```cpp
meta.content = config_.store_eep
    ? SurfaceContent::EarlyExercisePremium
    : SurfaceContent::RawPrice;
```

**Step 3: Add include and BUILD dep**

Add `#include "src/option/european_option.hpp"` to `price_table_builder.cpp`.

Add `"//src/option:european_option"` to deps in `src/option/table/BUILD.bazel:73-89`.

**Step 4: Plumb store_eep through factory methods**

The `from_vectors` and other factory methods construct `PriceTableConfig`. The `store_eep` field needs to be passed through. Add `bool store_eep = false` parameter to `from_vectors`, `from_strikes`, `from_chain`, `from_chain_auto`, `from_chain_auto_profile`. Or alternatively, make the config accessible. Check how `max_failure_rate` is passed — it's a parameter on factory methods. Follow the same pattern.

**Step 5: Run all existing tests plus new test**

Run: `bazel test //tests:price_table_builder_test --test_output=all`
Expected: ALL PASS (existing tests use default `store_eep = false`, behavior unchanged)

**Step 6: Commit**

```bash
git add src/option/table/price_table_builder.cpp src/option/table/price_table_builder.hpp \
        src/option/table/BUILD.bazel tests/price_table_builder_test.cc
git commit -m "Subtract European price in extract_tensor when store_eep enabled"
```

---

### Task 5: AmericanPriceSurface wrapper

**Files:**
- Create: `src/option/table/american_price_surface.hpp`
- Create: `src/option/table/american_price_surface.cpp`
- Create: `tests/american_price_surface_test.cc`
- Modify: `src/option/table/BUILD.bazel` (add target)
- Modify: `tests/BUILD.bazel` (add test target)

**Step 1: Write failing test**

Create `tests/american_price_surface_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/american_price_surface.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/option/table/price_table_axes.hpp"
#include "src/option/table/price_table_metadata.hpp"
#include "src/option/european_option.hpp"
#include <cmath>

namespace mango {
namespace {

// Rejects surface with wrong content type
TEST(AmericanPriceSurfaceTest, RejectsRawPriceSurface) {
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};
    axes.grids[1] = {0.25, 0.5, 1.0, 2.0};
    axes.grids[2] = {0.10, 0.20, 0.30, 0.40};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    std::vector<double> coeffs(5 * 4 * 4 * 4, 1.0);
    PriceTableMetadata meta{.K_ref = 100.0, .content = SurfaceContent::RawPrice};

    auto surface = PriceTableSurface<4>::build(axes, coeffs, meta);
    ASSERT_TRUE(surface.has_value());

    auto result = AmericanPriceSurface::create(*surface, OptionType::PUT);
    EXPECT_FALSE(result.has_value());
}

// Accepts surface with EEP content
TEST(AmericanPriceSurfaceTest, AcceptsEEPSurface) {
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};
    axes.grids[1] = {0.25, 0.5, 1.0, 2.0};
    axes.grids[2] = {0.10, 0.20, 0.30, 0.40};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    std::vector<double> coeffs(5 * 4 * 4 * 4, 0.5);  // Fake EEP values
    PriceTableMetadata meta{
        .K_ref = 100.0,
        .dividend_yield = 0.0,
        .m_min = 0.8,
        .m_max = 1.2,
        .content = SurfaceContent::EarlyExercisePremium
    };

    auto surface = PriceTableSurface<4>::build(axes, coeffs, meta);
    ASSERT_TRUE(surface.has_value());

    auto result = AmericanPriceSurface::create(*surface, OptionType::PUT);
    EXPECT_TRUE(result.has_value());
}

// Price = EEP * (K/K_ref) + P_EU(S, K, tau, sigma, r, q)
TEST(AmericanPriceSurfaceTest, PriceReconstructsCorrectly) {
    // Use a constant-EEP surface to verify reconstruction formula
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};
    axes.grids[1] = {0.25, 0.5, 1.0, 2.0};
    axes.grids[2] = {0.10, 0.20, 0.30, 0.40};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    // Constant EEP = 2.0 (for K_ref = 100)
    std::vector<double> coeffs(5 * 4 * 4 * 4, 2.0);
    PriceTableMetadata meta{
        .K_ref = 100.0,
        .dividend_yield = 0.02,
        .m_min = 0.8,
        .m_max = 1.2,
        .content = SurfaceContent::EarlyExercisePremium
    };

    auto surface = PriceTableSurface<4>::build(axes, coeffs, meta);
    ASSERT_TRUE(surface.has_value());

    auto aps = AmericanPriceSurface::create(*surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, r = 0.05;
    double price = aps->price(S, K, tau, sigma, r);

    // Expected: EEP_interp * (K/K_ref) + P_EU(S, K, tau, sigma, r, q)
    double eep_interp = surface->value()->value({S/K, tau, sigma, r});
    double p_eu = EuropeanOptionSolver(PricingParams{
        S, K, tau, r, 0.02, OptionType::PUT, sigma}).solve().value();
    double expected = eep_interp * (K / 100.0) + p_eu;

    EXPECT_NEAR(price, expected, 1e-10);
}

// Vega matches finite difference of price w.r.t. sigma
TEST(AmericanPriceSurfaceTest, VegaMatchesFiniteDiff) {
    // Build with constant EEP
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};
    axes.grids[1] = {0.25, 0.5, 1.0, 2.0};
    axes.grids[2] = {0.10, 0.20, 0.30, 0.40};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    std::vector<double> coeffs(5 * 4 * 4 * 4, 2.0);
    PriceTableMetadata meta{
        .K_ref = 100.0, .dividend_yield = 0.0,
        .m_min = 0.8, .m_max = 1.2,
        .content = SurfaceContent::EarlyExercisePremium
    };

    auto surface = PriceTableSurface<4>::build(axes, coeffs, meta);
    ASSERT_TRUE(surface.has_value());
    auto aps = AmericanPriceSurface::create(*surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, r = 0.05;
    double eps = 1e-5;
    double p_up = aps->price(S, K, tau, sigma + eps, r);
    double p_dn = aps->price(S, K, tau, sigma - eps, r);
    double fd_vega = (p_up - p_dn) / (2.0 * eps);

    EXPECT_NEAR(aps->vega(S, K, tau, sigma, r), fd_vega, 1e-4);
}

}  // namespace
}  // namespace mango
```

**Step 2: Implement AmericanPriceSurface**

Create `src/option/table/american_price_surface.hpp` and `american_price_surface.cpp`.

Key implementation details:
- `create()`: validate `meta.content == SurfaceContent::EarlyExercisePremium`
- `price()`: `eep * (K/K_ref) + eu.value()` where `eep = surface_->value({m, tau, sigma, r})`
- `delta()`: `(1.0/K_ref) * surface_->partial(0, coords) + eu.delta()`
- `gamma()`: use finite difference of `delta()` since `BSplineND` lacks second derivatives. `(delta(S+eps) - delta(S-eps)) / (2*eps)` with `eps = S * 1e-4`
- `vega()`: `(K/K_ref) * surface_->partial(2, coords) + eu.vega()`
- `theta()`: `(K/K_ref) * surface_->partial(1, coords) + eu.theta()`

Note on `partial(0, ...)`: `PriceTableSurface::partial(0, coords)` already applies the log-moneyness chain rule (divides by `m`), returning `∂E/∂m`. So `delta = (1/K_ref) * surface_->partial(0, coords)` is correct — the `1/K` from `∂m/∂S` combined with `K/K_ref` gives `1/K_ref`.

**Step 3: Add BUILD targets**

Add to `src/option/table/BUILD.bazel`:

```python
cc_library(
    name = "american_price_surface",
    srcs = ["american_price_surface.cpp"],
    hdrs = ["american_price_surface.hpp"],
    deps = [
        ":price_table_surface",
        ":price_table_metadata",
        "//src/option:european_option",
        "//src/option:option_spec",
        "//src/support:error_types",
    ],
    visibility = ["//visibility:public"],
)
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "american_price_surface_test",
    size = "small",
    srcs = ["american_price_surface_test.cc"],
    deps = [
        "//src/option/table:american_price_surface",
        "//src/option/table:price_table_surface",
        "//src/option/table:price_table_axes",
        "//src/option/table:price_table_metadata",
        "//src/option:european_option",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 4: Run tests**

Run: `bazel test //tests:american_price_surface_test --test_output=all`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/option/table/american_price_surface.hpp \
        src/option/table/american_price_surface.cpp \
        src/option/table/BUILD.bazel \
        tests/american_price_surface_test.cc tests/BUILD.bazel
git commit -m "Add AmericanPriceSurface with EEP reconstruction"
```

---

### Task 6: Wire IVSolverInterpolated to AmericanPriceSurface

**Files:**
- Modify: `src/option/iv_solver_interpolated.hpp:60-170`
- Modify: `src/option/iv_solver_interpolated.cpp:66-76,121-214`
- Modify: `src/option/BUILD.bazel:131-153` (add deps)
- Modify: `tests/iv_solver_interpolated_test.cc` (add EEP test)

**Step 1: Write failing test**

Add to `tests/iv_solver_interpolated_test.cc`:

```cpp
// Test that IV solver accepts AmericanPriceSurface
TEST_F(IVSolverInterpolatedTest, AcceptsAmericanPriceSurface) {
    // Build an EEP surface (set metadata content)
    PriceTableMetadata eep_meta = meta_;  // copy fixture's meta
    eep_meta.content = SurfaceContent::EarlyExercisePremium;

    // For this test, fill coefficients with EEP values (American - European)
    // ... (construct surface with EEP content)

    // auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    // auto solver = IVSolverInterpolated::create(aps.value());
    // EXPECT_TRUE(solver.has_value());
}
```

**Step 2: Add create() overload for AmericanPriceSurface**

Add to `iv_solver_interpolated.hpp`:

```cpp
/// Create solver from AmericanPriceSurface (EEP mode)
static std::expected<IVSolverInterpolated, ValidationError> create(
    AmericanPriceSurface surface,
    const IVSolverInterpolatedConfig& config = {});
```

The implementation delegates to `AmericanPriceSurface::price()` and `AmericanPriceSurface::vega()` instead of raw `surface_->value()` and `surface_->partial()`.

Implementation approach: store an `std::optional<AmericanPriceSurface>` in the solver. When present, `eval_price()` and `compute_vega()` delegate to it. When absent, use the existing raw surface path.

**Step 3: Modify eval_price and compute_vega**

In `iv_solver_interpolated.cpp`, modify `eval_price()` (line 66-70):

```cpp
double IVSolverInterpolated::eval_price(
    double moneyness, double maturity, double vol, double rate, double strike) const {
    if (american_surface_) {
        double spot = moneyness * strike;
        return american_surface_->price(spot, strike, maturity, vol, rate);
    }
    // Existing raw surface path
    double price_Kref = surface_->value({moneyness, maturity, vol, rate});
    return price_Kref * (strike / K_ref_);
}
```

Similarly for `compute_vega()` (line 72-76):

```cpp
double IVSolverInterpolated::compute_vega(
    double moneyness, double maturity, double vol, double rate, double strike) const {
    if (american_surface_) {
        double spot = moneyness * strike;
        return american_surface_->vega(spot, strike, maturity, vol, rate);
    }
    double vega_Kref = surface_->partial(2, {moneyness, maturity, vol, rate});
    return vega_Kref * (strike / K_ref_);
}
```

**Step 4: Add deps**

Add `"//src/option/table:american_price_surface"` to `src/option/BUILD.bazel` iv_solver_interpolated deps.

**Step 5: Run all IV solver tests**

Run: `bazel test //tests:iv_solver_interpolated_test --test_output=all`
Expected: ALL PASS (existing tests unchanged, new test passes)

**Step 6: Commit**

```bash
git add src/option/iv_solver_interpolated.hpp src/option/iv_solver_interpolated.cpp \
        src/option/BUILD.bazel tests/iv_solver_interpolated_test.cc
git commit -m "Wire IVSolverInterpolated to AmericanPriceSurface"
```

---

### Task 7: Workspace serialization format version bump

**Files:**
- Modify: `src/option/table/price_table_workspace.cpp:258-263,649-659` (save/load)
- Modify: `tests/price_table_workspace_test.cc` (add version test)

**Step 1: Write failing test**

Add to `tests/price_table_workspace_test.cc`:

```cpp
// Save with EEP content, reload, verify content field preserved
TEST(PriceTableWorkspaceTest, RoundTripSurfaceContent) {
    // Create workspace, save, load, check content == EarlyExercisePremium
    // (test will depend on the save/load API specifics)
}

// Load old format (version 1): content defaults to RawPrice
TEST(PriceTableWorkspaceTest, OldFormatDefaultsToRawPrice) {
    // Load a v1 file, verify content == RawPrice
}
```

**Step 2: Modify save path**

In `price_table_workspace.cpp` around line 258-263, add `surface_content` field to Arrow schema:

```cpp
arrow::field("surface_content", arrow::uint8()),  // SurfaceContent enum
```

Add builder and append the value during save.

**Step 3: Modify load path**

In `price_table_workspace.cpp` around line 649-659, after format version check:

- Accept `format_version == 1` or `format_version == 2`
- If version 1: default `content = SurfaceContent::RawPrice`
- If version 2: read `surface_content` field
- Write version 2 on save

**Step 4: Run tests**

Run: `bazel test //tests:price_table_workspace_test --test_output=all`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/option/table/price_table_workspace.cpp \
        tests/price_table_workspace_test.cc
git commit -m "Bump workspace format to v2 with SurfaceContent field"
```

---

### Task 8: Discrete dividend TODO markers

**Files:**
- Modify: `src/pde/operators/black_scholes_pde.hpp` (add TODO)
- Modify: `src/option/american_option_batch.cpp` (add TODO at rejection site)
- Modify: `src/option/table/price_table_builder.cpp` (add TODO for segmentation)

**Step 1: Add TODO comments**

In `src/pde/operators/black_scholes_pde.hpp`, near the constructor where `d` (dividend yield) is set:

```cpp
// TODO(discrete-dividends): Add discrete dividend handling — adjust spot
// at dividend dates via temporal event callbacks in PDESolver time-stepping.
```

In `src/option/american_option_batch.cpp`, at the rejection site (around line 56-61):

```cpp
// TODO(discrete-dividends): Remove this rejection once discrete dividend
// support is implemented via maturity segmentation with per-segment tables.
```

In `src/option/table/price_table_builder.cpp`, at the top of `build()`:

```cpp
// TODO(discrete-dividends): Segment maturity axis at dividend dates,
// building a separate table for each segment. Each segment uses continuous
// dividend yield only, so EEP decomposition applies per-segment.
```

**Step 2: Run all tests to verify nothing broken**

Run: `bazel test //...`
Expected: ALL PASS (comments only)

**Step 3: Commit**

```bash
git add src/pde/operators/black_scholes_pde.hpp \
        src/option/american_option_batch.cpp \
        src/option/table/price_table_builder.cpp
git commit -m "Add TODO markers for discrete dividend support"
```

---

### Task 9: End-to-end integration test

**Files:**
- Create: `tests/eep_integration_test.cc`
- Modify: `tests/BUILD.bazel` (add test target)

**Step 1: Write integration test**

Create `tests/eep_integration_test.cc`. This test builds a price table in both raw and EEP mode, runs IV queries, and compares accuracy against FDM ground truth.

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/american_price_surface.hpp"
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/european_option.hpp"

namespace mango {
namespace {

// Build table, reconstruct American price, compare to PDE
TEST(EEPIntegrationTest, ReconstructedPriceMatchesPDE) {
    // Build small EEP table
    std::vector<double> moneyness = {0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15};
    std::vector<double> maturity = {0.25, 0.5, 1.0};
    std::vector<double> volatility = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.03, 0.05};

    // Build EEP table (store_eep = true)
    auto setup = PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, volatility, rate, 100.0,
        GridAccuracyParams{}, OptionType::PUT, 0.0, 0.0 /*, store_eep=true*/);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = setup.value();
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());

    // Verify metadata
    EXPECT_EQ(result->surface->metadata().content,
              SurfaceContent::EarlyExercisePremium);

    // Wrap in AmericanPriceSurface
    auto aps = AmericanPriceSurface::create(result->surface, OptionType::PUT);
    ASSERT_TRUE(aps.has_value());

    // Compare reconstructed price to direct PDE solve at a few points
    // (tolerance should be within a few cents for 100-strike options)
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, r = 0.05;
    double reconstructed = aps->price(S, K, tau, sigma, r);

    // Direct PDE solve for comparison
    PricingParams params{S, K, tau, r, 0.0, OptionType::PUT, sigma};
    auto [grid_spec, time_domain] = estimate_grid_for_option(params);
    std::pmr::synchronized_pool_resource pool;
    auto ws = PDEWorkspace::create(grid_spec, &pool);
    ASSERT_TRUE(ws.has_value());
    AmericanOptionSolver solver(params, std::move(ws.value()));
    auto pde_result = solver.solve();
    ASSERT_TRUE(pde_result.has_value());

    // Reconstructed should be close to PDE result
    // Tolerance: ~1% of price (we're testing reconstruction, not grid accuracy)
    EXPECT_NEAR(reconstructed, pde_result->value(), pde_result->value() * 0.01);
}

// EEP values are non-negative in tensor
TEST(EEPIntegrationTest, SoftplusFloorEnsuresNonNegative) {
    std::vector<double> moneyness = {0.90, 1.00, 1.10};
    std::vector<double> maturity = {0.1, 0.5, 1.0};  // Short τ included
    std::vector<double> volatility = {0.15, 0.25};
    std::vector<double> rate = {0.05};

    auto setup = PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, volatility, rate, 100.0,
        GridAccuracyParams{}, OptionType::PUT, 0.0, 0.0 /*, store_eep=true*/);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = setup.value();
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());

    // Query EEP surface directly — all values should be non-negative
    auto& surface = result->surface;
    for (double m : moneyness) {
        for (double tau : maturity) {
            for (double sigma : volatility) {
                for (double r : rate) {
                    double eep = surface->value({m, tau, sigma, r});
                    EXPECT_GE(eep, 0.0) << "Negative EEP at m=" << m
                        << " tau=" << tau << " sigma=" << sigma << " r=" << r;
                }
            }
        }
    }
}

}  // namespace
}  // namespace mango
```

**Step 2: Add BUILD target**

```python
cc_test(
    name = "eep_integration_test",
    size = "medium",
    srcs = ["eep_integration_test.cc"],
    deps = [
        "//src/option/table:price_table_builder",
        "//src/option/table:american_price_surface",
        "//src/option:iv_solver_interpolated",
        "//src/option:european_option",
        "//src/option:american_option",
        "//src/pde/core:pde_workspace",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
    copts = ["-fopenmp"],
    linkopts = ["-fopenmp"],
)
```

**Step 3: Run**

Run: `bazel test //tests:eep_integration_test --test_output=all`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/eep_integration_test.cc tests/BUILD.bazel
git commit -m "Add EEP end-to-end integration test"
```

---

### Task 10: Run full test suite and verify nothing broken

**Step 1: Run all tests**

Run: `bazel test //...`
Expected: ALL PASS — no regressions. Existing tests use `store_eep = false` (default), so behavior is unchanged.

**Step 2: Build benchmarks and Python bindings**

Run: `bazel build //benchmarks/... //src/python:mango_option`
Expected: BUILD SUCCESS

**Step 3: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "Fix any test/build issues from EEP integration"
```

---

## Task Dependency Graph

```
Task 1 (concepts) ─────────────┐
                                ├─→ Task 5 (AmericanPriceSurface) ─→ Task 6 (IV solver) ─→ Task 9 (integration)
Task 2 (European solver) ──────┤                                                            │
                                ├─→ Task 4 (builder extract_tensor) ────────────────────────┘
Task 3 (metadata + config) ────┘                                                            │
                                                                                             │
Task 7 (workspace serialization) ──── independent                                           │
Task 8 (TODO markers) ──────────────── independent                                          │
                                                                                             ↓
                                                                                    Task 10 (full suite)
```

Tasks 1, 2, 3 can run in parallel. Tasks 7 and 8 are independent and can run at any point.
