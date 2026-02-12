# Interpolation Surface Greeks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add delta, gamma, theta, rho to all interpolated price surfaces (B-spline, Chebyshev, segmented).

**Architecture:** Generalize the existing vega_weights pattern to a greek_weights(Greek, ...) method on coordinate transforms. First-order Greeks (delta, theta, rho) use weighted sums of interpolant partials. Gamma uses a special formula with eval_second_partial (analytical for B-spline, FD fallback for Chebyshev). EEP layer adds European Greeks analytically. All new Greeks return std::expected.

**Tech Stack:** C++23, GoogleTest, Bazel

**Design doc:** `docs/plans/2026-02-11-interpolation-greeks-design.md`

---

### Task 1: Greek and GreekError types

**Files:**
- Create: `src/option/table/greek_types.hpp`
- Modify: `src/option/table/BUILD.bazel`
- Test: `tests/greek_types_test.cc`

**Step 1: Write the test**

Create `tests/greek_types_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/greek_types.hpp"

using namespace mango;

TEST(GreekTypesTest, EnumValues) {
    // Greek enum has the four first-order types
    EXPECT_NE(static_cast<int>(Greek::Delta), static_cast<int>(Greek::Vega));
    EXPECT_NE(static_cast<int>(Greek::Theta), static_cast<int>(Greek::Rho));
}

TEST(GreekTypesTest, GreekErrorValues) {
    GreekError e1 = GreekError::OutOfDomain;
    GreekError e2 = GreekError::NumericalFailure;
    EXPECT_NE(e1, e2);
}
```

**Step 2: Run test, verify it fails**

```bash
bazel test //tests:greek_types_test --test_output=all
```
Expected: BUILD FAILURE (header doesn't exist)

**Step 3: Create the header**

Create `src/option/table/greek_types.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

namespace mango {

/// First-order Greek type for coordinate transform weight dispatch.
enum class Greek { Delta, Vega, Theta, Rho };

/// Error codes for Greek computation.
enum class GreekError {
    OutOfDomain,        ///< Query point outside surface domain
    NumericalFailure,   ///< FD computation failed (e.g., near boundary)
};

}  // namespace mango
```

Add a `cc_library` target for `greek_types` in `src/option/table/BUILD.bazel` (header-only, no srcs). Add the test target to `tests/BUILD.bazel` with dep on `//src/option/table:greek_types`.

**Step 4: Run test, verify it passes**

```bash
bazel test //tests:greek_types_test --test_output=all
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/option/table/greek_types.hpp src/option/table/BUILD.bazel tests/greek_types_test.cc tests/BUILD.bazel
git commit -m "Add Greek and GreekError enum types"
```

---

### Task 2: Replace vega_weights with greek_weights in transforms

**Files:**
- Modify: `src/option/table/surface_concepts.hpp`
- Modify: `src/option/table/transforms/standard_4d.hpp`
- Modify: `src/option/table/transforms/dimensionless_3d.hpp`
- Test: `tests/surface_concepts_test.cc` (existing)

**Step 1: Write failing test**

Add to existing `tests/surface_concepts_test.cc`:

```cpp
#include "mango/option/table/greek_types.hpp"

TEST(SurfaceConceptsTest, StandardTransform4DGreekWeights) {
    mango::StandardTransform4D xform;
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, rate = 0.05;

    auto dw = xform.greek_weights(mango::Greek::Delta, S, K, tau, sigma, rate);
    EXPECT_NEAR(dw[0], 1.0 / S, 1e-12);  // dx/dS = 1/S
    EXPECT_EQ(dw[1], 0.0);
    EXPECT_EQ(dw[2], 0.0);
    EXPECT_EQ(dw[3], 0.0);

    auto vw = xform.greek_weights(mango::Greek::Vega, S, K, tau, sigma, rate);
    EXPECT_EQ(vw[0], 0.0);
    EXPECT_EQ(vw[1], 0.0);
    EXPECT_EQ(vw[2], 1.0);
    EXPECT_EQ(vw[3], 0.0);

    auto tw = xform.greek_weights(mango::Greek::Theta, S, K, tau, sigma, rate);
    EXPECT_EQ(tw[0], 0.0);
    EXPECT_EQ(tw[1], -1.0);
    EXPECT_EQ(tw[2], 0.0);
    EXPECT_EQ(tw[3], 0.0);

    auto rw = xform.greek_weights(mango::Greek::Rho, S, K, tau, sigma, rate);
    EXPECT_EQ(rw[0], 0.0);
    EXPECT_EQ(rw[1], 0.0);
    EXPECT_EQ(rw[2], 0.0);
    EXPECT_EQ(rw[3], 1.0);
}

TEST(SurfaceConceptsTest, DimensionlessTransform3DGreekWeights) {
    mango::DimensionlessTransform3D xform;
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, rate = 0.05;

    auto dw = xform.greek_weights(mango::Greek::Delta, S, K, tau, sigma, rate);
    EXPECT_NEAR(dw[0], 1.0 / S, 1e-12);
    EXPECT_EQ(dw[1], 0.0);
    EXPECT_EQ(dw[2], 0.0);

    auto vw = xform.greek_weights(mango::Greek::Vega, S, K, tau, sigma, rate);
    EXPECT_EQ(vw[0], 0.0);
    EXPECT_NEAR(vw[1], sigma * tau, 1e-12);
    EXPECT_NEAR(vw[2], -2.0 / sigma, 1e-12);

    auto tw = xform.greek_weights(mango::Greek::Theta, S, K, tau, sigma, rate);
    EXPECT_EQ(tw[0], 0.0);
    EXPECT_NEAR(tw[1], sigma * sigma / 2.0, 1e-12);
    EXPECT_EQ(tw[2], 0.0);

    auto rw = xform.greek_weights(mango::Greek::Rho, S, K, tau, sigma, rate);
    EXPECT_EQ(rw[0], 0.0);
    EXPECT_EQ(rw[1], 0.0);
    EXPECT_NEAR(rw[2], 1.0 / rate, 1e-12);
}
```

**Step 2: Run test, verify it fails**

```bash
bazel test //tests:surface_concepts_test --test_output=all
```
Expected: FAIL — no `greek_weights` method

**Step 3: Implement**

Update `src/option/table/transforms/standard_4d.hpp` — replace `vega_weights` with `greek_weights`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/greek_types.hpp"
#include <array>
#include <cmath>

namespace mango {

struct StandardTransform4D {
    static constexpr size_t kDim = 4;

    [[nodiscard]] std::array<double, 4> to_coords(
        double spot, double strike, double tau, double sigma, double rate) const noexcept {
        return {std::log(spot / strike), tau, sigma, rate};
    }

    [[nodiscard]] std::array<double, 4> greek_weights(
        Greek greek, double spot, double /*strike*/, double /*tau*/,
        double /*sigma*/, double /*rate*/) const noexcept {
        switch (greek) {
            case Greek::Delta: return {1.0 / spot, 0.0, 0.0, 0.0};
            case Greek::Vega:  return {0.0, 0.0, 1.0, 0.0};
            case Greek::Theta: return {0.0, -1.0, 0.0, 0.0};
            case Greek::Rho:   return {0.0, 0.0, 0.0, 1.0};
        }
        __builtin_unreachable();
    }
};

}  // namespace mango
```

Update `src/option/table/transforms/dimensionless_3d.hpp` — replace `vega_weights` with `greek_weights`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/greek_types.hpp"
#include <array>
#include <cmath>

namespace mango {

struct DimensionlessTransform3D {
    static constexpr size_t kDim = 3;

    [[nodiscard]] std::array<double, 3> to_coords(
        double spot, double strike, double tau, double sigma, double rate) const noexcept {
        return {std::log(spot / strike),
                sigma * sigma * tau / 2.0,
                std::log(2.0 * rate / (sigma * sigma))};
    }

    [[nodiscard]] std::array<double, 3> greek_weights(
        Greek greek, double spot, double /*strike*/, double tau,
        double sigma, double rate) const noexcept {
        switch (greek) {
            case Greek::Delta: return {1.0 / spot, 0.0, 0.0};
            case Greek::Vega:  return {0.0, sigma * tau, -2.0 / sigma};
            case Greek::Theta: return {0.0, sigma * sigma / 2.0, 0.0};
            case Greek::Rho:   return {0.0, 0.0, 1.0 / rate};
        }
        __builtin_unreachable();
    }
};

}  // namespace mango
```

Update `src/option/table/surface_concepts.hpp` — replace `vega_weights` with `greek_weights` in the `CoordinateTransform` concept:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/greek_types.hpp"
#include <array>
#include <cstddef>
#include <concepts>

namespace mango {

template <typename S, size_t N>
concept SurfaceInterpolant = requires(const S& s, std::array<double, N> coords) {
    { s.eval(coords) } -> std::same_as<double>;
    { s.partial(size_t{}, coords) } -> std::same_as<double>;
};

template <typename T>
concept CoordinateTransform = requires(const T& t, double spot, double strike,
                                        double tau, double sigma, double rate) {
    { T::kDim } -> std::convertible_to<size_t>;
    { t.to_coords(spot, strike, tau, sigma, rate) }
        -> std::same_as<std::array<double, T::kDim>>;
    { t.greek_weights(Greek{}, spot, strike, tau, sigma, rate) }
        -> std::same_as<std::array<double, T::kDim>>;
};

template <typename E>
concept EEPStrategy = requires(const E& e, double spot, double strike,
                                double tau, double sigma, double rate) {
    { e.european_price(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.european_vega(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
};

}  // namespace mango
```

**Step 4: Update TransformLeaf::vega() to use greek_weights**

In `src/option/table/transform_leaf.hpp`, change the `vega()` method to call `xform_.greek_weights(Greek::Vega, ...)` instead of `xform_.vega_weights(...)`. This maintains backward compatibility while switching to the new API.

```cpp
[[nodiscard]] double vega(double spot, double strike,
                           double tau, double sigma, double rate) const {
    auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
    double raw = interp_.eval(coords);
    if (raw <= 0.0) return 0.0;
    auto w = xform_.greek_weights(Greek::Vega, spot, strike, tau, sigma, rate);
    double v = 0.0;
    for (size_t i = 0; i < Xform::kDim; ++i)
        if (w[i] != 0.0)
            v += w[i] * interp_.partial(i, coords);
    return v * strike / K_ref_;
}
```

**Step 5: Run tests, verify they pass**

```bash
bazel test //tests:surface_concepts_test //tests:chebyshev_surface_test //tests:price_table_surface_test --test_output=errors
```
Expected: All PASS (vega still works through greek_weights)

**Step 6: Commit**

```bash
git add src/option/table/greek_types.hpp src/option/table/surface_concepts.hpp \
    src/option/table/transforms/standard_4d.hpp \
    src/option/table/transforms/dimensionless_3d.hpp \
    src/option/table/transform_leaf.hpp \
    tests/surface_concepts_test.cc
git commit -m "Replace vega_weights with greek_weights in transforms"
```

---

### Task 3: Add eval_second_partial to SharedBSplineInterp

**Files:**
- Modify: `src/option/table/bspline/bspline_surface.hpp` (SharedBSplineInterp)
- Test: `tests/bspline_nd_test.cc` (existing, add test)

**Step 1: Write failing test**

Add to `tests/bspline_nd_test.cc`:

```cpp
TEST(SharedBSplineInterpTest, EvalSecondPartial) {
    // Build a simple 4D spline (reuse existing test helper)
    // Then wrap in SharedBSplineInterp and verify eval_second_partial forwards correctly
    auto spline = /* create a BSplineND<double, 4> via BSplineND::create(...) */;
    auto shared = std::make_shared<const mango::BSplineND<double, 4>>(std::move(*spline));
    mango::SharedBSplineInterp<4> interp(shared);

    std::array<double, 4> coords = {0.0, 0.5, 0.20, 0.05};
    double direct = shared->eval_second_partial(0, coords);
    double via_interp = interp.eval_second_partial(0, coords);
    EXPECT_DOUBLE_EQ(direct, via_interp);
}
```

Note: Use the existing 4D spline construction pattern from `bspline_nd_test.cc` — look for a test that creates a `BSplineND<double, 4>` and reuse that setup.

**Step 2: Run test, verify it fails**

```bash
bazel test //tests:bspline_nd_test --test_output=all --test_filter=SharedBSplineInterpTest*
```
Expected: FAIL — `eval_second_partial` not a member of `SharedBSplineInterp`

**Step 3: Add forwarding method**

In `src/option/table/bspline/bspline_surface.hpp`, add to `SharedBSplineInterp<N>`:

```cpp
[[nodiscard]] double eval_second_partial(size_t axis, const std::array<double, N>& coords) const {
    return spline_->eval_second_partial(axis, coords);
}
```

**Step 4: Run test, verify it passes**

```bash
bazel test //tests:bspline_nd_test --test_output=all --test_filter=SharedBSplineInterpTest*
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/option/table/bspline/bspline_surface.hpp tests/bspline_nd_test.cc
git commit -m "Add eval_second_partial to SharedBSplineInterp"
```

---

### Task 4: Add Greeks to TransformLeaf

This is the core task. TransformLeaf gets a generic `compute_first_order_greek()` and a `compute_gamma()` method, both returning `std::expected<double, GreekError>`.

**Files:**
- Modify: `src/option/table/transform_leaf.hpp`
- Test: `tests/transform_leaf_greeks_test.cc` (new)

**Step 1: Write failing test**

Create `tests/transform_leaf_greeks_test.cc`. Build a BSplineTransformLeaf from a known 4D spline and test that Greeks return expected values. Use a polynomial test surface where derivatives are known analytically.

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/greek_types.hpp"
#include "mango/option/option_spec.hpp"

using namespace mango;

class TransformLeafGreeksTest : public ::testing::Test {
protected:
    // Build a 4D B-spline price table with known ATM put parameters.
    // We test that Greeks have correct signs and are finite.
    void SetUp() override {
        // Use PriceTableBuilder to build a small surface for testing.
        // This is an integration-level test — exact values checked against
        // FDM reference in a separate test.
    }
};

TEST_F(TransformLeafGreeksTest, DeltaHasCorrectSign) {
    // For a put option: delta should be negative
    // For a call option: delta should be positive
    // Test with a BSplineTransformLeaf (no EEP, just leaf)
}

TEST_F(TransformLeafGreeksTest, GammaIsPositive) {
    // Gamma is always positive for vanilla options
}

TEST_F(TransformLeafGreeksTest, ThetaIsFinite) {
    // Theta should be finite and typically negative for puts
}

TEST_F(TransformLeafGreeksTest, RhoHasCorrectSign) {
    // For puts: rho should be negative
    // For calls: rho should be positive
}
```

Detailed test code depends on the builder setup — the implementer should use the pattern from `tests/price_table_4d_integration_test.cc` to build a small surface, extract the `BSplineTransformLeaf` from it, and call the new Greek methods.

**Step 2: Run test, verify it fails**

```bash
bazel test //tests:transform_leaf_greeks_test --test_output=all
```
Expected: FAIL — methods don't exist

**Step 3: Implement TransformLeaf Greeks**

Update `src/option/table/transform_leaf.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/surface_concepts.hpp"
#include "mango/option/table/greek_types.hpp"
#include "mango/option/option_spec.hpp"
#include <algorithm>
#include <cmath>
#include <expected>

namespace mango {

template <typename Interp, CoordinateTransform Xform>
    requires SurfaceInterpolant<Interp, Xform::kDim>
class TransformLeaf {
public:
    TransformLeaf(Interp interp, Xform xform, double K_ref)
        : interp_(std::move(interp))
        , xform_(std::move(xform))
        , K_ref_(K_ref)
    {}

    [[nodiscard]] double price(double spot, double strike,
                                double tau, double sigma, double rate) const {
        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        return std::max(0.0, raw) * strike / K_ref_;
    }

    [[nodiscard]] double vega(double spot, double strike,
                               double tau, double sigma, double rate) const {
        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        if (raw <= 0.0) return 0.0;
        auto w = xform_.greek_weights(Greek::Vega, spot, strike, tau, sigma, rate);
        double v = 0.0;
        for (size_t i = 0; i < Xform::kDim; ++i)
            if (w[i] != 0.0)
                v += w[i] * interp_.partial(i, coords);
        return v * strike / K_ref_;
    }

    /// Compute a first-order Greek (delta, vega, theta, rho).
    /// Returns the leaf contribution only (no European add-back).
    [[nodiscard]] std::expected<double, GreekError>
    greek(Greek g, const PricingParams& params) const {
        double spot = params.spot, strike = params.strike;
        double tau = params.maturity, sigma = params.volatility;
        double rate = get_zero_rate(params.rate, params.maturity);

        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        if (raw <= 0.0) return 0.0;

        auto w = xform_.greek_weights(g, spot, strike, tau, sigma, rate);
        double result = 0.0;
        for (size_t i = 0; i < Xform::kDim; ++i)
            if (w[i] != 0.0)
                result += w[i] * interp_.partial(i, coords);
        return result * strike / K_ref_;
    }

    /// Compute gamma = d²V/dS².
    /// Uses analytical second partial if available, FD fallback otherwise.
    [[nodiscard]] std::expected<double, GreekError>
    gamma(const PricingParams& params) const {
        double spot = params.spot, strike = params.strike;
        double tau = params.maturity, sigma = params.volatility;
        double rate = get_zero_rate(params.rate, params.maturity);

        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        double raw = interp_.eval(coords);
        if (raw <= 0.0) return 0.0;

        double df_dx = interp_.partial(0, coords);
        double d2f_dx2 = compute_second_partial_x(coords);

        // d²V/dS² = (d²f/dx² - df/dx) / S² × strike/K_ref
        return (d2f_dx2 - df_dx) / (spot * spot) * strike / K_ref_;
    }

    [[nodiscard]] const Interp& interpolant() const noexcept { return interp_; }
    [[nodiscard]] double K_ref() const noexcept { return K_ref_; }

    /// Expose raw interpolant value for EEP layer guard
    [[nodiscard]] double raw_value(double spot, double strike,
                                    double tau, double sigma, double rate) const {
        auto coords = xform_.to_coords(spot, strike, tau, sigma, rate);
        return interp_.eval(coords);
    }

private:
    /// Compute d²f/dx² along moneyness axis.
    /// Analytical for interpolants with eval_second_partial, FD otherwise.
    [[nodiscard]] double compute_second_partial_x(
        const std::array<double, Xform::kDim>& coords) const {
        if constexpr (requires { interp_.eval_second_partial(size_t{0}, coords); }) {
            return interp_.eval_second_partial(0, coords);
        } else {
            // Central FD fallback
            double x = coords[0];
            double h = 1e-4;  // Fixed step in log-moneyness
            auto coords_up = coords;
            auto coords_dn = coords;
            coords_up[0] = x + h;
            coords_dn[0] = x - h;
            double f_up = interp_.eval(coords_up);
            double f_dn = interp_.eval(coords_dn);
            double f_mid = interp_.eval(coords);
            return (f_up - 2.0 * f_mid + f_dn) / (h * h);
        }
    }

    Interp interp_;
    Xform xform_;
    double K_ref_;
};

}  // namespace mango
```

**Step 4: Run test, verify it passes**

```bash
bazel test //tests:transform_leaf_greeks_test --test_output=all
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/option/table/transform_leaf.hpp tests/transform_leaf_greeks_test.cc tests/BUILD.bazel
git commit -m "Add greek() and gamma() to TransformLeaf"
```

---

### Task 5: Add European Greeks to AnalyticalEEP and EEPLayer

**Files:**
- Modify: `src/option/table/eep/analytical_eep.hpp`
- Modify: `src/option/table/eep/eep_layer.hpp`
- Modify: `src/option/table/surface_concepts.hpp` (EEPStrategy concept)
- Test: `tests/eep_greeks_test.cc` (new)

**Step 1: Write failing test**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"

using namespace mango;

TEST(AnalyticalEEPTest, EuropeanDeltaMatchesDirect) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, rate = 0.05;

    double eep_delta = eep.european_delta(S, K, tau, sigma, rate);

    // Compare against direct EuropeanOptionSolver
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = S, .strike = K, .maturity = tau,
            .rate = rate, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, sigma).solve().value();

    EXPECT_NEAR(eep_delta, eu.delta(), 1e-12);
}

TEST(AnalyticalEEPTest, EuropeanGammaMatchesDirect) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, rate = 0.05;
    double eep_gamma = eep.european_gamma(S, K, tau, sigma, rate);

    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = S, .strike = K, .maturity = tau,
            .rate = rate, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, sigma).solve().value();

    EXPECT_NEAR(eep_gamma, eu.gamma(), 1e-12);
}

TEST(AnalyticalEEPTest, EuropeanThetaMatchesDirect) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, rate = 0.05;
    double eep_theta = eep.european_theta(S, K, tau, sigma, rate);

    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = S, .strike = K, .maturity = tau,
            .rate = rate, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, sigma).solve().value();

    EXPECT_NEAR(eep_theta, eu.theta(), 1e-12);
}

TEST(AnalyticalEEPTest, EuropeanRhoMatchesDirect) {
    AnalyticalEEP eep(OptionType::PUT, 0.02);
    double S = 100.0, K = 100.0, tau = 1.0, sigma = 0.20, rate = 0.05;
    double eep_rho = eep.european_rho(S, K, tau, sigma, rate);

    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = S, .strike = K, .maturity = tau,
            .rate = rate, .dividend_yield = 0.02,
            .option_type = OptionType::PUT}, sigma).solve().value();

    EXPECT_NEAR(eep_rho, eu.rho(), 1e-12);
}
```

**Step 2: Run test, verify it fails**

```bash
bazel test //tests:eep_greeks_test --test_output=all
```
Expected: FAIL — no `european_delta` etc.

**Step 3: Implement**

Update `src/option/table/eep/analytical_eep.hpp` — add four methods. Each constructs a `EuropeanOptionSolver`, calls `.solve()`, and returns the relevant Greek. Follow the exact pattern of existing `european_price` and `european_vega`:

```cpp
[[nodiscard]] double european_delta(
    double spot, double strike, double tau, double sigma, double rate) const {
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_,
            .option_type = option_type_}, sigma).solve().value();
    return eu.delta();
}

[[nodiscard]] double european_gamma(
    double spot, double strike, double tau, double sigma, double rate) const {
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_,
            .option_type = option_type_}, sigma).solve().value();
    return eu.gamma();
}

[[nodiscard]] double european_theta(
    double spot, double strike, double tau, double sigma, double rate) const {
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_,
            .option_type = option_type_}, sigma).solve().value();
    return eu.theta();
}

[[nodiscard]] double european_rho(
    double spot, double strike, double tau, double sigma, double rate) const {
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = spot, .strike = strike, .maturity = tau,
            .rate = rate, .dividend_yield = dividend_yield_,
            .option_type = option_type_}, sigma).solve().value();
    return eu.rho();
}
```

Update `src/option/table/surface_concepts.hpp` — expand `EEPStrategy` concept:

```cpp
template <typename E>
concept EEPStrategy = requires(const E& e, double spot, double strike,
                                double tau, double sigma, double rate) {
    { e.european_price(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.european_vega(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.european_delta(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.european_gamma(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.european_theta(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
    { e.european_rho(spot, strike, tau, sigma, rate) } -> std::same_as<double>;
};
```

Update `src/option/table/eep/eep_layer.hpp` — add Greek methods with early guard:

```cpp
/// First-order Greek with EEP decomposition.
/// When leaf EEP is zero (deep OTM), returns European Greek only.
[[nodiscard]] std::expected<double, GreekError>
greek(Greek g, const PricingParams& params) const {
    double spot = params.spot, strike = params.strike;
    double tau = params.maturity, sigma = params.volatility;
    double rate = get_zero_rate(params.rate, params.maturity);

    // Early guard: if EEP surface reads zero, return European only
    double raw = leaf_.raw_value(spot, strike, tau, sigma, rate);
    double european = [&] {
        switch (g) {
            case Greek::Delta: return eep_.european_delta(spot, strike, tau, sigma, rate);
            case Greek::Vega:  return eep_.european_vega(spot, strike, tau, sigma, rate);
            case Greek::Theta: return eep_.european_theta(spot, strike, tau, sigma, rate);
            case Greek::Rho:   return eep_.european_rho(spot, strike, tau, sigma, rate);
        }
        __builtin_unreachable();
    }();

    if (raw <= 0.0) return european;

    auto leaf_greek = leaf_.greek(g, params);
    if (!leaf_greek.has_value()) return std::unexpected(leaf_greek.error());
    return *leaf_greek + european;
}

/// Gamma with EEP decomposition.
[[nodiscard]] std::expected<double, GreekError>
gamma(const PricingParams& params) const {
    double spot = params.spot, strike = params.strike;
    double tau = params.maturity, sigma = params.volatility;
    double rate = get_zero_rate(params.rate, params.maturity);

    double raw = leaf_.raw_value(spot, strike, tau, sigma, rate);
    double european_gamma = eep_.european_gamma(spot, strike, tau, sigma, rate);

    if (raw <= 0.0) return european_gamma;

    auto leaf_gamma = leaf_.gamma(params);
    if (!leaf_gamma.has_value()) return std::unexpected(leaf_gamma.error());
    return *leaf_gamma + european_gamma;
}
```

**Step 4: Run test, verify it passes**

```bash
bazel test //tests:eep_greeks_test --test_output=all
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/option/table/eep/analytical_eep.hpp src/option/table/eep/eep_layer.hpp \
    src/option/table/surface_concepts.hpp tests/eep_greeks_test.cc tests/BUILD.bazel
git commit -m "Add European Greeks to AnalyticalEEP and EEPLayer"
```

---

### Task 6: Add Greeks to PriceTable and SplitSurface

**Files:**
- Modify: `src/option/table/price_table.hpp`
- Modify: `src/option/table/split_surface.hpp`
- Modify: `src/option/table/chebyshev/chebyshev_table_builder.hpp` (variant visitor)
- Test: `tests/price_table_greeks_test.cc` (new)

**Step 1: Write failing test**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/standard_surface.hpp"

using namespace mango;

TEST(PriceTableGreeksTest, BSplineDeltaIsNegativeForPut) {
    // Build a small B-spline price table for puts
    // (Reuse pattern from price_table_4d_integration_test.cc)
    // ...build surface...

    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.5,
            .rate = 0.05, .dividend_yield = 0.0,
            .option_type = OptionType::PUT},
        0.20);

    auto delta = surface.delta(params);
    ASSERT_TRUE(delta.has_value()) << static_cast<int>(delta.error());
    EXPECT_LT(*delta, 0.0) << "Put delta should be negative";
    EXPECT_GT(*delta, -1.0) << "Put delta should be > -1";
}

TEST(PriceTableGreeksTest, BSplineGammaIsPositive) {
    // ...build surface...
    PricingParams params(/* ATM put */);
    auto gamma = surface.gamma(params);
    ASSERT_TRUE(gamma.has_value());
    EXPECT_GT(*gamma, 0.0);
}

TEST(PriceTableGreeksTest, BSplineThetaIsNegative) {
    // ...build surface...
    PricingParams params(/* ATM put */);
    auto theta = surface.theta(params);
    ASSERT_TRUE(theta.has_value());
    EXPECT_LT(*theta, 0.0) << "Theta should be negative (time decay)";
}

TEST(PriceTableGreeksTest, BSplineRhoIsNegativeForPut) {
    // ...build surface...
    PricingParams params(/* ATM put */);
    auto rho = surface.rho(params);
    ASSERT_TRUE(rho.has_value());
    EXPECT_LT(*rho, 0.0) << "Put rho should be negative";
}
```

**Step 2: Run test, verify it fails**

```bash
bazel test //tests:price_table_greeks_test --test_output=all
```
Expected: FAIL — no `delta()` etc. on PriceTable

**Step 3: Implement**

Update `src/option/table/price_table.hpp`:

```cpp
[[nodiscard]] std::expected<double, GreekError>
delta(const PricingParams& params) const { return inner_.greek(Greek::Delta, params); }

[[nodiscard]] std::expected<double, GreekError>
gamma(const PricingParams& params) const { return inner_.gamma(params); }

[[nodiscard]] std::expected<double, GreekError>
theta(const PricingParams& params) const { return inner_.greek(Greek::Theta, params); }

[[nodiscard]] std::expected<double, GreekError>
rho(const PricingParams& params) const { return inner_.greek(Greek::Rho, params); }
```

Add `#include "mango/option/table/greek_types.hpp"` to the header.

Update `src/option/table/split_surface.hpp` — add `greek()` and `gamma()` methods following the same bracket/route/combine pattern as `price()` and `vega()`:

```cpp
[[nodiscard]] std::expected<double, GreekError>
greek(Greek g, const PricingParams& params) const {
    double spot = params.spot, strike = params.strike;
    double tau = params.maturity, sigma = params.volatility;
    double rate = get_zero_rate(params.rate, params.maturity);

    auto br = split_.bracket(spot, strike, tau, sigma, rate);
    double result = 0.0;
    for (size_t i = 0; i < br.count; ++i) {
        auto [ls, lk, lt, lv, lr] = split_.to_local(
            br.entries[i].index, spot, strike, tau, sigma, rate);
        // Build local PricingParams for the piece
        PricingParams local_params(
            OptionSpec{.spot = ls, .strike = lk, .maturity = lt,
                .rate = lr, .dividend_yield = params.dividend_yield,
                .option_type = params.option_type},
            lv);
        auto piece_greek = pieces_[br.entries[i].index].greek(g, local_params);
        if (!piece_greek.has_value()) return std::unexpected(piece_greek.error());
        double norm = split_.normalize(br.entries[i].index, strike, *piece_greek);
        result += br.entries[i].weight * norm;
    }
    return split_.denormalize(result, spot, strike, tau, sigma, rate);
}

[[nodiscard]] std::expected<double, GreekError>
gamma(const PricingParams& params) const {
    // Same routing pattern but calls gamma() on pieces
    double spot = params.spot, strike = params.strike;
    double tau = params.maturity, sigma = params.volatility;
    double rate = get_zero_rate(params.rate, params.maturity);

    auto br = split_.bracket(spot, strike, tau, sigma, rate);
    double result = 0.0;
    for (size_t i = 0; i < br.count; ++i) {
        auto [ls, lk, lt, lv, lr] = split_.to_local(
            br.entries[i].index, spot, strike, tau, sigma, rate);
        PricingParams local_params(
            OptionSpec{.spot = ls, .strike = lk, .maturity = lt,
                .rate = lr, .dividend_yield = params.dividend_yield,
                .option_type = params.option_type},
            lv);
        auto piece_gamma = pieces_[br.entries[i].index].gamma(local_params);
        if (!piece_gamma.has_value()) return std::unexpected(piece_gamma.error());
        double norm = split_.normalize(br.entries[i].index, strike, *piece_gamma);
        result += br.entries[i].weight * norm;
    }
    return split_.denormalize(result, spot, strike, tau, sigma, rate);
}
```

Update `src/option/table/chebyshev/chebyshev_table_builder.hpp` — add `delta`, `gamma`, `theta`, `rho` variant visitors to `ChebyshevTableResult`, following the pattern of existing `price` and `vega` methods.

**Step 4: Run tests**

```bash
bazel test //tests:price_table_greeks_test --test_output=all
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/option/table/price_table.hpp src/option/table/split_surface.hpp \
    src/option/table/chebyshev/chebyshev_table_builder.hpp \
    tests/price_table_greeks_test.cc tests/BUILD.bazel
git commit -m "Add Greeks to PriceTable and SplitSurface"
```

---

### Task 7: Integration tests — Greeks accuracy against FDM reference

**Files:**
- Test: `tests/greeks_accuracy_test.cc` (new)

Build a B-spline price table for an ATM put, compute Greeks from the surface, and compare against FDM reference (`AmericanOptionResult::delta()`, `.gamma()`, `.theta()`). European Greeks compared against analytical Black-Scholes.

**Test cases:**
1. ATM put: delta ≈ FDM delta (within 1%)
2. ATM put: gamma ≈ FDM gamma (within 5% — gamma is noisier)
3. ATM put: theta ≈ FDM theta (within 5%)
4. Low dividend case (near-European): all Greeks ≈ Black-Scholes (within 1%)
5. Chebyshev surface: same sign/magnitude tests (looser tolerances for FD derivatives)

**Step 1: Write tests**

```cpp
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/standard_surface.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/european_option.hpp"

using namespace mango;

class GreeksAccuracyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Build a B-spline surface covering typical option parameter ranges
        // Use from_vectors with reasonable grids for moneyness, tau, sigma, rate
        // Store the built BSplinePriceTable as a member
    }

    // Helper: solve single American option via FDM for reference
    AmericanOptionResult solve_fdm(const PricingParams& params) {
        return *solve_american_option(params);
    }
};

TEST_F(GreeksAccuracyTest, DeltaMatchesFDM) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.5,
            .rate = 0.05, .dividend_yield = 0.0,
            .option_type = OptionType::PUT},
        0.20);

    auto surface_delta = surface_.delta(params);
    ASSERT_TRUE(surface_delta.has_value());

    auto fdm = solve_fdm(params);
    double fdm_delta = fdm.delta();

    // Within 1% relative error
    EXPECT_NEAR(*surface_delta, fdm_delta, std::abs(fdm_delta) * 0.01);
}

// Similar tests for gamma, theta
// Plus near-European test comparing against analytical
```

**Step 2: Run tests**

```bash
bazel test //tests:greeks_accuracy_test --test_output=all
```

**Step 3: Commit**

```bash
git add tests/greeks_accuracy_test.cc tests/BUILD.bazel
git commit -m "Add Greeks accuracy tests against FDM reference"
```

---

### Task 8: Full build verification

Run the full CI check:

```bash
bazel test //...
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

Fix any remaining compilation issues (Chebyshev 3D surface with DimensionlessTransform3D, Python bindings, etc.).

**Commit:**

```bash
git commit -m "Fix remaining build issues for Greeks"
```

---

### Summary of tasks

| # | Task | Files | Key change |
|---|------|-------|------------|
| 1 | Greek/GreekError types | new `greek_types.hpp` | Enum definitions |
| 2 | greek_weights in transforms | `standard_4d.hpp`, `dimensionless_3d.hpp`, `surface_concepts.hpp`, `transform_leaf.hpp` | Replace vega_weights |
| 3 | SharedBSplineInterp second partial | `bspline_surface.hpp` | Forward eval_second_partial |
| 4 | TransformLeaf Greeks | `transform_leaf.hpp` | greek() and gamma() methods |
| 5 | EEP layer Greeks | `analytical_eep.hpp`, `eep_layer.hpp`, `surface_concepts.hpp` | European Greeks + composition |
| 6 | PriceTable + SplitSurface | `price_table.hpp`, `split_surface.hpp`, `chebyshev_table_builder.hpp` | Forwarding/routing |
| 7 | Accuracy tests | new `greeks_accuracy_test.cc` | FDM reference comparison |
| 8 | Full build verification | — | CI check |
