# Yield Curve Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add time-varying interest rate support using log-linear discount interpolation.

**Architecture:** YieldCurve class stores tenor points with log-discount values. BlackScholesPDE templates on rate function for zero overhead. RateSpec variant in PricingParams provides backward-compatible API.

**Tech Stack:** C++23, std::expected, std::variant, std::span, GoogleTest

---

## Task 1: Create YieldCurve Core Class

**Files:**
- Create: `src/math/yield_curve.hpp`

**Step 1: Write the failing test**

Create `tests/yield_curve_test.cc`:

```cpp
#include "mango/math/yield_curve.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(YieldCurveTest, FlatCurveReturnsConstantRate) {
    auto curve = mango::YieldCurve::flat(0.05);

    EXPECT_DOUBLE_EQ(curve.rate(0.0), 0.05);
    EXPECT_DOUBLE_EQ(curve.rate(0.5), 0.05);
    EXPECT_DOUBLE_EQ(curve.rate(1.0), 0.05);
    EXPECT_DOUBLE_EQ(curve.rate(10.0), 0.05);
}

TEST(YieldCurveTest, FlatCurveDiscountFactor) {
    auto curve = mango::YieldCurve::flat(0.05);

    // D(t) = exp(-r*t)
    EXPECT_NEAR(curve.discount(0.0), 1.0, 1e-10);
    EXPECT_NEAR(curve.discount(1.0), std::exp(-0.05), 1e-10);
    EXPECT_NEAR(curve.discount(2.0), std::exp(-0.10), 1e-10);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:yield_curve_test --test_output=all`
Expected: BUILD FAILURE - file not found

**Step 3: Write minimal implementation**

Create `src/math/yield_curve.hpp`:

```cpp
/**
 * @file yield_curve.hpp
 * @brief Yield curve with log-linear discount interpolation
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace mango {

/// Point on a yield curve: tenor and log-discount factor
struct TenorPoint {
    double tenor;        // Time in years (0.0, 0.25, 0.5, 1.0, ...)
    double log_discount; // ln(D(t)) where D(t) = exp(-integral_0^t r(s)ds)
};

/// Yield curve with log-linear discount interpolation
///
/// Stores discrete tenor points and interpolates ln(D(t)) linearly.
/// This implies piecewise-constant forward rates between tenors,
/// which is arbitrage-free and industry-standard.
class YieldCurve {
    std::vector<TenorPoint> curve_;  // Sorted by tenor, curve_[0].tenor == 0

public:
    /// Default constructor (empty curve)
    YieldCurve() = default;

    /// Construct flat curve (constant rate)
    static YieldCurve flat(double rate) {
        YieldCurve curve;
        // Two points: t=0 and t=100 (far future)
        // ln(D(t)) = -r*t for flat curve
        curve.curve_.push_back({0.0, 0.0});
        curve.curve_.push_back({100.0, -rate * 100.0});
        return curve;
    }

    /// Instantaneous forward rate at time t
    double rate(double t) const {
        if (curve_.size() < 2) return 0.0;
        if (t <= 0.0) return rate_between(0);

        // Binary search for bracketing interval
        auto it = std::upper_bound(curve_.begin(), curve_.end(), t,
            [](double t, const TenorPoint& p) { return t < p.tenor; });

        if (it == curve_.begin()) return rate_between(0);
        if (it == curve_.end()) return rate_between(curve_.size() - 2);

        size_t idx = static_cast<size_t>(std::distance(curve_.begin(), it)) - 1;
        return rate_between(idx);
    }

    /// Discount factor D(t) = exp(ln_D(t))
    double discount(double t) const {
        return std::exp(log_discount(t));
    }

    /// Log discount factor ln(D(t)) via linear interpolation
    double log_discount(double t) const {
        if (curve_.size() < 2) return 0.0;
        if (t <= 0.0) return 0.0;

        // Binary search for bracketing interval
        auto it = std::upper_bound(curve_.begin(), curve_.end(), t,
            [](double t, const TenorPoint& p) { return t < p.tenor; });

        if (it == curve_.begin()) return 0.0;
        if (it == curve_.end()) {
            // Extrapolate flat beyond last tenor
            const auto& last = curve_.back();
            const auto& prev = curve_[curve_.size() - 2];
            double rate = -(last.log_discount - prev.log_discount) /
                          (last.tenor - prev.tenor);
            return last.log_discount - rate * (t - last.tenor);
        }

        // Linear interpolation
        const auto& right = *it;
        const auto& left = *std::prev(it);
        double alpha = (t - left.tenor) / (right.tenor - left.tenor);
        return left.log_discount + alpha * (right.log_discount - left.log_discount);
    }

private:
    /// Forward rate between curve_[idx] and curve_[idx+1]
    double rate_between(size_t idx) const {
        if (idx + 1 >= curve_.size()) return 0.0;
        const auto& left = curve_[idx];
        const auto& right = curve_[idx + 1];
        double dt = right.tenor - left.tenor;
        if (dt <= 0.0) return 0.0;
        return -(right.log_discount - left.log_discount) / dt;
    }
};

}  // namespace mango
```

**Step 4: Add BUILD target**

Add to `src/math/BUILD.bazel`:

```python
cc_library(
    name = "yield_curve",
    hdrs = ["yield_curve.hpp"],
    visibility = ["//visibility:public"],
)
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "yield_curve_test",
    size = "small",
    srcs = ["yield_curve_test.cc"],
    deps = [
        "//src/math:yield_curve",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:yield_curve_test --test_output=all`
Expected: PASS

**Step 6: Commit**

```bash
git add src/math/yield_curve.hpp src/math/BUILD.bazel tests/yield_curve_test.cc tests/BUILD.bazel
git commit -m "feat: add YieldCurve class with flat curve support"
```

---

## Task 2: Add YieldCurve Factory Methods

**Files:**
- Modify: `src/math/yield_curve.hpp`
- Modify: `tests/yield_curve_test.cc`

**Step 1: Write the failing test**

Add to `tests/yield_curve_test.cc`:

```cpp
TEST(YieldCurveTest, FromPointsCreatesValidCurve) {
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},   // D(1) = exp(-0.05) ~ 0.9512
        {2.0, -0.10}    // D(2) = exp(-0.10) ~ 0.9048
    };

    auto result = mango::YieldCurve::from_points(points);
    ASSERT_TRUE(result.has_value());

    auto& curve = result.value();
    EXPECT_NEAR(curve.discount(0.0), 1.0, 1e-10);
    EXPECT_NEAR(curve.discount(1.0), std::exp(-0.05), 1e-10);
    EXPECT_NEAR(curve.discount(2.0), std::exp(-0.10), 1e-10);
}

TEST(YieldCurveTest, FromPointsFailsWithoutZeroTenor) {
    std::vector<mango::TenorPoint> points = {
        {0.5, -0.025},
        {1.0, -0.05}
    };

    auto result = mango::YieldCurve::from_points(points);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("t=0") != std::string::npos);
}

TEST(YieldCurveTest, FromDiscountsCreatesValidCurve) {
    std::vector<double> tenors = {0.0, 0.5, 1.0, 2.0};
    std::vector<double> discounts = {1.0, 0.9753, 0.9512, 0.9048};

    auto result = mango::YieldCurve::from_discounts(tenors, discounts);
    ASSERT_TRUE(result.has_value());

    auto& curve = result.value();
    EXPECT_NEAR(curve.discount(0.0), 1.0, 1e-10);
    EXPECT_NEAR(curve.discount(0.5), 0.9753, 1e-4);
    EXPECT_NEAR(curve.discount(1.0), 0.9512, 1e-4);
}

TEST(YieldCurveTest, FromDiscountsFailsOnSizeMismatch) {
    std::vector<double> tenors = {0.0, 0.5, 1.0};
    std::vector<double> discounts = {1.0, 0.9753};  // Wrong size

    auto result = mango::YieldCurve::from_discounts(tenors, discounts);
    ASSERT_FALSE(result.has_value());
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:yield_curve_test --test_output=all`
Expected: FAIL - from_points and from_discounts not defined

**Step 3: Write minimal implementation**

Add to `YieldCurve` class in `src/math/yield_curve.hpp`:

```cpp
    /// Construct from tenor points (must include t=0 with log_discount=0)
    static std::expected<YieldCurve, std::string>
    from_points(std::vector<TenorPoint> points) {
        if (points.empty()) {
            return std::unexpected("Empty points vector");
        }

        // Sort by tenor
        std::sort(points.begin(), points.end(),
            [](const TenorPoint& a, const TenorPoint& b) {
                return a.tenor < b.tenor;
            });

        // Check for t=0
        if (points[0].tenor != 0.0) {
            return std::unexpected("First point must have t=0");
        }
        if (std::abs(points[0].log_discount) > 1e-10) {
            return std::unexpected("log_discount at t=0 must be 0");
        }

        YieldCurve curve;
        curve.curve_ = std::move(points);
        return curve;
    }

    /// Construct from discount factors (convenience)
    static std::expected<YieldCurve, std::string>
    from_discounts(std::span<const double> tenors,
                   std::span<const double> discounts) {
        if (tenors.size() != discounts.size()) {
            return std::unexpected("Tenors and discounts must have same size");
        }
        if (tenors.empty()) {
            return std::unexpected("Empty tenors vector");
        }

        std::vector<TenorPoint> points;
        points.reserve(tenors.size());

        for (size_t i = 0; i < tenors.size(); ++i) {
            if (discounts[i] <= 0.0) {
                return std::unexpected("Discount factors must be positive");
            }
            points.push_back({tenors[i], std::log(discounts[i])});
        }

        return from_points(std::move(points));
    }
```

**Step 4: Run test to verify it passes**

Run: `bazel test //tests:yield_curve_test --test_output=all`
Expected: PASS

**Step 5: Commit**

```bash
git add src/math/yield_curve.hpp tests/yield_curve_test.cc
git commit -m "feat: add YieldCurve factory methods from_points and from_discounts"
```

---

## Task 3: Add YieldCurve Rate Interpolation Tests

**Files:**
- Modify: `tests/yield_curve_test.cc`

**Step 1: Write tests for interpolation behavior**

Add to `tests/yield_curve_test.cc`:

```cpp
TEST(YieldCurveTest, RateInterpolation) {
    // Upward sloping curve: 5% for first year, 6% for second year
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},   // 5% for [0,1]
        {2.0, -0.11}    // 6% for [1,2] (total: -0.05 - 0.06 = -0.11)
    };

    auto result = mango::YieldCurve::from_points(points);
    ASSERT_TRUE(result.has_value());
    auto& curve = result.value();

    // Rate in first segment [0,1]: 5%
    EXPECT_NEAR(curve.rate(0.0), 0.05, 1e-10);
    EXPECT_NEAR(curve.rate(0.5), 0.05, 1e-10);
    EXPECT_NEAR(curve.rate(0.99), 0.05, 1e-10);

    // Rate in second segment [1,2]: 6%
    EXPECT_NEAR(curve.rate(1.0), 0.06, 1e-10);
    EXPECT_NEAR(curve.rate(1.5), 0.06, 1e-10);
    EXPECT_NEAR(curve.rate(2.0), 0.06, 1e-10);
}

TEST(YieldCurveTest, RateExtrapolation) {
    auto curve = mango::YieldCurve::flat(0.05);

    // Extrapolation beyond curve should continue flat
    EXPECT_NEAR(curve.rate(50.0), 0.05, 1e-10);
    EXPECT_NEAR(curve.rate(100.0), 0.05, 1e-10);
}

TEST(YieldCurveTest, DiscountInterpolation) {
    // Curve with known discount factors
    std::vector<double> tenors = {0.0, 1.0, 2.0};
    std::vector<double> discounts = {1.0, 0.95, 0.90};

    auto result = mango::YieldCurve::from_discounts(tenors, discounts);
    ASSERT_TRUE(result.has_value());
    auto& curve = result.value();

    // Midpoint: log-linear interpolation
    // ln(D(0.5)) = 0.5 * ln(0.95) = -0.0256
    // D(0.5) = exp(-0.0256) ~ 0.9747
    double expected_d05 = std::exp(0.5 * std::log(0.95));
    EXPECT_NEAR(curve.discount(0.5), expected_d05, 1e-6);
}
```

**Step 2: Run test to verify it passes**

Run: `bazel test //tests:yield_curve_test --test_output=all`
Expected: PASS (implementation already handles these cases)

**Step 3: Commit**

```bash
git add tests/yield_curve_test.cc
git commit -m "test: add YieldCurve interpolation and extrapolation tests"
```

---

## Task 4: Update OptionSpec with RateSpec Variant

**Files:**
- Modify: `src/option/option_spec.hpp`
- Create: `tests/rate_spec_test.cc`

**Step 1: Write the failing test**

Create `tests/rate_spec_test.cc`:

```cpp
#include "mango/option/option_spec.hpp"
#include "mango/math/yield_curve.hpp"
#include <gtest/gtest.h>

TEST(RateSpecTest, DefaultIsDouble) {
    mango::OptionSpec spec;
    EXPECT_TRUE(std::holds_alternative<double>(spec.rate));
    EXPECT_DOUBLE_EQ(std::get<double>(spec.rate), 0.0);
}

TEST(RateSpecTest, CanAssignDouble) {
    mango::OptionSpec spec;
    spec.rate = 0.05;

    EXPECT_TRUE(std::holds_alternative<double>(spec.rate));
    EXPECT_DOUBLE_EQ(std::get<double>(spec.rate), 0.05);
}

TEST(RateSpecTest, CanAssignYieldCurve) {
    mango::OptionSpec spec;
    spec.rate = mango::YieldCurve::flat(0.05);

    EXPECT_TRUE(std::holds_alternative<mango::YieldCurve>(spec.rate));
    auto& curve = std::get<mango::YieldCurve>(spec.rate);
    EXPECT_DOUBLE_EQ(curve.rate(0.5), 0.05);
}

TEST(RateSpecTest, MakeRateFnFromDouble) {
    mango::RateSpec spec = 0.05;
    auto fn = mango::make_rate_fn(spec);

    EXPECT_DOUBLE_EQ(fn(0.0), 0.05);
    EXPECT_DOUBLE_EQ(fn(1.0), 0.05);
    EXPECT_DOUBLE_EQ(fn(10.0), 0.05);
}

TEST(RateSpecTest, MakeRateFnFromCurve) {
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},
        {2.0, -0.11}
    };
    auto curve = mango::YieldCurve::from_points(points).value();
    mango::RateSpec spec = curve;

    auto fn = mango::make_rate_fn(spec);

    EXPECT_NEAR(fn(0.5), 0.05, 1e-10);  // First segment: 5%
    EXPECT_NEAR(fn(1.5), 0.06, 1e-10);  // Second segment: 6%
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:rate_spec_test --test_output=all`
Expected: FAIL - RateSpec not defined

**Step 3: Write minimal implementation**

Modify `src/option/option_spec.hpp`:

Add includes at top:
```cpp
#include <variant>
#include "mango/math/yield_curve.hpp"
```

Add before `OptionSpec` struct:
```cpp
/// Rate specification: constant or yield curve
using RateSpec = std::variant<double, YieldCurve>;

/// Helper to extract rate function from RateSpec
///
/// Returns a callable that takes time t and returns the rate at that time.
/// For constant rate, returns the constant regardless of t.
/// For YieldCurve, delegates to curve.rate(t).
inline auto make_rate_fn(const RateSpec& spec) {
    return std::visit([](const auto& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, double>) {
            return [r = arg](double) { return r; };
        } else {
            // Capture by value to ensure curve lifetime
            return [curve = arg](double t) { return curve.rate(t); };
        }
    }, spec);
}
```

Change `double rate = 0.0;` to `RateSpec rate = 0.0;` in OptionSpec.

**Step 4: Update BUILD.bazel**

Modify `src/option/BUILD.bazel` to add yield_curve dependency:
```python
cc_library(
    name = "option_spec",
    hdrs = ["option_spec.hpp"],
    srcs = ["option_spec.cc"],
    deps = [
        "//src/math:yield_curve",
        "//src/support:error_types",
    ],
    visibility = ["//visibility:public"],
)
```

Add to `tests/BUILD.bazel`:
```python
cc_test(
    name = "rate_spec_test",
    size = "small",
    srcs = ["rate_spec_test.cc"],
    deps = [
        "//src/option:option_spec",
        "//src/math:yield_curve",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:rate_spec_test --test_output=all`
Expected: PASS

**Step 6: Commit**

```bash
git add src/option/option_spec.hpp src/option/BUILD.bazel tests/rate_spec_test.cc tests/BUILD.bazel
git commit -m "feat: add RateSpec variant for constant rate or yield curve"
```

---

## Task 5: Update BlackScholesPDE for Time-Varying Rates

**Files:**
- Modify: `src/pde/operators/black_scholes_pde.hpp`
- Create: `tests/black_scholes_pde_rate_fn_test.cc`

**Step 1: Write the failing test**

Create `tests/black_scholes_pde_rate_fn_test.cc`:

```cpp
#include "mango/pde/operators/black_scholes_pde.hpp"
#include "mango/math/yield_curve.hpp"
#include <gtest/gtest.h>

TEST(BlackScholesPDERateFnTest, ConstantRateFnViaLambda) {
    double sigma = 0.20;
    double d = 0.02;
    double r = 0.05;

    auto rate_fn = [r](double) { return r; };
    mango::operators::BlackScholesPDE pde(sigma, rate_fn, d);

    // L(V) = (sigma^2/2)*V_xx + (r-d-sigma^2/2)*V_x - r*V
    double half_sigma_sq = 0.5 * sigma * sigma;  // 0.02
    double drift = r - d - half_sigma_sq;         // 0.05 - 0.02 - 0.02 = 0.01

    double V_xx = 1.0;
    double V_x = 1.0;
    double V = 1.0;
    double t = 0.5;

    double expected = half_sigma_sq * V_xx + drift * V_x - r * V;
    // = 0.02 * 1 + 0.01 * 1 - 0.05 * 1 = -0.02

    EXPECT_NEAR(pde(V_xx, V_x, V, t), expected, 1e-10);
}

TEST(BlackScholesPDERateFnTest, TimeVaryingRate) {
    double sigma = 0.20;
    double d = 0.02;

    // Rate function: 5% for t < 1, 6% for t >= 1
    auto rate_fn = [](double t) { return t < 1.0 ? 0.05 : 0.06; };
    mango::operators::BlackScholesPDE pde(sigma, rate_fn, d);

    double V_xx = 1.0, V_x = 1.0, V = 1.0;

    // At t=0.5: r=0.05
    double r1 = 0.05;
    double drift1 = r1 - d - 0.5 * sigma * sigma;
    double expected1 = 0.02 * V_xx + drift1 * V_x - r1 * V;
    EXPECT_NEAR(pde(V_xx, V_x, V, 0.5), expected1, 1e-10);

    // At t=1.5: r=0.06
    double r2 = 0.06;
    double drift2 = r2 - d - 0.5 * sigma * sigma;
    double expected2 = 0.02 * V_xx + drift2 * V_x - r2 * V;
    EXPECT_NEAR(pde(V_xx, V_x, V, 1.5), expected2, 1e-10);
}

TEST(BlackScholesPDERateFnTest, WithYieldCurve) {
    double sigma = 0.20;
    double d = 0.02;

    // Create yield curve
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {1.0, -0.05},
        {2.0, -0.11}
    };
    auto curve = mango::YieldCurve::from_points(points).value();
    auto rate_fn = [&curve](double t) { return curve.rate(t); };

    mango::operators::BlackScholesPDE pde(sigma, rate_fn, d);

    double V_xx = 1.0, V_x = 1.0, V = 1.0;

    // At t=0.5: r=0.05 (first segment)
    double expected = 0.02 * V_xx + (0.05 - d - 0.02) * V_x - 0.05 * V;
    EXPECT_NEAR(pde(V_xx, V_x, V, 0.5), expected, 1e-10);
}
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:black_scholes_pde_rate_fn_test --test_output=all`
Expected: FAIL - BlackScholesPDE doesn't accept callable

**Step 3: Modify BlackScholesPDE**

Modify `src/pde/operators/black_scholes_pde.hpp`:

```cpp
/**
 * @file black_scholes_pde.hpp
 * @brief Black-Scholes PDE operator in log-moneyness coordinates
 */

#pragma once

#include <type_traits>
#include <functional>

namespace mango::operators {

/**
 * BlackScholesPDE: Black-Scholes PDE operator in log-moneyness coordinates
 *
 * Implements the Black-Scholes PDE in log-moneyness coordinates x = ln(S/K):
 *   dV/dt = L(V)
 *   L(V) = (sigma^2/2)*d2V/dx2 + (r(t)-d-sigma^2/2)*dV/dx - r(t)*V
 *
 * Supports both constant rate and time-varying rate via callable.
 *
 * @tparam T Scalar type (double)
 * @tparam RateFn Rate function type: double(double) or similar callable
 */
template<typename T = double, typename RateFn = T>
class BlackScholesPDE {
public:
    /**
     * Construct with callable rate function
     *
     * @param sigma Volatility
     * @param rate_fn Rate function: rate_fn(t) -> r(t)
     * @param d Continuous dividend yield
     */
    template<typename Fn,
             typename = std::enable_if_t<std::is_invocable_r_v<T, Fn, double>>>
    BlackScholesPDE(T sigma, Fn&& rate_fn, T d)
        : half_sigma_sq_(T(0.5) * sigma * sigma)
        , dividend_(d)
        , rate_fn_(std::forward<Fn>(rate_fn))
    {}

    /**
     * Construct with constant rate (backward compatible)
     *
     * @param sigma Volatility
     * @param r Constant risk-free rate
     * @param d Continuous dividend yield
     */
    template<typename U = RateFn,
             typename = std::enable_if_t<std::is_same_v<U, T>>>
    BlackScholesPDE(T sigma, T r, T d)
        : half_sigma_sq_(T(0.5) * sigma * sigma)
        , dividend_(d)
        , rate_fn_(r)
    {}

    /**
     * Apply operator with time parameter (for time-varying rate)
     *
     * L(V) = (sigma^2/2)*V_xx + (r(t)-d-sigma^2/2)*V_x - r(t)*V
     *
     * @param d2v_dx2 Second derivative
     * @param dv_dx First derivative
     * @param v Value V
     * @param t Current time
     * @return L(V)
     */
    T operator()(T d2v_dx2, T dv_dx, T v, double t) const {
        T r = get_rate(t);
        T drift = r - dividend_ - half_sigma_sq_;
        return half_sigma_sq_ * d2v_dx2 + drift * dv_dx - r * v;
    }

    /**
     * Apply operator without time (backward compatible, for constant rate)
     */
    T operator()(T d2v_dx2, T dv_dx, T v) const {
        return (*this)(d2v_dx2, dv_dx, v, 0.0);
    }

    // Accessor methods
    T first_derivative_coeff(double t = 0.0) const {
        return get_rate(t) - dividend_ - half_sigma_sq_;
    }
    T second_derivative_coeff() const { return half_sigma_sq_; }
    T discount_rate(double t = 0.0) const { return get_rate(t); }

private:
    T get_rate(double t) const {
        if constexpr (std::is_invocable_v<RateFn, double>) {
            return rate_fn_(t);
        } else {
            return rate_fn_;  // Constant rate
        }
    }

    T half_sigma_sq_;
    T dividend_;
    RateFn rate_fn_;
};

// Deduction guides
template<typename T, typename Fn>
BlackScholesPDE(T, Fn&&, T) -> BlackScholesPDE<T, std::decay_t<Fn>>;

template<typename T>
BlackScholesPDE(T, T, T) -> BlackScholesPDE<T, T>;

}  // namespace mango::operators
```

**Step 4: Update BUILD.bazel**

Add to `tests/BUILD.bazel`:
```python
cc_test(
    name = "black_scholes_pde_rate_fn_test",
    size = "small",
    srcs = ["black_scholes_pde_rate_fn_test.cc"],
    deps = [
        "//src/pde/operators:black_scholes_pde",
        "//src/math:yield_curve",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test to verify it passes**

Run: `bazel test //tests:black_scholes_pde_rate_fn_test --test_output=all`
Expected: PASS

**Step 6: Commit**

```bash
git add src/pde/operators/black_scholes_pde.hpp tests/black_scholes_pde_rate_fn_test.cc tests/BUILD.bazel
git commit -m "feat: add time-varying rate support to BlackScholesPDE"
```

---

## Task 6: Run Full Test Suite and Fix Breakages

**Files:**
- Various files may need updates for backward compatibility

**Step 1: Run full test suite**

Run: `bazel test //... --test_output=errors`

**Step 2: Fix any compilation errors**

Common issues:
- Files using `spec.rate` as double need to use `std::get<double>(spec.rate)` or `make_rate_fn(spec.rate)`
- BlackScholesPDE callers may need time parameter

**Step 3: Run tests again**

Run: `bazel test //... --test_output=errors`
Expected: All tests pass

**Step 4: Commit fixes**

```bash
git add -A
git commit -m "fix: update callers for RateSpec variant and BlackScholesPDE changes"
```

---

## Task 7: Add Integration Test with YieldCurve

**Files:**
- Modify: `tests/american_option_test.cc`

**Step 1: Write integration test**

Add to `tests/american_option_test.cc`:

```cpp
TEST(AmericanOptionTest, PricingWithYieldCurve) {
    // Upward sloping curve
    std::vector<mango::TenorPoint> points = {
        {0.0, 0.0},
        {0.5, -0.025},   // 5% for first 6 months
        {1.0, -0.055}    // 6% for second 6 months
    };
    auto curve = mango::YieldCurve::from_points(points).value();

    mango::PricingParams params;
    params.spot = 100.0;
    params.strike = 100.0;
    params.maturity = 1.0;
    params.volatility = 0.20;
    params.rate = curve;
    params.dividend_yield = 0.02;
    params.type = mango::OptionType::PUT;

    std::pmr::synchronized_pool_resource pool;
    auto [grid_spec, time_domain] = mango::estimate_grid_for_option(params);
    auto workspace = mango::PDEWorkspace::create(grid_spec, &pool).value();

    auto solver = mango::AmericanOptionSolver::create(params, workspace);
    ASSERT_TRUE(solver.has_value());

    auto result = solver->solve();
    ASSERT_TRUE(result.has_value());

    // Price should be positive and reasonable
    double price = result->value();
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, params.strike);  // Put can't exceed strike

    // Compare with flat rate at average (5.5%)
    mango::PricingParams flat_params = params;
    flat_params.rate = 0.055;

    auto flat_workspace = mango::PDEWorkspace::create(grid_spec, &pool).value();
    auto flat_solver = mango::AmericanOptionSolver::create(flat_params, flat_workspace);
    auto flat_result = flat_solver->solve();

    // Prices should be close (within 1% for similar average rate)
    double flat_price = flat_result->value();
    EXPECT_NEAR(price, flat_price, flat_price * 0.01);
}
```

**Step 2: Run test**

Run: `bazel test //tests:american_option_test --test_output=all --test_filter=*YieldCurve*`

**Step 3: Fix any issues and commit**

```bash
git add tests/american_option_test.cc
git commit -m "test: add integration test for American option with yield curve"
```

---

## Task 8: Final Verification

**Step 1: Run full test suite**

Run: `bazel test //... --test_output=errors`
Expected: All tests pass

**Step 2: Build examples**

Run: `bazel build //examples/...`
Expected: All examples build

**Step 3: Build benchmarks**

Run: `bazel build //benchmarks/...`
Expected: All benchmarks build

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup for yield curve support"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Create YieldCurve core class | yield_curve.hpp, BUILD.bazel |
| 2 | Add factory methods | yield_curve.hpp |
| 3 | Add interpolation tests | yield_curve_test.cc |
| 4 | Add RateSpec variant | option_spec.hpp |
| 5 | Update BlackScholesPDE | black_scholes_pde.hpp |
| 6 | Fix backward compatibility | Various |
| 7 | Integration test | american_option_test.cc |
| 8 | Final verification | N/A |
