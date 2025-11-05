# Implied Volatility Calculation - C++20 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Brent's method root finder and IV solver for calculating implied volatility from American option market prices.

**Architecture:** Two-layer design: (1) Generic Brent's method template in `src/cpp/brent.hpp` for root-finding, (2) Domain-specific `IVSolver` class in `src/cpp/iv_solver.hpp` that uses Brent + AmericanOptionSolver. Includes comprehensive validation, adaptive bounds optimization, vega computation, and USDT tracing.

**Tech Stack:** C++20, GoogleTest, USDT (systemtap-sdt-dev), Bazel

**Prerequisites:**
- `AmericanOptionSolver` must be implemented first (see `docs/plans/2025-11-04-american-option-cpp-design.md`)
- Verify `src/cpp/root_finding.hpp` exists with `RootFindingConfig` and `RootFindingResult`

---

## Task 1: Add USDT Trace Probes for IV and Brent

**Files:**
- Modify: `src/mango_trace.h` (add IV and Brent probes)

**Step 1: Add MODULE_IV_SOLVER constant**

Add to `src/mango_trace.h` after other MODULE_* definitions:

```c
#define MODULE_IV_SOLVER 5
```

**Step 2: Add IV-specific USDT probes**

Add to `src/mango_trace.h` before the closing `#endif`:

```c
// IV-specific probes
#define MANGO_TRACE_IV_START(module_id, spot, strike, market_price) \
    DTRACE_PROBE4(MANGO_PROVIDER, iv_start, module_id, spot, strike, market_price)

#define MANGO_TRACE_IV_COMPLETE(module_id, converged, iv, iterations) \
    DTRACE_PROBE4(MANGO_PROVIDER, iv_complete, module_id, converged, iv, iterations)

#define MANGO_TRACE_IV_VALIDATION_ERROR(module_id, error_msg) \
    DTRACE_PROBE2(MANGO_PROVIDER, iv_validation_error, module_id, error_msg)

// Brent-specific probes
#define MANGO_TRACE_BRENT_START(module_id, a, b) \
    DTRACE_PROBE3(MANGO_PROVIDER, brent_start, module_id, a, b)

#define MANGO_TRACE_BRENT_ITER(module_id, iter, x, fx) \
    DTRACE_PROBE4(MANGO_PROVIDER, brent_iter, module_id, iter, x, fx)

#define MANGO_TRACE_BRENT_COMPLETE(module_id, iterations, root) \
    DTRACE_PROBE3(MANGO_PROVIDER, brent_complete, module_id, iterations, root)
```

**Step 3: Verify compilation**

```bash
bazel build //src:pde_solver
```

Expected: SUCCESS (no new compilation errors)

**Step 4: Commit**

```bash
git add src/mango_trace.h
git commit -m "feat(iv): add USDT trace probes for IV and Brent"
```

---

## Task 2: Implement Brent's Method Root Finder (Part 1: Structure)

**Files:**
- Create: `src/cpp/brent.hpp`
- Create: `tests/brent_test.cc`

**Step 1: Write failing test for simple polynomial root**

Create `tests/brent_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/cpp/brent.hpp"
#include "src/cpp/root_finding.hpp"
#include <cmath>
#include <limits>

namespace mango {
namespace {

TEST(BrentTest, SimplePolynomial) {
    // f(x) = x^2 - 4, root at x = 2
    auto f = [](double x) { return x * x - 4.0; };
    RootFindingConfig config{};
    config.max_iter = 100;
    config.tolerance = 1e-6;
    config.brent_tol_abs = 1e-6;

    auto result = brent_find_root(f, 0.0, 5.0, config);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 2.0, 1e-6);
    EXPECT_NEAR(result.f_root, 0.0, 1e-6);
    EXPECT_LT(result.iterations, 100);
}

}  // namespace
}  // namespace mango
```

**Step 2: Add test target to BUILD file**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "brent_test",
    size = "small",
    srcs = ["brent_test.cc"],
    deps = [
        "//src:pde_solver",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3: Run test to verify it fails**

```bash
bazel test //tests:brent_test --test_output=errors
```

Expected: FAIL with "brent.hpp: No such file or directory"

**Step 4: Create minimal brent.hpp header structure**

Create `src/cpp/brent.hpp`:

```cpp
#pragma once

#include "src/cpp/root_finding.hpp"
#include "src/mango_trace.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace mango {

// Concept: Callable that takes double and returns double
template<typename Fn>
concept BrentObjective = requires(Fn fn, double x) {
    { fn(x) } -> std::convertible_to<double>;
};

// Result structure extending RootFindingResult
struct BrentResult : RootFindingResult {
    double root = 0.0;      // Found root
    double f_root = 0.0;    // Function value at root
};

// Forward declaration - implementation follows
template<BrentObjective Fn>
BrentResult brent_find_root(Fn&& objective,
                           double a, double b,
                           const RootFindingConfig& config);

}  // namespace mango
```

**Step 5: Run test again**

```bash
bazel test //tests:brent_test --test_output=errors
```

Expected: FAIL with linker error (undefined reference to brent_find_root)

**Step 6: Commit structure**

```bash
git add src/cpp/brent.hpp tests/brent_test.cc tests/BUILD.bazel
git commit -m "feat(iv): add Brent method structure and first test"
```

---

## Task 3: Implement Brent's Method Root Finder (Part 2: Algorithm)

**Files:**
- Modify: `src/cpp/brent.hpp` (add full implementation)

**Step 1: Implement complete Brent's algorithm**

Add to `src/cpp/brent.hpp` after the forward declaration:

```cpp
template<BrentObjective Fn>
BrentResult brent_find_root(Fn&& objective,
                           double a, double b,
                           const RootFindingConfig& config) {
    BrentResult result{};

    // Trace start
    MANGO_TRACE_BRENT_START(MODULE_IV_SOLVER, a, b);

    // Evaluate at endpoints
    double fa = objective(a);
    double fb = objective(b);

    // Check for NaN
    if (std::isnan(fa) || std::isnan(fb)) {
        result.converged = false;
        result.failure_reason = "Objective returned NaN at bracket endpoints";
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IV_SOLVER,
            "Brent: NaN at endpoints");
        return result;
    }

    // Check bracketing: f(a) and f(b) must have opposite signs
    if (fa * fb > 0.0) {
        result.converged = false;
        result.failure_reason = "Root not bracketed: f(a) and f(b) same sign";
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IV_SOLVER,
            "Brent: root not bracketed");
        return result;
    }

    // Ensure |f(a)| >= |f(b)| (swap if needed)
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a;
    double fc = fa;
    bool use_bisection = true;
    double d = 0.0;  // Previous step size

    for (size_t iter = 0; iter < config.max_iter; ++iter) {
        // Check convergence: function value tolerance
        if (std::abs(fb) < config.brent_tol_abs) {
            result.root = b;
            result.f_root = fb;
            result.converged = true;
            result.iterations = iter;
            result.final_error = std::abs(fb);
            MANGO_TRACE_BRENT_COMPLETE(MODULE_IV_SOLVER, iter, b);
            return result;
        }

        // Check convergence: interval tolerance
        if (std::abs(b - a) < config.brent_tol_abs) {
            result.root = b;
            result.f_root = fb;
            result.converged = true;
            result.iterations = iter;
            result.final_error = std::abs(b - a);
            MANGO_TRACE_BRENT_COMPLETE(MODULE_IV_SOLVER, iter, b);
            return result;
        }

        double s;  // New estimate

        // Try inverse quadratic interpolation if three distinct points
        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method (two points)
            s = b - fb * (b - a) / (fb - fa);
        }

        // Determine if interpolation result is acceptable
        double lower = std::min(b, (3.0 * a + b) / 4.0);
        double upper = std::max(b, (3.0 * a + b) / 4.0);

        bool cond1 = (s < lower || s > upper);
        bool cond2 = use_bisection && (std::abs(s - b) >= std::abs(b - c) / 2.0);
        bool cond3 = !use_bisection && (std::abs(s - b) >= std::abs(c - d) / 2.0);
        bool cond4 = use_bisection && (std::abs(b - c) < config.brent_tol_abs);
        bool cond5 = !use_bisection && (std::abs(c - d) < config.brent_tol_abs);

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            // Use bisection
            s = (a + b) / 2.0;
            use_bisection = true;
        } else {
            use_bisection = false;
        }

        // Evaluate at new point
        double fs = objective(s);

        // Check for NaN during iteration
        if (std::isnan(fs)) {
            result.converged = false;
            result.failure_reason = "Objective returned NaN during iteration";
            MANGO_TRACE_RUNTIME_ERROR(MODULE_IV_SOLVER,
                "Brent: NaN during iteration");
            return result;
        }

        // Trace iteration
        MANGO_TRACE_BRENT_ITER(MODULE_IV_SOLVER, iter, s, fs);

        // Update for next iteration
        d = c;
        c = b;
        fc = fb;

        // Update bracket
        if (fa * fs < 0.0) {
            // Root in [a, s]
            b = s;
            fb = fs;
        } else {
            // Root in [s, b]
            a = s;
            fa = fs;
        }

        // Ensure |f(a)| >= |f(b)|
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }

    // Max iterations reached
    result.root = b;
    result.f_root = fb;
    result.converged = false;
    result.iterations = config.max_iter;
    result.final_error = std::abs(fb);
    result.failure_reason = "Max iterations reached";

    MANGO_TRACE_CONVERGENCE_FAILED(MODULE_IV_SOLVER, config.max_iter,
        std::abs(fb));

    return result;
}
```

**Step 2: Run test to verify it passes**

```bash
bazel test //tests:brent_test --test_output=all
```

Expected: PASS

**Step 3: Commit implementation**

```bash
git add src/cpp/brent.hpp
git commit -m "feat(iv): implement complete Brent's method algorithm"
```

---

## Task 4: Add Comprehensive Brent Tests

**Files:**
- Modify: `tests/brent_test.cc` (add edge case tests)

**Step 1: Add test for root not bracketed**

Add to `tests/brent_test.cc`:

```cpp
TEST(BrentTest, NotBracketed) {
    // f(x) = x^2 + 1, no real roots
    auto f = [](double x) { return x * x + 1.0; };
    RootFindingConfig config{};
    config.max_iter = 100;
    config.brent_tol_abs = 1e-6;

    auto result = brent_find_root(f, 0.0, 5.0, config);

    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.failure_reason.has_value());
    EXPECT_EQ(result.failure_reason.value(),
              "Root not bracketed: f(a) and f(b) same sign");
}

TEST(BrentTest, RootOutsideBracket) {
    // f(x) = x - 10, root at x=10, but bracket [0, 5]
    auto f = [](double x) { return x - 10.0; };
    RootFindingConfig config{};
    config.max_iter = 100;
    config.brent_tol_abs = 1e-6;

    auto result = brent_find_root(f, 0.0, 5.0, config);

    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.failure_reason.value(),
              "Root not bracketed: f(a) and f(b) same sign");
}

TEST(BrentTest, MaxIterations) {
    // Pathological function requiring many iterations
    auto f = [](double x) { return std::sin(100.0 * x); };
    RootFindingConfig config{};
    config.max_iter = 5;  // Force failure
    config.brent_tol_abs = 1e-6;

    auto result = brent_find_root(f, 0.0, 0.1, config);

    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.iterations, 5);
    EXPECT_EQ(result.failure_reason.value(), "Max iterations reached");
}

TEST(BrentTest, NaNFromObjective) {
    // Function returns NaN
    auto f = [](double x) {
        return (x < 0.5) ? std::sqrt(x)
                         : std::numeric_limits<double>::quiet_NaN();
    };
    RootFindingConfig config{};
    config.max_iter = 100;
    config.brent_tol_abs = 1e-6;

    auto result = brent_find_root(f, 0.0, 1.0, config);

    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.failure_reason->find("NaN") != std::string::npos);
}

TEST(BrentTest, RootAtBoundary) {
    // Root exactly at lower bound
    auto f = [](double x) { return x - 0.01; };
    RootFindingConfig config{};
    config.max_iter = 100;
    config.brent_tol_abs = 1e-6;

    auto result = brent_find_root(f, 0.01, 2.0, config);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 0.01, 1e-6);
}

TEST(BrentTest, ConvergenceRecovery) {
    // Test recovery after initial failure with too-narrow bracket
    auto f = [](double x) { return x - 3.0; };
    RootFindingConfig config{};
    config.max_iter = 100;
    config.brent_tol_abs = 1e-6;

    // First attempt: narrow bracket doesn't contain root
    auto result1 = brent_find_root(f, 0.0, 2.0, config);
    EXPECT_FALSE(result1.converged);

    // Second attempt: wider bracket contains root
    auto result2 = brent_find_root(f, 0.0, 5.0, config);
    EXPECT_TRUE(result2.converged);
    EXPECT_NEAR(result2.root, 3.0, 1e-6);

    // Third attempt: verify no stale state
    auto result3 = brent_find_root(f, 2.0, 4.0, config);
    EXPECT_TRUE(result3.converged);
    EXPECT_NEAR(result3.root, 3.0, 1e-6);
}
```

**Step 2: Run all Brent tests**

```bash
bazel test //tests:brent_test --test_output=all
```

Expected: All 7 tests PASS

**Step 3: Commit tests**

```bash
git add tests/brent_test.cc
git commit -m "test(iv): add comprehensive Brent edge case tests"
```

---

## Task 5: Implement IVSolver Structure (Part 1: Parameters and Validation)

**Files:**
- Create: `src/cpp/iv_solver.hpp`
- Create: `tests/iv_solver_test.cc`

**Step 1: Write failing test for IV recovery from known price**

Create `tests/iv_solver_test.cc`:

```cpp
#include <gtest/gtest.h>
#include "src/cpp/iv_solver.hpp"
#include "src/cpp/american_option.hpp"
#include <vector>

namespace mango {
namespace {

TEST(IVSolverTest, KnownVolatility) {
    // Price an option at σ = 0.20, then recover it
    AmericanOptionParams option{
        .strike = 100.0,
        .volatility = 0.20,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OptionType::PUT,
        .dividend_times = {},
        .dividend_amounts = {}
    };

    AmericanOptionGrid grid{
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = 1000
    };

    // Price the option
    AmericanOptionSolver pricer(option, grid);
    ASSERT_TRUE(pricer.solve());
    double market_price = pricer.price_at(1.0);  // ATM

    // Recover IV
    IVParams iv_params{
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .dividend_times = {},
        .dividend_amounts = {},
        .market_price = market_price,
        .option_type = OptionType::PUT
    };

    IVSolver iv_solver(iv_params, grid);
    auto result = iv_solver.solve();

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, 0.20, 1e-4);
    EXPECT_TRUE(result.error.empty());
}

}  // namespace
}  // namespace mango
```

**Step 2: Add test target**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "iv_solver_test",
    size = "medium",
    srcs = ["iv_solver_test.cc"],
    deps = [
        "//src:pde_solver",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3: Run test to verify it fails**

```bash
bazel test //tests:iv_solver_test --test_output=errors
```

Expected: FAIL with "iv_solver.hpp: No such file or directory"

**Step 4: Create iv_solver.hpp with parameter structures**

Create `src/cpp/iv_solver.hpp`:

```cpp
#pragma once

#include "src/cpp/brent.hpp"
#include "src/cpp/american_option.hpp"
#include "src/cpp/root_finding.hpp"
#include "src/mango_trace.h"
#include <span>
#include <vector>
#include <optional>
#include <string>
#include <stdexcept>
#include <cmath>

namespace mango {

// IV solver parameters
struct IVParams {
    double spot_price;
    double strike;
    double time_to_maturity;
    double risk_free_rate;

    // Dividend handling
    double dividend_yield = 0.0;
    std::span<const double> dividend_times;
    std::span<const double> dividend_amounts;

    double market_price;
    OptionType option_type;

    // LIFETIME CONTRACT: Backing vectors must remain valid until solve() completes
};

// IV solver configuration
struct IVConfig {
    double sigma_min = 0.01;
    double sigma_max = 2.0;
    RootFindingConfig root_config{};
    bool use_adaptive_bounds = true;
    bool compute_vega = false;
    double vega_epsilon = 0.001;
};

// IV solver result
struct IVResult {
    double implied_vol = 0.0;
    double vega = std::numeric_limits<double>::quiet_NaN();
    size_t iterations = 0;
    bool converged = false;
    std::string error;
};

// Forward declaration
class IVSolver;

}  // namespace mango
```

**Step 5: Add IVSolver class declaration**

Add to `src/cpp/iv_solver.hpp`:

```cpp
class IVSolver {
public:
    IVSolver(const IVParams& params,
            const AmericanOptionGrid& grid,
            const IVConfig& config = {})
        : params_(params)
        , grid_(grid)
        , config_(config)
    {
        validate_params();
    }

    IVResult solve();

private:
    IVParams params_;
    AmericanOptionGrid grid_;
    IVConfig config_;

    // Cached data for vega computation
    double cached_sigma_ = 0.0;
    std::vector<double> cached_div_times_;
    std::vector<double> cached_div_amounts_;

    // Error tracking
    mutable std::string last_objective_error_;

    void validate_params();
    std::optional<std::string> validate_market_price();
    double compute_intrinsic_value() const;
    double compute_max_value() const;
    std::pair<double, double> compute_adaptive_bounds();
    double objective(double sigma);
    double compute_vega(double sigma);
};
```

**Step 6: Run test again**

```bash
bazel test //tests:iv_solver_test --test_output=errors
```

Expected: FAIL with linker errors (undefined methods)

**Step 7: Commit structure**

```bash
git add src/cpp/iv_solver.hpp tests/iv_solver_test.cc tests/BUILD.bazel
git commit -m "feat(iv): add IVSolver structure and first test"
```

---

## Task 6: Implement IVSolver Validation Logic

**Files:**
- Modify: `src/cpp/iv_solver.hpp` (add validation methods)

**Step 1: Implement validate_params()**

Add to `src/cpp/iv_solver.hpp` after class declaration:

```cpp
inline void IVSolver::validate_params() {
    if (params_.spot_price <= 0.0) {
        throw std::invalid_argument("spot_price must be positive");
    }
    if (params_.strike <= 0.0) {
        throw std::invalid_argument("strike must be positive");
    }
    if (params_.time_to_maturity <= 0.0) {
        throw std::invalid_argument("time_to_maturity must be positive");
    }
    if (params_.market_price < 0.0) {
        throw std::invalid_argument("market_price cannot be negative");
    }
    if (config_.sigma_min <= 0.0 || config_.sigma_min >= config_.sigma_max) {
        throw std::invalid_argument("Invalid sigma bounds");
    }
    if (!params_.dividend_times.empty()) {
        if (params_.dividend_times.size() != params_.dividend_amounts.size()) {
            throw std::invalid_argument(
                "dividend_times and dividend_amounts size mismatch");
        }
    }
}
```

**Step 2: Implement compute_intrinsic_value()**

Add to `src/cpp/iv_solver.hpp`:

```cpp
inline double IVSolver::compute_intrinsic_value() const {
    // Adjust spot for continuous dividend yield
    double S_adj = params_.spot_price;
    if (params_.dividend_yield > 0.0) {
        S_adj *= std::exp(-params_.dividend_yield * params_.time_to_maturity);
    }

    // Adjust for discrete dividends
    for (size_t i = 0; i < params_.dividend_times.size(); ++i) {
        double div_time = params_.dividend_times[i];
        double div_amount = params_.dividend_amounts[i];

        if (div_time < params_.time_to_maturity) {
            double pv = div_amount * std::exp(-params_.risk_free_rate * div_time);
            S_adj -= pv;
        }
    }

    if (params_.option_type == OptionType::PUT) {
        return std::max(params_.strike - S_adj, 0.0);
    } else {
        return std::max(S_adj - params_.strike, 0.0);
    }
}
```

**Step 3: Implement compute_max_value()**

Add to `src/cpp/iv_solver.hpp`:

```cpp
inline double IVSolver::compute_max_value() const {
    if (params_.option_type == OptionType::PUT) {
        return params_.strike * std::exp(-params_.risk_free_rate
                                        * params_.time_to_maturity);
    } else {
        double S_adj = params_.spot_price;
        for (size_t i = 0; i < params_.dividend_times.size(); ++i) {
            double div_time = params_.dividend_times[i];
            double div_amount = params_.dividend_amounts[i];

            if (div_time < params_.time_to_maturity) {
                S_adj -= div_amount * std::exp(-params_.risk_free_rate * div_time);
            }
        }
        return S_adj;
    }
}
```

**Step 4: Implement validate_market_price()**

Add to `src/cpp/iv_solver.hpp`:

```cpp
inline std::optional<std::string> IVSolver::validate_market_price() {
    double intrinsic = compute_intrinsic_value();

    if (params_.market_price < intrinsic - 1e-6) {
        return "Market price below intrinsic value (arbitrage)";
    }

    double max_value = compute_max_value();
    if (params_.market_price > max_value + 1e-6) {
        return "Market price above maximum possible value";
    }

    if (std::abs(params_.market_price - intrinsic) < 1e-6) {
        return "Market price equals intrinsic (IV undefined, volatility → 0)";
    }

    return std::nullopt;
}
```

**Step 5: Write validation test**

Add to `tests/iv_solver_test.cc`:

```cpp
TEST(IVSolverTest, ValidationErrors) {
    AmericanOptionGrid grid{
        .x_min = -0.7, .x_max = 0.7, .n_points = 101,
        .dt = 0.001, .n_steps = 1000
    };

    // Negative spot price
    EXPECT_THROW({
        IVParams params{
            .spot_price = -100.0,
            .strike = 100.0,
            .time_to_maturity = 1.0,
            .risk_free_rate = 0.05,
            .dividend_yield = 0.0,
            .market_price = 10.0,
            .option_type = OptionType::PUT
        };
        IVSolver solver(params, grid);
    }, std::invalid_argument);

    // Invalid sigma bounds
    EXPECT_THROW({
        IVParams params{
            .spot_price = 100.0,
            .strike = 100.0,
            .time_to_maturity = 1.0,
            .risk_free_rate = 0.05,
            .dividend_yield = 0.0,
            .market_price = 10.0,
            .option_type = OptionType::PUT
        };
        IVConfig config;
        config.sigma_min = 0.5;
        config.sigma_max = 0.3;
        IVSolver solver(params, grid, config);
    }, std::invalid_argument);
}
```

**Step 6: Run validation test**

```bash
bazel test //tests:iv_solver_test --test_filter="*ValidationErrors*" --test_output=all
```

Expected: PASS

**Step 7: Commit validation**

```bash
git add src/cpp/iv_solver.hpp tests/iv_solver_test.cc
git commit -m "feat(iv): implement IVSolver validation logic"
```

---

## Task 7: Implement IVSolver Adaptive Bounds

**Files:**
- Modify: `src/cpp/iv_solver.hpp` (add adaptive bounds)

**Step 1: Implement compute_adaptive_bounds()**

Add to `src/cpp/iv_solver.hpp`:

```cpp
inline std::pair<double, double> IVSolver::compute_adaptive_bounds() {
    double intrinsic = compute_intrinsic_value();
    double time_value = params_.market_price - intrinsic;

    // Estimate lower bound
    double moneyness = params_.spot_price / params_.strike;
    double sigma_lower = std::sqrt(2.0 * M_PI / params_.time_to_maturity)
                       * (time_value / params_.spot_price) / std::sqrt(moneyness);

    sigma_lower = std::max(config_.sigma_min, sigma_lower * 0.5);

    // Upper bound
    double sigma_upper = config_.sigma_max;
    if (time_value < intrinsic * 0.1) {
        sigma_upper = std::min(sigma_upper, sigma_lower * 5.0);
    }

    return {sigma_lower, sigma_upper};
}
```

**Step 2: Write adaptive bounds test**

Add to `tests/iv_solver_test.cc`:

```cpp
TEST(IVSolverTest, AdaptiveBounds) {
    AmericanOptionParams option{
        .strike = 100.0,
        .volatility = 0.15,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OptionType::PUT,
        .dividend_times = {},
        .dividend_amounts = {}
    };

    AmericanOptionGrid grid{
        .x_min = -0.7, .x_max = 0.7, .n_points = 101,
        .dt = 0.001, .n_steps = 1000
    };

    AmericanOptionSolver pricer(option, grid);
    ASSERT_TRUE(pricer.solve());
    double market_price = pricer.price_at(0.9);  // ITM

    IVParams iv_params{
        .spot_price = 90.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .dividend_times = {},
        .dividend_amounts = {},
        .market_price = market_price,
        .option_type = OptionType::PUT
    };

    // Without adaptive bounds
    IVConfig config_no_adapt;
    config_no_adapt.use_adaptive_bounds = false;
    IVSolver solver_no_adapt(iv_params, grid, config_no_adapt);
    auto result_no_adapt = solver_no_adapt.solve();

    // With adaptive bounds
    IVConfig config_adapt;
    config_adapt.use_adaptive_bounds = true;
    IVSolver solver_adapt(iv_params, grid, config_adapt);
    auto result_adapt = solver_adapt.solve();

    // Both converge to same answer
    EXPECT_TRUE(result_no_adapt.converged);
    EXPECT_TRUE(result_adapt.converged);
    EXPECT_NEAR(result_no_adapt.implied_vol, result_adapt.implied_vol, 1e-5);

    // Adaptive takes fewer iterations
    EXPECT_LT(result_adapt.iterations, result_no_adapt.iterations);
}
```

**Step 3: Commit adaptive bounds**

```bash
git add src/cpp/iv_solver.hpp tests/iv_solver_test.cc
git commit -m "feat(iv): implement adaptive bounds optimization"
```

---

## Task 8: Implement IVSolver Core (Objective Function)

**Files:**
- Modify: `src/cpp/iv_solver.hpp` (add objective function)

**Step 1: Implement objective() method**

Add to `src/cpp/iv_solver.hpp`:

```cpp
inline double IVSolver::objective(double sigma) {
    // Input validation
    if (sigma <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Convert continuous dividend yield to discrete
    std::vector<double> div_times_vec;
    std::vector<double> div_amounts_vec;

    if (params_.dividend_yield > 0.0 && params_.dividend_times.empty()) {
        double div_time = params_.time_to_maturity / 2.0;
        double div_amount = params_.spot_price
                          * (1.0 - std::exp(-params_.dividend_yield
                                           * params_.time_to_maturity));
        div_times_vec.push_back(div_time);
        div_amounts_vec.push_back(div_amount);
    } else {
        div_times_vec.assign(params_.dividend_times.begin(),
                           params_.dividend_times.end());
        div_amounts_vec.assign(params_.dividend_amounts.begin(),
                             params_.dividend_amounts.end());
    }

    // Cache for vega
    cached_sigma_ = sigma;
    cached_div_times_ = div_times_vec;
    cached_div_amounts_ = div_amounts_vec;

    // Create option params
    AmericanOptionParams option_params{
        .strike = params_.strike,
        .volatility = sigma,
        .risk_free_rate = params_.risk_free_rate,
        .time_to_maturity = params_.time_to_maturity,
        .option_type = params_.option_type,
        .dividend_times = std::span(div_times_vec),
        .dividend_amounts = std::span(div_amounts_vec)
    };

    // Solve PDE
    try {
        AmericanOptionSolver solver(option_params, grid_);
        if (!solver.solve()) {
            last_objective_error_ = "AmericanOptionSolver failed at sigma="
                                   + std::to_string(sigma);
            MANGO_TRACE_RUNTIME_ERROR(MODULE_IV_SOLVER,
                last_objective_error_.c_str());
            return std::numeric_limits<double>::quiet_NaN();
        }

        double moneyness = params_.spot_price / params_.strike;
        double price = solver.price_at(moneyness);

        last_objective_error_.clear();
        return price - params_.market_price;

    } catch (const std::exception& e) {
        last_objective_error_ = std::string("Exception: ") + e.what();
        MANGO_TRACE_RUNTIME_ERROR(MODULE_IV_SOLVER,
            last_objective_error_.c_str());
        return std::numeric_limits<double>::quiet_NaN();
    }
}
```

**Step 2: Run test (will still fail on solve())**

```bash
bazel test //tests:iv_solver_test --test_filter="*KnownVolatility*" --test_output=errors
```

Expected: FAIL with linker error on solve()

**Step 3: Commit objective**

```bash
git add src/cpp/iv_solver.hpp
git commit -m "feat(iv): implement IVSolver objective function"
```

---

## Task 9: Implement IVSolver Core (solve() Method)

**Files:**
- Modify: `src/cpp/iv_solver.hpp` (add solve method)

**Step 1: Implement solve() method**

Add to `src/cpp/iv_solver.hpp`:

```cpp
inline IVResult IVSolver::solve() {
    MANGO_TRACE_IV_START(MODULE_IV_SOLVER,
        params_.spot_price, params_.strike, params_.market_price);

    // Pre-solve validation
    auto validation_error = validate_market_price();
    if (validation_error) {
        IVResult result{};
        result.converged = false;
        result.error = *validation_error;
        result.implied_vol = std::numeric_limits<double>::quiet_NaN();
        result.vega = std::numeric_limits<double>::quiet_NaN();
        result.iterations = 0;

        MANGO_TRACE_IV_VALIDATION_ERROR(MODULE_IV_SOLVER,
            result.error.c_str());
        return result;
    }

    // Adaptive bounds
    double sigma_min = config_.sigma_min;
    double sigma_max = config_.sigma_max;

    if (config_.use_adaptive_bounds) {
        auto bounds = compute_adaptive_bounds();
        sigma_min = bounds.first;
        sigma_max = bounds.second;
    }

    // Call Brent's method
    auto obj = [this](double sigma) { return objective(sigma); };
    auto brent_result = brent_find_root(obj, sigma_min, sigma_max,
                                       config_.root_config);

    // Compute vega if requested
    double vega = std::numeric_limits<double>::quiet_NaN();
    if (config_.compute_vega && brent_result.converged) {
        vega = compute_vega(brent_result.root);
    }

    // Convert to IVResult
    std::string error_msg = brent_result.failure_reason.value_or("");
    if (!brent_result.converged && !last_objective_error_.empty()) {
        error_msg += " (Underlying: " + last_objective_error_ + ")";
    }

    IVResult result{
        .implied_vol = brent_result.root,
        .vega = vega,
        .iterations = brent_result.iterations,
        .converged = brent_result.converged,
        .error = error_msg
    };

    MANGO_TRACE_IV_COMPLETE(MODULE_IV_SOLVER,
        result.converged ? 1 : 0, result.implied_vol, result.iterations);

    return result;
}
```

**Step 2: Run KnownVolatility test**

```bash
bazel test //tests:iv_solver_test --test_filter="*KnownVolatility*" --test_output=all
```

Expected: PASS (assuming AmericanOptionSolver is implemented)

**Step 3: Commit solve method**

```bash
git add src/cpp/iv_solver.hpp
git commit -m "feat(iv): implement IVSolver solve() method"
```

---

## Task 10: Implement Vega Computation

**Files:**
- Modify: `src/cpp/iv_solver.hpp` (add vega computation)

**Step 1: Implement compute_vega() method**

Add to `src/cpp/iv_solver.hpp`:

```cpp
inline double IVSolver::compute_vega(double sigma) {
    double eps = config_.vega_epsilon;

    // Ensure perturbed volatilities are positive
    if (sigma - eps <= 0.0) {
        eps = sigma / 2.0;
    }

    double f_plus = objective(sigma + eps) + params_.market_price;
    double f_minus = objective(sigma - eps) + params_.market_price;

    if (std::isnan(f_plus) || std::isnan(f_minus)) {
        MANGO_TRACE_RUNTIME_ERROR(MODULE_IV_SOLVER,
            "Vega computation: NaN from objective");
        return std::numeric_limits<double>::quiet_NaN();
    }

    return (f_plus - f_minus) / (2.0 * eps);
}
```

**Step 2: Write vega test**

Add to `tests/iv_solver_test.cc`:

```cpp
TEST(IVSolverTest, VegaComputation) {
    AmericanOptionParams option{
        .strike = 100.0,
        .volatility = 0.20,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OptionType::PUT,
        .dividend_times = {},
        .dividend_amounts = {}
    };

    AmericanOptionGrid grid{
        .x_min = -0.7, .x_max = 0.7, .n_points = 101,
        .dt = 0.001, .n_steps = 1000
    };

    AmericanOptionSolver pricer(option, grid);
    ASSERT_TRUE(pricer.solve());
    double market_price = pricer.price_at(1.0);

    IVParams iv_params{
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .dividend_times = {},
        .dividend_amounts = {},
        .market_price = market_price,
        .option_type = OptionType::PUT
    };

    IVConfig iv_config;
    iv_config.compute_vega = true;

    IVSolver iv_solver(iv_params, grid, iv_config);
    auto result = iv_solver.solve();

    EXPECT_TRUE(result.converged);
    EXPECT_FALSE(std::isnan(result.vega));
    EXPECT_GT(result.vega, 0.0);
    EXPECT_GT(result.vega, 20.0);
    EXPECT_LT(result.vega, 50.0);
}
```

**Step 3: Run vega test**

```bash
bazel test //tests:iv_solver_test --test_filter="*VegaComputation*" --test_output=all
```

Expected: PASS

**Step 4: Commit vega**

```bash
git add src/cpp/iv_solver.hpp tests/iv_solver_test.cc
git commit -m "feat(iv): implement vega computation via finite differences"
```

---

## Task 11: Add Dividend Handling Tests

**Files:**
- Modify: `tests/iv_solver_test.cc` (add dividend tests)

**Step 1: Add discrete dividend test**

Add to `tests/iv_solver_test.cc`:

```cpp
TEST(IVSolverTest, WithDiscreteDividends) {
    std::vector<double> div_times = {0.25, 0.75};
    std::vector<double> div_amounts = {1.0, 1.0};

    AmericanOptionParams option{
        .strike = 100.0,
        .volatility = 0.25,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OptionType::PUT,
        .dividend_times = div_times,
        .dividend_amounts = div_amounts
    };

    AmericanOptionGrid grid{
        .x_min = -0.7, .x_max = 0.7, .n_points = 101,
        .dt = 0.001, .n_steps = 1000
    };

    AmericanOptionSolver pricer(option, grid);
    ASSERT_TRUE(pricer.solve());
    double market_price = pricer.price_at(1.0);

    IVParams iv_params{
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .dividend_times = div_times,
        .dividend_amounts = div_amounts,
        .market_price = market_price,
        .option_type = OptionType::PUT
    };

    IVSolver iv_solver(iv_params, grid);
    auto result = iv_solver.solve();

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, 0.25, 1e-4);
}

TEST(IVSolverTest, WithContinuousDividendYield) {
    AmericanOptionParams option{
        .strike = 100.0,
        .volatility = 0.20,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OptionType::PUT,
        .dividend_times = {},
        .dividend_amounts = {}
    };

    AmericanOptionGrid grid{
        .x_min = -0.7, .x_max = 0.7, .n_points = 101,
        .dt = 0.001, .n_steps = 1000
    };

    AmericanOptionSolver pricer(option, grid);
    ASSERT_TRUE(pricer.solve());
    double market_price = pricer.price_at(1.0);

    IVParams iv_params{
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.02,
        .dividend_times = {},
        .dividend_amounts = {},
        .market_price = market_price,
        .option_type = OptionType::PUT
    };

    IVSolver iv_solver(iv_params, grid);
    auto result = iv_solver.solve();

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.15);
}
```

**Step 2: Run dividend tests**

```bash
bazel test //tests:iv_solver_test --test_filter="*Dividend*" --test_output=all
```

Expected: Both tests PASS

**Step 3: Commit dividend tests**

```bash
git add tests/iv_solver_test.cc
git commit -m "test(iv): add discrete and continuous dividend tests"
```

---

## Task 12: Add Out-of-Bounds Test

**Files:**
- Modify: `tests/iv_solver_test.cc` (add validation test)

**Step 1: Add out-of-bounds test**

Add to `tests/iv_solver_test.cc`:

```cpp
TEST(IVSolverTest, OutOfBounds) {
    AmericanOptionGrid grid{
        .x_min = -0.7, .x_max = 0.7, .n_points = 101,
        .dt = 0.001, .n_steps = 1000
    };

    IVParams params{
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .dividend_times = {},
        .dividend_amounts = {},
        .market_price = 200.0,  // Unrealistic
        .option_type = OptionType::PUT
    };

    IVSolver solver(params, grid);
    auto result = solver.solve();

    EXPECT_FALSE(result.converged);
    EXPECT_FALSE(result.error.empty());
    EXPECT_EQ(result.iterations, 0);
}
```

**Step 2: Run test**

```bash
bazel test //tests:iv_solver_test --test_filter="*OutOfBounds*" --test_output=all
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/iv_solver_test.cc
git commit -m "test(iv): add out-of-bounds market price test"
```

---

## Task 13: Run Full Test Suite

**Files:**
- None (verification only)

**Step 1: Run all Brent tests**

```bash
bazel test //tests:brent_test --test_output=all
```

Expected: All 7 tests PASS

**Step 2: Run all IV solver tests**

```bash
bazel test //tests:iv_solver_test --test_output=all
```

Expected: All 8 tests PASS (assuming AmericanOptionSolver implemented)

**Step 3: Run both test suites together**

```bash
bazel test //tests:brent_test //tests:iv_solver_test --test_output=summary
```

Expected: 15 total tests PASS

**Step 4: Check code coverage (optional)**

```bash
bazel coverage //tests:brent_test //tests:iv_solver_test
```

Expected: >90% coverage for brent.hpp and iv_solver.hpp

---

## Task 14: Add USDT Tracing Script

**Files:**
- Create: `scripts/tracing/iv_detailed.bt`

**Step 1: Create bpftrace script for IV monitoring**

Create `scripts/tracing/iv_detailed.bt`:

```bpftrace
#!/usr/bin/env bpftrace

BEGIN {
    printf("Monitoring IV calculations...\n");
}

usdt::mango:iv_start {
    @iv_start[arg0] = nsecs;
    printf("IV START: spot=%.2f strike=%.2f market=%.2f\n",
           arg1, arg2, arg3);
}

usdt::mango:brent_start {
    printf("  Brent bracket: [%.4f, %.4f]\n", arg1, arg2);
}

usdt::mango:brent_iter {
    printf("  Iter %d: sigma=%.6f f(sigma)=%.6e\n",
           arg1, arg2, arg3);
}

usdt::mango:brent_complete {
    printf("  Brent CONVERGED in %d iterations: sigma=%.6f\n",
           arg1, arg2);
}

usdt::mango:iv_complete {
    if (@iv_start[arg0]) {
        $duration_us = (nsecs - @iv_start[arg0]) / 1000;
        printf("IV COMPLETE: converged=%d iv=%.6f iters=%d duration=%dus\n",
               arg1, arg2, arg3, $duration_us);
        delete(@iv_start[arg0]);
    }
}

usdt::mango:iv_validation_error {
    printf("IV VALIDATION ERROR: %s\n", str(arg1));
}

END {
    clear(@iv_start);
}
```

**Step 2: Make script executable**

```bash
chmod +x scripts/tracing/iv_detailed.bt
```

**Step 3: Test script (requires example program)**

If you have an IV example program:

```bash
sudo ./scripts/tracing/iv_detailed.bt -c './bazel-bin/examples/example_iv'
```

Expected: Trace output showing IV calculation flow

**Step 4: Commit tracing script**

```bash
git add scripts/tracing/iv_detailed.bt
git commit -m "feat(iv): add bpftrace script for IV monitoring"
```

---

## Task 15: Update Documentation

**Files:**
- Modify: `CLAUDE.md` (add IV calculation usage)

**Step 1: Add IV calculation section to CLAUDE.md**

Add to `CLAUDE.md` after the American Option section:

```markdown
## Implied Volatility Calculation Workflow

The IV calculation module provides fast implied volatility solving using Brent's method with American option pricing.

### Typical Workflow

**1. Create IV parameters:**
```cpp
mango::IVParams iv_params{
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .dividend_yield = 0.0,        // Or use discrete dividends
    .dividend_times = {},
    .dividend_amounts = {},
    .market_price = 10.50,        // Observed market price
    .option_type = mango::OptionType::PUT
};

mango::AmericanOptionGrid grid{
    .x_min = -0.7,
    .x_max = 0.7,
    .n_points = 101,
    .dt = 0.001,
    .n_steps = 1000
};
```

**2. Solve for IV:**
```cpp
mango::IVSolver solver(iv_params, grid);
auto result = solver.solve();

if (result.converged) {
    std::cout << "Implied volatility: " << result.implied_vol << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
} else {
    std::cerr << "Failed: " << result.error << std::endl;
}
```

**3. Optional: Compute vega:**
```cpp
mango::IVConfig config;
config.compute_vega = true;

mango::IVSolver solver(iv_params, grid, config);
auto result = solver.solve();

std::cout << "Vega: " << result.vega << std::endl;
```

### Performance Characteristics

- **Brent + FDM:** ~200ms per IV calculation (10-15 iterations)
- **With adaptive bounds:** ~120-160ms (6-8 iterations)
- **Vega computation:** +2 PDE solves (~40ms extra)

### Monitoring with USDT

```bash
# Watch IV calculations in real-time
sudo ./scripts/tracing/iv_detailed.bt -c './my_program'

# Count convergence failures
sudo bpftrace -e 'usdt::mango:iv_validation_error {
    @errors[str(arg1)] = count();
}' -c './my_program'
```

### Common Issues

**Market price out of bounds:**
- Ensure market_price > intrinsic_value
- Ensure market_price < max_possible_value
- Check dividend adjustments

**Non-convergence:**
- Widen sigma bounds: `config.sigma_min = 0.001`, `config.sigma_max = 3.0`
- Increase max iterations: `config.root_config.max_iter = 200`
- Check if market price is valid

**Slow performance:**
- Enable adaptive bounds: `config.use_adaptive_bounds = true` (default)
- Reduce grid resolution for faster (less accurate) solves
- Future: Use price table optimization for 25,000x speedup
```

**Step 2: Commit documentation**

```bash
git add CLAUDE.md
git commit -m "docs(iv): add IV calculation workflow to CLAUDE.md"
```

---

## Task 16: Create Example Program (Optional)

**Files:**
- Create: `examples/example_iv_calculation.cc`
- Modify: `examples/BUILD.bazel`

**Step 1: Create example program**

Create `examples/example_iv_calculation.cc`:

```cpp
#include "src/cpp/iv_solver.hpp"
#include "src/cpp/american_option.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== Implied Volatility Calculation Example ===\n\n";

    // Market data
    double spot = 100.0;
    double strike = 100.0;
    double time_to_maturity = 1.0;
    double risk_free_rate = 0.05;
    double market_price = 7.50;

    // Grid configuration
    mango::AmericanOptionGrid grid{
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = 1000
    };

    // IV parameters
    mango::IVParams iv_params{
        .spot_price = spot,
        .strike = strike,
        .time_to_maturity = time_to_maturity,
        .risk_free_rate = risk_free_rate,
        .dividend_yield = 0.0,
        .dividend_times = {},
        .dividend_amounts = {},
        .market_price = market_price,
        .option_type = mango::OptionType::PUT
    };

    // Solve with vega
    mango::IVConfig config;
    config.compute_vega = true;

    std::cout << "Market data:\n";
    std::cout << "  Spot: " << spot << "\n";
    std::cout << "  Strike: " << strike << "\n";
    std::cout << "  Time: " << time_to_maturity << " years\n";
    std::cout << "  Rate: " << risk_free_rate << "\n";
    std::cout << "  Market price: " << market_price << "\n\n";

    std::cout << "Solving for implied volatility...\n";
    mango::IVSolver solver(iv_params, grid, config);
    auto result = solver.solve();

    std::cout << std::fixed << std::setprecision(6);
    if (result.converged) {
        std::cout << "\nSuccess!\n";
        std::cout << "  Implied volatility: " << result.implied_vol << "\n";
        std::cout << "  Vega: " << result.vega << "\n";
        std::cout << "  Iterations: " << result.iterations << "\n";
    } else {
        std::cout << "\nFailed to converge.\n";
        std::cout << "  Error: " << result.error << "\n";
        return 1;
    }

    return 0;
}
```

**Step 2: Add build target**

Add to `examples/BUILD.bazel`:

```python
cc_binary(
    name = "example_iv_calculation",
    srcs = ["example_iv_calculation.cc"],
    deps = ["//src:pde_solver"],
)
```

**Step 3: Build and run**

```bash
bazel build //examples:example_iv_calculation
./bazel-bin/examples/example_iv_calculation
```

Expected: Program outputs implied volatility calculation

**Step 4: Commit example**

```bash
git add examples/example_iv_calculation.cc examples/BUILD.bazel
git commit -m "feat(iv): add IV calculation example program"
```

---

## Task 17: Final Integration Test

**Files:**
- None (verification only)

**Step 1: Build entire project**

```bash
bazel build //...
```

Expected: SUCCESS with no warnings

**Step 2: Run all tests**

```bash
bazel test //...
```

Expected: All tests PASS

**Step 3: Verify USDT probes (if systemtap-sdt-dev installed)**

```bash
readelf -n bazel-bin/examples/example_iv_calculation | grep mango
```

Expected: Output shows `iv_start`, `iv_complete`, `brent_start`, etc.

**Step 4: Run example with tracing (optional)**

```bash
sudo ./scripts/tracing/iv_detailed.bt -c './bazel-bin/examples/example_iv_calculation'
```

Expected: Trace output showing IV solve flow

---

## Completion Checklist

- [ ] Task 1: USDT probes added
- [ ] Task 2-3: Brent's method implemented and tested
- [ ] Task 4: Brent edge case tests added
- [ ] Task 5-6: IVSolver structure and validation
- [ ] Task 7: Adaptive bounds implemented
- [ ] Task 8-9: Objective and solve methods
- [ ] Task 10: Vega computation
- [ ] Task 11-12: Dividend and validation tests
- [ ] Task 13: Full test suite passing (15 tests)
- [ ] Task 14: USDT tracing script
- [ ] Task 15: Documentation updated
- [ ] Task 16: Example program (optional)
- [ ] Task 17: Final integration test

**Total estimated time:** 3-5 days (assuming AmericanOptionSolver complete)

**Success criteria:**
- All 15 tests pass
- Brent's method converges in <15 iterations for typical cases
- IV recovered to within 0.01% of true volatility
- USDT tracing shows complete flow
- Documentation clear and example runnable
