<!-- SPDX-License-Identifier: MIT -->
# Yield Curve Support Design

## Overview

Add time-varying interest rate support to the American option pricing library. Rate data comes from external systems, is preprocessed into a `YieldCurve` object, and used in the PDE solver.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Internal representation | Log-linear discount interpolation | Arbitrage-free, industry standard, efficient |
| Data structure | `vector<TenorPoint>` | Keeps tenor/discount pairs together |
| PDE integration | Template on rate function | Zero overhead, works with constant or curve |
| API approach | `std::variant<double, YieldCurve>` | Unified API, backward compatible |

## Core Data Structure

```cpp
// src/math/yield_curve.hpp
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
    /// Construct from tenor points (must include t=0 with log_discount=0)
    static std::expected<YieldCurve, std::string>
    from_points(std::vector<TenorPoint> points);

    /// Construct from discount factors (convenience)
    static std::expected<YieldCurve, std::string>
    from_discounts(std::span<const double> tenors,
                   std::span<const double> discounts);

    /// Construct flat curve (constant rate)
    static YieldCurve flat(double rate);

    /// Instantaneous forward rate at time t
    double rate(double t) const;

    /// Discount factor D(t)
    double discount(double t) const;
};

} // namespace mango
```

## Implementation

### YieldCurve::rate()

```cpp
double YieldCurve::rate(double t) const {
    // Binary search for bracketing interval [t_i, t_{i+1}]
    auto it = std::upper_bound(curve_.begin(), curve_.end(), t,
        [](double t, const TenorPoint& p) { return t < p.tenor; });

    if (it == curve_.begin()) return 0.0;  // t <= 0
    if (it == curve_.end()) --it;          // Extrapolate flat

    auto& right = *it;
    auto& left = *std::prev(it);

    // Forward rate = -(ln D(t2) - ln D(t1)) / (t2 - t1)
    double dt = right.tenor - left.tenor;
    return -(right.log_discount - left.log_discount) / dt;
}
```

### BlackScholesPDE Changes

```cpp
template<typename T, typename RateFn>
class BlackScholesPDE {
    T half_sigma_sq_;
    T dividend_;
    RateFn rate_fn_;  // Always double(double), even for constant

public:
    template<typename Fn>
    BlackScholesPDE(T sigma, Fn&& rate_fn, T d)
        : half_sigma_sq_(0.5 * sigma * sigma)
        , dividend_(d)
        , rate_fn_(std::forward<Fn>(rate_fn)) {}

    // Operator now takes time parameter
    T operator()(T d2v_dx2, T dv_dx, T v, double t) const {
        T r = rate_fn_(t);
        T drift = r - dividend_ - half_sigma_sq_;
        return half_sigma_sq_ * d2v_dx2 + drift * dv_dx - r * v;
    }
};

// Convenience factory for constant rate
template<typename T>
auto make_bs_pde(T sigma, T r, T d) {
    return BlackScholesPDE<T, /*lambda type*/>(
        sigma, [r](double) { return r; }, d);
}
```

## API Changes

```cpp
// src/option/option_spec.hpp
namespace mango {

/// Rate specification: constant or yield curve
using RateSpec = std::variant<double, YieldCurve>;

struct PricingParams {
    double strike = 0.0;
    double spot = 0.0;
    double maturity = 0.0;
    double volatility = 0.0;
    RateSpec rate = 0.05;  // Default: constant 5%
    double continuous_dividend_yield = 0.0;
    OptionType type = OptionType::PUT;
};

// Helper to extract rate function from RateSpec
inline auto make_rate_fn(const RateSpec& spec) {
    return std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, double>) {
            return [r = arg](double) { return r; };
        } else {
            return [&curve = arg](double t) { return curve.rate(t); };
        }
    }, spec);
}

} // namespace mango
```

### Usage Examples

```cpp
// Constant rate (unchanged)
mango::PricingParams params{
    .strike = 100.0,
    .spot = 100.0,
    .maturity = 1.0,
    .volatility = 0.20,
    .rate = 0.05,
    .continuous_dividend_yield = 0.02,
    .type = OptionType::PUT
};

// Yield curve
std::vector<double> tenors = {0.0, 0.25, 0.5, 1.0, 2.0};
std::vector<double> discounts = {1.0, 0.9876, 0.9753, 0.9512, 0.9048};
auto curve = mango::YieldCurve::from_discounts(tenors, discounts).value();

mango::PricingParams params{
    .strike = 100.0,
    .spot = 100.0,
    .maturity = 1.0,
    .volatility = 0.20,
    .rate = curve,
    .continuous_dividend_yield = 0.02,
    .type = OptionType::PUT
};
```

## Files to Modify

| File | Change |
|------|--------|
| `src/math/yield_curve.hpp` | **New** - `TenorPoint`, `YieldCurve` class |
| `src/math/BUILD.bazel` | Add `yield_curve` target |
| `src/option/option_spec.hpp` | Add `RateSpec` variant, update `PricingParams` |
| `src/pde/operators/black_scholes_pde.hpp` | Template on rate function, add time param to `operator()` |
| `src/pde/core/pde_solver.hpp` | Pass current time `t` to operator |
| `src/option/american_option.hpp` | Use `make_rate_fn()` to extract rate |
| `tests/yield_curve_test.cc` | **New** - Unit tests for YieldCurve |
| `tests/american_option_test.cc` | Add tests with yield curve |

## Testing Strategy

1. Unit test `YieldCurve::rate()` against known values
2. Verify flat curve matches constant rate pricing
3. Compare upward-sloping curve vs flat at average rate
4. Validate boundary cases (t=0, t > max tenor)

## Mathematical Background

### Log-Linear Discount Interpolation

Given discount factors D(t_i) at tenor points t_i, we interpolate:

```
ln D(t) = ln D(t_i) + (t - t_i) / (t_{i+1} - t_i) * (ln D(t_{i+1}) - ln D(t_i))
```

The instantaneous forward rate between t_i and t_{i+1} is constant:

```
f(t) = -(ln D(t_{i+1}) - ln D(t_i)) / (t_{i+1} - t_i)
```

This approach:
- Preserves positive, monotone discount factors
- Implies piecewise-constant forward rates (arbitrage-free)
- Matches industry standard (QuantLib, OpenGamma, clearing houses)
