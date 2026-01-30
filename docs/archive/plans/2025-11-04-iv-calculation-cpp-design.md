<!-- SPDX-License-Identifier: MIT -->
# Implied Volatility Calculation - C++20 Design

**Date:** 2025-11-04
**Status:** Design Complete, Ready for Implementation
**Scope:** Migrate `src/brent.h` and `src/implied_volatility.{h,c}` to C++20

## Overview

This design adds implied volatility (IV) calculation to the C++20 codebase. The solver computes the volatility that makes the theoretical American option price equal the market price. It uses Brent's method for robust root-finding.

## Motivation

Implied volatility inverts the pricing function: given a market price, find the volatility. This is the primary use case for option traders. We implement this by:

1. Adding Brent's method root finder (derivative-free, robust)
2. Creating IV solver that uses American option pricing
3. Integrating with existing `RootFindingConfig` interface
4. Supporting future optimization via price table lookup

## Architecture

### Two-Layer Design

**Layer 1: Brent's Method Root Finder**
General-purpose root finder using bracketing and interpolation. Integrates with existing `RootFindingConfig` and `RootFindingResult`.

**Layer 2: IV Solver**
Domain-specific solver that uses Brent's method to find σ such that V(σ) = V_market. Uses American option pricing as the objective function.

## Component Specifications

### 1. Brent's Method Root Finder

**Concept definition:**
```cpp
template<typename Fn>
concept BrentObjective = requires(Fn fn, double x) {
    { fn(x) } -> std::convertible_to<double>;
};
```

**Result structure:**
```cpp
struct BrentResult : RootFindingResult {
    double root;      // Found root
    double f_root;    // Function value at root

    // Inherited from RootFindingResult:
    // bool converged;
    // size_t iterations;
    // double final_error;
    // std::optional<std::string> failure_reason;
};
```

**Root finder implementation:**
```cpp
template<BrentObjective Fn>
BrentResult brent_find_root(Fn&& objective,
                           double a, double b,
                           const RootFindingConfig& config);
```

**Algorithm:**
Combines three methods for robust convergence:
1. **Inverse quadratic interpolation** - Fast when three distinct points available
2. **Secant method** - Fallback when only two points available
3. **Bisection** - Guaranteed progress when interpolation fails

**Convergence criteria:**
- Absolute tolerance: `|f(x)| < config.brent_tol_abs`
- Interval tolerance: `|b - a| < config.brent_tol_abs`

**Error handling:**
- Root not bracketed: Returns failure with diagnostic message
- Max iterations: Returns best estimate with failure flag
- NaN from objective: Propagates to caller

### 2. IV Solver

**Parameters:**
```cpp
struct IVParams {
    double spot_price;
    double strike;
    double time_to_maturity;
    double risk_free_rate;
    double dividend_yield;      // For continuous yield approximation
    double market_price;
    OptionType option_type;     // CALL or PUT
};

struct IVConfig {
    double sigma_min = 0.01;    // Min vol (1%)
    double sigma_max = 2.0;     // Max vol (200%)
    RootFindingConfig root_config;  // Tolerances, max iterations
};

struct IVResult {
    double implied_vol;
    double vega;                // Option vega at solution (0 if not computed)
    size_t iterations;
    bool converged;
    std::string_view error;     // Empty if success
};
```

**Solver implementation:**
```cpp
class IVSolver {
public:
    IVSolver(const IVParams& params,
            const AmericanOptionGrid& grid,
            const IVConfig& config = {});

    IVResult solve();

private:
    IVParams params_;
    AmericanOptionGrid grid_;
    IVConfig config_;

    // Objective: V(σ) - V_market
    double objective(double sigma);
};
```

**Objective function:**
```cpp
double IVSolver::objective(double sigma) {
    // Create American option with this volatility
    AmericanOptionParams option_params{
        .strike = params_.strike,
        .volatility = sigma,
        .risk_free_rate = params_.risk_free_rate,
        .time_to_maturity = params_.time_to_maturity,
        .option_type = params_.option_type,
        .dividend_times = {},     // No discrete dividends for IV
        .dividend_amounts = {}
    };

    // Solve PDE
    AmericanOptionSolver solver(option_params, grid_);
    if (!solver.solve()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Interpolate at spot price
    double moneyness = params_.spot_price / params_.strike;
    double price = solver.price_at(moneyness);

    return price - params_.market_price;
}
```

**Solve implementation:**
```cpp
IVResult IVSolver::solve() {
    // Wrap objective as lambda
    auto obj = [this](double sigma) { return objective(sigma); };

    // Call Brent's method
    auto brent_result = brent_find_root(obj,
                                        config_.sigma_min,
                                        config_.sigma_max,
                                        config_.root_config);

    // Convert to IVResult
    return IVResult{
        .implied_vol = brent_result.root,
        .vega = 0.0,  // Not computed by Brent
        .iterations = brent_result.iterations,
        .converged = brent_result.converged,
        .error = brent_result.failure_reason.value_or("")
    };
}
```

### 3. Integration Points

**With existing code:**
- Uses `RootFindingConfig` from `src/cpp/root_finding.hpp`
- Uses `AmericanOptionSolver` (to be implemented from previous design)
- Uses `OptionType` enum (shared with American option)

**Future optimization:**
When price table is ready, add fast path:

```cpp
class IVSolver {
public:
    // Constructor with optional price table
    template<typename PriceTable>
    IVSolver(const IVParams& params,
            const AmericanOptionGrid& grid,
            const PriceTable* table,  // nullptr = use FDM only
            const IVConfig& config = {});

    IVResult solve() {
        // Try table-based Newton if available
        if (table_ && is_in_bounds()) {
            auto result = newton_solve();
            if (result.converged) {
                return result;  // Fast: ~10μs
            }
        }

        // Fallback: Brent + FDM (~250ms)
        return brent_solve();
    }

private:
    IVResult newton_solve();  // Future implementation
    IVResult brent_solve();   // Current implementation
};
```

## Implementation Order

1. **Implement Brent's method** - Header-only in `src/cpp/brent.hpp`
   - Template function with concept
   - Integration with `RootFindingConfig`
   - Unit tests for convergence

2. **Implement IVSolver** - In `src/cpp/iv_solver.hpp`
   - IVParams, IVConfig, IVResult structures
   - IVSolver class with Brent integration
   - Objective function using AmericanOptionSolver

3. **Add convenience API** - High-level interface
   - `calculate_iv(params, grid)` function
   - Default parameter handling

4. **Write tests**
   - Unit tests for Brent (known roots)
   - Integration tests for IV (known IV values)
   - Edge cases (out of bounds, non-convergence)

## Testing Strategy

**Brent's method tests:**
```cpp
TEST(BrentTest, SimplePolynomial) {
    // f(x) = x^2 - 4, root at x = 2
    auto f = [](double x) { return x * x - 4.0; };
    auto result = brent_find_root(f, 0.0, 5.0);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.root, 2.0, 1e-6);
}

TEST(BrentTest, NotBracketed) {
    auto f = [](double x) { return x * x + 1.0; };  // No real roots
    auto result = brent_find_root(f, 0.0, 5.0);

    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.failure_reason.has_value());
}
```

**IV solver tests:**
```cpp
TEST(IVSolverTest, KnownVolatility) {
    // Price an option at σ = 0.20, then recover it
    AmericanOptionParams option{
        .strike = 100.0,
        .volatility = 0.20,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OptionType::PUT
    };

    AmericanOptionGrid grid{
        .x_min = -0.7, .x_max = 0.7, .n_points = 101,
        .dt = 0.001, .n_steps = 1000
    };

    // Price the option
    AmericanOptionSolver pricer(option, grid);
    pricer.solve();
    double market_price = pricer.price_at(1.0);  // ATM

    // Recover IV
    IVParams iv_params{
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = market_price,
        .option_type = OptionType::PUT
    };

    IVSolver iv_solver(iv_params, grid);
    auto result = iv_solver.solve();

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.implied_vol, 0.20, 1e-4);
}

TEST(IVSolverTest, OutOfBounds) {
    // Price too high for any reasonable volatility
    IVParams params{
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 200.0,  // Unrealistic
        .option_type = OptionType::PUT
    };

    AmericanOptionGrid grid{/* ... */};
    IVSolver solver(params, grid);
    auto result = solver.solve();

    // Should fail to find root in [0.01, 2.0]
    EXPECT_FALSE(result.converged);
}
```

## Performance Characteristics

**Brent's method:**
- Iterations: Typically 5-15 for well-behaved functions
- Cost per iteration: One function evaluation
- Total for IV: ~10 iterations × 20ms = ~200ms

**Comparison to Newton:**
- Brent: No derivative required, more robust
- Newton: Faster (fewer iterations) but needs vega
- Trade-off: Robustness vs speed

**Future optimization with price table:**
- Newton + table: ~10μs (25,000x faster)
- Brent + FDM: ~250ms (fallback for out-of-bounds)

## Migration Notes

This design preserves the C version's algorithm while using modern C++:
- Brent's method identical to C implementation
- Uses concepts instead of function pointers
- Integrates with existing `RootFindingConfig`
- Type-safe via templates
- Zero overhead abstractions

The C++20 version gains:
- Compile-time optimization (inlined lambdas)
- Type safety (concepts catch errors)
- Consistent interface (RootFindingResult)
- Easier testing (no function pointer setup)

## References

- `src/brent.h` - C implementation
- `src/implied_volatility.{h,c}` - C IV solver
- `src/cpp/root_finding.hpp` - Existing root-finding interface
- Brent, R. (1973) - "Algorithms for Minimization without Derivatives"
