<!-- SPDX-License-Identifier: MIT -->
# American Option Pricing - C++20 Design

**Date:** 2025-11-04
**Status:** Design Complete, Ready for Implementation
**Scope:** Migrate `src/american_option.{h,c}` to C++20

## Overview

This design adds American option pricing to the C++20 PDE solver. The solver prices American calls and puts with discrete dividends using the TR-BDF2 time-stepping scheme with obstacle constraints.

## Motivation

American options allow early exercise, creating a free-boundary problem. The C version solves this with obstacle conditions (V ≥ payoff). We migrate this to C++ to enable:

1. Integration with the C++20 PDE solver infrastructure
2. Type-safe boundary condition dispatch
3. Zero-overhead abstractions via concepts
4. Discrete dividend handling for realistic pricing

## Architecture

### Three-Layer Design

**Layer 1: Obstacle Condition Interface**
Extends PDESolver to enforce V(x,t) ≥ ψ(x,t) after each Newton iteration.

**Layer 2: Dividend Handler**
Applies stock price jumps (S → S - D) via linear interpolation when crossing ex-dividend dates.

**Layer 3: American Option API**
High-level wrapper that configures PDE solver, boundary conditions, and obstacles.

## Component Specifications

### 1. Obstacle Conditions

**Concept definition:**
```cpp
template<typename Fn>
concept ObstacleCondition = requires(Fn fn, std::span<const double> x,
                                     double t, std::span<double> psi) {
    { fn(x, t, psi) } -> std::same_as<void>;
};
```

**No-op obstacle** (European options):
```cpp
struct NoObstacle {
    void operator()(std::span<const double>, double, std::span<double>) const {}
};
```

**American put obstacle:**
```cpp
struct AmericanPutObstacle {
    double strike;

    void operator()(std::span<const double> x, double t, std::span<double> psi) const {
        for (size_t i = 0; i < x.size(); ++i) {
            double S = strike * std::exp(x[i]);  // x = ln(S/K)
            psi[i] = std::max(strike - S, 0.0);
        }
    }
};
```

**American call obstacle:**
```cpp
struct AmericanCallObstacle {
    double strike;

    void operator()(std::span<const double> x, double t, std::span<double> psi) const {
        for (size_t i = 0; i < x.size(); ++i) {
            double S = strike * std::exp(x[i]);
            psi[i] = std::max(S - strike, 0.0);
        }
    }
};
```

**PDESolver modification:**
```cpp
template<typename BoundaryL, typename BoundaryR, typename SpatialOp,
         ObstacleCondition Obstacle>
class PDESolver {
public:
    PDESolver(/* ... existing parameters ... */,
              const Obstacle& obstacle)
        : /* ... */
        , obstacle_(obstacle)
    { }

private:
    Obstacle obstacle_;

    // Enforce u >= ψ pointwise
    void apply_obstacle(std::span<double> u, double t) {
        obstacle_(grid_, t, workspace_.psi_buffer());
        auto psi = workspace_.psi_buffer();

        for (size_t i = 0; i < n_; ++i) {
            u[i] = std::max(u[i], psi[i]);
        }
    }

    // Modified Newton iteration: project after each step
    bool solve_stage1(double t_n, double t_stage1, double dt) {
        for (size_t iter = 0; iter < root_config_.max_iter; ++iter) {
            newton_ws_.solve(/* ... */);

            apply_obstacle(u_stage_, t_stage1);  // PROJECT
            apply_boundary_conditions(u_stage_, t_stage1);

            if (converged) return true;
        }
        return false;
    }
};
```

**Workspace extension:**
```cpp
class WorkspaceStorage {
    std::vector<double> psi_;  // Obstacle values

public:
    std::span<double> psi_buffer() { return psi_; }
};
```

### 2. Discrete Dividends

**Dividend event:**
```cpp
struct DividendEvent {
    double time;    // Years
    double amount;  // Dollars

    auto operator<=>(const DividendEvent&) const = default;
};
```

**Dividend handler:**
```cpp
class DividendHandler {
public:
    DividendHandler(std::span<const double> times,
                   std::span<const double> amounts,
                   double strike)
        : strike_(strike)
    {
        events_.reserve(times.size());
        for (size_t i = 0; i < times.size(); ++i) {
            events_.push_back({times[i], amounts[i]});
        }
        std::ranges::sort(events_);
    }

    // Check if we crossed a dividend
    std::optional<DividendEvent> check_crossing(double t_old, double t_new) const {
        for (const auto& div : events_) {
            if (div.time > t_old && div.time <= t_new) {
                return div;
            }
        }
        return std::nullopt;
    }

    // Apply dividend jump: V_new(S) = V_old(S - D)
    void apply_jump(std::span<const double> x_grid,
                   std::span<const double> V_old,
                   std::span<double> V_new,
                   double dividend) const
    {
        for (size_t i = 0; i < x_grid.size(); ++i) {
            double S_pre = strike_ * std::exp(x_grid[i]);
            double S_post = S_pre - dividend;

            if (S_post <= 0.0) {
                V_new[i] = V_old[0];
                continue;
            }

            double x_post = std::log(S_post / strike_);
            V_new[i] = interpolate_linear(x_grid, V_old, x_post);
        }
    }

private:
    std::vector<DividendEvent> events_;
    double strike_;

    static double interpolate_linear(std::span<const double> x_grid,
                                     std::span<const double> values,
                                     double x_query)
    {
        if (x_query <= x_grid.front()) return values.front();
        if (x_query >= x_grid.back()) return values.back();

        auto it = std::ranges::lower_bound(x_grid, x_query);
        size_t j = std::distance(x_grid.begin(), it) - 1;

        double alpha = (x_query - x_grid[j]) / (x_grid[j+1] - x_grid[j]);
        return (1.0 - alpha) * values[j] + alpha * values[j+1];
    }
};
```

**Integration into PDESolver::solve():**
```cpp
bool PDESolver::solve() {
    double t = time_.t_start();
    const double dt = time_.dt();

    for (size_t step = 0; step < time_.n_steps(); ++step) {
        double t_next = t + dt;

        // Check for dividend crossing
        if (auto div = dividend_handler_.check_crossing(t, t_next)) {
            std::vector<double> u_temp(n_);
            dividend_handler_.apply_jump(grid_, u_current_, u_temp, div->amount);
            std::copy(u_temp.begin(), u_temp.end(), u_current_.begin());
        }

        // Normal TR-BDF2 step
        std::copy(u_current_.begin(), u_current_.end(), u_old_.begin());

        if (!solve_stage1(t, t + config_.gamma * dt, dt)) return false;
        if (!solve_stage2(t, t_next, dt)) return false;

        std::copy(u_next_.begin(), u_next_.end(), u_current_.begin());
        t = t_next;
    }
    return true;
}
```

### 3. American Option API

**Parameters:**
```cpp
enum class OptionType { CALL, PUT };

struct AmericanOptionParams {
    double strike;
    double volatility;
    double risk_free_rate;
    double time_to_maturity;
    OptionType option_type;

    std::vector<double> dividend_times;
    std::vector<double> dividend_amounts;
};

struct AmericanOptionGrid {
    double x_min;        // e.g., -0.7
    double x_max;        // e.g., 0.7
    size_t n_points;     // e.g., 101
    double dt;           // e.g., 0.001
    size_t n_steps;      // e.g., 1000
};
```

**Solver:**
```cpp
class AmericanOptionSolver {
public:
    AmericanOptionSolver(const AmericanOptionParams& params,
                        const AmericanOptionGrid& grid_spec);

    bool solve();

    // Access results
    std::span<const double> grid() const { return grid_.x(); }
    std::span<const double> solution() const;
    std::span<const double> deltas() const;
    std::span<const double> gammas() const;

    // Convenience interpolation
    double price_at(double moneyness) const;
    double delta_at(double moneyness) const;
    double gamma_at(double moneyness) const;

private:
    AmericanOptionParams params_;
    GridBuffer grid_;

    // Type-erased PDE solver
    struct PDESolverImpl;
    std::unique_ptr<PDESolverImpl> solver_;

    std::unique_ptr<SnapshotInterpolator> price_interp_;
};
```

## Implementation Order

1. **Extend WorkspaceStorage** - Add `psi_` buffer for obstacle values
2. **Add obstacle parameter to PDESolver** - Template parameter + apply_obstacle()
3. **Implement DividendHandler** - Event sorting, crossing detection, linear interpolation
4. **Integrate dividends into PDESolver::solve()** - Check and apply jumps
5. **Implement AmericanOptionSolver wrapper** - High-level API
6. **Add boundary condition helpers** - create_american_left_bc(), create_american_right_bc()
7. **Write tests** - Unit tests for obstacles, dividends, full American pricing

## Testing Strategy

**Unit tests:**
- Obstacle projection (verify u ≥ ψ after projection)
- Dividend jump interpolation (verify V_new(S) = V_old(S - D))
- Boundary condition values (verify theoretical limits)

**Integration tests:**
- American put without dividends (compare to known values)
- American call with dividends (verify early exercise optimal)
- Convergence tests (verify solver converges for all cases)

**Regression tests:**
- Compare against C version results (verify numerical equivalence)

## Migration Notes

This design preserves the C version's numerical methods while using modern C++:
- Obstacle projection remains identical (max(u, ψ))
- Dividend interpolation uses linear interpolation (same as C)
- Boundary conditions match C version's auto-detection logic
- Grid generation reuses existing GridSpec infrastructure

The C++20 version gains type safety, zero-overhead abstractions, and cleaner interfaces without changing the underlying numerics.

## References

- `src/american_option.{h,c}` - C implementation
- `src/cpp/pde_solver.hpp` - Existing C++ PDE solver
- `src/cpp/spatial_operators.hpp` - Black-Scholes operators
- Ascher, Ruuth, Wetton (1995) - TR-BDF2 scheme
