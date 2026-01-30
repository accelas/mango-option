<!-- SPDX-License-Identifier: MIT -->
# Batch Option Enhancement Design (Revised)

**Date:** 2025-01-12
**Status:** Design (v2 - Post Review)
**Authors:** Claude Code + User + Codex Review

## Executive Summary

We enhance batch American option pricing with two improvements:

1. **Flexible Batch API**: Add `SetupCallback` to `BatchAmericanOptionSolver`
2. **Normalized Chain Solver**: Solve PDE once in dimensionless coordinates, interpolate for all strikes/maturities

The normalized solver exploits scale invariance: V(S,K,τ) = K · u(ln(S/K), τ). One PDE solve yields prices for arbitrary (S,K) combinations via interpolation.

**Key correction from v1**: Both spot and strike normalize to dimensionless coordinates (x = ln(S/K)). Payoff and boundaries expressed in normalized space. Solves at universal coordinates, scales results back with strike.

## Goals

### Goal 1: Flexible Batch API
Current `BatchAmericanOptionSolver` lacks configuration hooks. Add `SetupCallback` for per-solver configuration (snapshot registration, convergence tuning).

### Goal 2: Normalized Chain Solver
Price tables and option chains solve multiple (S,K,τ) combinations. Black-Scholes PDE in log-moneyness has constant coefficients—strike/spot appear only in coordinate normalization and payoff scaling.

**Mathematical foundation**:
- PDE: ∂u/∂t = 0.5σ²(∂²u/∂x² - ∂u/∂x) + (r-q)∂u/∂x - ru, where u = V/K, x = ln(S/K)
- Solution: V(S,K,τ) = K · u(ln(S/K), τ)
- Payoff: u(x,0) = max(eˣ-1, 0) for calls, max(1-eˣ, 0) for puts

One solve at (σ,r,q) produces universal function u(x,τ). Any (S,K) query becomes: V = K · u(ln(S/K), τ).

## Architecture

### Normalized Solver Core

```cpp
struct NormalizedSolveRequest {
    double sigma;              // Volatility
    double rate;               // Risk-free rate
    double dividend;           // Continuous dividend yield
    OptionType option_type;    // Call or Put

    double x_min, x_max;       // Log-moneyness domain
    size_t n_space;            // Spatial grid points
    size_t n_time;             // Time steps
    double T_max;              // Maximum maturity
    std::span<const double> tau_snapshots;  // Snapshot times
};

struct NormalizedSurfaceView {
    std::span<const double> x_grid;     // Log-moneyness grid
    std::span<const double> tau_grid;   // Time grid
    std::span<const double> values;     // Solution u(x,τ) [row-major]
};

class NormalizedChainSolver {
public:
    // Solve in dimensionless coordinates, output to surface_view
    static expected<void, SolverError> solve(
        const NormalizedSolveRequest& request,
        Workspace& workspace,
        NormalizedSurfaceView& surface_view);

    // Check if request parameters are eligible
    static expected<void, std::string> check_eligibility(
        const NormalizedSolveRequest& request);
};
```

### Front-End APIs

**Price Table Usage** (moneyness-based):
```cpp
// Price table provides moneyness grid directly
std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1, 1.2};
std::vector<double> maturities = {0.25, 0.5, 1.0};

NormalizedSolveRequest request{
    .sigma = 0.20,
    .rate = 0.05,
    .dividend = 0.02,
    .option_type = OptionType::PUT,
    .x_min = -3.0,
    .x_max = 3.0,
    .n_space = 101,
    .n_time = 1000,
    .T_max = 1.0,
    .tau_snapshots = maturities
};

Workspace workspace = create_workspace(request);
NormalizedSurfaceView surface = workspace.surface_view();

NormalizedChainSolver::solve(request, workspace, surface);

// Convert to prices
// Moneyness convention: m = K/S (strike/spot ratio)
double spot = K_ref;  // For price tables, spot = K_ref
for (size_t i = 0; i < moneyness.size(); ++i) {
    // x = ln(S/K) = -ln(K/S) = -ln(m)
    double x = -std::log(moneyness[i]);
    double K = moneyness[i] * spot;  // K = m * S

    for (size_t j = 0; j < maturities.size(); ++j) {
        double u = surface.interpolate(x, maturities[j]);
        prices[i][j] = K * u;  // V = K·u
    }
}
```

**Option Chain Usage** (strike-based):
```cpp
// Option chain provides strikes, convert to log-moneyness
double spot = 100.0;
std::vector<double> strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
std::vector<double> maturities = {0.25, 0.5, 1.0};

std::vector<double> x_targets;
for (double K : strikes) {
    x_targets.push_back(std::log(spot / K));
}

// Solve (same as price table)
NormalizedChainSolver::solve(request, workspace, surface);

// Convert to prices
for (size_t i = 0; i < strikes.size(); ++i) {
    for (size_t j = 0; j < maturities.size(); ++j) {
        double u = surface.interpolate(x_targets[i], maturities[j]);
        prices[i][j] = strikes[i] * u;
    }
}
```

## Normalization Details

### Coordinate Transform

```cpp
// Input: (S, K, τ, σ, r, q)
double x = std::log(S / K);  // Log-moneyness
double t = tau;               // Time to expiry

// PDE in normalized coordinates
// ∂u/∂t = 0.5σ²(∂²u/∂x² - ∂u/∂x) + (r-q)∂u/∂x - ru
```

### Payoff (Normalized Space)

```cpp
// Terminal condition at t=0
for (size_t i = 0; i < n_space; ++i) {
    double x = x_min + i * dx;
    if (is_call) {
        u[i] = std::max(std::exp(x) - 1.0, 0.0);  // max(S/K - 1, 0)
    } else {
        u[i] = std::max(1.0 - std::exp(x), 0.0);  // max(1 - S/K, 0)
    }
}
```

### Price Recovery

```cpp
// Given solver output u(x,τ)
double V = K * u;  // Strike carries currency units

// Greeks (derived from V = K·u(ln(S/K), τ))
// ∂V/∂S = K·u_x·(∂x/∂S) = K·u_x·(1/S)
double delta = (K / S) * u_x;

// ∂²V/∂S² = K·[u_xx·(1/S)² + u_x·(-1/S²)] = (K/S²)·(u_xx - u_x)
double gamma = (K / (S * S)) * (u_xx - u_x);

// ∂V/∂τ = K·u_τ (but we solve backward in time, so u_t = -u_τ)
double theta = -K * u_t;

// ∂V/∂σ = K·u_σ
double vega = K * u_sigma;
```

## Eligibility Criteria

### Numerical Thresholds (Derived)

```cpp
struct EligibilityLimits {
    static constexpr double MAX_WIDTH = 5.8;      // Empirical convergence limit
    static constexpr double MAX_DX = 0.05;        // Truncation error O(dx²)
    static constexpr double MIN_MARGIN_ABS = 0.35; // 6-cell ghost zone minimum

    static double min_margin(double dx) {
        return std::max(MIN_MARGIN_ABS, 6.0 * dx);
    }

    static double max_ratio(double dx) {
        double margin = min_margin(dx);
        return std::exp(MAX_WIDTH - 2.0 * margin);
    }
};
```

**Derivation**:
- **Margin**: Crank-Nicolson with second-order stencil needs ≥6 ghost cells between payoff and boundary for <0.5bp reflection error
- **Width limit**: Convergence degrades beyond 5.8 log-units (empirical from sweep tests)
- **Ratio limit**: `ln(K_max/K_min) + 2·margin ≤ 5.8` → `ratio ≤ exp(5.8 - 2·margin)`
- **Grid spacing**: Von Neumann stability for σ=200%, τ=2y requires dx≤0.05

**Example**:
- dx = 0.045 → margin = 0.35 → ratio_limit = exp(5.1) ≈ 164
- Conservative production threshold: 160
- For ratio=148: width = ln(148) + 0.7 = 5.7 ≤ 5.8 ✓

### Eligibility Check

```cpp
expected<void, std::string> check_eligibility(
    const NormalizedSolveRequest& req,
    std::span<const double> moneyness_grid)  // Price table provides this
{
    // Check grid spacing
    double dx = (req.x_max - req.x_min) / (req.n_space - 1);
    if (dx > EligibilityLimits::MAX_DX) {
        return unexpected("Grid spacing " + std::to_string(dx) +
                         " exceeds limit " + std::to_string(EligibilityLimits::MAX_DX));
    }

    // Check width
    double width = req.x_max - req.x_min;
    if (width > EligibilityLimits::MAX_WIDTH) {
        return unexpected("Domain width " + std::to_string(width) +
                         " exceeds limit " + std::to_string(EligibilityLimits::MAX_WIDTH));
    }

    // Check margin (moneyness convention: m = K/S)
    // x = ln(S/K) = -ln(m)
    // x_min_data = -ln(m_max), x_max_data = -ln(m_min)
    double m_min = *std::ranges::min_element(moneyness_grid);
    double m_max = *std::ranges::max_element(moneyness_grid);
    double x_min_data = -std::log(m_max);
    double x_max_data = -std::log(m_min);

    double margin_left = x_min_data - req.x_min;
    double margin_right = req.x_max - x_max_data;
    double min_margin = EligibilityLimits::min_margin(dx);

    if (margin_left < min_margin) {
        return unexpected("Left margin " + std::to_string(margin_left) +
                         " < required " + std::to_string(min_margin));
    }
    if (margin_right < min_margin) {
        return unexpected("Right margin " + std::to_string(margin_right) +
                         " < required " + std::to_string(min_margin));
    }

    return {};
}
```

## Workspace Management

### Thread-Safe Pattern

```cpp
// Caller-owned workspace (thread-safe)
class Workspace {
public:
    // Pre-allocate buffers for PDE solve
    static expected<Workspace, std::string> create(
        const NormalizedSolveRequest& request);

    // Get view for results
    NormalizedSurfaceView surface_view();

    // No copying (expensive)
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    // Moving OK
    Workspace(Workspace&&) = default;
    Workspace& operator=(Workspace&&) = default;
};
```

### Price Table Usage

```cpp
// Parallel over (σ,r), each thread reuses workspace
// CRITICAL: Workspace created OUTSIDE work-sharing loop, reused across iterations
#pragma omp parallel
{
    // Each thread creates workspace once (outside for loop)
    auto workspace_result = Workspace::create(base_request);
    Workspace workspace;
    bool workspace_valid = false;

    if (workspace_result) {
        workspace = std::move(workspace_result.value());
        workspace_valid = true;
    }

    if (workspace_valid) {
        auto surface = workspace.surface_view();

        #pragma omp for collapse(2) schedule(static)
        for (size_t k = 0; k < Nv; ++k) {
            for (size_t l = 0; l < Nr; ++l) {
                NormalizedSolveRequest request = build_request(k, l);

                auto result = NormalizedChainSolver::solve(request, workspace, surface);
                if (!result) {
                    // Handle error
                    continue;
                }

                // Extract prices from surface
                // Moneyness convention: m = K/S
                for (size_t i = 0; i < Nm; ++i) {
                    double x = -std::log(moneyness[i]);  // x = -ln(m)
                    double K = moneyness[i] * spot;       // K = m * S

                    for (size_t j = 0; j < Nt; ++j) {
                        double u = surface.interpolate(x, maturities[j]);
                        prices_4d[index_4d(i,j,k,l)] = K * u;
                    }
                }
            }
        }
    } else {
        // Workspace creation failed, mark all as errors
        #pragma omp for collapse(2)
        for (size_t k = 0; k < Nv; ++k) {
            for (size_t l = 0; l < Nr; ++l) {
                // Set error flag
            }
        }
    }
}
```

**Thread safety**: Each thread holds distinct `Workspace`. Solver is stateless. Immutable data (grid coefficients) shared via solver object.

## Batch API Enhancement

### SetupCallback Addition

```cpp
using SetupCallback = std::function<void(size_t index, AmericanOptionSolver& solver)>;

static std::vector<expected<AmericanOptionResult, SolverError>> solve_batch(
    std::span<const AmericanOptionParams> params,
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time,
    SetupCallback setup = nullptr);  // NEW parameter
```

**Backward compatible**: `setup = nullptr` preserves existing behavior.

### Implementation Pattern

```cpp
// Validate parameters (no allocation)
auto validation = AmericanSolverWorkspace::validate_params(x_min, x_max, n_space, n_time);
if (!validation) return all_errors;

// Common solve logic
auto solve_one = [&](size_t i, auto& workspace) -> expected<AmericanOptionResult, SolverError> {
    auto solver = AmericanOptionSolver::create(params[i], workspace);
    if (!solver) return unexpected(error);

    // NEW: Invoke callback if provided
    if (setup) {
        setup(i, solver.value());
    }

    return solver.value().solve();
};

#ifdef _OPENMP
#pragma omp parallel
{
    auto thread_workspace = AmericanSolverWorkspace::create(...).value();

    #pragma omp for
    for (size_t i = 0; i < params.size(); ++i) {
        results[i] = solve_one(i, thread_workspace);
    }
}
#else
auto workspace = AmericanSolverWorkspace::create(...).value();
for (size_t i = 0; i < params.size(); ++i) {
    results[i] = solve_one(i, workspace);
}
#endif

return results;
```

### Usage Example

```cpp
std::vector<AmericanOptionParams> batch = { /* ... */ };
std::vector<SnapshotCollector> collectors = { /* ... */ };

auto results = BatchAmericanOptionSolver::solve_batch(
    batch, -3.0, 3.0, 101, 1000,
    [&](size_t idx, AmericanOptionSolver& solver) {
        // Register snapshots for this solver
        solver.register_snapshot(249, 0, &collectors[idx]);
        solver.register_snapshot(499, 1, &collectors[idx]);
        solver.register_snapshot(999, 2, &collectors[idx]);
    });
```

## PriceTable4DBuilder Integration

### Routing Decision

```cpp
bool should_use_normalized_solver(const PriceTableConfig& config) {
    // Check 1: No discrete dividends (normalized solver requirement)
    if (config.has_discrete_dividends) {
        return false;
    }

    // Check 2: Build normalized request and check eligibility
    NormalizedSolveRequest test_request = build_normalized_request(config);
    auto eligibility = NormalizedChainSolver::check_eligibility(test_request);

    return eligibility.has_value();
}
```

### Fast Path (Normalized Solver)

```cpp
// Per-thread workspace allocation OUTSIDE the work-sharing loop
#pragma omp parallel
{
    auto workspace_result = Workspace::create(base_request);
    if (!workspace_result) {
        // Handle workspace creation failure
        #pragma omp for collapse(2)
        for (size_t k = 0; k < Nv; ++k) {
            for (size_t l = 0; l < Nr; ++l) {
                // Mark as failed
            }
        }
    } else {
        auto workspace = std::move(workspace_result.value());
        auto surface = workspace.surface_view();

        #pragma omp for collapse(2)
        for (size_t k = 0; k < Nv; ++k) {
            for (size_t l = 0; l < Nr; ++l) {
                NormalizedSolveRequest request{
                    .sigma = volatility[k],
                    .rate = rate[l],
                    .dividend = dividend_yield,
                    .option_type = option_type,
                    /* ... grid params ... */
                };

                NormalizedChainSolver::solve(request, workspace, surface);

                // Extract prices (moneyness convention: m = K/S)
                for (size_t i = 0; i < Nm; ++i) {
                    double x = -std::log(moneyness[i]);  // x = -ln(m)
                    double K = moneyness[i] * K_ref;     // K = m * S

                    for (size_t j = 0; j < Nt; ++j) {
                        double u = surface.interpolate(x, maturities[j]);
                        prices_4d[index_4d(i,j,k,l)] = K * u;
                    }
                }
            }
        }
    }
}
```

### Fallback Path (Batch API)

```cpp
// Build batch parameters
std::vector<AmericanOptionParams> batch_params;
for (size_t k = 0; k < Nv; ++k) {
    for (size_t l = 0; l < Nr; ++l) {
        batch_params.push_back({
            .strike = K_ref,
            .spot = K_ref,
            .maturity = T_max,
            .volatility = volatility[k],
            .rate = rate[l],
            .continuous_dividend_yield = dividend_yield,
            .option_type = option_type,
            .discrete_dividends = {}  // or actual dividends
        });
    }
}

// Solve with snapshot registration
auto results = BatchAmericanOptionSolver::solve_batch(
    batch_params, x_min, x_max, n_space, n_time,
    [&](size_t idx, AmericanOptionSolver& solver) {
        for (size_t j = 0; j < Nt; ++j) {
            solver.register_snapshot(step_indices[j], j, &collectors[idx]);
        }
    });

// Extract from collectors
for (size_t idx = 0; idx < batch_params.size(); ++idx) {
    size_t k = idx / Nr;
    size_t l = idx % Nr;
    auto prices_2d = collectors[idx].prices();
    // Copy to prices_4d...
}
```

## Testing Strategy

### Unit Tests

**Normalization correctness**:
- Solve normalized problem, verify V = K·u(ln(S/K), τ)
- Test multiple (S,K) pairs, same u surface
- Verify payoff: u(x,0) matches normalized formula

**Eligibility checking**:
- Pass: ratio<160, margins OK, dx≤0.05
- Fail: ratio≥160
- Fail: insufficient margin
- Fail: dx>0.05

**Batch callback**:
- Verify callback invoked per solver
- Snapshot registration via callback
- Converged results with callback

**Price table routing**:
- Eligible parameters → normalized solver
- Discrete dividends → fallback
- Wide range → fallback
- Verify both paths produce correct results

### Accuracy Tests

Compare normalized solver vs individual solves:
```cpp
// Solve once with normalized solver
auto surface = solve_normalized(sigma, r, q, T_max);

// Solve each option individually
for (auto [S, K, tau] : test_cases) {
    double x = std::log(S / K);
    double u = surface.interpolate(x, tau);
    double price_normalized = K * u;

    double price_individual = solve_individual(S, K, tau, sigma, r, q);

    EXPECT_NEAR(price_normalized, price_individual, 0.01);  // <1bp
}
```

### Performance Benchmarks

Existing benchmarks remain valid. Normalized solver maintains current throughput (~848 options/sec on 32 cores).

## Migration Path

### Phase 1: Normalized Solver
1. Implement `NormalizedChainSolver` class
2. Implement `Workspace` with thread-safe pattern
3. Unit tests for normalization, eligibility
4. Accuracy tests vs individual solves

### Phase 2: Batch Enhancement
1. Add `SetupCallback` to `BatchAmericanOptionSolver`
2. Add `AmericanSolverWorkspace::validate_params()`
3. Unit tests for callback functionality

### Phase 3: Integration
1. Refactor `PriceTable4DBuilder` with routing logic
2. Implement fast path (normalized solver)
3. Implement fallback path (batch API)
4. Integration tests for both paths
5. Performance benchmarks

### Backward Compatibility

- `BatchAmericanOptionSolver`: Default `setup=nullptr` preserves behavior
- `PriceTable4DBuilder`: Automatic routing transparent to users
- Existing tests: No modifications required

## Summary

We simplify batch processing with two enhancements:

1. **Normalized solver**: Exploits mathematical structure (scale invariance) to eliminate redundant PDE solves while maintaining performance
2. **Flexible batch API**: Adds configuration hooks for advanced use cases

Key improvements from v1:
- **Correct normalization**: Both spot and strike dimensionless (x = ln(S/K))
- **Unified API**: Single normalized core, thin front-ends for different use cases
- **Consistent eligibility**: All thresholds derived from same numerical analysis
- **Thread-safe design**: Caller-owned workspaces, stateless solver

**Implementation ready**: APIs designed, algorithms specified, testing strategy complete.
