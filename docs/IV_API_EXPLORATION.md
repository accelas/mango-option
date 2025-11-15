# IV Solver API Exploration Summary

## Overview

This document summarizes the current implied volatility (IV) solver API structure, batch processing patterns, and the relationship between FDM-based and interpolation-based approaches.

## Current IV Solver Architecture

### 1. FDM-Based IV Solver (Primary)

**File:** `/home/kai/work/iv_calc/src/option/iv_solver.hpp`

#### IVSolver Class Structure

```cpp
class IVSolver {
public:
    explicit IVSolver(const IVParams& params, const IVConfig& config);
    IVResult solve();

private:
    IVParams params_;        // Problem definition
    IVConfig config_;        // Solver configuration
    
    // Helper methods
    expected<void, std::string> validate_params() const;
    double estimate_upper_bound() const;
    double estimate_lower_bound() const;
    double objective_function(double volatility) const;  // f(σ) = V(σ) - V_market
};
```

#### Input Parameters: `IVParams`

```cpp
struct IVParams {
    double spot_price;              // S (current stock price)
    double strike;                  // K
    double time_to_maturity;        // T (years)
    double risk_free_rate;          // r
    double market_price;            // Observed option price
    bool is_call;                   // true = call, false = put
};
```

#### Configuration: `IVConfig`

```cpp
struct IVConfig {
    RootFindingConfig root_config;  // Brent method parameters
    size_t grid_n_space = 101;      // Spatial grid points for PDE
    size_t grid_n_time = 1000;      // Time steps for PDE
    double grid_s_max = 200.0;      // Maximum spot price for grid
};
```

#### Output: `IVResult`

```cpp
struct IVResult {
    bool converged;                         // Success indicator
    size_t iterations;                      // Newton/Brent iterations
    double implied_vol;                     // Solved σ
    double final_error;                     // |V(σ) - V_market|
    std::optional<std::string> failure_reason;  // Error description
    std::optional<double> vega;             // ∂V/∂σ (if available)
};
```

#### Algorithm

- **Root-Finding Method:** Brent's method (bracketing + interpolation)
- **Bounds Estimation:** Adaptive based on time value analysis
  - High time value (>50%): σ_upper = 300%
  - Moderate time value (20-50%): σ_upper = 200%
  - Low time value (<20%, deep ITM): σ_upper = 150%
  - σ_lower = 1% (always)

- **Objective Function:** f(σ) = V(σ) - V_market
  - Evaluates American option via PDE for each candidate σ
  - Creates adaptive workspace per call
  - Returns NaN on solver failure

#### Performance

- Single IV calculation: ~143ms (43% faster than 250ms target)
- Grid configuration: 101 space × 1000 time steps
- No reuse between solves (each creates fresh workspace)

---

### 2. Interpolation-Based IV Solver (Fast)

**File:** `/home/kai/work/iv_calc/src/option/iv_solver_interpolated.hpp`

#### IVSolverInterpolated Class Structure

```cpp
class IVSolverInterpolated {
public:
    // Constructor with explicit bounds specification
    IVSolverInterpolated(
        const BSpline4D& price_surface,
        double K_ref,
        std::pair<double, double> m_range,    // moneyness bounds
        std::pair<double, double> tau_range,  // maturity bounds
        std::pair<double, double> sigma_range,// volatility bounds
        std::pair<double, double> r_range,    // rate bounds
        const IVSolverConfig& config = {});

    // Convenience constructor from PriceTableSurface
    IVSolverInterpolated(
        const PriceTableSurface& surface,
        const IVSolverConfig& config = {});

    IVResult solve(const IVQuery& query) const;

private:
    const BSpline4D& price_surface_;  // Pre-computed 4D B-spline
    double K_ref_;                    // Reference strike
    std::pair<double, double> m_range_, tau_range_, sigma_range_, r_range_;
    IVSolverConfig config_;
};
```

#### Query Parameters: `IVQuery`

```cpp
struct IVQuery {
    double market_price;            // Observed price
    double spot;                    // S (current price)
    double strike;                  // K (actual strike, may differ from K_ref)
    double maturity;                // T (years)
    double rate;                    // r
    OptionType option_type = OptionType::PUT;  // CALL or PUT
};
```

#### Configuration: `IVSolverConfig`

```cpp
struct IVSolverConfig {
    int max_iterations = 50;        // Newton iterations
    double tolerance = 1e-6;        // Price convergence tolerance
    double vega_epsilon = 1e-4;     // FD step for vega (if not using analytic)
    double sigma_min = 0.01;        // Min volatility bound (1%)
    double sigma_max = 3.0;         // Max volatility bound (300%)
};
```

#### Algorithm

- **Method:** Newton-Raphson iteration
  - σ_{n+1} = σ_n - f(σ_n) / f'(σ_n)
  - f(σ) = Price(m, τ, σ, r) - Market_Price
  - f'(σ) = Vega (computed via analytic B-spline derivative)

- **Bounds:** Adaptive based on time value
  - Same logic as FDM solver

- **Strike Scaling:** Uses K_ref for moneyness, scales prices
  - Surface built with m = S/K_ref
  - Price scaling: V(K) = V(K_ref) × (K/K_ref)
  - Vega scaling: ∂(V_ref × K/K_ref)/∂σ = (K/K_ref) × ∂V_ref/∂σ

#### Performance

- B-spline evaluation: ~500ns per price query
- Vega computation: ~1µs (analytic derivative via Cox-de Boor formula)
- Newton iterations: 3-5 typical
- Total IV solve: **10-30µs** (4,800× speedup vs FDM)

---

### 3. Price Table Surface Structure

**File:** `/home/kai/work/iv_calc/src/option/price_table_4d_builder.hpp`

#### PriceTableSurface Class

```cpp
class PriceTableSurface {
public:
    PriceTableSurface() = default;
    explicit PriceTableSurface(std::shared_ptr<PriceTableWorkspace> workspace);

    bool valid() const;
    double eval(double m, double tau, double sigma, double rate) const;
    double K_ref() const;
    double dividend_yield() const;
    
    // Range queries
    std::pair<double, double> moneyness_range() const;
    std::pair<double, double> maturity_range() const;
    std::pair<double, double> volatility_range() const;
    std::pair<double, double> rate_range() const;
    
    std::shared_ptr<PriceTableWorkspace> workspace() const;
};
```

#### Builder: PriceTable4DBuilder

```cpp
class PriceTable4DBuilder {
public:
    // Primary constructor: 4D grids + K_ref
    static PriceTable4DBuilder create(
        std::vector<double> moneyness,    // m = S/K, sorted >=4 points
        std::vector<double> maturity,     // τ (years), sorted >=4 points
        std::vector<double> volatility,   // σ, sorted >=4 points
        std::vector<double> rate,         // r, sorted >=4 points
        double K_ref);

    // Convenience: from strike prices
    static PriceTable4DBuilder from_strikes(
        double spot,
        std::vector<double> strikes,      // Auto-computes moneyness
        std::vector<double> maturities,
        std::vector<double> volatilities,
        std::vector<double> rates);

    // Convenience: from market chain data
    static PriceTable4DBuilder from_chain(const OptionChain& chain);

    // Pre-computation
    expected<PriceTable4DResult, std::string> precompute(
        OptionType option_type,
        size_t n_space = 101,
        size_t n_time = 1000);

    // Get evaluator
    PriceTableSurface get_surface();
};
```

#### Result: PriceTable4DResult

```cpp
struct PriceTable4DResult {
    PriceTableSurface surface;              // User-friendly interface
    std::shared_ptr<BSpline4D> evaluator;  // Fast evaluator
    std::vector<double> prices_4d;          // Raw 4D array
    size_t n_pde_solves;                    // Number of PDEs solved
    double precompute_time_seconds;         // Wall-clock time
    BSplineFittingStats fitting_stats;      // Diagnostics
};
```

---

## Batch Processing Patterns

### 1. FDM Batch IV Solver

**File:** `/home/kai/work/iv_calc/src/option/iv_solver.hpp` (lines 144-211)

#### BatchIVSolver Class

```cpp
class BatchIVSolver {
public:
    static std::vector<IVResult> solve_batch(
        std::span<const IVParams> params,
        const IVConfig& config)
    {
        std::vector<IVResult> results(params.size());
        
        MANGO_PRAGMA_PARALLEL_FOR
        for (size_t i = 0; i < params.size(); ++i) {
            IVSolver solver(params[i], config);
            results[i] = solver.solve();
        }
        
        return results;
    }

    // Vector overload
    static std::vector<IVResult> solve_batch(
        const std::vector<IVParams>& params,
        const IVConfig& config);
};
```

#### Performance

- Single-threaded: ~7 IVs/sec (101×1000 grid)
- Parallel (32 cores): ~107 IVs/sec (15.3× speedup)

#### Usage Pattern

```cpp
std::vector<IVParams> batch = { ... };
IVConfig config;  // Shared configuration

auto results = solve_implied_vol_batch(batch, config);

// Or explicitly use BatchIVSolver
auto results = BatchIVSolver::solve_batch(batch, config);
```

#### Key Characteristics

- **Simple:** One IVSolver per option
- **Embarrassingly parallel:** Each option independent
- **No workspace reuse:** Each solver creates own workspace
- **OpenMP:** Uses `MANGO_PRAGMA_PARALLEL_FOR`

---

### 2. American Option Batch Pattern (Template)

**File:** `/home/kai/work/iv_calc/src/option/american_option.hpp` (lines 247-400)

This shows a more sophisticated batch pattern that can inspire IV solver improvements:

#### BatchAmericanOptionSolver

```cpp
class BatchAmericanOptionSolver {
public:
    using SetupCallback = std::function<void(size_t index, AmericanOptionSolver&)>;

    static std::vector<expected<AmericanOptionResult, SolverError>> solve_batch(
        std::span<const AmericanOptionParams> params,
        double x_min, double x_max, size_t n_space, size_t n_time,
        SetupCallback setup = nullptr);
};
```

#### Key Features

1. **Per-thread Workspace Reuse:**
   ```cpp
   auto workspace_result = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);
   // Reuse across multiple solves within same thread
   ```

2. **Setup Callbacks:** Register snapshots, configure solvers before solve()

3. **Error Handling:**
   - Validates workspace parameters once upfront
   - Returns `expected<Result, SolverError>` for each option
   - Graceful degradation on OOM

4. **Thread Safety:**
   - Each thread has its own workspace
   - Workspace arrays are modified during solve (not thread-safe for sharing)

---

### 3. American Solver Workspace

**File:** `/home/kai/work/iv_calc/src/option/american_solver_workspace.hpp`

#### Key Design Points

```cpp
class AmericanSolverWorkspace {
public:
    static expected<AmericanSolverWorkspace, std::string> create(
        double x_min, double x_max, size_t n_space, size_t n_time);

    // Grid configuration accessors
    // PDEWorkspace members (inherited)
    std::span<double> u_current(), u_next(), u_stage();
    std::span<double> rhs(), lu(), psi_buffer();
};
```

#### Design Rationale

- **Reuse Pattern:** Each thread creates workspace once, reuses for all its solves
- **Per-thread Safety:** Multiple threads don't share workspace
- **Memory Savings:** ~1.6 KB saved per reuse (800 bytes grid + 800 bytes spacing)
- **Typical Case:** 200 solves = ~320 KB saved with batch solving

---

## Relating FDM and Interpolation Approaches

### Data Flow Diagram

```
Market Data (Strikes, Maturities, Prices, Rates)
    ↓
    ├─→ PriceTable4DBuilder.from_chain()
    │       ↓
    │   Define 4D grids (m, τ, σ, r)
    │       ↓
    │   precompute() [200× PDE solves]
    │       ↓
    │   Fit B-spline coefficients
    │       ↓
    │   PriceTableSurface (evaluator)
    │
    └─→ IVSolverInterpolated(surface)
            ↓
        IVQuery {market_price, spot, strike, ...}
            ↓
        Newton iteration (3-5 steps)
            ↓
        IVResult {implied_vol, ...}


OR (for single options):

    IVParams {spot_price, strike, maturity, price, rate}
        ↓
    IVSolver
        ↓
    Brent iteration with embedded PDE solves
        ↓
    IVResult
```

---

## API Patterns Summary

### 1. Constructor Patterns

| Component | Pattern | Notes |
|-----------|---------|-------|
| IVSolver | Explicit constructor | Takes IVParams + IVConfig |
| IVSolverInterpolated | Two constructors | Direct (explicit bounds) or from PriceTableSurface |
| AmericanOptionSolver | Workspace-based | Takes params + shared_ptr<Workspace> |
| BatchSolvers | Static factory | solve_batch(span, config) |

### 2. Result Patterns

| Component | Result Type | Key Fields |
|-----------|------------|------------|
| IVSolver | IVResult | converged, iterations, implied_vol, failure_reason, vega |
| AmericanOption | AmericanOptionResult | value, delta, gamma, theta, converged |
| BatchSolver | vector<Result> | Same as individual result type |

### 3. Configuration Patterns

| Component | Config | Purpose |
|-----------|--------|---------|
| IVSolver | IVConfig | Root-finding + PDE grid |
| IVSolverInterpolated | IVSolverConfig | Newton iterations + bounds |
| AmericanOptionSolver | TRBDF2Config + RootFindingConfig | Time-stepping + Newton |

### 4. Batch Patterns

| Component | Pattern | Thread Safety |
|-----------|---------|----------------|
| BatchIVSolver | Simple parallel loop | Each thread independent |
| BatchAmericanOption | Per-thread workspace | Validates params once upfront |
| PriceTable4DBuilder | OpenMP reduction | Parallel slice collection |

---

## Key Design Insights

### 1. Strike Handling

**Critical Point in IVSolverInterpolated:**
- Price surface built with moneyness m = S/K_ref
- For queries with different strikes:
  - Compute m = S/K_ref (not S/K)
  - Scale price: V(K) = V(K_ref) × (K/K_ref)
  - Scale vega: ∂V/∂σ scales by same factor
- Allows single surface to serve multiple strikes

### 2. Workspace Reuse Strategy

**American Option Pattern:**
```cpp
// Sequential: Reuse across many solves
auto workspace = AmericanSolverWorkspace::create(...).value();
for (auto params : options) {
    auto solver = AmericanOptionSolver::create(params, workspace).value();
    solver.solve();
}

// Parallel: Per-thread workspace (no sharing!)
#pragma omp parallel {
    auto workspace = AmericanSolverWorkspace::create(...).value();
    #pragma omp for
    for (...) {
        // Each thread owns workspace
    }
}
```

This pattern could improve FDM batch IV solver.

### 3. Error Handling

- **Exception-based:** Constructors validate and throw
- **Expected-based:** Factory methods return expected<T, Error>
- **Consistent:** Both IVSolver and AmericanOptionSolver support both patterns

### 4. Adaptive Bounds

Both FDM and interpolation solvers use identical adaptive bound logic:
- Analyze time value as % of market price
- High time value → search higher σ
- Low time value (deep ITM) → search lower σ
- Monotone computation of intrinsic value

---

## Files Referenced

### Primary IV Solver Files
- `/home/kai/work/iv_calc/src/option/iv_solver.hpp` - FDM solver interface + batch
- `/home/kai/work/iv_calc/src/option/iv_solver.cpp` - FDM solver implementation
- `/home/kai/work/iv_calc/src/option/iv_solver_interpolated.hpp` - Interpolation solver
- `/home/kai/work/iv_calc/src/option/iv_solver_interpolated.cpp` - Interpolation implementation

### Supporting Files
- `/home/kai/work/iv_calc/src/option/iv_types.hpp` - Shared types (IVResult)
- `/home/kai/work/iv_calc/src/option/american_option.hpp` - American option solver + batch pattern
- `/home/kai/work/iv_calc/src/option/american_solver_workspace.hpp` - Workspace pattern
- `/home/kai/work/iv_calc/src/option/price_table_4d_builder.hpp` - Price table builder
- `/home/kai/work/iv_calc/src/option/price_table_workspace.hpp` - Price table storage
- `/home/kai/work/iv_calc/src/option/snapshot.hpp` - Snapshot collection pattern

---

## Summary

The codebase provides **three complementary IV solving approaches:**

1. **FDM (IVSolver):** Ground truth, ~143ms per IV, suitable for single calculations
2. **Interpolation (IVSolverInterpolated):** Fast, ~20µs per IV, requires pre-computed table
3. **Batch (BatchIVSolver + BatchAmericanOption):** Parallelizes with OpenMP, per-thread workspace reuse

The **American option solver** shows a more sophisticated batch pattern with per-thread workspace reuse that could inform improvements to batch IV solving.
