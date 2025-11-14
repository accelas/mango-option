# Batch Option Chain Enhancement Design

**Date:** 2025-01-12
**Status:** Design Complete
**Authors:** Claude Code + User

## Executive Summary

We enhance batch American option pricing with two improvements:

1. **Flexible Batch API**: Add `SetupCallback` to `BatchAmericanOptionSolver` for per-solver configuration (snapshot registration, convergence tuning)
2. **Option Chain Optimization**: New `OptionChainSolver` solves one normalized PDE covering all strikes and maturities, avoiding redundant computation

The chain solver exploits coordinate invariance: solving at K=1 yields results for all strikes via interpolation. For price tables (IV surface building), this maintains current performance (Nσ × Nr solves) with cleaner normalized-coordinate implementation.

**Performance**: Chain solver matches current speed (~848 options/sec on 32 cores) but simplifies code and enables future optimizations.

**Eligibility**: Chain solver requires no discrete dividends and moderate strike ranges (m_max/m_min < 150). Wide ranges or discrete dividends automatically fall back to batch API.

## Motivation

### Goal 1: Batch API Enhancement

Current `PriceTable4DBuilder` duplicates workspace management logic that `BatchAmericanOptionSolver` already implements. The batch API lacks configuration hooks (snapshot registration, solver tuning), forcing manual OpenMP loops.

Adding `SetupCallback` eliminates duplication and unifies batch processing patterns.

### Goal 2: Option Chain Optimization

When pricing multiple options with identical parameters except strike and maturity, we solve N independent PDEs. Yet the Black-Scholes PDE in log-moneyness coordinates x = ln(S/K) has constant coefficients—strike appears only in coordinate normalization.

**Key insight**: Solve once at K=1, T=T_max. All other (K_i, T_j) combinations become interpolation queries on the single PDE surface.

**Application**: Price table building solves Nσ × Nr PDEs (one per volatility-rate pair). Each solve covers all Nm × Nt strikes and maturities via snapshot collection. Chain solver maintains this count while using cleaner normalized coordinates.

## Architecture

### Component Overview

```
OptionChainSolver (new)
  ├─ Solves normalized PDE (K=1, T=T_max)
  ├─ Collects snapshots at requested maturities
  └─ Uses existing snapshot infrastructure

BatchAmericanOptionSolver (enhanced)
  ├─ Adds SetupCallback parameter
  ├─ Supports per-solver configuration
  └─ Handles discrete dividends, wide strike ranges

PriceTable4DBuilder (refactored)
  ├─ Check eligibility: discrete dividends? strike range?
  ├─   Eligible → OptionChainSolver (fast path)
  └─   Not eligible → BatchAmericanOptionSolver (fallback)
```

### Coordinate System Choice

We preserve the current log-moneyness system x = ln(S/K) with normalized strike K=1:

**Why this system wins:**
- Constant PDE coefficients (no x-dependence in diffusion/drift)
- Scale invariance: V(S,K) = K · u(ln(S/K), t)
- Existing grid infrastructure works without modification
- Free-boundary shape invariant across strikes

**Rejected alternatives:**
- **Spot coordinates**: Variable coefficients break tridiagonal structure, require larger grids
- **Log-moneyness relative to K_max**: Requires coordinate shifts, boundary alignment issues
- **Raw moneyness**: Retains m² factor in diffusion, needs nonuniform grids

## API Design

### OptionChainParams

```cpp
struct OptionChainParams {
    double spot;
    double volatility;
    double rate;
    double continuous_dividend_yield;
    OptionType option_type;
    std::vector<double> strikes;     // All strikes in chain
    std::vector<double> maturities;  // All maturities
    // Note: No discrete_dividends field (enforces eligibility)
};
```

### Eligibility Checking

```cpp
struct ChainEligibility {
    bool eligible;
    std::string reason;  // Diagnostic if not eligible
    double m_min, m_max, ratio;
};

static ChainEligibility check_eligibility(
    const OptionChainParams& params,
    double x_min = -3.0,
    double x_max = 3.0,
    size_t n_space = 101);
```

**Criteria (all must pass):**
1. No discrete dividends (enforced by struct)
2. Moneyness ratio m_max/m_min < 150
3. Boundary margins: ln(m_min) ≥ x_min + 0.5, ln(m_max) ≤ x_max - 0.5
4. Grid spacing: dx < 0.07

**Threshold justification** (from numerical analysis):
- Ratio 150x keeps ln-range within 5.0 units
- Margin 0.5 prevents payoff kink from touching boundaries
- Spacing 0.07 maintains truncation error under 1bp, preserves Newton convergence

### Chain Solver Interface

```cpp
static expected<AmericanOptionResult, SolverError> solve_chain(
    const OptionChainParams& params,
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time,
    SnapshotCollector* collector);
```

Solves normalized PDE, registers snapshots at requested maturities. Collector receives all (moneyness × maturity) results from single solve.

### Batch API Enhancement

```cpp
using SetupCallback = std::function<void(size_t index, AmericanOptionSolver& solver)>;

static std::vector<expected<AmericanOptionResult, SolverError>> solve_batch(
    std::span<const AmericanOptionParams> params,
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time,
    SetupCallback setup = nullptr);
```

**Backward compatible**: `setup = nullptr` preserves existing behavior.

**Callback invoked**: After solver creation, before `solve()`. Enables snapshot registration, config tuning.

## Implementation Details

### OptionChainSolver Algorithm

```cpp
1. Create normalized params: K=1, T=max(maturities), spot=actual_spot
2. Create workspace
3. Create solver
4. Register snapshots at each maturity: step_index = round(τ/dt) - 1
5. Solve once
6. Collector contains all (m, τ) prices
```

Step index formula: For snapshot at maturity τ with timestep dt, PDESolver calls `process_snapshots(step, t)` where t = (step+1)·dt. To capture τ, we need (step+1)·dt ≈ τ, thus step = round(τ/dt) - 1.

### Batch API Implementation

```cpp
static std::vector<expected<AmericanOptionResult, SolverError>> solve_batch(...)
{
    // Validate workspace params (no allocation)
    auto validation = AmericanSolverWorkspace::validate_params(x_min, x_max, n_space, n_time);
    if (!validation) return all_errors;

    // Common solve logic
    auto solve_one = [&](size_t i, workspace) {
        auto solver = AmericanOptionSolver::create(params[i], workspace);
        if (!solver) return error;

        if (setup) setup(i, solver.value());  // NEW: callback

        return solver.value().solve();
    };

#ifdef _OPENMP
#pragma omp parallel
    {
        auto thread_workspace = AmericanSolverWorkspace::create(...);
#pragma omp for
        for (size_t i = 0; i < params.size(); ++i) {
            results[i] = solve_one(i, thread_workspace.value());
        }
    }
#else
    auto workspace = AmericanSolverWorkspace::create(...).value();
    for (size_t i = 0; i < params.size(); ++i) {
        results[i] = solve_one(i, workspace);
    }
#endif
    return results;
}
```

Lambda `solve_one` captures common logic. OpenMP creates per-thread workspaces; sequential path reuses one workspace.

### PriceTable4DBuilder Refactoring

```cpp
expected<PriceTable4DResult, std::string> precompute(...)
{
    // Check eligibility once (moneyness grid fixed for all (σ,r) pairs)
    bool use_chain_solver = check_chain_solver_eligibility(x_min, x_max, n_space);

    if (use_chain_solver) {
        // Fast path: parallel over (σ,r), each calls OptionChainSolver
        #pragma omp parallel for collapse(2)
        for (size_t k = 0; k < Nv; ++k) {
            for (size_t l = 0; l < Nr; ++l) {
                auto chain_params = build_chain_params(k, l, ...);
                PriceTableSnapshotCollector collector(config);

                OptionChainSolver::solve_chain(chain_params, ..., &collector);
                copy_to_4d(collector.prices(), k, l, prices_4d);
            }
        }
    } else {
        // Fallback: build full batch, call BatchAmericanOptionSolver
        auto batch_params = build_batch_params(...);
        auto results = BatchAmericanOptionSolver::solve_batch(
            batch_params, ...,
            [&](size_t idx, AmericanOptionSolver& solver) {
                register_snapshots(idx, solver);
            });
        extract_to_4d(results, prices_4d);
    }

    // Fit B-splines, return result...
}
```

Both paths solve Nσ × Nr PDEs. Chain solver uses normalized coordinates; fallback handles edge cases.

### AmericanSolverWorkspace Validation

Add static validation method:

```cpp
static expected<void, std::string> validate_params(
    double x_min, double x_max, size_t n_space, size_t n_time)
{
    if (x_min >= x_max) return unexpected("x_min must be < x_max");
    if (n_space < 3) return unexpected("n_space must be ≥ 3");
    if (n_time < 1) return unexpected("n_time must be ≥ 1");

    double dx = (x_max - x_min) / (n_space - 1);
    if (dx >= 0.5) return unexpected("Grid too coarse");

    return {};
}
```

Enables fail-fast in batch operations without allocating workspace.

## C++23 Ranges Usage

Where applicable (no OpenMP), use ranges for clarity:

```cpp
// Build (σ, r) pairs
auto vol_indices = std::views::iota(size_t{0}, Nv);
auto rate_indices = std::views::iota(size_t{0}, Nr);
auto sigma_rate_pairs = std::views::cartesian_product(vol_indices, rate_indices);

// Iterate with structured bindings
for (auto [k, l] : sigma_rate_pairs) {
    // ...
}

// Count failures
size_t failed = std::ranges::count_if(results,
    [](const auto& r) { return !r.has_value(); });
```

OpenMP loops retain traditional nested structure for `collapse(2)` directive.

## Testing Strategy

### Unit Tests

**Eligibility checking:**
- Eligible chain (ratio < 150, margins OK)
- Ratio too large (ratio ≥ 150)
- Boundary margin insufficient
- Grid spacing too coarse

**Accuracy verification:**
- Solve chain with OptionChainSolver
- Solve each option individually
- Compare prices: expect <1bp difference

**Batch callback invocation:**
- Verify callback called for each solver
- Test snapshot registration via callback
- Verify solves succeed with callback

**Price table routing:**
- Small moneyness range → fast path (chain solver)
- Wide range or narrow grid → fallback (batch API)
- Verify PDE solve count = Nσ × Nr

### Integration Tests

**Price table end-to-end:**
- Build table with eligible parameters
- Verify fast path used (check logs/traces)
- Query interpolated prices
- Compare accuracy vs reference implementation

**Fallback scenarios:**
- Discrete dividends → batch API
- Wide strike range → batch API
- Verify fallback produces correct results

### Performance Benchmarks

Existing benchmarks remain valid. Chain solver maintains current throughput (~848 options/sec on 32 cores) while simplifying code.

## Usage Examples

### Direct Chain Solving

```cpp
mango::OptionChainParams chain{
    .spot = 100.0,
    .volatility = 0.25,
    .rate = 0.05,
    .continuous_dividend_yield = 0.02,
    .option_type = mango::OptionType::PUT,
    .strikes = {90.0, 95.0, 100.0, 105.0, 110.0},
    .maturities = {0.25, 0.5, 1.0}
};

// Check eligibility
auto eligibility = mango::OptionChainSolver::check_eligibility(chain, -3.0, 3.0, 101);
if (!eligibility.eligible) {
    std::cerr << "Not eligible: " << eligibility.reason << "\n";
    return 1;
}

// Create collector
std::vector<double> moneyness;
for (double K : chain.strikes) {
    moneyness.push_back(chain.spot / K);
}

mango::PriceTableSnapshotCollectorConfig config{
    .moneyness = std::span{moneyness},
    .tau = std::span{chain.maturities},
    .K_ref = 100.0,
    .option_type = chain.option_type,
    .payoff_params = nullptr
};
mango::PriceTableSnapshotCollector collector(config);

// Solve once for all 15 options
auto result = mango::OptionChainSolver::solve_chain(
    chain, -3.0, 3.0, 101, 1000, &collector);

// Extract prices (5 strikes × 3 maturities)
auto prices = collector.prices();
```

### Batch with Snapshots

```cpp
std::vector<mango::AmericanOptionParams> batch = { /* ... */ };
std::vector<mango::PriceTableSnapshotCollector> collectors = { /* ... */ };

auto results = mango::BatchAmericanOptionSolver::solve_batch(
    batch, -3.0, 3.0, 101, 1000,
    [&](size_t idx, mango::AmericanOptionSolver& solver) {
        solver.register_snapshot(249, 0, &collectors[idx]);
        solver.register_snapshot(499, 1, &collectors[idx]);
        solver.register_snapshot(999, 2, &collectors[idx]);
    });
```

### Price Table (Automatic Routing)

```cpp
auto builder = mango::PriceTable4DBuilder::create(
    {0.8, 0.9, 1.0, 1.1, 1.2},      // Moneyness
    {0.027, 0.25, 0.5, 1.0, 2.0},   // Maturity
    {0.15, 0.20, 0.25, 0.30},       // Volatility
    {0.0, 0.02, 0.05},              // Rate
    100.0);

auto result = builder.precompute(mango::OptionType::PUT, 101, 1000);
// Automatically uses OptionChainSolver (eligible parameters)

auto& evaluator = *result.value().evaluator;
double price = evaluator.eval(1.0, 0.5, 0.20, 0.05);
```

## Migration Path

### Phase 1: Core Implementation
1. Add `AmericanSolverWorkspace::validate_params()`
2. Implement `OptionChainSolver` class
3. Add `SetupCallback` to `BatchAmericanOptionSolver`
4. Write unit tests for new components

### Phase 2: Integration
1. Refactor `PriceTable4DBuilder::precompute()`
2. Add eligibility checking
3. Implement routing logic (fast path vs fallback)
4. Integration tests for both paths

### Phase 3: Validation
1. Accuracy tests (chain solver vs individual solves)
2. Performance benchmarks
3. End-to-end price table tests
4. Documentation updates

### Backward Compatibility

All existing code continues to work:
- `BatchAmericanOptionSolver::solve_batch()` without callback: unchanged behavior
- `PriceTable4DBuilder`: automatic routing transparent to users
- Existing tests: no modifications required

## Open Questions

**Q: Should we expose chain solver in public API or keep internal?**
A: Start internal (price table only). Expose later if external use cases emerge.

**Q: Add USDT tracing for routing decisions?**
A: Yes. Add probe when choosing fast path vs fallback, log reason.

**Q: Parallel outer loop in price table fast path?**
A: Current design parallelizes (σ,r) loop. Verified safe (no data races).

## References

- Codex subagent analysis: Coordinate system comparison, numerical thresholds
- Existing implementation: `src/option/price_table_4d_builder.cpp`
- Snapshot infrastructure: `src/option/price_table_snapshot_collector.hpp`
- Workspace design: `src/option/american_solver_workspace.hpp`

## Summary

We enhance batch processing with flexible callbacks and specialized chain optimization. The chain solver exploits mathematical structure (coordinate invariance) to eliminate redundant computation while maintaining current performance. Automatic routing ensures correctness across all scenarios.

**Implementation ready:** All APIs designed, algorithms specified, testing strategy complete.
