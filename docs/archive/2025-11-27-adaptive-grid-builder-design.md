<!-- SPDX-License-Identifier: MIT -->
# Adaptive Grid Builder Design

**Date:** 2025-11-27
**Status:** Implemented
**Author:** Claude + Kai

## Overview

Replace the one-shot analytical grid estimator with a feedback-driven adaptive builder that iteratively refines grid density until target IV error is achieved.

## Problem

The current `estimate_grid_for_price_table()` uses curvature-based heuristics to guess grid sizes. This often over- or under-provisions points because:
- American option early exercise boundaries create non-smooth regions
- The h⁴ error theory assumes smooth functions
- No validation that the target is actually met

## Solution

A goal-seeking controller that:
1. Seeds with analytical estimate
2. Builds table, validates against **fresh FD solves** at sample points
3. Diagnoses which dimension contributes most error
4. Refines that dimension locally
5. Repeats until target met or limits reached

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Ground truth | Fresh FD American solves | Self-referential validation (spline vs spline) cannot detect grid inadequacy |
| Final validation | Mandatory | Target guarantee requires verification; "optional" undermines the contract |
| Error attribution | Bin-based aggregation | Simple, identifies problem regions directly |
| σ/r refinement | Incremental slice computation | Reuses existing PDE solves when only σ/r grid changes |
| m/τ refinement | Full rebuild required | Current builder doesn't store time lattice; future optimization possible |
| API | Separate `AdaptiveGridBuilder` class | Keeps `PriceTableBuilder` simple, enables rich diagnostics |
| Builder integration | Reuse axes/extraction helpers, manage PDE solves ourselves | `AdaptiveGridBuilder` enumerates σ/r pairs via builder helpers but runs subset solves directly so cache is effective |
| Error metric | Hybrid IV/price metric with vega floor | Uses vega-scaled IV approximation when safe, falls back to price bounds when vega is tiny |

## Critical Design Notes

### Ground Truth Must Be Independent

The validation loop compares interpolated prices against **fresh FD solves**, not against cubic spline interpolation of cached data. Comparing two interpolations of the same underlying tensor only measures interpolation scheme differences—it cannot reveal whether the (σ, τ, m, r) grid is dense enough to capture the true price surface.

**Cost implication:** Each iteration requires ~50-100 FD solves at sample points (~500ms-1s per pass). This is unavoidable—you cannot validate accuracy without an independent reference.

### Any Grid Change Requires PDE Resolves

The current `PriceTableBuilder` computes prices only at the maturity grid supplied in the axes. It does not store intermediate time snapshots. Therefore:

- **σ or r refinement:** Only new (σ, r) pairs need PDE solves; existing slices are reused
- **m or τ refinement:** All (σ, r) slices must be recomputed with the new grid

**Future optimization:** Modify `AmericanOptionSolver` to store the full time lattice during a single solve, enabling cheap τ refinement without re-solving. This is out of scope for the initial implementation.

### Validation Is Mandatory

The `target_met` flag and `achieved_max_error` tell callers whether the tolerance was achieved. If `target_met == false`, callers can decide to:
- Accept the best-effort result
- Retry with relaxed tolerance
- Fall back to manual grid specification

But the builder never silently skips verification.

### Integration with PriceTableBuilder

`PriceTableBuilder` exposes internal helpers via `*_for_testing()` suffixes. We promote these to an `_internal` API and use them as follows:

- `make_batch_internal(axes)` – enumerate all `(σ, r)` combinations in deterministic order.
- `extract_tensor_internal(merged_batch, axes)` – interpolate cached PDE slices onto the requested axes.
- `fit_coeffs_internal(tensor, axes)` – run the B-spline fit.

Key point: we **do not** call `solve_batch_for_testing()`, because that always solves the entire batch and would defeat caching. Instead, `AdaptiveGridBuilder` maps each `(σ, r)` pair to the corresponding `AmericanOptionParams` using `make_batch_internal()`, runs `BatchAmericanOptionSolver` only on the subset of pairs that are missing from the cache, and constructs a synthetic `BatchAmericanOptionResult` by ordering cached + new `AmericanOptionResult`s to match the axes layout before feeding them into `extract_tensor_internal()`.

### Error Metric: Hybrid IV / Price with Vega Floor

We still report errors in IV basis points, but avoid dividing by near-zero vegas:

```cpp
double price_error = std::abs(interpolated_price - reference_price);
double bs_vega = black_scholes_vega(spot, strike, m, τ, σ, r);
constexpr double kVegaFloor = 1e-4;  // price per 1% IV

double iv_error;
if (bs_vega >= kVegaFloor) {
    iv_error = price_error / bs_vega;           // ΔIV ≈ ΔP / vega
} else {
    // Vega too small: fall back to price tolerance derived from target IV
    double price_tol = target_iv_error * kVegaFloor;
    iv_error = price_error / kVegaFloor;        // reports worst-case IV deviation
    if (price_error <= price_tol) continue;     // treat as within tolerance
}
```

For low-vega regions (deep ITM/OTM, very short τ) IV itself is ill-defined; this fallback caps noise while still tracking whether price accuracy is within the implied tolerance. We additionally log raw price errors so future enhancements can swap in a true IV root-solve if needed.

### Mapping Validation Samples to Option Params

Validation points are sampled as `(m, τ, σ, r)` tuples. To run the fresh FD solve we convert them into `AmericanOptionParams` consistent with the table normalization:

- Use chain spot `S` as the anchor.
- Strike = `S / m` because table moneyness is defined as `spot / strike` (see `from_strikes`).
- Maturity = `τ`, volatility = `σ`, rate = `r`, option type/dividend = chain defaults.
- The option price returned by the FD solver is therefore directly comparable to the surface value at `(m, τ, σ, r)`.

### Cache Stores Raw PDE Data

The cache stores raw PDE outputs in the solver's native grid (log-moneyness), not the user-specified axes grid:

```cpp
struct SliceData {
    double sigma;
    double rate;
    // Raw PDE output - stored in solver's log-moneyness grid
    std::vector<double> pde_log_moneyness;  // e.g., 101 points from -3 to +3
    std::vector<double> pde_time_points;    // snapshot times
    std::vector<std::vector<double>> pde_prices;  // [time_idx][x_idx]
};
```

When building the tensor, we must interpolate from PDE grid to axes grid. This is handled by `extract_tensor_for_testing()`, which takes `BatchAmericanOptionResult` and interpolates each slice onto the requested `PriceTableAxes`.

**Cache workflow:**
1. Cache stores raw `AmericanOptionResult` (which contains PDE snapshots)
2. When axes change, construct synthetic `BatchAmericanOptionResult` from cached + new results
3. Call `extract_tensor_for_testing()` to interpolate onto current axes
4. Proceed with `fit_coeffs_for_testing()`

## Data Structures

```cpp
/// Configuration for adaptive grid refinement
struct AdaptiveGridParams {
    double target_iv_error = 0.0005;      // 5 bps default target
    size_t max_iterations = 5;            // Cap refinement passes
    size_t max_points_per_dim = 50;       // Per-dimension ceiling
    size_t validation_samples = 64;       // FD solves per validation pass
    double refinement_factor = 1.3;       // Grid growth per iteration
    size_t bins_per_dim = 5;              // For error attribution
};

/// Per-iteration diagnostics
struct IterationStats {
    size_t iteration;
    std::array<size_t, 4> grid_sizes;     // [m, tau, sigma, r]
    size_t pde_solves_table;              // Slices computed for table
    size_t pde_solves_validation;         // Fresh solves for validation
    double max_error;                     // Max IV error observed
    double avg_error;                     // Mean IV error
    size_t refined_dim;                   // Which dim was refined (-1 if none)
    double elapsed_seconds;
};

/// Final result with full diagnostics
struct AdaptiveResult {
    std::shared_ptr<const PriceTableSurface<4>> surface;
    PriceTableAxes<4> axes;
    std::vector<IterationStats> iterations;
    double achieved_max_error;
    double achieved_avg_error;
    bool target_met;                      // True iff achieved_max_error <= target
    size_t total_pde_solves;
};
```

## Algorithm

```
AdaptiveGridBuilder::build(chain, grid_spec, n_time, type):

1. SEED ESTIMATE
   - Use existing estimate_grid_from_chain_bounds() with initial params
   - Initialize result_cache as map<(σ, r), AmericanOptionResult>

2. MAIN LOOP (iteration = 0..max_iterations)
   a. BUILD/UPDATE TABLE
      - Create PriceTableBuilder with current config (used for axes/extraction/fitting)
      - Enumerate `(σ, r)` combos via `make_batch_internal(axes)` and map each to an index
      - Detect which combos are missing in the cache (or invalidated)
      - If m or τ grid changed: clear entire cache (results tied to old maturity)
      - Build a vector of `AmericanOptionParams` for missing combos and run `BatchAmericanOptionSolver` on that subset only (solver configured identically to builder)
      - Merge cached + new `AmericanOptionResult`s into a `BatchAmericanOptionResult` ordered exactly like the full batch
      - Extract tensor: `builder.extract_tensor_internal(merged, axes)`
      - Fit coeffs: `builder.fit_coeffs_internal(tensor, axes)`
      - Build surface: `PriceTableSurface<4>::build(axes, coeffs, metadata)`

   b. GENERATE VALIDATION SAMPLE
      - Latin hypercube over [m, τ, σ, r] domain (validation_samples points)
      - Include chain's actual (strike, maturity) combinations

   c. VALIDATE AGAINST FRESH FD SOLVES
      - For each sample point (m, τ, σ, r):
          * Convert to option params: `spot = chain.spot`, `strike = spot / m`
          * interpolated_price = current_surface.value(...)
          * reference_price = solve_american_fd(spot, strike, τ, σ, r)
          * iv_error = hybrid_metric(interpolated_price, reference_price, spot, strike, τ, σ, r)
      - Record max_error, avg_error

   d. CHECK CONVERGENCE
      - If max_error ≤ target_iv_error → break (success)
      - If iteration == max_iterations → break (best effort)

   e. DIAGNOSE & REFINE
      - Bin each high-error sample by position in each dimension
      - Identify dimension with most concentrated errors
      - Insert midpoints in problematic bins for that dimension
      - Note: if m or τ changed, next iteration clears cache

3. RETURN AdaptiveResult
   - target_met = (achieved_max_error <= target_iv_error)
   - Include full iteration history for diagnostics
```

## Slice Caching

```cpp
/// Cache keyed by (σ, r) values
/// Stores raw AmericanOptionResult which contains PDE snapshots
class SliceCache {
    std::map<std::pair<double, double>, AmericanOptionResult> results_;

    // Track current m/τ grids - cache invalid if these change
    std::vector<double> current_tau_grid_;

public:
    void add(double sigma, double rate, AmericanOptionResult result);
    std::optional<AmericanOptionResult> get(double sigma, double rate) const;
    void invalidate_if_tau_changed(const std::vector<double>& new_tau);

    // Build BatchAmericanOptionResult from cached + new results
    BatchAmericanOptionResult merge_with_new(
        const std::vector<std::pair<double, double>>& all_pairs,
        const BatchAmericanOptionResult& new_results,
        const std::vector<size_t>& new_indices);
};
```

**Cache invalidation rules:**
- σ or r grid change: Keep results for (σ, r) pairs that still exist; solve new pairs
- m grid change: Cache remains valid (extract_tensor handles interpolation)
- τ grid change: Invalidate entire cache (PDE solve is τ-dependent)

## Error Attribution

```cpp
struct ErrorBins {
    static constexpr size_t N_BINS = 5;
    std::array<std::array<size_t, N_BINS>, 4> bin_counts = {};
    std::array<double, 4> dim_error_mass = {};

    void record_error(const std::array<double, 4>& normalized_pos,
                      double iv_error, double threshold);
    size_t worst_dimension() const;
    std::vector<size_t> problematic_bins(size_t dim, size_t min_count = 2) const;
};
```

This localizes refinement: if short maturities have most errors, we add points there rather than uniformly.

## File Organization

```
src/option/table/
├── adaptive_grid_builder.hpp      # AdaptiveGridBuilder class, params, result structs
├── adaptive_grid_builder.cpp      # Implementation
├── error_attribution.hpp          # ErrorBins, refinement helpers (header-only)
├── slice_cache.hpp                # SliceCache class (header-only)

tests/
├── adaptive_grid_builder_test.cc  # Unit tests
├── adaptive_grid_integration_test.cc  # End-to-end with real chain data
```

## Test Cases

1. **Convergence test** - Given chain data and 5 bps target, verify `target_met == true` within max_iterations
2. **Cache reuse test** - After σ-only refinement, verify existing slices not recomputed
3. **Cache invalidation test** - After τ refinement, verify all slices recomputed
4. **m refinement cache test** - After m-only refinement, verify cache NOT invalidated (only extract_tensor re-runs)
5. **Error attribution test** - Inject artificial error at specific region, verify correct dimension/bin identified
6. **Fallback test** - With impossible target, verify `target_met == false` with best-effort result
7. **Determinism test** - Same inputs produce same output (Latin hypercube seeded)
8. **Vega scaling test** - Verify price_error/vega ≈ true IV error for known cases

## Public API

```cpp
class AdaptiveGridBuilder {
public:
    explicit AdaptiveGridBuilder(AdaptiveGridParams params);

    /// Build price table with adaptive grid refinement
    ///
    /// Returns AdaptiveResult with:
    /// - surface: The built price table (always populated, even if target not met)
    /// - target_met: True iff achieved_max_error <= target_iv_error
    /// - achieved_max_error: Actual max IV error from final validation
    /// - iterations: Full history for diagnostics
    std::expected<AdaptiveResult, PriceTableError>
    build(const OptionChain& chain,
          GridSpec<double> grid_spec,
          size_t n_time,
          OptionType type = OptionType::PUT);
};
```

## Dependencies

- Uses `PriceTableBuilder` internal methods (promote `_for_testing` to `_internal`)
- Uses `BatchAmericanOptionSolver` for incremental PDE solves
- Uses `AmericanOptionSolver` for validation FD solves
- Uses Black-Scholes vega for error scaling (existing `src/math/black_scholes.hpp`)
- Needs Latin hypercube sampling utility (may add `src/math/latin_hypercube.hpp`)
- No new external dependencies

## Future Optimizations (Out of Scope)

1. **Time lattice caching** - Modify `AmericanOptionSolver` to store snapshots at all τ during a single solve, enabling cheap τ refinement
2. **Parallel validation** - Run validation FD solves in parallel (currently serial)
3. **Smarter sampling** - Concentrate samples near early exercise boundary where errors are highest
