# Adaptive Grid Builder Design

**Date:** 2025-11-27
**Status:** Approved (rev 2)
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
   - Initialize slice_cache as map<(σ, r), SliceData> (keyed by actual values, not indices)

2. MAIN LOOP (iteration = 0..max_iterations)
   a. BUILD/UPDATE TABLE
      - Identify (σ, r) pairs not in slice_cache
      - If m or τ grid changed since last iteration: clear cache, rebuild all
      - Solve PDE for missing pairs, add to slice_cache
      - Extract tensor from all cached slices
      - Fit B-spline coefficients → current_surface

   b. GENERATE VALIDATION SAMPLE
      - Latin hypercube over [m, τ, σ, r] domain (validation_samples points)
      - Include chain's actual (strike, maturity) combinations

   c. VALIDATE AGAINST FRESH FD SOLVES
      - For each sample point (m, τ, σ, r):
          interpolated_price = current_surface.value(m, τ, σ, r)
          reference_price = solve_american_fd(m, τ, σ, r)  // Fresh solve
          iv_error = |implied_vol(interpolated) - implied_vol(reference)|
      - Record max_error, avg_error

   d. CHECK CONVERGENCE
      - If max_error ≤ target_iv_error → break (success)
      - If iteration == max_iterations → break (best effort)

   e. DIAGNOSE & REFINE
      - Bin each high-error sample by position in each dimension
      - Identify dimension with most concentrated errors
      - Insert midpoints in problematic bins for that dimension
      - Note: if m or τ changed, next iteration rebuilds all slices

3. RETURN AdaptiveResult
   - target_met = (achieved_max_error <= target_iv_error)
   - Include full iteration history for diagnostics
```

## Slice Caching

```cpp
/// Cached PDE solution for a (σ, r) slice
/// Keyed by actual (sigma, rate) values to survive grid remapping
struct SliceData {
    double sigma;
    double rate;
    std::vector<double> log_moneyness_grid;  // From PDE spatial grid
    std::vector<double> maturity_grid;       // τ points where we have data
    std::vector<std::vector<double>> prices; // prices[tau_idx][m_idx]
};

class AdaptiveGridBuilder {
private:
    // Cache keyed by (σ, r) values with tolerance for floating-point matching
    std::map<std::pair<double, double>, SliceData> slice_cache_;

    // Track current m/τ grids to detect when cache must be invalidated
    std::vector<double> current_m_grid_;
    std::vector<double> current_tau_grid_;

    bool grids_changed(const PriceTableAxes<4>& new_axes) const;
    void invalidate_cache();
};
```

**Cache invalidation rules:**
- σ or r grid change: Keep slices for (σ, r) pairs that still exist; compute new pairs
- m or τ grid change: Invalidate entire cache; rebuild all slices

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

tests/
├── adaptive_grid_builder_test.cc  # Unit tests
├── adaptive_grid_integration_test.cc  # End-to-end with real chain data
```

## Test Cases

1. **Convergence test** - Given chain data and 5 bps target, verify `target_met == true` within max_iterations
2. **Cache reuse test** - After σ-only refinement, verify existing slices not recomputed
3. **Cache invalidation test** - After τ refinement, verify all slices recomputed
4. **Error attribution test** - Inject artificial error at specific region, verify correct dimension/bin identified
5. **Fallback test** - With impossible target, verify `target_met == false` with best-effort result
6. **Determinism test** - Same inputs produce same output (Latin hypercube seeded)

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

- Uses `PriceTableBuilder` internally (composition)
- Uses `AmericanOptionSolver` for validation FD solves
- Needs Latin hypercube sampling utility (may add `src/math/latin_hypercube.hpp`)
- No new external dependencies

## Future Optimizations (Out of Scope)

1. **Time lattice caching** - Modify `AmericanOptionSolver` to store snapshots at all τ during a single solve, enabling cheap τ refinement
2. **Parallel validation** - Run validation FD solves in parallel (currently serial)
3. **Smarter sampling** - Concentrate samples near early exercise boundary where errors are highest
