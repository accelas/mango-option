# Price Table Builder Refactor

**Date:** 2025-01-24
**Status:** Design Complete
**Goal:** Comprehensive refactor addressing resource ownership, memory pressure, grid configuration bugs, parallelization, and validation.

## Problems Addressed

1. **Resource ownership** - PriceTableSurface stores data twice (workspace + evaluator)
2. **Memory explosion** - Full 4D tensor kept in result (>10 GB for large grids)
3. **Grid configuration ignored** - Custom x_bounds/n_space/n_time overridden by GridAccuracyParams
4. **Serial extraction bottleneck** - Extraction not parallelized despite parallel PDE solving
5. **No snapshot validation** - Invalid maturities corrupt 4D tensor silently

## Architecture Overview

### 1. Resource Ownership Chain

**Design:** Single ownership path eliminates duplicate storage.

```
PriceTableWorkspace (grids + coefficients)
         ↑ owned by
     BSpline4D (evaluator logic)
         ↑ wrapped by shared_ptr
  PriceTableSurface (user-facing API)
```

**Changes to PriceTableSurface:**
```cpp
class PriceTableSurface {
public:
    explicit PriceTableSurface(std::shared_ptr<BSpline4D> evaluator)
        : evaluator_(std::move(evaluator)) {}

    double eval(double m, double tau, double sigma, double rate) const {
        return evaluator_->eval(m, tau, sigma, rate);
    }

    // Expose workspace for serialization/diagnostics
    const PriceTableWorkspace& workspace() const {
        return evaluator_->workspace();
    }

private:
    std::shared_ptr<BSpline4D> evaluator_;
    // REMOVED: std::shared_ptr<PriceTableWorkspace> workspace_
    // REMOVED: std::unique_ptr<BSpline4D> evaluator_
};
```

**Changes to BSpline4D:**
```cpp
class BSpline4D {
public:
    static std::expected<BSpline4D, std::string> create(PriceTableWorkspace&& ws);

    double eval(double m, double tau, double sigma, double rate) const;

    const PriceTableWorkspace& workspace() const { return workspace_; }

private:
    PriceTableWorkspace workspace_;  // Owned, not shared
    // ... evaluation state
};
```

**Memory impact:** Eliminates duplicate coefficient storage. For 50×30×20×10 grid: saves ~2.4 MB per surface instance.

### 2. Lightweight Result Type

**Design:** Remove raw price tensor from result. Provide on-demand diagnostics.

**Changes to PriceTable4DResult:**
```cpp
struct PriceTable4DResult {
    PriceTableSurface surface;              // Lightweight evaluator wrapper
    size_t n_pde_solves;                    // Useful stat
    double precompute_time_seconds;         // Useful stat
    BSplineFittingStats fitting_stats;      // Fitting diagnostics

    // REMOVED: std::vector<double> prices_4d (can be >10 GB)
    // REMOVED: std::shared_ptr<BSpline4D> evaluator (redundant with surface)
};
```

**New diagnostic APIs:**
```cpp
class PriceTable4DBuilder {
public:
    // Existing precompute returns lightweight result
    std::expected<PriceTable4DResult, PriceTableError> precompute(
        const PriceTableConfig& config);

    // On-demand residual computation (does not store raw prices)
    std::expected<ResidualStats, PriceTableError> compute_residuals(
        const PriceTableSurface& surface) const;

    // Optional: stream raw prices to disk during precompute
    std::expected<void, PriceTableError> precompute_with_save(
        const PriceTableConfig& config,
        const std::filesystem::path& output_path,
        bool streaming = true);
};
```

**Memory impact:**
- 50×30×20×10 grid: removes 2.4 MB
- 200×100×50×20 grid: removes 1.6 GB

### 3. Grid Configuration Precedence

**Design:** When user supplies explicit `x_bounds`, respect them by building `GridSpec`/`TimeDomain` and passing as `custom_grid_config`. Fall back to auto-estimation only when bounds not provided.

**Validation and precedence logic:**
```cpp
std::expected<PriceTable4DResult, PriceTableError>
PriceTable4DBuilder::precompute(const PriceTableConfig& config)
{
    const double T_max = maturity_.back();
    std::optional<std::pair<GridSpec<double>, TimeDomain>> custom_grid_config;

    if (config.x_bounds.has_value()) {
        // USER-SPECIFIED BOUNDS TAKE PRECEDENCE
        auto [x_min, x_max] = config.x_bounds.value();

        // Validate bounds contain requested moneyness range
        const double x_min_requested = std::log(moneyness_.front());
        const double x_max_requested = std::log(moneyness_.back());

        if (x_min_requested < x_min || x_max_requested > x_max) {
            return std::unexpected(PriceTableError{
                .code = PriceTableErrorCode::INVALID_BOUNDS,
                .expected_value = x_min_requested,  // or x_max_requested
                .invalid_values = {x_min, x_max}
            });
        }

        // Build explicit GridSpec from config
        auto grid_result = GridSpec<double>::sinh_stretched(
            x_min, x_max, config.n_space, 2.0);
        if (!grid_result) {
            return std::unexpected(PriceTableError{
                .code = PriceTableErrorCode::INVALID_GRID_SPEC
            });
        }

        // Build TimeDomain from config
        TimeDomain time_domain = TimeDomain::from_n_steps(0.0, T_max, config.n_time);

        custom_grid_config = std::make_pair(grid_result.value(), time_domain);

        // Bypass GridAccuracyParams entirely
    } else {
        // FALL BACK TO AUTO-ESTIMATION
        // Use GridAccuracyParams (existing behavior)
        GridAccuracyParams accuracy;
        accuracy.min_spatial_points = config.n_space;
        accuracy.max_spatial_points = config.n_space;
        accuracy.max_time_steps = config.n_time;
        batch_solver.set_grid_accuracy(accuracy);
    }

    // ... build batch params with custom_grid_config
}
```

**API documentation update:**
```cpp
/// Configuration for PDE solves
///
/// Grid configuration precedence:
/// 1. If x_bounds is set: uses exact bounds, bypasses auto-estimation
/// 2. If x_bounds is nullopt: uses GridAccuracyParams for auto-estimation
///
/// Recommendation: Specify x_bounds explicitly for price table construction
/// to ensure moneyness grid fits within PDE domain.
struct PriceTableConfig {
    OptionType option_type = OptionType::PUT;
    size_t n_space = 101;
    size_t n_time = 1000;
    double dividend_yield = 0.0;
    std::optional<std::pair<double, double>> x_bounds;
};
```

**Impact:** Eliminates "moneyness outside PDE bounds" errors when users provide correct bounds.

### 4. Parallel Extraction Phase

**Design:** Flatten nested (Nv × Nr × Nt) iteration space into single parallel loop for better load balancing and scalability.

**Current structure (price_table_extraction.cpp):**
```cpp
// Outer parallel over (σ, r): 200 iterations
MANGO_PRAGMA_PARALLEL_FOR
for (size_t idx = 0; idx < Nv * Nr; ++idx) {
    // Inner SERIAL over maturity: 30 iterations
    for (size_t j = 0; j < Nt; ++j) {
        // Build spline, interpolate
    }
}
```

**New structure:**
```cpp
// Single parallel loop over all work: 6000 iterations
const size_t total_work = Nv * Nr * Nt;

MANGO_PRAGMA_PARALLEL_FOR
for (size_t work_idx = 0; work_idx < total_work; ++work_idx) {
    // Decode flat index to (vol_idx, r_idx, mat_idx)
    const size_t vol_idx = work_idx / (Nr * Nt);
    const size_t remainder = work_idx % (Nr * Nt);
    const size_t r_idx = remainder / Nt;
    const size_t mat_idx = remainder % Nt;

    // Get batch result
    const size_t batch_idx = vol_idx * Nr + r_idx;
    const auto& result_expected = batch_result.results[batch_idx];

    if (!result_expected.has_value() || !result_expected->converged) {
        continue;
    }

    const auto& result = result_expected.value();
    auto result_grid = result.grid();
    auto x_grid = result_grid->x();

    // Build spline for this (σ, r, τ) combination
    size_t step_idx = step_indices[mat_idx];
    std::span<const double> spatial_solution = result.at_time(step_idx);

    if (spatial_solution.empty()) continue;

    CubicSpline<double> spline;
    auto build_error = spline.build(x_grid, spatial_solution);

    // Interpolate to moneyness grid
    for (size_t m_idx = 0; m_idx < Nm; ++m_idx) {
        const double x = log_moneyness[m_idx];
        double V_norm = build_error ? boundary_value : spline.eval(x);
        prices_view[m_idx, mat_idx, vol_idx, r_idx] = K_ref * V_norm;
    }
}
```

**Rationale:**
- Single-level parallelism: simpler than nested OpenMP
- Better load balancing: 6000 work items vs 200
- Scales to 64+ cores: uniform ~12μs work packets
- Low overhead: grain size large enough that scheduling cost is negligible

**Performance impact:** Extraction phase now scales linearly with core count like PDE solving phase.

### 5. Snapshot Grid Validation

**Design:** Validate maturity grid up-front before any PDE work. Use structured error types without redundant messages.

**New error type (src/support/error_types.hpp):**
```cpp
enum class PriceTableErrorCode {
    INVALID_GRID_UNSORTED,
    INVALID_GRID_NEGATIVE,
    INVALID_GRID_MISSING_TERMINAL,
    INVALID_GRID_EXCEEDS_TERMINAL,
    INVALID_BOUNDS,
    INVALID_GRID_SPEC,
    WORKSPACE_CREATION_FAILED,
    BSPLINE_FIT_FAILED,
    PDE_SOLVE_FAILED
};

struct PriceTableError {
    PriceTableErrorCode code;
    std::vector<double> invalid_values;
    double expected_value = 0.0;
};
```

**Validation in precompute():**
```cpp
std::expected<PriceTable4DResult, PriceTableError>
PriceTable4DBuilder::precompute(const PriceTableConfig& config)
{
    const double T_max = maturity_.back();
    const double epsilon = 1e-10;

    // Validate maturity grid before solver setup

    // Check 1: Must be sorted
    if (!std::is_sorted(maturity_.begin(), maturity_.end())) {
        return std::unexpected(PriceTableError{
            .code = PriceTableErrorCode::INVALID_GRID_UNSORTED,
            .invalid_values = maturity_
        });
    }

    // Check 2: No negative values
    if (maturity_.front() < 0.0) {
        return std::unexpected(PriceTableError{
            .code = PriceTableErrorCode::INVALID_GRID_NEGATIVE,
            .invalid_values = {maturity_.front()}
        });
    }

    // Check 3: Must include T_max
    if (std::abs(maturity_.back() - T_max) >= epsilon) {
        return std::unexpected(PriceTableError{
            .code = PriceTableErrorCode::INVALID_GRID_MISSING_TERMINAL,
            .expected_value = T_max,
            .invalid_values = {maturity_.back()}
        });
    }

    // Check 4: All values <= T_max
    auto exceeds = std::find_if(maturity_.begin(), maturity_.end(),
                                [T_max, epsilon](double t) {
                                    return t > T_max + epsilon;
                                });
    if (exceeds != maturity_.end()) {
        std::vector<double> exceeding;
        std::copy_if(maturity_.begin(), maturity_.end(),
                     std::back_inserter(exceeding),
                     [T_max, epsilon](double t) {
                         return t > T_max + epsilon;
                     });

        return std::unexpected(PriceTableError{
            .code = PriceTableErrorCode::INVALID_GRID_EXCEEDS_TERMINAL,
            .expected_value = T_max,
            .invalid_values = exceeding
        });
    }

    // ... proceed with PDE solves
}
```

**Keep extraction assert as safety net:**
```cpp
// In extract_batch_results_to_4d (line 42)
assert(n_time != 0 && "No snapshots recorded (programming error)");
```

**Impact:** Invalid grids caught immediately with actionable errors, before expensive PDE work.

## Implementation Order

1. **Add PriceTableError type** - Foundation for all error handling
2. **Refactor BSpline4D ownership** - Own workspace, update constructor
3. **Refactor PriceTableSurface** - Wrap BSpline4D only, expose workspace
4. **Remove prices_4d from result** - Update PriceTable4DResult struct
5. **Add snapshot validation** - In precompute() before solver setup
6. **Fix grid configuration** - Respect x_bounds, build custom_grid_config
7. **Parallelize extraction** - Flatten loop in extract_batch_results_to_4d
8. **Add diagnostic methods** - compute_residuals(), precompute_with_save()
9. **Update tests** - All affected test cases
10. **Update examples/docs** - Reflect new API

## Testing Plan

### Unit Tests
- `price_table_error_test.cc` - Error type construction and handling
- `price_table_surface_test.cc` - Surface ownership and workspace access
- `price_table_validation_test.cc` - All snapshot validation cases
- `price_table_grid_config_test.cc` - x_bounds precedence logic

### Integration Tests
- `price_table_4d_builder_test.cc` - Update existing tests for new API
- Test error cases: unsorted grid, missing T_max, exceeds T_max
- Test x_bounds respected vs auto-estimation fallback
- Verify no prices_4d in result

### Performance Tests
- Measure extraction phase scaling with core count
- Verify memory usage reduction (no raw tensor in result)
- Benchmark with large grids (200×100×50×20)

## Migration Guide

### Breaking Changes

**1. PriceTable4DResult no longer contains prices_4d:**
```cpp
// OLD
auto result = builder.precompute(config);
auto& raw_prices = result.prices_4d;  // No longer exists

// NEW
auto result = builder.precompute(config);
// If you need residuals:
auto residuals = builder.compute_residuals(result.surface);
// If you need raw prices:
builder.precompute_with_save(config, "prices.parquet");
```

**2. Return type changed to PriceTableError:**
```cpp
// OLD
auto result = builder.precompute(config);
if (!result) {
    std::cerr << result.error() << "\n";  // string
}

// NEW
auto result = builder.precompute(config);
if (!result) {
    const auto& err = result.error();
    switch (err.code) {
        case PriceTableErrorCode::INVALID_GRID_UNSORTED:
            // Handle unsorted grid
            break;
        // ...
    }
}
```

**3. PriceTableSurface constructor changed:**
```cpp
// OLD (internal use only, should not affect users)
auto workspace = std::make_shared<PriceTableWorkspace>(...);
PriceTableSurface surface(workspace);

// NEW
auto evaluator = std::make_shared<BSpline4D>(
    BSpline4D::create(std::move(workspace)).value());
PriceTableSurface surface(evaluator);
```

### Non-Breaking Changes

**Grid configuration now respects x_bounds:**
```cpp
PriceTableConfig config;
config.x_bounds = {-1.0, 1.0};  // Now actually used!
config.n_space = 101;
config.n_time = 1000;

// Builds GridSpec with exact [-1.0, 1.0] bounds
auto result = builder.precompute(config);
```

**Better validation errors:**
```cpp
// OLD: Silent corruption or unclear errors
// NEW: Immediate, actionable errors
auto result = builder.precompute(config);
if (!result && result.error().code ==
    PriceTableErrorCode::INVALID_GRID_MISSING_TERMINAL) {
    // Grid missing T_max - append it and retry
}
```

## Success Metrics

- Memory usage: <100 MB for typical 50×30×20×10 grid (vs ~2.5 GB before)
- Extraction scaling: Linear speedup to 64 cores
- Error rate: Zero "moneyness outside PDE bounds" errors with explicit x_bounds
- Validation: 100% of invalid grids caught before PDE work
- API clarity: No duplicate storage, clear ownership chain

## Code Review Findings & Resolutions

### 1. Ownership Model - Move Semantics

**Issue:** `BSpline4D::create(PriceTableWorkspace&& ws)` moving large workspace may cause extra copies.

**Resolution:**
- Ensure `PriceTableWorkspace` has deleted copy constructor
- Factory returns `std::expected<BSpline4D, string>` by value (NRVO applies)
- Constructor is `explicit BSpline4D(PriceTableWorkspace&& ws) : workspace_(std::move(ws))`
- Thread safety: `shared_ptr<BSpline4D>` in `PriceTableSurface` is thread-safe for read-only eval()

### 2. Memory Safety - Parallel Loop

**Issue:** Flattened loop assumes `batch_result.results[batch_idx]` exists for all (vol,r) pairs.

**Resolution:**
- Add bounds check: `if (batch_idx >= batch_result.results.size()) continue;`
- Check result validity: `if (!result_expected.has_value() || !result_expected->converged) continue;`
- Each iteration writes to unique `prices_view[m_idx, mat_idx, vol_idx, r_idx]` - no contention
- Spline objects are stack-allocated per iteration - no shared state

### 3. API Breaking Changes - Serialization

**Issue:** Serialization code expects `PriceTableSurface(shared_ptr<workspace>)` constructor.

**Resolution:**
- Update serialization to deserialize workspace first, then construct BSpline4D:
  ```cpp
  auto workspace = load_workspace_from_arrow(file);
  auto evaluator_result = BSpline4D::create(std::move(workspace));
  auto surface = PriceTableSurface(
      std::make_shared<BSpline4D>(std::move(evaluator_result.value())));
  ```
- Add to implementation checklist: audit Arrow exporter, snapshot replay

### 4. Grid Configuration - Solver API

**Issue:** Does `BatchAmericanOptionSolver` handle both `custom_grid_config` and `GridAccuracyParams`?

**Resolution:**
- Per PR #244 API: `AmericanOptionSolver` constructor takes `custom_grid_config` parameter
- When `custom_grid_config` is provided, solver uses it directly (bypasses auto-estimation)
- When nullopt, solver uses `GridAccuracyParams` for auto-estimation
- These paths are mutually exclusive by design
- Add validation: when `x_bounds` provided, do NOT call `set_grid_accuracy()`

### 5. Error Handling - Rich Context

**Issue:** `PriceTableError` only has `invalid_values` and `expected_value` - may need more context.

**Resolution:**
- Add optional fields to `PriceTableError`:
  ```cpp
  struct PriceTableError {
      PriceTableErrorCode code;
      std::vector<double> invalid_values;
      double expected_value = 0.0;
      std::optional<std::string> details;  // For complex contexts
      size_t axis_index = 0;  // Which dimension failed (0=m, 1=τ, 2=σ, 3=r)
  };
  ```
- Most errors use just code + values, but complex cases can add `details`

### 6. Validation Placement - Solver Grid Modifications

**Issue:** Solver might modify time grid after validation (adaptive steps, padding).

**Resolution:**
- Validation ensures input is well-formed before solver
- If using `custom_grid_config`, grid is fixed (no adaptive modification)
- If using auto-estimation, validation only checks snapshot times are reasonable
- Add post-solve assertion in extraction: verify `result.grid()->num_snapshots() == maturity_.size()`
- If mismatch, error indicates solver dropped snapshots (programming error, not user error)

### 7. Diagnostics - Grid Access

**Issue:** `compute_residuals()` needs moneyness/maturity grids from builder.

**Resolution:**
- Builder stores grids as members: `moneyness_`, `maturity_`, `volatility_`, `rate_`
- `compute_residuals()` is a builder method, so it has access to these grids
- Implementation:
  ```cpp
  std::expected<ResidualStats, PriceTableError>
  PriceTable4DBuilder::compute_residuals(const PriceTableSurface& surface) const
  {
      // Use builder's moneyness_, maturity_ grids
      // Evaluate surface.eval(m, tau, sigma, r) at each grid point
      // Compare with re-evaluated spline
  }
  ```

### 8. Missing Updates - Downstream Consumers

**Issue:** Removing `prices_4d` breaks tests, Arrow export, CLI tools.

**Resolution:**
- Add to implementation plan (before step 4):
  - **3.5. Audit all consumers of prices_4d**
    - Find with: `grep -r "prices_4d" tests/ examples/ tools/`
    - Update tests to use `compute_residuals()` or remove raw price checks
    - Update Arrow exporter to use `precompute_with_save()` streaming path
    - Update CLI tools to use new diagnostic APIs
- Add to migration guide: deprecation timeline for `prices_4d` access

## References

- Recent API refactor: PR #244 (custom_grid_config support)
- Memory profiling: price_table_4d_builder.cpp:145 (10 GB tensor issue)
- Extraction bottleneck: price_table_extraction.cpp:70-122
- Grid config bug: price_table_4d_builder.cpp:175-182
- Snapshot validation: price_table_extraction.cpp:42 (existing assert)
