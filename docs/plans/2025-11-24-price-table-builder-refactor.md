# Price Table Builder Refactor

**Date:** 2025-11-24
**Status:** Design Complete
**Goal:** Comprehensive refactor addressing resource ownership, memory pressure, grid configuration bugs, parallelization, and validation.

## Naming Changes

Drop redundant "4D" suffix from all names since price tables are inherently 4-dimensional (moneyness × maturity × volatility × rate):

- `PriceTable4DBuilder` → `PriceTableBuilder`
- `PriceTable4DResult` → `PriceTableResult`
- `BSpline4D` → `BSplineEvaluator` (more descriptive)
- File: `price_table_4d_builder.hpp` → `price_table_builder.hpp`
- File: `price_table_4d_builder.cpp` → `price_table_builder.cpp`

## Problems Addressed

1. **Resource ownership** - PriceTableSurface stores data twice (workspace + evaluator)
2. **Memory explosion** - Full tensor kept in result (>10 GB for large grids)
3. **Grid configuration ignored** - Custom x_bounds/n_space/n_time overridden by GridAccuracyParams
4. **Serial extraction bottleneck** - Extraction not parallelized despite parallel PDE solving
5. **No snapshot validation** - Invalid maturities corrupt tensor silently

## Architecture Overview

### 1. Resource Ownership Chain

**Design:** Single ownership path eliminates duplicate storage.

```
PriceTableWorkspace (grids + coefficients)
         ↑ owned by
     BSplineEvaluator (evaluator logic)
         ↑ wrapped by shared_ptr
  PriceTableSurface (user-facing API)
```

**Changes to PriceTableSurface:**
```cpp
class PriceTableSurface {
public:
    explicit PriceTableSurface(std::shared_ptr<BSplineEvaluator> evaluator)
        : evaluator_(std::move(evaluator)) {}

    double eval(double m, double tau, double sigma, double rate) const {
        return evaluator_->eval(m, tau, sigma, rate);
    }

    // Expose workspace for serialization/diagnostics
    const PriceTableWorkspace& workspace() const {
        return evaluator_->workspace();
    }

private:
    std::shared_ptr<BSplineEvaluator> evaluator_;
    // REMOVED: std::shared_ptr<PriceTableWorkspace> workspace_
    // REMOVED: std::unique_ptr<BSplineEvaluator> evaluator_
};
```

**Changes to BSplineEvaluator:**
```cpp
class BSplineEvaluator {
public:
    // Factory returns shared_ptr directly to avoid double-move
    static std::expected<std::shared_ptr<BSplineEvaluator>, std::string>
    create(PriceTableWorkspace&& ws);

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

**Changes to PriceTableResult:**
```cpp
struct PriceTableResult {
    PriceTableSurface surface;              // Lightweight evaluator wrapper
    size_t n_pde_solves;                    // Useful stat
    double precompute_time_seconds;         // Useful stat
    BSplineFittingStats fitting_stats;      // Fitting diagnostics

    // REMOVED: std::vector<double> prices_4d (can be >10 GB)
    // REMOVED: std::shared_ptr<BSplineEvaluator> evaluator (redundant with surface)
};
```

**New diagnostic APIs:**
```cpp
class PriceTableBuilder {
public:
    // Existing precompute returns lightweight result
    std::expected<PriceTableResult, PriceTableError> precompute(
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
std::expected<PriceTableResult, PriceTableError>
PriceTableBuilder::precompute(const PriceTableConfig& config)
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
// Track failures explicitly (from resolution #2)
const size_t total_work = Nv * Nr * Nt;
std::atomic<size_t> failed_slices{0};

MANGO_PRAGMA_PARALLEL_FOR
for (size_t work_idx = 0; work_idx < total_work; ++work_idx) {
    // Decode flat index to (vol_idx, r_idx, mat_idx)
    const size_t vol_idx = work_idx / (Nr * Nt);
    const size_t remainder = work_idx % (Nr * Nt);
    const size_t r_idx = remainder / Nt;
    const size_t mat_idx = remainder % Nt;

    // Get batch result
    const size_t batch_idx = vol_idx * Nr + r_idx;

    if (batch_idx >= batch_result.results.size() ||
        !batch_result.results[batch_idx].has_value() ||
        !batch_result.results[batch_idx]->converged) {
        failed_slices.fetch_add(1, std::memory_order_relaxed);
        continue;  // Leave zeros in prices_view
    }

    const auto& result = batch_result.results[batch_idx].value();
    auto result_grid = result.grid();
    auto x_grid = result_grid->x();

    // Build spline for this (σ, r, τ) combination
    size_t step_idx = step_indices[mat_idx];
    std::span<const double> spatial_solution = result.at_time(step_idx);

    if (spatial_solution.empty()) {
        failed_slices.fetch_add(1, std::memory_order_relaxed);
        continue;
    }

    CubicSpline<double> spline;
    auto build_error = spline.build(x_grid, spatial_solution);

    // Interpolate to moneyness grid
    for (size_t m_idx = 0; m_idx < Nm; ++m_idx) {
        const double x = log_moneyness[m_idx];
        double V_norm = build_error ? boundary_value : spline.eval(x);
        prices_view[m_idx, mat_idx, vol_idx, r_idx] = K_ref * V_norm;
    }
}

// Check for failures after extraction
if (failed_slices.load() > 0) {
    return std::unexpected(PriceTableError{
        .code = PriceTableErrorCode::INCOMPLETE_RESULTS,
        .invalid_values = {static_cast<double>(failed_slices.load())},
        .details = std::to_string(failed_slices.load()) + " slices failed"
    });
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
    INVALID_GRID_EMPTY,
    INVALID_GRID_MISSING_TERMINAL,
    INVALID_GRID_EXCEEDS_TERMINAL,
    INVALID_BOUNDS,
    INVALID_GRID_SPEC,
    WORKSPACE_CREATION_FAILED,
    BSPLINE_FIT_FAILED,
    PDE_SOLVE_FAILED,
    INCOMPLETE_RESULTS,        // Some PDE solves failed
    SNAPSHOT_MISMATCH,         // Fixed grid snapshot count wrong
    SNAPSHOT_DROPPED           // Adaptive grid dropped snapshots
};

struct PriceTableError {
    PriceTableErrorCode code;
    std::vector<double> invalid_values;
    std::optional<double> expected_value;   // nullopt when not applicable
    std::optional<std::string> details;     // For complex contexts
    std::optional<size_t> axis_index;       // 0=m, 1=τ, 2=σ, 3=r, nullopt=not axis-specific
};
```

**Validation in precompute():**
```cpp
std::expected<PriceTableResult, PriceTableError>
PriceTableBuilder::precompute(const PriceTableConfig& config)
{
    const double epsilon = 1e-10;

    // Validate maturity grid before solver setup

    // Check 1: Must be sorted
    if (!std::is_sorted(maturity_.begin(), maturity_.end())) {
        return std::unexpected(PriceTableError{
            .code = PriceTableErrorCode::INVALID_GRID_UNSORTED,
            .invalid_values = maturity_,
            .axis_index = 1  // τ axis
        });
    }

    // Check 2: No negative values
    if (maturity_.front() < 0.0) {
        return std::unexpected(PriceTableError{
            .code = PriceTableErrorCode::INVALID_GRID_NEGATIVE,
            .invalid_values = {maturity_.front()},
            .axis_index = 1  // τ axis
        });
    }

    // Check 3: Grid must be non-empty
    if (maturity_.empty()) {
        return std::unexpected(PriceTableError{
            .code = PriceTableErrorCode::INVALID_GRID_EMPTY,
            .axis_index = 1  // τ axis
        });
    }

    // Note: Validation that maturity grid fits within PDE time domain happens
    // post-solve (see resolution #6) since T_max depends on solver configuration

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
2. **Refactor BSplineEvaluator ownership** - Own workspace, update constructor
3. **Refactor PriceTableSurface** - Wrap BSplineEvaluator only, expose workspace
4. **Remove prices_4d from result** - Update PriceTableResult struct
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
- `price_table_builder_test.cc` - Update existing tests for new API
- Test error cases: unsorted grid, missing T_max, exceeds T_max
- Test x_bounds respected vs auto-estimation fallback
- Verify no prices_4d in result

### Performance Tests
- Measure extraction phase scaling with core count
- Verify memory usage reduction (no raw tensor in result)
- Benchmark with large grids (200×100×50×20)

## Migration Guide

### Breaking Changes

**1. PriceTableResult no longer contains prices_4d:**
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

// NEW (with proper error handling)
auto evaluator_result = BSplineEvaluator::create(std::move(workspace));
if (!evaluator_result) {
    // Handle error: evaluator_result.error()
    return;
}
PriceTableSurface surface(evaluator_result.value());  // shared_ptr, cheap
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

**Issue:** `BSplineEvaluator::create(PriceTableWorkspace&& ws)` moving large workspace may cause extra copies. Returning `expected<BSplineEvaluator>` by value forces workspace to live inside expected temporarily.

**Resolution:**
- Change factory signature to return pointer directly:
  ```cpp
  static std::expected<std::shared_ptr<BSplineEvaluator>, std::string>
  create(PriceTableWorkspace&& ws);
  ```
- Implementation moves workspace directly into heap-allocated BSplineEvaluator:
  ```cpp
  std::expected<std::shared_ptr<BSplineEvaluator>, std::string>
  BSplineEvaluator::create(PriceTableWorkspace&& ws) {
      // Validate workspace...
      return std::make_shared<BSplineEvaluator>(std::move(ws));
  }
  ```
- Ensure `PriceTableWorkspace` has deleted copy constructor, move-only
- Thread safety: `shared_ptr<BSplineEvaluator>` is thread-safe for concurrent eval()
- **Testing:** Add instrumentation to count PriceTableWorkspace moves/copies, verify exactly 1 move occurs

### 2. Memory Safety - Parallel Loop

**Issue:** Flattened loop assumes `batch_result.results[batch_idx]` exists for all (vol,r) pairs. Silent `continue` on failures leaves uninitialized data.

**Resolution:**
- Track failed slices explicitly instead of silently skipping:
  ```cpp
  std::atomic<size_t> failed_slices{0};

  #pragma omp parallel for
  for (size_t work_idx = 0; work_idx < total_work; ++work_idx) {
      // Decode indices...

      if (batch_idx >= batch_result.results.size() ||
          !result_expected.has_value() ||
          !result_expected->converged) {
          failed_slices.fetch_add(1, std::memory_order_relaxed);
          continue;  // Leave zeros in prices_view
      }
      // ... spline interpolation
  }

  if (failed_slices.load() > 0) {
      return std::unexpected(PriceTableError{
          .code = PriceTableErrorCode::INCOMPLETE_RESULTS,
          .invalid_values = {static_cast<double>(failed_slices.load())},
          .details = std::to_string(failed_slices.load()) + " slices failed"
      });
  }
  ```
- Each iteration writes to unique `prices_view[m_idx, mat_idx, vol_idx, r_idx]` - no contention
- Spline objects are stack-allocated per iteration - no shared state
- Bounds check is per-slice (outer loop), not per-lattice-node (inner loop) - negligible overhead
- **Testing:** Inject batch failures, verify error is surfaced (not silent zeros)

### 3. API Breaking Changes - Serialization

**Issue:** Serialization code expects `PriceTableSurface(shared_ptr<workspace>)` constructor. Unconditional `.value()` on expected will terminate on failure.

**Resolution:**
- Update serialization with proper error handling:
  ```cpp
  std::expected<PriceTableSurface, std::string>
  deserialize_surface(const std::filesystem::path& file) {
      auto workspace_result = load_workspace_from_arrow(file);
      if (!workspace_result) {
          return std::unexpected("Failed to load workspace: " +
                                 workspace_result.error());
      }

      auto evaluator_result = BSplineEvaluator::create(std::move(workspace_result.value()));
      if (!evaluator_result) {
          return std::unexpected("Failed to create evaluator: " +
                                 evaluator_result.error());
      }

      return PriceTableSurface(evaluator_result.value());  // shared_ptr, cheap copy
  }
  ```
- Workspace move is cheap (moved into make_shared inside create())
- Add to implementation checklist: audit Arrow exporter, snapshot replay, scripting bindings
- **Testing:** Verify no extra allocations during deserialize (memory profiler or custom allocator)

### 4. Grid Configuration - Solver API

**Issue:** Are `custom_grid_config` and `GridAccuracyParams` truly mutually exclusive? What if both are set? What about adaptive solvers?

**Resolution:**
- Add validation to ensure mutual exclusivity:
  ```cpp
  if (config.x_bounds.has_value()) {
      // Build custom_grid_config...
      custom_grid_config = std::make_pair(grid_spec, time_domain);

      // Explicitly DO NOT call set_grid_accuracy() - paths are exclusive
  } else {
      // Use GridAccuracyParams
      GridAccuracyParams accuracy;
      // ...
      batch_solver.set_grid_accuracy(accuracy);

      // custom_grid_config remains nullopt
  }

  // Assert mutual exclusivity
  assert(!(custom_grid_config.has_value() && /* accuracy was set */));
  ```
- Document contract in PriceTableConfig:
  - If `x_bounds` is set: grid is FIXED, no adaptive modification allowed
  - If `x_bounds` is nullopt: auto-estimation with adaptive time steps allowed
- **Testing:** Test both paths independently, verify error if both supplied

### 5. Error Handling - Rich Context

**Issue:** `PriceTableError` with default values misleading (`axis_index=0` implies moneyness even when irrelevant, `expected_value=0.0` ambiguous).

**Resolution:**
- Use optional for ambiguous fields:
  ```cpp
  struct PriceTableError {
      PriceTableErrorCode code;
      std::vector<double> invalid_values;
      std::optional<double> expected_value;  // nullopt when not applicable
      std::optional<std::string> details;    // For complex contexts
      std::optional<size_t> axis_index;      // 0=m, 1=τ, 2=σ, 3=r, nullopt=not axis-specific
  };
  ```
- Example usage:
  ```cpp
  // Axis-specific error
  PriceTableError{
      .code = INVALID_GRID_UNSORTED,
      .invalid_values = maturity_,
      .axis_index = 1  // τ axis
  };

  // Scalar error
  PriceTableError{
      .code = INVALID_GRID_MISSING_TERMINAL,
      .invalid_values = {maturity_.back()},
      .expected_value = T_max
  };
  ```
- **Testing:** Verify serialization/deserialization of error structs, test ABI stability

### 6. Validation Placement - Solver Grid Modifications

**Issue:** Solver might modify time grid after validation (adaptive steps, padding).

**Resolution:**
- **With custom_grid_config (x_bounds set):**
  - Grid is FIXED, no adaptive modification
  - Post-solve: assert exact equality `num_snapshots() == maturity_.size()`
  - Mismatch indicates programming error

- **With auto-estimation (x_bounds nullopt):**
  - Adaptive padding allowed
  - Post-solve: verify requested snapshots are subset of actual snapshots
  - Check: `for each requested_time: exists actual_time within epsilon`
  - Mismatch indicates solver dropped snapshots (error)

- Update extraction to handle both modes:
  ```cpp
  if (custom_grid_config.has_value()) {
      if (result.grid()->num_snapshots() != maturity_.size()) {
          return std::unexpected(PriceTableError{
              .code = PriceTableErrorCode::SNAPSHOT_MISMATCH
          });
      }
  } else {
      for (double t : maturity_) {
          if (!/* t in result snapshots */) {
              return std::unexpected(PriceTableError{
                  .code = PriceTableErrorCode::SNAPSHOT_DROPPED,
                  .invalid_values = {t}
              });
          }
      }
  }
  ```
- **Testing:** Test both fixed and adaptive modes separately

### 7. Diagnostics - Grid Access

**Issue:** `compute_residuals()` needs grids from builder. Downstream consumers (Arrow export, CLI) need grid access after removing `prices_4d`.

**Resolution:**
- Builder stores grids as members: `moneyness_`, `maturity_`, `volatility_`, `rate_`
- `compute_residuals()` is builder method with grid access
- For massive grids (>10 GB): use streaming to avoid memory explosion:
  ```cpp
  std::expected<ResidualStats, PriceTableError>
  PriceTableBuilder::compute_residuals(
      const PriceTableSurface& surface,
      size_t subsample_factor = 1) const  // subsample for huge grids
  {
      ResidualStats stats;
      // Stream through grid points, don't materialize full tensor
      for (size_t m_idx = 0; m_idx < Nm; m_idx += subsample_factor) {
          for (size_t t_idx = 0; t_idx < Nt; t_idx += subsample_factor) {
              for (size_t v_idx = 0; v_idx < Nv; v_idx += subsample_factor) {
                  for (size_t r_idx = 0; r_idx < Nr; r_idx += subsample_factor) {
                      double actual = surface.eval(...);
                      // Compare with ground truth, update stats
                  }
              }
          }
      }
      return stats;
  }
  ```
- Expose grids via builder accessors for downstream consumers:
  ```cpp
  class PriceTableBuilder {
  public:
      std::span<const double> moneyness() const { return moneyness_; }
      std::span<const double> maturity() const { return maturity_; }
      std::span<const double> volatility() const { return volatility_; }
      std::span<const double> rate() const { return rate_; }
  };
  ```
- Arrow export uses `builder.moneyness()` + `surface.eval()` instead of `result.prices_4d`
- **Testing:** Test residuals with 1GB+ grids using subsampling

### 8. Missing Updates - Downstream Consumers

**Issue:** Removing `prices_4d` breaks tests, Arrow export, CLI tools. Need concrete migration strategy.

**Resolution:**
- Add to implementation plan (step 3.5 before removing prices_4d):
  - **Audit all consumers:**
    ```bash
    grep -r "prices_4d" tests/ examples/ tools/ benchmarks/
    grep -r "PriceTableResult" tests/ examples/ tools/
    ```
  - **Update each consumer category:**
    - **Unit tests comparing raw prices:** Use `compute_residuals()` instead
    - **Integration tests diffing tables:** Use `precompute_with_save()` to serialize, diff files
    - **Arrow export:** Replace `result.prices_4d` with loop over `builder.moneyness()` + `surface.eval()`
    - **CLI diagnostic tools:** Add `--compute-residuals` flag, remove `--dump-raw-prices`
    - **Benchmarks measuring memory:** Update expected values (should drop by ~2 GB)
  - **Migration path:**
    - Phase 1 (this PR): Add new APIs (`compute_residuals`, `builder.moneyness()`, `precompute_with_save`)
    - Phase 2 (this PR): Update all consumers to use new APIs
    - Phase 3 (this PR): Remove `prices_4d` from `PriceTableResult`
- **Testing strategy:**
  - Before removing `prices_4d`: add tests for all new APIs
  - After removing: verify no compilation errors, all tests pass
  - Add regression test: ensure `PriceTableResult` size < 1 KB (was ~2 GB)
- **Performance validation:**
  - Benchmark extraction phase before/after parallelization
  - Document overhead of bounds checks (should be <1%)
  - Profile `compute_residuals()` with 10 GB grid

## References

- Recent API refactor: PR #244 (custom_grid_config support)
- Memory profiling: price_table_builder.cpp:145 (10 GB tensor issue)
- Extraction bottleneck: price_table_extraction.cpp:70-122
- Grid config bug: price_table_builder.cpp:175-182
- Snapshot validation: price_table_extraction.cpp:42 (existing assert)
