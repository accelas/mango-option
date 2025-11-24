# Price Table Builder Refactor - Two-Phase Design

**Date:** 2025-11-24 (Revised: 2025-11-24)
**Status:** Design Complete - Two-Phase Approach
**Goal:** Replace specialized builder with generic architecture, then add targeted improvements

## Overview

This refactor is split into **two phases**:

### Phase 1: Migration with Feature Parity (Mandatory)
**Goal:** Replace specialized `PriceTable4DBuilder` with feature-complete generic `PriceTableBuilder<4>`, then delete old builder

**Why:** Generic architecture is superior:
- Clean separation: `PriceTableAxes<N>` → `PriceTableConfig` → `PriceTableSurface<N>`
- Better composability: Works with any N dimensions (N=4 implemented, others possible)
- No duplicate storage: Single ownership chain through `BSplineND<N>`
- Already uses `BatchAmericanOptionSolver` with normalized solving

**Scope (expanded based on Codex review):**
- **Add `PriceTableResult<N>` struct** with diagnostics (n_pde_solves, timing, BSplineFittingStats)
- **Fix grid configuration bug** (respect user's grid_estimator exactly)
- **Port validation logic** from specialized builder (≥4 points, positive values, domain coverage)
- **Add helper factories** for all entry points (from_vectors, from_strikes, from_chain)
- **Update all consumers** (tests, benchmarks, examples, docs)
- **Delete specialized builder** completely (not just deprecate)

**Critical fixes required for parity:**
1. **Replace AlignedArena with PMR** (lines 183-186) - Use `std::pmr::vector` like rest of codebase
2. Grid configuration must honor user's `config.grid_estimator` (lines 131-142 bug fix)
3. Return `PriceTableResult<N>` with full diagnostics (not just surface pointer)
4. Comprehensive validation (prevent 2-point grids, negative values, out-of-domain)
5. Helper factories that return both builder AND axes (axes ownership solved)

**Timeline:** Must be completed atomically (no intermediate broken state)

### Phase 2: Performance & Quality Improvements (Optional)
**Goal:** Add enhancements that don't affect basic functionality

**Remaining improvements:**
1. Parallelize extraction with OpenMP (serial → parallel bottleneck fix)
2. Add atomic failure tracking (count failed slices explicitly)
3. Add structured `PriceTableError` types (replace string errors)
4. Add `build_with_save()` for streaming tensor to disk

**Scope:** Each improvement can be implemented independently
**Timeline:** Incremental, based on priority

---

# Phase 1: Migration to Generic Builder

## Goals

1. Update all consumers (tests, examples, benchmarks) to use `PriceTableBuilder<4>`
2. Deprecate specialized `PriceTable4DBuilder` (mark with warnings, keep functional)
3. Document migration path
4. **Zero functional changes** - APIs behave identically

## API Comparison

### Specialized Builder (Current)
```cpp
// Create builder
auto builder_result = PriceTable4DBuilder::create(
    moneyness_vec,  // std::vector<double>
    maturity_vec,   // std::vector<double>
    vol_vec,        // std::vector<double>
    rate_vec,       // std::vector<double>
    K_ref           // double
);
if (!builder_result) {
    // Handle error (string)
    return;
}
auto& builder = builder_result.value();

// Build surface
auto result = builder.precompute(OptionType::PUT, 101, 1000);
if (!result) {
    // Handle error (string)
    return;
}

// Use surface
double price = result->surface.eval(1.0, 0.25, 0.20, 0.05);
```

### Generic Builder (Target)
```cpp
// Create axes
PriceTableAxes<4> axes;
axes.grids = {moneyness_vec, maturity_vec, vol_vec, rate_vec};
axes.names = {"moneyness", "maturity", "volatility", "rate"};

// Create config
PriceTableConfig config;
config.option_type = OptionType::PUT;
config.K_ref = K_ref;
config.grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value();
config.n_time = 1000;

// Build surface
PriceTableBuilder<4> builder(config);
auto result = builder.build(axes);
if (!result) {
    // Handle error (string)
    return;
}

// Use surface
auto surface = result.value().surface;
double price = surface->value({1.0, 0.25, 0.20, 0.05});
```

## Implementation Steps (Phase 1)

### Step 1: Replace AlignedArena with PMR
**Location:** `src/option/price_table_builder.cpp:178-194`

**Problem:** Generic builder uses custom `memory::AlignedArena`, but rest of codebase uses standard PMR allocators

**Solution:** Replace with `std::pmr::vector` pattern used elsewhere:

```cpp
// BEFORE (lines 178-194):
const size_t total_points = Nm * Nt * Nσ * Nr;
const size_t tensor_bytes = total_points * sizeof(double);
const size_t arena_bytes = tensor_bytes + 64;  // 64-byte alignment padding

auto arena = memory::AlignedArena::create(arena_bytes);
if (!arena.has_value()) {
    return std::unexpected("Failed to create arena: " + arena.error());
}

std::array<size_t, N> shape = {Nm, Nt, Nσ, Nr};
auto tensor_result = PriceTensor<N>::create(shape, arena.value());
if (!tensor_result.has_value()) {
    return std::unexpected("Failed to create tensor: " + tensor_result.error());
}

auto tensor = tensor_result.value();

// AFTER (use PMR with aligned allocation):
const size_t total_points = Nm * Nt * Nσ * Nr;
const size_t tensor_bytes = total_points * sizeof(double);

// CRITICAL: Preserve 64-byte alignment for AVX-512
// Solution: Use aligned_alloc to create backing buffer, wrap in PMR
constexpr size_t alignment = 64;
const size_t aligned_bytes = (tensor_bytes + alignment - 1) & ~(alignment - 1);

void* aligned_buffer = std::aligned_alloc(alignment, aligned_bytes);
if (!aligned_buffer) {
    return std::unexpected("Failed to allocate aligned buffer");
}

// Wrap aligned buffer in PMR resource
std::pmr::monotonic_buffer_resource pool(aligned_buffer, aligned_bytes);
std::pmr::vector<double> tensor_data(&pool);
tensor_data.resize(total_points, 0.0);

// Wrap in mdspan for N-D access
using std::experimental::mdspan;
using std::experimental::dextents;
mdspan<double, dextents<size_t, N>> tensor(tensor_data.data(), Nm, Nt, Nσ, Nr);

// IMPORTANT: Free aligned buffer after fitting
auto cleanup = [aligned_buffer]() { std::free(aligned_buffer); };
std::unique_ptr<void, decltype(cleanup)> buffer_guard(aligned_buffer, cleanup);
```

**Benefits:**
- ✅ Preserves 64-byte alignment for AVX-512 (critical for performance)
- ✅ Uses PMR for consistency with codebase patterns
- ✅ Standard C++17 facilities (aligned_alloc + PMR)
- ✅ RAII cleanup via unique_ptr

**Impact:** Best of both worlds - aligned allocation + PMR interface

**Alternative:** Keep `AlignedArena` if it already provides PMR-compatible interface

### Step 2: Add `PriceTableResult<N>` struct
**Location:** `src/option/price_table_builder.hpp`

```cpp
/// B-spline fitting diagnostics (extract from BSplineNDSeparable)
struct BSplineFittingStats {
    double max_residual_axis0 = 0.0;
    double max_residual_axis1 = 0.0;
    double max_residual_axis2 = 0.0;
    double max_residual_axis3 = 0.0;
    double max_residual_overall = 0.0;

    double condition_axis0 = 0.0;
    double condition_axis1 = 0.0;
    double condition_axis2 = 0.0;
    double condition_axis3 = 0.0;
    double condition_max = 0.0;

    size_t failed_slices_axis0 = 0;
    size_t failed_slices_axis1 = 0;
    size_t failed_slices_axis2 = 0;
    size_t failed_slices_axis3 = 0;
    size_t failed_slices_total = 0;
};

/// Result from price table build with diagnostics
template <size_t N>
struct PriceTableResult {
    std::shared_ptr<const PriceTableSurface<N>> surface;  // Immutable surface
    size_t n_pde_solves;                    // Number of PDE solves performed
    double precompute_time_seconds;         // Wall-clock build time
    BSplineFittingStats fitting_stats;      // B-spline fitting diagnostics
};
```

**Update:** Change `build()` signature to return `PriceTableResult<N>` instead of `shared_ptr<Surface<N>>`

**Instrumentation (HOW to populate the fields):**

1. **n_pde_solves**: Count from `BatchAmericanOptionResult`
   ```cpp
   // In build() after solve_batch()
   size_t n_pde_solves = Nσ * Nr;  // One solve per (σ, r) combination
   // OR: batch_result.results.size() - batch_result.failed_count
   ```

2. **precompute_time_seconds**: Wall-clock timer
   ```cpp
   // In build() - start timer before solve_batch()
   auto start_time = std::chrono::high_resolution_clock::now();

   // ... solve_batch(), extract_tensor(), fit_coeffs() ...

   auto end_time = std::chrono::high_resolution_clock::now();
   double elapsed = std::chrono::duration<double>(end_time - start_time).count();
   ```

3. **BSplineFittingStats**: Extract from `BSplineNDSeparable::fit()` return value
   ```cpp
   // Modify BSplineNDSeparable<T, N>::fit() to return stats
   struct BSplineFitResult {
       std::vector<T> coefficients;
       BSplineFittingStats stats;  // NEW: residuals, condition numbers, failures
   };

   // In build() after fit_coeffs()
   auto fit_result = fitter.fit(tensor_data);
   BSplineFittingStats stats = fit_result.value().stats;
   ```

**Implementation tasks:**
- Add `BSplineFitResult` struct to `bspline_nd_separable.hpp`
- Update `BSplineNDSeparable::fit()` to compute and return stats
- Add timer instrumentation at start/end of `build()`
- Count PDE solves from batch result size

### Step 3: Fix Grid Configuration Bug
**Location:** `src/option/price_table_builder.cpp:131-142`

```cpp
// BEFORE (buggy):
GridAccuracyParams accuracy;
accuracy.min_spatial_points = std::min(config_.grid_estimator.n_points(), size_t(100));
accuracy.max_spatial_points = std::max(config_.grid_estimator.n_points(), size_t(1200));

// AFTER (fixed):
GridAccuracyParams accuracy;
accuracy.min_spatial_points = config_.grid_estimator.n_points();  // Use exact value
accuracy.max_spatial_points = config_.grid_estimator.n_points();  // Use exact value
accuracy.max_time_steps = config_.n_time;

// Extract alpha parameter for sinh-spaced grids
if (config_.grid_estimator.type() == GridSpec<double>::Type::SinhSpaced) {
    accuracy.alpha = config_.grid_estimator.concentration();
}
```

### Step 4: Port Validation Logic
**Location:** `src/option/price_table_builder.cpp` (in `build()` method)

```cpp
// Add comprehensive validation from PriceTable4DBuilder::validate_grids()
auto validation = axes.validate();  // Existing: checks empty, monotonic
if (!validation.has_value()) {
    return std::unexpected("Validation failed: " + /* convert error */);
}

// NEW: Port from specialized builder
// 1. Check minimum 4 points per axis (B-spline requirement)
for (size_t i = 0; i < N; ++i) {
    if (axes.grids[i].size() < 4) {
        return std::unexpected("Axis " + std::to_string(i) +
                               " has only " + std::to_string(axes.grids[i].size()) +
                               " points (need ≥4 for cubic B-splines)");
    }
}

// 2. Check positive moneyness (needed for log)
if (axes.grids[0].front() <= 0.0) {
    return std::unexpected("Moneyness must be positive (needed for log)");
}

// 3. Check non-negative maturity
if (axes.grids[1].front() < 0.0) {
    return std::unexpected("Maturity cannot be negative");
}

// 4. Check positive volatility
if (axes.grids[2].front() <= 0.0) {
    return std::unexpected("Volatility must be positive");
}

// 5. Check K_ref > 0
if (config_.K_ref <= 0.0) {
    return std::unexpected("Reference strike K_ref must be positive");
}

// 6. Check PDE domain coverage
// CRITICAL: Validate that moneyness grid fits within PDE spatial domain
// Port logic from PriceTable4DBuilder::precompute (lines 120-142)
const double x_min_requested = std::log(axes.grids[0].front());  // log(moneyness_min)
const double x_max_requested = std::log(axes.grids[0].back());   // log(moneyness_max)

// Get PDE bounds from grid_spec in config
const double x_min = config_.grid_estimator.x_min();  // PDE domain lower bound
const double x_max = config_.grid_estimator.x_max();  // PDE domain upper bound

// Validate requested range fits within PDE domain
if (x_min_requested < x_min || x_max_requested > x_max) {
    return std::unexpected(
        "Requested moneyness range [" + std::to_string(axes.grids[0].front()) + ", " +
        std::to_string(axes.grids[0].back()) + "] in spot ratios "
        "maps to log-moneyness [" + std::to_string(x_min_requested) + ", " +
        std::to_string(x_max_requested) + "], "
        "which exceeds PDE grid bounds [" + std::to_string(x_min) + ", " +
        std::to_string(x_max) + "]. "
        "Narrow the moneyness grid or expand the PDE domain. "
        "Example: for moneyness [0.7, 1.5], use grid_spec with x_min <= " +
        std::to_string(x_min_requested) + " and x_max >= " +
        std::to_string(x_max_requested) + "."
    );
}

// This check prevents interpolation artifacts from cubic splines extrapolating
// outside their knot domain, which would produce arbitrary garbage values
```

### Step 5: Add Helper Factories
**Location:** `src/option/price_table_builder.hpp`

```cpp
template <size_t N>
class PriceTableBuilder {
public:
    // Existing constructor
    explicit PriceTableBuilder(PriceTableConfig config);

    // NEW: Factory that returns BOTH builder and axes (solves ownership problem)
    // PDE resolution control: caller specifies grid_spec and n_time
    static std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
    from_vectors(
        std::vector<double> moneyness,
        std::vector<double> maturity,
        std::vector<double> volatility,
        std::vector<double> rate,
        double K_ref,
        GridSpec<double> grid_spec,     // PDE spatial grid (e.g., uniform(-3, 3, 101))
        size_t n_time,                   // PDE time steps (e.g., 1000)
        OptionType type = OptionType::PUT,
        double dividend_yield = 0.0);

    // NEW: Factory from strikes (auto-computes moneyness)
    static std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
    from_strikes(
        double spot,
        std::vector<double> strikes,
        std::vector<double> maturities,
        std::vector<double> volatilities,
        std::vector<double> rates,
        GridSpec<double> grid_spec,     // PDE spatial grid
        size_t n_time,                   // PDE time steps
        OptionType type = OptionType::PUT,
        double dividend_yield = 0.0);

    // NEW: Factory from option chain
    static std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
    from_chain(
        const OptionChain& chain,
        GridSpec<double> grid_spec,     // PDE spatial grid
        size_t n_time,                   // PDE time steps
        OptionType type = OptionType::PUT);
};
```

**Implementation approach:** Each factory:
1. Constructs `PriceTableAxes<4>` from input vectors
2. Constructs `PriceTableConfig` with defaults
3. Runs validation
4. Returns both via `std::pair` (caller owns axes, passes to `build()`)

### Step 6: Update All Consumers
**Inventory of files to update:**
- `tests/price_table_4d_integration_test.cc` - Main integration tests
- `tests/price_table_end_to_end_performance_test.cc` - Performance tests
- `benchmarks/market_iv_e2e_benchmark.cc` - Benchmarks
- `examples/` - Any example files using builder
- `docs/API_GUIDE.md` - Documentation

**Migration pattern for each file:**
```cpp
// OLD
auto builder_result = PriceTable4DBuilder::create(vec1, vec2, vec3, vec4, K_ref);
auto result = builder_result->precompute(OptionType::PUT, 101, 1000);
auto& surface = result->surface;
auto& stats = result->fitting_stats;

// NEW (with explicit PDE resolution control)
auto grid_spec = GridSpec<double>::uniform(-3.0, 3.0, 101).value();  // Match old n_space=101
auto [builder, axes] = PriceTableBuilder<4>::from_vectors(
    vec1, vec2, vec3, vec4,
    K_ref,
    grid_spec,           // PDE spatial grid
    1000,                // PDE time steps (match old n_time=1000)
    OptionType::PUT
).value();

auto result = builder.build(axes).value();
auto surface = result.surface;
const auto& stats = result.fitting_stats;
```

### Step 7: Delete Specialized Builder
**Files to delete:**
- `src/option/price_table_4d_builder.hpp`
- `src/option/price_table_4d_builder.cpp`
- Update `BUILD` file to remove targets

**Verify:** Run `bazel test //...` to ensure no broken dependencies

### Step 8: Update Documentation
- `docs/API_GUIDE.md` - Replace all examples with generic builder
- Inline doc comments - Update to reference `PriceTableBuilder<4>`
- `CLAUDE.md` - Update quick reference examples

---

# Phase 2: Performance & Quality Improvements

**Prerequisite:** Phase 1 must be complete (generic builder is feature-complete, specialized builder deleted)

## Remaining Improvements

After Phase 1, the generic builder will have feature parity but still has performance and quality issues:

### Improvement 1: Parallelize Extraction
**Problem:** `extract_tensor()` loops over (σ,r) batches serially (src/option/price_table_builder.cpp:203-250)

**Impact:** Leaves cores idle during extraction. For 20×10 = 200 batches, this is embarrassingly parallel work.

**Solution:** Add OpenMP to outer loop:

```cpp
// BEFORE (serial):
for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
    for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
        // ... extract prices from batch result
    }
}

// AFTER (parallel):
MANGO_PRAGMA_PARALLEL_FOR
for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
    for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
        // ... extract prices from batch result
    }
}
```

**Expected speedup:** ~10-16× on 32-core machines

### Improvement 2: Add Atomic Failure Tracking
**Problem:** `extract_tensor()` fills NaN for failures but doesn't count them (lines 208-216, 232-238)

**Impact:** No way to detect partial failures or get useful error messages

**Solution:** Add atomic counter:

```cpp
std::atomic<size_t> failed_slices{0};

MANGO_PRAGMA_PARALLEL_FOR
for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
    for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
        if (!result_expected.has_value()) {
            failed_slices.fetch_add(Nt, std::memory_order_relaxed);  // Nt failures
            // Fill with NaN
            continue;
        }

        // ... per-maturity loop
        if (build_error.has_value()) {
            failed_slices.fetch_add(1, std::memory_order_relaxed);
            // Fill with NaN
            continue;
        }
    }
}

// After extraction
if (failed_slices.load() > 0) {
    return std::unexpected("Extraction had " + std::to_string(failed_slices.load()) +
                           " failed slices out of " + std::to_string(Nσ * Nr * Nt));
}
```

**Impact:** Clear error messages when PDE solves or spline fits fail

### Improvement 3: Add Structured Error Types
**Problem:** All methods return `std::string` errors (lines 17-67)

**Impact:** No programmatic error handling, hard to test specific error conditions

**Solution:** Add `PriceTableError` enum and struct:

```cpp
enum class PriceTableErrorCode {
    INVALID_GRID_UNSORTED,
    INVALID_GRID_NEGATIVE,
    INVALID_GRID_EMPTY,
    INVALID_GRID_TOO_FEW_POINTS,
    INVALID_K_REF,
    INVALID_BOUNDS,
    INCOMPLETE_RESULTS,        // Some PDE solves failed
    BSPLINE_FIT_FAILED,
    UNSUPPORTED_DIMENSION,
    IO_ERROR_WRITE_FAILED
};

struct PriceTableError {
    PriceTableErrorCode code;
    std::vector<double> invalid_values;
    std::optional<std::string> details;
    std::optional<size_t> axis_index;  // 0=m, 1=τ, 2=σ, 3=r
};
```

**Update:** Change all builder methods to return `std::expected<T, PriceTableError>`

### Improvement 4: Add `build_with_save()`
**Problem:** No way to stream temporary `PriceTensor<N>` to disk for external analysis

**Solution:** Add method that saves tensor before fitting:

```cpp
std::expected<PriceTableResult<N>, PriceTableError>
build_with_save(
    const PriceTableAxes<N>& axes,
    const std::filesystem::path& output_path,
    bool streaming = true);
```

**Implementation:**
1. Call existing build pipeline up to tensor extraction
2. Stream tensor to Parquet (temporary file → atomic rename)
3. Continue with B-spline fitting
4. Return same `PriceTableResult<N>` as `build()`

**Atomicity contract:**
- Remove stale `.tmp` files before starting (crash recovery)
- Write to `{output_path}.tmp` during extraction
- Atomic rename to `{output_path}` on success
- Delete `.tmp` on failure (idempotent retry)

**Use case:** External tools need raw PDE prices for validation/analysis
- Returns same result as `build()` but also streams `PriceTensor<N>` to disk before fitting
- Single execution provides both in-memory surface and archived raw tensor
- Tensor still discarded from memory after fitting (only saved to disk)

**Atomicity contract for `build_with_save()`:**
- **Before starting:** Remove any stale `{output_path}.tmp` (crash recovery)
- Writes to temporary file: `{output_path}.tmp`
- On success: atomic rename `{output_path}.tmp` → `{output_path}`
- On failure (PDE error, IO error, INCOMPLETE_RESULTS):
  - Temporary file deleted automatically in error path
  - Returns error, no file left on disk
  - Idempotent: safe to retry immediately
- **Crash recovery:** Stale .tmp files from crashed processes cleaned up on next run
- If output_path exists: overwritten atomically (no partial state visible)
- Thread-safe: multiple concurrent writes to different paths supported
- **Testing:** Inject failures at various stages, verify no corrupt files remain
- **Testing:** Create stale .tmp file, verify it's cleaned up on next run

**Memory impact (PriceTensor<N> size):**
- 50×30×20×10 grid: 2.4 MB (300,000 doubles × 8 bytes) - temporary during build only
- 200×100×50×20 grid: 160 MB (20,000,000 doubles × 8 bytes) - temporary during build only
- Very large grids (e.g., 150×80×60×30): can exceed 2 GB temporarily
- **Key improvement:** Tensor never included in result, always discarded after fitting

### 3. Fix Grid Configuration Override Bug

**Current bug (src/option/price_table_builder.cpp:131-142):**
```cpp
GridAccuracyParams accuracy;
// BUG: Overrides config_.grid_estimator!
accuracy.min_spatial_points = std::min(config_.grid_estimator.n_points(), size_t(100));
accuracy.max_spatial_points = std::max(config_.grid_estimator.n_points(), size_t(1200));
```

**Problem:** `GridAccuracyParams` forces min=100, max=1200, ignoring user's `config_.grid_estimator.n_points()`.

**Solution:** Respect user's `GridSpec` from config:

```cpp
template <size_t N>
BatchAmericanOptionResult
PriceTableBuilder<N>::solve_batch(
    const std::vector<AmericanOptionParams>& batch,
    const PriceTableAxes<N>& axes) const
{
    if constexpr (N != 4) {
        // Return empty result for N≠4
        return /* ... */;
    }

    BatchAmericanOptionSolver solver;

    // FIXED: Use grid_estimator from config directly
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = config_.grid_estimator.n_points();  // Use exact value
    accuracy.max_spatial_points = config_.grid_estimator.n_points();  // Use exact value
    accuracy.max_time_steps = config_.n_time;

    // Extract alpha parameter for sinh-spaced grids
    if (config_.grid_estimator.type() == GridSpec<double>::Type::SinhSpaced) {
        accuracy.alpha = config_.grid_estimator.concentration();
    }

    solver.set_grid_accuracy(accuracy);

    // Register maturity grid as snapshot times
    solver.set_snapshot_times(axes.grids[1]);  // maturity axis

    // Solve batch with shared grid (normalized chain solver)
    return solver.solve_batch(batch, true);  // use_shared_grid = true
}
```

**Impact:** User's grid specification from `PriceTableConfig::grid_estimator` is now respected exactly.

### 4. Parallelize Extraction Phase

**Current problem (src/option/price_table_builder.cpp:203-250):**
```cpp
// Serial loops over (σ, r) batches
for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
    for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
        // ... nested loop over maturity
        for (size_t j = 0; j < Nt; ++j) {
            // Build spline, interpolate
        }
    }
}
```

**Problem:** Serial execution leaves cores idle. For 20×10×30 = 6000 slices, this is embarrassingly parallel work.

**Solution:** Add OpenMP parallel loop over outer (σ, r) iteration:

```cpp
template <size_t N>
std::expected<PriceTensor<N>, std::string>
PriceTableBuilder<N>::extract_tensor(
    const BatchAmericanOptionResult& batch,
    const PriceTableAxes<N>& axes) const
{
    if constexpr (N != 4) {
        return std::unexpected("extract_tensor only supports N=4");
    }

    const size_t Nm = axes.grids[0].size();  // moneyness
    const size_t Nt = axes.grids[1].size();  // maturity
    const size_t Nσ = axes.grids[2].size();  // volatility
    const size_t Nr = axes.grids[3].size();  // rate

    // ... create tensor ...

    // Precompute log-moneyness
    std::vector<double> log_moneyness(Nm);
    for (size_t i = 0; i < Nm; ++i) {
        log_moneyness[i] = std::log(axes.grids[0][i]);
    }

    // Track failures atomically
    std::atomic<size_t> failed_slices{0};

    // FIXED: Parallelize outer loop over (σ, r) batches
    MANGO_PRAGMA_PARALLEL_FOR
    for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
        for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
            size_t batch_idx = σ_idx * Nr + r_idx;
            const auto& result_expected = batch.results[batch_idx];

            if (!result_expected.has_value()) {
                failed_slices.fetch_add(Nt, std::memory_order_relaxed);  // Nt failures
                // Fill with NaN
                for (size_t i = 0; i < Nm; ++i) {
                    for (size_t j = 0; j < Nt; ++j) {
                        tensor.view[i, j, σ_idx, r_idx] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
                continue;
            }

            const auto& result = result_expected.value();
            auto grid = result.grid();
            auto x_grid = grid->x();

            // For each maturity snapshot
            for (size_t j = 0; j < Nt; ++j) {
                std::span<const double> spatial_solution = result.at_time(j);

                // Build cubic spline
                CubicSpline<double> spline;
                auto build_error = spline.build(x_grid, spatial_solution);

                if (build_error.has_value()) {
                    // Spline build failed
                    failed_slices.fetch_add(1, std::memory_order_relaxed);
                    for (size_t i = 0; i < Nm; ++i) {
                        tensor.view[i, j, σ_idx, r_idx] = std::numeric_limits<double>::quiet_NaN();
                    }
                    continue;
                }

                // Interpolate and scale by K_ref
                const double K_ref = config_.K_ref;
                for (size_t i = 0; i < Nm; ++i) {
                    double normalized_price = spline.eval(log_moneyness[i]);
                    tensor.view[i, j, σ_idx, r_idx] = K_ref * normalized_price;
                }
            }
        }
    }

    // Check for failures
    if (failed_slices.load() > 0) {
        return std::unexpected("Extraction had " + std::to_string(failed_slices.load()) +
                               " failed slices out of " + std::to_string(Nσ * Nr * Nt));
    }

    return tensor;
}
```

**Performance impact:** Extraction phase now scales linearly with core count (previously serial bottleneck).

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
    SNAPSHOT_DROPPED,          // Adaptive grid dropped snapshots
    IO_ERROR_WRITE_FAILED,     // Failed to write to disk (precompute_with_save)
    IO_ERROR_READ_FAILED,      // Failed to read from disk (deserialize)
    IO_ERROR_INVALID_FORMAT    // File format invalid or corrupted
};

struct PriceTableError {
    PriceTableErrorCode code;
    std::vector<double> invalid_values;
    std::optional<double> expected_value;   // nullopt when not applicable
    std::optional<std::string> details;     // For complex contexts
    std::optional<size_t> axis_index;       // 0=m, 1=τ, 2=σ, 3=r, nullopt=not axis-specific
};
```

**Validation in build():**
```cpp
template <size_t N>
std::expected<PriceTableResult<N>, PriceTableError>
PriceTableBuilder<N>::build(const PriceTableAxes<N>& axes)
{
    if constexpr (N != 4) {
        return std::unexpected(PriceTableError{
            .code = PriceTableErrorCode::UNSUPPORTED_DIMENSION,
            .details = "Only N=4 currently supported"
        });
    }

    // Validate axes (delegates to PriceTableAxes::validate())
    auto validation = axes.validate();
    if (!validation.has_value()) {
        // Convert ValidationError to PriceTableError
        auto val_err = validation.error();
        return std::unexpected(PriceTableError{
            .code = (val_err.code == ValidationErrorCode::InvalidGridSize)
                    ? PriceTableErrorCode::INVALID_GRID_EMPTY
                    : PriceTableErrorCode::INVALID_GRID_UNSORTED,
            .invalid_values = {val_err.value},
            .axis_index = val_err.axis
        });
    }

    // Additional validation: moneyness must be positive (needed for log)
    if (axes.grids[0].front() <= 0.0) {
        return std::unexpected(PriceTableError{
            .code = PriceTableErrorCode::INVALID_GRID_NEGATIVE,
            .invalid_values = {axes.grids[0].front()},
            .details = "Moneyness must be positive (needed for log)",
            .axis_index = 0  // moneyness axis
        });
    }

    // Additional validation: maturity must be non-negative
    if (axes.grids[1].front() < 0.0) {
        return std::unexpected(PriceTableError{
            .code = PriceTableErrorCode::INVALID_GRID_NEGATIVE,
            .invalid_values = {axes.grids[1].front()},
            .axis_index = 1  // maturity axis
        });
    }

    // ... proceed with make_batch(), solve_batch(), extract_tensor(), fit_coeffs()
}
```

**Keep extraction assert as safety net:**
```cpp
// In extract_tensor() (line ~220)
// This should never trigger if validation is correct (defensive programming)
if (batch.results.empty()) {
    return std::unexpected("Batch results empty (programming error)");
}
```

**Impact:** Invalid grids caught immediately with actionable errors, before expensive PDE work.

## Implementation Order

Working with existing `PriceTableBuilder<N>` at `src/option/price_table_builder.{hpp,cpp}`:

1. **Add PriceTableError type** (src/support/error_types.hpp)
   - Define `PriceTableErrorCode` enum with all codes
   - Define `PriceTableError` struct with optional fields
   - Update all builder methods to return `PriceTableError` instead of `std::string`

2. **Add PriceTableResult<N> struct** (src/option/price_table_builder.hpp)
   - Define struct with `surface`, `n_pde_solves`, `precompute_time_seconds`, `fitting_stats`
   - Update `build()` signature to return `PriceTableResult<N>`

3. **Add BSplineFittingStats** (src/math/bspline_nd_separable.hpp)
   - Add diagnostics to `BSplineNDSeparable::fit()` return value
   - Track max residuals, condition numbers, failed slices per axis

4. **Fix grid configuration bug** (src/option/price_table_builder.cpp:131-142)
   - Remove min/max clamping in `GridAccuracyParams`
   - Use `config_.grid_estimator.n_points()` exactly

5. **Add snapshot validation** (src/option/price_table_builder.cpp:18-30)
   - Call `axes.validate()` at start of `build()`
   - Add moneyness positive check, maturity non-negative check
   - Return `PriceTableError` on validation failure

6. **Parallelize extraction** (src/option/price_table_builder.cpp:203-250)
   - Add `MANGO_PRAGMA_PARALLEL_FOR` to outer (σ, r) loop
   - Add `std::atomic<size_t> failed_slices` counter
   - Increment counter on PDE failure or spline build failure
   - Return error if `failed_slices > 0`

7. **Add build_with_save()** (new method in price_table_builder.cpp)
   - Call `build()` to get tensor
   - Stream tensor to Parquet before `fit_coeffs()`
   - Implement atomic write protocol (.tmp → rename)
   - Return same `PriceTableResult<N>` as `build()`

8. **Update tests** (tests/price_table_builder_test.cc)
   - Update for `PriceTableResult<N>` return type
   - Add error handling tests for `PriceTableError`
   - Add atomic write tests, schema migration tests

9. **Deprecate specialized builder** (mark in docs/comments)
   - Add deprecation warnings to `PriceTable4DBuilder`
   - Update examples to use `PriceTableBuilder<4>`
   - Migration guide for existing users

10. **Update documentation** (docs/API_GUIDE.md, inline docs)
    - Show generic builder usage patterns
    - Document error codes and handling
    - Performance characteristics

**Critical constraint:** No breaking changes to existing generic API (it's not used yet). Build new features incrementally.

## Testing Plan

### Unit Tests
- `price_table_error_test.cc` - Error type construction and handling for all codes
  - Test `PriceTableError` struct initialization
  - Test all `PriceTableErrorCode` values
  - Test optional field handling (`details`, `axis_index`, `expected_value`)

- `bspline_fitting_stats_test.cc` - B-spline fitting diagnostics
  - Test stats collection during `BSplineNDSeparable::fit()`
  - Test max residual tracking per axis
  - Test condition number estimation
  - Test failed slice counting

- `price_table_validation_test.cc` - Axes validation
  - Test empty grid detection (all 4 axes)
  - Test unsorted grid detection
  - Test negative moneyness detection
  - Test negative maturity detection
  - Test validation error conversion to `PriceTableError`

### Integration Tests
- `price_table_builder_test.cc` - End-to-end builder tests
  - Update for `PriceTableResult<N>` return type (not `shared_ptr<Surface>`)
  - Test error cases: empty grids, unsorted grids, negative values
  - Test grid configuration: verify `config_.grid_estimator` respected
  - Test diagnostics: verify `fitting_stats`, `n_pde_solves`, `precompute_time_seconds` populated
  - Test tensor NOT in result (verify it's temporary only)

- `price_table_extraction_test.cc` - Parallel extraction tests
  - Test atomic failure counter increments correctly
  - Test failed PDE solves tracked
  - Test failed spline builds tracked
  - Test error returned when `failed_slices > 0`
  - Test parallel execution (verify OpenMP pragma effective)

### Performance Tests
- Measure extraction phase scaling with core count (expect linear to 16-32 cores)
- Verify `PriceTensor<N>` temporary only (not in result)
- Benchmark with large grids (200×100×50×20 = 20M points)
- Verify atomic counter overhead negligible (< 1% of extraction time)

### Atomic Write Tests
- `price_table_atomic_write_test.cc` - Test failure modes for atomic write protocol:
  - **Inject write failure during serialization**: Verify `.tmp` file cleaned up, `IO_ERROR_WRITE_FAILED` returned
  - **Inject failure during atomic rename**: Verify cleanup, proper error propagation
  - **Pre-existing stale `.tmp` file**: Create stale `{output_path}.tmp` before calling `precompute_with_save()`, verify it's removed before starting
  - **Concurrent writes to different paths**: Multiple threads write to different files, verify no interference
  - **Idempotency**: Call `precompute_with_save()` twice with same path, verify second call succeeds (overwrites)
  - **Crash recovery simulation**: Leave stale `.tmp` file, restart process, verify cleanup on next run

### Schema Migration Tests
- `price_table_schema_test.cc` - Test schema versioning and migration:
  - **Load v1 (legacy) snapshot**: Create legacy Parquet file (no stats metadata), verify:
    - `deserialize_result()` succeeds
    - `n_pde_solves = 0`, `precompute_time_seconds = 0.0`
    - `fitting_stats.max_residual_overall = -1.0` (sentinel value)
    - Consumer can detect legacy file: `if (stats.max_residual_overall < 0) { /* legacy */ }`
  - **Load v2 (new) snapshot**: Create new Parquet file with full metadata, verify:
    - All fields present and correct
    - No sentinel values
  - **Migration tool roundtrip**:
    - Create v1 snapshot
    - Run `mango-upgrade-snapshots v1.parquet v2.parquet`
    - Load v2 snapshot, verify stats populated (re-computed from workspace)
  - **Schema version detection**: Verify loader correctly reads schema version from Arrow metadata
  - **Backward compatibility**: Verify v1 files still load after v2 implementation

## Migration Guide

### From Specialized PriceTable4DBuilder to Generic PriceTableBuilder<N>

**No immediate breaking changes** - specialized builder still works. Migrate incrementally:

#### Phase 1: Understand Architecture Differences

**Specialized (PriceTable4DBuilder):**
```cpp
auto builder_result = PriceTable4DBuilder::create(
    moneyness_vec, maturity_vec, vol_vec, rate_vec, K_ref);
auto result = builder_result->precompute(OptionType::PUT, 101, 1000);
```

**Generic (PriceTableBuilder<4>):**
```cpp
// 1. Build axes
PriceTableAxes<4> axes;
axes.grids[0] = moneyness_vec;
axes.grids[1] = maturity_vec;
axes.grids[2] = vol_vec;
axes.grids[3] = rate_vec;
axes.names = {"moneyness", "maturity", "volatility", "rate"};

// 2. Build config
PriceTableConfig config;
config.option_type = OptionType::PUT;
config.K_ref = K_ref;
config.grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value();
config.n_time = 1000;

// 3. Build
PriceTableBuilder<4> builder(config);
auto result = builder.build(axes);
```

#### Phase 2: Update Return Type Handling

**OLD (specialized builder, after refactor):**
```cpp
auto result = builder.precompute(config);
if (!result) {
    // Handle PriceTableError
}
const auto& surface = result->surface;
```

**NEW (generic builder):**
```cpp
auto result = builder.build(axes);
if (!result) {
    // Handle PriceTableError (same error type!)
}
const auto& result_val = result.value();
auto surface = result_val.surface;  // shared_ptr<const PriceTableSurface<4>>
const auto& stats = result_val.fitting_stats;
```

#### Phase 3: Access Diagnostics

**Both builders will support:**
```cpp
auto result = builder.build(axes);  // or precompute(config) for specialized
if (result) {
    std::cout << "PDE solves: " << result->n_pde_solves << "\n";
    std::cout << "Build time: " << result->precompute_time_seconds << "s\n";
    std::cout << "Max residual: " << result->fitting_stats.max_residual_overall << "\n";
}
```

### Non-Breaking Enhancements

**1. Grid configuration now respected:**
```cpp
PriceTableConfig config;
config.grid_estimator = GridSpec<double>::sinh_stretched(-2.0, 2.0, 201, 2.0).value();
config.n_time = 2000;

// Previously: ignored, clamped to [100, 1200]
// Now: used exactly as specified
```

**2. Better validation errors:**
```cpp
auto result = builder.build(axes);
if (!result) {
    const auto& err = result.error();
    if (err.axis_index.has_value()) {
        std::cout << "Error on axis " << err.axis_index.value() << ": ";
    }
    std::cout << static_cast<int>(err.code) << "\n";
}
```

**3. Parallel extraction (automatic):**
- Extraction now parallelized automatically
- No code changes needed
- Expect ~10-16× speedup on 32-core machines
}
```

### Phased Rollout Strategy

This refactor breaks the `PriceTableResult` API. Follow this three-phase migration:

**Phase 1: Add New APIs (Step 4 in Implementation Order)**
- Add `precompute_with_save(config, path) -> expected<PriceTableResult, PriceTableError>` to builder
- Add grid accessors to builder: `.moneyness()`, `.maturity()`, `.volatility()`, `.rate()`
- **Action:** Run `bazel test //tests:price_table_diagnostics_test` to verify new APIs work
- **Validation:** All new APIs have test coverage before Phase 2
- **Note:** Residuals already available via `result.fitting_stats` (no new API needed)

**Phase 2: Update All Consumers (Step 8 in Implementation Order)**
This phase happens after adding diagnostic APIs (Step 4) and before removing prices_4d (Step 9).
During this phase, also complete Steps 5-7 (validation, grid config, extraction parallelization).

- Find all uses: `grep -r "prices_4d" tests/ examples/ tools/ benchmarks/`
- **Update patterns:**
  - Tests: Replace `result.prices_4d` access with `result.fitting_stats` or direct `surface.eval()` calls
  - Arrow export: Replace `result.prices_4d` with loop over `builder.moneyness()` + `surface.eval()`
  - CLI diagnostic tools: Use `result.fitting_stats` for residuals, remove `--dump-raw-prices`
  - Benchmarks: Update expected memory values (varies by grid size: 2.4 MB to 160 MB+)
  - Examples/docs: Update to use `fitting_stats` and grid accessors (will be finalized in Step 10)
- **Action:** Run `bazel test //... && bazel build //...` to verify all consumers still work
- **Validation:** No compilation errors, all tests pass with old API still present

**Phase 3: Remove prices_4d (Step 9 in Implementation Order)**
- Remove `std::vector<double> prices_4d` field from `PriceTableResult` struct
- Remove `prices_4d` population code from `precompute()` implementation
- **Action:** Run `bazel test //... && bazel build //...` to verify nothing breaks
- **Validation:** Compilation succeeds (proves all consumers migrated), memory usage drops (2.4 MB to 160+ MB depending on grid)

**Rollback Plan:**
- If Phase 3 breaks: Revert removal, add deprecated warnings to `prices_4d` field
- If Phase 2 breaks specific consumer: Keep that consumer using old API, mark as technical debt

**Timeline:**
- Phases 1-3 happen in single PR (implementation order steps 4-9)
- No external API changes until Phase 3 completes
- Internal refactor only - no user-facing changes during Phases 1-2
- Steps 1-3 (error types, ownership) are prerequisites done first

## Success Metrics

**Memory usage:**
- Per-surface savings: ~2.4 MB eliminated by removing duplicate coefficient storage (Resolution #1)
  - Applies every time a surface is copied or stored
  - Grid-independent (coefficient storage size)
- Result size reduction (Resolution #2 removes prices_4d from PriceTableResult):
  - 50×30×20×10 grid: 2.4 MB saved
  - 200×100×50×20 grid: 160 MB saved
  - Very large grids (150×80×60×30+): can save >2 GB
  - One-time savings (not per-copy)

**Performance:**
- Extraction scaling: Linear speedup to 64 cores
- Error rate: Zero "moneyness outside PDE bounds" errors with explicit x_bounds
- Validation: 100% of invalid grids caught before PDE work

**API Quality:**
- No duplicate storage, clear ownership chain
- Structured error handling with PriceTableError

## Code Review Findings & Resolutions

### 1. Ownership Model - Move Semantics

**Issue:** `BSplineEvaluator::create(PriceTableWorkspace&& ws)` moving large workspace may cause extra copies. Returning `expected<BSplineEvaluator>` by value forces workspace to live inside expected temporarily.

**Resolution:**
- Change factory signature to return pointer with structured error:
  ```cpp
  static std::expected<std::shared_ptr<BSplineEvaluator>, PriceTableError>
  create(PriceTableWorkspace&& ws);
  ```
- Implementation moves workspace directly into heap-allocated BSplineEvaluator:
  ```cpp
  std::expected<std::shared_ptr<BSplineEvaluator>, PriceTableError>
  BSplineEvaluator::create(PriceTableWorkspace&& ws) {
      // Validate workspace...
      if (!ws.is_valid()) {
          return std::unexpected(PriceTableError{
              .code = PriceTableErrorCode::INVALID_GRID_SPEC
          });
      }
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
- Each iteration writes to unique `prices_view(m_idx, mat_idx, vol_idx, r_idx)` - no contention
- Spline objects are stack-allocated per iteration - no shared state
- Bounds check is per-slice (outer loop), not per-lattice-node (inner loop) - negligible overhead
- **Testing:** Inject batch failures, verify error is surfaced (not silent zeros)

### 3. API Breaking Changes - Serialization

**Issue:** Serialization code expects `PriceTableSurface(shared_ptr<workspace>)` constructor. Unconditional `.value()` on expected will terminate on failure. Additionally, deserializing only a surface loses `BSplineFittingStats` and other metadata.

**Resolution:**
- Update serialization to roundtrip full `PriceTableResult`:
  ```cpp
  // load_workspace_from_arrow now returns PriceTableError (not string)
  std::expected<PriceTableResult, PriceTableError>
  deserialize_result(const std::filesystem::path& file) {
      auto workspace_result = load_workspace_from_arrow(file);  // returns expected<..., PriceTableError>
      if (!workspace_result) {
          return std::unexpected(workspace_result.error());  // Already PriceTableError
      }

      // Extract metadata BEFORE moving workspace
      auto& ws = workspace_result.value();
      size_t n_solves = ws.n_pde_solves;
      double time = ws.precompute_time_seconds;
      BSplineFittingStats stats = ws.fitting_stats;

      auto evaluator_result = BSplineEvaluator::create(std::move(ws));
      if (!evaluator_result) {
          return std::unexpected(evaluator_result.error());  // Already PriceTableError
      }

      // Reconstruct full result (stats extracted before move)
      PriceTableResult result{
          .surface = PriceTableSurface(evaluator_result.value()),
          .n_pde_solves = n_solves,
          .precompute_time_seconds = time,
          .fitting_stats = stats
      };
      return result;
  }
  ```
- Arrow schema extended to include metadata: n_pde_solves, time, BSplineFittingStats
- Enables full quality validation on deserialized surfaces
- **Backward compatibility strategy:**
  - Schema version embedded in Arrow metadata (version=2, previous=1)
  - Version 1 (legacy): workspace only, no stats (pre-refactor snapshots)
  - Version 2 (new): workspace + n_pde_solves + time + BSplineFittingStats
  - Loader checks version, populates sentinel for missing fields:
    ```cpp
    if (schema_version == 1) {
        n_solves = 0;  // Unknown for legacy files
        time = 0.0;
        // Use sentinel values to distinguish "no data" from "zero residual"
        stats = BSplineFittingStats{};
        stats.max_residual_overall = -1.0;  // Sentinel: negative impossible for real residuals
        // Consumers check: if (stats.max_residual_overall < 0) { /* legacy file */ }
    }
    ```
  - Alternative: Add `bool has_fitting_stats = true;` field to schema v2 (explicit flag)
  - Migration tool provided: `mango-upgrade-snapshots` re-saves v1 → v2 with computed stats
- Add to implementation checklist: audit Arrow exporter, snapshot replay, scripting bindings
- **Testing:** Verify no extra allocations during deserialize (memory profiler or custom allocator)
- **Testing:** Load v1 snapshots, verify graceful degradation (no crash, usable surface)

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

### 7. Diagnostics - Grid Access & Residuals

**Issue:** Downstream consumers (Arrow export, CLI) need grid access after removing `prices_4d`. Users need residual diagnostics for quality assessment.

**Resolution:**
- Builder stores grids as members: `moneyness_`, `maturity_`, `volatility_`, `rate_`
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
- **Residuals available via `BSplineFittingStats`:**
  - Already computed during B-spline fitting process
  - Already included in `PriceTableResult.fitting_stats`
  - No need for separate `compute_residuals()` API (would require ground truth data)
- **Testing:** Test grid accessors return correct spans, fitting_stats populated correctly

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
    - **Unit tests comparing raw prices:** Use `result.fitting_stats` for quality checks or direct `surface.eval()` calls
    - **Integration tests diffing tables:** Use `precompute_with_save()` to serialize, diff files
    - **Arrow export:** Replace `result.prices_4d` with loop over `builder.moneyness()` + `surface.eval()`
    - **CLI diagnostic tools:** Use `result.fitting_stats` for residuals, remove `--dump-raw-prices`
    - **Benchmarks measuring memory:** Update expected values (varies by grid: 2.4 MB to 160 MB+)
  - **Migration path:**
    - Phase 1 (this PR): Add new APIs (`builder.moneyness()`, `precompute_with_save()`, grid accessors)
    - Phase 2 (this PR): Update all consumers to use new APIs and `fitting_stats`
    - Phase 3 (this PR): Remove `prices_4d` from `PriceTableResult`
- **Testing strategy:**
  - Before removing `prices_4d`: add tests for all new APIs
  - After removing: verify no compilation errors, all tests pass
  - Add regression test: ensure `PriceTableResult` size < 1 KB (varies by grid)
- **Performance validation:**
  - Benchmark extraction phase before/after parallelization
  - Document overhead of bounds checks (should be <1%)
  - Verify `BSplineFittingStats` computation overhead is negligible

## References

- Recent API refactor: PR #244 (custom_grid_config support)
- Memory profiling: price_table_builder.cpp:145 (10 GB tensor issue)
- Extraction bottleneck: price_table_extraction.cpp:70-122
- Grid config bug: price_table_builder.cpp:175-182
- Snapshot validation: price_table_extraction.cpp:42 (existing assert)
