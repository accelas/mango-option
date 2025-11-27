# ThreadWorkspaceBuffer Design

## Overview

Introduce `ThreadWorkspaceBuffer<T>` as a reusable per-thread memory buffer for parallel algorithms, and add workspace pooling to B-spline collocation to eliminate per-fit allocations.

## Problem Statement

### Current Issues

1. **B-spline collocation allocates per fit**: Each call to `BSplineCollocation1D::fit()` creates:
   - `BandedMatrix<T>` (~10n doubles)
   - `BandedLUWorkspace<T>` (~10n doubles + n pivot indices)
   - `std::vector<T> coeffs` (n doubles)

2. **Duplicated PMR pattern**: The per-thread PMR buffer pattern is repeated in `american_option_batch.cpp`:
   ```cpp
   MANGO_PRAGMA_PARALLEL
   {
       std::pmr::monotonic_buffer_resource thread_pool(size_bytes);
       std::pmr::vector<double> thread_buffer(&thread_pool);
       thread_buffer.resize(size_elements);
       // ...
   }
   ```

3. **Two arena classes with overlapping purpose**: `AlignedArena` and the ad-hoc PMR pattern both provide bump-pointer allocation.

### Impact

For a 4D price table with dimensions (20, 10, 20, 10):
- Axis 3: 4,000 fits × 2 allocations = 8,000 allocations
- Axis 2: 2,000 fits × 2 allocations = 4,000 allocations
- Total: ~24,000 allocations during B-spline fitting

## Design

### 1. ThreadWorkspaceBuffer<T>

A lightweight RAII wrapper for per-thread PMR buffers with automatic fallback.

```cpp
// src/support/thread_workspace.hpp

#pragma once

#include <memory_resource>
#include <span>
#include <vector>

namespace mango {

/// Per-thread workspace buffer with automatic fallback
///
/// Primary: monotonic_buffer_resource (fast bump allocation)
/// Fallback: thread-local synchronized_pool_resource (if exhausted)
///
/// Design principles:
/// - Buffer provides raw storage only
/// - Workspace consumers handle alignment via padding (like PDEWorkspace)
/// - Fallback is transparent - resize beyond capacity just works
///
/// Example:
///   MANGO_PRAGMA_PARALLEL
///   {
///       ThreadWorkspaceBuffer buffer(expected_size);
///       auto ws = MyWorkspace::from_buffer(buffer.span(), n);
///
///       MANGO_PRAGMA_FOR_STATIC
///       for (size_t i = 0; i < count; ++i) {
///           // Use ws - zero allocations
///       }
///   }
///
template<typename T = double>
class ThreadWorkspaceBuffer {
public:
    /// Construct with expected element count
    explicit ThreadWorkspaceBuffer(size_t element_count)
        : primary_pool_(element_count * sizeof(T), get_fallback_resource())
        , buffer_(&primary_pool_)
    {
        buffer_.resize(element_count);
    }

    /// Get span view of buffer
    std::span<T> span() noexcept { return buffer_; }
    std::span<const T> span() const noexcept { return buffer_; }

    /// Resize buffer - uses fallback if exceeds initial capacity
    void resize(size_t count) { buffer_.resize(count); }

    /// Release for reuse (resets monotonic offset)
    void release() {
        primary_pool_.release();
        buffer_.clear();
    }

    size_t size() const noexcept { return buffer_.size(); }

private:
    /// Thread-local synchronized pool for fallback allocations
    static std::pmr::memory_resource* get_fallback_resource() {
        thread_local std::pmr::synchronized_pool_resource pool;
        return &pool;
    }

    std::pmr::monotonic_buffer_resource primary_pool_;
    std::pmr::vector<T> buffer_;
};

} // namespace mango
```

### 2. BSplineCollocationWorkspace<T>

Workspace that slices a buffer into properly-aligned spans for B-spline collocation.

```cpp
// src/math/bspline_collocation_workspace.hpp

#pragma once

#include <span>
#include <expected>
#include <string>
#include <cstddef>

namespace mango {

/// Workspace for B-spline collocation solver
///
/// Slices external buffer into named spans with SIMD-safe padding.
/// Alignment is handled by padding, not buffer base address.
///
/// Required arrays for bandwidth=4 (cubic B-splines):
/// - band_storage: 10n doubles (LAPACK banded format: ldab=10)
/// - lapack_storage: 10n doubles (LU factorization copy)
/// - pivots: n integers (pivot indices)
/// - coeffs: n doubles (result buffer)
///
template<typename T>
struct BSplineCollocationWorkspace {
    static constexpr size_t SIMD_WIDTH = 8;
    static constexpr size_t BANDWIDTH = 4;
    static constexpr size_t LDAB = 10;  // 2*kl + ku + 1 for bandwidth=4

    static constexpr size_t pad_to_simd(size_t n) {
        return ((n + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
    }

    /// Calculate required buffer size in elements of type T
    static size_t required_size(size_t n) {
        size_t ldab_n = pad_to_simd(LDAB * n);      // band_storage
        size_t lapack = pad_to_simd(LDAB * n);      // lapack_storage
        size_t pivots_as_T = pad_to_simd((n * sizeof(int) + sizeof(T) - 1) / sizeof(T));
        size_t coeffs = pad_to_simd(n);
        return ldab_n + lapack + pivots_as_T + coeffs;
    }

    /// Create workspace from external buffer
    static std::expected<BSplineCollocationWorkspace, std::string>
    from_buffer(std::span<T> buffer, size_t n) {
        size_t required = required_size(n);
        if (buffer.size() < required) {
            return std::unexpected("Buffer too small for BSplineCollocationWorkspace");
        }

        BSplineCollocationWorkspace ws;
        ws.n_ = n;

        size_t offset = 0;
        size_t ldab_n = pad_to_simd(LDAB * n);
        size_t pivots_as_T = pad_to_simd((n * sizeof(int) + sizeof(T) - 1) / sizeof(T));
        size_t coeffs_n = pad_to_simd(n);

        ws.band_storage_ = buffer.subspan(offset, ldab_n);
        offset += ldab_n;

        ws.lapack_storage_ = buffer.subspan(offset, ldab_n);
        offset += ldab_n;

        ws.pivots_storage_ = buffer.subspan(offset, pivots_as_T);
        offset += pivots_as_T;

        ws.coeffs_ = buffer.subspan(offset, coeffs_n);

        return ws;
    }

    // Accessors - return logical size spans
    std::span<T> band_storage() { return band_storage_.subspan(0, LDAB * n_); }
    std::span<T> lapack_storage() { return lapack_storage_.subspan(0, LDAB * n_); }
    std::span<int> pivots() {
        return std::span<int>(reinterpret_cast<int*>(pivots_storage_.data()), n_);
    }
    std::span<T> coeffs() { return coeffs_.subspan(0, n_); }

    size_t size() const { return n_; }

private:
    size_t n_ = 0;
    std::span<T> band_storage_;
    std::span<T> lapack_storage_;
    std::span<T> pivots_storage_;
    std::span<T> coeffs_;
};

} // namespace mango
```

### 3. Modified BSplineCollocation1D

Add `fit_with_workspace()` method that uses external workspace.

```cpp
// In bspline_collocation.hpp

/// Fit with external workspace (zero-allocation variant)
[[nodiscard]] std::expected<BSplineCollocationResult<T>, InterpolationError>
fit_with_workspace(
    std::span<const T> values,
    BSplineCollocationWorkspace<T>& ws,
    const BSplineCollocationConfig<T>& config = {})
{
    // Use ws.band_storage(), ws.lapack_storage(), ws.pivots(), ws.coeffs()
    // instead of allocating BandedMatrix and BandedLUWorkspace
    // ...
}
```

### 4. Modified BSplineNDSeparable

Use ThreadWorkspaceBuffer and parallelize slice iteration.

```cpp
template<size_t Axis>
void fit_axis(
    std::vector<T>& coeffs,
    T tolerance,
    std::array<T, N>& max_residuals,
    std::array<T, N>& conditions,
    std::array<size_t, N>& failed)
{
    const size_t n_axis = dims_[Axis];
    const size_t ws_size = BSplineCollocationWorkspace<T>::required_size(n_axis);
    const size_t n_slices = compute_n_slices<Axis>();

    // Thread-safe statistics
    T global_max_residual = T{0};
    T global_max_condition = T{0};
    size_t global_failed = 0;

    MANGO_PRAGMA_PARALLEL
    {
        // Per-thread workspace buffer
        ThreadWorkspaceBuffer<T> buffer(ws_size);
        auto ws = BSplineCollocationWorkspace<T>::from_buffer(buffer.span(), n_axis).value();

        // Per-thread slice buffers
        std::vector<T> slice_buffer(n_axis);

        // Thread-local statistics
        T local_max_residual = T{0};
        T local_max_condition = T{0};
        size_t local_failed = 0;

        MANGO_PRAGMA_FOR_STATIC
        for (size_t slice_idx = 0; slice_idx < n_slices; ++slice_idx) {
            // Extract slice
            extract_slice<Axis>(coeffs, slice_idx, slice_buffer);

            // Fit with workspace (zero allocations)
            auto result = solvers_[Axis]->fit_with_workspace(
                slice_buffer, ws, BSplineCollocationConfig<T>{.tolerance = tolerance});

            if (result.has_value()) {
                local_max_residual = std::max(local_max_residual, result->max_residual);
                local_max_condition = std::max(local_max_condition, result->condition_estimate);

                // Write coefficients back
                write_slice<Axis>(coeffs, slice_idx, ws.coeffs());
            } else {
                ++local_failed;
            }
        }

        // Reduce thread-local statistics
        MANGO_PRAGMA_CRITICAL
        {
            global_max_residual = std::max(global_max_residual, local_max_residual);
            global_max_condition = std::max(global_max_condition, local_max_condition);
            global_failed += local_failed;
        }
    }

    max_residuals[Axis] = global_max_residual;
    conditions[Axis] = global_max_condition;
    failed[Axis] = global_failed;
}
```

## Dispatch Mechanism

Use **static dispatch** for all hot-path function calls to enable inlining and avoid vtable overhead.

### BSplineNDSeparable Solver Storage

```cpp
// WRONG: Dynamic dispatch via pointer (blocks inlining)
std::array<std::unique_ptr<BSplineCollocation1D<T>>, N> solvers_;
solvers_[Axis]->fit_with_workspace(...)  // vtable lookup

// CORRECT: Static dispatch via direct member
std::array<BSplineCollocation1D<T>, N> solvers_;
solvers_[Axis].fit_with_workspace(...)  // direct call, inlinable
```

### Why Static Dispatch

| Aspect | Dynamic (pointer) | Static (direct) |
|--------|-------------------|-----------------|
| Vtable lookup | ~3 cycles/call | 0 |
| Inlining | Blocked | Enabled |
| Per-fit overhead | 24k × 3 = 72k cycles | 0 |
| Code size | Smaller (one copy) | May increase (inlined) |

For 24,000 fits in a 4D price table, static dispatch eliminates ~72k cycles of overhead and enables the compiler to inline small functions like coefficient access.

### Template Instantiation

`fit_axis<Axis>()` is already a template, so the axis index is compile-time known:

```cpp
template<size_t Axis>
void fit_axis(...) {
    // Axis is constexpr - compiler generates specialized code per axis
    const size_t n_axis = dims_[Axis];  // constant-folded if dims_ is constexpr

    // Static dispatch to correct solver
    auto& solver = solvers_[Axis];  // no runtime indexing if Axis is constexpr
    solver.fit_with_workspace(...);  // direct call
}
```

## Memory Layout

```
ThreadWorkspaceBuffer (raw storage):
┌─────────────────────────────────────────────────────────────┐
│ [............................................................] │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
BSplineCollocationWorkspace::from_buffer() slices with padding:
┌────────────┬───┬────────────┬───┬──────────┬───┬──────────┬───┐
│ band_stor. │pad│ lapack_st. │pad│ pivots   │pad│ coeffs   │pad│
│ (10n)      │   │ (10n)      │   │ (n ints) │   │ (n)      │   │
└────────────┴───┴────────────┴───┴──────────┴───┴──────────┴───┘
      64-byte aligned spans (via padding, not base address)
```

## Fallback Behavior

```
Request fits in buffer:           Request exceeds buffer:
┌──────────────────────┐         ┌──────────────────────┐
│ monotonic_buffer     │         │ monotonic (exhausted)│
│ [used...][free.....]│         │ [full................]│
│ ← O(1) bump alloc    │         └──────────┬───────────┘
└──────────────────────┘                    │ upstream
                                            ▼
                                 ┌──────────────────────┐
                                 │ synchronized_pool    │
                                 │ (thread-local)       │
                                 │ ← Efficient pooling  │
                                 └──────────────────────┘
```

## OpenMP Pattern Analysis

Comprehensive audit of all OpenMP usage patterns in the codebase.

### Pattern Categories

#### Pattern 1: Simple Parallel For (No Per-Thread State)

These are trivially parallel with no workspace needed.

| File | Lines | Construct | Notes |
|------|-------|-----------|-------|
| `iv_solver_interpolated.cpp` | 220-227 | `MANGO_PRAGMA_PARALLEL_FOR` | Atomic counter only |
| `iv_solver_fdm.cpp` | 335-343 | `MANGO_PRAGMA_PARALLEL_FOR` | Atomic counter only |

**ThreadWorkspaceBuffer needed:** No

#### Pattern 2: SIMD Vectorization (Not Threading)

These are SIMD hints for vectorization, not thread parallelization.

| File | Lines | Construct | Notes |
|------|-------|-----------|-------|
| `american_pde_solver.hpp` | 80-83, 91-95, 175-179, 187-191 | `#pragma omp simd` | Payoff/obstacle |
| `price_table_extraction.cpp` | 59-62 | `MANGO_PRAGMA_SIMD` | Log-moneyness |

**ThreadWorkspaceBuffer needed:** No

#### Pattern 3: Parallel Region with Per-Thread Workspace (PRIMARY TARGET)

This is THE pattern ThreadWorkspaceBuffer replaces.

| File | Lines | Construct | Current Implementation |
|------|-------|-----------|----------------------|
| `american_option_batch.cpp` | 466-604 | `MANGO_PRAGMA_PARALLEL` + `MANGO_PRAGMA_FOR_STATIC` | Per-thread `monotonic_buffer_resource`, per-thread `pmr::vector<double>`, per-thread Grid, manual heap fallback |

**ThreadWorkspaceBuffer needed:** Yes (primary migration target)

```cpp
// Current pattern (to be replaced):
MANGO_PRAGMA_PARALLEL
{
    std::pmr::monotonic_buffer_resource thread_pool(workspace_size_bytes);
    std::pmr::vector<double> thread_buffer(&thread_pool);
    thread_buffer.resize(workspace_size_elements);

    MANGO_PRAGMA_FOR_STATIC
    for (size_t i = 0; i < params.size(); ++i) {
        auto workspace = PDEWorkspace::from_buffer(thread_buffer, n);
        // ... use workspace ...
        if (!use_shared_grid) {
            thread_pool.release();
        }
    }
}

// New pattern (with ThreadWorkspaceBuffer):
MANGO_PRAGMA_PARALLEL
{
    ThreadWorkspaceBuffer<double> buffer(workspace_size_elements);

    MANGO_PRAGMA_FOR_STATIC
    for (size_t i = 0; i < params.size(); ++i) {
        auto workspace = PDEWorkspace::from_buffer(buffer.span(), n);
        // ... use workspace ...
        buffer.release();  // Reset for next iteration
    }
}
```

#### Pattern 4: Parallel For with Per-Iteration Allocation

These allocate objects (CubicSpline) per iteration. Could benefit from workspace pooling.

| File | Lines | Construct | Per-Iteration Allocation |
|------|-------|-----------|-------------------------|
| `price_table_builder.cpp` | 358-419 | `MANGO_PRAGMA_PARALLEL` + `MANGO_PRAGMA_FOR_COLLAPSE2` | `CubicSpline<double>` per (σ,r) |
| `price_table_extraction.cpp` | 73-122 | `MANGO_PRAGMA_PARALLEL_FOR` | `CubicSpline<double>` per batch result |

**ThreadWorkspaceBuffer needed:** Optional (requires CubicSpline workspace API)

#### Pattern 5: Thread-Local Cache (Manual Implementation)

Manual workspace caching that could be simplified with ThreadWorkspaceBuffer.

| File | Lines | Construct | Current Implementation |
|------|-------|-----------|----------------------|
| `iv_solver_fdm.cpp` | 169-191 | `thread_local` cache | `thread_local std::unordered_map<size_t, std::pmr::vector<double>>` |

**ThreadWorkspaceBuffer needed:** Optional (could simplify, but works as-is)

### Summary Table

| File | Pattern | Needs Migration | Priority |
|------|---------|-----------------|----------|
| `american_option_batch.cpp` | Per-thread workspace | **Yes** | P0 (primary) |
| `bspline_nd_separable.hpp` | New parallelization | **Yes** | P0 (new feature) |
| `iv_solver_fdm.cpp` | Thread-local cache | Optional | P2 |
| `price_table_builder.cpp` | Per-iteration alloc | Optional | P2 |
| `price_table_extraction.cpp` | Per-iteration alloc | Optional | P2 |
| `iv_solver_interpolated.cpp` | Simple parallel | No | - |
| `american_pde_solver.hpp` | SIMD only | No | - |

## Migration Plan

### Prerequisites: Extend parallel.hpp

Add missing OpenMP macros to `src/support/parallel.hpp`:

```cpp
// Critical section for thread-safe reduction
#if defined(_OPENMP)
    #define MANGO_PRAGMA_CRITICAL _Pragma("omp critical")
#else
    #define MANGO_PRAGMA_CRITICAL
#endif
```

### Phase 1: Add ThreadWorkspaceBuffer (P0)
1. Create `src/support/thread_workspace.hpp`
2. Add unit tests for:
   - Basic allocation and span access
   - Fallback to synchronized pool when exhausted
   - Thread safety under OpenMP parallel regions
   - Release/reuse behavior
3. Refactor `american_option_batch.cpp` to use ThreadWorkspaceBuffer
   - Replace manual PMR pattern (lines 466-604)
   - Remove heap fallback code (ThreadWorkspaceBuffer handles this)
   - Verify identical results with existing tests

### Phase 2: Add BSplineCollocationWorkspace (P0)
1. Create `src/math/bspline_collocation_workspace.hpp`
2. Add `fit_with_workspace()` to `BSplineCollocation1D`
3. Modify `BSplineNDSeparable` to:
   - Use ThreadWorkspaceBuffer for per-thread workspace
   - Parallelize slice iteration with `MANGO_PRAGMA_PARALLEL` + `MANGO_PRAGMA_FOR_STATIC`
   - Use thread-local statistics with critical section reduction

### Phase 3: Optional Consolidation (P2)
1. **IVSolverFDM** (optional): Replace thread_local cache with ThreadWorkspaceBuffer
2. **PriceTableBuilder/Extraction** (optional): Add CubicSpline workspace API
3. **AlignedArena**: Keep for PriceTensor (SIMD alignment at base address may be needed)

## Performance Impact

| Stage | Before | After |
|-------|--------|-------|
| B-spline fit() | 2 allocs/fit | 0 allocs/fit |
| Slice iteration | Sequential | Parallel (OpenMP) |
| Memory fragmentation | High | Low (monotonic) |

For 4D price table (20×10×20×10):
- Before: ~24,000 allocations
- After: N (number of threads) allocations

## Testing

1. Unit tests for `ThreadWorkspaceBuffer`:
   - Basic allocation
   - Fallback to synchronized pool
   - Thread safety under OpenMP

2. Unit tests for `BSplineCollocationWorkspace`:
   - Buffer slicing
   - Alignment verification
   - Size calculation

3. Integration tests:
   - `BSplineNDSeparable` produces identical results with workspace
   - No memory leaks under parallel execution

4. Benchmark:
   - Compare B-spline fitting time before/after
   - Measure allocation overhead reduction
