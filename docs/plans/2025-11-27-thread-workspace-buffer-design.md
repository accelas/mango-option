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

### 1. ThreadWorkspaceBuffer (Byte-Based)

A lightweight RAII wrapper for per-thread raw byte storage with automatic fallback.

**Key design decisions:**
- Uses `std::byte` storage to avoid strict-aliasing issues when slicing into typed spans
- No `release()` method - buffer is allocated once and reused across iterations
- Workspace consumers create typed views via placement or `std::start_lifetime_as`

```cpp
// src/support/thread_workspace.hpp

#pragma once

#include <memory_resource>
#include <span>
#include <vector>
#include <cstddef>

namespace mango {

/// Per-thread workspace buffer with automatic fallback
///
/// Primary: monotonic_buffer_resource (fast bump allocation)
/// Fallback: thread-local synchronized_pool_resource (if exhausted)
///
/// Design principles:
/// - Buffer provides raw byte storage (std::byte)
/// - Workspace consumers handle alignment and typed access
/// - Buffer is allocated once per thread, reused across iterations
/// - No release() - memory stays valid for parallel region lifetime
///
/// Example:
///   MANGO_PRAGMA_PARALLEL
///   {
///       ThreadWorkspaceBuffer buffer(required_bytes);
///       auto ws = MyWorkspace::from_bytes(buffer.bytes(), n);
///
///       MANGO_PRAGMA_FOR_STATIC
///       for (size_t i = 0; i < count; ++i) {
///           // Use ws - same memory each iteration, zero allocations
///       }
///   }
///
class ThreadWorkspaceBuffer {
public:
    /// Construct with expected byte count
    explicit ThreadWorkspaceBuffer(size_t byte_count)
        : primary_pool_(byte_count, get_fallback_resource())
        , buffer_(&primary_pool_)
    {
        buffer_.resize(byte_count);
    }

    /// Get byte span view of buffer (stable for lifetime of object)
    std::span<std::byte> bytes() noexcept { return buffer_; }
    std::span<const std::byte> bytes() const noexcept { return buffer_; }

    /// Resize buffer - uses fallback if exceeds initial capacity
    void resize(size_t byte_count) { buffer_.resize(byte_count); }

    size_t size() const noexcept { return buffer_.size(); }

private:
    /// Thread-local synchronized pool for fallback allocations
    static std::pmr::memory_resource* get_fallback_resource() {
        thread_local std::pmr::synchronized_pool_resource pool;
        return &pool;
    }

    std::pmr::monotonic_buffer_resource primary_pool_;
    std::pmr::vector<std::byte> buffer_;
};

} // namespace mango
```

**Why no release():**
The original design called `release()` per iteration, which would invalidate the vector's internal pointers (use-after-free). Instead:
- Buffer is allocated once at parallel region start
- Same memory is reused across all iterations
- Workspace `from_bytes()` just creates typed views into stable storage
- Memory is freed when ThreadWorkspaceBuffer destructor runs at parallel region end

### 2. BSplineCollocationWorkspace<T>

Workspace that slices a byte buffer into properly-aligned typed spans for B-spline collocation.

**Key design decisions:**
- Takes `std::span<std::byte>` input to avoid type aliasing issues
- Uses `std::start_lifetime_as_array` (C++23) or placement new for typed access
- Separate storage for `int` pivots (not aliased through `T*`)
- Alignment via padding to 64-byte boundaries

```cpp
// src/math/bspline_collocation_workspace.hpp

#pragma once

#include <span>
#include <expected>
#include <string>
#include <cstddef>
#include <new>  // for std::launder, placement new

namespace mango {

/// Workspace for B-spline collocation solver
///
/// Slices external BYTE buffer into typed spans with proper alignment.
/// Uses placement new to start object lifetimes (avoids strict-aliasing UB).
///
/// Required arrays for bandwidth=4 (cubic B-splines):
/// - band_storage: 10n doubles (LAPACK banded format: ldab=10)
/// - lapack_storage: 10n doubles (LU factorization copy)
/// - pivots: n integers (pivot indices) - separate int storage
/// - coeffs: n doubles (result buffer)
///
template<typename T>
struct BSplineCollocationWorkspace {
    static constexpr size_t ALIGNMENT = 64;  // Cache line / AVX-512
    static constexpr size_t BANDWIDTH = 4;
    static constexpr size_t LDAB = 10;  // 2*kl + ku + 1 for bandwidth=4

    /// Align byte offset to specified alignment
    static constexpr size_t align_up(size_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    /// Calculate required buffer size in BYTES
    static size_t required_bytes(size_t n) {
        size_t offset = 0;

        // band_storage: 10n × sizeof(T), aligned
        offset = align_up(offset, alignof(T));
        offset += LDAB * n * sizeof(T);

        // lapack_storage: 10n × sizeof(T), aligned
        offset = align_up(offset, alignof(T));
        offset += LDAB * n * sizeof(T);

        // pivots: n × sizeof(int), aligned
        offset = align_up(offset, alignof(int));
        offset += n * sizeof(int);

        // coeffs: n × sizeof(T), aligned
        offset = align_up(offset, alignof(T));
        offset += n * sizeof(T);

        // Final alignment for SIMD
        return align_up(offset, ALIGNMENT);
    }

    /// Create workspace from external BYTE buffer
    static std::expected<BSplineCollocationWorkspace, std::string>
    from_bytes(std::span<std::byte> buffer, size_t n) {
        size_t required = required_bytes(n);
        if (buffer.size() < required) {
            return std::unexpected("Buffer too small for BSplineCollocationWorkspace");
        }

        BSplineCollocationWorkspace ws;
        ws.n_ = n;

        std::byte* ptr = buffer.data();
        size_t offset = 0;

        // band_storage
        offset = align_up(offset, alignof(T));
        ws.band_storage_ = std::span<T>(
            std::launder(reinterpret_cast<T*>(ptr + offset)), LDAB * n);
        offset += LDAB * n * sizeof(T);

        // lapack_storage
        offset = align_up(offset, alignof(T));
        ws.lapack_storage_ = std::span<T>(
            std::launder(reinterpret_cast<T*>(ptr + offset)), LDAB * n);
        offset += LDAB * n * sizeof(T);

        // pivots (int, NOT T)
        offset = align_up(offset, alignof(int));
        ws.pivots_ = std::span<int>(
            std::launder(reinterpret_cast<int*>(ptr + offset)), n);
        offset += n * sizeof(int);

        // coeffs
        offset = align_up(offset, alignof(T));
        ws.coeffs_ = std::span<T>(
            std::launder(reinterpret_cast<T*>(ptr + offset)), n);

        return ws;
    }

    // Accessors - return typed spans (lifetime started by from_bytes)
    std::span<T> band_storage() { return band_storage_; }
    std::span<T> lapack_storage() { return lapack_storage_; }
    std::span<int> pivots() { return pivots_; }  // Properly typed int span
    std::span<T> coeffs() { return coeffs_; }

    size_t size() const { return n_; }

private:
    size_t n_ = 0;
    std::span<T> band_storage_;
    std::span<T> lapack_storage_;
    std::span<int> pivots_;  // int, not T
    std::span<T> coeffs_;
};

} // namespace mango
```

**Strict-aliasing compliance:**
- `std::byte*` can alias any type (like `char*`)
- `std::launder` ensures the compiler recognizes the new object lifetime
- In C++23, can use `std::start_lifetime_as_array<T>(ptr, n)` instead
- Pivots are stored as `int` directly, not cast through `T`

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
    const size_t ws_bytes = BSplineCollocationWorkspace<T>::required_bytes(n_axis);
    const size_t n_slices = compute_n_slices<Axis>();

    // Thread-safe statistics
    T global_max_residual = T{0};
    T global_max_condition = T{0};
    size_t global_failed = 0;

    MANGO_PRAGMA_PARALLEL
    {
        // Per-thread workspace buffer (bytes, not T)
        ThreadWorkspaceBuffer buffer(ws_bytes);
        auto ws = BSplineCollocationWorkspace<T>::from_bytes(buffer.bytes(), n_axis).value();

        // Per-thread slice buffer
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
            // Note: solver is accessed via optional dereference (static dispatch)
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

**Problem:** `BSplineCollocation1D<T>` is not default-constructible. It's produced via a factory method that returns `std::expected<BSplineCollocation1D<T>, InterpolationError>` because construction can fail (e.g., insufficient points for spline degree).

**Solution:** Use `std::optional` for deferred initialization:

```cpp
// WRONG: Assumes default-constructible (compile error)
std::array<BSplineCollocation1D<T>, N> solvers_;

// WRONG: Dynamic dispatch via pointer (blocks inlining)
std::array<std::unique_ptr<BSplineCollocation1D<T>>, N> solvers_;
solvers_[Axis]->fit_with_workspace(...)  // vtable lookup

// CORRECT: Deferred initialization with static dispatch
std::array<std::optional<BSplineCollocation1D<T>>, N> solvers_;

// Initialization (in constructor or build phase):
for (size_t axis = 0; axis < N; ++axis) {
    auto result = BSplineCollocation1D<T>::create(knots[axis], degree);
    if (result.has_value()) {
        solvers_[axis].emplace(std::move(result.value()));
    } else {
        return std::unexpected(result.error());  // Propagate failure
    }
}

// Usage (in hot path):
solvers_[Axis]->fit_with_workspace(...)  // direct call through optional, inlinable
```

**Why `std::optional` enables static dispatch:**
- `std::optional<T>` stores `T` inline (no heap allocation)
- `operator->()` and `operator*()` return direct references (no vtable)
- Compiler can see through `optional` and inline the underlying call
- Cost: one byte for engaged flag + padding

### Why Static Dispatch

| Aspect | Dynamic (unique_ptr) | Static (optional) |
|--------|----------------------|-------------------|
| Storage | Heap pointer | Inline + 1 byte flag |
| Vtable lookup | ~3 cycles/call | 0 |
| Inlining | Blocked | Enabled |
| Per-fit overhead | 24k × 3 = 72k cycles | 0 |

For 24,000 fits in a 4D price table, static dispatch eliminates ~72k cycles of overhead and enables the compiler to inline small functions like coefficient access.

### Template Instantiation

`fit_axis<Axis>()` is already a template, so the axis index is compile-time known:

```cpp
template<size_t Axis>
void fit_axis(...) {
    // Axis is constexpr - compiler generates specialized code per axis
    const size_t n_axis = dims_[Axis];  // constant-folded if dims_ is constexpr

    // Static dispatch to correct solver (optional dereference inlined)
    auto& solver = *solvers_[Axis];  // Direct access, no runtime indexing
    solver.fit_with_workspace(...);  // direct call, inlinable
}
```

**Note:** The `*solvers_[Axis]` dereference is safe because:
1. Construction validates all solvers are created successfully
2. The class invariant guarantees all optionals are engaged after construction
3. If construction fails, the builder returns an error before reaching `fit_axis`

## Memory Layout

```
ThreadWorkspaceBuffer (std::byte storage):
┌─────────────────────────────────────────────────────────────┐
│ [std::byte..................................................] │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
BSplineCollocationWorkspace::from_bytes() slices with padding:
┌────────────┬───┬────────────┬───┬──────────┬───┬──────────┬───┐
│ band_stor. │pad│ lapack_st. │pad│ pivots   │pad│ coeffs   │pad│
│ T[10n]     │   │ T[10n]     │   │ int[n]   │   │ T[n]     │   │
└────────────┴───┴────────────┴───┴──────────┴───┴──────────┴───┘
      Typed views via std::launder (proper object lifetime)
```

**Key points:**
- Base storage is `std::byte` (can alias any type per standard)
- `from_bytes()` uses `std::launder` to reinterpret as typed spans
- Pivots are stored as `int[]` directly (not aliased through `T`)
- Each segment aligned to `alignof(T)` or `alignof(int)` as appropriate

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
    ThreadWorkspaceBuffer buffer(workspace_size_bytes);

    MANGO_PRAGMA_FOR_STATIC
    for (size_t i = 0; i < params.size(); ++i) {
        auto workspace = PDEWorkspace::from_bytes(buffer.bytes(), n).value();
        // ... use workspace ...
        // No release() - same buffer reused each iteration
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
