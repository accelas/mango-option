# ThreadWorkspaceBuffer Design

## Overview

Introduce `ThreadWorkspaceBuffer` as a reusable per-thread raw byte buffer for parallel algorithms, and add `BSplineCollocationWorkspace<T>` for zero-allocation B-spline fitting.

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
///       ThreadWorkspaceBuffer buffer(MyWorkspace::required_bytes(n));
///
///       // Create workspace ONCE per thread (starts object lifetimes)
///       auto ws = MyWorkspace::from_bytes(buffer.bytes(), n).value();
///
///       MANGO_PRAGMA_FOR_STATIC
///       for (size_t i = 0; i < count; ++i) {
///           // Reuse ws - solver overwrites spans each iteration.
///           // No re-initialization needed for trivial types.
///           solver.fit_with_workspace(input[i], ws);
///       }
///   }
///
/// Lifecycle notes:
/// - from_bytes() starts object lifetimes via start_array_lifetime()
/// - Workspace is created ONCE per thread, NOT per iteration
/// - Solver methods overwrite workspace arrays each iteration
/// - For B-spline fitting: band_storage, lapack_storage, pivots, coeffs
///   are all written fresh each fit - no stale state accumulates
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
    ///
    /// The returned span is valid until the ThreadWorkspaceBuffer is destroyed.
    /// All workspace objects created from this span remain valid for the same duration.
    std::span<std::byte> bytes() noexcept { return buffer_; }
    std::span<const std::byte> bytes() const noexcept { return buffer_; }

    size_t size() const noexcept { return buffer_.size(); }

    // Note: No resize() method. Callers must know the required size upfront.
    // This ensures bytes() spans remain stable for the buffer's lifetime.

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
#include <algorithm>    // for std::max
#include <memory>       // for std::start_lifetime_as_array (C++23), uninitialized_default_construct_n
#include <type_traits>  // for std::is_trivially_default_constructible_v
#include <new>          // for std::construct_at

namespace mango {

// C++23 feature detection and fallback
// std::start_lifetime_as_array is P2590R2, available in GCC 14+, Clang 18+
#ifdef __cpp_lib_start_lifetime_as
    // C++23: Use standard library function
    template<typename T>
    T* start_array_lifetime(void* p, [[maybe_unused]] size_t n) {
        return std::start_lifetime_as_array<T>(p, n);
    }
#else
    // Fallback: Start object lifetime without C++23 facility
    // IMPORTANT: Even trivial types need explicit lifetime start per [basic.life].
    // We use std::construct_at which is guaranteed to begin object lifetime.
    template<typename T>
    T* start_array_lifetime(void* p, size_t n) {
        // void* -> T* requires reinterpret_cast (static_cast is UB for unrelated types)
        auto* typed = reinterpret_cast<T*>(p);
        if constexpr (std::is_trivially_default_constructible_v<T>) {
            // Trivial types: construct_at is required to start lifetime but
            // doesn't actually write anything for trivial default constructors.
            // This is the minimal correct way to start lifetime pre-C++23.
            for (size_t i = 0; i < n; ++i) {
                std::construct_at(typed + i);
            }
        } else {
            // Non-trivial types: must explicitly construct
            std::uninitialized_default_construct_n(typed, n);
        }
        return typed;
    }
#endif

/// Workspace for B-spline collocation solver
///
/// Slices external BYTE buffer into typed spans with proper alignment.
/// Uses std::start_lifetime_as_array (C++23) or fallback to start object
/// lifetimes, avoiding strict-aliasing UB.
///
/// Required arrays for bandwidth=4 (cubic B-splines):
/// - band_storage: 10n doubles (LAPACK banded format: ldab=10)
/// - lapack_storage: 10n doubles (LU factorization copy)
/// - pivots: n integers (pivot indices) - separate int storage
/// - coeffs: n doubles (result buffer)
///
/// All storage regions are aligned to 64-byte boundaries for SIMD.
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

    /// Effective alignment for each block (max of SIMD alignment and type alignment)
    static constexpr size_t block_alignment_T = std::max(ALIGNMENT, alignof(T));
    static constexpr size_t block_alignment_int = std::max(ALIGNMENT, alignof(int));

    /// Calculate required buffer size in BYTES
    static size_t required_bytes(size_t n) {
        size_t offset = 0;

        // band_storage: 10n × sizeof(T), 64-byte aligned
        offset = align_up(offset, block_alignment_T);
        offset += LDAB * n * sizeof(T);

        // lapack_storage: 10n × sizeof(T), 64-byte aligned
        offset = align_up(offset, block_alignment_T);
        offset += LDAB * n * sizeof(T);

        // pivots: n × sizeof(int), 64-byte aligned
        offset = align_up(offset, block_alignment_int);
        offset += n * sizeof(int);

        // coeffs: n × sizeof(T), 64-byte aligned
        offset = align_up(offset, block_alignment_T);
        offset += n * sizeof(T);

        // Final alignment
        return align_up(offset, ALIGNMENT);
    }

    /// Create workspace from external BYTE buffer
    ///
    /// Uses std::start_lifetime_as_array (C++23) to properly start the lifetime
    /// of objects in the buffer. This is NOT equivalent to std::launder, which
    /// only provides pointer provenance but does NOT create objects.
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

        // band_storage - start lifetime of T[LDAB*n]
        offset = align_up(offset, block_alignment_T);
        ws.band_storage_ = std::span<T>(
            start_array_lifetime<T>(ptr + offset, LDAB * n), LDAB * n);
        offset += LDAB * n * sizeof(T);

        // lapack_storage - start lifetime of T[LDAB*n]
        offset = align_up(offset, block_alignment_T);
        ws.lapack_storage_ = std::span<T>(
            start_array_lifetime<T>(ptr + offset, LDAB * n), LDAB * n);
        offset += LDAB * n * sizeof(T);

        // pivots - start lifetime of int[n]
        offset = align_up(offset, block_alignment_int);
        ws.pivots_ = std::span<int>(
            start_array_lifetime<int>(ptr + offset, n), n);
        offset += n * sizeof(int);

        // coeffs - start lifetime of T[n]
        offset = align_up(offset, block_alignment_T);
        ws.coeffs_ = std::span<T>(
            start_array_lifetime<T>(ptr + offset, n), n);

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
- `std::start_lifetime_as_array<T>(ptr, n)` (C++23) starts object lifetimes
- **Important:** `std::launder` is NOT sufficient - it only provides pointer provenance but does NOT create objects. `std::start_lifetime_as_array` actually starts the lifetime.
- All arrays are 64-byte aligned via `block_alignment_T` / `block_alignment_int`
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

### BSplineCollocation1D API Hierarchy

The three `fit*` methods form a progression of allocation control:

| Method | Allocates | User Provides | Use Case |
|--------|-----------|---------------|----------|
| `fit()` | BandedMatrix + BandedLUWorkspace + coeffs | Nothing | Simple single-fit API |
| `fit_with_buffer()` | BandedMatrix + BandedLUWorkspace | coeffs span | Caller manages output only |
| `fit_with_workspace()` | Nothing | BSplineCollocationWorkspace | Zero-allocation hot path |

**Implementation relationship:**
```cpp
// fit() - highest level, allocates everything
auto result = solver.fit(values, config);
// Internally: creates BandedMatrix, BandedLUWorkspace, std::vector<T>

// fit_with_buffer() - mid level, caller provides coeffs output
std::vector<T> coeffs(n);
auto result = solver.fit_with_buffer(values, coeffs, config);
// Internally: creates BandedMatrix, BandedLUWorkspace; writes to coeffs

// fit_with_workspace() - lowest level, zero allocations (NEW)
auto ws = BSplineCollocationWorkspace<T>::from_bytes(buffer.bytes(), n);
auto result = solver.fit_with_workspace(values, ws.value(), config);
// Internally: uses ws.band_storage(), ws.lapack_storage(), ws.pivots()
// Writes coefficients to ws.coeffs()
```

**Migration guidance:**
- **Keep `fit()`** for simple single-option use cases
- **Keep `fit_with_buffer()`** for compatibility (deprecation NOT recommended)
- **Use `fit_with_workspace()`** in parallel hot paths (e.g., BSplineNDSeparable)

**Why not deprecate `fit_with_buffer()`?**
1. Lower barrier to entry—user only needs to allocate coefficients, not understand workspace layout
2. Adequate for moderate parallelism where workspace overhead is acceptable
3. Existing callers shouldn't be forced to migrate

### 4. Modified BSplineNDSeparable

Use ThreadWorkspaceBuffer and parallelize slice iteration.

```cpp
template<size_t Axis>
[[nodiscard]] std::expected<void, InterpolationError> fit_axis(
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

    // Track workspace creation errors (atomic for thread safety)
    std::atomic<bool> workspace_error{false};

    MANGO_PRAGMA_PARALLEL
    {
        // Per-thread workspace buffer (bytes, not T)
        ThreadWorkspaceBuffer buffer(ws_bytes);

        // Per-thread slice buffer
        std::vector<T> slice_buffer(n_axis);

        // Thread-local statistics
        T local_max_residual = T{0};
        T local_max_condition = T{0};
        size_t local_failed = 0;
        bool local_ws_error = false;

        // Create workspace ONCE per thread (outside the loop)
        // Object lifetimes are started here and remain valid for all iterations.
        // The workspace's spans point into buffer.bytes() which is stable.
        auto ws_result = BSplineCollocationWorkspace<T>::from_bytes(buffer.bytes(), n_axis);
        if (!ws_result.has_value()) {
            workspace_error.store(true, std::memory_order_relaxed);
            local_ws_error = true;
        }

        // Only enter loop if workspace was created successfully
        if (!local_ws_error) {
            auto& ws = ws_result.value();

            MANGO_PRAGMA_FOR_STATIC
            for (size_t slice_idx = 0; slice_idx < n_slices; ++slice_idx) {
                // Early exit if another thread hit a workspace error
                if (workspace_error.load(std::memory_order_relaxed)) {
                    break;  // Exit loop immediately, don't inflate failure count
                }

                // Extract slice into per-thread buffer
                extract_slice<Axis>(coeffs, slice_idx, slice_buffer);

                // Fit with workspace (zero allocations)
                // Note: solver is accessed via optional dereference (static dispatch)
                // Workspace spans are reused - solver overwrites them each iteration.
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
        }

        // Reduce thread-local statistics
        MANGO_PRAGMA_CRITICAL
        {
            global_max_residual = std::max(global_max_residual, local_max_residual);
            global_max_condition = std::max(global_max_condition, local_max_condition);
            global_failed += local_failed;
        }
    }

    // Always write statistics (even on error, so callers see partial progress)
    max_residuals[Axis] = global_max_residual;
    conditions[Axis] = global_max_condition;
    failed[Axis] = global_failed;

    // Return error if any workspace creation failed
    if (workspace_error.load()) {
        return std::unexpected(InterpolationError::AllocationFailed);
    }

    return {};  // Success
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
BSplineCollocationWorkspace::from_bytes() slices with 64-byte alignment:
┌────────────┬───────┬────────────┬───────┬──────────┬───────┬──────────┬───────┐
│ band_stor. │ pad64 │ lapack_st. │ pad64 │ pivots   │ pad64 │ coeffs   │ pad64 │
│ T[10n]     │       │ T[10n]     │       │ int[n]   │       │ T[n]     │       │
└────────────┴───────┴────────────┴───────┴──────────┴───────┴──────────┴───────┘
       ↑                 ↑                     ↑                  ↑
    64-byte           64-byte              64-byte            64-byte
    aligned           aligned              aligned            aligned
```

**Key points:**
- Base storage is `std::byte` (can alias any type per standard)
- `from_bytes()` uses `std::start_lifetime_as_array<T>()` to start object lifetimes
- **Every segment is 64-byte aligned** for cache line / AVX-512 optimization
- Pivots are stored as `int[]` directly (not aliased through `T`)
- `std::launder` is NOT sufficient - must use `std::start_lifetime_as_array` (C++23)

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

#### Pattern 3: Parallel Region with Per-Thread Workspace

Existing per-thread workspace patterns in the codebase.

| File | Lines | Construct | Status |
|------|-------|-----------|--------|
| `american_option_batch.cpp` | 466-604 | `MANGO_PRAGMA_PARALLEL` + `MANGO_PRAGMA_FOR_STATIC` | **OUT OF SCOPE** - existing PMR pattern works, N allocations acceptable |
| `bspline_nd_separable.hpp` | (new) | `MANGO_PRAGMA_PARALLEL` + `MANGO_PRAGMA_FOR_STATIC` | **PRIMARY TARGET** - 24,000 allocations → N allocations |

**american_option_batch.cpp (OUT OF SCOPE):**

The existing code uses per-thread `monotonic_buffer_resource` with `pmr::vector<double>`.
This works correctly and has acceptable performance (N allocations where N = batch size).
Migrating to ThreadWorkspaceBuffer would require adding `PDEWorkspace::from_bytes()`,
which is not justified given the low allocation count.

**bspline_nd_separable.hpp (PRIMARY TARGET):**

This is where ThreadWorkspaceBuffer provides the most value. Each B-spline fit currently
allocates BandedMatrix + BandedLUWorkspace. For a 4D price table, that's 24,000 allocations.
With ThreadWorkspaceBuffer + BSplineCollocationWorkspace, this drops to N (thread count).

```cpp
// PRIMARY TARGET: BSplineNDSeparable::fit_axis()
// See "4. Modified BSplineNDSeparable" section for full implementation

MANGO_PRAGMA_PARALLEL
{
    ThreadWorkspaceBuffer buffer(BSplineCollocationWorkspace<T>::required_bytes(n_axis));

    MANGO_PRAGMA_FOR_STATIC
    for (size_t slice_idx = 0; slice_idx < n_slices; ++slice_idx) {
        auto ws = BSplineCollocationWorkspace<T>::from_bytes(buffer.bytes(), n_axis);
        // Zero allocations in hot path
        solver.fit_with_workspace(slice_buffer, ws.value(), config);
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
| `bspline_nd_separable.hpp` | New parallelization | **Yes** | P0 (primary target) |
| `american_option_batch.cpp` | Per-thread workspace | No | - (existing PMR pattern works) |
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

### Phase 0: Parallel Primitives (P0)

Land the foundational components that all subsequent phases depend on:

1. **ThreadWorkspaceBuffer** (`src/support/thread_workspace.hpp`)
   - Non-templated RAII byte buffer with PMR fallback
   - Unit tests for allocation, fallback, thread safety, span stability

2. **Lifetime helpers** (`src/support/lifetime.hpp` or inline in thread_workspace.hpp)
   - `start_array_lifetime<T>(void*, size_t)` with C++23/fallback paths
   - Alignment utilities (`block_alignment<T>`, etc.)

3. **OpenMP macros** (`src/support/parallel.hpp`)
   - `MANGO_PRAGMA_CRITICAL` for thread-safe reductions

### Phase 1: B-Spline Collocation (P0)

Migrate the primary allocation hotspot (24,000 → N allocations):

1. **BSplineCollocationWorkspace<T>** (`src/math/bspline_collocation_workspace.hpp`)
   - `required_bytes(n)` static method
   - `from_bytes(span<byte>, n)` factory returning `std::expected`
   - Typed spans for coeffs, matrix, LU workspace

2. **BSplineCollocation1D** updates
   - Add `fit_with_workspace(span<const T>, BSplineCollocationWorkspace<T>&)`
   - Keep existing `fit()` and `fit_with_buffer()` for backward compatibility

3. **BSplineNDSeparable** updates
   - Use ThreadWorkspaceBuffer for per-thread workspace
   - Parallelize slice iteration with `MANGO_PRAGMA_PARALLEL` + `MANGO_PRAGMA_FOR_STATIC`
   - Thread-local statistics with critical section reduction

### Phase 2: American Batch PDE Workspace (P1)

Add byte-buffer interface to existing PDE workspace:

1. **AmericanPDEWorkspace** (new or extend `PDEWorkspace`)
   - `required_bytes(grid_spec)` static method
   - `from_bytes(span<byte>, grid_spec)` factory
   - Owns Grid, solution vectors, LU workspace as typed spans over byte buffer

2. **Unit tests**
   - Verify workspace correctly slices byte buffer
   - Compare results against existing PDEWorkspace::create()

### Phase 3: American Batch Solver Refactor (P1)

Migrate `american_option_batch.cpp` to use the new workspace:

1. **Replace per-thread PMR pools** with ThreadWorkspaceBuffer
   - Currently: `std::pmr::synchronized_pool_resource` per thread
   - After: `ThreadWorkspaceBuffer buffer(AmericanPDEWorkspace::required_bytes(grid_spec))`

2. **Use AmericanPDEWorkspace::from_bytes()**
   - Create workspace once per thread (outside loop)
   - Reuse across all options in batch

3. **Verify correctness**
   - Compare batch results before/after refactor
   - Benchmark allocation reduction (N pools → N buffers, but simpler allocation pattern)

### Phase 4: Optional Consumers (P2)

Lower-priority migrations for completeness:

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
