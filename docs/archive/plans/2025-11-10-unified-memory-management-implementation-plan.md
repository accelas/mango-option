# Unified Memory Management C++23 Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor workspace memory management to use std::pmr, full SoA layout, and std::experimental::simd with ISA dispatch for 3-8x performance improvement.

**Architecture:** Four-layer design: (1) UnifiedMemoryResource using std::pmr::monotonic_buffer_resource, (2) WorkspaceBase + PDEWorkspace with SoA layout and SIMD padding, (3) CenteredDifferenceSIMD operators with [[gnu::target_clones]], (4) CPU feature detection with XGETBV validation.

**Tech Stack:** C++23, std::pmr, std::experimental::simd, GCC/Clang target_clones, Bazel, GoogleTest

**Design Reference:** `docs/plans/2025-11-10-unified-memory-management-c++23-refactor.md`

---

## Phase 1: Core Memory Management

### Task 1: Create UnifiedMemoryResource header

**Files:**
- Create: `src/pde/memory/unified_memory_resource.hpp`
- Test: `tests/memory/unified_memory_resource_test.cc`
- Modify: `src/BUILD.bazel` (add memory library)
- Modify: `tests/BUILD.bazel` (add memory tests)

**Step 1: Write failing test for basic allocation**

Create `tests/memory/unified_memory_resource_test.cc`:

```cpp
#include "src/pde/memory/unified_memory_resource.hpp"
#include <gtest/gtest.h>

TEST(UnifiedMemoryResourceTest, BasicAllocation) {
    mango::memory::UnifiedMemoryResource resource(1024);

    void* ptr = resource.allocate(64, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);  // 64-byte aligned
    EXPECT_EQ(resource.bytes_allocated(), 64);
}

TEST(UnifiedMemoryResourceTest, MultipleAllocations) {
    mango::memory::UnifiedMemoryResource resource(1024);

    void* ptr1 = resource.allocate(32, 64);
    void* ptr2 = resource.allocate(32, 64);

    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2);
    EXPECT_EQ(resource.bytes_allocated(), 64);
}

TEST(UnifiedMemoryResourceTest, ResetClearsMemory) {
    mango::memory::UnifiedMemoryResource resource(1024);

    resource.allocate(128, 64);
    EXPECT_EQ(resource.bytes_allocated(), 128);

    resource.reset();
    EXPECT_EQ(resource.bytes_allocated(), 0);

    // Can allocate again after reset
    void* ptr = resource.allocate(64, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(resource.bytes_allocated(), 64);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests/memory:unified_memory_resource_test
```

Expected: Build fails with "No such file or directory: src/pde/memory/unified_memory_resource.hpp"

**Step 3: Create UnifiedMemoryResource implementation**

Create `src/pde/memory/unified_memory_resource.hpp`:

```cpp
#pragma once

#include <memory_resource>
#include <cstddef>

namespace mango::memory {

/**
 * RAII wrapper around std::pmr::monotonic_buffer_resource
 *
 * Provides workspace-owned allocator with:
 * - Zero-cost reset() between solves
 * - 64-byte default alignment for AVX-512
 * - Manual bytes_allocated() tracking
 *
 * Thread-safe: each workspace owns one instance (no shared state)
 */
class UnifiedMemoryResource {
public:
    explicit UnifiedMemoryResource(size_t initial_buffer_size = 1024 * 1024)
        : upstream_(std::pmr::get_default_resource())
        , monotonic_(initial_buffer_size, upstream_)
        , bytes_allocated_(0)
    {}

    /**
     * Allocate memory with specified alignment
     *
     * @param bytes Size in bytes
     * @param alignment Alignment requirement (default: 64 for AVX-512)
     * @return Pointer to allocated memory
     */
    void* allocate(size_t bytes, size_t alignment = 64) {
        void* ptr = monotonic_.allocate(bytes, alignment);
        bytes_allocated_ += bytes;
        return ptr;
    }

    /**
     * Reset for reuse (zero-cost between solves)
     *
     * WARNING: Invalidates all previously allocated pointers!
     */
    void reset() {
        monotonic_.release();
        bytes_allocated_ = 0;
    }

    /// Query total bytes allocated
    size_t bytes_allocated() const { return bytes_allocated_; }

private:
    std::pmr::memory_resource* upstream_;
    std::pmr::monotonic_buffer_resource monotonic_;
    size_t bytes_allocated_;  // Manual tracking (PMR doesn't expose this)
};

} // namespace mango::memory
```

**Step 4: Update BUILD.bazel files**

Modify `src/BUILD.bazel` - add after existing cc_library targets:

```python
cc_library(
    name = "unified_memory_resource",
    hdrs = ["memory/unified_memory_resource.hpp"],
    visibility = ["//visibility:public"],
    deps = [],
)
```

Modify `tests/BUILD.bazel` - add after existing cc_test targets:

```python
cc_test(
    name = "unified_memory_resource_test",
    srcs = ["memory/unified_memory_resource_test.cc"],
    deps = [
        "//src:unified_memory_resource",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test to verify it passes**

```bash
bazel test //tests/memory:unified_memory_resource_test --test_output=all
```

Expected: All tests PASS

**Step 6: Commit**

```bash
mkdir -p src/memory tests/memory
git add src/pde/memory/unified_memory_resource.hpp
git add tests/memory/unified_memory_resource_test.cc
git add src/BUILD.bazel tests/BUILD.bazel
git commit -m "feat: add UnifiedMemoryResource with std::pmr

RAII wrapper around std::pmr::monotonic_buffer_resource for workspace
memory management. Provides zero-cost reset() and 64-byte alignment.

- Implements allocate() with manual bytes_allocated() tracking
- Thread-safe via workspace-owned instances
- Tests verify alignment, multiple allocations, and reset

Part of unified memory management refactor (Phase 1/5)."
```

---

### Task 2: Create WorkspaceBase with tiling infrastructure

**Files:**
- Create: `src/pde/memory/workspace_base.hpp`
- Test: `tests/memory/workspace_base_test.cc`
- Modify: `src/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing test for tiling metadata**

Create `tests/memory/workspace_base_test.cc`:

```cpp
#include "src/pde/memory/workspace_base.hpp"
#include <gtest/gtest.h>

TEST(WorkspaceBaseTest, TileMetadataGeneration) {
    // Divide 100 elements into 3 tiles: 34, 33, 33
    auto tile0 = mango::WorkspaceBase::tile_info(100, 0, 3);
    EXPECT_EQ(tile0.tile_start, 0);
    EXPECT_EQ(tile0.tile_size, 34);
    EXPECT_EQ(tile0.padded_size, 40);  // Rounded to SIMD_WIDTH=8
    EXPECT_EQ(tile0.alignment, 64);

    auto tile1 = mango::WorkspaceBase::tile_info(100, 1, 3);
    EXPECT_EQ(tile1.tile_start, 34);
    EXPECT_EQ(tile1.tile_size, 33);
    EXPECT_EQ(tile1.padded_size, 40);

    auto tile2 = mango::WorkspaceBase::tile_info(100, 2, 3);
    EXPECT_EQ(tile2.tile_start, 67);
    EXPECT_EQ(tile2.tile_size, 33);
    EXPECT_EQ(tile2.padded_size, 40);
}

TEST(WorkspaceBaseTest, SIMDPadding) {
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(1), 8);
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(8), 8);
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(9), 16);
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(16), 16);
    EXPECT_EQ(mango::WorkspaceBase::pad_to_simd(17), 24);
}

TEST(WorkspaceBaseTest, TileInfoBoundsChecking) {
    // Debug mode should assert on invalid inputs
    #ifndef NDEBUG
    EXPECT_DEATH(mango::WorkspaceBase::tile_info(100, 0, 0), "num_tiles must be positive");
    EXPECT_DEATH(mango::WorkspaceBase::tile_info(100, 5, 3), "tile_idx out of bounds");
    #endif
}

TEST(WorkspaceBaseTest, BytesAllocatedTracking) {
    mango::WorkspaceBase workspace(1024);
    EXPECT_EQ(workspace.bytes_allocated(), 0);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests/memory:workspace_base_test
```

Expected: Build fails with "No such file or directory: src/pde/memory/workspace_base.hpp"

**Step 3: Create WorkspaceBase implementation**

Create `src/pde/memory/workspace_base.hpp`:

```cpp
#pragma once

#include "unified_memory_resource.hpp"
#include <cassert>
#include <cstddef>
#include <algorithm>

namespace mango {

/// Tile metadata for operator-level tiling
struct TileMetadata {
    size_t tile_start;      ///< Start index in original grid
    size_t tile_size;       ///< Actual elements (not padded)
    size_t padded_size;     ///< Rounded to SIMD_WIDTH
    size_t alignment;       ///< Byte alignment
};

/**
 * WorkspaceBase: Base class providing allocator and tiling infrastructure
 *
 * Provides reusable functionality for all workspace types:
 * - UnifiedMemoryResource allocator
 * - Tiling metadata generation
 * - SIMD padding utilities
 */
class WorkspaceBase {
public:
    explicit WorkspaceBase(size_t initial_buffer_size = 1024 * 1024)
        : resource_(initial_buffer_size)
    {}

    /**
     * Generate tile metadata for operator-level tiling
     *
     * Distributes n elements across num_tiles, with remainder spread
     * across first tiles. All tiles SIMD-padded.
     *
     * @param n Total number of elements
     * @param tile_idx Index of this tile [0, num_tiles)
     * @param num_tiles Total number of tiles
     * @return TileMetadata for this tile
     */
    static TileMetadata tile_info(size_t n, size_t tile_idx, size_t num_tiles) {
        assert(num_tiles > 0 && "num_tiles must be positive");
        assert(tile_idx < num_tiles && "tile_idx out of bounds");

        const size_t base_tile_size = n / num_tiles;
        const size_t remainder = n % num_tiles;
        const size_t tile_size = base_tile_size + (tile_idx < remainder ? 1 : 0);
        const size_t tile_start = tile_idx * base_tile_size + std::min(tile_idx, remainder);
        const size_t padded_size = pad_to_simd(tile_size);

        return {tile_start, tile_size, padded_size, 64};
    }

    /// AVX-512: 8 doubles per vector
    static constexpr size_t SIMD_WIDTH = 8;

    /// Round up to SIMD_WIDTH boundary
    static constexpr size_t pad_to_simd(size_t n) {
        return ((n + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
    }

    /// Query total bytes allocated
    size_t bytes_allocated() const { return resource_.bytes_allocated(); }

protected:
    memory::UnifiedMemoryResource resource_;
};

} // namespace mango
```

**Step 4: Update BUILD.bazel**

Modify `src/BUILD.bazel` - add library:

```python
cc_library(
    name = "workspace_base",
    hdrs = ["memory/workspace_base.hpp"],
    visibility = ["//visibility:public"],
    deps = [":unified_memory_resource"],
)
```

Modify `tests/BUILD.bazel` - add test:

```python
cc_test(
    name = "workspace_base_test",
    srcs = ["memory/workspace_base_test.cc"],
    deps = [
        "//src:workspace_base",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test to verify it passes**

```bash
bazel test //tests/memory:workspace_base_test --test_output=all
```

Expected: All tests PASS (tile_info bounds checking only in debug builds)

**Step 6: Commit**

```bash
git add src/pde/memory/workspace_base.hpp
git add tests/memory/workspace_base_test.cc
git add src/BUILD.bazel tests/BUILD.bazel
git commit -m "feat: add WorkspaceBase with tiling infrastructure

Base class for all workspace types providing:
- Tile metadata generation for operator-level tiling
- SIMD padding utilities (AVX-512: 8 doubles)
- UnifiedMemoryResource allocator access

Tests verify tiling math, SIMD padding, and bounds checking.

Part of unified memory management refactor (Phase 1/5)."
```

---

## Phase 2: PDEWorkspace with SoA Layout

### Task 3: Create PDEWorkspace header

**Files:**
- Create: `src/pde/memory/pde_workspace.hpp`
- Test: `tests/memory/pde_workspace_test.cc`
- Modify: `src/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing test for SoA allocation**

Create `tests/memory/pde_workspace_test.cc`:

```cpp
#include "src/pde/memory/pde_workspace.hpp"
#include "src/grid.hpp"
#include <gtest/gtest.h>
#include <algorithm>

TEST(PDEWorkspaceTest, BasicConstruction) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::PDEWorkspace workspace(101, grid.span());

    EXPECT_EQ(workspace.logical_size(), 101);
    EXPECT_EQ(workspace.padded_size(), 104);  // Rounded to 8

    // All array accessors should return valid spans
    EXPECT_EQ(workspace.u_current().size(), 101);
    EXPECT_EQ(workspace.u_next().size(), 101);
    EXPECT_EQ(workspace.u_stage().size(), 101);
    EXPECT_EQ(workspace.rhs().size(), 101);
    EXPECT_EQ(workspace.lu().size(), 101);
    EXPECT_EQ(workspace.psi_buffer().size(), 101);
}

TEST(PDEWorkspaceTest, PaddedAccessors) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::PDEWorkspace workspace(101, grid.span());

    EXPECT_EQ(workspace.u_current_padded().size(), 104);
    EXPECT_EQ(workspace.u_next_padded().size(), 104);
    EXPECT_EQ(workspace.lu_padded().size(), 104);

    // Padding should be zero-initialized
    auto u_padded = workspace.u_current_padded();
    EXPECT_DOUBLE_EQ(u_padded[101], 0.0);
    EXPECT_DOUBLE_EQ(u_padded[102], 0.0);
    EXPECT_DOUBLE_EQ(u_padded[103], 0.0);
}

TEST(PDEWorkspaceTest, GridSpacing) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 10.0, 6);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();  // 0, 2, 4, 6, 8, 10

    mango::PDEWorkspace workspace(6, grid.span());

    auto dx = workspace.dx();
    EXPECT_EQ(dx.size(), 5);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(dx[i], 2.0);
    }

    // Padded dx (size 5 → 8)
    auto dx_padded = workspace.dx_padded();
    EXPECT_EQ(dx_padded.size(), 8);
    EXPECT_DOUBLE_EQ(dx_padded[5], 0.0);  // Zero-padded tail
}

TEST(PDEWorkspaceTest, ArraysAreIndependent) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 10);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::PDEWorkspace workspace(10, grid.span());

    // Write to one array
    auto u = workspace.u_current();
    std::fill(u.begin(), u.end(), 1.0);

    // Other arrays should remain zero
    auto v = workspace.u_next();
    EXPECT_DOUBLE_EQ(v[0], 0.0);
}

TEST(PDEWorkspaceTest, ResetInvalidatesSpans) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 10);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::PDEWorkspace workspace(10, grid.span());

    auto u_before = workspace.u_current();
    u_before[0] = 999.0;

    workspace.reset();

    // Must re-acquire span after reset
    auto u_after = workspace.u_current();
    EXPECT_DOUBLE_EQ(u_after[0], 0.0);  // Freshly allocated
}

TEST(PDEWorkspaceTest, TileMetadata) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 100);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::PDEWorkspace workspace(100, grid.span());

    // 100 elements into 3 tiles: 34, 33, 33
    auto tile0 = workspace.tile_info(0, 3);
    EXPECT_EQ(tile0.tile_start, 0);
    EXPECT_EQ(tile0.tile_size, 34);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests/memory:pde_workspace_test
```

Expected: Build fails with "No such file or directory: src/pde/memory/pde_workspace.hpp"

**Step 3: Create PDEWorkspace implementation**

Create `src/pde/memory/pde_workspace.hpp`:

```cpp
#pragma once

#include "workspace_base.hpp"
#include <span>
#include <cassert>
#include <algorithm>

namespace mango {

/**
 * PDEWorkspace: workspace for PDE solver with SoA layout
 *
 * Full Structure-of-Arrays layout for SIMD-friendly access:
 * - Each state array separate and SIMD-padded
 * - Zero-initialized padding for safe tail processing
 * - Dual accessors: logical size and padded size
 *
 * LIFETIME REQUIREMENTS:
 * - The `grid` span passed to constructor must remain valid for the lifetime
 *   of this workspace (stored for reset() reinit).
 *
 * INVALIDATION WARNING:
 * - reset() invalidates all previously returned std::span objects.
 * - After reset(), caller MUST re-acquire spans via accessors.
 */
class PDEWorkspace : public WorkspaceBase {
public:
    explicit PDEWorkspace(size_t n, std::span<const double> grid,
                         size_t initial_buffer_size = 1024 * 1024)
        : WorkspaceBase(initial_buffer_size)
        , n_(n)
        , padded_n_(pad_to_simd(n))
        , grid_(grid)
    {
        assert(!grid.empty() && "grid must not be empty");
        assert(grid.size() == n && "grid size must match n");
        allocate_and_initialize();
    }

    // SoA array accessors (logical size)
    std::span<double> u_current() { return {u_current_, n_}; }
    std::span<const double> u_current() const { return {u_current_, n_}; }

    std::span<double> u_next() { return {u_next_, n_}; }
    std::span<const double> u_next() const { return {u_next_, n_}; }

    std::span<double> u_stage() { return {u_stage_, n_}; }
    std::span<const double> u_stage() const { return {u_stage_, n_}; }

    std::span<double> rhs() { return {rhs_, n_}; }
    std::span<const double> rhs() const { return {rhs_, n_}; }

    std::span<double> lu() { return {lu_, n_}; }
    std::span<const double> lu() const { return {lu_, n_}; }

    std::span<double> psi_buffer() { return {psi_, n_}; }
    std::span<const double> psi_buffer() const { return {psi_, n_}; }

    // Padded accessors for SIMD kernels
    std::span<double> u_current_padded() { return {u_current_, padded_n_}; }
    std::span<const double> u_current_padded() const { return {u_current_, padded_n_}; }

    std::span<double> u_next_padded() { return {u_next_, padded_n_}; }
    std::span<const double> u_next_padded() const { return {u_next_, padded_n_}; }

    std::span<double> lu_padded() { return {lu_, padded_n_}; }
    std::span<const double> lu_padded() const { return {lu_, padded_n_}; }

    // Grid spacing (SIMD-padded, zero-filled tail)
    std::span<const double> dx() const { return {dx_, n_ - 1}; }
    std::span<const double> dx_padded() const { return {dx_, pad_to_simd(n_ - 1)}; }

    /// Tile metadata for this workspace
    TileMetadata tile_info(size_t tile_idx, size_t num_tiles) const {
        return WorkspaceBase::tile_info(n_, tile_idx, num_tiles);
    }

    /**
     * Reset and reinitialize
     * WARNING: Invalidates all previously returned spans!
     */
    void reset() {
        resource_.reset();
        allocate_and_initialize();
    }

    size_t logical_size() const { return n_; }
    size_t padded_size() const { return padded_n_; }

private:
    void allocate_and_initialize() {
        allocate_arrays();
        precompute_grid_spacing();
    }

    void allocate_arrays() {
        const size_t array_bytes = padded_n_ * sizeof(double);
        u_current_ = static_cast<double*>(resource_.allocate(array_bytes));
        u_next_    = static_cast<double*>(resource_.allocate(array_bytes));
        u_stage_   = static_cast<double*>(resource_.allocate(array_bytes));
        rhs_       = static_cast<double*>(resource_.allocate(array_bytes));
        lu_        = static_cast<double*>(resource_.allocate(array_bytes));
        psi_       = static_cast<double*>(resource_.allocate(array_bytes));

        // Zero-initialize entire buffers (including padding)
        std::fill(u_current_, u_current_ + padded_n_, 0.0);
        std::fill(u_next_, u_next_ + padded_n_, 0.0);
        std::fill(u_stage_, u_stage_ + padded_n_, 0.0);
        std::fill(rhs_, rhs_ + padded_n_, 0.0);
        std::fill(lu_, lu_ + padded_n_, 0.0);
        std::fill(psi_, psi_ + padded_n_, 0.0);
    }

    void precompute_grid_spacing() {
        const size_t dx_padded = pad_to_simd(n_ - 1);
        const size_t dx_bytes = dx_padded * sizeof(double);
        dx_ = static_cast<double*>(resource_.allocate(dx_bytes));

        for (size_t i = 0; i < n_ - 1; ++i) {
            dx_[i] = grid_[i + 1] - grid_[i];
        }
        // Zero padding for safe SIMD tail
        std::fill(dx_ + (n_ - 1), dx_ + dx_padded, 0.0);
    }

    size_t n_;
    size_t padded_n_;
    std::span<const double> grid_;  // Caller must keep alive!

    // SoA arrays (separate, SIMD-aligned)
    double* u_current_;
    double* u_next_;
    double* u_stage_;
    double* rhs_;
    double* lu_;
    double* psi_;
    double* dx_;
};

} // namespace mango
```

**Step 4: Update BUILD.bazel**

Modify `src/BUILD.bazel`:

```python
cc_library(
    name = "pde_workspace",
    hdrs = ["memory/pde_workspace.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":workspace_base",
    ],
)
```

Modify `tests/BUILD.bazel`:

```python
cc_test(
    name = "pde_workspace_test",
    srcs = ["memory/pde_workspace_test.cc"],
    deps = [
        "//src:pde_workspace",
        "//src:grid",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test to verify it passes**

```bash
bazel test //tests/memory:pde_workspace_test --test_output=all
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/pde/memory/pde_workspace.hpp
git add tests/memory/pde_workspace_test.cc
git add src/BUILD.bazel tests/BUILD.bazel
git commit -m "feat: add PDEWorkspace with full SoA layout

PDE-specific workspace with Structure-of-Arrays layout:
- 6 separate SIMD-padded arrays (u_current, u_next, u_stage, rhs, lu, psi)
- Dual accessors: logical size and padded size for SIMD kernels
- Precomputed grid spacing (dx) with zero-padded tail
- Reset support with span invalidation

Tests verify SoA allocation, padding, grid spacing, array independence,
and reset behavior.

Part of unified memory management refactor (Phase 2/5)."
```

---

## Phase 3: SIMD Operators with ISA Dispatch

### Task 4: Create CPU feature detection

**Files:**
- Create: `src/support/cpu/feature_detection.hpp`
- Test: `tests/cpu/feature_detection_test.cc`
- Modify: `src/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write test for CPU feature detection**

Create `tests/cpu/feature_detection_test.cc`:

```cpp
#include "src/support/cpu/feature_detection.hpp"
#include <gtest/gtest.h>
#include <iostream>

TEST(CPUFeatureDetectionTest, DetectFeatures) {
    auto features = mango::cpu::detect_cpu_features();

    // x86-64 baseline guarantees SSE2
    EXPECT_TRUE(features.has_sse2);

    // Print detected features for diagnostic
    std::cout << "SSE2: " << features.has_sse2 << "\n";
    std::cout << "AVX2: " << features.has_avx2 << "\n";
    std::cout << "AVX512F: " << features.has_avx512f << "\n";
    std::cout << "FMA: " << features.has_fma << "\n";
}

TEST(CPUFeatureDetectionTest, SelectISATarget) {
    auto target = mango::cpu::select_isa_target();
    auto name = mango::cpu::isa_target_name(target);

    std::cout << "Selected ISA: " << name << "\n";

    // Should return one of the known targets
    EXPECT_TRUE(target == mango::cpu::ISATarget::DEFAULT ||
                target == mango::cpu::ISATarget::AVX2 ||
                target == mango::cpu::ISATarget::AVX512F);
}

TEST(CPUFeatureDetectionTest, ISATargetNames) {
    EXPECT_EQ(mango::cpu::isa_target_name(mango::cpu::ISATarget::DEFAULT), "SSE2");
    EXPECT_EQ(mango::cpu::isa_target_name(mango::cpu::ISATarget::AVX2), "AVX2+FMA");
    EXPECT_EQ(mango::cpu::isa_target_name(mango::cpu::ISATarget::AVX512F), "AVX512F");
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests/cpu:feature_detection_test
```

Expected: Build fails

**Step 3: Create CPU feature detection implementation**

Create `src/support/cpu/feature_detection.hpp`:

```cpp
#pragma once

#include <cpuid.h>
#include <string>
#include <iostream>
#include <immintrin.h>

namespace mango::cpu {

/// CPU feature flags detected at runtime
struct CPUFeatures {
    bool has_sse2 = false;
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_fma = false;
};

/**
 * ISA target enum for dispatch
 *
 * NOTE: This is DIAGNOSTIC ONLY. Actual dispatch happens via
 * [[gnu::target_clones]] IFUNC resolution at link time.
 */
enum class ISATarget {
    DEFAULT,   // SSE2 baseline
    AVX2,      // Haswell+ (2013+)
    AVX512F    // Skylake-X+ (2017+)
};

/**
 * Check if OS has enabled xsave for AVX/AVX-512 state
 *
 * AVX/AVX-512 require OS support for YMM/ZMM register state.
 * Without this check, CPUID may report AVX support but executing
 * AVX instructions will SIGILL.
 */
inline bool check_os_avx_support() {
    unsigned int eax, ebx, ecx, edx;

    // Check OSXSAVE bit (OS has enabled XSAVE)
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return false;
    }

    if ((ecx & bit_OSXSAVE) == 0) {
        return false;  // OS hasn't enabled xsave
    }

    // Check XCR0 register via XGETBV
    // XCR0[1] = SSE state, XCR0[2] = YMM state
    unsigned long long xcr0 = _xgetbv(0);

    // AVX requires SSE (bit 1) and YMM (bit 2)
    constexpr unsigned long long AVX_MASK = (1ULL << 1) | (1ULL << 2);

    return (xcr0 & AVX_MASK) == AVX_MASK;
}

/// Check if OS has enabled AVX-512 state
inline bool check_os_avx512_support() {
    unsigned long long xcr0 = _xgetbv(0);
    // AVX-512 requires SSE, YMM, and ZMM state (bits 5, 6, 7)
    constexpr unsigned long long AVX512_MASK = (1ULL << 1) | (1ULL << 2) |
                                               (1ULL << 5) | (1ULL << 6) | (1ULL << 7);
    return (xcr0 & AVX512_MASK) == AVX512_MASK;
}

/**
 * Detect CPU features once at program startup
 *
 * Uses CPUID instruction to query supported ISA extensions,
 * with OS support validation via XGETBV.
 */
inline CPUFeatures detect_cpu_features() {
    CPUFeatures features;

    unsigned int eax, ebx, ecx, edx;

    // Check for SSE2 (standard in x86-64)
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        features.has_sse2 = (edx & bit_SSE2) != 0;
        features.has_fma = (ecx & bit_FMA) != 0;
    }

    // Check OS support for AVX/AVX-512 state
    bool os_avx_support = check_os_avx_support();
    bool os_avx512_support = os_avx_support && check_os_avx512_support();

    // Check for AVX2 (requires OS support)
    if (os_avx_support && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        features.has_avx2 = (ebx & bit_AVX2) != 0;

        // Emit diagnostic if FMA is missing (AVX2 CPUs typically have it)
        if (features.has_avx2 && !features.has_fma) {
            #ifndef NDEBUG
            std::cerr << "Warning: AVX2 detected but FMA not available\n";
            #endif
        }
    }

    // Check for AVX-512 (requires OS support)
    if (os_avx512_support && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        features.has_avx512f = (ebx & bit_AVX512F) != 0;
    }

    return features;
}

/**
 * Select best ISA target for current CPU
 *
 * Called once at solver construction, result cached in solver.
 *
 * NOTE: This is DIAGNOSTIC ONLY for logging/stats. The actual kernel
 * dispatch happens automatically via [[gnu::target_clones]] IFUNC
 * resolution at runtime. This function merely reports what the IFUNC
 * resolver will choose.
 */
inline ISATarget select_isa_target() {
    static const CPUFeatures features = detect_cpu_features();

    if (features.has_avx512f) {
        return ISATarget::AVX512F;
    } else if (features.has_avx2 && features.has_fma) {
        return ISATarget::AVX2;
    } else {
        return ISATarget::DEFAULT;
    }
}

/// Get human-readable ISA target name (for logging/diagnostics)
inline std::string isa_target_name(ISATarget target) {
    switch (target) {
        case ISATarget::DEFAULT: return "SSE2";
        case ISATarget::AVX2: return "AVX2+FMA";
        case ISATarget::AVX512F: return "AVX512F";
        default: return "UNKNOWN";
    }
}

} // namespace mango::cpu
```

**Step 4: Update BUILD.bazel**

Modify `src/BUILD.bazel`:

```python
cc_library(
    name = "cpu_features",
    hdrs = ["cpu/feature_detection.hpp"],
    visibility = ["//visibility:public"],
    deps = [],
)
```

Modify `tests/BUILD.bazel`:

```python
cc_test(
    name = "feature_detection_test",
    srcs = ["cpu/feature_detection_test.cc"],
    deps = [
        "//src:cpu_features",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test to verify it passes**

```bash
mkdir -p src/cpu tests/cpu
bazel test //tests/cpu:feature_detection_test --test_output=all
```

Expected: Tests PASS with diagnostic output showing detected CPU features

**Step 6: Commit**

```bash
git add src/support/cpu/feature_detection.hpp
git add tests/cpu/feature_detection_test.cc
git add src/BUILD.bazel tests/BUILD.bazel
git commit -m "feat: add CPU feature detection with OS support validation

Implements runtime CPU feature detection using CPUID and XGETBV:
- Detects SSE2, AVX2, AVX-512F, FMA
- Validates OS support via XGETBV (prevents SIGILL on AVX/AVX-512)
- ISA target selection (diagnostic only, IFUNC does actual dispatch)

Tests verify feature detection and ISA target naming.

Part of unified memory management refactor (Phase 3/5)."
```

---

### Task 5: Create SIMD stencil operator with target_clones

**Files:**
- Create: `src/pde/operators/centered_difference_simd.hpp`
- Test: `tests/operators/centered_difference_simd_test.cc`
- Modify: `src/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

**Step 1: Write test for SIMD second derivative**

Create `tests/operators/centered_difference_simd_test.cc`:

```cpp
#include "src/pde/operators/centered_difference_simd.hpp"
#include "src/pde/operators/grid_spacing.hpp"
#include "src/grid.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

TEST(CenteredDifferenceSIMDTest, UniformSecondDerivative) {
    // Create uniform grid
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 11);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::operators::GridSpacing<double> spacing(grid.span());
    mango::operators::CenteredDifferenceSIMD<double> stencil(spacing);

    // Test function: u(x) = x^2, d2u/dx2 = 2.0 everywhere
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = grid.span()[i] * grid.span()[i];
    }

    std::vector<double> d2u_dx2(11, 0.0);

    // Compute second derivative (interior points only)
    stencil.compute_second_derivative_uniform(u, d2u_dx2, 1, 10);

    // Check interior points (should be 2.0)
    for (size_t i = 1; i < 10; ++i) {
        EXPECT_NEAR(d2u_dx2[i], 2.0, 1e-10);
    }
}

TEST(CenteredDifferenceSIMDTest, TiledComputation) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 10.0, 101);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::operators::GridSpacing<double> spacing(grid.span());
    mango::operators::CenteredDifferenceSIMD<double> stencil(spacing, 32);  // L1 tile size

    // u(x) = sin(x), d2u/dx2 = -sin(x)
    std::vector<double> u(101);
    for (size_t i = 0; i < 101; ++i) {
        u[i] = std::sin(grid.span()[i]);
    }

    std::vector<double> d2u_dx2(101, 0.0);

    stencil.compute_second_derivative_tiled(u, d2u_dx2, 1, 100);

    // Verify against analytical derivative
    for (size_t i = 1; i < 100; ++i) {
        double expected = -std::sin(grid.span()[i]);
        EXPECT_NEAR(d2u_dx2[i], expected, 1e-6);
    }
}

TEST(CenteredDifferenceSIMDTest, FirstDerivative) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 11);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::operators::GridSpacing<double> spacing(grid.span());
    mango::operators::CenteredDifferenceSIMD<double> stencil(spacing);

    // u(x) = x^3, du/dx = 3x^2
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = grid.span()[i] * grid.span()[i] * grid.span()[i];
    }

    std::vector<double> du_dx(11, 0.0);

    stencil.compute_first_derivative_uniform(u, du_dx, 1, 10);

    for (size_t i = 1; i < 10; ++i) {
        double x = grid.span()[i];
        double expected = 3.0 * x * x;
        EXPECT_NEAR(du_dx[i], expected, 1e-10);
    }
}

TEST(CenteredDifferenceSIMDTest, PaddedArraySafety) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 10);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    mango::operators::GridSpacing<double> spacing(grid.span());
    mango::operators::CenteredDifferenceSIMD<double> stencil(spacing);

    // Allocate with SIMD padding (10 → 16)
    std::vector<double> u(16, 0.0);
    std::vector<double> d2u_dx2(16, 0.0);

    // Fill logical portion
    for (size_t i = 0; i < 10; ++i) {
        u[i] = static_cast<double>(i);
    }

    // Compute should not crash on padded array
    stencil.compute_second_derivative_uniform(u, d2u_dx2, 1, 9);

    // Padding should remain zero
    EXPECT_DOUBLE_EQ(d2u_dx2[10], 0.0);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests/operators:centered_difference_simd_test
```

Expected: Build fails

**Step 3: Create SIMD operator implementation**

Create `src/pde/operators/centered_difference_simd.hpp`:

```cpp
#pragma once

#include "grid_spacing.hpp"
#include <experimental/simd>
#include <span>
#include <concepts>
#include <cassert>
#include <algorithm>
#include <cmath>

namespace mango::operators {

namespace stdx = std::experimental;

/**
 * CenteredDifferenceSIMD: Vectorized stencil operator
 *
 * Replaces scalar std::fma with std::experimental::simd operations.
 * Uses [[gnu::target_clones]] for ISA-specific code generation.
 *
 * REQUIREMENTS:
 * - Input spans must be PADDED (use workspace.u_current_padded(), etc.)
 * - start must be ≥ 1 (no boundary point)
 * - end must be ≤ u.size() - 1 (no boundary point)
 * - Boundary conditions handled separately by caller
 */
template<std::floating_point T = double>
class CenteredDifferenceSIMD {
public:
    using simd_t = stdx::native_simd<T>;
    static constexpr size_t simd_width = simd_t::size();

    explicit CenteredDifferenceSIMD(const GridSpacing<T>& spacing,
                                   size_t l1_tile_size = 1024)
        : spacing_(spacing)
        , l1_tile_size_(l1_tile_size)
    {}

    /**
     * Vectorized second derivative kernel (uniform grid)
     *
     * Marked with target_clones for ISA-specific code generation:
     * - default: SSE2 baseline (simd_width = 2 for double)
     * - avx2: Haswell+ (simd_width = 4 for double)
     * - avx512f: Skylake-X+ (simd_width = 8 for double)
     *
     * Verify with: objdump -d <binary> | grep -A20 compute_second_derivative_uniform
     */
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative_uniform(
        std::span<const T> u,
        std::span<T> d2u_dx2,
        size_t start,
        size_t end) const
    {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");

        const T dx2_inv = spacing_.spacing_inv_sq();
        const simd_t dx2_inv_vec(dx2_inv);
        const simd_t minus_two(T(-2));

        // Vectorized main loop
        size_t i = start;
        for (; i + simd_width <= end; i += simd_width) {
            // SoA layout ensures contiguous loads (no gather needed)
            simd_t u_left, u_center, u_right;
            u_left.copy_from(u.data() + i - 1, stdx::element_aligned);
            u_center.copy_from(u.data() + i, stdx::element_aligned);
            u_right.copy_from(u.data() + i + 1, stdx::element_aligned);

            // d2u/dx2 = (u[i+1] + u[i-1] - 2*u[i]) * dx2_inv
            const simd_t sum = u_left + u_right;
            const simd_t result = stdx::fma(sum, dx2_inv_vec,
                                           minus_two * u_center * dx2_inv_vec);

            result.copy_to(d2u_dx2.data() + i, stdx::element_aligned);
        }

        // Scalar tail (zero-padded arrays allow safe i+1 access)
        for (; i < end; ++i) {
            d2u_dx2[i] = std::fma(u[i+1] + u[i-1], dx2_inv, T(-2) * u[i] * dx2_inv);
        }
    }

    /**
     * Tiled second derivative (cache-friendly)
     *
     * Operator decides tile size based on stencil width and cache target.
     */
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative_tiled(
        std::span<const T> u,
        std::span<T> d2u_dx2,
        size_t start,
        size_t end) const
    {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");

        for (size_t tile_start = start; tile_start < end; tile_start += l1_tile_size_) {
            const size_t tile_end = std::min(tile_start + l1_tile_size_, end);
            compute_second_derivative_uniform(u, d2u_dx2, tile_start, tile_end);
        }
    }

    /**
     * First derivative (vectorized, uniform grid)
     */
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_first_derivative_uniform(
        std::span<const T> u,
        std::span<T> du_dx,
        size_t start,
        size_t end) const
    {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");

        const T half_dx_inv = spacing_.spacing_inv() * T(0.5);
        const simd_t half_dx_inv_vec(half_dx_inv);

        size_t i = start;
        for (; i + simd_width <= end; i += simd_width) {
            simd_t u_left, u_right;
            u_left.copy_from(u.data() + i - 1, stdx::element_aligned);
            u_right.copy_from(u.data() + i + 1, stdx::element_aligned);

            const simd_t result = (u_right - u_left) * half_dx_inv_vec;
            result.copy_to(du_dx.data() + i, stdx::element_aligned);
        }

        for (; i < end; ++i) {
            du_dx[i] = (u[i+1] - u[i-1]) * half_dx_inv;
        }
    }

    size_t tile_size() const { return l1_tile_size_; }

private:
    const GridSpacing<T>& spacing_;
    size_t l1_tile_size_;
};

} // namespace mango::operators
```

**Step 4: Update BUILD.bazel**

Modify `src/BUILD.bazel`:

```python
cc_library(
    name = "centered_difference_simd",
    hdrs = ["operators/centered_difference_simd.hpp"],
    visibility = ["//visibility:public"],
    copts = ["-std=c++23"],  # Required for std::experimental::simd
    deps = [":grid_spacing"],
)
```

Modify `tests/BUILD.bazel`:

```python
cc_test(
    name = "centered_difference_simd_test",
    srcs = ["operators/centered_difference_simd_test.cc"],
    copts = ["-std=c++23"],
    deps = [
        "//src:centered_difference_simd",
        "//src:grid",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test to verify it passes**

```bash
mkdir -p tests/operators
bazel test //tests/operators:centered_difference_simd_test --test_output=all
```

Expected: Tests PASS

**Step 6: Verify IFUNC clones generated**

```bash
bazel build //tests/operators:centered_difference_simd_test -c opt
objdump -d bazel-bin/tests/operators/centered_difference_simd_test | grep "compute_second_derivative_uniform"
```

Expected: Should see `.default`, `.avx2`, `.avx512f` suffixes

**Step 7: Commit**

```bash
git add src/pde/operators/centered_difference_simd.hpp
git add tests/operators/centered_difference_simd_test.cc
git add src/BUILD.bazel tests/BUILD.bazel
git commit -m "feat: add CenteredDifferenceSIMD with target_clones

SIMD-vectorized stencil operator using std::experimental::simd:
- First and second derivatives with target_clones (default/avx2/avx512f)
- Operator-level tiling for cache-friendly execution
- SoA-friendly: contiguous loads/stores, no gather/scatter
- Tests verify correctness against analytical derivatives

Part of unified memory management refactor (Phase 3/5)."
```

---

## Phase 4: Integration with Existing PDESolver

### Task 6: Update PDESolver to use PDEWorkspace

**Files:**
- Modify: `src/pde_solver.hpp` (replace WorkspaceStorage with PDEWorkspace)
- Modify: `tests/pde_solver_test.cc` (update tests)

**Step 1: Read current PDESolver implementation**

```bash
# Understand current workspace usage
grep -n "WorkspaceStorage" src/pde_solver.hpp | head -20
```

**Step 2: Create compatibility test**

Add to `tests/pde_solver_test.cc`:

```cpp
TEST(PDESolverTest, PDEWorkspaceIntegration) {
    // Verify PDEWorkspace is drop-in replacement for WorkspaceStorage
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 101);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    // This test will pass once we switch to PDEWorkspace
    mango::PDEWorkspace workspace(101, grid.span());

    EXPECT_EQ(workspace.u_current().size(), 101);
    EXPECT_EQ(workspace.dx().size(), 100);
}
```

**Step 3: Run test to verify current state**

```bash
bazel test //tests:pde_solver_test --test_filter="*PDEWorkspaceIntegration*"
```

Expected: May fail if PDEWorkspace not yet imported

**Step 4: Replace WorkspaceStorage with PDEWorkspace in PDESolver**

Modify `src/pde_solver.hpp`:

Find:
```cpp
#include "workspace.hpp"
```

Replace with:
```cpp
#include "memory/pde_workspace.hpp"
```

Find:
```cpp
WorkspaceStorage workspace_;
```

Replace with:
```cpp
PDEWorkspace workspace_;
```

**Step 5: Run all PDE solver tests**

```bash
bazel test //tests:pde_solver_test --test_output=errors
```

Expected: All tests PASS (PDEWorkspace is API-compatible)

**Step 6: Commit**

```bash
git add src/pde_solver.hpp tests/pde_solver_test.cc
git commit -m "refactor: migrate PDESolver to PDEWorkspace

Replace WorkspaceStorage with PDEWorkspace (drop-in compatible):
- Same API for u_current(), dx(), etc.
- Adds padded accessors for future SIMD operators
- std::pmr-based allocation replaces manual aligned_alloc/free

All existing tests pass without modification.

Part of unified memory management refactor (Phase 4/5)."
```

---

## Phase 5: Final Integration and Benchmarking

### Task 7: Add SIMD operator integration to PDESolver

**Files:**
- Modify: `src/pde_solver.hpp` (add CPU feature detection and SIMD operator)
- Create: `tests/pde_solver_simd_benchmark.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Add CPU feature detection to PDESolver**

Modify `src/pde_solver.hpp` - add includes:

```cpp
#include "cpu/feature_detection.hpp"
#include "operators/centered_difference_simd.hpp"
```

Add member to PDESolver class:

```cpp
private:
    cpu::ISATarget isa_target_;  // Diagnostic only
```

Update constructor to detect ISA:

```cpp
PDESolver(/* existing params */)
    : workspace_(n, grid)
    , isa_target_(cpu::select_isa_target())
    , /* other initialization */
{
    #ifndef NDEBUG
    std::cout << "PDESolver ISA target: "
              << cpu::isa_target_name(isa_target_) << "\n";
    #endif
}
```

**Step 2: Create SIMD benchmark test**

Create `tests/pde_solver_simd_benchmark.cc`:

```cpp
#include "src/pde_solver.hpp"
#include "src/support/cpu/feature_detection.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>

TEST(PDESolverSIMDBenchmark, PerformanceComparison) {
    auto grid_result = mango::GridSpec<>::uniform(0.0, 1.0, 1001);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value().generate();

    auto isa = mango::cpu::select_isa_target();
    std::cout << "Running on ISA: " << mango::cpu::isa_target_name(isa) << "\n";

    // Benchmark will be expanded with actual timing once SIMD operators integrated
    EXPECT_TRUE(isa == mango::cpu::ISATarget::DEFAULT ||
                isa == mango::cpu::ISATarget::AVX2 ||
                isa == mango::cpu::ISATarget::AVX512F);
}
```

**Step 3: Update BUILD.bazel**

Modify `tests/BUILD.bazel`:

```python
cc_test(
    name = "pde_solver_simd_benchmark",
    srcs = ["pde_solver_simd_benchmark.cc"],
    copts = ["-std=c++23"],
    deps = [
        "//src:pde_solver",
        "//src:cpu_features",
        "@googletest//:gtest_main",
    ],
)
```

**Step 4: Run benchmark**

```bash
bazel test //tests:pde_solver_simd_benchmark --test_output=all
```

Expected: Shows detected ISA target

**Step 5: Commit**

```bash
git add src/pde_solver.hpp
git add tests/pde_solver_simd_benchmark.cc
git add tests/BUILD.bazel
git commit -m "feat: integrate CPU feature detection into PDESolver

Add ISA target detection to PDESolver for diagnostic logging:
- Detects SSE2/AVX2/AVX-512F at construction
- Logs selected ISA in debug builds
- Add benchmark test for performance tracking

SIMD operators to be integrated in follow-up commits.

Part of unified memory management refactor (Phase 5/5)."
```

---

### Task 8: Deprecate old WorkspaceStorage

**Files:**
- Modify: `src/workspace.hpp` (add deprecation warning)
- Create: `docs/migration/workspace-migration-guide.md`

**Step 1: Add deprecation notice to WorkspaceStorage**

Modify `src/workspace.hpp`:

```cpp
/**
 * WorkspaceStorage: DEPRECATED - Use PDEWorkspace instead
 *
 * @deprecated Replaced by mango::PDEWorkspace with std::pmr allocation
 *             and full SoA layout. This class will be removed in v3.0.
 *
 * Migration: Replace `WorkspaceStorage` with `PDEWorkspace` (drop-in compatible)
 */
class [[deprecated("Use PDEWorkspace instead")]] WorkspaceStorage {
    // ... existing implementation ...
};
```

**Step 2: Create migration guide**

Create `docs/migration/workspace-migration-guide.md`:

```markdown
# Workspace Migration Guide

## Overview

The workspace memory management has been refactored to use modern C++23 features.
Old workspace classes are deprecated in favor of new PMR-based implementations.

## Migration Path

### WorkspaceStorage → PDEWorkspace

**Before:**
\`\`\`cpp
#include "src/workspace.hpp"
WorkspaceStorage workspace(n, grid);
auto u = workspace.u_current();
\`\`\`

**After:**
\`\`\`cpp
#include "src/pde/memory/pde_workspace.hpp"
PDEWorkspace workspace(n, grid);
auto u = workspace.u_current();  // Same API!
\`\`\`

### For SIMD Kernels

New: Use padded accessors for vectorized operations:

\`\`\`cpp
auto u_padded = workspace.u_current_padded();
auto lu_padded = workspace.lu_padded();
stencil.compute_second_derivative_tiled(u_padded, lu_padded, 1, n-1);
\`\`\`

## API Compatibility

PDEWorkspace is API-compatible with WorkspaceStorage for:
- `u_current()`, `u_next()`, `u_stage()`, `rhs()`, `lu()`, `psi_buffer()`
- `dx()` - precomputed grid spacing

New accessors:
- `u_current_padded()` - SIMD-friendly padded span
- `dx_padded()` - SIMD-friendly grid spacing
- `tile_info(idx, num_tiles)` - operator tiling metadata

## Deprecation Timeline

- v2.5 (current): WorkspaceStorage marked deprecated, warnings issued
- v2.6: Remove WorkspaceStorage, PDEWorkspace becomes default
- v3.0: Remove all deprecated workspace code

## Questions?

See `docs/plans/2025-11-10-unified-memory-management-c++23-refactor.md`
```

**Step 3: Run all tests with deprecation warnings**

```bash
bazel test //... --copt="-Wdeprecated-declarations" 2>&1 | grep "deprecated"
```

Expected: Shows warnings for any remaining WorkspaceStorage usage

**Step 4: Commit**

```bash
mkdir -p docs/migration
git add src/workspace.hpp
git add docs/migration/workspace-migration-guide.md
git commit -m "docs: deprecate WorkspaceStorage, add migration guide

Mark WorkspaceStorage as deprecated in favor of PDEWorkspace:
- Add [[deprecated]] attribute with migration message
- Create migration guide with before/after examples
- Document deprecation timeline (removal in v3.0)

Existing code continues to work with compiler warnings.

Part of unified memory management refactor (Phase 5/5)."
```

---

## Verification and Cleanup

### Task 9: Run full test suite and verify performance

**Step 1: Run all tests**

```bash
bazel test //... --test_output=errors
```

Expected: All tests PASS

**Step 2: Run with sanitizers**

```bash
# AddressSanitizer
bazel test //... --config=asan --test_output=errors

# UndefinedBehaviorSanitizer
bazel test //... --config=ubsan --test_output=errors
```

Expected: No errors

**Step 3: Verify IFUNC clones**

```bash
bazel build //src:pde_solver -c opt
objdump -d bazel-bin/src/libpde_solver.so | grep -E "\.avx2|\.avx512f|\.default" | head -20
```

Expected: See ISA-specific function variants

**Step 4: Performance benchmark**

Create `scripts/benchmark_simd.sh`:

```bash
#!/bin/bash
set -e

echo "Building optimized binaries..."
bazel build //tests:pde_solver_simd_benchmark -c opt

echo "Running SIMD benchmark..."
./bazel-bin/tests/pde_solver_simd_benchmark

echo "Verification complete!"
```

Run:

```bash
chmod +x scripts/benchmark_simd.sh
./scripts/benchmark_simd.sh
```

**Step 5: Document results**

Create `docs/benchmark/2025-11-10-simd-performance.md`:

```markdown
# SIMD Performance Benchmark Results

**Date:** 2025-11-10
**CPU:** [Your CPU model]
**ISA:** [Detected ISA: SSE2/AVX2/AVX-512F]

## Results

[Paste benchmark output here]

## Expected Performance

- AVX2: 3-4x speedup over scalar
- AVX-512F: 6-8x speedup over scalar

## Verification

- Correctness: ✓ All tests pass
- Memory safety: ✓ ASan/UBSan clean
- ISA dispatch: ✓ IFUNC clones verified
```

**Step 6: Final commit**

```bash
git add scripts/benchmark_simd.sh
git add docs/benchmark/2025-11-10-simd-performance.md
git commit -m "test: add SIMD performance verification

Complete verification suite:
- Full test suite (all tests pass)
- Sanitizer checks (ASan, UBSan clean)
- IFUNC clone verification (objdump)
- Performance benchmarks (document ISA-specific speedups)

Completes unified memory management refactor (Phase 5/5)."
```

---

## Summary

This plan implements the complete unified memory management refactor in 9 tasks:

1. **UnifiedMemoryResource** - std::pmr wrapper
2. **WorkspaceBase** - tiling infrastructure
3. **PDEWorkspace** - SoA layout with SIMD padding
4. **CPU feature detection** - runtime ISA selection
5. **SIMD operators** - target_clones with std::experimental::simd
6. **PDESolver integration** - migrate to PDEWorkspace
7. **ISA detection** - integrate into solver
8. **Deprecation** - mark old classes, migration guide
9. **Verification** - tests, sanitizers, benchmarks

Each task follows TDD (test-first), includes exact file paths and code, and commits incrementally.

**Total estimated time:** 2-3 days for experienced C++ developer

**Dependencies:**
- GCC 11+ or Clang 14+ (for std::experimental::simd)
- Bazel build system
- GoogleTest framework
- x86-64 CPU (for CPUID/XGETBV)
