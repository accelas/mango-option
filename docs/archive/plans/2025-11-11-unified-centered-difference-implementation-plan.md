<!-- SPDX-License-Identifier: MIT -->
# Unified CenteredDifference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify scalar and SIMD CenteredDifference implementations behind a single façade with automatic runtime ISA selection.

**Architecture:** Façade + Backend pattern. Public `CenteredDifference` class dispatches to `ScalarBackend` or `SimdBackend` via virtual interface. Mode::Auto uses CPU feature detection + OS XSAVE check.

**Tech Stack:** C++23, std::experimental::simd, [[gnu::target_clones]], GoogleTest

---

## Task 1: Create ScalarBackend (Extract from CenteredDifference)

**Files:**
- Create: `src/pde/operators/centered_difference_scalar.hpp`
- Reference: `src/pde/operators/centered_difference.hpp` (current scalar implementation)
- Test: Will add tests in Task 5

**Step 1: Copy current CenteredDifference to ScalarBackend template**

Create `src/pde/operators/centered_difference_scalar.hpp` with the class renamed to `ScalarBackend<T>`:

```cpp
#pragma once

#include "grid_spacing.hpp"
#include "../parallel.hpp"
#include <span>
#include <cmath>
#include <cassert>

namespace mango::operators {

/**
 * ScalarBackend: Scalar implementation with compiler auto-vectorization
 *
 * Uses #pragma omp simd hints for automatic vectorization.
 * For non-uniform grids, loads from precomputed GridSpacing arrays.
 */
template<std::floating_point T = double>
class ScalarBackend {
public:
    explicit ScalarBackend(const GridSpacing<T>& spacing)
        : spacing_(spacing)
    {}

    // Uniform grid second derivative
    void compute_second_derivative_uniform(
        std::span<const T> u, std::span<T> d2u_dx2,
        size_t start, size_t end) const
    {
        const T dx2_inv = spacing_.spacing_inv_sq();

        MANGO_PRAGMA_SIMD
        for (size_t i = start; i < end; ++i) {
            d2u_dx2[i] = std::fma(u[i+1] + u[i-1], dx2_inv,
                                 -T(2)*u[i]*dx2_inv);
        }
    }

    // Uniform grid first derivative
    void compute_first_derivative_uniform(
        std::span<const T> u, std::span<T> du_dx,
        size_t start, size_t end) const
    {
        const T half_dx_inv = spacing_.spacing_inv() * T(0.5);

        MANGO_PRAGMA_SIMD
        for (size_t i = start; i < end; ++i) {
            du_dx[i] = (u[i+1] - u[i-1]) * half_dx_inv;
        }
    }

    // Non-uniform grid second derivative - USES PRECOMPUTED ARRAYS
    void compute_second_derivative_non_uniform(
        std::span<const T> u, std::span<T> d2u_dx2,
        size_t start, size_t end) const
    {
        auto dx_left_inv = spacing_.dx_left_inv();
        auto dx_right_inv = spacing_.dx_right_inv();
        auto dx_center_inv = spacing_.dx_center_inv();

        MANGO_PRAGMA_SIMD
        for (size_t i = start; i < end; ++i) {
            const T dxl_inv = dx_left_inv[i - 1];
            const T dxr_inv = dx_right_inv[i - 1];
            const T dxc_inv = dx_center_inv[i - 1];

            const T forward_diff = (u[i+1] - u[i]) * dxr_inv;
            const T backward_diff = (u[i] - u[i-1]) * dxl_inv;
            d2u_dx2[i] = (forward_diff - backward_diff) * dxc_inv;
        }
    }

    // Non-uniform grid first derivative - USES PRECOMPUTED ARRAYS
    void compute_first_derivative_non_uniform(
        std::span<const T> u, std::span<T> du_dx,
        size_t start, size_t end) const
    {
        auto w_left = spacing_.w_left();
        auto w_right = spacing_.w_right();
        auto dx_left_inv = spacing_.dx_left_inv();
        auto dx_right_inv = spacing_.dx_right_inv();

        MANGO_PRAGMA_SIMD
        for (size_t i = start; i < end; ++i) {
            const T wl = w_left[i - 1];
            const T wr = w_right[i - 1];
            const T dxl_inv = dx_left_inv[i - 1];
            const T dxr_inv = dx_right_inv[i - 1];

            const T term1 = wl * (u[i] - u[i-1]) * dxl_inv;
            const T term2 = wr * (u[i+1] - u[i]) * dxr_inv;
            du_dx[i] = term1 + term2;
        }
    }

    // Auto-dispatch second derivative
    void compute_second_derivative(
        std::span<const T> u, std::span<T> d2u_dx2,
        size_t start, size_t end) const
    {
        if (spacing_.is_uniform()) {
            compute_second_derivative_uniform(u, d2u_dx2, start, end);
        } else {
            compute_second_derivative_non_uniform(u, d2u_dx2, start, end);
        }
    }

    // Auto-dispatch first derivative
    void compute_first_derivative(
        std::span<const T> u, std::span<T> du_dx,
        size_t start, size_t end) const
    {
        if (spacing_.is_uniform()) {
            compute_first_derivative_uniform(u, du_dx, start, end);
        } else {
            compute_first_derivative_non_uniform(u, du_dx, start, end);
        }
    }

    // Tiled second derivative
    void compute_second_derivative_tiled(
        std::span<const T> u, std::span<T> d2u_dx2,
        size_t start, size_t end) const
    {
        // Scalar backend doesn't use tiling (simple forwarding)
        compute_second_derivative(u, d2u_dx2, start, end);
    }

private:
    const GridSpacing<T>& spacing_;
};

} // namespace mango::operators
```

**Step 2: Verify file compiles**

Run: `bazel build //src/operators:centered_difference_scalar`

Expected: SUCCESS (header-only, will compile when included)

**Step 3: Commit**

```bash
git add src/pde/operators/centered_difference_scalar.hpp
git commit -m "feat: add ScalarBackend extracted from CenteredDifference

Extracted scalar implementation to ScalarBackend template.
Key change: non-uniform methods now use precomputed arrays
from GridSpacing (matches SIMD strategy).

Part of unified CenteredDifference façade implementation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Create SimdBackend (Rename CenteredDifferenceSIMD)

**Files:**
- Create: `src/pde/operators/centered_difference_simd_backend.hpp`
- Reference: `src/pde/operators/centered_difference_simd.hpp` (current SIMD implementation)

**Step 1: Copy current CenteredDifferenceSIMD to SimdBackend**

Create `src/pde/operators/centered_difference_simd_backend.hpp` by copying the entire contents of `centered_difference_simd.hpp` and renaming the class:

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
 * SimdBackend: Vectorized stencil operator (renamed from CenteredDifferenceSIMD)
 *
 * All implementation unchanged from CenteredDifferenceSIMD.
 * Uses [[gnu::target_clones]] for ISA-specific code generation.
 */
template<std::floating_point T = double>
class SimdBackend {
public:
    using simd_t = stdx::native_simd<T>;
    static constexpr size_t simd_width = simd_t::size();

    explicit SimdBackend(const GridSpacing<T>& spacing,
                        size_t l1_tile_size = 1024)
        : spacing_(spacing)
        , l1_tile_size_(l1_tile_size)
    {}

    // Copy ALL methods from CenteredDifferenceSIMD unchanged:
    // - compute_second_derivative_uniform
    // - compute_second_derivative_tiled
    // - compute_first_derivative_uniform
    // - compute_second_derivative_non_uniform
    // - compute_first_derivative_non_uniform
    // - compute_second_derivative (auto-dispatch wrapper)
    // - compute_first_derivative (auto-dispatch wrapper)
    // - tile_size()

    // [Copy entire implementation from centered_difference_simd.hpp]
    // Just replace class name CenteredDifferenceSIMD → SimdBackend

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

        size_t i = start;
        for (; i + simd_width <= end; i += simd_width) {
            simd_t u_left, u_center, u_right;
            u_left.copy_from(u.data() + i - 1, stdx::element_aligned);
            u_center.copy_from(u.data() + i, stdx::element_aligned);
            u_right.copy_from(u.data() + i + 1, stdx::element_aligned);

            const simd_t sum = u_left + u_right;
            const simd_t result = stdx::fma(sum, dx2_inv_vec,
                                           minus_two * u_center * dx2_inv_vec);

            result.copy_to(d2u_dx2.data() + i, stdx::element_aligned);
        }

        for (; i < end; ++i) {
            d2u_dx2[i] = std::fma(u[i+1] + u[i-1], dx2_inv, T(-2) * u[i] * dx2_inv);
        }
    }

    // [Copy all other methods verbatim from centered_difference_simd.hpp]
    // ... (continue with full implementation)

    size_t tile_size() const { return l1_tile_size_; }

private:
    const GridSpacing<T>& spacing_;
    size_t l1_tile_size_;
};

} // namespace mango::operators
```

**Note:** This step copies the ENTIRE implementation verbatim. Only the class name changes.

**Step 2: Verify file compiles**

Run: `bazel build //src/operators:centered_difference_simd_backend`

Expected: SUCCESS

**Step 3: Commit**

```bash
git add src/pde/operators/centered_difference_simd_backend.hpp
git commit -m "feat: add SimdBackend (rename of CenteredDifferenceSIMD)

Pure rename with zero implementation changes.
All [[gnu::target_clones]] attributes preserved.

Part of unified CenteredDifference façade implementation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Implement CenteredDifference Façade

**Files:**
- Create: `src/pde/operators/centered_difference_facade.hpp`
- Reference: `src/support/cpu/feature_detection.hpp` (for CPU detection)

**Step 1: Write façade header**

Create `src/pde/operators/centered_difference_facade.hpp`:

```cpp
#pragma once

#include "grid_spacing.hpp"
#include "centered_difference_scalar.hpp"
#include "centered_difference_simd_backend.hpp"
#include "../cpu/feature_detection.hpp"
#include <span>
#include <memory>
#include <cassert>

namespace mango::operators {

/**
 * CenteredDifference: Unified façade with automatic backend selection
 *
 * Mode::Auto (default) uses CPU feature detection to pick optimal backend.
 * Mode::Scalar/Simd force specific backend (for testing).
 *
 * Virtual dispatch overhead: ~5-10ns per call (negligible vs computation).
 */
class CenteredDifference {
public:
    enum class Mode { Auto, Scalar, Simd };

    explicit CenteredDifference(const GridSpacing<double>& spacing,
                                Mode mode = Mode::Auto);

    // Public API - keeps [[gnu::target_clones]] for consistent symbols
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative(std::span<const double> u,
                                   std::span<double> d2u_dx2,
                                   size_t start, size_t end) const {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");
        impl_->compute_second_derivative(u, d2u_dx2, start, end);
    }

    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_first_derivative(std::span<const double> u,
                                  std::span<double> du_dx,
                                  size_t start, size_t end) const {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");
        impl_->compute_first_derivative(u, du_dx, start, end);
    }

    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative_tiled(std::span<const double> u,
                                         std::span<double> d2u_dx2,
                                         size_t start, size_t end) const {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");
        impl_->compute_second_derivative_tiled(u, d2u_dx2, start, end);
    }

private:
    struct BackendInterface {
        virtual ~BackendInterface() = default;
        virtual void compute_second_derivative(
            std::span<const double> u, std::span<double> d2u_dx2,
            size_t start, size_t end) const = 0;
        virtual void compute_first_derivative(
            std::span<const double> u, std::span<double> du_dx,
            size_t start, size_t end) const = 0;
        virtual void compute_second_derivative_tiled(
            std::span<const double> u, std::span<double> d2u_dx2,
            size_t start, size_t end) const = 0;
    };

    template<typename Backend>
    struct BackendImpl final : BackendInterface {
        Backend backend_;

        explicit BackendImpl(const GridSpacing<double>& spacing)
            : backend_(spacing) {}

        void compute_second_derivative(
            std::span<const double> u, std::span<double> d2u_dx2,
            size_t start, size_t end) const override {
            backend_.compute_second_derivative(u, d2u_dx2, start, end);
        }

        void compute_first_derivative(
            std::span<const double> u, std::span<double> du_dx,
            size_t start, size_t end) const override {
            backend_.compute_first_derivative(u, du_dx, start, end);
        }

        void compute_second_derivative_tiled(
            std::span<const double> u, std::span<double> d2u_dx2,
            size_t start, size_t end) const override {
            backend_.compute_second_derivative_tiled(u, d2u_dx2, start, end);
        }
    };

    std::unique_ptr<BackendInterface> impl_;
};

// Constructor implementation
inline CenteredDifference::CenteredDifference(const GridSpacing<double>& spacing,
                                              Mode mode)
{
    if (mode == Mode::Auto) {
        // Check CPU features AND OS support
        auto features = cpu::detect_features();
        bool os_supports_avx = cpu::check_os_avx_support();

        // Use SIMD if both CPU and OS support it
        if ((features.has_avx2 || features.has_avx512f) && os_supports_avx) {
            mode = Mode::Simd;
        } else {
            mode = Mode::Scalar;
        }
    }

    switch (mode) {
        case Mode::Scalar:
            impl_ = std::make_unique<BackendImpl<ScalarBackend<double>>>(spacing);
            break;
        case Mode::Simd:
            impl_ = std::make_unique<BackendImpl<SimdBackend<double>>>(spacing);
            break;
        case Mode::Auto:
            // Already resolved above
            break;
    }
}

} // namespace mango::operators
```

**Step 2: Verify file compiles**

Run: `bazel build //src/operators:centered_difference_facade`

Expected: SUCCESS

**Step 3: Commit**

```bash
git add src/pde/operators/centered_difference_facade.hpp
git commit -m "feat: add CenteredDifference façade with Mode enum

Implements façade + backend pattern with automatic ISA selection:
- Mode::Auto: CPU detection + OS XSAVE check
- Mode::Scalar/Simd: Explicit backend selection (for tests)
- Virtual dispatch: ~5-10ns overhead (negligible)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Add BUILD.bazel Targets for New Files

**Files:**
- Modify: `src/pde/operators/BUILD.bazel`

**Step 1: Add cc_library targets for backends and façade**

Add to `src/pde/operators/BUILD.bazel`:

```python
cc_library(
    name = "centered_difference_scalar",
    hdrs = ["centered_difference_scalar.hpp"],
    deps = [
        ":grid_spacing",
        "//src:parallel",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "centered_difference_simd_backend",
    hdrs = ["centered_difference_simd_backend.hpp"],
    deps = [
        ":grid_spacing",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "centered_difference_facade",
    hdrs = ["centered_difference_facade.hpp"],
    deps = [
        ":centered_difference_scalar",
        ":centered_difference_simd_backend",
        ":grid_spacing",
        "//src/cpu:feature_detection",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 2: Verify targets build**

Run: `bazel build //src/operators:centered_difference_facade //src/operators:centered_difference_scalar //src/operators:centered_difference_simd_backend`

Expected: SUCCESS (all 3 targets build)

**Step 3: Commit**

```bash
git add src/pde/operators/BUILD.bazel
git commit -m "build: add Bazel targets for unified CenteredDifference

Added cc_library targets:
- centered_difference_scalar
- centered_difference_simd_backend
- centered_difference_facade

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Add Tests for Façade with Mode Selection

**Files:**
- Create: `tests/centered_difference_facade_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing test for Mode::Auto**

Create `tests/centered_difference_facade_test.cc`:

```cpp
#include "src/pde/operators/centered_difference_facade.hpp"
#include "src/pde/operators/grid_spacing.hpp"
#include "src/grid.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace mango::operators {
namespace {

TEST(CenteredDifferenceFacadeTest, AutoModeWorks) {
    // Create non-uniform grid
    std::vector<double> x(11);
    x[0] = -1.0; x[1] = -0.8; x[2] = -0.5; x[3] = -0.2; x[4] = -0.05;
    x[5] = 0.0; x[6] = 0.05; x[7] = 0.2; x[8] = 0.5; x[9] = 0.8; x[10] = 1.0;

    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    // Mode::Auto should select backend automatically
    CenteredDifference stencil(spacing);  // Mode::Auto by default

    // Test with f(x) = x²
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    std::vector<double> d2u_dx2(11, 0.0);
    stencil.compute_second_derivative(u, d2u_dx2, 1, 10);

    // d²(x²)/dx² = 2.0
    for (size_t i = 1; i < 10; ++i) {
        EXPECT_NEAR(d2u_dx2[i], 2.0, 1e-12)
            << "Mismatch at index " << i;
    }
}

TEST(CenteredDifferenceFacadeTest, ScalarModeWorks) {
    std::vector<double> x(11);
    x[0] = -1.0; x[1] = -0.8; x[2] = -0.5; x[3] = -0.2; x[4] = -0.05;
    x[5] = 0.0; x[6] = 0.05; x[7] = 0.2; x[8] = 0.5; x[9] = 0.8; x[10] = 1.0;

    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    // Force scalar backend
    CenteredDifference stencil(spacing, CenteredDifference::Mode::Scalar);

    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    std::vector<double> d2u_dx2(11, 0.0);
    stencil.compute_second_derivative(u, d2u_dx2, 1, 10);

    for (size_t i = 1; i < 10; ++i) {
        EXPECT_NEAR(d2u_dx2[i], 2.0, 1e-12)
            << "Scalar backend failed at index " << i;
    }
}

TEST(CenteredDifferenceFacadeTest, SimdModeWorks) {
    std::vector<double> x(11);
    x[0] = -1.0; x[1] = -0.8; x[2] = -0.5; x[3] = -0.2; x[4] = -0.05;
    x[5] = 0.0; x[6] = 0.05; x[7] = 0.2; x[8] = 0.5; x[9] = 0.8; x[10] = 1.0;

    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    // Force SIMD backend
    CenteredDifference stencil(spacing, CenteredDifference::Mode::Simd);

    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    std::vector<double> d2u_dx2(11, 0.0);
    stencil.compute_second_derivative(u, d2u_dx2, 1, 10);

    for (size_t i = 1; i < 10; ++i) {
        EXPECT_NEAR(d2u_dx2[i], 2.0, 1e-12)
            << "SIMD backend failed at index " << i;
    }
}

TEST(CenteredDifferenceFacadeTest, ScalarVsSimdMatch) {
    std::vector<double> x(11);
    x[0] = -1.0; x[1] = -0.8; x[2] = -0.5; x[3] = -0.2; x[4] = -0.05;
    x[5] = 0.0; x[6] = 0.05; x[7] = 0.2; x[8] = 0.5; x[9] = 0.8; x[10] = 1.0;

    auto grid = GridView<double>(x);
    auto spacing = GridSpacing<double>(grid);

    CenteredDifference scalar_stencil(spacing, CenteredDifference::Mode::Scalar);
    CenteredDifference simd_stencil(spacing, CenteredDifference::Mode::Simd);

    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = x[i] * x[i];
    }

    std::vector<double> d2u_scalar(11, 0.0), d2u_simd(11, 0.0);
    scalar_stencil.compute_second_derivative(u, d2u_scalar, 1, 10);
    simd_stencil.compute_second_derivative(u, d2u_simd, 1, 10);

    // Allow FP rounding differences
    for (size_t i = 1; i < 10; ++i) {
        EXPECT_NEAR(d2u_scalar[i], d2u_simd[i], 1e-14)
            << "Scalar/SIMD mismatch at index " << i;
    }
}

} // namespace
} // namespace mango::operators
```

**Step 2: Add test target to BUILD.bazel**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "centered_difference_facade_test",
    srcs = ["centered_difference_facade_test.cc"],
    deps = [
        "//src/operators:centered_difference_facade",
        "//src/operators:grid_spacing",
        "//src:grid",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3: Run test to verify it fails**

Run: `bazel test //tests:centered_difference_facade_test --test_output=all`

Expected: Tests PASS (implementation already complete from Task 3)

**Step 4: Commit**

```bash
git add tests/centered_difference_facade_test.cc tests/BUILD.bazel
git commit -m "test: add façade tests with Mode selection

Tests verify:
- Mode::Auto works correctly
- Mode::Scalar forces scalar backend
- Mode::Simd forces SIMD backend
- Scalar vs SIMD results match within FP tolerance

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Run Full Test Suite to Verify No Regressions

**Files:**
- None (verification only)

**Step 1: Run all tests**

Run: `bazel test //tests:all --test_output=errors`

Expected: All 40+ tests PASS (no regressions from new code)

**Step 2: If any tests fail, investigate and fix**

Common issues:
- Missing includes
- Template instantiation errors
- Linker errors

Fix any issues before proceeding.

**Step 3: Document verification**

```bash
# No commit needed - just verification checkpoint
echo "All tests passing after façade implementation"
```

---

## Task 7: Update Existing Tests to Use Mode Explicitly

**Files:**
- Modify: `tests/centered_difference_simd_test.cc`

**Step 1: Update SIMD tests to use façade with Mode::Simd**

Modify `tests/centered_difference_simd_test.cc`:

Change includes:
```cpp
// Old:
#include "src/pde/operators/centered_difference_simd.hpp"

// New:
#include "src/pde/operators/centered_difference_facade.hpp"
```

Change test instantiations:
```cpp
// Old:
CenteredDifferenceSIMD<double> simd_stencil(spacing);

// New:
CenteredDifference simd_stencil(spacing, CenteredDifference::Mode::Simd);
```

**Step 2: Run tests to verify they still pass**

Run: `bazel test //tests:centered_difference_simd_test --test_output=errors`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/centered_difference_simd_test.cc
git commit -m "test: update SIMD tests to use façade with Mode::Simd

Changed from direct CenteredDifferenceSIMD instantiation to
CenteredDifference(spacing, Mode::Simd).

All tests pass with façade.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Update CLAUDE.md Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add section on CenteredDifference automatic ISA selection**

Add after the "SIMD Vectorization" section in `CLAUDE.md`:

```markdown
### CenteredDifference: Automatic ISA Selection

The `CenteredDifference` stencil operator automatically selects the optimal backend based on CPU capabilities:

**Mode Enum:**
- **Mode::Auto** (default): Runtime CPU detection + OS XSAVE check chooses Scalar or SIMD
- **Mode::Scalar**: Force scalar backend (for testing/debugging)
- **Mode::Simd**: Force SIMD backend (for testing/benchmarking)

**Production Usage:**
```cpp
// Always use Mode::Auto in production code
auto spacing = GridSpacing<double>(grid);
auto stencil = CenteredDifference(spacing);  // Auto-selects optimal backend
```

**Test Usage:**
```cpp
// Tests can force specific backends for regression testing
auto scalar = CenteredDifference(spacing, CenteredDifference::Mode::Scalar);
auto simd = CenteredDifference(spacing, CenteredDifference::Mode::Simd);

// Compare results
scalar.compute_second_derivative(u, d2u_scalar, 1, n-1);
simd.compute_second_derivative(u, d2u_simd, 1, n-1);
EXPECT_NEAR(d2u_scalar[i], d2u_simd[i], 1e-14);  // Allow FP rounding
```

**Performance Characteristics:**
- Virtual dispatch overhead: ~5-10ns per call
- Negligible vs computation cost (~5,000ns for 100-point grid)
- Both backends use precomputed arrays on non-uniform grids
- SIMD backend: 3-6x speedup via explicit vectorization

**Architecture:**
- Façade + Backend pattern (similar to strategy pattern)
- ScalarBackend: `#pragma omp simd` for compiler auto-vectorization
- SimdBackend: `std::experimental::simd` + `[[gnu::target_clones]]` for multi-ISA
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document CenteredDifference automatic ISA selection

Added section explaining:
- Mode enum (Auto/Scalar/Simd)
- Production usage (Mode::Auto)
- Test usage (explicit Mode selection)
- Performance characteristics
- Architecture overview

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Deprecate Old Headers (Add Warnings)

**Files:**
- Modify: `src/pde/operators/centered_difference.hpp` (old scalar version)
- Modify: `src/pde/operators/centered_difference_simd.hpp` (old SIMD version)

**Step 1: Add deprecation warnings to old headers**

Add to top of `src/pde/operators/centered_difference.hpp`:

```cpp
#pragma once

#warning "This header is deprecated. Use centered_difference_facade.hpp with Mode::Scalar instead."

// Keep existing implementation for now...
```

Add to top of `src/pde/operators/centered_difference_simd.hpp`:

```cpp
#pragma once

#warning "This header is deprecated. Use centered_difference_facade.hpp with Mode::Simd instead."

// Keep existing implementation for now...
```

**Step 2: Verify warnings appear during build**

Run: `bazel build //tests:centered_difference_test --test_output=errors 2>&1 | grep deprecated`

Expected: See deprecation warnings

**Step 3: Commit**

```bash
git add src/pde/operators/centered_difference.hpp src/pde/operators/centered_difference_simd.hpp
git commit -m "deprecate: add warnings to old CenteredDifference headers

Added #warning directives to guide users toward façade.
Old headers remain functional during migration period.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Final Verification and Summary

**Files:**
- None (verification only)

**Step 1: Run full test suite**

Run: `bazel test //tests:all --test_output=errors`

Expected: All tests PASS (including new façade tests)

**Step 2: Verify build targets**

Run: `bazel build //src/operators:all`

Expected: All operators build successfully

**Step 3: Check commit log**

Run: `git log --oneline HEAD~10..HEAD`

Expected: See 10 commits from this implementation plan

**Step 4: Document completion**

Create completion summary:

```bash
echo "Unified CenteredDifference implementation complete!

Summary:
- ScalarBackend: Extracted with precomputed array support
- SimdBackend: Renamed from CenteredDifferenceSIMD
- Façade: Mode::Auto with CPU detection + XSAVE check
- Tests: 4 new façade tests, existing tests updated
- Docs: CLAUDE.md updated with usage examples
- Old headers: Deprecated with warnings

All 40+ tests passing.
Ready for production use with Mode::Auto."
```

---

## Post-Implementation Notes

### Future Cleanup (Optional)

After all call sites migrate to façade:

1. **Remove old headers:**
   - Delete `src/pde/operators/centered_difference.hpp` (old scalar)
   - Delete `src/pde/operators/centered_difference_simd.hpp` (old SIMD)

2. **Make backends truly internal:**
   - Move `centered_difference_scalar.hpp` → `internal/centered_difference_scalar.hpp`
   - Move `centered_difference_simd_backend.hpp` → `internal/centered_difference_simd_backend.hpp`
   - Update `centered_difference_facade.hpp` includes

3. **Update BUILD visibility:**
   - Set backend libraries to `visibility = ["//src/operators:__pkg__"]`
   - Only façade remains publicly visible

### Performance Verification

Benchmark virtual dispatch overhead:

```cpp
// Benchmark code
auto spacing = GridSpacing<double>(grid);
auto stencil = CenteredDifference(spacing, Mode::Auto);

auto start = std::chrono::steady_clock::now();
for (int i = 0; i < 1000; ++i) {
    stencil.compute_second_derivative(u, d2u, 1, n-1);
}
auto end = std::chrono::steady_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

// Expect: <1% overhead vs direct backend call
```

### Known Issues

- None at this stage

### References

- Design doc: `docs/plans/2025-11-11-unified-centered-difference-design.md`
- CPU detection: `src/support/cpu/feature_detection.hpp`
- Current SIMD impl: `src/pde/operators/centered_difference_simd.hpp`
