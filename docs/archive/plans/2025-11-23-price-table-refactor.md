# Price Table Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor price table infrastructure to use template-based N-dimensional design with proper ownership, discrete dividend support, and API consistency with solver conventions.

**Architecture:** Template-based design using `mdspan` for multi-dimensional arrays, `AlignedArena` for memory management, immutable `PriceTableSurface<N>` as the result type, and `PriceTableBuilder<N>` following config→builder→surface pattern.

**Tech Stack:** C++23, `std::mdspan`, PMR allocators, `std::variant`, `std::expected`, GoogleTest

---

## Phase 1: Foundation - AlignedArena

### Task 1.1: Create AlignedArena test file

**Files:**
- Create: `tests/aligned_arena_test.cc`

**Step 1: Write failing test for arena creation**

```cpp
#include <gtest/gtest.h>
#include "mango/support/memory/aligned_arena.hpp"

namespace mango {
namespace memory {
namespace {

TEST(AlignedArenaTest, CreateValidArena) {
    auto result = AlignedArena::create(1024, 64);
    ASSERT_TRUE(result.has_value());

    auto arena = result.value();
    EXPECT_NE(arena, nullptr);
}

TEST(AlignedArenaTest, AllocateAligned) {
    auto arena = AlignedArena::create(1024, 64).value();

    double* ptr = arena->allocate(10);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);  // Check 64-byte alignment
}

TEST(AlignedArenaTest, ShareArena) {
    auto arena1 = AlignedArena::create(1024, 64).value();
    auto arena2 = arena1->share();

    EXPECT_EQ(arena1.use_count(), 2);
}

} // namespace
} // namespace memory
} // namespace mango
```

**Step 2: Run test to verify it fails**

Run: `bazel test //tests:aligned_arena_test`
Expected: FAIL with "No such file: src/support/memory/aligned_arena.hpp"

**Step 3: Create AlignedArena header**

**Files:**
- Create: `src/support/memory/aligned_arena.hpp`

```cpp
#pragma once

#include <memory>
#include <vector>
#include <cstddef>
#include <expected>
#include <string>

namespace mango {
namespace memory {

/// Arena allocator with guaranteed alignment for SIMD operations
///
/// Provides 64-byte aligned memory allocation for AVX-512 compatibility.
/// Uses shared_ptr for automatic lifetime management.
class AlignedArena {
public:
    /// Factory method to create arena
    ///
    /// @param bytes Total size in bytes
    /// @param align Alignment requirement (default 64 for AVX-512)
    /// @return Shared pointer to arena or error message
    [[nodiscard]] static std::expected<std::shared_ptr<AlignedArena>, std::string>
    create(size_t bytes, size_t align = 64);

    /// Allocate aligned memory for count doubles
    ///
    /// @param count Number of doubles to allocate
    /// @return Pointer to aligned memory or nullptr if insufficient space
    [[nodiscard]] double* allocate(size_t count);

    /// Share this arena (increment reference count)
    ///
    /// @return Shared pointer to this arena
    [[nodiscard]] std::shared_ptr<AlignedArena> share();

    /// Get total capacity in bytes
    [[nodiscard]] size_t capacity() const noexcept { return buffer_.size(); }

    /// Get current offset in bytes
    [[nodiscard]] size_t offset() const noexcept { return offset_; }

private:
    explicit AlignedArena(size_t bytes, size_t align);

    std::vector<std::byte> buffer_;
    size_t align_;
    size_t offset_;
    std::shared_ptr<AlignedArena> self_;
};

} // namespace memory
} // namespace mango
```

**Step 4: Create AlignedArena implementation**

**Files:**
- Create: `src/support/memory/aligned_arena.cpp`

```cpp
#include "mango/support/memory/aligned_arena.hpp"
#include <cstring>

namespace mango {
namespace memory {

AlignedArena::AlignedArena(size_t bytes, size_t align)
    : buffer_(bytes), align_(align), offset_(0) {}

std::expected<std::shared_ptr<AlignedArena>, std::string>
AlignedArena::create(size_t bytes, size_t align) {
    if (bytes == 0) {
        return std::unexpected("Arena size must be positive");
    }
    if (align == 0 || (align & (align - 1)) != 0) {
        return std::unexpected("Alignment must be a power of 2");
    }

    auto arena = std::shared_ptr<AlignedArena>(new AlignedArena(bytes, align));
    arena->self_ = arena;
    return arena;
}

double* AlignedArena::allocate(size_t count) {
    const size_t bytes = count * sizeof(double);

    // Align offset to alignment boundary
    size_t aligned_offset = (offset_ + align_ - 1) & ~(align_ - 1);

    if (aligned_offset + bytes > buffer_.size()) {
        return nullptr;  // Out of memory
    }

    double* ptr = reinterpret_cast<double*>(buffer_.data() + aligned_offset);
    offset_ = aligned_offset + bytes;

    return ptr;
}

std::shared_ptr<AlignedArena> AlignedArena::share() {
    return self_;
}

} // namespace memory
} // namespace mango
```

**Step 5: Add BUILD.bazel entries**

**Files:**
- Modify: `src/support/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

Update `src/support/BUILD.bazel`:
```python
cc_library(
    name = "aligned_arena",
    srcs = ["memory/aligned_arena.cpp"],
    hdrs = ["memory/aligned_arena.hpp"],
    visibility = ["//visibility:public"],
)
```

Update `tests/BUILD.bazel`:
```python
cc_test(
    name = "aligned_arena_test",
    size = "small",
    srcs = ["aligned_arena_test.cc"],
    deps = [
        "//src/support:aligned_arena",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 6: Run test to verify it passes**

Run: `bazel test //tests:aligned_arena_test`
Expected: PASS (3/3 tests)

**Step 7: Commit**

```bash
git add src/support/memory/ tests/aligned_arena_test.cc src/support/BUILD.bazel tests/BUILD.bazel
git commit -m "Add AlignedArena for SIMD-aligned memory management

Implement 64-byte aligned memory arena for AVX-512 compatibility.
Uses shared_ptr for automatic lifetime management and supports
zero-copy sharing across price table components.

Part of price table refactor (#XXX)"
```

---

## Phase 2: PriceTableAxes Template

### Task 2.1: Create PriceTableAxes test

**Files:**
- Create: `tests/price_table_axes_test.cc`

**Step 1: Write failing test**

```cpp
#include <gtest/gtest.h>
#include "mango/option/price_table_axes.hpp"

namespace mango {
namespace {

TEST(PriceTableAxesTest, Create4DAxes) {
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};
    axes.grids[1] = {0.027, 0.1, 0.5, 1.0, 2.0};
    axes.grids[2] = {0.10, 0.20, 0.30};
    axes.grids[3] = {0.0, 0.05, 0.10};

    axes.names[0] = "moneyness";
    axes.names[1] = "maturity";
    axes.names[2] = "volatility";
    axes.names[3] = "rate";

    EXPECT_EQ(axes.grids[0].size(), 7);
    EXPECT_EQ(axes.names[0], "moneyness");
}

TEST(PriceTableAxesTest, TotalGridPoints) {
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.7, 0.8, 0.9, 1.0};  // 4 points
    axes.grids[1] = {0.1, 0.5};            // 2 points
    axes.grids[2] = {0.10, 0.20, 0.30};    // 3 points
    axes.grids[3] = {0.0, 0.05};           // 2 points

    size_t total = axes.total_points();
    EXPECT_EQ(total, 4 * 2 * 3 * 2);  // 48 points
}

TEST(PriceTableAxesTest, ValidateMonotonic) {
    PriceTableAxes<2> axes;
    axes.grids[0] = {1.0, 2.0, 3.0};
    axes.grids[1] = {0.1, 0.2, 0.3};

    auto result = axes.validate();
    EXPECT_TRUE(result.has_value());
}

TEST(PriceTableAxesTest, RejectNonMonotonic) {
    PriceTableAxes<2> axes;
    axes.grids[0] = {1.0, 3.0, 2.0};  // Non-monotonic
    axes.grids[1] = {0.1, 0.2, 0.3};

    auto result = axes.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("monotonic"), std::string::npos);
}

} // namespace
} // namespace mango
```

**Step 2: Run test**

Run: `bazel test //tests:price_table_axes_test`
Expected: FAIL with "No such file: src/option/price_table_axes.hpp"

**Step 3: Create PriceTableAxes header**

**Files:**
- Create: `src/option/price_table_axes.hpp`

```cpp
#pragma once

#include <array>
#include <vector>
#include <string>
#include <expected>
#include <algorithm>

namespace mango {

/// Metadata for N-dimensional price table axes
///
/// Stores grid points and optional axis names for each dimension.
/// All grids must be strictly monotonic increasing.
///
/// @tparam N Number of dimensions (axes)
template <size_t N>
struct PriceTableAxes {
    std::array<std::vector<double>, N> grids;  ///< Grid points per axis
    std::array<std::string, N> names;          ///< Optional names (e.g., "moneyness", "maturity")

    /// Calculate total number of grid points (product of all axis sizes)
    [[nodiscard]] size_t total_points() const noexcept {
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) {
            total *= grids[i].size();
        }
        return total;
    }

    /// Validate all grids are non-empty and strictly monotonic
    ///
    /// @return Empty expected on success, error message on failure
    [[nodiscard]] std::expected<void, std::string> validate() const {
        for (size_t i = 0; i < N; ++i) {
            if (grids[i].empty()) {
                return std::unexpected("Axis " + std::to_string(i) + " is empty");
            }

            // Check strict monotonicity
            for (size_t j = 1; j < grids[i].size(); ++j) {
                if (grids[i][j] <= grids[i][j-1]) {
                    return std::unexpected(
                        "Axis " + std::to_string(i) + " is not strictly monotonic at index " +
                        std::to_string(j));
                }
            }
        }
        return {};
    }

    /// Get shape (number of points per axis)
    [[nodiscard]] std::array<size_t, N> shape() const noexcept {
        std::array<size_t, N> s;
        for (size_t i = 0; i < N; ++i) {
            s[i] = grids[i].size();
        }
        return s;
    }
};

} // namespace mango
```

**Step 4: Add to BUILD.bazel**

**Files:**
- Modify: `src/option/BUILD.bazel`
- Modify: `tests/BUILD.bazel`

Update `src/option/BUILD.bazel`:
```python
cc_library(
    name = "price_table_axes",
    hdrs = ["price_table_axes.hpp"],
    visibility = ["//visibility:public"],
)
```

Update `tests/BUILD.bazel`:
```python
cc_test(
    name = "price_table_axes_test",
    size = "small",
    srcs = ["price_table_axes_test.cc"],
    deps = [
        "//src/option:price_table_axes",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test**

Run: `bazel test //tests:price_table_axes_test`
Expected: PASS (5/5 tests)

**Step 6: Commit**

```bash
git add src/option/price_table_axes.hpp tests/price_table_axes_test.cc src/option/BUILD.bazel tests/BUILD.bazel
git commit -m "Add PriceTableAxes template for N-dimensional grids

Template-based axis metadata supporting any number of dimensions.
Validates strict monotonicity and provides total point calculation.

Part of price table refactor (#XXX)"
```

---

## Phase 3: PriceTensor and mdspan Integration

### Task 3.1: Create PriceTensor test

**Files:**
- Create: `tests/price_tensor_test.cc`

**Step 1: Write failing test**

```cpp
#include <gtest/gtest.h>
#include "mango/option/price_tensor.hpp"
#include "mango/support/memory/aligned_arena.hpp"

namespace mango {
namespace {

TEST(PriceTensorTest, Create2DTensor) {
    auto arena = memory::AlignedArena::create(1024).value();

    auto result = PriceTensor<2>::create({3, 4}, arena);
    ASSERT_TRUE(result.has_value());

    auto tensor = result.value();
    EXPECT_EQ(tensor.view.extent(0), 3);
    EXPECT_EQ(tensor.view.extent(1), 4);
    EXPECT_EQ(tensor.arena, arena);
}

TEST(PriceTensorTest, AccessElements) {
    auto arena = memory::AlignedArena::create(1024).value();
    auto tensor = PriceTensor<2>::create({2, 3}, arena).value();

    // Write via mdspan
    tensor.view(0, 0) = 1.0;
    tensor.view(0, 1) = 2.0;
    tensor.view(1, 2) = 6.0;

    // Read via mdspan
    EXPECT_DOUBLE_EQ(tensor.view(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(tensor.view(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(tensor.view(1, 2), 6.0);
}

TEST(PriceTensorTest, Create4DTensor) {
    auto arena = memory::AlignedArena::create(10000).value();

    auto result = PriceTensor<4>::create({5, 4, 3, 2}, arena);
    ASSERT_TRUE(result.has_value());

    auto tensor = result.value();
    EXPECT_EQ(tensor.view.extent(0), 5);
    EXPECT_EQ(tensor.view.extent(1), 4);
    EXPECT_EQ(tensor.view.extent(2), 3);
    EXPECT_EQ(tensor.view.extent(3), 2);

    // Total elements = 5*4*3*2 = 120
    size_t total = 1;
    for (size_t i = 0; i < 4; ++i) {
        total *= tensor.view.extent(i);
    }
    EXPECT_EQ(total, 120);
}

TEST(PriceTensorTest, ArenaOutOfMemory) {
    auto arena = memory::AlignedArena::create(100).value();  // Too small

    auto result = PriceTensor<3>::create({100, 100, 100}, arena);  // Needs 100*100*100*8 bytes
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("memory"), std::string::npos);
}

} // namespace
} // namespace mango
```

**Step 2: Run test**

Run: `bazel test //tests:price_tensor_test`
Expected: FAIL with "No such file: src/option/price_tensor.hpp"

**Step 3: Create PriceTensor header**

**Files:**
- Create: `src/option/price_tensor.hpp`

```cpp
#pragma once

#include "mango/support/memory/aligned_arena.hpp"
#include <experimental/mdspan>
#include <memory>
#include <expected>
#include <string>
#include <array>

namespace mango {

/// N-dimensional tensor with arena ownership and mdspan view
///
/// Wraps aligned memory allocation with type-safe mdspan access.
/// The arena keeps memory alive via shared_ptr ownership.
///
/// @tparam N Number of dimensions
template <size_t N>
struct PriceTensor {
    std::shared_ptr<memory::AlignedArena> arena;  ///< Owns the memory
    std::experimental::mdspan<double, std::experimental::dextents<size_t, N>> view;  ///< Type-safe view

    /// Create tensor with given shape, allocating from arena
    ///
    /// @param shape Number of elements per dimension
    /// @param arena_ptr Shared pointer to memory arena
    /// @return PriceTensor or error message
    [[nodiscard]] static std::expected<PriceTensor, std::string>
    create(std::array<size_t, N> shape, std::shared_ptr<memory::AlignedArena> arena_ptr) {
        // Calculate total elements
        size_t total = 1;
        for (size_t dim = 0; dim < N; ++dim) {
            total *= shape[dim];
        }

        // Allocate from arena
        double* data = arena_ptr->allocate(total);
        if (!data) {
            return std::unexpected("Insufficient arena memory for tensor of size " +
                                 std::to_string(total * sizeof(double)) + " bytes");
        }

        // Create mdspan view
        PriceTensor tensor;
        tensor.arena = arena_ptr;

        // Construct mdspan with dextents
        std::experimental::dextents<size_t, N> extents;
        if constexpr (N == 1) {
            extents = std::experimental::dextents<size_t, 1>(shape[0]);
        } else if constexpr (N == 2) {
            extents = std::experimental::dextents<size_t, 2>(shape[0], shape[1]);
        } else if constexpr (N == 3) {
            extents = std::experimental::dextents<size_t, 3>(shape[0], shape[1], shape[2]);
        } else if constexpr (N == 4) {
            extents = std::experimental::dextents<size_t, 4>(shape[0], shape[1], shape[2], shape[3]);
        } else if constexpr (N == 5) {
            extents = std::experimental::dextents<size_t, 5>(shape[0], shape[1], shape[2], shape[3], shape[4]);
        } else {
            static_assert(N <= 5, "PriceTensor supports up to 5 dimensions");
        }

        tensor.view = std::experimental::mdspan<double, std::experimental::dextents<size_t, N>>(data, extents);

        return tensor;
    }
};

} // namespace mango
```

**Step 4: Add to BUILD.bazel**

Update `src/option/BUILD.bazel`:
```python
cc_library(
    name = "price_tensor",
    hdrs = ["price_tensor.hpp"],
    deps = [
        "//src/support:aligned_arena",
        "@mdspan",
    ],
    visibility = ["//visibility:public"],
)
```

Update `tests/BUILD.bazel`:
```python
cc_test(
    name = "price_tensor_test",
    size = "small",
    srcs = ["price_tensor_test.cc"],
    deps = [
        "//src/option:price_tensor",
        "//src/support:aligned_arena",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run test**

Run: `bazel test //tests:price_tensor_test`
Expected: PASS (4/4 tests)

**Step 6: Commit**

```bash
git add src/option/price_tensor.hpp tests/price_tensor_test.cc src/option/BUILD.bazel tests/BUILD.bazel
git commit -m "Add PriceTensor with mdspan and arena ownership

Template-based N-dimensional tensor using mdspan for type-safe
indexing and AlignedArena for aligned memory management.

Part of price table refactor (#XXX)"
```

---

## Phase 4: Immutable PriceTableSurface

### Task 4.1: Create Metadata structure test

**Files:**
- Create: `tests/price_table_metadata_test.cc`

**Step 1: Write failing test**

```cpp
#include <gtest/gtest.h>
#include "mango/option/price_table_metadata.hpp"

namespace mango {
namespace {

TEST(PriceTableMetadataTest, DefaultConstruction) {
    PriceTableMetadata meta;
    EXPECT_DOUBLE_EQ(meta.K_ref, 0.0);
    EXPECT_DOUBLE_EQ(meta.dividend_yield, 0.0);
    EXPECT_TRUE(meta.discrete_dividends.empty());
}

TEST(PriceTableMetadataTest, WithDiscreteDividends) {
    PriceTableMetadata meta{
        .K_ref = 100.0,
        .dividend_yield = 0.02,
        .discrete_dividends = {{0.25, 2.50}, {0.75, 2.50}}
    };

    EXPECT_DOUBLE_EQ(meta.K_ref, 100.0);
    EXPECT_DOUBLE_EQ(meta.dividend_yield, 0.02);
    EXPECT_EQ(meta.discrete_dividends.size(), 2);
    EXPECT_DOUBLE_EQ(meta.discrete_dividends[0].first, 0.25);
    EXPECT_DOUBLE_EQ(meta.discrete_dividends[0].second, 2.50);
}

} // namespace
} // namespace mango
```

**Step 2: Run test**

Run: `bazel test //tests:price_table_metadata_test`
Expected: FAIL

**Step 3: Create Metadata header**

**Files:**
- Create: `src/option/price_table_metadata.hpp`

```cpp
#pragma once

#include <vector>
#include <utility>

namespace mango {

/// Metadata for price table surface
///
/// Stores reference strike, dividend information, and discrete dividend schedule.
struct PriceTableMetadata {
    double K_ref = 0.0;                                     ///< Reference strike price
    double dividend_yield = 0.0;                            ///< Continuous dividend yield
    std::vector<std::pair<double, double>> discrete_dividends;  ///< (time, amount) pairs
};

} // namespace mango
```

**Step 4: Add to BUILD and run**

Update `src/option/BUILD.bazel`:
```python
cc_library(
    name = "price_table_metadata",
    hdrs = ["price_table_metadata.hpp"],
    visibility = ["//visibility:public"],
)
```

Run: `bazel test //tests:price_table_metadata_test`
Expected: PASS

**Step 5: Commit**

```bash
git add src/option/price_table_metadata.hpp tests/price_table_metadata_test.cc
git commit -m "Add PriceTableMetadata structure

Stores reference strike, dividend yield, and discrete dividend
schedule for price table surfaces.

Part of price table refactor (#XXX)"
```

### Task 4.2: Create PriceTableSurface test

**Files:**
- Create: `tests/price_table_surface_test.cc`

**Step 1: Write failing test**

```cpp
#include <gtest/gtest.h>
#include "mango/option/price_table_surface.hpp"
#include "mango/support/memory/aligned_arena.hpp"

namespace mango {
namespace {

TEST(PriceTableSurfaceTest, Build2DSurface) {
    PriceTableAxes<2> axes;
    axes.grids[0] = {0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5, 1.0};
    axes.names[0] = "moneyness";
    axes.names[1] = "maturity";

    // 3x3 = 9 coefficients (row-major: m varies fastest)
    std::vector<double> coeffs = {
        1.0, 2.0, 3.0,  // tau=0.1
        4.0, 5.0, 6.0,  // tau=0.5
        7.0, 8.0, 9.0   // tau=1.0
    };

    PriceTableMetadata meta{.K_ref = 100.0, .dividend_yield = 0.02};

    auto result = PriceTableSurface<2>::build(std::move(axes), std::move(coeffs), meta);
    ASSERT_TRUE(result.has_value());

    auto surface = result.value();
    EXPECT_EQ(surface->axes().grids[0].size(), 3);
    EXPECT_DOUBLE_EQ(surface->metadata().K_ref, 100.0);
}

TEST(PriceTableSurfaceTest, ValueInterpolation) {
    PriceTableAxes<2> axes;
    axes.grids[0] = {0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5, 1.0};

    // Simple linear coefficients for testing
    std::vector<double> coeffs(9);
    for (size_t i = 0; i < 9; ++i) {
        coeffs[i] = static_cast<double>(i + 1);
    }

    PriceTableMetadata meta{.K_ref = 100.0};
    auto surface = PriceTableSurface<2>::build(std::move(axes), std::move(coeffs), meta).value();

    // Query at grid point (should match coefficient)
    double val = surface->value({0.9, 0.1});
    EXPECT_NEAR(val, 1.0, 1e-10);
}

TEST(PriceTableSurfaceTest, RejectInvalidCoefficients) {
    PriceTableAxes<2> axes;
    axes.grids[0] = {0.9, 1.0, 1.1};  // 3 points
    axes.grids[1] = {0.1, 0.5};       // 2 points

    std::vector<double> coeffs = {1.0, 2.0};  // Only 2, need 3*2=6

    PriceTableMetadata meta{.K_ref = 100.0};
    auto result = PriceTableSurface<2>::build(std::move(axes), std::move(coeffs), meta);

    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("size"), std::string::npos);
}

} // namespace
} // namespace mango
```

**Step 2: Run test**

Run: `bazel test //tests:price_table_surface_test`
Expected: FAIL

**Step 3: Create PriceTableSurface header (simplified for now)**

**Files:**
- Create: `src/option/price_table_surface.hpp`

```cpp
#pragma once

#include "mango/option/price_table_axes.hpp"
#include "mango/option/price_table_tensor.hpp"
#include "mango/option/price_table_metadata.hpp"
#include "mango/math/bspline_nd.hpp"
#include <memory>
#include <expected>
#include <string>

namespace mango {

/// Immutable N-dimensional price surface with B-spline interpolation
///
/// Provides fast interpolation queries and partial derivatives.
/// Thread-safe after construction.
///
/// @tparam N Number of dimensions
template <size_t N>
class PriceTableSurface {
public:
    /// Build surface from axes, coefficients, and metadata
    ///
    /// @param axes Grid points and names for each dimension
    /// @param coeffs Flattened B-spline coefficients (row-major)
    /// @param metadata Reference strike, dividends, etc.
    /// @return Shared pointer to surface or error message
    [[nodiscard]] static std::expected<std::shared_ptr<const PriceTableSurface>, std::string>
    build(PriceTableAxes<N> axes, std::vector<double> coeffs, PriceTableMetadata metadata);

    /// Access axes
    [[nodiscard]] const PriceTableAxes<N>& axes() const noexcept { return axes_; }

    /// Access metadata
    [[nodiscard]] const PriceTableMetadata& metadata() const noexcept { return meta_; }

    /// Evaluate price at query point
    ///
    /// @param coords N-dimensional coordinates
    /// @return Interpolated value
    [[nodiscard]] double value(const std::array<double, N>& coords) const;

    /// Partial derivative along specified axis
    ///
    /// @param axis Axis index (0 to N-1)
    /// @param coords N-dimensional coordinates
    /// @return Partial derivative estimate
    [[nodiscard]] double partial(size_t axis, const std::array<double, N>& coords) const;

private:
    PriceTableSurface(PriceTableAxes<N> axes, PriceTableMetadata metadata,
                     std::unique_ptr<BSplineND<double, N>> spline);

    PriceTableAxes<N> axes_;
    PriceTableMetadata meta_;
    std::unique_ptr<BSplineND<double, N>> spline_;
};

} // namespace mango
```

**Step 4: Create implementation**

**Files:**
- Create: `src/option/price_table_surface.cpp`

```cpp
#include "mango/option/price_table_surface.hpp"
#include "mango/math/bspline_nd.hpp"

namespace mango {

template <size_t N>
PriceTableSurface<N>::PriceTableSurface(
    PriceTableAxes<N> axes,
    PriceTableMetadata metadata,
    std::unique_ptr<BSplineND<double, N>> spline)
    : axes_(std::move(axes))
    , meta_(std::move(metadata))
    , spline_(std::move(spline)) {}

template <size_t N>
std::expected<std::shared_ptr<const PriceTableSurface<N>>, std::string>
PriceTableSurface<N>::build(
    PriceTableAxes<N> axes,
    std::vector<double> coeffs,
    PriceTableMetadata metadata)
{
    // Validate axes
    if (auto valid = axes.validate(); !valid.has_value()) {
        return std::unexpected(valid.error());
    }

    // Check coefficient size matches axes
    size_t expected_size = axes.total_points();
    if (coeffs.size() != expected_size) {
        return std::unexpected(
            "Coefficient size " + std::to_string(coeffs.size()) +
            " does not match axes shape (expected " +
            std::to_string(expected_size) + ")");
    }

    // Create knot sequences for clamped cubic B-splines
    typename BSplineND<double, N>::KnotArray knots;
    for (size_t dim = 0; dim < N; ++dim) {
        const auto& grid = axes.grids[dim];
        size_t n = grid.size();

        // Clamped knots: repeat first/last knot 4 times
        knots[dim].resize(n + 4);
        for (size_t i = 0; i < 4; ++i) {
            knots[dim][i] = grid.front();
            knots[dim][n + i] = grid.back();
        }
        for (size_t i = 0; i < n; ++i) {
            knots[dim][i + 4] = grid[i];
        }
    }

    // Create BSplineND
    typename BSplineND<double, N>::GridArray grids_copy;
    for (size_t dim = 0; dim < N; ++dim) {
        grids_copy[dim] = axes.grids[dim];
    }

    auto spline_result = BSplineND<double, N>::create(
        std::move(grids_copy),
        std::move(knots),
        std::move(coeffs));

    if (!spline_result.has_value()) {
        return std::unexpected("Failed to create B-spline: " + spline_result.error());
    }

    auto spline = std::make_unique<BSplineND<double, N>>(std::move(spline_result.value()));

    auto surface = std::shared_ptr<const PriceTableSurface<N>>(
        new PriceTableSurface<N>(std::move(axes), std::move(metadata), std::move(spline)));

    return surface;
}

template <size_t N>
double PriceTableSurface<N>::value(const std::array<double, N>& coords) const {
    return spline_->eval(coords);
}

template <size_t N>
double PriceTableSurface<N>::partial(size_t axis, const std::array<double, N>& coords) const {
    // Simple finite difference for now (can optimize later with analytic derivatives)
    constexpr double h = 1e-8;

    std::array<double, N> coords_plus = coords;
    std::array<double, N> coords_minus = coords;

    coords_plus[axis] += h;
    coords_minus[axis] -= h;

    double f_plus = spline_->eval(coords_plus);
    double f_minus = spline_->eval(coords_minus);

    return (f_plus - f_minus) / (2.0 * h);
}

// Explicit template instantiations
template class PriceTableSurface<2>;
template class PriceTableSurface<3>;
template class PriceTableSurface<4>;
template class PriceTableSurface<5>;

} // namespace mango
```

**Step 5: Update BUILD and run**

Update `src/option/BUILD.bazel`:
```python
cc_library(
    name = "price_table_surface",
    srcs = ["price_table_surface.cpp"],
    hdrs = ["price_table_surface.hpp"],
    deps = [
        ":price_table_axes",
        ":price_table_tensor",
        ":price_table_metadata",
        "//src/math:bspline_nd",
    ],
    visibility = ["//visibility:public"],
)
```

Run: `bazel test //tests:price_table_surface_test`
Expected: PASS

**Step 6: Commit**

```bash
git add src/option/price_table_surface.* tests/price_table_surface_test.cc
git commit -m "Add immutable PriceTableSurface template

Immutable N-dimensional price surface with B-spline interpolation.
Provides value() and partial() queries with thread-safe access.

Part of price table refactor (#XXX)"
```

---

## Phase 5: PriceTableConfig and Builder Skeleton

### Task 5.1: Create PriceTableConfig test

**Files:**
- Create: `tests/price_table_config_test.cc`

**Step 1: Write failing test**

```cpp
#include <gtest/gtest.h>
#include "mango/option/price_table_config.hpp"

namespace mango {
namespace {

TEST(PriceTableConfigTest, DefaultValues) {
    PriceTableConfig config;
    EXPECT_EQ(config.option_type, OptionType::PUT);
    EXPECT_EQ(config.n_time, 1000);
    EXPECT_DOUBLE_EQ(config.dividend_yield, 0.0);
    EXPECT_TRUE(config.discrete_dividends.empty());
}

TEST(PriceTableConfigTest, WithDiscreteDividends) {
    PriceTableConfig config{
        .option_type = OptionType::CALL,
        .n_time = 500,
        .dividend_yield = 0.01,
        .discrete_dividends = {{0.25, 2.0}, {0.75, 2.0}}
    };

    EXPECT_EQ(config.option_type, OptionType::CALL);
    EXPECT_EQ(config.discrete_dividends.size(), 2);
}

} // namespace
} // namespace mango
```

**Step 2: Create config header**

**Files:**
- Create: `src/option/price_table_config.hpp`

```cpp
#pragma once

#include "mango/option/american_option.hpp"  // For OptionType
#include "mango/pde/core/grid_spec.hpp"
#include <vector>
#include <utility>
#include <optional>

namespace mango {

/// Configuration for price table pre-computation
struct PriceTableConfig {
    OptionType option_type = OptionType::PUT;  ///< Option type (call/put)
    GridSpec<double> grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value();  ///< Grid for PDE solves
    size_t n_time = 1000;                      ///< Time steps for TR-BDF2
    double dividend_yield = 0.0;               ///< Continuous dividend yield
    std::vector<std::pair<double, double>> discrete_dividends;  ///< (time, amount) schedule
};

} // namespace mango
```

**Step 3: Run and commit**

Run: `bazel test //tests:price_table_config_test`

```bash
git add src/option/price_table_config.hpp tests/price_table_config_test.cc
git commit -m "Add PriceTableConfig structure

Configuration for price table builder including grid specs,
time steps, and discrete dividend schedule.

Part of price table refactor (#XXX)"
```

### Task 5.2: Create PriceTableBuilder skeleton

**Files:**
- Create: `tests/price_table_builder_test.cc`

**Step 1: Write minimal test**

```cpp
#include <gtest/gtest.h>
#include "mango/option/price_table_builder.hpp"

namespace mango {
namespace {

TEST(PriceTableBuilderTest, ConstructFromConfig) {
    PriceTableConfig config;
    PriceTableBuilder<4> builder(config);

    // Just verify construction succeeds
    SUCCEED();
}

TEST(PriceTableBuilderTest, BuildEmpty4DSurface) {
    PriceTableConfig config;
    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5};
    axes.grids[2] = {0.15, 0.25};
    axes.grids[3] = {0.02, 0.05};

    // This will fail until we implement the pipeline
    // For now, just verify it returns an error
    auto result = builder.build(axes);
    EXPECT_FALSE(result.has_value());  // Not implemented yet
}

} // namespace
} // namespace mango
```

**Step 2: Create builder header skeleton**

**Files:**
- Create: `src/option/price_table_builder.hpp`

```cpp
#pragma once

#include "mango/option/price_table_config.hpp"
#include "mango/option/price_table_axes.hpp"
#include "mango/option/price_table_surface.hpp"
#include "mango/option/american_option.hpp"
#include <expected>
#include <string>

namespace mango {

/// Builder for N-dimensional price table surfaces
///
/// Orchestrates PDE solves across grid points, fits B-spline coefficients,
/// and constructs immutable PriceTableSurface.
///
/// @tparam N Number of dimensions
template <size_t N>
class PriceTableBuilder {
public:
    /// Construct builder with configuration
    explicit PriceTableBuilder(PriceTableConfig config);

    /// Build price table surface
    ///
    /// @param axes Grid points for each dimension
    /// @return Immutable surface or error message
    [[nodiscard]] std::expected<std::shared_ptr<const PriceTableSurface<N>>, std::string>
    build(const PriceTableAxes<N>& axes);

private:
    PriceTableConfig config_;
};

} // namespace mango
```

**Step 3: Create implementation stub**

**Files:**
- Create: `src/option/price_table_builder.cpp`

```cpp
#include "mango/option/price_table_builder.hpp"

namespace mango {

template <size_t N>
PriceTableBuilder<N>::PriceTableBuilder(PriceTableConfig config)
    : config_(std::move(config)) {}

template <size_t N>
std::expected<std::shared_ptr<const PriceTableSurface<N>>, std::string>
PriceTableBuilder<N>::build(const PriceTableAxes<N>& axes) {
    // TODO: Implement pipeline
    return std::unexpected("PriceTableBuilder::build() not yet implemented");
}

// Explicit instantiations
template class PriceTableBuilder<2>;
template class PriceTableBuilder<3>;
template class PriceTableBuilder<4>;
template class PriceTableBuilder<5>;

} // namespace mango
```

**Step 4: Run and commit**

Run: `bazel test //tests:price_table_builder_test`

```bash
git add src/option/price_table_builder.* tests/price_table_builder_test.cc
git commit -m "Add PriceTableBuilder skeleton

Template-based builder following config→builder→surface pattern.
Pipeline implementation to follow in subsequent commits.

Part of price table refactor (#XXX)"
```

---

## Phase 6: Recursion Helpers

### Task 6.1: Implement for_each_axis_index

**Files:**
- Create: `tests/recursion_helpers_test.cc`
- Create: `src/option/recursion_helpers.hpp`

**Step 1: Write test for for_each_axis_index**

```cpp
#include <gtest/gtest.h>
#include "mango/option/recursion_helpers.hpp"
#include <vector>

namespace mango {
namespace {

TEST(RecursionHelpersTest, ForEachAxisIndex2D) {
    PriceTableAxes<2> axes;
    axes.grids[0] = {0.9, 1.0};       // 2 points
    axes.grids[1] = {0.1, 0.5, 1.0};  // 3 points

    std::vector<std::array<size_t, 2>> indices;

    for_each_axis_index<0>(axes, [&](const std::array<size_t, 2>& idx) {
        indices.push_back(idx);
    });

    // Should have 2*3 = 6 combinations
    EXPECT_EQ(indices.size(), 6);

    // Check first and last
    EXPECT_EQ(indices[0][0], 0);
    EXPECT_EQ(indices[0][1], 0);
    EXPECT_EQ(indices[5][0], 1);
    EXPECT_EQ(indices[5][1], 2);
}

TEST(RecursionHelpersTest, ForEachAxisIndex4D) {
    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0};       // 2
    axes.grids[1] = {0.1, 0.5};       // 2
    axes.grids[2] = {0.15, 0.25};     // 2
    axes.grids[3] = {0.02};           // 1

    size_t count = 0;
    for_each_axis_index<0>(axes, [&](const std::array<size_t, 4>& idx) {
        ++count;
    });

    EXPECT_EQ(count, 2 * 2 * 2 * 1);  // 8 combinations
}

} // namespace
} // namespace mango
```

**Step 2: Implement recursion_helpers.hpp**

```cpp
#pragma once

#include "mango/option/price_table_axes.hpp"
#include <array>
#include <functional>

namespace mango {

/// Recursively iterate over all combinations of axis indices
///
/// Calls func for every combination of indices across N dimensions.
/// Uses compile-time recursion to unroll loops.
///
/// @tparam Axis Current axis (0 to N-1)
/// @tparam N Number of dimensions
/// @tparam Func Callable accepting std::array<size_t, N>
template <size_t Axis, size_t N, typename Func>
void for_each_axis_index_impl(
    const PriceTableAxes<N>& axes,
    std::array<size_t, N>& indices,
    Func&& func)
{
    if constexpr (Axis == N) {
        // Base case: all axes filled, call function
        func(indices);
    } else {
        // Recursive case: iterate over current axis
        for (size_t i = 0; i < axes.grids[Axis].size(); ++i) {
            indices[Axis] = i;
            for_each_axis_index_impl<Axis + 1>(axes, indices, std::forward<Func>(func));
        }
    }
}

/// Public entry point for axis index iteration
///
/// @tparam StartAxis Starting axis (usually 0)
/// @tparam N Number of dimensions
/// @tparam Func Callable accepting std::array<size_t, N>
template <size_t StartAxis, size_t N, typename Func>
void for_each_axis_index(const PriceTableAxes<N>& axes, Func&& func) {
    std::array<size_t, N> indices{};
    for_each_axis_index_impl<StartAxis>(axes, indices, std::forward<Func>(func));
}

} // namespace mango
```

**Step 3: Run and commit**

Run: `bazel test //tests:recursion_helpers_test`

```bash
git add src/option/recursion_helpers.hpp tests/recursion_helpers_test.cc
git commit -m "Add for_each_axis_index recursion helper

Compile-time recursive iteration over N-dimensional grid indices.
Enables dimension-agnostic batch parameter generation.

Part of price table refactor (#XXX)"
```

---

## Phase 7: Builder Pipeline Implementation

### Task 7.1: Implement make_batch

**Files:**
- Modify: `src/option/price_table_builder.cpp`
- Modify: `tests/price_table_builder_test.cc`

**Step 1: Add test for make_batch**

```cpp
TEST(PriceTableBuilderTest, MakeBatch4D) {
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {{0.25, 1.0}}
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0};      // moneyness: 2 points
    axes.grids[1] = {0.1, 0.5};      // maturity: 2 points
    axes.grids[2] = {0.20};          // volatility: 1 point
    axes.grids[3] = {0.05};          // rate: 1 point

    // Should create 2*2*1*1 = 4 option parameter sets
    auto batch = builder.make_batch_for_testing(axes, 100.0);

    EXPECT_EQ(batch.size(), 4);

    // Check first parameter set
    EXPECT_DOUBLE_EQ(batch[0].spot, 90.0);  // m=0.9, K_ref=100 => S=90
    EXPECT_DOUBLE_EQ(batch[0].strike, 100.0);
    EXPECT_DOUBLE_EQ(batch[0].maturity, 0.1);
    EXPECT_DOUBLE_EQ(batch[0].volatility, 0.20);
    EXPECT_DOUBLE_EQ(batch[0].rate, 0.05);
    EXPECT_DOUBLE_EQ(batch[0].continuous_dividend_yield, 0.02);

    // Check discrete dividends were copied
    EXPECT_EQ(batch[0].discrete_dividends.size(), 1);
    EXPECT_DOUBLE_EQ(batch[0].discrete_dividends[0].first, 0.25);
}
```

**Step 2: Implement make_batch in builder**

Add to `price_table_builder.hpp`:
```cpp
private:
    /// Generate batch of AmericanOptionParams from axes
    std::vector<AmericanOptionParams> make_batch(
        const PriceTableAxes<N>& axes, double K_ref) const;

public:  // For testing
    std::vector<AmericanOptionParams> make_batch_for_testing(
        const PriceTableAxes<N>& axes, double K_ref) const {
        return make_batch(axes, K_ref);
    }
```

Add to `price_table_builder.cpp`:
```cpp
template <size_t N>
std::vector<AmericanOptionParams>
PriceTableBuilder<N>::make_batch(const PriceTableAxes<N>& axes, double K_ref) const {
    static_assert(N >= 4, "PriceTableBuilder requires at least 4 dimensions");

    std::vector<AmericanOptionParams> batch;
    batch.reserve(axes.total_points());

    // Iterate over all grid point combinations
    for_each_axis_index<0>(axes, [&](const std::array<size_t, N>& indices) {
        double m = axes.grids[0][indices[0]];       // moneyness
        double tau = axes.grids[1][indices[1]];     // maturity
        double sigma = axes.grids[2][indices[2]];   // volatility
        double r = axes.grids[3][indices[3]];       // rate

        // Convert moneyness to spot: S = m * K_ref
        double spot = m * K_ref;

        AmericanOptionParams params{
            .strike = K_ref,
            .spot = spot,
            .maturity = tau,
            .volatility = sigma,
            .rate = r,
            .continuous_dividend_yield = config_.dividend_yield,
            .option_type = config_.option_type,
            .discrete_dividends = config_.discrete_dividends
        };

        batch.push_back(params);
    });

    return batch;
}
```

**Step 3: Run and commit**

Run: `bazel test //tests:price_table_builder_test`

```bash
git add src/option/price_table_builder.* tests/price_table_builder_test.cc
git commit -m "Implement make_batch for price table builder

Generate AmericanOptionParams from N-dimensional axes using
for_each_axis_index helper. Includes discrete dividend schedule.

Part of price table refactor (#XXX)"
```

---

## Remaining Phases Summary

Due to space constraints, I'll provide a high-level outline for the remaining phases:

### Phase 8: Solve Batch Integration
- Wire `make_batch` → `BatchAmericanOptionSolver`
- Extract results into `PriceTensor<N>`
- Handle snapshot time registration

### Phase 9: B-Spline Fitting
- Integrate `BSplineNDSeparable` for coefficient fitting
- Extract coefficients as flat vector
- Pass to `PriceTableSurface::build()`

### Phase 10: End-to-End Integration
- Complete `PriceTableBuilder::build()` pipeline
- Add integration test with 4D grid
- Verify interpolation accuracy

### Phase 11: Migration
- Create migration guide in `docs/`
- Add deprecation warnings to old API
- Update existing examples

### Phase 12: Documentation and Cleanup
- Update CLAUDE.md with new API patterns
- Add usage examples
- Performance benchmarks vs old implementation

---

## Execution Handoff

Plan saved to `docs/plans/2025-11-23-price-table-refactor.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
