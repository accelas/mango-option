# Chebyshev Interpolant Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a generic N-dimensional Chebyshev interpolant to `src/math/chebyshev/` with pluggable tensor storage (raw or Tucker-compressed).

**Architecture:** `ChebyshevInterpolant<N, Storage>` is a policy-based template. The interpolant handles Chebyshev node generation, domain clamping, and barycentric weight computation. The `Storage` policy (`RawTensor<N>` or `TuckerTensor<N>`) handles tensor data and contraction. Convenience aliases `ChebyshevTensor<N>` and `ChebyshevTucker<N>` provide clean names.

**Tech Stack:** C++23 templates, Eigen 3.4.0 (for Tucker SVD only), Bazel, GoogleTest

**Reference code:** The experiment branch at `.worktrees/chebyshev-tensor` has working 3D/4D implementations. Key files:
- `src/option/table/dimensionless/chebyshev_nodes.hpp` — node generation (copy as-is)
- `src/option/table/dimensionless/chebyshev_tucker.hpp` — 3D interpolant (generalize)
- `src/option/table/dimensionless/tucker_decomposition.hpp` — 3D HOSVD (generalize)
- `src/option/table/dimensionless/tucker_decomposition_4d.hpp` — 4D HOSVD (generalize)
- `tests/chebyshev_nodes_test.cc` — node tests (adapt include paths)
- `tests/chebyshev_tucker_test.cc` — interpolant tests (adapt to new API)

---

### Task 1: Add Eigen dependency and chebyshev_nodes

**Files:**
- Modify: `third_party/deps.bzl` — add Eigen http_archive
- Modify: `MODULE.bazel:51` — add `"eigen"` to `use_repo`
- Create: `src/math/chebyshev/BUILD.bazel`
- Create: `src/math/chebyshev/chebyshev_nodes.hpp` — copy from experiment branch
- Create: `tests/chebyshev_nodes_test.cc` — adapt from experiment branch
- Modify: `tests/BUILD.bazel` — add test target

**Step 1: Add Eigen to `third_party/deps.bzl`**

Add inside `_non_bcr_deps_impl`, after the mdspan block:

```python
    # Eigen 3.4.0 (header-only linear algebra)
    # Used by Chebyshev-Tucker for SVD in tensor decomposition
    http_archive(
        name = "eigen",
        urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"],
        strip_prefix = "eigen-3.4.0",
        sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
        build_file_content = """
cc_library(
    name = "eigen",
    hdrs = glob(["Eigen/**", "unsupported/Eigen/**"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)
""",
    )
```

**Step 2: Update `MODULE.bazel`**

Change line 51 from:
```python
use_repo(non_bcr_deps, "mdspan")
```
to:
```python
use_repo(non_bcr_deps, "mdspan", "eigen")
```

**Step 3: Create `src/math/chebyshev/chebyshev_nodes.hpp`**

Copy from `.worktrees/chebyshev-tensor/src/option/table/dimensionless/chebyshev_nodes.hpp` verbatim. The file is dimension-agnostic and needs no changes.

**Step 4: Create `src/math/chebyshev/BUILD.bazel`**

```python
# SPDX-License-Identifier: MIT
cc_library(
    name = "chebyshev_nodes",
    hdrs = ["chebyshev_nodes.hpp"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/math/chebyshev",
    include_prefix = "mango/math/chebyshev",
)
```

**Step 5: Create `tests/chebyshev_nodes_test.cc`**

Copy from `.worktrees/chebyshev-tensor/tests/chebyshev_nodes_test.cc`. Change the include from:
```cpp
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
```
to:
```cpp
#include "mango/math/chebyshev/chebyshev_nodes.hpp"
```

**Step 6: Add test target to `tests/BUILD.bazel`**

```python
cc_test(
    name = "chebyshev_nodes_test",
    size = "small",
    srcs = ["chebyshev_nodes_test.cc"],
    deps = [
        "//src/math/chebyshev:chebyshev_nodes",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 7: Run tests**

Run: `bazel test //tests:chebyshev_nodes_test --test_output=all`
Expected: 13 tests PASS

**Step 8: Run full test suite**

Run: `bazel test //...`
Expected: 129 tests pass (116 existing + 13 new)

**Step 9: Commit**

```bash
git add third_party/deps.bzl MODULE.bazel \
    src/math/chebyshev/BUILD.bazel src/math/chebyshev/chebyshev_nodes.hpp \
    tests/chebyshev_nodes_test.cc tests/BUILD.bazel
git commit -m "Add chebyshev_nodes and Eigen dependency"
```

---

### Task 2: RawTensor<N> storage policy

**Files:**
- Create: `src/math/chebyshev/raw_tensor.hpp`
- Modify: `src/math/chebyshev/BUILD.bazel` — add target
- Create: `tests/chebyshev_interpolant_test.cc` — start with RawTensor tests
- Modify: `tests/BUILD.bazel` — add test target

**Step 1: Create `src/math/chebyshev/raw_tensor.hpp`**

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace mango {

/// Raw (uncompressed) N-dimensional tensor storage.
/// Stores all values in row-major order. Contraction iterates
/// all elements weighted by the product of per-axis coefficients.
template <size_t N>
class RawTensor {
public:
    static RawTensor build(std::vector<double> values,
                           const std::array<size_t, N>& shape) {
        RawTensor t;
        t.values_ = std::move(values);
        t.shape_ = shape;
        return t;
    }

    /// Contract with per-axis coefficient vectors.
    /// coeffs[d] has length shape_[d].
    [[nodiscard]] double
    contract(const std::array<std::vector<double>, N>& coeffs) const {
        // Iterate all elements via flat index, compute N-dim subscript,
        // multiply by product of coeffs[d][subscript[d]]
        size_t total = values_.size();
        double result = 0.0;

        // Precompute strides: stride[d] = product of shape[d+1..N-1]
        std::array<size_t, N> strides{};
        strides[N - 1] = 1;
        for (int d = static_cast<int>(N) - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape_[d + 1];
        }

        for (size_t flat = 0; flat < total; ++flat) {
            double weight = values_[flat];
            size_t remaining = flat;
            for (size_t d = 0; d < N; ++d) {
                size_t idx = remaining / strides[d];
                remaining %= strides[d];
                weight *= coeffs[d][idx];
            }
            result += weight;
        }
        return result;
    }

    [[nodiscard]] size_t compressed_size() const {
        return values_.size();
    }

    [[nodiscard]] const std::array<size_t, N>& shape() const { return shape_; }

private:
    std::vector<double> values_;
    std::array<size_t, N> shape_{};
};

}  // namespace mango
```

**Step 2: Add BUILD target**

Add to `src/math/chebyshev/BUILD.bazel`:

```python
cc_library(
    name = "raw_tensor",
    hdrs = ["raw_tensor.hpp"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/math/chebyshev",
    include_prefix = "mango/math/chebyshev",
)
```

**Step 3: Create `tests/chebyshev_interpolant_test.cc` with RawTensor unit test**

```cpp
// SPDX-License-Identifier: MIT
#include "mango/math/chebyshev/raw_tensor.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

TEST(RawTensorTest, Contract2DIdentityWeights) {
    // 2x3 tensor: [[1,2,3],[4,5,6]]
    // Contract with identity weights [1,0] x [0,1,0] => element (0,1) = 2
    RawTensor<2> t = RawTensor<2>::build({1, 2, 3, 4, 5, 6}, {2, 3});
    std::array<std::vector<double>, 2> coeffs = {
        std::vector<double>{1.0, 0.0},
        std::vector<double>{0.0, 1.0, 0.0},
    };
    EXPECT_NEAR(t.contract(coeffs), 2.0, 1e-15);
}

TEST(RawTensorTest, Contract3DUniform) {
    // 2x2x2 tensor of all 1s, uniform weights [0.5,0.5] per axis
    // Result = 8 * 1.0 * 0.5^3 = 1.0
    std::vector<double> vals(8, 1.0);
    RawTensor<3> t = RawTensor<3>::build(std::move(vals), {2, 2, 2});
    std::array<std::vector<double>, 3> coeffs = {
        std::vector<double>{0.5, 0.5},
        std::vector<double>{0.5, 0.5},
        std::vector<double>{0.5, 0.5},
    };
    EXPECT_NEAR(t.contract(coeffs), 1.0, 1e-15);
}

TEST(RawTensorTest, CompressedSizeEqualsTotal) {
    RawTensor<3> t = RawTensor<3>::build(std::vector<double>(60, 0.0), {3, 4, 5});
    EXPECT_EQ(t.compressed_size(), 60u);
}

}  // namespace
}  // namespace mango
```

**Step 4: Add test target to `tests/BUILD.bazel`**

```python
cc_test(
    name = "chebyshev_interpolant_test",
    size = "small",
    srcs = ["chebyshev_interpolant_test.cc"],
    deps = [
        "//src/math/chebyshev:raw_tensor",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
```

**Step 5: Run tests**

Run: `bazel test //tests:chebyshev_interpolant_test --test_output=all`
Expected: 3 tests PASS

**Step 6: Commit**

```bash
git add src/math/chebyshev/raw_tensor.hpp src/math/chebyshev/BUILD.bazel \
    tests/chebyshev_interpolant_test.cc tests/BUILD.bazel
git commit -m "Add RawTensor<N> storage policy"
```

---

### Task 3: TuckerTensor<N> storage policy

**Files:**
- Create: `src/math/chebyshev/tucker_tensor.hpp`
- Modify: `src/math/chebyshev/BUILD.bazel` — add target
- Modify: `tests/chebyshev_interpolant_test.cc` — add Tucker tests
- Modify: `tests/BUILD.bazel` — add tucker dep

**Step 1: Create `src/math/chebyshev/tucker_tensor.hpp`**

This file contains:
- `TuckerResult<N>` struct (core, factors, shape, ranks)
- `mode_unfold<N>()` — generic mode unfolding via flat-index stride computation
- `tucker_hosvd<N>()` — sequential mode contraction with generic repacking
- `tucker_reconstruct<N>()` — full tensor reconstruction (for testing)
- `TuckerTensor<N>` class with `build()` and `contract()`

Reference the existing 3D (`tucker_decomposition.hpp`) and 4D
(`tucker_decomposition_4d.hpp`) implementations on the experiment branch.
Generalize the nested loops to N dimensions using flat-index iteration
with stride arrays.

The `contract()` method:
```cpp
double contract(const std::array<std::vector<double>, N>& coeffs) const {
    // For each axis d, compute contracted[d] of length ranks[d]:
    //   contracted[d][r] = sum_j coeffs[d][j] * factors[d](j, r)
    // Then iterate core tensor (R0 x R1 x ... x R_{N-1}) and sum:
    //   result += core[flat] * product(contracted[d][r_d])
}
```

Key difference from existing code: the existing `eval()` pre-computes
barycentric weights for each factor column individually. The new design
receives full-length coefficient vectors and does the `U^T * c` multiply
here. Same math, cleaner split.

**Step 2: Add BUILD target**

Add to `src/math/chebyshev/BUILD.bazel`:

```python
cc_library(
    name = "tucker_tensor",
    hdrs = ["tucker_tensor.hpp"],
    deps = ["@eigen//:eigen"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/math/chebyshev",
    include_prefix = "mango/math/chebyshev",
)
```

**Step 3: Add Tucker tests to `tests/chebyshev_interpolant_test.cc`**

```cpp
#include "mango/math/chebyshev/tucker_tensor.hpp"

TEST(TuckerTensorTest, RoundTripSmall3D) {
    // Build 3x3x3 tensor of f(i,j,k) = i + 2*j + 3*k
    std::vector<double> vals(27);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                vals[i*9 + j*3 + k] = i + 2.0*j + 3.0*k;

    auto tucker = TuckerTensor<3>::build(std::move(vals), {3, 3, 3}, 1e-12);

    // Contract with identity weights to recover element (1,2,0)
    std::array<std::vector<double>, 3> coeffs = {
        std::vector<double>{0, 1, 0},
        std::vector<double>{0, 0, 1},
        std::vector<double>{1, 0, 0},
    };
    // Expected: 1 + 2*2 + 3*0 = 5
    EXPECT_NEAR(tucker.contract(coeffs), 5.0, 1e-10);
}

TEST(TuckerTensorTest, CompressesLowRankTensor) {
    // Rank-1 tensor: f(i,j,k) = i * j * k (separable)
    std::vector<double> vals(8*8*8);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            for (size_t k = 0; k < 8; ++k)
                vals[i*64 + j*8 + k] = (i+1.0) * (j+1.0) * (k+1.0);

    auto tucker = TuckerTensor<3>::build(std::move(vals), {8, 8, 8}, 1e-10);

    EXPECT_LT(tucker.compressed_size(), 8u * 8 * 8);
    auto ranks = tucker.ranks();
    for (size_t r : ranks) {
        EXPECT_LE(r, 2u) << "Rank-1 tensor should have rank 1 (plus epsilon)";
    }
}
```

**Step 4: Update test BUILD deps**

Add `"//src/math/chebyshev:tucker_tensor"` to the `chebyshev_interpolant_test` deps.

**Step 5: Run tests**

Run: `bazel test //tests:chebyshev_interpolant_test --test_output=all`
Expected: 5 tests PASS (3 RawTensor + 2 TuckerTensor)

**Step 6: Commit**

```bash
git add src/math/chebyshev/tucker_tensor.hpp src/math/chebyshev/BUILD.bazel \
    tests/chebyshev_interpolant_test.cc tests/BUILD.bazel
git commit -m "Add TuckerTensor<N> storage with generic HOSVD"
```

---

### Task 4: ChebyshevInterpolant<N, Storage>

**Files:**
- Create: `src/math/chebyshev/chebyshev_interpolant.hpp`
- Modify: `src/math/chebyshev/BUILD.bazel` — add target + convenience targets
- Modify: `tests/chebyshev_interpolant_test.cc` — add interpolant tests
- Modify: `tests/BUILD.bazel` — update deps

**Step 1: Create `src/math/chebyshev/chebyshev_interpolant.hpp`**

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace mango {

template <size_t N, typename Storage>
class ChebyshevInterpolant {
public:
    struct Domain {
        std::array<std::array<double, 2>, N> bounds;
    };

    /// Build from pre-computed values on CGL nodes (row-major).
    /// storage_args forwarded to Storage::build().
    template <typename... Args>
    [[nodiscard]] static ChebyshevInterpolant
    build_from_values(std::span<const double> values,
                      const Domain& domain,
                      const std::array<size_t, N>& num_pts,
                      Args&&... storage_args) {
        ChebyshevInterpolant interp;
        interp.domain_ = domain;
        for (size_t d = 0; d < N; ++d) {
            auto [a, b] = domain.bounds[d];
            interp.nodes_[d] = chebyshev_nodes(num_pts[d], a, b);
            interp.weights_[d] = chebyshev_barycentric_weights(num_pts[d]);
        }
        std::vector<double> vals(values.begin(), values.end());
        interp.storage_ = Storage::build(
            std::move(vals), num_pts, std::forward<Args>(storage_args)...);
        return interp;
    }

    /// Build by sampling f on CGL tensor grid.
    template <typename... Args>
    [[nodiscard]] static ChebyshevInterpolant
    build(std::function<double(std::array<double, N>)> f,
          const Domain& domain,
          const std::array<size_t, N>& num_pts,
          Args&&... storage_args) {
        ChebyshevInterpolant interp;
        interp.domain_ = domain;
        for (size_t d = 0; d < N; ++d) {
            auto [a, b] = domain.bounds[d];
            interp.nodes_[d] = chebyshev_nodes(num_pts[d], a, b);
            interp.weights_[d] = chebyshev_barycentric_weights(num_pts[d]);
        }

        // Compute total size and sample
        size_t total = 1;
        for (size_t d = 0; d < N; ++d) total *= num_pts[d];

        // Precompute strides
        std::array<size_t, N> strides{};
        strides[N - 1] = 1;
        for (int d = static_cast<int>(N) - 2; d >= 0; --d)
            strides[d] = strides[d + 1] * num_pts[d + 1];

        std::vector<double> vals(total);
        for (size_t flat = 0; flat < total; ++flat) {
            std::array<double, N> pt{};
            size_t remaining = flat;
            for (size_t d = 0; d < N; ++d) {
                size_t idx = remaining / strides[d];
                remaining %= strides[d];
                pt[d] = interp.nodes_[d][idx];
            }
            vals[flat] = f(pt);
        }

        interp.storage_ = Storage::build(
            std::move(vals), num_pts, std::forward<Args>(storage_args)...);
        return interp;
    }

    /// Evaluate at query point. Out-of-domain queries clamped.
    [[nodiscard]] double eval(const std::array<double, N>& query) const {
        // Clamp
        std::array<double, N> q = query;
        for (size_t d = 0; d < N; ++d)
            q[d] = std::clamp(q[d], domain_.bounds[d][0], domain_.bounds[d][1]);

        // Compute per-axis barycentric coefficient vectors
        auto coeffs = barycentric_coeffs(q);
        return storage_.contract(coeffs);
    }

    /// Partial derivative via central finite difference.
    [[nodiscard]] double partial(size_t axis,
                                  const std::array<double, N>& coords) const {
        double lo = domain_.bounds[axis][0];
        double hi = domain_.bounds[axis][1];
        double h = 1e-6 * (hi - lo);

        auto fwd = coords, bwd = coords;
        fwd[axis] += h;
        bwd[axis] -= h;
        fwd[axis] = std::min(fwd[axis], hi);
        bwd[axis] = std::max(bwd[axis], lo);

        double dh = fwd[axis] - bwd[axis];
        if (dh <= 0.0) return 0.0;
        return (eval(fwd) - eval(bwd)) / dh;
    }

    [[nodiscard]] size_t compressed_size() const {
        return storage_.compressed_size();
    }

    [[nodiscard]] std::array<size_t, N> num_pts() const {
        std::array<size_t, N> result;
        for (size_t d = 0; d < N; ++d) result[d] = nodes_[d].size();
        return result;
    }

    [[nodiscard]] const Domain& domain() const { return domain_; }

private:
    [[nodiscard]] std::array<std::vector<double>, N>
    barycentric_coeffs(const std::array<double, N>& q) const {
        std::array<std::vector<double>, N> coeffs;
        for (size_t d = 0; d < N; ++d) {
            size_t n = nodes_[d].size();
            coeffs[d].assign(n, 0.0);

            // Check node coincidence
            bool at_node = false;
            for (size_t j = 0; j < n; ++j) {
                if (q[d] == nodes_[d][j]) {
                    coeffs[d][j] = 1.0;
                    at_node = true;
                    break;
                }
            }
            if (at_node) continue;

            // Barycentric weights
            double denom = 0.0;
            for (size_t j = 0; j < n; ++j) {
                coeffs[d][j] = weights_[d][j] / (q[d] - nodes_[d][j]);
                denom += coeffs[d][j];
            }
            for (size_t j = 0; j < n; ++j) {
                coeffs[d][j] /= denom;
            }
        }
        return coeffs;
    }

    Domain domain_{};
    std::array<std::vector<double>, N> nodes_;
    std::array<std::vector<double>, N> weights_;
    Storage storage_;
};

/// Chebyshev interpolant with raw (uncompressed) tensor storage.
template <size_t N>
using ChebyshevTensor = ChebyshevInterpolant<N, RawTensor<N>>;

/// Chebyshev interpolant with Tucker-compressed tensor storage.
template <size_t N>
using ChebyshevTucker = ChebyshevInterpolant<N, TuckerTensor<N>>;

}  // namespace mango
```

Note: The `ChebyshevTensor` and `ChebyshevTucker` aliases require the
user to include the relevant storage header. The `chebyshev_interpolant.hpp`
itself only depends on `chebyshev_nodes.hpp`.

**Step 2: Add BUILD targets**

Add to `src/math/chebyshev/BUILD.bazel`:

```python
cc_library(
    name = "chebyshev_interpolant",
    hdrs = ["chebyshev_interpolant.hpp"],
    deps = [":chebyshev_nodes"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/math/chebyshev",
    include_prefix = "mango/math/chebyshev",
)

# Convenience: interpolant + raw storage (no Eigen dep)
cc_library(
    name = "chebyshev_tensor",
    deps = [":chebyshev_interpolant", ":raw_tensor"],
    visibility = ["//visibility:public"],
)

# Convenience: interpolant + Tucker storage (pulls Eigen)
cc_library(
    name = "chebyshev_tucker",
    deps = [":chebyshev_interpolant", ":tucker_tensor"],
    visibility = ["//visibility:public"],
)
```

**Step 3: Add interpolant tests**

Add to `tests/chebyshev_interpolant_test.cc`:

```cpp
#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"
#include "mango/math/chebyshev/tucker_tensor.hpp"

// ===== ChebyshevTensor (raw) tests =====

TEST(ChebyshevTensorTest, ExactForLinear3D) {
    // f(x,y,z) = 2x + 3y - z + 1 is degree 1, exact with >= 2 pts/axis
    auto f = [](std::array<double, 3> p) {
        return 2*p[0] + 3*p[1] - p[2] + 1;
    };
    ChebyshevTensor<3>::Domain dom{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}}};

    auto interp = ChebyshevTensor<3>::build(f, dom, {4, 4, 4});

    for (double x : {-0.5, 0.0, 0.7})
        for (double y : {-0.3, 0.4})
            for (double z : {-0.8, 0.9})
                EXPECT_NEAR(interp.eval({x, y, z}), f({x, y, z}), 1e-12);
}

TEST(ChebyshevTensorTest, ExactForBilinear3D) {
    // f(x,y,z) = x*y + z, degree 2, exact with >= 2 pts/axis
    auto f = [](std::array<double, 3> p) {
        return p[0] * p[1] + p[2];
    };
    ChebyshevTensor<3>::Domain dom{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}}};

    auto interp = ChebyshevTensor<3>::build(f, dom, {4, 4, 4});

    for (double x : {-0.5, 0.0, 0.7})
        for (double y : {-0.3, 0.4})
            for (double z : {-0.8, 0.1, 0.9})
                EXPECT_NEAR(interp.eval({x, y, z}), f({x, y, z}), 1e-11);
}

TEST(ChebyshevTensorTest, SmoothFunctionConverges3D) {
    auto f = [](std::array<double, 3> p) {
        return std::exp(-p[0]*p[0]) * std::sin(p[1]) * std::cos(p[2]*0.5);
    };
    ChebyshevTensor<3>::Domain dom{
        .bounds = {{{-1.0, 1.0}, {0.0, M_PI}, {-1.0, 1.0}}}};

    auto coarse = ChebyshevTensor<3>::build(f, dom, {8, 8, 8});
    auto fine = ChebyshevTensor<3>::build(f, dom, {16, 16, 16});

    std::array<double, 3> pt = {0.5, 1.2, 0.3};
    double exact = f(pt);
    double err_coarse = std::abs(coarse.eval(pt) - exact);
    double err_fine = std::abs(fine.eval(pt) - exact);

    EXPECT_LT(err_fine, err_coarse * 0.01);
    EXPECT_LT(err_fine, 1e-12);
}

TEST(ChebyshevTensorTest, ExactForLinear4D) {
    auto f = [](std::array<double, 4> p) {
        return p[0] + 2*p[1] - 3*p[2] + 0.5*p[3];
    };
    ChebyshevTensor<4>::Domain dom{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}}};

    auto interp = ChebyshevTensor<4>::build(f, dom, {4, 4, 4, 4});

    for (double x : {-0.5, 0.7})
        for (double y : {-0.3, 0.4})
            for (double z : {-0.8, 0.9})
                for (double w : {-0.2, 0.6})
                    EXPECT_NEAR(interp.eval({x, y, z, w}),
                                f({x, y, z, w}), 1e-12);
}

TEST(ChebyshevTensorTest, PartialDerivatives3D) {
    // f(x,y,z) = sin(x)*cos(y)*z
    // df/dx = cos(x)*cos(y)*z
    // df/dy = -sin(x)*sin(y)*z
    // df/dz = sin(x)*cos(y)
    auto f = [](std::array<double, 3> p) {
        return std::sin(p[0]) * std::cos(p[1]) * p[2];
    };
    ChebyshevTensor<3>::Domain dom{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}}};

    auto interp = ChebyshevTensor<3>::build(f, dom, {16, 16, 16});

    for (double x : {-0.5, 0.3, 0.8})
        for (double y : {-0.4, 0.6}) {
            double z = 0.2;
            EXPECT_NEAR(interp.partial(0, {x, y, z}),
                        std::cos(x)*std::cos(y)*z, 1e-5);
            EXPECT_NEAR(interp.partial(1, {x, y, z}),
                        -std::sin(x)*std::sin(y)*z, 1e-5);
            EXPECT_NEAR(interp.partial(2, {x, y, z}),
                        std::sin(x)*std::cos(y), 1e-5);
        }
}

TEST(ChebyshevTensorTest, DomainClamping) {
    auto f = [](std::array<double, 3> p) { return p[0]; };
    ChebyshevTensor<3>::Domain dom{
        .bounds = {{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}}};

    auto interp = ChebyshevTensor<3>::build(f, dom, {8, 8, 8});

    // Query below domain should clamp to 0
    EXPECT_NEAR(interp.eval({-5.0, 0.5, 0.5}), 0.0, 1e-12);
    // Query above domain should clamp to 1
    EXPECT_NEAR(interp.eval({5.0, 0.5, 0.5}), 1.0, 1e-12);
}

// ===== ChebyshevTucker tests =====

TEST(ChebyshevTuckerTest, ExactForBilinear3D) {
    auto f = [](std::array<double, 3> p) {
        return p[0] * p[1] + p[2];
    };
    ChebyshevTucker<3>::Domain dom{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}}};

    auto interp = ChebyshevTucker<3>::build(f, dom, {4, 4, 4}, 1e-12);

    for (double x : {-0.5, 0.0, 0.7})
        for (double y : {-0.3, 0.4})
            for (double z : {-0.8, 0.1, 0.9})
                EXPECT_NEAR(interp.eval({x, y, z}), f({x, y, z}), 1e-10);
}

TEST(ChebyshevTuckerTest, CompressesSmooth3D) {
    auto f = [](std::array<double, 3> p) {
        return std::exp(-p[0]*p[0]) * std::sin(p[1]) * std::cos(p[2]*0.5);
    };
    ChebyshevTucker<3>::Domain dom{
        .bounds = {{{-2.0, 2.0}, {0.0, M_PI}, {-1.0, 1.0}}}};

    auto interp = ChebyshevTucker<3>::build(f, dom, {15, 15, 15}, 1e-8);

    EXPECT_LT(interp.compressed_size(), 15u * 15 * 15);
}

TEST(ChebyshevTuckerTest, ExactForLinear4D) {
    auto f = [](std::array<double, 4> p) {
        return p[0] + 2*p[1] - 3*p[2] + 0.5*p[3];
    };
    ChebyshevTucker<4>::Domain dom{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}}};

    auto interp = ChebyshevTucker<4>::build(f, dom, {4, 4, 4, 4}, 1e-12);

    for (double x : {-0.5, 0.7})
        for (double y : {-0.3, 0.4})
            for (double z : {-0.8, 0.9})
                for (double w : {-0.2, 0.6})
                    EXPECT_NEAR(interp.eval({x, y, z, w}),
                                f({x, y, z, w}), 1e-10);
}

TEST(ChebyshevTuckerTest, BuildFromValuesMatchesBuild) {
    auto f = [](std::array<double, 3> p) {
        return std::sin(p[0]) * std::cos(p[1]) * p[2];
    };
    ChebyshevTucker<3>::Domain dom{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}}};
    std::array<size_t, 3> npts = {8, 8, 8};

    auto from_fn = ChebyshevTucker<3>::build(f, dom, npts, 1e-12);

    // Manually sample the same grid
    auto nodes0 = chebyshev_nodes(8, -1.0, 1.0);
    auto nodes1 = chebyshev_nodes(8, -1.0, 1.0);
    auto nodes2 = chebyshev_nodes(8, -1.0, 1.0);
    std::vector<double> vals(512);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            for (size_t k = 0; k < 8; ++k)
                vals[i*64 + j*8 + k] =
                    f({nodes0[i], nodes1[j], nodes2[k]});

    auto from_vals = ChebyshevTucker<3>::build_from_values(
        vals, dom, npts, 1e-12);

    // Should produce identical results
    for (double x : {-0.5, 0.3})
        for (double y : {-0.4, 0.6})
            for (double z : {-0.7, 0.2})
                EXPECT_NEAR(from_fn.eval({x, y, z}),
                            from_vals.eval({x, y, z}), 1e-14);
}
```

**Step 4: Update test BUILD deps**

Update `chebyshev_interpolant_test` deps to include all three:
```python
    deps = [
        "//src/math/chebyshev:chebyshev_interpolant",
        "//src/math/chebyshev:raw_tensor",
        "//src/math/chebyshev:tucker_tensor",
        "//src/math/chebyshev:chebyshev_nodes",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
```

**Step 5: Run tests**

Run: `bazel test //tests:chebyshev_interpolant_test --test_output=all`
Expected: 14 tests PASS (3 RawTensor + 7 ChebyshevTensor + 4 ChebyshevTucker)

**Step 6: Verify SurfaceInterpolant concept satisfaction**

Add to `tests/chebyshev_interpolant_test.cc`:

```cpp
#include "mango/option/table/surface_concepts.hpp"

static_assert(SurfaceInterpolant<ChebyshevTensor<3>, 3>);
static_assert(SurfaceInterpolant<ChebyshevTensor<4>, 4>);
static_assert(SurfaceInterpolant<ChebyshevTucker<3>, 3>);
static_assert(SurfaceInterpolant<ChebyshevTucker<4>, 4>);
```

Add `"//src/option/table:surface_concepts"` to test deps.

**Step 7: Run full test suite**

Run: `bazel test //...`
Expected: all tests pass

Run: `bazel build //benchmarks/...`
Expected: all benchmarks build

Run: `bazel build //src/python:mango_option`
Expected: builds

**Step 8: Commit**

```bash
git add src/math/chebyshev/chebyshev_interpolant.hpp \
    src/math/chebyshev/BUILD.bazel \
    tests/chebyshev_interpolant_test.cc tests/BUILD.bazel
git commit -m "Add ChebyshevInterpolant<N, Storage> template

Generic N-dimensional Chebyshev interpolant with pluggable
storage. ChebyshevTensor<N> uses raw tensors (no Eigen dep),
ChebyshevTucker<N> adds Tucker compression via HOSVD.
Both satisfy SurfaceInterpolant<T, N>."
```

---

## Verification

1. `bazel test //...` — all tests pass (existing 116 + new ~27)
2. `bazel build //benchmarks/...` — all benchmarks build
3. `bazel build //src/python:mango_option` — Python bindings build
4. No existing code modified except `tests/BUILD.bazel`, `MODULE.bazel`, and `third_party/deps.bzl`
