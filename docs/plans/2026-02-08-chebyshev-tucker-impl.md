# Chebyshev-Tucker 3D Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone Chebyshev-Tucker 3D interpolant for the dimensionless EEP function and benchmark it head-to-head against B-spline interpolation.

**Architecture:** Chebyshev-Gauss-Lobatto nodes with barycentric interpolation, compressed via HOSVD (Tucker decomposition using Eigen SVD). Compared against cubic B-spline (unsegmented and 3-segment) at matched grid sizes. All code lives in `src/option/table/dimensionless/` with tests in `tests/` and benchmarks in `benchmarks/`.

**Tech Stack:** C++23, Bazel with Bzlmod, Eigen 3.4.0 (header-only via http_archive), GoogleTest, existing BatchAmericanOptionSolver + PriceTableSurfaceND<3> + BSplineNDSeparable<3>.

**Worktree:** `/home/kai/work/mango-option/.worktrees/chebyshev-tensor/`

**Design doc:** `docs/plans/2026-02-08-chebyshev-tucker-experiment.md`

**Prerequisite:** Branch is rebased on `experiment/dimensionless-3d` (PR 386). The dimensionless builder, European formula, adaptive builder, and tests are already available.

---

## Task 1: Add Eigen dependency via Bzlmod

Eigen is header-only. Follow the mdspan pattern in `third_party/deps.bzl`.

**Step 1: Modify `third_party/deps.bzl`**

Add after the mdspan `http_archive` block:

```python
    # Eigen 3.4.0 (header-only linear algebra)
    # Used by Chebyshev-Tucker experiment for SVD
    http_archive(
        name = "eigen",
        urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"],
        strip_prefix = "eigen-3.4.0",
        sha256 = "8586084f71f9bde545e7db6b14e1667468f61a3b082793a770e364368c2e759d4",
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

**Step 2: Modify `MODULE.bazel`**

Change line 51 from:

```python
use_repo(non_bcr_deps, "mdspan")
```

to:

```python
use_repo(non_bcr_deps, "mdspan", "eigen")
```

**Step 3: Write a compile test**

Create `tests/eigen_compile_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(EigenCompileTest, SVDWorks) {
    Eigen::MatrixXd A(3, 3);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 10;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    EXPECT_EQ(svd.singularValues().size(), 3);
    EXPECT_GT(svd.singularValues()(0), 0.0);
}
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "eigen_compile_test",
    size = "small",
    srcs = ["eigen_compile_test.cc"],
    deps = [
        "@eigen//:eigen",
        "@googletest//:gtest_main",
    ],
)
```

**Step 4: Build and run**

```bash
bazel test //tests:eigen_compile_test --test_output=all
```

Expected: PASS. If the sha256 is wrong, fetch the tarball and compute it:

```bash
curl -sL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | sha256sum
```

**Step 5: Commit**

```bash
git add third_party/deps.bzl MODULE.bazel tests/eigen_compile_test.cc tests/BUILD.bazel
git commit -m "Add Eigen 3.4.0 dependency for Chebyshev experiment"
```

---

## Task 2: Implement Chebyshev node generation and barycentric weights

Header-only. Generates Chebyshev-Gauss-Lobatto nodes on an arbitrary [a, b] interval and computes barycentric weights.

**Step 1: Write the failing test**

Create `tests/chebyshev_nodes_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

namespace mango {
namespace {

TEST(ChebyshevNodesTest, NodeCountMatchesNumPts) {
    auto nodes = chebyshev_nodes(10, -1.0, 1.0);
    EXPECT_EQ(nodes.size(), 10u);
}

TEST(ChebyshevNodesTest, EndpointsMatchDomain) {
    auto nodes = chebyshev_nodes(11, -2.0, 3.0);
    EXPECT_DOUBLE_EQ(nodes.front(), -2.0);
    EXPECT_DOUBLE_EQ(nodes.back(), 3.0);
}

TEST(ChebyshevNodesTest, NodesAreSortedAscending) {
    auto nodes = chebyshev_nodes(15, 0.0, 1.0);
    for (size_t i = 1; i < nodes.size(); ++i) {
        EXPECT_LT(nodes[i - 1], nodes[i]);
    }
}

TEST(ChebyshevNodesTest, StandardNodesOnMinusOneOne) {
    // CGL nodes: x_j = cos(j*pi/n), j=0..n, reversed to ascending
    auto nodes = chebyshev_nodes(5, -1.0, 1.0);  // degree 4, 5 nodes
    EXPECT_NEAR(nodes[0], -1.0, 1e-15);
    EXPECT_NEAR(nodes[1], -std::cos(M_PI / 4), 1e-15);  // cos(3pi/4) = -cos(pi/4)
    EXPECT_NEAR(nodes[2], 0.0, 1e-15);
    EXPECT_NEAR(nodes[3], std::cos(M_PI / 4), 1e-15);
    EXPECT_NEAR(nodes[4], 1.0, 1e-15);
}

TEST(ChebyshevNodesTest, BarycentricWeightsAlternateSign) {
    auto weights = chebyshev_barycentric_weights(10);
    for (size_t i = 1; i < weights.size(); ++i) {
        // Adjacent weights should have opposite signs
        EXPECT_LT(weights[i - 1] * weights[i], 0.0);
    }
}

TEST(ChebyshevNodesTest, BarycentricEndpointsHalfWeight) {
    // Endpoints get half the weight of interior nodes
    auto weights = chebyshev_barycentric_weights(5);
    // w_0 and w_n should be half magnitude of interior
    EXPECT_NEAR(std::abs(weights[0]) * 2, std::abs(weights[1]), 1e-15);
    EXPECT_NEAR(std::abs(weights[4]) * 2, std::abs(weights[3]), 1e-15);
}

TEST(ChebyshevNodesTest, BarycentricInterpolationExactForPolynomial) {
    // Interpolate f(x) = x^3 - 2x + 1 on [-1, 1] with 5 nodes (degree 4)
    // Should be exact since degree 3 < 4
    size_t num_pts = 5;
    auto nodes = chebyshev_nodes(num_pts, -1.0, 1.0);
    auto weights = chebyshev_barycentric_weights(num_pts);

    auto f = [](double x) { return x * x * x - 2.0 * x + 1.0; };
    std::vector<double> values(num_pts);
    for (size_t i = 0; i < num_pts; ++i) values[i] = f(nodes[i]);

    // Evaluate at off-grid points
    for (double x : {-0.7, -0.3, 0.0, 0.15, 0.8}) {
        double result = chebyshev_interpolate(x, nodes, values, weights);
        EXPECT_NEAR(result, f(x), 1e-13)
            << "Mismatch at x=" << x;
    }
}

TEST(ChebyshevNodesTest, BarycentricExactAtNodes) {
    size_t num_pts = 8;
    auto nodes = chebyshev_nodes(num_pts, 0.0, 5.0);
    auto weights = chebyshev_barycentric_weights(num_pts);

    std::vector<double> values(num_pts);
    for (size_t i = 0; i < num_pts; ++i) values[i] = std::sin(nodes[i]);

    // Should be exact at nodes themselves
    for (size_t i = 0; i < num_pts; ++i) {
        double result = chebyshev_interpolate(nodes[i], nodes, values, weights);
        EXPECT_NEAR(result, values[i], 1e-14);
    }
}

}  // namespace
}  // namespace mango
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "chebyshev_nodes_test",
    size = "small",
    srcs = ["chebyshev_nodes_test.cc"],
    deps = [
        "//src/option/table/dimensionless:chebyshev_nodes",
        "@googletest//:gtest_main",
    ],
)
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:chebyshev_nodes_test --test_output=all
```

Expected: BUILD FAIL (header doesn't exist yet)

**Step 3: Implement `chebyshev_nodes.hpp`**

Create `src/option/table/dimensionless/chebyshev_nodes.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <span>
#include <vector>

namespace mango {

/// Generate num_pts Chebyshev-Gauss-Lobatto nodes on [a, b], sorted ascending.
/// Polynomial degree = num_pts - 1.
[[nodiscard]] inline std::vector<double>
chebyshev_nodes(size_t num_pts, double a, double b) {
    const size_t n = num_pts - 1;
    std::vector<double> nodes(num_pts);
    // CGL nodes on [-1,1]: cos(j*pi/n), j=0..n (descending)
    // Map to [a,b] and reverse for ascending order
    for (size_t j = 0; j <= n; ++j) {
        double t = std::cos(static_cast<double>(j) * M_PI / static_cast<double>(n));
        // Map [-1,1] -> [a,b]: x = (b+a)/2 + (b-a)/2 * t
        nodes[n - j] = (b + a) / 2.0 + (b - a) / 2.0 * t;
    }
    return nodes;
}

/// Barycentric weights for num_pts Chebyshev-Gauss-Lobatto nodes.
/// w_j = (-1)^j * delta_j, where delta_j = 1/2 for endpoints, 1 otherwise.
/// Returned in ascending node order (reversed from standard CGL ordering).
[[nodiscard]] inline std::vector<double>
chebyshev_barycentric_weights(size_t num_pts) {
    const size_t n = num_pts - 1;
    std::vector<double> w(num_pts);
    for (size_t j = 0; j <= n; ++j) {
        // In standard CGL order (descending), w_j = (-1)^j * delta_j
        double sign = (j % 2 == 0) ? 1.0 : -1.0;
        double delta = (j == 0 || j == n) ? 0.5 : 1.0;
        // We store in reversed (ascending) order, so index is n-j
        w[n - j] = sign * delta;
    }
    return w;
}

/// Evaluate barycentric Chebyshev interpolant at point x.
/// nodes: ascending CGL nodes on [a,b] (from chebyshev_nodes())
/// values: function values at nodes
/// weights: barycentric weights (from chebyshev_barycentric_weights())
[[nodiscard]] inline double
chebyshev_interpolate(double x,
                      std::span<const double> nodes,
                      std::span<const double> values,
                      std::span<const double> weights) {
    // Check if x coincides with a node (avoid division by zero)
    for (size_t j = 0; j < nodes.size(); ++j) {
        if (x == nodes[j]) return values[j];
    }

    double numer = 0.0, denom = 0.0;
    for (size_t j = 0; j < nodes.size(); ++j) {
        double term = weights[j] / (x - nodes[j]);
        numer += term * values[j];
        denom += term;
    }
    return numer / denom;
}

}  // namespace mango
```

Add BUILD target in `src/option/table/dimensionless/BUILD.bazel`:

```python
cc_library(
    name = "chebyshev_nodes",
    hdrs = ["chebyshev_nodes.hpp"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option/table/dimensionless",
    include_prefix = "mango/option/table/dimensionless",
)
```

**Step 4: Run tests**

```bash
bazel test //tests:chebyshev_nodes_test --test_output=all
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/option/table/dimensionless/chebyshev_nodes.hpp \
        src/option/table/dimensionless/BUILD.bazel \
        tests/chebyshev_nodes_test.cc tests/BUILD.bazel
git commit -m "Add Chebyshev node generation and barycentric interpolation"
```

---

## Task 3: Implement Tucker decomposition via HOSVD

Uses Eigen for SVD. Input: a flat vector representing a 3D tensor. Output: core tensor + three factor matrices.

**Step 1: Write the failing test**

Create `tests/tucker_decomposition_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless/tucker_decomposition.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

// Helper: fill a rank-1 tensor f(i,j,k) = a_i * b_j * c_k
std::vector<double> rank1_tensor(const std::vector<double>& a,
                                 const std::vector<double>& b,
                                 const std::vector<double>& c) {
    std::vector<double> T(a.size() * b.size() * c.size());
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            for (size_t k = 0; k < c.size(); ++k)
                T[i * b.size() * c.size() + j * c.size() + k] = a[i] * b[j] * c[k];
    return T;
}

TEST(TuckerDecompositionTest, Rank1TensorGivesRank1Core) {
    std::array<size_t, 3> shape = {5, 4, 3};
    auto T = rank1_tensor({1, 2, 3, 4, 5}, {1, 0.5, 0.25, 0.125}, {1, -1, 0.5});

    auto result = tucker_hosvd(T, shape, 1e-8);

    // A rank-1 tensor should decompose to rank (1,1,1)
    EXPECT_EQ(result.ranks[0], 1u);
    EXPECT_EQ(result.ranks[1], 1u);
    EXPECT_EQ(result.ranks[2], 1u);
    EXPECT_EQ(result.core.size(), 1u);  // 1x1x1
}

TEST(TuckerDecompositionTest, ReconstructionMatchesOriginal) {
    // A low-rank tensor: sum of 2 rank-1 terms
    std::array<size_t, 3> shape = {6, 5, 4};
    std::vector<double> T(6 * 5 * 4, 0.0);

    // Term 1
    auto t1 = rank1_tensor({1, 2, 3, 4, 5, 6}, {1, 0.5, 0.25, 0.125, 0.0625}, {1, -1, 0.5, -0.5});
    // Term 2
    auto t2 = rank1_tensor({6, 5, 4, 3, 2, 1}, {0.1, 0.2, 0.3, 0.4, 0.5}, {0.5, 0.5, -0.5, -0.5});

    for (size_t i = 0; i < T.size(); ++i) T[i] = t1[i] + t2[i];

    auto result = tucker_hosvd(T, shape, 1e-10);

    // Reconstruct and compare
    auto reconstructed = tucker_reconstruct(result);
    EXPECT_EQ(reconstructed.size(), T.size());
    for (size_t i = 0; i < T.size(); ++i) {
        EXPECT_NEAR(reconstructed[i], T[i], 1e-10)
            << "Mismatch at index " << i;
    }
}

TEST(TuckerDecompositionTest, TruncationReducesRank) {
    // Full-rank random-ish tensor
    std::array<size_t, 3> shape = {8, 7, 6};
    std::vector<double> T(8 * 7 * 6);
    for (size_t i = 0; i < T.size(); ++i) {
        T[i] = std::sin(static_cast<double>(i) * 0.1) +
                std::cos(static_cast<double>(i) * 0.037);
    }

    auto tight = tucker_hosvd(T, shape, 1e-12);
    auto loose = tucker_hosvd(T, shape, 1e-2);

    // Loose threshold should give smaller or equal ranks
    EXPECT_LE(loose.ranks[0], tight.ranks[0]);
    EXPECT_LE(loose.ranks[1], tight.ranks[1]);
    EXPECT_LE(loose.ranks[2], tight.ranks[2]);

    // Loose reconstruction should still be reasonable
    auto recon = tucker_reconstruct(loose);
    double max_err = 0;
    for (size_t i = 0; i < T.size(); ++i) {
        max_err = std::max(max_err, std::abs(recon[i] - T[i]));
    }
    // With loose truncation, some error is expected but bounded
    EXPECT_LT(max_err, 1.0);
}

TEST(TuckerDecompositionTest, FullRankPreservesExactly) {
    std::array<size_t, 3> shape = {4, 3, 3};
    std::vector<double> T(4 * 3 * 3);
    for (size_t i = 0; i < T.size(); ++i) T[i] = std::sin(i * 0.7) * std::exp(-i * 0.01);

    // Very tight threshold => full rank => exact reconstruction
    auto result = tucker_hosvd(T, shape, 1e-15);
    auto recon = tucker_reconstruct(result);

    for (size_t i = 0; i < T.size(); ++i) {
        EXPECT_NEAR(recon[i], T[i], 1e-12);
    }
}

}  // namespace
}  // namespace mango
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "tucker_decomposition_test",
    size = "small",
    srcs = ["tucker_decomposition_test.cc"],
    deps = [
        "//src/option/table/dimensionless:tucker_decomposition",
        "@googletest//:gtest_main",
    ],
)
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:tucker_decomposition_test --test_output=all
```

Expected: BUILD FAIL

**Step 3: Implement `tucker_decomposition.hpp`**

Create `src/option/table/dimensionless/tucker_decomposition.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <vector>

namespace mango {

/// Result of Tucker (HOSVD) decomposition of a 3D tensor.
struct TuckerResult3D {
    std::vector<double> core;                  ///< Core tensor G, shape (R0, R1, R2)
    std::array<Eigen::MatrixXd, 3> factors;    ///< Factor matrices U_k, shape (N_k, R_k)
    std::array<size_t, 3> shape;               ///< Original tensor shape (N0, N1, N2)
    std::array<size_t, 3> ranks;               ///< Truncated ranks (R0, R1, R2)
};

/// Mode-k unfolding of a 3D tensor stored in row-major order.
/// Returns an Eigen matrix of shape (shape[mode], product of other dims).
inline Eigen::MatrixXd
mode_unfold(const std::vector<double>& T,
            const std::array<size_t, 3>& shape,
            size_t mode) {
    size_t n_rows = shape[mode];
    size_t n_cols = 1;
    for (size_t d = 0; d < 3; ++d)
        if (d != mode) n_cols *= shape[d];

    Eigen::MatrixXd M(n_rows, n_cols);

    for (size_t i0 = 0; i0 < shape[0]; ++i0) {
        for (size_t i1 = 0; i1 < shape[1]; ++i1) {
            for (size_t i2 = 0; i2 < shape[2]; ++i2) {
                size_t flat = i0 * shape[1] * shape[2] + i1 * shape[2] + i2;
                std::array<size_t, 3> idx = {i0, i1, i2};
                size_t row = idx[mode];

                // Column index: linearize remaining indices in order
                size_t col = 0;
                size_t stride = 1;
                for (int d = 2; d >= 0; --d) {
                    if (static_cast<size_t>(d) == mode) continue;
                    col += idx[d] * stride;
                    stride *= shape[d];
                }

                M(row, col) = T[flat];
            }
        }
    }
    return M;
}

/// HOSVD: Higher-Order Singular Value Decomposition for 3D tensors.
///
/// Decomposes T ≈ G ×₀ U₀ ×₁ U₁ ×₂ U₂ where each U_k is truncated
/// by keeping singular values with σ_i/σ_0 >= epsilon.
///
/// @param T Tensor values in row-major order, size shape[0]*shape[1]*shape[2]
/// @param shape Tensor dimensions (N0, N1, N2)
/// @param epsilon Relative singular value threshold for rank truncation
[[nodiscard]] inline TuckerResult3D
tucker_hosvd(const std::vector<double>& T,
             const std::array<size_t, 3>& shape,
             double epsilon) {
    TuckerResult3D result;
    result.shape = shape;

    // Step 1: Compute factor matrices via mode-k SVD
    for (size_t mode = 0; mode < 3; ++mode) {
        Eigen::MatrixXd M = mode_unfold(T, shape, mode);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU);

        const auto& sigma = svd.singularValues();
        double sigma_0 = sigma(0);

        // Determine rank: keep singular values above threshold
        size_t rank = 1;
        for (Eigen::Index i = 1; i < sigma.size(); ++i) {
            if (sigma(i) / sigma_0 >= epsilon) rank++;
            else break;
        }

        result.ranks[mode] = rank;
        result.factors[mode] = svd.matrixU().leftCols(rank);
    }

    // Step 2: Compute core tensor G = T ×₀ U₀ᵀ ×₁ U₁ᵀ ×₂ U₂ᵀ
    // Apply contractions sequentially along each mode
    size_t R0 = result.ranks[0], R1 = result.ranks[1], R2 = result.ranks[2];

    // Start with original tensor as a matrix (mode-0 unfolding)
    // G_0 = U₀ᵀ × unfold_0(T), shape: (R0, N1*N2)
    Eigen::MatrixXd M0 = mode_unfold(T, shape, 0);
    Eigen::MatrixXd G0 = result.factors[0].transpose() * M0;  // (R0, N1*N2)

    // Reshape G0 to (R0, N1, N2), then contract mode-1
    // unfold_1 of (R0, N1, N2) => (N1, R0*N2)
    std::array<size_t, 3> shape1 = {R0, shape[1], shape[2]};
    std::vector<double> G0_vec(R0 * shape[1] * shape[2]);
    for (size_t r = 0; r < R0; ++r)
        for (size_t j = 0; j < shape[1] * shape[2]; ++j)
            G0_vec[r * shape[1] * shape[2] + j] = G0(r, j);

    Eigen::MatrixXd M1 = mode_unfold(G0_vec, shape1, 1);
    Eigen::MatrixXd G1 = result.factors[1].transpose() * M1;  // (R1, R0*N2)

    // Reshape G1 to (R0, R1, N2), then contract mode-2
    std::array<size_t, 3> shape2 = {R0, R1, shape[2]};
    std::vector<double> G1_vec(R0 * R1 * shape[2]);
    for (size_t r1 = 0; r1 < R1; ++r1)
        for (size_t j = 0; j < R0 * shape[2]; ++j)
            G1_vec[j / shape[2] * R1 * shape[2] + r1 * shape[2] + j % shape[2]] = G1(r1, j);

    Eigen::MatrixXd M2 = mode_unfold(G1_vec, shape2, 2);
    Eigen::MatrixXd G2 = result.factors[2].transpose() * M2;  // (R2, R0*R1)

    // Store core tensor in row-major (R0, R1, R2)
    result.core.resize(R0 * R1 * R2);
    for (size_t r2 = 0; r2 < R2; ++r2)
        for (size_t j = 0; j < R0 * R1; ++j)
            result.core[j / R1 * R1 * R2 + j % R1 * R2 + r2] = G2(r2, j);

    return result;
}

/// Reconstruct full tensor from Tucker decomposition (for validation).
/// Returns flat row-major vector of shape (N0, N1, N2).
[[nodiscard]] inline std::vector<double>
tucker_reconstruct(const TuckerResult3D& tucker) {
    auto [N0, N1, N2] = tucker.shape;
    auto [R0, R1, R2] = tucker.ranks;
    const auto& U0 = tucker.factors[0];
    const auto& U1 = tucker.factors[1];
    const auto& U2 = tucker.factors[2];

    std::vector<double> T(N0 * N1 * N2, 0.0);

    for (size_t i = 0; i < N0; ++i) {
        for (size_t j = 0; j < N1; ++j) {
            for (size_t k = 0; k < N2; ++k) {
                double val = 0.0;
                for (size_t r0 = 0; r0 < R0; ++r0) {
                    for (size_t r1 = 0; r1 < R1; ++r1) {
                        for (size_t r2 = 0; r2 < R2; ++r2) {
                            val += tucker.core[r0 * R1 * R2 + r1 * R2 + r2]
                                 * U0(i, r0) * U1(j, r1) * U2(k, r2);
                        }
                    }
                }
                T[i * N1 * N2 + j * N2 + k] = val;
            }
        }
    }
    return T;
}

}  // namespace mango
```

Add BUILD target in `src/option/table/dimensionless/BUILD.bazel`:

```python
cc_library(
    name = "tucker_decomposition",
    hdrs = ["tucker_decomposition.hpp"],
    deps = ["@eigen//:eigen"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option/table/dimensionless",
    include_prefix = "mango/option/table/dimensionless",
)
```

**Step 4: Run tests**

```bash
bazel test //tests:tucker_decomposition_test --test_output=all
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/option/table/dimensionless/tucker_decomposition.hpp \
        src/option/table/dimensionless/BUILD.bazel \
        tests/tucker_decomposition_test.cc tests/BUILD.bazel
git commit -m "Add Tucker decomposition (HOSVD) via Eigen SVD"
```

---

## Task 4: Implement combined Chebyshev-Tucker interpolant

Ties together Chebyshev node generation, tensor sampling, Tucker compression, and barycentric evaluation. This is the main class for the experiment.

**Step 1: Write the failing test**

Create `tests/chebyshev_tucker_test.cc`:

```cpp
// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless/chebyshev_tucker.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

// Smooth 3D test function: known analytically
double smooth_3d(double x, double y, double z) {
    return std::exp(-x * x) * std::sin(y) * std::cos(z * 0.5);
}

TEST(ChebyshevTuckerTest, ExactForLowDegreePolynomial) {
    // f(x,y,z) = x*y + z should be exactly represented with degree >= 1
    auto f = [](double x, double y, double z) { return x * y + z; };

    ChebyshevTuckerDomain domain{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    ChebyshevTuckerConfig config{.num_pts = {4, 4, 4}, .epsilon = 1e-12};

    auto interp = ChebyshevTucker3D::build(f, domain, config);

    // Query at off-grid points
    for (double x : {-0.5, 0.0, 0.7}) {
        for (double y : {-0.3, 0.4}) {
            for (double z : {-0.8, 0.1, 0.9}) {
                double expected = f(x, y, z);
                double got = interp.eval({x, y, z});
                EXPECT_NEAR(got, expected, 1e-11)
                    << "at (" << x << ", " << y << ", " << z << ")";
            }
        }
    }
}

TEST(ChebyshevTuckerTest, SmoothFunctionConverges) {
    ChebyshevTuckerDomain domain{
        .bounds = {{{-2.0, 2.0}, {0.0, M_PI}, {-1.0, 1.0}}},
    };

    // Coarse
    auto coarse = ChebyshevTucker3D::build(
        smooth_3d, domain, {.num_pts = {6, 6, 6}, .epsilon = 1e-12});

    // Fine
    auto fine = ChebyshevTucker3D::build(
        smooth_3d, domain, {.num_pts = {12, 12, 12}, .epsilon = 1e-12});

    // Evaluate at a test point
    double x = 0.5, y = 1.2, z = 0.3;
    double exact = smooth_3d(x, y, z);
    double err_coarse = std::abs(coarse.eval({x, y, z}) - exact);
    double err_fine = std::abs(fine.eval({x, y, z}) - exact);

    // Fine should be significantly more accurate
    EXPECT_LT(err_fine, err_coarse * 0.1);
    EXPECT_LT(err_fine, 1e-10);
}

TEST(ChebyshevTuckerTest, TuckerCompressionReducesStorage) {
    ChebyshevTuckerDomain domain{
        .bounds = {{{-2.0, 2.0}, {0.0, M_PI}, {-1.0, 1.0}}},
    };
    ChebyshevTuckerConfig config{.num_pts = {15, 15, 15}, .epsilon = 1e-8};

    auto interp = ChebyshevTucker3D::build(smooth_3d, domain, config);

    size_t full_size = 15 * 15 * 15;
    size_t compressed = interp.compressed_size();
    EXPECT_LT(compressed, full_size)
        << "Tucker should compress a smooth function";
}

TEST(ChebyshevTuckerTest, RanksReportedCorrectly) {
    auto f = [](double x, double y, double z) { return x * y + z; };
    ChebyshevTuckerDomain domain{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    auto interp = ChebyshevTucker3D::build(
        f, domain, {.num_pts = {8, 8, 8}, .epsilon = 1e-10});

    auto ranks = interp.ranks();
    // x*y + z is rank-2 in mode-0/1 unfolding, rank-2 in mode-2
    // Ranks should be small
    for (size_t r : ranks) {
        EXPECT_LE(r, 4u) << "Rank too high for a simple bilinear function";
    }
}

TEST(ChebyshevTuckerTest, EvalFullMatchesEvalTucker) {
    ChebyshevTuckerDomain domain{
        .bounds = {{{-2.0, 2.0}, {0.0, M_PI}, {-1.0, 1.0}}},
    };

    auto full = ChebyshevTucker3D::build(
        smooth_3d, domain, {.num_pts = {10, 10, 10}, .epsilon = 1e-15});
    auto compressed = ChebyshevTucker3D::build(
        smooth_3d, domain, {.num_pts = {10, 10, 10}, .epsilon = 1e-8});

    // Full (epsilon near machine precision) should match direct evaluation closely
    double x = 0.3, y = 2.0, z = -0.5;
    double exact = smooth_3d(x, y, z);
    double full_err = std::abs(full.eval({x, y, z}) - exact);
    double comp_err = std::abs(compressed.eval({x, y, z}) - exact);

    EXPECT_LT(full_err, 1e-12) << "Full-rank should be near-exact";
    EXPECT_LT(comp_err, 1e-6) << "Compressed should still be accurate";
}

}  // namespace
}  // namespace mango
```

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "chebyshev_tucker_test",
    size = "small",
    srcs = ["chebyshev_tucker_test.cc"],
    deps = [
        "//src/option/table/dimensionless:chebyshev_tucker",
        "@googletest//:gtest_main",
    ],
)
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:chebyshev_tucker_test --test_output=all
```

Expected: BUILD FAIL

**Step 3: Implement `chebyshev_tucker.hpp`**

Create `src/option/table/dimensionless/chebyshev_tucker.hpp`:

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/table/dimensionless/tucker_decomposition.hpp"

#include <array>
#include <functional>
#include <vector>

namespace mango {

/// Domain bounds for 3D Chebyshev-Tucker interpolation.
struct ChebyshevTuckerDomain {
    std::array<std::array<double, 2>, 3> bounds;  ///< {{a0,b0}, {a1,b1}, {a2,b2}}
};

/// Configuration for Chebyshev-Tucker interpolant.
struct ChebyshevTuckerConfig {
    std::array<size_t, 3> num_pts = {10, 10, 10};  ///< Sample points per axis
    double epsilon = 1e-8;                           ///< Tucker truncation threshold
};

/// 3D Chebyshev interpolant with Tucker compression.
///
/// Build: sample function on Chebyshev tensor grid, compress via HOSVD.
/// Eval: barycentric interpolation contracted with Tucker factors.
class ChebyshevTucker3D {
public:
    using SampleFn = std::function<double(double, double, double)>;

    /// Build interpolant by sampling f on Chebyshev nodes.
    [[nodiscard]] static ChebyshevTucker3D
    build(SampleFn f, const ChebyshevTuckerDomain& domain,
          const ChebyshevTuckerConfig& config) {
        ChebyshevTucker3D interp;
        interp.domain_ = domain;

        // Generate nodes and weights per axis
        for (size_t d = 0; d < 3; ++d) {
            auto [a, b] = domain.bounds[d];
            interp.nodes_[d] = chebyshev_nodes(config.num_pts[d], a, b);
            interp.weights_[d] = chebyshev_barycentric_weights(config.num_pts[d]);
        }

        // Sample function on full tensor grid
        auto& n = config.num_pts;
        std::vector<double> T(n[0] * n[1] * n[2]);
        for (size_t i = 0; i < n[0]; ++i)
            for (size_t j = 0; j < n[1]; ++j)
                for (size_t k = 0; k < n[2]; ++k)
                    T[i * n[1] * n[2] + j * n[2] + k] =
                        f(interp.nodes_[0][i], interp.nodes_[1][j], interp.nodes_[2][k]);

        // Tucker compress
        interp.tucker_ = tucker_hosvd(T, {n[0], n[1], n[2]}, config.epsilon);

        return interp;
    }

    /// Evaluate at a 3D point using Tucker-contracted barycentric interpolation.
    [[nodiscard]] double eval(const std::array<double, 3>& query) const {
        auto [R0, R1, R2] = tucker_.ranks;

        // Step 1: Barycentric weights -> factor-contracted coefficients per axis
        // For each axis d: compute c_r = Σ_j bary_weight(x, j) * U_d(j, r)
        //                                / Σ_j bary_weight(x, j)
        // But we need to handle the barycentric formula correctly:
        // The interpolant of column r of U_d is:
        //   c_r = Σ_j [w_j/(x-x_j)] * U_d(j,r) / Σ_j [w_j/(x-x_j)]

        std::array<std::vector<double>, 3> contracted;

        for (size_t d = 0; d < 3; ++d) {
            size_t R = tucker_.ranks[d];
            contracted[d].resize(R);

            // Check if query coincides with a node
            bool at_node = false;
            size_t node_idx = 0;
            for (size_t j = 0; j < nodes_[d].size(); ++j) {
                if (query[d] == nodes_[d][j]) {
                    at_node = true;
                    node_idx = j;
                    break;
                }
            }

            if (at_node) {
                // Exact at node: c_r = U_d(node_idx, r)
                for (size_t r = 0; r < R; ++r) {
                    contracted[d][r] = tucker_.factors[d](node_idx, r);
                }
            } else {
                // Barycentric interpolation of each column of U_d
                double denom = 0.0;
                for (size_t j = 0; j < nodes_[d].size(); ++j) {
                    denom += weights_[d][j] / (query[d] - nodes_[d][j]);
                }
                for (size_t r = 0; r < R; ++r) {
                    double numer = 0.0;
                    for (size_t j = 0; j < nodes_[d].size(); ++j) {
                        double term = weights_[d][j] / (query[d] - nodes_[d][j]);
                        numer += term * tucker_.factors[d](j, r);
                    }
                    contracted[d][r] = numer / denom;
                }
            }
        }

        // Step 2: Contract with core tensor
        double result = 0.0;
        for (size_t r0 = 0; r0 < R0; ++r0)
            for (size_t r1 = 0; r1 < R1; ++r1)
                for (size_t r2 = 0; r2 < R2; ++r2)
                    result += tucker_.core[r0 * R1 * R2 + r1 * R2 + r2]
                            * contracted[0][r0] * contracted[1][r1] * contracted[2][r2];

        return result;
    }

    /// Number of stored coefficients (core + factor entries).
    [[nodiscard]] size_t compressed_size() const {
        auto [R0, R1, R2] = tucker_.ranks;
        size_t core_size = R0 * R1 * R2;
        size_t factor_size = 0;
        for (size_t d = 0; d < 3; ++d)
            factor_size += nodes_[d].size() * tucker_.ranks[d];
        return core_size + factor_size;
    }

    /// Tucker ranks per mode.
    [[nodiscard]] std::array<size_t, 3> ranks() const { return tucker_.ranks; }

    /// Number of sample points per axis.
    [[nodiscard]] std::array<size_t, 3> num_pts() const {
        return {nodes_[0].size(), nodes_[1].size(), nodes_[2].size()};
    }

private:
    ChebyshevTuckerDomain domain_;
    std::array<std::vector<double>, 3> nodes_;
    std::array<std::vector<double>, 3> weights_;
    TuckerResult3D tucker_;
};

}  // namespace mango
```

Add BUILD target in `src/option/table/dimensionless/BUILD.bazel`:

```python
cc_library(
    name = "chebyshev_tucker",
    hdrs = ["chebyshev_tucker.hpp"],
    deps = [
        ":chebyshev_nodes",
        ":tucker_decomposition",
    ],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option/table/dimensionless",
    include_prefix = "mango/option/table/dimensionless",
)
```

**Step 4: Run tests**

```bash
bazel test //tests:chebyshev_tucker_test --test_output=all
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/option/table/dimensionless/chebyshev_tucker.hpp \
        src/option/table/dimensionless/BUILD.bazel \
        tests/chebyshev_tucker_test.cc tests/BUILD.bazel
git commit -m "Add Chebyshev-Tucker 3D interpolant"
```

---

## Task 5: Implement head-to-head benchmark

Sweeps `num_pts` from 6 to 30, builds all four interpolant configurations, evaluates at 520 probes, and prints a convergence table. This is the core experiment.

**Step 1: Write the benchmark**

Create `benchmarks/chebyshev_vs_bspline.cc`:

```cpp
// SPDX-License-Identifier: MIT
//
// Head-to-head benchmark: Chebyshev-Tucker vs B-spline for dimensionless EEP.
// Sweeps grid density and prints convergence table.

#include "mango/option/table/dimensionless/chebyshev_tucker.hpp"
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_european.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/math/cubic_spline_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace mango {
namespace {

// ===========================================================================
// Domain constants (from DimensionlessAdaptiveParams defaults)
// ===========================================================================

constexpr double SIGMA_MIN = 0.10, SIGMA_MAX = 0.80;
constexpr double RATE_MIN = 0.005, RATE_MAX = 0.10;
constexpr double TAU_MIN = 7.0 / 365, TAU_MAX = 2.0;
constexpr double MONEYNESS_MIN = 0.65, MONEYNESS_MAX = 1.50;

// Dimensionless domain bounds
const double X_MIN = std::log(MONEYNESS_MIN);           // -0.431
const double X_MAX = std::log(MONEYNESS_MAX);            //  0.405
const double TP_MIN = std::max(
    SIGMA_MIN * SIGMA_MIN * TAU_MIN / 2.0, 0.005);       //  0.005
const double TP_MAX = SIGMA_MAX * SIGMA_MAX * TAU_MAX / 2.0;  // 0.64
const double LK_MIN = std::log(2.0 * RATE_MIN / (SIGMA_MAX * SIGMA_MAX));  // -5.37
const double LK_MAX = std::log(2.0 * RATE_MAX / (SIGMA_MIN * SIGMA_MIN));  //  2.996

constexpr double K_REF = 100.0;
constexpr auto OPTION_TYPE = OptionType::PUT;

// Segment boundaries for config 2 (matching PR 386 logic)
constexpr size_t N_SEGMENTS = 3;

// Tucker epsilon sweep (pinned in design doc)
constexpr std::array<double, 6> TUCKER_EPSILONS = {
    1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12
};

// ===========================================================================
// Reference EEP solve
// ===========================================================================

double reference_eep(double x0, double tp0, double lk0) {
    double kappa = std::exp(lk0);
    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
    std::vector<double> snap = {tp0};
    solver.set_snapshot_times(std::span<const double>{snap});
    double tp_max = std::max(tp0 * 1.01, 0.02);
    std::vector<PricingParams> batch;
    batch.emplace_back(
        OptionSpec{.spot = K_REF, .strike = K_REF, .maturity = tp_max,
                   .rate = kappa, .dividend_yield = 0.0, .option_type = OPTION_TYPE},
        std::sqrt(2.0));
    auto result = solver.solve_batch(batch, false);
    if (result.results.empty() || !result.results[0].has_value()) return 0.0;
    const auto& sol = result.results[0].value();
    if (sol.num_snapshots() < 1) return 0.0;
    CubicSpline<double> spline;
    if (spline.build(sol.grid()->x(), sol.at_time(0)).has_value()) return 0.0;
    double am = spline.eval(x0);
    double eu = dimensionless_european(x0, tp0, kappa, OPTION_TYPE);
    return std::max(am - eu, 0.0);
}

// ===========================================================================
// Probe generation (520 probes: 500 LHS + 8 corners + 12 edge midpoints)
// ===========================================================================

struct Probe { double x, tp, lk, true_eep; };

std::vector<Probe> generate_probes() {
    std::vector<Probe> probes;

    // 500 seeded LHS probes
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dx(X_MIN, X_MAX);
    std::uniform_real_distribution<double> dtp(TP_MIN, TP_MAX);
    std::uniform_real_distribution<double> dlk(LK_MIN, LK_MAX);

    for (size_t i = 0; i < 500; ++i) {
        double x = dx(rng), tp = dtp(rng), lk = dlk(rng);
        probes.push_back({x, tp, lk, reference_eep(x, tp, lk)});
    }

    // 8 corner points
    for (double x : {X_MIN, X_MAX})
        for (double tp : {TP_MIN, TP_MAX})
            for (double lk : {LK_MIN, LK_MAX})
                probes.push_back({x, tp, lk, reference_eep(x, tp, lk)});

    // 12 edge midpoints (vary one axis, fix other two at midpoint)
    double x_mid = (X_MIN + X_MAX) / 2;
    double tp_mid = (TP_MIN + TP_MAX) / 2;
    double lk_mid = (LK_MIN + LK_MAX) / 2;

    for (double x : {X_MIN, X_MAX}) {
        probes.push_back({x, tp_mid, lk_mid, reference_eep(x, tp_mid, lk_mid)});
    }
    for (double tp : {TP_MIN, TP_MAX}) {
        probes.push_back({x_mid, tp, lk_mid, reference_eep(x_mid, tp, lk_mid)});
    }
    for (double lk : {LK_MIN, LK_MAX}) {
        probes.push_back({x_mid, tp_mid, lk, reference_eep(x_mid, tp_mid, lk)});
    }
    // Remaining 6 edge midpoints: vary two axes at extremes, fix one at mid
    for (double x : {X_MIN, X_MAX}) {
        for (double tp : {TP_MIN, TP_MAX}) {
            probes.push_back({x, tp, lk_mid, reference_eep(x, tp, lk_mid)});
        }
    }
    // That gives: 500 + 8 + 2 + 2 + 2 + 4 = 518.
    // Add the last 2: vary (x, lk) at extremes, tp at mid
    for (double x : {X_MIN, X_MAX}) {
        probes.push_back({x, tp_mid, LK_MIN, reference_eep(x, tp_mid, LK_MIN)});
    }
    // Total: 520

    return probes;
}

// ===========================================================================
// B-spline headroom helper
// ===========================================================================

double spline_headroom(double domain_width, size_t n_knots) {
    size_t n = std::max(n_knots, size_t{4});
    return 3.0 * domain_width / static_cast<double>(n - 1);
}

std::vector<double> linspace(double lo, double hi, size_t n) {
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
    return v;
}

// ===========================================================================
// Config 1: B-spline unsegmented
// ===========================================================================

struct ErrorResult {
    double max_err, avg_err;
    double build_seconds;
    int n_pde_solves;
};

ErrorResult eval_bspline_unsegmented(
    size_t num_pts, const std::vector<Probe>& probes)
{
    // Generate grids with headroom
    double hx = spline_headroom(X_MAX - X_MIN, num_pts);
    double htp = spline_headroom(TP_MAX - TP_MIN, num_pts);
    double hlk = spline_headroom(LK_MAX - LK_MIN, num_pts);
    auto x_grid = linspace(X_MIN - hx, X_MAX + hx, num_pts);
    auto tp_grid = linspace(std::max(TP_MIN - htp, 1e-3), TP_MAX + htp, num_pts);
    auto lk_grid = linspace(LK_MIN - hlk, LK_MAX + hlk, num_pts);

    DimensionlessAxes axes{x_grid, tp_grid, lk_grid};

    auto t0 = std::chrono::steady_clock::now();
    auto result = build_dimensionless_surface(axes, K_REF, OPTION_TYPE,
                                              SurfaceContent::EarlyExercisePremium);
    auto t1 = std::chrono::steady_clock::now();

    if (!result.has_value()) {
        std::fprintf(stderr, "  B-spline unseg build failed at num_pts=%zu\n", num_pts);
        return {999.0, 999.0, 0.0, 0};
    }

    double max_err = 0, sum_err = 0;
    for (const auto& p : probes) {
        double interp = result->surface->value({p.x, p.tp, p.lk});
        double err = std::abs(interp - p.true_eep);
        max_err = std::max(max_err, err);
        sum_err += err;
    }

    return {
        max_err, sum_err / probes.size(),
        std::chrono::duration<double>(t1 - t0).count(),
        result->n_pde_solves
    };
}

// ===========================================================================
// Config 2: B-spline 3-segment (matching PR 386 boundary logic)
// ===========================================================================

ErrorResult eval_bspline_segmented(
    size_t num_pts, const std::vector<Probe>& probes)
{
    double hx = spline_headroom(X_MAX - X_MIN, num_pts);
    double htp = spline_headroom(TP_MAX - TP_MIN, num_pts);
    auto x_grid = linspace(X_MIN - hx, X_MAX + hx, num_pts);
    auto tp_grid = linspace(std::max(TP_MIN - htp, 1e-3), TP_MAX + htp, num_pts);

    double lk_seg_width = (LK_MAX - LK_MIN) / static_cast<double>(N_SEGMENTS);

    auto t0 = std::chrono::steady_clock::now();

    std::vector<SegmentedDimensionlessSurface::Segment> segments;
    int total_solves = 0;

    for (size_t s = 0; s < N_SEGMENTS; ++s) {
        double seg_lk_min_phys = LK_MIN + lk_seg_width * static_cast<double>(s);
        double seg_lk_max_phys = LK_MIN + lk_seg_width * static_cast<double>(s + 1);
        double hlk = spline_headroom(seg_lk_max_phys - seg_lk_min_phys, num_pts);
        auto lk_grid = linspace(seg_lk_min_phys - hlk, seg_lk_max_phys + hlk, num_pts);

        DimensionlessAxes axes{x_grid, tp_grid, lk_grid};
        auto result = build_dimensionless_surface(axes, K_REF, OPTION_TYPE,
                                                  SurfaceContent::EarlyExercisePremium);
        if (!result.has_value()) {
            std::fprintf(stderr, "  B-spline seg[%zu] build failed at num_pts=%zu\n", s, num_pts);
            return {999.0, 999.0, 0.0, 0};
        }

        segments.push_back({result->surface, seg_lk_min_phys, seg_lk_max_phys});
        total_solves += result->n_pde_solves;
    }

    auto surface = std::make_shared<SegmentedDimensionlessSurface>(std::move(segments));
    auto t1 = std::chrono::steady_clock::now();

    double max_err = 0, sum_err = 0;
    for (const auto& p : probes) {
        double interp = surface->value({p.x, p.tp, p.lk});
        double err = std::abs(interp - p.true_eep);
        max_err = std::max(max_err, err);
        sum_err += err;
    }

    return {
        max_err, sum_err / probes.size(),
        std::chrono::duration<double>(t1 - t0).count(),
        total_solves
    };
}

// ===========================================================================
// Config 3: Chebyshev full tensor (no Tucker)
// ===========================================================================

ErrorResult eval_chebyshev_full(
    size_t num_pts, const std::vector<Probe>& probes)
{
    ChebyshevTuckerDomain domain{
        .bounds = {{{X_MIN, X_MAX}, {TP_MIN, TP_MAX}, {LK_MIN, LK_MAX}}},
    };
    // epsilon near machine precision => effectively full rank
    ChebyshevTuckerConfig config{
        .num_pts = {num_pts, num_pts, num_pts},
        .epsilon = 1e-15,
    };

    auto eep_fn = [](double x, double tp, double lk) -> double {
        return reference_eep(x, tp, lk);
    };

    auto t0 = std::chrono::steady_clock::now();
    auto interp = ChebyshevTucker3D::build(eep_fn, domain, config);
    auto t1 = std::chrono::steady_clock::now();

    double max_err = 0, sum_err = 0;
    for (const auto& p : probes) {
        double val = interp.eval({p.x, p.tp, p.lk});
        double err = std::abs(val - p.true_eep);
        max_err = std::max(max_err, err);
        sum_err += err;
    }

    return {
        max_err, sum_err / probes.size(),
        std::chrono::duration<double>(t1 - t0).count(),
        static_cast<int>(num_pts * num_pts * num_pts)  // One solve per grid point
    };
}

// ===========================================================================
// Config 4: Chebyshev-Tucker (epsilon sweep)
// ===========================================================================

struct TuckerSweepResult {
    double epsilon;
    std::array<size_t, 3> ranks;
    double max_err, avg_err;
    size_t compressed_size;
};

std::vector<TuckerSweepResult> eval_chebyshev_tucker(
    size_t num_pts, const std::vector<Probe>& probes)
{
    ChebyshevTuckerDomain domain{
        .bounds = {{{X_MIN, X_MAX}, {TP_MIN, TP_MAX}, {LK_MIN, LK_MAX}}},
    };

    auto eep_fn = [](double x, double tp, double lk) -> double {
        return reference_eep(x, tp, lk);
    };

    std::vector<TuckerSweepResult> results;

    for (double eps : TUCKER_EPSILONS) {
        ChebyshevTuckerConfig config{
            .num_pts = {num_pts, num_pts, num_pts},
            .epsilon = eps,
        };

        auto interp = ChebyshevTucker3D::build(eep_fn, domain, config);

        double max_err = 0, sum_err = 0;
        for (const auto& p : probes) {
            double val = interp.eval({p.x, p.tp, p.lk});
            double err = std::abs(val - p.true_eep);
            max_err = std::max(max_err, err);
            sum_err += err;
        }

        results.push_back({
            eps, interp.ranks(), max_err, sum_err / probes.size(),
            interp.compressed_size()
        });
    }

    return results;
}

}  // anonymous namespace
}  // namespace mango

// ===========================================================================
// Main
// ===========================================================================

int main() {
    using namespace mango;

    std::printf("=== Chebyshev-Tucker vs B-spline benchmark ===\n");
    std::printf("Domain: x=[%.3f,%.3f] tp=[%.4f,%.3f] lk=[%.3f,%.3f]\n",
                X_MIN, X_MAX, TP_MIN, TP_MAX, LK_MIN, LK_MAX);

    std::printf("\nGenerating reference probes...\n");
    auto probes = generate_probes();
    std::printf("  %zu probes generated\n\n", probes.size());

    // Sweep num_pts
    for (size_t num_pts : {6, 8, 10, 12, 15, 18, 20, 25, 30}) {
        std::printf("=== num_pts = %zu ===\n", num_pts);

        // Config 1: B-spline unsegmented
        auto bs1 = eval_bspline_unsegmented(num_pts, probes);
        std::printf("  B-spline unseg:  max_err=%.6f  avg_err=%.6f  "
                     "solves=%d  build=%.1fs\n",
                     bs1.max_err, bs1.avg_err, bs1.n_pde_solves, bs1.build_seconds);

        // Config 2: B-spline 3-segment
        auto bs3 = eval_bspline_segmented(num_pts, probes);
        std::printf("  B-spline 3-seg:  max_err=%.6f  avg_err=%.6f  "
                     "solves=%d  build=%.1fs\n",
                     bs3.max_err, bs3.avg_err, bs3.n_pde_solves, bs3.build_seconds);

        // Config 3: Chebyshev full (only at smaller sizes due to cost)
        if (num_pts <= 20) {
            auto ch_full = eval_chebyshev_full(num_pts, probes);
            std::printf("  Cheb full:       max_err=%.6f  avg_err=%.6f  "
                         "solves=%d  build=%.1fs\n",
                         ch_full.max_err, ch_full.avg_err,
                         ch_full.n_pde_solves, ch_full.build_seconds);
        }

        // Config 4: Chebyshev-Tucker sweep (only at a few sizes)
        if (num_pts == 10 || num_pts == 15 || num_pts == 20) {
            auto tucker_results = eval_chebyshev_tucker(num_pts, probes);
            for (const auto& tr : tucker_results) {
                std::printf("  Cheb-Tucker e=%.0e: max=%.6f avg=%.6f "
                             "R=(%zu,%zu,%zu) size=%zu\n",
                             tr.epsilon, tr.max_err, tr.avg_err,
                             tr.ranks[0], tr.ranks[1], tr.ranks[2],
                             tr.compressed_size);
            }
        }

        std::printf("\n");
    }

    return 0;
}
```

Add to `benchmarks/BUILD.bazel`:

```python
cc_binary(
    name = "chebyshev_vs_bspline",
    srcs = ["chebyshev_vs_bspline.cc"],
    copts = ["-fopenmp", "-pthread"],
    linkopts = ["-fopenmp", "-pthread"],
    deps = [
        "//src/option/table/dimensionless:chebyshev_tucker",
        "//src/option/table/dimensionless:dimensionless_builder",
        "//src/option/table/dimensionless:dimensionless_european",
        "//src/option:american_option_batch",
        "//src/math:cubic_spline_solver",
    ],
)
```

**Step 2: Build the benchmark**

```bash
bazel build //benchmarks:chebyshev_vs_bspline
```

Expected: BUILD SUCCESS

**Step 3: Commit**

```bash
git add benchmarks/chebyshev_vs_bspline.cc benchmarks/BUILD.bazel
git commit -m "Add Chebyshev vs B-spline head-to-head benchmark"
```

**Step 4: Run the benchmark**

```bash
bazel-bin/benchmarks/chebyshev_vs_bspline 2>&1 | tee benchmark_results.txt
```

This will take a while (many PDE solves). Expected output: a convergence table showing error vs num_pts for all four configurations. Examine whether Chebyshev error continues dropping where B-spline plateaus.

**Step 5: Commit results**

```bash
git add benchmark_results.txt
git commit -m "Record initial benchmark results"
```

---

## Task 6: Verify all tests pass

Final verification that nothing is broken.

**Step 1: Run full test suite**

```bash
bazel test //... --test_output=errors
```

Expected: All tests PASS (existing + new Chebyshev/Tucker tests).

**Step 2: Build benchmarks and Python bindings**

```bash
bazel build //benchmarks/...
bazel build //src/python:mango_option
```

Expected: BUILD SUCCESS

**Step 3: Commit if any fixes needed**

---

## Task Summary

| Task | Description | Key files |
|------|-------------|-----------|
| 1 | Add Eigen 3.4.0 via Bzlmod | `third_party/deps.bzl`, `MODULE.bazel` |
| 2 | Chebyshev nodes + barycentric interpolation | `chebyshev_nodes.hpp` |
| 3 | Tucker decomposition (HOSVD) | `tucker_decomposition.hpp` |
| 4 | Combined Chebyshev-Tucker interpolant | `chebyshev_tucker.hpp` |
| 5 | Head-to-head benchmark | `benchmarks/chebyshev_vs_bspline.cc` |
| 6 | Final verification | — |

## Residual Risks (from design review)

1. **Config 2 segment boundaries** must exactly match PR 386: uniform split of `[LK_MIN, LK_MAX]` into 3 equal segments with `spline_headroom()` per segment. The benchmark code above replicates this logic directly.
2. **Probe domain bounds** are explicit constants at the top of the benchmark file, matching `DimensionlessAdaptiveParams` defaults. Edge midpoints are enumerated explicitly.
