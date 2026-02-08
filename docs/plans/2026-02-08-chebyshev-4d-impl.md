# 4D Chebyshev-Tucker EEP Surface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a 4D Chebyshev-Tucker interpolant for EEP surfaces and integrate it into the existing IV benchmarks.

**Architecture:** Extend the existing 3D Tucker HOSVD and Chebyshev-Tucker interpolation classes to 4D (ln(S/K), tau, sigma, rate). Keeping sigma as its own axis eliminates the coordinate coupling that limited the 3D dimensionless approach to ~312 bps RMS. The 4D interpolant uses the same barycentric-Chebyshev + Tucker-core pattern, just generalized from 3 to 4 modes.

**Tech Stack:** C++23, Eigen (SVD), Bazel, GoogleTest, Google Benchmark

---

### Task 1: Add `tucker_decomposition_4d.hpp`

Extend the existing 3D HOSVD to 4D. Same algorithm: unfold along each of 4 modes, SVD + epsilon truncation, sequential core contraction.

**Files:**
- Create: `src/option/table/dimensionless/tucker_decomposition_4d.hpp`
- Reference: `src/option/table/dimensionless/tucker_decomposition.hpp`

**Step 1: Create the 4D Tucker decomposition header**

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <vector>

namespace mango {

struct TuckerResult4D {
    std::vector<double> core;
    std::array<Eigen::MatrixXd, 4> factors;
    std::array<size_t, 4> shape;
    std::array<size_t, 4> ranks;
};

inline Eigen::MatrixXd
mode_unfold_4d(const std::vector<double>& T,
               const std::array<size_t, 4>& shape,
               size_t mode) {
    size_t n_rows = shape[mode];
    size_t n_cols = 1;
    for (size_t d = 0; d < 4; ++d)
        if (d != mode) n_cols *= shape[d];

    Eigen::MatrixXd M(n_rows, n_cols);

    for (size_t i0 = 0; i0 < shape[0]; ++i0)
        for (size_t i1 = 0; i1 < shape[1]; ++i1)
            for (size_t i2 = 0; i2 < shape[2]; ++i2)
                for (size_t i3 = 0; i3 < shape[3]; ++i3) {
                    std::array<size_t, 4> idx = {i0, i1, i2, i3};
                    size_t row = idx[mode];
                    size_t col = 0;
                    size_t stride = 1;
                    for (int d = 3; d >= 0; --d) {
                        if (static_cast<size_t>(d) == mode) continue;
                        col += idx[d] * stride;
                        stride *= shape[d];
                    }
                    size_t flat = i0 * shape[1] * shape[2] * shape[3]
                                + i1 * shape[2] * shape[3]
                                + i2 * shape[3] + i3;
                    M(row, col) = T[flat];
                }
    return M;
}

[[nodiscard]] inline TuckerResult4D
tucker_hosvd_4d(const std::vector<double>& T,
                const std::array<size_t, 4>& shape,
                double epsilon) {
    TuckerResult4D result;
    result.shape = shape;

    // Step 1: SVD each mode, truncate
    for (size_t mode = 0; mode < 4; ++mode) {
        Eigen::MatrixXd M = mode_unfold_4d(T, shape, mode);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU);
        const auto& sigma = svd.singularValues();
        double sigma_0 = sigma(0);
        size_t rank = 1;
        for (Eigen::Index i = 1; i < sigma.size(); ++i) {
            if (sigma(i) / sigma_0 >= epsilon) rank++;
            else break;
        }
        result.ranks[mode] = rank;
        result.factors[mode] = svd.matrixU().leftCols(rank);
    }

    // Step 2: Sequential core contraction
    // G = U0^T x_0 U1^T x_1 U2^T x_2 U3^T x_3 T
    // Contract mode by mode, shrinking the tensor each time.

    auto [R0, R1, R2, R3] = result.ranks;

    // Contract mode-0: shape (R0, N1, N2, N3)
    Eigen::MatrixXd M0 = mode_unfold_4d(T, shape, 0);
    Eigen::MatrixXd G0 = result.factors[0].transpose() * M0;

    std::array<size_t, 4> shape1 = {R0, shape[1], shape[2], shape[3]};
    size_t size1 = R0 * shape[1] * shape[2] * shape[3];
    std::vector<double> G0_vec(size1);
    for (size_t r = 0; r < R0; ++r)
        for (size_t j = 0; j < shape[1] * shape[2] * shape[3]; ++j)
            G0_vec[r * shape[1] * shape[2] * shape[3] + j] = G0(r, j);

    // Contract mode-1: shape (R0, R1, N2, N3)
    Eigen::MatrixXd M1 = mode_unfold_4d(G0_vec, shape1, 1);
    Eigen::MatrixXd G1 = result.factors[1].transpose() * M1;

    std::array<size_t, 4> shape2 = {R0, R1, shape[2], shape[3]};
    size_t size2 = R0 * R1 * shape[2] * shape[3];
    std::vector<double> G1_vec(size2);
    // G1 rows = R1, cols = R0 * N2 * N3
    // Repack into row-major (R0, R1, N2, N3)
    for (size_t r1 = 0; r1 < R1; ++r1)
        for (size_t j = 0; j < R0 * shape[2] * shape[3]; ++j) {
            size_t r0 = j / (shape[2] * shape[3]);
            size_t rem = j % (shape[2] * shape[3]);
            G1_vec[r0 * R1 * shape[2] * shape[3] + r1 * shape[2] * shape[3] + rem] = G1(r1, j);
        }

    // Contract mode-2: shape (R0, R1, R2, N3)
    Eigen::MatrixXd M2 = mode_unfold_4d(G1_vec, shape2, 2);
    Eigen::MatrixXd G2 = result.factors[2].transpose() * M2;

    std::array<size_t, 4> shape3 = {R0, R1, R2, shape[3]};
    size_t size3 = R0 * R1 * R2 * shape[3];
    std::vector<double> G2_vec(size3);
    // G2 rows = R2, cols = R0 * R1 * N3
    for (size_t r2 = 0; r2 < R2; ++r2)
        for (size_t j = 0; j < R0 * R1 * shape[3]; ++j) {
            size_t r0r1 = j / shape[3];
            size_t r0 = r0r1 / R1;
            size_t r1 = r0r1 % R1;
            size_t i3 = j % shape[3];
            G2_vec[r0 * R1 * R2 * shape[3] + r1 * R2 * shape[3] + r2 * shape[3] + i3] = G2(r2, j);
        }

    // Contract mode-3: shape (R0, R1, R2, R3)
    Eigen::MatrixXd M3 = mode_unfold_4d(G2_vec, shape3, 3);
    Eigen::MatrixXd G3 = result.factors[3].transpose() * M3;

    result.core.resize(R0 * R1 * R2 * R3);
    // G3 rows = R3, cols = R0 * R1 * R2
    for (size_t r3 = 0; r3 < R3; ++r3)
        for (size_t j = 0; j < R0 * R1 * R2; ++j) {
            size_t r0 = j / (R1 * R2);
            size_t r1 = (j % (R1 * R2)) / R2;
            size_t r2 = j % R2;
            result.core[r0 * R1 * R2 * R3 + r1 * R2 * R3 + r2 * R3 + r3] = G3(r3, j);
        }

    return result;
}

[[nodiscard]] inline std::vector<double>
tucker_reconstruct_4d(const TuckerResult4D& tucker) {
    auto [N0, N1, N2, N3] = tucker.shape;
    auto [R0, R1, R2, R3] = tucker.ranks;
    const auto& U0 = tucker.factors[0];
    const auto& U1 = tucker.factors[1];
    const auto& U2 = tucker.factors[2];
    const auto& U3 = tucker.factors[3];

    std::vector<double> T(N0 * N1 * N2 * N3, 0.0);
    for (size_t i = 0; i < N0; ++i)
        for (size_t j = 0; j < N1; ++j)
            for (size_t k = 0; k < N2; ++k)
                for (size_t l = 0; l < N3; ++l) {
                    double val = 0.0;
                    for (size_t r0 = 0; r0 < R0; ++r0)
                        for (size_t r1 = 0; r1 < R1; ++r1)
                            for (size_t r2 = 0; r2 < R2; ++r2)
                                for (size_t r3 = 0; r3 < R3; ++r3)
                                    val += tucker.core[r0*R1*R2*R3 + r1*R2*R3 + r2*R3 + r3]
                                         * U0(i, r0) * U1(j, r1) * U2(k, r2) * U3(l, r3);
                    T[i*N1*N2*N3 + j*N2*N3 + k*N3 + l] = val;
                }
    return T;
}

}  // namespace mango
```

**Step 2: Add BUILD target**

In `src/option/table/dimensionless/BUILD.bazel`, add after the `tucker_decomposition` target:

```python
cc_library(
    name = "tucker_decomposition_4d",
    hdrs = ["tucker_decomposition_4d.hpp"],
    deps = ["@eigen//:eigen"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option/table/dimensionless",
    include_prefix = "mango/option/table/dimensionless",
)
```

**Step 3: Verify it compiles**

Run: `bazel build //src/option/table/dimensionless:tucker_decomposition_4d`
Expected: BUILD SUCCESSFUL

**Step 4: Commit**

```bash
git add src/option/table/dimensionless/tucker_decomposition_4d.hpp src/option/table/dimensionless/BUILD.bazel
git commit -m "Add 4D Tucker HOSVD decomposition"
```

---

### Task 2: Add tests for 4D Tucker HOSVD

**Files:**
- Create: `tests/tucker_decomposition_4d_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the test file**

```cpp
// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless/tucker_decomposition_4d.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

TEST(TuckerDecomposition4DTest, ExactRoundtripForRank1) {
    // f(i,j,k,l) = (i+1) * (j+1) * (k+1) * (l+1) is rank-1
    std::array<size_t, 4> shape = {4, 5, 3, 4};
    std::vector<double> T(4 * 5 * 3 * 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 5; ++j)
            for (size_t k = 0; k < 3; ++k)
                for (size_t l = 0; l < 4; ++l)
                    T[i*5*3*4 + j*3*4 + k*4 + l] =
                        (i + 1.0) * (j + 1.0) * (k + 1.0) * (l + 1.0);

    auto tucker = tucker_hosvd_4d(T, shape, 1e-10);

    // Rank-1 tensor: each mode should have rank 1
    for (size_t d = 0; d < 4; ++d) {
        EXPECT_EQ(tucker.ranks[d], 1u) << "mode " << d;
    }

    auto reconstructed = tucker_reconstruct_4d(tucker);
    for (size_t i = 0; i < T.size(); ++i) {
        EXPECT_NEAR(reconstructed[i], T[i], 1e-10) << "at flat index " << i;
    }
}

TEST(TuckerDecomposition4DTest, CompressesLowRankTensor) {
    // f = a*b*c*d + e*f*g*h (rank-2)
    std::array<size_t, 4> shape = {6, 6, 6, 6};
    size_t total = 6 * 6 * 6 * 6;
    std::vector<double> T(total);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 6; ++j)
            for (size_t k = 0; k < 6; ++k)
                for (size_t l = 0; l < 6; ++l)
                    T[i*6*6*6 + j*6*6 + k*6 + l] =
                        std::sin(i * 0.5) * std::cos(j * 0.3) *
                        std::exp(-k * 0.2) * (l + 1.0);

    auto tucker = tucker_hosvd_4d(T, shape, 1e-8);

    // Should compress well
    size_t core_size = tucker.ranks[0] * tucker.ranks[1] *
                       tucker.ranks[2] * tucker.ranks[3];
    EXPECT_LT(core_size, total);

    auto reconstructed = tucker_reconstruct_4d(tucker);
    for (size_t i = 0; i < T.size(); ++i) {
        EXPECT_NEAR(reconstructed[i], T[i], 1e-6) << "at flat index " << i;
    }
}

TEST(TuckerDecomposition4DTest, ModeUnfoldDimensionsCorrect) {
    std::array<size_t, 4> shape = {3, 4, 5, 2};
    std::vector<double> T(3 * 4 * 5 * 2, 1.0);

    for (size_t mode = 0; mode < 4; ++mode) {
        auto M = mode_unfold_4d(T, shape, mode);
        EXPECT_EQ(static_cast<size_t>(M.rows()), shape[mode]) << "mode " << mode;
        size_t expected_cols = 1;
        for (size_t d = 0; d < 4; ++d)
            if (d != mode) expected_cols *= shape[d];
        EXPECT_EQ(static_cast<size_t>(M.cols()), expected_cols) << "mode " << mode;
    }
}

}  // namespace
}  // namespace mango
```

**Step 2: Add test target to `tests/BUILD.bazel`**

Add after the `chebyshev_tucker_test` target:

```python
cc_test(
    name = "tucker_decomposition_4d_test",
    size = "small",
    srcs = ["tucker_decomposition_4d_test.cc"],
    deps = [
        "//src/option/table/dimensionless:tucker_decomposition_4d",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3: Run the test**

Run: `bazel test //tests:tucker_decomposition_4d_test --test_output=all`
Expected: 3 tests PASS

**Step 4: Commit**

```bash
git add tests/tucker_decomposition_4d_test.cc tests/BUILD.bazel
git commit -m "Add tests for 4D Tucker HOSVD"
```

---

### Task 3: Add `chebyshev_tucker_4d.hpp`

Mirror `ChebyshevTucker3D` but with 4 axes. Uses barycentric Chebyshev interpolation contracted with Tucker core.

**Files:**
- Create: `src/option/table/dimensionless/chebyshev_tucker_4d.hpp`
- Modify: `src/option/table/dimensionless/BUILD.bazel`

**Step 1: Create the 4D Chebyshev-Tucker class**

```cpp
// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/table/dimensionless/tucker_decomposition_4d.hpp"

#include <array>
#include <functional>
#include <span>
#include <vector>

namespace mango {

struct ChebyshevTucker4DDomain {
    std::array<std::array<double, 2>, 4> bounds;
};

struct ChebyshevTucker4DConfig {
    std::array<size_t, 4> num_pts = {10, 10, 10, 6};
    double epsilon = 1e-8;
};

class ChebyshevTucker4D {
public:
    using SampleFn = std::function<double(double, double, double, double)>;

    [[nodiscard]] static ChebyshevTucker4D
    build(SampleFn f, const ChebyshevTucker4DDomain& domain,
          const ChebyshevTucker4DConfig& config) {
        ChebyshevTucker4D interp;
        interp.domain_ = domain;

        for (size_t d = 0; d < 4; ++d) {
            auto [a, b] = domain.bounds[d];
            interp.nodes_[d] = chebyshev_nodes(config.num_pts[d], a, b);
            interp.weights_[d] = chebyshev_barycentric_weights(config.num_pts[d]);
        }

        auto& n = config.num_pts;
        std::vector<double> T(n[0] * n[1] * n[2] * n[3]);
        for (size_t i = 0; i < n[0]; ++i)
            for (size_t j = 0; j < n[1]; ++j)
                for (size_t k = 0; k < n[2]; ++k)
                    for (size_t l = 0; l < n[3]; ++l)
                        T[i*n[1]*n[2]*n[3] + j*n[2]*n[3] + k*n[3] + l] =
                            f(interp.nodes_[0][i], interp.nodes_[1][j],
                              interp.nodes_[2][k], interp.nodes_[3][l]);

        interp.tucker_ = tucker_hosvd_4d(T, {n[0], n[1], n[2], n[3]}, config.epsilon);
        return interp;
    }

    [[nodiscard]] static ChebyshevTucker4D
    build_from_values(std::span<const double> values,
                      const ChebyshevTucker4DDomain& domain,
                      const ChebyshevTucker4DConfig& config) {
        ChebyshevTucker4D interp;
        interp.domain_ = domain;

        for (size_t d = 0; d < 4; ++d) {
            auto [a, b] = domain.bounds[d];
            interp.nodes_[d] = chebyshev_nodes(config.num_pts[d], a, b);
            interp.weights_[d] = chebyshev_barycentric_weights(config.num_pts[d]);
        }

        auto& n = config.num_pts;
        std::vector<double> T(values.begin(), values.end());
        interp.tucker_ = tucker_hosvd_4d(T, {n[0], n[1], n[2], n[3]}, config.epsilon);
        return interp;
    }

    [[nodiscard]] double eval(const std::array<double, 4>& query) const {
        std::array<double, 4> q = query;
        for (size_t d = 0; d < 4; ++d) {
            q[d] = std::clamp(q[d], domain_.bounds[d][0], domain_.bounds[d][1]);
        }

        auto [R0, R1, R2, R3] = tucker_.ranks;

        std::array<std::vector<double>, 4> contracted;

        for (size_t d = 0; d < 4; ++d) {
            size_t R = tucker_.ranks[d];
            contracted[d].resize(R);

            bool at_node = false;
            size_t node_idx = 0;
            for (size_t j = 0; j < nodes_[d].size(); ++j) {
                if (q[d] == nodes_[d][j]) {
                    at_node = true;
                    node_idx = j;
                    break;
                }
            }

            if (at_node) {
                for (size_t r = 0; r < R; ++r) {
                    contracted[d][r] = tucker_.factors[d](node_idx, r);
                }
            } else {
                double denom = 0.0;
                for (size_t j = 0; j < nodes_[d].size(); ++j) {
                    denom += weights_[d][j] / (q[d] - nodes_[d][j]);
                }
                for (size_t r = 0; r < R; ++r) {
                    double numer = 0.0;
                    for (size_t j = 0; j < nodes_[d].size(); ++j) {
                        double term = weights_[d][j] / (q[d] - nodes_[d][j]);
                        numer += term * tucker_.factors[d](j, r);
                    }
                    contracted[d][r] = numer / denom;
                }
            }
        }

        double result = 0.0;
        for (size_t r0 = 0; r0 < R0; ++r0)
            for (size_t r1 = 0; r1 < R1; ++r1)
                for (size_t r2 = 0; r2 < R2; ++r2)
                    for (size_t r3 = 0; r3 < R3; ++r3)
                        result += tucker_.core[r0*R1*R2*R3 + r1*R2*R3 + r2*R3 + r3]
                                * contracted[0][r0] * contracted[1][r1]
                                * contracted[2][r2] * contracted[3][r3];

        return result;
    }

    [[nodiscard]] double partial(size_t axis,
                                  const std::array<double, 4>& coords) const {
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
        auto [R0, R1, R2, R3] = tucker_.ranks;
        size_t core_size = R0 * R1 * R2 * R3;
        size_t factor_size = 0;
        for (size_t d = 0; d < 4; ++d)
            factor_size += nodes_[d].size() * tucker_.ranks[d];
        return core_size + factor_size;
    }

    [[nodiscard]] std::array<size_t, 4> ranks() const { return tucker_.ranks; }

    [[nodiscard]] std::array<size_t, 4> num_pts() const {
        return {nodes_[0].size(), nodes_[1].size(),
                nodes_[2].size(), nodes_[3].size()};
    }

private:
    ChebyshevTucker4DDomain domain_;
    std::array<std::vector<double>, 4> nodes_;
    std::array<std::vector<double>, 4> weights_;
    TuckerResult4D tucker_;
};

}  // namespace mango
```

**Step 2: Add BUILD target**

In `src/option/table/dimensionless/BUILD.bazel`, add:

```python
cc_library(
    name = "chebyshev_tucker_4d",
    hdrs = ["chebyshev_tucker_4d.hpp"],
    deps = [
        ":chebyshev_nodes",
        ":tucker_decomposition_4d",
    ],
    visibility = ["//visibility:public"],
    strip_include_prefix = "/src/option/table/dimensionless",
    include_prefix = "mango/option/table/dimensionless",
)
```

**Step 3: Verify it compiles**

Run: `bazel build //src/option/table/dimensionless:chebyshev_tucker_4d`
Expected: BUILD SUCCESSFUL

**Step 4: Commit**

```bash
git add src/option/table/dimensionless/chebyshev_tucker_4d.hpp src/option/table/dimensionless/BUILD.bazel
git commit -m "Add 4D Chebyshev-Tucker interpolant"
```

---

### Task 4: Add tests for `ChebyshevTucker4D`

**Files:**
- Create: `tests/chebyshev_tucker_4d_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write the test file**

```cpp
// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless/chebyshev_tucker_4d.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

TEST(ChebyshevTucker4DTest, ExactForLowDegreePolynomial) {
    // f(x,y,z,w) = x*y + z*w
    auto f = [](double x, double y, double z, double w) {
        return x * y + z * w;
    };

    ChebyshevTucker4DDomain domain{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    ChebyshevTucker4DConfig config{.num_pts = {4, 4, 4, 4}, .epsilon = 1e-12};
    auto interp = ChebyshevTucker4D::build(f, domain, config);

    for (double x : {-0.5, 0.3}) {
        for (double y : {-0.4, 0.7}) {
            for (double z : {-0.8, 0.1}) {
                for (double w : {-0.3, 0.6}) {
                    double expected = f(x, y, z, w);
                    double got = interp.eval({x, y, z, w});
                    EXPECT_NEAR(got, expected, 1e-10)
                        << "at (" << x << "," << y << "," << z << "," << w << ")";
                }
            }
        }
    }
}

TEST(ChebyshevTucker4DTest, SmoothFunctionConverges) {
    auto f = [](double x, double y, double z, double w) {
        return std::exp(-x * x) * std::sin(y) * std::cos(z * 0.5) * (1.0 + w);
    };

    ChebyshevTucker4DDomain domain{
        .bounds = {{{-1.0, 1.0}, {0.0, 3.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };

    auto coarse = ChebyshevTucker4D::build(
        f, domain, {.num_pts = {6, 6, 6, 6}, .epsilon = 1e-14});
    auto fine = ChebyshevTucker4D::build(
        f, domain, {.num_pts = {12, 12, 12, 6}, .epsilon = 1e-14});

    double x = 0.5, y = 1.2, z = 0.3, w = 0.4;
    double exact = f(x, y, z, w);
    double err_coarse = std::abs(coarse.eval({x, y, z, w}) - exact);
    double err_fine = std::abs(fine.eval({x, y, z, w}) - exact);

    EXPECT_LT(err_fine, err_coarse * 0.01);
    EXPECT_LT(err_fine, 1e-10);
}

TEST(ChebyshevTucker4DTest, TuckerCompressionReducesStorage) {
    auto f = [](double x, double y, double z, double w) {
        return std::exp(-x * x) * std::sin(y) * std::cos(z * 0.5) * (1.0 + w);
    };
    ChebyshevTucker4DDomain domain{
        .bounds = {{{-2.0, 2.0}, {0.0, 3.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    ChebyshevTucker4DConfig config{.num_pts = {10, 10, 10, 6}, .epsilon = 1e-8};
    auto interp = ChebyshevTucker4D::build(f, domain, config);

    size_t full_size = 10 * 10 * 10 * 6;
    size_t compressed = interp.compressed_size();
    EXPECT_LT(compressed, full_size);
}

TEST(ChebyshevTucker4DTest, PartialDerivativeAccuracy) {
    // f(x,y,z,w) = sin(x) * cos(y) * z * (1+w)
    // df/dx = cos(x) * cos(y) * z * (1+w)
    // df/dy = -sin(x) * sin(y) * z * (1+w)
    // df/dz = sin(x) * cos(y) * (1+w)
    // df/dw = sin(x) * cos(y) * z
    auto f = [](double x, double y, double z, double w) {
        return std::sin(x) * std::cos(y) * z * (1.0 + w);
    };

    ChebyshevTucker4DDomain domain{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    auto interp = ChebyshevTucker4D::build(
        f, domain, {.num_pts = {14, 14, 14, 6}, .epsilon = 1e-14});

    double x = 0.3, y = -0.4, z = 0.5, w = 0.2;
    double dx = interp.partial(0, {x, y, z, w});
    double dy = interp.partial(1, {x, y, z, w});
    double dz = interp.partial(2, {x, y, z, w});
    double dw = interp.partial(3, {x, y, z, w});

    double dx_exact = std::cos(x) * std::cos(y) * z * (1.0 + w);
    double dy_exact = -std::sin(x) * std::sin(y) * z * (1.0 + w);
    double dz_exact = std::sin(x) * std::cos(y) * (1.0 + w);
    double dw_exact = std::sin(x) * std::cos(y) * z;

    EXPECT_NEAR(dx, dx_exact, 1e-5);
    EXPECT_NEAR(dy, dy_exact, 1e-5);
    EXPECT_NEAR(dz, dz_exact, 1e-5);
    EXPECT_NEAR(dw, dw_exact, 1e-5);
}

TEST(ChebyshevTucker4DTest, DomainClampingDoesNotCrash) {
    auto f = [](double x, double y, double z, double w) { return x + y + z + w; };
    ChebyshevTucker4DDomain domain{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    auto interp = ChebyshevTucker4D::build(
        f, domain, {.num_pts = {4, 4, 4, 4}, .epsilon = 1e-12});

    // Out-of-domain queries should be clamped, not crash
    double v = interp.eval({5.0, -5.0, 10.0, -10.0});
    EXPECT_TRUE(std::isfinite(v));
}

}  // namespace
}  // namespace mango
```

**Step 2: Add test target**

In `tests/BUILD.bazel`, add:

```python
cc_test(
    name = "chebyshev_tucker_4d_test",
    size = "small",
    srcs = ["chebyshev_tucker_4d_test.cc"],
    deps = [
        "//src/option/table/dimensionless:chebyshev_tucker_4d",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3: Run the tests**

Run: `bazel test //tests:chebyshev_tucker_4d_test --test_output=all`
Expected: 5 tests PASS

**Step 4: Commit**

```bash
git add tests/chebyshev_tucker_4d_test.cc tests/BUILD.bazel
git commit -m "Add tests for 4D Chebyshev-Tucker interpolant"
```

---

### Task 5: Create `chebyshev_4d_eep_inner.hpp` benchmark adapter

Benchmark-local header that wraps `ChebyshevTucker4D` with:
- `Chebyshev4DEEPInner`: price/vega adapter using EEP decomposition
- `build_chebyshev_4d_eep()`: builder that runs N_sigma × N_rate PDE solves

The 4D parameterization uses (ln(S/K), tau, sigma, rate) — sigma is its own axis, so Newton vega uses `partial(2, coords)` directly.

**Files:**
- Create: `benchmarks/chebyshev_4d_eep_inner.hpp`
- Reference: `benchmarks/chebyshev_eep_inner.hpp` (3D adapter)
- Reference: `src/option/table/eep_transform.cpp` (EEP softplus)
- Reference: `src/option/table/eep_transform.hpp:60-78` (EEPPriceTableInner::price/vega)

**Step 1: Create the 4D benchmark adapter**

```cpp
// SPDX-License-Identifier: MIT
//
// Benchmark-local adapter and builder for 4D Chebyshev-Tucker EEP surfaces.
// Mirrors EEPPriceTableInner but wraps ChebyshevTucker4D instead of PriceTableSurface.
//
// 4D axes: (ln(S/K), tau, sigma, rate) — sigma is its own axis.
// Benchmark experiment only.
#pragma once

#include "mango/option/table/dimensionless/chebyshev_tucker_4d.hpp"
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/table/price_query.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/math/cubic_spline_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <vector>

namespace mango {

// ============================================================================
// Chebyshev4DEEPInner: price/vega adapter for ChebyshevTucker4D EEP surface
// ============================================================================

class Chebyshev4DEEPInner {
public:
    Chebyshev4DEEPInner(ChebyshevTucker4D interp, OptionType type,
                         double K_ref, double dividend_yield)
        : interp_(std::move(interp)), type_(type),
          K_ref_(K_ref), dividend_yield_(dividend_yield) {}

    // Mirrors EEPPriceTableInner::price (eep_transform.cpp:60-68)
    [[nodiscard]] double price(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        double eep = interp_.eval({x, q.tau, q.sigma, q.rate});
        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();
        return eep * (q.strike / K_ref_) + eu.value();
    }

    // Mirrors EEPPriceTableInner::vega (eep_transform.cpp:70-78)
    // sigma is axis 2 — Newton only sweeps along this one dimension.
    [[nodiscard]] double vega(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        double eep_vega = (q.strike / K_ref_) *
            interp_.partial(2, {x, q.tau, q.sigma, q.rate});
        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();
        return eep_vega + eu.vega();
    }

    [[nodiscard]] const ChebyshevTucker4D& interp() const { return interp_; }

private:
    ChebyshevTucker4D interp_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
};

// ============================================================================
// Builder: batch-PDE 4D Chebyshev EEP surface
// ============================================================================

struct Chebyshev4DEEPConfig {
    size_t num_x = 10;       // ln(S/K) nodes
    size_t num_tau = 10;     // tau nodes
    size_t num_sigma = 15;   // sigma nodes
    size_t num_rate = 6;     // rate nodes
    double epsilon = 1e-8;

    // Domain bounds — same as 4D B-spline
    double x_min = -0.50;     // ln(0.60)
    double x_max = 0.40;      // ln(1.50)
    double tau_min = 0.019;   // ~7 days
    double tau_max = 2.0;     // 2 years
    double sigma_min = 0.05;
    double sigma_max = 0.50;
    double rate_min = 0.01;
    double rate_max = 0.10;

    double dividend_yield = 0.0;  // q for EEP decomposition
};

struct Chebyshev4DEEPResult {
    ChebyshevTucker4D interp;
    int n_pde_solves;
    double build_seconds;
};

inline Chebyshev4DEEPResult build_chebyshev_4d_eep(
    const Chebyshev4DEEPConfig& cfg,
    double K_ref,
    OptionType option_type)
{
    // Per-axis headroom: 3 * domain_width / (n-1) per side
    auto headroom = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };

    double hx = headroom(cfg.x_min, cfg.x_max, cfg.num_x);
    double ht = headroom(cfg.tau_min, cfg.tau_max, cfg.num_tau);
    double hs = headroom(cfg.sigma_min, cfg.sigma_max, cfg.num_sigma);
    double hr = headroom(cfg.rate_min, cfg.rate_max, cfg.num_rate);

    // Extended bounds with per-axis floor clamping
    double x_lo = cfg.x_min - hx, x_hi = cfg.x_max + hx;
    double tau_lo = std::max(cfg.tau_min - ht, 1e-4);
    double tau_hi = cfg.tau_max + ht;
    double sig_lo = std::max(cfg.sigma_min - hs, 0.01);
    double sig_hi = cfg.sigma_max + hs;
    double rate_lo = std::max(cfg.rate_min - hr, -0.05);
    double rate_hi = cfg.rate_max + hr;

    ChebyshevTucker4DDomain dom{
        .bounds = {{{x_lo, x_hi}, {tau_lo, tau_hi},
                    {sig_lo, sig_hi}, {rate_lo, rate_hi}}}};
    ChebyshevTucker4DConfig tcfg{
        .num_pts = {cfg.num_x, cfg.num_tau, cfg.num_sigma, cfg.num_rate},
        .epsilon = cfg.epsilon};

    auto x_nodes = chebyshev_nodes(cfg.num_x, x_lo, x_hi);
    auto tau_nodes = chebyshev_nodes(cfg.num_tau, tau_lo, tau_hi);
    auto sigma_nodes = chebyshev_nodes(cfg.num_sigma, sig_lo, sig_hi);
    auto rate_nodes = chebyshev_nodes(cfg.num_rate, rate_lo, rate_hi);

    auto t0 = std::chrono::steady_clock::now();

    // One PDE per (sigma, rate) pair: N_sigma * N_rate solves
    // Each PDE uses snapshot times at tau_nodes, solving for S around K_ref
    int n_pde = 0;
    size_t total = cfg.num_x * cfg.num_tau * cfg.num_sigma * cfg.num_rate;
    std::vector<double> tensor(total, 0.0);

    // Group by (sigma, rate) — one PDE batch per pair
    for (size_t si = 0; si < cfg.num_sigma; ++si) {
        double sigma = sigma_nodes[si];

        // Build batch of rate PDEs for this sigma
        std::vector<PricingParams> batch;
        batch.reserve(cfg.num_rate);
        for (size_t ri = 0; ri < cfg.num_rate; ++ri) {
            double rate = rate_nodes[ri];
            // Max tau for this PDE (slightly beyond tau_max node for safety)
            double max_tau = tau_nodes.back() * 1.01;
            batch.emplace_back(
                OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = max_tau,
                           .rate = rate, .dividend_yield = cfg.dividend_yield,
                           .option_type = option_type},
                sigma);
        }

        BatchAmericanOptionSolver solver;
        solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
        solver.set_snapshot_times(std::span<const double>{tau_nodes});
        auto batch_result = solver.solve_batch(batch, /*use_shared_grid=*/true);
        n_pde += static_cast<int>(cfg.num_rate);

        for (size_t ri = 0; ri < cfg.num_rate; ++ri) {
            if (!batch_result.results[ri].has_value()) continue;
            const auto& result = batch_result.results[ri].value();
            auto x_grid = result.grid()->x();
            double rate = rate_nodes[ri];

            for (size_t ti = 0; ti < cfg.num_tau; ++ti) {
                double tau = tau_nodes[ti];
                auto spatial = result.at_time(ti);
                CubicSpline<double> spline;
                if (spline.build(x_grid, spatial).has_value()) continue;

                for (size_t xi = 0; xi < cfg.num_x; ++xi) {
                    double am = spline.eval(x_nodes[xi]);
                    double spot = K_ref * std::exp(x_nodes[xi]);

                    // European at this point
                    auto eu = EuropeanOptionSolver(
                        OptionSpec{.spot = spot, .strike = K_ref, .maturity = tau,
                                   .rate = rate, .dividend_yield = cfg.dividend_yield,
                                   .option_type = option_type},
                        sigma).solve().value();

                    double eep_raw = am - eu.value();

                    // Debiased softplus (matches eep_transform.cpp:41-49)
                    constexpr double kSharpness = 100.0;
                    double eep;
                    if (kSharpness * eep_raw > 500.0) {
                        eep = eep_raw;
                    } else {
                        double softplus = std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
                        double bias = std::log(2.0) / kSharpness;
                        eep = std::max(0.0, softplus - bias);
                    }

                    // Row-major: [x, tau, sigma, rate]
                    tensor[xi * cfg.num_tau * cfg.num_sigma * cfg.num_rate
                         + ti * cfg.num_sigma * cfg.num_rate
                         + si * cfg.num_rate + ri] = eep;
                }
            }
        }
    }

    auto interp = ChebyshevTucker4D::build_from_values(tensor, dom, tcfg);
    auto t1 = std::chrono::steady_clock::now();

    return {std::move(interp), n_pde,
            std::chrono::duration<double>(t1 - t0).count()};
}

}  // namespace mango
```

**Step 2: Verify it compiles (needs benchmark BUILD integration)**

This file will be verified as part of Task 7 (BUILD.bazel update). For now, just save it.

**Step 3: Commit**

```bash
git add benchmarks/chebyshev_4d_eep_inner.hpp
git commit -m "Add 4D Chebyshev EEP benchmark adapter"
```

---

### Task 6: Integrate into `interp_iv_safety.cc`

Add a Chebyshev 4D section after the existing Chebyshev 3D section. Same pattern:
`build_chebyshev_4d_surface()` → `compute_errors_chebyshev_4d()` → print heatmap.

**Files:**
- Modify: `benchmarks/interp_iv_safety.cc`

**Step 1: Add the include at the top**

After `#include "chebyshev_eep_inner.hpp"`, add:

```cpp
#include "chebyshev_4d_eep_inner.hpp"
```

**Step 2: Add builder and error function**

After `compute_errors_chebyshev()` (around line 567), add:

```cpp
// ============================================================================
// Step 4c: Chebyshev-Tucker 4D surface (ln(S/K), tau, sigma, rate)
// ============================================================================

static Chebyshev4DEEPInner build_chebyshev_4d_surface() {
    Chebyshev4DEEPConfig cfg;  // use defaults: 10x10x15x6, epsilon=1e-8

    auto t0 = std::chrono::steady_clock::now();
    auto result = build_chebyshev_4d_eep(cfg, kSpot, OptionType::PUT);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    auto ranks = result.interp.ranks();
    std::printf("  Chebyshev 4D surface: %d PDE solves, %.3fs build, "
                "ranks=(%zu,%zu,%zu,%zu)\n",
                result.n_pde_solves, elapsed,
                ranks[0], ranks[1], ranks[2], ranks[3]);

    return Chebyshev4DEEPInner(
        std::move(result.interp), OptionType::PUT, kSpot, 0.0);
}

static ErrorTable compute_errors_chebyshev_4d(const PriceGrid& prices,
                                               const Chebyshev4DEEPInner& inner,
                                               size_t vol_idx) {
    ErrorTable errors{};
    IVSolverConfig fdm_config;
    IVSolver fdm_solver(fdm_config);

    for (size_t ti = 0; ti < kNT; ++ti) {
        for (size_t si = 0; si < kNS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) {
                errors[ti][si] = std::nan("");
                continue;
            }

            IVQuery fdm_q;
            fdm_q.spot = kSpot;
            fdm_q.strike = kStrikes[si];
            fdm_q.maturity = kMaturities[ti];
            fdm_q.rate = kRate;
            fdm_q.dividend_yield = 0.0;
            fdm_q.option_type = OptionType::PUT;
            fdm_q.market_price = price;

            auto fdm_result = fdm_solver.solve(fdm_q);
            if (!fdm_result) {
                errors[ti][si] = std::nan("");
                continue;
            }

            double iv_cheb4d = brent_solve_iv(
                [&](double vol) -> double {
                    PriceQuery q{.spot = kSpot, .strike = kStrikes[si],
                                 .tau = kMaturities[ti], .sigma = vol,
                                 .rate = kRate};
                    return inner.price(q);
                },
                price);

            if (!std::isfinite(iv_cheb4d)) {
                errors[ti][si] = std::nan("");
                continue;
            }

            errors[ti][si] = std::abs(iv_cheb4d - fdm_result->implied_vol) * 10000.0;
        }
    }
    return errors;
}
```

**Step 3: Add to main()**

After the Chebyshev 3D section (around line 721), before `return 0;`, add:

```cpp
    // ================================================================
    // Chebyshev-Tucker 4D (ln(S/K), tau, sigma, rate)
    // ================================================================
    std::printf("\n\n================================================================\n");
    std::printf("Chebyshev-Tucker 4D — q=0, no dividends\n");
    std::printf("================================================================\n\n");

    std::printf("--- Building Chebyshev-Tucker 4D surface...\n");
    auto inner_cheb4d = build_chebyshev_4d_surface();

    std::printf("--- Computing Chebyshev 4D IV errors...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title_cheb4d[128];
        std::snprintf(title_cheb4d, sizeof(title_cheb4d),
                      "Chebyshev-Tucker 4D IV Error (bps) — σ=%.0f%%, q=0",
                      kVols[vi] * 100);

        auto errors_cheb4d = compute_errors_chebyshev_4d(q0_prices, inner_cheb4d, vi);
        print_heatmap(title_cheb4d, errors_cheb4d);
    }
```

**Step 4: Verify it compiles (after Task 7)**

**Step 5: Commit**

```bash
git add benchmarks/interp_iv_safety.cc
git commit -m "Add Chebyshev 4D to interp_iv_safety benchmark"
```

---

### Task 7: Integrate into `iv_interpolation_sweep.cc`

Add a `BM_Chebyshev4D_IV` benchmark mirroring `BM_Chebyshev_IV`.

**Files:**
- Modify: `benchmarks/iv_interpolation_sweep.cc`

**Step 1: Add the include**

After `#include "chebyshev_eep_inner.hpp"`, add:

```cpp
#include "chebyshev_4d_eep_inner.hpp"
```

**Step 2: Add the 4D entry and solver**

After `BM_Chebyshev_IV` registration (around line 610), add:

```cpp
// ============================================================================
// BM_Chebyshev4D_IV: Chebyshev-Tucker 4D surface (q=0)
// ============================================================================

struct Chebyshev4DEntry {
    std::shared_ptr<Chebyshev4DEEPInner> inner;
    double build_time_ms = 0;
    int n_pde_solves = 0;
};

static const Chebyshev4DEntry& get_chebyshev_4d_solver() {
    static Chebyshev4DEntry entry = [] {
        Chebyshev4DEEPConfig cfg;  // use defaults

        auto t0 = std::chrono::steady_clock::now();
        auto result = build_chebyshev_4d_eep(cfg, kSpot, OptionType::PUT);
        auto t1 = std::chrono::steady_clock::now();

        return Chebyshev4DEntry{
            .inner = std::make_shared<Chebyshev4DEEPInner>(
                std::move(result.interp), OptionType::PUT, kSpot, 0.0),
            .build_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(),
            .n_pde_solves = result.n_pde_solves,
        };
    }();
    return entry;
}

static double solve_iv_newton_chebyshev_4d(const Chebyshev4DEEPInner& inner,
                                            double spot, double strike, double tau,
                                            double rate, double market_price) {
    double sigma = 0.20;
    for (int iter = 0; iter < 30; ++iter) {
        PriceQuery q{.spot = spot, .strike = strike, .tau = tau,
                     .sigma = sigma, .rate = rate};
        double price = inner.price(q);
        double vega = inner.vega(q);
        if (std::abs(vega) < 1e-10) break;
        double step = (price - market_price) / vega;
        sigma -= step;
        sigma = std::clamp(sigma, 0.01, 5.0);
        if (std::abs(step) < 1e-8) break;
    }
    return sigma;
}

static void BM_Chebyshev4D_IV(benchmark::State& state) {
    size_t si = static_cast<size_t>(state.range(0));
    double K = kStrikes[si];
    double ref_price = get_q0_reference_prices()[si];
    if (!std::isfinite(ref_price)) {
        state.SkipWithError("q=0 reference not available");
        return;
    }

    const auto& entry = get_chebyshev_4d_solver();
    double last_iv = 0;
    for (auto _ : state) {
        last_iv = solve_iv_newton_chebyshev_4d(*entry.inner, kSpot, K, kMaturity, kRate, ref_price);
        benchmark::DoNotOptimize(last_iv);
    }

    state.SetLabel(std::format("K={:.0f} Cheb4D", K));
    state.counters["strike"] = K;
    state.counters["iv"] = last_iv;
    state.counters["iv_err_bps"] = std::abs(last_iv - kTrueVol) * 10000.0;
    state.counters["build_ms"] = entry.build_time_ms;
    state.counters["n_pde_solves"] = static_cast<double>(entry.n_pde_solves);
}

BENCHMARK(BM_Chebyshev4D_IV)
    ->DenseRange(0, static_cast<int>(kStrikes.size()) - 1, 1)
    ->Unit(benchmark::kMicrosecond);
```

**Step 3: Commit**

```bash
git add benchmarks/iv_interpolation_sweep.cc
git commit -m "Add Chebyshev 4D to IV interpolation sweep"
```

---

### Task 8: Update `benchmarks/BUILD.bazel`

Add the 4D Chebyshev deps to both benchmark targets.

**Files:**
- Modify: `benchmarks/BUILD.bazel`

**Step 1: Update `interp_iv_safety` target**

In the `srcs` list, add `"chebyshev_4d_eep_inner.hpp"`.
In the `deps` list, add `"//src/option/table/dimensionless:chebyshev_tucker_4d"`.

**Step 2: Update `iv_interpolation_sweep` target**

In the `srcs` list, add `"chebyshev_4d_eep_inner.hpp"`.
In the `deps` list, add `"//src/option/table/dimensionless:chebyshev_tucker_4d"`.

**Step 3: Verify everything compiles**

Run: `bazel build //benchmarks:interp_iv_safety //benchmarks:iv_interpolation_sweep`
Expected: BUILD SUCCESSFUL

**Step 4: Commit**

```bash
git add benchmarks/BUILD.bazel
git commit -m "Add Chebyshev 4D deps to benchmark BUILD targets"
```

---

### Task 9: Run all tests and benchmarks

Verify no regressions across the entire build.

**Step 1: Run all tests**

Run: `bazel test //...`
Expected: All tests pass (including new `tucker_decomposition_4d_test` and `chebyshev_tucker_4d_test`)

**Step 2: Build all benchmarks**

Run: `bazel build //benchmarks/...`
Expected: All benchmarks compile

**Step 3: Build Python bindings**

Run: `bazel build //src/python:mango_option`
Expected: BUILD SUCCESSFUL

**Step 4: Run the interp_iv_safety benchmark**

Run: `bazel run //benchmarks:interp_iv_safety`
Expected: See Chebyshev 4D heatmaps. Look for:
- Interior (K=90-110, T>=60d): error comparable to B-spline (~1-10 bps)
- Deep ITM/OTM: more probes solved than Chebyshev 3D (56/72)
- Any newly solved probes should be <50 bps error
