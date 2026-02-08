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

    for (size_t i0 = 0; i0 < shape[0]; ++i0) {
        for (size_t i1 = 0; i1 < shape[1]; ++i1) {
            for (size_t i2 = 0; i2 < shape[2]; ++i2) {
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
            }
        }
    }
    return M;
}

[[nodiscard]] inline TuckerResult4D
tucker_hosvd_4d(const std::vector<double>& T,
                const std::array<size_t, 4>& shape,
                double epsilon) {
    TuckerResult4D result;
    result.shape = shape;

    // Compute truncated SVD for each mode
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

    size_t R0 = result.ranks[0];
    size_t R1 = result.ranks[1];
    size_t R2 = result.ranks[2];
    size_t R3 = result.ranks[3];
    size_t N1 = shape[1], N2 = shape[2], N3 = shape[3];

    // Contract mode-0: G0 = U0^T x_0 T  -> shape (R0, N1, N2, N3)
    Eigen::MatrixXd M0 = mode_unfold_4d(T, shape, 0);
    Eigen::MatrixXd G0 = result.factors[0].transpose() * M0;

    // G0 rows = R0, cols = N1*N2*N3 (column order matches mode-0 unfold)
    // For mode-0 unfold, col = i3 + i2*N3 + i1*N2*N3
    // Row-major flat: r0*N1*N2*N3 + i1*N2*N3 + i2*N3 + i3
    // Since mode-0 unfold col order is i3 + i2*N3 + i1*N2*N3 (right-to-left),
    // col j directly maps to the non-mode-0 portion of row-major order.
    std::array<size_t, 4> shape1 = {R0, N1, N2, N3};
    std::vector<double> G0_vec(R0 * N1 * N2 * N3);
    for (size_t r = 0; r < R0; ++r)
        for (size_t j = 0; j < N1 * N2 * N3; ++j)
            G0_vec[r * N1 * N2 * N3 + j] = G0(r, j);

    // Contract mode-1: G1 = U1^T x_1 G0  -> shape (R0, R1, N2, N3)
    // Mode-1 unfold of (R0, N1, N2, N3): rows=N1, cols=R0*N2*N3
    // Column order (right-to-left skipping mode-1): col = i3 + i2*N3 + i0*N2*N3
    // So col j: i0 = j / (N2*N3), remainder = j % (N2*N3)
    // Repack to row-major (R0, R1, N2, N3): flat = i0*R1*N2*N3 + r1*N2*N3 + rem
    Eigen::MatrixXd M1 = mode_unfold_4d(G0_vec, shape1, 1);
    Eigen::MatrixXd G1 = result.factors[1].transpose() * M1;

    std::array<size_t, 4> shape2 = {R0, R1, N2, N3};
    std::vector<double> G1_vec(R0 * R1 * N2 * N3);
    size_t tail1 = N2 * N3;
    for (size_t r1 = 0; r1 < R1; ++r1)
        for (size_t j = 0; j < R0 * tail1; ++j) {
            size_t i0 = j / tail1;
            size_t rem = j % tail1;
            G1_vec[i0 * R1 * tail1 + r1 * tail1 + rem] = G1(r1, j);
        }

    // Contract mode-2: G2 = U2^T x_2 G1  -> shape (R0, R1, R2, N3)
    // Mode-2 unfold of (R0, R1, N2, N3): rows=N2, cols=R0*R1*N3
    // Column order (right-to-left skipping mode-2): col = i3 + i1*N3 + i0*R1*N3
    // So col j: i0 = j / (R1*N3), rem1 = j % (R1*N3), i1 = rem1 / N3, i3 = rem1 % N3
    // Repack to row-major (R0, R1, R2, N3): flat = i0*R1*R2*N3 + i1*R2*N3 + r2*N3 + i3
    Eigen::MatrixXd M2 = mode_unfold_4d(G1_vec, shape2, 2);
    Eigen::MatrixXd G2 = result.factors[2].transpose() * M2;

    std::array<size_t, 4> shape3 = {R0, R1, R2, N3};
    std::vector<double> G2_vec(R0 * R1 * R2 * N3);
    size_t tail2 = R1 * N3;
    for (size_t r2 = 0; r2 < R2; ++r2)
        for (size_t j = 0; j < R0 * tail2; ++j) {
            size_t i0 = j / tail2;
            size_t rem1 = j % tail2;
            size_t i1 = rem1 / N3;
            size_t i3 = rem1 % N3;
            G2_vec[i0 * R1 * R2 * N3 + i1 * R2 * N3 + r2 * N3 + i3] =
                G2(r2, j);
        }

    // Contract mode-3: core = U3^T x_3 G2  -> shape (R0, R1, R2, R3)
    // Mode-3 unfold of (R0, R1, R2, N3): rows=N3, cols=R0*R1*R2
    // Column order (right-to-left skipping mode-3): col = i2 + i1*R2 + i0*R1*R2
    // So col j: i0 = j / (R1*R2), rem = j % (R1*R2), i1 = rem / R2, i2 = rem % R2
    // Repack to row-major (R0, R1, R2, R3): flat = i0*R1*R2*R3 + i1*R2*R3 + i2*R3 + r3
    Eigen::MatrixXd M3 = mode_unfold_4d(G2_vec, shape3, 3);
    Eigen::MatrixXd G3 = result.factors[3].transpose() * M3;

    result.core.resize(R0 * R1 * R2 * R3);
    size_t tail3 = R1 * R2;
    for (size_t r3 = 0; r3 < R3; ++r3)
        for (size_t j = 0; j < R0 * tail3; ++j) {
            size_t i0 = j / tail3;
            size_t rem = j % tail3;
            size_t i1 = rem / R2;
            size_t i2 = rem % R2;
            result.core[i0 * R1 * R2 * R3 + i1 * R2 * R3 + i2 * R3 + r3] =
                G3(r3, j);
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
                                    val += tucker.core[r0 * R1 * R2 * R3
                                                     + r1 * R2 * R3
                                                     + r2 * R3 + r3]
                                         * U0(i, r0) * U1(j, r1)
                                         * U2(k, r2) * U3(l, r3);
                    T[i * N1 * N2 * N3 + j * N2 * N3 + k * N3 + l] = val;
                }
    return T;
}

}  // namespace mango
