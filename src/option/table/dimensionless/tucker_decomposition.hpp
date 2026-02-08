// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <vector>

namespace mango {

struct TuckerResult3D {
    std::vector<double> core;
    std::array<Eigen::MatrixXd, 3> factors;
    std::array<size_t, 3> shape;
    std::array<size_t, 3> ranks;
};

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
                std::array<size_t, 3> idx = {i0, i1, i2};
                size_t row = idx[mode];
                size_t col = 0;
                size_t stride = 1;
                for (int d = 2; d >= 0; --d) {
                    if (static_cast<size_t>(d) == mode) continue;
                    col += idx[d] * stride;
                    stride *= shape[d];
                }
                size_t flat = i0 * shape[1] * shape[2] + i1 * shape[2] + i2;
                M(row, col) = T[flat];
            }
        }
    }
    return M;
}

[[nodiscard]] inline TuckerResult3D
tucker_hosvd(const std::vector<double>& T,
             const std::array<size_t, 3>& shape,
             double epsilon) {
    TuckerResult3D result;
    result.shape = shape;

    for (size_t mode = 0; mode < 3; ++mode) {
        Eigen::MatrixXd M = mode_unfold(T, shape, mode);
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

    size_t R0 = result.ranks[0], R1 = result.ranks[1], R2 = result.ranks[2];

    // Contract mode-0: G0 = U0^T x_0 T  -> shape (R0, N1, N2)
    Eigen::MatrixXd M0 = mode_unfold(T, shape, 0);
    Eigen::MatrixXd G0 = result.factors[0].transpose() * M0;

    std::array<size_t, 3> shape1 = {R0, shape[1], shape[2]};
    std::vector<double> G0_vec(R0 * shape[1] * shape[2]);
    for (size_t r = 0; r < R0; ++r)
        for (size_t j = 0; j < shape[1] * shape[2]; ++j)
            G0_vec[r * shape[1] * shape[2] + j] = G0(r, j);

    // Contract mode-1: G1 = U1^T x_1 G0  -> shape (R0, R1, N2)
    Eigen::MatrixXd M1 = mode_unfold(G0_vec, shape1, 1);
    Eigen::MatrixXd G1 = result.factors[1].transpose() * M1;

    std::array<size_t, 3> shape2 = {R0, R1, shape[2]};
    std::vector<double> G1_vec(R0 * R1 * shape[2]);
    for (size_t r1 = 0; r1 < R1; ++r1)
        for (size_t j = 0; j < R0 * shape[2]; ++j)
            G1_vec[j / shape[2] * R1 * shape[2] + r1 * shape[2] + j % shape[2]] = G1(r1, j);

    // Contract mode-2: core = U2^T x_2 G1  -> shape (R0, R1, R2)
    Eigen::MatrixXd M2 = mode_unfold(G1_vec, shape2, 2);
    Eigen::MatrixXd G2 = result.factors[2].transpose() * M2;

    result.core.resize(R0 * R1 * R2);
    for (size_t r2 = 0; r2 < R2; ++r2)
        for (size_t j = 0; j < R0 * R1; ++j)
            result.core[j / R1 * R1 * R2 + j % R1 * R2 + r2] = G2(r2, j);

    return result;
}

[[nodiscard]] inline std::vector<double>
tucker_reconstruct(const TuckerResult3D& tucker) {
    auto [N0, N1, N2] = tucker.shape;
    auto [R0, R1, R2] = tucker.ranks;
    const auto& U0 = tucker.factors[0];
    const auto& U1 = tucker.factors[1];
    const auto& U2 = tucker.factors[2];

    std::vector<double> T(N0 * N1 * N2, 0.0);
    for (size_t i = 0; i < N0; ++i)
        for (size_t j = 0; j < N1; ++j)
            for (size_t k = 0; k < N2; ++k) {
                double val = 0.0;
                for (size_t r0 = 0; r0 < R0; ++r0)
                    for (size_t r1 = 0; r1 < R1; ++r1)
                        for (size_t r2 = 0; r2 < R2; ++r2)
                            val += tucker.core[r0 * R1 * R2 + r1 * R2 + r2]
                                 * U0(i, r0) * U1(j, r1) * U2(k, r2);
                T[i * N1 * N2 + j * N2 + k] = val;
            }
    return T;
}

}  // namespace mango
