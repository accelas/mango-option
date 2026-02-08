// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/table/dimensionless/tucker_decomposition_4d.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <span>
#include <vector>

namespace mango {

/// Domain bounds for 4D Chebyshev-Tucker interpolation.
struct ChebyshevTucker4DDomain {
    std::array<std::array<double, 2>, 4> bounds;  ///< {{a0,b0}, {a1,b1}, {a2,b2}, {a3,b3}}
};

/// Configuration for 4D Chebyshev-Tucker interpolant.
struct ChebyshevTucker4DConfig {
    std::array<size_t, 4> num_pts = {10, 10, 10, 6};  ///< Sample points per axis
    double epsilon = 1e-8;                              ///< Tucker truncation threshold
};

/// 4D Chebyshev interpolant with Tucker compression.
///
/// Build: sample function on Chebyshev tensor grid, compress via HOSVD.
/// Eval: barycentric interpolation contracted with Tucker factors.
class ChebyshevTucker4D {
public:
    using SampleFn = std::function<double(double, double, double, double)>;

    /// Build interpolant by sampling f on Chebyshev nodes.
    [[nodiscard]] static ChebyshevTucker4D
    build(SampleFn f, const ChebyshevTucker4DDomain& domain,
          const ChebyshevTucker4DConfig& config) {
        ChebyshevTucker4D interp;
        interp.domain_ = domain;

        // Generate nodes and weights per axis
        for (size_t d = 0; d < 4; ++d) {
            auto [a, b] = domain.bounds[d];
            interp.nodes_[d] = chebyshev_nodes(config.num_pts[d], a, b);
            interp.weights_[d] = chebyshev_barycentric_weights(config.num_pts[d]);
        }

        // Sample function on full tensor grid (row-major)
        auto& n = config.num_pts;
        std::vector<double> T(n[0] * n[1] * n[2] * n[3]);
        for (size_t i0 = 0; i0 < n[0]; ++i0)
            for (size_t i1 = 0; i1 < n[1]; ++i1)
                for (size_t i2 = 0; i2 < n[2]; ++i2)
                    for (size_t i3 = 0; i3 < n[3]; ++i3)
                        T[i0 * n[1] * n[2] * n[3] + i1 * n[2] * n[3]
                          + i2 * n[3] + i3] =
                            f(interp.nodes_[0][i0], interp.nodes_[1][i1],
                              interp.nodes_[2][i2], interp.nodes_[3][i3]);

        // Tucker compress
        interp.tucker_ =
            tucker_hosvd_4d(T, {n[0], n[1], n[2], n[3]}, config.epsilon);

        return interp;
    }

    /// Build interpolant from pre-computed tensor values on Chebyshev nodes.
    /// values: row-major tensor of shape num_pts[0] x ... x num_pts[3],
    /// sampled at CGL nodes on the domain (same ordering as build() would use).
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
        interp.tucker_ =
            tucker_hosvd_4d(T, {n[0], n[1], n[2], n[3]}, config.epsilon);

        return interp;
    }

    /// Evaluate at a 4D point using Tucker-contracted barycentric interpolation.
    /// Out-of-domain queries are clamped to domain boundaries.
    [[nodiscard]] double eval(const std::array<double, 4>& query) const {
        // Clamp to domain (matches B-spline boundary behavior)
        std::array<double, 4> q = query;
        for (size_t d = 0; d < 4; ++d) {
            q[d] = std::clamp(q[d], domain_.bounds[d][0], domain_.bounds[d][1]);
        }

        auto [R0, R1, R2, R3] = tucker_.ranks;

        // Step 1: Barycentric weights -> factor-contracted coefficients per axis
        // For each axis d, interpolate each column r of U_d using barycentric formula:
        //   c_r = sum_j [w_j/(x-x_j)] * U_d(j,r) / sum_j [w_j/(x-x_j)]
        std::array<std::vector<double>, 4> contracted;

        for (size_t d = 0; d < 4; ++d) {
            size_t R = tucker_.ranks[d];
            contracted[d].resize(R);

            // Check if query coincides with a node
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
                // Exact at node: c_r = U_d(node_idx, r)
                for (size_t r = 0; r < R; ++r) {
                    contracted[d][r] = tucker_.factors[d](node_idx, r);
                }
            } else {
                // Barycentric interpolation of each column of U_d
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

        // Step 2: Contract with core tensor
        double result = 0.0;
        for (size_t r0 = 0; r0 < R0; ++r0)
            for (size_t r1 = 0; r1 < R1; ++r1)
                for (size_t r2 = 0; r2 < R2; ++r2)
                    for (size_t r3 = 0; r3 < R3; ++r3)
                        result +=
                            tucker_.core[r0 * R1 * R2 * R3 + r1 * R2 * R3
                                         + r2 * R3 + r3]
                            * contracted[0][r0] * contracted[1][r1]
                            * contracted[2][r2] * contracted[3][r3];

        return result;
    }

    /// Partial derivative along one axis via central finite difference.
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

    /// Number of stored coefficients (core + factor entries).
    [[nodiscard]] size_t compressed_size() const {
        auto [R0, R1, R2, R3] = tucker_.ranks;
        size_t core_size = R0 * R1 * R2 * R3;
        size_t factor_size = 0;
        for (size_t d = 0; d < 4; ++d)
            factor_size += nodes_[d].size() * tucker_.ranks[d];
        return core_size + factor_size;
    }

    /// Tucker ranks per mode.
    [[nodiscard]] std::array<size_t, 4> ranks() const { return tucker_.ranks; }

    /// Number of sample points per axis.
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
