// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/table/dimensionless/tucker_decomposition.hpp"

#include <array>
#include <functional>
#include <span>
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

    /// Build interpolant from pre-computed tensor values on Chebyshev nodes.
    /// values: row-major tensor of shape num_pts[0] x num_pts[1] x num_pts[2],
    /// sampled at CGL nodes on the domain (same ordering as build() would use).
    [[nodiscard]] static ChebyshevTucker3D
    build_from_values(std::span<const double> values,
                      const ChebyshevTuckerDomain& domain,
                      const ChebyshevTuckerConfig& config) {
        ChebyshevTucker3D interp;
        interp.domain_ = domain;

        for (size_t d = 0; d < 3; ++d) {
            auto [a, b] = domain.bounds[d];
            interp.nodes_[d] = chebyshev_nodes(config.num_pts[d], a, b);
            interp.weights_[d] = chebyshev_barycentric_weights(config.num_pts[d]);
        }

        auto& n = config.num_pts;
        std::vector<double> T(values.begin(), values.end());
        interp.tucker_ = tucker_hosvd(T, {n[0], n[1], n[2]}, config.epsilon);

        return interp;
    }

    /// Evaluate at a 3D point using Tucker-contracted barycentric interpolation.
    [[nodiscard]] double eval(const std::array<double, 3>& query) const {
        auto [R0, R1, R2] = tucker_.ranks;

        // Step 1: Barycentric weights -> factor-contracted coefficients per axis
        // For each axis d, interpolate each column r of U_d using barycentric formula:
        //   c_r = sum_j [w_j/(x-x_j)] * U_d(j,r) / sum_j [w_j/(x-x_j)]
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
