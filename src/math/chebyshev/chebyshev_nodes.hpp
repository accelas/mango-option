// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <span>
#include <vector>

namespace mango {

/// Generate num_pts Chebyshev-Gauss-Lobatto nodes on [a, b], sorted ascending.
/// Polynomial degree = num_pts - 1.  Requires num_pts >= 2.
[[nodiscard]] inline std::vector<double>
chebyshev_nodes(size_t num_pts, double a, double b) {
    if (num_pts < 2) return {};
    const size_t n = num_pts - 1;
    std::vector<double> nodes(num_pts);
    for (size_t j = 0; j <= n; ++j) {
        double t = std::cos(static_cast<double>(j) * M_PI / static_cast<double>(n));
        nodes[n - j] = (b + a) / 2.0 + (b - a) / 2.0 * t;
    }
    return nodes;
}

/// Barycentric weights for num_pts Chebyshev-Gauss-Lobatto nodes.
/// w_j = (-1)^j * delta_j, where delta_j = 1/2 for endpoints, 1 otherwise.
/// Returned in ascending node order (reversed from standard CGL ordering).
/// Requires num_pts >= 2.
[[nodiscard]] inline std::vector<double>
chebyshev_barycentric_weights(size_t num_pts) {
    if (num_pts < 2) return {};
    const size_t n = num_pts - 1;
    std::vector<double> w(num_pts);
    for (size_t j = 0; j <= n; ++j) {
        double sign = (j % 2 == 0) ? 1.0 : -1.0;
        double delta = (j == 0 || j == n) ? 0.5 : 1.0;
        w[n - j] = sign * delta;
    }
    return w;
}

/// Evaluate barycentric Chebyshev interpolant at point x.
[[nodiscard]] inline double
chebyshev_interpolate(double x,
                      std::span<const double> nodes,
                      std::span<const double> values,
                      std::span<const double> weights) {
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

/// Generate Clenshaw-Curtis nodes at level l on [a, b].
/// Returns 2^l + 1 nodes (same as CGL nodes at that count), sorted ascending.
/// Levels are nested: every node at level l appears at level l+1.
[[nodiscard]] inline std::vector<double>
cc_level_nodes(size_t level, double a, double b) {
    return chebyshev_nodes((1u << level) + 1, a, b);
}

/// Return only the NEW nodes introduced at level l (not present at level l-1).
/// At level 0, returns all 2 nodes (both endpoints).
/// At level l >= 1, returns 2^(l-1) new interior nodes, sorted ascending.
[[nodiscard]] inline std::vector<double>
cc_new_nodes_at_level(size_t level, double a, double b) {
    auto all = cc_level_nodes(level, a, b);
    if (level == 0) return all;
    auto prev = cc_level_nodes(level - 1, a, b);
    std::vector<double> result;
    result.reserve(all.size() - prev.size());
    size_t pi = 0;
    for (double node : all) {
        if (pi < prev.size() && std::abs(node - prev[pi]) < 1e-14 * (b - a + 1.0)) {
            ++pi;
        } else {
            result.push_back(node);
        }
    }
    return result;
}

}  // namespace mango
