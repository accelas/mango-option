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
    for (size_t j = 0; j <= n; ++j) {
        double t = std::cos(static_cast<double>(j) * M_PI / static_cast<double>(n));
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

}  // namespace mango
