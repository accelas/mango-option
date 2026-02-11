// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/support/parallel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace mango {

/// Domain specification for N-dimensional Chebyshev interpolant.
/// Each axis has a [lo, hi] interval.
template <size_t N>
struct Domain {
    std::array<double, N> lo{};
    std::array<double, N> hi{};
};

/// N-dimensional Chebyshev interpolant with pluggable storage policy.
///
/// Storage must provide:
///   static Storage build(vector<double> values, array<size_t, N> shape, Args...)
///   double contract(array<vector<double>, N> coeffs) const
///   size_t compressed_size() const
///
/// ChebyshevTensor<N> = ChebyshevInterpolant<N, RawTensor<N>>   (no Eigen dep)
/// ChebyshevTucker<N> = ChebyshevInterpolant<N, TuckerTensor<N>> (Tucker compression)
template <size_t N, typename Storage>
class ChebyshevInterpolant {
public:
    /// Build from pre-computed function values at Chebyshev nodes.
    ///
    /// values: flat row-major array of function values at the tensor product
    ///         of Chebyshev nodes, with shape num_pts[0] x ... x num_pts[N-1].
    /// domain: axis bounds [lo, hi] per dimension.
    /// num_pts: number of Chebyshev nodes per axis.
    /// storage_args: forwarded to Storage::build (e.g., epsilon for Tucker).
    template <typename... Args>
    [[nodiscard]] static ChebyshevInterpolant
    build_from_values(std::span<const double> values,
                      const Domain<N>& domain,
                      const std::array<size_t, N>& num_pts,
                      Args&&... storage_args) {
        ChebyshevInterpolant interp;
        interp.domain_ = domain;
        interp.num_pts_ = num_pts;

        // Generate Chebyshev nodes and barycentric weights per axis
        for (size_t d = 0; d < N; ++d) {
            interp.nodes_[d] = chebyshev_nodes(num_pts[d], domain.lo[d], domain.hi[d]);
            interp.weights_[d] = chebyshev_barycentric_weights(num_pts[d]);
        }

        // Build storage from values
        std::vector<double> vals(values.begin(), values.end());
        interp.storage_ = Storage::build(
            std::move(vals), num_pts, std::forward<Args>(storage_args)...);

        return interp;
    }

    /// Build by sampling a function at Chebyshev nodes.
    ///
    /// f: function mapping N-dim coordinates to scalar.
    /// domain: axis bounds [lo, hi] per dimension.
    /// num_pts: number of Chebyshev nodes per axis.
    /// storage_args: forwarded to Storage::build (e.g., epsilon for Tucker).
    template <typename... Args>
    [[nodiscard]] static ChebyshevInterpolant
    build(std::function<double(std::array<double, N>)> f,
          const Domain<N>& domain,
          const std::array<size_t, N>& num_pts,
          Args&&... storage_args) {
        // Compute total number of tensor product nodes
        size_t total = 1;
        for (size_t d = 0; d < N; ++d) total *= num_pts[d];

        // Generate nodes per axis
        std::array<std::vector<double>, N> nodes;
        for (size_t d = 0; d < N; ++d)
            nodes[d] = chebyshev_nodes(num_pts[d], domain.lo[d], domain.hi[d]);

        // Compute strides (row-major)
        std::array<size_t, N> strides{};
        strides[N - 1] = 1;
        for (int d = static_cast<int>(N) - 2; d >= 0; --d)
            strides[d] = strides[d + 1] * num_pts[d + 1];

        // Sample function at all tensor product nodes
        std::vector<double> values(total);
        for (size_t flat = 0; flat < total; ++flat) {
            std::array<double, N> coords{};
            size_t remaining = flat;
            for (size_t d = 0; d < N; ++d) {
                size_t idx = remaining / strides[d];
                remaining %= strides[d];
                coords[d] = nodes[d][idx];
            }
            values[flat] = f(coords);
        }

        return build_from_values(
            std::span<const double>(values),
            domain, num_pts, std::forward<Args>(storage_args)...);
    }

    /// Evaluate the interpolant at a query point.
    /// Coordinates are clamped to the domain.
    MANGO_TARGET_CLONES("default", "avx2", "avx512f")
    [[nodiscard]] double eval(std::array<double, N> query) const {
        // Clamp to domain
        for (size_t d = 0; d < N; ++d) {
            query[d] = std::clamp(query[d], domain_.lo[d], domain_.hi[d]);
        }

        // Compute barycentric coefficients per axis
        std::array<std::vector<double>, N> coeffs;
        for (size_t d = 0; d < N; ++d) {
            coeffs[d] = barycentric_coeffs(query[d], d);
        }

        return storage_.contract(coeffs);
    }

    /// Compute partial derivative along a given axis using central FD.
    /// h = 1e-6 * (hi - lo) for the given axis.
    [[nodiscard]] double partial(size_t axis, std::array<double, N> coords) const {
        double span = domain_.hi[axis] - domain_.lo[axis];
        double h = 1e-6 * span;

        auto coords_plus = coords;
        auto coords_minus = coords;
        coords_plus[axis] += h;
        coords_minus[axis] -= h;

        return (eval(coords_plus) - eval(coords_minus)) / (2.0 * h);
    }

    /// Number of stored doubles in the underlying storage.
    [[nodiscard]] size_t compressed_size() const {
        return storage_.compressed_size();
    }

    /// Access the domain.
    [[nodiscard]] const Domain<N>& domain() const { return domain_; }

    /// Access the number of points per axis.
    [[nodiscard]] const std::array<size_t, N>& num_pts() const { return num_pts_; }

private:
    /// Compute barycentric interpolation coefficients for value x on axis d.
    /// Returns a vector of length num_pts_[d] such that the interpolated
    /// value is sum_j coeffs[j] * f_j.
    [[nodiscard]] std::vector<double>
    barycentric_coeffs(double x, size_t d) const {
        const auto& nodes = nodes_[d];
        const auto& weights = weights_[d];
        size_t n = nodes.size();

        // Check for exact node match
        for (size_t j = 0; j < n; ++j) {
            if (x == nodes[j]) {
                std::vector<double> c(n, 0.0);
                c[j] = 1.0;
                return c;
            }
        }

        // Barycentric formula: c_j = w_j / (x - x_j) / sum_k w_k / (x - x_k)
        const double* w = weights.data();
        const double* nd = nodes.data();
        std::vector<double> c(n);
        double denom = 0.0;
        MANGO_PRAGMA_SIMD
        for (size_t j = 0; j < n; ++j) {
            c[j] = w[j] / (x - nd[j]);
        }
        for (size_t j = 0; j < n; ++j) {
            denom += c[j];
        }
        double inv_denom = 1.0 / denom;
        MANGO_PRAGMA_SIMD
        for (size_t j = 0; j < n; ++j) {
            c[j] *= inv_denom;
        }
        return c;
    }

    Domain<N> domain_{};
    std::array<size_t, N> num_pts_{};
    std::array<std::vector<double>, N> nodes_;
    std::array<std::vector<double>, N> weights_;
    Storage storage_;
};

}  // namespace mango
