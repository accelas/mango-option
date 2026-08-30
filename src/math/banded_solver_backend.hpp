// SPDX-License-Identifier: MIT
/**
 * @file banded_solver_backend.hpp
 * @brief Backend-agnostic vocabulary for banded linear-solver backends
 *
 * Defines `BandedResult<T>` and the `BandedSolverBackend` concept that a
 * banded-solver backend (e.g. LAPACK, Eigen) must satisfy. No LAPACK
 * includes here — this header is deliberately implementation-neutral so
 * that consumers (e.g. bspline_collocation.hpp) can be templated on a
 * backend policy type without depending on any particular linear-algebra
 * library.
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <optional>
#include <span>
#include <string_view>

namespace mango {

/// Result type for banded matrix operations
template<std::floating_point T>
struct BandedResult {
    bool success;
    std::optional<std::string_view> error;

    /// Implicit conversion to bool for easy checking
    constexpr explicit operator bool() const noexcept { return success; }

    /// Check if operation succeeded
    [[nodiscard]] constexpr bool ok() const noexcept { return success; }

    /// Get error message (empty if successful)
    [[nodiscard]] constexpr std::string_view message() const noexcept {
        return error.value_or("");
    }

    /// Create success result
    [[nodiscard]] static constexpr BandedResult ok_result() noexcept {
        return BandedResult{.success = true, .error = std::nullopt};
    }

    /// Create error result
    [[nodiscard]] static constexpr BandedResult error_result(std::string_view msg) noexcept {
        return BandedResult{.success = false, .error = msg};
    }
};

/// A banded linear-solver backend: a stateless policy type owning its
/// factor-storage layout, its pivot representation, its sizing, and every
/// call into the underlying linear-algebra library.
///
/// `band_rows` is the solver's neutral band form: n×bandwidth row-major
/// values plus a per-row first-column index (exactly what
/// BSplineCollocation1D::build_collocation_matrix produces).
template<typename B, typename T>
concept BandedSolverBackend =
    std::floating_point<T> &&
    requires(std::span<const T> band_rows, std::span<const int> col_start,
             std::span<T> factors, std::span<typename B::pivot_type> pivots,
             std::span<const T> cfactors,
             std::span<const typename B::pivot_type> cpivots,
             std::span<T> x, T norm, std::size_t n, std::size_t bandwidth)
{
    typename B::pivot_type;
    { B::factor_storage_size(n, bandwidth) } -> std::convertible_to<std::size_t>;
    { B::pivot_storage_size(n, bandwidth) } -> std::convertible_to<std::size_t>;
    { B::pack(band_rows, col_start, n, bandwidth, factors) };
    { B::factorize(factors, pivots, n, bandwidth) } -> std::same_as<BandedResult<T>>;
    { B::solve(cfactors, cpivots, x, n, bandwidth) } -> std::same_as<BandedResult<T>>;
    { B::condition(cfactors, cpivots, norm, n, bandwidth) } -> std::convertible_to<T>;
};

}  // namespace mango
