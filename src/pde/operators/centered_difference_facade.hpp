#pragma once

#include "src/pde/core/grid.hpp"
#include "centered_difference_scalar.hpp"
#include <span>
#include <concepts>

namespace mango::operators {

/**
 * CenteredDifference: Simplified facade using ScalarBackend only
 *
 * Uses OpenMP SIMD with [[gnu::target_clones]] for automatic ISA selection.
 * No Mode enum, no virtual dispatch - direct calls to ScalarBackend.
 *
 * @tparam T Floating-point type (float, double, long double)
 */
template<std::floating_point T = double>
class CenteredDifference {
public:
    explicit CenteredDifference(const GridSpacing<T>& spacing)
        : backend_(spacing)
    {}

    // Movable and copyable (ScalarBackend is copyable)
    CenteredDifference(const CenteredDifference&) = default;
    CenteredDifference& operator=(const CenteredDifference&) = default;
    CenteredDifference(CenteredDifference&&) = default;
    CenteredDifference& operator=(CenteredDifference&&) = default;

    // Public API - direct call to ScalarBackend (no virtual dispatch)
    void compute_second_derivative(std::span<const T> u,
                                   std::span<T> d2u_dx2,
                                   size_t start, size_t end) const {
        backend_.compute_second_derivative(u, d2u_dx2, start, end);
    }

    void compute_first_derivative(std::span<const T> u,
                                  std::span<T> du_dx,
                                  size_t start, size_t end) const {
        backend_.compute_first_derivative(u, du_dx, start, end);
    }

private:
    ScalarBackend<T> backend_;
};

} // namespace mango::operators
