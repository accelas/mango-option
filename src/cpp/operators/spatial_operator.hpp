#pragma once

#include "grid_spacing.hpp"
#include "centered_difference.hpp"
#include <span>
#include <memory>
#include <type_traits>
#include <cassert>

namespace mango::operators {

/// Helper to describe stencil interior range
struct StencilInterior {
    size_t start;  // First interior point
    size_t end;    // One past last interior point
};

/// Type trait to detect time-dependent PDEs
template<typename PDE, typename = void>
struct has_time_param : std::false_type {};

template<typename PDE>
struct has_time_param<PDE, std::void_t<
    decltype(std::declval<PDE>()(
        std::declval<double>(),  // t
        std::declval<double>(),  // d2u
        std::declval<double>(),  // du
        std::declval<double>()   // u
    ))
>> : std::true_type {};

template<typename PDE>
inline constexpr bool has_time_param_v = has_time_param<PDE>::value;

/// SpatialOperator: Composes PDE, GridSpacing, and CenteredDifference
template<typename PDE, typename T = double>
class SpatialOperator {
public:
    SpatialOperator(PDE pde, std::shared_ptr<GridSpacing<T>> spacing)
        : pde_(std::move(pde))
        , spacing_(std::move(spacing))
        , stencil_(*spacing_)
    {}

    /// Get interior range for this stencil (3-point: [1, n-1))
    /// Precondition: n >= GridSpacing<T>::min_stencil_size() (i.e., n >= 3)
    StencilInterior interior_range(size_t n) const {
        assert(n >= GridSpacing<T>::min_stencil_size() && "Grid too small for stencil");
        return {1, n - 1};  // 3-point stencil width
    }

    /// Apply operator to full grid (convenience)
    void apply(double t, std::span<const T> u, std::span<T> Lu) const {
        const auto range = interior_range(u.size());
        apply_interior(t, u, Lu, range.start, range.end);
    }

    /// Apply operator to interior points only [start, end)
    /// Used by both full-grid and cache-blocked evaluation
    void apply_interior(double t,
                       std::span<const T> u,
                       std::span<T> Lu,
                       size_t start,
                       size_t end) const {
        // Create evaluator lambda that handles time parameter
        auto eval = [&](T d2u, T du, T val) -> T {
            if constexpr (has_time_param_v<PDE>) {
                return pde_(t, d2u, du, val);  // Time-dependent PDE
            } else {
                return pde_(d2u, du, val);     // Time-independent PDE
            }
        };

        // Dispatch to appropriate stencil strategy
        if (spacing_->is_uniform()) {
            stencil_.apply_uniform(u, Lu, start, end, eval);
        } else {
            stencil_.apply_non_uniform(u, Lu, start, end, eval);
        }
    }

    /// Greeks computation (delegates to stencil)
    void compute_first_derivative(std::span<const T> u,
                                 std::span<T> du_dx) const {
        const auto range = interior_range(u.size());
        stencil_.compute_all_first(u, du_dx, range.start, range.end);
    }

    void compute_second_derivative(std::span<const T> u,
                                  std::span<T> d2u_dx2) const {
        const auto range = interior_range(u.size());
        stencil_.compute_all_second(u, d2u_dx2, range.start, range.end);
    }

private:
    PDE pde_;  // Owned by value (PDEs are typically small)
    std::shared_ptr<GridSpacing<T>> spacing_;
    CenteredDifference<T> stencil_;
};

} // namespace mango::operators
