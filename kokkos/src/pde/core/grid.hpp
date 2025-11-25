#pragma once

#include <Kokkos_Core.hpp>
#include <expected>
#include <string>
#include <cmath>
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

/// Grid error codes
enum class GridError {
    InvalidSize,
    InvalidBounds,
    AllocationFailed
};

/// Grid with Kokkos View storage
///
/// Template on MemSpace for CPU/GPU portability.
/// Owns spatial coordinates and solution arrays.
template <typename MemSpace>
class Grid {
public:
    using view_type = Kokkos::View<double*, MemSpace>;

    /// Factory: uniform grid
    [[nodiscard]] static std::expected<Grid, GridError>
    uniform(double x_min, double x_max, size_t n_points) {
        if (n_points < 2) {
            return std::unexpected(GridError::InvalidSize);
        }
        if (x_min >= x_max) {
            return std::unexpected(GridError::InvalidBounds);
        }

        Grid grid;
        grid.n_points_ = n_points;
        grid.x_min_ = x_min;
        grid.x_max_ = x_max;

        // Allocate Views
        grid.x_ = view_type("x", n_points);
        grid.u_current_ = view_type("u_current", n_points);
        grid.u_prev_ = view_type("u_prev", n_points);

        // Initialize x coordinates (on host, then copy if needed)
        auto x_host = Kokkos::create_mirror_view(grid.x_);
        double dx = (x_max - x_min) / static_cast<double>(n_points - 1);
        for (size_t i = 0; i < n_points; ++i) {
            x_host(i) = x_min + static_cast<double>(i) * dx;
        }
        Kokkos::deep_copy(grid.x_, x_host);

        return grid;
    }

    /// Factory: sinh-spaced grid (concentrates points at center)
    [[nodiscard]] static std::expected<Grid, GridError>
    sinh_spaced(double x_min, double x_max, size_t n_points, double alpha = 2.0) {
        if (n_points < 2) {
            return std::unexpected(GridError::InvalidSize);
        }
        if (x_min >= x_max) {
            return std::unexpected(GridError::InvalidBounds);
        }
        if (alpha <= 0.0) {
            return std::unexpected(GridError::InvalidBounds);
        }

        Grid grid;
        grid.n_points_ = n_points;
        grid.x_min_ = x_min;
        grid.x_max_ = x_max;

        grid.x_ = view_type("x", n_points);
        grid.u_current_ = view_type("u_current", n_points);
        grid.u_prev_ = view_type("u_prev", n_points);

        auto x_host = Kokkos::create_mirror_view(grid.x_);
        double x_mid = 0.5 * (x_min + x_max);
        double L = 0.5 * (x_max - x_min);

        for (size_t i = 0; i < n_points; ++i) {
            double xi = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n_points - 1);
            x_host(i) = x_mid + L * std::sinh(alpha * xi) / std::sinh(alpha);
        }
        Kokkos::deep_copy(grid.x_, x_host);

        return grid;
    }

    // Accessors
    [[nodiscard]] size_t n_points() const { return n_points_; }
    [[nodiscard]] double x_min() const { return x_min_; }
    [[nodiscard]] double x_max() const { return x_max_; }

    [[nodiscard]] view_type x() const { return x_; }
    [[nodiscard]] view_type u_current() const { return u_current_; }
    [[nodiscard]] view_type u_prev() const { return u_prev_; }

    /// Swap current and previous solution
    void swap_solutions() {
        std::swap(u_current_, u_prev_);
    }

private:
    Grid() = default;

    size_t n_points_ = 0;
    double x_min_ = 0.0;
    double x_max_ = 0.0;

    view_type x_;
    view_type u_current_;
    view_type u_prev_;
};

}  // namespace mango::kokkos
