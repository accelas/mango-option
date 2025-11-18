#pragma once

#include <span>
#include <variant>
#include <expected>
#include <string>
#include <cmath>
#include <vector>

namespace mango {

struct UniformSpacing {
    double dx;
    double dx_inv;
    double dx_inv_sq;
};

struct NonUniformSpacing {
    std::span<const double> grid;
    std::span<const double> dx;
    // Precomputed arrays stored in workspace, viewed here
    std::vector<double> dx_left_inv_data;
    std::vector<double> dx_right_inv_data;
    std::vector<double> dx_center_inv_data;
    std::vector<double> w_left_data;
    std::vector<double> w_right_data;
};

class GridSpacing {
public:
    static std::expected<GridSpacing, std::string>
    create(std::span<const double> grid, std::span<const double> dx) {
        if (grid.size() < 2) {
            return std::unexpected("Grid must have at least 2 points");
        }
        if (dx.size() != grid.size() - 1) {
            return std::unexpected("dx size must be grid.size() - 1");
        }

        // Check if uniform
        const double dx0 = dx[0];
        constexpr double tol = 1e-10;
        bool uniform = true;
        for (size_t i = 1; i < dx.size(); ++i) {
            if (std::abs(dx[i] - dx0) > tol) {
                uniform = false;
                break;
            }
        }

        if (uniform) {
            return GridSpacing(UniformSpacing{
                dx0, 1.0 / dx0, 1.0 / (dx0 * dx0)
            });
        } else {
            return GridSpacing(grid, dx);
        }
    }

    bool is_uniform() const {
        return std::holds_alternative<UniformSpacing>(spacing_);
    }

    double spacing() const {
        return std::get<UniformSpacing>(spacing_).dx;
    }

    double spacing_inv() const {
        return std::get<UniformSpacing>(spacing_).dx_inv;
    }

    std::span<const double> dx_left_inv() const {
        const auto& nu = std::get<NonUniformSpacing>(spacing_);
        return nu.dx_left_inv_data;
    }

    std::span<const double> dx_right_inv() const {
        const auto& nu = std::get<NonUniformSpacing>(spacing_);
        return nu.dx_right_inv_data;
    }

    std::span<const double> dx_center_inv() const {
        const auto& nu = std::get<NonUniformSpacing>(spacing_);
        return nu.dx_center_inv_data;
    }

    std::span<const double> w_left() const {
        const auto& nu = std::get<NonUniformSpacing>(spacing_);
        return nu.w_left_data;
    }

    std::span<const double> w_right() const {
        const auto& nu = std::get<NonUniformSpacing>(spacing_);
        return nu.w_right_data;
    }

private:
    explicit GridSpacing(UniformSpacing us)
        : spacing_(us)
    {}

    GridSpacing(std::span<const double> grid, std::span<const double> dx)
        : spacing_(create_nonuniform(grid, dx))
    {}

    static NonUniformSpacing create_nonuniform(
        std::span<const double> grid,
        std::span<const double> dx)
    {
        size_t n = grid.size();
        NonUniformSpacing nu;
        nu.grid = grid;
        nu.dx = dx;

        // Allocate precomputed arrays
        nu.dx_left_inv_data.resize(n - 2);
        nu.dx_right_inv_data.resize(n - 2);
        nu.dx_center_inv_data.resize(n - 2);
        nu.w_left_data.resize(n - 2);
        nu.w_right_data.resize(n - 2);

        // Precompute values
        for (size_t i = 0; i < n - 2; ++i) {
            double dx_left = dx[i];
            double dx_right = dx[i + 1];
            double dx_center = dx_left + dx_right;

            nu.dx_left_inv_data[i] = 1.0 / dx_left;
            nu.dx_right_inv_data[i] = 1.0 / dx_right;
            nu.dx_center_inv_data[i] = 1.0 / dx_center;
            nu.w_left_data[i] = dx_right / dx_center;
            nu.w_right_data[i] = dx_left / dx_center;
        }

        return nu;
    }

    std::variant<UniformSpacing, NonUniformSpacing> spacing_;
};

}  // namespace mango
