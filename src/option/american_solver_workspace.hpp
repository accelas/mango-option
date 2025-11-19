/**
 * @file american_solver_workspace.hpp
 * @brief Reusable workspace for American option solving with PMR memory allocation
 */

#pragma once

#include "src/pde/core/pde_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include <memory>
#include <expected>
#include <string>
#include <memory_resource>

namespace mango {

/**
 * Workspace for American option solving with PMR-based memory allocation.
 *
 * Provides unified workspace for American option pricing with:
 * - PDEWorkspace allocated from provided memory resource
 * - GridSpacing<double> for spatial operators
 * - Grid configuration (spatial + temporal parameters)
 *
 * Example usage:
 * ```cpp
 * std::pmr::synchronized_pool_resource pool;
 * auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
 *
 * auto workspace = AmericanSolverWorkspace::create(
 *     grid_spec.value(), 1000, &pool);
 *
 * if (!workspace.has_value()) {
 *     std::cerr << "Failed: " << workspace.error() << "\n";
 *     return;
 *     }
 *
 * // Use workspace with solver
 * auto solver = AmericanPutSolver(params, workspace.value());
 * ```
 *
 * Thread safety: **NOT thread-safe for concurrent solving**.
 * Use BatchAmericanOptionSolver for parallel option pricing.
 */
class AmericanSolverWorkspace {
public:
    /**
     * Validate workspace parameters without allocation.
     */
    static std::expected<void, std::string> validate_params(
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time);

    /**
     * Factory method creates workspace from spatial grid bounds.
     *
     * @param x_min Minimum log-moneyness
     * @param x_max Maximum log-moneyness
     * @param n_space Number of spatial grid points
     * @param n_time Number of time steps
     * @return Expected containing shared workspace on success, error message on failure
     */
    static std::expected<std::shared_ptr<AmericanSolverWorkspace>, std::string>
    create(double x_min, double x_max, size_t n_space, size_t n_time);

    /**
     * Factory method creates workspace from GridSpec.
     *
     * @param grid_spec Grid specification for spatial domain
     * @param n_time Number of time steps
     * @param resource PMR memory resource for workspace allocation
     * @return Expected containing shared workspace on success, error message on failure
     */
    static std::expected<std::shared_ptr<AmericanSolverWorkspace>, std::string>
    create(const GridSpec<double>& grid_spec,
           size_t n_time,
           std::pmr::memory_resource* resource);

    PDEWorkspace* pde_workspace() const { return pde_workspace_.get(); }
    GridSpacing<double> grid_spacing() const { return *grid_spacing_; }

    std::span<const double> grid() const {
        return grid_buffer_.span();
    }

    std::span<const double> grid_span() const {
        return grid_buffer_.span();
    }

    size_t n_space() const { return grid_buffer_.size(); }
    size_t n_time() const { return n_time_; }

    double x_min() const {
        auto g = grid();
        return g.empty() ? 0.0 : g[0];
    }

    double x_max() const {
        auto g = grid();
        return g.empty() ? 0.0 : g[g.size() - 1];
    }

private:
    AmericanSolverWorkspace(GridBuffer<double> grid_buf,
                           std::shared_ptr<PDEWorkspace> pde_ws,
                           std::shared_ptr<GridSpacing<double>> spacing,
                           size_t n_time)
        : grid_buffer_(std::move(grid_buf))
        , pde_workspace_(std::move(pde_ws))
        , grid_spacing_(std::move(spacing))
        , n_time_(n_time)
    {}

    GridBuffer<double> grid_buffer_;  // Must come before pde_workspace_ (owns grid data)
    std::shared_ptr<PDEWorkspace> pde_workspace_;
    std::shared_ptr<GridSpacing<double>> grid_spacing_;
    size_t n_time_;
};

}  // namespace mango
