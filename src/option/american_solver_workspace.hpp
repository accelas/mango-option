/**
 * @file american_solver_workspace.hpp
 * @brief Reusable workspace for American option solving with PMR memory allocation
 */

#pragma once

#include "src/pde/core/pde_workspace_pmr.hpp"
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

    std::shared_ptr<PDEWorkspace> pde_workspace() const { return pde_workspace_; }
    std::shared_ptr<GridSpacing<double>> grid_spacing() const { return grid_spacing_; }

    std::span<const double> grid() const {
        return pde_workspace_->grid().subspan(0, pde_workspace_->logical_size());
    }

    std::span<const double> grid_span() const {
        return pde_workspace_->grid().subspan(0, pde_workspace_->logical_size());
    }

    size_t n_space() const { return pde_workspace_->logical_size(); }
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
    AmericanSolverWorkspace(std::shared_ptr<PDEWorkspace> pde_ws,
                           std::shared_ptr<GridSpacing<double>> spacing,
                           size_t n_time)
        : pde_workspace_(std::move(pde_ws))
        , grid_spacing_(std::move(spacing))
        , n_time_(n_time)
    {}

    std::shared_ptr<PDEWorkspace> pde_workspace_;
    std::shared_ptr<GridSpacing<double>> grid_spacing_;
    size_t n_time_;
};

}  // namespace mango
