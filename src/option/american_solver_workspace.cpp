#include "src/option/american_solver_workspace.hpp"

namespace mango {

std::expected<std::shared_ptr<AmericanSolverWorkspace>, std::string>
AmericanSolverWorkspace::create(const GridSpec<double>& grid_spec,
                                size_t n_time,
                                std::pmr::memory_resource* resource) {
    if (!resource) {
        return std::unexpected("Memory resource cannot be null");
    }

    if (n_time == 0) {
        return std::unexpected("n_time must be positive");
    }

    // Create PDEWorkspace with PMR
    auto pde_ws_result = PDEWorkspace::create(grid_spec, resource);
    if (!pde_ws_result.has_value()) {
        return std::unexpected(pde_ws_result.error());
    }
    auto pde_ws = pde_ws_result.value();

    // Create GridSpacing from PDEWorkspace grid
    auto grid_buffer = grid_spec.generate();
    auto grid_spacing = std::make_shared<GridSpacing<double>>(grid_buffer.view());

    return std::shared_ptr<AmericanSolverWorkspace>(
        new AmericanSolverWorkspace(std::move(grid_buffer), pde_ws, std::move(grid_spacing), n_time));
}

}  // namespace mango
