#include "src/option/american_solver_workspace.hpp"

namespace mango {

std::expected<void, std::string>
AmericanSolverWorkspace::validate_params(double x_min, double x_max, size_t n_space, size_t n_time) {
    if (x_min >= x_max) {
        return std::unexpected("x_min must be < x_max");
    }
    if (n_space < 3) {
        return std::unexpected("n_space must be >= 3");
    }
    if (n_time < 1) {
        return std::unexpected("n_time must be >= 1");
    }
    return {};
}

std::expected<std::shared_ptr<AmericanSolverWorkspace>, std::string>
AmericanSolverWorkspace::create(size_t n_space, size_t n_time) {
    // Use standard log-moneyness bounds [-3.0, 3.0]
    return create(-3.0, 3.0, n_space, n_time);
}

std::expected<std::shared_ptr<AmericanSolverWorkspace>, std::string>
AmericanSolverWorkspace::create(double x_min, double x_max, size_t n_space, size_t n_time) {
    // Use uniform grid (value_at() interpolation assumes uniform spacing)
    auto grid_spec = GridSpec<double>::uniform(x_min, x_max, n_space);
    if (!grid_spec.has_value()) {
        return std::unexpected(grid_spec.error());
    }

    return create(grid_spec.value(), n_time, std::pmr::get_default_resource());
}

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
