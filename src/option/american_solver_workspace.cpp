#include "src/option/american_solver_workspace.hpp"
#include "src/pde/core/time_domain.hpp"
#include <memory_resource>

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

    // Generate grid
    auto grid_buffer = grid_spec.generate();
    size_t n = grid_buffer.size();

    // Create TimeDomain for GridWithSolution
    // For American options, we solve backward from maturity (t_start=0) to present (t_end=maturity)
    // The time domain represents backward time τ = T - t
    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, 1.0, n_time);

    // Create GridWithSolution (Grid + solution storage)
    auto grid_with_sol = GridWithSolution<double>::create(grid_spec, time_domain);
    if (!grid_with_sol.has_value()) {
        return std::unexpected(grid_with_sol.error());
    }

    // Allocate contiguous PMR buffer for PDEWorkspaceSpans
    size_t buffer_size = PDEWorkspaceSpans::required_size(n);
    std::pmr::vector<double> pmr_buffer(buffer_size, 0.0, resource);

    // Create PDEWorkspaceSpans from buffer and grid
    auto workspace_spans_result = PDEWorkspaceSpans::from_buffer_and_grid(
        std::span{pmr_buffer.data(), pmr_buffer.size()},
        grid_with_sol.value()->x(),
        n
    );

    if (!workspace_spans_result.has_value()) {
        return std::unexpected(workspace_spans_result.error());
    }

    return std::shared_ptr<AmericanSolverWorkspace>(
        new AmericanSolverWorkspace(
            grid_with_sol.value(),
            std::move(pmr_buffer),
            workspace_spans_result.value()
        )
    );
}

}  // namespace mango
