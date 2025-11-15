#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/core/spatial_operators.hpp"
#include "src/pde/core/root_finding.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // Heat equation: ∂u/∂t = ∂²u/∂x²
    // Domain: x ∈ [0, 1], t ∈ [0, 0.1]
    // BCs: u(0, t) = 0, u(1, t) = 0
    // IC: u(x, 0) = sin(πx)

    const size_t n = 101;
    auto grid_spec_result = mango::GridSpec<double>::uniform(0.0, 1.0, n);
    if (!grid_spec_result) {
        std::cerr << "Failed to create grid spec: " << grid_spec_result.error() << "\n";
        return 1;
    }
    auto grid_buffer = grid_spec_result->generate();

    mango::TimeDomain time(0.0, 0.1, 0.001);

    mango::TRBDF2Config trbdf2_config{
        .max_iter = 20,
        .tolerance = 1e-6,
        .jacobian_fd_epsilon = 1e-7
    };

    // Boundary conditions
    mango::DirichletBC left_bc{[](double, double) { return 0.0; }};
    mango::DirichletBC right_bc{[](double, double) { return 0.0; }};

    // Spatial operator: L(u) = ∂²u/∂x²
    mango::LaplacianOperator spatial_op{1.0};  // Diffusion coefficient D = 1.0

    // Create solver with Newton integration
    mango::PDESolver solver(grid_buffer.span(), time, trbdf2_config,
                           left_bc, right_bc, spatial_op);

    // Initial condition: u(x, 0) = sin(πx)
    auto initial_condition = [](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(M_PI * x[i]);
        }
    };
    solver.initialize(initial_condition);

    std::cout << "Solving heat equation with Newton-Raphson...\n";
    std::cout << "Grid size: " << n << "\n";
    std::cout << "Time steps: " << time.n_steps() << "\n";
    std::cout << "Newton config: max_iter=" << trbdf2_config.max_iter
              << ", tol=" << trbdf2_config.tolerance << "\n\n";

    auto status = solver.solve();

    if (status) {
        std::cout << "Solver converged successfully!\n\n";

        auto solution = solver.solution();

        // Print solution at a few points
        std::cout << "Solution at t=" << time.t_end() << ":\n";
        for (size_t i = 0; i < n; i += 20) {
            std::cout << "  u(" << grid_buffer[i] << ") = " << solution[i] << "\n";
        }
    } else {
        std::cout << "Solver failed to converge: " << status.error().message << "\n";
        return 1;
    }

    return 0;
}
