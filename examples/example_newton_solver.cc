#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/math/root_finding.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// Helper class to use PDESolver with CRTP pattern
template<typename LeftBC, typename RightBC, typename SpatialOp>
class ExamplePDESolver : public mango::PDESolver<ExamplePDESolver<LeftBC, RightBC, SpatialOp>> {
public:
    ExamplePDESolver(std::span<const double> grid,
                     const mango::TimeDomain& time,
                     const mango::TRBDF2Config& config,
                     LeftBC left_bc,
                     RightBC right_bc,
                     SpatialOp spatial_op)
        : mango::PDESolver<ExamplePDESolver>(
              grid, time, config, std::nullopt, nullptr, {})
        , left_bc_(std::move(left_bc))
        , right_bc_(std::move(right_bc))
        , spatial_op_(std::move(spatial_op))
    {}

    // CRTP interface - called by PDESolver base class
    const LeftBC& left_boundary() const { return left_bc_; }
    const RightBC& right_boundary() const { return right_bc_; }
    const SpatialOp& spatial_operator() const { return spatial_op_; }

private:
    LeftBC left_bc_;
    RightBC right_bc_;
    SpatialOp spatial_op_;
};

// Helper function to create solver with deduced types
template<typename LeftBC, typename RightBC, typename SpatialOp>
auto make_solver(std::span<const double> grid,
                 const mango::TimeDomain& time,
                 const mango::TRBDF2Config& config,
                 LeftBC left_bc,
                 RightBC right_bc,
                 SpatialOp spatial_op) {
    return ExamplePDESolver<LeftBC, RightBC, SpatialOp>(
        grid, time, config, std::move(left_bc), std::move(right_bc), std::move(spatial_op));
}

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
    auto pde = mango::operators::LaplacianPDE<double>(1.0);
    auto grid_view = mango::GridView<double>(grid_buffer.span());
    auto spatial_op = mango::operators::create_spatial_operator(std::move(pde), grid_view);  // Diffusion coefficient D = 1.0

    // Create solver with Newton integration using CRTP helper
    auto solver = make_solver(grid_buffer.span(), time, trbdf2_config,
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
