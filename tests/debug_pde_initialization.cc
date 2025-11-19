#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>

namespace mango {

// Minimal solver to test initialization
class TestSolver : public PDESolver<TestSolver> {
public:
    TestSolver(std::span<const double> grid, const TimeDomain& time)
        : PDESolver<TestSolver>(grid, time)
        , left_bc_([](double, double) { return 0.0; })
        , right_bc_([](double, double) { return 0.0; })
        , spacing_(GridSpacing<double>::uniform(0.0, 1.0, grid.size()).value())
    {
        auto pde = operators::LaplacianPDE(0.1);
        auto spacing_ptr = std::make_shared<GridSpacing<double>>(spacing_);
        spatial_op_ = operators::create_spatial_operator(std::move(pde), spacing_ptr);
    }

    const auto& left_boundary() const { return left_bc_; }
    const auto& right_boundary() const { return right_bc_; }
    const auto& spatial_operator() const { return spatial_op_; }

    double x_min() const { return 0.0; }
    double x_max() const { return 1.0; }
    size_t n_space() const { return grid_.size(); }
    size_t n_time() const { return 10; }

private:
    std::span<const double> grid_;
    DirichletBC<std::function<double(double,double)>> left_bc_;
    DirichletBC<std::function<double(double,double)>> right_bc_;
    GridSpacing<double> spacing_;
    operators::SpatialOperator<operators::LaplacianPDE<double>, double> spatial_op_;
};

TEST(PDEInitializationDebug, CheckUninitialized) {
    // Create grid
    std::vector<double> grid_storage(101);
    for (size_t i = 0; i < 101; ++i) {
        grid_storage[i] = i * 0.01;
    }
    std::span<const double> grid{grid_storage};

    TimeDomain time = TimeDomain::from_n_steps(0.0, 1.0, 10);

    TestSolver solver(grid, time);

    std::cout << "\n=== PDE Solution at t=0 (before any time steps) ===" << std::endl;
    std::cout << "First 10 values of u_current:" << std::endl;

    auto solution = solver.solution();
    for (size_t i = 0; i < std::min(size_t(10), solution.size()); ++i) {
        std::cout << "u[" << i << "] = " << solution[i] << std::endl;
    }

    std::cout << "\n=== Analysis ===" << std::endl;
    std::cout << "If all zeros → default initialization" << std::endl;
    std::cout << "If garbage → uninitialized memory" << std::endl;
    std::cout << "The solve() method is called WITHOUT initialize()!" << std::endl;
}

}  // namespace mango
