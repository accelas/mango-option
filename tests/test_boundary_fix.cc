#include "src/option/american_pde_solver.hpp"
#include "src/option/american_solver_workspace.hpp"
#include "src/option/option_spec.hpp"
#include "src/pde/core/pde_solver.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>

namespace mango {

// Test solver with time-dependent boundary condition
class TestPutSolver : public PDESolver<TestPutSolver> {
public:
    TestPutSolver(const PricingParams& params,
                 std::shared_ptr<AmericanSolverWorkspace> workspace)
        : PDESolver<TestPutSolver>(
              workspace->grid_span(),
              TimeDomain::from_n_steps(0.0, params.maturity, workspace->n_time()),
              create_obstacle(),
              workspace->pde_workspace())
        , params_(params)
        , workspace_(std::move(workspace))
        , grid_spacing_(workspace_->grid_spacing())
        , left_bc_(create_left_bc())
        , right_bc_(create_right_bc())
        , spatial_op_(create_spatial_op())
    {}

    const auto& left_boundary() const { return left_bc_; }
    const auto& right_boundary() const { return right_bc_; }
    const auto& spatial_operator() const { return spatial_op_; }

    double x_min() const { return workspace_->x_min(); }
    double x_max() const { return workspace_->x_max(); }
    size_t n_space() const { return workspace_->n_space(); }
    size_t n_time() const { return workspace_->n_time(); }

private:
    struct LeftBCFunction {
        double r;
        double T;

        double operator()(double t, double x) const {
            double tau = T - t;  // Time to maturity
            double discount = std::exp(-r * tau);
            // Use discounted intrinsic value
            return std::max(discount * (1.0 - std::exp(x)), 0.0);
        }
    };

    struct RightBCFunction {
        double operator()(double /*t*/, double /*x*/) const {
            return 0.0;
        }
    };

    static ObstacleCallback create_obstacle() {
        return [](double /*t*/, std::span<const double> x, std::span<double> psi) {
            for (size_t i = 0; i < x.size(); ++i) {
                psi[i] = std::max(1.0 - std::exp(x[i]), 0.0);
            }
        };
    }

    DirichletBC<LeftBCFunction> create_left_bc() const {
        return DirichletBC(LeftBCFunction{params_.rate, params_.maturity});
    }

    DirichletBC<RightBCFunction> create_right_bc() const {
        return DirichletBC(RightBCFunction{});
    }

    operators::SpatialOperator<operators::BlackScholesPDE<double>, double> create_spatial_op() const {
        auto pde = operators::BlackScholesPDE<double>(
            params_.volatility,
            params_.rate,
            params_.dividend_yield);
        auto spacing_ptr = std::make_shared<GridSpacing<double>>(grid_spacing_);
        return operators::create_spatial_operator(std::move(pde), spacing_ptr);
    }

    PricingParams params_;
    std::shared_ptr<AmericanSolverWorkspace> workspace_;
    GridSpacing<double> grid_spacing_;
    DirichletBC<LeftBCFunction> left_bc_;
    DirichletBC<RightBCFunction> right_bc_;
    operators::SpatialOperator<operators::BlackScholesPDE<double>, double> spatial_op_;
};

TEST(BoundaryFixTest, TestDiscountedBoundary) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::sinh_spaced(-7.0, 2.0, 301, 2.0);
    ASSERT_TRUE(grid_spec.has_value());
    auto workspace = AmericanSolverWorkspace::create(grid_spec.value(), 1500, &pool);
    ASSERT_TRUE(workspace.has_value());

    PricingParams params(
        0.25,   // spot deep ITM
        100.0,  // strike
        0.75,   // maturity
        0.05,   // rate
        0.0,    // dividend yield
        OptionType::PUT,
        0.2     // volatility
    );

    TestPutSolver solver(params, workspace.value());
    auto solve_result = solver.solve();
    ASSERT_TRUE(solve_result.has_value());

    auto solution = solver.solution();
    double intrinsic = params.strike - params.spot;

    // Find closest grid point to current spot
    double x_current = std::log(params.spot / params.strike);
    auto grid = workspace.value()->grid();
    size_t closest_idx = 0;
    double min_dist = std::abs(grid[0] - x_current);
    for (size_t i = 1; i < grid.size(); ++i) {
        double dist = std::abs(grid[i] - x_current);
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }

    double value = solution[closest_idx] * params.strike;
    double excess = value - intrinsic;

    std::cout << "\n=== Test with Discounted Left Boundary ===" << std::endl;
    std::cout << "Intrinsic value: " << intrinsic << std::endl;
    std::cout << "Computed value: " << value << std::endl;
    std::cout << "Excess: " << excess << std::endl;
    std::cout << "(Original excess was +16.22)" << std::endl;

    // If hypothesis is correct, excess should be reduced
    EXPECT_LT(excess, 10.0) << "Discounted BC should reduce excess";
}

}  // namespace mango
