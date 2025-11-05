/**
 * @file american_option.cpp
 * @brief American option pricing solver implementation
 */

#include "src/cpp/american_option.hpp"
#include "src/cpp/spatial_operators.hpp"
#include "src/cpp/boundary_conditions.hpp"
#include "src/cpp/grid.hpp"
#include "src/cpp/time_domain.hpp"
#include "src/cpp/pde_solver.hpp"
#include <algorithm>
#include <span>
#include <cmath>

namespace mango {

AmericanOptionSolver::AmericanOptionSolver(
    const AmericanOptionParams& params,
    const AmericanOptionGrid& grid,
    const TRBDF2Config& trbdf2_config,
    const RootFindingConfig& root_config)
    : params_(params)
    , grid_(grid)
    , trbdf2_config_(trbdf2_config)
    , root_config_(root_config)
{
    // Validate parameters
    params_.validate();
    grid_.validate();
}

void AmericanOptionSolver::register_dividend(double time, double amount) {
    if (time < 0.0 || time > params_.maturity) {
        throw std::invalid_argument("Dividend time must be in [0, maturity]");
    }
    if (amount < 0.0) {
        throw std::invalid_argument("Dividend amount must be non-negative");
    }
    dividends_.push_back({time, amount});

    // Sort by time (early dividends first)
    std::sort(dividends_.begin(), dividends_.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
}

AmericanOptionResult AmericanOptionSolver::solve() {
    // 1. Generate grid in log-moneyness coordinates
    auto grid_buffer = GridSpec<>::uniform(grid_.x_min, grid_.x_max, grid_.n_space).generate();
    auto x_grid = grid_buffer.span();

    // 2. Setup time domain
    // For option pricing: solve forward in PDE time (backward in calendar time)
    // t=0: terminal payoff at maturity
    // t=T: present value
    TimeDomain time_domain(0.0, params_.maturity, params_.maturity / grid_.n_time);

    // 3. Create Black-Scholes operator in log-moneyness coordinates
    LogMoneynessBlackScholesOperator bs_op(
        params_.volatility,
        params_.rate,
        params_.dividend_yield
    );

    // 4. Setup boundary conditions (NORMALIZED by K=1)
    // For log-moneyness: x → -∞ (S → 0), x → +∞ (S → ∞)
    // LEFT boundary (x → -∞, S → 0):
    auto left_bc = DirichletBC([this](double, double x) {
        if (params_.option_type == OptionType::PUT) {
            // Deep ITM put: V/K ≈ 1 - exp(x) → 1 as x → -∞
            return 1.0 - std::exp(x);
        } else {
            // Deep OTM call: V/K → 0 as S → 0
            return 0.0;
        }
    });

    // RIGHT boundary (x → +∞, S → ∞):
    auto right_bc = DirichletBC([this](double, double x) {
        if (params_.option_type == OptionType::CALL) {
            // Deep ITM call: V/K ≈ exp(x) - 1
            return std::exp(x) - 1.0;
        } else {
            // Deep OTM put: V/K → 0 as S → ∞
            return 0.0;
        }
    });

    // 5. Setup obstacle condition
    AmericanOptionResult result;

    if (params_.option_type == OptionType::PUT) {
        // Create PDESolver with obstacle
        PDESolver solver(x_grid, time_domain, trbdf2_config_, root_config_,
                        left_bc, right_bc, bs_op,
                        [](double t, auto x, auto psi) {
                            AmericanPutObstacle obstacle;
                            obstacle(t, x, psi);
                        });

        // 6. Register dividends as temporal events
        for (const auto& [time, amount] : dividends_) {
            DividendJump div_jump(amount, params_.strike);
            solver.add_temporal_event(time,
                [div_jump](double t, auto x, auto u) {
                    div_jump(t, x, u);
                });
        }

        // 7. Initialize with terminal condition (payoff at maturity)
        // In log-moneyness, obstacle is normalized: ψ = max(1 - exp(x), 0)
        // Initial condition must match (already normalized by K=1)
        solver.initialize([](auto x, auto u) {
            for (size_t i = 0; i < x.size(); ++i) {
                // Normalized payoff: max(1 - exp(x), 0) where x = ln(S/K)
                u[i] = std::max(1.0 - std::exp(x[i]), 0.0);
            }
        });

        // 8. Solve the PDE
        result.converged = solver.solve();

        // 9. Extract solution
        auto solution_view = solver.solution();
        solution_ = std::vector<double>(solution_view.begin(), solution_view.end());
        solved_ = true;

        // 10. Interpolate to current spot price and denormalize
        double current_moneyness = std::log(params_.spot / params_.strike);
        double normalized_value = interpolate_solution(current_moneyness, x_grid);

        result.value = normalized_value * params_.strike;  // Denormalize

    } else {  // CALL
        // Create PDESolver with obstacle
        PDESolver solver(x_grid, time_domain, trbdf2_config_, root_config_,
                        left_bc, right_bc, bs_op,
                        [](double t, auto x, auto psi) {
                            AmericanCallObstacle obstacle;
                            obstacle(t, x, psi);
                        });

        // 6. Register dividends as temporal events
        for (const auto& [time, amount] : dividends_) {
            DividendJump div_jump(amount, params_.strike);
            solver.add_temporal_event(time,
                [div_jump](double t, auto x, auto u) {
                    div_jump(t, x, u);
                });
        }

        // 7. Initialize with terminal condition (payoff at maturity)
        // In log-moneyness, obstacle is normalized: ψ = max(exp(x) - 1, 0)
        solver.initialize([](auto x, auto u) {
            for (size_t i = 0; i < x.size(); ++i) {
                // Normalized payoff: max(exp(x) - 1, 0) where x = ln(S/K)
                u[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
            }
        });

        // 8. Solve the PDE
        result.converged = solver.solve();

        // 9. Extract solution
        auto solution_view = solver.solution();
        solution_ = std::vector<double>(solution_view.begin(), solution_view.end());
        solved_ = true;

        // 10. Interpolate to current spot price and denormalize
        double current_moneyness = std::log(params_.spot / params_.strike);
        double normalized_value = interpolate_solution(current_moneyness, x_grid);
        result.value = normalized_value * params_.strike;  // Denormalize
    }

    // 11. Compute Greeks (stub implementation for now - Task 9)
    result.delta = compute_delta();
    result.gamma = compute_gamma();
    result.theta = compute_theta();

    return result;
}

double AmericanOptionSolver::interpolate_solution(double x_target,
                                                   std::span<const double> x_grid) const {
    const size_t n = solution_.size();

    // Boundary cases
    if (x_target <= x_grid[0]) return solution_[0];
    if (x_target >= x_grid[n-1]) return solution_[n-1];

    // Find bracketing indices
    size_t i = 0;
    while (i < n-1 && x_grid[i+1] < x_target) {
        i++;
    }

    // Linear interpolation
    double t = (x_target - x_grid[i]) / (x_grid[i+1] - x_grid[i]);
    return (1.0 - t) * solution_[i] + t * solution_[i+1];
}

std::vector<double> AmericanOptionSolver::get_solution() const {
    if (!solved_) {
        throw std::runtime_error("Solver has not been run yet");
    }
    return solution_;
}

double AmericanOptionSolver::compute_delta() const {
    // TODO: Implement in Task 9
    return 0.0;
}

double AmericanOptionSolver::compute_gamma() const {
    // TODO: Implement in Task 9
    return 0.0;
}

double AmericanOptionSolver::compute_theta() const {
    // TODO: Implement in Task 9
    return 0.0;
}

}  // namespace mango
