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
#include <vector>

namespace mango {

// ============================================================================
// Internal implementation details (not exposed in public API)
// ============================================================================

namespace {

/**
 * American put option obstacle in log-moneyness coordinates.
 *
 * Intrinsic value: ψ(x) = max(1 - exp(x), 0)
 * where x = ln(S/K).
 */
class AmericanPutObstacle {
public:
    void operator()(double, std::span<const double> x,
                    std::span<double> psi) const {
        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            psi[i] = std::max(1.0 - std::exp(x[i]), 0.0);
        }
    }
};

/**
 * American call option obstacle in log-moneyness coordinates.
 *
 * Intrinsic value: ψ(x) = max(exp(x) - 1, 0)
 * where x = ln(S/K).
 */
class AmericanCallObstacle {
public:
    void operator()(double, std::span<const double> x,
                    std::span<double> psi) const {
        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            psi[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
        }
    }
};

/**
 * Dividend jump event for discrete dividend payments.
 *
 * When dividend D is paid, stock price drops: S → S - D
 * causing jump in log-moneyness: x = ln(S/K) → x' = ln((S-D)/K)
 */
class DividendJump {
public:
    DividendJump(double dividend, double strike)
        : dividend_(dividend), strike_(strike) {}

    void operator()(double, std::span<const double> x,
                    std::span<double> u) const {
        const size_t n = x.size();
        std::vector<double> u_old(u.begin(), u.end());
        std::vector<double> x_new(n);

        // Compute new x positions after dividend
        for (size_t i = 0; i < n; ++i) {
            const double S = strike_ * std::exp(x[i]);
            const double S_new = S - dividend_;
            x_new[i] = (S_new <= 0.0) ? -10.0 : std::log(S_new / strike_);
        }

        // Interpolate u values to new positions
        for (size_t i = 0; i < n; ++i) {
            u[i] = interpolate(x, u_old, x_new[i]);
        }
    }

private:
    double dividend_;
    double strike_;

    /// Linear interpolation
    static double interpolate(std::span<const double> x,
                              std::span<const double> u,
                              double x_target) {
        const size_t n = x.size();
        if (x_target <= x[0]) return u[0];
        if (x_target >= x[n-1]) return u[n-1];

        auto it = std::lower_bound(x.begin(), x.end(), x_target);
        size_t j = std::distance(x.begin(), it);
        if (j == 0) j = 1;
        size_t i = j - 1;

        double dx = x[j] - x[i];
        double weight = (x_target - x[i]) / dx;
        return (1.0 - weight) * u[i] + weight * u[j];
    }
};

}  // anonymous namespace

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
    // Validate parameters (includes discrete dividend validation)
    params_.validate();
    grid_.validate();
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
        params_.continuous_dividend_yield
    );

    // 4. Setup boundary conditions (NORMALIZED by K=1)
    // For log-moneyness: x → -∞ (S → 0), x → +∞ (S → ∞)
    // IMPORTANT: Boundaries must account for time evolution via discounting
    //
    // LEFT boundary (x → -∞, S → 0):
    auto left_bc = DirichletBC([this](double t, double x) {
        const double tau = t;  // Time to maturity (backward PDE time)
        const double discount = std::exp(-params_.rate * tau);

        if (params_.option_type == OptionType::PUT) {
            // Deep ITM put: V = K·e^(-r*τ) - S ≈ K·e^(-r*τ) as S → 0
            // Normalized: V/K = e^(-r*τ) - e^(x - r*τ) ≈ e^(-r*τ) as x → -∞
            return discount - std::exp(x) * discount;
        } else {
            // Deep OTM call: V → 0 as S → 0
            return 0.0;
        }
    });

    // RIGHT boundary (x → +∞, S → ∞):
    auto right_bc = DirichletBC([this](double t, double x) {
        const double tau = t;  // Time to maturity (backward PDE time)
        const double discount = std::exp(-params_.rate * tau);

        if (params_.option_type == OptionType::CALL) {
            // Deep ITM call: V = S - K·e^(-r*τ)
            // Normalized: V/K = (S/K) - e^(-r*τ) = e^x - e^(-r*τ)
            return std::exp(x) - discount;
        } else {
            // Deep OTM put: V → 0 as S → ∞
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

        // 6. Register discrete dividends as temporal events
        // Convert from calendar time (years from now) to solver time (backward time)
        for (const auto& [calendar_time, amount] : params_.discrete_dividends) {
            // Solver time: t=0 at maturity, t=T at present
            // Calendar time: time=0 now, time=T at maturity
            double solver_time = params_.maturity - calendar_time;

            // Skip dividends at or beyond maturity (solver_time <= 0)
            if (solver_time <= 1e-10) continue;

            DividendJump div_jump(amount, params_.strike);
            solver.add_temporal_event(solver_time,
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
        // Note: left_bc and right_bc already defined above with time-dependent discounting
        PDESolver solver(x_grid, time_domain, trbdf2_config_, root_config_,
                        left_bc, right_bc, bs_op,
                        [](double t, auto x, auto psi) {
                            AmericanCallObstacle obstacle;
                            obstacle(t, x, psi);
                        });

        // 6. Register discrete dividends as temporal events
        // Convert from calendar time (years from now) to solver time (backward time)
        for (const auto& [calendar_time, amount] : params_.discrete_dividends) {
            // Solver time: t=0 at maturity, t=T at present
            // Calendar time: time=0 now, time=T at maturity
            double solver_time = params_.maturity - calendar_time;

            // Skip dividends at or beyond maturity (solver_time <= 0)
            if (solver_time <= 1e-10) continue;

            DividendJump div_jump(amount, params_.strike);
            solver.add_temporal_event(solver_time,
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
    if (!solved_) {
        return 0.0;  // No solution available
    }

    const size_t n = solution_.size();
    const double dx = (grid_.x_max - grid_.x_min) / (n - 1);

    // Find current spot in grid
    double current_moneyness = std::log(params_.spot / params_.strike);

    // Find the grid point closest to current_moneyness
    // Use the same approach as interpolate_solution
    size_t i = 0;
    while (i < n-1 && grid_.x_min + (i+1)*dx < current_moneyness) {
        i++;
    }

    // Ensure we're in valid interior range for centered differences
    if (i == 0) i = 1;
    if (i >= n-1) i = n-2;

    // Compute ∂V/∂x using centered finite difference
    // Note: solution_ stores V/K (normalized)
    double dVdx = (solution_[i+1] - solution_[i-1]) / (2.0 * dx);

    // Transform from log-moneyness to spot
    // V_dollar = V_norm * K
    // Delta = ∂V_dollar/∂S = K * ∂V_norm/∂x * ∂x/∂S
    //       = K * dVdx * (1/S)
    //       = (K/S) * dVdx
    double delta = (params_.strike / params_.spot) * dVdx;

    return delta;
}

double AmericanOptionSolver::compute_gamma() const {
    if (!solved_) {
        return 0.0;  // No solution available
    }

    const size_t n = solution_.size();
    const double dx = (grid_.x_max - grid_.x_min) / (n - 1);

    // Find current spot in grid
    double current_moneyness = std::log(params_.spot / params_.strike);

    // Find the grid point closest to current_moneyness
    size_t i = 0;
    while (i < n-1 && grid_.x_min + (i+1)*dx < current_moneyness) {
        i++;
    }

    // Ensure we're in valid interior range for centered differences
    if (i == 0) i = 1;
    if (i >= n-1) i = n-2;

    // Centered second derivative: [V(i+1) - 2*V(i) + V(i-1)] / dx²
    double d2Vdx2 = (solution_[i+1] - 2.0*solution_[i] + solution_[i-1]) / (dx * dx);
    // Centered first derivative: [V(i+1) - V(i-1)] / (2*dx)
    double dVdx = (solution_[i+1] - solution_[i-1]) / (2.0 * dx);

    // Transform from log-moneyness to spot using chain rule
    // x = ln(S/K), so ∂x/∂S = 1/S and ∂²x/∂S² = -1/S²
    //
    // V_dollar(S) = K * V_norm(x(S))
    //
    // First derivative:
    // dV/dS = K * dV_norm/dx * dx/dS = K * dV_norm/dx * (1/S)
    //
    // Second derivative:
    // d²V/dS² = d/dS[K * dV_norm/dx * (1/S)]
    //         = K * d/dS[dV_norm/dx * (1/S)]
    //         = K * [d²V_norm/dx² * (dx/dS) * (1/S) + dV_norm/dx * d/dS(1/S)]
    //         = K * [d²V_norm/dx² * (1/S²) + dV_norm/dx * (-1/S²)]
    //         = (K/S²) * [d²V_norm/dx² - dV_norm/dx]
    //
    double S = params_.spot;
    double K = params_.strike;
    double gamma = (K / (S * S)) * (d2Vdx2 - dVdx);

    return gamma;
}

double AmericanOptionSolver::compute_theta() const {
    // Theta is time decay: ∂V/∂t
    // For American options with no closed form, accurate theta requires:
    // 1. Re-solving at slightly different time, or
    // 2. Evaluating the PDE operator: ∂V/∂t = L(V)
    //
    // Both approaches are expensive and complex. For now, return 0.0 as stub.
    // Future enhancement could evaluate the BS operator on the solution surface.

    return 0.0;  // Stub implementation
}

}  // namespace mango
