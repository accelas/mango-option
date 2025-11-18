/**
 * @file american_option.cpp
 * @brief American option pricing solver implementation
 */

#include "src/option/american_option.hpp"
#include "src/option/american_solver_workspace.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/pde/core/pde_solver.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/support/parallel.hpp"
// BlackScholesPDE now defined in american_option.hpp
#include <algorithm>
#include <span>
#include <cmath>
#include <vector>
#include <optional>
#include <variant>

namespace mango {

// ============================================================================
// Internal implementation details (not exposed in public API)
// ============================================================================

namespace {

/**
 * Internal collector for gathering all time steps during solve.
 */
class AllTimeStepsCollector : public SnapshotCollector {
public:
    explicit AllTimeStepsCollector(size_t n_space, size_t n_time)
        : n_space_(n_space), n_time_(n_time) {
        data_.resize(n_space * n_time);
    }

    void collect(const Snapshot& snapshot) override {
        size_t time_idx = snapshot.user_index;
        if (time_idx >= n_time_) return;

        const auto& solution = snapshot.solution;
        std::copy(solution.begin(), solution.end(),
                  data_.begin() + time_idx * n_space_);
    }

    std::vector<double> extract() && {
        return std::move(data_);
    }

private:
    size_t n_space_;
    size_t n_time_;
    std::vector<double> data_;
};

/**
 * Strategy pattern for option-specific behavior (obstacle + initial condition).
 * Using std::variant eliminates ~130 lines of code duplication in solve().
 */

/**
 * American put option strategy in log-moneyness coordinates.
 * Intrinsic value: ψ(x) = max(1 - exp(x), 0) where x = ln(S/K).
 */
struct PutStrategy {
    void obstacle(double, std::span<const double> x, std::span<double> psi) const {
        MANGO_PRAGMA_SIMD
        for (size_t i = 0; i < x.size(); ++i) {
            psi[i] = std::max(1.0 - std::exp(x[i]), 0.0);
        }
    }

    void initial_condition(std::span<const double> x, std::span<double> u) const {
        MANGO_PRAGMA_SIMD
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(1.0 - std::exp(x[i]), 0.0);
        }
    }

    double left_boundary(double, double x) const {
        // Deep ITM put: exercise immediately ⇒ V/K = 1 - e^x
        return std::max(1.0 - std::exp(x), 0.0);
    }

    double right_boundary(double, double, double) const {
        // Deep OTM put: V → 0 as S → ∞
        return 0.0;
    }
};

/**
 * American call option strategy in log-moneyness coordinates.
 * Intrinsic value: ψ(x) = max(exp(x) - 1, 0) where x = ln(S/K).
 */
struct CallStrategy {
    void obstacle(double, std::span<const double> x, std::span<double> psi) const {
        MANGO_PRAGMA_SIMD
        for (size_t i = 0; i < x.size(); ++i) {
            psi[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
        }
    }

    void initial_condition(std::span<const double> x, std::span<double> u) const {
        MANGO_PRAGMA_SIMD
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
        }
    }

    double left_boundary(double, double) const {
        // Deep OTM call: V → 0 as S → 0
        return 0.0;
    }

    double right_boundary(double t, double x, double rate) const {
        // Deep ITM call: V = S - K·e^(-r*τ)
        // Normalized: V/K = e^x - e^(-r*τ)
        const double discount = std::exp(-rate * t);
        return std::exp(x) - discount;
    }
};

using OptionStrategy = std::variant<CallStrategy, PutStrategy>;

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
    std::shared_ptr<AmericanSolverWorkspace> workspace)
    : params_(params)
    , workspace_(std::move(workspace))
{
    // Validate parameters using unified validation
    auto validation = validate_pricing_params(params_);
    if (!validation) {
        throw std::invalid_argument(validation.error());
    }

    // Validate workspace is not null
    if (!workspace_) {
        throw std::invalid_argument("Workspace cannot be null");
    }
}

// ============================================================================
// Factory methods with expected-based validation
// ============================================================================

std::expected<AmericanOptionSolver, std::string> AmericanOptionSolver::create(
    const AmericanOptionParams& params,
    std::shared_ptr<AmericanSolverWorkspace> workspace) {

    // Validate workspace first
    if (!workspace) {
        return std::unexpected("Workspace cannot be null");
    }

    // Chain validation and construction using monadic operations
    return validate_pricing_params(params)
        .and_then([&]() -> std::expected<AmericanOptionSolver, std::string> {
            try {
                return AmericanOptionSolver(params, workspace);
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Failed to create solver: ") + e.what());
            }
        });
}

// ============================================================================
// Public API
// ============================================================================

std::expected<AmericanOptionResult, SolverError> AmericanOptionSolver::solve() {
    // 1. Create strategy based on option type
    OptionStrategy strategy;
    switch (params_.type) {
        case OptionType::CALL:
            strategy = CallStrategy{};
            break;
        case OptionType::PUT:
            strategy = PutStrategy{};
            break;
        default:
            return std::unexpected(SolverError{
                .code = SolverErrorCode::InvalidConfiguration,
                .message = "Unknown option type",
                .iterations = 0
            });
    }

    // 2. Acquire grid from workspace
    std::span<const double> x_grid = workspace_->grid_span();
    std::shared_ptr<GridSpacing<double>> shared_spacing = workspace_->grid_spacing();
    PDEWorkspace* external_workspace = workspace_.get();

    // 3. Setup time domain
    // For option pricing: solve forward in PDE time (backward in calendar time)
    // t=0: terminal payoff at maturity, t=T: present value
    TimeDomain time_domain(0.0, params_.maturity, params_.maturity / workspace_->n_time());

    // 4. Create Black-Scholes operator in log-moneyness coordinates
    auto make_operator = [&]() {
        auto pde = BlackScholesPDE<double>(
            params_.volatility,
            params_.rate,
            params_.dividend_yield);
        if (shared_spacing) {
            return operators::create_spatial_operator(std::move(pde), shared_spacing);
        }
        auto grid_view = GridView<double>(x_grid);
        return operators::create_spatial_operator(std::move(pde), grid_view);
    };
    auto bs_op = make_operator();

    // 5. Get grid dimensions
    size_t n_space = workspace_->n_space();
    size_t n_time = workspace_->n_time();

    // 6. Create collector for all time steps
    AllTimeStepsCollector all_time_collector(n_space, n_time);

    // 7. Single unified solver path using std::visit
    return std::visit(
        [&](const auto& strat) -> std::expected<AmericanOptionResult, SolverError> {
            // Setup boundary conditions using strategy (NORMALIZED by K=1)
            // For log-moneyness: x → -∞ (S → 0), x → +∞ (S → ∞)
            // IMPORTANT: Boundaries must account for time evolution via discounting
            auto left_bc = DirichletBC([&strat](double t, double x) {
                return strat.left_boundary(t, x);
            });


            auto right_bc = DirichletBC([&strat, this](double t, double x) {
                return strat.right_boundary(t, x, params_.rate);
            });

            // Create PDESolver with strategy-specific obstacle
            PDESolver solver(
                x_grid, time_domain, TRBDF2Config{},
                left_bc, right_bc, bs_op,
                [&strat](double t, auto x, auto psi) { strat.obstacle(t, x, psi); },
                external_workspace
            );

            // Register discrete dividends as temporal events
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

            // Register collector for all time steps
            for (size_t step_idx = 0; step_idx < n_time; ++step_idx) {
                solver.register_snapshot(step_idx, step_idx, &all_time_collector);
            }

            // Initialize with strategy-specific terminal condition (payoff at maturity)
            // In log-moneyness, initial condition is normalized by K=1
            solver.initialize([&strat](auto x, auto u) {
                strat.initial_condition(x, u);
            });

            // Solve the PDE
            auto solve_result = solver.solve();
            if (!solve_result) {
                return std::unexpected(solve_result.error());
            }

            // Extract solution
            AmericanOptionResult result;
            result.converged = true;

            auto solution_view = solver.solution();
            solution_.assign(solution_view.begin(), solution_view.end());
            solved_ = true;

            // Store solution surface and grid information
            result.surface_2d = std::move(all_time_collector).extract();
            result.n_space = n_space;
            result.n_time = n_time;
            result.x_min = workspace_->x_min();
            result.x_max = workspace_->x_max();
            result.strike = params_.strike;

            return result;
        },
        strategy
    );
}

std::expected<AmericanOptionGreeks, SolverError> AmericanOptionSolver::compute_greeks() const {
    if (!solved_) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidState,
            .message = "Cannot compute Greeks: solve() has not been called or did not converge",
            .iterations = 0
        });
    }

    AmericanOptionGreeks greeks;
    greeks.delta = compute_delta();
    greeks.gamma = compute_gamma();
    greeks.theta = compute_theta();

    return greeks;
}

double AmericanOptionResult::value_at(double spot) const {
    // Convert spot to log-moneyness
    double x_target = std::log(spot / strike);

    // Get final time step (present value)
    // PDE time: t=0 at maturity, t=n_time-1 at present
    if (surface_2d.empty() || n_space == 0 || n_time == 0) {
        return 0.0;
    }

    // Extract final time step surface
    std::span<const double> final_surface = at_time(n_time - 1);

    // Compute grid spacing (uniform grid)
    const double dx = (x_max - x_min) / (n_space - 1);

    // Boundary cases
    if (x_target <= x_min) {
        return final_surface[0] * strike;  // Denormalize
    }
    if (x_target >= x_max) {
        return final_surface[n_space-1] * strike;  // Denormalize
    }

    // Find bracketing indices
    size_t i = 0;
    while (i < n_space-1 && x_min + (i+1)*dx < x_target) {
        i++;
    }

    // Linear interpolation
    double x_i = x_min + i * dx;
    double x_i1 = x_min + (i+1) * dx;
    double t = (x_target - x_i) / (x_i1 - x_i);
    double normalized_value = (1.0 - t) * final_surface[i] + t * final_surface[i+1];

    return normalized_value * strike;  // Denormalize
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
    const double x_min = workspace_->x_min();
    const double x_max = workspace_->x_max();
    const double dx = (x_max - x_min) / (n - 1);

    // Find current spot in grid
    double current_moneyness = std::log(params_.spot / params_.strike);

    // Find the grid point closest to current_moneyness
    // Use the same approach as interpolate_solution
    size_t i = 0;
    while (i < n-1 && x_min + (i+1)*dx < current_moneyness) {
        i++;
    }

    // Ensure we're in valid interior range for centered differences
    if (i == 0) i = 1;
    if (i >= n-1) i = n-2;

    // Compute ∂V/∂x using centered finite difference
    // Note: solution_ stores V/K (normalized)
    const double half_dx_inv = 1.0 / (2.0 * dx);
    double dVdx = (solution_[i+1] - solution_[i-1]) * half_dx_inv;

    // Transform from log-moneyness to spot
    // V_dollar = V_norm * K
    // Delta = ∂V_dollar/∂S = K * ∂V_norm/∂x * ∂x/∂S
    //       = K * dVdx * (1/S)
    //       = (K/S) * dVdx
    // Use FMA: (K/S) * dVdx
    const double K_over_S = params_.strike / params_.spot;
    double delta = K_over_S * dVdx;

    return delta;
}

double AmericanOptionSolver::compute_gamma() const {
    if (!solved_) {
        return 0.0;  // No solution available
    }

    const size_t n = solution_.size();
    const double x_min = workspace_->x_min();
    const double x_max = workspace_->x_max();
    const double dx = (x_max - x_min) / (n - 1);

    // Find current spot in grid
    double current_moneyness = std::log(params_.spot / params_.strike);

    // Find the grid point closest to current_moneyness
    size_t i = 0;
    while (i < n-1 && x_min + (i+1)*dx < current_moneyness) {
        i++;
    }

    // Ensure we're in valid interior range for centered differences
    if (i == 0) i = 1;
    if (i >= n-1) i = n-2;

    // Centered second derivative: [V(i+1) - 2*V(i) + V(i-1)] / dx²
    // Use FMA: (V(i+1) + V(i-1)) / dx² - 2*V(i) / dx²
    const double dx2_inv = 1.0 / (dx * dx);
    double d2Vdx2 = std::fma(solution_[i+1] + solution_[i-1], dx2_inv, -2.0*solution_[i]*dx2_inv);
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
    // Use FMA: (K/S²) * (d2Vdx2 - dVdx) = (K/S²)*d2Vdx2 - (K/S²)*dVdx
    const double K_over_S2 = K / (S * S);
    double gamma = std::fma(K_over_S2, d2Vdx2, -K_over_S2 * dVdx);

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
