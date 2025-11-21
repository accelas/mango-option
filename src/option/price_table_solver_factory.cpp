/**
 * @file price_table_solver_factory.cpp
 * @brief Implementation of price table solver factory
 */

#include "src/option/price_table_solver_factory.hpp"
#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include "src/support/parallel.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include <ranges>
#include <stdexcept>


namespace mango {

namespace {

/// Extract prices from batch results into 4D array
///
/// Shared logic for both normalized and batch solvers.
/// Interpolates spatial solutions to moneyness grid using cubic splines.
void extract_batch_results_to_4d(
    const BatchAmericanOptionResult& batch_result,
    std::span<double> prices_4d,
    const PriceTableGrid& grid,
    double K_ref)
{
    const size_t Nm = grid.moneyness.size();
    const size_t Nt = grid.maturity.size();
    const size_t Nv = grid.volatility.size();
    const size_t Nr = grid.rate.size();
    const double T_max = grid.maturity.back();

    // Get n_time from first successful result (all share same grid)
    size_t n_time = 0;
    for (const auto& result_expected : batch_result.results) {
        if (result_expected.has_value() && result_expected->converged) {
            n_time = result_expected->grid()->num_snapshots();
            break;
        }
    }

    // Precompute step indices for each maturity
    const double dt = T_max / n_time;
    std::vector<size_t> step_indices(Nt);
    for (size_t j = 0; j < Nt; ++j) {
        double step_exact = grid.maturity[j] / dt - 1.0;
        long long step_rounded = std::llround(step_exact);

        if (step_rounded < 0) {
            step_indices[j] = 0;
        } else if (step_rounded >= static_cast<long long>(n_time)) {
            step_indices[j] = n_time - 1;
        } else {
            step_indices[j] = static_cast<size_t>(step_rounded);
        }
    }

    // Precompute log-moneyness values
    std::vector<double> log_moneyness(Nm);
    for (size_t i = 0; i < Nm; ++i) {
        log_moneyness[i] = std::log(grid.moneyness[i]);
    }

    // Extract prices from surfaces for each (σ, r) result
    const size_t slice_stride = Nv * Nr;
    for (size_t idx = 0; idx < batch_result.results.size(); ++idx) {
        const auto& result_expected = batch_result.results[idx];
        if (!result_expected.has_value() || !result_expected->converged) {
            continue;  // Leave zeros for failed solves
        }
        const auto& result = result_expected.value();

        // Extract grid from result
        auto result_grid = result.grid();
        auto x_grid = result_grid->x();  // Span of spatial grid points
        const size_t n_space = x_grid.size();
        const double x_min = x_grid.front();

        // For each maturity time step
        for (size_t j = 0; j < grid.maturity.size(); ++j) {
            size_t step_idx = step_indices[j];
            std::span<const double> spatial_solution = result.at_time(step_idx);

            if (spatial_solution.empty()) {
                continue;
            }

            // Build cubic spline for this time step
            CubicSpline<double> spline;
            auto build_error = spline.build(x_grid, spatial_solution);
            if (build_error.has_value()) {
                // Fall back to boundary values if spline build fails
                for (size_t m_idx = 0; m_idx < Nm; ++m_idx) {
                    const double x = log_moneyness[m_idx];
                    double V_norm = (x <= x_min) ? spatial_solution[0] : spatial_solution[n_space - 1];
                    size_t table_idx = (m_idx * Nt + j) * slice_stride + idx;
                    prices_4d[table_idx] = K_ref * V_norm;
                }
                continue;
            }

            // Interpolate spatial solution to moneyness grid using cubic spline
            for (size_t m_idx = 0; m_idx < Nm; ++m_idx) {
                const double x = log_moneyness[m_idx];
                double V_norm = spline.eval(x);

                // Store denormalized price
                size_t table_idx = (m_idx * Nt + j) * slice_stride + idx;
                prices_4d[table_idx] = K_ref * V_norm;
            }
        }
    }
}

} // anonymous namespace

// ============================================================================
// Normalized Chain Solver Implementation
// ============================================================================

class NormalizedPriceTableSolver : public IPriceTableSolver {
public:
    explicit NormalizedPriceTableSolver(const OptionSolverGrid& config)
        : config_(config) {}

    std::expected<void, std::string> solve(
        std::span<double> prices_4d,
        const PriceTableGrid& grid) override;

    const char* strategy_name() const override {
        return "NormalizedChainSolver";
    }

private:
    OptionSolverGrid config_;
};

std::expected<void, std::string> NormalizedPriceTableSolver::solve(
    std::span<double> prices_4d,
    const PriceTableGrid& grid)
{
    const size_t Nm = grid.moneyness.size();
    const size_t Nt = grid.maturity.size();
    const size_t Nv = grid.volatility.size();
    const size_t Nr = grid.rate.size();
    const double T_max = grid.maturity.back();

    // Precompute log-moneyness values for interpolation queries
    std::vector<double> log_moneyness(Nm);
    for (size_t i = 0; i < Nm; ++i) {
        log_moneyness[i] = std::log(grid.moneyness[i]);
    }

    // Pre-allocate workspaces and surfaces for all (σ, r) combinations
    const size_t batch_size = Nv * Nr;
    std::vector<std::optional<NormalizedWorkspace>> workspaces(batch_size);
    std::vector<std::optional<NormalizedSurfaceView>> surfaces(batch_size);

    // Create workspaces outside parallel region
    size_t workspace_failed = 0;
    namespace views = std::views;
    for (auto [k, l] : views::cartesian_product(views::iota(size_t{0}, Nv),
                                                 views::iota(size_t{0}, Nr))) {
        size_t idx = k + l * Nv;

        NormalizedSolveRequest request;
        request.sigma = grid.volatility[k];
        request.rate = grid.rate[l];
        request.dividend = config_.dividend_yield;
        request.option_type = config_.option_type;
        request.x_min = config_.x_min;
        request.x_max = config_.x_max;
        request.n_space = config_.n_space;
        request.n_time = config_.n_time;
        request.T_max = T_max;
        request.tau_snapshots = grid.maturity;

        auto ws_result = NormalizedWorkspace::create(request);
        if (ws_result.has_value()) {
            workspaces[idx] = std::move(ws_result.value());
        } else {
            ++workspace_failed;
        }
    }

    // Parallel solve for each (σ, r) combination
    size_t failed_count = workspace_failed;
    #pragma omp parallel for reduction(+:failed_count) collapse(2)
    for (size_t k = 0; k < Nv; ++k) {
        for (size_t l = 0; l < Nr; ++l) {
            size_t idx = k + l * Nv;

            // Skip if workspace creation failed
            if (!workspaces[idx].has_value()) {
                continue;
            }

            NormalizedSolveRequest request;
            request.sigma = grid.volatility[k];
            request.rate = grid.rate[l];
            request.dividend = config_.dividend_yield;
            request.option_type = config_.option_type;
            request.x_min = config_.x_min;
            request.x_max = config_.x_max;
            request.n_space = config_.n_space;
            request.n_time = config_.n_time;
            request.T_max = T_max;
            request.tau_snapshots = grid.maturity;

            NormalizedSurfaceView surface_view(
                std::span<const double>{},
                std::span<const double>{},
                std::span<const double>{});

            auto result = NormalizedChainSolver::solve(
                request, workspaces[idx].value(), surface_view);

            if (result.has_value()) {
                surfaces[idx] = surface_view;
            } else {
                failed_count++;
            }
        }
    }

    if (failed_count > 0) {
        return std::unexpected("Failed to solve " + std::to_string(failed_count) +
                         " out of " + std::to_string(batch_size) + " normalized PDEs");
    }

    // Extract prices by interpolating each surface
    const size_t slice_stride = Nv * Nr;
    for (size_t k = 0; k < Nv; ++k) {
        for (size_t l = 0; l < Nr; ++l) {
            size_t idx = k + l * Nv;
            const auto& surface = surfaces[idx];
            if (!surface.has_value()) {
                continue;  // Leave zeros for failed solves
            }

            // For each (moneyness, maturity) query
            for (size_t m_idx = 0; m_idx < Nm; ++m_idx) {
                for (size_t t_idx = 0; t_idx < Nt; ++t_idx) {
                    double x = log_moneyness[m_idx];
                    double tau = grid.maturity[t_idx];

                    // Interpolate normalized value u(x, τ)
                    double u_norm = surface->interpolate(x, tau);

                    // Denormalize: V = K_ref × u
                    size_t table_idx = (m_idx * Nt + t_idx) * slice_stride + idx;
                    prices_4d[table_idx] = grid.K_ref * u_norm;
                }
            }
        }
    }

    return {};
}

// ============================================================================
// Batch API Solver Implementation
// ============================================================================

class BatchPriceTableSolver : public IPriceTableSolver {
public:
    explicit BatchPriceTableSolver(const OptionSolverGrid& config)
        : config_(config) {}

    std::expected<void, std::string> solve(
        std::span<double> prices_4d,
        const PriceTableGrid& grid) override;

    const char* strategy_name() const override {
        return "BatchAmericanOptionSolver";
    }

private:
    OptionSolverGrid config_;
};

std::expected<void, std::string> BatchPriceTableSolver::solve(
    std::span<double> prices_4d,
    const PriceTableGrid& grid)
{
    const size_t Nv = grid.volatility.size();
    const size_t Nr = grid.rate.size();
    const double T_max = grid.maturity.back();

    // Zero out entire output array upfront (failed solves leave zeros)
    std::ranges::fill(prices_4d, 0.0);

    // Build batch parameters (all (σ,r) combinations)
    const size_t batch_size = Nv * Nr;
    std::vector<AmericanOptionParams> batch_params;
    batch_params.reserve(batch_size);

    namespace views = std::views;
    for (auto [k, l] : views::cartesian_product(views::iota(size_t{0}, Nv),
                                                 views::iota(size_t{0}, Nr))) {
        AmericanOptionParams params;
        params.spot = grid.K_ref;
        params.strike = grid.K_ref;
        params.maturity = T_max;
        params.rate = grid.rate[l];
        params.dividend_yield = config_.dividend_yield;
        params.type = config_.option_type;
        params.volatility = grid.volatility[k];
        params.discrete_dividends = {};
        batch_params.push_back(params);
    }

    // Use BatchAmericanOptionSolver with shared grid (use_shared_grid=true)
    // Grid is automatically computed, surfaces are collected
    auto batch_result = BatchAmericanOptionSolver().solve_batch(batch_params, true);

    // Check failures
    if (batch_result.failed_count > 0) {
        return std::unexpected("Failed to solve " + std::to_string(batch_result.failed_count) +
                         " out of " + std::to_string(batch_size) + " PDEs");
    }

    // Extract prices using shared logic
    extract_batch_results_to_4d(batch_result, prices_4d, grid, grid.K_ref);

    return {};
}

// ============================================================================
// Factory Implementation
// ============================================================================

std::expected<void, std::string> PriceTableSolverFactory::validate_config(
    const OptionSolverGrid& config)
{
    if (config.n_space < 4) {
        return std::unexpected("n_space must be >= 4");
    }
    if (config.n_time < 2) {
        return std::unexpected("n_time must be >= 2");
    }
    if (config.x_min >= config.x_max) {
        return std::unexpected("x_min must be < x_max");
    }
    if (config.dividend_yield < 0.0) {
        return std::unexpected("dividend_yield must be non-negative");
    }
    return {};
}

bool PriceTableSolverFactory::is_normalized_solver_eligible(
    const OptionSolverGrid& config,
    std::span<const double> moneyness)
{
    // Check normalized solver eligibility
    NormalizedSolveRequest test_request{
        .sigma = 0.20,  // Test value
        .rate = 0.05,   // Test value
        .dividend = config.dividend_yield,
        .option_type = config.option_type,
        .x_min = config.x_min,
        .x_max = config.x_max,
        .n_space = config.n_space,
        .n_time = config.n_time,
        .T_max = 1.0,  // Test value
        .tau_snapshots = std::span<const double>{}  // Will be set per-solve
    };

    auto eligibility = NormalizedChainSolver::check_eligibility(
        test_request, moneyness);

    return eligibility.has_value();
}

std::expected<std::unique_ptr<IPriceTableSolver>, std::string>
PriceTableSolverFactory::create(
    const OptionSolverGrid& config,
    std::span<const double> moneyness,
    bool force_batch)
{
    // Step 1: Validate configuration
    auto validation = validate_config(config);
    if (!validation) {
        return std::unexpected("Invalid configuration: " + validation.error());
    }

    // Step 2: Check eligibility for normalized solver (fast path)
    if (!force_batch && is_normalized_solver_eligible(config, moneyness)) {
        return std::make_unique<NormalizedPriceTableSolver>(config);
    }

    // Step 3: Fall back to batch solver
    return std::make_unique<BatchPriceTableSolver>(config);
}

} // namespace mango
