/**
 * @file price_table_solver_factory.cpp
 * @brief Implementation of price table solver factory
 */

#include "src/option/price_table_solver_factory.hpp"
#include "src/option/normalized_chain_solver.hpp"
#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include "src/option/american_solver_workspace.hpp"
#include "src/support/parallel.hpp"
#include <ranges>
#include <stdexcept>


namespace mango {



// ============================================================================
// Normalized Chain Solver Implementation
// ============================================================================

class NormalizedPriceTableSolver : public IPriceTableSolver {
public:
    explicit NormalizedPriceTableSolver(const OptionSolverGrid& config)
        : config_(config) {}

    std::expected<void, std::string> solve(
        std::span<double> prices_4d,
        std::span<const double> moneyness,
        std::span<const double> maturity,
        std::span<const double> volatility,
        std::span<const double> rate,
        double K_ref) override;

    const char* strategy_name() const override {
        return "NormalizedChainSolver";
    }

private:
    OptionSolverGrid config_;
};

std::expected<void, std::string> NormalizedPriceTableSolver::solve(
    std::span<double> prices_4d,
    std::span<const double> moneyness,
    std::span<const double> maturity,
    std::span<const double> volatility,
    std::span<const double> rate,
    double K_ref)
{
    const size_t Nm = moneyness.size();
    const size_t Nt = maturity.size();
    const size_t Nv = volatility.size();
    const size_t Nr = rate.size();
    const double T_max = maturity.back();

    size_t failed_count = 0;

    MANGO_PRAGMA_PARALLEL
    {
        // Create normalized request template (per-thread)
        NormalizedSolveRequest base_request{
            .sigma = 0.20,  // Placeholder, set in loop
            .rate = 0.05,   // Placeholder, set in loop
            .dividend = config_.dividend_yield,
            .option_type = config_.option_type,
            .x_min = config_.x_min,
            .x_max = config_.x_max,
            .n_space = config_.n_space,
            .n_time = config_.n_time,
            .T_max = T_max,
            .tau_snapshots = maturity
        };

        // Create workspace once per thread
        auto workspace_result = NormalizedWorkspace::create(base_request);

        if (!workspace_result) {
            // Workspace creation failed
            MANGO_PRAGMA_FOR_COLLAPSE2
            for (size_t k = 0; k < Nv; ++k) {
                for (size_t l = 0; l < Nr; ++l) {
                    MANGO_PRAGMA_ATOMIC
                    ++failed_count;
                }
            }
        } else {
            auto workspace = std::move(workspace_result.value());
            auto surface = workspace.surface_view();

            MANGO_PRAGMA_FOR_COLLAPSE2_DYNAMIC
            for (size_t k = 0; k < Nv; ++k) {
                for (size_t l = 0; l < Nr; ++l) {
                    // Set (σ, r) for this solve
                    NormalizedSolveRequest request = base_request;
                    request.sigma = volatility[k];
                    request.rate = rate[l];

                    // Solve normalized PDE
                    auto solve_result = NormalizedChainSolver::solve(
                        request, workspace, surface);

                    if (!solve_result) {
                        MANGO_PRAGMA_ATOMIC
                        ++failed_count;
                        continue;
                    }

                    // Extract prices from surface
                    namespace views = std::views;
                    for (auto [i, j] : views::cartesian_product(views::iota(size_t{0}, Nm),
                                                                 views::iota(size_t{0}, Nt))) {
                        double x = std::log(moneyness[i]);
                        double u = surface.interpolate(x, maturity[j]);
                        size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;
                        prices_4d[idx_4d] = K_ref * u;
                    }
                }
            }
        }
    }

    if (failed_count > 0) {
        return std::unexpected("Failed to solve " + std::to_string(failed_count) +
                         " out of " + std::to_string(Nv * Nr) + " PDEs");
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
        std::span<const double> moneyness,
        std::span<const double> maturity,
        std::span<const double> volatility,
        std::span<const double> rate,
        double K_ref) override;

    const char* strategy_name() const override {
        return "BatchAmericanOptionSolver";
    }

private:
    OptionSolverGrid config_;
};

std::expected<void, std::string> BatchPriceTableSolver::solve(
    std::span<double> prices_4d,
    std::span<const double> moneyness,
    std::span<const double> maturity,
    std::span<const double> volatility,
    std::span<const double> rate,
    double K_ref)
{
    const size_t Nm = moneyness.size();
    const size_t Nt = maturity.size();
    const size_t Nv = volatility.size();
    const size_t Nr = rate.size();
    const double T_max = maturity.back();
    const double dt = T_max / config_.n_time;

    // Zero out entire output array upfront (failed solves leave zeros)
    std::ranges::fill(prices_4d, 0.0);

    // Precompute step indices for each maturity
    std::vector<size_t> step_indices(Nt);
    for (size_t j = 0; j < Nt; ++j) {
        double step_exact = maturity[j] / dt - 1.0;
        long long step_rounded = std::llround(step_exact);

        if (step_rounded < 0) {
            step_indices[j] = 0;
        } else if (step_rounded >= static_cast<long long>(config_.n_time)) {
            step_indices[j] = config_.n_time - 1;
        } else {
            step_indices[j] = static_cast<size_t>(step_rounded);
        }
    }

    // Precompute log-moneyness values
    std::vector<double> log_moneyness(Nm);
    for (size_t i = 0; i < Nm; ++i) {
        log_moneyness[i] = std::log(moneyness[i]);
    }

    // Build batch parameters (all (σ,r) combinations)
    std::vector<AmericanOptionParams> batch_params;
    batch_params.reserve(Nv * Nr);

    namespace views = std::views;
    for (auto [k, l] : views::cartesian_product(views::iota(size_t{0}, Nv),
                                                 views::iota(size_t{0}, Nr))) {
        AmericanOptionParams params;
        params.spot = K_ref;
        params.strike = K_ref;
        params.maturity = T_max;
        params.rate = rate[l];
        params.dividend_yield = config_.dividend_yield;
        params.type = config_.option_type;
        params.volatility = volatility[k];
        params.discrete_dividends = {};
        batch_params.push_back(params);
    }

    // Solve batch (surface always collected)
    auto batch_result = BatchAmericanOptionSolver::solve_batch_with_grid(
        std::span{batch_params}, config_.x_min, config_.x_max, config_.n_space, config_.n_time);

    // Check failure count
    if (batch_result.failed_count > 0) {
        return std::unexpected("Failed to solve " + std::to_string(batch_result.failed_count) +
                         " out of " + std::to_string(Nv * Nr) + " PDEs");
    }

    // Extract prices from surface_2d for each result
    const size_t slice_stride = Nv * Nr;
    for (size_t idx = 0; idx < batch_result.results.size(); ++idx) {
        const auto& result_expected = batch_result.results[idx];
        if (!result_expected.has_value()) {
            continue;  // Leave zeros for failed solves
        }

        const auto& result = result_expected.value();
        if (!result.converged) {
            continue;  // Leave zeros for failed solves
        }

        // For each maturity time step
        for (size_t j = 0; j < Nt; ++j) {
            size_t step_idx = step_indices[j];
            std::span<const double> spatial_solution = result.at_time(step_idx);

            if (spatial_solution.empty()) {
                continue;
            }

            // Interpolate spatial solution to moneyness grid
            const double dx = (result.x_max - result.x_min) / (result.n_space - 1);
            for (size_t m_idx = 0; m_idx < Nm; ++m_idx) {
                const double x = log_moneyness[m_idx];

                // Linear interpolation
                double V_norm = 0.0;
                if (x <= result.x_min) {
                    V_norm = spatial_solution[0];
                } else if (x >= result.x_max) {
                    V_norm = spatial_solution[result.n_space - 1];
                } else {
                    // Find bracketing indices
                    size_t i = 0;
                    while (i < result.n_space - 1 && result.x_min + (i + 1) * dx < x) {
                        i++;
                    }

                    // Interpolate
                    double x_i = result.x_min + i * dx;
                    double x_i1 = result.x_min + (i + 1) * dx;
                    double t = (x - x_i) / (x_i1 - x_i);
                    V_norm = (1.0 - t) * spatial_solution[i] + t * spatial_solution[i + 1];
                }

                // Store denormalized price
                size_t table_idx = (m_idx * Nt + j) * slice_stride + idx;
                prices_4d[table_idx] = K_ref * V_norm;
            }
        }
    }

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
    std::span<const double> moneyness)
{
    // Step 1: Validate configuration
    auto validation = validate_config(config);
    if (!validation) {
        return std::unexpected("Invalid configuration: " + validation.error());
    }

    // Step 2: Check eligibility for normalized solver (fast path)
    if (is_normalized_solver_eligible(config, moneyness)) {
        return std::make_unique<NormalizedPriceTableSolver>(config);
    }

    // Step 3: Fall back to batch solver
    return std::make_unique<BatchPriceTableSolver>(config);
}

} // namespace mango
