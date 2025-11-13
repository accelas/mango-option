/**
 * @file price_table_4d_builder.cpp
 * @brief Implementation of 4D price table builder
 */

#include "src/option/price_table_4d_builder.hpp"
#include "src/option/price_table_snapshot_collector.hpp"
#include "src/option/american_option.hpp"
#include "src/option/american_solver_workspace.hpp"
#include "src/option/normalized_chain_solver.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mango {

expected<void, std::string> PriceTable4DBuilder::validate_grids() const {
    if (moneyness_.size() < 4) {
        return unexpected("Moneyness grid must have ≥4 points for cubic B-splines");
    }
    if (maturity_.size() < 4) {
        return unexpected("Maturity grid must have ≥4 points for cubic B-splines");
    }
    if (volatility_.size() < 4) {
        return unexpected("Volatility grid must have ≥4 points for cubic B-splines");
    }
    if (rate_.size() < 4) {
        return unexpected("Rate grid must have ≥4 points for cubic B-splines");
    }
    if (K_ref_ <= 0.0) {
        return unexpected("Reference strike K_ref must be positive");
    }

    // Verify sorted
    auto is_sorted = [](const std::vector<double>& v) {
        return std::is_sorted(v.begin(), v.end());
    };

    if (!is_sorted(moneyness_)) {
        return unexpected("Moneyness grid must be sorted");
    }
    if (!is_sorted(maturity_)) {
        return unexpected("Maturity grid must be sorted");
    }
    if (!is_sorted(volatility_)) {
        return unexpected("Volatility grid must be sorted");
    }
    if (!is_sorted(rate_)) {
        return unexpected("Rate grid must be sorted");
    }

    // Verify positive
    if (maturity_.front() <= 0.0) {
        return unexpected("Maturity must be positive");
    }
    if (volatility_.front() <= 0.0) {
        return unexpected("Volatility must be positive");
    }

    // Verify moneyness values are positive
    // CRITICAL: PDE works in log-moneyness x = ln(m), so m must be > 0
    // Moneyness grid should represent S/K_ref ratios, not raw spots
    for (size_t i = 0; i < moneyness_.size(); ++i) {
        if (moneyness_[i] <= 0.0) {
            return unexpected(
                "Moneyness values must be positive (m = S/K_ref > 0). "
                "Found m[" + std::to_string(i) + "] = " + std::to_string(moneyness_[i]) + ". "
                "Note: moneyness represents spot ratios S/K_ref, not log-moneyness x = ln(S/K_ref)."
            );
        }
    }

    return {};
}

bool PriceTable4DBuilder::should_use_normalized_solver(
    double x_min,
    double x_max,
    size_t n_space,
    const std::vector<std::pair<double, double>>& discrete_dividends) const
{
    // Check 1: No discrete dividends (normalized solver requirement)
    if (!discrete_dividends.empty()) {
        return false;
    }

    // Check 2: Build test request and check eligibility
    // Use first volatility/rate for eligibility check (grid params are same for all)
    NormalizedSolveRequest test_request{
        .sigma = volatility_.front(),
        .rate = rate_.front(),
        .dividend = 0.0,  // Will be set per-solve
        .option_type = OptionType::PUT,  // Doesn't affect eligibility
        .x_min = x_min,
        .x_max = x_max,
        .n_space = n_space,
        .n_time = 1000,  // Typical value
        .T_max = maturity_.back(),
        .tau_snapshots = std::span{maturity_}
    };

    auto eligibility = NormalizedChainSolver::check_eligibility(
        test_request, std::span{moneyness_});

    return eligibility.has_value();
}

expected<PriceTable4DResult, std::string> PriceTable4DBuilder::precompute(
    OptionType option_type,
    size_t n_space,
    size_t n_time,
    double dividend_yield)
{
    // Standard bounds: [-3.0, 3.0] log-moneyness
    constexpr double x_min = -3.0;
    constexpr double x_max = 3.0;
    return precompute(option_type, x_min, x_max, n_space, n_time, dividend_yield);
}

expected<PriceTable4DResult, std::string> PriceTable4DBuilder::precompute(
    OptionType option_type,
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time,
    double dividend_yield)
{
    const size_t Nm = moneyness_.size();
    const size_t Nt = maturity_.size();
    const size_t Nv = volatility_.size();
    const size_t Nr = rate_.size();

    // Validate that requested moneyness range fits within PDE grid bounds
    // CRITICAL: PDE works in log-moneyness x = ln(m), and SnapshotInterpolator
    // uses natural cubic splines that extrapolate unpredictably outside knot domain.
    // If any requested ln(m) lies outside [x_min, x_max], the interpolation
    // will produce arbitrary extrapolation artifacts.

    const double x_min_requested = std::log(moneyness_.front());
    const double x_max_requested = std::log(moneyness_.back());

    if (x_min_requested < x_min || x_max_requested > x_max) {
        return unexpected(
            "Requested moneyness range [" + std::to_string(moneyness_.front()) + ", " +
            std::to_string(moneyness_.back()) + "] in spot ratios "
            "maps to log-moneyness [" + std::to_string(x_min_requested) + ", " +
            std::to_string(x_max_requested) + "], "
            "which exceeds PDE grid bounds [" + std::to_string(x_min) + ", " +
            std::to_string(x_max) + "]. "
            "Either narrow the moneyness grid or expand the PDE x_min/x_max bounds. "
            "Example: for moneyness [0.7, 1.5], use x_min <= " +
            std::to_string(x_min_requested) + " and x_max >= " +
            std::to_string(x_max_requested) + "."
        );
    }

    // Allocate 4D price array
    std::vector<double> prices_4d(Nm * Nt * Nv * Nr, 0.0);

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Routing decision: normalized solver or batch API?
    bool use_normalized_solver = should_use_normalized_solver(
        x_min, x_max, n_space, {});  // No discrete dividends

    size_t failed_count = 0;

    if (use_normalized_solver) {
        // FAST PATH: Normalized solver
        const double T_max = maturity_.back();

#pragma omp parallel
        {
            // Create normalized request template (per-thread)
            NormalizedSolveRequest base_request{
                .sigma = 0.20,  // Placeholder, set in loop
                .rate = 0.05,   // Placeholder, set in loop
                .dividend = dividend_yield,
                .option_type = option_type,
                .x_min = x_min,
                .x_max = x_max,
                .n_space = n_space,
                .n_time = n_time,
                .T_max = T_max,
                .tau_snapshots = std::span{maturity_}
            };

            // Create workspace once per thread (OUTSIDE work-sharing loop)
            auto workspace_result = NormalizedWorkspace::create(base_request);

            if (!workspace_result) {
                // Workspace creation failed, mark all as errors
#pragma omp for collapse(2)
                for (size_t k = 0; k < Nv; ++k) {
                    for (size_t l = 0; l < Nr; ++l) {
#pragma omp atomic
                        ++failed_count;
                    }
                }
            } else {
                auto workspace = std::move(workspace_result.value());
                auto surface = workspace.surface_view();

#pragma omp for collapse(2) schedule(dynamic, 1)
                for (size_t k = 0; k < Nv; ++k) {
                    for (size_t l = 0; l < Nr; ++l) {
                        // Set (σ, r) for this solve
                        NormalizedSolveRequest request = base_request;
                        request.sigma = volatility_[k];
                        request.rate = rate_[l];

                        // Solve normalized PDE
                        auto solve_result = NormalizedChainSolver::solve(
                            request, workspace, surface);

                        if (!solve_result) {
#pragma omp atomic
                            ++failed_count;
                            continue;
                        }

                        // Extract prices from surface
                        // Moneyness convention: m = S/K_ref, strike is always K_ref
                        // Identity: V(S,K_ref,τ) = K_ref · u(ln(m), τ)
                        for (size_t i = 0; i < Nm; ++i) {
                            double x = std::log(moneyness_[i]);  // x = ln(m) = ln(S/K_ref)

                            for (size_t j = 0; j < Nt; ++j) {
                                double u = surface.interpolate(x, maturity_[j]);
                                size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;
                                prices_4d[idx_4d] = K_ref_ * u;  // V = K_ref·u (strike is constant)
                            }
                        }
                    }
                }
            }
        }

    } else {
        // FALLBACK PATH: Batch API with snapshots
        const double T_max = maturity_.back();
        const double dt = T_max / n_time;

        // Precompute step indices for each maturity
        std::vector<size_t> step_indices(Nt);
        for (size_t j = 0; j < Nt; ++j) {
            double step_exact = maturity_[j] / dt - 1.0;
            long long step_rounded = std::llround(step_exact);

            if (step_rounded < 0) {
                step_indices[j] = 0;
            } else if (step_rounded >= static_cast<long long>(n_time)) {
                step_indices[j] = n_time - 1;
            } else {
                step_indices[j] = static_cast<size_t>(step_rounded);
            }
        }

        // Build batch parameters (all (σ,r) combinations)
        std::vector<AmericanOptionParams> batch_params;
        batch_params.reserve(Nv * Nr);

        for (size_t k = 0; k < Nv; ++k) {
            for (size_t l = 0; l < Nr; ++l) {
                batch_params.push_back({
                    .strike = K_ref_,
                    .spot = K_ref_,
                    .maturity = T_max,
                    .volatility = volatility_[k],
                    .rate = rate_[l],
                    .continuous_dividend_yield = dividend_yield,
                    .option_type = option_type,
                    .discrete_dividends = {}
                });
            }
        }

        // Create collectors for each batch item
        std::vector<PriceTableSnapshotCollector> collectors;
        collectors.reserve(Nv * Nr);

        for (size_t idx = 0; idx < Nv * Nr; ++idx) {
            PriceTableSnapshotCollectorConfig collector_config{
                .moneyness = std::span{moneyness_},
                .tau = std::span{maturity_},
                .K_ref = K_ref_,
                .option_type = option_type,
                .payoff_params = nullptr
            };
            collectors.emplace_back(collector_config);
        }

        // Solve batch with snapshot registration via callback
        auto results = BatchAmericanOptionSolver::solve_batch(
            batch_params, x_min, x_max, n_space, n_time,
            [&](size_t idx, AmericanOptionSolver& solver) {
                // Register snapshots for all maturities
                for (size_t j = 0; j < Nt; ++j) {
                    solver.register_snapshot(step_indices[j], j, &collectors[idx]);
                }
            });

        // Extract prices from collectors
        for (size_t idx = 0; idx < Nv * Nr; ++idx) {
            size_t k = idx / Nr;
            size_t l = idx % Nr;

            if (!results[idx].has_value()) {
                ++failed_count;
                continue;
            }

            auto prices_2d = collectors[idx].prices();
            for (size_t i = 0; i < Nm; ++i) {
                for (size_t j = 0; j < Nt; ++j) {
                    size_t idx_2d = i * Nt + j;
                    size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;
                    prices_4d[idx_4d] = prices_2d[idx_2d];
                }
            }
        }
    }  // End of else (FALLBACK PATH)

    if (failed_count > 0) {
        return unexpected("Failed to solve " + std::to_string(failed_count) +
                         " out of " + std::to_string(Nv * Nr) + " PDEs");
    }

    // End timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time);

    // Fit B-spline coefficients using factory pattern
    auto fitter_result = BSplineFitter4D::create(moneyness_, maturity_, volatility_, rate_);
    if (!fitter_result.has_value()) {
        return unexpected("B-spline fitter creation failed: " + fitter_result.error());
    }
    auto fit_result = fitter_result.value().fit(prices_4d);

    if (!fit_result.success) {
        return unexpected("B-spline fitting failed: " + fit_result.error_message);
    }

    // Create evaluator
    auto evaluator = std::make_unique<BSpline4D_FMA>(
        moneyness_, maturity_, volatility_, rate_, fit_result.coefficients);

    // Populate fitting statistics from result
    BSplineFittingStats fitting_stats{
        .max_residual_m = fit_result.max_residual_m,
        .max_residual_tau = fit_result.max_residual_tau,
        .max_residual_sigma = fit_result.max_residual_sigma,
        .max_residual_r = fit_result.max_residual_r,
        .max_residual_overall = fit_result.max_residual,
        .condition_m = fit_result.condition_m,
        .condition_tau = fit_result.condition_tau,
        .condition_sigma = fit_result.condition_sigma,
        .condition_r = fit_result.condition_r,
        .condition_max = std::max({
            fit_result.condition_m,
            fit_result.condition_tau,
            fit_result.condition_sigma,
            fit_result.condition_r
        }),
        .failed_slices_m = fit_result.failed_slices_m,
        .failed_slices_tau = fit_result.failed_slices_tau,
        .failed_slices_sigma = fit_result.failed_slices_sigma,
        .failed_slices_r = fit_result.failed_slices_r,
        .failed_slices_total = fit_result.failed_slices_m +
                               fit_result.failed_slices_tau +
                               fit_result.failed_slices_sigma +
                               fit_result.failed_slices_r
    };

    return PriceTable4DResult{
        .evaluator = std::move(evaluator),
        .prices_4d = std::move(prices_4d),
        .n_pde_solves = Nv * Nr,  // Now correct: O(Nσ × Nr) not O(Nm × Nt × Nσ × Nr)
        .precompute_time_seconds = duration.count(),
        .fitting_stats = fitting_stats
    };
}

}  // namespace mango
