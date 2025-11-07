/**
 * @file price_table_4d_builder.cpp
 * @brief Implementation of 4D price table builder
 */

#include "price_table_4d_builder.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mango {

void PriceTable4DBuilder::validate_grids() const {
    if (moneyness_.size() < 4) {
        throw std::invalid_argument("Moneyness grid must have ≥4 points for cubic B-splines");
    }
    if (maturity_.size() < 4) {
        throw std::invalid_argument("Maturity grid must have ≥4 points for cubic B-splines");
    }
    if (volatility_.size() < 4) {
        throw std::invalid_argument("Volatility grid must have ≥4 points for cubic B-splines");
    }
    if (rate_.size() < 4) {
        throw std::invalid_argument("Rate grid must have ≥4 points for cubic B-splines");
    }
    if (K_ref_ <= 0.0) {
        throw std::invalid_argument("Reference strike K_ref must be positive");
    }

    // Verify sorted
    auto is_sorted = [](const std::vector<double>& v) {
        return std::is_sorted(v.begin(), v.end());
    };

    if (!is_sorted(moneyness_)) {
        throw std::invalid_argument("Moneyness grid must be sorted");
    }
    if (!is_sorted(maturity_)) {
        throw std::invalid_argument("Maturity grid must be sorted");
    }
    if (!is_sorted(volatility_)) {
        throw std::invalid_argument("Volatility grid must be sorted");
    }
    if (!is_sorted(rate_)) {
        throw std::invalid_argument("Rate grid must be sorted");
    }

    // Verify positive
    if (maturity_.front() <= 0.0) {
        throw std::invalid_argument("Maturity must be positive");
    }
    if (volatility_.front() <= 0.0) {
        throw std::invalid_argument("Volatility must be positive");
    }
}

PriceTable4DResult PriceTable4DBuilder::precompute(
    OptionType option_type,
    const AmericanOptionGrid& grid_config,
    double dividend_yield)
{
    const size_t Nm = moneyness_.size();
    const size_t Nt = maturity_.size();
    const size_t Nv = volatility_.size();
    const size_t Nr = rate_.size();

    // Allocate 4D price array
    std::vector<double> prices_4d(Nm * Nt * Nv * Nr, 0.0);

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create batch of option parameters for each (σ, r) combination
    std::vector<AmericanOptionParams> batch;
    batch.reserve(Nv * Nr);

    for (size_t k = 0; k < Nv; ++k) {
        for (size_t l = 0; l < Nr; ++l) {
            // Use midpoint moneyness and maturity for reference spot calculation
            const double m_mid = moneyness_[Nm / 2];
            const double tau_mid = maturity_[Nt / 2];

            AmericanOptionParams params{
                .strike = K_ref_,
                .spot = m_mid * K_ref_,          // Representative spot
                .maturity = tau_mid,              // Representative maturity
                .volatility = volatility_[k],
                .rate = rate_[l],
                .continuous_dividend_yield = dividend_yield,
                .option_type = option_type
            };

            batch.push_back(params);
        }
    }

    // Solve all options using BatchAmericanOptionSolver
    BatchAmericanOptionSolver solver;
    auto results = solver.solve(batch, grid_config);

    // Process results and fill 4D price array
    size_t result_idx = 0;
    for (size_t k = 0; k < Nv; ++k) {
        for (size_t l = 0; l < Nr; ++l) {
            const auto& result = results[result_idx++];

            if (!result.converged) {
                // For now, continue with zero prices (or could throw)
                continue;
            }

            // For each (m, tau) point, compute price
            // We need to re-solve with actual parameters since batch used midpoint
            // This is a simplification - ideally we'd solve for each (m, tau, σ, r)
            // For now, use the batch result as a first approximation

            // TODO: Implement proper multi-point evaluation
            // For now, fill with the single result value
            for (size_t i = 0; i < Nm; ++i) {
                for (size_t j = 0; j < Nt; ++j) {
                    size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;

                    // Scale by moneyness (rough approximation)
                    const double m = moneyness_[i];
                    const double tau = maturity_[j];

                    // Simple scaling: price scales linearly with moneyness and sqrt(tau)
                    // This is placeholder - proper implementation needs individual solves
                    double scale = m * std::sqrt(tau / maturity_[Nt / 2]);
                    prices_4d[idx_4d] = result.value * scale;
                }
            }
        }
    }

    // End timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time);

    // Fit B-spline coefficients
    BSplineFitter4D fitter(moneyness_, maturity_, volatility_, rate_);
    auto fit_result = fitter.fit(prices_4d);

    if (!fit_result.success) {
        throw std::runtime_error("B-spline fitting failed: " + fit_result.error_message);
    }

    // Create evaluator
    auto evaluator = std::make_unique<BSpline4D_FMA>(
        moneyness_, maturity_, volatility_, rate_, fit_result.coefficients);

    return PriceTable4DResult{
        .evaluator = std::move(evaluator),
        .prices_4d = std::move(prices_4d),
        .n_pde_solves = Nv * Nr,
        .precompute_time_seconds = duration.count()
    };
}

}  // namespace mango
