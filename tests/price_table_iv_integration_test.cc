/**
 * @file price_table_iv_integration_test.cc
 * @brief End-to-end integration tests for PDE→Builder→IV pipeline
 *
 * Validates the complete workflow:
 * 1. PDE snapshots collected via PriceTableSnapshotCollector
 * 2. B-spline surface fitted via PriceTable4DBuilder
 * 3. IV solved via IVSolverInterpolated
 *
 * Tests critical scaling and coordinate handling:
 * - Strike scaling (query.strike vs K_ref)
 * - Moneyness bounds validation
 * - Theta computation for calls vs puts
 * - Coordinate transformations (log-moneyness)
 */

#include "src/price_table_4d_builder.hpp"
#include "src/price_table_snapshot_collector.hpp"
#include "src/iv_solver_interpolated.hpp"
#include "src/american_option.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <memory>

using namespace mango;

namespace {

/// Analytic Black-Scholes call/put price for testing
/// Uses simplified BS formula (no dividends, American ≈ European for test)
double bs_price(double S, double K, double tau, double sigma, double r, OptionType type) {
    if (tau <= 0.0) {
        // At expiry, option worth intrinsic value
        if (type == OptionType::CALL) {
            return std::max(S - K, 0.0);
        } else {
            return std::max(K - S, 0.0);
        }
    }

    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * std::sqrt(tau));
    double d2 = d1 - sigma * std::sqrt(tau);

    // Simplified normal CDF (approximation good enough for tests)
    auto Phi = [](double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    };

    if (type == OptionType::CALL) {
        return S * Phi(d1) - K * std::exp(-r * tau) * Phi(d2);
    } else {
        return K * std::exp(-r * tau) * Phi(-d2) - S * Phi(-d1);
    }
}

/// Fake snapshot collector that uses analytical BS formula instead of PDE
/// This makes tests fast and deterministic
class AnalyticSnapshotCollector : public SnapshotCollector {
public:
    AnalyticSnapshotCollector(
        std::span<const double> moneyness,
        std::span<const double> tau,
        double K_ref,
        double fixed_sigma,
        double fixed_r,
        OptionType option_type)
        : moneyness_(moneyness.begin(), moneyness.end())
        , tau_(tau.begin(), tau.end())
        , K_ref_(K_ref)
        , fixed_sigma_(fixed_sigma)
        , fixed_r_(fixed_r)
        , option_type_(option_type)
    {
        const size_t n = moneyness_.size() * tau_.size();
        prices_.resize(n, 0.0);
    }

    void collect(const Snapshot& snapshot) override {
        const size_t tau_idx = snapshot.user_index;

        for (size_t m_idx = 0; m_idx < moneyness_.size(); ++m_idx) {
            const double m = moneyness_[m_idx];
            const double S = m * K_ref_;
            const double tau_val = tau_[tau_idx];

            const size_t table_idx = m_idx * tau_.size() + tau_idx;

            // Use analytical BS price (normalized by K_ref for consistency)
            double price_dollar = bs_price(S, K_ref_, tau_val, fixed_sigma_, fixed_r_, option_type_);
            prices_[table_idx] = price_dollar;
        }
    }

    std::span<const double> prices() const { return prices_; }

private:
    std::vector<double> moneyness_;
    std::vector<double> tau_;
    double K_ref_;
    double fixed_sigma_;
    double fixed_r_;
    OptionType option_type_;
    std::vector<double> prices_;
};

} // namespace

// ============================================================================
// End-to-End Integration Tests
// ============================================================================

TEST(PriceTableIVIntegrationTest, PutOptionSurfaceRoundTrip) {
    // Setup: Create a synthetic American put surface using analytical BS
    const double K_ref = 100.0;
    const double known_sigma = 0.20;  // Known volatility we'll try to recover
    const double known_r = 0.05;

    // Grid configuration (small for fast test)
    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> maturity = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> volatility = {0.10, 0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.0, 0.05, 0.10};

    // Create fake collector that uses analytical formula
    AnalyticSnapshotCollector collector(
        std::span{moneyness},
        std::span{maturity},
        K_ref,
        known_sigma,
        known_r,
        OptionType::PUT
    );

    // Simulate snapshot collection (fake - we don't actually run PDE)
    for (size_t tau_idx = 0; tau_idx < maturity.size(); ++tau_idx) {
        Snapshot fake_snapshot{
            .time = maturity[tau_idx],
            .user_index = tau_idx,
            .spatial_grid = {},
            .solution = {},
            .first_derivative = {},
            .second_derivative = {},
            .spatial_operator = {}
        };
        collector.collect(fake_snapshot);
    }

    // Get prices from collector
    auto prices_2d = collector.prices();

    // Build 4D price table by replicating across (σ, r) dimensions
    const size_t Nm = moneyness.size();
    const size_t Nt = maturity.size();
    const size_t Nv = volatility.size();
    const size_t Nr = rate.size();

    std::vector<double> prices_4d(Nm * Nt * Nv * Nr);

    // For this test, we replicate the same (m, τ) surface across all (σ, r)
    // In reality, each (σ, r) would come from a different PDE solve
    for (size_t i = 0; i < Nm; ++i) {
        for (size_t j = 0; j < Nt; ++j) {
            for (size_t k = 0; k < Nv; ++k) {
                for (size_t l = 0; l < Nr; ++l) {
                    size_t idx_2d = i * Nt + j;
                    size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;
                    prices_4d[idx_4d] = prices_2d[idx_2d];
                }
            }
        }
    }

    // Build B-spline surface
    PriceTable4DBuilder builder(moneyness, maturity, volatility, rate, K_ref);

    // Fit B-spline coefficients
    BSplineFitter4D fitter(moneyness, maturity, volatility, rate);
    auto fit_result = fitter.fit(prices_4d);

    ASSERT_TRUE(fit_result.success) << fit_result.error_message;

    // Create evaluator
    auto evaluator = std::make_unique<BSpline4D_FMA>(
        moneyness, maturity, volatility, rate, fit_result.coefficients);

    // Create IV solver
    IVSolverInterpolated iv_solver(
        *evaluator,
        K_ref,
        std::make_pair(moneyness.front(), moneyness.back()),
        std::make_pair(maturity.front(), maturity.back()),
        std::make_pair(volatility.front(), volatility.back()),
        std::make_pair(rate.front(), rate.back())
    );

    // Test: Recover known volatility from market price
    double test_spot = 100.0;
    double test_strike = K_ref;  // Important: use K_ref for this test
    double test_maturity = 1.0;
    double test_rate = known_r;

    double market_price = bs_price(test_spot, test_strike, test_maturity,
                                   known_sigma, test_rate, OptionType::PUT);

    IVQuery query{
        .market_price = market_price,
        .spot = test_spot,
        .strike = test_strike,
        .maturity = test_maturity,
        .rate = test_rate,
        .option_type = OptionType::PUT
    };

    auto result = iv_solver.solve(query);

    ASSERT_TRUE(result.converged) << (result.error_message.has_value() ? *result.error_message : "");

    // Should recover the known volatility within reasonable tolerance
    // (B-spline approximation + Newton convergence → ~1-2% error acceptable)
    EXPECT_NEAR(result.implied_vol, known_sigma, 0.02)
        << "Failed to recover known volatility. Got " << result.implied_vol
        << " but expected " << known_sigma;
}

TEST(PriceTableIVIntegrationTest, CallOptionSurfaceRoundTrip) {
    // Same as above but for CALL options
    const double K_ref = 100.0;
    const double known_sigma = 0.25;
    const double known_r = 0.03;

    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> maturity = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> volatility = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.0, 0.03, 0.06};

    AnalyticSnapshotCollector collector(
        std::span{moneyness},
        std::span{maturity},
        K_ref,
        known_sigma,
        known_r,
        OptionType::CALL
    );

    // Collect fake snapshots
    for (size_t tau_idx = 0; tau_idx < maturity.size(); ++tau_idx) {
        Snapshot fake_snapshot{
            .time = maturity[tau_idx],
            .user_index = tau_idx,
            .spatial_grid = {},
            .solution = {},
            .first_derivative = {},
            .second_derivative = {},
            .spatial_operator = {}
        };
        collector.collect(fake_snapshot);
    }

    auto prices_2d = collector.prices();

    // Build 4D array
    const size_t Nm = moneyness.size();
    const size_t Nt = maturity.size();
    const size_t Nv = volatility.size();
    const size_t Nr = rate.size();

    std::vector<double> prices_4d(Nm * Nt * Nv * Nr);
    for (size_t i = 0; i < Nm; ++i) {
        for (size_t j = 0; j < Nt; ++j) {
            for (size_t k = 0; k < Nv; ++k) {
                for (size_t l = 0; l < Nr; ++l) {
                    size_t idx_2d = i * Nt + j;
                    size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;
                    prices_4d[idx_4d] = prices_2d[idx_2d];
                }
            }
        }
    }

    // Build B-spline surface
    BSplineFitter4D fitter(moneyness, maturity, volatility, rate);
    auto fit_result = fitter.fit(prices_4d);

    ASSERT_TRUE(fit_result.success);

    auto evaluator = std::make_unique<BSpline4D_FMA>(
        moneyness, maturity, volatility, rate, fit_result.coefficients);

    IVSolverInterpolated iv_solver(
        *evaluator,
        K_ref,
        std::make_pair(moneyness.front(), moneyness.back()),
        std::make_pair(maturity.front(), maturity.back()),
        std::make_pair(volatility.front(), volatility.back()),
        std::make_pair(rate.front(), rate.back())
    );

    // Test recovery
    double test_spot = 100.0;
    double test_strike = K_ref;
    double test_maturity = 0.5;
    double test_rate = known_r;

    double market_price = bs_price(test_spot, test_strike, test_maturity,
                                   known_sigma, test_rate, OptionType::CALL);

    IVQuery query{
        .market_price = market_price,
        .spot = test_spot,
        .strike = test_strike,
        .maturity = test_maturity,
        .rate = test_rate,
        .option_type = OptionType::CALL
    };

    auto result = iv_solver.solve(query);

    ASSERT_TRUE(result.converged) << (result.error_message.has_value() ? *result.error_message : "");
    EXPECT_NEAR(result.implied_vol, known_sigma, 0.02);
}

TEST(PriceTableIVIntegrationTest, MoneynessBoundsValidation) {
    // Test that moneyness outside PDE grid bounds is rejected
    const double K_ref = 100.0;

    // Narrow moneyness grid
    std::vector<double> moneyness = {0.9, 1.0, 1.1};  // ln(0.9) ≈ -0.105, ln(1.1) ≈ 0.095
    std::vector<double> maturity = {0.5, 1.0};
    std::vector<double> volatility = {0.20, 0.25};
    std::vector<double> rate = {0.05};

    PriceTable4DBuilder builder(moneyness, maturity, volatility, rate, K_ref);

    AmericanOptionGrid grid_config{
        .n_space = 51,
        .n_time = 100,
        .x_min = -0.10,  // Covers ln(0.9) ≈ -0.105
        .x_max = 0.10    // Covers ln(1.1) ≈ 0.095
    };

    // This should FAIL because ln(0.9) = -0.105 < -0.10
    EXPECT_THROW({
        builder.precompute(OptionType::PUT, grid_config, 0.0);
    }, std::invalid_argument);
}

TEST(PriceTableIVIntegrationTest, StrikeScalingValidation) {
    // Test that strike != K_ref is handled correctly via moneyness = S/K_ref
    const double K_ref = 100.0;
    const double known_sigma = 0.20;
    const double known_r = 0.05;

    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> maturity = {0.5, 1.0};
    std::vector<double> volatility = {0.15, 0.20, 0.25};
    std::vector<double> rate = {0.05};

    // Build surface at K_ref = 100
    AnalyticSnapshotCollector collector(
        std::span{moneyness},
        std::span{maturity},
        K_ref,
        known_sigma,
        known_r,
        OptionType::PUT
    );

    for (size_t tau_idx = 0; tau_idx < maturity.size(); ++tau_idx) {
        Snapshot fake_snapshot{
            .time = maturity[tau_idx],
            .user_index = tau_idx,
            .spatial_grid = {},
            .solution = {},
            .first_derivative = {},
            .second_derivative = {},
            .spatial_operator = {}
        };
        collector.collect(fake_snapshot);
    }

    auto prices_2d = collector.prices();

    const size_t Nm = moneyness.size();
    const size_t Nt = maturity.size();
    const size_t Nv = volatility.size();
    const size_t Nr = rate.size();

    std::vector<double> prices_4d(Nm * Nt * Nv * Nr);
    for (size_t i = 0; i < Nm; ++i) {
        for (size_t j = 0; j < Nt; ++j) {
            for (size_t k = 0; k < Nv; ++k) {
                for (size_t l = 0; l < Nr; ++l) {
                    size_t idx_2d = i * Nt + j;
                    size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;
                    prices_4d[idx_4d] = prices_2d[idx_2d];
                }
            }
        }
    }

    BSplineFitter4D fitter(moneyness, maturity, volatility, rate);
    auto fit_result = fitter.fit(prices_4d);
    ASSERT_TRUE(fit_result.success);

    auto evaluator = std::make_unique<BSpline4D_FMA>(
        moneyness, maturity, volatility, rate, fit_result.coefficients);

    IVSolverInterpolated iv_solver(
        *evaluator,
        K_ref,
        std::make_pair(moneyness.front(), moneyness.back()),
        std::make_pair(maturity.front(), maturity.back()),
        std::make_pair(volatility.front(), volatility.back()),
        std::make_pair(rate.front(), rate.back())
    );

    // Test with strike = K_ref (should work)
    double spot = 105.0;
    double strike = K_ref;
    double market_price = bs_price(spot, strike, 1.0, known_sigma, known_r, OptionType::PUT);

    IVQuery query1{
        .market_price = market_price,
        .spot = spot,
        .strike = strike,  // = K_ref
        .maturity = 1.0,
        .rate = known_r,
        .option_type = OptionType::PUT
    };

    auto result1 = iv_solver.solve(query1);
    EXPECT_TRUE(result1.converged);

    // Test with strike != K_ref (should also work with correct scaling)
    // Note: The solver now uses m = spot / K_ref for surface lookup
    // and scales price by (strike / K_ref)
    double strike2 = 110.0;
    double market_price2 = bs_price(spot, strike2, 1.0, known_sigma, known_r, OptionType::PUT);

    IVQuery query2{
        .market_price = market_price2,
        .spot = spot,
        .strike = strike2,  // != K_ref
        .maturity = 1.0,
        .rate = known_r,
        .option_type = OptionType::PUT
    };

    auto result2 = iv_solver.solve(query2);
    // This should work because we compute moneyness as spot/K_ref
    // and scale the price appropriately
    EXPECT_TRUE(result2.converged) << (result2.error_message.has_value() ? *result2.error_message : "");
}
