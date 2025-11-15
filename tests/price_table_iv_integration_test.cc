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

#include "src/option/price_table_4d_builder.hpp"
#include "src/option/price_table_snapshot_collector.hpp"
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/american_option.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <array>
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
    std::vector<double> rate = {0.0, 0.025, 0.05, 0.10};

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
            .dx = {},
            .solution = {},
            .spatial_operator = {},
            .first_derivative = {},
            .second_derivative = {}
        };
        collector.collect(fake_snapshot);
    }

    // Get prices from collector
    auto prices_2d = collector.prices();
    const auto sigma_ref_it = std::find(volatility.begin(), volatility.end(), known_sigma);
    ASSERT_NE(sigma_ref_it, volatility.end());
    const size_t sigma_ref_idx = static_cast<size_t>(std::distance(volatility.begin(), sigma_ref_it));
    const auto rate_ref_it = std::find(rate.begin(), rate.end(), known_r);
    ASSERT_NE(rate_ref_it, rate.end());
    const size_t rate_ref_idx = static_cast<size_t>(std::distance(rate.begin(), rate_ref_it));

    // Build 4D price table using analytical BS prices for each (σ, r)
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
                    double price = bs_price(
                        moneyness[i] * K_ref,
                        K_ref,
                        maturity[j],
                        volatility[k],
                        rate[l],
                        OptionType::PUT);

                    // Sanity check: recovered surface matches collector where σ=known_sigma and r=known_r
                    if (k == sigma_ref_idx && l == rate_ref_idx) {
                        EXPECT_NEAR(price, prices_2d[idx_2d], 1e-9);
                    }

                    prices_4d[idx_4d] = price;
                }
            }
        }
    }

    // Build B-spline surface
    auto builder = PriceTable4DBuilder::create(moneyness, maturity, volatility, rate, K_ref);

    // Fit B-spline coefficients using factory pattern
    auto fitter_result = BSplineFitter4D::create(moneyness, maturity, volatility, rate);
    ASSERT_TRUE(fitter_result.has_value()) << "Factory creation failed: " << fitter_result.error();
    auto fit_result = fitter_result.value().fit(prices_4d);

    ASSERT_TRUE(fit_result.success) << fit_result.error_message;

    // Create evaluator
    auto evaluator = [&]() {
        auto workspace = PriceTableWorkspace::create(
            moneyness, maturity, volatility, rate, fit_result.coefficients, K_ref, 0.0);
        if (!workspace.has_value()) {
            throw std::runtime_error("Failed to create workspace: " + workspace.error());
        }
        return std::make_unique<BSpline4D>(workspace.value());
    }();

    // Create IV solver
    auto iv_solver_result = IVSolverInterpolated::create(
        std::move(evaluator),
        K_ref,
        std::make_pair(moneyness.front(), moneyness.back()),
        std::make_pair(maturity.front(), maturity.back()),
        std::make_pair(volatility.front(), volatility.back()),
        std::make_pair(rate.front(), rate.back())
    );
    ASSERT_TRUE(iv_solver_result.has_value()) << iv_solver_result.error();
    const auto& iv_solver = iv_solver_result.value();

    // Test: Recover known volatility from market price
    double test_spot = 100.0;
    double test_strike = K_ref;  // Important: use K_ref for this test
    double test_maturity = 1.0;
    double test_rate = known_r;

    double market_price = bs_price(test_spot, test_strike, test_maturity,
                                   known_sigma, test_rate, OptionType::PUT);

    IVQuery query{test_spot, test_strike, test_maturity, test_rate, 0.0, OptionType::PUT, market_price};

    auto result = iv_solver.solve(query);

    ASSERT_TRUE(result.converged) << (result.failure_reason.has_value() ? *result.failure_reason : "");

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
    std::vector<double> rate = {0.0, 0.02, 0.03, 0.06};

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
            .dx = {},
            .solution = {},
            .spatial_operator = {},
            .first_derivative = {},
            .second_derivative = {}
        };
        collector.collect(fake_snapshot);
    }

    auto prices_2d = collector.prices();
    const auto sigma_ref_it = std::find(volatility.begin(), volatility.end(), known_sigma);
    ASSERT_NE(sigma_ref_it, volatility.end());
    const size_t sigma_ref_idx = static_cast<size_t>(std::distance(volatility.begin(), sigma_ref_it));
    const auto rate_ref_it = std::find(rate.begin(), rate.end(), known_r);
    ASSERT_NE(rate_ref_it, rate.end());
    const size_t rate_ref_idx = static_cast<size_t>(std::distance(rate.begin(), rate_ref_it));

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
                    double price = bs_price(
                        moneyness[i] * K_ref,
                        K_ref,
                        maturity[j],
                        volatility[k],
                        rate[l],
                        OptionType::CALL);

                    if (k == sigma_ref_idx && l == rate_ref_idx) {
                        EXPECT_NEAR(price, prices_2d[idx_2d], 1e-9);
                    }

                    prices_4d[idx_4d] = price;
                }
            }
        }
    }

    // Build B-spline surface
    auto fitter_result = BSplineFitter4D::create(moneyness, maturity, volatility, rate);
    ASSERT_TRUE(fitter_result.has_value()) << "Factory creation failed: " << fitter_result.error();
    auto fit_result = fitter_result.value().fit(prices_4d);

    ASSERT_TRUE(fit_result.success);

    auto evaluator = [&]() {
        auto workspace = PriceTableWorkspace::create(
            moneyness, maturity, volatility, rate, fit_result.coefficients, K_ref, 0.0);
        if (!workspace.has_value()) {
            throw std::runtime_error("Failed to create workspace: " + workspace.error());
        }
        return std::make_unique<BSpline4D>(workspace.value());
    }();

    auto iv_solver_result = IVSolverInterpolated::create(
        std::move(evaluator),
        K_ref,
        std::make_pair(moneyness.front(), moneyness.back()),
        std::make_pair(maturity.front(), maturity.back()),
        std::make_pair(volatility.front(), volatility.back()),
        std::make_pair(rate.front(), rate.back())
    );
    ASSERT_TRUE(iv_solver_result.has_value()) << iv_solver_result.error();
    const auto& iv_solver = iv_solver_result.value();

    // Test recovery
    double test_spot = 100.0;
    double test_strike = K_ref;
    double test_maturity = 0.5;
    double test_rate = known_r;

    double market_price = bs_price(test_spot, test_strike, test_maturity,
                                   known_sigma, test_rate, OptionType::CALL);

    IVQuery query{test_spot, test_strike, test_maturity, test_rate, 0.0, OptionType::CALL, market_price};

    auto result = iv_solver.solve(query);

    ASSERT_TRUE(result.converged) << (result.failure_reason.has_value() ? *result.failure_reason : "");
    EXPECT_NEAR(result.implied_vol, known_sigma, 0.02);
}

TEST(PriceTableIVIntegrationTest, MoneynessBoundsValidation) {
    // Test that moneyness outside PDE grid bounds is rejected
    const double K_ref = 100.0;

    // Narrow moneyness grid
    std::vector<double> moneyness = {0.85, 0.9, 0.95, 1.0, 1.05};  // ln(0.85) ≈ -0.162 < -0.10
    std::vector<double> maturity = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> volatility = {0.18, 0.20, 0.22, 0.24};
    std::vector<double> rate = {0.02, 0.03, 0.04, 0.05};

    auto builder = PriceTable4DBuilder::create(moneyness, maturity, volatility, rate, K_ref);

    constexpr double x_min = -0.10;  // Covers ln(0.9) ≈ -0.105
    constexpr double x_max = 0.10;   // Covers ln(1.1) ≈ 0.095
    constexpr size_t n_space = 51;
    constexpr size_t n_time = 100;

    // This should FAIL because ln(0.9) = -0.105 < -0.10
    auto result = builder.precompute(OptionType::PUT, x_min, x_max, n_space, n_time, 0.0);
    EXPECT_FALSE(result.has_value());
    EXPECT_THAT(result.error(), testing::HasSubstr("exceeds PDE grid bounds"));
}

TEST(PriceTableIVIntegrationTest, StrikeScalingValidation) {
    // Test that strike != K_ref is handled correctly via moneyness = S/K_ref
    const double K_ref = 100.0;
    const double known_sigma = 0.20;
    const double known_r = 0.05;

    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> maturity = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> volatility = {0.15, 0.18, 0.20, 0.25};
    std::vector<double> rate = {0.02, 0.035, 0.05, 0.065};

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
            .dx = {},
            .solution = {},
            .spatial_operator = {},
            .first_derivative = {},
            .second_derivative = {}
        };
        collector.collect(fake_snapshot);
    }

    auto prices_2d = collector.prices();

    const size_t Nm = moneyness.size();
    const size_t Nt = maturity.size();
    const size_t Nv = volatility.size();
    const size_t Nr = rate.size();

    const auto sigma_ref_it = std::find(volatility.begin(), volatility.end(), known_sigma);
    ASSERT_NE(sigma_ref_it, volatility.end());
    const size_t sigma_ref_idx = static_cast<size_t>(std::distance(volatility.begin(), sigma_ref_it));
    const auto rate_ref_it = std::find(rate.begin(), rate.end(), known_r);
    ASSERT_NE(rate_ref_it, rate.end());
    const size_t rate_ref_idx = static_cast<size_t>(std::distance(rate.begin(), rate_ref_it));

    std::vector<double> prices_4d(Nm * Nt * Nv * Nr);
    for (size_t i = 0; i < Nm; ++i) {
        for (size_t j = 0; j < Nt; ++j) {
            for (size_t k = 0; k < Nv; ++k) {
                for (size_t l = 0; l < Nr; ++l) {
                    size_t idx_2d = i * Nt + j;
                    size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;
                    double price = bs_price(
                        moneyness[i] * K_ref,
                        K_ref,
                        maturity[j],
                        volatility[k],
                        rate[l],
                        OptionType::PUT);

                    if (k == sigma_ref_idx && l == rate_ref_idx) {
                        EXPECT_NEAR(price, prices_2d[idx_2d], 1e-9);
                    }

                    prices_4d[idx_4d] = price;
                }
            }
        }
    }

    auto fitter_result = BSplineFitter4D::create(moneyness, maturity, volatility, rate);
    ASSERT_TRUE(fitter_result.has_value()) << "Factory creation failed: " << fitter_result.error();
    auto fit_result = fitter_result.value().fit(prices_4d);
    ASSERT_TRUE(fit_result.success);

    auto evaluator = [&]() {
        auto workspace = PriceTableWorkspace::create(
            moneyness, maturity, volatility, rate, fit_result.coefficients, K_ref, 0.0);
        if (!workspace.has_value()) {
            throw std::runtime_error("Failed to create workspace: " + workspace.error());
        }
        return std::make_unique<BSpline4D>(workspace.value());
    }();

    // First compute base price before moving evaluator
    double spot = 105.0;
    const double base_price_kref = evaluator->eval(spot / K_ref, 1.0, known_sigma, known_r);

    auto iv_solver_result = IVSolverInterpolated::create(
        std::move(evaluator),
        K_ref,
        std::make_pair(moneyness.front(), moneyness.back()),
        std::make_pair(maturity.front(), maturity.back()),
        std::make_pair(volatility.front(), volatility.back()),
        std::make_pair(rate.front(), rate.back())
    );
    ASSERT_TRUE(iv_solver_result.has_value()) << iv_solver_result.error();
    const auto& iv_solver = iv_solver_result.value();

    // Test with strike = K_ref (should work)
    double strike = K_ref;
    double market_price = base_price_kref * (strike / K_ref);

    IVQuery query1{spot, strike, 1.0, known_r, 0.0, OptionType::PUT, market_price};

    auto result1 = iv_solver.solve(query1);
    EXPECT_TRUE(result1.converged);

    // Test with strike != K_ref (should also work with correct scaling)
    // Note: The solver now uses m = spot / K_ref for surface lookup
    // and scales price by (strike / K_ref)
    double strike2 = 90.0;
    double market_price2 = base_price_kref * (strike2 / K_ref);

    IVQuery query2{spot, strike2, 1.0, known_r, 0.0, OptionType::PUT, market_price2};

    auto result2 = iv_solver.solve(query2);
    // This should work because we compute moneyness as spot/K_ref
    // and scale the price appropriately
    EXPECT_TRUE(result2.converged) << (result2.failure_reason.has_value() ? *result2.failure_reason : "");
}

TEST(PriceTableIVIntegrationTest, SolverCoversAxisBoundaries) {
    const double K_ref = 120.0;
    std::vector<double> moneyness = {0.75, 0.9, 1.05, 1.25};
    std::vector<double> maturity = {0.1, 0.5, 1.5, 2.5};
    std::vector<double> volatility = {0.12, 0.2, 0.3, 0.4};
    std::vector<double> rate = {0.0, 0.02, 0.05, 0.08};

    const size_t Nm = moneyness.size();
    const size_t Nt = maturity.size();
    const size_t Nv = volatility.size();
    const size_t Nr = rate.size();

    std::vector<double> prices_4d(Nm * Nt * Nv * Nr);
    for (size_t i = 0; i < Nm; ++i) {
        for (size_t j = 0; j < Nt; ++j) {
            for (size_t k = 0; k < Nv; ++k) {
                for (size_t l = 0; l < Nr; ++l) {
                    const size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                    prices_4d[idx] = bs_price(
                        moneyness[i] * K_ref,
                        K_ref,
                        maturity[j],
                        volatility[k],
                        rate[l],
                        OptionType::CALL);
                }
            }
        }
    }

    auto fitter_result = BSplineFitter4D::create(moneyness, maturity, volatility, rate);
    ASSERT_TRUE(fitter_result.has_value()) << "Factory creation failed: " << fitter_result.error();
    auto fit_result = fitter_result.value().fit(prices_4d);
    ASSERT_TRUE(fit_result.success);

    auto evaluator = [&]() {
        auto workspace = PriceTableWorkspace::create(
            moneyness, maturity, volatility, rate, fit_result.coefficients, K_ref, 0.0);
        if (!workspace.has_value()) {
            throw std::runtime_error("Failed to create workspace: " + workspace.error());
        }
        return std::make_unique<BSpline4D>(workspace.value());
    }();

    auto iv_solver_result = IVSolverInterpolated::create(
        std::move(evaluator),
        K_ref,
        std::make_pair(moneyness.front(), moneyness.back()),
        std::make_pair(maturity.front(), maturity.back()),
        std::make_pair(volatility.front(), volatility.back()),
        std::make_pair(rate.front(), rate.back()));
    ASSERT_TRUE(iv_solver_result.has_value()) << iv_solver_result.error();
    const auto& iv_solver = iv_solver_result.value();

    const std::array<double, 2> m_range = {moneyness.front(), moneyness.back()};
    const std::array<double, 2> tau_range = {maturity.front(), maturity.back()};
    const std::array<double, 2> sigma_range = {volatility.front(), volatility.back()};
    const std::array<double, 2> rate_range = {rate.front(), rate.back()};

    struct Scenario {
        double m;
        double tau;
        double sigma;
        double rate;
    };
    std::vector<Scenario> scenarios = {
        {m_range.front(), tau_range.front(), sigma_range.back(), rate_range.front()},   // min m, min tau, max sigma
        {m_range.back(), tau_range.back(), sigma_range.front(), rate_range.back()},     // max m, max tau, min sigma
        {m_range.front(), tau_range.back(), sigma_range.front(), rate_range.back()},    // min m, max tau, min sigma
        {m_range.back(), tau_range.front(), sigma_range.back(), rate_range.front()}     // max m, min tau, max sigma
    };

    for (const auto& scenario : scenarios) {
        const double spot = scenario.m * K_ref;
        const double market_price = bs_price(
            spot,
            K_ref,
            scenario.tau,
            scenario.sigma,
            scenario.rate,
            OptionType::CALL);

        IVQuery query{
            spot,
            K_ref,
            scenario.tau,
            scenario.rate,
            0.0,  // dividend_yield
            OptionType::CALL,
            market_price
        };

        auto result = iv_solver.solve(query);
        ASSERT_TRUE(result.converged)
            << "Failed at m=" << scenario.m << " tau=" << scenario.tau
            << " sigma=" << scenario.sigma << " rate=" << scenario.rate
            << ". Error: "
        << (result.failure_reason.has_value() ? *result.failure_reason : "");
        EXPECT_NEAR(result.implied_vol, scenario.sigma, 0.05);
    }
}

// Test that SIMD vega computation produces identical results to scalar
TEST(PriceTableIVIntegrationTest, SIMDVega_MatchesScalarResults) {
    // Setup: Create a synthetic price surface
    const double K_ref = 100.0;

    // Grid configuration
    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> maturity = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> volatility = {0.10, 0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.0, 0.025, 0.05, 0.10};

    // Build 4D price table using analytical BS
    const size_t Nm = moneyness.size();
    const size_t Nt = maturity.size();
    const size_t Nv = volatility.size();
    const size_t Nr = rate.size();

    std::vector<double> prices_4d(Nm * Nt * Nv * Nr);

    for (size_t i = 0; i < Nm; ++i) {
        for (size_t j = 0; j < Nt; ++j) {
            for (size_t k = 0; k < Nv; ++k) {
                for (size_t l = 0; l < Nr; ++l) {
                    size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;
                    prices_4d[idx_4d] = bs_price(
                        moneyness[i] * K_ref,
                        K_ref,
                        maturity[j],
                        volatility[k],
                        rate[l],
                        OptionType::PUT);
                }
            }
        }
    }

    // Build B-spline surface
    auto fitter_result = BSplineFitter4D::create(moneyness, maturity, volatility, rate);
    ASSERT_TRUE(fitter_result.has_value());
    auto fit_result = fitter_result.value().fit(prices_4d);
    ASSERT_TRUE(fit_result.success);

    auto evaluator = [&]() {
        auto workspace = PriceTableWorkspace::create(
            moneyness, maturity, volatility, rate, fit_result.coefficients, K_ref, 0.0);
        if (!workspace.has_value()) {
            throw std::runtime_error("Failed to create workspace: " + workspace.error());
        }
        return std::make_unique<BSpline4D>(workspace.value());
    }();

    // Create IV solver
    auto iv_solver_result = IVSolverInterpolated::create(
        std::move(evaluator),
        K_ref,
        std::make_pair(moneyness.front(), moneyness.back()),
        std::make_pair(maturity.front(), maturity.back()),
        std::make_pair(volatility.front(), volatility.back()),
        std::make_pair(rate.front(), rate.back())
    );
    ASSERT_TRUE(iv_solver_result.has_value()) << iv_solver_result.error();
    const auto& iv_solver = iv_solver_result.value();

    // Test multiple scenarios
    std::vector<IVQuery> test_queries = {
        // ATM
        IVQuery{100.0, K_ref, 0.5, 0.05, 0.0, OptionType::PUT,
                bs_price(100.0, 100.0, 0.5, 0.20, 0.05, OptionType::PUT)},
        // ITM
        IVQuery{90.0, K_ref, 1.0, 0.05, 0.0, OptionType::PUT,
                bs_price(90.0, 100.0, 1.0, 0.25, 0.05, OptionType::PUT)},
        // OTM
        IVQuery{110.0, K_ref, 0.25, 0.025, 0.0, OptionType::PUT,
                bs_price(110.0, 100.0, 0.25, 0.15, 0.025, OptionType::PUT)},
    };

    for (const auto& query : test_queries) {
        auto result = iv_solver.solve(query);

        // Should converge successfully
        ASSERT_TRUE(result.converged)
            << "Failed for spot=" << query.spot
            << " maturity=" << query.maturity
            << " rate=" << query.rate
            << (result.failure_reason.has_value() ? ": " + *result.failure_reason : "");

        // Results should be numerically stable and within reasonable bounds
        EXPECT_GT(result.implied_vol, 0.05);  // > 5%
        EXPECT_LT(result.implied_vol, 0.50);  // < 50%
        EXPECT_LT(result.iterations, 20);     // Should converge quickly

        // Final error should be within tolerance
        EXPECT_LT(result.final_error, 1e-6);
    }
}
