// SPDX-License-Identifier: MIT
/**
 * @file interpolation_greek_accuracy.cc
 * @brief Accuracy benchmark: interpolated Greeks vs PDE solver reference
 *
 * Builds an EEP price table, then compares interpolated Greeks from
 * AmericanPriceSurface against PDE-solver Greeks across a grid of
 * (strike, maturity) combinations.
 *
 * Reports max/mean absolute error and basis points for each Greek.
 *
 * Usage:
 *   bazel run //benchmarks:interpolation_greek_accuracy
 */

#include "src/option/american_option.hpp"
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/option/table/american_price_surface.hpp"
#include <benchmark/benchmark.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace mango;

namespace {

// ---------------------------------------------------------------------------
// Fixture: build EEP surface once (expensive), reuse across iterations
// ---------------------------------------------------------------------------

struct EEPFixture {
    AmericanPriceSurface aps;
    double K_ref;
    double dividend_yield;
};

const EEPFixture& GetEEPFixture() {
    static EEPFixture* fixture = [] {
        // Dense grids for good interpolation coverage
        auto linspace = [](double lo, double hi, int n) {
            std::vector<double> v(n);
            for (int i = 0; i < n; ++i)
                v[i] = lo + (hi - lo) * i / (n - 1);
            return v;
        };

        auto m_grid   = linspace(0.70, 1.40, 15);
        auto tau_grid = linspace(0.05, 2.50, 10);
        auto vol_grid = linspace(0.08, 0.50, 8);
        auto rate_grid = linspace(0.00, 0.12, 6);

        double K_ref = 100.0;
        double q = 0.02;

        auto result = PriceTableBuilder<4>::from_vectors(
            m_grid, tau_grid, vol_grid, rate_grid, K_ref,
            GridAccuracyParams{}, OptionType::PUT, q);
        if (!result) {
            throw std::runtime_error("Failed to create PriceTableBuilder");
        }
        auto [builder, axes] = std::move(result.value());
        auto table = builder.build(axes);
        if (!table) {
            throw std::runtime_error("Failed to build price table");
        }

        auto aps = AmericanPriceSurface::create(table->surface, OptionType::PUT);
        if (!aps) {
            throw std::runtime_error("Failed to create AmericanPriceSurface");
        }

        return new EEPFixture{std::move(*aps), K_ref, q};
    }();

    return *fixture;
}

// ---------------------------------------------------------------------------
// Error accumulator
// ---------------------------------------------------------------------------

struct GreekErrors {
    double max_err = 0.0;
    double sum_err = 0.0;
    double max_bps = 0.0;  // basis points relative to price
    int count = 0;

    void record(double interp, double ref, double ref_price) {
        double err = std::abs(interp - ref);
        max_err = std::max(max_err, err);
        sum_err += err;
        if (ref_price > 0.01) {
            max_bps = std::max(max_bps, err / ref_price * 10000.0);
        }
        ++count;
    }

    double mean_err() const { return count > 0 ? sum_err / count : 0.0; }
};

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

static void BM_InterpolationGreekAccuracy(benchmark::State& state) {
    const auto& fix = GetEEPFixture();

    constexpr double spot = 100.0;
    constexpr double sigma = 0.20;
    constexpr double rate = 0.05;

    const std::vector<double> strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
    const std::vector<double> maturities = {0.25, 0.5, 1.0, 2.0};

    // Pre-compute PDE reference (expensive, do once)
    struct RefGreeks {
        double price, delta, gamma, theta;
    };
    std::vector<RefGreeks> refs;
    refs.reserve(strikes.size() * maturities.size());

    for (double K : strikes) {
        for (double tau : maturities) {
            AmericanOptionParams params(
                spot, K, tau, rate, fix.dividend_yield, OptionType::PUT, sigma);
            auto [grid_spec, time_domain] = estimate_grid_for_option(params);
            size_t n = grid_spec.n_points();
            std::pmr::synchronized_pool_resource pool;
            std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);
            auto workspace = PDEWorkspace::from_buffer(buffer, n);
            if (!workspace) {
                throw std::runtime_error("Failed to create workspace");
            }
            AmericanOptionSolver solver(params, workspace.value());
            auto result = solver.solve();
            if (!result) {
                throw std::runtime_error("PDE solver failed");
            }
            refs.push_back({
                result->value_at(spot),
                result->delta(),
                result->gamma(),
                result->theta()
            });
        }
    }

    // Benchmark loop: compute interpolated Greeks and accumulate errors
    GreekErrors price_err, delta_err, gamma_err, theta_err;

    for (auto _ : state) {
        price_err = {};
        delta_err = {};
        gamma_err = {};
        theta_err = {};

        size_t idx = 0;
        for (double K : strikes) {
            for (double tau : maturities) {
                const auto& ref = refs[idx++];

                double i_price = fix.aps.price(spot, K, tau, sigma, rate);
                double i_delta = fix.aps.delta(spot, K, tau, sigma, rate);
                double i_gamma = fix.aps.gamma(spot, K, tau, sigma, rate);
                double i_theta = fix.aps.theta(spot, K, tau, sigma, rate);

                benchmark::DoNotOptimize(i_price);
                benchmark::DoNotOptimize(i_delta);
                benchmark::DoNotOptimize(i_gamma);
                benchmark::DoNotOptimize(i_theta);

                price_err.record(i_price, ref.price, ref.price);
                delta_err.record(i_delta, ref.delta, ref.price);
                gamma_err.record(i_gamma, ref.gamma, ref.price);
                // PDE theta is dV/dt (calendar), interpolated theta is dP/d(tau).
                // They differ by sign: dV/dt = -dV/d(tau). Compare with sign flip.
                theta_err.record(i_theta, -ref.theta, ref.price);
            }
        }
    }

    state.counters["n_combos"] = static_cast<double>(refs.size());
    state.counters["max_price_err"] = price_err.max_err;
    state.counters["mean_price_err"] = price_err.mean_err();
    state.counters["max_price_bps"] = price_err.max_bps;
    state.counters["max_delta_err"] = delta_err.max_err;
    state.counters["mean_delta_err"] = delta_err.mean_err();
    state.counters["max_delta_bps"] = delta_err.max_bps;
    state.counters["max_gamma_err"] = gamma_err.max_err;
    state.counters["mean_gamma_err"] = gamma_err.mean_err();
    state.counters["max_gamma_bps"] = gamma_err.max_bps;
    state.counters["max_theta_err"] = theta_err.max_err;
    state.counters["mean_theta_err"] = theta_err.mean_err();
    state.counters["max_theta_bps"] = theta_err.max_bps;
    state.SetLabel(std::format("{} combos", refs.size()));
}
BENCHMARK(BM_InterpolationGreekAccuracy)->Iterations(1);

}  // namespace

// ---------------------------------------------------------------------------
// Custom reporter: print accuracy summary table
// ---------------------------------------------------------------------------

class AccuracySummaryReporter : public benchmark::ConsoleReporter {
public:
    bool ReportContext(const Context& context) override {
        return ConsoleReporter::ReportContext(context);
    }

    void ReportRuns(const std::vector<Run>& reports) override {
        ConsoleReporter::ReportRuns(reports);

        for (const auto& run : reports) {
            if (run.counters.empty()) continue;

            auto get = [&](const char* name) -> double {
                auto it = run.counters.find(name);
                return it != run.counters.end() ? static_cast<double>(it->second) : 0.0;
            };

            auto& out = GetOutputStream();
            out << "\n";
            out << "Interpolation Greek Accuracy vs PDE Solver\n";
            out << "-------------------------------------------\n";
            out << std::format("  Combos tested: {:.0f}\n", get("n_combos"));
            out << "\n";
            out << std::format("  {:>8s}  {:>12s}  {:>12s}  {:>10s}\n",
                               "Greek", "Max Error", "Mean Error", "Max (bps)");
            out << std::format("  {:>8s}  {:>12s}  {:>12s}  {:>10s}\n",
                               "--------", "------------", "------------", "----------");
            out << std::format("  {:>8s}  {:>12.6f}  {:>12.6f}  {:>10.1f}\n",
                               "Price", get("max_price_err"), get("mean_price_err"), get("max_price_bps"));
            out << std::format("  {:>8s}  {:>12.6f}  {:>12.6f}  {:>10.1f}\n",
                               "Delta", get("max_delta_err"), get("mean_delta_err"), get("max_delta_bps"));
            out << std::format("  {:>8s}  {:>12.6f}  {:>12.6f}  {:>10.1f}\n",
                               "Gamma", get("max_gamma_err"), get("mean_gamma_err"), get("max_gamma_bps"));
            out << std::format("  {:>8s}  {:>12.6f}  {:>12.6f}  {:>10.1f}\n",
                               "Theta", get("max_theta_err"), get("mean_theta_err"), get("max_theta_bps"));
            out << "\n";
            out << "  bps = |error| / reference_price * 10000\n";
        }
    }
};

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    AccuracySummaryReporter reporter;
    benchmark::RunSpecifiedBenchmarks(&reporter);
    benchmark::Shutdown();
    return 0;
}
