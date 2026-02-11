// SPDX-License-Identifier: MIT
/**
 * @file interpolation_greek_accuracy.cc
 * @brief Accuracy benchmark: interpolated Greeks vs PDE solver reference
 *
 * Builds an EEP price table, then compares interpolated Greeks from
 * EEP decomposition against PDE-solver Greeks across a grid of
 * (strike, maturity) combinations.
 *
 * Reports max/mean absolute error and basis points for each Greek.
 *
 * Usage:
 *   bazel run //benchmarks:interpolation_greek_accuracy
 */

#include "mango/option/american_option.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/table/bspline/bspline_tensor_accessor.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
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
    BSplinePriceTable wrapper;
    std::shared_ptr<const PriceTableSurface> surface;
    double K_ref;
    double dividend_yield;
    OptionType type;
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

        auto result = PriceTableBuilder::from_vectors(
            m_grid, tau_grid, vol_grid, rate_grid, K_ref,
            GridAccuracyParams{}, OptionType::PUT, q);
        if (!result) {
            throw std::runtime_error("Failed to create PriceTableBuilderND");
        }
        auto [builder, axes] = std::move(result.value());
        auto table = builder.build(axes,
            [&](PriceTensor& tensor, const PriceTableAxes& a) {
                BSplineTensorAccessor accessor(tensor, a, K_ref);
                eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, q));
            });
        if (!table) {
            throw std::runtime_error("Failed to build price table");
        }

        auto wrapper = make_bspline_surface(table->surface, OptionType::PUT);
        if (!wrapper) {
            throw std::runtime_error("Failed to create BSplinePriceTable");
        }

        return new EEPFixture{std::move(*wrapper), table->surface, K_ref, q, OptionType::PUT};
    }();

    return *fixture;
}

// ---------------------------------------------------------------------------
// Error accumulator
// ---------------------------------------------------------------------------

struct GreekErrors {
    double max_err = 0.0;
    double sum_err = 0.0;
    double max_rel_pct = 0.0;  // max relative error in percent
    int count = 0;

    void record(double interp, double ref) {
        double err = std::abs(interp - ref);
        max_err = std::max(max_err, err);
        sum_err += err;
        if (std::abs(ref) > 1e-6) {
            max_rel_pct = std::max(max_rel_pct, err / std::abs(ref) * 100.0);
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
    // PDE solver gives delta/gamma from the spatial grid at tau=T.
    // For theta, AmericanOptionResult::theta() returns dV/dt near expiry (τ≈0),
    // NOT at the query tau. So we compute theta via finite differences on price:
    //   theta_fd = (P(tau+eps) - P(tau-eps)) / (2*eps)  [in tau space]
    struct RefGreeks {
        double price, delta, gamma, theta;
    };
    std::vector<RefGreeks> refs;
    refs.reserve(strikes.size() * maturities.size());

    auto solve_price = [&](double K, double tau) -> double {
        PricingParams p{OptionSpec{.spot = spot, .strike = K, .maturity = tau, .rate = rate, .dividend_yield = fix.dividend_yield, .option_type = OptionType::PUT}, sigma};
        auto [gs, td] = estimate_pde_grid(p);
        size_t n = gs.n_points();
        std::pmr::synchronized_pool_resource pool;
        std::pmr::vector<double> buf(PDEWorkspace::required_size(n), &pool);
        auto ws = PDEWorkspace::from_buffer(buf, n);
        if (!ws) throw std::runtime_error("Failed to create workspace");
        auto solver = AmericanOptionSolver::create(p, ws.value()).value();
        auto r = solver.solve();
        if (!r) throw std::runtime_error("PDE solver failed");
        return r->value_at(spot);
    };

    constexpr double theta_eps = 1.0 / 365.0;  // 1 day

    for (double K : strikes) {
        for (double tau : maturities) {
            PricingParams params{OptionSpec{.spot = spot, .strike = K, .maturity = tau, .rate = rate, .dividend_yield = fix.dividend_yield, .option_type = OptionType::PUT}, sigma};
            auto [grid_spec, time_domain] = estimate_pde_grid(params);
            size_t n = grid_spec.n_points();
            std::pmr::synchronized_pool_resource pool;
            std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);
            auto workspace = PDEWorkspace::from_buffer(buffer, n);
            if (!workspace) {
                throw std::runtime_error("Failed to create workspace");
            }
            auto solver = AmericanOptionSolver::create(params, workspace.value()).value();
            auto result = solver.solve();
            if (!result) {
                throw std::runtime_error("PDE solver failed");
            }

            // FD theta in calendar time: dV/dt = -(dP/d(tau))
            // Guard: tau - eps must stay positive
            double tau_lo = std::max(tau - theta_eps, theta_eps);
            double tau_hi = tau + theta_eps;
            double p_up = solve_price(K, tau_hi);
            double p_dn = solve_price(K, tau_lo);
            double fd_theta = -(p_up - p_dn) / (tau_hi - tau_lo);

            refs.push_back({
                result->value_at(spot),
                result->delta(),
                result->gamma(),
                fd_theta
            });
        }
    }

    // EEP Greek helpers: inline the EEP decomposition formulas.
    // The wrapper has price() but not delta/gamma/theta, so we compute
    // those from the B-spline partials + European Greeks directly.
    const auto& surf = *fix.surface;
    const double K_ref = fix.K_ref;
    const double q = fix.dividend_yield;
    const OptionType type = fix.type;

    auto make_european = [&](double S, double K, double tau_val,
                             double sig, double r) {
        return EuropeanOptionSolver(
            OptionSpec{.spot = S, .strike = K, .maturity = tau_val,
                       .rate = r, .dividend_yield = q,
                       .option_type = type}, sig).solve().value();
    };

    auto eep_delta = [&](double S, double K, double tau_val,
                         double sig, double r) -> double {
        double x = std::log(S / K);
        double dEdx = surf.partial(0, {x, tau_val, sig, r});
        double d = (K / (K_ref * S)) * dEdx;
        auto eu = make_european(S, K, tau_val, sig, r);
        return d + eu.delta();
    };

    auto eep_gamma = [&](double S, double K, double tau_val,
                         double sig, double r) -> double {
        double x = std::log(S / K);
        double dEdx = surf.partial(0, {x, tau_val, sig, r});
        double d2Edx2 = surf.second_partial(0, {x, tau_val, sig, r});
        double g = (K / K_ref) * (d2Edx2 - dEdx) / (S * S);
        auto eu = make_european(S, K, tau_val, sig, r);
        return g + eu.gamma();
    };

    auto eep_theta = [&](double S, double K, double tau_val,
                         double sig, double r) -> double {
        double x = std::log(S / K);
        double dtau = (K / K_ref) * surf.partial(1, {x, tau_val, sig, r});
        auto eu = make_european(S, K, tau_val, sig, r);
        return -dtau + eu.theta();
    };

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

                double i_price = fix.wrapper.price(spot, K, tau, sigma, rate);
                double i_delta = eep_delta(spot, K, tau, sigma, rate);
                double i_gamma = eep_gamma(spot, K, tau, sigma, rate);
                double i_theta = eep_theta(spot, K, tau, sigma, rate);

                benchmark::DoNotOptimize(i_price);
                benchmark::DoNotOptimize(i_delta);
                benchmark::DoNotOptimize(i_gamma);
                benchmark::DoNotOptimize(i_theta);

                price_err.record(i_price, ref.price);
                delta_err.record(i_delta, ref.delta);
                gamma_err.record(i_gamma, ref.gamma);
                theta_err.record(i_theta, ref.theta);
            }
        }
    }

    state.counters["n_combos"] = static_cast<double>(refs.size());
    state.counters["max_price_err"] = price_err.max_err;
    state.counters["mean_price_err"] = price_err.mean_err();
    state.counters["max_price_rel%"] = price_err.max_rel_pct;
    state.counters["max_delta_err"] = delta_err.max_err;
    state.counters["mean_delta_err"] = delta_err.mean_err();
    state.counters["max_delta_rel%"] = delta_err.max_rel_pct;
    state.counters["max_gamma_err"] = gamma_err.max_err;
    state.counters["mean_gamma_err"] = gamma_err.mean_err();
    state.counters["max_gamma_rel%"] = gamma_err.max_rel_pct;
    state.counters["max_theta_err"] = theta_err.max_err;
    state.counters["mean_theta_err"] = theta_err.mean_err();
    state.counters["max_theta_rel%"] = theta_err.max_rel_pct;
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
                               "Greek", "Max Error", "Mean Error", "Max Rel%");
            out << std::format("  {:>8s}  {:>12s}  {:>12s}  {:>10s}\n",
                               "--------", "------------", "------------", "----------");
            out << std::format("  {:>8s}  {:>12.6f}  {:>12.6f}  {:>10.2f}\n",
                               "Price", get("max_price_err"), get("mean_price_err"), get("max_price_rel%"));
            out << std::format("  {:>8s}  {:>12.6f}  {:>12.6f}  {:>10.2f}\n",
                               "Delta", get("max_delta_err"), get("mean_delta_err"), get("max_delta_rel%"));
            out << std::format("  {:>8s}  {:>12.6f}  {:>12.6f}  {:>10.2f}\n",
                               "Gamma", get("max_gamma_err"), get("mean_gamma_err"), get("max_gamma_rel%"));
            out << std::format("  {:>8s}  {:>12.6f}  {:>12.6f}  {:>10.2f}\n",
                               "Theta", get("max_theta_err"), get("mean_theta_err"), get("max_theta_rel%"));
            out << "\n";
            out << "  Rel% = |error| / |reference| * 100\n";
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
