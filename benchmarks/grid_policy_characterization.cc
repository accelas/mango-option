// SPDX-License-Identifier: MIT
// Fixed #487 characterization population; CSV includes every failed solve.
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/grid_spec_types.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <vector>

using namespace mango;

int batch_characterization() {
    std::puts("type,case,contracts,unique_solutions,nx,nt,mean_batch_us,max_price_error,status");
    for (auto type : {OptionType::PUT, OptionType::CALL}) {
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<PricingParams> contracts;
            for (int i = 0; i < 20; ++i) {
                contracts.emplace_back(OptionSpec{.spot = 100.0,
                    .strike = 90.0 + i, .maturity = 0.5, .rate = 0.05,
                    .option_type = type}, 0.2);
            }
            BatchAmericanOptionSolver solver;
            GridAccuracyParams accuracy;
            if (mode == 1) accuracy.log_moneyness_coverage = LogMoneynessRange{-3.0, 3.0};
            if (mode == 2) accuracy.min_spatial_points = accuracy.max_spatial_points = 3;
            solver.set_grid_accuracy(accuracy);
            std::optional<PDEGridSpec> grid;
            if (mode == 2) grid = PDEGridConfig{
                GridSpec<double>::uniform(-1.0, 1.0, 401).value(), 1000};
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 10; ++i) {
                auto result = solver.solve_batch(contracts, true, nullptr, grid);
                if (result.failed_count) return 1;
            }
            const double us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count() / 10.0;
            auto result = solver.solve_batch(contracts, true, nullptr, grid);
            std::set<const Grid<double>*> unique;
            for (const auto& r : result.results) if (r) unique.insert(r->grid().get());
            double err = 0.0;
            for (size_t i = 0; i < contracts.size(); ++i) {
                auto ref_solver = AmericanOptionSolver::create(contracts[i],
                    make_grid_accuracy(GridAccuracyProfile::High));
                auto ref = ref_solver->solve();
                if (!ref || !result.results[i]) return 1;
                err = std::max(err, std::abs(ref->value() - result.results[i]->value()));
            }
            const auto& first = result.results[0]->grid();
            std::printf("%s,%s,%zu,%zu,%zu,%zu,%.3f,%.9g,ok\n",
                type == OptionType::PUT ? "put" : "call",
                mode == 0 ? "automatic" : mode == 1 ? "wide_coverage" : "explicit_override",
                contracts.size(), unique.size(), first->n_space(),
                first->time().n_steps(), us, err);
        }
    }
    return 0;
}

int main(int argc, char**) {
    if (argc > 1) return batch_characterization();
    struct Case { const char* name; double sigma; double maturity; bool cash; };
    constexpr std::array cases{
        Case{"low_short", 0.01, 0.03, false},
        Case{"low_wide", 0.05, 0.5, false},
        Case{"moderate", 0.2, 0.5, false},
        Case{"wide_domain", 0.5, 2.0, false},
        Case{"cash_before", 0.1, 0.2501, true},
        Case{"cash_after", 0.1, 0.2499, true},
        Case{"cash_long", 0.2, 1.0, true}};
    constexpr std::array spots{36.787944117144235, 100.0, 271.8281828459045};
    std::puts("type,case,policy,width,nx,nt,cap_hit,solve_us,max_price_error,reference_change,status");
    for (auto type : {OptionType::PUT, OptionType::CALL}) {
        for (const auto& c : cases) {
            PricingParams p(OptionSpec{.spot = 100.0, .strike = 100.0,
                .maturity = c.maturity, .rate = 0.05,
                .dividend_yield = c.cash ? 0.02 : 0.0, .option_type = type}, c.sigma);
            // The after-event cohort has no remaining cash event. An expired
            // dividend is invalid input, not an event to silently retain.
            if (c.cash && c.maturity > 0.25) p.discrete_dividends = {{0.25, 1.5}};
            auto run = [&](const GridAccuracyParams& a)
                -> std::expected<AmericanOptionResult, SolverError> {
                auto solver = AmericanOptionSolver::create(p, a);
                if (!solver) return std::unexpected(SolverError{
                    .code = SolverErrorCode::InvalidConfiguration});
                return solver->solve();
            };
            GridAccuracyParams ref;
            ref.log_moneyness_coverage = LogMoneynessRange{-1.0, 1.0};
            ref.coverage_clearance_sigmas = 8.0;
            ref.min_spatial_points = ref.max_spatial_points = 2001;
            ref.max_time_steps = 8000;
            ref.c_t = 0.25;
            auto coarse_ref = run(ref);
            ref.min_spatial_points = ref.max_spatial_points = 4001;
            ref.max_time_steps = 16000;
            auto fine_ref = run(ref);
            double ref_change = 0;
            if (coarse_ref && fine_ref) {
                for (double spot : spots) ref_change = std::max(ref_change,
                    std::abs(coarse_ref->value_at(spot) - fine_ref->value_at(spot)));
            } else ref_change = std::numeric_limits<double>::quiet_NaN();
            for (int mode = 0; mode < 4; ++mode) {
                GridAccuracyParams a;
                a.log_moneyness_coverage = LogMoneynessRange{-1.0, 1.0};
                a.coverage_clearance_sigmas = mode == 0 ? 1.0 : mode == 2 ? 6.0 : 3.0;
                if (mode == 3) {
                    auto [g, td] = estimate_pde_grid(p, a);
                    const double reach = std::max(std::abs(g.x_min()), std::abs(g.x_max()));
                    a.alpha = optimal_sinh_alpha(reach / (p.volatility * std::sqrt(p.maturity)));
                }
                auto [g, td] = estimate_pde_grid(p, a);
                const auto begin = std::chrono::steady_clock::now();
                auto result = run(a);
                const double us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - begin).count();
                double err = 0.0;
                if (result && fine_ref) {
                    for (double spot : spots) err = std::max(err,
                        std::abs(result->value_at(spot) - fine_ref->value_at(spot)));
                } else err = std::numeric_limits<double>::quiet_NaN();
                std::printf("%s,%s,%s,%.9g,%zu,%zu,%d,%.3f,%.9g,%.9g,%s\n",
                    type == OptionType::PUT ? "put" : "call", c.name,
                    mode == 0 ? "clearance1" : mode == 1 ? "default" :
                        mode == 2 ? "clearance6" : "geometry_alpha",
                    g.x_max()-g.x_min(), g.n_points(), td.n_steps(),
                    g.n_points() >= a.max_spatial_points-1, us, err, ref_change,
                    result && fine_ref ? "ok" : "failed");
                std::fflush(stdout);
            }
        }
    }
}
