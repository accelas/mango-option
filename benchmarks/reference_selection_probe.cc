// SPDX-License-Identifier: MIT
#include "mango/option/table/reference_strike_evaluator.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "";
    const bool cache_cash = mode == "cache-ratio-cash";
    const bool cache_case = cache_cash || mode == "cache-ratio";
    if (cache_case && argc != 2) return 2;
    const bool tracing = mode == "ordinary-trace" || cache_case;
    const bool ordinary = mode == "ordinary-trace" || mode == "ordinary";
    std::vector<size_t> counts;
    for (int i = ordinary || cache_case ? 2 : 1; i < argc; ++i) {
        const size_t count = std::strtoull(argv[i], nullptr, 10);
        if (count < 2 || count > 65) return 2;
        counts.push_back(count);
    }
    if (counts.empty()) counts.push_back(cache_case && !cache_cash ? 1 : 2);
    mango::SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = mango::OptionType::PUT,
        .dividend_yield = 0.02, .discrete_dividends = {{0.5, ordinary ? 1.0 : 3.0}},
        .maturity = 1.0, .strike_bounds = mango::StrikeBounds{90.0, 110.0}};
    mango::SurfaceBounds bounds{
        .m_min = std::log(0.92), .m_max = std::log(1.08),
        .tau_min = 0.0, .tau_max = 1.0,
        .sigma_min = ordinary ? 0.2 : 0.05, .sigma_max = ordinary ? 0.3 : 0.1,
        .rate_min = 0.05, .rate_max = 0.05,
        .strike_bounds = mango::StrikeBounds{90.0, 110.0},
        .ratio_bounds = mango::MoneynessBounds{0.92, 1.08}};
    std::vector<std::pair<double, double>> times{{0.0, 0.4995}, {0.5005, 1.0}};
    if (cache_case) {
        config.spot = 97.1;
        config.dividend_yield = 0.0;
        config.discrete_dividends = cache_cash ? std::vector<mango::Dividend>{{0.5, 0.005}}
                                               : std::vector<mango::Dividend>{};
        config.kref_config.K_refs = cache_cash ? std::vector<double>{95.0, 100.0}
                                               : std::vector<double>{97.1};
        config.strike_bounds = mango::StrikeBounds{97.1, 97.1};
        bounds.m_min = bounds.m_max = std::log(0.8);
        bounds.sigma_min = bounds.sigma_max = 0.2;
        bounds.strike_bounds = config.strike_bounds;
        bounds.ratio_bounds = mango::MoneynessBounds{0.8, 0.8};
        if (!cache_cash) times = {{0.0, 1.0}};
    }
    const mango::SurfaceHandle fitted{.price = [](double spot, double strike,
        double, double, double) { return std::max(strike - spot, 0.0); }};
    size_t active_count = 0;
    mango::ReferenceSequenceObserver observer;
    if (tracing) observer = [&](const mango::ReferenceSequenceTrace& row) {
        std::cout << std::setprecision(17) << "sequence," << active_count << ',' << row.quantity
            << ',' << row.spot << ',' << row.strike << ',' << row.reference_strike
            << ',' << row.tau << ',' << row.sigma << ',' << row.rate << ',' << row.bump << ',' << row.round;
        for (const auto& axis : {row.space, row.time, row.domain})
            for (double value : axis) std::cout << ',' << value;
        std::cout << ',';
        if (row.direct_sequence_allowance) std::cout << *row.direct_sequence_allowance;
        std::cout << '\n';
    };
    auto evaluator = mango::ReferenceStrikeEvaluator::create(config, bounds,
        std::move(times), .01, 2e-5, 1e-4, std::move(observer));
    if (!evaluator) return 3;
    auto print = [](const char* name, const std::optional<mango::ReferenceErrorSummary>& summary) {
        if (!summary) return;
        const auto& x = *summary;
        std::cout << name << " requested=" << x.requested << " measured=" << x.measured
            << " filtered=" << x.filtered << " unresolved=" << x.unresolved << " refused=" << x.refused
            << " exact=" << x.structurally_exact << " untested=" << x.untested;
        if (x.max_error) std::cout << " max=" << *x.max_error;
        if (x.rms_error) std::cout << " rms=" << *x.rms_error;
        if (x.max_uncertainty) std::cout << " uncertainty=" << *x.max_uncertainty;
        std::cout << '\n';
    };
    std::cout << "case=" << (cache_case ? mode : ordinary ? "ordinary-put" : "low-vol-put")
        << " sigma_min=" << bounds.sigma_min << " sigma_max=" << bounds.sigma_max
        << " cash=" << (config.discrete_dividends.empty() ? 0.0 : config.discrete_dividends.front().amount) << '\n';
    for (size_t count : counts) {
        active_count = count;
        std::vector<double> refs = cache_case ? config.kref_config.K_refs : std::vector<double>{};
        if (!cache_case) {
            for (size_t i = 0; i < count; ++i)
                refs.push_back(std::lerp(90.0, 110.0, static_cast<double>(i) / (count - 1)));
        }
        auto result = evaluator->evaluate(refs, cache_case && !cache_cash ? &fitted : nullptr);
        if (!result) return 4;
        std::cout << std::setprecision(17) << "refs=" << count
            << " decision=" << static_cast<int>(result->decision)
            << " prefer_refinement=" << result->prefer_refinement
            << " solves=" << result->pde_solves << " seconds=" << result->elapsed_seconds << '\n';
        print("ideal_price", result->ideal_blend.price);
        print("ideal_iv", result->ideal_blend.iv);
        print("total_price", result->total.price);
        print("total_iv", result->total.iv);
        if (result->provider_work) {
            const auto& work = *result->provider_work;
            std::cout << "provider_work attempted=" << work.attempted
                << " completed=" << work.completed << " failed=" << work.failed << '\n';
        }
        if (result->qualification_paths) {
            const auto& paths = *result->qualification_paths;
            std::cout << "empirical_paths residual_direct=" << paths.residual_direct
                << " residual_component=" << paths.residual_component
                << " vega_direct=" << paths.vega_direct
                << " vega_component=" << paths.vega_component << '\n';
        }
        std::cout.flush();
    }

}
