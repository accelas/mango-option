// SPDX-License-Identifier: MIT
#include "mango/option/table/reference_strike_evaluator.hpp"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    const bool tracing = argc > 1 && std::string_view(argv[1]) == "ordinary-trace";
    const bool ordinary = tracing || (argc > 1 && std::string_view(argv[1]) == "ordinary");
    std::vector<size_t> counts;
    for (int i = ordinary ? 2 : 1; i < argc; ++i) {
        const size_t count = std::strtoull(argv[i], nullptr, 10);
        if (count < 2 || count > 65) return 2;
        counts.push_back(count);
    }
    if (counts.empty()) counts.push_back(2);
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
        {{0.0, 0.4995}, {0.5005, 1.0}}, .01, 2e-5, 1e-4, std::move(observer));
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
    std::cout << "case=" << (ordinary ? "ordinary-put" : "low-vol-put")
        << " sigma_min=" << bounds.sigma_min << " sigma_max=" << bounds.sigma_max
        << " cash=" << config.discrete_dividends.front().amount << '\n';
    for (size_t count : counts) {
    active_count = count;
    std::vector<double> refs;
    for (size_t i = 0; i < count; ++i)
        refs.push_back(std::lerp(90.0, 110.0, static_cast<double>(i) / (count - 1)));
    auto result = evaluator->evaluate(refs);
    if (!result) return 4;
    std::cout << std::setprecision(17) << "refs=" << count
        << " decision=" << static_cast<int>(result->decision)
        << " prefer_refinement=" << result->prefer_refinement
        << " solves=" << result->pde_solves << " seconds=" << result->elapsed_seconds << '\n';
        print("ideal_price", result->ideal_blend.price);
        print("ideal_iv", result->ideal_blend.iv);
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
