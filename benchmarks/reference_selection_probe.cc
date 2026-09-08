// SPDX-License-Identifier: MIT
#include "mango/option/table/reference_strike_evaluator.hpp"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    std::vector<size_t> counts;
    for (int i = 1; i < argc; ++i) {
        const size_t count = std::strtoull(argv[i], nullptr, 10);
        if (count < 2 || count > 65) return 2;
        counts.push_back(count);
    }
    if (counts.empty()) counts.push_back(2);
    mango::SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = mango::OptionType::PUT,
        .dividend_yield = 0.02, .discrete_dividends = {{0.5, 3.0}},
        .maturity = 1.0, .strike_bounds = mango::StrikeBounds{90.0, 110.0}};
    mango::SurfaceBounds bounds{
        .m_min = std::log(0.92), .m_max = std::log(1.08),
        .tau_min = 0.0, .tau_max = 1.0, .sigma_min = 0.05, .sigma_max = 0.1,
        .rate_min = 0.05, .rate_max = 0.05,
        .strike_bounds = mango::StrikeBounds{90.0, 110.0},
        .ratio_bounds = mango::MoneynessBounds{0.92, 1.08}};
    auto evaluator = mango::ReferenceStrikeEvaluator::create(config, bounds, {{0.0, 0.4995}, {0.5005, 1.0}});
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
    for (size_t count : counts) {
    std::vector<double> refs;
    for (size_t i = 0; i < count; ++i)
        refs.push_back(std::lerp(90.0, 110.0, static_cast<double>(i) / (count - 1)));
    auto result = evaluator->evaluate(refs);
    if (!result) return 4;
    std::cout << std::setprecision(17) << "refs=" << count
        << " decision=" << static_cast<int>(result->decision)
        << " solves=" << result->pde_solves << " seconds=" << result->elapsed_seconds << '\n';
        print("ideal_price", result->ideal_blend.price);
        print("ideal_iv", result->ideal_blend.iv);
        std::cout.flush();
    }

}
