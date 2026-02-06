// SPDX-License-Identifier: MIT
#pragma once
#include "iv_benchmark_common.hpp"
#include <ql/quantlib.hpp>
#include <vector>

namespace mango::bench {

namespace ql = QuantLib;

// Fixed evaluation date for QuantLib reproducibility
inline const ql::Date kEvalDate(1, ql::January, 2024);

// American put via QuantLib FD (vanilla, no discrete dividends)
inline double price_ql(double spot, double strike, double vol, double maturity,
                       double rate, double div_yield,
                       size_t grid_steps, size_t time_steps) {
    ql::Date today = kEvalDate;
    ql::Settings::instance().evaluationDate() = today;

    auto maturity_date = today + ql::Period(static_cast<int>(maturity * 365), ql::Days);

    auto exercise = ql::ext::make_shared<ql::AmericanExercise>(today, maturity_date);
    auto payoff = ql::ext::make_shared<ql::PlainVanillaPayoff>(ql::Option::Put, strike);
    ql::VanillaOption option(payoff, exercise);

    auto spot_h = ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(spot));
    auto rate_h = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, rate, ql::Actual365Fixed()));
    auto div_h = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, div_yield, ql::Actual365Fixed()));
    auto vol_h = ql::Handle<ql::BlackVolTermStructure>(
        ql::ext::make_shared<ql::BlackConstantVol>(today, ql::NullCalendar(), vol, ql::Actual365Fixed()));

    auto process = ql::ext::make_shared<ql::BlackScholesMertonProcess>(spot_h, div_h, rate_h, vol_h);

    option.setPricingEngine(
        ql::ext::make_shared<ql::FdBlackScholesVanillaEngine>(process, time_steps, grid_steps));

    return option.NPV();
}

// American put via QuantLib FD with discrete dividends
inline double price_ql_div(double spot, double strike, double vol, double maturity,
                           double rate, double div_yield,
                           const std::vector<Dividend>& divs,
                           size_t grid_steps, size_t time_steps) {
    ql::Date today = kEvalDate;
    ql::Settings::instance().evaluationDate() = today;

    auto maturity_date = today + ql::Period(static_cast<int>(maturity * 365), ql::Days);

    auto exercise = ql::ext::make_shared<ql::AmericanExercise>(today, maturity_date);
    auto payoff = ql::ext::make_shared<ql::PlainVanillaPayoff>(ql::Option::Put, strike);
    ql::VanillaOption option(payoff, exercise);

    auto spot_h = ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(spot));
    auto rate_h = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, rate, ql::Actual365Fixed()));
    auto div_h = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, div_yield, ql::Actual365Fixed()));
    auto vol_h = ql::Handle<ql::BlackVolTermStructure>(
        ql::ext::make_shared<ql::BlackConstantVol>(today, ql::NullCalendar(), vol, ql::Actual365Fixed()));

    auto process = ql::ext::make_shared<ql::BlackScholesMertonProcess>(spot_h, div_h, rate_h, vol_h);

    std::vector<ql::Date> div_dates;
    std::vector<ql::Real> div_amounts;
    for (const auto& d : divs) {
        div_dates.push_back(today + ql::Period(static_cast<int>(d.calendar_time * 365), ql::Days));
        div_amounts.push_back(d.amount);
    }

    option.setPricingEngine(
        ql::MakeFdBlackScholesVanillaEngine(process)
            .withTGrid(time_steps)
            .withXGrid(grid_steps)
            .withCashDividends(div_dates, div_amounts));

    return option.NPV();
}

}  // namespace mango::bench
