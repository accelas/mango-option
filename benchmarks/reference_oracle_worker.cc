// SPDX-License-Identifier: MIT
// A persistent reference worker. No interpolation-table implementation is linked.
#include "mango/option/american_option.hpp"
#include "mango/option/grid_spec_types.hpp"
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/version.hpp>
#include <ql/quantlib.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;

std::string quote(const std::string& value) {
    std::string out = "\"";
    for (char c : value) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c;
        }
    }
    return out + '"';
}

template<class R>
std::array<R, 6> analytic_european(const mango::PricingParams& p) {
    const R S(p.spot), K(p.strike), T(p.maturity), sigma(p.volatility);
    const R r(std::get<double>(p.rate)), q(p.dividend_yield);
    const R root_t = sqrt(T), discount = exp(-r * T), yield_discount = exp(-q * T);
    const R d1 = (log(S / K) + (r - q + sigma * sigma / 2) * T) / (sigma * root_t);
    const R d2 = d1 - sigma * root_t;
    const R sqrt_two = sqrt(R(2));
    const R n1 = boost::math::erfc(-d1 / sqrt_two) / 2;
    const R n2 = boost::math::erfc(-d2 / sqrt_two) / 2;
    const R density = exp(-d1 * d1 / 2) / sqrt(2 * boost::math::constants::pi<R>());
    const R gamma = yield_discount * density / (S * sigma * root_t);
    const R vega = S * yield_discount * root_t * density;
    const R diffusion_theta = -S * yield_discount * density * sigma / (2 * root_t);
    if (p.option_type == mango::OptionType::CALL) {
        return {S * yield_discount * n1 - K * discount * n2, yield_discount * n1,
                gamma, vega, diffusion_theta + q * S * yield_discount * n1 - r * K * discount * n2,
                K * T * discount * n2};
    }
    // Evaluate negative tails directly; 1-Phi(d) loses deep-tail precision.
    const R put_n1 = boost::math::erfc(d1 / sqrt_two) / 2;
    const R put_n2 = boost::math::erfc(d2 / sqrt_two) / 2;
    return {K * discount * put_n2 - S * yield_discount * put_n1, -yield_discount * put_n1,
            gamma, vega, diffusion_theta - q * S * yield_discount * put_n1 + r * K * discount * put_n2,
            -K * T * discount * put_n2};
}

long aligned_days(double years) {
    const double days = years * 365.0;
    const long n = std::lround(days);
    const double tol = 32 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(days));
    if (n <= 0 || std::abs(days - static_cast<double>(n)) > tol)
        throw std::invalid_argument("quantlib_requires_integer_days");
    return n;
}

double quantlib_price(const mango::PricingParams& p, size_t nx, size_t nt) {
    // Cash model equivalence requires separate qualification. Never silently
    // substitute a vanilla engine for a dividend-bearing reference.
    if (!p.discrete_dividends.empty())
        throw std::invalid_argument("quantlib_cash_model_not_qualified");
    namespace ql = QuantLib;
    const ql::Date today(1, ql::January, 2024);
    ql::Settings::instance().evaluationDate() = today;
    const ql::Date expiry = today + static_cast<int>(aligned_days(p.maturity));
    const auto payoff = ql::ext::make_shared<ql::PlainVanillaPayoff>(
        p.option_type == mango::OptionType::CALL ? ql::Option::Call : ql::Option::Put, p.strike);
    ql::VanillaOption option(payoff, ql::ext::make_shared<ql::AmericanExercise>(today, expiry));
    const auto spot = ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(p.spot));
    const auto rates = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, std::get<double>(p.rate), ql::Actual365Fixed()));
    const auto yield = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, p.dividend_yield, ql::Actual365Fixed()));
    const auto vol = ql::Handle<ql::BlackVolTermStructure>(
        ql::ext::make_shared<ql::BlackConstantVol>(today, ql::NullCalendar(), p.volatility, ql::Actual365Fixed()));
    const auto process = ql::ext::make_shared<ql::BlackScholesMertonProcess>(spot, yield, rates, vol);
    option.setPricingEngine(ql::ext::make_shared<ql::FdBlackScholesVanillaEngine>(
        process, nt, nx, 2, ql::FdmSchemeDesc::CrankNicolson()));
    return option.NPV();
}

void run(const std::string& line) {
    // provider S K tau sigma rate q type grid_kind nx nt radius alpha n_div [time amount]...
    std::istringstream in(line);
    std::string provider, grid_kind;
    mango::PricingParams p;
    double rate, radius, alpha;
    int type;
    size_t nx, nt, ndiv;
    if (!(in >> provider >> p.spot >> p.strike >> p.maturity >> p.volatility
          >> rate >> p.dividend_yield >> type >> grid_kind >> nx >> nt >> radius >> alpha >> ndiv)
        || (type != 0 && type != 1) || ndiv > 100)
        throw std::invalid_argument("malformed_reference_request");
    p.rate = rate;
    p.option_type = type == 0 ? mango::OptionType::CALL : mango::OptionType::PUT;
    for (size_t i = 0; i < ndiv; ++i) {
        mango::Dividend d;
        if (!(in >> d.calendar_time >> d.amount)) throw std::invalid_argument("malformed_dividend");
        p.discrete_dividends.push_back(d);
    }
    std::string extra;
    if (in >> extra) throw std::invalid_argument("trailing_reference_input");
    const auto valid = mango::validate_pricing_params(p);
    if (!valid) throw std::invalid_argument("invalid_option_spec_" + std::to_string(int(valid.error().code)));
    const auto start = Clock::now();
    if (provider == "analytic") {
        const bool call_identity = type == 0 && rate >= 0 && p.dividend_yield == 0;
        // Healy (2021), Proposition 2: no early exercise when r<=0 and r<=q.
        // This applies only without future cash dividends.
        const bool put_identity = type == 1 && rate <= 0 && rate <= p.dividend_yield;
        if ((!call_identity && !put_identity) || !p.discrete_dividends.empty())
            throw std::invalid_argument("analytic_regime_not_applicable");
        using Low = boost::multiprecision::cpp_dec_float_50;
        using High = boost::multiprecision::cpp_dec_float_100;
        const auto low = analytic_european<Low>(p);
        const auto high = analytic_european<High>(p);
        const std::array<const char*, 6> names = {"price", "delta", "gamma", "vega", "theta", "rho"};
        std::cout << "{\"ok\":true,\"provider\":\"analytic-bsm-50-100\"";
        for (size_t i = 0; i < names.size(); ++i) {
            const double value = static_cast<double>(high[i]);
            const High disagreement = abs(High(low[i]) - high[i]) + abs(High(value) - high[i]);
            const double error = static_cast<double>(disagreement)
                + 64 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(value));
            std::cout << ",\"" << names[i] << "\":" << value
                      << ",\"" << names[i] << "_error\":" << error;
        }
        std::cout << ",\"seconds\":" << std::chrono::duration<double>(Clock::now() - start).count() << "}\n";
        return;
    }
    if (provider == "ql") {
        const double value = quantlib_price(p, nx, nt);
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite_quantlib_result");
        std::cout << "{\"ok\":true,\"provider\":\"quantlib-vanilla-date-aligned\",\"price\":" << value
                  << ",\"nx\":" << nx << ",\"nt\":" << nt
                  << ",\"seconds\":" << std::chrono::duration<double>(Clock::now() - start).count() << "}\n";
        return;
    }
    mango::PDEGridSpec grid;
    if (grid_kind == "H" || grid_kind == "U") {
        grid = mango::make_grid_accuracy(grid_kind == "H"
            ? mango::GridAccuracyProfile::High : mango::GridAccuracyProfile::Ultra);
    } else if (grid_kind == "S") {
        if (nt == 0 || !std::isfinite(radius) || radius <= std::abs(std::log(p.spot / p.strike)))
            throw std::invalid_argument("invalid_controlled_domain");
        auto spec = mango::GridSpec<double>::sinh_spaced(-radius, radius, nx, alpha);
        if (!spec) throw std::invalid_argument("invalid_controlled_grid");
        if (provider == "grid") {
            auto xs = spec->generate();
            std::cout << "{\"ok\":true,\"nodes\":[";
            for (size_t i = 0; i < xs.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << xs.span()[i];
            }
            std::cout << "]}\n";
            return;
        }
        grid = mango::PDEGridConfig{*spec, nt, {}};
    } else throw std::invalid_argument("unknown_grid_kind");
    if (provider != "fd") throw std::invalid_argument("unknown_reference_provider");
    auto solver = mango::AmericanOptionSolver::create(p, grid);
    if (!solver) throw std::runtime_error("fd_create_" + std::to_string(int(solver.error().code)));
    auto result = solver->solve();
    if (!result) throw std::runtime_error("fd_solve_" + std::to_string(int(result.error().code)));
    const double value = result->value();
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite_fd_result");
    const auto g = result->grid();
    std::cout << "{\"ok\":true,\"provider\":\"direct-fde\",\"price\":" << value
              << ",\"nx\":" << g->n_space() << ",\"nt\":" << g->time().n_steps()
              << ",\"lo\":" << g->x().front() << ",\"hi\":" << g->x().back()
              << ",\"work\":" << double(g->n_space()) * double(g->time().n_steps())
              << ",\"seconds\":" << std::chrono::duration<double>(Clock::now() - start).count() << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
    std::cout << std::setprecision(17);
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "{\"protocol\":\"reference-v1\",\"boost\":" << quote(BOOST_LIB_VERSION)
                  << ",\"quantlib\":" << quote(QL_VERSION)
                  << ",\"compiler\":" << quote(__VERSION__)
#ifdef __OPTIMIZE__
                  << ",\"optimized\":true}\n";
#else
                  << ",\"optimized\":false}\n";
#endif
        return 0;
    }
    for (std::string line; std::getline(std::cin, line);) {
        try { run(line); }
        catch (const std::exception& error) {
            std::cout << "{\"ok\":false,\"error\":" << quote(error.what()) << "}\n";
        }
        std::cout.flush();
    }
}
