/**
 * @file simple_yfinance_example.cpp
 * @brief End-to-end example: yfinance data → volatility smile
 */

#include "src/simple/simple.hpp"
#include <iostream>
#include <iomanip>
#include <ranges>

using namespace mango::simple;

int main() {
    // ============================================
    // Step 1: Simulated yfinance data
    // ============================================

    // In practice, this comes from Python via pybind11
    Converter<YFinanceSource>::RawOption spy_calls[] = {
        {.expiry = "2024-06-21", .strike = 575.0, .bid = 6.10, .ask = 6.25, .lastPrice = 6.15, .volume = 15420, .openInterest = 28300, .impliedVolatility = 0.142},
        {.expiry = "2024-06-21", .strike = 580.0, .bid = 2.85, .ask = 2.92, .lastPrice = 2.88, .volume = 42150, .openInterest = 51200, .impliedVolatility = 0.128},
        {.expiry = "2024-06-21", .strike = 585.0, .bid = 0.95, .ask = 1.02, .lastPrice = 0.98, .volume = 31200, .openInterest = 39100, .impliedVolatility = 0.135},
        {.expiry = "2024-06-21", .strike = 590.0, .bid = 0.22, .ask = 0.28, .lastPrice = 0.25, .volume = 18900, .openInterest = 22400, .impliedVolatility = 0.148},
    };

    Converter<YFinanceSource>::RawOption spy_puts[] = {
        {.expiry = "2024-06-21", .strike = 570.0, .bid = 0.18, .ask = 0.24, .lastPrice = 0.21, .volume = 12300, .openInterest = 18700, .impliedVolatility = 0.152},
        {.expiry = "2024-06-21", .strike = 575.0, .bid = 0.52, .ask = 0.58, .lastPrice = 0.55, .volume = 28400, .openInterest = 35600, .impliedVolatility = 0.138},
        {.expiry = "2024-06-21", .strike = 580.0, .bid = 2.30, .ask = 2.42, .lastPrice = 2.35, .volume = 38700, .openInterest = 48200, .impliedVolatility = 0.126},
        {.expiry = "2024-06-21", .strike = 585.0, .bid = 5.40, .ask = 5.55, .lastPrice = 5.48, .volume = 21500, .openInterest = 31400, .impliedVolatility = 0.132},
    };

    // ============================================
    // Step 2: Build option chain using type-safe builder
    // ============================================

    auto builder = ChainBuilder<YFinanceSource>{}
        .symbol("SPY")
        .spot(580.50)
        .quote_time("2024-06-21T10:30:00")
        .settlement(Settlement::PM)
        .dividend_yield(0.013);

    for (const auto& call : spy_calls) {
        builder.add_call(call.expiry, call);
    }
    for (const auto& put : spy_puts) {
        builder.add_put(put.expiry, put);
    }

    auto chain = builder.build();

    // ============================================
    // Step 3: Set up market context
    // ============================================

    MarketContext ctx;
    ctx.rate = 0.053;  // 5.3% Fed Funds
    ctx.valuation_time = Timestamp{"2024-06-21T10:30:00"};

    // ============================================
    // Step 4: Display chain structure
    // ============================================

    // Note: For actual IV computation, you'd need to provide an IVSolverInterpolated
    // with a precomputed price table. Here we just demonstrate the data flow from
    // yfinance format through the mango::simple API.

    std::cout << "# SPY Option Chain\n";
    std::cout << "# Symbol: " << chain.symbol << "\n";
    std::cout << "# Spot: " << chain.spot->to_double() << "\n";
    std::cout << "# Quote Time: " << chain.quote_time->to_string() << "\n";
    std::cout << "# Dividend Yield: " << std::get<double>(*chain.dividends) << "\n";
    std::cout << "# Rate: " << std::get<double>(*ctx.rate) << "\n\n";

    // Display chain data organized by expiry
    for (const auto& expiry : chain.expiries) {
        double tau = compute_tau(*ctx.valuation_time, expiry.expiry);
        std::cout << "## Expiry: " << expiry.expiry.to_string()
                  << " (tau=" << std::fixed << std::setprecision(6)
                  << tau << " years)\n";
        std::cout << "## Settlement: "
                  << (*expiry.settlement == Settlement::PM ? "PM" : "AM") << "\n\n";

        std::cout << "### Calls (" << std::ranges::distance(expiry.calls()) << " strikes)\n";
        std::cout << "strike,bid,ask,last,volume,oi,moneyness\n";
        for (const auto& call : expiry.calls()) {
            double strike = call.strike.to_double();
            double moneyness = std::log(strike / chain.spot->to_double());
            std::cout << std::setprecision(2) << std::fixed << strike << ","
                      << call.bid->to_double() << ","
                      << call.ask->to_double() << ","
                      << call.last->to_double() << ","
                      << *call.volume << ","
                      << *call.open_interest << ","
                      << std::setprecision(6) << moneyness << "\n";
        }

        std::cout << "\n### Puts (" << std::ranges::distance(expiry.puts()) << " strikes)\n";
        std::cout << "strike,bid,ask,last,volume,oi,moneyness\n";
        for (const auto& put : expiry.puts()) {
            double strike = put.strike.to_double();
            double moneyness = std::log(strike / chain.spot->to_double());
            std::cout << std::setprecision(2) << std::fixed << strike << ","
                      << put.bid->to_double() << ","
                      << put.ask->to_double() << ","
                      << put.last->to_double() << ","
                      << *put.volume << ","
                      << *put.open_interest << ","
                      << std::setprecision(6) << moneyness << "\n";
        }
    }

    std::cout << "\nChain built successfully!\n";
    std::cout << "- Expiries: " << chain.expiries.size() << "\n";
    std::cout << "- Total calls: " << std::ranges::distance(chain.expiries[0].calls()) << "\n";
    std::cout << "- Total puts: " << std::ranges::distance(chain.expiries[0].puts()) << "\n";
    std::cout << "\nTo compute implied volatilities, provide an IVSolverInterpolated\n";
    std::cout << "with a precomputed price table to compute_vol_surface().\n";

    return 0;
}
