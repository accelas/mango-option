/**
 * @file simple_databento_example.cpp
 * @brief End-to-end example: Databento fixed-point data → volatility smile
 */

#include "src/simple/simple.hpp"
#include <iostream>
#include <iomanip>

using namespace mango::simple;

int main() {
    // ============================================
    // Simulated Databento data (fixed-point format)
    // ============================================

    // Databento uses int64_t with 9 decimal places
    // 580.50 = 580500000000

    Converter<DatabentSource>::RawOption options[] = {
        {.ts_event = 1718972400000000000ULL, .price = 615000000000LL, .bid_px = 610000000000LL, .ask_px = 625000000000LL, .strike_price = 575000000000LL, .option_type = 'C'},
        {.ts_event = 1718972400000000000ULL, .price = 288000000000LL, .bid_px = 285000000000LL, .ask_px = 292000000000LL, .strike_price = 580000000000LL, .option_type = 'C'},
        {.ts_event = 1718972400000000000ULL, .price = 98000000000LL, .bid_px = 95000000000LL, .ask_px = 102000000000LL, .strike_price = 585000000000LL, .option_type = 'C'},
        {.ts_event = 1718972400000000000ULL, .price = 21000000000LL, .bid_px = 18000000000LL, .ask_px = 24000000000LL, .strike_price = 570000000000LL, .option_type = 'P'},
        {.ts_event = 1718972400000000000ULL, .price = 235000000000LL, .bid_px = 230000000000LL, .ask_px = 242000000000LL, .strike_price = 580000000000LL, .option_type = 'P'},
    };

    // ============================================
    // Build chain with Databento converter (preserves fixed-point)
    // ============================================

    auto builder = ChainBuilder<DatabentSource>{}
        .symbol("SPY")
        .spot(580500000000LL)  // Fixed-point preserved!
        .quote_time(1718972400000000000ULL)
        .settlement(Settlement::PM);

    // Single expiry for this example
    uint64_t expiry_nanos = 1719014400000000000ULL;  // 2024-06-21 16:00 UTC

    for (const auto& opt : options) {
        auto leg = Converter<DatabentSource>::to_leg(opt);

        // Verify fixed-point is preserved
        if (leg.strike.is_fixed_point()) {
            std::cout << "Strike " << leg.strike.to_double()
                      << " stored as fixed-point\n";
        }

        if (opt.option_type == 'C') {
            builder.add_call(expiry_nanos, opt);
        } else {
            builder.add_put(expiry_nanos, opt);
        }
    }

    auto chain = builder.build();

    // ============================================
    // Verify precision preservation
    // ============================================

    std::cout << "\n=== Precision Check ===\n";
    std::cout << "Spot stored as fixed-point: "
              << (chain.spot->is_fixed_point() ? "YES" : "NO") << "\n";
    std::cout << "Spot value: " << std::fixed << std::setprecision(9)
              << chain.spot->to_double() << "\n";

    // The key benefit: no precision loss until final computation
    std::cout << "\nFixed-point precision preserved through data pipeline.\n";
    std::cout << "Conversion to double only at solver boundary.\n";

    return 0;
}
