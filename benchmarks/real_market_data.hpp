// Auto-generated real market data for benchmarks
// Generated: 2025-11-27T12:07:03.758086
// Symbol: SPY
// DO NOT EDIT - regenerate with: python scripts/download_benchmark_data.py SPY

#pragma once

#include <array>
#include <cstddef>

namespace mango::benchmark_data {

// Market snapshot
constexpr const char* SYMBOL = "SPY";
constexpr double SPOT = 679.68;
constexpr double RISK_FREE_RATE = 0.0400;
constexpr double DIVIDEND_YIELD = 0.0109;

// Option data structure
struct RealOptionData {
    double strike;
    double maturity;
    double market_price;
    bool is_call;
};

// Real put options for batch benchmarks (64 options)
constexpr std::array<RealOptionData, 64> REAL_PUTS = {{
    {595.00, 0.019178, 0.0650, false},
    {640.00, 0.019178, 0.2550, false},
    {649.00, 0.019178, 0.4100, false},
    {658.00, 0.019178, 0.7400, false},
    {667.00, 0.019178, 1.4900, false},
    {676.00, 0.019178, 3.1750, false},
    {685.00, 0.019178, 6.9900, false},
    {694.00, 0.019178, 13.9150, false},
    {703.00, 0.019178, 22.8650, false},
    {725.00, 0.019178, 44.8650, false},
    {610.00, 0.027397, 0.1350, false},
    {631.00, 0.027397, 0.2550, false},
    {640.00, 0.027397, 0.3750, false},
    {649.00, 0.027397, 0.5900, false},
    {658.00, 0.027397, 1.0200, false},
    {667.00, 0.027397, 1.8950, false},
    {595.00, 0.019178, 0.0650, false},
    {642.00, 0.019178, 0.2850, false},
    {653.00, 0.019178, 0.5250, false},
    {664.00, 0.019178, 1.1700, false},
    {675.00, 0.019178, 2.9150, false},
    {686.00, 0.019178, 7.5750, false},
    {697.00, 0.019178, 16.8650, false},
    {708.00, 0.019178, 27.8650, false},
    {600.00, 0.027397, 0.1050, false},
    {631.00, 0.027397, 0.2550, false},
    {642.00, 0.027397, 0.4100, false},
    {653.00, 0.027397, 0.7400, false},
    {664.00, 0.027397, 1.5300, false},
    {675.00, 0.027397, 3.4250, false},
    {697.00, 0.027397, 16.8650, false},
    {645.00, 0.030137, 0.6100, false},
    {595.00, 0.027397, 0.0950, false},
    {630.00, 0.027397, 0.2450, false},
    {641.00, 0.027397, 0.3900, false},
    {652.00, 0.027397, 0.7000, false},
    {663.00, 0.027397, 1.4250, false},
    {674.00, 0.027397, 3.1750, false},
    {691.00, 0.027397, 11.4350, false},
    {640.00, 0.030137, 0.4800, false},
    {695.00, 0.030137, 15.0500, false},
    {641.00, 0.038356, 1.0500, false},
    {652.00, 0.038356, 1.7050, false},
    {663.00, 0.038356, 2.9050, false},
    {674.00, 0.038356, 5.1050, false},
    {685.00, 0.038356, 9.2750, false},
    {696.00, 0.038356, 16.5350, false},
    {710.00, 0.038356, 29.8650, false},
    {595.00, 0.030137, 0.1250, false},
    {625.00, 0.030137, 0.2750, false},
    {655.00, 0.030137, 1.0500, false},
    {685.00, 0.030137, 7.7850, false},
    {610.00, 0.038356, 0.3800, false},
    {640.00, 0.038356, 1.0000, false},
    {646.00, 0.038356, 1.3000, false},
    {652.00, 0.038356, 1.7050, false},
    {658.00, 0.038356, 2.2750, false},
    {664.00, 0.038356, 3.0550, false},
    {670.00, 0.038356, 4.1450, false},
    {676.00, 0.038356, 5.6750, false},
    {682.00, 0.038356, 7.8550, false},
    {688.00, 0.038356, 10.9350, false},
    {694.00, 0.038356, 14.9700, false},
    {700.00, 0.038356, 19.9650, false}
}}};

// Sample ATM put for single option benchmark
constexpr RealOptionData ATM_PUT = {675.00, 0.090411, 9.5400, false};

// Real call options (16 options)
constexpr std::array<RealOptionData, 16> REAL_CALLS = {{
    {595.00, 0.019178, 85.9200, true},
    {649.00, 0.019178, 32.3150, true},
    {667.00, 0.019178, 15.4300, true},
    {685.00, 0.019178, 2.8600, true},
    {703.00, 0.019178, 0.0950, true},
    {765.00, 0.019178, 0.0150, true},
    {664.00, 0.027397, 18.5550, true},
    {682.00, 0.027397, 4.8700, true},
    {595.00, 0.019178, 85.9200, true},
    {652.00, 0.019178, 29.4750, true},
    {673.00, 0.019178, 10.4000, true},
    {694.00, 0.019178, 0.5600, true},
    {735.00, 0.019178, 0.0150, true},
    {660.00, 0.027397, 22.1800, true},
    {682.00, 0.027397, 4.8700, true},
    {720.00, 0.027397, 0.0350, true}
}}};

}}  // namespace mango::benchmark_data
