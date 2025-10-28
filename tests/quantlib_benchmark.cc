#include <benchmark/benchmark.h>
#include <cmath>
#include <iostream>

extern "C" {
#include "../src/american_option.h"
#include "../src/implied_volatility.h"
}

// QuantLib includes
#include <ql/quantlib.hpp>

using namespace QuantLib;

// Helper function to create QuantLib flat rate term structure
ext::shared_ptr<YieldTermStructure> flatRate(const Date& today,
                                               const ext::shared_ptr<Quote>& forward,
                                               const DayCounter& dc) {
    return ext::shared_ptr<YieldTermStructure>(
        new FlatForward(today, Handle<Quote>(forward), dc));
}

// Helper function to create QuantLib flat volatility term structure
ext::shared_ptr<BlackVolTermStructure> flatVol(const Date& today,
                                                 const ext::shared_ptr<Quote>& vol,
                                                 const DayCounter& dc) {
    return ext::shared_ptr<BlackVolTermStructure>(
        new BlackConstantVol(today, NullCalendar(), Handle<Quote>(vol), dc));
}

// Benchmark: Price American put option with our implementation
static void BM_IVCalc_AmericanPut(benchmark::State& state) {
    const double spot = 100.0;
    const double strike = 100.0;
    const double volatility = 0.25;
    const double risk_free_rate = 0.05;
    const double time_to_maturity = 1.0;

    OptionData option = {
        .strike = strike,
        .volatility = volatility,
        .risk_free_rate = risk_free_rate,
        .time_to_maturity = time_to_maturity,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 141,
        .dt = 0.001,
        .n_steps = 1000
    };

    for (auto _ : state) {
        AmericanOptionResult result = american_option_price(&option, &grid);
        double value = american_option_get_value_at_spot(result.solver, spot, strike);
        benchmark::DoNotOptimize(value);
        pde_solver_destroy(result.solver);
    }
}
BENCHMARK(BM_IVCalc_AmericanPut);

// Benchmark: Price American put option with QuantLib
static void BM_QuantLib_AmericanPut(benchmark::State& state) {
    const double spot_value = 100.0;
    const double strike = 100.0;
    const double volatility_value = 0.25;
    const double risk_free_rate = 0.05;
    const double dividend_yield = 0.0;

    Date today = Date::todaysDate();
    Settings::instance().evaluationDate() = today;
    DayCounter dc = Actual365Fixed();

    ext::shared_ptr<SimpleQuote> spot(new SimpleQuote(spot_value));
    ext::shared_ptr<SimpleQuote> qRate(new SimpleQuote(dividend_yield));
    ext::shared_ptr<SimpleQuote> rRate(new SimpleQuote(risk_free_rate));
    ext::shared_ptr<SimpleQuote> vol(new SimpleQuote(volatility_value));

    ext::shared_ptr<YieldTermStructure> qTS = flatRate(today, qRate, dc);
    ext::shared_ptr<YieldTermStructure> rTS = flatRate(today, rRate, dc);
    ext::shared_ptr<BlackVolTermStructure> volTS = flatVol(today, vol, dc);

    ext::shared_ptr<BlackScholesMertonProcess> stochProcess(
        new BlackScholesMertonProcess(
            Handle<Quote>(spot),
            Handle<YieldTermStructure>(qTS),
            Handle<YieldTermStructure>(rTS),
            Handle<BlackVolTermStructure>(volTS)
        )
    );

    ext::shared_ptr<StrikedTypePayoff> payoff(
        new PlainVanillaPayoff(Option::Put, strike)
    );

    Date maturityDate = today + Period(1, Years);
    ext::shared_ptr<Exercise> exercise(
        new AmericanExercise(today, maturityDate)
    );

    VanillaOption option(payoff, exercise);

    for (auto _ : state) {
        // Create fresh engine for each iteration
        // Using 1000 time steps to match our implementation (fair comparison)
        ext::shared_ptr<PricingEngine> engine(
            new FdBlackScholesVanillaEngine(stochProcess, 1000, 400)
        );
        option.setPricingEngine(engine);
        Real value = option.NPV();
        benchmark::DoNotOptimize(value);
    }
}
BENCHMARK(BM_QuantLib_AmericanPut);

// Benchmark: Price American call option with our implementation
static void BM_IVCalc_AmericanCall(benchmark::State& state) {
    const double spot = 100.0;
    const double strike = 100.0;
    const double volatility = 0.25;
    const double risk_free_rate = 0.05;
    const double time_to_maturity = 1.0;

    OptionData option = {
        .strike = strike,
        .volatility = volatility,
        .risk_free_rate = risk_free_rate,
        .time_to_maturity = time_to_maturity,
        .option_type = OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 141,
        .dt = 0.001,
        .n_steps = 1000
    };

    for (auto _ : state) {
        AmericanOptionResult result = american_option_price(&option, &grid);
        double value = american_option_get_value_at_spot(result.solver, spot, strike);
        benchmark::DoNotOptimize(value);
        pde_solver_destroy(result.solver);
    }
}
BENCHMARK(BM_IVCalc_AmericanCall);

// Benchmark: Price American call option with QuantLib
static void BM_QuantLib_AmericanCall(benchmark::State& state) {
    const double spot_value = 100.0;
    const double strike = 100.0;
    const double volatility_value = 0.25;
    const double risk_free_rate = 0.05;
    const double dividend_yield = 0.0;

    Date today = Date::todaysDate();
    Settings::instance().evaluationDate() = today;
    DayCounter dc = Actual365Fixed();

    ext::shared_ptr<SimpleQuote> spot(new SimpleQuote(spot_value));
    ext::shared_ptr<SimpleQuote> qRate(new SimpleQuote(dividend_yield));
    ext::shared_ptr<SimpleQuote> rRate(new SimpleQuote(risk_free_rate));
    ext::shared_ptr<SimpleQuote> vol(new SimpleQuote(volatility_value));

    ext::shared_ptr<YieldTermStructure> qTS = flatRate(today, qRate, dc);
    ext::shared_ptr<YieldTermStructure> rTS = flatRate(today, rRate, dc);
    ext::shared_ptr<BlackVolTermStructure> volTS = flatVol(today, vol, dc);

    ext::shared_ptr<BlackScholesMertonProcess> stochProcess(
        new BlackScholesMertonProcess(
            Handle<Quote>(spot),
            Handle<YieldTermStructure>(qTS),
            Handle<YieldTermStructure>(rTS),
            Handle<BlackVolTermStructure>(volTS)
        )
    );

    ext::shared_ptr<StrikedTypePayoff> payoff(
        new PlainVanillaPayoff(Option::Call, strike)
    );

    Date maturityDate = today + Period(1, Years);
    ext::shared_ptr<Exercise> exercise(
        new AmericanExercise(today, maturityDate)
    );

    VanillaOption option(payoff, exercise);

    for (auto _ : state) {
        // Using 1000 time steps to match our implementation (fair comparison)
        ext::shared_ptr<PricingEngine> engine(
            new FdBlackScholesVanillaEngine(stochProcess, 1000, 400)
        );
        option.setPricingEngine(engine);
        Real value = option.NPV();
        benchmark::DoNotOptimize(value);
    }
}
BENCHMARK(BM_QuantLib_AmericanCall);

// Comparison test: Verify our implementation matches QuantLib
static void CompareImplementations() {
    const double spot_value = 100.0;
    const double strike = 100.0;
    const double volatility_value = 0.25;
    const double risk_free_rate = 0.05;
    const double dividend_yield = 0.0;
    const double time_to_maturity = 1.0;

    // Our implementation
    OptionData option = {
        .strike = strike,
        .volatility = volatility_value,
        .risk_free_rate = risk_free_rate,
        .time_to_maturity = time_to_maturity,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 141,
        .dt = 0.001,
        .n_steps = 1000
    };

    AmericanOptionResult result = american_option_price(&option, &grid);
    double our_value = american_option_get_value_at_spot(result.solver, spot_value, strike);

    // QuantLib implementation
    Date today = Date::todaysDate();
    Settings::instance().evaluationDate() = today;
    DayCounter dc = Actual365Fixed();

    ext::shared_ptr<SimpleQuote> spot(new SimpleQuote(spot_value));
    ext::shared_ptr<SimpleQuote> qRate(new SimpleQuote(dividend_yield));
    ext::shared_ptr<SimpleQuote> rRate(new SimpleQuote(risk_free_rate));
    ext::shared_ptr<SimpleQuote> vol(new SimpleQuote(volatility_value));

    ext::shared_ptr<YieldTermStructure> qTS = flatRate(today, qRate, dc);
    ext::shared_ptr<YieldTermStructure> rTS = flatRate(today, rRate, dc);
    ext::shared_ptr<BlackVolTermStructure> volTS = flatVol(today, vol, dc);

    ext::shared_ptr<BlackScholesMertonProcess> stochProcess(
        new BlackScholesMertonProcess(
            Handle<Quote>(spot),
            Handle<YieldTermStructure>(qTS),
            Handle<YieldTermStructure>(rTS),
            Handle<BlackVolTermStructure>(volTS)
        )
    );

    ext::shared_ptr<StrikedTypePayoff> payoff(
        new PlainVanillaPayoff(Option::Put, strike)
    );

    Date maturityDate = today + Period(1, Years);
    ext::shared_ptr<Exercise> exercise(
        new AmericanExercise(today, maturityDate)
    );

    VanillaOption ql_option(payoff, exercise);
    // Using 1000 time steps to match our implementation for fair comparison
    ext::shared_ptr<PricingEngine> engine(
        new FdBlackScholesVanillaEngine(stochProcess, 1000, 400)
    );
    ql_option.setPricingEngine(engine);
    Real ql_value = ql_option.NPV();

    std::cout << "\n=== Implementation Comparison ===\n";
    std::cout << "Parameters:\n";
    std::cout << "  Spot: " << spot_value << "\n";
    std::cout << "  Strike: " << strike << "\n";
    std::cout << "  Volatility: " << volatility_value << "\n";
    std::cout << "  Risk-free rate: " << risk_free_rate << "\n";
    std::cout << "  Time to maturity: " << time_to_maturity << " years\n";
    std::cout << "\nResults:\n";
    std::cout << "  IV Calc value: " << our_value << "\n";
    std::cout << "  QuantLib value: " << ql_value << "\n";
    std::cout << "  Difference: " << std::abs(our_value - ql_value) << "\n";
    std::cout << "  Relative error: " << std::abs((our_value - ql_value) / ql_value * 100.0) << "%\n";
    std::cout << "================================\n\n";

    // Cleanup
    pde_solver_destroy(result.solver);
}

int main(int argc, char** argv) {
    // Run comparison first
    CompareImplementations();

    // Run benchmarks
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
