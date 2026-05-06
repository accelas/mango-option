// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "mango/option/price_table_factory.hpp"
#include "mango/option/table/serialization/price_table_data.hpp"

namespace mango {
namespace {

unsigned long current_process_id() {
#if defined(_WIN32)
    return static_cast<unsigned long>(_getpid());
#else
    return static_cast<unsigned long>(getpid());
#endif
}

class TempDirectory {
public:
    TempDirectory() {
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("mango_any_price_table_test_" + std::to_string(current_process_id()) + "_" +
                 std::to_string(timestamp));
        std::filesystem::create_directory(path_);
    }

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

IVSolverFactoryConfig bspline_4d_config() {
    IVSolverFactoryConfig config;
    config.option_type = OptionType::PUT;
    config.spot = 100.0;
    config.dividend_yield = 0.02;
    config.grid.moneyness = {0.8, 0.9, 1.0, 1.1, 1.2};
    config.grid.vol = {0.10, 0.20, 0.30, 0.40};
    config.grid.rate = {0.01, 0.03, 0.05, 0.07};
    BSplineBackend backend;
    backend.maturity_grid = {0.1, 0.25, 0.5, 1.0};
    config.backend = backend;
    return config;
}

PricingParams off_grid_pricing_params() {
    PricingParams p;
    p.spot = 100.0;
    p.strike = 97.0;      // S/K is between the 1.0 and 1.1 moneyness knots
    p.maturity = 0.37;    // between maturity knots
    p.volatility = 0.23;  // between volatility knots
    p.rate = 0.037;       // between rate knots
    p.dividend_yield = 0.02;
    p.option_type = OptionType::PUT;
    return p;
}

IVQuery off_grid_iv_query(const PricingParams& params, double market_price) {
    IVQuery query;
    query.spot = params.spot;
    query.strike = params.strike;
    query.maturity = params.maturity;
    query.rate = params.rate;
    query.dividend_yield = params.dividend_yield;
    query.option_type = params.option_type;
    query.market_price = market_price;
    return query;
}

IVQuery yield_curve_iv_query(const PricingParams& params, double market_price) {
    auto query = off_grid_iv_query(params, market_price);
    query.rate = YieldCurve::flat(0.037);
    return query;
}

TEST(PriceTableFactoryTest, BuildsContinuous4DBSplineAndEvaluatesOffGridPoint) {
    auto table_result = make_price_table(bspline_4d_config());
    ASSERT_TRUE(table_result.has_value()) << "make_price_table failed";

    auto table = std::move(*table_result);
    auto params = off_grid_pricing_params();

    EXPECT_EQ(table.surface_type(), surface_types::kBSpline4D);
    EXPECT_EQ(table.option_type(), OptionType::PUT);
    EXPECT_NEAR(table.dividend_yield(), 0.02, 1e-15);
    EXPECT_GT(table.price(params), 0.0);
    EXPECT_GT(table.vega(params), 0.0);
    EXPECT_TRUE(table.delta(params).has_value());
    EXPECT_TRUE(table.gamma(params).has_value());
    EXPECT_TRUE(table.theta(params).has_value());
    EXPECT_TRUE(table.rho(params).has_value());

    auto query = off_grid_iv_query(params, table.price(params));
    auto iv_direct = table.solve_iv(query);
    ASSERT_TRUE(iv_direct.has_value()) << "table.solve_iv failed";
    EXPECT_NEAR(iv_direct->implied_vol, params.volatility, 0.08);

    auto solver = table.make_iv_solver();
    ASSERT_TRUE(solver.has_value()) << "make_iv_solver failed";
    auto iv_via_solver = solver->solve(query);
    ASSERT_TRUE(iv_via_solver.has_value()) << "solver.solve failed";
    EXPECT_NEAR(iv_via_solver->implied_vol, params.volatility, 0.08);
}

TEST(PriceTableFactoryTest, ReportsInterpolatedIVApproximationAndVegaFailures) {
    auto table_result = make_price_table(bspline_4d_config());
    ASSERT_TRUE(table_result.has_value()) << "make_price_table failed";
    auto table = std::move(*table_result);

    auto params = off_grid_pricing_params();
    const double market_price = table.price(params);

    auto curve_iv = table.solve_iv(yield_curve_iv_query(params, market_price));
    ASSERT_TRUE(curve_iv.has_value()) << "yield curve IV solve failed";
    EXPECT_TRUE(curve_iv->used_rate_approximation);

    InterpolatedIVSolverConfig config;
    config.vega_threshold = 1e12;
    auto solver = table.make_iv_solver(config);
    ASSERT_TRUE(solver.has_value()) << "make_iv_solver failed";
    auto low_vega = solver->solve(off_grid_iv_query(params, market_price));
    ASSERT_FALSE(low_vega.has_value());
    EXPECT_EQ(low_vega.error().code, IVErrorCode::VegaTooSmall);
}

TEST(PriceTableFactoryTest, ValidatesReusableTablePricingQueries) {
    auto table_result = make_price_table(bspline_4d_config());
    ASSERT_TRUE(table_result.has_value()) << "make_price_table failed";
    auto table = std::move(*table_result);

    auto params = off_grid_pricing_params();
    EXPECT_TRUE(table.validate_pricing_params(params).has_value());

    auto wrong_type = params;
    wrong_type.option_type = OptionType::CALL;
    auto type_error = table.validate_pricing_params(wrong_type);
    ASSERT_FALSE(type_error.has_value());
    EXPECT_EQ(type_error.error().code, ValidationErrorCode::OptionTypeMismatch);

    auto out_of_range = params;
    out_of_range.volatility = 0.75;
    auto bounds_error = table.validate_pricing_params(out_of_range);
    ASSERT_FALSE(bounds_error.has_value());
    EXPECT_EQ(bounds_error.error().code, ValidationErrorCode::OutOfRange);
    EXPECT_EQ(bounds_error.error().index, 2u);
}

TEST(PriceTableFactoryTest, ParquetRoundTripPreserves4DBSplineSurface) {
    auto table_result = make_price_table(bspline_4d_config());
    ASSERT_TRUE(table_result.has_value());
    auto table = std::move(*table_result);
    auto params = off_grid_pricing_params();
    const double price_before = table.price(params);

    TempDirectory temp_dir;
    auto path = temp_dir.path() / "price_table.parquet";
    auto save_result = table.save(path);
    ASSERT_TRUE(save_result.has_value()) << "save failed";

    auto loaded_result = load_price_table(path);
    ASSERT_TRUE(loaded_result.has_value()) << "load failed";
    auto loaded = std::move(*loaded_result);

    EXPECT_EQ(table.surface_type(), surface_types::kBSpline4D);
    EXPECT_EQ(loaded.surface_type(), surface_types::kBSpline4D);
    EXPECT_NEAR(loaded.price(params), price_before, 1e-10);
    EXPECT_TRUE(loaded.delta(params).has_value());
    EXPECT_TRUE(loaded.make_iv_solver().has_value());
}

TEST(PriceTableFactoryTest, InterpolatedIVSolverConvenienceStillWorks) {
    auto solver_result = make_interpolated_iv_solver(bspline_4d_config());
    ASSERT_TRUE(solver_result.has_value()) << "legacy convenience factory failed";

    auto table_result = make_price_table(bspline_4d_config());
    ASSERT_TRUE(table_result.has_value()) << "make_price_table failed";

    auto params = off_grid_pricing_params();
    auto query = off_grid_iv_query(params, (*table_result).price(params));
    auto iv = solver_result->solve(query);
    ASSERT_TRUE(iv.has_value()) << "solver.solve failed";
    EXPECT_NEAR(iv->implied_vol, params.volatility, 0.08);
}

}  // namespace
}  // namespace mango
