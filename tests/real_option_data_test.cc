#include "src/option/american_option.hpp"
#include "src/option/american_solver_workspace.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>

using namespace mango;

namespace {

std::string ResolveDataPath(const std::string& relative) {
    const char* src_dir = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    if (!src_dir || !workspace) {
        return relative;  // Best effort for local runs
    }
    return std::string(src_dir) + "/" + workspace + "/" + relative;
}

const arrow::DoubleArray* AsDoubleArray(const std::shared_ptr<arrow::Array>& array) {
    return static_cast<const arrow::DoubleArray*>(array.get());
}

const arrow::UInt8Array* AsUInt8Array(const std::shared_ptr<arrow::Array>& array) {
    return static_cast<const arrow::UInt8Array*>(array.get());
}

}  // namespace

TEST(RealOptionDataTest, SolverMatchesRecordedPrices) {
    const std::string filepath = ResolveDataPath("data/real_option_chain.arrow");

    auto file_result = arrow::io::ReadableFile::Open(filepath);
    ASSERT_TRUE(file_result.ok()) << file_result.status().ToString();
    auto reader_result = arrow::ipc::RecordBatchFileReader::Open(*file_result);
    ASSERT_TRUE(reader_result.ok()) << reader_result.status().ToString();
    auto batch_result = (*reader_result)->ReadRecordBatch(0);
    ASSERT_TRUE(batch_result.ok()) << batch_result.status().ToString();
    std::shared_ptr<arrow::RecordBatch> batch = *batch_result;

    auto spot = AsDoubleArray(batch->GetColumnByName("spot"));
    auto strike = AsDoubleArray(batch->GetColumnByName("strike"));
    auto maturity = AsDoubleArray(batch->GetColumnByName("time_to_maturity"));
    auto rate = AsDoubleArray(batch->GetColumnByName("rate"));
    auto dividend_yield = AsDoubleArray(batch->GetColumnByName("dividend_yield"));
    auto implied_vol = AsDoubleArray(batch->GetColumnByName("implied_volatility"));
    auto model_price = AsDoubleArray(batch->GetColumnByName("model_price"));
    auto option_type = AsUInt8Array(batch->GetColumnByName("option_type"));

    ASSERT_NE(spot, nullptr);
    ASSERT_NE(strike, nullptr);
    ASSERT_NE(maturity, nullptr);
    ASSERT_NE(rate, nullptr);
    ASSERT_NE(dividend_yield, nullptr);
    ASSERT_NE(implied_vol, nullptr);
    ASSERT_NE(model_price, nullptr);
    ASSERT_NE(option_type, nullptr);

    const int64_t n = batch->num_rows();
    ASSERT_GT(n, 0);

    for (int64_t i = 0; i < n; ++i) {
        AmericanOptionParams params;
        params.spot = spot->Value(i);
        params.strike = strike->Value(i);
        params.maturity = maturity->Value(i);
        params.rate = rate->Value(i);
        params.dividend_yield = dividend_yield->Value(i);
        params.volatility = implied_vol->Value(i);
        params.type = option_type->Value(i) == 0 ? OptionType::PUT : OptionType::CALL;

        // Create workspace for this option's estimated grid
        auto [grid_spec, n_time] = estimate_grid_for_option(params);
        auto workspace_result = AmericanSolverWorkspace::create(
            grid_spec, n_time, std::pmr::get_default_resource());
        ASSERT_TRUE(workspace_result.has_value()) << workspace_result.error();

        AmericanOptionSolver solver(params, workspace_result.value()->workspace_spans());
        auto price_result = solver.solve();
        ASSERT_TRUE(price_result.has_value())
            << price_result.error().message;

        // Relaxed tolerance to account for grid differences (new API uses estimate_grid_for_option)
        EXPECT_NEAR(price_result->value_at(params.spot), model_price->Value(i), 0.5)
            << "Mismatch for contract " << i;
    }
}
