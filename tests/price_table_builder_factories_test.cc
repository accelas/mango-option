#include <gtest/gtest.h>
#include "src/option/table/price_table_builder.hpp"
#include "src/option/option_chain.hpp"

TEST(PriceTableFactoriesTest, FromVectorsCreatesBuilderAndAxes) {
    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> maturity = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> vol = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.03, 0.04, 0.05, 0.06};

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, vol, rate,
        100.0,      // K_ref
        grid_spec,
        100,        // n_time
        mango::OptionType::PUT
    );

    ASSERT_TRUE(result.has_value()) << "Factory failed: " << result.error();
    auto& [builder, axes] = result.value();

    EXPECT_EQ(axes.grids[0].size(), 4);
    EXPECT_EQ(axes.grids[1].size(), 4);
    EXPECT_EQ(axes.grids[2].size(), 4);
    EXPECT_EQ(axes.grids[3].size(), 4);
}

TEST(PriceTableFactoriesTest, FromVectorsSortsAndDedupes) {
    std::vector<double> moneyness = {1.0, 0.8, 0.9, 1.0, 1.1};  // Has duplicate
    std::vector<double> maturity = {0.5, 0.25, 1.0, 0.75};     // Out of order
    std::vector<double> vol = {0.20, 0.15, 0.30, 0.25};
    std::vector<double> rate = {0.04, 0.03, 0.06, 0.05};

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, vol, rate,
        100.0, grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_TRUE(result.has_value());
    auto& [builder, axes] = result.value();

    // Should have 4 unique moneyness values after deduplication
    EXPECT_EQ(axes.grids[0].size(), 4);

    // Should be sorted ascending
    EXPECT_NEAR(axes.grids[0][0], 0.8, 1e-10);
    EXPECT_NEAR(axes.grids[0][1], 0.9, 1e-10);
    EXPECT_NEAR(axes.grids[0][2], 1.0, 1e-10);
    EXPECT_NEAR(axes.grids[0][3], 1.1, 1e-10);

    EXPECT_NEAR(axes.grids[1][0], 0.25, 1e-10);
    EXPECT_NEAR(axes.grids[1][1], 0.5, 1e-10);
    EXPECT_NEAR(axes.grids[1][2], 0.75, 1e-10);
    EXPECT_NEAR(axes.grids[1][3], 1.0, 1e-10);
}

TEST(PriceTableFactoriesTest, FromVectorsRejectsNegativeMoneyness) {
    std::vector<double> moneyness = {-0.1, 0.9, 1.0, 1.1};
    std::vector<double> maturity = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> vol = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.03, 0.04, 0.05, 0.06};

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, vol, rate,
        100.0, grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("positive") != std::string::npos);
}

TEST(PriceTableFactoriesTest, FromVectorsRejectsNegativeMaturity) {
    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> maturity = {-0.1, 0.5, 0.75, 1.0};
    std::vector<double> vol = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.03, 0.04, 0.05, 0.06};

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, vol, rate,
        100.0, grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("positive") != std::string::npos);
}

TEST(PriceTableFactoriesTest, FromVectorsRejectsNegativeVolatility) {
    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> maturity = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> vol = {-0.1, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.03, 0.04, 0.05, 0.06};

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, vol, rate,
        100.0, grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("positive") != std::string::npos);
}

TEST(PriceTableFactoriesTest, FromVectorsRejectsZeroKRef) {
    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> maturity = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> vol = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.03, 0.04, 0.05, 0.06};

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, vol, rate,
        0.0,  // Invalid K_ref
        grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("K_ref") != std::string::npos ||
                result.error().find("positive") != std::string::npos);
}

TEST(PriceTableFactoriesTest, FromVectorsAcceptsNegativeRates) {
    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> maturity = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> vol = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {-0.02, -0.01, 0.0, 0.01};  // Negative rates allowed

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, vol, rate,
        100.0, grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_TRUE(result.has_value()) << "Factory rejected valid negative rates: " << result.error();
}

TEST(PriceTableFactoriesTest, FromStrikesComputesMoneyness) {
    double spot = 100.0;
    std::vector<double> strikes = {80.0, 90.0, 100.0, 110.0, 120.0};
    std::vector<double> maturities = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> vols = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rates = {0.03, 0.04, 0.05, 0.06};

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_strikes(
        spot, strikes, maturities, vols, rates,
        grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_TRUE(result.has_value()) << "Factory failed: " << result.error();
    auto& [builder, axes] = result.value();

    // Moneyness = spot/strike, sorted ascending
    // strikes [80, 90, 100, 110, 120] → moneyness [100/120, 100/110, 1.0, 100/90, 100/80]
    //                                  → sorted: [0.833, 0.909, 1.0, 1.111, 1.25]
    EXPECT_EQ(axes.grids[0].size(), 5);
    EXPECT_NEAR(axes.grids[0][0], 100.0/120.0, 1e-6);  // 0.833...
    EXPECT_NEAR(axes.grids[0][1], 100.0/110.0, 1e-6);  // 0.909...
    EXPECT_NEAR(axes.grids[0][2], 1.0, 1e-6);          // 1.0
    EXPECT_NEAR(axes.grids[0][3], 100.0/90.0, 1e-6);   // 1.111...
    EXPECT_NEAR(axes.grids[0][4], 100.0/80.0, 1e-6);   // 1.25
}

TEST(PriceTableFactoriesTest, FromStrikesRejectsNegativeSpot) {
    double spot = -100.0;
    std::vector<double> strikes = {80.0, 90.0, 100.0, 110.0};
    std::vector<double> maturities = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> vols = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rates = {0.03, 0.04, 0.05, 0.06};

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_strikes(
        spot, strikes, maturities, vols, rates,
        grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Spot") != std::string::npos ||
                result.error().find("positive") != std::string::npos);
}

TEST(PriceTableFactoriesTest, FromStrikesRejectsNegativeStrikes) {
    double spot = 100.0;
    std::vector<double> strikes = {-80.0, 90.0, 100.0, 110.0};
    std::vector<double> maturities = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> vols = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rates = {0.03, 0.04, 0.05, 0.06};

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_strikes(
        spot, strikes, maturities, vols, rates,
        grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().find("Strikes") != std::string::npos ||
                result.error().find("positive") != std::string::npos);
}

TEST(PriceTableFactoriesTest, FromChainExtractsFields) {
    mango::OptionChain chain;
    chain.ticker = "AAPL";
    chain.spot = 150.0;
    chain.strikes = {140.0, 145.0, 150.0, 155.0, 160.0};
    chain.maturities = {0.25, 0.5, 0.75, 1.0};
    chain.implied_vols = {0.20, 0.22, 0.25, 0.28};
    chain.rates = {0.04, 0.045, 0.05, 0.055};
    chain.dividend_yield = 0.01;

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_chain(
        chain, grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_TRUE(result.has_value()) << "Factory failed: " << result.error();
    auto& [builder, axes] = result.value();

    EXPECT_EQ(axes.grids[0].size(), 5);  // 5 strikes → 5 moneyness
    EXPECT_EQ(axes.grids[1].size(), 4);  // 4 maturities
    EXPECT_EQ(axes.grids[2].size(), 4);  // 4 vols
    EXPECT_EQ(axes.grids[3].size(), 4);  // 4 rates
}

TEST(PriceTableFactoriesTest, FromChainUsesDividendYield) {
    mango::OptionChain chain;
    chain.ticker = "AAPL";
    chain.spot = 150.0;
    chain.strikes = {140.0, 145.0, 150.0, 155.0};
    chain.maturities = {0.25, 0.5, 0.75, 1.0};
    chain.implied_vols = {0.20, 0.22, 0.25, 0.28};
    chain.rates = {0.04, 0.045, 0.05, 0.055};
    chain.dividend_yield = 0.01;

    auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value();

    auto result = mango::PriceTableBuilder<4>::from_chain(
        chain, grid_spec, 100, mango::OptionType::PUT
    );

    ASSERT_TRUE(result.has_value());
    // We can't directly check the dividend_yield is used, but we can verify
    // the factory succeeds and returns valid axes
    auto& [builder, axes] = result.value();
    EXPECT_EQ(axes.grids[0].size(), 4);
}
