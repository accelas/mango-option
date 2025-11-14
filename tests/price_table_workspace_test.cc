#include "src/option/price_table_workspace.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(PriceTableWorkspace, ConstructsFromGridData) {
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02);

    ASSERT_TRUE(ws_result.has_value());
    auto& ws = ws_result.value();

    EXPECT_EQ(ws.moneyness().size(), 4);
    EXPECT_EQ(ws.maturity().size(), 4);
    EXPECT_EQ(ws.coefficients().size(), 256);
    EXPECT_DOUBLE_EQ(ws.K_ref(), 100.0);
}

TEST(PriceTableWorkspace, RejectsInsufficientGridPoints) {
    std::vector<double> m_grid = {0.9, 1.0, 1.1};  // Only 3 points
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(3 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02);

    EXPECT_FALSE(ws_result.has_value());
    EXPECT_EQ(ws_result.error(), "Moneyness grid must have >= 4 points");
}

TEST(PriceTableWorkspace, ValidatesArenaAlignment) {
    std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1};
    std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
    std::vector<double> sigma_grid = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r_grid = {0.02, 0.03, 0.04, 0.05};
    std::vector<double> coeffs(4 * 4 * 4 * 4, 1.0);

    auto ws_result = mango::PriceTableWorkspace::create(
        m_grid, tau_grid, sigma_grid, r_grid, coeffs, 100.0, 0.02);

    ASSERT_TRUE(ws_result.has_value());
    auto& ws = ws_result.value();

    // Check 64-byte alignment for SIMD
    auto addr = reinterpret_cast<std::uintptr_t>(ws.moneyness().data());
    EXPECT_EQ(addr % 64, 0) << "Moneyness grid not 64-byte aligned";
}
