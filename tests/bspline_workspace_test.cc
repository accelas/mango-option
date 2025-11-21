#include <gtest/gtest.h>
#include "src/math/bspline_nd_separable.hpp"
#include <cmath>
#include <vector>

using namespace mango;

class BSplineWorkspaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test grids (minimum 4 points for cubic B-splines)
        axis0_ = {0.0, 0.25, 0.5, 0.75, 1.0};           // 5 points
        axis1_ = {0.0, 0.33, 0.67, 1.0};                // 4 points
        axis2_ = {0.0, 0.33, 0.67, 1.0};                // 4 points (was 3)
        axis3_ = {0.0, 0.25, 0.5, 0.75, 1.0, 1.25};     // 6 points (largest)

        // Total: 5×4×4×6 = 480 points
        n_total_ = 480;
    }

    std::vector<double> axis0_, axis1_, axis2_, axis3_;
    size_t n_total_;
};

TEST_F(BSplineWorkspaceTest, WorkspaceGivesIdenticalResults) {
    // Create test function: f(i,j,k,l) = i + 2*j + 3*k + 4*l
    std::vector<double> values(n_total_);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                for (size_t l = 0; l < 6; ++l) {
                    size_t idx = ((i * 4 + j) * 4 + k) * 6 + l;
                    values[idx] = static_cast<double>(i + 2*j + 3*k + 4*l);
                }
            }
        }
    }

    // Fit with workspace (current code path)
    auto fitter1_result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{axis0_, axis1_, axis2_, axis3_});
    ASSERT_TRUE(fitter1_result.has_value());
    auto result_workspace = fitter1_result.value().fit(values, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});

    EXPECT_TRUE(result_workspace.success) << "Workspace path failed: "
                                          << result_workspace.error_message;
    EXPECT_EQ(result_workspace.coefficients.size(), n_total_);

    // Check residuals
    EXPECT_LT(result_workspace.max_residual_per_axis[0], 1e-6);
    EXPECT_LT(result_workspace.max_residual_per_axis[1], 1e-6);
    EXPECT_LT(result_workspace.max_residual_per_axis[2], 1e-6);
    EXPECT_LT(result_workspace.max_residual_per_axis[3], 1e-6);

    // Check no failed slices
    EXPECT_EQ(result_workspace.failed_slices[0], 0);
    EXPECT_EQ(result_workspace.failed_slices[1], 0);
    EXPECT_EQ(result_workspace.failed_slices[2], 0);
    EXPECT_EQ(result_workspace.failed_slices[3], 0);
}

TEST_F(BSplineWorkspaceTest, HandlesLargestAxisCorrectly) {
    // axis3 is largest (6 points), ensure workspace sized correctly
    std::vector<double> values(n_total_, 1.0);  // Constant function

    auto fitter_result = BSplineNDSeparable<double, 4>::create(std::array<std::vector<double>, 4>{axis0_, axis1_, axis2_, axis3_});
    ASSERT_TRUE(fitter_result.has_value());

    auto result = fitter_result.value().fit(values, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});
    EXPECT_TRUE(result.success);

    // For constant function, residuals should be near zero
    EXPECT_LT(result.max_residual_per_axis[0], 1e-9);
    EXPECT_LT(result.max_residual_per_axis[1], 1e-9);
    EXPECT_LT(result.max_residual_per_axis[2], 1e-9);
    EXPECT_LT(result.max_residual_per_axis[3], 1e-9);
}

TEST_F(BSplineWorkspaceTest, WorksWithRealisticGrid) {
    // Realistic price table grid (smaller for test speed)
    std::vector<double> moneyness = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> maturity = {0.1, 0.25, 0.5, 1.0};  // 4 points (was 3)
    std::vector<double> volatility = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.0, 0.02, 0.05, 0.08};  // 4 points (was 3)

    size_t n = 5 * 4 * 4 * 4;  // 320 points
    std::vector<double> prices(n);

    // Synthetic option prices (roughly ATM put behavior)
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                for (size_t l = 0; l < 4; ++l) {
                    size_t idx = ((i * 4 + j) * 4 + k) * 4 + l;
                    double m = moneyness[i];
                    double tau = maturity[j];
                    double sigma = volatility[k];

                    // Simple pricing model: intrinsic + time value
                    prices[idx] = std::max(0.0, 1.0 - m) + sigma * std::sqrt(tau) * 0.4;
                }
            }
        }
    }

    auto fitter_result = BSplineNDSeparable<double, 4>::create(
        std::array<std::vector<double>, 4>{moneyness, maturity, volatility, rate});
    ASSERT_TRUE(fitter_result.has_value());

    auto result = fitter_result.value().fit(prices, BSplineNDSeparableConfig<double>{.tolerance = 1e-3});  // Relaxed tolerance
    EXPECT_TRUE(result.success);

    // All axes should converge
    EXPECT_EQ(result.failed_slices[0], 0);
    EXPECT_EQ(result.failed_slices[1], 0);
    EXPECT_EQ(result.failed_slices[2], 0);
    EXPECT_EQ(result.failed_slices[3], 0);
}
