#include <gtest/gtest.h>
#include "src/option/table/price_table_surface.hpp"
#include "src/support/memory/aligned_arena.hpp"

namespace mango {
namespace {

TEST(PriceTableSurfaceTest, Build2DSurface) {
    PriceTableAxes<2> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};  // Need at least 4 points for cubic B-splines
    axes.grids[1] = {0.1, 0.5, 1.0, 1.5};
    axes.names[0] = "moneyness";
    axes.names[1] = "maturity";

    // 5x4 = 20 coefficients (row-major: m varies fastest)
    std::vector<double> coeffs(20);
    for (size_t i = 0; i < 20; ++i) {
        coeffs[i] = static_cast<double>(i + 1);
    }

    PriceTableMetadata meta{.K_ref = 100.0, .dividend_yield = 0.02};

    auto result = PriceTableSurface<2>::build(std::move(axes), std::move(coeffs), meta);
    ASSERT_TRUE(result.has_value()) << "Error: " << result.error();

    auto surface = result.value();
    EXPECT_EQ(surface->axes().grids[0].size(), 5);
    EXPECT_DOUBLE_EQ(surface->metadata().K_ref, 100.0);
}

TEST(PriceTableSurfaceTest, ValueInterpolation) {
    PriceTableAxes<2> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5, 1.0, 1.5};

    // Simple linear coefficients for testing
    size_t total = 4 * 4;
    std::vector<double> coeffs(total);
    for (size_t i = 0; i < total; ++i) {
        coeffs[i] = static_cast<double>(i + 1);
    }

    PriceTableMetadata meta{.K_ref = 100.0};
    auto surface = PriceTableSurface<2>::build(std::move(axes), std::move(coeffs), meta).value();

    // Query at grid point (should match coefficient)
    double val = surface->value({0.8, 0.1});
    EXPECT_NEAR(val, 1.0, 1e-10);
}

TEST(PriceTableSurfaceTest, RejectInvalidCoefficients) {
    PriceTableAxes<2> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};  // 4 points
    axes.grids[1] = {0.1, 0.5, 1.0, 1.5};  // 4 points

    std::vector<double> coeffs = {1.0, 2.0};  // Only 2, need 4*4=16

    PriceTableMetadata meta{.K_ref = 100.0};
    auto result = PriceTableSurface<2>::build(std::move(axes), std::move(coeffs), meta);

    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("size"), std::string::npos);
}

// REGRESSION TEST: Verify build() return type includes template parameter <N>
// Issue caught during review: Missing <N> in return type would cause type mismatch
// This test ensures the return type is shared_ptr<const PriceTableSurface<N>>, not
// shared_ptr<const PriceTableSurface> (which would be invalid)
TEST(PriceTableSurfaceTest, BuildReturnsCorrectTemplateType) {
    PriceTableAxes<2> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5, 1.0, 1.5};

    std::vector<double> coeffs(16, 1.0);
    PriceTableMetadata meta{.K_ref = 100.0};

    auto result = PriceTableSurface<2>::build(std::move(axes), std::move(coeffs), meta);
    ASSERT_TRUE(result.has_value());

    // Compile-time type verification: ensure template parameter is present
    // This would fail to compile if build() returned shared_ptr<const PriceTableSurface>
    // without the <N> template parameter
    std::shared_ptr<const PriceTableSurface<2>> surface = result.value();
    EXPECT_NE(surface, nullptr);

    // Runtime verification: surface should be usable
    EXPECT_EQ(surface->axes().grids[0].size(), 4);
}

// REGRESSION TEST: Verify 3D surface also has correct template type
TEST(PriceTableSurfaceTest, Build3DReturnsCorrectTemplateType) {
    PriceTableAxes<3> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5, 1.0, 1.5};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};

    std::vector<double> coeffs(64, 1.0);  // 4*4*4 = 64
    PriceTableMetadata meta{.K_ref = 100.0};

    auto result = PriceTableSurface<3>::build(std::move(axes), std::move(coeffs), meta);
    ASSERT_TRUE(result.has_value());

    // Compile-time type verification
    std::shared_ptr<const PriceTableSurface<3>> surface = result.value();
    EXPECT_NE(surface, nullptr);
}

} // namespace
} // namespace mango
