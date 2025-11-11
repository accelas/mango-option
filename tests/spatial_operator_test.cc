#include "src/pde/operators/spatial_operator.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"
#include "src/pde/operators/grid_spacing.hpp"
#include "src/pde/core/grid.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>

TEST(SpatialOperatorTest, UniformGridApplication) {
    // Create uniform grid [0, 1] with dx = 0.1
    std::vector<double> x(11);
    for (size_t i = 0; i < 11; ++i) {
        x[i] = i * 0.1;
    }
    auto grid = mango::GridView<double>(x);
    auto spacing = std::make_shared<mango::operators::GridSpacing<double>>(grid);

    // Create Black-Scholes PDE: sigma=0.2, r=0.05, d=0.01
    auto pde = mango::operators::BlackScholesPDE<double>(0.2, 0.05, 0.01);

    // Create spatial operator
    auto spatial_op = mango::operators::SpatialOperator<
        mango::operators::BlackScholesPDE<double>, double>(pde, spacing);

    // Test function: u = exp(-x) (arbitrary smooth function)
    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = std::exp(-x[i]);
    }

    // Apply operator
    std::vector<double> Lu(11, 0.0);
    spatial_op.apply(0.0, u, Lu);  // t = 0.0

    // Verify interior point i=5 (x=0.5) was computed
    // For u = exp(-x), analytically:
    // L(u) = (σ²/2)·u'' + (r-d-σ²/2)·u' - r·u
    //      = 0.02·exp(-x) + 0.02·(-exp(-x)) - 0.05·exp(-x)
    //      = -0.05·exp(-x)
    // At x=0.5: L(u) ≈ -0.0303
    const double expected = -0.05 * std::exp(-0.5);  // ≈ -0.0303
    EXPECT_NEAR(Lu[5], expected, 1e-4);  // Numerical error tolerance

    // Boundaries should be untouched (left as 0.0)
    EXPECT_EQ(Lu[0], 0.0);
    EXPECT_EQ(Lu[10], 0.0);
}

TEST(SpatialOperatorTest, InteriorRange) {
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    auto grid = mango::GridView<double>(x);
    auto spacing = std::make_shared<mango::operators::GridSpacing<double>>(grid);
    auto pde = mango::operators::BlackScholesPDE<double>(0.2, 0.05, 0.01);
    auto spatial_op = mango::operators::SpatialOperator<
        mango::operators::BlackScholesPDE<double>, double>(pde, spacing);

    auto range = spatial_op.interior_range(5);
    EXPECT_EQ(range.start, 1);
    EXPECT_EQ(range.end, 4);  // [1, 4) excludes boundaries 0 and 4
}

TEST(SpatialOperatorTest, ApplyInteriorMatchesApply) {
    // Verify apply() and apply_interior() produce same results
    std::vector<double> x(11);
    for (size_t i = 0; i < 11; ++i) {
        x[i] = i * 0.1;
    }
    auto grid = mango::GridView<double>(x);
    auto spacing = std::make_shared<mango::operators::GridSpacing<double>>(grid);
    auto pde = mango::operators::BlackScholesPDE<double>(0.2, 0.05, 0.01);
    auto spatial_op = mango::operators::SpatialOperator<
        mango::operators::BlackScholesPDE<double>, double>(pde, spacing);

    std::vector<double> u(11);
    for (size_t i = 0; i < 11; ++i) {
        u[i] = std::exp(-x[i]);
    }

    // Apply via apply()
    std::vector<double> Lu1(11, 0.0);
    spatial_op.apply(0.0, u, Lu1);

    // Apply via apply_interior()
    std::vector<double> Lu2(11, 0.0);
    spatial_op.apply_interior(0.0, u, Lu2, 1, 10);

    // Results should match in interior
    for (size_t i = 1; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(Lu1[i], Lu2[i]);
    }
}

TEST(SpatialOperatorTest, LifetimeSafety) {
    // Test that operator owns PDE and GridSpacing safely
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
    auto grid = mango::GridView<double>(x);

    // Create operator with temporary PDE
    auto spatial_op = mango::operators::SpatialOperator<
        mango::operators::BlackScholesPDE<double>, double>(
            mango::operators::BlackScholesPDE<double>(0.2, 0.05, 0.01),  // Temporary!
            std::make_shared<mango::operators::GridSpacing<double>>(grid)
    );

    // Operator should still work (PDE owned by value, not dangling reference)
    std::vector<double> u = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> Lu(4, 0.0);
    spatial_op.apply(0.0, u, Lu);  // Should not crash

    // Should produce non-zero result in interior (just verify computation happened)
    // Exact value depends on irregular spacing, so we only check != 0
    EXPECT_NE(Lu[1], 0.0) << "Interior point should be computed";
}
