#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"
#include "src/pde/core/grid.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(OperatorFactoryTest, CreateFromGrid) {
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    auto grid = mango::GridView<double>(x);

    auto pde = mango::operators::BlackScholesPDE<double>(0.2, 0.05, 0.01);
    auto spatial_op = mango::operators::create_spatial_operator(pde, grid);

    // Verify operator was created and works
    std::vector<double> u = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> Lu(5, 0.0);
    spatial_op.apply(0.0, u, Lu);

    // Should produce non-zero result in interior (uniform grid: x = [0,1,2,3,4])
    // Just verify computation happened - exact value depends on grid spacing
    EXPECT_NE(Lu[2], 0.0) << "Interior point should be computed";
}

TEST(OperatorFactoryTest, CreateWithTemporaryPDE) {
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    auto grid = mango::GridView<double>(x);

    // Pass temporary PDE - should be safe (factory passes by value)
    auto spatial_op = mango::operators::create_spatial_operator(
        mango::operators::BlackScholesPDE<double>(0.2, 0.05, 0.01),
        grid
    );

    std::vector<double> u = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> Lu(5, 0.0);
    spatial_op.apply(0.0, u, Lu);

    // Should not crash, should work correctly even with temporary PDE
    EXPECT_NE(Lu[2], 0.0) << "Interior point should be computed (PDE owned by value)";
}

TEST(OperatorFactoryTest, CreateWithSharedSpacing) {
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    auto grid = mango::GridView<double>(x);

    // Create shared spacing for reuse
    auto spacing = std::make_shared<mango::operators::GridSpacing<double>>(grid);

    // Create multiple operators sharing the same spacing
    auto pde1 = mango::operators::BlackScholesPDE<double>(0.2, 0.05, 0.01);
    auto pde2 = mango::operators::BlackScholesPDE<double>(0.3, 0.05, 0.01);  // Different sigma

    auto op1 = mango::operators::create_spatial_operator(pde1, spacing);
    auto op2 = mango::operators::create_spatial_operator(pde2, spacing);

    // Both operators should work independently
    std::vector<double> u = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> Lu1(5, 0.0);
    std::vector<double> Lu2(5, 0.0);

    op1.apply(0.0, u, Lu1);
    op2.apply(0.0, u, Lu2);

    EXPECT_NE(Lu1[2], 0.0);
    EXPECT_NE(Lu2[2], 0.0);
    // Results should differ (different PDEs)
    EXPECT_NE(Lu1[2], Lu2[2]);
}
