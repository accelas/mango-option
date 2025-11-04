#include "src/cpp/boundary_conditions.hpp"
#include <gtest/gtest.h>

TEST(BoundaryConditionTest, DirichletTagExists) {
    // Just verify tags exist and are distinct types
    [[maybe_unused]] mango::bc::dirichlet_tag d;
    [[maybe_unused]] mango::bc::neumann_tag n;
    [[maybe_unused]] mango::bc::robin_tag r;

    // Tags should be empty types
    EXPECT_EQ(sizeof(mango::bc::dirichlet_tag), 1);
    EXPECT_EQ(sizeof(mango::bc::neumann_tag), 1);
    EXPECT_EQ(sizeof(mango::bc::robin_tag), 1);
}

TEST(BoundaryConditionTest, BoundarySideEnum) {
    auto left = mango::bc::BoundarySide::Left;
    auto right = mango::bc::BoundarySide::Right;

    EXPECT_NE(left, right);
}

TEST(DirichletBCTest, ConstantValue) {
    auto bc = mango::DirichletBC([](double, double) { return 5.0; });

    // Test natural interface
    EXPECT_DOUBLE_EQ(bc.value(0.0, 0.0), 5.0);
    EXPECT_DOUBLE_EQ(bc.value(1.0, 0.5), 5.0);
}

TEST(DirichletBCTest, TimeDependent) {
    auto bc = mango::DirichletBC([](double t, double) { return 2.0 * t; });

    EXPECT_DOUBLE_EQ(bc.value(0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(bc.value(1.0, 0.0), 2.0);
    EXPECT_DOUBLE_EQ(bc.value(2.5, 0.0), 5.0);
}

TEST(DirichletBCTest, ApplyMethod) {
    auto bc = mango::DirichletBC([](double, double x) { return x * x; });

    double u = 999.0;  // Will be overwritten
    bc.apply(u, 3.0, 0.0, 0.1, 0.0, 0.0, mango::bc::BoundarySide::Left);

    EXPECT_DOUBLE_EQ(u, 9.0);  // x^2 with x=3
}
