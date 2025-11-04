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

TEST(NeumannBCTest, LinearFunctionLeftBoundary) {
    // Analytical: u(x) = 2x + 3
    // du/dx = 2 everywhere
    // At left boundary (x=0): u[0] = 3, u[1] = 2*dx + 3
    // Neumann BC: du/dx = 2
    // Formula: u[0] = u[1] - g*dx

    auto bc = mango::NeumannBC([](double, double) { return 2.0; }, 1.0);

    const double dx = 0.1;
    const double u1 = 2.0 * dx + 3.0;  // u[1] = 3.2
    double u0 = 999.0;

    bc.apply(u0, 0.0, 0.0, dx, u1, 1.0, mango::bc::BoundarySide::Left);

    EXPECT_DOUBLE_EQ(u0, 3.0);  // Should match analytical u(0) = 3
}

TEST(NeumannBCTest, LinearFunctionRightBoundary) {
    // Analytical: u(x) = 2x + 3 on [0, 1]
    // du/dx = 2 everywhere
    // At right boundary (x=1): u[n-1] = 5, u[n-2] = 5 - 2*dx
    // Neumann BC: du/dx = 2
    // Formula: u[n-1] = u[n-2] + g*dx

    auto bc = mango::NeumannBC([](double, double) { return 2.0; }, 1.0);

    const double dx = 0.1;
    const double u_n2 = 5.0 - 2.0 * dx;  // u[n-2] = 4.8
    double u_n1 = 999.0;

    bc.apply(u_n1, 1.0, 0.0, dx, u_n2, 1.0, mango::bc::BoundarySide::Right);

    EXPECT_DOUBLE_EQ(u_n1, 5.0);  // Should match analytical u(1) = 5
}

TEST(NeumannBCTest, InsulatedBoundary) {
    // Zero flux: du/dx = 0
    // Both boundaries should equal interior value
    auto bc = mango::NeumannBC([](double, double) { return 0.0; }, 1.0);

    const double dx = 0.1;
    const double u_interior = 7.5;
    double u_boundary = 999.0;

    // Left boundary
    bc.apply(u_boundary, 0.0, 0.0, dx, u_interior, 1.0, mango::bc::BoundarySide::Left);
    EXPECT_DOUBLE_EQ(u_boundary, u_interior);

    // Right boundary
    u_boundary = 999.0;
    bc.apply(u_boundary, 1.0, 0.0, dx, u_interior, 1.0, mango::bc::BoundarySide::Right);
    EXPECT_DOUBLE_EQ(u_boundary, u_interior);
}
