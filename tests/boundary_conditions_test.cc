#include "src/boundary_conditions.hpp"
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

TEST(RobinBCTest, PureNeumannLeft) {
    // Robin with outward normal convention at left boundary:
    // a*u[0] - b*(u[1]-u[0])/dx = g
    // With a=0, b=-1: -(−1)*(u[1]-u[0])/dx = g  →  (u[1]-u[0])/dx = g
    // This reduces to standard Neumann: du/dx = g
    auto bc = mango::RobinBC([](double, double) { return 2.0; }, 0.0, -1.0);

    const double dx = 0.1;
    const double u_interior = 5.0;
    double u = 999.0;

    bc.apply(u, 0.0, 0.0, dx, u_interior, 1.0, mango::bc::BoundarySide::Left);

    // Should behave like Neumann: u[0] = u[1] - g*dx
    EXPECT_DOUBLE_EQ(u, 5.0 - 2.0 * 0.1);
}

TEST(RobinBCTest, PureDirichletLeft) {
    // Robin: a*u + b*du/dx = g
    // With a=1, b=0: u = g (reduces to Dirichlet)
    auto bc = mango::RobinBC([](double, double) { return 7.0; }, 1.0, 0.0);

    double u = 999.0;
    bc.apply(u, 0.0, 0.0, 0.1, 5.0, 1.0, mango::bc::BoundarySide::Left);

    EXPECT_DOUBLE_EQ(u, 7.0);  // Should be g/a = 7/1
}

TEST(RobinBCTest, MixedLeft) {
    // Robin: 2*u + u/dx = 10
    // At left: 2*u[0] - (u[1]-u[0])/dx = 10
    // Solve: u[0] = (10 + u[1]/dx) / (2 + 1/dx)
    auto bc = mango::RobinBC([](double, double) { return 10.0; }, 2.0, 1.0);

    const double dx = 0.5;
    const double u_interior = 4.0;
    double u = 999.0;

    bc.apply(u, 0.0, 0.0, dx, u_interior, 1.0, mango::bc::BoundarySide::Left);

    // Expected: (10 + 4/0.5) / (2 + 1/0.5) = (10 + 8) / (2 + 2) = 18/4 = 4.5
    EXPECT_DOUBLE_EQ(u, 4.5);
}

TEST(RobinBCTest, PureNeumannRight) {
    // Robin with a=0, b=1 at right boundary should reduce to Neumann
    // Formula: u = (g + b*u_interior/dx) / (a + b/dx)
    // With a=0, b=1: u = (g + u_interior/dx) / (1/dx) = g*dx + u_interior
    auto bc = mango::RobinBC([](double, double) { return 2.0; }, 0.0, 1.0);

    const double dx = 0.1;
    const double u_interior = 5.0;
    double u = 999.0;

    bc.apply(u, 1.0, 0.0, dx, u_interior, 1.0, mango::bc::BoundarySide::Right);

    // Should behave like Neumann: u = u_interior + g*dx
    EXPECT_DOUBLE_EQ(u, 5.0 + 2.0 * 0.1);
}

TEST(RobinBCTest, PureDirichletRight) {
    // Robin with a=1, b=0 should reduce to Dirichlet
    // Formula: u = (g + b*u_interior/dx) / (a + b/dx)
    // With a=1, b=0: u = g / 1 = g
    auto bc = mango::RobinBC([](double, double) { return 7.0; }, 1.0, 0.0);

    double u = 999.0;
    bc.apply(u, 1.0, 0.0, 0.1, 5.0, 1.0, mango::bc::BoundarySide::Right);

    EXPECT_DOUBLE_EQ(u, 7.0);  // Should be g
}

TEST(RobinBCTest, MixedRight) {
    // Robin: 2*u + du/dx = 10 at right boundary
    // Formula: u = (g + b*u_interior/dx) / (a + b/dx)
    // With a=2, b=1, g=10, dx=0.5, u_interior=4:
    // u = (10 + 1*4/0.5) / (2 + 1/0.5) = (10 + 8) / (2 + 2) = 18/4 = 4.5
    auto bc = mango::RobinBC([](double, double) { return 10.0; }, 2.0, 1.0);

    const double dx = 0.5;
    const double u_interior = 4.0;
    double u = 999.0;

    bc.apply(u, 1.0, 0.0, dx, u_interior, 1.0, mango::bc::BoundarySide::Right);

    // Expected: (10 + 4/0.5) / (2 + 1/0.5) = (10 + 8) / (2 + 2) = 18/4 = 4.5
    EXPECT_DOUBLE_EQ(u, 4.5);
}

// Concept verification tests
TEST(BoundaryConditionConceptTest, DirichletSatisfiesConcept) {
    auto bc = mango::DirichletBC([](double, double) { return 1.0; });
    static_assert(mango::BoundaryCondition<decltype(bc)>,
                  "DirichletBC must satisfy BoundaryCondition concept");
    SUCCEED();  // If we compile, the test passes
}

TEST(BoundaryConditionConceptTest, NeumannSatisfiesConcept) {
    auto bc = mango::NeumannBC([](double, double) { return 0.0; }, 1.0);
    static_assert(mango::BoundaryCondition<decltype(bc)>,
                  "NeumannBC must satisfy BoundaryCondition concept");
    SUCCEED();
}

TEST(BoundaryConditionConceptTest, RobinSatisfiesConcept) {
    auto bc = mango::RobinBC([](double, double) { return 1.0; }, 1.0, 1.0);
    static_assert(mango::BoundaryCondition<decltype(bc)>,
                  "RobinBC must satisfy BoundaryCondition concept");
    SUCCEED();
}

TEST(BoundaryConditionConceptTest, TagTypesAreDistinct) {
    using DTag = typename decltype(mango::DirichletBC([](double, double) { return 0.0; }))::tag;
    using NTag = typename decltype(mango::NeumannBC([](double, double) { return 0.0; }, 1.0))::tag;
    using RTag = typename decltype(mango::RobinBC([](double, double) { return 0.0; }, 1.0, 1.0))::tag;

    static_assert(std::is_same_v<DTag, mango::bc::dirichlet_tag>);
    static_assert(std::is_same_v<NTag, mango::bc::neumann_tag>);
    static_assert(std::is_same_v<RTag, mango::bc::robin_tag>);

    static_assert(!std::is_same_v<DTag, NTag>);
    static_assert(!std::is_same_v<NTag, RTag>);
    static_assert(!std::is_same_v<DTag, RTag>);

    SUCCEED();
}
