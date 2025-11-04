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
