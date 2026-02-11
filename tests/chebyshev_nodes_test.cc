// SPDX-License-Identifier: MIT
#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

namespace mango {
namespace {

TEST(ChebyshevNodesTest, NodeCountMatchesNumPts) {
    auto nodes = chebyshev_nodes(10, -1.0, 1.0);
    EXPECT_EQ(nodes.size(), 10u);
}

TEST(ChebyshevNodesTest, EndpointsMatchDomain) {
    auto nodes = chebyshev_nodes(11, -2.0, 3.0);
    EXPECT_DOUBLE_EQ(nodes.front(), -2.0);
    EXPECT_DOUBLE_EQ(nodes.back(), 3.0);
}

TEST(ChebyshevNodesTest, NodesAreSortedAscending) {
    auto nodes = chebyshev_nodes(15, 0.0, 1.0);
    for (size_t i = 1; i < nodes.size(); ++i) {
        EXPECT_LT(nodes[i - 1], nodes[i]);
    }
}

TEST(ChebyshevNodesTest, StandardNodesOnMinusOneOne) {
    auto nodes = chebyshev_nodes(5, -1.0, 1.0);
    EXPECT_NEAR(nodes[0], -1.0, 1e-15);
    EXPECT_NEAR(nodes[1], -std::cos(M_PI / 4), 1e-15);
    EXPECT_NEAR(nodes[2], 0.0, 1e-15);
    EXPECT_NEAR(nodes[3], std::cos(M_PI / 4), 1e-15);
    EXPECT_NEAR(nodes[4], 1.0, 1e-15);
}

TEST(ChebyshevNodesTest, BarycentricWeightsAlternateSign) {
    auto weights = chebyshev_barycentric_weights(10);
    for (size_t i = 1; i < weights.size(); ++i) {
        EXPECT_LT(weights[i - 1] * weights[i], 0.0);
    }
}

TEST(ChebyshevNodesTest, BarycentricEndpointsHalfWeight) {
    auto weights = chebyshev_barycentric_weights(5);
    EXPECT_NEAR(std::abs(weights[0]) * 2, std::abs(weights[1]), 1e-15);
    EXPECT_NEAR(std::abs(weights[4]) * 2, std::abs(weights[3]), 1e-15);
}

TEST(ChebyshevNodesTest, BarycentricInterpolationExactForPolynomial) {
    size_t num_pts = 5;
    auto nodes = chebyshev_nodes(num_pts, -1.0, 1.0);
    auto weights = chebyshev_barycentric_weights(num_pts);
    auto f = [](double x) { return x * x * x - 2.0 * x + 1.0; };
    std::vector<double> values(num_pts);
    for (size_t i = 0; i < num_pts; ++i) values[i] = f(nodes[i]);
    for (double x : {-0.7, -0.3, 0.0, 0.15, 0.8}) {
        double result = chebyshev_interpolate(x, nodes, values, weights);
        EXPECT_NEAR(result, f(x), 1e-13) << "Mismatch at x=" << x;
    }
}

TEST(ChebyshevNodesTest, BarycentricExactAtNodes) {
    size_t num_pts = 8;
    auto nodes = chebyshev_nodes(num_pts, 0.0, 5.0);
    auto weights = chebyshev_barycentric_weights(num_pts);
    std::vector<double> values(num_pts);
    for (size_t i = 0; i < num_pts; ++i) values[i] = std::sin(nodes[i]);
    for (size_t i = 0; i < num_pts; ++i) {
        double result = chebyshev_interpolate(nodes[i], nodes, values, weights);
        EXPECT_NEAR(result, values[i], 1e-14);
    }
}

TEST(ChebyshevNodesTest, CCLevelZeroGivesTwoNodes) {
    auto nodes = cc_level_nodes(0, -1.0, 1.0);
    ASSERT_EQ(nodes.size(), 2u);
    EXPECT_DOUBLE_EQ(nodes[0], -1.0);
    EXPECT_DOUBLE_EQ(nodes[1], 1.0);
}

TEST(ChebyshevNodesTest, CCLevelNodesMatchChebyshevNodes) {
    for (size_t l = 0; l <= 4; ++l) {
        size_t n = (1u << l) + 1;
        auto cc = cc_level_nodes(l, 0.0, 5.0);
        auto cgl = chebyshev_nodes(n, 0.0, 5.0);
        ASSERT_EQ(cc.size(), n) << "Level " << l;
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(cc[i], cgl[i], 1e-15)
                << "Level " << l << ", node " << i;
        }
    }
}

TEST(ChebyshevNodesTest, CCLevelsAreNested) {
    for (size_t l = 0; l <= 3; ++l) {
        auto coarse = cc_level_nodes(l, -2.0, 3.0);
        auto fine = cc_level_nodes(l + 1, -2.0, 3.0);
        for (double c : coarse) {
            bool found = false;
            for (double f : fine) {
                if (std::abs(c - f) < 1e-14) { found = true; break; }
            }
            EXPECT_TRUE(found) << "Level " << l << " node " << c
                               << " not found at level " << l + 1;
        }
    }
}

TEST(ChebyshevNodesTest, CCNewNodesAtLevel) {
    auto new2 = cc_new_nodes_at_level(2, -1.0, 1.0);
    auto full2 = cc_level_nodes(2, -1.0, 1.0);
    auto full1 = cc_level_nodes(1, -1.0, 1.0);
    EXPECT_EQ(new2.size(), 2u);
    for (double n : new2) {
        bool in_fine = false, in_coarse = false;
        for (double f : full2) if (std::abs(n - f) < 1e-14) in_fine = true;
        for (double c : full1) if (std::abs(n - c) < 1e-14) in_coarse = true;
        EXPECT_TRUE(in_fine) << "New node " << n << " not in level 2";
        EXPECT_FALSE(in_coarse) << "New node " << n << " already in level 1";
    }
}

TEST(ChebyshevNodesTest, CCNewNodesAtLevelZeroReturnsAll) {
    auto new0 = cc_new_nodes_at_level(0, -1.0, 1.0);
    EXPECT_EQ(new0.size(), 2u);
}

}  // namespace
}  // namespace mango
