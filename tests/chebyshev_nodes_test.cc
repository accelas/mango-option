// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
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

}  // namespace
}  // namespace mango
