// SPDX-License-Identifier: MIT
#include "mango/math/chebyshev/raw_tensor.hpp"
#include "mango/math/chebyshev/tucker_tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>

namespace mango {
namespace {

// ===========================================================================
// RawTensor tests
// ===========================================================================

TEST(RawTensorTest, Contract2DIdentityWeights) {
    // 2x3 tensor: [[1,2,3],[4,5,6]]
    // Contract with weights [1,0] x [0,1,0] => element (0,1) = 2
    RawTensor<2> t = RawTensor<2>::build({1, 2, 3, 4, 5, 6}, {2, 3});
    std::array<std::vector<double>, 2> coeffs = {
        std::vector<double>{1.0, 0.0},
        std::vector<double>{0.0, 1.0, 0.0},
    };
    EXPECT_NEAR(t.contract(coeffs), 2.0, 1e-15);
}

TEST(RawTensorTest, Contract3DUniform) {
    // 2x2x2 tensor of all 1s, uniform weights [0.5,0.5] per axis
    // Result = 8 * 1.0 * 0.5^3 = 1.0
    std::vector<double> vals(8, 1.0);
    RawTensor<3> t = RawTensor<3>::build(std::move(vals), {2, 2, 2});
    std::array<std::vector<double>, 3> coeffs = {
        std::vector<double>{0.5, 0.5},
        std::vector<double>{0.5, 0.5},
        std::vector<double>{0.5, 0.5},
    };
    EXPECT_NEAR(t.contract(coeffs), 1.0, 1e-15);
}

TEST(RawTensorTest, CompressedSizeEqualsTotal) {
    RawTensor<3> t = RawTensor<3>::build(std::vector<double>(60, 0.0), {3, 4, 5});
    EXPECT_EQ(t.compressed_size(), 60u);
}

// ===========================================================================
// TuckerTensor tests
// ===========================================================================

TEST(TuckerTensorTest, RoundTripSmall3D) {
    // 3x3x3 tensor: f(i,j,k) = i + 2*j + 3*k
    std::vector<double> vals(27);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                vals[i * 9 + j * 3 + k] = i + 2.0 * j + 3.0 * k;

    auto tucker = TuckerTensor<3>::build(std::move(vals), {3, 3, 3}, 1e-12);

    // Contract with delta weights to recover element (1,2,0)
    std::array<std::vector<double>, 3> coeffs = {
        std::vector<double>{0, 1, 0},
        std::vector<double>{0, 0, 1},
        std::vector<double>{1, 0, 0},
    };
    // Expected: 1 + 2*2 + 3*0 = 5
    EXPECT_NEAR(tucker.contract(coeffs), 5.0, 1e-10);
}

TEST(TuckerTensorTest, CompressesLowRankTensor) {
    // Rank-1 tensor: f(i,j,k) = (i+1)*(j+1)*(k+1) (separable)
    std::vector<double> vals(8 * 8 * 8);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            for (size_t k = 0; k < 8; ++k)
                vals[i * 64 + j * 8 + k] =
                    (i + 1.0) * (j + 1.0) * (k + 1.0);

    auto tucker = TuckerTensor<3>::build(std::move(vals), {8, 8, 8}, 1e-10);

    EXPECT_LT(tucker.compressed_size(), 8u * 8 * 8);
    auto ranks = tucker.ranks();
    for (size_t r : ranks) {
        EXPECT_LE(r, 2u) << "Rank-1 tensor should have rank ~1";
    }
}

TEST(TuckerTensorTest, ReconstructMatches3D) {
    // Build and reconstruct, verify round-trip
    std::vector<double> vals(4 * 5 * 3);
    for (size_t i = 0; i < vals.size(); ++i)
        vals[i] = std::sin(static_cast<double>(i) * 0.1);

    std::array<size_t, 3> shape = {4, 5, 3};
    auto result = tucker_hosvd<3>(vals, shape, 1e-14);
    auto reconstructed = tucker_reconstruct<3>(result);

    for (size_t i = 0; i < vals.size(); ++i)
        EXPECT_NEAR(reconstructed[i], vals[i], 1e-10) << "at index " << i;
}

TEST(TuckerTensorTest, RoundTrip4D) {
    // 3x3x3x3 tensor: f = i + 2j + 3k + 4l
    std::vector<double> vals(81);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                for (size_t l = 0; l < 3; ++l)
                    vals[i * 27 + j * 9 + k * 3 + l] =
                        i + 2.0 * j + 3.0 * k + 4.0 * l;

    auto tucker = TuckerTensor<4>::build(std::move(vals), {3, 3, 3, 3}, 1e-12);

    // Recover element (2, 1, 0, 2): expected = 2 + 2*1 + 3*0 + 4*2 = 12
    std::array<std::vector<double>, 4> coeffs = {
        std::vector<double>{0, 0, 1},
        std::vector<double>{0, 1, 0},
        std::vector<double>{1, 0, 0},
        std::vector<double>{0, 0, 1},
    };
    EXPECT_NEAR(tucker.contract(coeffs), 12.0, 1e-10);
}

}  // namespace
}  // namespace mango
