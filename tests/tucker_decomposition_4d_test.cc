// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless/tucker_decomposition_4d.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

TEST(TuckerDecomposition4DTest, ExactRoundtripForRank1) {
    // f(i,j,k,l) = (i+1) * (j+1) * (k+1) * (l+1) is rank-1
    std::array<size_t, 4> shape = {4, 5, 3, 4};
    std::vector<double> T(4 * 5 * 3 * 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 5; ++j)
            for (size_t k = 0; k < 3; ++k)
                for (size_t l = 0; l < 4; ++l)
                    T[i*5*3*4 + j*3*4 + k*4 + l] =
                        (i + 1.0) * (j + 1.0) * (k + 1.0) * (l + 1.0);

    auto tucker = tucker_hosvd_4d(T, shape, 1e-10);

    for (size_t d = 0; d < 4; ++d) {
        EXPECT_EQ(tucker.ranks[d], 1u) << "mode " << d;
    }

    auto reconstructed = tucker_reconstruct_4d(tucker);
    for (size_t i = 0; i < T.size(); ++i) {
        EXPECT_NEAR(reconstructed[i], T[i], 1e-10) << "at flat index " << i;
    }
}

TEST(TuckerDecomposition4DTest, CompressesLowRankTensor) {
    // Rank-2 tensor: sum of two separable rank-1 components
    std::array<size_t, 4> shape = {6, 6, 6, 6};
    size_t total = 6 * 6 * 6 * 6;
    std::vector<double> T(total);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 6; ++j)
            for (size_t k = 0; k < 6; ++k)
                for (size_t l = 0; l < 6; ++l) {
                    double di = static_cast<double>(i);
                    double dj = static_cast<double>(j);
                    double dk = static_cast<double>(k);
                    double dl = static_cast<double>(l);
                    T[i*6*6*6 + j*6*6 + k*6 + l] =
                        (di + 1.0) * (dj + 1.0) * (dk + 1.0) * (dl + 1.0)
                        + (6.0 - di) * std::cos(dj * 0.5) *
                          std::exp(-dk * 0.2) * (dl + 2.0);
                }

    auto tucker = tucker_hosvd_4d(T, shape, 1e-8);

    size_t core_size = tucker.ranks[0] * tucker.ranks[1] *
                       tucker.ranks[2] * tucker.ranks[3];
    EXPECT_LT(core_size, total);

    auto reconstructed = tucker_reconstruct_4d(tucker);
    for (size_t i = 0; i < T.size(); ++i) {
        EXPECT_NEAR(reconstructed[i], T[i], 1e-6) << "at flat index " << i;
    }
}

TEST(TuckerDecomposition4DTest, ModeUnfoldDimensionsCorrect) {
    std::array<size_t, 4> shape = {3, 4, 5, 2};
    std::vector<double> T(3 * 4 * 5 * 2, 1.0);

    for (size_t mode = 0; mode < 4; ++mode) {
        auto M = mode_unfold_4d(T, shape, mode);
        EXPECT_EQ(static_cast<size_t>(M.rows()), shape[mode]) << "mode " << mode;
        size_t expected_cols = 1;
        for (size_t d = 0; d < 4; ++d)
            if (d != mode) expected_cols *= shape[d];
        EXPECT_EQ(static_cast<size_t>(M.cols()), expected_cols) << "mode " << mode;
    }
}

}  // namespace
}  // namespace mango
