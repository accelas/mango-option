// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless/tucker_decomposition.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

std::vector<double> rank1_tensor(const std::vector<double>& a,
                                 const std::vector<double>& b,
                                 const std::vector<double>& c) {
    std::vector<double> T(a.size() * b.size() * c.size());
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            for (size_t k = 0; k < c.size(); ++k)
                T[i * b.size() * c.size() + j * c.size() + k] = a[i] * b[j] * c[k];
    return T;
}

TEST(TuckerDecompositionTest, Rank1TensorGivesRank1Core) {
    std::array<size_t, 3> shape = {5, 4, 3};
    auto T = rank1_tensor({1, 2, 3, 4, 5}, {1, 0.5, 0.25, 0.125}, {1, -1, 0.5});
    auto result = tucker_hosvd(T, shape, 1e-8);
    EXPECT_EQ(result.ranks[0], 1u);
    EXPECT_EQ(result.ranks[1], 1u);
    EXPECT_EQ(result.ranks[2], 1u);
    EXPECT_EQ(result.core.size(), 1u);
}

TEST(TuckerDecompositionTest, ReconstructionMatchesOriginal) {
    std::array<size_t, 3> shape = {6, 5, 4};
    std::vector<double> T(6 * 5 * 4, 0.0);
    auto t1 = rank1_tensor({1, 2, 3, 4, 5, 6}, {1, 0.5, 0.25, 0.125, 0.0625}, {1, -1, 0.5, -0.5});
    auto t2 = rank1_tensor({6, 5, 4, 3, 2, 1}, {0.1, 0.2, 0.3, 0.4, 0.5}, {0.5, 0.5, -0.5, -0.5});
    for (size_t i = 0; i < T.size(); ++i) T[i] = t1[i] + t2[i];
    auto result = tucker_hosvd(T, shape, 1e-10);
    auto reconstructed = tucker_reconstruct(result);
    EXPECT_EQ(reconstructed.size(), T.size());
    for (size_t i = 0; i < T.size(); ++i) {
        EXPECT_NEAR(reconstructed[i], T[i], 1e-10) << "Mismatch at index " << i;
    }
}

TEST(TuckerDecompositionTest, TruncationReducesRank) {
    std::array<size_t, 3> shape = {8, 7, 6};
    std::vector<double> T(8 * 7 * 6);
    for (size_t i = 0; i < T.size(); ++i) {
        T[i] = std::sin(static_cast<double>(i) * 0.1) +
                std::cos(static_cast<double>(i) * 0.037);
    }
    auto tight = tucker_hosvd(T, shape, 1e-12);
    auto loose = tucker_hosvd(T, shape, 1e-2);
    EXPECT_LE(loose.ranks[0], tight.ranks[0]);
    EXPECT_LE(loose.ranks[1], tight.ranks[1]);
    EXPECT_LE(loose.ranks[2], tight.ranks[2]);
    auto recon = tucker_reconstruct(loose);
    double max_err = 0;
    for (size_t i = 0; i < T.size(); ++i) {
        max_err = std::max(max_err, std::abs(recon[i] - T[i]));
    }
    EXPECT_LT(max_err, 1.0);
}

TEST(TuckerDecompositionTest, FullRankPreservesExactly) {
    std::array<size_t, 3> shape = {4, 3, 3};
    std::vector<double> T(4 * 3 * 3);
    for (size_t i = 0; i < T.size(); ++i) {
        double di = static_cast<double>(i);
        T[i] = std::sin(di * 0.7) * std::exp(-di * 0.01);
    }
    auto result = tucker_hosvd(T, shape, 1e-15);
    auto recon = tucker_reconstruct(result);
    for (size_t i = 0; i < T.size(); ++i) {
        EXPECT_NEAR(recon[i], T[i], 1e-12);
    }
}

}  // namespace
}  // namespace mango
