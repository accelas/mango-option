// SPDX-License-Identifier: MIT
#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(EigenCompileTest, SVDWorks) {
    Eigen::MatrixXd A(3, 3);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 10;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    EXPECT_EQ(svd.singularValues().size(), 3);
    EXPECT_GT(svd.singularValues()(0), 0.0);
}
