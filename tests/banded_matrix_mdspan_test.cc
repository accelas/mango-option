#include "src/math/banded_matrix_solver.hpp"
#include <gtest/gtest.h>

namespace mango {
namespace {

TEST(BandedMatrixMdspan, Construction) {
    // Create 5x5 tridiagonal matrix (kl=1, ku=1)
    BandedMatrix<double> matrix(5, 1, 1);

    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.kl(), 1);
    EXPECT_EQ(matrix.ku(), 1);
    EXPECT_EQ(matrix.ldab(), 4);  // 2*1 + 1 + 1
}

TEST(BandedMatrixMdspan, ElementAccess) {
    BandedMatrix<double> matrix(4, 1, 1);

    // Fill diagonal
    for (size_t i = 0; i < 4; ++i) {
        matrix(i, i) = static_cast<double>(i + 1);
    }

    // Fill super-diagonal
    for (size_t i = 0; i < 3; ++i) {
        matrix(i, i + 1) = 0.5;
    }

    // Fill sub-diagonal
    for (size_t i = 1; i < 4; ++i) {
        matrix(i, i - 1) = 0.25;
    }

    // Verify values via const access
    const auto& cmatrix = matrix;
    EXPECT_EQ(cmatrix(0, 0), 1.0);
    EXPECT_EQ(cmatrix(1, 1), 2.0);
    EXPECT_EQ(cmatrix(0, 1), 0.5);
    EXPECT_EQ(cmatrix(1, 0), 0.25);
}

TEST(BandedMatrixMdspan, LapackDataPointer) {
    BandedMatrix<double> matrix(5, 2, 1);

    // Set value via mdspan operator()
    matrix(1, 1) = 42.0;

    // Verify it's stored in correct LAPACK location
    // A(1,1) -> AB(kl + ku + 1 - 1, 1) = AB(2 + 1 + 0, 1) = AB(3, 1)
    // Offset = 3 + 1 * ldab = 3 + 1 * 6 = 9
    const double* lapack_data = matrix.lapack_data();
    EXPECT_EQ(lapack_data[9], 42.0);
}

}  // namespace
}  // namespace mango
