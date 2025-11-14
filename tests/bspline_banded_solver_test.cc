#include <gtest/gtest.h>
#include "src/interpolation/bspline_fitter_4d.hpp"
#include <vector>
#include <cmath>

namespace mango {
namespace {

// Test fixture for banded solver testing
class BandedSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simple test case: fit cubic B-spline to quadratic function
        // y = x^2 on [0, 1] with 11 points
        n_ = 11;
        x_.resize(n_);
        y_.resize(n_);

        for (size_t i = 0; i < n_; ++i) {
            x_[i] = static_cast<double>(i) / (n_ - 1);
            y_[i] = x_[i] * x_[i];  // Quadratic function
        }

        // Create knot vector for cubic B-spline (degree 3)
        // Use clamped_knots_cubic utility function
        knots_ = clamped_knots_cubic(x_);
    }

    size_t n_;
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> knots_;
};

TEST_F(BandedSolverTest, DenseSolverBaseline) {
    // This test uses the CURRENT dense solver as baseline
    // We'll compare banded solver results against this

    // Create fitter (uses current dense implementation)
    // NOTE: Adjust constructor call based on actual BSplineFitter4D API
    // This is a placeholder - actual API may differ

    // For now, test that we can construct the test case
    EXPECT_EQ(x_.size(), n_);
    EXPECT_EQ(y_.size(), n_);
    EXPECT_EQ(knots_.size(), n_ + 4);

    // Verify knot vector structure (clamped at endpoints)
    EXPECT_DOUBLE_EQ(knots_[0], x_.front());
    EXPECT_DOUBLE_EQ(knots_[3], x_.front());  // Multiplicity 4 at start
    EXPECT_DOUBLE_EQ(knots_[knots_.size() - 1], x_.back());
    EXPECT_DOUBLE_EQ(knots_[knots_.size() - 4], x_.back());  // Multiplicity 4 at end

    // Verify monotonicity
    for (size_t i = 1; i < knots_.size(); ++i) {
        EXPECT_GE(knots_[i], knots_[i-1]) << "Knots not monotonic at index " << i;
    }
}

TEST_F(BandedSolverTest, BandedStorageStructure) {
    // Test that banded matrix is stored compactly (4 diagonals × n entries)
    // instead of dense n×n storage

    mango::BandedMatrixStorage mat(n_);

    // Verify compact storage: 4n entries, not n²
    EXPECT_EQ(mat.band_values().size(), 4 * n_);
    EXPECT_EQ(mat.col_starts().size(), n_);

    // Test accessor for simple 3×3 case
    mango::BandedMatrixStorage small(3);
    small.set_col_start(0, 0);
    small.set_col_start(1, 0);
    small.set_col_start(2, 0);

    // Set diagonal entries
    small(0, 0) = 1.0;
    small(1, 1) = 2.0;
    small(2, 2) = 3.0;

    EXPECT_DOUBLE_EQ(small(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(small(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(small(2, 2), 3.0);
}

TEST_F(BandedSolverTest, BandedLUSolveSimple) {
    // Test banded LU solve on simple 3×3 tridiagonal system
    // A = [2 -1  0]     b = [1]
    //     [-1 2 -1]         [0]
    //     [0 -1  2]         [1]
    // Solution: x = [1, 1, 1]

    mango::BandedMatrixStorage A(3);
    std::vector<double> b = {1.0, 0.0, 1.0};
    std::vector<double> x(3);

    // Build tridiagonal matrix
    A.set_col_start(0, 0);  // Row 0: columns [0, 1]
    A.set_col_start(1, 0);  // Row 1: columns [0, 1, 2]
    A.set_col_start(2, 1);  // Row 2: columns [1, 2]

    A(0, 0) = 2.0; A(0, 1) = -1.0;
    A(1, 0) = -1.0; A(1, 1) = 2.0; A(1, 2) = -1.0;
    A(2, 1) = -1.0; A(2, 2) = 2.0;

    // Solve
    mango::banded_lu_solve(A, b, x);

    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 1.0, 1e-10);
    EXPECT_NEAR(x[2], 1.0, 1e-10);
}

TEST_F(BandedSolverTest, BandedLUSolveLarger) {
    // Test on larger system (n=10) with random RHS
    const size_t n = 10;
    mango::BandedMatrixStorage A(n);
    std::vector<double> b(n);
    std::vector<double> x(n);

    // Build symmetric positive-definite tridiagonal matrix
    for (size_t i = 0; i < n; ++i) {
        A.set_col_start(i, (i > 0) ? i - 1 : 0);

        if (i > 0) A(i, i - 1) = -1.0;  // Lower diagonal
        A(i, i) = 3.0;                    // Main diagonal (diagonally dominant)
        if (i < n - 1) A(i, i + 1) = -1.0; // Upper diagonal

        b[i] = static_cast<double>(i + 1);  // RHS: [1, 2, ..., n]
    }

    // Make a copy of A for residual verification (LU modifies A in-place)
    std::vector<double> A_copy(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        size_t col_start = A.col_start(i);
        for (size_t k = 0; k < 4; ++k) {
            size_t col = col_start + k;
            if (col < n) {
                A_copy[i * n + col] = A(i, col);
            }
        }
    }

    // Solve
    mango::banded_lu_solve(A, b, x);

    // Verify residual ||Ax - b|| is small
    std::vector<double> residual(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            residual[i] += A_copy[i * n + j] * x[j];
        }
        residual[i] -= b[i];
    }

    // Compute max residual
    double max_residual = 0.0;
    for (size_t i = 0; i < n; ++i) {
        max_residual = std::max(max_residual, std::abs(residual[i]));
    }

    EXPECT_LT(max_residual, 1e-10) << "Residual too large for n=10 system";
}

// ============================================================================
// Task 5: Integration Tests
// ============================================================================

TEST_F(BandedSolverTest, CollocationUseBandedSolver) {
    // Test that BSplineCollocation1D can use banded solver
    // and produces valid results

    // Create collocation solver
    auto solver_result = mango::BSplineCollocation1D::create(x_);
    ASSERT_TRUE(solver_result.has_value()) << "Failed to create solver: " << solver_result.error();

    auto& solver = solver_result.value();

    // Enable banded solver mode (this will fail until we implement it)
    solver.set_use_banded_solver(true);

    // Fit quadratic function
    auto fit_result = solver.fit(y_, 1e-9);

    EXPECT_TRUE(fit_result.success) << "Fit failed: " << fit_result.error_message;
    EXPECT_LT(fit_result.max_residual, 1e-6) << "Residual too large";
    EXPECT_EQ(fit_result.coefficients.size(), n_);
}

TEST_F(BandedSolverTest, BandedVsDenseIdenticalResults) {
    // Verify banded solver produces identical results to dense solver

    // Create solver
    auto solver_result = mango::BSplineCollocation1D::create(x_);
    ASSERT_TRUE(solver_result.has_value()) << "Failed to create solver: " << solver_result.error();

    auto& solver = solver_result.value();

    // Solve with dense solver (current default)
    solver.set_use_banded_solver(false);
    auto dense_result = solver.fit(y_, 1e-9);
    ASSERT_TRUE(dense_result.success) << "Dense solver failed: " << dense_result.error_message;

    // Solve with banded solver
    solver.set_use_banded_solver(true);
    auto banded_result = solver.fit(y_, 1e-9);
    ASSERT_TRUE(banded_result.success) << "Banded solver failed: " << banded_result.error_message;

    // Compare coefficients (should match to floating-point precision)
    ASSERT_EQ(dense_result.coefficients.size(), banded_result.coefficients.size());

    for (size_t i = 0; i < dense_result.coefficients.size(); ++i) {
        EXPECT_NEAR(dense_result.coefficients[i], banded_result.coefficients[i], 1e-14)
            << "Coefficients differ at index " << i;
    }

    // Compare residuals
    EXPECT_NEAR(dense_result.max_residual, banded_result.max_residual, 1e-14)
        << "Residuals differ between solvers";
}

} // namespace
} // namespace mango
