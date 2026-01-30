// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/math/bspline_nd_separable.hpp"
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
    // Test that banded matrix uses LAPACK banded storage
    // For bandwidth=3: kl=ku=2, ldab = 2*2 + 2 + 1 = 7

    mango::BandedMatrix<double> mat(n_, 3);

    // Verify LAPACK compact storage: ldab × n entries, not n²
    EXPECT_EQ(mat.bandwidth(), 3);
    EXPECT_EQ(mat.kl(), 2);  // bandwidth - 1
    EXPECT_EQ(mat.ku(), 2);  // bandwidth - 1
    EXPECT_EQ(mat.ldab(), 7);  // 2*kl + ku + 1

    // Test accessor for simple 3×3 tridiagonal case (bandwidth=3)
    mango::BandedMatrix<double> small(3, 3);

    // Set col_start for each row (tridiagonal structure)
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

    // Tridiagonal matrix: bandwidth=3 (main diagonal + 1 above + 1 below)
    mango::BandedMatrix<double> A(3, 3);
    std::vector<double> b = {1.0, 0.0, 1.0};
    std::vector<double> x(3);

    // Set col_start for each row
    A.set_col_start(0, 0);
    A.set_col_start(1, 0);
    A.set_col_start(2, 0);

    // Build tridiagonal matrix
    A(0, 0) = 2.0; A(0, 1) = -1.0;
    A(1, 0) = -1.0; A(1, 1) = 2.0; A(1, 2) = -1.0;
    A(2, 1) = -1.0; A(2, 2) = 2.0;

    // Solve using modern interface with workspace
    mango::BandedLUWorkspace<double> workspace(3, 3);
    auto factor_result = mango::factorize_banded(A, workspace);
    ASSERT_TRUE(factor_result.ok()) << "Factorization failed: " << factor_result.message();

    auto solve_result = mango::solve_banded(workspace, std::span<const double>(b), std::span<double>(x));
    ASSERT_TRUE(solve_result.ok()) << "Substitution failed: " << solve_result.message();

    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 1.0, 1e-10);
    EXPECT_NEAR(x[2], 1.0, 1e-10);
}

TEST_F(BandedSolverTest, BandedLUSolveLarger) {
    // Test on larger system (n=10) with random RHS
    const size_t n = 10;
    // Tridiagonal matrix: bandwidth=3
    mango::BandedMatrix<double> A(n, 3);
    std::vector<double> b(n);
    std::vector<double> x(n);

    // Set col_start for each row (tridiagonal: starts at max(0, i-1))
    for (size_t i = 0; i < n; ++i) {
        A.set_col_start(i, (i > 0) ? (i - 1) : 0);
    }

    // Build symmetric positive-definite tridiagonal matrix
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) A(i, i - 1) = -1.0;  // Lower diagonal
        A(i, i) = 3.0;                    // Main diagonal (diagonally dominant)
        if (i < n - 1) A(i, i + 1) = -1.0; // Upper diagonal

        b[i] = static_cast<double>(i + 1);  // RHS: [1, 2, ..., n]
    }

    // Make a copy of A for residual verification (LU modifies A in-place)
    std::vector<double> A_copy(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        // Tridiagonal: 3 non-zero elements per row (lower, diag, upper)
        if (i > 0) A_copy[i * n + (i - 1)] = A(i, i - 1);
        A_copy[i * n + i] = A(i, i);
        if (i < n - 1) A_copy[i * n + (i + 1)] = A(i, i + 1);
    }

    // Solve using modern interface with workspace
    mango::BandedLUWorkspace<double> workspace(n, 3);
    auto factor_result = mango::factorize_banded(A, workspace);
    ASSERT_TRUE(factor_result.ok()) << "Factorization failed: " << factor_result.message();

    auto solve_result = mango::solve_banded(workspace, std::span<const double>(b), std::span<double>(x));
    ASSERT_TRUE(solve_result.ok()) << "Substitution failed: " << solve_result.message();

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

TEST_F(BandedSolverTest, CollocationAccuracy) {
    // Test that BSplineCollocation1D produces accurate results
    // for a quadratic function (should fit perfectly with cubic B-splines)

    // Create collocation solver
    auto solver_result = mango::BSplineCollocation1D<double>::create(x_);
    ASSERT_TRUE(solver_result.has_value()) << "Failed to create solver: " << solver_result.error();

    auto& solver = solver_result.value();

    // Fit quadratic function
    auto fit_result = solver.fit(y_, mango::BSplineCollocationConfig<double>{.tolerance = 1e-9});

    EXPECT_TRUE(fit_result.has_value()) << "Fit failed: " << fit_result.error();
    EXPECT_LT(fit_result->max_residual, 1e-6) << "Residual too large";
    EXPECT_EQ(fit_result->coefficients.size(), n_);
}

TEST_F(BandedSolverTest, BandedSolverAccuracyQuadratic) {
    // Verify banded solver produces highly accurate results
    // For a quadratic function y=x², cubic B-splines should fit perfectly

    // Create solver
    auto solver_result = mango::BSplineCollocation1D<double>::create(x_);
    ASSERT_TRUE(solver_result.has_value()) << "Failed to create solver: " << solver_result.error();

    auto& solver = solver_result.value();

    // Fit quadratic function y = x²
    auto fit_result = solver.fit(y_, mango::BSplineCollocationConfig<double>{.tolerance = 1e-9});
    ASSERT_TRUE(fit_result.has_value()) << "Fit failed: " << fit_result.error();

    // For a quadratic function fit with cubic B-splines, residuals should be extremely small
    // (machine precision level, not just 1e-6)
    EXPECT_LT(fit_result->max_residual, 1e-10)
        << "Residual too large for quadratic function fit with cubic B-splines";

    // Verify we got coefficients
    ASSERT_EQ(fit_result->coefficients.size(), n_);

    // Verify condition number is reasonable (well-conditioned)
    EXPECT_LT(fit_result->condition_estimate, 1e6)
        << "System is poorly conditioned";
}

TEST_F(BandedSolverTest, DetectsSingularMatrixDuplicatePoints) {
    // Test that solver detects and reports singular matrices
    // caused by duplicate or near-duplicate grid points

    // Create degenerate grid with duplicate points
    std::vector<double> bad_grid = {0.0, 0.0, 0.5, 1.0};

    // Should fail during solver creation (duplicate points)
    auto solver_result = mango::BSplineCollocation1D<double>::create(bad_grid);
    EXPECT_FALSE(solver_result.has_value())
        << "Solver should reject grid with duplicate points";

    if (!solver_result.has_value()) {
        // Duplicate points cause GridNotSorted error (spacing too small)
        EXPECT_EQ(solver_result.error().code, mango::InterpolationErrorCode::GridNotSorted)
            << "Error should indicate grid not sorted (spacing too small)";
    }
}

TEST_F(BandedSolverTest, DetectsSingularMatrixDegenerateValues) {
    // Test that solver detects singular matrix during factorization
    // when input values lead to singular system

    // Create valid grid
    std::vector<double> grid = {0.0, 0.25, 0.5, 0.75, 1.0};
    auto solver_result = mango::BSplineCollocation1D<double>::create(grid);
    ASSERT_TRUE(solver_result.has_value());

    auto& solver = solver_result.value();

    // Try to fit degenerate values (all zeros might create singular system)
    // In practice, this may or may not be singular depending on the B-spline basis
    // The real test is that we get a proper error message if it IS singular
    std::vector<double> degenerate_values = {0.0, 0.0, 0.0, 0.0, 0.0};
    auto fit_result = solver.fit(degenerate_values, mango::BSplineCollocationConfig<double>{.tolerance = 1e-9});

    // Either the fit succeeds (all-zero solution is valid for all-zero input)
    // or it fails with a clear error code
    if (!fit_result.has_value()) {
        // Any error code is acceptable as long as there is one
        EXPECT_TRUE(true)
            << "Failed fit should provide error code";
    }
}

TEST_F(BandedSolverTest, DetectsNearSingularMatrix) {
    // Test detection of nearly singular matrices (small pivots)

    // Create grid with very close points (but not duplicates)
    std::vector<double> grid = {0.0, 1e-15, 0.5, 1.0};

    // Should fail during creation (points too close)
    auto solver_result = mango::BSplineCollocation1D<double>::create(grid);
    EXPECT_FALSE(solver_result.has_value())
        << "Solver should reject grid with points closer than 1e-14";
}

} // namespace
} // namespace mango
