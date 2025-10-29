#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

extern "C" {
#include "../src/tridiagonal.h"
}

// Test fixture for tridiagonal solver tests
class TridiagonalTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-10;

    // Helper: Compute matrix-vector product for verification
    std::vector<double> matvec_tridiagonal(const std::vector<double>& lower,
                                          const std::vector<double>& diag,
                                          const std::vector<double>& upper,
                                          const std::vector<double>& x) {
        size_t n = diag.size();
        std::vector<double> result(n, 0.0);

        // First row
        result[0] = diag[0] * x[0] + upper[0] * x[1];

        // Interior rows
        for (size_t i = 1; i < n - 1; i++) {
            result[i] = lower[i-1] * x[i-1] + diag[i] * x[i] + upper[i] * x[i+1];
        }

        // Last row
        result[n-1] = lower[n-2] * x[n-2] + diag[n-1] * x[n-1];

        return result;
    }
};

// Test 2x2 system
TEST_F(TridiagonalTest, Size2System) {
    // System: [2  1] [x1]   [3]
    //         [1  3] [x2] = [4]
    // Solution: x1 = 1, x2 = 1

    std::vector<double> lower = {1.0};
    std::vector<double> diag = {2.0, 3.0};
    std::vector<double> upper = {1.0};
    std::vector<double> rhs = {3.0, 4.0};
    std::vector<double> solution(2);

    solve_tridiagonal(2, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    EXPECT_NEAR(solution[0], 1.0, tolerance);
    EXPECT_NEAR(solution[1], 1.0, tolerance);
}

// Test 3x3 system
TEST_F(TridiagonalTest, Size3System) {
    // System: [2  1  0] [x1]   [5]
    //         [1  2  1] [x2] = [6]
    //         [0  1  2] [x3]   [5]
    // Solution: x1 = 2, x2 = 1, x3 = 2

    std::vector<double> lower = {1.0, 1.0};
    std::vector<double> diag = {2.0, 2.0, 2.0};
    std::vector<double> upper = {1.0, 1.0};
    std::vector<double> rhs = {5.0, 6.0, 5.0};
    std::vector<double> solution(3);

    solve_tridiagonal(3, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    EXPECT_NEAR(solution[0], 2.0, tolerance);
    EXPECT_NEAR(solution[1], 1.0, tolerance);
    EXPECT_NEAR(solution[2], 2.0, tolerance);
}

// Test diagonal matrix (special case)
TEST_F(TridiagonalTest, DiagonalMatrix) {
    size_t n = 5;
    std::vector<double> lower(n-1, 0.0);
    std::vector<double> diag = {2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> upper(n-1, 0.0);
    std::vector<double> rhs = {4.0, 9.0, 12.0, 15.0, 18.0};
    std::vector<double> solution(n);

    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Solution should be rhs[i] / diag[i]
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(solution[i], rhs[i] / diag[i], tolerance);
    }
}

// Test identity matrix
TEST_F(TridiagonalTest, IdentityMatrix) {
    size_t n = 5;
    std::vector<double> lower(n-1, 0.0);
    std::vector<double> diag(n, 1.0);
    std::vector<double> upper(n-1, 0.0);
    std::vector<double> rhs = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> solution(n);

    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Solution should equal rhs
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(solution[i], rhs[i], tolerance);
    }
}

// Test symmetric positive definite matrix
TEST_F(TridiagonalTest, SymmetricPositiveDefinite) {
    // Matrix from discrete Laplacian: diag=2, off-diag=-1
    size_t n = 10;
    std::vector<double> lower(n-1, -1.0);
    std::vector<double> diag(n, 2.0);
    std::vector<double> upper(n-1, -1.0);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);

    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Verify: A*x = b
    std::vector<double> b_check = matvec_tridiagonal(lower, diag, upper, solution);

    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(b_check[i], rhs[i], tolerance);
    }
}

// Test with known analytical solution
TEST_F(TridiagonalTest, AnalyticalSolution) {
    // Set up system where we know x = [1, 2, 3, 4, 5]
    size_t n = 5;
    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0, 5.0};

    std::vector<double> lower = {1.0, 2.0, 1.0, 2.0};
    std::vector<double> diag = {4.0, 5.0, 6.0, 5.0, 4.0};
    std::vector<double> upper = {2.0, 1.0, 2.0, 1.0};

    // Compute rhs = A * x_true
    std::vector<double> rhs = matvec_tridiagonal(lower, diag, upper, x_true);

    // Solve system
    std::vector<double> solution(n);
    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Should recover x_true
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(solution[i], x_true[i], tolerance);
    }
}

// Test larger system
TEST_F(TridiagonalTest, LargeSystem) {
    size_t n = 1000;
    std::vector<double> lower(n-1, -1.0);
    std::vector<double> diag(n, 3.0);
    std::vector<double> upper(n-1, -1.0);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);

    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Verify residual is small
    std::vector<double> b_check = matvec_tridiagonal(lower, diag, upper, solution);

    double max_error = 0.0;
    for (size_t i = 0; i < n; i++) {
        max_error = std::max(max_error, std::abs(b_check[i] - rhs[i]));
    }

    EXPECT_LT(max_error, 1e-8);
}

// Test non-uniform coefficients
TEST_F(TridiagonalTest, NonUniformCoefficients) {
    size_t n = 7;
    std::vector<double> lower = {0.5, 1.0, 1.5, 2.0, 1.0, 0.5};
    std::vector<double> diag = {5.0, 6.0, 7.0, 8.0, 7.0, 6.0, 5.0};
    std::vector<double> upper = {1.0, 1.5, 2.0, 1.5, 1.0, 0.5};
    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0};

    // Compute rhs
    std::vector<double> rhs = matvec_tridiagonal(lower, diag, upper, x_true);

    // Solve
    std::vector<double> solution(n);
    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Verify
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(solution[i], x_true[i], tolerance);
    }
}

// Test with negative coefficients
TEST_F(TridiagonalTest, NegativeCoefficients) {
    size_t n = 5;
    std::vector<double> lower = {-1.0, -0.5, -1.0, -0.5};
    std::vector<double> diag = {3.0, 3.0, 4.0, 3.0, 3.0};
    std::vector<double> upper = {-0.5, -1.0, -0.5, -1.0};
    std::vector<double> x_true = {1.0, 1.0, 1.0, 1.0, 1.0};

    std::vector<double> rhs = matvec_tridiagonal(lower, diag, upper, x_true);

    std::vector<double> solution(n);
    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(solution[i], x_true[i], tolerance);
    }
}

// Test diagonally dominant matrix (guaranteed stability)
TEST_F(TridiagonalTest, DiagonallyDominant) {
    size_t n = 10;
    std::vector<double> lower(n-1, 1.0);
    std::vector<double> diag(n, 10.0);  // |diag| > |lower| + |upper|
    std::vector<double> upper(n-1, 1.0);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);

    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Verify solution
    std::vector<double> b_check = matvec_tridiagonal(lower, diag, upper, solution);

    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(b_check[i], rhs[i], tolerance);
    }
}

// Test oscillating right-hand side
TEST_F(TridiagonalTest, OscillatingRHS) {
    size_t n = 20;
    std::vector<double> lower(n-1, -1.0);
    std::vector<double> diag(n, 3.0);
    std::vector<double> upper(n-1, -1.0);
    std::vector<double> rhs(n);

    // Oscillating rhs
    for (size_t i = 0; i < n; i++) {
        rhs[i] = std::sin(2.0 * M_PI * i / n);
    }

    std::vector<double> solution(n);
    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Verify
    std::vector<double> b_check = matvec_tridiagonal(lower, diag, upper, solution);

    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(b_check[i], rhs[i], tolerance);
    }
}

// Numerical stability: Very small off-diagonals
TEST_F(TridiagonalTest, SmallOffDiagonals) {
    size_t n = 5;
    std::vector<double> lower(n-1, 1e-10);
    std::vector<double> diag(n, 1.0);
    std::vector<double> upper(n-1, 1e-10);
    std::vector<double> rhs = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> solution(n);

    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Should be close to diagonal solution (relaxed tolerance for numerical precision)
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(solution[i], rhs[i], 1e-8);
    }
}

// Numerical stability: Very large diagonal
TEST_F(TridiagonalTest, LargeDiagonal) {
    size_t n = 5;
    std::vector<double> lower(n-1, 1.0);
    std::vector<double> diag(n, 1e10);
    std::vector<double> upper(n-1, 1.0);
    std::vector<double> rhs = {1e10, 2e10, 3e10, 4e10, 5e10};
    std::vector<double> solution(n);

    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Should be close to rhs/diag ≈ [1, 2, 3, 4, 5] (relaxed for numerical precision)
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(solution[i], (double)(i + 1), 1e-2);
    }
}

// Test with zero rhs
TEST_F(TridiagonalTest, ZeroRHS) {
    size_t n = 5;
    std::vector<double> lower(n-1, -1.0);
    std::vector<double> diag(n, 2.0);
    std::vector<double> upper(n-1, -1.0);
    std::vector<double> rhs(n, 0.0);
    std::vector<double> solution(n);

    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Solution should be zero
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(solution[i], 0.0, tolerance);
    }
}

// Test consistency: multiple solves with same matrix
TEST_F(TridiagonalTest, MultipleConsistentSolves) {
    size_t n = 5;
    std::vector<double> lower(n-1, -1.0);
    std::vector<double> diag(n, 3.0);
    std::vector<double> upper(n-1, -1.0);
    std::vector<double> rhs = {1.0, 2.0, 3.0, 4.0, 5.0};

    std::vector<double> solution1(n);
    std::vector<double> solution2(n);

    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution1.data(), nullptr);
    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution2.data(), nullptr);

    // Solutions should be identical
    for (size_t i = 0; i < n; i++) {
        EXPECT_DOUBLE_EQ(solution1[i], solution2[i]);
    }
}

// Test heat equation discretization matrix
TEST_F(TridiagonalTest, HeatEquationMatrix) {
    // Typical Crank-Nicolson matrix: (I + 0.5*dt*A)
    size_t n = 50;
    double dt = 0.001;
    double dx = 0.02;
    double alpha = dt / (dx * dx);

    std::vector<double> lower(n-1, -0.5 * alpha);
    std::vector<double> diag(n, 1.0 + alpha);
    std::vector<double> upper(n-1, -0.5 * alpha);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);

    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Verify
    std::vector<double> b_check = matvec_tridiagonal(lower, diag, upper, solution);

    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(b_check[i], rhs[i], tolerance);
    }
}

// Test Black-Scholes discretization matrix
TEST_F(TridiagonalTest, BlackScholesMatrix) {
    size_t n = 100;
    double dt = 0.01;
    double r = 0.05;
    double sigma = 0.2;

    // Simplified coefficients
    std::vector<double> lower(n-1);
    std::vector<double> diag(n);
    std::vector<double> upper(n-1);

    for (size_t i = 0; i < n-1; i++) {
        lower[i] = -0.25 * dt * (sigma * sigma * (i+1) * (i+1) - r * (i+1));
        upper[i] = -0.25 * dt * (sigma * sigma * (i+1) * (i+1) + r * (i+1));
    }

    for (size_t i = 0; i < n; i++) {
        diag[i] = 1.0 + 0.5 * dt * (r + sigma * sigma * i * i);
    }

    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);

    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Verify no NaN or Inf
    for (size_t i = 0; i < n; i++) {
        EXPECT_FALSE(std::isnan(solution[i]));
        EXPECT_FALSE(std::isinf(solution[i]));
    }
}

// Stress test: Weakly diagonally dominant
TEST_F(TridiagonalTest, WeaklyDiagonallyDominant) {
    size_t n = 10;
    std::vector<double> lower(n-1, 0.49);
    std::vector<double> diag(n, 1.0);
    std::vector<double> upper(n-1, 0.49);
    std::vector<double> x_true(n, 1.0);

    std::vector<double> rhs = matvec_tridiagonal(lower, diag, upper, x_true);

    std::vector<double> solution(n);
    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(solution[i], x_true[i], 1e-8);
    }
}

// Stress test: Near-singular matrix (but still solvable)
TEST_F(TridiagonalTest, NearSingular) {
    size_t n = 5;
    std::vector<double> lower(n-1, -1.0);
    std::vector<double> diag = {2.0, 2.0, 2.0, 2.0, 2.000001};  // Almost singular
    std::vector<double> upper(n-1, -1.0);
    std::vector<double> rhs = {1.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<double> solution(n);

    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Verify residual
    std::vector<double> b_check = matvec_tridiagonal(lower, diag, upper, solution);

    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(b_check[i], rhs[i], 1e-6);
    }
}

// Stress test: Alternating signs in rhs
TEST_F(TridiagonalTest, AlternatingRHS) {
    size_t n = 10;
    std::vector<double> lower(n-1, -1.0);
    std::vector<double> diag(n, 3.0);
    std::vector<double> upper(n-1, -1.0);
    std::vector<double> rhs(n);

    for (size_t i = 0; i < n; i++) {
        rhs[i] = (i % 2 == 0) ? 1.0 : -1.0;
    }

    std::vector<double> solution(n);
    solve_tridiagonal(n, lower.data(), diag.data(), upper.data(), rhs.data(), solution.data(), nullptr);

    // Verify
    std::vector<double> b_check = matvec_tridiagonal(lower, diag, upper, solution);

    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(b_check[i], rhs[i], tolerance);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
