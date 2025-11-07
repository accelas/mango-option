/**
 * @file eigen_banded_solver_test.cc
 * @brief Unit tests for Eigen-based banded linear system solver
 *
 * Validates:
 * - Tridiagonal system solving (comparison with ThomasSolver)
 * - Pentadiagonal system solving
 * - B-spline collocation matrices
 * - Error handling and edge cases
 * - Performance benchmarks
 */

#include "src/eigen_banded_solver.hpp"
#include "src/thomas_solver.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <chrono>

using namespace mango;

namespace {

constexpr double kTolerance = 1e-10;

/// Compute residual norm ||Ax - b|| / ||b||
double compute_relative_residual(
    size_t n,
    const double* subdiag2,
    const double* subdiag1,
    const double* diag,
    const double* superdiag1,
    const double* superdiag2,
    const std::vector<double>& x,
    const std::vector<double>& b)
{
    std::vector<double> Ax(n, 0.0);

    // Compute Ax
    for (size_t i = 0; i < n; ++i) {
        Ax[i] += diag[i] * x[i];

        if (i > 0 && subdiag1) {
            Ax[i] += subdiag1[i-1] * x[i-1];
        }
        if (i > 1 && subdiag2) {
            Ax[i] += subdiag2[i-2] * x[i-2];
        }
        if (i < n-1 && superdiag1) {
            Ax[i] += superdiag1[i] * x[i+1];
        }
        if (i < n-2 && superdiag2) {
            Ax[i] += superdiag2[i] * x[i+2];
        }
    }

    // Compute ||Ax - b||
    double residual_norm = 0.0;
    double b_norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = Ax[i] - b[i];
        residual_norm += diff * diff;
        b_norm += b[i] * b[i];
    }

    residual_norm = std::sqrt(residual_norm);
    b_norm = std::sqrt(b_norm);

    return (b_norm > 1e-14) ? residual_norm / b_norm : residual_norm;
}

}  // namespace

// ============================================================================
// Construction and Error Handling Tests
// ============================================================================

TEST(EigenBandedSolverTest, Construction) {
    EXPECT_NO_THROW({
        EigenBandedSolver solver(100, 2, 2);
        EXPECT_EQ(solver.size(), 100UL);
        EXPECT_EQ(solver.lower_bandwidth(), 2UL);
        EXPECT_EQ(solver.upper_bandwidth(), 2UL);
    });
}

TEST(EigenBandedSolverTest, InvalidConstruction) {
    EXPECT_THROW(EigenBandedSolver(0, 1, 1), std::invalid_argument);  // Zero size
    EXPECT_THROW(EigenBandedSolver(10, 10, 1), std::invalid_argument);  // kl >= n
    EXPECT_THROW(EigenBandedSolver(10, 1, 10), std::invalid_argument);  // ku >= n
}

TEST(EigenBandedSolverTest, SolveBeforeSetMatrix) {
    EigenBandedSolver solver(10, 2, 2);

    std::vector<double> rhs(10, 1.0);
    auto result = solver.solve(rhs);

    EXPECT_FALSE(result.success);
}

// ============================================================================
// Tridiagonal System Tests
// ============================================================================

TEST(EigenBandedSolverTest, TridiagonalSimple) {
    // Solve tridiagonal system with known solution
    // Matrix: [2 -1  0  0]    [x0]   [1]
    //         [-1 2 -1  0]    [x1] = [0]
    //         [0 -1  2 -1]    [x2]   [0]
    //         [0  0 -1  2]    [x3]   [1]

    constexpr size_t n = 4;
    std::vector<double> subdiag(n-1, -1.0);
    std::vector<double> diag(n, 2.0);
    std::vector<double> superdiag(n-1, -1.0);
    std::vector<double> rhs = {1.0, 0.0, 0.0, 1.0};

    auto result = solve_tridiagonal(n, subdiag.data(), diag.data(),
                                    superdiag.data(), rhs);

    ASSERT_TRUE(result.success) << "Error: " << result.error_message;
    EXPECT_EQ(result.solution.size(), n);

    // Verify residual
    double residual = compute_relative_residual(
        n, nullptr, subdiag.data(), diag.data(), superdiag.data(), nullptr,
        result.solution, rhs);

    EXPECT_LT(residual, 1e-10) << "Large residual: " << residual;
}

TEST(EigenBandedSolverTest, TridiagonalVsThomasSolver) {
    // Compare Eigen solver with ThomasSolver for tridiagonal system
    constexpr size_t n = 50;

    std::vector<double> subdiag(n-1);
    std::vector<double> diag(n);
    std::vector<double> superdiag(n-1);
    std::vector<double> rhs(n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Create diagonally dominant matrix
    for (size_t i = 0; i < n; ++i) {
        diag[i] = 10.0 + dist(rng);
        rhs[i] = dist(rng);

        if (i < n-1) {
            subdiag[i] = dist(rng);
            superdiag[i] = dist(rng);
        }
    }

    // Solve with Eigen
    auto eigen_result = solve_tridiagonal(n, subdiag.data(), diag.data(),
                                          superdiag.data(), rhs);
    ASSERT_TRUE(eigen_result.success);

    // Solve with Thomas
    ThomasSolver<double> thomas_solver;
    auto thomas_result = thomas_solver.solve(n, subdiag.data(), diag.data(),
                                             superdiag.data(), rhs.data());
    ASSERT_TRUE(thomas_result.has_value());

    // Compare solutions
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(eigen_result.solution[i], thomas_result.value()[i], 1e-10)
            << "Mismatch at index " << i;
    }
}

// ============================================================================
// Pentadiagonal System Tests
// ============================================================================

TEST(EigenBandedSolverTest, PentadiagonalSimple) {
    // Solve small pentadiagonal system
    constexpr size_t n = 6;

    std::vector<double> subdiag2(n-2, 0.1);
    std::vector<double> subdiag1(n-1, -1.0);
    std::vector<double> diag(n, 4.0);
    std::vector<double> superdiag1(n-1, -1.0);
    std::vector<double> superdiag2(n-2, 0.1);
    std::vector<double> rhs(n, 1.0);

    auto result = solve_pentadiagonal(n, subdiag2.data(), subdiag1.data(),
                                      diag.data(), superdiag1.data(),
                                      superdiag2.data(), rhs);

    ASSERT_TRUE(result.success) << "Error: " << result.error_message;

    // Verify residual
    double residual = compute_relative_residual(
        n, subdiag2.data(), subdiag1.data(), diag.data(),
        superdiag1.data(), superdiag2.data(), result.solution, rhs);

    EXPECT_LT(residual, 1e-10) << "Large residual: " << residual;
}

TEST(EigenBandedSolverTest, PentadiagonalDiagonallyDominant) {
    // Test with randomly generated diagonally dominant pentadiagonal matrix
    constexpr size_t n = 100;

    std::vector<double> subdiag2(n-2);
    std::vector<double> subdiag1(n-1);
    std::vector<double> diag(n);
    std::vector<double> superdiag1(n-1);
    std::vector<double> superdiag2(n-2);
    std::vector<double> rhs(n);

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < n; ++i) {
        // Make diagonally dominant
        diag[i] = 20.0 + dist(rng);
        rhs[i] = dist(rng);

        if (i < n-1) {
            subdiag1[i] = dist(rng);
            superdiag1[i] = dist(rng);
        }
        if (i < n-2) {
            subdiag2[i] = dist(rng);
            superdiag2[i] = dist(rng);
        }
    }

    auto result = solve_pentadiagonal(n, subdiag2.data(), subdiag1.data(),
                                      diag.data(), superdiag1.data(),
                                      superdiag2.data(), rhs);

    ASSERT_TRUE(result.success) << "Error: " << result.error_message;

    double residual = compute_relative_residual(
        n, subdiag2.data(), subdiag1.data(), diag.data(),
        superdiag1.data(), superdiag2.data(), result.solution, rhs);

    EXPECT_LT(residual, 1e-8) << "Large residual: " << residual;
}

// ============================================================================
// B-Spline Collocation Matrix Tests
// ============================================================================

TEST(EigenBandedSolverTest, BSplineCollocationMatrix) {
    // Test with realistic B-spline collocation matrix structure
    // Cubic B-splines produce pentadiagonal systems
    constexpr size_t n = 50;

    std::vector<double> subdiag2(n-2, 1.0);
    std::vector<double> subdiag1(n-1, 4.0);
    std::vector<double> diag(n, 6.0);
    std::vector<double> superdiag1(n-1, 4.0);
    std::vector<double> superdiag2(n-2, 1.0);

    // Modify boundaries (clamped B-splines)
    diag[0] = diag[n-1] = 1.0;
    superdiag1[0] = subdiag1[n-2] = 0.0;

    std::vector<double> rhs(n);
    for (size_t i = 0; i < n; ++i) {
        rhs[i] = std::sin(2.0 * M_PI * i / (n - 1));
    }

    auto result = solve_pentadiagonal(n, subdiag2.data(), subdiag1.data(),
                                      diag.data(), superdiag1.data(),
                                      superdiag2.data(), rhs);

    ASSERT_TRUE(result.success) << "Error: " << result.error_message;

    double residual = compute_relative_residual(
        n, subdiag2.data(), subdiag1.data(), diag.data(),
        superdiag1.data(), superdiag2.data(), result.solution, rhs);

    EXPECT_LT(residual, 1e-10) << "Large residual: " << residual;
}

// ============================================================================
// Reuse and Multiple RHS Tests
// ============================================================================

TEST(EigenBandedSolverTest, ReuseFactorization) {
    // Test solving multiple RHS with same matrix (reuse factorization)
    constexpr size_t n = 30;

    std::vector<double> subdiag2(n-2, 0.5);
    std::vector<double> subdiag1(n-1, -2.0);
    std::vector<double> diag(n, 10.0);
    std::vector<double> superdiag1(n-1, -2.0);
    std::vector<double> superdiag2(n-2, 0.5);

    EigenBandedSolver solver(n, 2, 2);
    solver.set_pentadiagonal_matrix(subdiag2.data(), subdiag1.data(),
                                    diag.data(), superdiag1.data(),
                                    superdiag2.data());

    // Solve first RHS
    std::vector<double> rhs1(n, 1.0);
    auto result1 = solver.solve(rhs1);
    ASSERT_TRUE(result1.success);

    // Solve second RHS (should reuse factorization)
    std::vector<double> rhs2(n);
    for (size_t i = 0; i < n; ++i) {
        rhs2[i] = static_cast<double>(i);
    }
    auto result2 = solver.solve(rhs2);
    ASSERT_TRUE(result2.success);

    // Verify both solutions
    double residual1 = compute_relative_residual(
        n, subdiag2.data(), subdiag1.data(), diag.data(),
        superdiag1.data(), superdiag2.data(), result1.solution, rhs1);
    double residual2 = compute_relative_residual(
        n, subdiag2.data(), subdiag1.data(), diag.data(),
        superdiag1.data(), superdiag2.data(), result2.solution, rhs2);

    EXPECT_LT(residual1, 1e-10);
    EXPECT_LT(residual2, 1e-10);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(EigenBandedSolverTest, WrongRHSSize) {
    EigenBandedSolver solver(10, 2, 2);

    std::vector<double> diag(10, 1.0);
    solver.set_pentadiagonal_matrix(nullptr, nullptr, diag.data(), nullptr, nullptr);

    std::vector<double> rhs(5, 1.0);  // Wrong size!
    auto result = solver.solve(rhs);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
}

TEST(EigenBandedSolverTest, SingularMatrix) {
    // Test with singular matrix (all zeros)
    constexpr size_t n = 10;

    std::vector<double> diag(n, 0.0);
    std::vector<double> rhs(n, 1.0);

    auto result = solve_pentadiagonal(n, nullptr, nullptr, diag.data(),
                                      nullptr, nullptr, rhs);

    EXPECT_FALSE(result.success);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(EigenBandedSolverTest, PerformanceLarge) {
    // Benchmark large pentadiagonal system
    constexpr size_t n = 1000;

    std::vector<double> subdiag2(n-2, 0.1);
    std::vector<double> subdiag1(n-1, -1.0);
    std::vector<double> diag(n, 4.0);
    std::vector<double> superdiag1(n-1, -1.0);
    std::vector<double> superdiag2(n-2, 0.1);
    std::vector<double> rhs(n, 1.0);

    auto start = std::chrono::high_resolution_clock::now();

    auto result = solve_pentadiagonal(n, subdiag2.data(), subdiag1.data(),
                                      diag.data(), superdiag1.data(),
                                      superdiag2.data(), rhs);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    ASSERT_TRUE(result.success);
    std::cout << "Eigen pentadiagonal solve (n=1000): " << duration_us << " µs\n";

    // Target: <100µs for n=1000
    EXPECT_LT(duration_us, 100)
        << "Performance regression: " << duration_us << " µs";
}

TEST(EigenBandedSolverTest, PerformanceSmall) {
    // Benchmark small system (typical for 1D B-spline fitting)
    constexpr size_t n = 50;

    std::vector<double> subdiag2(n-2, 1.0);
    std::vector<double> subdiag1(n-1, 4.0);
    std::vector<double> diag(n, 6.0);
    std::vector<double> superdiag1(n-1, 4.0);
    std::vector<double> superdiag2(n-2, 1.0);
    std::vector<double> rhs(n, 1.0);

    constexpr int n_trials = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int trial = 0; trial < n_trials; ++trial) {
        auto result = solve_pentadiagonal(n, subdiag2.data(), subdiag1.data(),
                                          diag.data(), superdiag1.data(),
                                          superdiag2.data(), rhs);
        (void)result;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();
    double us_per_solve = static_cast<double>(duration_us) / n_trials;

    std::cout << "Eigen pentadiagonal solve (n=50): " << us_per_solve << " µs\n";

    // Target: <10µs for n=50
    EXPECT_LT(us_per_solve, 10.0)
        << "Performance regression: " << us_per_solve << " µs";
}
