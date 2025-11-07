/**
 * @file pentadiagonal_test.cc
 * @brief Unit tests for pentadiagonal solver
 *
 * Validates:
 * - Correctness against known solutions
 * - Tridiagonal case (comparison with ThomasSolver)
 * - B-spline collocation matrices
 * - Symmetric positive definite systems
 * - Error handling
 * - Performance benchmarks
 */

#include "src/pentadiagonal_solver.hpp"
#include "src/thomas_solver.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <chrono>

using namespace mango;

namespace {

constexpr double kTolerance = 1e-10;

/// Helper: Create a simple pentadiagonal test system
/// Diagonally dominant for stability
struct SimplePentadiagonalSystem {
    std::vector<double> e{0.2, 0.2, 0.2};        // 2nd subdiagonal
    std::vector<double> a{0.5, 0.5, 0.5, 0.5};   // 1st subdiagonal
    std::vector<double> b{10.0, 10.0, 10.0, 10.0, 10.0};  // Main diagonal
    std::vector<double> c{0.5, 0.5, 0.5, 0.5};   // 1st superdiagonal
    std::vector<double> d{0.2, 0.2, 0.2};        // 2nd superdiagonal
    std::vector<double> rhs{1.0, 2.0, 3.0, 4.0, 5.0};
    size_t n = 5;
};

/// Helper: Create tridiagonal system as pentadiagonal (zeros in e and d)
struct TridiagonalAsPentaSystem {
    std::vector<double> e{0.0, 0.0};          // Zero 2nd subdiagonal
    std::vector<double> a{1.0, 1.0, 1.0};     // 1st subdiagonal
    std::vector<double> b{2.0, 2.0, 2.0, 2.0}; // Main diagonal
    std::vector<double> c{1.0, 1.0, 1.0};     // 1st superdiagonal
    std::vector<double> d{0.0, 0.0};          // Zero 2nd superdiagonal
    std::vector<double> rhs{1.0, 2.0, 3.0, 4.0};
    size_t n = 4;
};

}  // namespace

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(PentadiagonalTest, SimpleSolve) {
    SimplePentadiagonalSystem sys;

    std::vector<double> solution(sys.n);

    auto result = solve_pentadiagonal_alloc<double>(
        std::span{sys.e},
        std::span{sys.a},
        std::span{sys.b},
        std::span{sys.c},
        std::span{sys.d},
        std::span{sys.rhs},
        std::span{solution});

    ASSERT_TRUE(result.ok()) << result.message();

    // Verify: A*x = b by reconstructing matrix-vector product
    std::vector<double> Ax(sys.n, 0.0);

    for (size_t i = 0; i < sys.n; ++i) {
        // Main diagonal
        Ax[i] += sys.b[i] * solution[i];

        // 1st subdiagonal
        if (i >= 1) {
            Ax[i] += sys.a[i-1] * solution[i-1];
        }

        // 2nd subdiagonal
        if (i >= 2) {
            Ax[i] += sys.e[i-2] * solution[i-2];
        }

        // 1st superdiagonal
        if (i < sys.n - 1) {
            Ax[i] += sys.c[i] * solution[i+1];
        }

        // 2nd superdiagonal
        if (i < sys.n - 2) {
            Ax[i] += sys.d[i] * solution[i+2];
        }
    }

    // Check Ax ≈ b
    for (size_t i = 0; i < sys.n; ++i) {
        EXPECT_NEAR(Ax[i], sys.rhs[i], kTolerance)
            << "Residual too large at index " << i;
    }
}

TEST(PentadiagonalTest, TridiagonalVsThomasSolver) {
    TridiagonalAsPentaSystem sys;

    // Solve using pentadiagonal solver
    std::vector<double> penta_solution(sys.n);
    auto penta_result = solve_pentadiagonal_alloc<double>(
        std::span{sys.e},
        std::span{sys.a},
        std::span{sys.b},
        std::span{sys.c},
        std::span{sys.d},
        std::span{sys.rhs},
        std::span{penta_solution});

    ASSERT_TRUE(penta_result.ok()) << penta_result.message();

    // Solve using Thomas solver for comparison
    std::vector<double> thomas_solution(sys.n);
    auto thomas_result = solve_thomas_alloc<double>(
        std::span{sys.a},
        std::span{sys.b},
        std::span{sys.c},
        std::span{sys.rhs},
        std::span{thomas_solution});

    ASSERT_TRUE(thomas_result.ok()) << thomas_result.message();

    // Compare solutions
    for (size_t i = 0; i < sys.n; ++i) {
        EXPECT_NEAR(penta_solution[i], thomas_solution[i], kTolerance)
            << "Mismatch with ThomasSolver at index " << i;
    }
}

// ============================================================================
// B-Spline Collocation Matrix Tests
// ============================================================================

TEST(PentadiagonalTest, BSplineCollocationMatrix) {
    // B-spline collocation matrices are symmetric positive definite
    // with specific band structure from basis function overlap

    const size_t n = 20;

    // Create a symmetric positive definite pentadiagonal matrix
    std::vector<double> e(n-2, 0.1);
    std::vector<double> a(n-1, 0.25);
    std::vector<double> b(n, 1.0);
    std::vector<double> c(n-1, 0.25);  // Symmetric
    std::vector<double> d(n-2, 0.1);   // Symmetric

    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);

    auto result = solve_pentadiagonal_alloc<double>(
        std::span{e},
        std::span{a},
        std::span{b},
        std::span{c},
        std::span{d},
        std::span{rhs},
        std::span{solution});

    ASSERT_TRUE(result.ok()) << result.message();

    // Verify residual
    std::vector<double> Ax(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        Ax[i] += b[i] * solution[i];

        if (i >= 1) Ax[i] += a[i-1] * solution[i-1];
        if (i >= 2) Ax[i] += e[i-2] * solution[i-2];
        if (i < n - 1) Ax[i] += c[i] * solution[i+1];
        if (i < n - 2) Ax[i] += d[i] * solution[i+2];
    }

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(Ax[i], rhs[i], kTolerance)
            << "Residual error at index " << i;
    }
}

TEST(PentadiagonalTest, AsymmetricPentadiagonal) {
    // Test with non-symmetric bands
    const size_t n = 10;

    std::vector<double> e(n-2, 0.1);
    std::vector<double> a(n-1, 0.3);
    std::vector<double> b(n, 2.0);
    std::vector<double> c(n-1, 0.2);   // Different from a
    std::vector<double> d(n-2, 0.05);  // Different from e

    std::vector<double> rhs(n);
    for (size_t i = 0; i < n; ++i) {
        rhs[i] = static_cast<double>(i + 1);
    }

    std::vector<double> solution(n);

    auto result = solve_pentadiagonal_alloc<double>(
        std::span{e},
        std::span{a},
        std::span{b},
        std::span{c},
        std::span{d},
        std::span{rhs},
        std::span{solution});

    ASSERT_TRUE(result.ok()) << result.message();

    // Verify residual
    std::vector<double> Ax(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        Ax[i] = b[i] * solution[i];

        if (i >= 1) Ax[i] += a[i-1] * solution[i-1];
        if (i >= 2) Ax[i] += e[i-2] * solution[i-2];
        if (i < n - 1) Ax[i] += c[i] * solution[i+1];
        if (i < n - 2) Ax[i] += d[i] * solution[i+2];
    }

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(Ax[i], rhs[i], kTolerance)
            << "Residual error at index " << i;
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(PentadiagonalTest, SingularMatrix) {
    const size_t n = 5;

    std::vector<double> e(n-2, 0.1);
    std::vector<double> a(n-1, 0.1);
    std::vector<double> b{1.0, 0.0, 1.0, 1.0, 1.0};  // Zero diagonal!
    std::vector<double> c(n-1, 0.1);
    std::vector<double> d(n-2, 0.1);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);

    auto result = solve_pentadiagonal_alloc<double>(
        std::span{e},
        std::span{a},
        std::span{b},
        std::span{c},
        std::span{d},
        std::span{rhs},
        std::span{solution});

    EXPECT_FALSE(result.ok());
    EXPECT_NE(result.message().find("Singular"), std::string_view::npos);
}

TEST(PentadiagonalTest, InvalidDimensions) {
    std::vector<double> e{0.1};  // Wrong size
    std::vector<double> a{0.1, 0.1, 0.1};
    std::vector<double> b{1.0, 1.0, 1.0, 1.0};
    std::vector<double> c{0.1, 0.1, 0.1};
    std::vector<double> d{0.1, 0.1};
    std::vector<double> rhs{1.0, 1.0, 1.0, 1.0};
    std::vector<double> solution(4);

    auto result = solve_pentadiagonal_alloc<double>(
        std::span{e},
        std::span{a},
        std::span{b},
        std::span{c},
        std::span{d},
        std::span{rhs},
        std::span{solution});

    EXPECT_FALSE(result.ok());
}

TEST(PentadiagonalTest, DiagonalDominanceCheck) {
    const size_t n = 5;

    // Create non-diagonally dominant matrix
    std::vector<double> e(n-2, 2.0);  // Large off-diagonal
    std::vector<double> a(n-1, 2.0);
    std::vector<double> b(n, 1.0);    // Small diagonal
    std::vector<double> c(n-1, 2.0);
    std::vector<double> d(n-2, 2.0);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);

    PentadiagonalConfig<double> config;
    config.check_diagonal_dominance = true;

    std::vector<double> workspace(5 * n);
    auto result = solve_pentadiagonal<double>(
        std::span{e},
        std::span{a},
        std::span{b},
        std::span{c},
        std::span{d},
        std::span{rhs},
        std::span{solution},
        std::span{workspace},
        config);

    EXPECT_FALSE(result.ok());
    EXPECT_NE(result.message().find("dominant"), std::string_view::npos);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(PentadiagonalTest, SingleElement) {
    std::vector<double> e{};
    std::vector<double> a{};
    std::vector<double> b{5.0};
    std::vector<double> c{};
    std::vector<double> d{};
    std::vector<double> rhs{10.0};
    std::vector<double> solution(1);

    auto result = solve_pentadiagonal_alloc<double>(
        std::span{e},
        std::span{a},
        std::span{b},
        std::span{c},
        std::span{d},
        std::span{rhs},
        std::span{solution});

    ASSERT_TRUE(result.ok()) << result.message();
    EXPECT_NEAR(solution[0], 2.0, kTolerance);
}

TEST(PentadiagonalTest, TwoElements) {
    std::vector<double> e{};
    std::vector<double> a{1.0};
    std::vector<double> b{2.0, 2.0};
    std::vector<double> c{1.0};
    std::vector<double> d{};
    std::vector<double> rhs{3.0, 3.0};
    std::vector<double> solution(2);

    auto result = solve_pentadiagonal_alloc<double>(
        std::span{e},
        std::span{a},
        std::span{b},
        std::span{c},
        std::span{d},
        std::span{rhs},
        std::span{solution});

    ASSERT_TRUE(result.ok()) << result.message();

    // Verify: [2 1][x0] = [3]
    //         [1 2][x1]   [3]
    // Solution: x0=x1=1
    EXPECT_NEAR(solution[0], 1.0, kTolerance);
    EXPECT_NEAR(solution[1], 1.0, kTolerance);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(PentadiagonalTest, PerformanceLargeSystem) {
    const size_t n = 1000;

    // Create diagonally dominant matrix
    std::vector<double> e(n-2, 0.1);
    std::vector<double> a(n-1, 0.2);
    std::vector<double> b(n, 2.0);
    std::vector<double> c(n-1, 0.2);
    std::vector<double> d(n-2, 0.1);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = solve_pentadiagonal_alloc<double>(
        std::span{e},
        std::span{a},
        std::span{b},
        std::span{c},
        std::span{d},
        std::span{rhs},
        std::span{solution});
    auto end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(result.ok()) << result.message();

    const auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    std::cout << "Large system solve (n=" << n << "): " << time_us << " µs\n";

    // Target: <500µs for n=1000
    EXPECT_LT(time_us, 500)
        << "Large system solve too slow";
}

TEST(PentadiagonalTest, Performance50PointBSplineFit) {
    // Simulate 50-point B-spline fit (typical for separable fitter)
    const size_t n = 50;

    std::vector<double> e(n-2, 0.15);
    std::vector<double> a(n-1, 0.25);
    std::vector<double> b(n, 1.0);
    std::vector<double> c(n-1, 0.25);
    std::vector<double> d(n-2, 0.15);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = solve_pentadiagonal_alloc<double>(
        std::span{e},
        std::span{a},
        std::span{b},
        std::span{c},
        std::span{d},
        std::span{rhs},
        std::span{solution});
    auto end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(result.ok()) << result.message();

    const auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    std::cout << "50-point fit solve: " << time_us << " µs\n";

    // Target: <50µs for 50-point fit
    EXPECT_LT(time_us, 50)
        << "50-point solve exceeds target";
}

TEST(PentadiagonalTest, ReusableWorkspace) {
    const size_t n = 100;

    std::vector<double> e(n-2, 0.1);
    std::vector<double> a(n-1, 0.2);
    std::vector<double> b(n, 2.0);
    std::vector<double> c(n-1, 0.2);
    std::vector<double> d(n-2, 0.1);
    std::vector<double> solution(n);

    PentadiagonalWorkspace<double> workspace(n);

    // Solve multiple times with same workspace
    for (int trial = 0; trial < 10; ++trial) {
        std::vector<double> rhs(n);
        for (size_t i = 0; i < n; ++i) {
            rhs[i] = static_cast<double>(trial + i);
        }

        auto result = solve_pentadiagonal<double>(
            std::span{e},
            std::span{a},
            std::span{b},
            std::span{c},
            std::span{d},
            std::span{rhs},
            std::span{solution},
            workspace.get());

        ASSERT_TRUE(result.ok()) << "Trial " << trial << ": " << result.message();

        // Verify residual
        double max_residual = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double Ax_i = b[i] * solution[i];
            if (i >= 1) Ax_i += a[i-1] * solution[i-1];
            if (i >= 2) Ax_i += e[i-2] * solution[i-2];
            if (i < n - 1) Ax_i += c[i] * solution[i+1];
            if (i < n - 2) Ax_i += d[i] * solution[i+2];

            max_residual = std::max(max_residual, std::abs(Ax_i - rhs[i]));
        }

        EXPECT_LT(max_residual, kTolerance)
            << "Trial " << trial << " has large residual";
    }
}
