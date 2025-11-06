/**
 * @file banded_lu_test.cc
 * @brief Unit tests for banded LU solver
 *
 * Validates:
 * - Correctness against known solutions
 * - Comparison with ThomasSolver for tridiagonal case
 * - B-spline collocation matrices (width-4 bands)
 * - Diagonal dominance handling
 * - Error conditions
 * - Performance benchmarks
 */

#include "src/banded_lu_solver.hpp"
#include "src/thomas_solver.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <chrono>
#include <numbers>

using namespace mango;

namespace {

constexpr double kTolerance = 1e-10;

/// Helper: Create a simple tridiagonal test system
/// Matrix: [2 1 0]  RHS: [1]  Solution: [0, 0, 1]
///         [1 2 1]        [1]
///         [0 1 2]        [3]
struct TridiagonalTestSystem {
    std::vector<double> lower{1.0, 1.0};
    std::vector<double> diag{2.0, 2.0, 2.0};
    std::vector<double> upper{1.0, 1.0};
    std::vector<double> rhs{1.0, 1.0, 3.0};
    std::vector<double> expected_solution{0.0, 0.0, 1.0};
    size_t n = 3;
};

/// Helper: Create a pentadiagonal test system (width-5, kl=2, ku=2)
/// Diagonally dominant for stability
struct PentadiagonalTestSystem {
    std::vector<std::vector<double>> lower{
        {1.0, 1.0, 1.0, 1.0},  // 1st subdiagonal
        {0.5, 0.5, 0.5}         // 2nd subdiagonal
    };
    std::vector<double> diag{10.0, 10.0, 10.0, 10.0, 10.0};
    std::vector<std::vector<double>> upper{
        {1.0, 1.0, 1.0, 1.0},  // 1st superdiagonal
        {0.5, 0.5, 0.5}         // 2nd superdiagonal
    };
    std::vector<double> rhs{1.0, 2.0, 3.0, 4.0, 5.0};
    size_t n = 5;
};

}  // namespace

// ============================================================================
// Construction and Basic Tests
// ============================================================================

TEST(BandedLUTest, Construction) {
    BandedLU<double> solver(10, 2, 2);  // Pentadiagonal, n=10

    EXPECT_EQ(solver.size(), 10UL);
    EXPECT_EQ(solver.lower_bandwidth(), 2UL);
    EXPECT_EQ(solver.upper_bandwidth(), 2UL);
    EXPECT_FALSE(solver.is_factorized());
}

TEST(BandedLUTest, TridiagonalVsThomasSolver) {
    TridiagonalTestSystem sys;

    // Solve using BandedLU (kl=1, ku=1 for tridiagonal)
    BandedLU<double> banded_solver(sys.n, 1, 1);

    std::vector<std::span<const double>> lower_bands = {
        std::span{sys.lower}
    };
    std::vector<std::span<const double>> upper_bands = {
        std::span{sys.upper}
    };

    auto factor_result = banded_solver.factorize(
        std::span{lower_bands},
        std::span{sys.diag},
        std::span{upper_bands});

    ASSERT_TRUE(factor_result.ok()) << factor_result.message();
    EXPECT_TRUE(banded_solver.is_factorized());

    std::vector<double> banded_solution(sys.n);
    auto solve_result = banded_solver.solve(std::span{sys.rhs}, std::span{banded_solution});
    ASSERT_TRUE(solve_result.ok()) << solve_result.message();

    // Solve using ThomasSolver for comparison
    std::vector<double> thomas_solution(sys.n);
    auto thomas_result = solve_thomas_alloc<double>(
        std::span{sys.lower},
        std::span{sys.diag},
        std::span{sys.upper},
        std::span{sys.rhs},
        std::span{thomas_solution});

    ASSERT_TRUE(thomas_result.ok()) << thomas_result.message();

    // Compare solutions
    for (size_t i = 0; i < sys.n; ++i) {
        EXPECT_NEAR(banded_solution[i], thomas_solution[i], kTolerance)
            << "Mismatch at index " << i;
    }

    // Also check against known solution
    for (size_t i = 0; i < sys.n; ++i) {
        EXPECT_NEAR(banded_solution[i], sys.expected_solution[i], kTolerance)
            << "Wrong solution at index " << i;
    }
}

TEST(BandedLUTest, PentadiagonalSolve) {
    PentadiagonalTestSystem sys;

    BandedLU<double> solver(sys.n, 2, 2);

    std::vector<std::span<const double>> lower_bands;
    for (auto& band : sys.lower) {
        lower_bands.push_back(std::span{band});
    }

    std::vector<std::span<const double>> upper_bands;
    for (auto& band : sys.upper) {
        upper_bands.push_back(std::span{band});
    }

    auto factor_result = solver.factorize(
        std::span{lower_bands},
        std::span{sys.diag},
        std::span{upper_bands});

    ASSERT_TRUE(factor_result.ok()) << factor_result.message();

    std::vector<double> solution(sys.n);
    auto solve_result = solver.solve(std::span{sys.rhs}, std::span{solution});
    ASSERT_TRUE(solve_result.ok()) << solve_result.message();

    // Verify: A*x = b by reconstructing matrix-vector product
    std::vector<double> Ax(sys.n, 0.0);

    for (size_t i = 0; i < sys.n; ++i) {
        // Diagonal
        Ax[i] += sys.diag[i] * solution[i];

        // Lower bands
        for (size_t k = 0; k < 2 && i >= k + 1; ++k) {
            Ax[i] += sys.lower[k][i - k - 1] * solution[i - k - 1];
        }

        // Upper bands
        for (size_t k = 0; k < 2 && i + k + 1 < sys.n; ++k) {
            Ax[i] += sys.upper[k][i] * solution[i + k + 1];
        }
    }

    // Check Ax ≈ b
    for (size_t i = 0; i < sys.n; ++i) {
        EXPECT_NEAR(Ax[i], sys.rhs[i], kTolerance)
            << "Residual too large at index " << i;
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(BandedLUTest, SingularMatrix) {
    // Create singular matrix (zero diagonal)
    std::vector<std::vector<double>> lower{{1.0, 1.0}};
    std::vector<double> diag{1.0, 0.0, 1.0};  // Zero in middle!
    std::vector<std::vector<double>> upper{{1.0, 1.0}};

    BandedLU<double> solver(3, 1, 1);

    std::vector<std::span<const double>> lower_spans;
    for (auto& band : lower) lower_spans.push_back(std::span{band});

    std::vector<std::span<const double>> upper_spans;
    for (auto& band : upper) upper_spans.push_back(std::span{band});

    auto result = solver.factorize(
        std::span{lower_spans},
        std::span{diag},
        std::span{upper_spans});

    EXPECT_FALSE(result.ok());
    EXPECT_FALSE(solver.is_factorized());
}

TEST(BandedLUTest, SolveWithoutFactorization) {
    BandedLU<double> solver(3, 1, 1);

    std::vector<double> rhs{1.0, 2.0, 3.0};
    std::vector<double> solution(3);

    auto result = solver.solve(std::span{rhs}, std::span{solution});

    EXPECT_FALSE(result.ok());
    EXPECT_NE(result.message().find("not factorized"), std::string_view::npos);
}

TEST(BandedLUTest, InvalidDimensions) {
    BandedLU<double> solver(5, 2, 2);

    std::vector<std::vector<double>> lower{{1.0}};  // Wrong size!
    std::vector<double> diag{1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<std::vector<double>> upper{{1.0}};

    std::vector<std::span<const double>> lower_spans{std::span{lower[0]}};
    std::vector<std::span<const double>> upper_spans{std::span{upper[0]}};

    // Should fail due to wrong number of bands
    auto result = solver.factorize(
        std::span{lower_spans},
        std::span{diag},
        std::span{upper_spans});

    EXPECT_FALSE(result.ok());
}

// ============================================================================
// B-Spline Collocation Matrix Tests
// ============================================================================

TEST(BandedLUTest, BSplineCollocationMatrix) {
    // B-spline collocation matrices have width-4 bands (kl=1, ku=2 typically)
    // Create a simple test case simulating B-spline basis overlap

    const size_t n = 20;
    BandedLU<double> solver(n, 1, 2);  // Asymmetric bands

    // Create a diagonally dominant test matrix
    std::vector<std::vector<double>> lower(1);
    lower[0].resize(n - 1, 0.25);

    std::vector<double> diag(n, 1.0);

    std::vector<std::vector<double>> upper(2);
    upper[0].resize(n - 1, 0.25);
    upper[1].resize(n - 2, 0.1);

    std::vector<std::span<const double>> lower_spans;
    for (auto& band : lower) lower_spans.push_back(std::span{band});

    std::vector<std::span<const double>> upper_spans;
    for (auto& band : upper) upper_spans.push_back(std::span{band});

    auto factor_result = solver.factorize(
        std::span{lower_spans},
        std::span{diag},
        std::span{upper_spans});

    ASSERT_TRUE(factor_result.ok()) << factor_result.message();

    // Solve with a known RHS
    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);

    auto solve_result = solver.solve(std::span{rhs}, std::span{solution});
    ASSERT_TRUE(solve_result.ok()) << solve_result.message();

    // Verify residual Ax - b
    std::vector<double> Ax(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        Ax[i] += diag[i] * solution[i];

        if (i > 0) {
            Ax[i] += lower[0][i - 1] * solution[i - 1];
        }

        if (i < n - 1) {
            Ax[i] += upper[0][i] * solution[i + 1];
        }

        if (i < n - 2) {
            Ax[i] += upper[1][i] * solution[i + 2];
        }
    }

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(Ax[i], rhs[i], kTolerance)
            << "Residual error at index " << i;
    }
}

// ============================================================================
// Diagonal Dominance Test
// ============================================================================

TEST(BandedLUTest, DiagonalDominanceCheck) {
    // Create a matrix that is NOT diagonally dominant
    std::vector<std::vector<double>> lower{{5.0, 5.0}};  // Large off-diagonal
    std::vector<double> diag{1.0, 1.0, 1.0};             // Small diagonal
    std::vector<std::vector<double>> upper{{5.0, 5.0}};

    BandedLU<double> solver(3, 1, 1);

    std::vector<std::span<const double>> lower_spans;
    for (auto& band : lower) lower_spans.push_back(std::span{band});

    std::vector<std::span<const double>> upper_spans;
    for (auto& band : upper) upper_spans.push_back(std::span{band});

    BandedLUConfig<double> config;
    config.check_diagonal_dominance = true;

    auto result = solver.factorize(
        std::span{lower_spans},
        std::span{diag},
        std::span{upper_spans},
        config);

    EXPECT_FALSE(result.ok());
    EXPECT_NE(result.message().find("dominant"), std::string_view::npos);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(BandedLUTest, PerformanceLargeSystem) {
    const size_t n = 1000;
    const size_t kl = 3;
    const size_t ku = 3;

    // Create diagonally dominant matrix
    std::vector<std::vector<double>> lower(kl);
    for (size_t k = 0; k < kl; ++k) {
        lower[k].resize(n - k - 1, 0.1);
    }

    std::vector<double> diag(n, 10.0);

    std::vector<std::vector<double>> upper(ku);
    for (size_t k = 0; k < ku; ++k) {
        upper[k].resize(n - k - 1, 0.1);
    }

    std::vector<std::span<const double>> lower_spans;
    for (auto& band : lower) lower_spans.push_back(std::span{band});

    std::vector<std::span<const double>> upper_spans;
    for (auto& band : upper) upper_spans.push_back(std::span{band});

    BandedLU<double> solver(n, kl, ku);

    // Benchmark factorization
    auto start = std::chrono::high_resolution_clock::now();
    auto factor_result = solver.factorize(
        std::span{lower_spans},
        std::span{diag},
        std::span{upper_spans});
    auto end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(factor_result.ok()) << factor_result.message();

    const auto factor_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    std::cout << "Factorization time (n=" << n << ", bw=7): "
              << factor_time_us << " µs\n";

    // Target: <5ms for n=1000, width=7
    EXPECT_LT(factor_time_us, 5000)
        << "Factorization too slow";

    // Benchmark solve
    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);

    start = std::chrono::high_resolution_clock::now();
    auto solve_result = solver.solve(std::span{rhs}, std::span{solution});
    end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(solve_result.ok()) << solve_result.message();

    const auto solve_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    std::cout << "Solve time (n=" << n << "): " << solve_time_us << " µs\n";

    // Target: <500µs for n=1000
    EXPECT_LT(solve_time_us, 500)
        << "Solve too slow";
}

TEST(BandedLUTest, Performance50PointBSplineFit) {
    // Simulate 50-point B-spline fit (typical for separable fitter)
    const size_t n = 50;
    const size_t kl = 3;
    const size_t ku = 3;

    std::vector<std::vector<double>> lower(kl);
    for (size_t k = 0; k < kl; ++k) {
        lower[k].resize(n - k - 1, 0.2);
    }

    std::vector<double> diag(n, 2.0);

    std::vector<std::vector<double>> upper(ku);
    for (size_t k = 0; k < ku; ++k) {
        upper[k].resize(n - k - 1, 0.2);
    }

    std::vector<std::span<const double>> lower_spans;
    for (auto& band : lower) lower_spans.push_back(std::span{band});

    std::vector<std::span<const double>> upper_spans;
    for (auto& band : upper) upper_spans.push_back(std::span{band});

    BandedLU<double> solver(n, kl, ku);

    auto start = std::chrono::high_resolution_clock::now();
    auto factor_result = solver.factorize(
        std::span{lower_spans},
        std::span{diag},
        std::span{upper_spans});
    auto end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(factor_result.ok()) << factor_result.message();

    const auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    std::cout << "50-point fit factorization: " << time_us << " µs\n";

    // Target: <1ms for 50-point fit
    EXPECT_LT(time_us, 1000)
        << "50-point factorization exceeds 1ms target";
}

// ============================================================================
// Convenience Function Test
// ============================================================================

TEST(BandedLUTest, ConvenienceFunction) {
    TridiagonalTestSystem sys;

    std::vector<std::span<const double>> lower_bands{std::span{sys.lower}};
    std::vector<std::span<const double>> upper_bands{std::span{sys.upper}};

    std::vector<double> solution(sys.n);

    auto result = solve_banded<double>(
        sys.n, 1, 1,
        std::span{lower_bands},
        std::span{sys.diag},
        std::span{upper_bands},
        std::span{sys.rhs},
        std::span{solution});

    ASSERT_TRUE(result.ok()) << result.message();

    for (size_t i = 0; i < sys.n; ++i) {
        EXPECT_NEAR(solution[i], sys.expected_solution[i], kTolerance)
            << "Solution mismatch at index " << i;
    }
}
