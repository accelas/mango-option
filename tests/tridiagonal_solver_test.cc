// SPDX-License-Identifier: MIT
#include "mango/math/thomas_solver.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

TEST(TridiagonalSolverTest, Simple3x3System) {
    // System:
    // 2x + 1y       = 1
    // 1x + 2y + 1z  = 0
    //      1y + 2z  = 1
    // Solution: x=1, y=-1, z=1

    std::vector<double> lower = {1.0, 1.0};      // size n-1
    std::vector<double> diag = {2.0, 2.0, 2.0};  // size n
    std::vector<double> upper = {1.0, 1.0};      // size n-1
    std::vector<double> rhs = {1.0, 0.0, 1.0};
    std::vector<double> solution(3);
    std::vector<double> workspace(6);  // 2n

    auto result = mango::solve_thomas<double>(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{solution}, std::span{workspace}
    );

    EXPECT_TRUE(result.ok());
    EXPECT_NEAR(solution[0], 1.0, 1e-10);
    EXPECT_NEAR(solution[1], -1.0, 1e-10);
    EXPECT_NEAR(solution[2], 1.0, 1e-10);
}

TEST(TridiagonalSolverTest, SingularMatrix) {
    // All zeros diagonal - should detect singularity
    std::vector<double> lower = {1.0};
    std::vector<double> diag = {0.0, 0.0};  // Singular!
    std::vector<double> upper = {1.0};
    std::vector<double> rhs = {1.0, 1.0};
    std::vector<double> solution(2);
    std::vector<double> workspace(4);

    auto result = mango::solve_thomas<double>(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{solution}, std::span{workspace}
    );

    EXPECT_FALSE(result.ok());  // Should fail
}

TEST(TridiagonalSolverTest, HeatEquationDiscretization) {
    // Heat equation: ∂u/∂t = D·∂²u/∂x²
    // Implicit Euler: u^{n+1} - dt·D·∂²u^{n+1}/∂x² = u^n
    // With D=1, dt=0.01, dx=0.1, central difference:
    // u_i - 0.01·(u_{i-1} - 2u_i + u_{i+1})/(0.1)² = rhs_i
    // (1 + 2·0.01/0.01)·u_i - (0.01/0.01)·u_{i±1} = rhs_i
    // 3u_i - u_{i-1} - u_{i+1} = rhs_i

    const size_t n = 5;
    std::vector<double> lower(n-1, -1.0);
    std::vector<double> diag(n, 3.0);
    std::vector<double> upper(n-1, -1.0);
    std::vector<double> rhs = {1.0, 2.0, 3.0, 2.0, 1.0};
    std::vector<double> solution(n);
    std::vector<double> workspace(2*n);

    auto result = mango::solve_thomas<double>(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{solution}, std::span{workspace}
    );

    EXPECT_TRUE(result.ok());
    // Verify solution satisfies the system (spot check middle point)
    double check = lower[1] * solution[1] + diag[2] * solution[2]
                   + upper[2] * solution[3];
    EXPECT_NEAR(check, rhs[2], 1e-9);
}

TEST(TridiagonalSolverTest, DiagonallyDominant) {
    // Diagonally dominant matrix (guaranteed stable)
    // |a_ii| >= sum(|a_ij|) for all i
    const size_t n = 10;
    std::vector<double> lower(n-1, -1.0);
    std::vector<double> diag(n, 10.0);  // >> 2 (sum of off-diag)
    std::vector<double> upper(n-1, -1.0);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> solution(n);
    std::vector<double> workspace(2*n);

    auto result = mango::solve_thomas<double>(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{solution}, std::span{workspace}
    );

    EXPECT_TRUE(result.ok());
    // Should converge without issue
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FALSE(std::isnan(solution[i]));
        EXPECT_FALSE(std::isinf(solution[i]));
    }
}

// ===========================================================================
// Projected Thomas solver (Brennan-Schwartz) tests
// ===========================================================================

TEST(ProjectedThomasTest, NoActiveConstraints) {
    // When obstacle is below unconstrained solution, projected = standard
    std::vector<double> lower = {1.0, 1.0};
    std::vector<double> diag = {2.0, 2.0, 2.0};
    std::vector<double> upper = {1.0, 1.0};
    std::vector<double> rhs = {1.0, 0.0, 1.0};
    std::vector<double> psi = {-100.0, -100.0, -100.0};  // Far below solution
    std::vector<double> solution(3);
    std::vector<double> workspace(6);

    auto result = mango::solve_thomas_projected<double>(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{psi}, std::span{solution}, std::span{workspace}
    );

    ASSERT_TRUE(result.ok());
    // Should match standard Thomas: [1.0, -1.0, 1.0]
    EXPECT_NEAR(solution[0], 1.0, 1e-10);
    EXPECT_NEAR(solution[1], -1.0, 1e-10);
    EXPECT_NEAR(solution[2], 1.0, 1e-10);
}

TEST(ProjectedThomasTest, AllConstraintsActive) {
    // When obstacle is above unconstrained solution everywhere,
    // solution should equal obstacle
    std::vector<double> lower = {1.0, 1.0};
    std::vector<double> diag = {2.0, 2.0, 2.0};
    std::vector<double> upper = {1.0, 1.0};
    std::vector<double> rhs = {1.0, 0.0, 1.0};
    std::vector<double> psi = {100.0, 100.0, 100.0};  // Far above solution
    std::vector<double> solution(3);
    std::vector<double> workspace(6);

    auto result = mango::solve_thomas_projected<double>(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{psi}, std::span{solution}, std::span{workspace}
    );

    ASSERT_TRUE(result.ok());
    // Each component should equal obstacle (obstacle far above unconstrained solution)
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(solution[i], psi[i], 1e-10);
    }
}

TEST(ProjectedThomasTest, PartialConstraints) {
    // Middle node has high obstacle, endpoints don't
    std::vector<double> lower = {1.0, 1.0};
    std::vector<double> diag = {2.0, 2.0, 2.0};
    std::vector<double> upper = {1.0, 1.0};
    std::vector<double> rhs = {1.0, 0.0, 1.0};
    // Unconstrained solution: [1.0, -1.0, 1.0]
    // Force middle to 5.0 via obstacle
    std::vector<double> psi = {-100.0, 5.0, -100.0};
    std::vector<double> solution(3);
    std::vector<double> workspace(6);

    auto result = mango::solve_thomas_projected<double>(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{psi}, std::span{solution}, std::span{workspace}
    );

    ASSERT_TRUE(result.ok());
    // Middle node must respect obstacle
    EXPECT_GE(solution[1], 5.0 - 1e-10);
}

TEST(ProjectedThomasTest, SolutionRespectsBound) {
    // Property test: solution[i] >= psi[i] for all i
    const size_t n = 20;
    std::vector<double> lower(n - 1, -0.5);
    std::vector<double> diag(n, 2.0);
    std::vector<double> upper(n - 1, -0.5);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> psi(n);
    for (size_t i = 0; i < n; ++i) {
        psi[i] = 0.3 * std::sin(static_cast<double>(i));  // Oscillating obstacle
    }
    std::vector<double> solution(n);
    std::vector<double> workspace(2 * n);

    auto result = mango::solve_thomas_projected<double>(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{psi}, std::span{solution}, std::span{workspace}
    );

    ASSERT_TRUE(result.ok());
    for (size_t i = 0; i < n; ++i) {
        EXPECT_GE(solution[i], psi[i] - 1e-10)
            << "Constraint violated at index " << i;
    }
}

TEST(ProjectedThomasTest, SingularMatrix) {
    std::vector<double> lower = {1.0};
    std::vector<double> diag = {0.0, 0.0};
    std::vector<double> upper = {1.0};
    std::vector<double> rhs = {1.0, 1.0};
    std::vector<double> psi = {0.0, 0.0};
    std::vector<double> solution(2);
    std::vector<double> workspace(4);

    auto result = mango::solve_thomas_projected<double>(
        std::span{lower}, std::span{diag}, std::span{upper},
        std::span{rhs}, std::span{psi}, std::span{solution}, std::span{workspace}
    );

    EXPECT_FALSE(result.ok());
}

// ===========================================================================
// ThomasWorkspace tests
// ===========================================================================

TEST(ThomasWorkspaceTest, ConstructAndUse) {
    mango::ThomasWorkspace<double> ws(10);
    EXPECT_EQ(ws.size(), 10);

    auto span = ws.get();
    EXPECT_EQ(span.size(), 20);  // Workspace is 2n
}

TEST(ThomasWorkspaceTest, Resize) {
    mango::ThomasWorkspace<double> ws(5);
    EXPECT_EQ(ws.size(), 5);

    ws.resize(20);
    EXPECT_EQ(ws.size(), 20);

    auto span = ws.get();
    EXPECT_EQ(span.size(), 40);  // Workspace is 2n
}

// ===========================================================================
// Regression tests for bugs found during code review (issue #441)
// ===========================================================================

// Regression: n==0 must be trivial success, not a size error
// Bug: `lower.size() != n - 1` ran before the n==0 check; with n==0 the
//      subtraction wrapped to SIZE_MAX, so empty systems returned
//      "Lower diagonal size must be n-1" and the success path was dead code.
TEST(ThomasSolverTest, EmptySystemSucceeds) {
    std::span<const double> empty_c;
    std::span<double> empty_m;
    auto result = mango::solve_thomas<double>(
        empty_c, empty_c, empty_c, empty_c, empty_m, empty_m);
    EXPECT_TRUE(result.ok());
}

// Regression: empty diag with nonempty companion spans must still error
// Bug: naive hoisting of the n==0 check above validation would have
//      accepted malformed calls (empty diag, nonempty rhs) as success.
TEST(ThomasSolverTest, EmptyDiagNonemptyRhsRejected) {
    std::vector<double> rhs = {1.0};
    std::vector<double> sol(1);
    std::vector<double> ws(2);
    std::span<const double> empty_c;
    auto result = mango::solve_thomas<double>(
        empty_c, empty_c, empty_c,
        std::span<const double>{rhs}, std::span<double>{sol},
        std::span<double>{ws});
    EXPECT_FALSE(result.ok());
}

// Regression: same n==0 ordering bug in the projected (obstacle) variant
// Bug: duplicate of the wrap-around in solve_thomas_projected.
TEST(ThomasSolverTest, ProjectedEmptySystemSucceeds) {
    std::span<const double> empty_c;
    std::span<double> empty_m;
    auto result = mango::solve_thomas_projected<double>(
        empty_c, empty_c, empty_c, empty_c, empty_c, empty_m, empty_m);
    EXPECT_TRUE(result.ok());
}
