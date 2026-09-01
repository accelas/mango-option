// SPDX-License-Identifier: MIT
//
// Unit tests for the Il'in exponentially-fitted diffusion helper (#472).
// The floating-point contract lives in
// docs/superpowers/specs/2026-08-31-drift-upwinding-472-design.md.

#include "mango/pde/internal/fitted_diffusion.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango::operators::detail {
namespace {

TEST(FittedDiffusionTest, ZeroDriftReturnsAExactly) {
    auto fd = fitted_diffusion(0.02, 0.0, 0.1, 0.2);
    EXPECT_EQ(fd.a_f, 0.02);  // bit-exact: LaplacianPDE path must not move
    EXPECT_EQ(fd.z, 0.0);
}

// Regression: sigma > 0 passes public validation but 0.5*sigma*sigma can
// underflow to 0. Bug guard: a == 0, b != 0 must hit the convection-limit
// branch (a_f = z), not divide by zero.
TEST(FittedDiffusionTest, ZeroDiffusionConvectionLimit) {
    auto fd = fitted_diffusion(0.0, 0.05, 0.1, 0.1);
    EXPECT_EQ(fd.z, 0.5 * 0.05 * 0.1);
    EXPECT_EQ(fd.a_f, fd.z);
}

TEST(FittedDiffusionTest, SeriesBranchNearIdentityForTinyDrift) {
    const double a = 0.02;
    auto fd = fitted_diffusion(a, 1e-8, 0.01, 0.01);
    EXPECT_GE(fd.a_f, a);
    EXPECT_NEAR(fd.a_f, a, a * 1e-12);
}

TEST(FittedDiffusionTest, SeriesAndDirectAgreeAtCutoff) {
    // rho = z/a straddling the 1e-4 series cutoff: branch must be seamless.
    const double a = 1.0;
    for (double rho : {0.99e-4, 1.01e-4}) {
        const double b = 2.0 * rho * a;  // h = 1 => z = b/2 = rho*a
        auto fd = fitted_diffusion(a, b, 1.0, 1.0);
        const double exact = fd.z / std::tanh(rho);
        EXPECT_NEAR(fd.a_f, exact, a * 1e-14) << "rho=" << rho;
    }
}

TEST(FittedDiffusionTest, LargeRhoApproachesZExactly) {
    // sigma = 1e-4 -> a = 5e-9; rho = z/a huge; tanh saturates to 1.
    const double a = 5e-9;
    const double b = 0.05;
    auto fd = fitted_diffusion(a, b, 1.0, 1.0);
    EXPECT_EQ(fd.z, 0.025);
    EXPECT_TRUE(std::isfinite(fd.a_f));
    EXPECT_GE(fd.a_f - fd.z, 0.0);   // binding numerator, exact in FP
    EXPECT_EQ(fd.a_f, fd.z);         // tanh(rho) == 1.0 here
}

TEST(FittedDiffusionTest, ClampInvariantsHoldAcrossSweep) {
    for (double a : {0.0, 5e-9, 5e-5, 0.005, 0.02, 0.08}) {
        for (double b : {-0.25, -0.03, -1e-6, 0.0, 1e-6, 0.03, 0.25}) {
            for (double dxl : {0.01, 0.1, 0.5}) {
                for (double dxr : {0.01, 0.1, 0.5}) {
                    auto fd = fitted_diffusion(a, b, dxl, dxr);
                    EXPECT_GE(fd.a_f, a);
                    EXPECT_GE(fd.a_f - fd.z, 0.0)
                        << "a=" << a << " b=" << b
                        << " dxl=" << dxl << " dxr=" << dxr;
                    EXPECT_TRUE(std::isfinite(fd.a_f));
                }
            }
        }
    }
}

TEST(FittedDiffusionTest, BindingSideFollowsDriftSign) {
    // b > 0 binds dx_right; b < 0 binds dx_left. Mirror symmetry:
    const double a = 0.005;
    auto pos = fitted_diffusion(a, 0.25, 0.05, 0.2);
    auto neg = fitted_diffusion(a, -0.25, 0.2, 0.05);
    EXPECT_EQ(pos.a_f, neg.a_f);
    EXPECT_EQ(pos.z, neg.z);
    EXPECT_EQ(pos.z, 0.5 * 0.25 * 0.2);  // dx_right for b > 0
}

// Guards the spec's C1-at-crossing claim on an asymmetric cell:
// continuity at b = 0 and the quadratic correction z^2/(3a) for small b.
TEST(FittedDiffusionTest, NearZeroDriftContinuityAndQuadraticCorrection) {
    const double a = 0.02, dxl = 0.05, dxr = 0.2;
    const double eps = 1e-12;
    auto minus = fitted_diffusion(a, -eps, dxl, dxr);
    auto zero  = fitted_diffusion(a, 0.0, dxl, dxr);
    auto plus  = fitted_diffusion(a, eps, dxl, dxr);
    EXPECT_NEAR(minus.a_f, zero.a_f, a * 1e-15);
    EXPECT_NEAR(plus.a_f, zero.a_f, a * 1e-15);

    const double b = 1e-3;  // small but in the quadratic regime
    auto fd = fitted_diffusion(a, b, dxl, dxr);
    const double z = 0.5 * b * dxr;
    EXPECT_NEAR(fd.a_f - a, z * z / (3.0 * a), 0.01 * z * z / (3.0 * a));
}

}  // namespace
}  // namespace mango::operators::detail
