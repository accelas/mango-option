// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless/chebyshev_tucker.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

// Smooth 3D test function: separable, known analytically.
// exp(-x^2) * sin(y) * cos(z/2) is rank-1 in Tucker sense.
double smooth_3d(double x, double y, double z) {
    return std::exp(-x * x) * std::sin(y) * std::cos(z * 0.5);
}

TEST(ChebyshevTuckerTest, ExactForLowDegreePolynomial) {
    // f(x,y,z) = x*y + z should be exactly represented with degree >= 1
    auto f = [](double x, double y, double z) { return x * y + z; };

    ChebyshevTuckerDomain domain{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    ChebyshevTuckerConfig config{.num_pts = {4, 4, 4}, .epsilon = 1e-12};

    auto interp = ChebyshevTucker3D::build(f, domain, config);

    // Query at off-grid points
    for (double x : {-0.5, 0.0, 0.7}) {
        for (double y : {-0.3, 0.4}) {
            for (double z : {-0.8, 0.1, 0.9}) {
                double expected = f(x, y, z);
                double got = interp.eval({x, y, z});
                EXPECT_NEAR(got, expected, 1e-11)
                    << "at (" << x << ", " << y << ", " << z << ")";
            }
        }
    }
}

TEST(ChebyshevTuckerTest, SmoothFunctionConverges) {
    // Use a narrower domain for exp(-x^2) so Chebyshev converges faster
    ChebyshevTuckerDomain domain{
        .bounds = {{{-1.0, 1.0}, {0.0, M_PI}, {-1.0, 1.0}}},
    };

    // Coarse
    auto coarse = ChebyshevTucker3D::build(
        smooth_3d, domain, {.num_pts = {8, 8, 8}, .epsilon = 1e-14});

    // Fine
    auto fine = ChebyshevTucker3D::build(
        smooth_3d, domain, {.num_pts = {16, 16, 16}, .epsilon = 1e-14});

    // Evaluate at a test point
    double x = 0.5, y = 1.2, z = 0.3;
    double exact = smooth_3d(x, y, z);
    double err_coarse = std::abs(coarse.eval({x, y, z}) - exact);
    double err_fine = std::abs(fine.eval({x, y, z}) - exact);

    // Fine should be significantly more accurate (spectral convergence)
    EXPECT_LT(err_fine, err_coarse * 0.01);
    // 16 CGL nodes on [-1,1] for exp(-x^2) gives sub-machine precision
    EXPECT_LT(err_fine, 1e-12);
}

TEST(ChebyshevTuckerTest, TuckerCompressionReducesStorage) {
    ChebyshevTuckerDomain domain{
        .bounds = {{{-2.0, 2.0}, {0.0, M_PI}, {-1.0, 1.0}}},
    };
    ChebyshevTuckerConfig config{.num_pts = {15, 15, 15}, .epsilon = 1e-8};

    auto interp = ChebyshevTucker3D::build(smooth_3d, domain, config);

    size_t full_size = 15 * 15 * 15;
    size_t compressed = interp.compressed_size();
    EXPECT_LT(compressed, full_size)
        << "Tucker should compress a smooth function";
}

TEST(ChebyshevTuckerTest, RanksReportedCorrectly) {
    auto f = [](double x, double y, double z) { return x * y + z; };
    ChebyshevTuckerDomain domain{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    auto interp = ChebyshevTucker3D::build(
        f, domain, {.num_pts = {8, 8, 8}, .epsilon = 1e-10});

    auto ranks = interp.ranks();
    // x*y + z is rank-2 in mode-0/1 unfolding, rank-2 in mode-2
    // Ranks should be small
    for (size_t r : ranks) {
        EXPECT_LE(r, 4u) << "Rank too high for a simple bilinear function";
    }
}

TEST(ChebyshevTuckerTest, EvalFullMatchesEvalTucker) {
    // Use [-1,1] domain where Chebyshev converges rapidly for exp(-x^2)
    ChebyshevTuckerDomain domain{
        .bounds = {{{-1.0, 1.0}, {0.0, M_PI}, {-1.0, 1.0}}},
    };

    // Full rank (epsilon near machine precision)
    auto full = ChebyshevTucker3D::build(
        smooth_3d, domain, {.num_pts = {16, 16, 16}, .epsilon = 1e-15});
    // Compressed
    auto compressed = ChebyshevTucker3D::build(
        smooth_3d, domain, {.num_pts = {16, 16, 16}, .epsilon = 1e-8});

    double x = 0.3, y = 2.0, z = -0.5;
    double exact = smooth_3d(x, y, z);
    double full_err = std::abs(full.eval({x, y, z}) - exact);
    double comp_err = std::abs(compressed.eval({x, y, z}) - exact);

    // With 16 CGL nodes, Chebyshev interpolation error is ~1e-10 for this function.
    // Tucker with epsilon=1e-15 preserves all singular values, so the combined
    // error is dominated by Chebyshev polynomial truncation.
    EXPECT_LT(full_err, 1e-9) << "Full-rank should be near-exact";
    EXPECT_LT(comp_err, 1e-6) << "Compressed should still be accurate";
}

TEST(ChebyshevTuckerTest, PartialDerivativeAccuracy) {
    // f(x,y,z) = sin(x) * cos(y) * z
    // df/dx = cos(x) * cos(y) * z
    // df/dy = -sin(x) * sin(y) * z
    // df/dz = sin(x) * cos(y)
    auto f = [](double x, double y, double z) {
        return std::sin(x) * std::cos(y) * z;
    };

    ChebyshevTuckerDomain domain{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    auto interp = ChebyshevTucker3D::build(
        f, domain, {.num_pts = {16, 16, 16}, .epsilon = 1e-14});

    for (double x : {-0.5, 0.3, 0.8}) {
        for (double y : {-0.4, 0.6}) {
            for (double z : {-0.7, 0.2}) {
                double dx = interp.partial(0, {x, y, z});
                double dy = interp.partial(1, {x, y, z});
                double dz = interp.partial(2, {x, y, z});

                double dx_exact = std::cos(x) * std::cos(y) * z;
                double dy_exact = -std::sin(x) * std::sin(y) * z;
                double dz_exact = std::sin(x) * std::cos(y);

                EXPECT_NEAR(dx, dx_exact, 1e-5)
                    << "df/dx at (" << x << "," << y << "," << z << ")";
                EXPECT_NEAR(dy, dy_exact, 1e-5)
                    << "df/dy at (" << x << "," << y << "," << z << ")";
                EXPECT_NEAR(dz, dz_exact, 1e-5)
                    << "df/dz at (" << x << "," << y << "," << z << ")";
            }
        }
    }
}

}  // namespace
}  // namespace mango
