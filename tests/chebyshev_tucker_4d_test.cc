// SPDX-License-Identifier: MIT
#include "mango/option/table/dimensionless/chebyshev_tucker_4d.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace mango {
namespace {

TEST(ChebyshevTucker4DTest, ExactForLowDegreePolynomial) {
    auto f = [](double x, double y, double z, double w) {
        return x * y + z * w;
    };

    ChebyshevTucker4DDomain domain{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    ChebyshevTucker4DConfig config{.num_pts = {4, 4, 4, 4}, .epsilon = 1e-12};
    auto interp = ChebyshevTucker4D::build(f, domain, config);

    for (double x : {-0.5, 0.3}) {
        for (double y : {-0.4, 0.7}) {
            for (double z : {-0.8, 0.1}) {
                for (double w : {-0.3, 0.6}) {
                    double expected = f(x, y, z, w);
                    double got = interp.eval({x, y, z, w});
                    EXPECT_NEAR(got, expected, 1e-10)
                        << "at (" << x << "," << y << "," << z << "," << w << ")";
                }
            }
        }
    }
}

TEST(ChebyshevTucker4DTest, SmoothFunctionConverges) {
    auto f = [](double x, double y, double z, double w) {
        return std::exp(-x * x) * std::sin(y) * std::cos(z * 0.5) * (1.0 + w);
    };

    ChebyshevTucker4DDomain domain{
        .bounds = {{{-1.0, 1.0}, {0.0, 3.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };

    auto coarse = ChebyshevTucker4D::build(
        f, domain, {.num_pts = {6, 6, 6, 6}, .epsilon = 1e-14});
    auto fine = ChebyshevTucker4D::build(
        f, domain, {.num_pts = {12, 12, 12, 6}, .epsilon = 1e-14});

    double x = 0.5, y = 1.2, z = 0.3, w = 0.4;
    double exact = f(x, y, z, w);
    double err_coarse = std::abs(coarse.eval({x, y, z, w}) - exact);
    double err_fine = std::abs(fine.eval({x, y, z, w}) - exact);

    EXPECT_LT(err_fine, err_coarse * 0.01);
    EXPECT_LT(err_fine, 1e-5);
}

TEST(ChebyshevTucker4DTest, TuckerCompressionReducesStorage) {
    auto f = [](double x, double y, double z, double w) {
        return std::exp(-x * x) * std::sin(y) * std::cos(z * 0.5) * (1.0 + w);
    };
    ChebyshevTucker4DDomain domain{
        .bounds = {{{-2.0, 2.0}, {0.0, 3.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    ChebyshevTucker4DConfig config{.num_pts = {10, 10, 10, 6}, .epsilon = 1e-8};
    auto interp = ChebyshevTucker4D::build(f, domain, config);

    size_t full_size = 10 * 10 * 10 * 6;
    size_t compressed = interp.compressed_size();
    EXPECT_LT(compressed, full_size);
}

TEST(ChebyshevTucker4DTest, PartialDerivativeAccuracy) {
    auto f = [](double x, double y, double z, double w) {
        return std::sin(x) * std::cos(y) * z * (1.0 + w);
    };

    ChebyshevTucker4DDomain domain{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    auto interp = ChebyshevTucker4D::build(
        f, domain, {.num_pts = {14, 14, 14, 6}, .epsilon = 1e-14});

    double x = 0.3, y = -0.4, z = 0.5, w = 0.2;
    double dx = interp.partial(0, {x, y, z, w});
    double dy = interp.partial(1, {x, y, z, w});
    double dz = interp.partial(2, {x, y, z, w});
    double dw = interp.partial(3, {x, y, z, w});

    double dx_exact = std::cos(x) * std::cos(y) * z * (1.0 + w);
    double dy_exact = -std::sin(x) * std::sin(y) * z * (1.0 + w);
    double dz_exact = std::sin(x) * std::cos(y) * (1.0 + w);
    double dw_exact = std::sin(x) * std::cos(y) * z;

    EXPECT_NEAR(dx, dx_exact, 1e-5);
    EXPECT_NEAR(dy, dy_exact, 1e-5);
    EXPECT_NEAR(dz, dz_exact, 1e-5);
    EXPECT_NEAR(dw, dw_exact, 1e-5);
}

TEST(ChebyshevTucker4DTest, DomainClampingDoesNotCrash) {
    auto f = [](double x, double y, double z, double w) { return x + y + z + w; };
    ChebyshevTucker4DDomain domain{
        .bounds = {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}},
    };
    auto interp = ChebyshevTucker4D::build(
        f, domain, {.num_pts = {4, 4, 4, 4}, .epsilon = 1e-12});

    double v = interp.eval({5.0, -5.0, 10.0, -10.0});
    EXPECT_TRUE(std::isfinite(v));
}

}  // namespace
}  // namespace mango
