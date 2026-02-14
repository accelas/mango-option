// SPDX-License-Identifier: MIT
#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"
#include "mango/option/table/surface_concepts.hpp"
#include <cmath>
#include <gtest/gtest.h>

namespace mango {
namespace {

// Type aliases used throughout the tests
template <size_t N>
using ChebyshevTensor = ChebyshevInterpolant<N, RawTensor<N>>;

// ===========================================================================
// RawTensor tests
// ===========================================================================

TEST(RawTensorTest, Contract2DIdentityWeights) {
    // 2x3 tensor: [[1,2,3],[4,5,6]]
    // Contract with weights [1,0] x [0,1,0] => element (0,1) = 2
    RawTensor<2> t = RawTensor<2>::build({1, 2, 3, 4, 5, 6}, {2, 3});
    std::array<std::vector<double>, 2> coeffs = {
        std::vector<double>{1.0, 0.0},
        std::vector<double>{0.0, 1.0, 0.0},
    };
    EXPECT_NEAR(t.contract(coeffs), 2.0, 1e-15);
}

TEST(RawTensorTest, Contract3DUniform) {
    // 2x2x2 tensor of all 1s, uniform weights [0.5,0.5] per axis
    // Result = 8 * 1.0 * 0.5^3 = 1.0
    std::vector<double> vals(8, 1.0);
    RawTensor<3> t = RawTensor<3>::build(std::move(vals), {2, 2, 2});
    std::array<std::vector<double>, 3> coeffs = {
        std::vector<double>{0.5, 0.5},
        std::vector<double>{0.5, 0.5},
        std::vector<double>{0.5, 0.5},
    };
    EXPECT_NEAR(t.contract(coeffs), 1.0, 1e-15);
}

TEST(RawTensorTest, CompressedSizeEqualsTotal) {
    RawTensor<3> t = RawTensor<3>::build(std::vector<double>(60, 0.0), {3, 4, 5});
    EXPECT_EQ(t.compressed_size(), 60u);
}

// ===========================================================================
// ChebyshevTensor (raw storage) tests
// ===========================================================================

TEST(ChebyshevTensorTest, ExactForLinear3D) {
    // f(x,y,z) = 2x + 3y - z + 1
    // Linear functions are exact with degree >= 1 (i.e., num_pts >= 2).
    // Use 4 pts/axis for comfort.
    auto f = [](std::array<double, 3> c) {
        return 2.0 * c[0] + 3.0 * c[1] - c[2] + 1.0;
    };
    Domain<3> dom{.lo = {-1.0, 0.0, 0.5}, .hi = {1.0, 2.0, 3.0}};
    std::array<size_t, 3> npts = {4, 4, 4};

    auto interp = ChebyshevTensor<3>::build(f, dom, npts);

    // Test at several interior points
    std::array<double, 3> q1 = {0.3, 1.2, 1.7};
    EXPECT_NEAR(interp.eval(q1), f(q1), 1e-12);

    std::array<double, 3> q2 = {-0.5, 0.1, 2.9};
    EXPECT_NEAR(interp.eval(q2), f(q2), 1e-12);

    std::array<double, 3> q3 = {0.0, 1.0, 1.75};
    EXPECT_NEAR(interp.eval(q3), f(q3), 1e-12);
}

TEST(ChebyshevTensorTest, ExactForBilinear3D) {
    // f(x,y,z) = x*y + z
    // Bilinear = degree 2 in the product sense. With 4 nodes per axis
    // (degree 3) this should be exact.
    auto f = [](std::array<double, 3> c) {
        return c[0] * c[1] + c[2];
    };
    Domain<3> dom{.lo = {0.0, 0.0, 0.0}, .hi = {2.0, 3.0, 1.0}};
    std::array<size_t, 3> npts = {4, 4, 4};

    auto interp = ChebyshevTensor<3>::build(f, dom, npts);

    std::array<double, 3> q1 = {0.7, 1.5, 0.3};
    EXPECT_NEAR(interp.eval(q1), f(q1), 1e-12);

    std::array<double, 3> q2 = {1.9, 2.8, 0.95};
    EXPECT_NEAR(interp.eval(q2), f(q2), 1e-12);
}

TEST(ChebyshevTensorTest, SmoothFunctionConverges3D) {
    // f(x,y,z) = exp(-x^2) * sin(y) * cos(z/2)
    // Verify: finer grid gives smaller error than coarser grid.
    auto f = [](std::array<double, 3> c) {
        return std::exp(-c[0] * c[0]) * std::sin(c[1]) * std::cos(c[2] / 2.0);
    };
    Domain<3> dom{.lo = {-1.0, 0.0, 0.0}, .hi = {1.0, M_PI, 2.0}};

    auto coarse = ChebyshevTensor<3>::build(f, dom, {5, 5, 5});
    auto fine   = ChebyshevTensor<3>::build(f, dom, {9, 9, 9});

    // Evaluate at test points and measure max error
    double coarse_err = 0.0, fine_err = 0.0;
    std::array<std::array<double, 3>, 5> test_pts = {{
        {0.3, 1.0, 0.5},
        {-0.7, 2.5, 1.8},
        {0.0, 0.5, 1.0},
        {0.9, 3.0, 0.1},
        {-0.2, 1.5, 1.5},
    }};
    for (const auto& q : test_pts) {
        double exact = f(q);
        coarse_err = std::max(coarse_err, std::abs(coarse.eval(q) - exact));
        fine_err   = std::max(fine_err, std::abs(fine.eval(q) - exact));
    }

    EXPECT_LT(fine_err, coarse_err)
        << "Fine grid should be more accurate than coarse";
    EXPECT_LT(fine_err, 1e-4)
        << "Fine grid (9 pts) should be accurate for this smooth function";
}

TEST(ChebyshevTensorTest, ExactForLinear4D) {
    // f(x,y,z,w) = x + 2y - 3z + 0.5w
    auto f = [](std::array<double, 4> c) {
        return c[0] + 2.0 * c[1] - 3.0 * c[2] + 0.5 * c[3];
    };
    Domain<4> dom{.lo = {0.0, -1.0, 0.0, 1.0}, .hi = {1.0, 1.0, 2.0, 3.0}};
    std::array<size_t, 4> npts = {4, 4, 4, 4};

    auto interp = ChebyshevTensor<4>::build(f, dom, npts);

    std::array<double, 4> q1 = {0.5, 0.0, 1.0, 2.0};
    EXPECT_NEAR(interp.eval(q1), f(q1), 1e-12);

    std::array<double, 4> q2 = {0.1, -0.8, 1.9, 1.2};
    EXPECT_NEAR(interp.eval(q2), f(q2), 1e-12);
}

TEST(ChebyshevTensorTest, PartialDerivatives3D) {
    // f(x,y,z) = sin(x) * cos(y) * z
    // df/dx = cos(x) * cos(y) * z
    // df/dy = -sin(x) * sin(y) * z
    // df/dz = sin(x) * cos(y)
    auto f = [](std::array<double, 3> c) {
        return std::sin(c[0]) * std::cos(c[1]) * c[2];
    };
    Domain<3> dom{.lo = {0.0, 0.0, 0.5}, .hi = {M_PI, M_PI, 2.0}};
    std::array<size_t, 3> npts = {12, 12, 12};

    auto interp = ChebyshevTensor<3>::build(f, dom, npts);

    std::array<double, 3> q = {1.0, 0.5, 1.2};

    double df_dx = std::cos(q[0]) * std::cos(q[1]) * q[2];
    double df_dy = -std::sin(q[0]) * std::sin(q[1]) * q[2];
    double df_dz = std::sin(q[0]) * std::cos(q[1]);

    // FD partial derivatives (h = 1e-6 * span) should match analytical
    // to ~1e-4 or better for a well-resolved interpolant.
    EXPECT_NEAR(interp.partial(0, q), df_dx, 1e-4)
        << "Partial w.r.t. x";
    EXPECT_NEAR(interp.partial(1, q), df_dy, 1e-4)
        << "Partial w.r.t. y";
    EXPECT_NEAR(interp.partial(2, q), df_dz, 1e-4)
        << "Partial w.r.t. z";
}

TEST(ChebyshevTensorTest, DomainClamping) {
    // f(x,y,z) = x + y + z on [0,1]^3
    // Out-of-bounds queries should be clamped to the boundary.
    auto f = [](std::array<double, 3> c) {
        return c[0] + c[1] + c[2];
    };
    Domain<3> dom{.lo = {0.0, 0.0, 0.0}, .hi = {1.0, 1.0, 1.0}};
    std::array<size_t, 3> npts = {4, 4, 4};

    auto interp = ChebyshevTensor<3>::build(f, dom, npts);

    // Query below domain: should clamp to (0, 0, 0) => f = 0
    std::array<double, 3> below = {-1.0, -2.0, -0.5};
    EXPECT_NEAR(interp.eval(below), 0.0, 1e-12);

    // Query above domain: should clamp to (1, 1, 1) => f = 3
    std::array<double, 3> above = {5.0, 3.0, 2.0};
    EXPECT_NEAR(interp.eval(above), 3.0, 1e-12);

    // Mixed: (-0.5, 0.5, 1.5) => clamp to (0, 0.5, 1) => f = 1.5
    std::array<double, 3> mixed = {-0.5, 0.5, 1.5};
    EXPECT_NEAR(interp.eval(mixed), 1.5, 1e-12);
}

// ===========================================================================
// SurfaceInterpolant concept verification
// ===========================================================================

static_assert(SurfaceInterpolant<ChebyshevTensor<3>, 3>);
static_assert(SurfaceInterpolant<ChebyshevTensor<4>, 4>);

}  // namespace
}  // namespace mango
