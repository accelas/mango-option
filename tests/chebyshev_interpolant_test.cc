// SPDX-License-Identifier: MIT
#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"
#include "mango/option/table/surface_concepts.hpp"
#include "mango/support/error_types.hpp"
#include <cmath>
#include <limits>
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

    auto r = ChebyshevTensor<3>::build(f, dom, npts);
    ASSERT_TRUE(r.has_value());
    auto& interp = *r;

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

    auto r = ChebyshevTensor<3>::build(f, dom, npts);
    ASSERT_TRUE(r.has_value());
    auto& interp = *r;

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

    auto coarse_r = ChebyshevTensor<3>::build(f, dom, {5, 5, 5});
    auto fine_r   = ChebyshevTensor<3>::build(f, dom, {9, 9, 9});
    ASSERT_TRUE(coarse_r.has_value());
    ASSERT_TRUE(fine_r.has_value());
    auto& coarse = *coarse_r;
    auto& fine   = *fine_r;

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

    auto r = ChebyshevTensor<4>::build(f, dom, npts);
    ASSERT_TRUE(r.has_value());
    auto& interp = *r;

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

    auto r = ChebyshevTensor<3>::build(f, dom, npts);
    ASSERT_TRUE(r.has_value());
    auto& interp = *r;

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

    auto r = ChebyshevTensor<3>::build(f, dom, npts);
    ASSERT_TRUE(r.has_value());
    auto& interp = *r;

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

// ===========================================================================
// Regression tests for issue #426 (build_from_values silently fit NaN input)
// ===========================================================================

// Regression: builds succeeded with 15-20% NaN input during the #419 incident
// Bug: no input validation and no error path (object returned directly)
TEST(ChebyshevInterpolantGuardTest, BuildFromValuesRejectsNaN) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    std::vector<double> values(9, 0.0);
    values[4] = std::nan("");
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::NaNInput);
    EXPECT_EQ(r.error().index, 4u);
}

// Regression: builds succeeded with 15-20% NaN input during the #419 incident
// Bug: no input validation and no error path (object returned directly)
TEST(ChebyshevInterpolantGuardTest, BuildFromValuesRejectsInf) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    std::vector<double> values(9, 0.0);
    values[2] = std::numeric_limits<double>::infinity();
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::InfInput);
    EXPECT_EQ(r.error().index, 2u);
}

// Regression: no size validation between values buffer and declared shape
// Bug: mismatched sizes were never checked before indexing into storage
TEST(ChebyshevInterpolantGuardTest, RejectsSizeMismatch) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    std::vector<double> values(8, 0.0);  // needs 9
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::ValueSizeMismatch);
}

// Regression: num_pts < 2 on any axis produced a degenerate/undefined interpolant
// Bug: no lower-bound check on num_pts before generating Chebyshev nodes
TEST(ChebyshevInterpolantGuardTest, RejectsNumPtsBelowTwo) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    std::vector<double> values(3, 0.0);
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {1, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::InsufficientGridPoints);
}

// Regression: NaN domain bounds were never checked before generating nodes
// Bug: no finiteness check on domain.lo/hi
TEST(ChebyshevInterpolantGuardTest, RejectsNaNDomain) {
    mango::Domain<2> dom{.lo = {std::nan(""), 0.0}, .hi = {1.0, 1.0}};
    std::vector<double> values(9, 0.0);
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::NaNInput);
}

// Regression: infinite domain bounds were never checked before generating nodes
// Bug: no finiteness check on domain.lo/hi
TEST(ChebyshevInterpolantGuardTest, RejectsInfDomain) {
    mango::Domain<2> dom{.lo = {0.0, 0.0},
                         .hi = {std::numeric_limits<double>::infinity(), 1.0}};
    std::vector<double> values(9, 0.0);
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::InfInput);
}

// Regression: reversed domain bounds (lo > hi) were never checked
// Bug: no ordering check on domain.lo/hi
TEST(ChebyshevInterpolantGuardTest, RejectsReversedDomain) {
    mango::Domain<2> dom{.lo = {0.0, 1.0}, .hi = {1.0, 0.5}};  // axis 1 reversed
    std::vector<double> values(9, 0.0);
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::GridNotSorted);
}

// Regression: zero-width domain axes were never checked
// Bug: no width check on domain.lo/hi, leading to division by zero in node generation
TEST(ChebyshevInterpolantGuardTest, RejectsZeroWidthDomain) {
    mango::Domain<2> dom{.lo = {0.0, 0.5}, .hi = {1.0, 0.5}};
    std::vector<double> values(9, 0.0);
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        values, dom, {3, 3});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::ZeroWidthGrid);
}

// Regression: unchecked product of num_pts could overflow size_t
TEST(ChebyshevInterpolantGuardTest, RejectsShapeProductOverflow) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    std::array<size_t, 2> huge = {std::numeric_limits<size_t>::max() / 2, 4};
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build_from_values(
        std::span<const double>{}, dom, huge);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::ValueSizeMismatch);
}

// Sampling overload: NaN from the sampled function is rejected too
TEST(ChebyshevInterpolantGuardTest, BuildRejectsNaNSampledFunction) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    auto f = [](std::array<double, 2> c) {
        return (c[0] > 0.5) ? std::nan("") : 1.0;
    };
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build(f, dom, {4, 4});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, mango::InterpolationErrorCode::NaNInput);
}

// Sampling overload must validate shape/domain BEFORE invoking f (or allocating)
TEST(ChebyshevInterpolantGuardTest, SamplingValidatesBeforeInvokingF) {
    using Cheb2 = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>;
    int calls = 0;
    auto f = [&calls](std::array<double, 2>) { ++calls; return 1.0; };

    mango::Domain<2> reversed{.lo = {0.0, 1.0}, .hi = {1.0, 0.5}};
    auto r1 = Cheb2::build(f, reversed, {3, 3});
    EXPECT_FALSE(r1.has_value());
    EXPECT_EQ(calls, 0);

    mango::Domain<2> ok{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    auto r2 = Cheb2::build(f, ok, {1, 3});
    EXPECT_FALSE(r2.has_value());
    EXPECT_EQ(calls, 0);

    std::array<size_t, 2> huge = {std::numeric_limits<size_t>::max() / 2, 4};
    auto r3 = Cheb2::build(f, ok, huge);
    EXPECT_FALSE(r3.has_value());
    EXPECT_EQ(calls, 0);
}

// Locks existing behavior: NaN queries propagate through barycentric eval
TEST(ChebyshevInterpolantGuardTest, EvalPropagatesNaNQuery) {
    mango::Domain<2> dom{.lo = {0.0, 0.0}, .hi = {1.0, 1.0}};
    auto f = [](std::array<double, 2> c) { return c[0] + c[1]; };
    auto r = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>::build(f, dom, {5, 5});
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(std::isnan(r->eval({std::nan(""), 0.5})));
}

}  // namespace
}  // namespace mango
