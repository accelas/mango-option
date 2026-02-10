// SPDX-License-Identifier: MIT
#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"
#include "mango/math/chebyshev/tucker_tensor.hpp"
#include "mango/option/table/surface_concepts.hpp"
#include <cmath>
#include <gtest/gtest.h>

namespace mango {
namespace {

// Type aliases used throughout the tests
template <size_t N>
using ChebyshevTensor = ChebyshevInterpolant<N, RawTensor<N>>;
template <size_t N>
using ChebyshevTucker = ChebyshevInterpolant<N, TuckerTensor<N>>;

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
// TuckerTensor tests
// ===========================================================================

TEST(TuckerTensorTest, RoundTripSmall3D) {
    // 3x3x3 tensor: f(i,j,k) = i + 2*j + 3*k
    std::vector<double> vals(27);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                vals[i * 9 + j * 3 + k] = i + 2.0 * j + 3.0 * k;

    auto tucker = TuckerTensor<3>::build(std::move(vals), {3, 3, 3}, 1e-12);

    // Contract with delta weights to recover element (1,2,0)
    std::array<std::vector<double>, 3> coeffs = {
        std::vector<double>{0, 1, 0},
        std::vector<double>{0, 0, 1},
        std::vector<double>{1, 0, 0},
    };
    // Expected: 1 + 2*2 + 3*0 = 5
    EXPECT_NEAR(tucker.contract(coeffs), 5.0, 1e-10);
}

TEST(TuckerTensorTest, CompressesLowRankTensor) {
    // Rank-1 tensor: f(i,j,k) = (i+1)*(j+1)*(k+1) (separable)
    std::vector<double> vals(8 * 8 * 8);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            for (size_t k = 0; k < 8; ++k)
                vals[i * 64 + j * 8 + k] =
                    (i + 1.0) * (j + 1.0) * (k + 1.0);

    auto tucker = TuckerTensor<3>::build(std::move(vals), {8, 8, 8}, 1e-10);

    EXPECT_LT(tucker.compressed_size(), 8u * 8 * 8);
    auto ranks = tucker.ranks();
    for (size_t r : ranks) {
        EXPECT_LE(r, 2u) << "Rank-1 tensor should have rank ~1";
    }
}

TEST(TuckerTensorTest, ReconstructMatches3D) {
    // Build and reconstruct, verify round-trip
    std::vector<double> vals(4 * 5 * 3);
    for (size_t i = 0; i < vals.size(); ++i)
        vals[i] = std::sin(static_cast<double>(i) * 0.1);

    std::array<size_t, 3> shape = {4, 5, 3};
    auto result = tucker_hosvd<3>(vals, shape, 1e-14);
    auto reconstructed = tucker_reconstruct<3>(result);

    for (size_t i = 0; i < vals.size(); ++i)
        EXPECT_NEAR(reconstructed[i], vals[i], 1e-10) << "at index " << i;
}

TEST(TuckerTensorTest, RoundTrip4D) {
    // 3x3x3x3 tensor: f = i + 2j + 3k + 4l
    std::vector<double> vals(81);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                for (size_t l = 0; l < 3; ++l)
                    vals[i * 27 + j * 9 + k * 3 + l] =
                        i + 2.0 * j + 3.0 * k + 4.0 * l;

    auto tucker = TuckerTensor<4>::build(std::move(vals), {3, 3, 3, 3}, 1e-12);

    // Recover element (2, 1, 0, 2): expected = 2 + 2*1 + 3*0 + 4*2 = 12
    std::array<std::vector<double>, 4> coeffs = {
        std::vector<double>{0, 0, 1},
        std::vector<double>{0, 1, 0},
        std::vector<double>{1, 0, 0},
        std::vector<double>{0, 0, 1},
    };
    EXPECT_NEAR(tucker.contract(coeffs), 12.0, 1e-10);
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
// ChebyshevTucker (Tucker storage) tests
// ===========================================================================

TEST(ChebyshevTuckerTest, ExactForBilinear3D) {
    // f(x,y,z) = x*y + z with Tucker compression
    auto f = [](std::array<double, 3> c) {
        return c[0] * c[1] + c[2];
    };
    Domain<3> dom{.lo = {0.0, 0.0, 0.0}, .hi = {2.0, 3.0, 1.0}};
    std::array<size_t, 3> npts = {4, 4, 4};

    auto interp = ChebyshevTucker<3>::build(f, dom, npts, 1e-12);

    std::array<double, 3> q1 = {0.7, 1.5, 0.3};
    EXPECT_NEAR(interp.eval(q1), f(q1), 1e-10);

    std::array<double, 3> q2 = {1.9, 2.8, 0.95};
    EXPECT_NEAR(interp.eval(q2), f(q2), 1e-10);
}

TEST(ChebyshevTuckerTest, CompressesSmooth3D) {
    // exp(-x^2)*sin(y)*cos(z) is smooth and should compress well
    auto f = [](std::array<double, 3> c) {
        return std::exp(-c[0] * c[0]) * std::sin(c[1]) * std::cos(c[2]);
    };
    Domain<3> dom{.lo = {-1.0, 0.0, 0.0}, .hi = {1.0, M_PI, M_PI}};
    std::array<size_t, 3> npts = {8, 8, 8};

    auto raw   = ChebyshevTensor<3>::build(f, dom, npts);
    auto tucker = ChebyshevTucker<3>::build(f, dom, npts, 1e-8);

    EXPECT_LT(tucker.compressed_size(), raw.compressed_size())
        << "Tucker compression should reduce storage for smooth functions";

    // Verify accuracy is preserved
    std::array<double, 3> q = {0.3, 1.5, 1.0};
    EXPECT_NEAR(tucker.eval(q), raw.eval(q), 1e-6);
}

TEST(ChebyshevTuckerTest, ExactForLinear4D) {
    // f(x,y,z,w) = x + 2y - 3z + 0.5w
    auto f = [](std::array<double, 4> c) {
        return c[0] + 2.0 * c[1] - 3.0 * c[2] + 0.5 * c[3];
    };
    Domain<4> dom{.lo = {0.0, -1.0, 0.0, 1.0}, .hi = {1.0, 1.0, 2.0, 3.0}};
    std::array<size_t, 4> npts = {4, 4, 4, 4};

    auto interp = ChebyshevTucker<4>::build(f, dom, npts, 1e-12);

    std::array<double, 4> q1 = {0.5, 0.0, 1.0, 2.0};
    EXPECT_NEAR(interp.eval(q1), f(q1), 1e-10);

    std::array<double, 4> q2 = {0.1, -0.8, 1.9, 1.2};
    EXPECT_NEAR(interp.eval(q2), f(q2), 1e-10);
}

TEST(ChebyshevTuckerTest, BuildFromValuesMatchesBuild) {
    // Both construction paths should give the same results
    auto f = [](std::array<double, 3> c) {
        return c[0] * c[0] + std::sin(c[1]) + c[2];
    };
    Domain<3> dom{.lo = {0.0, 0.0, 0.0}, .hi = {1.0, M_PI, 2.0}};
    std::array<size_t, 3> npts = {5, 5, 5};

    // Build via sampling
    auto from_build = ChebyshevTucker<3>::build(f, dom, npts, 1e-12);

    // Build via pre-computed values: manually sample at Chebyshev nodes
    size_t total = 1;
    for (size_t d = 0; d < 3; ++d) total *= npts[d];

    std::array<std::vector<double>, 3> nodes;
    for (size_t d = 0; d < 3; ++d)
        nodes[d] = chebyshev_nodes(npts[d], dom.lo[d], dom.hi[d]);

    std::array<size_t, 3> strides = {npts[1] * npts[2], npts[2], 1};
    std::vector<double> values(total);
    for (size_t flat = 0; flat < total; ++flat) {
        std::array<double, 3> coords{};
        size_t remaining = flat;
        for (size_t d = 0; d < 3; ++d) {
            size_t idx = remaining / strides[d];
            remaining %= strides[d];
            coords[d] = nodes[d][idx];
        }
        values[flat] = f(coords);
    }

    auto from_values = ChebyshevTucker<3>::build_from_values(
        std::span<const double>(values), dom, npts, 1e-12);

    // Both should give the same result at query points
    std::array<double, 3> q1 = {0.3, 1.0, 0.8};
    std::array<double, 3> q2 = {0.9, 2.5, 1.5};
    EXPECT_NEAR(from_build.eval(q1), from_values.eval(q1), 1e-10);
    EXPECT_NEAR(from_build.eval(q2), from_values.eval(q2), 1e-10);
}

// ===========================================================================
// SurfaceInterpolant concept verification
// ===========================================================================

static_assert(SurfaceInterpolant<ChebyshevTensor<3>, 3>);
static_assert(SurfaceInterpolant<ChebyshevTensor<4>, 4>);
static_assert(SurfaceInterpolant<ChebyshevTucker<3>, 3>);
static_assert(SurfaceInterpolant<ChebyshevTucker<4>, 4>);

}  // namespace
}  // namespace mango
