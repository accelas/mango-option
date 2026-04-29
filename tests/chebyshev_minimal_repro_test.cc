// SPDX-License-Identifier: MIT
//
// Minimal reproducer for the local-vs-CI divergence found while
// debugging the PDEWorkspace refactor. The original failing test
// (SegmentedChebyshevGapRoutesNearest) goes through PDE + spline +
// chebyshev + segment-routing. Probes during debugging confirmed
// that PDE outputs are bit-identical local vs CI; the divergence is
// downstream. This test exercises the chebyshev-only portion with
// hand-crafted inputs to see if the bug reproduces here.
//
// If this test passes locally and fails in CI, the bug is in
// ChebyshevInterpolant::eval / RawTensor::contract (both are
// target_clones'd "default,avx2,avx512f").
//
// If this test passes in both environments, the bug is upstream of
// chebyshev (cached cubic spline, segment routing, or the build
// transformation) — and we add more probes there.

#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdio>
#include <vector>

namespace {

// Construct a 4D tensor with the same shape as the failing test
// (Nm=33, Nt=4, Ns=5, Nr=3) and values that mimic American put
// price magnitudes (~0.0 to ~1.0 in V/K_ref units).
//
// Uses a deterministic synthetic function: f(m, tau, sigma, rate)
//   = max(0, exp(-tau) * (1 - exp(m)) + sigma * sqrt(tau) - rate * tau)
// — chosen to produce realistic-looking values without invoking any
// PDE machinery. The exact values don't matter; what matters is that
// they are non-zero and realistic in magnitude.
double synthetic_value(double m, double tau, double sigma, double rate) {
    double intrinsic = std::max(0.0, -std::expm1(m));  // (1 - exp(m))_+
    double time_value = sigma * std::sqrt(tau);
    double rate_term = rate * tau;
    return std::max(0.0,
        std::exp(-tau) * intrinsic + time_value - rate_term);
}

TEST(ChebyshevMinimalReproTest, FourDimensionalSurface_KnownInputsProduceNonZeroOutputs) {
    using Interp = mango::ChebyshevInterpolant<4, mango::RawTensor<4>>;

    constexpr size_t Nm = 33;   // moneyness
    constexpr size_t Nt = 4;    // tau
    constexpr size_t Ns = 5;    // sigma
    constexpr size_t Nr = 3;    // rate

    mango::Domain<4> domain{
        .lo = {-0.30, 0.05, 0.10, 0.01},
        .hi = { 0.30, 1.00, 0.40, 0.07},
    };
    std::array<size_t, 4> num_pts = {Nm, Nt, Ns, Nr};

    // Generate values at CGL nodes using the synthetic function
    auto m_nodes  = mango::chebyshev_nodes(Nm, domain.lo[0], domain.hi[0]);
    auto t_nodes  = mango::chebyshev_nodes(Nt, domain.lo[1], domain.hi[1]);
    auto s_nodes  = mango::chebyshev_nodes(Ns, domain.lo[2], domain.hi[2]);
    auto r_nodes  = mango::chebyshev_nodes(Nr, domain.lo[3], domain.hi[3]);

    std::vector<double> values(Nm * Nt * Ns * Nr);
    for (size_t mi = 0; mi < Nm; ++mi) {
        for (size_t ti = 0; ti < Nt; ++ti) {
            for (size_t si = 0; si < Ns; ++si) {
                for (size_t ri = 0; ri < Nr; ++ri) {
                    size_t flat = mi*(Nt*Ns*Nr) + ti*(Ns*Nr) + si*Nr + ri;
                    values[flat] = synthetic_value(
                        m_nodes[mi], t_nodes[ti], s_nodes[si], r_nodes[ri]);
                }
            }
        }
    }

    // Quick sanity check on the inputs themselves
    double values_min = values[0], values_max = values[0], values_sum = 0;
    for (double v : values) {
        values_min = std::min(values_min, v);
        values_max = std::max(values_max, v);
        values_sum += v;
    }
    std::fprintf(stderr,
        "[REPRO] input values: n=%zu min=%.6g max=%.6g sum=%.6g\n",
        values.size(), values_min, values_max, values_sum);

    // Build the interpolant
    auto interp = Interp::build_from_values(
        std::span<const double>(values), domain, num_pts);

    // Sample the surface at a battery of off-node query points
    struct Query { double m, t, s, r; const char* label; };
    std::array<Query, 6> queries = {{
        { 0.00, 0.50, 0.20, 0.05, "ATM_mid_tau" },
        { 0.10, 0.50, 0.20, 0.05, "OTM_put_mid_tau" },
        {-0.10, 0.50, 0.20, 0.05, "ITM_put_mid_tau" },
        { 0.00, 0.10, 0.20, 0.05, "ATM_short_tau" },
        { 0.00, 0.95, 0.20, 0.05, "ATM_long_tau" },
        { 0.00, 0.50, 0.30, 0.05, "ATM_high_vol" },
    }};

    for (const auto& q : queries) {
        double v = interp.eval({q.m, q.t, q.s, q.r});
        double truth = synthetic_value(q.m, q.t, q.s, q.r);
        std::fprintf(stderr,
            "[REPRO] %s: eval=%.10g truth=%.10g diff=%.6g\n",
            q.label, v, truth, std::abs(v - truth));

        EXPECT_TRUE(std::isfinite(v)) << "Non-finite at " << q.label;
        EXPECT_GT(v, 0.0) << "Zero or negative at " << q.label;
        // Loose accuracy: truth-vs-interp within 0.01 on values ~0.05-0.5
        EXPECT_NEAR(v, truth, 0.01) << "Drift at " << q.label;
    }
}

// Even smaller case: 2D, no SIMD width relevance, sanity check.
TEST(ChebyshevMinimalReproTest, TwoDimensionalSurface_QuickSanity) {
    using Interp = mango::ChebyshevInterpolant<2, mango::RawTensor<2>>;

    constexpr size_t Nx = 9, Ny = 5;
    mango::Domain<2> domain{ .lo = {0.0, 0.0}, .hi = {1.0, 1.0} };
    std::array<size_t, 2> num_pts = {Nx, Ny};

    auto x_nodes = mango::chebyshev_nodes(Nx, 0.0, 1.0);
    auto y_nodes = mango::chebyshev_nodes(Ny, 0.0, 1.0);
    std::vector<double> values(Nx * Ny);
    for (size_t i = 0; i < Nx; ++i) {
        for (size_t j = 0; j < Ny; ++j) {
            // Smooth function: f(x, y) = (1-x)*y + 0.5
            values[i*Ny + j] = (1.0 - x_nodes[i]) * y_nodes[j] + 0.5;
        }
    }

    auto interp = Interp::build_from_values(
        std::span<const double>(values), domain, num_pts);
    double v = interp.eval({0.3, 0.4});
    double truth = (1.0 - 0.3) * 0.4 + 0.5;  // = 0.78

    std::fprintf(stderr, "[REPRO-2D] eval=%.10g truth=%.10g\n", v, truth);
    EXPECT_NEAR(v, truth, 1e-10);
}

// Direct RawTensor contract test — bypass even ChebyshevInterpolant.
// If this fails in CI, the bug is in RawTensor::contract.
TEST(ChebyshevMinimalReproTest, RawTensorContract_DirectTest) {
    constexpr size_t N = 4;
    std::array<size_t, N> shape = {3, 3, 3, 3};
    size_t total = 81;

    // Tensor: f(i, j, k, l) = (i + j + k + l) / 4.0
    std::vector<double> values(total);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                for (size_t l = 0; l < 3; ++l)
                    values[i*27 + j*9 + k*3 + l]
                        = (static_cast<double>(i) + j + k + l) / 4.0;

    auto tensor = mango::RawTensor<4>::build(std::move(values), shape);

    // Contract with all-uniform coefficients: should give average.
    std::array<std::vector<double>, N> uniform_coeffs;
    for (size_t d = 0; d < N; ++d) uniform_coeffs[d] = {1.0/3, 1.0/3, 1.0/3};
    double avg = tensor.contract(uniform_coeffs);
    double expected_avg = 0.0;
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 3; ++k)
                for (size_t l = 0; l < 3; ++l)
                    expected_avg += (i+j+k+l) / 4.0;
    expected_avg /= 81.0;

    std::fprintf(stderr, "[REPRO-RAW] avg_contract=%.17g expected=%.17g\n",
        avg, expected_avg);
    EXPECT_NEAR(avg, expected_avg, 1e-12);

    // Contract with delta-coefficient: should pick exact element
    std::array<std::vector<double>, N> delta_coeffs;
    for (size_t d = 0; d < N; ++d) delta_coeffs[d] = {0.0, 1.0, 0.0};
    double mid = tensor.contract(delta_coeffs);
    double expected_mid = (1.0 + 1.0 + 1.0 + 1.0) / 4.0;  // = 1.0
    std::fprintf(stderr, "[REPRO-RAW] delta_contract=%.17g expected=%.17g\n",
        mid, expected_mid);
    EXPECT_NEAR(mid, expected_mid, 1e-14);
}

}  // namespace
