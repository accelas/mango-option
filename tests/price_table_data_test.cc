// SPDX-License-Identifier: MIT

// Round-trip tests for PriceTableData serialization: to_data -> from_data -> verify
// Tests all 7+1 surface types (including Chebyshev3DRaw).

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <optional>
#include <vector>

#include "mango/option/table/serialization/to_data.hpp"
#include "mango/option/table/serialization/from_data.hpp"

// Surface type headers
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"

// Dimensionless 3D builder
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_3d_accessor.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/math/bspline/bspline_nd_separable.hpp"
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/math/bspline/bspline_basis.hpp"
#include "mango/math/chebyshev/chebyshev_nodes.hpp"

namespace mango {
namespace {

// ===========================================================================
// Helper: convert S/K moneyness to log-moneyness
// ===========================================================================

std::vector<double> to_log_m(std::initializer_list<double> sk) {
    std::vector<double> out;
    out.reserve(sk.size());
    for (double m : sk) {
        out.push_back(std::log(m));
    }
    return out;
}

// ===========================================================================
// Helper: verify prices match between two surfaces at sample points (4D)
// ===========================================================================

template <typename Surface1, typename Surface2>
void verify_prices_match_4d(const Surface1& original, const Surface2& loaded,
                            double K_ref, double tolerance = 1e-10) {
    struct TestPoint { double spot, strike, tau, sigma, rate; };
    std::vector<TestPoint> test_points = {
        {K_ref,       K_ref,       0.5,  0.25, 0.05},
        {K_ref * 0.9, K_ref,       0.3,  0.20, 0.04},
        {K_ref * 1.1, K_ref,       0.8,  0.30, 0.03},
        {K_ref,       K_ref * 1.1, 1.0,  0.20, 0.05},
        {K_ref,       K_ref * 0.9, 0.25, 0.35, 0.04},
    };

    for (const auto& p : test_points) {
        double orig = original.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        double load = loaded.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        EXPECT_NEAR(orig, load, tolerance)
            << "Mismatch at spot=" << p.spot << " strike=" << p.strike
            << " tau=" << p.tau << " sigma=" << p.sigma << " rate=" << p.rate;
    }
}

// ===========================================================================
// Helper: verify prices match between two surfaces at sample points (3D)
// ===========================================================================

template <typename Surface1, typename Surface2>
void verify_prices_match_3d(const Surface1& original, const Surface2& loaded,
                            double K_ref, double tolerance = 1e-10) {
    struct TestPoint { double spot, strike, tau, sigma, rate; };
    std::vector<TestPoint> test_points = {
        {K_ref,       K_ref,       1.0,  0.20, 0.05},
        {K_ref * 0.9, K_ref,       0.5,  0.25, 0.03},
        {K_ref * 1.1, K_ref,       1.5,  0.15, 0.04},
    };

    for (const auto& p : test_points) {
        double orig = original.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        double load = loaded.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        EXPECT_NEAR(orig, load, tolerance)
            << "Mismatch at spot=" << p.spot << " strike=" << p.strike
            << " tau=" << p.tau << " sigma=" << p.sigma << " rate=" << p.rate;
    }
}

// ===========================================================================
// Test 1: BSpline 4D round-trip
// ===========================================================================

TEST(PriceTableDataTest, BSpline4DRoundTrip) {
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value()) << "from_vectors failed";
    auto& [builder, axes] = *setup;

    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value()) << "build failed";

    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value()) << "make_bspline_surface failed";

    auto data = to_data(*surface);

    EXPECT_EQ(data.surface_type, "bspline_4d");
    ASSERT_EQ(data.segments.size(), 1u);
    EXPECT_EQ(data.segments[0].interp_type, "bspline");
    EXPECT_EQ(data.segments[0].ndim, 4u);
    EXPECT_EQ(data.option_type, OptionType::PUT);
    EXPECT_NEAR(data.dividend_yield, 0.02, 1e-15);

    auto loaded = from_data<BSplineLeaf>(data);
    ASSERT_TRUE(loaded.has_value()) << "from_data failed";

    verify_prices_match_4d(*surface, *loaded, 100.0);
}

// ===========================================================================
// Test 2: ChebyshevRaw 4D round-trip (Raw -> to_data -> from_data<Raw>)
// ===========================================================================

TEST(PriceTableDataTest, ChebyshevRaw4DRoundTrip) {
    ChebyshevTableConfig config{
        .num_pts = {8, 6, 6, 4},
        .domain = Domain<4>{
            .lo = {-0.30, 0.02, 0.10, 0.02},
            .hi = { 0.30, 1.50, 0.40, 0.08},
        },
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .tucker_epsilon = 0.0,  // 0 = raw tensor
    };
    auto result = build_chebyshev_table(config);
    ASSERT_TRUE(result.has_value()) << "build_chebyshev_table failed";
    auto& surface = std::get<ChebyshevRawSurface>(result->surface);

    auto data = to_data(surface);

    EXPECT_EQ(data.surface_type, "chebyshev_4d_raw");
    ASSERT_EQ(data.segments.size(), 1u);
    EXPECT_EQ(data.segments[0].interp_type, "chebyshev");
    EXPECT_EQ(data.segments[0].ndim, 4u);

    auto loaded = from_data<ChebyshevRawLeaf>(data);
    ASSERT_TRUE(loaded.has_value()) << "from_data failed";

    verify_prices_match_4d(surface, *loaded, 100.0);
}

// ===========================================================================
// Test 3: Chebyshev Tucker 4D -> to_data -> from_data<ChebyshevRawLeaf>
// ===========================================================================

TEST(PriceTableDataTest, ChebyshevTucker4DToRawRoundTrip) {
    ChebyshevTableConfig config{
        .num_pts = {8, 6, 6, 4},
        .domain = Domain<4>{
            .lo = {-0.30, 0.02, 0.10, 0.02},
            .hi = { 0.30, 1.50, 0.40, 0.08},
        },
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .tucker_epsilon = 1e-8,
    };
    auto result = build_chebyshev_table(config);
    ASSERT_TRUE(result.has_value()) << "build_chebyshev_table failed";
    auto& tucker_surface = std::get<ChebyshevSurface>(result->surface);

    auto data = to_data(tucker_surface);

    EXPECT_EQ(data.surface_type, "chebyshev_4d");
    ASSERT_EQ(data.segments.size(), 1u);
    EXPECT_EQ(data.segments[0].interp_type, "chebyshev");

    // from_data<ChebyshevRawLeaf> should accept "chebyshev_4d"
    auto raw_result = from_data<ChebyshevRawLeaf>(data);
    ASSERT_TRUE(raw_result.has_value()) << "from_data<ChebyshevRawLeaf> failed";

    // Prices should be identical (Tucker expand -> raw storage, same math)
    verify_prices_match_4d(tucker_surface, *raw_result, 100.0);
}

// ===========================================================================
// Test 4: Tucker from_data<ChebyshevLeaf> always returns error
// ===========================================================================

TEST(PriceTableDataTest, TuckerFromDataReturnsError) {
    ChebyshevTableConfig config{
        .num_pts = {8, 6, 6, 4},
        .domain = Domain<4>{
            .lo = {-0.30, 0.02, 0.10, 0.02},
            .hi = { 0.30, 1.50, 0.40, 0.08},
        },
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .tucker_epsilon = 1e-8,
    };
    auto result = build_chebyshev_table(config);
    ASSERT_TRUE(result.has_value()) << "build_chebyshev_table failed";
    auto& tucker_surface = std::get<ChebyshevSurface>(result->surface);

    auto data = to_data(tucker_surface);

    auto tucker_result = from_data<ChebyshevLeaf>(data);
    EXPECT_FALSE(tucker_result.has_value())
        << "from_data<ChebyshevLeaf> should fail (Tucker not recoverable)";
}

// ===========================================================================
// Test 5: BSpline 3D round-trip
// ===========================================================================

TEST(PriceTableDataTest, BSpline3DRoundTrip) {
    constexpr double K_ref = 100.0;

    DimensionlessAxes axes;
    axes.log_moneyness = {-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30};
    axes.tau_prime = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16};
    axes.ln_kappa = {-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8};

    auto pde = solve_dimensionless_pde(axes, K_ref, OptionType::PUT);
    ASSERT_TRUE(pde.has_value())
        << "PDE solve failed: code=" << static_cast<int>(pde.error().code);

    Dimensionless3DAccessor accessor(pde->values, axes, K_ref);
    eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));

    std::array<std::vector<double>, 3> grids = {
        axes.log_moneyness, axes.tau_prime, axes.ln_kappa};
    auto fitter = BSplineNDSeparable<double, 3>::create(grids);
    ASSERT_TRUE(fitter.has_value());
    auto fit = fitter->fit(std::move(pde->values));
    ASSERT_TRUE(fit.has_value());

    std::array<std::vector<double>, 3> bspline_grids = {
        axes.log_moneyness, axes.tau_prime, axes.ln_kappa};
    std::array<std::vector<double>, 3> bspline_knots;
    for (size_t i = 0; i < 3; ++i) {
        bspline_knots[i] = clamped_knots_cubic(bspline_grids[i]);
    }
    auto spline = BSplineND<double, 3>::create(
        bspline_grids, std::move(bspline_knots), std::move(fit->coefficients));
    ASSERT_TRUE(spline.has_value());

    auto spline_ptr = std::make_shared<const BSplineND<double, 3>>(
        std::move(spline.value()));

    SharedBSplineInterp<3> interp(std::move(spline_ptr));
    DimensionlessTransform3D xform;
    BSpline3DTransformLeaf leaf(std::move(interp), xform, K_ref);
    AnalyticalEEP eep(OptionType::PUT, 0.0);
    BSpline3DLeaf eep_leaf(std::move(leaf), std::move(eep));

    const double sigma_min = 0.10;
    const double sigma_max = 0.80;
    SurfaceBounds bounds{
        .m_min = axes.log_moneyness.front(),
        .m_max = axes.log_moneyness.back(),
        .tau_min = 2.0 * axes.tau_prime.front() / (sigma_max * sigma_max),
        .tau_max = 2.0 * axes.tau_prime.back() / (sigma_min * sigma_min),
        .sigma_min = sigma_min,
        .sigma_max = sigma_max,
        .rate_min = 0.005,
        .rate_max = 0.10,
    };

    BSpline3DPriceTable surface(
        std::move(eep_leaf), bounds, OptionType::PUT, 0.0);

    auto data = to_data(surface);

    EXPECT_EQ(data.surface_type, "bspline_3d");
    ASSERT_EQ(data.segments.size(), 1u);
    EXPECT_EQ(data.segments[0].interp_type, "bspline");
    EXPECT_EQ(data.segments[0].ndim, 3u);
    EXPECT_EQ(data.option_type, OptionType::PUT);

    auto loaded = from_data<BSpline3DLeaf>(data);
    ASSERT_TRUE(loaded.has_value()) << "from_data failed";

    verify_prices_match_3d(surface, *loaded, K_ref);
}

// ===========================================================================
// Test 6: Chebyshev 3D Tucker -> to_data -> from_data<Chebyshev3DRawLeaf>
// ===========================================================================

TEST(PriceTableDataTest, Chebyshev3DTuckerToRawRoundTrip) {
    constexpr double K_ref = 100.0;

    DimensionlessAxes axes;
    axes.log_moneyness = {-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30};
    axes.tau_prime = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16};
    axes.ln_kappa = {-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8};

    auto pde = solve_dimensionless_pde(axes, K_ref, OptionType::PUT);
    ASSERT_TRUE(pde.has_value())
        << "PDE solve failed: code=" << static_cast<int>(pde.error().code);

    Dimensionless3DAccessor accessor(pde->values, axes, K_ref);
    eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, 0.0));

    std::array<size_t, 3> num_pts = {
        axes.log_moneyness.size(),
        axes.tau_prime.size(),
        axes.ln_kappa.size(),
    };
    Domain<3> domain{
        .lo = {axes.log_moneyness.front(), axes.tau_prime.front(), axes.ln_kappa.front()},
        .hi = {axes.log_moneyness.back(), axes.tau_prime.back(), axes.ln_kappa.back()},
    };

    auto cheb = ChebyshevInterpolant<3, TuckerTensor<3>>::build_from_values(
        std::span<const double>(pde->values),
        domain, num_pts, 1e-8);

    DimensionlessTransform3D xform;
    Chebyshev3DTransformLeaf tleaf(std::move(cheb), xform, K_ref);
    AnalyticalEEP eep_fn(OptionType::PUT, 0.0);
    Chebyshev3DLeaf eep_leaf(std::move(tleaf), std::move(eep_fn));

    const double sigma_min = 0.10;
    const double sigma_max = 0.80;
    SurfaceBounds bounds{
        .m_min = axes.log_moneyness.front(),
        .m_max = axes.log_moneyness.back(),
        .tau_min = 2.0 * axes.tau_prime.front() / (sigma_max * sigma_max),
        .tau_max = 2.0 * axes.tau_prime.back() / (sigma_min * sigma_min),
        .sigma_min = sigma_min,
        .sigma_max = sigma_max,
        .rate_min = 0.005,
        .rate_max = 0.10,
    };

    Chebyshev3DPriceTable tucker_surface(
        std::move(eep_leaf), bounds, OptionType::PUT, 0.0);

    auto data = to_data(tucker_surface);

    EXPECT_EQ(data.surface_type, "chebyshev_3d");
    ASSERT_EQ(data.segments.size(), 1u);
    EXPECT_EQ(data.segments[0].interp_type, "chebyshev");
    EXPECT_EQ(data.segments[0].ndim, 3u);

    // from_data<Chebyshev3DLeaf> should fail (Tucker not recoverable)
    auto tucker_result = from_data<Chebyshev3DLeaf>(data);
    EXPECT_FALSE(tucker_result.has_value());

    // from_data<Chebyshev3DRawLeaf> should succeed
    auto raw_result = from_data<Chebyshev3DRawLeaf>(data);
    ASSERT_TRUE(raw_result.has_value()) << "from_data<Chebyshev3DRawLeaf> failed";

    verify_prices_match_3d(tucker_surface, *raw_result, K_ref);
}

// ===========================================================================
// Test 7: BSpline segmented round-trip
// ===========================================================================

TEST(PriceTableDataTest, BSplineSegmentedRoundTrip) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.02,
            .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}},
        },
        .grid = IVGrid{
            .moneyness = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.04, 0.05},
        },
        .maturity = 1.0,
    };

    auto bspline_seg = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(bspline_seg.has_value()) << "SegmentedPriceTableBuilder failed";

    auto multi = build_multi_kref_surface({BSplineMultiKRefEntry{
        .K_ref = 100.0, .surface = std::move(*bspline_seg)}});
    ASSERT_TRUE(multi.has_value()) << "build_multi_kref_surface failed";

    SurfaceBounds bounds{
        .m_min = to_log_m({0.8}).front(),
        .m_max = to_log_m({1.2}).front(),
        .tau_min = 0.01,
        .tau_max = 1.0,
        .sigma_min = 0.15,
        .sigma_max = 0.40,
        .rate_min = 0.02,
        .rate_max = 0.05,
    };

    BSplineMultiKRefSurface surface(
        std::move(*multi), bounds, OptionType::PUT, 0.02);

    auto data = to_data(surface);

    EXPECT_EQ(data.surface_type, "bspline_4d_segmented");
    EXPECT_GT(data.segments.size(), 0u);
    EXPECT_EQ(data.option_type, OptionType::PUT);
    EXPECT_NEAR(data.dividend_yield, 0.02, 1e-15);

    for (const auto& seg : data.segments) {
        EXPECT_EQ(seg.interp_type, "bspline");
        EXPECT_EQ(seg.ndim, 4u);
    }

    auto loaded = from_data<BSplineMultiKRefInner>(data);
    ASSERT_TRUE(loaded.has_value()) << "from_data failed";

    struct TestPoint { double spot, strike, tau, sigma, rate; };
    std::vector<TestPoint> test_points = {
        {100.0, 100.0, 0.3, 0.25, 0.04},
        {100.0, 100.0, 0.8, 0.20, 0.03},
    };
    for (const auto& p : test_points) {
        double orig = surface.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        double load = loaded->price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        EXPECT_NEAR(orig, load, 1e-10)
            << "Mismatch at tau=" << p.tau << " sigma=" << p.sigma;
    }
}

// ===========================================================================
// Test 8: Chebyshev segmented round-trip
// ===========================================================================

TEST(PriceTableDataTest, ChebyshevSegmentedRoundTrip) {
    SegmentedAdaptiveConfig seg_config{
        .spot = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {Dividend{.calendar_time = 0.5, .amount = 2.0}},
        .maturity = 1.0,
        .kref_config = {.K_refs = {100.0}},
    };

    IVGrid grid{
        .moneyness = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2}),
        .vol = {0.10, 0.20, 0.30},
        .rate = {0.03, 0.05},
    };

    auto surface_result = build_chebyshev_segmented_manual(seg_config, grid);
    ASSERT_TRUE(surface_result.has_value()) << "build_chebyshev_segmented_manual failed";
    auto& surface = *surface_result;

    auto data = to_data(surface);

    EXPECT_EQ(data.surface_type, "chebyshev_4d_segmented");
    EXPECT_GT(data.segments.size(), 0u);
    EXPECT_EQ(data.option_type, OptionType::PUT);
    EXPECT_NEAR(data.dividend_yield, 0.02, 1e-15);

    for (const auto& seg : data.segments) {
        EXPECT_EQ(seg.interp_type, "chebyshev");
        EXPECT_EQ(seg.ndim, 4u);
    }

    auto loaded = from_data<ChebyshevMultiKRefInner>(data);
    ASSERT_TRUE(loaded.has_value()) << "from_data failed";

    struct TestPoint { double spot, strike, tau, sigma, rate; };
    std::vector<TestPoint> test_points = {
        {100.0, 100.0, 0.3, 0.20, 0.05},
        {100.0, 100.0, 0.8, 0.25, 0.04},
    };
    for (const auto& p : test_points) {
        double orig = surface.price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        double load = loaded->price(p.spot, p.strike, p.tau, p.sigma, p.rate);
        EXPECT_NEAR(orig, load, 1e-10)
            << "Mismatch at tau=" << p.tau << " sigma=" << p.sigma;
    }
}

// ===========================================================================
// Test 9: Type mismatch errors
// ===========================================================================

TEST(PriceTableDataTest, TypeMismatchReturnsError) {
    // Build minimal PriceTableData with bspline_4d type without doing PDE solves.
    PriceTableData data;
    data.surface_type = "bspline_4d";
    data.option_type = OptionType::PUT;
    data.dividend_yield = 0.0;
    data.segments.resize(1);
    data.segments[0].interp_type = "bspline";
    data.segments[0].ndim = 4;

    // B-spline data -> from_data<ChebyshevRawLeaf> -> error
    auto result1 = from_data<ChebyshevRawLeaf>(data);
    EXPECT_FALSE(result1.has_value())
        << "ChebyshevRawLeaf should reject bspline_4d data";

    // B-spline data -> from_data<BSplineMultiKRefInner> -> error
    auto result2 = from_data<BSplineMultiKRefInner>(data);
    EXPECT_FALSE(result2.has_value())
        << "BSplineMultiKRefInner should reject bspline_4d data";

    // B-spline data -> from_data<BSpline3DLeaf> -> error
    auto result3 = from_data<BSpline3DLeaf>(data);
    EXPECT_FALSE(result3.has_value())
        << "BSpline3DLeaf should reject bspline_4d data";

    // B-spline data -> from_data<Chebyshev3DRawLeaf> -> error
    auto result4 = from_data<Chebyshev3DRawLeaf>(data);
    EXPECT_FALSE(result4.has_value())
        << "Chebyshev3DRawLeaf should reject bspline_4d data";

    // from_data<ChebyshevLeaf> always fails
    data.surface_type = "chebyshev_4d";
    auto result5 = from_data<ChebyshevLeaf>(data);
    EXPECT_FALSE(result5.has_value())
        << "ChebyshevLeaf always returns error (Tucker not recoverable)";

    // from_data<Chebyshev3DLeaf> always fails
    data.surface_type = "chebyshev_3d";
    auto result6 = from_data<Chebyshev3DLeaf>(data);
    EXPECT_FALSE(result6.has_value())
        << "Chebyshev3DLeaf always returns error (Tucker not recoverable)";
}

// ===========================================================================
// Test 10: Segment metadata verification (BSpline 4D)
// ===========================================================================

TEST(PriceTableDataTest, BSpline4DSegmentMetadata) {
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    auto data = to_data(*surface);
    ASSERT_EQ(data.segments.size(), 1u);
    const auto& seg = data.segments[0];

    EXPECT_EQ(seg.segment_id, 0);
    EXPECT_NEAR(seg.K_ref, 100.0, 1e-10);
    EXPECT_GT(seg.tau_max, seg.tau_min);
    EXPECT_GT(seg.tau_max, 0.0);

    for (size_t d = 0; d < seg.ndim; ++d) {
        EXPECT_LT(seg.domain_lo[d], seg.domain_hi[d])
            << "domain_lo >= domain_hi for dim " << d;
    }

    // B-spline should have grids and knots
    EXPECT_EQ(seg.grids.size(), 4u);
    EXPECT_EQ(seg.knots.size(), 4u);
    EXPECT_GT(seg.values.size(), 0u);
}

// ===========================================================================
// Test 11: Segment metadata verification (Chebyshev 4D)
// ===========================================================================

TEST(PriceTableDataTest, Chebyshev4DSegmentMetadata) {
    ChebyshevTableConfig config{
        .num_pts = {8, 6, 6, 4},
        .domain = Domain<4>{
            .lo = {-0.30, 0.02, 0.10, 0.02},
            .hi = { 0.30, 1.50, 0.40, 0.08},
        },
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .tucker_epsilon = 0.0,
    };
    auto result = build_chebyshev_table(config);
    ASSERT_TRUE(result.has_value());
    auto& surface = std::get<ChebyshevRawSurface>(result->surface);

    auto data = to_data(surface);
    ASSERT_EQ(data.segments.size(), 1u);
    const auto& seg = data.segments[0];

    EXPECT_EQ(seg.segment_id, 0);
    EXPECT_NEAR(seg.K_ref, 100.0, 1e-10);

    // Chebyshev should NOT have grids/knots
    EXPECT_TRUE(seg.grids.empty());
    EXPECT_TRUE(seg.knots.empty());

    // But should have num_pts and values
    EXPECT_EQ(seg.num_pts.size(), 4u);
    EXPECT_GT(seg.values.size(), 0u);
}

// ===========================================================================
// Test 12: Segmented segment metadata verification
// ===========================================================================

TEST(PriceTableDataTest, SegmentedSegmentMetadata) {
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {
            .dividend_yield = 0.02,
            .discrete_dividends = {{.calendar_time = 0.5, .amount = 2.0}},
        },
        .grid = IVGrid{
            .moneyness = to_log_m({0.8, 0.9, 1.0, 1.1, 1.2}),
            .vol = {0.15, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.03, 0.04, 0.05},
        },
        .maturity = 1.0,
    };

    auto bspline_seg = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(bspline_seg.has_value());

    auto multi = build_multi_kref_surface({BSplineMultiKRefEntry{
        .K_ref = 100.0, .surface = std::move(*bspline_seg)}});
    ASSERT_TRUE(multi.has_value());

    SurfaceBounds bounds{
        .m_min = to_log_m({0.8}).front(),
        .m_max = to_log_m({1.2}).front(),
        .tau_min = 0.01,
        .tau_max = 1.0,
        .sigma_min = 0.15,
        .sigma_max = 0.40,
        .rate_min = 0.02,
        .rate_max = 0.05,
    };

    BSplineMultiKRefSurface surface(
        std::move(*multi), bounds, OptionType::PUT, 0.02);

    auto data = to_data(surface);
    EXPECT_GE(data.segments.size(), 2u)
        << "Segmented surface with 1 dividend should have >= 2 segments";

    for (const auto& seg : data.segments) {
        EXPECT_GE(seg.tau_start, 0.0);
        EXPECT_GT(seg.tau_end, seg.tau_start);
        EXPECT_NEAR(seg.K_ref, 100.0, 1e-10);
    }
}

// ===========================================================================
// Test 13: Data preserves maturity and option type
// ===========================================================================

TEST(PriceTableDataTest, MetadataPreservation) {
    auto setup = PriceTableBuilder::from_vectors(
        {-0.3, -0.1, 0.0, 0.1, 0.3},
        {0.1, 0.5, 1.0, 1.5},
        {0.10, 0.20, 0.30, 0.40},
        {0.02, 0.04, 0.06, 0.08},
        100.0, GridAccuracyParams{}, OptionType::PUT, 0.02);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = *setup;
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
    auto surface = make_bspline_surface(
        result->spline, result->K_ref, result->dividends.dividend_yield,
        OptionType::PUT);
    ASSERT_TRUE(surface.has_value());

    auto data = to_data(*surface);

    EXPECT_EQ(data.option_type, OptionType::PUT);
    EXPECT_NEAR(data.dividend_yield, 0.02, 1e-15);
    EXPECT_GT(data.maturity, 0.0);

    auto loaded = from_data<BSplineLeaf>(data);
    ASSERT_TRUE(loaded.has_value());

    EXPECT_EQ(loaded->option_type(), OptionType::PUT);
    EXPECT_NEAR(loaded->dividend_yield(), 0.02, 1e-15);
}

}  // namespace
}  // namespace mango
