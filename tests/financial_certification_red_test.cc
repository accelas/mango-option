// SPDX-License-Identifier: MIT
// Gate #459 public REDs. Explicit manual target until public activation follows
// the final #460/#461 parents; internal proof tests run independently.
#include "mango/option/table/bspline/bspline_surface.hpp"
#include <gtest/gtest.h>
#include <array>
#include <memory>
#include <limits>
#include "mango/option/price_table_factory.hpp"
#include "mango/option/table/serialization/from_data.hpp"
#include "mango/option/table/serialization/to_data.hpp"
#include "mango/option/table/serialization/extract_segments.hpp"
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"

namespace {
using namespace mango;

std::shared_ptr<const BSplineND<double,4>> hidden_pocket_eep() {
    const std::array<std::pair<double,double>,4> ranges{{
        {-.001,.001},{.1,.11},{.2,.20001},{.04,.06}}};
    BSplineND<double,4>::GridArray grid;
    BSplineND<double,4>::KnotArray knots;
    for (std::size_t d=0;d<4;++d) {
        const auto [lo,hi]=ranges[d];
        for (int i=0;i<4;++i) grid[d].push_back(lo+(hi-lo)*i/3);
        knots[d]={lo,lo,lo,lo,hi,hi,hi,hi};
    }
    constexpr double a=17.0/32, c=a*a-1.0/(128*128);
    // A positive, sub-dollar EEP; its narrow negative-vega pocket outweighs
    // the European vega at the middle of the admitted short-maturity domain.
    const std::array<double,4> sigma_coeff{.1,.1+c,.1+2*c-a,1.1+3*c-3*a};
    std::vector<double> coefficients(256);
    for (std::size_t i=0;i<256;++i) coefficients[i]=sigma_coeff[(i/4)%4];
    auto spline=BSplineND<double,4>::create(std::move(grid),std::move(knots),std::move(coefficients));
    if (!spline) return {};
    return std::make_shared<const BSplineND<double,4>>(std::move(*spline));
}

TEST(FinancialCertificationRedTest, ManualPublicationRejectsHiddenPhysicalVegaPocket) {
    auto spline=hidden_pocket_eep();
    ASSERT_TRUE(spline);
    BSplineLeaf raw(BSplineTransformLeaf(SharedBSplineInterp<4>(spline),StandardTransform4D{},100),
                    AnalyticalEEP(OptionType::PUT,0));
    constexpr double lo=.2, hi=.20001;
    const double center=lo+(hi-lo)*17/32;
    ASSERT_LT(raw.vega(100,100,.105,center,.05),0);
    for (int i=1;i<=16;++i) {
        const double before=lo+(hi-lo)*(i-1)/16;
        const double after=lo+(hi-lo)*i/16;
        ASSERT_GT(raw.price(100,100,.105,after,.05),raw.price(100,100,.105,before,.05));
    }
    auto published=make_bspline_surface(spline,100,0,OptionType::PUT);
    ASSERT_FALSE(published.has_value())
        << "Financial publication must prove physical sigma monotonicity, not accept a 17-point scan";
    EXPECT_EQ(published.error().code, PriceTableErrorCode::NonMonotoneSurface);
}
TEST(FinancialCertificationRedTest, ManualPublicationRetainsACertifiedPositivePayload) {
    const auto original = hidden_pocket_eep();
    ASSERT_TRUE(original);
    BSplineND<double, 4>::GridArray grids;
    BSplineND<double, 4>::KnotArray knots;
    for (std::size_t d = 0; d < 4; ++d) {
        grids[d] = original->grid(d);
        knots[d] = original->knots(d);
    }
    auto spline = BSplineND<double, 4>::create(
        std::move(grids), std::move(knots), std::vector<double>(256, 0.0));
    ASSERT_TRUE(spline);
    auto published = make_bspline_surface(
        std::make_shared<const BSplineND<double, 4>>(std::move(*spline)),
        100.0, 0.0, OptionType::PUT);
    ASSERT_TRUE(published);
    EXPECT_EQ(published->proof_status(), PriceProofStatus::Certified);
    EXPECT_GT(published->price(100, 100, .105, .200005, .05), 0.0);
    EXPECT_GT(published->vega(100, 100, .105, .200005, .05), 0.0);
}

TEST(FinancialCertificationRedTest, LoadingReprovesTheStoredPhysicalPayload) {
    const auto spline = hidden_pocket_eep();
    ASSERT_TRUE(spline);
    BSplineLeaf leaf(BSplineTransformLeaf(SharedBSplineInterp<4>(spline),
        StandardTransform4D{}, 100.0), AnalyticalEEP(OptionType::PUT, 0.0));
    const SurfaceBounds bounds{-.001, .001, .1, .11, .2, .20001, .04, .06};
    // The numerical record can come from a historical/manual producer. Its
    // current physical sigma shape must be proved again before publication.
    PriceTableData record;
    record.surface_type = surface_types::kBSpline4D;
    record.option_type = OptionType::PUT;
    record.dividend_yield = 0;
    record.bounds_m_min = bounds.m_min;
    record.bounds_m_max = bounds.m_max;
    record.bounds_tau_min = bounds.tau_min;
    record.bounds_tau_max = record.maturity = bounds.tau_max;
    record.bounds_sigma_min = bounds.sigma_min;
    record.bounds_sigma_max = bounds.sigma_max;
    record.bounds_rate_min = bounds.rate_min;
    record.bounds_rate_max = bounds.rate_max;
    record.ratio_bounds = MoneynessBounds{std::exp(bounds.m_min), std::exp(bounds.m_max)};
    extract_segments(leaf, record.segments, 100, 0, bounds.tau_max,
        bounds.tau_min, bounds.tau_max);
    auto loaded = from_data<BSplineLeaf>(record);
    ASSERT_FALSE(loaded);
    EXPECT_EQ(loaded.error().code, PriceTableErrorCode::NonMonotoneSurface);
}

TEST(FinancialCertificationRedTest, AllRepresentationsRecomputeEvidenceAndPreserveFlatPrices) {
    auto check = []<class Inner>(const PriceTableData& record, bool segmented) {
        auto loaded = from_data<Inner>(record);
        ASSERT_TRUE(loaded) << static_cast<int>(loaded.error().code);
        EXPECT_EQ(loaded->proof_status(), PriceProofStatus::Certified);
        EXPECT_GT(loaded->proof_work(), 0u);
        using Table = PriceTable<Inner>;
        auto iv = InterpolatedIVSolver<Table>::create(*loaded);
        ASSERT_TRUE(iv);
        auto any_iv = make_any_interpolated_solver(std::move(*iv));
        EXPECT_EQ(any_iv.proof_status(), loaded->proof_status());
        EXPECT_EQ(any_iv.proof_work(), loaded->proof_work());
        using View = detail::SharedPriceTableSurface<Table>;
        auto shared_iv = InterpolatedIVSolver<View>::create(
            View(std::make_shared<const Table>(*loaded)));
        ASSERT_TRUE(shared_iv);
        auto any_shared = make_any_interpolated_solver(std::move(*shared_iv));
        EXPECT_EQ(any_shared.proof_status(), loaded->proof_status());
        EXPECT_EQ(any_shared.proof_work(), loaded->proof_work());
        auto original_handle = *loaded;
        auto retained_handle = std::move(original_handle);
        auto empty_iv = InterpolatedIVSolver<Table>::create(std::move(original_handle));
        ASSERT_FALSE(empty_iv);
        EXPECT_EQ(empty_iv.error().code, ValidationErrorCode::CertificationIndeterminate);
        EXPECT_TRUE(InterpolatedIVSolver<Table>::create(std::move(retained_handle)));
        if (!segmented) {
            auto fixed = record;
            fixed.bounds_m_min = fixed.bounds_m_max = 0;
            fixed.ratio_bounds = MoneynessBounds{1, 1};
            fixed.bounds_tau_min = fixed.bounds_tau_max = fixed.maturity = .3;
            fixed.bounds_rate_min = fixed.bounds_rate_max = .05;
            auto fixed_table = from_data<Inner>(fixed);
            ASSERT_TRUE(fixed_table);
            EXPECT_TRUE(InterpolatedIVSolver<Table>::create(*fixed_table));
            EXPECT_FALSE(InterpolatedIVSolver<Table>::create(*fixed_table, {},
                std::vector<Dividend>{{.1, 1.0}}));
        }
        const double price = loaded->price(100, 100, .3, .25, .05);
        EXPECT_TRUE(std::isfinite(price));
        EXPECT_GT(price, 0.0);
        if (segmented) {
            EXPECT_DOUBLE_EQ(price, 10.0);
            // Cubic basis differentiation may leave binary64 cancellation
            // noise even when the stored polynomial is structurally flat.
            EXPECT_NEAR(loaded->vega(100, 100, .3, .25, .05), 0.0, 1e-12);
        }
    };
    for (bool modal : {false, true}) for (int variant = 0; variant < 3; ++variant) {
        const bool dimensionless = variant == 1, segmented = variant == 2;
        SCOPED_TRACE(::testing::Message() << "modal=" << modal << " variant=" << variant);
        PriceTableData record;
        record.surface_type = modal
            ? (segmented ? surface_types::kChebyshev4DSegmented : dimensionless
                ? surface_types::kChebyshev3DRaw : surface_types::kChebyshev4DRaw)
            : (segmented ? surface_types::kBSpline4DSegmented : dimensionless
                ? surface_types::kBSpline3D : surface_types::kBSpline4D);
        record.option_type = OptionType::PUT;
        record.dividend_yield = 0.0;
        record.ratio_bounds = MoneynessBounds{.99, 1.01};
        record.bounds_m_min = std::log(.99);
        record.bounds_m_max = std::log(1.01);
        record.bounds_tau_min = .2;
        record.bounds_tau_max = record.maturity = .4;
        record.bounds_sigma_min = .2;
        record.bounds_sigma_max = .3;
        record.bounds_rate_min = .04;
        record.bounds_rate_max = .06;
        if (segmented) {
            record.strike_bounds = StrikeBounds{95, 105};
            record.fixed_expiry = FixedExpiryMetadata{1, {}};
        }
        for (double reference : segmented ? std::vector<double>{80, 120}
                                           : std::vector<double>{100}) {
            PriceTableData::Segment segment;
            segment.segment_id = static_cast<int32_t>(record.segments.size());
            segment.K_ref = reference;
            segment.tau_start = 0;
            segment.tau_end = segment.tau_max = 1;
            segment.tau_min = 0;
            segment.ndim = dimensionless ? 3 : 4;
            segment.interp_type = modal ? "chebyshev_modal" : "bspline";
            segment.domain_lo = dimensionless ? std::vector<double>{-.02, .001, -2}
                                              : std::vector<double>{-.02, 0, .1, .01};
            segment.domain_hi = dimensionless ? std::vector<double>{.02, .2, 2}
                                              : std::vector<double>{.02, 1, .4, .1};
            segment.num_pts.assign(segment.ndim, modal ? 2 : 4);
            std::size_t count = 1;
            for (std::size_t d = 0; d < segment.num_pts.size(); ++d) {
                count *= segment.num_pts[d];
                if (modal) continue;
                const double lo = segment.domain_lo[d], hi = segment.domain_hi[d];
                segment.grids.push_back({lo, lo + (hi-lo)/3, lo + 2*(hi-lo)/3, hi});
                segment.knots.push_back({lo,lo,lo,lo,hi,hi,hi,hi});
            }
            segment.values.assign(count, segmented && !modal ? .1 : 0.0);
            if (segmented && modal) segment.values.front() = .1;
            record.segments.push_back(std::move(segment));
        }
        if (modal) {
            if (segmented) check.template operator()<ChebyshevMultiKRefInner>(record, true);
            else if (dimensionless) check.template operator()<Chebyshev3DLeaf>(record, false);
            else check.template operator()<ChebyshevLeaf>(record, false);
        } else {
            if (segmented) check.template operator()<BSplineMultiKRefInner>(record, true);
            else if (dimensionless) check.template operator()<BSpline3DLeaf>(record, false);
            else check.template operator()<BSplineLeaf>(record, false);
        }
    }
}

TEST(FinancialCertificationRedTest, ManualChebyshevNeverPublishesUnprovenNumerics) {
    const ChebyshevTableConfig config{
        .num_pts = {2, 2, 2, 2},
        .domain = Domain<4>{{-.01, .1, .2, .04}, {.01, .11, .21, .05}},
        .K_ref = 100,
        .option_type = OptionType::PUT,
        .dividend_yield = 0,
    };
    const auto result = build_chebyshev_table(config);
    if (result) {
        EXPECT_EQ(result->surface.proof_status(), PriceProofStatus::Certified);
    } else {
        // This is a construction-gate test, not a required-fit accuracy claim.
        EXPECT_TRUE(result.error().code == PriceTableErrorCode::NonMonotoneSurface ||
                    result.error().code == PriceTableErrorCode::CertificationIndeterminate);
    }
}

TEST(FinancialCertificationRedTest, DimensionlessPublicationRequiresTheExactZeroYieldModel) {
    IVSolverFactoryConfig config;
    config.option_type = OptionType::PUT;
    config.dividend_yield = 1e-13;
    config.grid.moneyness = {.9, 1.0, 1.1, 1.2};
    config.grid.vol = {.2, .21, .22, .23};
    config.grid.rate = {.03, .04, .05, .06};
    config.backend = DimensionlessBackend{.maturity = .5};
    const auto result = make_price_table(config);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ValidationErrorCode::InvalidDividend);
}

TEST(FinancialCertificationRedTest, DirectUncheckedPriceTableConstructionIsUnavailable) {
    EXPECT_FALSE((std::is_constructible_v<BSplinePriceTable, const BSplineLeaf&,
        const SurfaceBounds&, OptionType, double>));
}

struct ClaimedCertifiedCallback {
    PriceProofStatus proof_status() const { return PriceProofStatus::Certified; }
    double price(double, double, double, double sigma, double) const { return 10 + sigma; }
    double vega(double, double, double, double, double) const { return 1; }
    double m_min() const { return -.1; }
    double m_max() const { return .1; }
    double tau_min() const { return .1; }
    double tau_max() const { return 1; }
    double sigma_min() const { return .1; }
    double sigma_max() const { return .5; }
    double rate_min() const { return .01; }
    double rate_max() const { return .1; }
    OptionType option_type() const { return OptionType::PUT; }
    double dividend_yield() const { return 0; }
};

TEST(FinancialCertificationRedTest, IVAdmissionDoesNotTrustACallbackCertificationClaim) {
    auto solver = InterpolatedIVSolver<ClaimedCertifiedCallback>::create(ClaimedCertifiedCallback{});
    ASSERT_FALSE(solver);
    EXPECT_EQ(solver.error().code, ValidationErrorCode::UnsupportedRepresentation);
}

}

TEST(FinancialCertificationRedTest, FactoryAdmitsTheRequestedRatioEndpointBeforeLogRoundtrip) {
    using namespace mango;
    IVSolverFactoryConfig config;
    config.option_type=OptionType::PUT;
    config.spot=100;
    config.grid.moneyness={.1,.11,.12,.13};
    config.grid.vol={.18,.2,.22,.24};
    config.grid.rate={.02,.04,.06,.08};
    config.backend=BSplineBackend{.maturity_grid={.1,.2,.3,.4}};
    auto table=make_price_table(config);
    ASSERT_TRUE(table);
    PricingParams query(OptionSpec{.spot=10,.strike=100,.maturity=.2,.rate=.04,
        .dividend_yield=0,.option_type=OptionType::PUT},.2);
    EXPECT_TRUE(table->validate_pricing_params(query));
}

TEST(FinancialCertificationRedTest, QuoteRoundingCannotAdmitAnUnrelatedSubnormalRatio) {
    using namespace mango;
    const MoneynessDomain domain({std::exp(-.5), std::exp(.5)});
    const double quantum=std::numeric_limits<double>::denorm_min();
    ASSERT_EQ(quantum/(2*quantum),.5);
    EXPECT_FALSE(domain.contains_quote(quantum,2*quantum));
}
