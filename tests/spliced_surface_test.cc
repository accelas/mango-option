// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <cmath>
#include "mango/option/table/spliced_surface.hpp"

namespace mango {
namespace {

// Simple mock surface for testing
struct MockSurface {
    double offset = 0.0;

    double price(const PriceQuery& q) const {
        return q.spot / q.strike + q.tau + q.sigma + q.rate + offset;
    }

    double price(double spot, double strike, double tau, double sigma, double rate) const {
        return spot / strike + tau + sigma + rate + offset;
    }

    double vega(const PriceQuery& q) const {
        return q.tau * 100.0 + offset;  // Simple vega for testing
    }

    double vega(double, double, double tau, double, double) const {
        return tau * 100.0 + offset;
    }
};

static_assert(PriceSurface<MockSurface>, "MockSurface must satisfy PriceSurface");

// ===========================================================================
// Concept verification tests
// ===========================================================================

TEST(SplicedSurfaceTest, ConceptsCompile) {
    static_assert(SplitStrategy<SegmentLookup>);
    static_assert(SplitStrategy<LinearBracket>);
    static_assert(SliceTransform<IdentityTransform>);
    static_assert(SliceTransform<MaturityTransform>);
    static_assert(CombineStrategy<WeightedSum>);
    SUCCEED();
}

// ===========================================================================
// MaturityTransform tests
// ===========================================================================

TEST(MaturityTransformTest, SatisfiesSliceTransformConcept) {
    static_assert(SliceTransform<MaturityTransform>);
    SUCCEED();
}

TEST(MaturityTransformTest, ToLocalReturnsQueryUnchanged) {
    MaturityTransform xform{.option_type = OptionType::PUT, .dividend_yield = 0.02};
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.5, .sigma = 0.2, .rate = 0.05};

    PriceQuery local = xform.to_local(0, q);
    EXPECT_DOUBLE_EQ(local.spot, q.spot);
    EXPECT_DOUBLE_EQ(local.strike, q.strike);
    EXPECT_DOUBLE_EQ(local.tau, q.tau);
    EXPECT_DOUBLE_EQ(local.sigma, q.sigma);
    EXPECT_DOUBLE_EQ(local.rate, q.rate);
}

TEST(MaturityTransformTest, NormalizeValueAddsEuropeanPrice) {
    MaturityTransform xform{.option_type = OptionType::PUT, .dividend_yield = 0.0};
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.5, .sigma = 0.2, .rate = 0.05};

    // Compute expected European put price using bs_price
    double p_eu = bs_price(q.spot, q.strike, q.tau, q.sigma, q.rate, 0.0, OptionType::PUT);

    // With EEP = 0, the American price should equal the European price
    double american = xform.normalize_value(0, q, 0.0);
    EXPECT_DOUBLE_EQ(american, p_eu);

    // With non-zero EEP, the American price = EEP + P_Eu
    double eep = 1.5;
    american = xform.normalize_value(0, q, eep);
    EXPECT_DOUBLE_EQ(american, eep + p_eu);
}

TEST(MaturityTransformTest, WorksWithDividendYield) {
    MaturityTransform xform{.option_type = OptionType::PUT, .dividend_yield = 0.02};
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.5, .sigma = 0.2, .rate = 0.05};

    // European price with dividend yield
    double p_eu = bs_price(q.spot, q.strike, q.tau, q.sigma, q.rate, 0.02, OptionType::PUT);

    double eep = 2.0;
    double american = xform.normalize_value(0, q, eep);
    EXPECT_DOUBLE_EQ(american, eep + p_eu);
}

TEST(MaturityTransformTest, WorksWithCallOptions) {
    MaturityTransform xform{.option_type = OptionType::CALL, .dividend_yield = 0.03};
    PriceQuery q{.spot = 100.0, .strike = 95.0, .tau = 0.25, .sigma = 0.25, .rate = 0.04};

    double p_eu = bs_price(q.spot, q.strike, q.tau, q.sigma, q.rate, 0.03, OptionType::CALL);

    double eep = 0.5;
    double american = xform.normalize_value(0, q, eep);
    EXPECT_DOUBLE_EQ(american, eep + p_eu);
}

// ===========================================================================
// Surface adapter concept checks
// ===========================================================================

TEST(SurfaceAdapterTest, PriceTableSurface3DAdapterSatisfiesConcept) {
    static_assert(PriceSurface<PriceTableSurface3DAdapter>);
    SUCCEED();
}

TEST(SurfaceAdapterTest, AmericanPriceSurfaceAdapterSatisfiesConcept) {
    static_assert(PriceSurface<AmericanPriceSurfaceAdapter>);
    SUCCEED();
}

// ===========================================================================
// SegmentLookup tests
// ===========================================================================

TEST(SegmentLookupTest, FindsCorrectSegment) {
    SegmentLookup lookup({0.0, 0.5, 1.0}, {0.5, 1.0, 2.0});

    // Query in first segment
    auto br = lookup.bracket(0.25);
    EXPECT_EQ(br.size, 1);
    EXPECT_EQ(br.items[0].index, 0);
    EXPECT_DOUBLE_EQ(br.items[0].weight, 1.0);

    // Query in second segment
    br = lookup.bracket(0.75);
    EXPECT_EQ(br.size, 1);
    EXPECT_EQ(br.items[0].index, 1);

    // Query in third segment
    br = lookup.bracket(1.5);
    EXPECT_EQ(br.size, 1);
    EXPECT_EQ(br.items[0].index, 2);
}

TEST(SegmentLookupTest, ClampsToFirstSegment) {
    SegmentLookup lookup({0.5, 1.0}, {1.0, 2.0});

    auto br = lookup.bracket(0.1);  // Below first segment
    EXPECT_EQ(br.size, 1);
    EXPECT_EQ(br.items[0].index, 0);
}

TEST(SegmentLookupTest, ClampsToLastSegment) {
    SegmentLookup lookup({0.5, 1.0}, {1.0, 2.0});

    auto br = lookup.bracket(5.0);  // Above last segment
    EXPECT_EQ(br.size, 1);
    EXPECT_EQ(br.items[0].index, 1);
}

// ===========================================================================
// LinearBracket tests
// ===========================================================================

TEST(LinearBracketTest, InterpolatesMidpoint) {
    LinearBracket bracket({0.0, 1.0, 2.0});

    auto br = bracket.bracket(0.5);
    EXPECT_EQ(br.size, 2);
    EXPECT_EQ(br.items[0].index, 0);
    EXPECT_EQ(br.items[1].index, 1);
    EXPECT_NEAR(br.items[0].weight, 0.5, 1e-10);
    EXPECT_NEAR(br.items[1].weight, 0.5, 1e-10);
}

TEST(LinearBracketTest, InterpolatesQuarter) {
    LinearBracket bracket({0.0, 1.0});

    auto br = bracket.bracket(0.25);
    EXPECT_EQ(br.size, 2);
    EXPECT_NEAR(br.items[0].weight, 0.75, 1e-10);
    EXPECT_NEAR(br.items[1].weight, 0.25, 1e-10);
}

TEST(LinearBracketTest, ClampsBelow) {
    LinearBracket bracket({1.0, 2.0, 3.0});

    auto br = bracket.bracket(0.5);
    EXPECT_EQ(br.size, 1);
    EXPECT_EQ(br.items[0].index, 0);
    EXPECT_DOUBLE_EQ(br.items[0].weight, 1.0);
}

TEST(LinearBracketTest, ClampsAbove) {
    LinearBracket bracket({1.0, 2.0, 3.0});

    auto br = bracket.bracket(5.0);
    EXPECT_EQ(br.size, 1);
    EXPECT_EQ(br.items[0].index, 2);
    EXPECT_DOUBLE_EQ(br.items[0].weight, 1.0);
}

TEST(LinearBracketTest, AtGridPoint) {
    LinearBracket bracket({0.0, 1.0, 2.0});

    // At exact grid point tau=1.0
    auto br = bracket.bracket(1.0);
    EXPECT_EQ(br.size, 2);
    // Weight should be 0 for lower, 1 for upper (or could return single sample)
    double total = br.items[0].weight + br.items[1].weight;
    EXPECT_NEAR(total, 1.0, 1e-10);
}

// ===========================================================================
// KRefBracket tests
// ===========================================================================

TEST(KRefBracketTest, HandlesEmptyKRefs) {
    KRefBracket bracket({});  // Empty K_refs

    auto br = bracket.bracket(100.0);
    EXPECT_EQ(br.size, 0);  // Should return empty bracket, not crash
}

TEST(KRefBracketTest, HandlesSingleKRef) {
    KRefBracket bracket({100.0});  // Single K_ref

    auto br = bracket.bracket(80.0);
    EXPECT_EQ(br.size, 1);
    EXPECT_EQ(br.items[0].index, 0);
    EXPECT_DOUBLE_EQ(br.items[0].weight, 1.0);

    br = bracket.bracket(100.0);
    EXPECT_EQ(br.size, 1);
    EXPECT_EQ(br.items[0].index, 0);
    EXPECT_DOUBLE_EQ(br.items[0].weight, 1.0);

    br = bracket.bracket(120.0);
    EXPECT_EQ(br.size, 1);
    EXPECT_EQ(br.items[0].index, 0);
    EXPECT_DOUBLE_EQ(br.items[0].weight, 1.0);
}

TEST(KRefBracketTest, InterpolatesMidpoint) {
    KRefBracket bracket({80.0, 100.0, 120.0});

    auto br = bracket.bracket(90.0);  // Midpoint between 80 and 100
    EXPECT_EQ(br.size, 2);
    EXPECT_EQ(br.items[0].index, 0);
    EXPECT_EQ(br.items[1].index, 1);
    EXPECT_NEAR(br.items[0].weight, 0.5, 1e-10);
    EXPECT_NEAR(br.items[1].weight, 0.5, 1e-10);

    // Verify weights sum to 1.0
    double total = br.items[0].weight + br.items[1].weight;
    EXPECT_NEAR(total, 1.0, 1e-10);
}

TEST(KRefBracketTest, InterpolatesQuarter) {
    KRefBracket bracket({80.0, 120.0});

    auto br = bracket.bracket(90.0);  // 25% of the way from 80 to 120
    EXPECT_EQ(br.size, 2);
    EXPECT_NEAR(br.items[0].weight, 0.75, 1e-10);
    EXPECT_NEAR(br.items[1].weight, 0.25, 1e-10);

    // Verify weights sum to 1.0
    double total = br.items[0].weight + br.items[1].weight;
    EXPECT_NEAR(total, 1.0, 1e-10);
}

TEST(KRefBracketTest, ClampsBelow) {
    KRefBracket bracket({80.0, 100.0, 120.0});

    auto br = bracket.bracket(50.0);  // Below first K_ref
    EXPECT_EQ(br.size, 1);
    EXPECT_EQ(br.items[0].index, 0);
    EXPECT_DOUBLE_EQ(br.items[0].weight, 1.0);
}

TEST(KRefBracketTest, ClampsAbove) {
    KRefBracket bracket({80.0, 100.0, 120.0});

    auto br = bracket.bracket(150.0);  // Above last K_ref
    EXPECT_EQ(br.size, 1);
    EXPECT_EQ(br.items[0].index, 2);
    EXPECT_DOUBLE_EQ(br.items[0].weight, 1.0);
}

TEST(KRefBracketTest, AtGridPoint) {
    KRefBracket bracket({80.0, 100.0, 120.0});

    // At exact grid point K=100
    auto br = bracket.bracket(100.0);
    EXPECT_EQ(br.size, 2);
    // Weight should be 0 for lower, 1 for upper
    double total = br.items[0].weight + br.items[1].weight;
    EXPECT_NEAR(total, 1.0, 1e-10);
}

TEST(KRefBracketTest, ConceptCheck) {
    static_assert(SplitStrategy<KRefBracket>);
    SUCCEED();
}

// ===========================================================================
// SplicedSurface integration tests
// ===========================================================================

TEST(SplicedSurfaceTest, SegmentLookupWithMockSurface) {
    std::vector<MockSurface> slices = {{.offset = 0.0}, {.offset = 1.0}};
    SegmentLookup split({0.0, 0.5}, {0.5, 1.0});
    IdentityTransform xform;
    WeightedSum combine;

    SplicedSurface<MockSurface, SegmentLookup, IdentityTransform, WeightedSum>
        surface(std::move(slices), std::move(split), xform, combine);

    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.25, .sigma = 0.2, .rate = 0.05};
    double p1 = surface.price(q);

    // Should use first slice (offset=0)
    // Expected: 100/100 + 0.25 + 0.2 + 0.05 + 0 = 1.5
    EXPECT_NEAR(p1, 1.5, 1e-10);

    q.tau = 0.75;
    double p2 = surface.price(q);
    // Should use second slice (offset=1)
    // Expected: 100/100 + 0.75 + 0.2 + 0.05 + 1 = 3.0
    EXPECT_NEAR(p2, 3.0, 1e-10);
}

TEST(SplicedSurfaceTest, LinearInterpolationWithMockSurface) {
    std::vector<MockSurface> slices = {{.offset = 0.0}, {.offset = 2.0}};
    LinearBracket split({0.0, 1.0});
    IdentityTransform xform;
    WeightedSum combine;

    SplicedSurface<MockSurface, LinearBracket, IdentityTransform, WeightedSum>
        surface(std::move(slices), std::move(split), xform, combine);

    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.5, .sigma = 0.2, .rate = 0.05};

    // At tau=0.5, weights are (0.5, 0.5)
    // Slice 0: 1 + 0.5 + 0.2 + 0.05 + 0 = 1.75
    // Slice 1: 1 + 0.5 + 0.2 + 0.05 + 2 = 3.75
    // Interpolated: 0.5 * 1.75 + 0.5 * 3.75 = 2.75
    double p = surface.price(q);
    EXPECT_NEAR(p, 2.75, 1e-10);
}

TEST(SplicedSurfaceTest, VegaWorks) {
    std::vector<MockSurface> slices = {{.offset = 0.0}, {.offset = 10.0}};
    LinearBracket split({0.0, 1.0});
    IdentityTransform xform;
    WeightedSum combine;

    SplicedSurface<MockSurface, LinearBracket, IdentityTransform, WeightedSum>
        surface(std::move(slices), std::move(split), xform, combine);

    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.5, .sigma = 0.2, .rate = 0.05};

    // Vega at tau=0.5:
    // Slice 0: 0.5 * 100 + 0 = 50
    // Slice 1: 0.5 * 100 + 10 = 60
    // Interpolated: 0.5 * 50 + 0.5 * 60 = 55
    double v = surface.vega(q);
    EXPECT_NEAR(v, 55.0, 1e-10);
}

// ===========================================================================
// Composition test (nested SplicedSurface)
// ===========================================================================

TEST(SplicedSurfaceTest, CompositionWorks) {
    // Inner level: linear interpolation over tau
    using InnerSurface = SplicedSurface<MockSurface, LinearBracket, IdentityTransform, WeightedSum>;

    // Create two inner surfaces with different offsets
    std::vector<MockSurface> inner1_slices = {{.offset = 0.0}, {.offset = 1.0}};
    std::vector<MockSurface> inner2_slices = {{.offset = 10.0}, {.offset = 11.0}};

    InnerSurface inner1(std::move(inner1_slices), LinearBracket({0.0, 1.0}),
                        IdentityTransform{}, WeightedSum{});
    InnerSurface inner2(std::move(inner2_slices), LinearBracket({0.0, 1.0}),
                        IdentityTransform{}, WeightedSum{});

    // Outer level: segment lookup (like dividend segments)
    std::vector<InnerSurface> outer_slices;
    outer_slices.push_back(std::move(inner1));
    outer_slices.push_back(std::move(inner2));

    using OuterSurface = SplicedSurface<InnerSurface, SegmentLookup, IdentityTransform, WeightedSum>;

    OuterSurface outer(std::move(outer_slices),
                       SegmentLookup({0.0, 0.5}, {0.5, 1.0}),
                       IdentityTransform{}, WeightedSum{});

    // Query in first segment (uses inner1)
    PriceQuery q{.spot = 100.0, .strike = 100.0, .tau = 0.25, .sigma = 0.2, .rate = 0.05};
    double p1 = outer.price(q);

    // Query in second segment (uses inner2)
    q.tau = 0.75;
    double p2 = outer.price(q);

    // Second segment should have +10 offset from inner2
    EXPECT_GT(p2, p1 + 9.0);  // At least 10 more due to offset difference
}

// ===========================================================================
// Regression tests for Codex review critical issues
// ===========================================================================

// Regression: KRefTransform::denormalize() must be called after combining
// Bug: denormalize() existed but was never invoked by SplicedSurface
TEST(KRefTransformTest, DenormalizeIsCalled) {
    // Mock surface returns normalized price (price / K_ref)
    struct NormalizedMockSurface {
        double k_ref = 100.0;
        double price(const PriceQuery& q) const {
            // Return price as if already normalized by K_ref
            return q.spot / k_ref;  // e.g., spot=100 -> 1.0
        }
        double vega(const PriceQuery&) const { return 0.01; }
    };
    static_assert(PriceSurface<NormalizedMockSurface>);

    std::vector<NormalizedMockSurface> slices = {{.k_ref = 100.0}};
    LinearBracket split({0.5});  // Single point
    KRefTransform xform{.k_refs = {100.0}};
    WeightedSum combine;

    SplicedSurface<NormalizedMockSurface, LinearBracket, KRefTransform, WeightedSum>
        surface(std::move(slices), std::move(split), xform, combine);

    // Query with spot=100, strike=120
    PriceQuery q{.spot = 100.0, .strike = 120.0, .tau = 0.5, .sigma = 0.2, .rate = 0.05};
    double p = surface.price(q);

    // Without denormalize: would return (100/100) / 100 = 0.01
    // With denormalize: (100/100) / 100 * strike = 0.01 * 120 = 1.2
    //
    // Actually the flow is:
    // 1. slice.price(q) = 100/100 = 1.0 (normalized mock returns spot/k_ref)
    // 2. normalize_value: 1.0 / k_refs[0] = 1.0 / 100 = 0.01
    // 3. combine: 0.01
    // 4. denormalize: 0.01 * strike = 0.01 * 120 = 1.2
    EXPECT_NEAR(p, 1.2, 1e-10);
}

// Regression: SegmentLookup::bracket() must handle empty slices
// Bug: Called tau_start_.front() without checking if vector is empty
TEST(SegmentLookupTest, HandlesEmptySlices) {
    SegmentLookup lookup({}, {});  // Empty vectors

    auto br = lookup.bracket(0.5);
    EXPECT_EQ(br.size, 0);  // Should return empty bracket, not crash
}

// Regression: LinearBracket::bracket() must handle empty grid
TEST(LinearBracketTest, HandlesEmptyGrid) {
    LinearBracket bracket({});  // Empty grid

    auto br = bracket.bracket(0.5);
    EXPECT_EQ(br.size, 0);  // Should return empty bracket, not crash
}

// ===========================================================================
// Unified type alias tests
// ===========================================================================

// Verify PerMaturitySurface type alias compiles
TEST(UnifiedTypeAliasTest, PerMaturitySurfaceCompiles) {
    // Static assert that the type alias resolves correctly
    static_assert(std::is_same_v<
        PerMaturitySurface,
        SplicedSurface<PriceTableSurface3DAdapter, LinearBracket, MaturityTransform, WeightedSum>>);
    SUCCEED();
}

// Verify SegmentedSurface type alias compiles with default template parameter
TEST(UnifiedTypeAliasTest, SegmentedSurfaceDefaultCompiles) {
    // Default Inner = AmericanPriceSurfaceAdapter
    static_assert(std::is_same_v<
        SegmentedSurface<>,
        SplicedSurface<AmericanPriceSurfaceAdapter, SegmentLookup, SegmentedTransform, WeightedSum>>);
    SUCCEED();
}

// Verify SegmentedSurface type alias compiles with explicit MockSurface
TEST(UnifiedTypeAliasTest, SegmentedSurfaceWithMockCompiles) {
    static_assert(std::is_same_v<
        SegmentedSurface<MockSurface>,
        SplicedSurface<MockSurface, SegmentLookup, SegmentedTransform, WeightedSum>>);
    SUCCEED();
}

// Verify MultiKRefSurface type alias compiles with default template parameter
TEST(UnifiedTypeAliasTest, MultiKRefSurfaceDefaultCompiles) {
    // Default Inner = SegmentedSurface<> = SegmentedSurface<AmericanPriceSurfaceAdapter>
    static_assert(std::is_same_v<
        MultiKRefSurface<>,
        SplicedSurface<SegmentedSurface<>, KRefBracket, KRefTransform, WeightedSum>>);
    SUCCEED();
}

// Verify MultiKRefSurface type alias compiles with explicit MockSurface
TEST(UnifiedTypeAliasTest, MultiKRefSurfaceWithMockCompiles) {
    static_assert(std::is_same_v<
        MultiKRefSurface<MockSurface>,
        SplicedSurface<MockSurface, KRefBracket, KRefTransform, WeightedSum>>);
    SUCCEED();
}

// Verify nested composition: MultiKRefSurface<SegmentedSurface<PerMaturitySurface>>
TEST(UnifiedTypeAliasTest, FullyNestedCompositionCompiles) {
    // This represents the full composition for discrete dividends with per-maturity surfaces
    using NestedType = MultiKRefSurface<SegmentedSurface<PerMaturitySurface>>;

    // Verify it expands to the expected type
    static_assert(std::is_same_v<
        NestedType,
        SplicedSurface<
            SplicedSurface<
                SplicedSurface<PriceTableSurface3DAdapter, LinearBracket, MaturityTransform, WeightedSum>,
                SegmentLookup,
                SegmentedTransform,
                WeightedSum>,
            KRefBracket,
            KRefTransform,
            WeightedSum>>);
    SUCCEED();
}

// Integration test: SegmentedSurface with MockSurface can be instantiated
TEST(UnifiedTypeAliasTest, SegmentedSurfaceInstantiates) {
    std::vector<MockSurface> slices = {{.offset = 0.0}, {.offset = 1.0}};
    SegmentLookup split({0.0, 0.5}, {0.5, 1.0});
    SegmentedTransform xform{
        .tau_start = {0.0, 0.5},
        .tau_min = {0.0, 0.0},
        .tau_max = {0.5, 0.5},
        .content = {SurfaceContent::EarlyExercisePremium, SurfaceContent::EarlyExercisePremium},
        .dividends = {},
        .K_ref = 100.0,
        .T = 1.0
    };
    WeightedSum combine;

    SegmentedSurface<MockSurface> surface(
        std::move(slices), std::move(split), std::move(xform), combine);

    EXPECT_EQ(surface.num_slices(), 2);
}

// Integration test: MultiKRefSurface with MockSurface can be instantiated
TEST(UnifiedTypeAliasTest, MultiKRefSurfaceInstantiates) {
    std::vector<MockSurface> slices = {{.offset = 0.0}, {.offset = 1.0}};
    KRefBracket split({80.0, 100.0});
    KRefTransform xform{.k_refs = {80.0, 100.0}};
    WeightedSum combine;

    MultiKRefSurface<MockSurface> surface(
        std::move(slices), std::move(split), std::move(xform), combine);

    EXPECT_EQ(surface.num_slices(), 2);
}

} // namespace
} // namespace mango
