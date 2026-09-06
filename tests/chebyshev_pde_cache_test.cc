// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include "mango/option/american_option.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/chebyshev/chebyshev_pde_cache.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"

using namespace mango;

TEST(ChebyshevPDECacheTest, MissingPairsReturnsAllInitially) {
    ChebyshevPDECache cache;
    std::vector<double> sigmas = {0.10, 0.20, 0.30};
    std::vector<double> rates = {0.03, 0.05};
    auto missing = cache.missing_pairs(sigmas, rates);
    EXPECT_EQ(missing.size(), 6u);  // 3 x 2
}

TEST(ChebyshevPDECacheTest, StoreAndRetrieveSlice) {
    ChebyshevPDECache cache;
    std::vector<double> x = {0.0, 0.5, 1.0};
    std::vector<double> v = {1.0, 1.5, 2.0};
    cache.store_slice(0.20, 0.05, /*tau_idx=*/0, x, v);

    auto* spline = cache.get_slice(0.20, 0.05, 0);
    ASSERT_NE(spline, nullptr);
    EXPECT_NEAR(spline->eval(0.25), 1.25, 0.1);
}

TEST(ChebyshevPDECacheTest, MissingPairsExcludesCached) {
    ChebyshevPDECache cache;
    std::vector<double> x = {0.0, 0.5, 1.0};
    std::vector<double> v = {1.0, 1.5, 2.0};
    cache.store_slice(0.20, 0.05, 0, x, v);

    std::vector<double> sigmas = {0.10, 0.20, 0.30};
    std::vector<double> rates = {0.03, 0.05};
    auto missing = cache.missing_pairs(sigmas, rates);
    // (0.20, 0.05) is cached, so 5 remain
    EXPECT_EQ(missing.size(), 5u);
}

TEST(ChebyshevPDECacheTest, QuantizationMatchesCrossLevel) {
    ChebyshevPDECache cache;
    std::vector<double> x = {0.0, 0.5, 1.0};
    std::vector<double> v = {1.0, 1.5, 2.0};
    // Store at a value computed one way
    double sigma = 0.05 + (0.50 - 0.05) * 0.5;  // 0.275
    cache.store_slice(sigma, 0.05, 0, x, v);

    // Query with a value computed a different way (same physical value)
    double sigma2 = 0.275;
    auto* spline = cache.get_slice(sigma2, 0.05, 0);
    ASSERT_NE(spline, nullptr);
}

// ===========================================================================
// Regression tests for the #419 incident closure (D6)
// ===========================================================================

// Regression: a NaN PDE slice was stored as invalid, then extraction did
// `if (!spline) continue` over a zero-initialized tensor — the surface built
// "successfully" out of silent zeros
// Bug: ChebyshevPDECache::store_slice discarded the CubicSpline build error
// and build_segment_leaves treated missing slices as skippable
TEST(ChebyshevPDECacheTest, InvalidSliceFailsSegmentExtraction) {
    mango::ChebyshevPDECache cache;
    std::vector<double> x = {-0.5, 0.0, 0.5, 1.0};
    std::vector<double> bad = {0.1, std::nan(""), 0.2, 0.3};
    cache.store_slice(0.2, 0.05, 0, x, bad);
    ASSERT_EQ(cache.get_slice(0.2, 0.05, 0), nullptr);  // marked invalid

    std::vector<double> seg_bounds = {0.0, 1.0};
    std::vector<bool> seg_is_gap = {false};
    std::vector<double> m = {-0.5, 0.0, 0.5};
    std::vector<double> tau = {0.5};
    std::vector<double> sigma = {0.2};
    std::vector<double> rate = {0.05};

    auto leaves = mango::detail::build_segment_leaves(
        cache, /*K_ref=*/100.0, seg_bounds, seg_is_gap, /*include_gaps=*/false,
        m, tau, sigma, rate);
    ASSERT_FALSE(leaves.has_value());
    EXPECT_EQ(leaves.error().code, mango::PriceTableErrorCode::ExtractionFailed);
}

// Regression: a non-gap segment containing no tau nodes silently became a
// zeros-placeholder leaf, pricing the whole real segment as 0
// Bug: the Nt_seg == 0 placeholder branch did not check seg_is_gap
TEST(ChebyshevPDECacheTest, EmptyRealSegmentFailsExtraction) {
    mango::ChebyshevPDECache cache;
    std::vector<double> x = {-0.5, 0.0, 0.5, 1.0};
    std::vector<double> good = {0.1, 0.15, 0.2, 0.3};
    // Only one tau node is supplied below (tau = {0.75}), so it maps to
    // tau_idx 0 within build_segment_leaves.
    cache.store_slice(0.2, 0.05, 0, x, good);
    ASSERT_NE(cache.get_slice(0.2, 0.05, 0), nullptr);  // valid slice

    // Two real segments: [0.0, 0.5) and [0.5, 1.0]. The single tau node
    // falls entirely in the second segment, so the first segment has zero
    // tau nodes despite being a real (non-gap) segment.
    std::vector<double> seg_bounds = {0.0, 0.5, 1.0};
    std::vector<bool> seg_is_gap = {false, false};
    std::vector<double> m = {-0.5, 0.0, 0.5};
    std::vector<double> tau = {0.75};
    std::vector<double> sigma = {0.2};
    std::vector<double> rate = {0.05};

    auto leaves = mango::detail::build_segment_leaves(
        cache, /*K_ref=*/100.0, seg_bounds, seg_is_gap, /*include_gaps=*/true,
        m, tau, sigma, rate);
    ASSERT_FALSE(leaves.has_value());
    EXPECT_EQ(leaves.error().code, mango::PriceTableErrorCode::ExtractionFailed);
}

// Regression #485: cardinal evaluations expose the raw sampled rows, so
// no off-node tensor fit or reference-strike blend can mask timeline errors.
TEST(ChebyshevPDECacheTest, FixedExpiryRowsMatchRolledContracts) {
    const std::vector<Dividend> dividends = {{0.25, 3.0}};
    const std::vector<double> bounds = {0.1, 0.7495, 0.7505, 1.0};
    const std::vector<bool> gaps = {false, true, false};
    const std::vector<double> m = {-0.7, 0.0, 0.7};
    const std::vector<double> tau = {0.1, 0.42475, 0.7495, 0.7505, 0.87525, 1.0};
    const std::vector<double> sigma = {0.1, 0.2};
    const std::vector<double> rate = {0.03, 0.05};
    for (auto type : {OptionType::PUT, OptionType::CALL}) {
        auto pieces = build_chebyshev_segmented_pieces(
            100.0, type, 0.0, dividends, bounds, gaps, m, tau, sigma, rate);
        ASSERT_TRUE(pieces.has_value());
        EXPECT_EQ(pieces->pde_solves, 4u);
        for (size_t j = 0; j < tau.size(); ++j) {
            for (double vol : sigma) {
                PricingParams p(OptionSpec{.spot = 100.0, .strike = 100.0,
                    .maturity = tau[j], .rate = 0.05, .option_type = type}, vol);
                // Independent fixed-expiry oracle: event at backward .75.
                if (tau[j] > 0.75) p.discrete_dividends = {{tau[j] - 0.75, 3.0}};
                auto acc = make_grid_accuracy(GridAccuracyProfile::High);
                auto direct = AmericanOptionSolver::create(p, PDEGridSpec{acc});
                ASSERT_TRUE(direct.has_value());
                auto ref = direct->solve();
                ASSERT_TRUE(ref.has_value());
                acc = make_grid_accuracy(GridAccuracyProfile::Ultra);
                auto finer = AmericanOptionSolver::create(p, PDEGridSpec{acc});
                ASSERT_TRUE(finer.has_value());
                auto converged = finer->solve();
                ASSERT_TRUE(converged.has_value());
                ASSERT_NEAR(ref->value(), converged->value(), 0.001);
                const size_t leaf = j < 3 ? 0 : 1;
                const double local_tau = tau[j] - pieces->tau_split.tau_start()[leaf];
                const double got = 100.0 * pieces->leaves[leaf].price(
                    100.0, 100.0, local_tau, vol, 0.05);
                std::cout << "TIMELINE type=" << static_cast<int>(type)
                          << " tau=" << tau[j] << " sigma=" << vol
                          << " got=" << got << " fixed=" << converged->value()
                          << " convergence=" << std::abs(ref->value() - converged->value())
                          << '\n';
                EXPECT_NEAR(got, converged->value(), 0.003);
            }
        }
    }
}

// Regression #485: the leaf and the router must use the same time origin.
TEST(ChebyshevPDECacheTest, SegmentedRoutingPreservesLabelledTime) {
    ChebyshevPDECache cache;
    std::vector<double> m = {-0.2, 0.2}, tau = {0.01, 0.4995, 0.5005, 1.0};
    std::vector<double> sigmas = {0.1, 0.2}, rates = {0.03, 0.05};
    std::vector<double> bounds = {0.01, 0.4995, 0.5005, 1.0};
    std::vector<bool> gaps = {false, true, false};
    for (double sigma : sigmas) for (double rate : rates) {
        for (size_t j = 0; j < tau.size(); ++j) {
            std::vector<double> x = {-0.3, 0.0, 0.3};
            std::vector<double> values(3, tau[j]);
            cache.store_slice(sigma, rate, j, x, values);
        }
    }
    auto leaves = detail::build_segment_leaves(cache, 100.0, bounds, gaps,
        false, m, tau, sigmas, rates);
    ASSERT_TRUE(leaves.has_value());
    ChebyshevTauSegmented surface(std::move(*leaves),
        make_tau_split_from_segments(bounds, gaps, 100.0));
    EXPECT_NEAR(surface.price(100.0, 100.0, 0.7, 0.15, 0.04), 70.0, 1e-11);
    // An omitted event neighborhood cannot silently become another time.
    EXPECT_FALSE(std::isfinite(surface.price(100.0, 100.0, 0.5, 0.15, 0.04)));
    PriceTable<ChebyshevTauSegmented> table(std::move(surface),
        SurfaceBounds{-0.2, 0.2, 0.01, 1.0, 0.1, 0.2, 0.03, 0.05},
        OptionType::PUT, 0.0);
    EXPECT_FALSE(table.contains_maturity(0.5));
    PricingParams p(OptionSpec{.spot = 100.0, .strike = 100.0,
        .maturity = 0.5, .rate = 0.04, .option_type = OptionType::PUT}, 0.15);
    auto gamma = table.gamma(p);
    ASSERT_FALSE(gamma.has_value());
    EXPECT_EQ(gamma.error(), GreekError::OutOfDomain);
    auto solver = InterpolatedIVSolver<PriceTable<ChebyshevTauSegmented>>::create(
        std::move(table));
    ASSERT_TRUE(solver.has_value());
    IVQuery query(static_cast<const OptionSpec&>(p), 5.0);
    auto iv = solver->solve(query);
    ASSERT_FALSE(iv.has_value());
    EXPECT_EQ(iv.error().code, IVErrorCode::InvalidGridConfig);
}

TEST(ChebyshevPDECacheTest, RejectsEventInsideUnsplittableLeaf) {
    SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = OptionType::PUT,
        .discrete_dividends = {{0.9895, 1.0}}, .maturity = 1.0,
        .kref_config = {.K_refs = {100.0}}};
    IVGrid domain{.moneyness = {-0.2, 0.0, 0.2},
        .vol = {0.1, 0.2}, .rate = {0.03, 0.05}};
    // Backward event .0105 is too close to the published lower bound .01
    // for this topology's inset. Dropping the split would cross the jump.
    auto builder = ChebyshevSegmentedBuilder::create(config, domain);
    ASSERT_FALSE(builder.has_value());
    EXPECT_EQ(builder.error().code, PriceTableErrorCode::InvalidConfig);
}
