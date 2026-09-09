// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include "mango/option/american_option.hpp"
#include "mango/math/chebyshev/chebyshev_nodes.hpp"
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

// #486: an unrepresentable manual CC level is a configuration error.
// The node helper formerly shifted 1u by the unchecked requested level.
TEST(ChebyshevPDECacheTest, RejectsUnrepresentableManualLevels) {
    SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = OptionType::PUT,
        .discrete_dividends = {{0.01, 1.0}}, .maturity = 0.025,
        .kref_config = {.K_refs = {100.0}},
        .strike_bounds = StrikeBounds{100.0, 100.0}};
    IVGrid domain{.moneyness = {-0.1, 0.0, 0.1},
        .vol = {0.15, 0.25}, .rate = {0.03, 0.05}};
    auto builder = ChebyshevSegmentedBuilder::create(config, domain);
    ASSERT_TRUE(builder.has_value());
    std::vector<std::array<size_t, 4>> invalid_levels;
    for (size_t axis = 0; axis < 4; ++axis) {
        for (size_t invalid : {size_t{std::numeric_limits<unsigned>::digits},
                               size_t{std::numeric_limits<size_t>::digits},
                               std::numeric_limits<size_t>::max()}) {
            std::array<size_t, 4> levels{1, 1, 1, 1};
            levels[axis] = invalid;
            invalid_levels.push_back(levels);
        }
    }
    invalid_levels.push_back({20, 20, 20, 20});  // overflowing tensor cardinality
    for (const auto& levels : invalid_levels) {
        auto table = builder->build(levels);
        ASSERT_FALSE(table.has_value());
        EXPECT_EQ(table.error().code, PriceTableErrorCode::InvalidConfig);
        auto convenience = build_chebyshev_segmented_manual(config, domain, levels);
        ASSERT_FALSE(convenience.has_value());
        EXPECT_EQ(convenience.error().code, PriceTableErrorCode::InvalidConfig);
    }
}

// #486: an explicit manual level request is a constraint, even when defaults
// use more nodes. Requested-accuracy acceptance is a separate builder policy.
TEST(ChebyshevPDECacheTest, ManualNumericalSupportContainsRequestedLowEndpoints) {
    const std::array domains{
        IVGrid{.moneyness = {std::log(.001), std::log(.005), std::log(.02)},
            .vol = {.15, .25}, .rate = {-.10, -.06}},
        IVGrid{.moneyness = {-.05, .0, .05},
            .vol = {.005, .15}, .rate = {-.10, -.06}}};
    for (size_t i = 0; i < domains.size(); ++i) {
        SCOPED_TRACE(i);
        SegmentedAdaptiveConfig config{.spot = 100.0, .option_type = OptionType::PUT,
            .dividend_yield = 0.0, .discrete_dividends = {{.01, .1}}, .maturity = .025,
            .kref_config = {.K_refs = {100.0}}, .strike_bounds = StrikeBounds{100.0, 100.0}};
        auto builder = ChebyshevSegmentedBuilder::create(config, domains[i]);
        ASSERT_TRUE(builder);
        auto table = builder->build({3, 2, 1, 1});
        ASSERT_TRUE(table) << table.error();
        EXPECT_EQ(table->m_min(), domains[i].moneyness.front());
        EXPECT_EQ(table->sigma_min(), domains[i].vol.front());
        EXPECT_EQ(table->rate_min(), domains[i].rate.front());
        for (const auto& segment : table->inner().pieces().front().pieces()) {
            const auto& stored = segment.interpolant().domain();
            EXPECT_LE(stored.lo[0], domains[i].moneyness.front());
            EXPECT_GE(stored.hi[0], domains[i].moneyness.back());
            EXPECT_GT(stored.lo[2], 0.0);
            EXPECT_LE(stored.lo[2], domains[i].vol.front());
            EXPECT_GE(stored.hi[2], domains[i].vol.back());
            EXPECT_LE(stored.lo[3], domains[i].rate.front());
            EXPECT_GE(stored.hi[3], domains[i].rate.back());
            EXPECT_EQ(segment.interpolant().num_pts(), (std::array<size_t, 4>{9, 5, 3, 3}));
        }
    }
}

TEST(ChebyshevPDECacheTest, ExplicitManualLevelsRemainConstraints) {
    SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = OptionType::PUT,
        .discrete_dividends = {{0.01, 1.0}}, .maturity = 0.025,
        .kref_config = {.K_refs = {100.0}},
        .strike_bounds = StrikeBounds{100.0, 100.0}};
    IVGrid domain{.moneyness = {std::log(0.9), 0.0, std::log(1.1)},
        .vol = {0.15, 0.25}, .rate = {0.03, 0.05}};
    auto builder = ChebyshevSegmentedBuilder::create(config, domain);
    ASSERT_TRUE(builder.has_value());
    auto table = builder->build({3, 2, 1, 1});
    ASSERT_TRUE(table.has_value());
    const auto& segments = table->inner().pieces().front();
    ASSERT_EQ(segments.num_pieces(), 2u);
    for (const auto& leaf : segments.pieces()) {
        EXPECT_EQ(leaf.interpolant().num_pts(), (std::array<size_t, 4>{9, 5, 3, 3}));
    }
}

TEST(ChebyshevPDECacheTest, RejectsEventInsideUnsplittableLeaf) {
    SegmentedAdaptiveConfig config{
        .spot = 100.0, .option_type = OptionType::PUT,
        .discrete_dividends = {{0.9995, 1.0}}, .maturity = 1.0,
        .kref_config = {.K_refs = {100.0}},
        .strike_bounds = StrikeBounds{100.0, 100.0}};
    IVGrid domain{.moneyness = {-0.2, 0.0, 0.2},
        .vol = {0.1, 0.2}, .rate = {0.03, 0.05}};
    // Backward event .0005 is too close to the construction lower bound 0
    // for this topology's inset. Dropping the split would cross the jump.
    auto builder = ChebyshevSegmentedBuilder::create(config, domain);
    ASSERT_FALSE(builder.has_value());
    EXPECT_EQ(builder.error().code, PriceTableErrorCode::InvalidConfig);
}

// Regression #485: a sample without a real segment owner cannot be relabelled
// as a segment-zero row, changing the apparent CGL cardinality of both leaves.
TEST(ChebyshevPDECacheTest, RejectsUnmatchedSampleInsteadOfChangingItsOwner) {
    ChebyshevPDECache cache;
    std::vector<double> m = {-0.2, 0.2}, tau = {0.1, 0.4, 0.5, 0.6, 1.0};
    std::vector<double> sigma = {0.1, 0.2}, rate = {0.03, 0.05};
    for (double s : sigma) for (double r : rate) {
        for (size_t j = 0; j < tau.size(); ++j) {
            std::vector<double> x = {-0.3, 0.0, 0.3}, values(3, tau[j]);
            cache.store_slice(s, r, j, x, values);
        }
    }
    auto leaves = detail::build_segment_leaves(cache, 100.0,
        {0.1, 0.4, 0.6, 1.0}, {false, true, false}, false, m, tau, sigma, rate);
    ASSERT_FALSE(leaves.has_value());
    EXPECT_EQ(leaves.error().code, PriceTableErrorCode::ExtractionFailed);
}

TEST(ChebyshevPDECacheTest, GeneratedSegmentNodesKeepTheirCardinality) {
    ChebyshevPDECache cache;
    const std::vector<double> bounds = {0.01, 0.1495, 0.1505, 0.25};
    std::vector<double> tau = cc_level_nodes(3, bounds[0], bounds[1]);
    auto later = cc_level_nodes(3, bounds[2], bounds[3]);
    tau.insert(tau.end(), later.begin(), later.end());
    const std::vector<double> m = {-0.2, 0.2}, sigma = {0.1, 0.2}, rate = {0.03, 0.05};
    for (double s : sigma) for (double r : rate) {
        for (size_t j = 0; j < tau.size(); ++j) {
            std::vector<double> x = {-0.3, 0.0, 0.3}, values(3, tau[j]);
            cache.store_slice(s, r, j, x, values);
        }
    }
    auto leaves = detail::build_segment_leaves(cache, 100.0,
        bounds, {false, true, false}, false, m, tau, sigma, rate);
    ASSERT_TRUE(leaves.has_value());
    ASSERT_EQ(leaves->size(), 2u);
    for (const auto& leaf : *leaves) EXPECT_EQ(leaf.interpolant().num_pts()[1], 9u);
}

TEST(ChebyshevPDECacheTest, SegmentedPayoffRowUsesAnalyticalIntrinsicValues) {
    const auto m = mango::cc_level_nodes(2, -.00005, .01);
    const auto tau = mango::cc_level_nodes(2, 0.0, 1.0);
    const auto sigma = mango::cc_level_nodes(1, .1, .3);
    const auto rate = mango::cc_level_nodes(1, .03, .05);
    for (auto type : {mango::OptionType::PUT, mango::OptionType::CALL}) {
        auto pieces = mango::build_chebyshev_segmented_pieces(
            100.0, type, 0.0, {}, {0.0, 1.0}, {false}, m, tau, sigma, rate);
        ASSERT_TRUE(pieces.has_value());
        for (double x : m) {
            const double spot = 100.0 * std::exp(x);
            const double expected = mango::intrinsic_value(spot, 100.0, type);
            const double got = 100.0 * pieces->leaves.front().price(
                spot, 100.0, 0.0, .2, .04);
            EXPECT_NEAR(got, expected, 1e-10) << "x=" << x;
        }
        EXPECT_GT(pieces->leaves.front().price(100.0, 100.0, 1.0, .2, .04), 0.0);
    }
}
