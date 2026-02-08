// SPDX-License-Identifier: MIT
#include "chebyshev_4d_incremental_builder.hpp"
#include "chebyshev_4d_eep_inner.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <random>

namespace mango {
namespace {

// Phase A verification gate 1: cached build matches fresh build.
TEST(Chebyshev4DIncrementalTest, CachedBuildMatchesFreshBuild) {
    // Build fresh (no cache) with 9 sigma nodes, 5 rate nodes
    // (CC level 3 = 2^3+1=9, CC level 2 = 2^2+1=5)
    Chebyshev4DEEPConfig fresh_cfg;
    fresh_cfg.num_x = 15;
    fresh_cfg.num_tau = 9;
    fresh_cfg.num_sigma = 9;   // CC level 3
    fresh_cfg.num_rate = 5;    // CC level 2
    fresh_cfg.use_tucker = false;
    fresh_cfg.dividend_yield = 0.0;

    auto fresh = build_chebyshev_4d_eep(fresh_cfg, 100.0, OptionType::PUT);

    // Build incrementally with same final levels.
    // Headroom refs must match the fresh builder's node counts so extended
    // domains are identical.
    IncrementalBuildConfig inc_cfg;
    inc_cfg.num_x = 15;
    inc_cfg.num_tau = 9;
    inc_cfg.sigma_level = 3;  // 9 nodes
    inc_cfg.rate_level = 2;   // 5 nodes
    inc_cfg.use_tucker = false;
    inc_cfg.dividend_yield = 0.0;
    inc_cfg.sigma_headroom_ref = 9;   // match fresh num_sigma
    inc_cfg.rate_headroom_ref = 5;    // match fresh num_rate

    PDESliceCache cache;
    auto inc = build_chebyshev_4d_eep_incremental(inc_cfg, cache, 100.0, OptionType::PUT);

    // Compare at 50 random probe points in the USER domain (not extended)
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> ux(fresh_cfg.x_min, fresh_cfg.x_max);
    std::uniform_real_distribution<double> ut(fresh_cfg.tau_min, fresh_cfg.tau_max);
    std::uniform_real_distribution<double> us(fresh_cfg.sigma_min, fresh_cfg.sigma_max);
    std::uniform_real_distribution<double> ur(fresh_cfg.rate_min, fresh_cfg.rate_max);

    double max_diff = 0.0;
    for (int i = 0; i < 50; ++i) {
        double x = ux(rng);
        double tau = ut(rng);
        double sigma = us(rng);
        double rate = ur(rng);

        double p_fresh = fresh.interp.eval({x, tau, sigma, rate});
        double p_inc = inc.interp.eval({x, tau, sigma, rate});
        max_diff = std::max(max_diff, std::abs(p_fresh - p_inc));
    }

    // Must match to ~1e-12 (accumulated spline interpolation noise).
    EXPECT_LT(max_diff, 1e-12)
        << "Cached build diverges from fresh build: max diff = " << max_diff;
}

// Phase A verification gate 2: incremental cost accounting.
TEST(Chebyshev4DIncrementalTest, IncrementalSolvesOnlyNewPairs) {
    IncrementalBuildConfig cfg;
    cfg.num_x = 10;
    cfg.num_tau = 5;
    cfg.sigma_level = 1;  // 3 nodes
    cfg.rate_level = 1;   // 3 nodes
    cfg.use_tucker = false;
    // Fixed headroom refs ensure extended domain stays constant across levels
    cfg.sigma_headroom_ref = 15;
    cfg.rate_headroom_ref = 9;

    // First build: empty cache -> should solve 3 x 3 = 9 PDEs
    PDESliceCache cache;
    auto r1 = build_chebyshev_4d_eep_incremental(cfg, cache, 100.0, OptionType::PUT);
    EXPECT_EQ(r1.new_pde_solves, 9u);
    EXPECT_EQ(cache.total_pde_solves(), 9u);

    // Refine sigma to level 2 (5 nodes). Rate stays at level 1 (3 nodes).
    // New sigma nodes: 2. New pairs: 2 x 3 = 6.
    cfg.sigma_level = 2;
    auto r2 = build_chebyshev_4d_eep_incremental(cfg, cache, 100.0, OptionType::PUT);
    EXPECT_EQ(r2.new_pde_solves, 6u);
    EXPECT_EQ(cache.total_pde_solves(), 15u);

    // Refine rate to level 2 (5 nodes). Sigma at level 2 (5 nodes).
    // New rate nodes: 2. New pairs: 5 x 2 = 10.
    cfg.rate_level = 2;
    auto r3 = build_chebyshev_4d_eep_incremental(cfg, cache, 100.0, OptionType::PUT);
    EXPECT_EQ(r3.new_pde_solves, 10u);
    EXPECT_EQ(cache.total_pde_solves(), 25u);
}

}  // namespace
}  // namespace mango
