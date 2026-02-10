// SPDX-License-Identifier: MIT
// Diagnostic: trace vanilla IV interpolation error through each layer
#include <cstdio>
#include <cmath>

#include "mango/option/american_option.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/adaptive_grid_builder.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/math/cubic_spline_solver.hpp"
#include "mango/option/table/standard_surface.hpp"
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/eep/eep_decomposer.hpp"

using namespace mango;

// Single test point: ATM, 1Y, 20% vol, 5% rate
static constexpr double kSpot = 100.0;
static constexpr double kStrike = 100.0;
static constexpr double kTau = 1.0;
static constexpr double kSigma = 0.20;
static constexpr double kRate = 0.05;
static constexpr double kDivYield = 0.02;

int main() {
    std::printf("=== Vanilla IV Interpolation Diagnostic ===\n\n");

    // Layer 1: FDM reference price
    std::printf("--- Layer 1: FDM Reference Price ---\n");
    PricingParams fdm_params;
    fdm_params.spot = kSpot;
    fdm_params.strike = kStrike;
    fdm_params.maturity = kTau;
    fdm_params.rate = kRate;
    fdm_params.dividend_yield = kDivYield;
    fdm_params.option_type = OptionType::PUT;
    fdm_params.volatility = kSigma;

    auto fdm_result = solve_american_option(fdm_params);
    if (!fdm_result.has_value()) {
        std::fprintf(stderr, "FDM solve failed\n");
        return 1;
    }
    double fdm_price = fdm_result->value_at(kSpot);
    std::printf("  FDM American price: %.6f\n", fdm_price);

    // Layer 2: European price (for EEP decomposition)
    std::printf("\n--- Layer 2: European Price ---\n");
    auto eu_result = EuropeanOptionSolver(
        OptionSpec{.spot = kSpot, .strike = kStrike, .maturity = kTau,
                   .rate = kRate, .dividend_yield = kDivYield,
                   .option_type = OptionType::PUT}, kSigma).solve();
    if (!eu_result.has_value()) {
        std::fprintf(stderr, "European solve failed\n");
        return 1;
    }
    double eu_price = eu_result->value();
    double eep = fdm_price - eu_price;
    std::printf("  European price: %.6f\n", eu_price);
    std::printf("  EEP = Am - Eu: %.6f\n", eep);

    // Layer 3: Build price table with PriceTableBuilderND directly
    std::printf("\n--- Layer 3: PriceTableBuilderND Surface ---\n");
    // Need at least 4 points per axis for cubic B-spline
    std::vector<double> moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};
    std::vector<double> maturities = {0.25, 0.5, 1.0, 1.5, 2.0};
    std::vector<double> vols = {0.10, 0.15, 0.20, 0.25, 0.30};
    std::vector<double> rates = {0.02, 0.04, 0.05, 0.06, 0.08};

    auto setup = PriceTableBuilder::from_vectors(
        moneyness, maturities, vols, rates,
        kSpot, GridAccuracyParams{}, OptionType::PUT, kDivYield);
    if (!setup.has_value()) {
        std::fprintf(stderr, "PriceTableBuilderND setup failed\n");
        return 1;
    }

    auto& [builder, axes] = *setup;
    EEPDecomposer decomposer{OptionType::PUT, kSpot, kDivYield};
    auto table_result = builder.build(axes, SurfaceContent::EarlyExercisePremium,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            decomposer.decompose(tensor, a);
        });
    if (!table_result.has_value()) {
        std::fprintf(stderr, "PriceTableBuilderND build failed: error code %d\n",
                     static_cast<int>(table_result.error().code));
        return 1;
    }

    auto surface = table_result->surface;
    auto& meta = surface->metadata();
    std::printf("  Surface content: %s\n",
                meta.content == SurfaceContent::EarlyExercisePremium ? "EEP" : "NormalizedPrice");
    std::printf("  K_ref: %.2f\n", meta.K_ref);

    // Query the raw surface at our test point
    double m = kSpot / kStrike;  // = 1.0 for ATM
    double raw_value = surface->value({m, kTau, kSigma, kRate});
    std::printf("  Raw surface value at (m=%.2f, tau=%.2f, sigma=%.2f, rate=%.2f): %.6f\n",
                m, kTau, kSigma, kRate, raw_value);

    // Layer 4: StandardSurface reconstruction
    std::printf("\n--- Layer 4: StandardSurface Reconstruction ---\n");
    auto wrapper = make_standard_surface(surface, OptionType::PUT);
    if (!wrapper.has_value()) {
        std::fprintf(stderr, "make_standard_surface failed\n");
        return 1;
    }

    double wrapper_price = wrapper->price(kSpot, kStrike, kTau, kSigma, kRate);
    std::printf("  StandardSurface::price(): %.6f\n", wrapper_price);
    std::printf("  Error vs FDM: %.6f (%.2f bps in price)\n",
                std::abs(wrapper_price - fdm_price),
                std::abs(wrapper_price - fdm_price) * 10000 / fdm_price);

    // Compute what the reconstruction formula gives
    double reconstructed = raw_value * (kStrike / meta.K_ref) + eu_price;
    std::printf("  Manual reconstruction (EEP * K/Kref + Eu): %.6f\n", reconstructed);

    // Layer 5: Check if interpolation is the issue
    std::printf("\n--- Layer 5: Grid Point Check ---\n");
    std::printf("  Is (m=1.0, tau=1.0, sigma=0.20, rate=0.05) on grid?\n");
    std::printf("  Moneyness grid: ");
    for (double v : moneyness) std::printf("%.2f ", v);
    std::printf("\n  Maturity grid: ");
    for (double v : maturities) std::printf("%.2f ", v);
    std::printf("\n  Vol grid: ");
    for (double v : vols) std::printf("%.2f ", v);
    std::printf("\n  Rate grid: ");
    for (double v : rates) std::printf("%.2f ", v);
    std::printf("\n");

    bool m_on_grid = false, tau_on_grid = false, sigma_on_grid = false, rate_on_grid = false;
    for (double v : moneyness) if (std::abs(v - m) < 1e-6) m_on_grid = true;
    for (double v : maturities) if (std::abs(v - kTau) < 1e-6) tau_on_grid = true;
    for (double v : vols) if (std::abs(v - kSigma) < 1e-6) sigma_on_grid = true;
    for (double v : rates) if (std::abs(v - kRate) < 1e-6) rate_on_grid = true;

    std::printf("  m=1.0 on grid: %s\n", m_on_grid ? "YES" : "NO");
    std::printf("  tau=1.0 on grid: %s\n", tau_on_grid ? "YES" : "NO");
    std::printf("  sigma=0.20 on grid: %s\n", sigma_on_grid ? "YES" : "NO");
    std::printf("  rate=0.05 on grid: %s\n", rate_on_grid ? "YES" : "NO");

    // Layer 6: Check what AdaptiveGridBuilder produces
    std::printf("\n--- Layer 6: AdaptiveGridBuilder Surface ---\n");
    OptionGrid chain;
    chain.spot = kSpot;
    chain.dividend_yield = kDivYield;
    chain.strikes = {kSpot / 0.8, kSpot / 0.9, kSpot / 1.0, kSpot / 1.1, kSpot / 1.2};
    chain.maturities = {0.5, 1.0, 1.5, 2.0};
    chain.implied_vols = {0.10, 0.20, 0.30};
    chain.rates = {0.03, 0.05, 0.07};

    auto grid_spec = GridSpec<double>::sinh_spaced(-3.0, 3.0, 101, 2.0);
    AdaptiveGridParams params{.target_iv_error = 2e-5};
    AdaptiveGridBuilder adaptive_builder(params);

    auto adaptive_result = adaptive_builder.build(chain, *grid_spec, 500, OptionType::PUT);
    if (!adaptive_result.has_value()) {
        std::fprintf(stderr, "AdaptiveGridBuilder failed\n");
        return 1;
    }

    auto adaptive_surface = adaptive_result->surface;
    auto& adaptive_meta = adaptive_surface->metadata();
    std::printf("  Surface content: %s\n",
                adaptive_meta.content == SurfaceContent::EarlyExercisePremium ? "EEP" : "NormalizedPrice");
    std::printf("  K_ref: %.2f\n", adaptive_meta.K_ref);

    double adaptive_raw = adaptive_surface->value({m, kTau, kSigma, kRate});
    std::printf("  Raw surface value: %.6f\n", adaptive_raw);

    auto adaptive_wrapper = make_standard_surface(adaptive_surface, OptionType::PUT);
    if (!adaptive_wrapper.has_value()) {
        std::fprintf(stderr, "make_standard_surface failed for adaptive\n");
        return 1;
    }

    double adaptive_price = adaptive_wrapper->price(kSpot, kStrike, kTau, kSigma, kRate);
    std::printf("  StandardSurface::price(): %.6f\n", adaptive_price);
    std::printf("  Error vs FDM: %.6f (%.2f bps in price)\n",
                std::abs(adaptive_price - fdm_price),
                std::abs(adaptive_price - fdm_price) * 10000 / fdm_price);

    // Summary
    std::printf("\n=== Summary ===\n");
    std::printf("  FDM reference:       %.6f\n", fdm_price);
    std::printf("  PriceTableBuilderND:   %.6f (error: %.2f bps)\n",
                wrapper_price, std::abs(wrapper_price - fdm_price) * 10000 / fdm_price);
    std::printf("  AdaptiveGridBuilder: %.6f (error: %.2f bps)\n",
                adaptive_price, std::abs(adaptive_price - fdm_price) * 10000 / fdm_price);

    // Layer 7: Check what GridAccuracyParams the builder uses
    std::printf("\n--- Layer 7: Grid Accuracy Investigation ---\n");
    std::printf("  Expected EEP (Am - Eu): %.6f\n", eep);
    std::printf("  PriceTableBuilderND raw EEP: %.6f\n", raw_value);
    std::printf("  Difference: %.6f\n", std::abs(eep - raw_value));
    std::printf("  This IS a grid point, so error is from PDE solver accuracy\n");

    // Compare with default vs high-accuracy grid
    std::printf("\n--- Layer 8: High-Accuracy Grid Test ---\n");
    GridAccuracyParams high_acc{.tol = 1e-6};  // High accuracy mode
    auto setup_hi = PriceTableBuilder::from_vectors(
        moneyness, maturities, vols, rates,
        kSpot, high_acc, OptionType::PUT, kDivYield);
    if (setup_hi.has_value()) {
        auto& [builder_hi, axes_hi] = *setup_hi;
        EEPDecomposer decomposer_hi{OptionType::PUT, kSpot, kDivYield};
        auto result_hi = builder_hi.build(axes_hi, SurfaceContent::EarlyExercisePremium,
            [&](PriceTensor& tensor, const PriceTableAxes& a) {
                decomposer_hi.decompose(tensor, a);
            });
        if (result_hi.has_value()) {
            double raw_hi = result_hi->surface->value({m, kTau, kSigma, kRate});
            std::printf("  High-accuracy raw EEP (tol=1e-6): %.6f\n", raw_hi);
            std::printf("  Error vs expected: %.6f\n", std::abs(eep - raw_hi));

            auto wrapper_hi = make_standard_surface(result_hi->surface, OptionType::PUT);
            if (wrapper_hi.has_value()) {
                double price_hi = wrapper_hi->price(kSpot, kStrike, kTau, kSigma, kRate);
                std::printf("  High-accuracy price: %.6f\n", price_hi);
                std::printf("  Error vs FDM: %.6f (%.2f bps)\n",
                            std::abs(price_hi - fdm_price),
                            std::abs(price_hi - fdm_price) * 10000 / fdm_price);
            }
        }
    }

    // Layer 9: Direct comparison - what does PDE compute at this point?
    std::printf("\n--- Layer 9: Direct PDE EEP Check ---\n");
    // PriceTableBuilderND computes EEP as: am_price - eu_price
    // where am_price = K_ref * normalized_spline_value
    // Let's check what the raw PDE returns

    // First compute what the European component should be
    auto eu_at_grid = EuropeanOptionSolver(
        OptionSpec{.spot = kSpot, .strike = kSpot, .maturity = kTau,
                   .rate = kRate, .dividend_yield = kDivYield,
                   .option_type = OptionType::PUT}, kSigma).solve().value();
    std::printf("  EU at (S=K=100, tau=1, sigma=0.2, rate=0.05): %.6f\n", eu_at_grid.value());

    // Now the American at the same point
    PricingParams grid_params;
    grid_params.spot = kSpot;
    grid_params.strike = kSpot;  // K_ref = spot for moneyness=1
    grid_params.maturity = kTau;
    grid_params.rate = kRate;
    grid_params.dividend_yield = kDivYield;
    grid_params.option_type = OptionType::PUT;
    grid_params.volatility = kSigma;

    auto am_at_grid = solve_american_option(grid_params);
    if (am_at_grid.has_value()) {
        double am_val = am_at_grid->value_at(kSpot);
        std::printf("  AM at (S=K=100, tau=1, sigma=0.2, rate=0.05): %.6f\n", am_val);
        std::printf("  EEP = AM - EU: %.6f\n", am_val - eu_at_grid.value());
        std::printf("  Expected EEP from Layer 2: %.6f\n", eep);
        std::printf("  Surface stores: %.6f\n", raw_value);
    }

    // Layer 10: Check the raw spline interpolation vs value_at
    std::printf("\n--- Layer 10: Spline vs value_at Check ---\n");
    // What if there's something wrong with cubic spline interpolation?
    // Let's check the direct B-spline value vs the raw tensor value

    // Query off-grid to force interpolation
    double off_grid_m = 0.95;  // Between 0.9 and 1.0
    double off_grid_tau = 0.75;  // Between 0.5 and 1.0

    double on_grid_val = surface->value({m, kTau, kSigma, kRate});
    double off_grid_val = surface->value({off_grid_m, off_grid_tau, kSigma, kRate});

    std::printf("  On-grid value (m=1.0, tau=1.0): %.6f\n", on_grid_val);
    std::printf("  Off-grid value (m=0.95, tau=0.75): %.6f\n", off_grid_val);

    // Compare against direct FDM solve at off-grid point
    PricingParams off_params;
    off_params.spot = off_grid_m * kSpot;
    off_params.strike = kSpot;
    off_params.maturity = off_grid_tau;
    off_params.rate = kRate;
    off_params.dividend_yield = kDivYield;
    off_params.option_type = OptionType::PUT;
    off_params.volatility = kSigma;

    auto off_am = solve_american_option(off_params);
    auto off_eu = EuropeanOptionSolver(
        OptionSpec{.spot = off_grid_m * kSpot, .strike = kSpot, .maturity = off_grid_tau,
                   .rate = kRate, .dividend_yield = kDivYield,
                   .option_type = OptionType::PUT}, kSigma).solve();

    if (off_am.has_value() && off_eu.has_value()) {
        double off_eep = off_am->value_at(off_grid_m * kSpot) - off_eu->value();
        std::printf("  Direct FDM EEP at off-grid point: %.6f\n", off_eep);
        std::printf("  Surface EEP at off-grid point: %.6f\n", off_grid_val);
        std::printf("  Error: %.6f\n", std::abs(off_eep - off_grid_val));
    }

    // Layer 11: Direct batch solve check
    std::printf("\n--- Layer 11: Batch Solver Direct Check ---\n");
    // The batch solver uses normalized coordinates (S=K=K_ref)
    // So the normalized price at x=0 should equal the EEP + EU(S=K=K_ref)

    PricingParams normalized_params;
    normalized_params.spot = kSpot;  // = K_ref
    normalized_params.strike = kSpot;  // = K_ref
    normalized_params.maturity = 2.0;  // max maturity to get all snapshots
    normalized_params.rate = kRate;
    normalized_params.dividend_yield = kDivYield;
    normalized_params.option_type = OptionType::PUT;
    normalized_params.volatility = kSigma;

    // Solve with snapshots at our test maturities using batch solver
    std::vector<double> snapshot_times = {0.25, 0.50, 1.0, 1.5, 2.0};
    BatchAmericanOptionSolver batch_solver;
    batch_solver.set_snapshot_times(snapshot_times);

    auto batch_result = batch_solver.solve_batch({normalized_params}, true);
    if (!batch_result.results.empty() && batch_result.results[0].has_value()) {
        const auto& am_result = batch_result.results[0].value();
        std::printf("  Batch solve succeeded\n");
        std::printf("  Snapshot times: ");
        for (double t : am_result.snapshot_times()) {
            std::printf("%.2f ", t);
        }
        std::printf("\n");

        // Get the solution at tau=1.0 (index 2)
        auto snapshot = am_result.at_time(2);  // tau=1.0
        std::printf("  Snapshot at tau=1.0 has %zu points\n", snapshot.size());

        // Find x=0 (ATM) in the grid
        auto grid = am_result.grid();
        auto x_grid = grid->x();
        std::printf("  Grid x range: [%.4f, %.4f] with %zu points\n",
                    x_grid.front(), x_grid.back(), x_grid.size());

        // Find index closest to x=0
        size_t atm_idx = 0;
        double min_dist = std::abs(x_grid[0]);
        for (size_t i = 1; i < x_grid.size(); ++i) {
            if (std::abs(x_grid[i]) < min_dist) {
                min_dist = std::abs(x_grid[i]);
                atm_idx = i;
            }
        }
        std::printf("  ATM index: %zu, x=%.6f\n", atm_idx, x_grid[atm_idx]);
        std::printf("  Raw normalized price at ATM, tau=1.0: %.6f\n", snapshot[atm_idx]);
        std::printf("  Denormalized (x K_ref): %.6f\n", snapshot[atm_idx] * kSpot);

        // The value_at function should give the same thing
        double val_at = am_result.value_at(kSpot);
        std::printf("  value_at(spot=100): %.6f\n", val_at);

        // Check: value_at uses cubic spline on the FINAL (tau=0) solution
        // But snapshots are at intermediate maturities
        std::printf("\n  Note: value_at() interpolates on FINAL solution (tau=0)\n");
        std::printf("  Snapshots are stored at intermediate maturities\n");
    }

    // Layer 12: Direct spline evaluation test
    std::printf("\n--- Layer 12: Direct Spline Evaluation ---\n");
    if (!batch_result.results.empty() && batch_result.results[0].has_value()) {
        const auto& am_result = batch_result.results[0].value();
        auto grid = am_result.grid();
        auto x_grid = grid->x();
        auto snapshot = am_result.at_time(2);  // tau=1.0

        // Build cubic spline just like PriceTableBuilderND
        CubicSpline<double> spline;
        auto build_err = spline.build(x_grid, snapshot);
        if (build_err.has_value()) {
            std::printf("  Spline build failed!\n");
        } else {
            // Evaluate at x=0 (ATM)
            double spline_at_0 = spline.eval(0.0);
            std::printf("  Spline eval at x=0: %.6f\n", spline_at_0);
            std::printf("  Direct grid value at x≈0: %.6f\n", snapshot[71]);  // ATM index

            // Evaluate at several x values
            std::printf("  Spline eval at x=-0.1: %.6f\n", spline.eval(-0.1));
            std::printf("  Spline eval at x=0.1: %.6f\n", spline.eval(0.1));

            // What is the American price after denormalization?
            double am_from_spline = spline_at_0 * kSpot;
            std::printf("  American price from spline: %.6f\n", am_from_spline);
            std::printf("  Expected American price: %.6f\n", fdm_price);

            // Compute EEP
            double eep_from_spline = am_from_spline - eu_price;
            std::printf("  EEP from spline: %.6f\n", eep_from_spline);
            std::printf("  Surface stores: %.6f\n", raw_value);
        }
    }

    // Layer 13: Trace through PriceTableBuilderND batch solve
    std::printf("\n--- Layer 13: PriceTableBuilderND Batch Solve Trace ---\n");
    // Use the same parameters as PriceTableBuilderND would
    // For sigma=0.20, rate=0.05, it creates a batch entry with K_ref as strike

    PricingParams batch_params;
    batch_params.spot = kSpot;  // = K_ref
    batch_params.strike = kSpot;  // = K_ref  (normalized solve)
    batch_params.maturity = 2.0;  // max maturity
    batch_params.rate = kRate;
    batch_params.dividend_yield = kDivYield;
    batch_params.option_type = OptionType::PUT;
    batch_params.volatility = kSigma;

    // Create batch solver with snapshot times matching PriceTableBuilderND
    BatchAmericanOptionSolver batch_solver2;
    batch_solver2.set_snapshot_times(maturities);
    batch_solver2.set_grid_accuracy({.tol = 1e-2});  // Same as default

    auto batch_result2 = batch_solver2.solve_batch({batch_params}, true);
    if (!batch_result2.results.empty() && batch_result2.results[0].has_value()) {
        const auto& am_result2 = batch_result2.results[0].value();

        // Find tau=1.0 snapshot index
        auto times = am_result2.snapshot_times();
        size_t tau_idx = 2;  // Should be index 2 for tau=1.0

        std::printf("  Snapshot times: ");
        for (size_t i = 0; i < times.size(); ++i) {
            std::printf("%.2f%s ", times[i], i == tau_idx ? "*" : "");
        }
        std::printf("\n");

        auto snapshot2 = am_result2.at_time(tau_idx);
        auto grid2 = am_result2.grid();
        auto x_grid2 = grid2->x();

        std::printf("  Grid: [%.4f, %.4f] with %zu points\n",
                    x_grid2.front(), x_grid2.back(), x_grid2.size());

        // Build spline and evaluate at x=0
        CubicSpline<double> spline2;
        spline2.build(x_grid2, snapshot2);
        double norm_price_at_0 = spline2.eval(0.0);
        double am_price_at_0 = norm_price_at_0 * kSpot;

        std::printf("  Spline at x=0: %.6f\n", norm_price_at_0);
        std::printf("  American price: %.6f\n", am_price_at_0);

        // Compute European at S=K=100, tau=1.0
        double spot_for_eu = 1.0 * kSpot;  // m=1.0
        auto eu_at_1 = EuropeanOptionSolver(
            OptionSpec{.spot = spot_for_eu, .strike = kSpot, .maturity = 1.0,
                       .rate = kRate, .dividend_yield = kDivYield,
                       .option_type = OptionType::PUT}, kSigma).solve().value();

        double eep_raw = am_price_at_0 - eu_at_1.value();
        std::printf("  European at m=1, tau=1: %.6f\n", eu_at_1.value());
        std::printf("  EEP raw: %.6f\n", eep_raw);

        // Apply softplus
        constexpr double kSharpness = 100.0;
        double stored;
        if (kSharpness * eep_raw > 500.0) {
            stored = eep_raw;
        } else {
            stored = std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
        }
        std::printf("  After softplus: %.6f\n", stored);
        std::printf("  Surface stores: %.6f\n", raw_value);
    }

    // Layer 14: Check tensor vs surface at a grid point
    std::printf("\n--- Layer 14: Tensor vs Surface Direct Comparison ---\n");
    // Re-run the build step by step to extract the tensor value

    auto [builder2, axes2] = *PriceTableBuilder::from_vectors(
        moneyness, maturities, vols, rates,
        kSpot, GridAccuracyParams{}, OptionType::PUT, kDivYield);

    // Access internal state by building
    auto result2 = builder2.build(axes2);
    if (result2.has_value()) {
        auto& res = result2.value();
        auto surf = res.surface;

        // Query at the on-grid point
        double surf_val = surf->value({m, kTau, kSigma, kRate});
        std::printf("  Surface value at (1.0, 1.0, 0.20, 0.05): %.6f\n", surf_val);

        // Unfortunately we can't access the tensor directly from outside
        // But we CAN check the fitting stats
        std::printf("  Surface metadata content: %s\n",
                    surf->metadata().content == SurfaceContent::EarlyExercisePremium ? "EEP" : "NormalizedPrice");
        std::printf("  K_ref: %.2f\n", surf->metadata().K_ref);

        // Check if this differs from the direct batch solve
        std::printf("  Expected from batch solve: 0.324765\n");
        std::printf("  Difference: %.6f\n", std::abs(surf_val - 0.324765));
    }

    return 0;
}
