// SPDX-License-Identifier: MIT
// Calibration of the reference oracle's convergence order and uncertainty
// floor, and a check of the declared High profile (spec D8, rev 6).
//
// This test IS the measurement: it prints the calibration table that
// `docs/MATHEMATICAL_FOUNDATIONS.md` records, and it asserts the two
// calibrated constants (`kReferenceConvergenceOrder`,
// `kReferenceUncertaintyFloor`) stay consistent with what the oracle
// actually does.  Every number here is an empirical outcome on the declared
// six-point set, never a bound on the oracle's error.
//
// Grids (rev 6): `G` is production's fine grid and `G½` production's coarse
// partner -- the pair `δ̂` is computed from -- and `2G`/`4G` are two nested
// refinements of `G` built by `refine_grid_config`.  Triple A
// `(G½, G, 2G)` is the production pair's own observed order and is what sets
// the constant; triple B `(G, 2G, 4G)` checks the order holds one level
// finer.  Rev 5 calibrated downward instead (`G, G½, G¼, G⅛`); those coarser
// grids are ones production never uses, and on the 30-day contract they were
// pre-asymptotic (measured p_obs < 0), so the order they reported said
// nothing about the pair that `δ̂` actually uses.
//
// Orientation of `p_obs`: for a triple (coarse, mid, fine) under exact
// halving, `V_mid − V_coarse = 2^p · (V_fine − V_mid)` in the asymptotic
// regime, so the observed order is `log2(d_coarse / d_fine)` with
// `d_coarse = V_mid − V_coarse` and `d_fine = V_fine − V_mid`.  The series
// here runs coarse to fine, which is the reverse of rev 5's listing, so the
// ratio is taken coarse-difference over fine-difference to keep `p_obs`
// positive for a convergent series.  The insufficient-signal and
// oscillatory tests are symmetric in the two differences and are unchanged.
//
// STATUS (2026-09-21, rev 6 measurement): one assertion FAILS, and the
// constants were deliberately left untuned -- D8's failure policy is that a
// failed calibration is a statement about the oracle family, not a licence
// to move a constant.  Moving to the production pair fixed what rev 5
// found: every p_obs is now positive, no triple is oscillatory, all six
// points carry both triples usable at the base sigma, and the triple-A
// minimum is 1.317 (`atm-1y-3div`, tau_iv = 5e-4, sigma-lo), so the shipped
// `kReferenceConvergenceOrder = 1.0` is comfortably under it.  What remains:
//
//   * Order stability |p_A - p_B| <= 0.5 fails at `otm-30d`, tau_iv = 1e-3,
//     all three sigma: p_A = 1.685 / 1.687 / 1.685 against p_B = 0.671 /
//     0.676 / 0.677.  It is not time-step quantization -- forcing the
//     tau_iv = 1e-3 grid to the tau_iv = 5e-4 step count (249, and 250)
//     leaves p_B at 0.676 and 0.675 -- and it is not a missing refinement:
//     an 8G level gives p over successive triples of 1.397 / 0.982 at
//     tau_iv = 5e-4 and 0.676 / 1.349 at tau_iv = 1e-3, i.e. the observed
//     order at this contract wanders in [0.68, 1.40] rather than settling.
//     Both families do agree on the answer: their 8G prices differ by
//     6e-9.  The reading is an oscillatory O(h) component from the free
//     boundary crossing grid nodes riding on the smooth term, which the
//     0.5 allowance is tighter than.
//
// The open decision is the allowance or the oracle family for the 30-day
// contract class, not the constant.
#include <gtest/gtest.h>

#include "mango/option/american_option.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/option/table/adaptive_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace mango;

namespace {

struct CalPoint {
    const char* name;
    const char* key;  // RecordProperty-safe form of `name`
    double S, K, tau, sigma, r;
    OptionType type;
    std::vector<Dividend> divs;
    double T;  // reference (fixed) expiry the schedule is anchored to
};

// Spec D8's declared calibration set.
const std::vector<CalPoint>& points() {
    static const std::vector<CalPoint> kPoints = {
        {"atm-1y-3div", "atm_1y_3div", 100, 100, 1.0, 0.20, 0.05,
         OptionType::PUT, {{0.25, 0.5}, {0.5, 0.5}, {0.75, 0.5}}, 1.0},
        {"500-trigger", "trigger_500", 100, 113.71897954276989,
         0.27191459370080351, 0.1396857726802572, 0.055127327935524113,
         OptionType::PUT, {{0.25, 0.5}, {0.5, 0.5}, {0.75, 0.5}}, 1.0},
        {"otm-30d", "otm_30d", 100, 92, 30.0 / 365, 0.25, 0.05,
         OptionType::PUT, {}, 30.0 / 365},
        {"deep-otm-7d", "deep_otm_7d", 100, 85, 7.0 / 365, 0.30, 0.05,
         OptionType::PUT, {}, 7.0 / 365},
        {"itm-2y", "itm_2y", 100, 120, 2.0, 0.20, 0.05, OptionType::PUT, {}, 2.0},
        {"atm-6m-call", "atm_6m_call", 100, 100, 0.5, 0.20, 0.05,
         OptionType::CALL, {}, 0.5},
    };
    return kPoints;
}

constexpr double kDividendYield = 0.02;

// The four grids of one series, coarse to fine.
enum GridLevel { kGHalf = 0, kG = 1, k2G = 2, k4G = 3, kNumGrids = 4 };
const char* kGridName[kNumGrids] = {"Ghalf", "G", "2G", "4G"};

enum class Triple { Insufficient, Oscillatory, Usable };

const char* label(Triple t) {
    switch (t) {
        case Triple::Insufficient: return "insufficient";
        case Triple::Oscillatory:  return "oscillatory";
        case Triple::Usable:       return "usable";
    }
    return "?";
}

// Spec D8 classification of one triple (coarse, mid, fine), with
// `d_coarse = V_mid - V_coarse`, `d_fine = V_fine - V_mid` and threshold
// theta = 2^-40 * K.
Triple classify(double d_coarse, double d_fine, double theta, double* p) {
    if (std::abs(d_coarse) <= theta || std::abs(d_fine) <= theta) {
        return Triple::Insufficient;
    }
    if (d_coarse * d_fine < 0) return Triple::Oscillatory;
    *p = std::log(d_coarse / d_fine) / std::log(2.0);
    return Triple::Usable;
}

// The shipped estimate, floor included (adaptive_metrics.cpp).
double delta_hat(double fine, double coarse, double strike) {
    const double two_grid = kRichardsonSafetyFactor * std::abs(fine - coarse)
                          / (std::pow(2.0, kReferenceConvergenceOrder) - 1.0);
    return std::max(two_grid, kReferenceUncertaintyFloor * strike);
}

std::string num(double v, int prec = 12) {
    std::ostringstream os;
    os << std::setprecision(prec) << v;
    return os.str();
}

std::string tau_key(double tau_iv) { return tau_iv == 5e-4 ? "t5em4" : "t1em3"; }

// One stencil member's price series over the four grids.
struct Series {
    double v[kNumGrids] = {};
    Triple ca = Triple::Insufficient, cb = Triple::Insufficient;
    double pa = std::numeric_limits<double>::quiet_NaN();
    double pb = std::numeric_limits<double>::quiet_NaN();
};

// The calibration table, one CSV-ish line per measurement, prefixed so it
// can be sieved out of the test log with `grep '^CAL|'`.
void row(const std::string& s) { std::cout << "CAL| " << s << "\n"; }

}  // namespace

// Spec D8.  Complete production stencils on `G½, G, 2G, 4G`: for each of the
// six points and each tau_iv, all three stencil sigma on the grid family
// selected at sigma0 + tau_iv, so the endpoint uncertainties that control
// admission are the ones calibrated.
TEST(ReferenceOracleCalibration, OrderIsUsableStableAndAboveConstant) {
    const auto acc = make_grid_accuracy(kReferenceAccuracy);
    const auto ultra_acc = make_grid_accuracy(GridAccuracyProfile::Ultra);

    double min_usable_a = std::numeric_limits<double>::infinity();
    double max_usable_a = -std::numeric_limits<double>::infinity();
    double min_usable_any = std::numeric_limits<double>::infinity();
    double max_rel_profile_gap = 0.0;  // max |V_High - V_Ultra| / K
    size_t points_with_both_usable = 0;
    size_t usable_count = 0, oscillatory_count = 0, insufficient_count = 0;

    row("point,tau_iv,sigma,grid,n_points,n_time,value");

    for (const auto& pt : points()) {
        ReferenceOracle oracle{.dividend_yield = kDividendYield,
                               .option_type = pt.type,
                               .discrete_dividends = pt.divs,
                               .reference_maturity = pt.T,
                               .accuracy = acc};
        const double theta = std::ldexp(1.0, -40) * pt.K;

        // The production stencil at tau_iv = 5e-4, base sigma: its G and G-half
        // prices are the fine/coarse pair the shipped code uses, so delta_hat
        // below is the deployed estimate, not a re-derivation.
        std::optional<Series> production;
        std::optional<PDEGridConfig> production_g, production_2g;

        for (double tau_iv : {5e-4, 1e-3}) {
            const auto widest =
                oracle.contract(pt.S, pt.K, pt.tau, pt.sigma + tau_iv, pt.r);
            auto fam = make_reference_grid_family(widest, acc, 1);
            ASSERT_TRUE(fam.has_value()) << pt.name;
            ASSERT_EQ(fam->levels.size(), 2u) << pt.name;
            // Recorded, not asserted: D8's assertion set is the list below,
            // so a stray failure here would not muddy the calibration
            // signal.  It is worth watching all the same: `itm-2y` at High
            // rounds its fine count down (the estimate of 3495 cannot round
            // up to 3505 under the strict 3500 cap, so the family takes
            // 3489), which is what D1's rounded_down note now records.
            RecordProperty(std::string(pt.key) + "_" + tau_key(tau_iv) + "_rounded_down",
                           fam->rounded_down ? "true" : "false");
            if (fam->rounded_down) {
                row(std::string(pt.name) + "," + num(tau_iv, 3)
                    + ",rounded_down,true,n=" + std::to_string(fam->point_counts[0]));
            }

            // G-half, G, then two nested refinements of G above the cap.
            PDEGridConfig grids[kNumGrids];
            grids[kGHalf] = fam->levels[1];
            grids[kG] = fam->levels[0];
            auto g2 = refine_grid_config(fam->levels[0], 2);
            ASSERT_TRUE(g2.has_value()) << pt.name;
            auto g4 = refine_grid_config(fam->levels[0], 4);
            ASSERT_TRUE(g4.has_value()) << pt.name;
            grids[k2G] = std::move(*g2);
            grids[k4G] = std::move(*g4);

            const double sigmas[3] = {pt.sigma - tau_iv, pt.sigma, pt.sigma + tau_iv};
            for (size_t si = 0; si < 3; ++si) {
                const double s = sigmas[si];
                const char* slabel = (si == 0) ? "lo" : (si == 1) ? "base" : "hi";
                Series ser;
                for (size_t k = 0; k < kNumGrids; ++k) {
                    auto r = oracle.solve(
                        oracle.contract(pt.S, pt.K, pt.tau, s, pt.r), grids[k]);
                    ASSERT_TRUE(r.has_value())
                        << pt.name << " tau_iv=" << tau_iv << " sigma=" << s
                        << " grid=" << kGridName[k];
                    ser.v[k] = *r;
                    row(std::string(pt.name) + "," + num(tau_iv, 3) + "," + slabel
                        + "," + kGridName[k] + ","
                        + std::to_string(grids[k].grid_spec.n_points()) + ","
                        + std::to_string(grids[k].n_time) + "," + num(ser.v[k], 17));
                    RecordProperty(std::string(pt.key) + "_" + tau_key(tau_iv) + "_"
                                       + slabel + "_" + kGridName[k],
                                   num(ser.v[k], 17));
                }

                // Triple A: the production pair plus one refinement.
                ser.ca = classify(ser.v[kG] - ser.v[kGHalf],
                                  ser.v[k2G] - ser.v[kG], theta, &ser.pa);
                // Triple B: one level finer throughout.
                ser.cb = classify(ser.v[k2G] - ser.v[kG],
                                  ser.v[k4G] - ser.v[k2G], theta, &ser.pb);

                const std::string tag =
                    std::string(pt.key) + "_" + tau_key(tau_iv) + "_" + slabel;
                RecordProperty(tag + "_classA", label(ser.ca));
                RecordProperty(tag + "_classB", label(ser.cb));
                RecordProperty(tag + "_pA", num(ser.pa, 6));
                RecordProperty(tag + "_pB", num(ser.pb, 6));
                row(std::string(pt.name) + "," + num(tau_iv, 3) + "," + slabel
                    + ",classify," + label(ser.ca) + "," + label(ser.cb) + ",pA="
                    + num(ser.pa, 6) + ",pB=" + num(ser.pb, 6));

                for (Triple c : {ser.ca, ser.cb}) {
                    if (c == Triple::Usable) ++usable_count;
                    else if (c == Triple::Oscillatory) ++oscillatory_count;
                    else ++insufficient_count;
                }

                EXPECT_NE(ser.ca, Triple::Oscillatory)
                    << pt.name << " tau_iv=" << tau_iv << " sigma=" << slabel
                    << " triple A (Ghalf, G, 2G)";
                EXPECT_NE(ser.cb, Triple::Oscillatory)
                    << pt.name << " tau_iv=" << tau_iv << " sigma=" << slabel
                    << " triple B (G, 2G, 4G)";
                if (ser.ca == Triple::Usable) {
                    EXPECT_TRUE(std::isfinite(ser.pa) && ser.pa > 0.0)
                        << pt.name << " pA=" << ser.pa
                        << " (a non-positive usable order on triple A means the"
                           " profile's grid is pre-asymptotic for this contract)";
                    min_usable_a = std::min(min_usable_a, ser.pa);
                    max_usable_a = std::max(max_usable_a, ser.pa);
                    min_usable_any = std::min(min_usable_any, ser.pa);
                }
                if (ser.cb == Triple::Usable) {
                    EXPECT_TRUE(std::isfinite(ser.pb) && ser.pb > 0.0)
                        << pt.name << " pB=" << ser.pb;
                    min_usable_any = std::min(min_usable_any, ser.pb);
                }
                if (ser.ca == Triple::Usable && ser.cb == Triple::Usable) {
                    EXPECT_NEAR(ser.pa, ser.pb, 0.5)
                        << pt.name << " tau_iv=" << tau_iv << " sigma=" << slabel
                        << " order not stable";
                }

                if (si == 1 && tau_iv == 5e-4) {
                    production = ser;
                    production_g = grids[kG];
                    production_2g = grids[k2G];
                    if (ser.ca == Triple::Usable && ser.cb == Triple::Usable) {
                        ++points_with_both_usable;
                    }
                }
            }
        }

        ASSERT_TRUE(production.has_value()) << pt.name;
        const double v_high = production->v[kG];
        const double v_half = production->v[kGHalf];
        const double delta = delta_hat(v_high, v_half, pt.K);

        // Profile adequacy: High against Ultra, inside the shipped estimate.
        ReferenceOracle ultra = oracle;
        ultra.accuracy = ultra_acc;
        const auto ultra_contract = ultra.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r);
        auto ultra_fam = make_reference_grid_family(ultra_contract, ultra_acc, 1);
        ASSERT_TRUE(ultra_fam.has_value()) << pt.name;
        auto v_ultra = ultra.solve(ultra_contract, ultra_fam->levels[0]);
        ASSERT_TRUE(v_ultra.has_value()) << pt.name;
        const double profile_gap = std::abs(v_high - *v_ultra);
        max_rel_profile_gap = std::max(max_rel_profile_gap, profile_gap / pt.K);

        // Effective sensitivity (a): x-domain widened by one sigma*sqrt(T).
        GridAccuracyParams wide = acc;
        wide.n_sigma += 1.0;
        ReferenceOracle wide_oracle = oracle;
        wide_oracle.accuracy = wide;
        const auto wide_widest =
            wide_oracle.contract(pt.S, pt.K, pt.tau, pt.sigma + 5e-4, pt.r);
        auto wide_fam = make_reference_grid_family(wide_widest, wide, 1);
        ASSERT_TRUE(wide_fam.has_value()) << pt.name;
        auto v_wide = wide_oracle.solve(
            wide_oracle.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r), wide_fam->levels[0]);
        ASSERT_TRUE(v_wide.has_value()) << pt.name;
        const double domain_shift = std::abs(*v_wide - v_high);

        // Effective sensitivity (b): spatial-only and temporal-only
        // refinement of G, to attribute the G -> 2G shift between the axes.
        PDEGridConfig spatial_only{.grid_spec = production_2g->grid_spec,
                                   .n_time = production_g->n_time,
                                   .mandatory_times = {}};
        PDEGridConfig temporal_only{.grid_spec = production_g->grid_spec,
                                    .n_time = production_2g->n_time,
                                    .mandatory_times = {}};
        const auto base_contract = oracle.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r);
        auto v_spatial = oracle.solve(base_contract, spatial_only);
        auto v_temporal = oracle.solve(base_contract, temporal_only);
        ASSERT_TRUE(v_spatial.has_value()) << pt.name;
        ASSERT_TRUE(v_temporal.has_value()) << pt.name;

        const std::string tag(pt.key);
        RecordProperty(tag + "_V_G", num(v_high, 17));
        RecordProperty(tag + "_V_Ghalf", num(v_half, 17));
        RecordProperty(tag + "_V_2G", num(production->v[k2G], 17));
        RecordProperty(tag + "_V_4G", num(production->v[k4G], 17));
        RecordProperty(tag + "_V_ultra", num(*v_ultra, 17));
        RecordProperty(tag + "_delta_hat", num(delta, 12));
        RecordProperty(tag + "_profile_gap", num(profile_gap, 12));
        RecordProperty(tag + "_profile_gap_rel_K", num(profile_gap / pt.K, 12));
        RecordProperty(tag + "_domain_shift", num(domain_shift, 12));
        RecordProperty(tag + "_spatial_only_shift", num(std::abs(*v_spatial - v_high), 12));
        RecordProperty(tag + "_temporal_only_shift", num(std::abs(*v_temporal - v_high), 12));
        RecordProperty(tag + "_refined_shift", num(std::abs(production->v[k2G] - v_high), 12));

        row(std::string(pt.name) + ",profile,V_G=" + num(v_high, 17)
            + ",V_Ghalf=" + num(v_half, 17) + ",V_ultra=" + num(*v_ultra, 17)
            + ",delta_hat=" + num(delta, 12) + ",|V_High-V_Ultra|=" + num(profile_gap, 12)
            + ",rel_K=" + num(profile_gap / pt.K, 12));
        row(std::string(pt.name) + ",sensitivity,domain_shift=" + num(domain_shift, 12)
            + ",spatial_only=" + num(std::abs(*v_spatial - v_high), 12)
            + ",temporal_only=" + num(std::abs(*v_temporal - v_high), 12)
            + ",both(2G)=" + num(std::abs(production->v[k2G] - v_high), 12));

        EXPECT_LE(profile_gap, delta) << pt.name << " High vs Ultra outside the estimate";
        EXPECT_LE(domain_shift, delta) << pt.name << " domain sensitivity";
    }

    RecordProperty("min_usable_order_tripleA", num(min_usable_a, 6));
    RecordProperty("max_usable_order_tripleA", num(max_usable_a, 6));
    RecordProperty("min_usable_order_any", num(min_usable_any, 6));
    RecordProperty("max_profile_gap_rel_K", num(max_rel_profile_gap, 12));
    RecordProperty("points_with_both_usable_at_base_sigma",
                   std::to_string(points_with_both_usable));
    RecordProperty("usable_triples", std::to_string(usable_count));
    RecordProperty("oscillatory_triples", std::to_string(oscillatory_count));
    RecordProperty("insufficient_triples", std::to_string(insufficient_count));

    row("summary,min_usable_A=" + num(min_usable_a, 6)
        + ",max_usable_A=" + num(max_usable_a, 6)
        + ",min_usable_any=" + num(min_usable_any, 6)
        + ",max_profile_gap_rel_K=" + num(max_rel_profile_gap, 12)
        + ",points_with_both_usable=" + std::to_string(points_with_both_usable)
        + ",usable=" + std::to_string(usable_count)
        + ",oscillatory=" + std::to_string(oscillatory_count)
        + ",insufficient=" + std::to_string(insufficient_count));

    // Coverage rule: at least four of the six points carry both triples
    // usable at the base sigma of the tau_iv = 5e-4 stencil.
    EXPECT_GE(points_with_both_usable, 4u);
    // The constant is set from triple A: the production pair's own order.
    EXPECT_LE(kReferenceConvergenceOrder, min_usable_a);
    // The floor is a calibrated constant (spec D1, rev 5): it must cover the
    // oracle's own High-vs-Ultra discrepancy scale on this set.
    EXPECT_GE(kReferenceUncertaintyFloor, max_rel_profile_gap);
}

// Effective sensitivity (c): the solver's own complementarity diagnostic on
// the production fine grid, for two points.  Recorded, not asserted -- it
// says how far the projected solve's active set is from satisfying the full
// KKT conditions, which is one contributor to the bias a two-grid estimate
// cannot see.
TEST(ReferenceOracleCalibration, ComplementarityReportOnTheReferenceGrid) {
    const auto acc = make_grid_accuracy(kReferenceAccuracy);
    for (size_t i = 0; i < 2; ++i) {
        const auto& pt = points()[i];
        ReferenceOracle oracle{.dividend_yield = kDividendYield,
                               .option_type = pt.type,
                               .discrete_dividends = pt.divs,
                               .reference_maturity = pt.T,
                               .accuracy = acc};
        const auto widest = oracle.contract(pt.S, pt.K, pt.tau, pt.sigma + 5e-4, pt.r);
        auto fam = make_reference_grid_family(widest, acc, 1);
        ASSERT_TRUE(fam.has_value()) << pt.name;
        const auto base = oracle.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r);
        auto solver = AmericanOptionSolver::create(base, PDEGridSpec{fam->levels[0]});
        ASSERT_TRUE(solver.has_value()) << pt.name;
        auto result = solver->solve();
        ASSERT_TRUE(result.has_value()) << pt.name;
        const auto& kkt = solver->complementarity_report();
        const std::string tag(pt.key);
        RecordProperty(tag + "_kkt_violation_count", std::to_string(kkt.violation_count));
        RecordProperty(tag + "_kkt_max_violation", num(kkt.max_violation, 12));
        RecordProperty(tag + "_kkt_worst_kind", std::to_string(kkt.worst_kind));
        row(std::string(pt.name) + ",kkt,violation_count="
            + std::to_string(kkt.violation_count) + ",max_violation="
            + num(kkt.max_violation, 12) + ",worst_kind="
            + std::to_string(kkt.worst_kind));
    }
}
