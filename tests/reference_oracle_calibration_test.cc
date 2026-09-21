// SPDX-License-Identifier: MIT
// Calibration of the reference oracle's convergence order and uncertainty
// floor, and a check of the declared High profile (spec D8, rev 5).
//
// This test IS the measurement: it prints the calibration table that
// `docs/MATHEMATICAL_FOUNDATIONS.md` records, and it asserts the two
// calibrated constants (`kReferenceConvergenceOrder`,
// `kReferenceUncertaintyFloor`) stay consistent with what the oracle
// actually does.  Every number here is an empirical outcome on the declared
// six-point set, never a bound on the oracle's error.
//
// STATUS (2026-09-21, first measurement): this test FAILS on the shipped
// constants, and the constants were deliberately left untuned -- spec D8's
// failure policy is that a failed calibration is a statement about the
// oracle family, not a licence to move `kReferenceConvergenceOrder` down to
// whatever the family happens to produce.  What the run measured:
//
//   * `otm-30d` (30-day OTM put) is pre-asymptotic on the High profile's own
//     grid: at tau_iv = 5e-4 the finest triple gives p_obs = -0.19..-0.16,
//     because the level-0 -> level-1 shift is *larger* than the level-1 ->
//     level-2 shift.  Re-running that point with a 2x and 4x finer level 0
//     gives p_obs = 1.44 / 0.80 and 1.44 / 1.50, so the order is recoverable
//     -- with a finer family, not with more levels.  Extending the shipped
//     family to levels 4-5 (D8's first suggested remedy) walks the *coarse*
//     way and turns triples (2,3,4) and (3,4,5) oscillatory.
//   * Order stability |p1 - p2| <= 0.5 also fails at `500-trigger` (sigma-lo,
//     both tau_iv; 0.91 and 0.95) and `itm-2y` (all three sigma at
//     tau_iv = 1e-3; 0.73..0.78).
//   * Everything else passes: no oscillatory triple, 6/6 points carry two
//     usable triples at the base sigma, |V_High - V_Ultra| <= delta-hat and
//     the widened-domain shift <= delta-hat at every point, and
//     max |V_High - V_Ultra| / K = 6.34e-8 <= kReferenceUncertaintyFloor.
//
// The open decision is the oracle family (profile or domain rule), not the
// constants.
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

enum class Triple { Insufficient, Oscillatory, Usable };

const char* label(Triple t) {
    switch (t) {
        case Triple::Insufficient: return "insufficient";
        case Triple::Oscillatory:  return "oscillatory";
        case Triple::Usable:       return "usable";
    }
    return "?";
}

// Spec D8 classification of one consecutive triple (k, k+1, k+2):
// d_a = V_{k+2} - V_{k+1}, d_b = V_{k+1} - V_k, threshold theta = 2^-40 * K.
Triple classify(double d_a, double d_b, double theta, double* p) {
    if (std::abs(d_a) <= theta || std::abs(d_b) <= theta) return Triple::Insufficient;
    if (d_a * d_b < 0) return Triple::Oscillatory;
    *p = std::log(d_a / d_b) / std::log(2.0);
    return Triple::Usable;
}

// The shipped estimate, floor included (adaptive_metrics.cpp:315).
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

// One stencil member's price series over the four nested levels.
struct Series {
    double v[4] = {};
    Triple c1 = Triple::Insufficient, c2 = Triple::Insufficient;
    double p1 = std::numeric_limits<double>::quiet_NaN();
    double p2 = std::numeric_limits<double>::quiet_NaN();
};

// The calibration table, one CSV-ish line per measurement, prefixed so it
// can be sieved out of the test log with `grep '^CAL|'`.
void row(const std::string& s) { std::cout << "CAL| " << s << "\n"; }

}  // namespace

// Spec D8.  Complete production stencils on four nested levels: for each of
// the six points and each tau_iv, all three stencil sigma solved on the grid
// family selected at sigma0 + tau_iv, so the endpoint uncertainties that
// control admission are the ones calibrated.
TEST(ReferenceOracleCalibration, OrderIsUsableStableAndAboveConstant) {
    const auto acc = make_grid_accuracy(kReferenceAccuracy);
    const auto ultra_acc = make_grid_accuracy(GridAccuracyProfile::Ultra);

    double min_usable = std::numeric_limits<double>::infinity();
    double max_usable = -std::numeric_limits<double>::infinity();
    double max_rel_profile_gap = 0.0;  // max |V_High - V_Ultra| / K
    size_t points_with_two_usable = 0;
    size_t usable_count = 0, oscillatory_count = 0, insufficient_count = 0;

    row("point,tau_iv,sigma,level,n_points,n_time,value");

    for (const auto& pt : points()) {
        ReferenceOracle oracle{.dividend_yield = kDividendYield,
                               .option_type = pt.type,
                               .discrete_dividends = pt.divs,
                               .reference_maturity = pt.T,
                               .accuracy = acc};
        const double theta = std::ldexp(1.0, -40) * pt.K;

        // The production stencil at tau_iv = 5e-4, base sigma: its level-0
        // and level-1 prices are the fine/coarse pair the shipped code uses,
        // so delta_hat below is the deployed estimate, not a re-derivation.
        std::optional<Series> production;
        std::optional<ReferenceGridFamily> production_family;

        for (double tau_iv : {5e-4, 1e-3}) {
            const auto widest =
                oracle.contract(pt.S, pt.K, pt.tau, pt.sigma + tau_iv, pt.r);
            auto fam = make_reference_grid_family(widest, acc, 3);
            ASSERT_TRUE(fam.has_value()) << pt.name;
            ASSERT_EQ(fam->levels.size(), 4u) << pt.name;
            // Recorded, not asserted: D8's assertion set is the list below,
            // so a stray failure here would not muddy the calibration
            // signal.  It is worth watching all the same -- D1 says the
            // shipped profiles never round the fine count down, and
            // `itm-2y` at High measurably does (n0 lands in [3490, 3500],
            // the next count = 1 (mod 16) is 3505, and the strict cap is
            // 3500, so the family falls back to 3489).
            RecordProperty(std::string(pt.key) + "_" + tau_key(tau_iv) + "_rounded_down",
                           fam->rounded_down ? "true" : "false");
            if (fam->rounded_down) {
                row(std::string(pt.name) + "," + num(tau_iv, 3)
                    + ",rounded_down,true,n=" + std::to_string(fam->point_counts[0]));
            }

            const double sigmas[3] = {pt.sigma - tau_iv, pt.sigma, pt.sigma + tau_iv};
            for (size_t si = 0; si < 3; ++si) {
                const double s = sigmas[si];
                const char* slabel = (si == 0) ? "lo" : (si == 1) ? "base" : "hi";
                Series ser;
                for (size_t k = 0; k < 4; ++k) {
                    auto r = oracle.solve(
                        oracle.contract(pt.S, pt.K, pt.tau, s, pt.r), fam->levels[k]);
                    ASSERT_TRUE(r.has_value())
                        << pt.name << " tau_iv=" << tau_iv << " sigma=" << s
                        << " level=" << k;
                    ser.v[k] = *r;
                    row(std::string(pt.name) + "," + num(tau_iv, 3) + "," + slabel
                        + "," + std::to_string(k) + ","
                        + std::to_string(fam->point_counts[k]) + ","
                        + std::to_string(fam->time_steps[k]) + "," + num(ser.v[k], 17));
                    RecordProperty(std::string(pt.key) + "_" + tau_key(tau_iv) + "_"
                                       + slabel + "_v" + std::to_string(k),
                                   num(ser.v[k], 17));
                }

                ser.c1 = classify(ser.v[2] - ser.v[1], ser.v[1] - ser.v[0], theta, &ser.p1);
                ser.c2 = classify(ser.v[3] - ser.v[2], ser.v[2] - ser.v[1], theta, &ser.p2);

                const std::string tag =
                    std::string(pt.key) + "_" + tau_key(tau_iv) + "_" + slabel;
                RecordProperty(tag + "_class1", label(ser.c1));
                RecordProperty(tag + "_class2", label(ser.c2));
                RecordProperty(tag + "_p1", num(ser.p1, 6));
                RecordProperty(tag + "_p2", num(ser.p2, 6));
                row(std::string(pt.name) + "," + num(tau_iv, 3) + "," + slabel
                    + ",classify," + label(ser.c1) + "," + label(ser.c2) + ",p1="
                    + num(ser.p1, 6) + ",p2=" + num(ser.p2, 6));

                for (Triple c : {ser.c1, ser.c2}) {
                    if (c == Triple::Usable) ++usable_count;
                    else if (c == Triple::Oscillatory) ++oscillatory_count;
                    else ++insufficient_count;
                }

                EXPECT_NE(ser.c1, Triple::Oscillatory)
                    << pt.name << " tau_iv=" << tau_iv << " sigma=" << slabel
                    << " triple (0,1,2)";
                EXPECT_NE(ser.c2, Triple::Oscillatory)
                    << pt.name << " tau_iv=" << tau_iv << " sigma=" << slabel
                    << " triple (1,2,3)";
                if (ser.c1 == Triple::Usable) {
                    EXPECT_TRUE(std::isfinite(ser.p1) && ser.p1 > 0.0)
                        << pt.name << " p1=" << ser.p1;
                    min_usable = std::min(min_usable, ser.p1);
                    max_usable = std::max(max_usable, ser.p1);
                }
                if (ser.c2 == Triple::Usable) {
                    EXPECT_TRUE(std::isfinite(ser.p2) && ser.p2 > 0.0)
                        << pt.name << " p2=" << ser.p2;
                    min_usable = std::min(min_usable, ser.p2);
                    max_usable = std::max(max_usable, ser.p2);
                }
                if (ser.c1 == Triple::Usable && ser.c2 == Triple::Usable) {
                    EXPECT_NEAR(ser.p1, ser.p2, 0.5)
                        << pt.name << " tau_iv=" << tau_iv << " sigma=" << slabel
                        << " order not stable";
                }

                if (si == 1 && tau_iv == 5e-4) {
                    production = ser;
                    production_family = *fam;
                    if (ser.c1 == Triple::Usable && ser.c2 == Triple::Usable) {
                        ++points_with_two_usable;
                    }
                }
            }
        }

        ASSERT_TRUE(production.has_value()) << pt.name;
        const double v_high = production->v[0];
        const double v_half = production->v[1];
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

        // Effective sensitivity (b): spatial-only and temporal-only halvings,
        // to attribute the level-0 -> level-1 shift between the two axes.
        PDEGridConfig spatial_only{.grid_spec = production_family->levels[1].grid_spec,
                                   .n_time = production_family->levels[0].n_time,
                                   .mandatory_times = {}};
        PDEGridConfig temporal_only{.grid_spec = production_family->levels[0].grid_spec,
                                    .n_time = production_family->levels[1].n_time,
                                    .mandatory_times = {}};
        const auto base_contract = oracle.contract(pt.S, pt.K, pt.tau, pt.sigma, pt.r);
        auto v_spatial = oracle.solve(base_contract, spatial_only);
        auto v_temporal = oracle.solve(base_contract, temporal_only);
        ASSERT_TRUE(v_spatial.has_value()) << pt.name;
        ASSERT_TRUE(v_temporal.has_value()) << pt.name;

        const std::string tag(pt.key);
        RecordProperty(tag + "_V_high", num(v_high, 17));
        RecordProperty(tag + "_V_half", num(v_half, 17));
        RecordProperty(tag + "_V_ultra", num(*v_ultra, 17));
        RecordProperty(tag + "_delta_hat", num(delta, 12));
        RecordProperty(tag + "_profile_gap", num(profile_gap, 12));
        RecordProperty(tag + "_profile_gap_rel_K", num(profile_gap / pt.K, 12));
        RecordProperty(tag + "_domain_shift", num(domain_shift, 12));
        RecordProperty(tag + "_spatial_only_shift", num(std::abs(*v_spatial - v_high), 12));
        RecordProperty(tag + "_temporal_only_shift", num(std::abs(*v_temporal - v_high), 12));
        RecordProperty(tag + "_coarse_shift", num(std::abs(v_half - v_high), 12));

        row(std::string(pt.name) + ",profile,V_high=" + num(v_high, 17)
            + ",V_half=" + num(v_half, 17) + ",V_ultra=" + num(*v_ultra, 17)
            + ",delta_hat=" + num(delta, 12) + ",|V_high-V_ultra|=" + num(profile_gap, 12)
            + ",rel_K=" + num(profile_gap / pt.K, 12));
        row(std::string(pt.name) + ",sensitivity,domain_shift=" + num(domain_shift, 12)
            + ",spatial_only=" + num(std::abs(*v_spatial - v_high), 12)
            + ",temporal_only=" + num(std::abs(*v_temporal - v_high), 12)
            + ",coarse=" + num(std::abs(v_half - v_high), 12));

        EXPECT_LE(profile_gap, delta) << pt.name << " High vs Ultra outside the estimate";
        EXPECT_LE(domain_shift, delta) << pt.name << " domain sensitivity";
    }

    RecordProperty("min_usable_order", num(min_usable, 6));
    RecordProperty("max_usable_order", num(max_usable, 6));
    RecordProperty("max_profile_gap_rel_K", num(max_rel_profile_gap, 12));
    RecordProperty("points_with_two_usable_at_base_sigma",
                   std::to_string(points_with_two_usable));
    RecordProperty("usable_triples", std::to_string(usable_count));
    RecordProperty("oscillatory_triples", std::to_string(oscillatory_count));
    RecordProperty("insufficient_triples", std::to_string(insufficient_count));

    row("summary,min_usable_order=" + num(min_usable, 6)
        + ",max_usable_order=" + num(max_usable, 6)
        + ",max_profile_gap_rel_K=" + num(max_rel_profile_gap, 12)
        + ",points_with_two_usable=" + std::to_string(points_with_two_usable)
        + ",usable=" + std::to_string(usable_count)
        + ",oscillatory=" + std::to_string(oscillatory_count)
        + ",insufficient=" + std::to_string(insufficient_count));

    // Coverage rule: at least four of the six points carry two usable triples
    // at the base sigma of the tau_iv = 5e-4 stencil.
    EXPECT_GE(points_with_two_usable, 4u);
    EXPECT_LE(kReferenceConvergenceOrder, min_usable);
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
