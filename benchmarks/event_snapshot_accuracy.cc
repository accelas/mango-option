// SPDX-License-Identifier: MIT
// Common-query accuracy comparison. Generate references once, then evaluate
// both revisions against that same CSV. All reference solves are independent
// of table sampling; High/Ultra disagreement is reported rather than hidden.
#include "mango/option/american_option.hpp"
#include "mango/option/dividend_utils.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/bspline/bspline_adaptive.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace mango;

namespace {
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

struct Point {
    size_t id;
    int scenario;
    std::string region;
    double strike, tau, sigma;
    double high = nan, ultra = nan, vega = nan;
};

SegmentedAdaptiveConfig config_for(int scenario) {
    return {.spot = 100, .option_type = OptionType::PUT, .dividend_yield = 0.02,
            .discrete_dividends = scenario == 0
                ? std::vector<Dividend>{{0.25, 0.5}, {0.5, 0.5}, {0.75, 0.5}}
                : std::vector<Dividend>{{10.0 / 365, 0.5}},
            .maturity = scenario == 0 ? 1.0 : 30.0 / 365,
            .kref_config = {.K_refs = {90, 92.5, 95, 97.5, 100, 102.5, 105, 107.5, 110}}};
}

IVGrid domain() {
    return {.moneyness = {std::log(0.92), std::log(0.95), 0, std::log(1.05), std::log(1.08)},
            .vol = {0.1, 0.15, 0.2, 0.3}, .rate = {0.02, 0.03, 0.05, 0.07}};
}

PricingParams pricing(const Point& pt, double sigma) {
    auto config = config_for(pt.scenario);
    PricingParams p(OptionSpec{.spot = 100, .strike = pt.strike,
        .maturity = pt.tau, .rate = 0.04, .dividend_yield = 0.02,
        .option_type = OptionType::PUT}, sigma);
    p.discrete_dividends = rolled_dividends(config.discrete_dividends, config.maturity, pt.tau);
    return p;
}

double reference(const Point& pt, double sigma, GridAccuracyProfile profile) {
    auto solver = AmericanOptionSolver::create(pricing(pt, sigma),
        PDEGridSpec{make_grid_accuracy(profile)});
    if (!solver) return nan;
    auto result = solver->solve();
    return result ? result->value() : nan;
}

std::vector<Point> points() {
    std::vector<Point> out;
    for (int scenario : {0, 1}) {
        const auto config = config_for(scenario);
        std::vector<std::pair<std::string, double>> times;
        const std::vector<double> interior = scenario == 0
            ? std::vector<double>{0.03, 0.10, 0.33, 0.60, 0.90, 1.0}
            : std::vector<double>{1.0/365, 7.0/365, 14.0/365, 25.0/365, 30.0/365};
        for (double tau : interior) times.push_back({"interior", tau});
        for (const auto& dividend : config.discrete_dividends) {
            const double event = config.maturity - dividend.calendar_time;
            // The edge points are outside the old +/-0.0005 exclusion.
            for (double dt : {-0.0006, -0.0001, 0.0, 0.0001, 0.0006}) {
                times.push_back({std::abs(dt) > 0.0005 ? "edge" : "gap", event + dt});
            }
        }
        for (const auto& [region, tau] : times) {
            // Two strike midpoints and one anchor; all vols and the rate are
            // off the supplied seed knots, so this tests real interpolation.
            for (double strike : {96.25, 100.0, 103.75}) {
                for (double sigma : {0.125, 0.225, 0.275}) {
                    out.push_back({out.size(), scenario, region, strike, tau, sigma});
                }
            }
        }
    }
    return out;
}

int make_references() {
    auto pts = points();
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < pts.size(); ++i) {
        auto& p = pts[i];
        p.high = reference(p, p.sigma, GridAccuracyProfile::High);
        p.ultra = reference(p, p.sigma, GridAccuracyProfile::Ultra);
        const double h = 0.01 * p.sigma;
        const double up = reference(p, p.sigma + h, GridAccuracyProfile::Ultra);
        const double down = reference(p, p.sigma - h, GridAccuracyProfile::Ultra);
        p.vega = (up - down) / (2 * h);
    }
    std::cout << "id,scenario,region,strike,tau,sigma,high,ultra,vega\n";
    size_t failed = 0;
    for (const auto& p : pts) {
        std::cout << p.id << ',' << p.scenario << ',' << p.region << ','
                  << p.strike << ',' << p.tau << ',' << p.sigma << ','
                  << p.high << ',' << p.ultra << ',' << p.vega << '\n';
        failed += !std::isfinite(p.high) || !std::isfinite(p.ultra) || !std::isfinite(p.vega);
    }
    std::cerr << "references=" << pts.size() << " failed=" << failed << '\n';
    return failed ? 1 : 0;
}

std::vector<Point> read_references(const char* file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("Cannot open reference CSV");
    std::vector<Point> out;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        std::vector<std::string> fields;
        std::istringstream row(line);
        std::string value;
        while (std::getline(row, value, ',')) fields.push_back(value);
        if (fields.size() != 9) throw std::runtime_error("Invalid reference CSV row");
        out.push_back({std::stoul(fields[0]), std::stoi(fields[1]), fields[2],
            std::stod(fields[3]), std::stod(fields[4]), std::stod(fields[5]),
            std::stod(fields[6]), std::stod(fields[7]), std::stod(fields[8])});
    }
    return out;
}

template <typename Inner>
void evaluate(const std::string& fixture, const PriceTable<Inner>& table,
              const std::vector<Point>& pts, int scenario, bool anchor_only) {
    auto solver = InterpolatedIVSolver<PriceTable<Inner>>::create(
        table, {}, config_for(scenario).discrete_dividends);
    if (!solver) throw std::runtime_error("Cannot wrap table for IV solving");
    for (const auto& p : pts) {
        if (p.scenario != scenario || (anchor_only && p.strike != 100)) continue;
        const double price = table.price(100, p.strike, p.tau, p.sigma, 0.04);
        const auto spec = pricing(p, p.sigma);
        auto iv = solver->solve(IVQuery(spec, p.ultra, spec.discrete_dividends));
        auto iv_high = solver->solve(IVQuery(spec, p.high, spec.discrete_dividends));
        std::cout << fixture << ',' << p.id << ',' << table.contains_maturity(p.tau)
                  << ',' << price << ',' << (iv ? iv->implied_vol : nan)
                  << ',' << (iv ? -1 : static_cast<int>(iv.error().code))
                  << ',' << (iv_high ? iv_high->implied_vol : nan) << '\n';
    }
}

int evaluate_placements() {
    size_t failed = 0;
    std::cout << "first_days,maturity_days,built,max_iv_error,target_met\n";
    for (double first : {10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 91.25}) {
        for (double days : {7.0, 14.0, 30.0, 60.0, 90.0, 180.0, 365.0, 730.0}) {
            auto config = config_for(1);
            config.maturity = days / 365.0;
            config.discrete_dividends.clear();
            for (double d = first; d < days; d += 91.25)
                config.discrete_dividends.push_back({d / 365.0, 0.5});
            auto result = build_adaptive_bspline_segmented(
                AdaptiveGridParams{.target_iv_error = 1e-3}, config, domain());
            std::cout << first << ',' << days << ',' << result.has_value() << ','
                      << (result ? result->achieved_max_error : nan) << ','
                      << (result && result->target_met) << std::endl;
            failed += !result;
        }
    }
    std::cerr << "placements=56 failed=" << failed << '\n';
    return failed ? 1 : 0;
}

int evaluate_tables(const char* file) {
    const auto pts = read_references(file);
    std::cout << "fixture,id,supported,price,iv,iv_error_code,iv_high\n";
    for (int scenario : {0, 1}) {
        auto config = config_for(scenario);
        auto result = build_adaptive_bspline_segmented(
            AdaptiveGridParams{.target_iv_error = 1e-3}, config, domain());
        if (!result) throw std::runtime_error("Adaptive table build refused");
        BSplineMultiKRefSurface table(result->surface, result->sample_bounds, OptionType::PUT, 0.02);
        evaluate(scenario == 0 ? "adaptive_1y" : "adaptive_30d", table, pts, scenario, false);
    }
    auto grid = domain();
    grid.moneyness.clear();
    for (int i = 0; i < 60; ++i) grid.moneyness.push_back(
        std::lerp(std::log(0.92), std::log(1.08), i / 59.0));
    SegmentedPriceTableBuilder::Config config{};
    config.K_ref = 100;
    config.option_type = OptionType::PUT;
    config.dividends.dividend_yield = 0.02;
    config.dividends.discrete_dividends = config_for(0).discrete_dividends;
    config.grid = grid;
    config.maturity = 1.0;
    auto built = SegmentedPriceTableBuilder::build(config);
    if (!built) throw std::runtime_error("Manual table build refused");
    PriceTable<BSplineSegmentedSurface> table(std::move(*built),
        SurfaceBounds{std::log(0.92), std::log(1.08), 0, 1, 0.1, 0.3, 0.02, 0.07},
        OptionType::PUT, 0.02);
    evaluate("manual_1y", table, pts, 0, true);
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    std::cout << std::setprecision(17);
    if (argc == 2 && std::string(argv[1]) == "--references") return make_references();
    if (argc == 2 && std::string(argv[1]) == "--placements") return evaluate_placements();
    if (argc == 3 && std::string(argv[1]) == "--evaluate") return evaluate_tables(argv[2]);
    std::cerr << "Usage: event_snapshot_accuracy --references | --evaluate refs.csv | --placements\n";
    return 2;
}
