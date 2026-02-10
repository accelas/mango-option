// SPDX-License-Identifier: MIT
#include "mango/option/table/chebyshev/chebyshev_table_builder.hpp"

#include "mango/math/chebyshev/chebyshev_nodes.hpp"
#include "mango/math/cubic_spline_solver.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/table/eep/eep_decomposer.hpp"

#include <chrono>
#include <cmath>

namespace mango {

namespace {

/// EEP accessor over pre-built cubic splines at CGL nodes.
///
/// Flat layout: [m][tau][sigma][rate] (row-major, rate innermost)
/// to match B-spline tensor layout for cache-friendly access.
class ChebyshevSplineAccessor {
public:
    ChebyshevSplineAccessor(
        std::span<const double> m_nodes,
        std::span<const double> tau_nodes,
        std::span<const double> sigma_nodes,
        std::span<const double> rate_nodes,
        // splines[si * n_rate * n_tau + ri * n_tau + ti]
        std::span<const CubicSpline<double>> splines,
        double K_ref,
        std::span<double> out)
        : m_(m_nodes), tau_(tau_nodes), sigma_(sigma_nodes), rate_(rate_nodes),
          splines_(splines), K_ref_(K_ref), out_(out),
          Nm_(m_nodes.size()), Nt_(tau_nodes.size()),
          Nv_(sigma_nodes.size()), Nr_(rate_nodes.size()) {}

    size_t size() const { return Nm_ * Nt_ * Nv_ * Nr_; }
    double strike() const { return K_ref_; }

    double american_price(size_t i) const {
        auto [mi, ti, vi, ri] = to_4d(i);
        const auto& spline = splines_[vi * Nr_ * Nt_ + ri * Nt_ + ti];
        return spline.eval(m_[mi]) * K_ref_;
    }

    double spot(size_t i) const {
        return std::exp(m_[to_4d(i).mi]) * K_ref_;
    }

    double tau(size_t i) const { return tau_[to_4d(i).ti]; }
    double sigma(size_t i) const { return sigma_[to_4d(i).vi]; }
    double rate(size_t i) const { return rate_[to_4d(i).ri]; }

    void set_value(size_t i, double v) {
        // Output layout matches input: [m][tau][sigma][rate]
        out_[i] = v;
    }

private:
    struct Idx4D { size_t mi, ti, vi, ri; };

    Idx4D to_4d(size_t flat) const {
        size_t ri = flat % Nr_;  flat /= Nr_;
        size_t vi = flat % Nv_;  flat /= Nv_;
        size_t ti = flat % Nt_;
        size_t mi = flat / Nt_;
        return {mi, ti, vi, ri};
    }

    std::span<const double> m_, tau_, sigma_, rate_;
    std::span<const CubicSpline<double>> splines_;
    double K_ref_;
    std::span<double> out_;
    size_t Nm_, Nt_, Nv_, Nr_;
};

}  // namespace

std::expected<ChebyshevTableResult, PriceTableError>
build_chebyshev_table(const ChebyshevTableConfig& config) {
    auto t0 = std::chrono::steady_clock::now();

    const size_t n_m     = config.num_pts[0];
    const size_t n_tau   = config.num_pts[1];
    const size_t n_sigma = config.num_pts[2];
    const size_t n_rate  = config.num_pts[3];

    // Generate CGL nodes per axis
    auto m_nodes     = chebyshev_nodes(n_m,     config.domain.lo[0], config.domain.hi[0]);
    auto tau_nodes   = chebyshev_nodes(n_tau,   config.domain.lo[1], config.domain.hi[1]);
    auto sigma_nodes = chebyshev_nodes(n_sigma, config.domain.lo[2], config.domain.hi[2]);
    auto rate_nodes  = chebyshev_nodes(n_rate,  config.domain.lo[3], config.domain.hi[3]);

    // Build batch: one PricingParams per (sigma, rate) pair
    std::vector<PricingParams> batch;
    batch.reserve(n_sigma * n_rate);
    for (size_t si = 0; si < n_sigma; ++si) {
        for (size_t ri = 0; ri < n_rate; ++ri) {
            batch.emplace_back(
                OptionSpec{
                    .spot = config.K_ref,
                    .strike = config.K_ref,
                    .maturity = config.domain.hi[1],  // max tau for full time domain
                    .rate = rate_nodes[ri],
                    .dividend_yield = config.dividend_yield,
                    .option_type = config.option_type,
                },
                sigma_nodes[si]);
        }
    }

    // Solve batch with snapshots at tau CGL nodes
    BatchAmericanOptionSolver solver;
    solver.set_snapshot_times(std::span<const double>(tau_nodes));
    auto batch_result = solver.solve_batch(
        std::span<const PricingParams>(batch), /*use_shared_grid=*/true);

    size_t n_pde_solves = batch.size() - batch_result.failed_count;

    // Phase 1: Build cubic splines from PDE solutions.
    // Layout: splines[si * n_rate * n_tau + ri * n_tau + ti]
    std::vector<CubicSpline<double>> splines(n_sigma * n_rate * n_tau);

    for (size_t si = 0; si < n_sigma; ++si) {
        for (size_t ri = 0; ri < n_rate; ++ri) {
            size_t batch_idx = si * n_rate + ri;
            const auto& res = batch_result.results[batch_idx];

            if (!res.has_value()) {
                return std::unexpected(PriceTableError{
                    PriceTableErrorCode::ExtractionFailed,
                    batch_idx, 0});
            }

            auto grid = res->grid();
            auto x_grid = grid->x();

            for (size_t ti = 0; ti < n_tau; ++ti) {
                auto solution = res->at_time(ti);
                auto& spline = splines[si * n_rate * n_tau + ri * n_tau + ti];
                auto build_error = spline.build(x_grid, solution);
                if (build_error.has_value()) {
                    return std::unexpected(PriceTableError{
                        PriceTableErrorCode::ExtractionFailed,
                        batch_idx, ti});
                }
            }
        }
    }

    // Phase 2: EEP decomposition via accessor.
    size_t total = n_m * n_tau * n_sigma * n_rate;
    std::vector<double> eep_values(total);

    ChebyshevSplineAccessor accessor(
        m_nodes, tau_nodes, sigma_nodes, rate_nodes,
        splines, config.K_ref, eep_values);
    analytical_eep_decompose(accessor, config.option_type, config.dividend_yield);

    // Build Chebyshev interpolant from EEP values
    auto interp = ChebyshevInterpolant<4, TuckerTensor<4>>::build_from_values(
        std::span<const double>(eep_values),
        config.domain, config.num_pts, config.tucker_epsilon);

    // Wrap in EEPSurfaceAdapter + PriceTable
    ChebyshevLeaf leaf(
        std::move(interp),
        StandardTransform4D{},
        AnalyticalEEP(config.option_type, config.dividend_yield),
        config.K_ref);

    SurfaceBounds bounds{
        .m_min = config.domain.lo[0], .m_max = config.domain.hi[0],
        .tau_min = config.domain.lo[1], .tau_max = config.domain.hi[1],
        .sigma_min = config.domain.lo[2], .sigma_max = config.domain.hi[2],
        .rate_min = config.domain.lo[3], .rate_max = config.domain.hi[3],
    };

    ChebyshevSurface surface(
        std::move(leaf), bounds, config.option_type, config.dividend_yield);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    return ChebyshevTableResult{
        .surface = std::move(surface),
        .n_pde_solves = n_pde_solves,
        .build_seconds = elapsed,
    };
}

}  // namespace mango
