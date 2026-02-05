// SPDX-License-Identifier: MIT

#include "mango/option/table/per_maturity_price_surface.hpp"
#include <algorithm>
#include <cmath>

namespace mango {

PerMaturityPriceSurface::PerMaturityPriceSurface(
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces,
    std::vector<double> tau_grid,
    PriceTableMetadata metadata)
    : surfaces_(std::move(surfaces))
    , tau_grid_(std::move(tau_grid))
    , meta_(std::move(metadata))
{}

std::expected<std::shared_ptr<const PerMaturityPriceSurface>, PriceTableError>
PerMaturityPriceSurface::build(
    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces,
    std::vector<double> tau_grid,
    PriceTableMetadata metadata)
{
    if (surfaces.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    if (surfaces.size() != tau_grid.size()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    if (tau_grid.size() < 2) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // Verify tau_grid is sorted
    for (size_t i = 1; i < tau_grid.size(); ++i) {
        if (tau_grid[i] <= tau_grid[i - 1]) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        }
    }

    // Verify all surfaces are valid
    for (const auto& surf : surfaces) {
        if (!surf) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        }
    }

    auto ptr = std::shared_ptr<const PerMaturityPriceSurface>(
        new PerMaturityPriceSurface(std::move(surfaces), std::move(tau_grid), std::move(metadata)));
    return ptr;
}

void PerMaturityPriceSurface::find_tau_bracket(
    double tau, size_t& lo, size_t& hi, double& t) const
{
    // Clamp to grid bounds
    if (tau <= tau_grid_.front()) {
        lo = 0;
        hi = 0;
        t = 0.0;
        return;
    }
    if (tau >= tau_grid_.back()) {
        lo = tau_grid_.size() - 1;
        hi = lo;
        t = 0.0;
        return;
    }

    // Binary search for bracketing interval
    auto it = std::upper_bound(tau_grid_.begin(), tau_grid_.end(), tau);
    hi = static_cast<size_t>(it - tau_grid_.begin());
    lo = hi - 1;

    // Interpolation weight in [0, 1]
    double tau_lo = tau_grid_[lo];
    double tau_hi = tau_grid_[hi];
    t = (tau - tau_lo) / (tau_hi - tau_lo);
}

double PerMaturityPriceSurface::value(double m, double tau, double sigma, double rate) const
{
    size_t lo, hi;
    double t;
    find_tau_bracket(tau, lo, hi, t);

    std::array<double, 3> coords_3d = {m, sigma, rate};

    if (lo == hi) {
        // At boundary - return single surface value
        return surfaces_[lo]->value(coords_3d);
    }

    // Linear interpolation between bracketing surfaces
    // (Catmull-Rom upgrade would use 4 points)
    double v_lo = surfaces_[lo]->value(coords_3d);
    double v_hi = surfaces_[hi]->value(coords_3d);

    return v_lo + t * (v_hi - v_lo);
}

double PerMaturityPriceSurface::partial(
    size_t axis, double m, double tau, double sigma, double rate) const
{
    size_t lo, hi;
    double t;
    find_tau_bracket(tau, lo, hi, t);

    std::array<double, 3> coords_3d = {m, sigma, rate};

    if (axis == 1) {
        // ∂/∂τ: derivative of linear interpolation
        if (lo == hi) {
            // At boundary - use finite difference with adjacent slice
            if (lo == 0 && tau_grid_.size() > 1) {
                double v0 = surfaces_[0]->value(coords_3d);
                double v1 = surfaces_[1]->value(coords_3d);
                return (v1 - v0) / (tau_grid_[1] - tau_grid_[0]);
            } else if (lo == tau_grid_.size() - 1 && tau_grid_.size() > 1) {
                size_t n = tau_grid_.size();
                double v0 = surfaces_[n - 2]->value(coords_3d);
                double v1 = surfaces_[n - 1]->value(coords_3d);
                return (v1 - v0) / (tau_grid_[n - 1] - tau_grid_[n - 2]);
            }
            return 0.0;
        }

        double v_lo = surfaces_[lo]->value(coords_3d);
        double v_hi = surfaces_[hi]->value(coords_3d);
        return (v_hi - v_lo) / (tau_grid_[hi] - tau_grid_[lo]);
    }

    // For other axes, interpolate the partial derivatives from 3D surfaces
    // Map 4D axis to 3D axis: 0→0 (m), 2→1 (sigma), 3→2 (rate)
    size_t axis_3d;
    if (axis == 0) {
        axis_3d = 0;  // moneyness
    } else if (axis == 2) {
        axis_3d = 1;  // sigma
    } else if (axis == 3) {
        axis_3d = 2;  // rate
    } else {
        return 0.0;  // invalid axis
    }

    if (lo == hi) {
        return surfaces_[lo]->partial(axis_3d, coords_3d);
    }

    double d_lo = surfaces_[lo]->partial(axis_3d, coords_3d);
    double d_hi = surfaces_[hi]->partial(axis_3d, coords_3d);

    return d_lo + t * (d_hi - d_lo);
}

}  // namespace mango
