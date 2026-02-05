// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/price_table_surface.hpp"
#include "mango/support/error_types.hpp"
#include <memory>
#include <vector>
#include <expected>

namespace mango {

/// Per-maturity price surface with τ interpolation
///
/// Stores separate 3D B-spline surfaces (m × σ × r) for each maturity point,
/// then interpolates across τ using cubic splines. This avoids global 4D
/// smoothing that causes bias near the American exercise boundary.
///
/// Thread-safe after construction.
class PerMaturityPriceSurface {
public:
    /// Build from per-maturity 3D surfaces
    ///
    /// @param surfaces 3D surfaces (moneyness × vol × rate), one per maturity
    /// @param tau_grid Maturity grid (must match surfaces.size())
    /// @param metadata Shared metadata (K_ref, dividends, etc.)
    /// @return Surface or error
    [[nodiscard]] static std::expected<std::shared_ptr<const PerMaturityPriceSurface>, PriceTableError>
    build(std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces,
          std::vector<double> tau_grid,
          PriceTableMetadata metadata);

    /// Access maturity grid
    [[nodiscard]] const std::vector<double>& tau_grid() const noexcept { return tau_grid_; }

    /// Access metadata
    [[nodiscard]] const PriceTableMetadata& metadata() const noexcept { return meta_; }

    /// Number of maturity slices
    [[nodiscard]] size_t num_maturities() const noexcept { return surfaces_.size(); }

    /// Evaluate price at query point
    ///
    /// Uses cubic spline interpolation across τ.
    ///
    /// @param m Moneyness (S/K)
    /// @param tau Time to maturity
    /// @param sigma Volatility
    /// @param rate Interest rate
    /// @return Interpolated price
    [[nodiscard]] double value(double m, double tau, double sigma, double rate) const;

    /// Evaluate with 4D coordinate array (for compatibility with PriceTableSurface<4>)
    [[nodiscard]] double value(const std::array<double, 4>& coords) const {
        return value(coords[0], coords[1], coords[2], coords[3]);
    }

    /// Partial derivative along specified axis
    ///
    /// @param axis 0=moneyness, 1=tau, 2=sigma, 3=rate
    /// @param m Moneyness
    /// @param tau Time to maturity
    /// @param sigma Volatility
    /// @param rate Interest rate
    /// @return Partial derivative
    [[nodiscard]] double partial(size_t axis, double m, double tau, double sigma, double rate) const;

    /// Partial derivative with 4D coordinate array
    [[nodiscard]] double partial(size_t axis, const std::array<double, 4>& coords) const {
        return partial(axis, coords[0], coords[1], coords[2], coords[3]);
    }

private:
    PerMaturityPriceSurface(
        std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces,
        std::vector<double> tau_grid,
        PriceTableMetadata metadata);

    /// Find bracketing τ indices and interpolation weight
    void find_tau_bracket(double tau, size_t& lo, size_t& hi, double& t) const;

    std::vector<std::shared_ptr<const PriceTableSurface<3>>> surfaces_;
    std::vector<double> tau_grid_;
    PriceTableMetadata meta_;
};

}  // namespace mango
