// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_factory.hpp
 * @brief Factory function that hides the two IV solver paths
 *
 * Provides make_interpolated_iv_solver() which builds the appropriate price surface
 * (AmericanPriceSurface for continuous dividends, MultiKRefSurface<>
 * for discrete dividends) and wraps it in a type-erased AnyIVSolver.
 *
 * Grid density is controlled via IVGrid.  When `adaptive` is set, the grid
 * values serve as domain bounds for automatic refinement; otherwise they are
 * exact interpolation knots.
 */

#pragma once

#include <optional>
#include <vector>
#include <variant>
#include <expected>
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/american_price_surface.hpp"
#include "mango/option/table/spliced_surface.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/support/error_types.hpp"

namespace mango {

/// Standard path: continuous dividends only, maturity grid for interpolation
struct StandardIVPath {
    std::vector<double> maturity_grid;
};

/// Segmented path: discrete dividends with multi-K_ref surface
struct SegmentedIVPath {
    double maturity = 1.0;
    std::vector<Dividend> discrete_dividends;
    MultiKRefConfig kref_config;  ///< defaults to auto
    std::vector<double> strike_grid;  ///< optional explicit strikes for per-strike surfaces
};

/// Configuration for the IV solver factory
struct IVSolverFactoryConfig {
    OptionType option_type = OptionType::PUT;
    double spot = 100.0;
    double dividend_yield = 0.0;
    IVGrid grid;                                    ///< Grid points (exact or domain bounds)
    std::optional<AdaptiveGridParams> adaptive;     ///< If set, refine grid adaptively
    InterpolatedIVSolverConfig solver_config;       ///< Newton config
    std::variant<StandardIVPath, SegmentedIVPath> path;
};

/// Type-erased IV solver wrapping either path
class AnyIVSolver {
public:
    /// Solve for implied volatility (single query)
    std::expected<IVSuccess, IVError> solve(const IVQuery& query) const;

    /// Solve for implied volatility (batch with OpenMP)
    BatchIVResult solve_batch(const std::vector<IVQuery>& queries) const;

    /// Constructor from standard solver
    explicit AnyIVSolver(InterpolatedIVSolver<AmericanPriceSurface> solver);

    /// Constructor from segmented solver (spliced surface)
    explicit AnyIVSolver(InterpolatedIVSolver<MultiKRefSurfaceWrapper<>> solver);
    /// Constructor from per-strike solver (spliced surface)
    explicit AnyIVSolver(InterpolatedIVSolver<StrikeSurfaceWrapper<>> solver);

private:
    using SolverVariant = std::variant<
        InterpolatedIVSolver<AmericanPriceSurface>,
        InterpolatedIVSolver<MultiKRefSurfaceWrapper<>>,
        InterpolatedIVSolver<StrikeSurfaceWrapper<>>
    >;
    SolverVariant solver_;
};

/// Factory function: build price surface and IV solver from config
///
/// If path holds StandardIVPath, uses the AmericanPriceSurface path.
/// If path holds SegmentedIVPath, uses the MultiKRefSurface path.
/// If adaptive is set, uses AdaptiveGridBuilder
/// to automatically refine grid density until the target IV error is met.
///
/// @param config Solver configuration
/// @return Type-erased AnyIVSolver or ValidationError
std::expected<AnyIVSolver, ValidationError> make_interpolated_iv_solver(const IVSolverFactoryConfig& config);

}  // namespace mango
