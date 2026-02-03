// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_factory.hpp
 * @brief Factory function that hides the two IV solver paths
 *
 * Provides make_interpolated_iv_solver() which builds the appropriate price surface
 * (AmericanPriceSurface for continuous dividends, SegmentedMultiKRefSurface
 * for discrete dividends) and wraps it in a type-erased AnyIVSolver.
 *
 * Grid density is controlled via IVGridSpec: ManualGrid for explicit grid
 * points, or AdaptiveGrid for automatic refinement to a target IV accuracy.
 */

#pragma once

#include <vector>
#include <variant>
#include <expected>
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/american_price_surface.hpp"
#include "mango/option/table/segmented_multi_kref_surface.hpp"
#include "mango/option/table/segmented_multi_kref_builder.hpp"
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
};

/// Manual grid specification: explicit grid points for each axis.
/// Requires >= 4 points per axis (B-spline minimum).
struct ManualGrid {
    std::vector<double> moneyness;
    std::vector<double> vol;
    std::vector<double> rate;
};

/// Adaptive grid specification: automatic grid density tuning.
/// User provides domain bounds and a target IV accuracy; the builder
/// iteratively refines until the target is met.
struct AdaptiveGrid {
    AdaptiveGridParams params;

    /// Domain bounds (min/max extracted automatically).
    /// Defaults cover typical equity option ranges.
    std::vector<double> moneyness = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};
    std::vector<double> vol = {0.05, 0.10, 0.20, 0.30, 0.50};
    std::vector<double> rate = {0.01, 0.03, 0.05, 0.10};
};

/// Grid specification: manual or adaptive
using IVGridSpec = std::variant<ManualGrid, AdaptiveGrid>;

/// Configuration for the IV solver factory
struct IVSolverFactoryConfig {
    OptionType option_type = OptionType::PUT;
    double spot = 100.0;
    double dividend_yield = 0.0;
    IVGridSpec grid;                           ///< Grid points or adaptive spec
    InterpolatedIVSolverConfig solver_config;  ///< Newton config
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

    /// Constructor from segmented solver
    explicit AnyIVSolver(InterpolatedIVSolver<SegmentedMultiKRefSurface> solver);

private:
    using SolverVariant = std::variant<
        InterpolatedIVSolver<AmericanPriceSurface>,
        InterpolatedIVSolver<SegmentedMultiKRefSurface>
    >;
    SolverVariant solver_;
};

/// Factory function: build price surface and IV solver from config
///
/// If path holds StandardIVPath, uses the AmericanPriceSurface path.
/// If path holds SegmentedIVPath, uses the SegmentedMultiKRefSurface path.
/// If grid holds AdaptiveGrid, uses AdaptiveGridBuilder
/// to automatically refine grid density until the target IV error is met.
///
/// @param config Solver configuration
/// @return Type-erased AnyIVSolver or ValidationError
std::expected<AnyIVSolver, ValidationError> make_interpolated_iv_solver(const IVSolverFactoryConfig& config);

}  // namespace mango
