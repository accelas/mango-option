// SPDX-License-Identifier: MIT
#pragma once

/// @file adaptive_metrics.hpp
/// @brief Option-domain adapters for the generic adaptive refinement loop.
///
/// `run_refinement` (adaptive_refinement.hpp) is deliberately ignorant of
/// American options: every domain specific enters through its callbacks.
/// This header owns the mango-side implementations of those callbacks --
/// the FD reference solver, the FD-vega reference generator, and the
/// IV-error scoring metric -- so the loop itself never links the PDE
/// solver.

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/option_spec.hpp"
#include <vector>

namespace mango {

/// Compute IV error from price error and vega, with floor and cap.
double compute_iv_error(double price_error, double vega,
                        double vega_floor, double target_iv_error);

/// Produce ErrorRefs (FD American price + FD central-difference vega) for
/// one point: base solve + two sigma-bump solves.
/// 2 extra PDE solves per point — acceptable at build time.
/// Any failed or non-finite solve => unexpected.
PrepareRefsFn make_fd_vega_refs_fn(const AdaptiveGridParams& params,
                                    const ValidateFn& validate_fn);

/// Score an interpolated price against cached ErrorRefs using the TV/K
/// filter (skips points where TV/K < 1e-4; IV undefined there) and
/// `compute_iv_error` arithmetic (vega floor + target-level noise clamp).
/// Filtered points return `std::nullopt`, never 0.0: a skip is the absence of
/// a measurement, not a perfect one.
ScoreErrorFn make_iv_score_fn(const AdaptiveGridParams& params,
                              OptionType option_type);

/// Create a ValidateFn that solves a single American option via FD.
ValidateFn make_validate_fn(double dividend_yield,
                            OptionType option_type,
                            const std::vector<Dividend>& discrete_dividends = {});

}  // namespace mango
