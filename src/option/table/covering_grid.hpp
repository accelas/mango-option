// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/grid_spec_types.hpp"
#include "mango/option/option_spec.hpp"
#include <span>

namespace mango::detail {

/// Ensure accuracy.n_sigma is large enough that a shared grid estimated by
/// estimate_batch_pde_grid(batch, accuracy) for a normalized batch
/// (spot = strike = K_ref, x0 = 0) spans every log-moneyness node in
/// `log_moneyness_nodes`: that grid's baseline half-width is
/// n_sigma * max(sigma*sqrt(T)) over `batch`, and any node beyond it would
/// be evaluated by cubic-spline extrapolation of the slice.  The reach is
/// max(|min|, |max|) over the nodes in ANY order (Chebyshev/CC node arrays
/// are handed in directly).  Callers must actually solve on such an
/// estimated grid, passed as a concrete custom grid: the batch solver's
/// gridless routing re-estimates per normalized group (or unions the
/// unwidened per-param grids) and does NOT realize this width.  No-op when
/// either span is empty.
void ensure_moneyness_coverage(GridAccuracyParams& accuracy,
                               std::span<const PricingParams> batch,
                               std::span<const double> log_moneyness_nodes);

/// Widen `accuracy` for moneyness coverage, estimate the shared batch
/// grid, and return it as a concrete PDEGridConfig suitable for
/// solve_batch's custom_grid parameter, which propagates it verbatim into
/// every normalized group and into the regular shared path.
/// mandatory_times stays empty: the batch solver reconstructs per-contract
/// dividend taus itself.
PDEGridSpec materialize_covering_grid(GridAccuracyParams accuracy,
                                      std::span<const PricingParams> batch,
                                      std::span<const double> log_moneyness_nodes);

}  // namespace mango::detail
