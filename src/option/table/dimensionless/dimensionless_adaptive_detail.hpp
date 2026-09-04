// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"

namespace mango::detail {

/// Ground-truth probe for the dimensionless adaptive loop: the normalized
/// early-exercise premium max(V/K - European/K, 0) at a single
/// dimensionless point (x0, tau'_0, ln kappa_0), from a direct PDE solve
/// with sigma_eff = sqrt(2), r = kappa, q = 0.  Returns 0.0 if the solve
/// or the slice spline fails.  Exposed only so the probe's own PDE domain
/// coverage can be regression-tested (#480).
double dimensionless_reference_eep(double x0, double tau_prime_0,
                                   double ln_kappa_0, double K_ref,
                                   OptionType option_type);

}  // namespace mango::detail
