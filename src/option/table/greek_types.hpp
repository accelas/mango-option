// SPDX-License-Identifier: MIT
#pragma once

namespace mango {

/// First-order Greek type for coordinate transform weight dispatch.
enum class Greek { Delta, Vega, Theta, Rho };

/// Error codes for Greek computation.
enum class GreekError {
    OutOfDomain,       ///< Query point outside surface domain
    NumericalFailure,  ///< FD computation failed (e.g., near boundary)
};

}  // namespace mango
