// SPDX-License-Identifier: MIT
/**
 * @file default_banded_backend.hpp
 * @brief The library-wide default banded-solver backend selection
 *
 * This is the single configuration point for which backend the banded
 * fitting layers use by default. Consumers (e.g. BSplineCollocation1D)
 * name only `DefaultBandedBackend`, never a concrete implementation —
 * to swap the library onto a different backend (e.g. an Eigen-based
 * one), change the alias here and nothing else.
 */

#pragma once

#include "mango/math/lapack_banded_backend.hpp"

namespace mango {

/// The backend used when a banded-solver consumer does not specify one.
using DefaultBandedBackend = LapackBandedBackend;

}  // namespace mango
