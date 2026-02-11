// SPDX-License-Identifier: MIT
// Chebyshev code moved to chebyshev/chebyshev_adaptive.cpp
// B-spline code moved to bspline/bspline_adaptive.cpp
// This file retains only the class shell (constructor/destructor/move ops).
#include "mango/option/table/adaptive_grid_builder.hpp"
#include "mango/option/table/bspline/bspline_slice_cache.hpp"

namespace mango {

AdaptiveGridBuilder::AdaptiveGridBuilder(AdaptiveGridParams params)
    : params_(std::move(params))
    , cache_(std::make_unique<SliceCache>())
{}

AdaptiveGridBuilder::~AdaptiveGridBuilder() = default;
AdaptiveGridBuilder::AdaptiveGridBuilder(AdaptiveGridBuilder&&) noexcept = default;
AdaptiveGridBuilder& AdaptiveGridBuilder::operator=(AdaptiveGridBuilder&&) noexcept = default;

}  // namespace mango
