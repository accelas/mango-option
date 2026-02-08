// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/price_table_axes.hpp"
#include <array>
#include <functional>

namespace mango {

/// Recursively iterate over all combinations of axis indices
///
/// Calls func for every combination of indices across N dimensions.
/// Uses compile-time recursion to unroll loops.
///
/// @tparam Axis Current axis (0 to N-1)
/// @tparam N Number of dimensions
/// @tparam Func Callable accepting std::array<size_t, N>
template <size_t Axis, size_t N, typename Func>
void for_each_axis_index_impl(
    const PriceTableAxesND<N>& axes,
    std::array<size_t, N>& indices,
    Func&& func)
{
    if constexpr (Axis == N) {
        // Base case: all axes filled, call function
        func(indices);
    } else {
        // Recursive case: iterate over current axis
        for (size_t i = 0; i < axes.grids[Axis].size(); ++i) {
            indices[Axis] = i;
            for_each_axis_index_impl<Axis + 1>(axes, indices, std::forward<Func>(func));
        }
    }
}

/// Public entry point for axis index iteration
///
/// @tparam StartAxis Starting axis (usually 0)
/// @tparam N Number of dimensions
/// @tparam Func Callable accepting std::array<size_t, N>
template <size_t StartAxis, size_t N, typename Func>
void for_each_axis_index(const PriceTableAxesND<N>& axes, Func&& func) {
    std::array<size_t, N> indices{};
    for_each_axis_index_impl<StartAxis>(axes, indices, std::forward<Func>(func));
}

} // namespace mango
