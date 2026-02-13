// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <memory>

namespace mango {

/// Generic adapter that wraps shared_ptr<const T> to satisfy
/// SurfaceInterpolant.  Preserves shared ownership semantics so
/// the same interpolant can be exposed via PriceTableResult while
/// a surface wraps it for evaluation.
///
/// eval_second_partial() is conditionally available: it exists on
/// SharedInterp<T, N> if and only if T provides it.  This interacts
/// correctly with the `if constexpr (requires { ... })` check in
/// TransformLeaf — when T lacks the method the FD fallback triggers.
///
/// @tparam T  Concrete interpolant type (e.g. BSplineND<double, 4>)
/// @tparam N  Number of interpolation dimensions
template <typename T, size_t N>
class SharedInterp {
public:
    explicit SharedInterp(std::shared_ptr<const T> ptr)
        : ptr_(std::move(ptr)) {}

    [[nodiscard]] double eval(const std::array<double, N>& coords) const {
        return ptr_->eval(coords);
    }

    [[nodiscard]] double partial(size_t axis, const std::array<double, N>& coords) const {
        return ptr_->partial(axis, coords);
    }

    [[nodiscard]] double eval_second_partial(size_t axis, const std::array<double, N>& coords) const
        requires requires(const T& t) { t.eval_second_partial(axis, coords); }
    {
        return ptr_->eval_second_partial(axis, coords);
    }

    /// Access the underlying interpolant.
    [[nodiscard]] const T& get() const { return *ptr_; }

private:
    std::shared_ptr<const T> ptr_;
};

} // namespace mango
