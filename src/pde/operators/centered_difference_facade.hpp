#pragma once

#include "grid_spacing.hpp"
#include "centered_difference_scalar.hpp"
#include "centered_difference_simd_backend.hpp"
#include "src/support/cpu/feature_detection.hpp"
#include <span>
#include <memory>
#include <cassert>
#include <concepts>

namespace mango::operators {

/**
 * CenteredDifference: Unified façade with automatic backend selection
 *
 * Mode::Auto (default) uses CPU feature detection to pick optimal backend.
 * Mode::Scalar/Simd force specific backend (for testing).
 *
 * Virtual dispatch overhead: ~5-10ns per call (negligible vs computation).
 *
 * @tparam T Floating-point type (float, double, long double)
 */
template<std::floating_point T = double>
class CenteredDifference {
public:
    enum class Mode { Auto, Scalar, Simd };

    explicit CenteredDifference(const GridSpacing<T>& spacing,
                                Mode mode = Mode::Auto);

    // Movable but not copyable (owns unique_ptr)
    CenteredDifference(const CenteredDifference&) = delete;
    CenteredDifference& operator=(const CenteredDifference&) = delete;
    CenteredDifference(CenteredDifference&&) = default;
    CenteredDifference& operator=(CenteredDifference&&) = default;

    // Public API - virtual dispatch happens after IFUNC resolution
    void compute_second_derivative(std::span<const T> u,
                                   std::span<T> d2u_dx2,
                                   size_t start, size_t end) const {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");
        impl_->compute_second_derivative(u, d2u_dx2, start, end);
    }

    void compute_first_derivative(std::span<const T> u,
                                  std::span<T> du_dx,
                                  size_t start, size_t end) const {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");
        impl_->compute_first_derivative(u, du_dx, start, end);
    }

private:
    struct BackendInterface {
        virtual ~BackendInterface() = default;
        virtual void compute_second_derivative(
            std::span<const T> u, std::span<T> d2u_dx2,
            size_t start, size_t end) const = 0;
        virtual void compute_first_derivative(
            std::span<const T> u, std::span<T> du_dx,
            size_t start, size_t end) const = 0;
    };

    template<typename Backend>
    struct BackendImpl final : BackendInterface {
        Backend backend_;

        explicit BackendImpl(const GridSpacing<T>& spacing)
            : backend_(spacing) {}

        void compute_second_derivative(
            std::span<const T> u, std::span<T> d2u_dx2,
            size_t start, size_t end) const override {
            backend_.compute_second_derivative(u, d2u_dx2, start, end);
        }

        void compute_first_derivative(
            std::span<const T> u, std::span<T> du_dx,
            size_t start, size_t end) const override {
            backend_.compute_first_derivative(u, du_dx, start, end);
        }
    };

    std::unique_ptr<BackendInterface> impl_;
};

// Constructor implementation
template<std::floating_point T>
inline CenteredDifference<T>::CenteredDifference(const GridSpacing<T>& spacing,
                                                  Mode mode)
{
    if (mode == Mode::Auto) {
        // Check CPU features AND OS support
        auto features = cpu::detect_cpu_features();
        bool os_supports_avx = cpu::check_os_avx_support();

        // Use SIMD if both CPU and OS support it
        if ((features.has_avx2 || features.has_avx512f) && os_supports_avx) {
            mode = Mode::Simd;
        } else {
            mode = Mode::Scalar;
        }
    }

    switch (mode) {
        case Mode::Scalar:
            impl_ = std::make_unique<BackendImpl<ScalarBackend<T>>>(spacing);
            break;
        case Mode::Simd:
            impl_ = std::make_unique<BackendImpl<SimdBackend<T>>>(spacing);
            break;
        case Mode::Auto:
            // Already resolved above
            break;
    }
}

} // namespace mango::operators
