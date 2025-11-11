#pragma once

#include "grid_spacing.hpp"
#include "centered_difference_scalar.hpp"
#include "centered_difference_simd_backend.hpp"
#include "../cpu/feature_detection.hpp"
#include <span>
#include <memory>
#include <cassert>

namespace mango::operators {

/**
 * CenteredDifference: Unified façade with automatic backend selection
 *
 * Mode::Auto (default) uses CPU feature detection to pick optimal backend.
 * Mode::Scalar/Simd force specific backend (for testing).
 *
 * Virtual dispatch overhead: ~5-10ns per call (negligible vs computation).
 */
class CenteredDifference {
public:
    enum class Mode { Auto, Scalar, Simd };

    explicit CenteredDifference(const GridSpacing<double>& spacing,
                                Mode mode = Mode::Auto);

    // Public API - keeps [[gnu::target_clones]] for consistent symbols
    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative(std::span<const double> u,
                                   std::span<double> d2u_dx2,
                                   size_t start, size_t end) const {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");
        impl_->compute_second_derivative(u, d2u_dx2, start, end);
    }

    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_first_derivative(std::span<const double> u,
                                  std::span<double> du_dx,
                                  size_t start, size_t end) const {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");
        impl_->compute_first_derivative(u, du_dx, start, end);
    }

    [[gnu::target_clones("default","avx2","avx512f")]]
    void compute_second_derivative_tiled(std::span<const double> u,
                                         std::span<double> d2u_dx2,
                                         size_t start, size_t end) const {
        assert(start >= 1 && "start must allow u[i-1] access");
        assert(end <= u.size() - 1 && "end must allow u[i+1] access");
        impl_->compute_second_derivative_tiled(u, d2u_dx2, start, end);
    }

private:
    struct BackendInterface {
        virtual ~BackendInterface() = default;
        virtual void compute_second_derivative(
            std::span<const double> u, std::span<double> d2u_dx2,
            size_t start, size_t end) const = 0;
        virtual void compute_first_derivative(
            std::span<const double> u, std::span<double> du_dx,
            size_t start, size_t end) const = 0;
        virtual void compute_second_derivative_tiled(
            std::span<const double> u, std::span<double> d2u_dx2,
            size_t start, size_t end) const = 0;
    };

    template<typename Backend>
    struct BackendImpl final : BackendInterface {
        Backend backend_;

        explicit BackendImpl(const GridSpacing<double>& spacing)
            : backend_(spacing) {}

        void compute_second_derivative(
            std::span<const double> u, std::span<double> d2u_dx2,
            size_t start, size_t end) const override {
            backend_.compute_second_derivative(u, d2u_dx2, start, end);
        }

        void compute_first_derivative(
            std::span<const double> u, std::span<double> du_dx,
            size_t start, size_t end) const override {
            backend_.compute_first_derivative(u, du_dx, start, end);
        }

        void compute_second_derivative_tiled(
            std::span<const double> u, std::span<double> d2u_dx2,
            size_t start, size_t end) const override {
            backend_.compute_second_derivative_tiled(u, d2u_dx2, start, end);
        }
    };

    std::unique_ptr<BackendInterface> impl_;
};

// Constructor implementation
inline CenteredDifference::CenteredDifference(const GridSpacing<double>& spacing,
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
            impl_ = std::make_unique<BackendImpl<ScalarBackend<double>>>(spacing);
            break;
        case Mode::Simd:
            impl_ = std::make_unique<BackendImpl<SimdBackend<double>>>(spacing);
            break;
        case Mode::Auto:
            // Already resolved above
            break;
    }
}

} // namespace mango::operators
