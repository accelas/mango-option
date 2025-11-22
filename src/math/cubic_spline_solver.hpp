/**
 * @file cubic_spline_solver.hpp
 * @brief Modern C++20 natural cubic spline interpolation
 *
 * Replaces legacy C implementation with type-safe, cache-friendly design.
 */

#pragma once

#include "src/math/thomas_solver.hpp"
#include "src/support/parallel.hpp"
#include <span>
#include <vector>
#include <optional>
#include <concepts>
#include <algorithm>
#include <cmath>
#include <string>

namespace mango {

/// Cubic spline interpolation result
template<std::floating_point T>
struct SplineEvalResult {
    T value;
    bool success;

    [[nodiscard]] constexpr explicit operator bool() const noexcept { return success; }
};

/// Cubic spline boundary condition type
enum class SplineBoundary {
    NATURAL,      ///< Natural spline: f''(x₀) = f''(xₙ) = 0
    CLAMPED,      ///< Clamped spline: f'(x₀) and f'(xₙ) specified
    NOT_A_KNOT    ///< Not-a-knot: f'''(x₁⁻) = f'''(x₁⁺), f'''(xₙ₋₁⁻) = f'''(xₙ₋₁⁺)
};

/// Configuration for cubic spline construction
template<std::floating_point T>
struct CubicSplineConfig {
    SplineBoundary boundary_type = SplineBoundary::NATURAL;
    T left_derivative = 0;    ///< For CLAMPED boundary at left
    T right_derivative = 0;   ///< For CLAMPED boundary at right
    ThomasConfig<T> thomas_config = {};  ///< Thomas solver configuration
};

/// Natural Cubic Spline Interpolator (Modern C++20)
///
/// Constructs a piecewise cubic polynomial that:
/// - Passes through all data points (x[i], y[i])
/// - Has continuous first and second derivatives
/// - Satisfies specified boundary conditions
///
/// For n points, creates n-1 cubic polynomials of the form:
///   S[i](x) = a[i] + b[i]·Δx + c[i]·Δx² + d[i]·Δx³
///   where Δx = x - x[i]
///
/// Memory layout: struct-of-arrays for cache efficiency
///
/// @tparam T Floating point type (float, double, long double)
template<std::floating_point T>
class CubicSpline {
public:
    /// Default constructor (empty spline)
    CubicSpline() = default;

    /// Construct spline from data points
    ///
    /// @param x X-coordinates (must be strictly increasing)
    /// @param y Y-coordinates
    /// @param config Spline configuration
    /// @return Optional error message (nullopt on success)
    [[nodiscard]] std::optional<std::string_view> build(
        std::span<const T> x,
        std::span<const T> y,
        const CubicSplineConfig<T>& config = {})
    {
        const size_t n = x.size();

        // Validate input
        if (x.size() != y.size()) {
            return "X and Y sizes must match";
        }
        if (n < 2) {
            return "Need at least 2 points for interpolation";
        }

        // Check that x is strictly increasing
        for (size_t i = 1; i < n; ++i) {
            if (x[i] <= x[i-1]) {
                return "X coordinates must be strictly increasing";
            }
        }

        // Store grid points
        x_.assign(x.begin(), x.end());
        y_.assign(y.begin(), y.end());

        // Allocate coefficient storage (n-1 intervals)
        const size_t n_intervals = n - 1;
        a_.resize(n_intervals);
        b_.resize(n_intervals);
        c_.resize(n);  // n values (including boundary)
        d_.resize(n_intervals);

        // Compute interval widths (h[i] = x[i+1] - x[i])
        h_.resize(n_intervals);
        MANGO_PRAGMA_SIMD
        for (size_t i = 0; i < n_intervals; ++i) {
            h_[i] = x[i+1] - x[i];
        }

        // Store config for rebuild_same_grid
        config_ = config;

        // Build tridiagonal system for second derivatives
        // Natural boundary: c[0] = c[n-1] = 0
        if (config.boundary_type == SplineBoundary::NATURAL) {
            if (!build_natural_spline(h_, config.thomas_config)) {
                return "Thomas solver failed (singular matrix)";
            }
        } else {
            return "Only NATURAL boundary conditions implemented (CLAMPED and NOT_A_KNOT coming soon)";
        }

        // Compute cubic coefficients from second derivatives
        compute_coefficients(h_);

        return std::nullopt;  // Success
    }

    /// Rebuild spline with new y-values on the same x-grid
    ///
    /// PERFORMANCE: Reuses cached interval widths and grid structure.
    /// Much faster than build() when only y-values change.
    ///
    /// @param y New Y-coordinates (must match existing grid size)
    /// @return Optional error message (nullopt on success)
    ///
    /// @pre build() must have been called successfully at least once
    [[nodiscard]] std::optional<std::string_view> rebuild_same_grid(
        std::span<const T> y)
    {
        if (x_.empty()) {
            return "Must call build() before rebuild_same_grid()";
        }
        if (y.size() != y_.size()) {
            return "Y size must match existing grid";
        }

        // Update y-values
        y_.assign(y.begin(), y.end());

        // Rebuild spline using cached interval widths
        if (config_.boundary_type == SplineBoundary::NATURAL) {
            if (!build_natural_spline(h_, config_.thomas_config)) {
                return "Thomas solver failed (singular matrix)";
            }
        } else {
            return "Only NATURAL boundary conditions implemented";
        }

        // Compute cubic coefficients from second derivatives
        compute_coefficients(h_);

        return std::nullopt;  // Success
    }

    /// Evaluate spline at point x
    ///
    /// @param x_eval Evaluation point
    /// @return Interpolated value (or extrapolated if outside domain)
    [[nodiscard]] T eval(T x_eval) const noexcept {
        if (x_.empty()) return static_cast<T>(0);
        if (x_.size() == 1) return y_[0];

        // Find interval containing x_eval (binary search)
        const size_t i = find_interval(x_eval);

        // Evaluate cubic polynomial in interval i
        const T dx = x_eval - x_[i];
        return a_[i] + b_[i] * dx + c_[i] * (dx * dx) + d_[i] * (dx * dx * dx);
    }

    /// Evaluate first derivative at point x
    ///
    /// @param x_eval Evaluation point
    /// @return Spline derivative
    [[nodiscard]] T eval_derivative(T x_eval) const noexcept {
        if (x_.empty()) return static_cast<T>(0);
        if (x_.size() == 1) return static_cast<T>(0);

        const size_t i = find_interval(x_eval);
        const T dx = x_eval - x_[i];

        // S'(x) = b + 2c·Δx + 3d·Δx²
        return b_[i] + static_cast<T>(2) * c_[i] * dx +
               static_cast<T>(3) * d_[i] * (dx * dx);
    }

    /// Evaluate second derivative at point x
    ///
    /// @param x_eval Evaluation point
    /// @return Spline second derivative
    [[nodiscard]] T eval_second_derivative(T x_eval) const noexcept {
        if (x_.empty()) return static_cast<T>(0);
        if (x_.size() == 1) return static_cast<T>(0);

        const size_t i = find_interval(x_eval);
        const T dx = x_eval - x_[i];

        // S''(x) = 2c + 6d·Δx
        return static_cast<T>(2) * c_[i] + static_cast<T>(6) * d_[i] * dx;
    }

    /// Get interpolation domain
    [[nodiscard]] std::pair<T, T> domain() const noexcept {
        if (x_.empty()) return {0, 0};
        return {x_.front(), x_.back()};
    }

    /// Get number of knot points
    [[nodiscard]] size_t size() const noexcept { return x_.size(); }

    /// Check if spline is initialized
    [[nodiscard]] bool empty() const noexcept { return x_.empty(); }

private:
    // Grid data
    std::vector<T> x_;  ///< Knot x-coordinates
    std::vector<T> y_;  ///< Knot y-coordinates

    // Spline coefficients (struct-of-arrays for cache efficiency)
    std::vector<T> a_;  ///< Constant terms (n-1)
    std::vector<T> b_;  ///< Linear terms (n-1)
    std::vector<T> c_;  ///< Quadratic terms (n)
    std::vector<T> d_;  ///< Cubic terms (n-1)

    // Cached data for rebuild_same_grid
    std::vector<T> h_;  ///< Cached interval widths (n-1)
    CubicSplineConfig<T> config_;  ///< Cached spline configuration

    /// Build natural cubic spline (second derivative boundary conditions)
    bool build_natural_spline(
        std::span<const T> h,
        const ThomasConfig<T>& thomas_config)
    {
        const size_t n = x_.size();

        // Natural boundary: c[0] = c[n-1] = 0
        c_[0] = 0;
        c_[n-1] = 0;

        // For n > 2, solve tridiagonal system for interior c values
        if (n == 2) {
            // Linear interpolation (only 1 interval)
            return true;
        }

        // Build tridiagonal system: A·c = rhs
        // where c = [c[1], c[2], ..., c[n-2]]
        //
        // Row i (for i=1..n-2):
        //   h[i-1]·c[i-1] + 2(h[i-1] + h[i])·c[i] + h[i]·c[i+1] =
        //     3·((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

        const size_t m = n - 2;  // Interior points
        std::vector<T> lower(m - 1);
        std::vector<T> diag(m);
        std::vector<T> upper(m - 1);
        std::vector<T> rhs(m);
        std::vector<T> c_interior(m);

        // Build tridiagonal coefficients
        for (size_t i = 0; i < m; ++i) {
            const size_t k = i + 1;  // Original index in full array

            diag[i] = static_cast<T>(2) * (h[k-1] + h[k]);

            if (i > 0) {
                lower[i-1] = h[k-1];
            }
            if (i < m - 1) {
                upper[i] = h[k];
            }

            // RHS
            const T slope_right = (y_[k+1] - y_[k]) / h[k];
            const T slope_left = (y_[k] - y_[k-1]) / h[k-1];
            rhs[i] = static_cast<T>(3) * (slope_right - slope_left);
        }

        // Solve tridiagonal system
        ThomasWorkspace<T> workspace(m);
        auto result = solve_thomas<T>(
            std::span{lower},
            std::span{diag},
            std::span{upper},
            std::span{rhs},
            std::span{c_interior},
            workspace.get(),
            thomas_config
        );

        if (!result.ok()) {
            return false;
        }

        // Copy interior second derivatives to full array
        MANGO_PRAGMA_SIMD
        for (size_t i = 0; i < m; ++i) {
            c_[i + 1] = c_interior[i];
        }

        return true;
    }

    /// Compute cubic coefficients from second derivatives
    void compute_coefficients(std::span<const T> h) {
        const size_t n_intervals = x_.size() - 1;

        MANGO_PRAGMA_SIMD
        for (size_t i = 0; i < n_intervals; ++i) {
            // For interval [x[i], x[i+1]]:
            // S[i](x) = a[i] + b[i]·Δx + c[i]·Δx² + d[i]·Δx³

            a_[i] = y_[i];

            b_[i] = (y_[i+1] - y_[i]) / h[i] -
                    h[i] * (static_cast<T>(2) * c_[i] + c_[i+1]) / static_cast<T>(3);

            // c_[i] already computed (second derivative / 2)

            d_[i] = (c_[i+1] - c_[i]) / (static_cast<T>(3) * h[i]);
        }
    }

    /// Find interval containing x using binary search
    [[nodiscard]] size_t find_interval(T x_eval) const noexcept {
        // Boundary cases
        if (x_eval <= x_[0]) return 0;
        if (x_eval >= x_.back()) return x_.size() - 2;

        // Binary search
        auto it = std::lower_bound(x_.begin(), x_.end(), x_eval);
        size_t idx = std::distance(x_.begin(), it);

        // lower_bound returns first element >= x_eval
        // We want the interval [x[i], x[i+1]] containing x_eval
        if (idx > 0 && (idx == x_.size() || x_[idx] > x_eval)) {
            --idx;
        }

        return std::min(idx, x_.size() - 2);
    }
};

/// Separable 2D Cubic Spline Interpolation
///
/// Interpolates a 2D surface z(x,y) using separable cubic splines:
/// 1. Builds and caches x-direction splines (one per y-slice)
/// 2. Evaluates x-splines at query point to get intermediate values
/// 3. Builds y-direction spline from intermediate values
/// 4. Evaluates y-spline at query point
///
/// This provides C² continuity in both dimensions with natural boundary conditions.
///
/// @tparam T Floating point type (float, double, long double)
template<std::floating_point T>
class CubicSpline2D {
public:
    /// Default constructor (empty spline)
    CubicSpline2D() = default;

    /// Construct and build 2D spline from gridded data
    ///
    /// @param x X-coordinates (must be strictly increasing)
    /// @param y Y-coordinates (must be strictly increasing)
    /// @param z Z-values in row-major order: z[i*ny + j] = z(x[i], y[j])
    /// @param config Spline configuration (applied to both dimensions)
    /// @return Optional error message (nullopt on success)
    [[nodiscard]] std::optional<std::string> build(
        std::span<const T> x,
        std::span<const T> y,
        std::span<const T> z,
        const CubicSplineConfig<T>& config = {})
    {
        const size_t nx = x.size();
        const size_t ny = y.size();

        if (nx < 2 || ny < 2) {
            return "Grid too small for 2D cubic spline (need at least 2 points per dimension)";
        }

        if (z.size() != nx * ny) {
            return "Z array size must equal nx * ny";
        }

        // Store grid coordinates
        x_ = std::vector<T>(x.begin(), x.end());
        y_ = std::vector<T>(y.begin(), y.end());

        // Build x-direction splines (one per y-slice)
        x_splines_.resize(ny);
        for (size_t j = 0; j < ny; ++j) {
            // Extract z-values for this y-slice: z(x_i, y_j)
            std::vector<T> z_slice(nx);
            MANGO_PRAGMA_SIMD
            for (size_t i = 0; i < nx; ++i) {
                z_slice[i] = z[i * ny + j];
            }

            // Build spline for this slice
            auto error = x_splines_[j].build(x_, z_slice, config);
            if (error.has_value()) {
                return "Failed to build x-spline for y-slice " + std::to_string(j) + ": " + std::string(error.value());
            }
        }

        return std::nullopt;
    }

    /// Evaluate interpolated value at (x_eval, y_eval)
    ///
    /// Uses separable cubic spline interpolation with natural boundary conditions.
    /// Extrapolation uses nearest boundary value.
    ///
    /// @param x_eval X-coordinate to evaluate
    /// @param y_eval Y-coordinate to evaluate
    /// @return Interpolated value z(x_eval, y_eval), or 0.0 if not built
    [[nodiscard]] T eval(T x_eval, T y_eval) const {
        if (!is_built()) {
            return T{0};
        }

        const size_t ny = y_.size();

        // Step 1: Evaluate all x-direction splines at x_eval
        std::vector<T> y_values(ny);
        for (size_t j = 0; j < ny; ++j) {
            y_values[j] = x_splines_[j].eval(x_eval);
        }

        // Step 2: Build y-direction spline from interpolated values
        CubicSpline<T> y_spline;
        auto error = y_spline.build(y_, y_values);
        if (error.has_value()) {
            // Fallback to linear interpolation if spline build fails
            auto it = std::lower_bound(y_.begin(), y_.end(), y_eval);
            if (it == y_.begin()) {
                return y_values[0];
            } else if (it == y_.end()) {
                return y_values[ny - 1];
            }
            size_t j = std::distance(y_.begin(), it) - 1;
            T t = (y_eval - y_[j]) / (y_[j + 1] - y_[j]);
            return (static_cast<T>(1) - t) * y_values[j] + t * y_values[j + 1];
        }

        // Step 3: Evaluate y-direction spline at y_eval
        return y_spline.eval(y_eval);
    }

    /// Check if spline is built and ready
    [[nodiscard]] bool is_built() const noexcept {
        return !x_splines_.empty();
    }

private:
    std::vector<T> x_;                        ///< X-coordinates
    std::vector<T> y_;                        ///< Y-coordinates
    std::vector<CubicSpline<T>> x_splines_;   ///< Cached x-direction splines (one per y-slice)
};

}  // namespace mango
