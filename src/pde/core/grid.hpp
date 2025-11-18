#pragma once

#include <vector>
#include <span>
#include <memory>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <expected>
#include "src/support/error_types.hpp"

namespace mango {

// Forward declarations
template<typename T = double>
class GridBuffer;

template<typename T = double>
class GridView;

template<typename T = double>
class GridSpec;

/**
 * GridSpec: Immutable grid specification (how to generate a grid)
 *
 * This is a value type that describes grid generation parameters.
 * It doesn't own data - call generate() to create a GridBuffer.
 */
template<typename T>
class GridSpec {
public:
    enum class Type {
        Uniform,      // Equally spaced points
        LogSpaced,    // Logarithmically spaced
        SinhSpaced    // Hyperbolic sine spacing (concentrates points at center)
    };

    // Factory methods for common grid types
    static std::expected<GridSpec, std::string> uniform(T x_min, T x_max, size_t n_points) {
        if (n_points < 2) {
            return std::unexpected<std::string>("Grid must have at least 2 points");
        }
        if (x_min >= x_max) {
            return std::unexpected<std::string>("x_min must be less than x_max");
        }
        return GridSpec(Type::Uniform, x_min, x_max, n_points);
    }

    static std::expected<GridSpec, std::string> log_spaced(T x_min, T x_max, size_t n_points) {
        if (n_points < 2) {
            return std::unexpected<std::string>("Grid must have at least 2 points");
        }
        if (x_min <= 0 || x_max <= 0) {
            return std::unexpected<std::string>("Log-spaced grid requires positive bounds");
        }
        if (x_min >= x_max) {
            return std::unexpected<std::string>("x_min must be less than x_max");
        }
        return GridSpec(Type::LogSpaced, x_min, x_max, n_points);
    }

    static std::expected<GridSpec, std::string> sinh_spaced(T x_min, T x_max, size_t n_points, T concentration = T(1.0)) {
        if (n_points < 2) {
            return std::unexpected<std::string>("Grid must have at least 2 points");
        }
        if (x_min >= x_max) {
            return std::unexpected<std::string>("x_min must be less than x_max");
        }
        if (concentration <= 0) {
            return std::unexpected<std::string>("Concentration parameter must be positive");
        }
        return GridSpec(Type::SinhSpaced, x_min, x_max, n_points, concentration);
    }

    // Generate the actual grid
    GridBuffer<T> generate() const;

    // Accessors
    Type type() const { return type_; }
    T x_min() const { return x_min_; }
    T x_max() const { return x_max_; }
    size_t n_points() const { return n_points_; }
    T concentration() const { return concentration_; }

private:
    GridSpec(Type type, T x_min, T x_max, size_t n_points, T concentration = T(1.0))
        : type_(type), x_min_(x_min), x_max_(x_max),
          n_points_(n_points), concentration_(concentration) {}

    Type type_;
    T x_min_;
    T x_max_;
    size_t n_points_;
    T concentration_;  // Only used for sinh spacing
};

/**
 * GridView: Non-owning view of grid data (cheap to copy)
 *
 * This is a lightweight wrapper around std::span that provides
 * grid-specific operations. It doesn't own data and is cheap to copy.
 */
template<typename T>
class GridView {
public:
    // Construct from span
    explicit GridView(std::span<const T> data) : data_(data) {}

    // Copyable and movable (cheap - just a span)
    GridView(const GridView&) = default;
    GridView& operator=(const GridView&) = default;
    GridView(GridView&&) noexcept = default;
    GridView& operator=(GridView&&) noexcept = default;

    // Access
    size_t size() const { return data_.size(); }
    const T& operator[](size_t i) const { return data_[i]; }

    std::span<const T> span() const { return data_; }
    const T* data() const { return data_.data(); }

    // Grid properties
    T x_min() const { return data_[0]; }
    T x_max() const { return data_[data_.size() - 1]; }

    // Check if grid is uniform (within tolerance)
    bool is_uniform(T tolerance = T(1e-10)) const {
        if (data_.size() < 2) return true;
        const T expected_dx = (x_max() - x_min()) / static_cast<T>(data_.size() - 1);
        for (size_t i = 1; i < data_.size(); ++i) {
            const T actual_dx = data_[i] - data_[i-1];
            if (std::abs(actual_dx - expected_dx) > tolerance) {
                return false;
            }
        }
        return true;
    }

private:
    std::span<const T> data_;
};

/**
 * GridBuffer: Owns grid data (movable, not copyable by default)
 *
 * This is the storage container for grid points. It owns a std::vector
 * and provides span-based access. GridBuffer is movable but explicitly
 * not copyable (use share() for shared ownership).
 */
template<typename T>
class GridBuffer {
public:
    // Construct from vector (takes ownership)
    explicit GridBuffer(std::vector<T> data) : data_(std::move(data)) {}

    // Movable
    GridBuffer(GridBuffer&&) noexcept = default;
    GridBuffer& operator=(GridBuffer&&) noexcept = default;

    // Not copyable (use share() for shared ownership)
    GridBuffer(const GridBuffer&) = delete;
    GridBuffer& operator=(const GridBuffer&) = delete;

    // Access
    size_t size() const { return data_.size(); }
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    std::span<T> span() { return data_; }
    std::span<const T> span() const { return data_; }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    // Create non-owning view
    GridView<T> view() const {
        return GridView<T>(std::span<const T>(data_));
    }

    // Create shared ownership (for reuse across solvers)
    std::shared_ptr<GridBuffer<T>> share() && {
        return std::make_shared<GridBuffer<T>>(std::move(*this));
    }

private:
    std::vector<T> data_;
};

template<typename T>
GridBuffer<T> GridSpec<T>::generate() const {
    std::vector<T> points;
    points.reserve(n_points_);

    switch (type_) {
        case Type::Uniform: {
            const T dx = (x_max_ - x_min_) / static_cast<T>(n_points_ - 1);
            for (size_t i = 0; i < n_points_; ++i) {
                points.push_back(x_min_ + static_cast<T>(i) * dx);
            }
            break;
        }

        case Type::LogSpaced: {
            const T log_min = std::log(x_min_);
            const T log_max = std::log(x_max_);
            const T d_log = (log_max - log_min) / static_cast<T>(n_points_ - 1);
            for (size_t i = 0; i < n_points_; ++i) {
                points.push_back(std::exp(log_min + static_cast<T>(i) * d_log));
            }
            break;
        }

        case Type::SinhSpaced: {
            // Sinh spacing: concentrates points at center
            // x(eta) = x_min + (x_max - x_min) * [1 + sinh(c*(eta - 0.5)) / sinh(c/2)] / 2
            // where eta goes from 0 to 1
            const T c = concentration_;
            const T sinh_half_c = std::sinh(c / T(2.0));
            for (size_t i = 0; i < n_points_; ++i) {
                const T eta = static_cast<T>(i) / static_cast<T>(n_points_ - 1);
                const T sinh_term = std::sinh(c * (eta - T(0.5))) / sinh_half_c;
                const T normalized = (T(1.0) + sinh_term) / T(2.0);
                points.push_back(x_min_ + (x_max_ - x_min_) * normalized);
            }
            break;
        }
    }

    return GridBuffer<T>(std::move(points));
}

/**
 * GridHolder: Helper to initialize grid before derived classes
 *
 * This class is designed for use as a base class when you need to
 * initialize grid data before initializing other base classes that
 * depend on the grid. It uses the base-from-member idiom.
 *
 * Example usage:
 * ```cpp
 * class MyWorkspace : private GridHolder, public PDEWorkspace {
 *     MyWorkspace(double x_min, double x_max, size_t n_space, size_t n_time)
 *         : GridHolder(x_min, x_max, n_space)  // Initialize grid first
 *         , PDEWorkspace(grid_view_, n_time)   // Then use grid_view_
 *     {}
 * };
 * ```
 */
class GridHolder {
protected:
    GridBuffer<double> grid_buffer_;
    GridView<double> grid_view_;

    GridHolder(double x_min, double x_max, size_t n_space)
        : grid_buffer_(GridSpec<>::uniform(x_min, x_max, n_space).value().generate())
        , grid_view_(grid_buffer_.span())
    {}
};

/**
 * GridSpacing: Grid spacing information for finite difference operators
 *
 * For UNIFORM grids:
 *   - Stores constant spacing (dx, dx_inv, dx_inv_sq)
 *   - Zero memory overhead for precomputed arrays
 *
 * For NON-UNIFORM grids:
 *   - Eagerly precomputes weight arrays during construction:
 *     * dx_left_inv[i]   = 1 / (x[i] - x[i-1])
 *     * dx_right_inv[i]  = 1 / (x[i+1] - x[i])
 *     * dx_center_inv[i] = 2 / (dx_left + dx_right)
 *     * w_left[i]        = dx_right / (dx_left + dx_right)
 *     * w_right[i]       = dx_left / (dx_left + dx_right)
 *   - Single contiguous buffer (5×(n-2)×8 bytes, ~4KB for n=100)
 *   - Zero-copy span accessors (fail-fast if called on uniform grid)
 *
 * USE CASE:
 *   Tanh-clustered grids for adaptive mesh refinement around strikes/barriers
 *   in option pricing. Grids are fixed during PDE solve, so one-time
 *   precomputation cost (~1-2 µs) is amortized over many time steps.
 *
 * SIMD INTEGRATION:
 *   CenteredDifferenceSIMD loads precomputed arrays via element_aligned spans,
 *   avoiding per-lane divisions. Expected speedup: 3-6x over scalar non-uniform.
 */
template<typename T = double>
class GridSpacing {
public:
    /**
     * Create grid spacing from a grid view
     * @param grid Grid points (non-owning view)
     */
    explicit GridSpacing(GridView<T> grid)
        : grid_(grid)
        , is_uniform_(grid.is_uniform())
        , n_(grid.size())
    {
        if (n_ < 2) return;

        if (is_uniform_) {
            // Uniform grid: compute once
            dx_uniform_ = (grid.x_max() - grid.x_min()) / static_cast<T>(n_ - 1);
            dx_uniform_inv_ = T(1) / dx_uniform_;
            dx_uniform_inv_sq_ = dx_uniform_inv_ * dx_uniform_inv_;
        } else {
            // Non-uniform grid: pre-compute all spacings
            dx_array_.reserve(n_ - 1);
            for (size_t i = 0; i < n_ - 1; ++i) {
                dx_array_.push_back(grid[i + 1] - grid[i]);
            }

            // Precompute non-uniform data for SIMD kernels
            precompute_non_uniform_data();
        }
    }

    // Query if grid is uniform
    bool is_uniform() const { return is_uniform_; }

    // Get uniform spacing (only valid if is_uniform())
    T spacing() const {
        assert(is_uniform_ && "spacing() requires uniform grid");
        return dx_uniform_;
    }

    T spacing_inv() const {
        assert(is_uniform_ && "spacing_inv() requires uniform grid");
        return dx_uniform_inv_;
    }

    T spacing_inv_sq() const {
        assert(is_uniform_ && "spacing_inv_sq() requires uniform grid");
        return dx_uniform_inv_sq_;
    }

    // Get spacing at point i: dx[i] = x[i+1] - x[i]
    // Valid for i in [0, n-2]
    T spacing_at(size_t i) const {
        assert(i < grid_.size() - 1 && "spacing_at(i) requires i < n-1");
        if (is_uniform_) {
            return dx_uniform_;
        } else {
            return dx_array_[i];
        }
    }

    // Get all spacings (for non-uniform grids)
    std::span<const T> spacings() const {
        return dx_array_;
    }

    // Left and right spacing for non-uniform centered differences
    // Preconditions:
    //   left_spacing(i):  requires 1 <= i < size()
    //   right_spacing(i): requires 0 <= i < size()-1
    T left_spacing(size_t i) const {
        assert(i >= 1 && i < grid_.size() && "left_spacing: index out of bounds");
        if (is_uniform_) {
            return dx_uniform_;
        } else {
            return dx_array_[i - 1];  // dx[i-1] = x[i] - x[i-1]
        }
    }

    T right_spacing(size_t i) const {
        assert(i < grid_.size() - 1 && "right_spacing: index out of bounds");
        if (is_uniform_) {
            return dx_uniform_;
        } else {
            return dx_array_[i];  // dx[i] = x[i+1] - x[i]
        }
    }

    // Minimum size for 3-point stencil
    static constexpr size_t min_stencil_size() { return 3; }

    // Access to underlying grid
    const GridView<T>& grid() const { return grid_; }
    size_t size() const { return grid_.size(); }

    // Zero-copy accessors (fail-fast if called on uniform grid)
    // Returns precomputed values for interior points i=1..n-2 (n-2 points)
    std::span<const T> dx_left_inv() const {
        assert(!is_uniform_ && "dx_left_inv only available for non-uniform grids");
        assert(n_ >= min_stencil_size() && "Grid too small for stencil operations");
        const size_t interior_count = n_ - 2;
        return {precomputed_.data(), interior_count};
    }

    std::span<const T> dx_right_inv() const {
        assert(!is_uniform_ && "dx_right_inv only available for non-uniform grids");
        assert(n_ >= min_stencil_size() && "Grid too small for stencil operations");
        const size_t interior_count = n_ - 2;
        return {precomputed_.data() + interior_count, interior_count};
    }

    std::span<const T> dx_center_inv() const {
        assert(!is_uniform_ && "dx_center_inv only available for non-uniform grids");
        assert(n_ >= min_stencil_size() && "Grid too small for stencil operations");
        const size_t interior_count = n_ - 2;
        return {precomputed_.data() + 2 * interior_count, interior_count};
    }

    std::span<const T> w_left() const {
        assert(!is_uniform_ && "w_left only available for non-uniform grids");
        assert(n_ >= min_stencil_size() && "Grid too small for stencil operations");
        const size_t interior_count = n_ - 2;
        return {precomputed_.data() + 3 * interior_count, interior_count};
    }

    std::span<const T> w_right() const {
        assert(!is_uniform_ && "w_right only available for non-uniform grids");
        assert(n_ >= min_stencil_size() && "Grid too small for stencil operations");
        const size_t interior_count = n_ - 2;
        return {precomputed_.data() + 4 * interior_count, interior_count};
    }

private:
    GridView<T> grid_;
    bool is_uniform_;
    size_t n_;

    // Uniform grid: single spacing value (pre-computed)
    T dx_uniform_{};
    T dx_uniform_inv_{};     // 1/dx
    T dx_uniform_inv_sq_{};  // 1/dx²

    // Non-uniform grid: array of spacings (pre-computed)
    std::vector<T> dx_array_;

    // Single buffer: [dx_left_inv | dx_right_inv | dx_center_inv | w_left | w_right]
    std::vector<T> precomputed_;

    void precompute_non_uniform_data() {
        const size_t interior_count = n_ - 2;  // Points i=1..n-2 (n-2 points with both neighbors)
        precomputed_.resize(5 * interior_count);

        // Compute all arrays in one loop (for interior points i=1..n-2)
        for (size_t i = 1; i <= n_ - 2; ++i) {
            const T dx_left = left_spacing(i);     // x[i] - x[i-1]
            const T dx_right = right_spacing(i);   // x[i+1] - x[i]
            const T dx_center = T(0.5) * (dx_left + dx_right);

            const size_t idx = i - 1;  // Index into precomputed arrays

            precomputed_[idx] = T(1) / dx_left;
            precomputed_[interior_count + idx] = T(1) / dx_right;
            precomputed_[2 * interior_count + idx] = T(1) / dx_center;
            precomputed_[3 * interior_count + idx] = dx_right / (dx_left + dx_right);
            precomputed_[4 * interior_count + idx] = dx_left / (dx_left + dx_right);
        }
    }
};

} // namespace mango
