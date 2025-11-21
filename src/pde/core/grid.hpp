#pragma once

#include <vector>
#include <span>
#include <memory>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <expected>
#include <variant>
#include "src/support/error_types.hpp"
#include "src/pde/core/time_domain.hpp"

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

/// Uniform grid spacing data (minimal storage: 4 values)
///
/// For uniform grids, spacing is constant everywhere.
/// Memory: 32 bytes (3 doubles + 1 size_t)
template<typename T = double>
struct UniformSpacing {
    T dx;           ///< Grid spacing
    T dx_inv;       ///< 1/dx (precomputed for performance)
    T dx_inv_sq;    ///< 1/dx² (precomputed for performance)
    size_t n;       ///< Number of grid points

    /// Construct from spacing and grid size
    ///
    /// @param spacing Grid spacing (dx)
    /// @param size Number of grid points
    UniformSpacing(T spacing, size_t size)
        : dx(spacing)
        , dx_inv(T(1) / spacing)
        , dx_inv_sq(dx_inv * dx_inv)
        , n(size)
    {}
};

/// Non-uniform grid spacing data (precomputed weight arrays)
///
/// For non-uniform grids, precomputes all spacing-dependent values
/// needed for finite difference operators.
///
/// Memory layout: [dx_left_inv | dx_right_inv | dx_center_inv | w_left | w_right]
/// Each section has size (n-2) for interior points
///
/// Memory: ~40 bytes overhead + 5×(n-2)×sizeof(T)
///         For n=100, double: ~4 KB
template<typename T = double>
struct NonUniformSpacing {
    size_t n;  ///< Number of grid points

    /// Precomputed arrays (single contiguous buffer)
    /// Layout: [dx_left_inv | dx_right_inv | dx_center_inv | w_left | w_right]
    std::vector<T> precomputed;

    /// Construct from non-uniform grid points
    ///
    /// @param x Grid points (must be sorted, size >= 3)
    explicit NonUniformSpacing(std::span<const T> x)
        : n(x.size())
    {
        const size_t interior = n - 2;
        precomputed.resize(5 * interior);

        // Precompute all spacing arrays for interior points i=1..n-2
        for (size_t i = 1; i <= n - 2; ++i) {
            const T dx_left = x[i] - x[i-1];
            const T dx_right = x[i+1] - x[i];
            const T dx_center = T(0.5) * (dx_left + dx_right);

            const size_t idx = i - 1;  // Index into arrays (0-based)

            precomputed[idx] = T(1) / dx_left;
            precomputed[interior + idx] = T(1) / dx_right;
            precomputed[2 * interior + idx] = T(1) / dx_center;
            precomputed[3 * interior + idx] = dx_right / (dx_left + dx_right);
            precomputed[4 * interior + idx] = dx_left / (dx_left + dx_right);
        }
    }

    /// Get inverse left spacing for each interior point
    /// Returns: 1/(x[i] - x[i-1]) for i=1..n-2
    std::span<const T> dx_left_inv() const {
        const size_t interior = n - 2;
        return {precomputed.data(), interior};
    }

    /// Get inverse right spacing for each interior point
    /// Returns: 1/(x[i+1] - x[i]) for i=1..n-2
    std::span<const T> dx_right_inv() const {
        const size_t interior = n - 2;
        return {precomputed.data() + interior, interior};
    }

    /// Get inverse center spacing for each interior point
    /// Returns: 2/(dx_left + dx_right) for i=1..n-2
    std::span<const T> dx_center_inv() const {
        const size_t interior = n - 2;
        return {precomputed.data() + 2 * interior, interior};
    }

    /// Get left weight for weighted first derivative
    /// Returns: dx_right/(dx_left + dx_right) for i=1..n-2
    std::span<const T> w_left() const {
        const size_t interior = n - 2;
        return {precomputed.data() + 3 * interior, interior};
    }

    /// Get right weight for weighted first derivative
    /// Returns: dx_left/(dx_left + dx_right) for i=1..n-2
    std::span<const T> w_right() const {
        const size_t interior = n - 2;
        return {precomputed.data() + 4 * interior, interior};
    }
};

/**
 * GridSpacing: Grid spacing information for finite difference operators
 *
 * Uses std::variant to store either UniformSpacing or NonUniformSpacing.
 * Memory efficient: only stores active alternative.
 *
 * For UNIFORM grids:
 *   - Stores constant spacing (dx, dx_inv, dx_inv_sq)
 *   - Memory: ~40 bytes
 *
 * For NON-UNIFORM grids:
 *   - Precomputes weight arrays during construction
 *   - Memory: ~40 bytes + 5×(n-2)×8 bytes (~4KB for n=100)
 *
 * SIMD INTEGRATION:
 *   CenteredDifference operators load precomputed arrays via element_aligned spans.
 */
template<typename T = double>
class GridSpacing {
public:
    using SpacingVariant = std::variant<UniformSpacing<T>, NonUniformSpacing<T>>;

    /**
     * Create grid spacing from a grid view
     *
     * Auto-detects uniform vs non-uniform and constructs appropriate variant.
     *
     * @param grid Grid points (non-owning view)
     */
    explicit GridSpacing(GridView<T> grid)
        : grid_(grid)
        , spacing_(compute_spacing(grid))
    {
    }

    // Query if grid is uniform (zero-cost - checks variant index)
    bool is_uniform() const {
        return std::holds_alternative<UniformSpacing<T>>(spacing_);
    }

    // Get size
    size_t size() const {
        return std::visit([](const auto& s) { return s.n; }, spacing_);
    }

    // Minimum grid size for stencil operations
    static constexpr size_t min_stencil_size() { return 3; }

    // Get uniform spacing (only valid if is_uniform())
    T spacing() const {
        return std::get<UniformSpacing<T>>(spacing_).dx;
    }

    T spacing_inv() const {
        return std::get<UniformSpacing<T>>(spacing_).dx_inv;
    }

    T spacing_inv_sq() const {
        return std::get<UniformSpacing<T>>(spacing_).dx_inv_sq;
    }

    // Get non-uniform arrays (only valid if !is_uniform())
    std::span<const T> dx_left_inv() const {
        return std::get<NonUniformSpacing<T>>(spacing_).dx_left_inv();
    }

    std::span<const T> dx_right_inv() const {
        return std::get<NonUniformSpacing<T>>(spacing_).dx_right_inv();
    }

    std::span<const T> dx_center_inv() const {
        return std::get<NonUniformSpacing<T>>(spacing_).dx_center_inv();
    }

    std::span<const T> w_left() const {
        return std::get<NonUniformSpacing<T>>(spacing_).w_left();
    }

    std::span<const T> w_right() const {
        return std::get<NonUniformSpacing<T>>(spacing_).w_right();
    }

    // Access to underlying grid
    const GridView<T>& grid() const { return grid_; }

private:
    static SpacingVariant compute_spacing(GridView<T> grid) {
        const size_t n = grid.size();

        if (n < 2) {
            // Degenerate case: treat as uniform with zero spacing
            return UniformSpacing<T>(T(0), n);
        }

        // Check uniformity (within tolerance)
        const T expected_dx = (grid.x_max() - grid.x_min()) / static_cast<T>(n - 1);
        constexpr T tolerance = T(1e-10);
        bool is_uniform = true;

        for (size_t i = 1; i < n; ++i) {
            const T actual_dx = grid[i] - grid[i-1];
            if (std::abs(actual_dx - expected_dx) > tolerance) {
                is_uniform = false;
                break;
            }
        }

        // Construct appropriate variant alternative
        if (is_uniform) {
            return UniformSpacing<T>(expected_dx, n);
        } else {
            return NonUniformSpacing<T>(grid.span());
        }
    }

    GridView<T> grid_;
    SpacingVariant spacing_;
};

// ============================================================================
// Grid: Main grid class with solution storage
// ============================================================================

/// Grid with persistent solution storage and metadata
/// Outlives PDESolver, passed via shared_ptr for lifetime management
template<typename T = double>
class Grid {
public:
    /// Create grid with solution storage
    /// @param grid_spec Grid specification (uniform, sinh, etc)
    /// @param time_domain Time domain information
    /// @return Grid instance or error message
    static std::expected<std::shared_ptr<Grid<T>>, std::string>
    create(const GridSpec<T>& grid_spec, const TimeDomain& time_domain) {
        // Generate grid buffer
        auto grid_buffer = grid_spec.generate();
        auto grid_view = grid_buffer.view();

        // Create GridSpacing
        auto spacing = GridSpacing<T>(grid_view);

        size_t n = grid_view.size();

        // Allocate solution storage (2 × n for current + previous)
        std::vector<T> solution(2 * n);

        // Create instance (private constructor, so use new)
        auto grid = std::shared_ptr<Grid<T>>(
            new Grid<T>(
                std::move(grid_buffer),
                std::move(spacing),
                time_domain,
                std::move(solution)
            )
        );

        return grid;
    }

    // Accessors

    /// Spatial grid points (read-only)
    std::span<const T> x() const {
        return grid_buffer_.span();
    }

    /// Grid spacing object (reference, safe since Grid outlives solver)
    const GridSpacing<T>& spacing() const {
        return spacing_;
    }

    /// Time domain information
    const TimeDomain& time() const {
        return time_;
    }

    /// Current solution (last time step)
    std::span<T> solution() {
        return std::span{solution_.data(), n_space()};
    }

    std::span<const T> solution() const {
        return std::span{solution_.data(), n_space()};
    }

    /// Previous solution (second-to-last time step)
    std::span<T> solution_prev() {
        return std::span{solution_.data() + n_space(), n_space()};
    }

    std::span<const T> solution_prev() const {
        return std::span{solution_.data() + n_space(), n_space()};
    }

    /// Number of spatial points
    size_t n_space() const {
        return grid_buffer_.size();
    }

    /// Time step size
    double dt() const {
        return time_.dt();
    }

private:
    // Private constructor (use factory method)
    Grid(GridBuffer<T>&& grid_buffer,
         GridSpacing<T>&& spacing,
         const TimeDomain& time,
         std::vector<T>&& solution)
        : grid_buffer_(std::move(grid_buffer))
        , spacing_(std::move(spacing))
        , time_(time)
        , solution_(std::move(solution))
    {}

    GridBuffer<T> grid_buffer_;     // Spatial grid points
    GridSpacing<T> spacing_;        // Grid spacing (uniform or non-uniform)
    TimeDomain time_;               // Time domain metadata
    std::vector<T> solution_;       // [u_current | u_prev] (2 × n_space)
};

} // namespace mango
