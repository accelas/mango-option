// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <span>
#include <memory>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <expected>
#include <variant>
#include <optional>
#include <algorithm>
#include <format>
#include <experimental/mdspan>
#include "src/support/aligned_allocator.hpp"
#include "src/support/error_types.hpp"
#include "src/pde/core/time_domain.hpp"

namespace mango {

/// Multi-sinh cluster: specifies a concentration region in composite grids
///
/// Used to concentrate grid points at multiple locations (e.g., ATM and deep ITM)
/// while still using a single shared PDE grid for batch solving.
template<typename T = double>
struct MultiSinhCluster {
    T center_x;   ///< Log-moneyness center for this cluster
    T alpha;      ///< Concentration strength (must be > 0)
    T weight;     ///< Relative contribution (must be > 0)
};

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
        Uniform,         // Equally spaced points
        LogSpaced,       // Logarithmically spaced
        SinhSpaced,      // Hyperbolic sine spacing (concentrates points at center)
        MultiSinhSpaced  // Composite multi-sinh (multiple concentration regions)
    };

    // Factory methods for common grid types
    static std::expected<GridSpec, ValidationError> uniform(T x_min, T x_max, size_t n_points) {
        if (n_points < 2) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidGridSize,
                static_cast<double>(n_points)));
        }
        if (x_min >= x_max) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidBounds,
                static_cast<double>(x_min)));
        }
        return GridSpec(Type::Uniform, x_min, x_max, n_points);
    }

    static std::expected<GridSpec, ValidationError> log_spaced(T x_min, T x_max, size_t n_points) {
        if (n_points < 2) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidGridSize,
                static_cast<double>(n_points)));
        }
        if (x_min <= 0 || x_max <= 0) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidBounds,
                static_cast<double>(x_min)));
        }
        if (x_min >= x_max) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidBounds,
                static_cast<double>(x_min)));
        }
        return GridSpec(Type::LogSpaced, x_min, x_max, n_points);
    }

    static std::expected<GridSpec, ValidationError> sinh_spaced(T x_min, T x_max, size_t n_points, T concentration = T(1.0)) {
        if (n_points < 2) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidGridSize,
                static_cast<double>(n_points)));
        }
        if (x_min >= x_max) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidBounds,
                static_cast<double>(x_min)));
        }
        if (concentration <= 0) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidGridSpacing,
                static_cast<double>(concentration)));
        }
        return GridSpec(Type::SinhSpaced, x_min, x_max, n_points, concentration);
    }

    static std::expected<GridSpec, ValidationError> multi_sinh_spaced(
        T x_min, T x_max, size_t n_points,
        std::vector<MultiSinhCluster<T>> clusters,
        bool auto_merge = true) {

        if (n_points < 2) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidGridSize,
                static_cast<double>(n_points)));
        }
        if (x_min >= x_max) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidBounds,
                static_cast<double>(x_min)));
        }
        if (clusters.empty()) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::InvalidGridSize,
                0.0));
        }

        // Validate each cluster
        for (size_t i = 0; i < clusters.size(); ++i) {
            if (clusters[i].alpha <= 0) {
                return std::unexpected(ValidationError(
                    ValidationErrorCode::InvalidGridSpacing,
                    static_cast<double>(clusters[i].alpha),
                    i));
            }
            if (clusters[i].weight <= 0) {
                return std::unexpected(ValidationError(
                    ValidationErrorCode::InvalidGridSpacing,
                    static_cast<double>(clusters[i].weight),
                    i));
            }
            if (clusters[i].center_x < x_min || clusters[i].center_x > x_max) {
                return std::unexpected(ValidationError(
                    ValidationErrorCode::OutOfRange,
                    static_cast<double>(clusters[i].center_x),
                    i));
            }
        }

        // Merge nearby clusters to prevent wasted resolution (unless bypassed)
        if (auto_merge) {
            merge_nearby_clusters(clusters);
        }

        // No automatic recentering - preserve the merged weighted-average position
        // Auto-merge only deduplicates overlapping centers, it doesn't recenter them

        return GridSpec(Type::MultiSinhSpaced, x_min, x_max, n_points,
                        T(1.0), std::move(clusters));
    }

    // Generate the actual grid
    GridBuffer<T> generate() const;

    // Accessors
    Type type() const { return type_; }
    T x_min() const { return x_min_; }
    T x_max() const { return x_max_; }
    size_t n_points() const { return n_points_; }
    T concentration() const { return concentration_; }
    std::span<const MultiSinhCluster<T>> clusters() const { return clusters_; }

private:
    GridSpec(Type type, T x_min, T x_max, size_t n_points, T concentration = T(1.0),
             std::vector<MultiSinhCluster<T>> clusters = {})
        : type_(type), x_min_(x_min), x_max_(x_max),
          n_points_(n_points), concentration_(concentration),
          clusters_(std::move(clusters)) {}

    /// Merge nearby clusters to prevent wasted resolution
    ///
    /// When clusters are too close (Δx < 0.3/α_avg), merges them to avoid
    /// overlapping concentration regions that waste grid resolution.
    ///
    /// @param clusters Input cluster list (will be modified in-place)
    static void merge_nearby_clusters(std::vector<MultiSinhCluster<T>>& clusters) {
        if (clusters.size() <= 1) {
            return;  // Nothing to merge
        }

        // Keep merging until no more merges are possible
        bool merged = true;
        while (merged) {
            merged = false;

            // Scan all pairs, restart after each merge (avoids iterator invalidation)
            for (size_t i = 0; i < clusters.size() && !merged; ++i) {
                for (size_t j = i + 1; j < clusters.size() && !merged; ++j) {
                    const T delta_x = std::abs(clusters[i].center_x - clusters[j].center_x);
                    const T alpha_avg = (clusters[i].alpha + clusters[j].alpha) / T(2.0);
                    const T threshold = T(0.3) / alpha_avg;

                    // Use slightly permissive comparison to handle floating point edge cases
                    const T epsilon = T(1e-10);
                    if (delta_x <= threshold + epsilon) {
                        // Merge clusters i and j into a new cluster
                        const T total_weight = clusters[i].weight + clusters[j].weight;
                        const T w_i = clusters[i].weight / total_weight;
                        const T w_j = clusters[j].weight / total_weight;

                        MultiSinhCluster<T> merged_cluster{
                            .center_x = w_i * clusters[i].center_x + w_j * clusters[j].center_x,
                            .alpha = w_i * clusters[i].alpha + w_j * clusters[j].alpha,
                            .weight = total_weight
                        };

                        // Replace cluster i with merged result, remove cluster j
                        clusters[i] = merged_cluster;
                        clusters.erase(clusters.begin() + j);

                        // Set flag and break to restart outer loop (avoids iterator invalidation)
                        merged = true;
                        break;
                    }
                }
            }
        }
    }

    /// Enforce strict monotonicity in grid points
    ///
    /// Ensures x[i+1] > x[i] for all i, while preserving endpoints.
    /// Uses iterative smoothing to fix non-monotonic regions.
    static void enforce_monotonicity(std::vector<T>& points, T x_min, T x_max) {
        const size_t n = points.size();
        if (n < 2) return;

        // Clamp endpoints
        points[0] = x_min;
        points[n-1] = x_max;

        // Iterative monotonicity enforcement (max 100 passes)
        for (int pass = 0; pass < 100; ++pass) {
            bool modified = false;

            for (size_t i = 1; i < n; ++i) {
                if (points[i] <= points[i-1]) {
                    // Fix violation: interpolate between neighbors
                    T right = (i < n-1) ? points[i+1] : x_max;
                    points[i] = (points[i-1] + right) / T(2.0);
                    modified = true;
                }
            }

            if (!modified) break;
        }

        // Final pass: ensure minimum spacing (avoid dx → 0)
        const T min_spacing = (x_max - x_min) / static_cast<T>(n * 100);

        // Clamp endpoints
        points[0] = x_min;
        points[n-1] = x_max;

        // Backward pass: ensure no point exceeds the next point minus min_spacing
        for (size_t i = n - 1; i > 1; --i) {
            if (points[i-1] >= points[i] - min_spacing) {
                points[i-1] = points[i] - min_spacing;
            }
        }

        // Forward pass: ensure no point is less than previous point plus min_spacing
        for (size_t i = 1; i < n - 1; ++i) {
            if (points[i] <= points[i-1] + min_spacing) {
                points[i] = points[i-1] + min_spacing;
            }
        }

        // Final clamp of endpoints
        points[0] = x_min;
        points[n-1] = x_max;
    }

    Type type_;
    T x_min_;
    T x_max_;
    size_t n_points_;
    T concentration_;  // Only used for sinh spacing
    std::vector<MultiSinhCluster<T>> clusters_;  // Empty for non-composite grids
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

        case Type::MultiSinhSpaced: {
            // Handle single cluster as special case (most common)
            // Note: Single clusters may be off-center after auto-merging
            if (clusters_.size() == 1) {
                const auto& cluster = clusters_[0];
                const T c = cluster.alpha;
                const T center = cluster.center_x;
                const T range = x_max_ - x_min_;
                const T sinh_half_c = std::sinh(c / T(2.0));

                // Compute normalized center position
                const T eta_center = (center - x_min_) / range;

                // Check if cluster is centered (eta_center ≈ 0.5)
                const bool is_centered = std::abs(eta_center - T(0.5)) < T(1e-10);

                if (is_centered) {
                    // Centered cluster: use standard sinh formula (guaranteed in-bounds)
                    for (size_t i = 0; i < n_points_; ++i) {
                        const T eta = static_cast<T>(i) / static_cast<T>(n_points_ - 1);
                        const T sinh_term = std::sinh(c * (eta - T(0.5))) / sinh_half_c;
                        const T normalized = (T(1.0) + sinh_term) / T(2.0);
                        points.push_back(x_min_ + range * normalized);
                    }
                } else {
                    // Off-center cluster: use generalized formula + monotonicity enforcement
                    const T offset = center - (x_min_ + x_max_) / T(2.0);
                    std::vector<T> raw_points(n_points_);
                    for (size_t i = 0; i < n_points_; ++i) {
                        const T eta = static_cast<T>(i) / static_cast<T>(n_points_ - 1);
                        const T sinh_term = std::sinh(c * (eta - eta_center)) / sinh_half_c;
                        const T normalized = (T(1.0) + sinh_term) / T(2.0);
                        raw_points[i] = x_min_ + range * normalized + offset;
                    }

                    // Enforce monotonicity and bounds
                    enforce_monotonicity(raw_points, x_min_, x_max_);

                    // Transfer to output
                    for (const auto& x : raw_points) {
                        points.push_back(x);
                    }
                }
            } else {
                // Multi-cluster: combine weighted sinh transforms
                std::vector<T> raw_points(n_points_);

                // Normalize weights
                T total_weight = T(0);
                for (const auto& cluster : clusters_) {
                    total_weight += cluster.weight;
                }

                const T range = x_max_ - x_min_;

                for (size_t i = 0; i < n_points_; ++i) {
                    // Map i to eta ∈ [0, 1]
                    const T eta = static_cast<T>(i) / static_cast<T>(n_points_ - 1);

                    // Weighted combination of sinh transforms
                    T weighted_x = T(0);
                    for (const auto& cluster : clusters_) {
                        const T c = cluster.alpha;
                        const T center = cluster.center_x;
                        const T w = cluster.weight / total_weight;
                        const T sinh_half_c = std::sinh(c / T(2.0));

                        // Compute normalized center position for this cluster
                        const T eta_center = (center - x_min_) / range;

                        // Apply sinh transform centered at eta_center
                        const T sinh_term = std::sinh(c * (eta - eta_center)) / sinh_half_c;
                        const T normalized = (T(1.0) + sinh_term) / T(2.0);

                        // Transform to [x_min, x_max] with offset to place peak at center_x
                        const T offset = center - (x_min_ + x_max_) / T(2.0);
                        const T x_i = x_min_ + range * normalized + offset;

                        weighted_x += w * x_i;
                    }

                    raw_points[i] = weighted_x;
                }

                // Enforce monotonicity with smoothing pass
                enforce_monotonicity(raw_points, x_min_, x_max_);

                // Transfer to output
                points = std::move(raw_points);
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

    /// 2D view showing 5-section structure
    using SectionView = std::experimental::mdspan<
        T,
        std::experimental::dextents<size_t, 2>,
        std::experimental::layout_right
    >;
    SectionView sections_view_;  // Shape: (5, interior)

    /// Construct from non-uniform grid points
    ///
    /// @param x Grid points (must be sorted, size >= 3)
    explicit NonUniformSpacing(std::span<const T> x)
        : n(x.size())
        , sections_view_(nullptr, std::experimental::dextents<size_t, 2>{0, 0})
    {
        const size_t interior = n - 2;
        precomputed.resize(5 * interior);

        // Create 2D view: 5 sections × interior points
        sections_view_ = SectionView(precomputed.data(), 5, interior);

        // Fill sections using self-documenting 2D indexing
        for (size_t i = 1; i <= n - 2; ++i) {
            const T dx_left = x[i] - x[i-1];
            const T dx_right = x[i+1] - x[i];
            const T dx_center = T(0.5) * (dx_left + dx_right);

            const size_t idx = i - 1;

            // Clear, self-documenting section assignments:
            sections_view_[0, idx] = T(1) / dx_left;              // dx_left_inv
            sections_view_[1, idx] = T(1) / dx_right;             // dx_right_inv
            sections_view_[2, idx] = T(1) / dx_center;            // dx_center_inv
            sections_view_[3, idx] = dx_right / (dx_left + dx_right);  // w_left
            sections_view_[4, idx] = dx_left / (dx_left + dx_right);   // w_right
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

namespace {

/// Convert snapshot times to state indices with validation and deduplication
///
/// @param times Requested snapshot times (may be off-grid)
/// @param time_domain Time domain specification
/// @return Pair of (sorted unique state indices, snapped times) or error
inline std::expected<std::pair<std::vector<size_t>, std::vector<double>>, ValidationError>
convert_times_to_indices(std::span<const double> times,
                         const TimeDomain& time_domain) {
    const double t_start = time_domain.t_start();
    const double t_end = time_domain.t_end();
    const size_t n_steps = time_domain.n_steps();
    const double dt = time_domain.dt();

    // Validate preconditions
    if (n_steps == 0) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidGridSize,
            0.0));
    }
    if (dt <= 0.0) {
        return std::unexpected(ValidationError(
            ValidationErrorCode::InvalidGridSpacing,
            dt));
    }

    std::vector<size_t> indices;
    std::vector<double> snapped_times;
    indices.reserve(times.size());
    snapped_times.reserve(times.size());

    for (double t : times) {
        // Validate range
        if (t < t_start || t > t_end) {
            return std::unexpected(ValidationError(
                ValidationErrorCode::OutOfRange,
                t));
        }

        size_t state_idx;
        double snapped_t;

        if (time_domain.has_time_points()) {
            // Non-uniform: binary search through stored time points
            const auto& pts = time_domain.time_points_ref();
            auto it = std::lower_bound(pts.begin(), pts.end(), t - 1e-10);
            size_t idx = static_cast<size_t>(std::distance(pts.begin(), it));
            // Find nearest point
            if (idx > 0 && (idx == pts.size() ||
                std::abs(pts[idx-1] - t) < std::abs(pts[idx] - t))) {
                idx--;
            }
            state_idx = std::min(idx, n_steps);
            snapped_t = pts[state_idx];
        } else {
            // Uniform: existing arithmetic path
            double step_exact = (t - t_start) / dt;
            state_idx = static_cast<size_t>(std::floor(step_exact + 0.5));
            state_idx = std::min(state_idx, n_steps);
            snapped_t = t_start + state_idx * dt;
        }

        indices.push_back(state_idx);
        snapped_times.push_back(snapped_t);
    }

    // Sort and deduplicate
    std::vector<std::pair<size_t, double>> paired;
    for (size_t i = 0; i < indices.size(); ++i) {
        paired.push_back({indices[i], snapped_times[i]});
    }
    std::sort(paired.begin(), paired.end());
    paired.erase(std::unique(paired.begin(), paired.end()), paired.end());

    indices.clear();
    snapped_times.clear();
    for (const auto& [idx, time] : paired) {
        indices.push_back(idx);
        snapped_times.push_back(time);
    }

    return std::make_pair(indices, snapped_times);
}

}  // namespace

/// Grid with persistent solution storage and metadata
/// Outlives PDESolver, passed via shared_ptr for lifetime management
template<typename T = double>
class Grid {
public:
    /// Create grid with solution storage
    /// @param grid_spec Grid specification (uniform, sinh, etc)
    /// @param time_domain Time domain information
    /// @param snapshot_times Optional times to record snapshots
    /// @return Grid instance or error message
    static std::expected<std::shared_ptr<Grid<T>>, ValidationError>
    create(const GridSpec<T>& grid_spec, const TimeDomain& time_domain,
           std::span<const double> snapshot_times = {}) {
        // Generate grid buffer
        auto grid_buffer = grid_spec.generate();
        auto grid_view = grid_buffer.view();

        // Create GridSpacing
        auto spacing = GridSpacing<T>(grid_view);

        size_t n = grid_view.size();

        // Allocate solution storage (2 × n for current + previous), 64-byte aligned
        AlignedVector<T> solution(2 * n);

        // Create instance (private constructor, so use new)
        auto grid = std::shared_ptr<Grid<T>>(
            new Grid<T>(
                std::move(grid_buffer),
                std::move(spacing),
                time_domain,
                std::move(solution)
            )
        );

        // If snapshot times provided, convert to indices and initialize storage
        if (!snapshot_times.empty()) {
            auto conversion = convert_times_to_indices(snapshot_times, time_domain);
            if (!conversion.has_value()) {
                return std::unexpected(conversion.error());
            }

            auto [indices, times] = conversion.value();
            grid->snapshot_indices_ = std::move(indices);
            grid->snapshot_times_ = std::move(times);

            // Allocate snapshot storage
            size_t total_size = grid->snapshot_indices_.size() * n;
            grid->surface_history_ = std::vector<T>(total_size);
        }

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

    // Snapshot query API
    bool has_snapshots() const {
        return surface_history_.has_value();
    }

    size_t num_snapshots() const {
        return snapshot_indices_.size();
    }

    std::span<const T> at(size_t snapshot_idx) const {
        if (!surface_history_.has_value() || snapshot_idx >= num_snapshots()) {
            return {};
        }
        size_t offset = snapshot_idx * n_space();
        return std::span<const T>(surface_history_->data() + offset, n_space());
    }

    std::span<const double> snapshot_times() const {
        return snapshot_times_;
    }

    // Recording API (for PDESolver)
    bool should_record(size_t state_idx) const {
        return find_snapshot_index(state_idx).has_value();
    }

    void record(size_t state_idx, std::span<const T> solution) {
        auto snap_idx = find_snapshot_index(state_idx);
        if (!snap_idx.has_value() || !surface_history_.has_value()) {
            return;  // Silently ignore if not a snapshot state
        }

        // Copy solution to snapshot storage
        size_t offset = snap_idx.value() * n_space();
        std::copy(solution.begin(), solution.end(),
                  surface_history_->begin() + offset);
    }

private:
    // Private constructor (use factory method)
    Grid(GridBuffer<T>&& grid_buffer,
         GridSpacing<T>&& spacing,
         const TimeDomain& time,
         AlignedVector<T>&& solution)
        : grid_buffer_(std::move(grid_buffer))
        , spacing_(std::move(spacing))
        , time_(time)
        , solution_(std::move(solution))
    {}

    // Helper: Find snapshot index for state index
    std::optional<size_t> find_snapshot_index(size_t state_idx) const {
        auto it = std::lower_bound(snapshot_indices_.begin(),
                                   snapshot_indices_.end(),
                                   state_idx);
        if (it != snapshot_indices_.end() && *it == state_idx) {
            return std::distance(snapshot_indices_.begin(), it);
        }
        return std::nullopt;
    }

    GridBuffer<T> grid_buffer_;     // Spatial grid points
    GridSpacing<T> spacing_;        // Grid spacing (uniform or non-uniform)
    TimeDomain time_;               // Time domain metadata
    AlignedVector<T> solution_;     // [u_current | u_prev] (2 × n_space), 64-byte aligned

    // Snapshot storage
    std::vector<size_t> snapshot_indices_;       // State indices to record
    std::vector<double> snapshot_times_;         // Actual times (after snapping)
    std::optional<std::vector<T>> surface_history_;  // Snapshots (row-major)
};

} // namespace mango
