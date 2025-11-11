#pragma once

#include <vector>
#include <span>
#include <memory>
#include <cmath>
#include <stdexcept>
#include "src/support/expected.hpp"

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
    static expected<GridSpec, std::string> uniform(T x_min, T x_max, size_t n_points) {
        if (n_points < 2) {
            return unexpected<std::string>("Grid must have at least 2 points");
        }
        if (x_min >= x_max) {
            return unexpected<std::string>("x_min must be less than x_max");
        }
        return GridSpec(Type::Uniform, x_min, x_max, n_points);
    }

    static expected<GridSpec, std::string> log_spaced(T x_min, T x_max, size_t n_points) {
        if (n_points < 2) {
            return unexpected<std::string>("Grid must have at least 2 points");
        }
        if (x_min <= 0 || x_max <= 0) {
            return unexpected<std::string>("Log-spaced grid requires positive bounds");
        }
        if (x_min >= x_max) {
            return unexpected<std::string>("x_min must be less than x_max");
        }
        return GridSpec(Type::LogSpaced, x_min, x_max, n_points);
    }

    static expected<GridSpec, std::string> sinh_spaced(T x_min, T x_max, size_t n_points, T concentration = T(1.0)) {
        if (n_points < 2) {
            return unexpected<std::string>("Grid must have at least 2 points");
        }
        if (x_min >= x_max) {
            return unexpected<std::string>("x_min must be less than x_max");
        }
        if (concentration <= 0) {
            return unexpected<std::string>("Concentration parameter must be positive");
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

} // namespace mango
