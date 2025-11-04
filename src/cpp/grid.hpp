#pragma once

#include <vector>
#include <span>
#include <memory>
#include <cmath>
#include <stdexcept>

namespace mango {

// Forward declarations
template<typename T = double>
class GridBuffer;

template<typename T = double>
class GridView;

/**
 * GridSpec: Immutable grid specification (how to generate a grid)
 *
 * This is a value type that describes grid generation parameters.
 * It doesn't own data - call generate() to create a GridBuffer.
 */
template<typename T = double>
class GridSpec {
public:
    enum class Type {
        Uniform,      // Equally spaced points
        LogSpaced,    // Logarithmically spaced
        SinhSpaced    // Hyperbolic sine spacing (concentrates points at center)
    };

    // Factory methods for common grid types
    static GridSpec uniform(T x_min, T x_max, size_t n_points) {
        if (n_points < 2) {
            throw std::invalid_argument("Grid must have at least 2 points");
        }
        if (x_min >= x_max) {
            throw std::invalid_argument("x_min must be less than x_max");
        }
        return GridSpec(Type::Uniform, x_min, x_max, n_points);
    }

    static GridSpec log_spaced(T x_min, T x_max, size_t n_points) {
        if (n_points < 2) {
            throw std::invalid_argument("Grid must have at least 2 points");
        }
        if (x_min <= 0 || x_max <= 0) {
            throw std::invalid_argument("Log-spaced grid requires positive bounds");
        }
        if (x_min >= x_max) {
            throw std::invalid_argument("x_min must be less than x_max");
        }
        return GridSpec(Type::LogSpaced, x_min, x_max, n_points);
    }

    static GridSpec sinh_spaced(T x_min, T x_max, size_t n_points, T concentration = T(1.0)) {
        if (n_points < 2) {
            throw std::invalid_argument("Grid must have at least 2 points");
        }
        if (x_min >= x_max) {
            throw std::invalid_argument("x_min must be less than x_max");
        }
        if (concentration <= 0) {
            throw std::invalid_argument("Concentration parameter must be positive");
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

} // namespace mango
