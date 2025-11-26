/**
 * @file price.hpp
 * @brief Price type with deferred double conversion
 */

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

namespace mango::simple {

/// Price storage format
enum class PriceFormat {
    Double,       // Native double
    FixedPoint9   // Databento: price * 10^9
};

/// Price with deferred conversion
///
/// Stores prices in their original format to preserve precision.
/// Conversion to double happens only when needed (at solver boundary).
class Price {
public:
    /// Construct from double
    explicit Price(double value) : value_(value) {}

    /// Construct from fixed-point
    Price(int64_t value, PriceFormat format) {
        if (format == PriceFormat::FixedPoint9) {
            value_ = FixedPoint9{value};
        } else {
            value_ = static_cast<double>(value);
        }
    }

    /// Convert to double (deferred conversion point)
    [[nodiscard]] double to_double() const {
        return std::visit([](const auto& v) -> double {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, double>) {
                return v;
            } else {
                return static_cast<double>(v.value) * 1e-9;
            }
        }, value_);
    }

    /// Compute midpoint of two prices
    ///
    /// If both prices are same format, preserves that format.
    /// Otherwise, converts to double.
    static std::optional<Price> midpoint(const Price& a, const Price& b) {
        // Check if both are same fixed-point format
        if (auto* fa = std::get_if<FixedPoint9>(&a.value_)) {
            if (auto* fb = std::get_if<FixedPoint9>(&b.value_)) {
                // Average in fixed-point to preserve precision
                int64_t mid = (fa->value + fb->value) / 2;
                return Price{mid, PriceFormat::FixedPoint9};
            }
        }
        // Fall back to double
        return Price{(a.to_double() + b.to_double()) / 2.0};
    }

    /// Check if stored as fixed-point
    [[nodiscard]] bool is_fixed_point() const {
        return std::holds_alternative<FixedPoint9>(value_);
    }

private:
    struct FixedPoint9 {
        int64_t value;
    };

    std::variant<double, FixedPoint9> value_;
};

}  // namespace mango::simple
