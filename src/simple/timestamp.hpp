// SPDX-License-Identifier: MIT
/**
 * @file timestamp.hpp
 * @brief Timestamp type with multiple format support
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <variant>

namespace mango::simple {

/// Timestamp storage format
enum class TimestampFormat {
    ISO,          // "2024-06-21" or "2024-06-21T10:30:00"
    Compact,      // "20240621"
    Nanoseconds   // uint64_t nanoseconds since epoch
};

/// Timestamp with multiple format support
///
/// Stores timestamps in original format, converts on demand.
class Timestamp {
public:
    using TimePoint = std::chrono::system_clock::time_point;

    /// Construct from ISO date string (auto-detect format)
    explicit Timestamp(std::string value, TimestampFormat format = TimestampFormat::ISO)
        : value_(StringValue{std::move(value), format}) {}

    /// Construct from nanoseconds since epoch
    explicit Timestamp(uint64_t nanos)
        : value_(nanos) {}

    /// Construct from time_point
    explicit Timestamp(TimePoint tp)
        : value_(tp) {}

    /// Get current time
    static Timestamp now() {
        return Timestamp{std::chrono::system_clock::now()};
    }

    /// Convert to time_point
    [[nodiscard]] std::expected<TimePoint, std::string> to_timepoint() const;

    /// Convert to string for display
    [[nodiscard]] std::string to_string() const;

private:
    struct StringValue {
        std::string str;
        TimestampFormat format;
    };

    std::variant<StringValue, uint64_t, TimePoint> value_;

    // Parse helpers
    static std::expected<TimePoint, std::string> parse_iso(const std::string& s);
    static std::expected<TimePoint, std::string> parse_compact(const std::string& s);
};

/// Compute time to expiry in years (calendar time basis)
///
/// Uses calendar time (365 * 24 hours) for consistency with
/// market-quoted implied volatilities.
///
/// @param valuation Current valuation time
/// @param expiry Option expiry time
/// @return Time to expiry in years
double compute_tau(const Timestamp& valuation, const Timestamp& expiry);

}  // namespace mango::simple
