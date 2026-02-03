// SPDX-License-Identifier: MIT
#include "mango/simple/timestamp.hpp"
#include <charconv>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace mango::simple {

std::expected<Timestamp::TimePoint, std::string> Timestamp::to_timepoint() const {
    return std::visit([](const auto& v) -> std::expected<TimePoint, std::string> {
        using T = std::decay_t<decltype(v)>;

        if constexpr (std::is_same_v<T, TimePoint>) {
            return v;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            // Nanoseconds since epoch
            auto duration = std::chrono::nanoseconds(v);
            return TimePoint{std::chrono::duration_cast<TimePoint::duration>(duration)};
        } else {
            // String value
            if (v.format == TimestampFormat::Compact) {
                return parse_compact(v.str);
            } else {
                return parse_iso(v.str);
            }
        }
    }, value_);
}

std::expected<Timestamp::TimePoint, std::string> Timestamp::parse_iso(const std::string& s) {
    std::tm tm = {};
    std::istringstream ss(s);

    // Try with time component first
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) {
        // Try date only
        ss.clear();
        ss.str(s);
        ss >> std::get_time(&tm, "%Y-%m-%d");
        if (ss.fail()) {
            return std::unexpected("Failed to parse ISO timestamp: " + s);
        }
    }

    // Use timegm for UTC time (non-standard but widely available)
    // or convert manually
    #ifdef _WIN32
    auto time = _mkgmtime(&tm);
    #else
    auto time = timegm(&tm);
    #endif

    if (time == -1) {
        return std::unexpected("Invalid timestamp: " + s);
    }

    return std::chrono::system_clock::from_time_t(time);
}

std::expected<Timestamp::TimePoint, std::string> Timestamp::parse_compact(const std::string& s) {
    if (s.length() != 8) {
        return std::unexpected("Compact format must be 8 digits: " + s);
    }

    int year, month, day;
    auto r1 = std::from_chars(s.data(), s.data() + 4, year);
    auto r2 = std::from_chars(s.data() + 4, s.data() + 6, month);
    auto r3 = std::from_chars(s.data() + 6, s.data() + 8, day);

    if (r1.ec != std::errc{} || r2.ec != std::errc{} || r3.ec != std::errc{}) {
        return std::unexpected("Failed to parse compact date: " + s);
    }

    std::tm tm = {};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;

    #ifdef _WIN32
    auto time = _mkgmtime(&tm);
    #else
    auto time = timegm(&tm);
    #endif

    if (time == -1) {
        return std::unexpected("Invalid date: " + s);
    }

    return std::chrono::system_clock::from_time_t(time);
}

std::string Timestamp::to_string() const {
    auto tp_result = to_timepoint();
    if (!tp_result) {
        return "<invalid>";
    }

    auto time = std::chrono::system_clock::to_time_t(*tp_result);
    std::tm tm;
#ifdef _WIN32
    gmtime_s(&tm, &time);  // Thread-safe Windows version
#else
    gmtime_r(&time, &tm);  // Thread-safe POSIX version
#endif

    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

double compute_tau(const Timestamp& valuation, const Timestamp& expiry) {
    auto val_tp = valuation.to_timepoint();
    auto exp_tp = expiry.to_timepoint();

    if (!val_tp || !exp_tp) {
        return 0.0;  // Error case
    }

    auto duration = *exp_tp - *val_tp;

    // Convert to seconds as floating point to avoid truncation
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    double hours = static_cast<double>(seconds) / 3600.0;

    // Calendar time: hours / (365 * 24)
    return hours / (365.0 * 24.0);
}

}  // namespace mango::simple
