// SPDX-License-Identifier: MIT
#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace mango {

/// Requested S/K interval. These are original ratio inputs, not exp(log(S/K))
/// reconstructions when a factory already knows the caller's ratio endpoints.
struct MoneynessBounds {
    double min = 0;
    double max = 0;
    [[nodiscard]] bool valid() const noexcept {
        return std::isfinite(min) && std::isfinite(max) && min > 0 && min <= max;
    }
};

/// Quote admission and a conservative real S/K enclosure for proof.
/// Retains rounded S=K*ratio endpoint semantics, while rejecting materially
/// unrelated ratios caused by underflow/overflow in that quote construction.
class MoneynessDomain {
  public:
    explicit MoneynessDomain(MoneynessBounds requested) noexcept
        : requested_(requested), enclosure_(requested), valid_(requested.valid()) {
        if (!valid_)
            return;
        for (int i = 0; i < 2; ++i) {
            enclosure_.min = std::nextafter(enclosure_.min, 0.0);
            enclosure_.max =
                std::nextafter(enclosure_.max, std::numeric_limits<double>::infinity());
        }
        if (enclosure_.min == 0)
            enclosure_.min = std::numeric_limits<double>::denorm_min();
        if (!std::isfinite(enclosure_.max))
            enclosure_.max = std::numeric_limits<double>::max();
    }
    [[nodiscard]] const MoneynessBounds &requested() const noexcept { return requested_; }
    [[nodiscard]] const MoneynessBounds &enclosure() const noexcept { return enclosure_; }
    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] bool contains_quote(double spot, double strike) const noexcept {
        return valid_ && std::isfinite(spot) && std::isfinite(strike) && spot > 0 && strike > 0 &&
               spot >= strike * requested_.min && spot <= strike * requested_.max &&
               compare_product(spot, strike, enclosure_.min) >= 0 &&
               compare_product(spot, strike, enclosure_.max) <= 0;
    }

  private:
    using Wide = unsigned __int128;
    struct Binary {
        std::uint64_t significand;
        int exponent;
    };
    static Binary binary(double value) noexcept {
        static_assert(sizeof(double) == 8 && std::numeric_limits<double>::digits == 53 &&
                      std::numeric_limits<double>::is_iec559);
        const auto bits = std::bit_cast<std::uint64_t>(value);
        const auto exponent = static_cast<int>((bits >> 52) & 0x7ff);
        auto significand = bits & ((std::uint64_t{1} << 52) - 1);
        if (exponent)
            significand |= std::uint64_t{1} << 52;
        return {significand, exponent ? exponent - 1023 - 52 : -1074};
    }
    static int width(Wide value) noexcept {
        const auto high = static_cast<std::uint64_t>(value >> 64);
        return high ? 64 + std::bit_width(high) : std::bit_width(static_cast<std::uint64_t>(value));
    }
    // Exact comparison of finite positive binary64 x and a*b. The product's
    // 106 significant bits fit in Wide; exponent comparison avoids overflow
    // or underflow and does not assume a particular hardware rounding mode.
    static int compare_product(double x, double a, double b) noexcept {
        const auto lhs = binary(x), aa = binary(a), bb = binary(b);
        const Wide product = Wide{aa.significand} * bb.significand;
        const int left_width = width(lhs.significand), right_width = width(product);
        const int left_order = lhs.exponent + left_width;
        const int right_order = aa.exponent + bb.exponent + right_width;
        if (left_order != right_order)
            return left_order < right_order ? -1 : 1;
        const Wide left = Wide{lhs.significand} << (128 - left_width);
        const Wide right = product << (128 - right_width);
        return left < right ? -1 : left > right ? 1 : 0;
    }
    MoneynessBounds requested_, enclosure_;
    bool valid_;
};
} // namespace mango
