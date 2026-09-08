// SPDX-License-Identifier: MIT
#pragma once
#include <mpfr.h>

namespace mango::detail::proof {

/// Private proof arithmetic. Binary64 constants are imported exactly; proof
/// endpoints remain at 128 bits through all sign decisions. This is an
/// inclusion interval, not a market/model uncertainty estimate.
class Interval {
  public:
    static constexpr mpfr_prec_t precision = 128;
    explicit Interval(double value = 0.0);
    Interval(const Interval &other);
    Interval(Interval &&other) noexcept;
    Interval &operator=(const Interval &other);
    Interval &operator=(Interval &&other) noexcept;
    ~Interval();

    [[nodiscard]] static Interval hull(double lower, double upper);
    [[nodiscard]] bool finite() const;
    [[nodiscard]] bool exact_zero() const;
    [[nodiscard]] bool nonnegative() const;
    [[nodiscard]] bool nonpositive() const;
    /// Retained MPFR endpoint values for partitioning enclosing domains.
    [[nodiscard]] Interval lower_endpoint() const;
    [[nodiscard]] Interval upper_endpoint() const;
    [[nodiscard]] static Interval pi();
    [[nodiscard]] bool strictly_negative() const;
    /// Directed binary64 conversion for diagnostics only, never sign decisions.
    [[nodiscard]] double lower_bound() const;
    [[nodiscard]] double upper_bound() const;
    [[nodiscard]] bool strictly_positive() const;
    [[nodiscard]] bool contains(double value) const;
    friend Interval hull(const Interval &a, const Interval &b);
    friend Interval operator*(const Interval &a, const Interval &b);
    friend Interval operator/(const Interval &a, const Interval &b);
    friend Interval positive_part(const Interval &value);
    friend Interval square(const Interval &value);
    friend Interval erfc(const Interval &value);
    friend Interval intersection(const Interval &a, const Interval &b);
    friend Interval acos(const Interval &value);
    friend Interval cos(const Interval &value);
    friend Interval exp(const Interval &value);
    friend Interval log(const Interval &value);
    friend Interval sqrt(const Interval &value);
    friend Interval operator+(const Interval &a, const Interval &b);
    friend Interval operator-(const Interval &a, const Interval &b);

  private:
    static Interval invalid();
    mpfr_t lower_, upper_;
};
} // namespace mango::detail::proof
