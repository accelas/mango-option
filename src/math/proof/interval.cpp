// SPDX-License-Identifier: MIT
#include "mango/math/proof/interval.hpp"
#include <cmath>
#include <initializer_list>
#include <limits>

namespace mango::detail::proof {
Interval::Interval(double value) {
    mpfr_init2(lower_, precision);
    mpfr_init2(upper_, precision);
    // An application can change MPFR's exponent range. A rounded import
    // would certify a different stored constant, so fail closed instead.
    const int lower_inexact = mpfr_set_d(lower_, value, MPFR_RNDN);
    const int upper_inexact = mpfr_set_d(upper_, value, MPFR_RNDN);
    if (lower_inexact || upper_inexact) {
        mpfr_set_nan(lower_);
        mpfr_set_nan(upper_);
    }
}
Interval::Interval(const Interval &other) : Interval() {
    mpfr_set(lower_, other.lower_, MPFR_RNDD);
    mpfr_set(upper_, other.upper_, MPFR_RNDU);
}
Interval::Interval(Interval &&other) noexcept : Interval() {
    mpfr_swap(lower_, other.lower_);
    mpfr_swap(upper_, other.upper_);
}
Interval &Interval::operator=(const Interval &other) {
    if (this != &other) {
        mpfr_set(lower_, other.lower_, MPFR_RNDD);
        mpfr_set(upper_, other.upper_, MPFR_RNDU);
    }
    return *this;
}
Interval &Interval::operator=(Interval &&other) noexcept {
    mpfr_swap(lower_, other.lower_);
    mpfr_swap(upper_, other.upper_);
    return *this;
}
Interval::~Interval() {
    mpfr_clear(lower_);
    mpfr_clear(upper_);
}
bool Interval::finite() const { return mpfr_number_p(lower_) && mpfr_number_p(upper_); }
bool Interval::strictly_positive() const { return finite() && mpfr_sgn(lower_) > 0; }
bool Interval::contains(double value) const {
    return finite() && std::isfinite(value) && mpfr_cmp_d(lower_, value) <= 0 &&
           mpfr_cmp_d(upper_, value) >= 0;
}
Interval operator+(const Interval &a, const Interval &b) {
    if (!a.finite() || !b.finite())
        return Interval::invalid();
    Interval result;
    mpfr_add(result.lower_, a.lower_, b.lower_, MPFR_RNDD);
    mpfr_add(result.upper_, a.upper_, b.upper_, MPFR_RNDU);
    return result;
}
Interval operator-(const Interval &a, const Interval &b) {
    if (!a.finite() || !b.finite())
        return Interval::invalid();
    Interval result;
    mpfr_sub(result.lower_, a.lower_, b.upper_, MPFR_RNDD);
    mpfr_sub(result.upper_, a.upper_, b.lower_, MPFR_RNDU);
    return result;
}

Interval Interval::invalid() { return Interval(std::numeric_limits<double>::quiet_NaN()); }
Interval Interval::hull(double lower, double upper) {
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper)
        return invalid();
    Interval result(lower);
    if (mpfr_set_d(result.upper_, upper, MPFR_RNDN))
        return invalid();
    return result;
}
bool Interval::exact_zero() const { return finite() && mpfr_zero_p(lower_) && mpfr_zero_p(upper_); }
bool Interval::nonnegative() const { return finite() && mpfr_sgn(lower_) >= 0; }
bool Interval::strictly_negative() const { return finite() && mpfr_sgn(upper_) < 0; }
double Interval::lower_bound() const { return mpfr_get_d(lower_, MPFR_RNDD); }
double Interval::upper_bound() const { return mpfr_get_d(upper_, MPFR_RNDU); }
Interval hull(const Interval &a, const Interval &b) {
    if (!a.finite() || !b.finite())
        return Interval::invalid();
    Interval result;
    mpfr_min(result.lower_, a.lower_, b.lower_, MPFR_RNDD);
    mpfr_max(result.upper_, a.upper_, b.upper_, MPFR_RNDU);
    return result;
}
Interval operator*(const Interval &a, const Interval &b) {
    if (!a.finite() || !b.finite())
        return Interval::invalid();
    Interval result, term;
    mpfr_set_inf(result.lower_, 1);
    mpfr_set_inf(result.upper_, -1);
    for (auto x : {a.lower_, a.upper_}) {
        for (auto y : {b.lower_, b.upper_}) {
            mpfr_mul(term.lower_, x, y, MPFR_RNDD);
            mpfr_mul(term.upper_, x, y, MPFR_RNDU);
            mpfr_min(result.lower_, result.lower_, term.lower_, MPFR_RNDD);
            mpfr_max(result.upper_, result.upper_, term.upper_, MPFR_RNDU);
        }
    }
    return result;
}
Interval operator/(const Interval &a, const Interval &b) {
    if (!a.finite() || !b.finite() || b.contains(0))
        return Interval::invalid();
    Interval result, term;
    mpfr_set_inf(result.lower_, 1);
    mpfr_set_inf(result.upper_, -1);
    for (auto x : {a.lower_, a.upper_}) {
        for (auto y : {b.lower_, b.upper_}) {
            mpfr_div(term.lower_, x, y, MPFR_RNDD);
            mpfr_div(term.upper_, x, y, MPFR_RNDU);
            mpfr_min(result.lower_, result.lower_, term.lower_, MPFR_RNDD);
            mpfr_max(result.upper_, result.upper_, term.upper_, MPFR_RNDU);
        }
    }
    return result;
}
Interval exp(const Interval &value) {
    if (!value.finite())
        return Interval::invalid();
    Interval result;
    mpfr_exp(result.lower_, value.lower_, MPFR_RNDD);
    mpfr_exp(result.upper_, value.upper_, MPFR_RNDU);
    return result;
}
Interval log(const Interval &value) {
    if (!value.strictly_positive())
        return Interval::invalid();
    Interval result;
    mpfr_log(result.lower_, value.lower_, MPFR_RNDD);
    mpfr_log(result.upper_, value.upper_, MPFR_RNDU);
    return result;
}
Interval sqrt(const Interval &value) {
    if (!value.nonnegative())
        return Interval::invalid();
    Interval result;
    mpfr_sqrt(result.lower_, value.lower_, MPFR_RNDD);
    mpfr_sqrt(result.upper_, value.upper_, MPFR_RNDU);
    return result;
}
} // namespace mango::detail::proof
