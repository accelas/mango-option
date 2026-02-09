// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cmath>
#include <algorithm>

namespace mango {

/// C∞ bump function: ψ(t) = exp(-1 / (1 - (2t-1)²)) for |2t-1| < 1, else 0.
/// Normalized CDF: Ψ(t) = ∫₀ᵗ ψ(s) ds / ∫₀¹ ψ(s) ds.
/// Returns the right-side weight w_right ∈ [0, 1].
/// At t=0: returns 0 (pure left).  At t=1: returns 1 (pure right).
inline double bump_blend_weight(double t) {
    t = std::clamp(t, 0.0, 1.0);

    constexpr int N = 256;
    static const auto table = [] {
        auto psi = [](double s) -> double {
            double u = 2.0 * s - 1.0;
            double u2 = u * u;
            if (u2 >= 1.0) return 0.0;
            return std::exp(-1.0 / (1.0 - u2));
        };

        std::array<double, N + 1> cdf{};
        cdf[0] = 0.0;
        double h = 1.0 / N;
        for (int i = 0; i < N; ++i) {
            double a = i * h;
            double b = (i + 1) * h;
            double mid = (a + b) / 2.0;
            cdf[i + 1] = cdf[i] + (h / 6.0) * (psi(a) + 4.0 * psi(mid) + psi(b));
        }
        double total = cdf[N];
        for (auto& v : cdf) v /= total;
        return cdf;
    }();

    double idx = t * N;
    int lo = static_cast<int>(idx);
    lo = std::clamp(lo, 0, N - 1);
    double frac = idx - lo;
    return table[lo] * (1.0 - frac) + table[lo + 1] * frac;
}

/// Convenience: given x in overlap zone [a, b], return the right-side weight.
/// Outside [a, b]: clamps to 0 or 1.
inline double overlap_weight_right(double x, double a, double b) {
    if (b <= a) return 0.5;
    double t = (x - a) / (b - a);
    return bump_blend_weight(t);
}

}  // namespace mango
