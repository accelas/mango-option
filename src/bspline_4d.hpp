/**
 * @file bspline_4d.hpp
 * @brief 4D tensor-product B-spline interpolation with FMA optimization
 *
 * Implements efficient 4D B-spline evaluation for option price surfaces.
 * Uses clamped cubic B-splines with Cox-de Boor recursion and fused
 * multiply-add (FMA) instructions for performance.
 *
 * Key features:
 * - Clamped knot vectors (interpolates at boundaries)
 * - Tensor-product structure for 4D evaluation
 * - FMA optimization for fast evaluation (~100ns per query)
 * - Proper boundary handling with nextafter for right endpoint
 *
 * Usage:
 *   std::vector<double> m_grid = {...};     // moneyness
 *   std::vector<double> tau_grid = {...};   // maturity
 *   std::vector<double> sigma_grid = {...}; // volatility
 *   std::vector<double> r_grid = {...};     // rate
 *   std::vector<double> coeffs = {...};     // from fitting
 *
 *   BSpline4D_FMA spline(m_grid, tau_grid, sigma_grid, r_grid, coeffs);
 *   double price = spline.eval(1.05, 0.25, 0.20, 0.05);
 *
 * Note: This class handles evaluation only. Coefficient fitting requires
 * a separate least-squares solver (see SeparableBSplineFitter4D).
 *
 * References:
 * - de Boor, "A Practical Guide to Splines" (2001)
 * - Piegl & Tiller, "The NURBS Book" (1997)
 */

#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace mango {

/// Create clamped knot vector for cubic B-splines
///
/// For n data points, creates n+4 knots with repeated endpoints:
/// [x[0], x[0], x[0], x[0], x[1], ..., x[n-2], x[n-1], x[n-1], x[n-1], x[n-1]]
///
/// @param x Grid points (must be sorted)
/// @return Clamped knot vector
inline std::vector<double> clamped_knots_cubic(const std::vector<double>& x) {
    const int n = static_cast<int>(x.size());
    std::vector<double> t(n + 4);

    // Left clamp: repeat first point 4 times
    std::fill_n(t.begin(), 4, x.front());

    // Interior knots
    for (int i = 1; i < n - 1; ++i) {
        t[3 + i] = x[i];
    }

    // Right clamp: repeat last point 4 times
    std::fill_n(t.end() - 4, 4, x.back());

    return t;
}

/// Find knot span containing x using binary search
///
/// Returns index i such that t[i] <= x < t[i+1]
///
/// @param t Knot vector
/// @param x Query point
/// @return Knot span index
inline int find_span_cubic(const std::vector<double>& t, double x) {
    auto it = std::upper_bound(t.begin(), t.end(), x);
    int i = static_cast<int>(std::distance(t.begin(), it)) - 1;

    // Clamp to valid range
    if (i < 0) i = 0;
    if (i >= static_cast<int>(t.size()) - 2) {
        i = static_cast<int>(t.size()) - 2;
    }

    return i;
}

/// Evaluate cubic basis functions using Cox-de Boor recursion
///
/// Computes the 4 nonzero cubic basis functions at x for knot span i.
/// Uses de Boor's recursive formula with proper handling of zero denominators.
///
/// @param t Knot vector
/// @param i Knot span index
/// @param x Evaluation point
/// @param N Output: 4 basis function values N[0..3]
///          N[0] corresponds to basis i, N[1] to i-1, N[2] to i-2, N[3] to i-3
inline void cubic_basis_nonuniform(const std::vector<double>& t, int i, double x, double N[4]) {
    // Degree 0: piecewise constants
    double N0[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        int idx = i - k;
        if (idx >= 0 && idx + 1 < static_cast<int>(t.size())) {
            N0[k] = (t[idx] <= x && x < t[idx + 1]) ? 1.0 : 0.0;
        }
    }

    // Degree 1: linear combination
    double N1[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        int idx = i - k;
        double leftDen  = t[idx + 1] - t[idx];
        double rightDen = t[idx + 2] - t[idx + 1];

        double left  = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N0[k] : 0.0;
        double right = (rightDen > 0.0 && k > 0) ? (t[idx + 2] - x) / rightDen * N0[k - 1] : 0.0;

        N1[k] = left + right;
    }

    // Degree 2: quadratic
    double N2[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        int idx = i - k;
        double leftDen  = t[idx + 2] - t[idx];
        double rightDen = t[idx + 3] - t[idx + 1];

        double left  = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N1[k] : 0.0;
        double right = (rightDen > 0.0 && k > 0) ? (t[idx + 3] - x) / rightDen * N1[k - 1] : 0.0;

        N2[k] = left + right;
    }

    // Degree 3: cubic
    for (int k = 0; k < 4; ++k) {
        int idx = i - k;
        double leftDen  = t[idx + 3] - t[idx];
        double rightDen = t[idx + 4] - t[idx + 1];

        double left  = (leftDen > 0.0) ? (x - t[idx]) / leftDen * N2[k] : 0.0;
        double right = (rightDen > 0.0 && k > 0) ? (t[idx + 4] - x) / rightDen * N2[k - 1] : 0.0;

        N[k] = left + right;
    }
}

/// Clamp query point to valid domain
///
/// For right boundary, uses nextafter to ensure x < xmax (not x <= xmax)
/// to avoid issues with half-open interval [xmin, xmax)
///
/// @param x Query point
/// @param xmin Minimum value
/// @param xmax Maximum value
/// @return Clamped value
inline double clamp_query(double x, double xmin, double xmax) {
    if (x <= xmin) return xmin;
    if (x >= xmax) {
        return std::nextafter(xmax, -std::numeric_limits<double>::infinity());
    }
    return x;
}

/// 4D Tensor-Product B-Spline with FMA Optimization
///
/// Evaluates 4D B-spline surfaces using tensor-product structure:
///   f(m, τ, σ, r) = Σ Σ Σ Σ c[i,j,k,l] · B_i(m) · B_j(τ) · B_k(σ) · B_l(r)
///
/// where B_i are cubic B-spline basis functions.
///
/// Performance: ~100-200ns per evaluation with FMA instructions
///
/// Memory layout: coefficients stored in row-major order (r varies fastest)
///   index = ((i * Nt + j) * Nv + k) * Nr + l
class BSpline4D_FMA {
public:
    /// Construct 4D B-spline from grids and coefficients
    ///
    /// @param m Moneyness grid (sorted, ≥4 points)
    /// @param t Maturity grid (sorted, ≥4 points)
    /// @param v Volatility grid (sorted, ≥4 points)
    /// @param r Rate grid (sorted, ≥4 points)
    /// @param coeff Coefficients (size must be Nm × Nt × Nv × Nr)
    BSpline4D_FMA(std::vector<double> m,
                  std::vector<double> t,
                  std::vector<double> v,
                  std::vector<double> r,
                  std::vector<double> coeff)
        : m_(std::move(m)),
          t_(std::move(t)),
          v_(std::move(v)),
          r_(std::move(r)),
          tm_(clamped_knots_cubic(m_)),
          tt_(clamped_knots_cubic(t_)),
          tv_(clamped_knots_cubic(v_)),
          tr_(clamped_knots_cubic(r_)),
          c_(std::move(coeff)),
          Nm_(static_cast<int>(m_.size())),
          Nt_(static_cast<int>(t_.size())),
          Nv_(static_cast<int>(v_.size())),
          Nr_(static_cast<int>(r_.size()))
    {
        assert(Nm_ >= 4 && "Moneyness grid must have ≥4 points");
        assert(Nt_ >= 4 && "Maturity grid must have ≥4 points");
        assert(Nv_ >= 4 && "Volatility grid must have ≥4 points");
        assert(Nr_ >= 4 && "Rate grid must have ≥4 points");
        assert(c_.size() == static_cast<std::size_t>(Nm_) * Nt_ * Nv_ * Nr_ &&
               "Coefficient size must match grid dimensions");
    }

    /// Evaluate B-spline at query point
    ///
    /// @param mq Moneyness query
    /// @param tq Maturity query
    /// @param vq Volatility query
    /// @param rq Rate query
    /// @return Interpolated value
    double eval(double mq, double tq, double vq, double rq) const {
        // Clamp queries to domain
        mq = clamp_query(mq, m_.front(), m_.back());
        tq = clamp_query(tq, t_.front(), t_.back());
        vq = clamp_query(vq, v_.front(), v_.back());
        rq = clamp_query(rq, r_.front(), r_.back());

        // Find knot spans
        int im = find_span_cubic(tm_, mq);
        int jt = find_span_cubic(tt_, tq);
        int kv = find_span_cubic(tv_, vq);
        int lr = find_span_cubic(tr_, rq);

        // Evaluate basis functions
        double wm[4], wt[4], wv[4], wr[4];
        cubic_basis_nonuniform(tm_, im, mq, wm);
        cubic_basis_nonuniform(tt_, jt, tq, wt);
        cubic_basis_nonuniform(tv_, kv, vq, wv);
        cubic_basis_nonuniform(tr_, lr, rq, wr);

        // Tensor-product evaluation with FMA
        double sum = 0.0;

        for (int a = 0; a < 4; ++a) {
            int im_idx = im - a;
            if (static_cast<unsigned>(im_idx) >= static_cast<unsigned>(Nm_)) continue;

            double wma = wm[a];

            for (int b = 0; b < 4; ++b) {
                int jt_idx = jt - b;
                if (static_cast<unsigned>(jt_idx) >= static_cast<unsigned>(Nt_)) continue;

                double wtab = std::fma(wma, wt[b], 0.0);  // wma * wt[b]

                for (int c = 0; c < 4; ++c) {
                    int kv_idx = kv - c;
                    if (static_cast<unsigned>(kv_idx) >= static_cast<unsigned>(Nv_)) continue;

                    double wtabc = std::fma(wtab, wv[c], 0.0);  // wtab * wv[c]

                    // Compute base index for coefficient array
                    const std::size_t base =
                        (((std::size_t)im_idx * Nt_ + jt_idx) * Nv_ + kv_idx) * Nr_;

                    // FMA accumulation for innermost dimension
                    // sum += coeff[...] * wtabc * wr[d]

                    if (int lr0 = lr - 0; static_cast<unsigned>(lr0) < static_cast<unsigned>(Nr_)) {
                        sum = std::fma(c_[base + lr0], wtabc * wr[0], sum);
                    }

                    if (int lr1 = lr - 1; static_cast<unsigned>(lr1) < static_cast<unsigned>(Nr_)) {
                        sum = std::fma(c_[base + lr1], wtabc * wr[1], sum);
                    }

                    if (int lr2 = lr - 2; static_cast<unsigned>(lr2) < static_cast<unsigned>(Nr_)) {
                        sum = std::fma(c_[base + lr2], wtabc * wr[2], sum);
                    }

                    if (int lr3 = lr - 3; static_cast<unsigned>(lr3) < static_cast<unsigned>(Nr_)) {
                        sum = std::fma(c_[base + lr3], wtabc * wr[3], sum);
                    }
                }
            }
        }

        return sum;
    }

    /// Get grid dimensions
    [[nodiscard]] std::tuple<int, int, int, int> dimensions() const noexcept {
        return {Nm_, Nt_, Nv_, Nr_};
    }

    /// Get moneyness grid
    [[nodiscard]] const std::vector<double>& moneyness_grid() const noexcept { return m_; }

    /// Get maturity grid
    [[nodiscard]] const std::vector<double>& maturity_grid() const noexcept { return t_; }

    /// Get volatility grid
    [[nodiscard]] const std::vector<double>& volatility_grid() const noexcept { return v_; }

    /// Get rate grid
    [[nodiscard]] const std::vector<double>& rate_grid() const noexcept { return r_; }

private:
    std::vector<double> m_;   ///< Moneyness grid
    std::vector<double> t_;   ///< Maturity grid
    std::vector<double> v_;   ///< Volatility grid
    std::vector<double> r_;   ///< Rate grid

    std::vector<double> tm_;  ///< Moneyness knot vector
    std::vector<double> tt_;  ///< Maturity knot vector
    std::vector<double> tv_;  ///< Volatility knot vector
    std::vector<double> tr_;  ///< Rate knot vector

    std::vector<double> c_;   ///< Coefficients (Nm × Nt × Nv × Nr)

    int Nm_;  ///< Number of moneyness points
    int Nt_;  ///< Number of maturity points
    int Nv_;  ///< Number of volatility points
    int Nr_;  ///< Number of rate points
};

}  // namespace mango
