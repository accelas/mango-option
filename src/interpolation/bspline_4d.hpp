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
 * - Optimized vega computation via scalar triple evaluation and analytic derivatives
 *
 * Performance Summary (Intel Xeon, 256 FMAs per evaluation):
 * - Single price eval: ~135ns
 * - Vega via FD (3 evals): 515ns (baseline)
 * - Vega via scalar triple: 272ns (1.89× speedup) ⭐ RECOMMENDED
 * - Vega via analytic derivative: 275ns (1.87× speedup, exact) ⭐ RECOMMENDED
 *
 * Vega Computation Methods:
 * 1. **Scalar triple FD** (`eval_price_and_vega_triple()`):
 *    - Evaluates (σ-ε, σ, σ+ε) simultaneously with 3 independent accumulators
 *    - Exploits CPU's 3-4 parallel FMA units via instruction-level parallelism (ILP)
 *    - 272ns per query (1.89× faster than baseline)
 *
 * 2. **Analytic derivative** (`eval_price_and_vega_analytic()`):
 *    - Uses Cox-de Boor derivative formula: B'_{i,3} = 3/(t[i+3]-t[i]) B_{i,2} - ...
 *    - Exact derivative (no finite difference truncation error)
 *    - No epsilon parameter tuning required
 *    - Better boundary accuracy (no σ±ε clamping bias)
 *    - 275ns per query (same performance as scalar FD, mathematically superior)
 *
 * Why Analytic ≈ Scalar FD Performance:
 * - Analytic uses 2 accumulators vs FD's 3, reducing ILP slightly
 * - Basis recursion is <5% of total work (256-FMA tensor loop dominates)
 * - Both hit latency-bound ceiling (~275ns = 73% of theoretical 160ns minimum)
 * - See docs/analytic-vega-analysis.md for detailed expert analysis
 *
 * Design Rationale - Why Scalar Beats SIMD:
 * Vertical SIMD experiments (removed as of PR #157) showed 2.3× performance
 * regression due to:
 * 1. ILP loss: 3 independent scalar chains → 1 SIMD dependency chain
 * 2. Broadcast overhead: 56 scalar→SIMD broadcasts serialize with computation
 * 3. Stack packing latency: Materializing basis weights to memory vs registers
 *
 * For complete SIMD analysis, see git history (commit 331e2ab) and
 * docs/analytic-vega-analysis.md.
 *
 * Usage:
 *   std::vector<double> m_grid = {...};     // moneyness
 *   std::vector<double> tau_grid = {...};   // maturity
 *   std::vector<double> sigma_grid = {...}; // volatility
 *   std::vector<double> r_grid = {...};     // rate
 *   std::vector<double> coeffs = {...};     // from fitting
 *
 *   BSpline4D_FMA spline(m_grid, tau_grid, sigma_grid, r_grid, coeffs);
 *
 *   // Scalar triple FD (fast, tunable epsilon):
 *   double price, vega;
 *   spline.eval_price_and_vega_triple(1.05, 0.25, 0.20, 0.05, 1e-4, price, vega);
 *
 *   // Analytic derivative (exact, no epsilon):
 *   spline.eval_price_and_vega_analytic(1.05, 0.25, 0.20, 0.05, price, vega);
 *
 * Note: This class handles evaluation only. Coefficient fitting requires
 * a separate least-squares solver (see BSplineFitter4D).
 *
 * References:
 * - de Boor, "A Practical Guide to Splines" (2001)
 * - Piegl & Tiller, "The NURBS Book" (1997)
 */

#pragma once

#include "src/interpolation/bspline_utils.hpp"
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace mango {

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

                    // Compute valid range for rate dimension (d ∈ [0,3])
                    // We access coefficient at index (lr - d), which must satisfy 0 <= lr - d < Nr_
                    // This gives: max(0, lr - Nr_ + 1) <= d <= min(3, lr)
                    const int d_min = std::max(0, lr - (Nr_ - 1));
                    const int d_max = std::min(3, lr);

                    // Get pointer to coefficient block for efficient streaming
                    const double* coeff_block = c_.data() + base;

                    // Stream coefficients with no per-iteration branches
                    // This replaces 4 separate if blocks with a tight loop,
                    // improving ILP and enabling CPU prefetching
                    for (int d = d_min; d <= d_max; ++d) {
                        const int lr_idx = lr - d;
                        sum = std::fma(coeff_block[lr_idx], wtabc * wr[d], sum);
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

    /// Evaluate price and vega in single pass (scalar version)
    ///
    /// Computes V(σ) and vega = ∂V/∂σ via centered finite difference.
    /// Single-pass implementation shares coefficient loads.
    ///
    /// @param mq Moneyness query point
    /// @param tq Maturity query point
    /// @param vq Volatility query point (σ)
    /// @param rq Rate query point
    /// @param epsilon Finite difference step for vega
    /// @param[out] price Output: V(σ)
    /// @param[out] vega Output: ∂V/∂σ ≈ (V(σ+ε) - V(σ-ε))/(2ε)
    void eval_price_and_vega_triple(
        double mq, double tq, double vq, double rq,
        double epsilon,
        double& price, double& vega) const
    {
        // Clamp queries to domain
        mq = clamp_query(mq, m_.front(), m_.back());
        tq = clamp_query(tq, t_.front(), t_.back());
        vq = clamp_query(vq, v_.front(), v_.back());
        rq = clamp_query(rq, r_.front(), r_.back());

        // Find knot spans (shared across 3 sigma values)
        const int im = find_span_cubic(tm_, mq);
        const int jt = find_span_cubic(tt_, tq);
        const int lr = find_span_cubic(tr_, rq);

        // Evaluate basis for m, tau, rate (shared across 3 sigma values)
        double wm[4], wt[4], wr[4];
        cubic_basis_nonuniform(tm_, im, mq, wm);
        cubic_basis_nonuniform(tt_, jt, tq, wt);
        cubic_basis_nonuniform(tr_, lr, rq, wr);

        // Clamp shifted sigma values to prevent extrapolation outside grid
        const double v_down = clamp_query(vq - epsilon, v_.front(), v_.back());
        const double v_up = clamp_query(vq + epsilon, v_.front(), v_.back());

        // Find sigma span (may differ for shifted values at boundaries)
        const int kv = find_span_cubic(tv_, vq);

        // Evaluate basis for 3 sigma values (with clamped shifts)
        double wv_down[4], wv_base[4], wv_up[4];
        cubic_basis_nonuniform(tv_, kv, v_down, wv_down);
        cubic_basis_nonuniform(tv_, kv, vq, wv_base);
        cubic_basis_nonuniform(tv_, kv, v_up, wv_up);

        // Accumulate 3 results in parallel
        double price_down = 0.0;
        double price_base = 0.0;
        double price_up = 0.0;

        // 4D tensor product (256 iterations total)
        for (int a = 0; a < 4; ++a) {
            int im_idx = im - a;
            if (static_cast<unsigned>(im_idx) >= static_cast<unsigned>(Nm_)) continue;

            for (int b = 0; b < 4; ++b) {
                int jt_idx = jt - b;
                if (static_cast<unsigned>(jt_idx) >= static_cast<unsigned>(Nt_)) continue;

                const double wm_wt = wm[a] * wt[b];

                for (int c = 0; c < 4; ++c) {
                    int kv_idx = kv - c;
                    if (static_cast<unsigned>(kv_idx) >= static_cast<unsigned>(Nv_)) continue;

                    // Pack 3 sigma weights
                    const double w_down = wm_wt * wv_down[c];
                    const double w_base = wm_wt * wv_base[c];
                    const double w_up = wm_wt * wv_up[c];

                    // Compute base index for coefficient array
                    const std::size_t base =
                        (((std::size_t)im_idx * Nt_ + jt_idx) * Nv_ + kv_idx) * Nr_;

                    // Compute valid range for rate dimension
                    const int d_min = std::max(0, lr - (Nr_ - 1));
                    const int d_max = std::min(3, lr);

                    const double* coeff_block = c_.data() + base;

                    for (int d = d_min; d <= d_max; ++d) {
                        const int lr_idx = lr - d;
                        const double coeff = coeff_block[lr_idx];
                        const double w_r = wr[d];

                        price_down = std::fma(coeff, w_down * w_r, price_down);
                        price_base = std::fma(coeff, w_base * w_r, price_base);
                        price_up = std::fma(coeff, w_up * w_r, price_up);
                    }
                }
            }
        }

        price = price_base;
        vega = (price_up - price_down) / (2.0 * epsilon);
    }

    /// Evaluate price and vega using analytic B-spline derivative
    ///
    /// Computes V(σ) and vega = ∂V/∂σ exactly using B-spline derivative formula.
    /// Single evaluation vs 3 for finite difference - expected ~45% faster than scalar triple.
    ///
    /// Uses Cox-de Boor derivative formula: B'_{i,p}(x) = p/(t[i+p]-t[i])*B_{i,p-1}(x) - ...
    /// For cubic B-splines, derivatives are expressed in terms of quadratic basis functions.
    ///
    /// @param mq Moneyness query point
    /// @param tq Maturity query point
    /// @param vq Volatility query point (σ)
    /// @param rq Rate query point
    /// @param[out] price Output: V(σ)
    /// @param[out] vega Output: exact ∂V/∂σ
    void eval_price_and_vega_analytic(
        double mq, double tq, double vq, double rq,
        double& price, double& vega) const
    {
        // Clamp queries to domain
        mq = clamp_query(mq, m_.front(), m_.back());
        tq = clamp_query(tq, t_.front(), t_.back());
        vq = clamp_query(vq, v_.front(), v_.back());
        rq = clamp_query(rq, r_.front(), r_.back());

        // Find knot spans
        const int im = find_span_cubic(tm_, mq);
        const int jt = find_span_cubic(tt_, tq);
        const int kv = find_span_cubic(tv_, vq);
        const int lr = find_span_cubic(tr_, rq);

        // Evaluate basis for m, tau, rate
        double wm[4], wt[4], wr[4];
        cubic_basis_nonuniform(tm_, im, mq, wm);
        cubic_basis_nonuniform(tt_, jt, tq, wt);
        cubic_basis_nonuniform(tr_, lr, rq, wr);

        // Evaluate basis AND derivative for volatility
        double wv[4], dwv[4];
        cubic_basis_nonuniform(tv_, kv, vq, wv);
        cubic_basis_derivative_nonuniform(tv_, kv, vq, dwv);

        // Accumulate price and vega simultaneously
        price = 0.0;
        vega = 0.0;

        // 4D tensor product (256 iterations total)
        for (int a = 0; a < 4; ++a) {
            int im_idx = im - a;
            if (static_cast<unsigned>(im_idx) >= static_cast<unsigned>(Nm_)) continue;

            for (int b = 0; b < 4; ++b) {
                int jt_idx = jt - b;
                if (static_cast<unsigned>(jt_idx) >= static_cast<unsigned>(Nt_)) continue;

                const double wm_wt = wm[a] * wt[b];

                for (int c = 0; c < 4; ++c) {
                    int kv_idx = kv - c;
                    if (static_cast<unsigned>(kv_idx) >= static_cast<unsigned>(Nv_)) continue;

                    // Weights for price and vega
                    const double w_price = wm_wt * wv[c];
                    const double w_vega = wm_wt * dwv[c];

                    // Compute base index for coefficient array
                    const std::size_t base =
                        (((std::size_t)im_idx * Nt_ + jt_idx) * Nv_ + kv_idx) * Nr_;

                    // Compute valid range for rate dimension
                    const int d_min = std::max(0, lr - (Nr_ - 1));
                    const int d_max = std::min(3, lr);

                    const double* coeff_block = c_.data() + base;

                    for (int d = d_min; d <= d_max; ++d) {
                        const int lr_idx = lr - d;
                        const double coeff = coeff_block[lr_idx];
                        const double w_r = wr[d];

                        // Two independent FMA chains (exploits ILP)
                        price = std::fma(coeff, w_price * w_r, price);
                        vega = std::fma(coeff, w_vega * w_r, vega);
                    }
                }
            }
        }
    }

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
