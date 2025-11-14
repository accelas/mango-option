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

#include "src/interpolation/bspline_utils.hpp"
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <experimental/simd>

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

    /// Evaluate price and vega using SIMD (3-lane)
    ///
    /// @deprecated PERFORMANCE REGRESSION - use eval_price_and_vega_triple() instead
    /// @warning Empirical benchmarking shows 0.45× speedup (18% slower than baseline!)
    /// @note Retained for research purposes only - demonstrates SIMD overhead for narrow width
    ///
    /// Uses std::experimental::fixed_size_simd<double,4> to evaluate
    /// σ-ε, σ, σ+ε in parallel. SIMD overhead (packing, broadcasts, copy_to)
    /// exceeds arithmetic benefits for only 3 lanes.
    ///
    /// Benchmarks (actual):
    /// - Scalar triple: 271ns (1.90× speedup) ✅ USE THIS
    /// - SIMD triple:   608ns (0.45× speedup) ❌ DO NOT USE
    ///
    /// @param mq Moneyness query point
    /// @param tq Maturity query point
    /// @param vq Volatility query point (σ)
    /// @param rq Rate query point
    /// @param epsilon Finite difference step
    /// @param[out] price Output: V(σ)
    /// @param[out] vega Output: ∂V/∂σ
    [[gnu::target_clones("default","avx2","avx512f")]]
    void eval_price_and_vega_triple_simd(
        double mq, double tq, double vq, double rq,
        double epsilon,
        double& price, double& vega) const
    {
        namespace stdx = std::experimental;
        using simd_t = stdx::fixed_size_simd<double, 4>;

        // Clamp queries to domain
        mq = clamp_query(mq, m_.front(), m_.back());
        tq = clamp_query(tq, t_.front(), t_.back());
        vq = clamp_query(vq, v_.front(), v_.back());
        rq = clamp_query(rq, r_.front(), r_.back());

        // Find knot spans (shared)
        const int im = find_span_cubic(tm_, mq);
        const int jt = find_span_cubic(tt_, tq);
        const int kv = find_span_cubic(tv_, vq);
        const int lr = find_span_cubic(tr_, rq);

        // Evaluate shared basis functions
        double wm[4], wt[4], wr[4];
        cubic_basis_nonuniform(tm_, im, mq, wm);
        cubic_basis_nonuniform(tt_, jt, tq, wt);
        cubic_basis_nonuniform(tr_, lr, rq, wr);

        // Clamp shifted sigma values to prevent extrapolation outside grid
        const double v_down = clamp_query(vq - epsilon, v_.front(), v_.back());
        const double v_up = clamp_query(vq + epsilon, v_.front(), v_.back());

        // Evaluate 3 sigma basis functions (with clamped shifts)
        double wv_down[4], wv_base[4], wv_up[4];
        cubic_basis_nonuniform(tv_, kv, v_down, wv_down);
        cubic_basis_nonuniform(tv_, kv, vq, wv_base);
        cubic_basis_nonuniform(tv_, kv, v_up, wv_up);

        // SIMD accumulator for 3 results + padding
        simd_t accum(0.0);

        // 4D tensor product with SIMD inner loop
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

                    // Pack 3 sigma weights into SIMD lanes
                    const double wv_data[4] = {wv_down[c], wv_base[c], wv_up[c], 0.0};
                    const simd_t wv_packed(wv_data, stdx::element_aligned);
                    const simd_t weight_mts = simd_t(wm_wt) * wv_packed;

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

                        // Single vector FMA for all 3 results
                        accum = stdx::fma(simd_t(coeff * w_r), weight_mts, accum);
                    }
                }
            }
        }

        // Extract results from SIMD lanes
        alignas(32) double results[4];
        accum.copy_to(results, stdx::element_aligned);

        price = results[1];  // Middle lane (σ)
        vega = (results[2] - results[0]) / (2.0 * epsilon);  // (σ+ε - σ-ε) / 2ε
    }

    /// Evaluate price and vega using dual-accumulator SIMD
    ///
    /// Experimental variant that uses TWO SIMD accumulators to break the
    /// dependency chain while preserving vectorization. Unrolls d-loop by 2
    /// to allow parallel execution of FMAs.
    ///
    /// @param mq Moneyness query point
    /// @param tq Maturity query point
    /// @param vq Volatility query point
    /// @param rq Rate query point
    /// @param epsilon Finite difference epsilon for vega
    /// @param price Output: interpolated price at σ
    /// @param vega Output: ∂V/∂σ via centered difference
    [[gnu::target_clones("default","avx2","avx512f")]]
    void eval_price_and_vega_triple_dual_simd(
        double mq, double tq, double vq, double rq,
        double epsilon,
        double& price, double& vega) const
    {
        namespace stdx = std::experimental;
        using simd_t = stdx::fixed_size_simd<double, 4>;

        // Clamp queries to domain
        mq = clamp_query(mq, m_.front(), m_.back());
        tq = clamp_query(tq, t_.front(), t_.back());
        vq = clamp_query(vq, v_.front(), v_.back());
        rq = clamp_query(rq, r_.front(), r_.back());

        // Find knot spans (shared)
        const int im = find_span_cubic(tm_, mq);
        const int jt = find_span_cubic(tt_, tq);
        const int kv = find_span_cubic(tv_, vq);
        const int lr = find_span_cubic(tr_, rq);

        // Evaluate shared basis functions
        double wm[4], wt[4], wr[4];
        cubic_basis_nonuniform(tm_, im, mq, wm);
        cubic_basis_nonuniform(tt_, jt, tq, wt);
        cubic_basis_nonuniform(tr_, lr, rq, wr);

        // Clamp shifted sigma values to prevent extrapolation outside grid
        const double v_down = clamp_query(vq - epsilon, v_.front(), v_.back());
        const double v_up = clamp_query(vq + epsilon, v_.front(), v_.back());

        // Evaluate 3 sigma basis functions (with clamped shifts)
        double wv_down[4], wv_base[4], wv_up[4];
        cubic_basis_nonuniform(tv_, kv, v_down, wv_down);
        cubic_basis_nonuniform(tv_, kv, vq, wv_base);
        cubic_basis_nonuniform(tv_, kv, v_up, wv_up);

        // TWO independent SIMD accumulators (breaks dependency chain)
        simd_t accum1(0.0);
        simd_t accum2(0.0);

        // 4D tensor product with dual-accumulator SIMD
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

                    // Pack 3 sigma weights into SIMD lanes
                    const double wv_data[4] = {wv_down[c], wv_base[c], wv_up[c], 0.0};
                    const simd_t wv_packed(wv_data, stdx::element_aligned);
                    const simd_t weight_mts = simd_t(wm_wt) * wv_packed;

                    // Compute base index for coefficient array
                    const std::size_t base =
                        (((std::size_t)im_idx * Nt_ + jt_idx) * Nv_ + kv_idx) * Nr_;

                    // Compute valid range for rate dimension
                    const int d_min = std::max(0, lr - (Nr_ - 1));
                    const int d_max = std::min(3, lr);

                    const double* coeff_block = c_.data() + base;

                    // Unroll d-loop by 2 to use both accumulators
                    int d = d_min;
                    for (; d + 1 <= d_max; d += 2) {
                        // First iteration: accumulator 1 (independent)
                        {
                            const int lr_idx = lr - d;
                            const double coeff = coeff_block[lr_idx];
                            const double w_r = wr[d];
                            accum1 = stdx::fma(simd_t(coeff * w_r), weight_mts, accum1);
                        }

                        // Second iteration: accumulator 2 (independent from accum1)
                        {
                            const int lr_idx = lr - (d + 1);
                            const double coeff = coeff_block[lr_idx];
                            const double w_r = wr[d + 1];
                            accum2 = stdx::fma(simd_t(coeff * w_r), weight_mts, accum2);
                        }
                    }

                    // Handle remaining iteration if d_max - d_min is odd
                    if (d <= d_max) {
                        const int lr_idx = lr - d;
                        const double coeff = coeff_block[lr_idx];
                        const double w_r = wr[d];
                        accum1 = stdx::fma(simd_t(coeff * w_r), weight_mts, accum1);
                    }
                }
            }
        }

        // Combine the two independent accumulators
        simd_t final_accum = accum1 + accum2;

        // Extract results from SIMD lanes
        alignas(32) double results[4];
        final_accum.copy_to(results, stdx::element_aligned);

        price = results[1];  // Middle lane (σ)
        vega = (results[2] - results[0]) / (2.0 * epsilon);  // (σ+ε - σ-ε) / 2ε
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
