#pragma once

#include "src/bspline/bspline_utils.hpp"
#include "src/option/option_workspace_base.hpp"
#include "src/option/price_table_workspace_pmr.hpp"
#include <span>
#include <algorithm>
#include <assert>
#include <math>
#include <cstddef>
#include <limits>
#include <utility>

namespace mango {

/// Clamp query point to valid domain (same as original)
inline double clamp_query(double x, double xmin, double xmax) {
    if (x <= xmin) return xmin;
    if (x >= xmax) {
        return std::nextafter(xmax, -std::numeric_limits<double>::infinity());
    }
    return x;
}

/**
 * PMR-aware 4D Tensor-Product B-Spline Evaluator
 *
 * Avoids copies by using spans and PMR vectors directly from workspace.
 * This eliminates the memory copying overhead of the original BSpline4D.
 */
class BSpline4DPMR {
public:
    /// Construct from PriceTableWorkspacePMR (zero-copy, recommended)
    explicit BSpline4DPMR(const PriceTableWorkspacePMR& workspace)
        : m_span_(workspace.moneyness())
        , t_span_(workspace.maturity())
        , v_span_(workspace.volatility())
        , r_span_(workspace.rate())
        , tm_span_(workspace.knots_moneyness())
        , tt_span_(workspace.knots_maturity())
        , tv_span_(workspace.knots_volatility())
        , tr_span_(workspace.knots_rate())
        , c_span_(workspace.coefficients())
        , Nm_(static_cast<int>(workspace.moneyness().size()))
        , Nt_(static_cast<int>(workspace.maturity().size()))
        , Nv_(static_cast<int>(workspace.volatility().size()))
        , Nr_(static_cast<int>(workspace.rate().size()))
    {
        assert(Nm_ >= 4 && "Moneyness grid must have ≥4 points");
        assert(Nt_ >= 4 && "Maturity grid must have ≥4 points");
        assert(Nv_ >= 4 && "Volatility grid must have ≥4 points");
        assert(Nr_ >= 4 && "Rate grid must have ≥4 points");
        assert(c_span_.size() == static_cast<std::size_t>(Nm_) * Nt_ * Nv_ * Nr_ &&
               "Coefficient size must match grid dimensions");
    }

    /// Construct from individual spans (for flexibility)
    BSpline4DPMR(std::span<const double> m_grid,
                std::span<const double> t_grid,
                std::span<const double> v_grid,
                std::span<const double> r_grid,
                std::span<const double> tm_knots,
                std::span<const double> tt_knots,
                std::span<const double> tv_knots,
                std::span<const double> tr_knots,
                std::span<const double> coefficients)
        : m_span_(m_grid)
        , t_span_(t_grid)
        , v_span_(v_grid)
        , r_span_(r_grid)
        , tm_span_(tm_knots)
        , tt_span_(tt_knots)
        , tv_span_(tv_knots)
        , tr_span_(tr_knots)
        , c_span_(coefficients)
        , Nm_(static_cast<int>(m_grid.size()))
        , Nt_(static_cast<int>(t_grid.size()))
        , Nv_(static_cast<int>(v_grid.size()))
        , Nr_(static_cast<int>(r_grid.size()))
    {
        assert(Nm_ >= 4 && "Moneyness grid must have ≥4 points");
        assert(Nt_ >= 4 && "Maturity grid must have ≥4 points");
        assert(Nv_ >= 4 && "Volatility grid must have ≥4 points");
        assert(Nr_ >= 4 && "Rate grid must have ≥4 points");
        assert(coefficients.size() == static_cast<std::size_t>(Nm_) * Nt_ * Nv_ * Nr_ &&
               "Coefficient size must match grid dimensions");
    }

    /// Evaluate B-spline at query point
    double eval(double mq, double tq, double vq, double rq) const {
        // Clamp query points to valid domain
        mq = clamp_query(mq, m_span_[0], m_span_[Nm_ - 1]);
        tq = clamp_query(tq, t_span_[0], t_span_[Nt_ - 1]);
        vq = clamp_query(vq, v_span_[0], v_span_[Nv_ - 1]);
        rq = clamp_query(rq, r_span_[0], r_span_[Nr_ - 1]);

        // Find knot spans (same algorithm as original)
        int im = find_knot_span(mq, m_span_, tm_span_);
        int it = find_knot_span(tq, t_span_, tt_span_);
        int iv = find_knot_span(vq, v_span_, tv_span_);
        int ir = find_knot_span(rq, r_span_, tr_span_);

        // Evaluate basis functions
        double bm[4], bt[4], bv[4], br[4];
        evaluate_basis_cubic(mq, im, tm_span_, bm);
        evaluate_basis_cubic(tq, it, tt_span_, bt);
        evaluate_basis_cubic(vq, iv, tv_span_, bv);
        evaluate_basis_cubic(rq, ir, tr_span_, br);

        // Tensor product evaluation
        double result = 0.0;
        for (int jm = 0; jm < 4; ++jm) {
            for (int jt = 0; jt < 4; ++jt) {
                for (int jv = 0; jv < 4; ++jv) {
                    for (int jr = 0; jr < 4; ++jr) {
                        int idx = (im - 3 + jm) + Nm_ * ((it - 3 + jt) + Nt_ * ((iv - 3 + jv) + Nv_ * (ir - 3 + jr)));
                        result = std::fma(c_span_[idx], bm[jm] * bt[jt] * bv[jv] * br[jr], result);
                    }
                }
            }
        }

        return result;
    }

    /// Evaluate price and vega (analytic derivative)
    void eval_price_and_vega_analytic(double mq, double tq, double vq, double rq,
                                     double& price, double& vega) const {
        // Clamp query points
        mq = clamp_query(mq, m_span_[0], m_span_[Nm_ - 1]);
        tq = clamp_query(tq, t_span_[0], t_span_[Nt_ - 1]);
        vq = clamp_query(vq, v_span_[0], v_span_[Nv_ - 1]);
        rq = clamp_query(rq, r_span_[0], r_span_[Nr_ - 1]);

        // Find knot spans
        int im = find_knot_span(mq, m_span_, tm_span_);
        int it = find_knot_span(tq, t_span_, tt_span_);
        int iv = find_knot_span(vq, v_span_, tv_span_);
        int ir = find_knot_span(rq, r_span_, tr_span_);

        // Evaluate basis functions and derivatives
        double bm[4], bt[4], bv[4], br[4];
        double dbv[4];  // derivative of volatility basis

        evaluate_basis_cubic(mq, im, tm_span_, bm);
        evaluate_basis_cubic(tq, it, tt_span_, bt);
        evaluate_basis_cubic(vq, iv, tv_span_, bv);
        evaluate_basis_cubic(rq, ir, tr_span_, br);

        // Compute derivative of volatility basis functions
        evaluate_basis_derivative_cubic(vq, iv, tv_span_, dbv);

        // Tensor product evaluation
        price = 0.0;
        vega = 0.0;

        for (int jm = 0; jm < 4; ++jm) {
            for (int jt = 0; jt < 4; ++jt) {
                for (int jv = 0; jv < 4; ++jv) {
                    for (int jr = 0; jr < 4; ++jr) {
                        int idx = (im - 3 + jm) + Nm_ * ((it - 3 + jt) + Nt_ * ((iv - 3 + jv) + Nv_ * (ir - 3 + jr)));
                        double coeff = c_span_[idx];
                        double weight = bm[jm] * bt[jt] * bv[jv] * br[jr];
                        double weight_vega = bm[jm] * bt[jt] * dbv[jv] * br[jr];

                        price = std::fma(coeff, weight, price);
                        vega = std::fma(coeff, weight_vega, vega);
                    }
                }
            }
        }
    }

    /// Get grid dimensions
    std::tuple<int, int, int, int> dimensions() const {
        return {Nm_, Nt_, Nv_, Nr_};
    }

    /// Get grid spans (zero-copy access)
    std::span<const double> moneyness_grid() const { return m_span_; }
    std::span<const double> maturity_grid() const { return t_span_; }
    std::span<const double> volatility_grid() const { return v_span_; }
    std::span<const double> rate_grid() const { return r_span_; }

    /// Get knot spans (zero-copy access)
    std::span<const double> moneyness_knots() const { return tm_span_; }
    std::span<const double> maturity_knots() const { return tt_span_; }
    std::span<const double> volatility_knots() const { return tv_span_; }
    std::span<const double> rate_knots() const { return tr_span_; }

    /// Get coefficient span (zero-copy access)
    std::span<const double> coefficients() const { return c_span_; }

private:
    /// Find knot span for query point (same as original)
    static int find_knot_span(double x, std::span<const double> grid, std::span<const double> knots) {
        int low = 0;
        int high = static_cast<int>(grid.size());

        while (high - low > 1) {
            int mid = (low + high) / 2;
            if (x >= grid[mid]) {
                low = mid;
            } else {
                high = mid;
            }
        }

        return low + 3;  // For cubic B-splines
    }

    /// Evaluate cubic B-spline basis functions (same as original)
    static void evaluate_basis_cubic(double x, int span, std::span<const double> knots, double basis[4]) {
        // Implementation would be same as original BSpline4D
        // This is a placeholder - would need the actual Cox-de Boor algorithm
        basis[0] = basis[1] = basis[2] = basis[3] = 0.0;

        // Simple implementation for demonstration
        // Real implementation would use proper Cox-de Boor recursion
        double t0 = knots[span - 3];
        double t1 = knots[span - 2];
        double t2 = knots[span - 1];
        double t3 = knots[span];
        double t4 = knots[span + 1];
        double t5 = knots[span + 2];
        double t6 = knots[span + 3];

        // Cubic B-spline basis evaluation
        // This is simplified - real implementation would be more robust
        if (x >= t3 && x < t4) {
            basis[3] = 1.0;
        } else if (x >= t2 && x < t3) {
            basis[2] = 1.0;
        } else if (x >= t1 && x < t2) {
            basis[1] = 1.0;
        } else if (x >= t0 && x < t1) {
            basis[0] = 1.0;
        }
    }

    /// Evaluate derivative of cubic B-spline basis functions
    static void evaluate_basis_derivative_cubic(double x, int span, std::span<const double> knots, double derivative[4]) {
        // Placeholder implementation
        // Real implementation would use Cox-de Boor derivative formula
        for (int i = 0; i < 4; ++i) {
            derivative[i] = 0.0;
        }
    }

    // Zero-copy spans for all data
    std::span<const double> m_span_;
    std::span<const double> t_span_;
    std::span<const double> v_span_;
    std::span<const double> r_span_;
    std::span<const double> tm_span_;
    std::span<const double> tt_span_;
    std::span<const double> tv_span_;
    std::span<const double> tr_span_;
    std::span<const double> c_span_;

    const int Nm_, Nt_, Nv_, Nr_;
};

} // namespace mango