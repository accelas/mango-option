#ifndef MANGO_DIVIDEND_JUMP_HPP
#define MANGO_DIVIDEND_JUMP_HPP

#include <span>
#include <cmath>
#include <vector>
#include <algorithm>

namespace mango {

/**
 * Dividend jump event for discrete dividend payments.
 *
 * When a dividend D is paid at time t_div, the stock price drops
 * from S to S - D, causing a jump in log-moneyness coordinates:
 *   x = ln(S/K) → x_new = ln((S-D)/K) = ln(exp(x)*K - D) - ln(K)
 *
 * Option values must be interpolated to the new grid positions.
 */
class DividendJump {
public:
    /**
     * Constructor.
     *
     * @param dividend Dividend amount (in dollars)
     * @param strike Strike price (for coordinate transformation)
     */
    DividendJump(double dividend, double strike)
        : dividend_(dividend), strike_(strike) {}

    /**
     * Apply dividend jump to option values.
     *
     * This is a TemporalEventCallback compatible method.
     *
     * @param t Time of dividend payment
     * @param x Grid points (log-moneyness values)
     * @param u Option values (modified in-place)
     */
    void operator()(double t, std::span<const double> x,
                    std::span<double> u) const {
        [[maybe_unused]] auto unused = t;

        const size_t n = x.size();

        // Store original values
        std::vector<double> u_old(u.begin(), u.end());

        // Compute new x positions after dividend
        std::vector<double> x_new(n);
        for (size_t i = 0; i < n; ++i) {
            const double S = strike_ * std::exp(x[i]);
            const double S_new = S - dividend_;

            // Avoid negative stock prices
            if (S_new <= 0.0) {
                x_new[i] = -10.0;  // Very deep OTM
            } else {
                x_new[i] = std::log(S_new / strike_);
            }
        }

        // Interpolate u values to new positions
        for (size_t i = 0; i < n; ++i) {
            u[i] = interpolate(x, u_old, x_new[i]);
        }
    }

private:
    double dividend_;  ///< Dividend amount (dollars)
    double strike_;    ///< Strike price (dollars)

    /**
     * Linear interpolation of u at position x_target.
     *
     * @param x Grid points (must be sorted ascending)
     * @param u Values at grid points
     * @param x_target Target position for interpolation
     * @return Interpolated value
     */
    double interpolate(std::span<const double> x,
                       std::span<const double> u,
                       double x_target) const {
        const size_t n = x.size();

        // Boundary cases
        if (x_target <= x[0]) return u[0];
        if (x_target >= x[n-1]) return u[n-1];

        // Find bracketing indices using binary search
        auto it = std::lower_bound(x.begin(), x.end(), x_target);
        size_t j = std::distance(x.begin(), it);

        // Ensure j > 0 (it will be since x_target > x[0])
        if (j == 0) j = 1;

        size_t i = j - 1;

        // Linear interpolation
        const double dx = x[j] - x[i];
        const double weight = (x_target - x[i]) / dx;

        return (1.0 - weight) * u[i] + weight * u[j];
    }
};

}  // namespace mango

#endif  // MANGO_DIVIDEND_JUMP_HPP
