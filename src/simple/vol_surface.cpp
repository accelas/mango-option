#include "src/simple/vol_surface.hpp"
#include <cmath>

namespace mango::simple {

std::optional<double> VolatilitySurface::iv_at(double strike, double tau) const {
    // Simple linear interpolation - could be enhanced with B-spline
    if (smiles.empty()) return std::nullopt;

    // Find bracketing smiles by tau
    const VolatilitySmile* left = nullptr;
    const VolatilitySmile* right = nullptr;

    for (const auto& smile : smiles) {
        if (smile.tau <= tau) {
            if (!left || smile.tau > left->tau) {
                left = &smile;
            }
        }
        if (smile.tau >= tau) {
            if (!right || smile.tau < right->tau) {
                right = &smile;
            }
        }
    }

    if (!left && !right) return std::nullopt;
    if (!left) left = right;
    if (!right) right = left;

    // For now, use nearest smile
    const auto& smile = (std::abs(left->tau - tau) < std::abs(right->tau - tau)) ? *left : *right;

    // Find nearest strike in smile
    double moneyness = std::log(strike / spot.to_double());
    std::optional<double> best_iv;
    double best_dist = std::numeric_limits<double>::max();

    for (const auto& pt : smile.calls) {
        double dist = std::abs(pt.moneyness - moneyness);
        if (dist < best_dist && pt.iv_mid) {
            best_dist = dist;
            best_iv = pt.iv_mid;
        }
    }
    for (const auto& pt : smile.puts) {
        double dist = std::abs(pt.moneyness - moneyness);
        if (dist < best_dist && pt.iv_mid) {
            best_dist = dist;
            best_iv = pt.iv_mid;
        }
    }

    return best_iv;
}

std::expected<VolatilitySurface, ComputeError> compute_vol_surface(
    const OptionChain& chain,
    const MarketContext& ctx,
    const mango::IVSolverInterpolated* solver,
    const IVComputeConfig& config,
    PriceSource price_source)
{
    if (!chain.spot) {
        return std::unexpected(ComputeError{"Missing spot price"});
    }

    if (config.method == IVComputeConfig::Method::Interpolated && !solver) {
        return std::unexpected(ComputeError{"Interpolated method requires solver"});
    }

    VolatilitySurface surface;
    surface.symbol = chain.symbol;
    surface.spot = *chain.spot;
    if (chain.quote_time) {
        surface.quote_time = *chain.quote_time;
    }

    // Get valuation time
    Timestamp val_time = ctx.valuation_time.value_or(
        chain.quote_time.value_or(Timestamp::now()));

    // Get rate
    double rate = 0.05;  // Default
    if (ctx.rate) {
        rate = mango::get_zero_rate(*ctx.rate, 1.0);  // Approximate
    }

    // Get dividend yield
    double div_yield = 0.0;
    auto div_spec = ctx.dividends.value_or(
        chain.dividends.value_or(DividendSpec{0.0}));
    if (std::holds_alternative<double>(div_spec)) {
        div_yield = std::get<double>(div_spec);
    }

    double spot = chain.spot->to_double();
    size_t failed_count = 0;

    for (const auto& slice : chain.expiries) {
        VolatilitySmile smile;
        smile.expiry = slice.expiry;
        smile.tau = compute_tau(val_time, slice.expiry);
        smile.spot = *chain.spot;

        if (smile.tau <= 0) continue;  // Skip expired options

        // Process calls
        for (const auto& leg : slice.calls) {
            auto price_opt = (price_source == PriceSource::Mid) ? leg.mid() :
                             (price_source == PriceSource::Bid) ? leg.bid :
                             (price_source == PriceSource::Ask) ? leg.ask :
                             leg.last;

            if (!price_opt) continue;

            double strike = leg.strike.to_double();
            double market_price = price_opt->to_double();

            VolatilitySmile::Point pt;
            pt.strike = leg.strike;
            pt.moneyness = std::log(strike / spot);

            mango::IVQuery query;
            query.spot = spot;
            query.strike = strike;
            query.maturity = smile.tau;
            query.rate = rate;
            query.dividend_yield = div_yield;
            query.type = mango::OptionType::CALL;
            query.market_price = market_price;

            if (solver) {
                auto result = solver->solve_impl(query);
                if (result) {
                    pt.iv_mid = result->implied_vol;
                } else {
                    ++failed_count;
                }
            }

            smile.calls.push_back(pt);
        }

        // Process puts
        for (const auto& leg : slice.puts) {
            auto price_opt = (price_source == PriceSource::Mid) ? leg.mid() :
                             (price_source == PriceSource::Bid) ? leg.bid :
                             (price_source == PriceSource::Ask) ? leg.ask :
                             leg.last;

            if (!price_opt) continue;

            double strike = leg.strike.to_double();
            double market_price = price_opt->to_double();

            VolatilitySmile::Point pt;
            pt.strike = leg.strike;
            pt.moneyness = std::log(strike / spot);

            mango::IVQuery query;
            query.spot = spot;
            query.strike = strike;
            query.maturity = smile.tau;
            query.rate = rate;
            query.dividend_yield = div_yield;
            query.type = mango::OptionType::PUT;
            query.market_price = market_price;

            if (solver) {
                auto result = solver->solve_impl(query);
                if (result) {
                    pt.iv_mid = result->implied_vol;
                } else {
                    ++failed_count;
                }
            }

            smile.puts.push_back(pt);
        }

        if (!smile.calls.empty() || !smile.puts.empty()) {
            surface.smiles.push_back(std::move(smile));
        }
    }

    return surface;
}

}  // namespace mango::simple
