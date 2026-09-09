// SPDX-License-Identifier: MIT
#include "mango/option/table/accuracy/population.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>

namespace mango::detail::accuracy {
namespace {
using Interval=std::pair<double,double>;
bool ordered_finite(double lo,double hi) {
    return std::isfinite(lo) && std::isfinite(hi) && lo<=hi;
}
std::vector<double> endpoints(double lo,double hi) {
    return lo==hi ? std::vector{lo} : std::vector{lo,hi};
}
// Exclude all known possible reference nodes before candidate assessment. A
// representable interior point is selected in the closest available gap.
double interior_strike(double lo,double hi,double fraction,std::span<const double> excluded) {
    const double desired=std::lerp(lo,hi,fraction);
    if (desired>lo && desired<hi && !std::binary_search(excluded.begin(),excluded.end(),desired))
        return desired;
    double best=desired, distance=std::numeric_limits<double>::infinity();
    double left=lo;
    auto consider=[&](double right) {
        const double middle=std::midpoint(left,right);
        if (middle>left && middle<right && std::abs(middle-desired)<distance) {
            best=middle; distance=std::abs(middle-desired);
        }
        left=right;
    };
    for (double value:excluded) if (value>lo && value<hi) consider(value);
    consider(hi);
    return best;
}
} // namespace

std::expected<AccuracyPopulation,PopulationFailure>
make_accuracy_population(const PopulationRequest& request) {
    const auto fail=[&](PopulationFailureReason reason,size_t required=0,bool exact=true) {
        return std::unexpected(PopulationFailure{reason,required,exact,request.limits,{}});
    };
    const auto& b=request.domain;
    if (!request.limits.valid() || request.max_moneyness_nodes==0 ||
        request.max_moneyness_nodes==std::numeric_limits<size_t>::max() ||
        !ordered_finite(b.m_min,b.m_max) || !ordered_finite(b.tau_min,b.tau_max) ||
        b.tau_min<0. || b.tau_max<=0. || !ordered_finite(b.sigma_min,b.sigma_max) ||
        b.sigma_min<=0. || !ordered_finite(b.rate_min,b.rate_max) ||
        !std::isfinite(request.spot) || request.spot<=0. ||
        !std::isfinite(request.dividend_yield) || request.dividend_yield<0. ||
        (request.option_type!=OptionType::CALL && request.option_type!=OptionType::PUT))
        return fail(PopulationFailureReason::InvalidInput);
    const auto ratios=b.ratio_bounds.value_or(MoneynessBounds{std::exp(b.m_min),std::exp(b.m_max)});
    if (!ratios.valid()) return fail(PopulationFailureReason::InvalidInput);
    const auto strikes=b.strike_bounds.value_or(
        StrikeBounds{request.spot/ratios.max,request.spot/ratios.min});
    if (!strikes.valid()) return fail(PopulationFailureReason::InvalidInput);
    auto excluded=request.possible_reference_strikes;
    if (!std::ranges::all_of(excluded,[](double k) { return std::isfinite(k) && k>0.; }))
        return fail(PopulationFailureReason::InvalidInput);
    std::ranges::sort(excluded);
    excluded.erase(std::unique(excluded.begin(),excluded.end()),excluded.end());

    std::vector<Interval> intervals;
    if (request.fixed_expiry) {
        const auto& fixed=*request.fixed_expiry;
        if (!fixed.valid(b.tau_max)) return fail(PopulationFailureReason::InvalidInput);
        const auto boundaries=compute_segment_boundaries(fixed.discrete_dividends,
            fixed.reference_maturity,0.,fixed.reference_maturity);
        const auto split=make_tau_split_from_segments(boundaries.bounds,boundaries.is_gap,1.);
        if (b.tau_min==b.tau_max) {
            if (split.contains_maturity(b.tau_min)) intervals.emplace_back(b.tau_min,b.tau_max);
        } else intervals=admitted_maturity_intervals(split,b.tau_min,b.tau_max);
    } else intervals.emplace_back(b.tau_min,b.tau_max);
    if (intervals.empty()) return fail(PopulationFailureReason::UnrepresentableDomain);
    const double first_positive=intervals.front().first>0. ? intervals.front().first
        : std::max(std::lerp(0.,intervals.front().second,1e-4),
                   std::nextafter(0.,intervals.front().second));
    const auto admitted=[&](double tau) {
        return tau>0. && std::ranges::any_of(intervals,[&](const auto& part) {
            return tau>=part.first && tau<=part.second;
        });
    };
    if (!admitted(first_positive)) return fail(PopulationFailureReason::UnrepresentableDomain);

    const double alpha=(3.-std::sqrt(3.))/6., beta=1.-alpha;
    const std::array interior_m{
        std::lerp(ratios.min,ratios.max,alpha),std::midpoint(ratios.min,ratios.max),
        std::lerp(ratios.min,ratios.max,beta)};
    const std::array interior_k{
        interior_strike(strikes.min,strikes.max,alpha,excluded),
        interior_strike(strikes.min,strikes.max,beta,excluded)};
    using Key=std::tuple<double,double,double,double,double>;
    std::map<Key,PopulationRow> rows;
    bool representable=true;
    bool limit_exceeded=false;
    const auto add=[&](double m,double tau,double sigma,double rate,double strike,
                       AccuracyStratum stratum,PopulationExpectation expectation) {
        if (limit_exceeded) return;
        const double spot=m*strike;
        if (!std::isfinite(spot) || spot<=0. || !std::isfinite(tau)) { representable=false; return; }
        const Key key{spot,strike,tau,sigma,rate};
        auto [it,inserted]=rows.try_emplace(key);
        auto& row=it->second;
        if (inserted) {
            row.query=PricingParams(OptionSpec{.spot=spot,.strike=strike,.maturity=tau,.rate=rate,
                .dividend_yield=request.dividend_yield,.option_type=request.option_type},sigma);
            if (request.fixed_expiry) row.query.discrete_dividends=rolled_dividends(
                request.fixed_expiry->discrete_dividends,
                request.fixed_expiry->reference_maturity,tau);
            row.expectation=expectation;
        } else if (expectation==PopulationExpectation::RequiredPrice)
            row.expectation=expectation;
        row.strata|=static_cast<uint32_t>(stratum);
        limit_exceeded=rows.size()>request.limits.max_population_rows;
    };
    for (double m:endpoints(ratios.min,ratios.max))
        for (double tau:endpoints(first_positive,intervals.back().second))
            for (double sigma:endpoints(b.sigma_min,b.sigma_max))
                for (double rate:endpoints(b.rate_min,b.rate_max))
                    for (double strike:endpoints(strikes.min,strikes.max))
                        add(m,tau,sigma,rate,strike,AccuracyStratum::GlobalCorner,
                            PopulationExpectation::RequiredPrice);
    if (limit_exceeded) return fail(PopulationFailureReason::RowLimit,rows.size(),false);
    const auto interior_tau=[](double lo,double hi,double fraction) {
        const double tau=std::lerp(lo,hi,fraction);
        return tau>0. ? tau : std::nextafter(0.,hi);
    };
    for (const auto& [lo,hi]:intervals) {
        for (size_t i=0;i<2;++i) {
            const double u=i ? beta : alpha;
            for (double m:interior_m) add(m,interior_tau(lo,hi,u),
                std::lerp(b.sigma_min,b.sigma_max,u),std::lerp(b.rate_min,b.rate_max,1.-u),
                interior_k[i],AccuracyStratum::RegimeInterior,PopulationExpectation::RequiredPrice);
        }
        if (b.rate_min<0. && b.rate_max>0.)
            for (double m:interior_m) add(m,interior_tau(lo,hi,.5),
                std::midpoint(b.sigma_min,b.sigma_max),0.,interior_k[0],
                AccuracyStratum::ZeroRate,PopulationExpectation::RequiredPrice);
        if (limit_exceeded) return fail(PopulationFailureReason::RowLimit,rows.size(),false);
    }
    if (request.fixed_expiry) {
        const double sigma=std::midpoint(b.sigma_min,b.sigma_max),
            rate=std::midpoint(b.rate_min,b.rate_max);
        for (const auto& event:request.fixed_expiry->discrete_dividends) {
            const double tau=request.fixed_expiry->reference_maturity-event.calendar_time;
            for (double offset:{-1./365.,1./365.})
                for (double m:interior_m) add(m,tau+offset,sigma,rate,interior_k[0],
                    AccuracyStratum::EventSide,admitted(tau+offset)
                        ? PopulationExpectation::RequiredPrice : PopulationExpectation::Refusal);
            for (double offset:{-1e-6,0.,1e-6})
                add(interior_m[1],tau+offset,sigma,rate,interior_k[0],
                    AccuracyStratum::EventAdmission,PopulationExpectation::Admission);
            if (limit_exceeded) return fail(PopulationFailureReason::RowLimit,rows.size(),false);
        }
    }
    if (b.tau_min==0.) add(interior_m[1],0.,std::midpoint(b.sigma_min,b.sigma_max),
        std::midpoint(b.rate_min,b.rate_max),interior_k[0],AccuracyStratum::ExpiryAdmission,
        PopulationExpectation::Refusal);

    if (limit_exceeded) return fail(PopulationFailureReason::RowLimit,rows.size(),false);
    const size_t strip_count=request.max_moneyness_nodes+1;
    if (ratios.min<ratios.max) {
        const auto [lo,hi]=intervals.back();
        const auto ratio_at=[&](size_t i) {
            return std::lerp(ratios.min,ratios.max,
                static_cast<double>(i)/(static_cast<double>(strip_count)+1.));
        };
        const auto spot_at=[&](size_t i) { return ratio_at(i)*interior_k[0]; };
        size_t i=1;
        while (true) {
            const double m=ratio_at(i), spot=spot_at(i);
            if (m>ratios.min && m<ratios.max) add(m,interior_tau(lo,hi,alpha),
                std::lerp(b.sigma_min,b.sigma_max,alpha),std::lerp(b.rate_min,b.rate_max,beta),
                interior_k[0],AccuracyStratum::MoneynessStrip,PopulationExpectation::RequiredPrice);
            const bool finished=i==strip_count || spot_at(strip_count)<=spot;
            if (limit_exceeded)
                return fail(PopulationFailureReason::RowLimit,rows.size(),finished);
            if (finished) break;
            // Lerp and multiplication by positive K are monotone. Skip equal
            // rounded physical spots by bounded binary search instead of
            // iterating a huge node ceiling over a tiny representable domain.
            size_t next=i+1, last=strip_count;
            while (next<last) {
                const size_t mid=next+(last-next)/2;
                if (spot_at(mid)<=spot) next=mid+1;
                else last=mid;
            }
            i=next;
        }
    }
    if (!representable) return fail(PopulationFailureReason::UnrepresentableDomain);
    if (rows.size()>request.limits.max_population_rows)
        return fail(PopulationFailureReason::RowLimit,rows.size());
    AccuracyPopulationCounts counts;
    counts.declared_rows=rows.size(); counts.temporal_regimes=intervals.size();
    counts.declared_moneyness_node_limit=request.max_moneyness_nodes;
    std::vector<PopulationRow> owned;
    owned.reserve(rows.size());
    for (auto& [key,row]:rows) {
        counts.pricing_rows+=row.expectation==PopulationExpectation::RequiredPrice;
        counts.admission_rows+=row.expectation==PopulationExpectation::Admission;
        counts.refusal_rows+=row.expectation==PopulationExpectation::Refusal;
        counts.moneyness_strip_rows+=has_stratum(row.strata,AccuracyStratum::MoneynessStrip);
        owned.push_back(std::move(row));
    }
    counts.off_node_moneyness_available=counts.moneyness_strip_rows>request.max_moneyness_nodes;
    return AccuracyPopulation(std::move(owned),request.limits,counts);
}

} // namespace mango::detail::accuracy
