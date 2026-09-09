// SPDX-License-Identifier: MIT
#include "mango/option/price_table_factory.hpp"
#include "mango/option/table/parquet/parquet_io.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/serialization/from_data.hpp"
#include "mango/option/table/certification/continuous_cell.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace mango;
template<class T> void field(const char* key,const T& value) { std::cout<<key<<'\t'<<value<<'\n'; }
double greek(const std::expected<double,GreekError>& v) { return v ? *v : std::numeric_limits<double>::quiet_NaN(); }

int main(int argc,char** argv) {
    if (argc!=5) return 2;
    const double delta=std::stod(argv[2]);
    if (!std::isfinite(delta) || delta<0) return 2;
    const std::string prefix=argv[3];
    std::cout<<std::setprecision(17)<<std::unitbuf;
    auto original=read_parquet(argv[1]); auto baseline=load_price_table(argv[1]);
    if (!original || !baseline || original->segments.size()!=1) return 3;
    auto changed=*original;
    long double max_shift=0,min_shift=std::numeric_limits<long double>::infinity(),rounding=0;
    size_t changed_count=0;
    for (auto& coefficient:changed.segments[0].values) {
        const double old=coefficient; coefficient=old-delta;
        const long double shift=static_cast<long double>(old)-coefficient;
        max_shift=std::max(max_shift,shift); min_shift=std::min(min_shift,shift);
        rounding=std::max(rounding,std::abs(shift-static_cast<long double>(delta)));
        changed_count+=coefficient!=old;
    }
    field("requested_raw_dollar_shift",delta); field("actual_shift_min",min_shift);
    field("actual_shift_max",max_shift); field("max_shift_rounding",rounding);
    field("changed_coefficients",changed_count); field("coefficient_count",changed.segments[0].values.size());
    field("required_K_scale_max",1.2);
    field("coefficient_price_movement_bound",max_shift*1.2L);
    if (!write_parquet(changed,prefix+".parquet",{.compression=ParquetCompression::NONE})) return 4;
    auto candidate=load_price_table(prefix+".parquet"); auto typed=from_data<BSplineLeaf>(changed);
    if (!candidate || !typed) return 5;
    const SurfaceBounds bounds{changed.bounds_m_min,changed.bounds_m_max,changed.bounds_tau_min,
        changed.bounds_tau_max,changed.bounds_sigma_min,changed.bounds_sigma_max,
        changed.bounds_rate_min,changed.bounds_rate_max,changed.strike_bounds,changed.ratio_bounds};
    const auto started=std::chrono::steady_clock::now();
    const auto proof=detail::certification::prove_continuous_bspline(
        typed->inner().interpolant().get(),changed.segments[0].K_ref,changed.option_type,
        changed.dividend_yield,bounds,{.max_nodes=4096,.max_depth=24});
    field("proof_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());
    field("proof_status",proof.status==PriceProofStatus::Certified ? "Certified"
        : proof.status==PriceProofStatus::NegativeWitness ? "NegativeWitness" : "Indeterminate");
    field("proof_nodes",proof.nodes); field("proof_reason",static_cast<int>(proof.reason));
    if (proof.witness) {
        const auto& w=*proof.witness;
        field("witness_bound_lower",proof.witness_vega_per_strike.lower_bound());
        field("witness_bound_upper",proof.witness_vega_per_strike.upper_bound());
        field("witness_spot",w.spot); field("witness_strike",w.strike);
        field("witness_tau",w.maturity); field("witness_sigma",w.volatility);
        field("witness_rate",std::get<double>(w.rate));
        field("witness_admitted",candidate->validate_pricing_params(w).has_value());
        field("witness_price",candidate->price(w)); field("witness_vega",candidate->vega(w));
    }
    auto old_iv=baseline->make_iv_solver(),new_iv=candidate->make_iv_solver();
    std::ifstream input(argv[4]); std::ofstream output(prefix+".rows.tsv");
    output<<std::setprecision(17);
    output<<"id\tadmitted\tbaseline_price\tcandidate_price\tbaseline_delta\tcandidate_delta\tbaseline_gamma\tcandidate_gamma\tbaseline_theta\tcandidate_theta\tbaseline_rho\tcandidate_rho\tbaseline_iv\tcandidate_iv\tbaseline_iv_error_code\tcandidate_iv_error_code\n";
    std::string id;
    double spot,strike,tau,sigma,rate,reference,gd,gg,gt,gr; int price_ok,iv_kind,qd,qg,qt,qr;
    size_t rows=0;
    while (input>>id>>spot>>strike>>tau>>sigma>>rate>>reference>>price_ok>>iv_kind>>gd>>qd>>gg>>qg>>gt>>qt>>gr>>qr) {
        PricingParams p(OptionSpec{.spot=spot,.strike=strike,.maturity=tau,.rate=rate,
            .dividend_yield=changed.dividend_yield,.option_type=changed.option_type},sigma);
        const bool admitted=candidate->validate_pricing_params(p).has_value()
            && baseline->validate_pricing_params(p).has_value();
        const double nan=std::numeric_limits<double>::quiet_NaN();
        double old_vol=nan,new_vol=nan; int old_error=-1,new_error=-1;
        if (iv_kind==1 && price_ok && admitted) {
            IVQuery query(p,reference,{});
            if (old_iv) { auto value=old_iv->solve(query); if (value) old_vol=value->implied_vol; else old_error=static_cast<int>(value.error().code); }
            if (new_iv) { auto value=new_iv->solve(query); if (value) new_vol=value->implied_vol; else new_error=static_cast<int>(value.error().code); }
        }
        output<<id<<'\t'<<admitted<<'\t'<<(admitted?baseline->price(p):nan)<<'\t'<<(admitted?candidate->price(p):nan)
            <<'\t'<<greek(baseline->delta(p))<<'\t'<<greek(candidate->delta(p))
            <<'\t'<<greek(baseline->gamma(p))<<'\t'<<greek(candidate->gamma(p))
            <<'\t'<<greek(baseline->theta(p))<<'\t'<<greek(candidate->theta(p))
            <<'\t'<<greek(baseline->rho(p))<<'\t'<<greek(candidate->rho(p))
            <<'\t'<<old_vol<<'\t'<<new_vol<<'\t'<<old_error<<'\t'<<new_error<<'\n';
        ++rows;
    }
    field("queried_rows",rows);
    return input.eof() ? 0 : 6;
}
