// SPDX-License-Identifier: MIT
#include "mango/option/price_table_factory.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/serialization/from_data.hpp"
#include "mango/option/table/certification/continuous_cell.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <algorithm>

using namespace mango;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
template<class T> void field(const std::string& key, const T& value) { std::cout << key << '\t' << value << '\n'; }

IVSolverFactoryConfig configuration(const std::string& id) {
    const bool short_term=id.starts_with("CS-");
    IVSolverFactoryConfig config;
    config.spot=100;
    config.option_type=id.ends_with("CALL") ? OptionType::CALL : OptionType::PUT;
    config.dividend_yield=id.starts_with("C0-") ? 0.0 : 0.02;
    config.grid.moneyness=short_term ? std::vector<double>{.9,.95,1,1.05,1.1}
                                   : std::vector<double>{.7,.8,.9,1,1.1,1.2,1.3};
    config.grid.vol={.05,.1,.2,.3,.5};
    config.grid.rate={-.05,0,.05,.1};
    config.backend=BSplineBackend{.maturity_grid=short_term
        ? std::vector<double>{1./365,2./365,4./365,7./365}
        : std::vector<double>{7./365,.1,.5,1,2}};
    config.adaptive=AdaptiveGridParams{};
    return config;
}

void metadata(const PriceTableData& data) {
    field("surface_type",data.surface_type);
    field("m_min",data.bounds_m_min); field("m_max",data.bounds_m_max);
    if (data.ratio_bounds) { field("ratio_min",data.ratio_bounds->min); field("ratio_max",data.ratio_bounds->max); }
    field("tau_min",data.bounds_tau_min); field("tau_max",data.bounds_tau_max);
    field("sigma_min",data.bounds_sigma_min); field("sigma_max",data.bounds_sigma_max);
    field("rate_min",data.bounds_rate_min); field("rate_max",data.bounds_rate_max);
    field("segments",data.segments.size());
    if (!data.segments.empty()) {
        const auto& segment=data.segments.front();
        field("K_ref",segment.K_ref); field("coefficients",segment.values.size());
        for (size_t d=0; d<segment.grids.size(); ++d) {
            field("grid_"+std::to_string(d)+"_size",segment.grids[d].size());
            field("grid_"+std::to_string(d)+"_min",segment.grids[d].front());
            field("grid_"+std::to_string(d)+"_max",segment.grids[d].back());
            for (size_t i=0;i<segment.grids[d].size();++i)
                field("grid_"+std::to_string(d)+"_node_"+std::to_string(i),segment.grids[d][i]);
        }
    }
}

void witness_evidence(AnyPriceTable& table, const PricingParams& p) {
    field("witness_spot",p.spot); field("witness_strike",p.strike);
    field("witness_tau",p.maturity); field("witness_sigma",p.volatility);
    field("witness_rate",std::get<double>(p.rate)); field("witness_q",p.dividend_yield);
    field("witness_type",p.option_type==OptionType::CALL ? "CALL" : "PUT");
    field("witness_admitted",table.validate_pricing_params(p).has_value());
    field("witness_price",table.price(p)); field("witness_vega",table.vega(p));
    const auto delta=table.delta(p), gamma=table.gamma(p), theta=table.theta(p), rho=table.rho(p);
    if (delta) field("witness_delta",*delta);
    if (gamma) field("witness_gamma",*gamma);
    if (theta) field("witness_theta",*theta);
    if (rho) field("witness_rho",*rho);
    for (double h : {1e-4,5e-5,1e-5}) {
        auto lower=p, upper=p; lower.volatility-=h; upper.volatility+=h;
        if (table.validate_pricing_params(lower) && table.validate_pricing_params(upper))
            field("witness_price_fd_vega_"+std::to_string(h),(table.price(upper)-table.price(lower))/(2*h));
    }
    // Convergence diagnostics only; these two profiles do not automatically
    // qualify an independent accuracy or Greek oracle.
    for (auto profile : {GridAccuracyProfile::High, GridAccuracyProfile::Ultra}) {
        const std::string label=profile==GridAccuracyProfile::High ? "high" : "ultra";
        auto solver=AmericanOptionSolver::create(p,PDEGridSpec{make_grid_accuracy(profile)});
        if (!solver) { field("direct_"+label+"_create_failed",true); continue; }
        auto result=solver->solve();
        if (!result) { field("direct_"+label+"_solve_failed",true); continue; }
        field("direct_"+label+"_price",result->value());
        field("direct_"+label+"_delta",result->delta());
        field("direct_"+label+"_gamma",result->gamma());
        field("direct_"+label+"_theta",result->theta());
    }
}

int main(int argc,char** argv) {
    if (argc!=4 && argc!=5) return 2;
    const std::string mode=argv[1], id=argv[2];
    if (id!="C0-CALL" && id!="CS-PUT" && id!="C2-PUT" && id!="CS-CALL") return 2;
    std::cout << std::setprecision(17) << std::unitbuf;
    const auto path=std::filesystem::path(argv[3])/(id+".parquet");
    field("case",id); field("mode",mode); field("payload",path.string());
    auto started=Clock::now();
    if (mode=="build") {
        auto config=configuration(id);
        const size_t rate_sites=argc==5 ? std::stoul(argv[4]) : 4;
        if (rate_sites!=4 && rate_sites!=7 && rate_sites!=13 && rate_sites!=25) return 2;
        if (rate_sites!=4 && id!="C0-CALL") return 2;
        while (config.grid.rate.size()<rate_sites) {
            auto expanded=config.grid.rate;
            for (size_t i=1;i<config.grid.rate.size();++i)
                expanded.push_back((config.grid.rate[i-1]+config.grid.rate[i])/2);
            std::sort(expanded.begin(),expanded.end());
            config.grid.rate=std::move(expanded);
        }
        field("rate_seed_count",config.grid.rate.size());
        for (size_t i=0;i<config.grid.rate.size();++i) field("rate_seed_"+std::to_string(i),config.grid.rate[i]);
        auto table=make_price_table(config);
        field("build_seconds",seconds(started));
        if (!table) { field("build_status","refused"); field("build_error",static_cast<int>(table.error().code)); return 3; }
        field("build_status","built");
        const auto data=table->to_data(); metadata(data);
        const auto& returned_rates=data.segments.front().grids[3];
        field("all_rate_seeds_retained",std::ranges::all_of(config.grid.rate,[&](double rate) {
            return std::ranges::find(returned_rates,rate)!=returned_rates.end();
        }));
        if (auto diagnostics=table->build_diagnostics()) {
            field("adaptive_target_met",diagnostics->target_met);
            field("adaptive_achieved_error",diagnostics->achieved_max_error);
            field("adaptive_iterations",diagnostics->total_iterations);
            field("adaptive_measured",diagnostics->holdout_points_measured);
            field("adaptive_invalid",diagnostics->holdout_points_invalid);
            field("reference_preparations",diagnostics->work.references.requests);
            field("failed_reference_preparations",diagnostics->work.references.failed_requests);
            const auto attempts=diagnostics->work.total_pde_attempts();
            field("actual_pde_work_known",attempts.has_value());
            if (attempts) field("actual_pde_attempts",*attempts);
        }
        auto saved=table->save(path,PriceTableCompression::NONE);
        field("save_ok",saved.has_value());
        return saved ? 0 : 4;
    }
    if (mode!="proof") return 2;
    auto table=load_price_table(path);
    if (!table) { field("load_status","failed"); return 5; }
    const auto data=table->to_data();
    auto typed=from_data<BSplineLeaf>(data);
    if (!typed) { field("reconstruct_status","failed"); return 6; }
    metadata(data);
    SurfaceBounds bounds{data.bounds_m_min,data.bounds_m_max,
        data.bounds_tau_min,data.bounds_tau_max,data.bounds_sigma_min,data.bounds_sigma_max,
        data.bounds_rate_min,data.bounds_rate_max,data.strike_bounds,data.ratio_bounds};
    field("proof_max_nodes",4096); field("proof_max_depth",24);
    started=Clock::now();
    const auto result=detail::certification::prove_continuous_bspline(
        typed->inner().interpolant().get(),data.segments.front().K_ref,data.option_type,
        data.dividend_yield,bounds,{.max_nodes=4096,.max_depth=24});
    field("proof_seconds",seconds(started));
    field("proof_status",result.status==PriceProofStatus::Certified ? "Certified"
        : result.status==PriceProofStatus::NegativeWitness ? "NegativeWitness" : "Indeterminate");
    field("proof_reason",static_cast<int>(result.reason)); field("proof_nodes",result.nodes);
    if (result.witness) {
        field("witness_bound_lower",result.witness_vega_per_strike.lower_bound());
        field("witness_bound_upper",result.witness_vega_per_strike.upper_bound());
        witness_evidence(*table,*result.witness);
    }
    return 0;
}
