// SPDX-License-Identifier: MIT
#include "mango/option/price_table_factory.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/serialization/from_data.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace mango;
double direct(const PricingParams& p,GridAccuracyProfile profile) {
    auto solver=AmericanOptionSolver::create(p,PDEGridSpec{make_grid_accuracy(profile)});
    if (!solver) return std::numeric_limits<double>::quiet_NaN();
    auto result=solver->solve();
    return result ? result->value() : std::numeric_limits<double>::quiet_NaN();
}
int main(int argc,char** argv) {
    if (argc!=3) return 2;
    auto table=load_price_table(argv[1]); if (!table) return 3;
    const auto data=table->to_data(); auto typed=from_data<BSplineLeaf>(data);
    if (!typed || data.segments[0].grids[3].size()!=4) return 4;
    const auto& spline=typed->inner().interpolant().get();
    const auto& rates=data.segments[0].grids[3];
    AnalyticalEEP euro(data.option_type,data.dividend_yield);
    std::cout<<std::setprecision(17)<<std::unitbuf;
    std::cout<<"id\tlocation\trate\tlagrange_weight\traw_fitted_premium_Kref\tquote_european\tdirect_high\tdirect_ultra\treference_premium_Kref\treference_basis\n";
    std::ifstream input(argv[2]);std::string id;
    double S,K,tau,sigma,r,reference,gd,gg,gt,gr;int price_ok,ivkind,qd,qg,qt,qr;
    while (input>>id>>S>>K>>tau>>sigma>>r>>reference>>price_ok>>ivkind>>gd>>qd>>gg>>qg>>gt>>qt>>gr>>qr) {
        PricingParams p(OptionSpec{.spot=S,.strike=K,.maturity=tau,.rate=r,
            .dividend_yield=data.dividend_yield,.option_type=data.option_type},sigma);
        for (size_t i=0;i<=rates.size();++i) {
            const bool query=i==rates.size(); const double rate=query?r:rates[i]; p.rate=rate;
            double weight=1;
            if (!query) for(size_t j=0;j<rates.size();++j) if(i!=j) weight*=(r-rates[j])/(rates[i]-rates[j]);
            const double raw=spline.eval({std::log(S/K),tau,sigma,rate});
            const double ep=euro.european_price(S,K,tau,sigma,rate);
            const double high=direct(p,GridAccuracyProfile::High),ultra=direct(p,GridAccuracyProfile::Ultra);
            const bool exact=rate>=0 && data.dividend_yield==0 && data.option_type==OptionType::CALL;
            std::cout<<id<<'\t'<<(query?"query":"node")<<'\t'<<rate<<'\t'<<weight<<'\t'<<raw
                <<'\t'<<ep<<'\t'<<high<<'\t'<<ultra<<'\t'
                <<(exact?0:(ultra-ep)*data.segments[0].K_ref/K)<<'\t'
                <<(exact?"analytic_zero":"Ultra_FDE_diagnostic")<<'\n';
        }
    }
    return input.eof()?0:5;
}
