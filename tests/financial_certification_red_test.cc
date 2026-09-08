// SPDX-License-Identifier: MIT
// Gate #459 public REDs. Explicit manual target until public activation follows
// the final #460/#461 parents; internal proof tests run independently.
#include "mango/option/table/bspline/bspline_surface.hpp"
#include <gtest/gtest.h>
#include <array>
#include <memory>
#include <limits>
#include "mango/option/price_table_factory.hpp"

namespace {
using namespace mango;

std::shared_ptr<const BSplineND<double,4>> hidden_pocket_eep() {
    const std::array<std::pair<double,double>,4> ranges{{
        {-.001,.001},{.1,.11},{.2,.20001},{.04,.06}}};
    BSplineND<double,4>::GridArray grid;
    BSplineND<double,4>::KnotArray knots;
    for (std::size_t d=0;d<4;++d) {
        const auto [lo,hi]=ranges[d];
        for (int i=0;i<4;++i) grid[d].push_back(lo+(hi-lo)*i/3);
        knots[d]={lo,lo,lo,lo,hi,hi,hi,hi};
    }
    constexpr double a=17.0/32, c=a*a-1.0/(128*128);
    // A positive, sub-dollar EEP; its narrow negative-vega pocket outweighs
    // the European vega at the middle of the admitted short-maturity domain.
    const std::array<double,4> sigma_coeff{.1,.1+c,.1+2*c-a,1.1+3*c-3*a};
    std::vector<double> coefficients(256);
    for (std::size_t i=0;i<256;++i) coefficients[i]=sigma_coeff[(i/4)%4];
    auto spline=BSplineND<double,4>::create(std::move(grid),std::move(knots),std::move(coefficients));
    if (!spline) return {};
    return std::make_shared<const BSplineND<double,4>>(std::move(*spline));
}

TEST(FinancialCertificationRedTest, ManualPublicationRejectsHiddenPhysicalVegaPocket) {
    auto spline=hidden_pocket_eep();
    ASSERT_TRUE(spline);
    BSplineLeaf raw(BSplineTransformLeaf(SharedBSplineInterp<4>(spline),StandardTransform4D{},100),
                    AnalyticalEEP(OptionType::PUT,0));
    constexpr double lo=.2, hi=.20001;
    const double center=lo+(hi-lo)*17/32;
    ASSERT_LT(raw.vega(100,100,.105,center,.05),0);
    for (int i=1;i<=16;++i) {
        const double before=lo+(hi-lo)*(i-1)/16;
        const double after=lo+(hi-lo)*i/16;
        ASSERT_GT(raw.price(100,100,.105,after,.05),raw.price(100,100,.105,before,.05));
    }
    auto published=make_bspline_surface(spline,100,0,OptionType::PUT);
    EXPECT_FALSE(published.has_value())
        << "Financial publication must prove physical sigma monotonicity, not accept a 17-point scan";
}
}

TEST(FinancialCertificationRedTest, FactoryAdmitsTheRequestedRatioEndpointBeforeLogRoundtrip) {
    using namespace mango;
    IVSolverFactoryConfig config;
    config.option_type=OptionType::PUT;
    config.spot=100;
    config.grid.moneyness={.1,.11,.12,.13};
    config.grid.vol={.18,.2,.22,.24};
    config.grid.rate={.02,.04,.06,.08};
    config.backend=BSplineBackend{.maturity_grid={.1,.2,.3,.4}};
    auto table=make_price_table(config);
    ASSERT_TRUE(table);
    PricingParams query(OptionSpec{.spot=10,.strike=100,.maturity=.2,.rate=.04,
        .dividend_yield=0,.option_type=OptionType::PUT},.2);
    EXPECT_TRUE(table->validate_pricing_params(query));
}

TEST(FinancialCertificationRedTest, QuoteRoundingCannotAdmitAnUnrelatedSubnormalRatio) {
    using namespace mango;
    auto spline=hidden_pocket_eep();
    ASSERT_TRUE(spline);
    BSplineLeaf raw(BSplineTransformLeaf(SharedBSplineInterp<4>(spline),StandardTransform4D{},100),
                    AnalyticalEEP(OptionType::PUT,0));
    const SurfaceBounds bounds{-.5,.5,.1,.11,.2,.20001,.04,.06};
    BSplinePriceTable table(std::move(raw),bounds,OptionType::PUT,0);
    const double quantum=std::numeric_limits<double>::denorm_min();
    ASSERT_EQ(quantum/(2*quantum),.5);
    EXPECT_FALSE(table.contains_moneyness(quantum,2*quantum));
}
