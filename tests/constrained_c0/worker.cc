// SPDX-License-Identifier: MIT
#include "mango/math/bspline/bspline_basis.hpp"
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/parquet/parquet_io.hpp"
#include "tests/price_table_builder_test_access.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace mango;
std::vector<double> rate_grid() { return {-.05, -.0375, -.025, 0, .05, .1}; }
std::vector<double> rate_knots() { return {-.05, -.05, -.05, -.05, 0, 0, .1, .1, .1, .1}; }
std::vector<double> midpoints(std::vector<double> x) {
  auto y = x;
  for (size_t i = 1; i < x.size(); ++i)
    y.push_back((x[i - 1] + x[i]) / 2);
  std::sort(y.begin(), y.end());
  return y;
}
std::vector<BSplineND<double, 1>> basis(const std::vector<double> &grid,
                                        const std::vector<double> &knots) {
  std::vector<BSplineND<double, 1>> out;
  for (size_t k = 0; k < grid.size(); ++k) {
    std::vector<double> c(grid.size(), 0);
    c[k] = 1;
    out.push_back(
        BSplineND<double, 1>::create({grid}, {knots}, std::move(c)).value());
  }
  return out;
}
int main(int argc, char **argv) {
  std::cout << std::setprecision(17) << std::unitbuf;
  if (argc == 2 && std::string(argv[1]) == "representation") {
    auto s = BSplineND<double, 1>::create({rate_grid()}, {rate_knots()},
                                          {2, 1, 0, 0, 0, 0});
    if (!s)
      return 3;
    for (double r : {0., 1e-6, .025, .1})
      if (s->eval({r}) != 0 || s->partial(0, {r}) != 0) {
        std::cout << "nonzero_analytic_branch\t" << r << '\t' << s->eval({r})
                  << '\n';
        return 9;
      }
    if (std::abs(s->eval({-.025}) - .625) > 1e-14 ||
        std::abs(s->partial(0, {-.025}) + 45) > 1e-12)
      return 10;
    std::cout << "representation\tpassed\n";
    return 0;
  }
  if (argc != 4 && argc != 5)
    return 2;
  const std::string mode = argv[1], prefix = argv[3];
  auto data = read_parquet(argv[2]);
  if (!data || data->segments.size() != 1)
    return 3;
  auto &seg = data->segments[0];
  const auto start = std::chrono::steady_clock::now();
  if (mode == "compose" && argc == 5) {
    std::ifstream in(argv[4]);
    std::vector<double> c;
    double value;
    while (in >> value)
      c.push_back(value);
    const auto expected =
        seg.grids[0].size() * seg.grids[1].size() * seg.grids[2].size() * 6;
    if (!in.eof() || c.size() != expected)
      return 4;
    seg.grids[3] = rate_grid();
    seg.knots[3] = rate_knots();
    seg.values = std::move(c);
    if (!write_parquet(*data, prefix + ".parquet",
                       {.compression = ParquetCompression::NONE}))
      return 5;
    std::cout << "coefficients\t" << expected << "\ncompose_seconds\t"
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << '\n';
    return 0;
  }
  if (mode != "sample")
    return 2;
  const std::vector<double> rates{-.05, -.0375, -.025, -.0125};
  auto setup = PriceTableBuilder::from_vectors(
      seg.grids[0], seg.grids[1], seg.grids[2], rates, 100,
      GridAccuracyParams{}, OptionType::CALL, 0, 0);
  if (!setup)
    return 6;
  auto &[builder, axes] = *setup;
  auto params = builder.make_batch(axes);
  auto batch =
      testing::PriceTableBuilderAccess<4>::solve_batch(builder, params, axes);
  std::cout << "sampling_parameter_combinations\t" << params.size()
            << "\nfailed_pde_results\t" << batch.failed_count << '\n';
  if (batch.failed_count)
    return 7;
  auto extraction = builder.extract_tensor(batch, axes);
  if (!extraction)
    return 8;
  std::cout << "failed_extraction_pde\t" << extraction->failed_pde.size()
            << "\nfailed_extraction_spline\t"
            << extraction->failed_spline.size() << '\n';
  if (!extraction->failed_pde.empty() || !extraction->failed_spline.empty())
    return 8;
  AnalyticalEEP euro(OptionType::CALL, 0);
  std::ofstream samples(prefix + ".samples.tsv"), cuts(prefix + ".cuts.tsv"),
      grids(prefix + ".axes.tsv");
  samples << std::setprecision(17);
  cuts << std::setprecision(17);
  grids << std::setprecision(17);
  for (size_t d = 0; d < 3; ++d) {
    auto b = basis(seg.grids[d], seg.knots[d]);
    std::ofstream matrix(prefix + ".matrix" + std::to_string(d) + ".tsv");
    matrix << std::setprecision(17);
    for (size_t i = 0; i < seg.grids[d].size(); ++i) {
      grids << d << '\t' << i << '\t' << seg.grids[d][i] << '\n';
      for (size_t j = 0; j < b.size(); ++j)
        matrix << (j ? "\t" : "") << b[j].eval({seg.grids[d][i]});
      matrix << '\n';
    }
  }
  auto sg = midpoints(seg.grids[2]);
  auto rg = rates;
  rg.push_back(0);
  rg = midpoints(rg);
  auto sb = basis(seg.grids[2], seg.knots[2]);
  std::ofstream sigma_basis(prefix + ".sigma_cuts.tsv");
  sigma_basis << std::setprecision(17);
  for (double sig : sg) {
    sigma_basis << sig;
    for (auto &b : sb)
      sigma_basis << '\t' << b.eval({sig});
    for (auto &b : sb)
      sigma_basis << '\t' << b.partial(0, {sig});
    sigma_basis << '\n';
  }
  for (size_t i = 0; i < seg.grids[0].size(); ++i)
    for (size_t j = 0; j < seg.grids[1].size(); ++j) {
      const double spot = 100 * std::exp(seg.grids[0][i]),
                   tau = seg.grids[1][j];
      for (size_t s = 0; s < seg.grids[2].size(); ++s)
        for (size_t r = 0; r < rates.size(); ++r) {
          const double sigma = seg.grids[2][s], rate = rates[r];
          const double am = 100 * extraction->tensor.view[i, j, s, r];
          const double eu = euro.european_price(spot, 100, tau, sigma, rate);
          if (!std::isfinite(am) || !std::isfinite(eu))
            return 11;
          samples << i << '\t' << j << '\t' << s << '\t' << r << '\t'
                  << std::max(0., am - eu) << '\t' << am << '\t' << eu << '\n';
        }
      for (size_t s = 0; s < sg.size(); ++s)
        for (size_t r = 0; r < rg.size(); ++r)
          cuts << i << '\t' << j << '\t' << s << '\t' << r << '\t' << rg[r]
               << '\t' << euro.european_vega(spot, 100, tau, sg[s], rg[r])
               << '\n';
    }
  std::cout << "sample_rows\t"
            << seg.grids[0].size() * seg.grids[1].size() * seg.grids[2].size() *
                   4
            << "\nsampling_seconds\t"
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             start)
                   .count()
            << '\n';
}
