// SPDX-License-Identifier: MIT
// Arithmetic-only characterization. This does not certify financial tables.
#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/chebyshev_polynomial.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"
#include "mango/math/proof/chebyshev.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using mango::detail::ChebyshevPolynomial;
template <std::size_t N> using Nodal = mango::ChebyshevInterpolant<N, mango::RawTensor<N>>;
double halton(std::size_t n, std::size_t base) {
    double value = 0, factor = 1;
    while (n) {
        factor /= base;
        value += factor * (n % base);
        n /= base;
    }
    return value;
}
template <class F> double latency(F function, std::size_t count) {
    std::array<double, 3> trials{};
    double sum = 0;
    for (auto &trial : trials) {
        const auto start = Clock::now();
        for (std::size_t i = 0; i < count; ++i)
            sum += function(i);
        trial = std::chrono::duration<double, std::nano>(Clock::now() - start).count() / count;
    }
    std::sort(trials.begin(), trials.end());
    // Observable use prevents dead-code elimination in the standalone binary.
    if (!std::isfinite(sum))
        std::cerr << "nonfinite checksum " << sum << '\n';
    return trials[1];
}
void one_dimensional(std::size_t n) {
    const mango::Domain<1> domain{{-1}, {1}};
    const auto nodes = mango::chebyshev_nodes(n, -1, 1);
    std::vector<double> values;
    for (double x : nodes)
        values.push_back(.5 + 2 * x + 3 * x * x + 4 * x * x * x);
    auto start = Clock::now();
    const auto old = Nodal<1>::build_from_values(values, domain, {n}).value();
    const double old_build =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    start = Clock::now();
    const auto modal = ChebyshevPolynomial::from_cgl_values({n}, {{-1, 1}}, values).value();
    const double modal_build =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::vector<std::array<double, 1>> queries{
        {-1}, {1}, {std::nextafter(-1., 0.)}, {std::nextafter(1., 0.)}, {0}};
    for (std::size_t i = 1; i <= 600; ++i)
        queries.push_back({2 * halton(i, 2) - 1});
    std::array<double, 3> old_error{}, modal_error{};
    for (const auto &q : queries) {
        const long double x = q[0];
        const std::array<double, 3> truth{
            static_cast<double>(.5L + 2 * x + 3 * x * x + 4 * x * x * x),
            static_cast<double>(2 + 6 * x + 12 * x * x), static_cast<double>(6 + 24 * x)};
        const std::array<double, 3> old_values{old.eval(q), old.partial(0, q),
                                               old.eval_second_partial(0, q)};
        const std::array<double, 3> modal_values{modal.eval(q), modal.partial(0, q),
                                                 modal.eval_second_partial(0, q)};
        for (std::size_t d = 0; d < 3; ++d) {
            old_error[d] = std::max(old_error[d], std::abs(old_values[d] - truth[d]));
            modal_error[d] = std::max(modal_error[d], std::abs(modal_values[d] - truth[d]));
        }
    }
    double nodal_residual = 0;
    for (std::size_t i = 0; i < n; ++i)
        nodal_residual = std::max(
            nodal_residual, std::abs(modal.eval(std::array<double, 1>{nodes[i]}) - values[i]));
    const auto count = std::size_t{2000};
    std::cout << "1d nodes=" << n << " payload_bytes=" << n * sizeof(double)
              << " build_ms=" << old_build << ',' << modal_build
              << " nodal_residual=" << nodal_residual << '\n';
    std::cout << " max_error value/first/second old=" << old_error[0] << ',' << old_error[1] << ','
              << old_error[2] << " modal=" << modal_error[0] << ',' << modal_error[1] << ','
              << modal_error[2] << '\n';
    std::cout
        << " query_ns value/first/second old="
        << latency([&](auto i) { return old.eval(queries[i % queries.size()]); }, count) << ','
        << latency([&](auto i) { return old.partial(0, queries[i % queries.size()]); }, count)
        << ','
        << latency([&](auto i) { return old.eval_second_partial(0, queries[i % queries.size()]); },
                   count)
        << " modal="
        << latency([&](auto i) { return modal.eval(queries[i % queries.size()]); }, count) << ','
        << latency([&](auto i) { return modal.partial(0, queries[i % queries.size()]); }, count)
        << ','
        << latency(
               [&](auto i) { return modal.eval_second_partial(0, queries[i % queries.size()]); },
               count)
        << '\n';
}
void four_dimensional() {
    const std::array<std::size_t, 4> shape{257, 9, 9, 5};
    const mango::Domain<4> domain{{-.5, .02, .05, -.02}, {.5, .8, .45, .08}};
    std::array<std::vector<double>, 4> nodes;
    std::vector<ChebyshevPolynomial::Bounds> bounds;
    std::size_t total = 1;
    for (std::size_t d = 0; d < 4; ++d) {
        nodes[d] = mango::chebyshev_nodes(shape[d], domain.lo[d], domain.hi[d]);
        bounds.emplace_back(domain.lo[d], domain.hi[d]);
        total *= shape[d];
    }
    auto function = [](const std::array<double, 4> &x) {
        return 10 * std::exp(.3 * x[0]) * (1 + x[1]) * (1 + x[2] + .2 * x[2] * x[2]) * (1 + x[3]);
    };
    std::vector<double> values(total);
    for (std::size_t i = 0; i < total; ++i) {
        std::array<double, 4> q{};
        auto index = i;
        for (std::size_t d = 4; d-- > 0;) {
            q[d] = nodes[d][index % shape[d]];
            index /= shape[d];
        }
        values[i] = function(q);
    }
    auto start = Clock::now();
    const auto old = Nodal<4>::build_from_values(values, domain, shape).value();
    const double old_build =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    start = Clock::now();
    const auto modal = ChebyshevPolynomial::from_cgl_values({257, 9, 9, 5}, bounds, values).value();
    const double modal_build =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::vector<std::array<double, 4>> queries;
    for (std::size_t i = 0; i < 616; ++i) {
        std::array<double, 4> q{};
        constexpr std::array<std::size_t, 4> primes{2, 3, 5, 7};
        for (std::size_t d = 0; d < 4; ++d) {
            const double u = i < 16 ? static_cast<double>((i >> d) & 1) : halton(i - 15, primes[d]);
            q[d] = domain.lo[d] + u * (domain.hi[d] - domain.lo[d]);
        }
        queries.push_back(q);
    }
    std::array<double, 3> old_error{}, modal_error{};
    for (const auto &q : queries) {
        const double price = function(q);
        const double vega = 10 * std::exp(.3 * q[0]) * (1 + q[1]) * (1 + .4 * q[2]) * (1 + q[3]);
        const std::array<double, 3> truth{price, vega, .09 * price};
        const std::array<double, 3> old_values{old.eval(q), old.partial(2, q),
                                               old.eval_second_partial(0, q)};
        const std::array<double, 3> modal_values{modal.eval(q), modal.partial(2, q),
                                                 modal.eval_second_partial(0, q)};
        for (std::size_t d = 0; d < 3; ++d) {
            old_error[d] = std::max(old_error[d], std::abs(old_values[d] - truth[d]));
            modal_error[d] = std::max(modal_error[d], std::abs(modal_values[d] - truth[d]));
        }
    }
    std::cout << "4d shape=257x9x9x5 payload_bytes=" << total * sizeof(double)
              << " build_ms=" << old_build << ',' << modal_build << '\n';
    std::cout << " max_error value/vol_partial/m_second old=" << old_error[0] << ',' << old_error[1]
              << ',' << old_error[2] << " modal=" << modal_error[0] << ',' << modal_error[1] << ','
              << modal_error[2] << '\n';
    std::cout
        << " query_ns value/vol_partial/m_second old="
        << latency([&](auto i) { return old.eval(queries[i % queries.size()]); }, 1000) << ','
        << latency([&](auto i) { return old.partial(2, queries[i % queries.size()]); }, 1000) << ','
        << latency([&](auto i) { return old.eval_second_partial(0, queries[i % queries.size()]); },
                   1000)
        << " modal="
        << latency([&](auto i) { return modal.eval(queries[i % queries.size()]); }, 1000) << ','
        << latency([&](auto i) { return modal.partial(2, queries[i % queries.size()]); }, 1000)
        << ','
        << latency(
               [&](auto i) { return modal.eval_second_partial(0, queries[i % queries.size()]); },
               1000)
        << '\n';
    start = Clock::now();
    const auto proof =
        mango::detail::proof::prove_chebyshev_partial(modal, 2, {.max_nodes = 64}).value();
    std::cout << " modal_vol_proof status=" << static_cast<int>(proof.status)
              << " nodes=" << proof.nodes
              << " ms=" << std::chrono::duration<double, std::milli>(Clock::now() - start).count()
              << '\n';
}
void proof_vectors() {
    std::vector<double> coefficients(257);
    for (std::size_t i = 0; i < coefficients.size(); ++i)
        coefficients[i] = (static_cast<int>(i % 17) - 8) / 1024.0;
    const auto polynomial =
        ChebyshevPolynomial::from_coefficients({257}, {{-2, 2}}, coefficients).value();
    for (int i = 0; i <= 64; ++i) {
        const double u = i / 64.0;
        const mango::detail::proof::UnitBox box{{u, u}};
        const auto value = mango::detail::proof::enclose_chebyshev(polynomial, box).value();
        const auto derivative = mango::detail::proof::enclose_chebyshev(polynomial, box, 0).value();
        std::cout << "proof_vector " << std::hexfloat << u << ' ' << value.lower_bound() << ' '
                  << value.upper_bound() << ' ' << derivative.lower_bound() << ' '
                  << derivative.upper_bound() << std::defaultfloat << '\n';
    }
}
} // namespace
int main() {
    std::cout << std::setprecision(10);
    for (std::size_t n : {9, 33, 65, 257})
        one_dimensional(n);
    four_dimensional();
    proof_vectors();
}
