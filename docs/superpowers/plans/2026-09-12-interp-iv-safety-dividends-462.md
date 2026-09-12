# interp_iv_safety Dividends Retune and K_ref Sweep (#462) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `interp_iv_safety --path=dividends` build and report honestly on both backends, and add `--path=kref`, a K_ref-spacing sweep with a same-query FDM blend control that measures the `MultiKRefSplit` blend policy's error.

**Architecture:** All changes live in `benchmarks/interp_iv_safety.cc` (plus its BUILD deps, one shared helper, the API guide, and two cross-reference comments). The benchmark's per-path containers become generic over the strike count so the dividends path can carry its own seven strikes. The dividends path switches to the documented adaptive grid with a fixed quarterly $0.50 calendar and gets one FDM reference grid per backend (per-maturity contract for B-spline, rolled fixed-expiry contract for Chebyshev). The sweep builds manual segmented B-spline surfaces at four K_ref spacings and decomposes each mid-anchor price error into a surface term and a blend-policy term against FDM references. No library code changes.

**Tech Stack:** C++23, Bazel, existing `SegmentedPriceTableBuilder`, `build_multi_kref_surface`, `BSplineMultiKRefSurface`, `InterpolatedIVSolver`, `AmericanOptionSolver`, `rolled_dividends`.

**Spec:** `docs/superpowers/specs/2026-09-12-interp-iv-safety-dividends-462-design.md` (revision 4). Read it first; every D-number below refers to it.

## Global Constraints

- No library source changes under `src/`. Only `benchmarks/`, `docs/`, and (Task 6, conditional) `tests/iv_solver_factory_slow_test.cc`.
- `make_div_schedule` in `benchmarks/iv_benchmark_common.hpp` is unchanged (`iv_fdm_sweep` uses it).
- Every new C++ file or header starts with `// SPDX-License-Identifier: MIT` (no new files are planned).
- Build with `bazel build -c opt //benchmarks:interp_iv_safety` (the target is `manual`; wildcards skip it). Run the binary from `bazel-bin/benchmarks/interp_iv_safety`, never via `bazel run` with a pipe.
- Run long benchmark invocations with `TMPDIR` set to the session scratch dir: `D="/tmp/codex-skills/$CLAUDE_CODE_SESSION_ID"; mkdir -p "$D"`.
- Commit messages: imperative subject ≤ 50 chars, body wrapped at 72, no attribution lines.
- The scratch driver `benchmarks/scratch_462.cc` and its `cc_binary` block at the end of `benchmarks/BUILD.bazel` are throwaway and must be removed in Task 7; never commit them. `MODULE.bazel.lock` churn is never committed.
- Verification of a benchmark is its output: each task states the exact lines to look for. There is no GoogleTest for the benchmark.
- The baseline (before any change) is 156/156 on `bazel test //...` in this worktree.

---

### Task 0: Capture baseline outputs for AC5

**Files:**
- None modified. Outputs go to `$D/baseline-<path>.txt`.

**Interfaces:**
- Produces: `$D/baseline-bspline.txt`, `$D/baseline-chebyshev.txt`, `$D/baseline-q0.txt`, used by Task 7's AC5 diff.

- [ ] **Step 1: Build the benchmark unchanged**

```bash
cd /home/kai/work/mango-option/.worktrees/462-dividends-retune
D="/tmp/codex-skills/$CLAUDE_CODE_SESSION_ID"; mkdir -p "$D"
bazel build -c opt //benchmarks:interp_iv_safety
```
Expected: `Build completed successfully`.

- [ ] **Step 2: Run the three unaffected paths in the background**

```bash
B=bazel-bin/benchmarks/interp_iv_safety
for p in bspline chebyshev q0; do
  (TMPDIR="$D" OMP_NUM_THREADS=8 $B --path=$p > "$D/baseline-$p.txt" 2>&1; echo "rc=$?" >> "$D/baseline-$p.txt") &
done
```
These take 10–40 minutes each. Continue with Task 1; Task 7 waits for them.

---

### Task 1: Generic strike-count containers and honest empty aggregates

**Files:**
- Modify: `benchmarks/interp_iv_safety.cc` (types near line 68, `generate_prices` 72–125, `compute_errors_vanilla` 258–316, `compute_errors_div` 318–383, `print_heatmap` 389–428, `TVKMask`/`compute_tvk_mask` 434–455, `AlgoErrors`/`print_tvk_comparison` 459–508, `compute_errors_via_solver` 536–590, every call site in `run_chebyshev_*` and `main`).

**Interfaces:**
- Produces (used by every later task):

```cpp
template <size_t NS> using PriceGridN  = std::array<std::array<std::array<double, NS>, kNT>, kNV>;
template <size_t NS> using ErrorTableN = std::array<std::array<double, NS>, kNT>;
template <size_t NS> using TVKMaskN    = std::array<std::array<bool, NS>, kNT>;
using PriceGrid  = PriceGridN<kNS>;   // vanilla/q0 keep these aliases
using ErrorTable = ErrorTableN<kNS>;

/// Maturity -> schedule for that contract; nullopt = maturity not covered
/// by this path (its prices are NaN).
using ScheduleFn = std::function<std::optional<std::vector<Dividend>>(double maturity)>;

template <size_t NS>
PriceGridN<NS> generate_prices(const std::array<double, NS>& strikes,
                               const ScheduleFn& schedule, double div_yield);

template <size_t NS>
void print_heatmap(const char* title, const std::array<double, NS>& strikes,
                   const ErrorTableN<NS>& errors,
                   const std::array<std::string, kNT>* row_suffix = nullptr);

template <size_t NS>
struct AlgoErrorsN { const char* label; const ErrorTableN<NS>* errors; };  // null => n/a

template <size_t NS>
void print_tvk_comparison(const PriceGridN<NS>& prices,
                          const std::array<double, NS>& strikes, size_t vol_idx,
                          std::span<const AlgoErrorsN<NS>> algos);
```

- [ ] **Step 1: Introduce the templated aliases and `ScheduleFn`**

Replace the `PriceGrid` alias (line ~70) and `ErrorTable` alias (line ~256) with the block above, placed once after the `kMatLabels` definition. Add `#include <functional>`, `#include <optional>`, `#include <string>` if missing.

- [ ] **Step 2: Rewrite `generate_prices`**

```cpp
template <size_t NS>
static PriceGridN<NS> generate_prices(const std::array<double, NS>& strikes,
                                      const ScheduleFn& schedule,
                                      double div_yield) {
    PriceGridN<NS> prices{};
    for (auto& v : prices) for (auto& t : v) t.fill(std::nan(""));

    BatchAmericanOptionSolver batch_solver;
    std::vector<PricingParams> all_params;
    std::vector<std::array<size_t, 3>> index;  // (vi, ti, si)
    all_params.reserve(kNV * kNT * NS);

    for (size_t vi = 0; vi < kNV; ++vi) {
        for (size_t ti = 0; ti < kNT; ++ti) {
            auto divs = schedule(kMaturities[ti]);
            if (!divs) continue;  // maturity not covered: row stays NaN
            for (size_t si = 0; si < NS; ++si) {
                PricingParams p;
                p.spot = kSpot;
                p.strike = strikes[si];
                p.maturity = kMaturities[ti];
                p.rate = kRate;
                p.dividend_yield = div_yield;
                p.option_type = OptionType::PUT;
                p.volatility = kVols[vi];
                p.discrete_dividends = *divs;
                all_params.push_back(std::move(p));
                index.push_back({vi, ti, si});
            }
        }
    }

    auto result = batch_solver.solve_batch(all_params, /*use_shared_grid=*/true);
    for (size_t i = 0; i < index.size(); ++i) {
        auto [vi, ti, si] = index[i];
        if (result.results[i].has_value())
            prices[vi][ti][si] = result.results[i]->value();
    }
    return prices;
}

static const ScheduleFn kNoDividends = [](double) {
    return std::optional<std::vector<Dividend>>{std::vector<Dividend>{}};
};
```

Update the vanilla call sites: `generate_prices(kStrikes, kNoDividends, kDivYield)` and, for q0, `generate_prices(kStrikes, kNoDividends, 0.0)`. Leave the dividends call site as `generate_prices(kStrikes, [](double T){ return std::optional{make_div_schedule(T)}; }, kDivYield)` for now; Task 3 replaces it.

- [ ] **Step 3: Template the error computations**

`compute_errors_vanilla`, `compute_errors_via_solver` and `compute_errors_div` gain `template <size_t NS>` and take `const PriceGridN<NS>& prices, const std::array<double, NS>& strikes` (replace every `kStrikes[si]` inside with `strikes[si]` and every `kNS` with `NS`). Return `ErrorTableN<NS>`. Call sites pass `kStrikes` explicitly.

- [ ] **Step 4: Template `print_heatmap`, add `n/a` and row suffixes**

```cpp
template <size_t NS>
static void print_heatmap(const char* title, const std::array<double, NS>& strikes,
                          const ErrorTableN<NS>& errors,
                          const std::array<std::string, kNT>* row_suffix = nullptr) {
    std::printf("\n=== %s ===\n", title);
    std::printf("          ");
    for (double K : strikes) {
        // Integral strikes keep the historical "K=100" header; fractional
        // ones (dividends path) print one decimal.
        if (std::fmod(K, 1.0) == 0.0) std::printf("  K=%-3.0f ", K);
        else                          std::printf(" K=%-5.1f", K);
    }
    std::printf("\n");

    size_t n_total = 0, n_failed = 0;
    double sum_sq = 0;
    for (size_t ti = 0; ti < kNT; ++ti) {
        std::printf("  T=%s  ", kMatLabels[ti]);
        for (size_t si = 0; si < NS; ++si) {
            double e = errors[ti][si];
            n_total++;
            if (std::isnan(e)) { std::printf("   ---  "); n_failed++; continue; }
            const char* marker = e > 200 ? "***" : e > 50 ? "**" : e > 10 ? "*" : "";
            std::printf("%6.1f%-3s", e, marker);
            sum_sq += e * e;
        }
        if (row_suffix) std::printf("  %s", (*row_suffix)[ti].c_str());
        std::printf("\n");
    }
    size_t n_valid = n_total - n_failed;
    std::printf("\n  Legend: * >10bps  ** >50bps  *** >200bps  --- solve failed\n");
    if (n_valid == 0)
        std::printf("  Overall RMS: n/a (0/%zu succeeded)\n", n_total);
    else
        std::printf("  Overall RMS: %.1f bps (%zu/%zu succeeded)\n",
                    std::sqrt(sum_sq / n_valid), n_valid, n_total);
}
```

- [ ] **Step 5: Template the TV/K helpers with null tables and `n/a`**

```cpp
template <size_t NS>
static TVKMaskN<NS> compute_tvk_mask(const PriceGridN<NS>& prices,
                                     const std::array<double, NS>& strikes,
                                     size_t vol_idx, double threshold) {
    TVKMaskN<NS> mask{};
    for (size_t ti = 0; ti < kNT; ++ti)
        for (size_t si = 0; si < NS; ++si) {
            double price = prices[vol_idx][ti][si];
            if (std::isnan(price) || price <= 0) { mask[ti][si] = false; continue; }
            double intrinsic = std::max(strikes[si] - kSpot, 0.0);  // put
            mask[ti][si] = ((price - intrinsic) / strikes[si]) >= threshold;
        }
    return mask;
}

template <size_t NS>
struct AlgoErrorsN { const char* label; const ErrorTableN<NS>* errors; };

template <size_t NS>
static void print_tvk_comparison(const PriceGridN<NS>& prices,
                                 const std::array<double, NS>& strikes,
                                 size_t vol_idx,
                                 std::span<const AlgoErrorsN<NS>> algos) {
    static constexpr double kThresholds[] = {0.0, 1e-4, 1e-3, 5e-3};
    static constexpr const char* kThreshLabels[] = {"none", "1e-4", "1e-3", "5e-3"};
    std::printf("\n  TV/K filtered RMS (σ=%.0f%%):\n", kVols[vol_idx] * 100);
    std::printf("  %-20s", "TV/K >=");
    for (const auto& a : algos) std::printf("  %14s", a.label);
    std::printf("\n");
    for (size_t fi = 0; fi < 4; ++fi) {
        auto mask = compute_tvk_mask(prices, strikes, vol_idx, kThresholds[fi]);
        size_t mask_count = 0;
        for (auto& row : mask) for (bool b : row) mask_count += b;
        std::printf("  %-12s [%2zu/%zu]", kThreshLabels[fi], mask_count, kNT * NS);
        for (const auto& a : algos) {
            if (!a.errors) { std::printf("  %14s", "n/a (no surf)"); continue; }
            double sum_sq = 0; size_t n = 0;
            for (size_t ti = 0; ti < kNT; ++ti)
                for (size_t si = 0; si < NS; ++si) {
                    if (!mask[ti][si]) continue;
                    double e = (*a.errors)[ti][si];
                    if (std::isnan(e)) continue;
                    sum_sq += e * e; n++;
                }
            char buf[32];
            if (n == 0) std::snprintf(buf, sizeof(buf), "n/a (0)");
            else        std::snprintf(buf, sizeof(buf), "%.1f (%zu)", std::sqrt(sum_sq / n), n);
            std::printf("  %14s", buf);
        }
        std::printf("\n");
    }
}
```

Update all three TV/K blocks in `main` to build `std::vector<AlgoErrorsN<kNS>>` and pass `kStrikes`.

- [ ] **Step 6: Build and smoke-run**

```bash
bazel build -c opt //benchmarks:interp_iv_safety 2>&1 | tail -3
```
Expected: `Build completed successfully`, no new warnings (compare warning count with Task 0's build log; `-Wall -Wextra` are on).

A full path run is slow; verify structure with the fastest path once Task 3 lands. For now confirm the binary starts and prints the banner:

```bash
timeout 5 bazel-bin/benchmarks/interp_iv_safety --path=none | head -12
```
Expected: banner lines through `Usage: interp_iv_safety [...]`, then nothing (no path matched), exit 0.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/interp_iv_safety.cc
git commit -m "Make interp_iv_safety tables generic over strikes

The dividends path needs its own strike set inside the documented
band, and an aggregate with no contributing points must print n/a
instead of 0.0, which is what hid the failed dividend builds."
```

---

### Task 2: Failure reporting and exit status (D3)

**Files:**
- Modify: `benchmarks/interp_iv_safety.cc` (`build_div_solvers` skip lines ~188–193, `run_chebyshev_dividends` failure ~730, `run_chebyshev_adaptive` ~652, `build_chebyshev_surface` ~528, `main` return).

**Interfaces:**
- Produces:

```cpp
static const char* code_name(PriceTableErrorCode c);   // exhaustive switch
static void report_build_failure(const char* what, const PriceTableError& e);
static void report_wrap_failure(const char* what, const ValidationError& e);
static bool g_build_failed = false;   // set by the two helpers; main returns 1 if set
```

- [ ] **Step 1: Add the helpers after the includes**

```cpp
static bool g_build_failed = false;

static const char* code_name(PriceTableErrorCode c) {
    switch (c) {
        case PriceTableErrorCode::InvalidConfig:          return "InvalidConfig";
        case PriceTableErrorCode::InsufficientGridPoints: return "InsufficientGridPoints";
        case PriceTableErrorCode::GridNotSorted:          return "GridNotSorted";
        case PriceTableErrorCode::NonPositiveValue:       return "NonPositiveValue";
        case PriceTableErrorCode::EmptyBatch:             return "EmptyBatch";
        case PriceTableErrorCode::ExtractionFailed:       return "ExtractionFailed";
        case PriceTableErrorCode::RepairFailed:           return "RepairFailed";
        case PriceTableErrorCode::FittingFailed:          return "FittingFailed";
        case PriceTableErrorCode::SurfaceBuildFailed:     return "SurfaceBuildFailed";
        case PriceTableErrorCode::SerializationFailed:    return "SerializationFailed";
        case PriceTableErrorCode::ArenaAllocationFailed:  return "ArenaAllocationFailed";
        case PriceTableErrorCode::TensorCreationFailed:   return "TensorCreationFailed";
        case PriceTableErrorCode::ValidationFailed:       return "ValidationFailed";
        case PriceTableErrorCode::NoViableSurface:        return "NoViableSurface";
    }
    return "?";  // unreachable; -Wswitch flags a new enumerator above
}

static void report_build_failure(const char* what, const PriceTableError& e) {
    g_build_failed = true;
    std::fprintf(stderr, "  [FAILED] %s: %s (axis=%zu count=%zu)\n",
                 what, code_name(e.code), e.axis_index, e.count);
    std::printf("  [FAILED] %s: %s (axis=%zu count=%zu)\n",
                what, code_name(e.code), e.axis_index, e.count);
}

static void report_wrap_failure(const char* what, const ValidationError& e) {
    g_build_failed = true;
    std::fprintf(stderr, "  [FAILED] %s: ValidationErrorCode %d\n",
                 what, static_cast<int>(e.code));
    std::printf("  [FAILED] %s: ValidationErrorCode %d\n",
                what, static_cast<int>(e.code));
}
```

(`ValidationError::code` is a `ValidationErrorCode`; see `src/support/error_types.hpp:63`.)

- [ ] **Step 2: Route every build failure through the helpers**

- `build_div_solvers`: replace the `[skip] ... adaptive build failed` `fprintf` with `report_build_failure(label, result.error())` where `label` is a `char[32]` formatted as `"B-spline dividends T=%s"`; the solver-wrap failure becomes `report_wrap_failure(label, solver.error())`.
- `run_chebyshev_dividends`: `report_build_failure("Chebyshev dividends", result.error())`; solver wrap → `report_wrap_failure`.
- `run_chebyshev_adaptive`, `run_chebyshev_4d`, `build_chebyshev_surface`, `build_vanilla_solver`, `build_bspline_q0`, `build_dimless_3d` call sites: keep their existing `std::exit(1)` or early return, but set `g_build_failed = true` and print the code before returning (vanilla/q0 factory errors are `ValidationError` → `report_wrap_failure`).
- End of `main`: `return g_build_failed ? 1 : 0;`

- [ ] **Step 3: Print which maturities built**

At the end of `build_div_solvers`, before `return solvers;`:

```cpp
std::printf("  built %zu/%zu per-maturity solvers:", solvers.size(), kNT);
for (const auto& [ti, _] : solvers) std::printf(" %s", kMatLabels[ti]);
std::printf("\n");
```

- [ ] **Step 4: Verify with the current (still failing) dividends config**

```bash
bazel build -c opt //benchmarks:interp_iv_safety 2>&1 | tail -1
TMPDIR="$D" OMP_NUM_THREADS=8 timeout 60 bazel-bin/benchmarks/interp_iv_safety --path=dividends 2>/dev/null | grep -E "FAILED|built [0-9]+/8"; echo "rc=${PIPESTATUS[0]}"
```
Expected: eight lines `[FAILED] B-spline dividends T=...: NoViableSurface (axis=0 count=0)`, then `built 0/8 per-maturity solvers:`. The `timeout` kills the Chebyshev build; that is fine for this check (rc will be 124 from timeout). AC3 proper is verified in Task 7.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/interp_iv_safety.cc
git commit -m "Report interp_iv_safety build failures by code

A failed adaptive build printed a bare skip line and the summary
tables showed 0.0, so a dead dividends path looked like a perfect
one. Print the PriceTableErrorCode and exit non-zero."
```

---

### Task 3: Retune the dividends path (D1, D2)

**Files:**
- Modify: `benchmarks/iv_benchmark_common.hpp` (add `quarterly_div_schedule` after `make_div_schedule`)
- Modify: `benchmarks/interp_iv_safety.cc` (`kDivStrikes` constants, `build_div_solvers`, `compute_errors_div`, `run_chebyshev_dividends`, dividends blocks in `main`)
- Modify: `benchmarks/BUILD.bazel` (`interp_iv_safety` deps: add `"//src/option:dividend_utils"`, `"//src/option/table/bspline:bspline_adaptive"`)

**Interfaces:**
- Consumes: Task 1 templates, Task 2 helpers.
- Produces:

```cpp
// iv_benchmark_common.hpp
inline std::vector<Dividend> quarterly_div_schedule(double maturity);  // $0.50 at 0.25, 0.50, ... < maturity

// interp_iv_safety.cc
static constexpr std::array<double, 7> kDivStrikes = {93.0, 95.0, 97.5, 100.0, 102.5, 105.0, 107.0};
static constexpr size_t kNDS = kDivStrikes.size();
static const std::vector<double> kDocMoneyness = {0.92, 0.95, 1.0, 1.05, 1.08};   // S/K
static const std::vector<double> kDocVols      = {0.10, 0.15, 0.20, 0.30};
static const std::vector<double> kDocRates     = {0.02, 0.03, 0.05, 0.07};
static const std::vector<double> kDocKRefs     = {90, 92.5, 95, 97.5, 100, 102.5, 105, 107.5, 110};
static constexpr double kDocTargetIVError = 1e-3;
static const ScheduleFn kQuarterlyPerMaturity;   // T -> quarterly_div_schedule(T)
static const ScheduleFn kRolledFrom1y;           // T<=1 -> rolled_dividends(quarterly_div_schedule(1.0), 1.0, T); else nullopt
```

- [ ] **Step 1: Add the calendar helper**

In `benchmarks/iv_benchmark_common.hpp` after `make_div_schedule`:

```cpp
// Fixed quarterly $0.50 calendar, filtered to the option's life. Used by
// interp_iv_safety's dividends path (#462): unlike make_div_schedule it does
// not scale with maturity, so a 7-day option carries no dividend and a
// 1-year option carries three. See the retune spec for why the scaled
// schedule refuses on the segmented B-spline path at short maturities (#501).
inline std::vector<Dividend> quarterly_div_schedule(double maturity) {
    std::vector<Dividend> out;
    for (double t = 0.25; t < maturity; t += 0.25)
        out.push_back(Dividend{.calendar_time = t, .amount = 0.50});
    return out;
}
```

- [ ] **Step 2: Add the documented-config constants and schedule functions**

After `kMatLabels` in `interp_iv_safety.cc`:

```cpp
// ---------------------------------------------------------------------------
// Dividends path: the documented adaptive discrete-dividend configuration.
// Grid, K_refs and target are verbatim from documented_adaptive_dividend_config()
// in tests/iv_solver_factory_slow_test.cc (the nightly pin); if that helper
// changes, change these too. The yield (kDivYield) and the schedule
// (quarterly_div_schedule) are the benchmark's own.
// ---------------------------------------------------------------------------
static constexpr std::array<double, 7> kDivStrikes = {
    93.0, 95.0, 97.5, 100.0, 102.5, 105.0, 107.0};   // all inside S/K in [0.92, 1.08]
static constexpr size_t kNDS = kDivStrikes.size();
static const std::vector<double> kDocMoneyness = {0.92, 0.95, 1.0, 1.05, 1.08};
static const std::vector<double> kDocVols      = {0.10, 0.15, 0.20, 0.30};
static const std::vector<double> kDocRates     = {0.02, 0.03, 0.05, 0.07};
static const std::vector<double> kDocKRefs     = {90.0, 92.5, 95.0, 97.5, 100.0,
                                                  102.5, 105.0, 107.5, 110.0};
static constexpr double kDocTargetIVError = 1e-3;

static std::vector<double> doc_log_moneyness() {
    std::vector<double> out;
    for (double m : kDocMoneyness) out.push_back(std::log(m));
    return out;
}

static const ScheduleFn kQuarterlyPerMaturity = [](double T) {
    return std::optional{quarterly_div_schedule(T)};
};
static const ScheduleFn kRolledFrom1y = [](double T) -> std::optional<std::vector<Dividend>> {
    if (T > 1.0 + 1e-9) return std::nullopt;
    return rolled_dividends(quarterly_div_schedule(1.0), 1.0, T);
};

static std::array<std::string, kNT> dividend_count_labels(const ScheduleFn& schedule) {
    std::array<std::string, kNT> out;
    for (size_t ti = 0; ti < kNT; ++ti) {
        auto d = schedule(kMaturities[ti]);
        out[ti] = d ? "(" + std::to_string(d->size()) + " div)" : "(not covered)";
    }
    return out;
}
```

Add `#include "mango/option/dividend_utils.hpp"`.

- [ ] **Step 3: Rewrite `build_div_solvers`**

```cpp
static std::vector<std::pair<size_t, BSplineDivSolver>> build_div_solvers() {
    std::vector<std::pair<size_t, BSplineDivSolver>> solvers;
    const auto log_m = doc_log_moneyness();
    const AdaptiveGridParams adaptive{.target_iv_error = kDocTargetIVError};

    for (size_t ti = 0; ti < kNT; ++ti) {
        const double mat = kMaturities[ti];
        auto divs = quarterly_div_schedule(mat);
        char label[48];
        std::snprintf(label, sizeof(label), "B-spline dividends T=%s", kMatLabels[ti]);

        SegmentedAdaptiveConfig seg_config{
            .spot = kSpot,
            .option_type = OptionType::PUT,
            .dividend_yield = kDivYield,
            .discrete_dividends = divs,
            .maturity = mat,
            .kref_config = {.K_refs = kDocKRefs},
        };
        auto result = build_adaptive_bspline_segmented(
            adaptive, seg_config, {log_m, kDocVols, kDocRates});
        if (!result.has_value()) { report_build_failure(label, result.error()); continue; }

        std::printf("  T=%s (%zu div): iters=%zu target_met=%s max_err=%.1f bps "
                    "avg_err=%.1f bps measured=%zu PDE=%zu%s\n",
                    kMatLabels[ti], divs.size(), result->iterations.size(),
                    result->target_met ? "yes" : "no",
                    result->achieved_max_error * 1e4, result->achieved_avg_error * 1e4,
                    result->diagnostics.holdout_points_measured,
                    result->total_pde_solves, result->used_retry ? " (retry)" : "");

        // Published bounds are the builder's measured sample domain (spec D2
        // of #454), not the input arrays.
        auto wrapper = BSplineMultiKRefSurface(
            std::move(result->surface), result->sample_bounds, OptionType::PUT, kDivYield);
        auto solver = BSplineDivSolver::create(std::move(wrapper), {}, divs);
        if (!solver.has_value()) { report_wrap_failure(label, solver.error()); continue; }
        solvers.emplace_back(ti, std::move(*solver));
    }
    std::printf("  built %zu/%zu per-maturity solvers:", solvers.size(), kNT);
    for (const auto& [ti, _] : solvers) std::printf(" %s", kMatLabels[ti]);
    std::printf("\n");
    return solvers;
}
```

- [ ] **Step 4: Populate query schedules in `compute_errors_div`**

In the per-maturity loop, replace `auto divs = make_div_schedule(maturity);` with `auto divs = quarterly_div_schedule(maturity);` and add `q.discrete_dividends = divs;` when building each `IVQuery`. The function is already templated on `NS` from Task 1; the dividends call site passes `kDivStrikes`.

- [ ] **Step 5: Rewrite the Chebyshev dividends config and error loop**

In `run_chebyshev_dividends(const PriceGridN<kNDS>& prices)`:

```cpp
    AdaptiveGridParams params{.target_iv_error = kDocTargetIVError};
    const auto divs_1y = quarterly_div_schedule(1.0);
    SegmentedAdaptiveConfig config{
        .spot = kSpot,
        .option_type = OptionType::PUT,
        .dividend_yield = kDivYield,
        .discrete_dividends = divs_1y,
        .maturity = 1.0,
        .kref_config = {.K_refs = kDocKRefs},
    };
    IVGrid domain{.moneyness = doc_log_moneyness(), .vol = kDocVols, .rate = kDocRates};
    std::printf("--- Building segmented Chebyshev surface (documented config, "
                "target=%.1f bps, %zu div)...\n", params.target_iv_error * 1e4, divs_1y.size());
    auto result = build_adaptive_chebyshev_segmented(params, config, domain);
    if (!result.has_value()) {
        report_build_failure("Chebyshev dividends", result.error());
        return {};   // callers treat an empty optional/table as "no surface" (see main)
    }
```

Keep the iteration printout. Change the T = 1 point diagnostic loop to iterate `kDivStrikes` and print `K=%5.1f`. Wrap with the schedule:

```cpp
    auto solver = InterpolatedIVSolver<ChebyshevMultiKRefSurface>::create(
        std::move(result->surface), {}, divs_1y);
    if (!solver.has_value()) { report_wrap_failure("Chebyshev dividends", solver.error()); return {}; }
```

Error loop: replace the hand-rolled `future_divs` with

```cpp
            auto rolled = kRolledFrom1y(tau);
            if (!rolled) continue;                       // T > 1: not covered
            for (size_t si = 0; si < kNDS; ++si) {
                double price = prices[vi][ti][si];       // from the rolled reference grid
                if (std::isnan(price) || price <= 0) continue;
                double fdm_iv = solve_fdm_iv_div(kDivStrikes[si], tau, price, *rolled);
                if (std::isnan(fdm_iv)) continue;
                IVQuery q; /* spot/strike/maturity/rate/yield/type as before */
                q.market_price = price;
                q.discrete_dividends = *rolled;
                auto iv_result = solver->solve(q);
                if (!iv_result.has_value()) continue;
                errors[ti][si] = std::abs(iv_result->implied_vol - fdm_iv) * 10000.0;
            }
```

Return type: `std::optional<std::array<ErrorTableN<kNDS>, kNV>>` (nullopt when no surface), and print each σ heatmap with `print_heatmap(title, kDivStrikes, errors, &labels)` where `labels = dividend_count_labels(kRolledFrom1y)`.

- [ ] **Step 6: Wire `main`**

Replace the dividends blocks:

```cpp
    PriceGridN<kNDS> bs_div_prices{}, cheb_div_prices{};
    if (need_divs) {
        std::printf("--- Generating dividend reference prices: per-maturity quarterly calendar (B-spline)...\n");
        bs_div_prices = generate_prices(kDivStrikes, kQuarterlyPerMaturity, kDivYield);
        std::printf("--- Generating dividend reference prices: 1y calendar rolled to each maturity (Chebyshev)...\n");
        cheb_div_prices = generate_prices(kDivStrikes, kRolledFrom1y, kDivYield);
    }
    std::optional<std::array<ErrorTableN<kNDS>, kNV>> div_errors, cheb_div_errors;
    ...
    if (run_all || path == "dividends") {
        auto div_solvers = build_div_solvers();
        if (!div_solvers.empty()) {
            div_errors.emplace();
            auto labels = dividend_count_labels(kQuarterlyPerMaturity);
            for (size_t vi = 0; vi < kNV; ++vi) {
                (*div_errors)[vi] = compute_errors_div(bs_div_prices, kDivStrikes, div_solvers, vi);
                char title[160];
                std::snprintf(title, sizeof(title),
                    "Interpolation IV Error (bps) — σ=%.0f%%, quarterly $0.50 calendar (B-spline per-maturity)",
                    kVols[vi] * 100);
                print_heatmap(title, kDivStrikes, (*div_errors)[vi], &labels);
            }
        }
        cheb_div_errors = run_chebyshev_dividends(cheb_div_prices);
    }
```

TV/K block for dividends: two separate sub-blocks, each with its own price grid and one algorithm:

```cpp
    if (need_divs) {
        std::printf("\n=== TV/K Filtered Comparison — discrete dividends ===\n");
        for (size_t vi = 0; vi < kNV; ++vi) {
            std::printf("\n  [B-spline per-maturity, reference = quarterly calendar per maturity]");
            std::array<AlgoErrorsN<kNDS>, 1> a{{{"B-spline(div)", div_errors ? &(*div_errors)[vi] : nullptr}}};
            print_tvk_comparison<kNDS>(bs_div_prices, kDivStrikes, vi, a);
            std::printf("\n  [Chebyshev fixed-expiry 1y, reference = 1y calendar rolled]");
            std::array<AlgoErrorsN<kNDS>, 1> c{{{"Cheb(div)", cheb_div_errors ? &(*cheb_div_errors)[vi] : nullptr}}};
            print_tvk_comparison<kNDS>(cheb_div_prices, kDivStrikes, vi, c);
        }
    }
```

Also print the dividends strike set in the banner (`Dividend strikes: 93.0 95.0 ...`) and add `kref` to the usage line (Task 4 fills it in).

- [ ] **Step 7: Add the cross-reference comment in the slow test**

In `tests/iv_solver_factory_slow_test.cc`, above `documented_adaptive_dividend_config()`, append to the existing doc comment:

```cpp
/// benchmarks/interp_iv_safety.cc copies this grid, K_ref list and target
/// (kDoc* constants) for its dividends path; keep them in step.
```

- [ ] **Step 8: Build and run the dividends path (AC2 pilot)**

```bash
bazel build -c opt //benchmarks:interp_iv_safety 2>&1 | tail -1
TMPDIR="$D" OMP_NUM_THREADS=16 bazel-bin/benchmarks/interp_iv_safety --path=dividends > "$D/dividends.txt" 2>&1; echo "rc=$?"
grep -E "T=.*\(.*div\): iters|built [0-9]+/8|FAILED|Overall RMS|target_met" "$D/dividends.txt"
```
Expected (about 12–15 minutes, dominated by the Chebyshev build):
- eight `T=... (n div): iters=...` lines, `built 8/8 per-maturity solvers`;
- no `[FAILED]` line; `rc=0`;
- four `Overall RMS:` lines with a numeric value (two B-spline σ, two Chebyshev σ);
- Chebyshev `target_met: no` with a max error near 70 bps (69.6 measured in the spec).

Then check AC2 coverage: count non-`---` cells in the four heatmaps (7 strikes × 8 rows B-spline, × 7 rows Chebyshev, × 2 σ = 210 attempted):

```bash
awk '/=== Interpolation IV Error.*B-spline/,/Overall RMS/' "$D/dividends.txt" | grep -c -- '---'
awk '/=== Cheb Dividend IV Error/,/Overall RMS/' "$D/dividends.txt" | grep -c -- '---'
```
Expected: total `---` count ≤ 52 (i.e. ≥ 75% of 210 succeed) and every row has at least one value. If not, record the per-row counts in the commit body and in the PR; do not change the config (spec AC2).

- [ ] **Step 9: Commit**

```bash
git add benchmarks/iv_benchmark_common.hpp benchmarks/interp_iv_safety.cc benchmarks/BUILD.bazel tests/iv_solver_factory_slow_test.cc
git commit -m "Retune interp_iv_safety dividends path

Use the documented adaptive discrete-dividend grid with a fixed
quarterly calendar, give each backend reference prices for the
contract it represents, and publish the builders' sample bounds.
The old config paired a +/-30% moneyness range with K_refs spanning
80-120 and scaled three dividends into a 7-day option, which the
segmented builders refuse (#462, #501)."
```

---

### Task 4: `--path=kref` spacing sweep (D4)

**Files:**
- Modify: `benchmarks/interp_iv_safety.cc` (new section before `main`; `main` dispatch; usage line)
- Modify: `benchmarks/BUILD.bazel` (`interp_iv_safety` deps: add `"//src/option/table/bspline:bspline_segmented_builder"`)

**Interfaces:**
- Consumes: `quarterly_div_schedule`, `solve_fdm_iv_div(strike, maturity, price, divs)` from `iv_benchmark_common.hpp`, Task 2 helpers.
- Produces: `static void run_kref_sweep();` and the CLI value `--path=kref` (also part of `all`).

- [ ] **Step 1: Add the sweep section**

Insert before `// CLI path selection`:

```cpp
// ============================================================================
// K_ref spacing sweep (--path=kref): blend policy vs surface, with a
// same-query FDM control. Spec: docs/superpowers/specs/2026-09-12-interp-iv-safety-dividends-462-design.md D4.
//
// Terms: an *anchor* is a strike equal to a K_ref; a *mid-anchor* is the
// midpoint between two adjacent K_refs; the *blend policy* is
// MultiKRefSplit (query each bracketing K_ref surface at (spot, K_ref),
// normalize by K_ref, interpolate linearly in strike, multiply by strike).
// ============================================================================
namespace kref {

constexpr double kSpanLo = 80.0, kSpanHi = 120.0;
constexpr double kWindowLo = 85.0, kWindowHi = 115.0;
constexpr std::array<double, 4> kSpacings = {10.0, 5.0, 2.5, 1.25};
constexpr std::array<double, 4> kSweepMaturities = {0.20, 0.30, 0.60, 1.0};
constexpr std::array<double, 2> kSweepVols = {0.15, 0.30};
constexpr double kVegaFloor = 1e-4;      // AdaptiveGridParams::vega_floor default
constexpr double kTVKThreshold = 1e-4;   // make_iv_score_fn's threshold
constexpr double kQualifyBps = 10.0;     // D5 classification threshold
constexpr size_t kBaseMoneynessKnots = 41;
constexpr int kBaseTauPoints = 5;

struct Ref { double price = 0.0, vega = 0.0; bool ok = false; };

/// FDM reference price and central-bump vega (same bump as
/// make_fd_vega_refs_fn: eps = max(1e-4, 0.01*sigma)). `accuracy` nullopt =
/// the solver's automatic grid; set = an explicit GridAccuracyParams.
static Ref fdm_ref(double K, double T, double sigma,
                   const std::vector<Dividend>& divs,
                   std::optional<GridAccuracyParams> accuracy) {
    auto price_at = [&](double sg) -> std::optional<double> {
        PricingParams p;
        p.spot = kSpot; p.strike = K; p.maturity = T; p.rate = kRate;
        p.dividend_yield = kDivYield; p.option_type = OptionType::PUT;
        p.volatility = sg; p.discrete_dividends = divs;
        std::optional<PDEGridSpec> grid;
        if (accuracy) grid = PDEGridSpec{*accuracy};
        auto solver = AmericanOptionSolver::create(p, grid);
        if (!solver) return std::nullopt;
        auto r = solver->solve();
        if (!r || !std::isfinite(r->value())) return std::nullopt;
        return r->value();
    };
    Ref out;
    const double eps = std::max(1e-4, 0.01 * sigma);
    auto p0 = price_at(sigma);
    auto pu = price_at(sigma + eps);
    auto pd = price_at(std::max(1e-4, sigma - eps));
    if (!p0 || !pu || !pd) return out;
    out.price = *p0;
    out.vega = (*pu - *pd) / ((sigma + eps) - std::max(1e-4, sigma - eps));
    out.ok = std::isfinite(out.vega);
    return out;
}

static std::vector<double> krefs_for(double delta) {
    std::vector<double> ks;
    for (double k = kSpanLo; k <= kSpanHi + 1e-9; k += delta) ks.push_back(k);
    return ks;
}

/// Manual (non-adaptive) multi-K_ref segmented B-spline surface on fixed
/// input knots. Mirrors build_multi_kref_manual + manual_segmented_bounds in
/// src/option/price_table_factory.cpp.
static std::expected<BSplineMultiKRefSurface, PriceTableError>
build_manual(const std::vector<double>& krefs, double T, size_t n_m, int tau_pts) {
    std::vector<double> log_m(n_m);
    for (size_t i = 0; i < n_m; ++i)
        log_m[i] = -0.30 + 0.60 * static_cast<double>(i) / static_cast<double>(n_m - 1);
    const std::vector<double> vols  = {0.10, 0.15, 0.20, 0.30, 0.50};
    const std::vector<double> rates = {0.02, 0.03, 0.05, 0.07};  // builder needs >= 4 knots
    DividendSpec dividends{.dividend_yield = kDivYield,
                           .discrete_dividends = quarterly_div_schedule(T)};
    std::vector<BSplineMultiKRefEntry> entries;
    for (double k : krefs) {
        SegmentedPriceTableBuilder::Config cfg{
            .K_ref = k, .option_type = OptionType::PUT, .dividends = dividends,
            .grid = IVGrid{.moneyness = log_m, .vol = vols, .rate = rates},
            .maturity = T, .tau_points_per_segment = tau_pts,
        };
        auto surface = SegmentedPriceTableBuilder::build(cfg);
        if (!surface) return std::unexpected(surface.error());
        entries.push_back({.K_ref = k, .surface = std::move(*surface)});
    }
    auto inner = build_multi_kref_surface(std::move(entries));
    if (!inner) return std::unexpected(inner.error());
    SurfaceBounds bounds{.m_min = -0.30, .m_max = 0.30, .tau_min = 0.0, .tau_max = T,
                         .sigma_min = vols.front(), .sigma_max = vols.back(),
                         .rate_min = rates.front(), .rate_max = rates.back()};
    return BSplineMultiKRefSurface(std::move(*inner), bounds, OptionType::PUT, kDivYield);
}

struct Query { double K; bool anchor; double L, H; };

static std::vector<Query> queries_for(const std::vector<double>& krefs) {
    std::vector<Query> qs;
    for (size_t i = 0; i < krefs.size(); ++i) {
        if (krefs[i] >= kWindowLo && krefs[i] <= kWindowHi)
            qs.push_back({krefs[i], true, krefs[i], krefs[i]});   // anchor: control = itself
        if (i + 1 < krefs.size()) {
            double mid = 0.5 * (krefs[i] + krefs[i + 1]);
            if (mid >= kWindowLo && mid <= kWindowHi)
                qs.push_back({mid, false, krefs[i], krefs[i + 1]});
        }
    }
    return qs;
}

/// Accumulator for one (delta, T, sigma, anchor/mid) population.
struct Stat {
    size_t q = 0, elig = 0, ref_fail = 0, low_vega = 0, low_tv = 0, surf_nonfinite = 0;
    double blend_max = 0, blend_sq = 0;   // |B_fdm - P_fdm| / vega, bps
    double surf_max = 0, surf_sq = 0;     // |P_hat - B_fdm| / vega, bps
    size_t surf_n = 0;
    size_t inv_n = 0, inv_fail = 0; double inv_max = 0;
    double blend_max_fine = 0; size_t fine_n = 0;   // ref-sens (mid-anchors only)
    bool complete() const { return elig >= 1 && 100 * elig >= 90 * q; }
    double blend_rms() const { return elig ? std::sqrt(blend_sq / elig) : std::nan(""); }
    double surf_rms()  const { return surf_n ? std::sqrt(surf_sq / surf_n) : std::nan(""); }
};

static void fmt(char* buf, size_t n, double v) {
    if (std::isnan(v)) std::snprintf(buf, n, "%9s", "n/a");
    else std::snprintf(buf, n, "%9.1f", v);
}

/// Reference cache keyed by (T, sigma, K, fine): each (K, sigma, T) is solved
/// once per accuracy no matter how many spacings share it.
using RefKey = std::tuple<int, int, long, int>;
static Ref cached_ref(std::map<RefKey, Ref>& cache, double K, double T, double sigma,
                      const std::vector<Dividend>& divs, bool fine) {
    RefKey key{static_cast<int>(std::lround(T * 1e4)), static_cast<int>(std::lround(sigma * 1e4)),
               std::lround(K * 1e3), fine ? 1 : 0};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto r = fdm_ref(K, T, sigma, divs,
                     fine ? std::optional{make_grid_accuracy(GridAccuracyProfile::Ultra)} : std::nullopt);
    cache.emplace(key, r);
    return r;
}

struct RowResult { Stat mid, anchor; bool built = false; double seconds = 0; };

static RowResult run_row(double delta, double T, double sigma, size_t n_m, int tau_pts,
                         std::map<RefKey, Ref>& cache) {
    RowResult row;
    auto t0 = std::chrono::steady_clock::now();
    const auto divs = quarterly_div_schedule(T);
    const auto krefs = krefs_for(delta);
    auto surface = build_manual(krefs, T, n_m, tau_pts);
    if (!surface) {
        char what[64]; std::snprintf(what, sizeof(what), "kref sweep delta=%.2f T=%.2f", delta, T);
        report_build_failure(what, surface.error());
        return row;
    }
    // InterpolatedIVSolver keeps its surface private, so keep a copy for
    // direct pricing (PriceTable and SplitSurface are value types).
    const BSplineMultiKRefSurface surf = *surface;
    auto solver = InterpolatedIVSolver<BSplineMultiKRefSurface>::create(std::move(*surface), {}, divs);
    if (!solver) {
        char what[64]; std::snprintf(what, sizeof(what), "kref sweep delta=%.2f T=%.2f", delta, T);
        report_wrap_failure(what, solver.error());
        return row;
    }
    row.built = true;

    for (const Query& qy : queries_for(krefs)) {
        Stat& st = qy.anchor ? row.anchor : row.mid;
        st.q++;
        Ref rk = cached_ref(cache, qy.K, T, sigma, divs, false);
        Ref rl = qy.anchor ? rk : cached_ref(cache, qy.L, T, sigma, divs, false);
        Ref rh = qy.anchor ? rk : cached_ref(cache, qy.H, T, sigma, divs, false);
        if (!rk.ok || !rl.ok || !rh.ok) { st.ref_fail++; continue; }
        const double intrinsic = std::max(qy.K - kSpot, 0.0);
        if ((rk.price - intrinsic) / qy.K < kTVKThreshold) { st.low_tv++; continue; }
        if (rk.vega < kVegaFloor) { st.low_vega++; continue; }
        st.elig++;

        const double w = qy.anchor ? 0.0 : (qy.K - qy.L) / (qy.H - qy.L);
        const double b_fdm = qy.K * ((1.0 - w) * rl.price / qy.L + w * rh.price / qy.H);
        const double blend_bps = std::abs(b_fdm - rk.price) / rk.vega * 1e4;
        st.blend_max = std::max(st.blend_max, blend_bps);
        st.blend_sq += blend_bps * blend_bps;

        const double p_hat = surf.price(kSpot, qy.K, T, sigma, kRate);
        if (!std::isfinite(p_hat)) { st.surf_nonfinite++; }
        else {
            const double surf_bps = std::abs(p_hat - b_fdm) / rk.vega * 1e4;
            st.surf_max = std::max(st.surf_max, surf_bps);
            st.surf_sq += surf_bps * surf_bps; st.surf_n++;

            double iv_fdm = solve_fdm_iv_div(qy.K, T, rk.price, divs);
            IVQuery q; q.spot = kSpot; q.strike = qy.K; q.maturity = T; q.rate = kRate;
            q.dividend_yield = kDivYield; q.option_type = OptionType::PUT;
            q.market_price = rk.price; q.discrete_dividends = divs;
            auto iv = solver->solve(q);
            if (std::isnan(iv_fdm) || !iv) st.inv_fail++;
            else { st.inv_n++; st.inv_max = std::max(st.inv_max, std::abs(iv->implied_vol - iv_fdm) * 1e4); }
        }

        if (!qy.anchor) {   // ref-sens: same query, finer references
            Ref fk = cached_ref(cache, qy.K, T, sigma, divs, true);
            Ref fl = cached_ref(cache, qy.L, T, sigma, divs, true);
            Ref fh = cached_ref(cache, qy.H, T, sigma, divs, true);
            if (fk.ok && fl.ok && fh.ok && fk.vega >= kVegaFloor) {
                const double b_fine = qy.K * ((1.0 - w) * fl.price / qy.L + w * fh.price / qy.H);
                st.blend_max_fine = std::max(st.blend_max_fine, std::abs(b_fine - fk.price) / fk.vega * 1e4);
                st.fine_n++;
            }
        }
    }
    row.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return row;
}

static void print_row(const char* delta_label, const char* t_label, const RowResult& r) {
    char b1[16], b2[16], b3[16], b4[16], b5[16], b6[16], b7[16], b8[16];
    if (!r.built) { std::printf("  %5s %5s   (no surface)\n", delta_label, t_label); return; }
    const Stat& m = r.mid; const Stat& a = r.anchor;
    fmt(b1, 16, m.elig ? m.blend_max : std::nan("")); fmt(b2, 16, m.blend_rms());
    fmt(b3, 16, m.surf_n ? m.surf_max : std::nan("")); fmt(b4, 16, m.surf_rms());
    fmt(b5, 16, m.inv_n ? m.inv_max : std::nan(""));
    fmt(b6, 16, a.surf_n ? a.surf_max : std::nan("")); fmt(b7, 16, a.surf_rms());
    fmt(b8, 16, a.inv_n ? a.inv_max : std::nan(""));
    const double sens = (m.fine_n && m.elig) ? std::abs(m.blend_max_fine - m.blend_max) : std::nan("");
    const char* status = !m.complete() ? "incomplete"
        : (std::isnan(sens) || std::abs(m.blend_max - kQualifyBps) <= sens) ? "inconclusive"
        : (m.blend_max <= kQualifyBps ? "pass" : "fail");
    std::printf("  %5s %5s %3zu %4zu %s %s %s %s %5zu %s | %3zu %4zu %s %s %5zu %s | sens %6.1f  %-12s %.0fs\n",
                delta_label, t_label, m.q, m.elig, b1, b2, b3, b4, m.inv_n, b5,
                a.q, a.elig, b6, b7, a.inv_n, b8, sens, status, r.seconds);
    if (m.ref_fail || m.low_vega || m.low_tv || m.surf_nonfinite || m.inv_fail ||
        a.ref_fail || a.low_vega || a.low_tv || a.surf_nonfinite || a.inv_fail)
        std::printf("        excluded: mid ref-fail=%zu low-vega=%zu low-tv=%zu surf-nonfinite=%zu inv-fail=%zu"
                    " | anchor ref-fail=%zu low-vega=%zu low-tv=%zu surf-nonfinite=%zu inv-fail=%zu\n",
                    m.ref_fail, m.low_vega, m.low_tv, m.surf_nonfinite, m.inv_fail,
                    a.ref_fail, a.low_vega, a.low_tv, a.surf_nonfinite, a.inv_fail);
}

}  // namespace kref

static void run_kref_sweep() {
    using namespace kref;
    std::printf("\n================================================================\n");
    std::printf("K_ref spacing sweep — manual segmented B-spline, quarterly $0.50 calendar\n");
    std::printf("window K in [%.0f, %.0f]; bps = |price| / FD vega (estimate) except 'inv' = IV inversion error\n",
                kWindowLo, kWindowHi);
    std::printf("status: pass/fail = mid-anchor blend max vs %.0f bps; inconclusive = within ref sensitivity; "
                "incomplete = eligible mid-anchors < 90%%\n", kQualifyBps);
    std::printf("================================================================\n");
    std::map<RefKey, Ref> cache;
    for (double sigma : kSweepVols) {
        std::printf("\n  σ=%.0f%%\n", sigma * 100);
        std::printf("  %5s %5s %3s %4s %9s %9s %9s %9s %5s %9s | %3s %4s %9s %9s %5s %9s | %s\n",
                    "Δ", "T", "q", "elig", "blendmax", "blendrms", "surfmax", "surfrms", "inv n", "inv max",
                    "q", "elig", "surfmax", "surfrms", "inv n", "inv max", "ref-sens / status / time");
        for (double delta : kSpacings) {
            for (double T : kSweepMaturities) {
                char dl[8], tl[8];
                std::snprintf(dl, sizeof(dl), "%.2f", delta);
                std::snprintf(tl, sizeof(tl), "%.2f", T);
                auto row = run_row(delta, T, sigma, kBaseMoneynessKnots, kBaseTauPoints, cache);
                print_row(dl, tl, row);
            }
        }
        auto fine = run_row(2.5, 1.0, sigma, 81, 9, cache);
        print_row("fine", "1.00", fine);
        std::printf("  (fine = Δ 2.5, T 1.00 with 81 moneyness knots and 9 tau points per segment; "
                    "a surf max change > 2x means the surface floor is not converged in those axes)\n");
    }
}
```

Add `#include <chrono>`, `#include <map>`, `#include <tuple>`, `#include "mango/option/table/bspline/bspline_segmented_builder.hpp"`, `#include "mango/option/grid_spec_types.hpp"`.

`InterpolatedIVSolver` has no surface accessor (its `surface_` is private); the copy above is the intended approach. Do not add an accessor to the library.

- [ ] **Step 2: Wire the CLI**

In `main`: `bool need_kref = run_all || path == "kref";` and after the q0 block `if (need_kref) run_kref_sweep();`. Update the usage line to `[--path=all|bspline|chebyshev|q0|dividends|kref]`.

- [ ] **Step 3: Build and run the sweep (AC4)**

```bash
bazel build -c opt //benchmarks:interp_iv_safety 2>&1 | tail -1
TMPDIR="$D" OMP_NUM_THREADS=16 bazel-bin/benchmarks/interp_iv_safety --path=kref > "$D/kref.txt" 2>&1; echo "rc=$?"
grep -E "^\s+(10\.00|5\.00|2\.50|1\.25|fine) " "$D/kref.txt" | head -40
grep -c "ref-fail=[1-9]" "$D/kref.txt"
```
Expected: `rc=0`; 17 rows per σ (16 + fine) with numeric `blendmax` columns; the last grep prints `0` (no reference failures). `incomplete`/`inconclusive` statuses are acceptable outcomes. Note the total runtime printed per row; if the whole sweep exceeds 15 minutes, record it in the PR and do not shrink the design.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/interp_iv_safety.cc benchmarks/BUILD.bazel
git commit -m "Add K_ref spacing sweep to interp_iv_safety

Measure the MultiKRefSplit blend policy against a same-query FDM
control at four K_ref spacings, separating the blend's own error
from per-surface error and from IV inversion error. This is the
measured baseline #460 needs for the sparse-K_ref finding (#462)."
```

---

### Task 5: Documentation and the #460 baseline (D5, D6)

**Files:**
- Modify: `docs/API_GUIDE.md` (K_ref paragraphs under "Discrete Dividends with Adaptive Grid", near line 761)
- Modify: `benchmarks/interp_iv_safety.cc` (file header comment: mention `--path=kref` and the dividends config source)
- GitHub: comment on #460.

- [ ] **Step 1: Extract the table from the committed sweep run**

From `$D/kref.txt` collect, per σ and per (Δ, T): `blendmax`, `blendrms`, status, ref-sens. Also record `git rev-parse --short HEAD`, the date, and the build flags (`-c opt`, `OMP_NUM_THREADS`).

- [ ] **Step 2: Add the guide section**

After the paragraph that ends "…strikes outside the K_ref span, where the blend clamps to a single K_ref, and…" add:

```markdown
**Measured K_ref spacing baseline (2026-09-DD, `interp_iv_safety --path=kref`,
commit `<sha>`, `-c opt`).** Manual segmented B-spline surfaces, PUT, spot 100,
r = 5%, q = 2%, quarterly $0.50 calendar, query window K ∈ [85, 115], mid-anchor
strikes, eligibility = FDM reference with TV/K ≥ 1e-4 and vega ≥ 1e-4. `blend`
is the `MultiKRefSplit` policy applied to exact FDM prices, divided by FD vega
(an IV-equivalent estimate); status compares `blend max` with 10 bps and marks a
row `inconclusive` when the finer-reference sensitivity straddles it.

| Δ (Δ/spot) | T | σ | blend max (bps) | blend rms | status |
|---|---|---|---|---|---|
| 10 (10%) | 0.20 | 15% | … | … | … |
| … | | | | | |

Spacings that stay at or below 10 bps at every measured (T, σ) with every row
complete and none inconclusive: <list, or "none of the tested spacings">.
Rows marked incomplete or inconclusive: <list>. These numbers hold only under
the conditions above; they are a baseline for the spot-scaling work in #460,
not a guarantee for other spots, option types, rates or schedules.
```

Fill every `…` and `<…>` from the run. Never leave a placeholder.

- [ ] **Step 3: Post the same table to #460**

```bash
gh issue comment 460 --body-file "$D/kref-baseline-comment.md"
```
where the file holds the table and the two sentences above, plus the sentence "Produced by #462's `--path=kref`; see the API guide section of the same name."

- [ ] **Step 4: Update the benchmark's file header**

Extend the `@file` comment at the top of `interp_iv_safety.cc` with the two dividends reference grids, the source of the dividends config (`documented_adaptive_dividend_config()`), and the `kref` path.

- [ ] **Step 5: Commit**

```bash
git add docs/API_GUIDE.md benchmarks/interp_iv_safety.cc
git commit -m "Document the measured K_ref spacing baseline

Record the sweep's blend-policy errors by spacing and maturity in the
API guide's reference-strike section as a conditional baseline for
#460, with the conditions under which the numbers hold."
```

---

### Task 6: nightly viability case (D6, accepted by the user at the go/no-go)

**Files:**
- Modify: `tests/iv_solver_factory_slow_test.cc` (after `DocumentedBSplineConfigReportsAccuracyAndSolves`)

- [ ] **Step 1: Add the test**

```cpp
// Regression: interp_iv_safety's dividends path must keep building.
// Bug: the benchmark's B-spline per-maturity config rotted into a
// configuration the #454 viability gate refuses (moneyness range wider
// than its K_ref span, and three dividends scaled into a 7-day option),
// and nothing outside the manual benchmark noticed (#462). The documented
// pin above uses a different yield, schedule and a single maturity, so it
// cannot catch this. Viability only: no accuracy number is pinned here.
TEST(IVSolverFactorySegmented, BenchmarkDividendsConfigBuildsAtEveryMaturity) {
    // Mirrors benchmarks/interp_iv_safety.cc kDoc* constants + quarterly_div_schedule.
    const std::vector<double> maturities = {7.0 / 365, 14.0 / 365, 30.0 / 365, 60.0 / 365,
                                            90.0 / 365, 180.0 / 365, 1.0, 2.0};
    for (double T : maturities) {
        std::vector<Dividend> divs;
        for (double t = 0.25; t < T; t += 0.25) divs.push_back({.calendar_time = t, .amount = 0.50});
        IVSolverFactoryConfig config{
            .option_type = OptionType::PUT,
            .spot = 100.0,
            .dividend_yield = 0.02,
            .grid = IVGrid{.moneyness = {0.92, 0.95, 1.0, 1.05, 1.08},
                           .vol = {0.10, 0.15, 0.20, 0.30},
                           .rate = {0.02, 0.03, 0.05, 0.07}},
            .adaptive = AdaptiveGridParams{.target_iv_error = 0.001},
            .backend = BSplineBackend{},
            .discrete_dividends = DiscreteDividendConfig{
                .maturity = T, .discrete_dividends = divs,
                .kref_config = {.K_refs = {90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5, 110.0}}},
        };
        auto solver = make_interpolated_iv_solver(config);
        ASSERT_TRUE(solver.has_value()) << "T=" << T << " code " << static_cast<int>(solver.error().code);
        auto diag = solver->build_diagnostics();
        ASSERT_TRUE(diag.has_value());
        EXPECT_GT(diag->holdout_points_measured, 0u) << "T=" << T;
    }
}
```

- [ ] **Step 2: Run it**

```bash
TMPDIR="$D" bazel test //tests:iv_solver_factory_slow_test --test_filter='*BenchmarkDividendsConfigBuildsAtEveryMaturity*' --test_output=errors
```
Expected: PASS (about 10–15 s: eight B-spline builds at ~1 s each).

- [ ] **Step 3: Commit**

```bash
git add tests/iv_solver_factory_slow_test.cc
git commit -m "Pin viability of the benchmark dividends config nightly

The documented-config pin exercises a different yield, schedule and
maturity, so it could not catch the benchmark's config rotting into a
refusal (#462)."
```

---

### Task 7: Final verification and cleanup (AC1, AC3, AC5, AC6)

**Files:**
- Delete: `benchmarks/scratch_462.cc`; remove the `scratch_462` `cc_binary` block from `benchmarks/BUILD.bazel`; `git checkout MODULE.bazel.lock`.

- [ ] **Step 1: Remove the scratch driver**

```bash
rm benchmarks/scratch_462.cc
python3 - <<'EOF'
p='benchmarks/BUILD.bazel'; s=open(p).read()
i=s.index('\ncc_binary(\n    name = "scratch_462"'); j=s.index(')\n', i)+2
open(p,'w').write(s[:i]+s[j:])
EOF
git checkout MODULE.bazel.lock
git status --short   # expect only intentional changes, or nothing
```

- [ ] **Step 2: AC1 and warnings**

```bash
bazel build -c opt //benchmarks:interp_iv_safety 2>&1 | grep -c "warning:"
bazel build //benchmarks/... 2>&1 | tail -1
```
Expected: warning count equal to Task 0's build (record both numbers in the PR), `Build completed successfully`.

- [ ] **Step 3: AC3 (once, by hand, not committed)**

Temporarily change `kDocKRefs` to `{80.0, 100.0, 120.0}` and `kDocMoneyness` to `{0.70, 1.0, 1.30}` in a scratch edit, build, run `--path=dividends`, confirm `[FAILED] ... NoViableSurface`, `n/a (no surf)` in the TV/K block and `rc=1`. Then `git checkout benchmarks/interp_iv_safety.cc` and rebuild.

- [ ] **Step 4: AC5 diff against Task 0 baselines**

```bash
B=bazel-bin/benchmarks/interp_iv_safety
for p in bspline chebyshev q0; do
  TMPDIR="$D" OMP_NUM_THREADS=8 $B --path=$p > "$D/after-$p.txt" 2>&1
  diff <(grep -vE "Build time|elapsed|seconds|Usage:|Dividend strikes" "$D/baseline-$p.txt") \
       <(grep -vE "Build time|elapsed|seconds|Usage:|Dividend strikes" "$D/after-$p.txt") | head -20
done
```
Expected: no differing error values or table layout; the only differences allowed are the filtered lines, added count lines, and `n/a` where the baseline printed `0.0 (0)`.

- [ ] **Step 5: AC6**

```bash
TMPDIR="$D" bazel test //tests/... --test_tag_filters=-manual,-slow 2>&1 | tail -2
TMPDIR="$D" bazel test //tests:iv_solver_factory_slow_test 2>&1 | tail -2
```
Expected: all pass; the fast count equals `main`'s (run the same filter on `main` in the reference root if unsure).

- [ ] **Step 6: Commit cleanup, if anything changed**

```bash
git add -A benchmarks
git commit -m "Remove #462 scratch driver"   # only if the diff is non-empty
```

---

## Self-review

- **Spec coverage:** D1 → Task 3; D2 → Tasks 1, 3; D3 → Task 2 (+ `n/a` in Task 1); D4 → Task 4; D5 → Task 5; D6 → Task 3 step 7 and Task 6 (conditional); AC1/3/5/6 → Task 7; AC2 → Task 3 step 8; AC4 → Task 4 step 3; AC7 → Task 5.
- **Placeholders:** the API-guide template in Task 5 has `…`/`<…>` cells by design, to be filled from the run; the step says so explicitly.
- **Type consistency:** `PriceGridN<NS>`, `ErrorTableN<NS>`, `AlgoErrorsN<NS>`, `ScheduleFn`, `kDivStrikes`/`kNDS`, `kDoc*`, `quarterly_div_schedule`, `report_build_failure`/`report_wrap_failure`, `g_build_failed` are used with the same names in every task.
- **Resolved during planning:** `InterpolatedIVSolver` has no surface accessor (Task 4 copies the surface); `ValidationError::code` is the field name.
