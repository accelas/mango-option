# Piecewise Chebyshev 4D on x-axis — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the x = ln(S/K) axis into 3 spectral elements, each with its own
ChebyshevTucker4D interpolant, to reduce per-segment EEP dynamic range and eliminate
the monotonic wing error (currently 20-65 bps at K=110-120 for T>=60d).

**Architecture:** Reuse the same 90 PDE batch. Each segment gets its own CGL nodes on
its x-sub-domain, shares (tau, sigma, rate) axes. Two-phase builder: (1) solve PDEs and
cache splines per (sigma, rate, tau) triple, (2) resample each segment's CGL nodes from
the cached splines. Query-time routing picks the segment containing the query x.

**Tech Stack:** ChebyshevTucker4D (existing), benchmark headers, interp_iv_safety.cc.

**Note:** Production codebase has `SplicedSurface` (src/option/table/spliced_surface.hpp)
for this pattern. For the benchmark we use a self-contained wrapper.

---

### Task 1: Add piecewise config, result, and inner class

**Files:**
- Modify: `benchmarks/chebyshev_4d_eep_inner.hpp` (after line 157, `Chebyshev4DEEPResult`)

**Step 1: Add config, result, and inner class after `Chebyshev4DEEPResult`**

Insert this code after `Chebyshev4DEEPResult` (line 157) and before
`build_chebyshev_4d_eep` (line 163):

```cpp
// ============================================================================
// Piecewise Chebyshev 4D: spectral elements along x-axis
// ============================================================================

struct PiecewiseChebyshev4DConfig {
    // Segment boundaries for x-axis (N+1 values for N segments, ascending)
    std::vector<double> x_breaks = {-0.50, -0.10, 0.15, 0.40};
    size_t num_x_per_seg = 15;    // CGL nodes per segment

    // Shared axes (same as Chebyshev4DEEPConfig)
    size_t num_tau = 15;
    size_t num_sigma = 15;
    size_t num_rate = 6;
    double epsilon = 1e-8;

    double tau_min = 0.019;
    double tau_max = 2.0;
    double sigma_min = 0.05;
    double sigma_max = 0.50;
    double rate_min = 0.01;
    double rate_max = 0.10;

    double dividend_yield = 0.0;
    bool use_hard_max = true;
};

struct PiecewiseChebyshev4DResult {
    std::vector<ChebyshevTucker4D> segments;
    std::vector<double> x_bounds;   // N+1 boundaries (with outer headroom)
    int n_pde_solves;
    double build_seconds;
};

class PiecewiseChebyshev4DEEPInner {
public:
    PiecewiseChebyshev4DEEPInner(std::vector<ChebyshevTucker4D> segments,
                                  std::vector<double> x_bounds,
                                  OptionType type, double K_ref,
                                  double dividend_yield)
        : segments_(std::move(segments)), x_bounds_(std::move(x_bounds)),
          type_(type), K_ref_(K_ref), dividend_yield_(dividend_yield) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        size_t seg = find_segment(x);
        double eep = segments_[seg].eval({x, q.tau, q.sigma, q.rate});

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return eep * (q.strike / K_ref_) + eu.value();
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        size_t seg = find_segment(x);
        std::array<double, 4> coords = {x, q.tau, q.sigma, q.rate};

        double eep_vega = (q.strike / K_ref_) * segments_[seg].partial(2, coords);

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return eep_vega + eu.vega();
    }

private:
    [[nodiscard]] size_t find_segment(double x) const {
        size_t n = segments_.size();
        for (size_t i = 0; i + 1 < n; ++i) {
            if (x < x_bounds_[i + 1]) return i;
        }
        return n - 1;
    }

    std::vector<ChebyshevTucker4D> segments_;
    std::vector<double> x_bounds_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
};
```

**Step 2: Verify compilation**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel build //benchmarks:interp_iv_safety`
Expected: compiles (new types defined but not yet used)

**Step 3: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor
git add benchmarks/chebyshev_4d_eep_inner.hpp
git commit -m "Add piecewise Chebyshev 4D config and inner class"
```

---

### Task 2: Add two-phase piecewise builder

**Files:**
- Modify: `benchmarks/chebyshev_4d_eep_inner.hpp` (insert after `PiecewiseChebyshev4DEEPInner`,
  before the existing `build_chebyshev_4d_eep` function)

The builder uses a two-phase strategy:
1. **Phase 1:** Solve PDE batch, build and cache one cubic spline per (sigma, rate, tau) triple
2. **Phase 2:** For each segment, resample each cached spline at that segment's CGL x-nodes

This avoids rebuilding N_seg × N_sigma × N_rate × N_tau = 3 × 90 × 15 = 4050 splines
and instead builds only 90 × 15 = 1350 once, reusing for all 3 segments.

**Step 1: Add builder function**

Insert after `PiecewiseChebyshev4DEEPInner` closing brace, before `build_chebyshev_4d_eep`:

```cpp
inline PiecewiseChebyshev4DResult build_piecewise_chebyshev_4d_eep(
    const PiecewiseChebyshev4DConfig& cfg,
    double K_ref,
    OptionType option_type)
{
    const size_t n_seg = cfg.x_breaks.size() - 1;

    // ---- 1. Shared axes: headroom + CGL nodes ----
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double htau   = headroom_fn(cfg.tau_min,   cfg.tau_max,   cfg.num_tau);
    double hsigma = headroom_fn(cfg.sigma_min, cfg.sigma_max, cfg.num_sigma);
    double hrate  = headroom_fn(cfg.rate_min,  cfg.rate_max,  cfg.num_rate);

    double tau_lo   = std::max(cfg.tau_min - htau, 1e-4);
    double tau_hi   = cfg.tau_max + htau;
    double sigma_lo = std::max(cfg.sigma_min - hsigma, 0.01);
    double sigma_hi = cfg.sigma_max + hsigma;
    double rate_lo  = std::max(cfg.rate_min - hrate, -0.05);
    double rate_hi  = cfg.rate_max + hrate;

    auto tau_nodes   = chebyshev_nodes(cfg.num_tau,   tau_lo,   tau_hi);
    auto sigma_nodes = chebyshev_nodes(cfg.num_sigma, sigma_lo, sigma_hi);
    auto rate_nodes  = chebyshev_nodes(cfg.num_rate,  rate_lo,  rate_hi);

    // ---- 2. Per-segment x-domains (headroom on outer edges only) ----
    std::vector<double> x_bounds(n_seg + 1);
    std::vector<std::vector<double>> seg_x_nodes(n_seg);

    for (size_t s = 0; s < n_seg; ++s) {
        double lo = cfg.x_breaks[s];
        double hi = cfg.x_breaks[s + 1];

        if (s == 0) {
            double h = headroom_fn(lo, hi, cfg.num_x_per_seg);
            lo -= h;
        }
        if (s == n_seg - 1) {
            double h = headroom_fn(cfg.x_breaks[s], cfg.x_breaks[s + 1],
                                    cfg.num_x_per_seg);
            hi += h;
        }

        x_bounds[s] = lo;
        if (s == n_seg - 1) x_bounds[s + 1] = hi;

        seg_x_nodes[s] = chebyshev_nodes(cfg.num_x_per_seg, lo, hi);
    }
    // Interior boundaries use the break values directly
    for (size_t s = 1; s < n_seg; ++s) {
        x_bounds[s] = cfg.x_breaks[s];
    }

    auto t0 = std::chrono::steady_clock::now();

    // ---- 3. Shared PDE batch: N_sigma x N_rate ----
    const double tau_solve = tau_nodes.back() * 1.01;

    std::vector<PricingParams> batch;
    batch.reserve(cfg.num_sigma * cfg.num_rate);
    for (size_t sv = 0; sv < cfg.num_sigma; ++sv) {
        for (size_t rv = 0; rv < cfg.num_rate; ++rv) {
            batch.emplace_back(
                OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = tau_solve,
                           .rate = rate_nodes[rv],
                           .dividend_yield = cfg.dividend_yield,
                           .option_type = option_type},
                sigma_nodes[sv]);
        }
    }

    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
    solver.set_snapshot_times(std::span<const double>{tau_nodes});
    auto batch_result = solver.solve_batch(batch, /*use_shared_grid=*/true);

    // ---- 4. Phase 1: cache splines per (sigma, rate, tau) ----
    // Layout: spline_cache[batch_idx][tau_idx] — one CubicSpline per triple
    const size_t Ns = cfg.num_sigma;
    const size_t Nr = cfg.num_rate;
    const size_t Nt = cfg.num_tau;
    const size_t Nx = cfg.num_x_per_seg;

    struct SplineEntry {
        CubicSpline<double> spline;
        bool valid = false;
    };
    std::vector<std::vector<SplineEntry>> spline_cache(Ns * Nr);

    for (size_t sv = 0; sv < Ns; ++sv) {
        for (size_t rv = 0; rv < Nr; ++rv) {
            size_t batch_idx = sv * Nr + rv;
            auto& cache = spline_cache[batch_idx];
            cache.resize(Nt);

            if (!batch_result.results[batch_idx].has_value()) continue;

            const auto& result = batch_result.results[batch_idx].value();
            auto x_grid = result.grid()->x();

            for (size_t j = 0; j < Nt; ++j) {
                auto spatial = result.at_time(j);
                if (!cache[j].spline.build(x_grid, spatial).has_value()) {
                    cache[j].valid = true;
                }
            }
        }
    }

    // ---- 5. Phase 2: fill per-segment tensors from cached splines ----
    std::vector<ChebyshevTucker4D> segments;
    segments.reserve(n_seg);

    for (size_t seg = 0; seg < n_seg; ++seg) {
        std::vector<double> tensor(Nx * Nt * Ns * Nr, 0.0);
        const auto& x_nodes = seg_x_nodes[seg];

        for (size_t sv = 0; sv < Ns; ++sv) {
            double sigma = sigma_nodes[sv];
            for (size_t rv = 0; rv < Nr; ++rv) {
                double rate = rate_nodes[rv];
                size_t batch_idx = sv * Nr + rv;
                const auto& cache = spline_cache[batch_idx];

                for (size_t j = 0; j < Nt; ++j) {
                    if (!cache[j].valid) continue;
                    double tau = tau_nodes[j];

                    for (size_t i = 0; i < Nx; ++i) {
                        double am = cache[j].spline.eval(x_nodes[i]) * K_ref;

                        double spot_local = std::exp(x_nodes[i]) * K_ref;
                        auto eu = EuropeanOptionSolver(
                            OptionSpec{.spot = spot_local, .strike = K_ref,
                                       .maturity = tau, .rate = rate,
                                       .dividend_yield = cfg.dividend_yield,
                                       .option_type = option_type},
                            sigma).solve().value();

                        double eep_raw = am - eu.value();

                        constexpr double kSharpness = 100.0;
                        double eep;
                        if (kSharpness * eep_raw > 500.0) {
                            eep = eep_raw;
                        } else {
                            double softplus =
                                std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
                            double bias = std::log(2.0) / kSharpness;
                            eep = cfg.use_hard_max
                                ? std::max(0.0, softplus - bias)
                                : (softplus - bias);
                        }

                        tensor[i * Nt * Ns * Nr + j * Ns * Nr + sv * Nr + rv] = eep;
                    }
                }
            }
        }

        double seg_x_lo = x_bounds[seg];
        double seg_x_hi = x_bounds[seg + 1];

        ChebyshevTucker4DDomain dom{
            .bounds = {{{seg_x_lo, seg_x_hi}, {tau_lo, tau_hi},
                        {sigma_lo, sigma_hi}, {rate_lo, rate_hi}}}};
        ChebyshevTucker4DConfig tcfg{
            .num_pts = {Nx, Nt, Ns, Nr},
            .epsilon = cfg.epsilon};

        segments.push_back(
            ChebyshevTucker4D::build_from_values(tensor, dom, tcfg));
    }

    auto t1 = std::chrono::steady_clock::now();

    return {std::move(segments), x_bounds,
            static_cast<int>(cfg.num_sigma * cfg.num_rate),
            std::chrono::duration<double>(t1 - t0).count()};
}
```

**Step 2: Verify compilation**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel build //benchmarks:interp_iv_safety`
Expected: compiles

**Step 3: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor
git add benchmarks/chebyshev_4d_eep_inner.hpp
git commit -m "Add two-phase piecewise Chebyshev 4D builder"
```

---

### Task 3: Add cheb4d-pw section to interp_iv_safety

**Files:**
- Modify: `benchmarks/interp_iv_safety.cc`

**Step 1: Add builder helper and section runner**

Insert after `run_cheb4d_diag()` (line 1113) and before `// Main` (line 1115):

```cpp
static PiecewiseChebyshev4DEEPInner build_piecewise_chebyshev_4d_surface() {
    PiecewiseChebyshev4DConfig cfg;
    // Default breaks: [-0.50, -0.10, 0.15, 0.40], 15 nodes/seg

    auto t0 = std::chrono::steady_clock::now();
    auto result = build_piecewise_chebyshev_4d_eep(cfg, kSpot, OptionType::PUT);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::printf("  Piecewise 4D: %d PDE solves, %zu segments, %.3fs build\n",
                result.n_pde_solves, result.segments.size(), elapsed);
    for (size_t s = 0; s < result.segments.size(); ++s) {
        auto r = result.segments[s].ranks();
        std::printf("    seg %zu [%.2f, %.2f]: ranks=(%zu,%zu,%zu,%zu)\n",
                    s, result.x_bounds[s], result.x_bounds[s + 1],
                    r[0], r[1], r[2], r[3]);
    }

    return PiecewiseChebyshev4DEEPInner(
        std::move(result.segments), std::move(result.x_bounds),
        OptionType::PUT, kSpot, 0.0);
}

static void run_cheb4d_pw() {
    std::printf("\n================================================================\n");
    std::printf("Piecewise Chebyshev 4D (Brent) — q=0, no dividends\n");
    std::printf("================================================================\n\n");

    const auto& q0_prices = get_q0_prices();

    std::printf("--- Building Piecewise Chebyshev 4D surface...\n");
    auto inner = build_piecewise_chebyshev_4d_surface();

    std::printf("--- Computing IV errors (Brent)...\n");
    for (size_t vi = 0; vi < kNV; ++vi) {
        char title[128];
        std::snprintf(title, sizeof(title),
                      "Piecewise Cheb 4D IV Error (bps, Brent) — σ=%.0f%%, q=0",
                      kVols[vi] * 100);
        auto errors = compute_errors_brent(q0_prices,
            [&](double S, double K, double tau, double sigma, double r) {
                PriceQuery q{.spot = S, .strike = K, .tau = tau,
                             .sigma = sigma, .rate = r};
                return inner.price(q);
            }, vi);
        print_heatmap(title, errors);
    }
}
```

**Step 2: Register section in kSections array**

Change line 1119-1122:
```cpp
static constexpr const char* kSections[] = {
    "vanilla", "dividends", "bspline-3d", "bspline-4d", "cheb3d", "cheb4d",
    "cheb4d-diag", "cheb4d-pw"
};
```

**Step 3: Add dispatch in main**

After `if (want("cheb4d-diag")) run_cheb4d_diag();` (line 1161), add:
```cpp
    if (want("cheb4d-pw"))   run_cheb4d_pw();
```

**Step 4: Build**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel build //benchmarks:interp_iv_safety`
Expected: compiles

**Step 5: Commit**

```bash
cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor
git add benchmarks/interp_iv_safety.cc
git commit -m "Add piecewise Chebyshev 4D benchmark section"
```

---

### Task 4: Run benchmark comparison

**Step 1: Run piecewise alongside global and B-spline**

Run: `cd /home/kai/work/mango-option/.worktrees/chebyshev-tensor && bazel run //benchmarks:interp_iv_safety -- bspline-4d cheb4d cheb4d-pw`
Expected: three sets of heatmaps. Compare σ=30% column for K=110-120, T>=60d:

| Method | K=110, T=60d | K=120, T=60d | K=120, T=1y |
|--------|-------------|-------------|-------------|
| B-spline 4D | 3.7 | 14.3 | 1.3 |
| Cheb 4D global | 20.4 | 64.6 | 23.3 |
| **Cheb 4D piecewise** | **target <10** | **target <15** | **target <5** |

**Step 2: If wing errors improved, commit results as comment in source**

No code changes needed — just verify the heatmap output matches expectations.

---

### Task 5: Tune segment boundaries (data-driven)

Only do this task if Task 4 results show specific segments need adjustment.

Possible adjustments:
1. **Move break points**: If ATM segment is too wide, try `{-0.50, -0.05, 0.10, 0.40}`
2. **Add a 4th segment**: If short-tau × deep-ITM still bad, split ITM into two
3. **Vary nodes per segment**: If one segment has higher ranks, give it more nodes

**Step 1: Modify `PiecewiseChebyshev4DConfig` defaults in `build_piecewise_chebyshev_4d_surface()`**
**Step 2: Rebuild and rerun**
**Step 3: Compare heatmaps**
**Step 4: Commit best configuration**

This task is iterative — run, observe, adjust, repeat.

---

## Why no blending at segment boundaries

During Brent's method for IV recovery, x = ln(S/K) is **fixed** for each root-finding
call. Brent varies σ while S, K, τ, r are constant. Since x doesn't change during the
solve, the query always hits the same segment — no risk of segment-bouncing causing
discontinuities in the Brent objective function.

If piecewise surfaces are later used for delta/gamma (where x varies), C¹ blending would
matter. Not needed for the IV benchmark experiment.

## Files summary

| File | Action |
|------|--------|
| `benchmarks/chebyshev_4d_eep_inner.hpp` | Add config, inner class, two-phase builder |
| `benchmarks/interp_iv_safety.cc` | Add `cheb4d-pw` section + registration |
| `benchmarks/BUILD.bazel` | No changes needed (deps already present) |

## Verification

1. `bazel build //benchmarks:interp_iv_safety` — compiles
2. `bazel run //benchmarks:interp_iv_safety -- cheb4d-pw` — runs, prints heatmaps
3. Compare σ=30% T>=60d wing errors vs global Chebyshev and B-spline
4. `bazel test //...` — no regressions (no library changes)

## Success criteria

- σ=30%, T>=60d, K=110-120: below 10 bps (currently 20-65 bps with global)
- σ=30%, T>=1y, all strikes: below 5 bps (currently 5-29 bps with global)
- PDE count: still 90 (unchanged)
- No new library code (benchmark-only experiment)
