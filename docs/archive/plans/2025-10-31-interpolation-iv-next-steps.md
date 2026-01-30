<!-- SPDX-License-Identifier: MIT -->
# Interpolation-Based American IV - Next Steps

**Status:** Deferred to future milestone

**Goal:** Implement fast American IV queries (~7.5µs) via 3D price table inversion

**Prerequisites:**
1. ✅ FDM-based American IV (ground truth for validation)
2. ✅ Let's Be Rational (for comparison)
3. ⏳ Extended price_table to support 3D grids (x, T, σ)

**Planned Approach:**

See design document `docs/plans/2025-10-31-american-iv-implementation-design.md`
Section: "Component 3: Interpolation-Based American IV"

**Grid specifications:**
- 100 × 80 × 40 (log-moneyness × maturity × volatility)
- ~2.7 MB memory
- 1bp accuracy target

**Implementation tasks:**
1. Extend OptionPriceTable to 3D
2. Implement precomputation workflow
3. Implement calculate_iv_interpolated()
4. Add validation tests (FDM vs interpolation)
5. Add performance benchmarks

**Validation criteria:**
- < 1bp difference from FDM IV on test set
- < 10µs query time
- > 30,000x speedup vs FDM

**Reference:**
- `docs/IV_SURFACE_PRECOMPUTATION_GUIDE.md` for grid sizing
- Issue #40 for coordinate transformations
