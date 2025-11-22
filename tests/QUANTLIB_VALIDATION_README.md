# QuantLib Validation Framework

Unified testing framework for validating mango-iv pricing and implied volatility solvers against QuantLib reference implementation.

## Overview

This framework provides:
- **Generic testing utilities** for option pricing and IV accuracy
- **Batch validation** for testing multiple scenarios at once
- **Support for multiple solvers**: FDM pricing, FDM IV, Interpolated IV
- **Auto-estimation mode** testing (validates production behavior)

## Files

### Framework

- **`quantlib_validation_framework.hpp`** - Core testing framework
  - `validate_pricing()` - Test option pricing accuracy
  - `validate_iv_fdm()` - Test FDM-based IV accuracy
  - `validate_batch()` - Batch test multiple scenarios
  - `get_standard_test_scenarios()` - 8 standard test cases

### Tests

- **`quantlib_accuracy_batch_test.cc`** - Unified batch validation tests
  - FDM pricing + IV (16 tests)
  - Greeks accuracy
  - Grid convergence
  - Interpolated IV (future work)

- **`quantlib_accuracy_test.cc`** - Legacy individual tests (manual tag)

## Running Tests

```bash
# Run batch validation (pricing + IV)
bazel test //tests:quantlib_accuracy_batch_test

# Run specific test
bazel test //tests:quantlib_accuracy_batch_test --test_filter="*FDM*"
```

## Test Results

**StandardScenarios_Pricing_And_IV_FDM**: ✅ PASSED (16/16 tests)
- Validates both pricing and IV accuracy for 8 standard scenarios
- Uses auto-estimation mode (production behavior)
- All tests converge within 1-2% tolerance

**GridConvergence**: ✅ PASSED
- Auto-estimated grid converges to within 1% of high-resolution reference

**Greeks_ATM**: ✅ PASSED
- Delta within 2%, Gamma within 5% of QuantLib

## Standard Test Scenarios

1. **ATM Put 1Y** - S=100, K=100, T=1.0, σ=0.20
2. **OTM Put 3M** - S=110, K=100, T=0.25, σ=0.30
3. **ITM Put 2Y** - S=90, K=100, T=2.0, σ=0.25
4. **ATM Call 1Y** - S=100, K=100, T=1.0, σ=0.20
5. **Deep ITM Put 6M** - S=80, K=100, T=0.5, σ=0.25
6. **High Vol Put 1Y** - S=100, K=100, T=1.0, σ=0.50
7. **Low Vol Put 1Y** - S=100, K=100, T=1.0, σ=0.10
8. **Long Maturity Put 5Y** - S=100, K=100, T=5.0, σ=0.20

## Adding New Test Scenarios

```cpp
#include "tests/quantlib_validation_framework.hpp"

TEST(MyTest, CustomScenario) {
    std::vector<OptionTestScenario> scenarios = {
        {"My Scenario", spot, strike, maturity, vol, rate, div, is_call,
         /*price_tol=*/1.0, /*greeks_tol=*/2.0, /*iv_tol=*/2.0}
    };

    auto summary = validate_batch(scenarios);
    summary.print_summary();
    EXPECT_TRUE(summary.all_passed());
}
```

## Performance

**Batch validation** (8 scenarios):
- Pricing validation: ~250 ms (8 × ~30ms per FDM solve)
- IV validation: ~900 ms (8 × ~110ms per FDM IV solve)
- Total: ~1.15 seconds for 16 validations

## Future Work

### Interpolated IV Testing

The `DISABLED_StandardScenarios_IV_Interpolated` test validates the B-spline-based fast IV solver. Currently disabled due to:

**Issue**: Newton-Raphson fails with "Derivative too small (flat region)"

**Root Cause**: Price table grid may need:
1. Finer volatility grid spacing (currently 0.05 steps)
2. Better initial guess for Newton solver
3. Adaptive grid refinement near critical points

**Performance Target**: ~30µs per IV (vs ~143ms for FDM, 4800× speedup)

**Next Steps**:
1. Increase price table resolution (0.05 → 0.02 volatility steps)
2. Implement adaptive Newton initialization
3. Add vega bounds checking

## Dependencies

- **QuantLib** - Reference implementation (`libquantlib0-dev`)
- **GoogleTest** - Testing framework
- **mango-iv** - Option pricing library
  - `american_option` - FDM pricing solver
  - `iv_solver_fdm` - FDM-based IV solver
  - `iv_solver_interpolated` - B-spline IV solver
  - `price_table_4d_builder` - Price table construction
