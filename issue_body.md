## Summary

Research and design for a comprehensive error tracking system to monitor and report:
- Slow convergence events
- Internal errors (memory allocation, validation failures)
- Numerical instability (NaN, Inf, conservation violations)
- Convergence failures with actionable diagnostics

## Current State Analysis

### Existing Error Handling

The library currently uses:

1. **Return Codes**: Integer status codes (0 = success, -1 = failure)
   - `pde_solver_solve()` returns status from convergence failures
   - `pde_solver_create()` returns nullptr on validation/allocation errors
   - Limited context about failure reasons

2. **USDT Tracing Probes**: Zero-overhead runtime tracing
   - `IVCALC_TRACE_CONVERGENCE_FAILED` - convergence failures
   - `IVCALC_TRACE_VALIDATION_ERROR` - input validation errors
   - `IVCALC_TRACE_CONVERGENCE_ITER` - iteration progress
   - Pros: Zero overhead when disabled, production-safe
   - Cons: Requires root access, external tooling (bpftrace), not persistent

3. **Test Coverage**: Comprehensive stability tests
   - Stiff equation stability
   - NaN/Inf detection
   - Maximum principle preservation
   - Mass conservation
   - Long-time stability
   - Non-negativity preservation

### Gaps in Current System

1. **No structured error logging** - Errors are not persisted or aggregated
2. **Limited error context** - What parameters led to failure?
3. **No error classification** - Hard to categorize and analyze failures
4. **No runtime diagnostics** - Limited visibility into solver health during execution
5. **No error recovery guidance** - Users don't know how to fix issues
6. **No floating-point exception detection** - NaN/Inf not detected until too late

## Research Findings

### Industry Best Practices

#### 1. PETSc Error Handling System

PETSc (Portable, Extensible Toolkit for Scientific Computation) provides:

- **Monitor callbacks**: `KSPMonitorSet()`, `SNESMonitorSet()` for iteration monitoring
- **Converged reason codes**: `SNESConvergedReason`, `KSPConvergedReason` with detailed failure reasons
- **Runtime options**: `-snes_monitor`, `-ksp_monitor`, `-log_view` for diagnostics
- **Error checking macros**: `CHKERRQ(ierr)` for consistent error handling
- **Diagnostic tools**: `-snes_test_jacobian`, `-snes_linesearch_monitor` for debugging

**Key Insight**: Provide both callback-based monitoring AND reason codes for failures.

#### 2. IEEE 754 Exception Handling

From research on floating-point exception handling:

- **Five exception types**: Invalid operation, division by zero, overflow, underflow, inexact
- **Detection methods**:
  - Use `fenv.h` functions: `fetestexcept()`, `feclearexcept()`
  - Check for NaN: `isnan()`, `isinf()`, `isfinite()`
  - NaN payload propagation for diagnostic information
- **Tools**: Ariadne (auto-detection in GSL), GPU-FPX (GPU exception tracking)

**Key Insight**: Proactively detect NaN/Inf during computation, not just at the end.

#### 3. Modern Observability Patterns (2024)

From observability research:

- **Structured logging**: JSON format with key-value pairs for machine-parseability
- **Log aggregation**: Centralized storage with query capabilities
- **Three pillars**: Metrics, logs, traces (OpenTelemetry standard)
- **Error context**: Include timestamps, service names, error codes, user IDs
- **Real-time processing**: Filtering, enrichment, normalization of telemetry data

**Key Insight**: Structure error data for aggregation and analysis.

## Proposed Error Tracking System

### Design Philosophy

**Hybrid Approach**: Combine zero-overhead tracing with optional structured error logging

1. **Keep USDT probes** for real-time monitoring (zero overhead when disabled)
2. **Add error callback API** for structured error collection
3. **Provide error classification** with reason codes and context
4. **Enable user customization** - users choose their error handling strategy

### Core Components

#### 1. Error Classification System

```c
// Error categories
typedef enum {
    ERROR_CATEGORY_CONVERGENCE,    // Iterative solver convergence
    ERROR_CATEGORY_VALIDATION,     // Input validation
    ERROR_CATEGORY_NUMERICAL,      // Numerical instability
    ERROR_CATEGORY_MEMORY,         // Memory allocation
    ERROR_CATEGORY_INTERNAL        // Internal logic errors
} ErrorCategory;

// Detailed error codes
typedef enum {
    // Convergence errors (100-199)
    ERROR_CONVERGENCE_FAILED = 100,
    ERROR_CONVERGENCE_SLOW = 101,
    ERROR_CONVERGENCE_STAGNATED = 102,

    // Numerical errors (200-299)
    ERROR_NUMERICAL_NAN = 200,
    ERROR_NUMERICAL_INF = 201,
    ERROR_NUMERICAL_OVERFLOW = 202,
    ERROR_NUMERICAL_UNDERFLOW = 203,
    ERROR_NUMERICAL_LOSS_OF_PRECISION = 204,

    // Stability violations (300-399)
    ERROR_STABILITY_MAXIMUM_PRINCIPLE = 300,
    ERROR_STABILITY_MASS_CONSERVATION = 301,
    ERROR_STABILITY_NON_NEGATIVITY = 302,

    // Validation errors (400-499)
    ERROR_VALIDATION_PARAMETER = 400,
    ERROR_VALIDATION_BOUNDARY_CONFIG = 401,
    ERROR_VALIDATION_GRID = 402,

    // Memory errors (500-599)
    ERROR_MEMORY_ALLOCATION = 500,
    ERROR_MEMORY_ALIGNMENT = 501
} ErrorCode;

// Error severity levels
typedef enum {
    ERROR_SEVERITY_WARNING,    // Non-fatal, solver can continue
    ERROR_SEVERITY_ERROR,      // Fatal, solver cannot continue
    ERROR_SEVERITY_CRITICAL    // System-level failure
} ErrorSeverity;
```

#### 2. Error Context Structure

```c
// Comprehensive error context
typedef struct {
    ErrorCategory category;
    ErrorCode code;
    ErrorSeverity severity;
    int module_id;              // MODULE_PDE_SOLVER, etc.

    // Temporal context
    double timestamp;           // System time when error occurred
    size_t step;                // Time step number (if applicable)
    size_t iteration;           // Iteration number (if applicable)

    // Numerical context
    double error_metric;        // Convergence error, NaN location index, etc.
    double tolerance;           // Expected tolerance
    double *state_snapshot;     // Optional: snapshot of solution state
    size_t state_size;

    // Descriptive information
    const char *message;        // Human-readable error message
    const char *suggestion;     // Suggested fix/mitigation
    const char *location;       // File:line where error occurred
} ErrorContext;
```

#### 3. Error Callback API

```c
// User-provided error handler callback
typedef void (*ErrorHandlerFunc)(const ErrorContext *error, void *user_data);

// Global error handler configuration
typedef struct {
    ErrorHandlerFunc handler;
    void *user_data;
    ErrorSeverity min_severity;  // Only report errors >= this severity
    bool enable_state_snapshots; // Include solution state in context
} ErrorHandlerConfig;

// API functions
void pde_set_error_handler(const ErrorHandlerConfig *config);
void pde_clear_error_handler(void);
void pde_report_error(const ErrorContext *error);
```

#### 4. Enhanced USDT Probes

Add new probes for detailed error tracking:

```c
// Numerical stability probes
#define IVCALC_TRACE_NUMERICAL_ERROR(module_id, error_code, location_index, value) \
    DTRACE_PROBE4(IVCALC_PROVIDER, numerical_error, module_id, error_code, location_index, value)

#define IVCALC_TRACE_STABILITY_VIOLATION(module_id, violation_type, metric, threshold) \
    DTRACE_PROBE4(IVCALC_PROVIDER, stability_violation, module_id, violation_type, metric, threshold)

// Convergence health probes
#define IVCALC_TRACE_CONVERGENCE_SLOW(module_id, step, iter_percent, error) \
    DTRACE_PROBE4(IVCALC_PROVIDER, convergence_slow, module_id, step, iter_percent, error)

#define IVCALC_TRACE_CONVERGENCE_STAGNATED(module_id, step, stagnant_iters, error) \
    DTRACE_PROBE4(IVCALC_PROVIDER, convergence_stagnated, module_id, step, stagnant_iters, error)
```

#### 5. Runtime Diagnostics

Add solver health monitoring:

```c
typedef struct {
    // Convergence statistics
    size_t total_iterations;
    size_t failed_steps;
    size_t slow_convergence_warnings;
    double avg_iterations_per_step;
    double max_iterations_per_step;

    // Numerical health
    size_t nan_detections;
    size_t inf_detections;
    size_t stability_violations;

    // Performance metrics
    double total_solve_time;
    double avg_step_time;
} SolverDiagnostics;

// API to query diagnostics
const SolverDiagnostics* pde_solver_get_diagnostics(const PDESolver *solver);
void pde_solver_reset_diagnostics(PDESolver *solver);
```

### Implementation Priorities

#### Phase 1: Core Error Infrastructure (High Priority)

1. **Error classification system** - Define error codes, categories, severity levels
2. **Error context structure** - Standardize error information
3. **Error callback API** - Allow users to hook into error reporting
4. **Enhance existing USDT probes** - Add error code parameters

**Files to modify:**
- `src/error_tracking.h` (new) - Error types and API
- `src/error_tracking.c` (new) - Error handler implementation
- `src/pde_solver.c` - Integrate error reporting
- `src/ivcalc_trace.h` - Add new probe definitions

#### Phase 2: Numerical Stability Detection (High Priority)

1. **NaN/Inf detection** - Check solution after each iteration
2. **Convergence health monitoring** - Detect slow/stagnant convergence
3. **Stability violation checks** - Maximum principle, mass conservation (optional)

**Files to modify:**
- `src/pde_solver.c` - Add stability checks in `solve_implicit_step()`
- `src/stability_checks.c` (new) - Standalone stability test functions

#### Phase 3: Diagnostics and Reporting (Medium Priority)

1. **Solver diagnostics** - Accumulate statistics during solve
2. **Diagnostic query API** - Let users access solver health metrics
3. **Structured error logging** - Built-in file logger (optional)

**Files to modify:**
- `src/pde_solver.h` - Add `SolverDiagnostics` to solver struct
- `src/pde_solver.c` - Update diagnostics counters
- `src/error_logger.c` (new) - JSON/CSV error logger

#### Phase 4: User Tools and Documentation (Low Priority)

1. **Error analysis scripts** - Aggregate and analyze error logs
2. **bpftrace scripts** - Monitor errors in real-time
3. **User guide** - How to use error tracking system
4. **Example programs** - Demonstrate error handling

**New files:**
- `scripts/error_analysis.py` - Parse error logs, generate reports
- `scripts/tracing/error_monitor.bt` - Real-time error monitoring
- `docs/ERROR_TRACKING.md` - User guide
- `examples/error_handling_example.c` - Demo program

### Design Decisions

#### Why Hybrid Approach?

1. **Zero overhead for production**: USDT probes remain no-op when not traced
2. **Flexibility**: Users choose their own error handling strategy
3. **Backward compatibility**: Existing API unchanged, new features opt-in
4. **Library philosophy**: No stdout/stderr pollution, users control output

#### Why Error Callbacks?

1. **Customization**: Users can log to files, databases, monitoring systems
2. **Framework integration**: Easy to integrate with existing logging systems
3. **Testing**: Tests can capture errors programmatically
4. **Production**: Can aggregate errors for analysis

#### Why Not Always-On Logging?

1. **Performance**: I/O overhead in hot loops is unacceptable
2. **Library design**: Libraries shouldn't dictate logging strategy
3. **User control**: Users should choose when/where to log
4. **USDT covers it**: Real-time monitoring via USDT is sufficient

### Usage Examples

#### Example 1: Custom Error Logger

```c
void my_error_handler(const ErrorContext *error, void *user_data) {
    FILE *log = (FILE*)user_data;
    fprintf(log, "{\"timestamp\": %f, \"category\": %d, \"code\": %d, "
                 "\"step\": %zu, \"error\": %g, \"message\": \"%s\"}\n",
            error->timestamp, error->category, error->code,
            error->step, error->error_metric, error->message);
}

int main() {
    FILE *error_log = fopen("errors.jsonl", "w");

    ErrorHandlerConfig config = {
        .handler = my_error_handler,
        .user_data = error_log,
        .min_severity = ERROR_SEVERITY_WARNING,
        .enable_state_snapshots = false
    };

    pde_set_error_handler(&config);

    // ... run solver ...

    pde_clear_error_handler();
    fclose(error_log);
}
```

#### Example 2: Real-Time Error Monitoring with bpftrace

```bash
# Monitor all numerical errors
sudo bpftrace -e '
usdt:./my_program:ivcalc:numerical_error {
    printf("[NUMERICAL ERROR] Module=%d Code=%d Location=%d Value=%f\n",
           arg0, arg1, arg2, arg3);
}

usdt:./my_program:ivcalc:convergence_slow {
    printf("[SLOW CONVERGENCE] Module=%d Step=%d IterPercent=%d%% Error=%f\n",
           arg0, arg1, arg2, arg3);
}' -c './my_program'
```

#### Example 3: Diagnostics Query

```c
pde_solver_solve(solver);

const SolverDiagnostics *diag = pde_solver_get_diagnostics(solver);

printf("Total iterations: %zu\n", diag->total_iterations);
printf("Failed steps: %zu\n", diag->failed_steps);
printf("Slow convergence warnings: %zu\n", diag->slow_convergence_warnings);
printf("Avg iterations/step: %.2f\n", diag->avg_iterations_per_step);
printf("NaN detections: %zu\n", diag->nan_detections);
printf("Stability violations: %zu\n", diag->stability_violations);
```

### Open Questions

1. **State snapshots**: Should we include full solution state in error context? (memory overhead)
2. **Error recovery**: Should we attempt automatic recovery (e.g., reduce dt on convergence failure)?
3. **Error aggregation**: Should library provide built-in log aggregation, or leave to users?
4. **Thread safety**: How to handle errors in multi-threaded environments?
5. **Stability checks**: Which checks should be always-on vs. optional? (performance impact)

### Testing Strategy

1. **Unit tests**: Test error reporting for each error code
2. **Integration tests**: Verify error callbacks are invoked correctly
3. **Stability tests**: Add tests that intentionally trigger numerical errors
4. **Performance tests**: Ensure zero overhead when error handler not set
5. **Tracing tests**: Verify USDT probes fire correctly (requires bpftrace)

### Backward Compatibility

All new features are **opt-in**:
- Existing code continues to work unchanged
- Error handler is `nullptr` by default (no callbacks)
- USDT probes remain no-op when not traced
- Return codes unchanged
- ABI compatibility maintained

### Documentation Requirements

1. **User guide**: `docs/ERROR_TRACKING.md` - How to use the system
2. **API reference**: Document new error types, callbacks, diagnostics API
3. **bpftrace guide**: How to monitor errors in real-time
4. **Troubleshooting**: Map error codes to solutions
5. **Examples**: Demonstrate error handling patterns

## Success Criteria

The error tracking system should:

1. ✅ **Report all critical events**: Convergence failures, NaN/Inf, validation errors
2. ✅ **Provide actionable context**: What went wrong, what parameters, suggested fixes
3. ✅ **Zero overhead when disabled**: No performance impact in production
4. ✅ **Flexible integration**: Works with any logging/monitoring system
5. ✅ **Production-ready**: Safe to use in production environments
6. ✅ **Well-documented**: Clear guides and examples
7. ✅ **Backward compatible**: Existing code continues to work

## References

- **PETSc Manual**: https://petsc.org/release/manual/manual.pdf
- **IEEE 754 Exception Handling**: SEI CERT C Coding Standard FLP03-C
- **Floating-Point Exception Detection**: "Automatic detection of floating-point exceptions" (POPL 2013)
- **Structured Logging Best Practices**: Better Stack Community Guide
- **OpenTelemetry Observability**: https://opentelemetry.io/docs/concepts/observability-primer/

## Next Steps

1. Review and approve design
2. Implement Phase 1 (core error infrastructure)
3. Add comprehensive tests
4. Update documentation
5. Roll out incrementally with feature flags

---

**Research conducted by:** Claude
**Date:** 2025-10-29
