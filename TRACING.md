# USDT Tracing Guide for iv_calc

This document explains how to use User Statically-Defined Tracing (USDT) probes in the iv_calc library for dynamic debugging, performance analysis, and production monitoring.

## Overview

The iv_calc library includes USDT probe points that provide zero-overhead tracing capabilities. When tracing is disabled (the default), probes compile to single NOP instructions with negligible performance impact. When enabled via tracing tools, probes capture detailed runtime information without requiring library recompilation.

## Benefits of USDT Tracing

- **Zero overhead when disabled**: Probes are NOPs when not actively traced
- **Dynamic enablement**: Enable/disable tracing at runtime without recompilation
- **Production-safe**: Can be used in production environments
- **Rich context**: Captures solver state, convergence metrics, and performance data
- **Standard tooling**: Works with bpftrace, systemtap, perf, and other Linux tracing tools

## Building with USDT Support

### Prerequisites

On Debian/Ubuntu systems:
```bash
sudo apt-get install systemtap-sdt-dev
```

On RHEL/Fedora systems:
```bash
sudo yum install systemtap-sdt-devel
```

### Build Commands

**Standard build (USDT probes are no-ops):**
```bash
bazel build //src:pde_solver
```

**USDT-enabled build (requires systemtap-sdt-dev):**
```bash
bazel build //src:pde_solver_usdt
```

The USDT-enabled version includes fully functional probe points that can be traced with bpftrace, systemtap, or other USDT-aware tools.

## Available Probe Points

### Solver Lifecycle Probes

**`pde:solver_start`** - Fired when `pde_solver_solve()` begins
- Parameters:
  - `arg0`: t_start (starting time)
  - `arg1`: t_end (ending time)
  - `arg2`: dt (time step size)
  - `arg3`: n_steps (total number of steps)

**`pde:solver_progress`** - Fired periodically (every 10% of steps)
- Parameters:
  - `arg0`: step (current step number)
  - `arg1`: n_steps (total steps)
  - `arg2`: t_current (current time value)

**`pde:solver_complete`** - Fired when solve completes successfully
- Parameters:
  - `arg0`: total_steps (steps executed)
  - `arg1`: final_time (final time reached)

### Convergence Tracking Probes

**`pde:implicit_iter`** - Fired on each implicit solver iteration
- Parameters:
  - `arg0`: step (time step number)
  - `arg1`: iter (iteration number)
  - `arg2`: error (current relative error)
  - `arg3`: tolerance (convergence threshold)

**`pde:implicit_converged`** - Fired when implicit solver converges
- Parameters:
  - `arg0`: step (time step number)
  - `arg1`: final_iter (iterations required)
  - `arg2`: final_error (final error achieved)

**`pde:implicit_failed`** - Fired when implicit solver fails to converge
- Parameters:
  - `arg0`: step (step where failure occurred)
  - `arg1`: t_current (time value)
  - `arg2`: max_iter (maximum iterations attempted)

### Error Event Probes

**`pde:spline_error`** - Fired when spline creation fails
- Parameters:
  - `arg0`: n_points (points provided)
  - `arg1`: min_required (minimum required, always 2)

## Using USDT Probes with bpftrace

### Installation

```bash
# Debian/Ubuntu
sudo apt-get install bpftrace

# RHEL/Fedora
sudo yum install bpftrace
```

### Listing Available Probes

```bash
# List all probes in the library
sudo bpftrace -l 'usdt:/path/to/libpde_solver.so:*'

# Or if running an executable
sudo bpftrace -l 'usdt:/path/to/example_heat_equation:*'
```

### Example: Monitor Solver Execution

```bash
#!/usr/bin/env bpftrace

// Monitor all solver lifecycle events
usdt:/path/to/example_heat_equation:pde:solver_start
{
    printf("Solver starting: t=[%.6f, %.6f], dt=%.6f, n_steps=%d\n",
           arg0, arg1, arg2, arg3);
}

usdt:/path/to/example_heat_equation:pde:solver_progress
{
    printf("Progress: step %d/%d (%.1f%%), t=%.6f\n",
           arg0, arg1, (arg0 * 100.0) / arg1, arg2);
}

usdt:/path/to/example_heat_equation:pde:solver_complete
{
    printf("Solver completed: %d steps, final time=%.6f\n",
           arg0, arg1);
}
```

### Example: Convergence Analysis

```bash
#!/usr/bin/env bpftrace

BEGIN {
    printf("Monitoring convergence behavior...\n");
}

// Track iterations per step
usdt:/path/to/example_heat_equation:pde:implicit_converged
{
    @iters[arg0] = arg1;  // step -> iterations
    @error[arg0] = arg2;  // step -> final error
}

// Detect convergence failures
usdt:/path/to/example_heat_equation:pde:implicit_failed
{
    printf("CONVERGENCE FAILURE at step %d, t=%.6f, max_iter=%d\n",
           arg0, arg1, arg2);
}

END {
    printf("\nConvergence Statistics:\n");
    print(@iters);
    print(@error);
}
```

### Example: Performance Profiling

```bash
#!/usr/bin/env bpftrace

BEGIN {
    printf("Profiling solver performance...\n");
}

// Measure time per step
usdt:/path/to/example_heat_equation:pde:solver_progress
{
    if (@last_step_time) {
        $dt = nsecs - @last_step_time;
        @step_times = hist($dt);
        @total_time += $dt;
        @step_count++;
    }
    @last_step_time = nsecs;
}

END {
    printf("\nPerformance Report:\n");
    printf("Total steps: %d\n", @step_count);
    printf("Average time per step: %d ns\n", @total_time / @step_count);
    printf("\nTime distribution (nanoseconds):\n");
    print(@step_times);
}
```

### Example: Real-time Convergence Monitoring

```bash
#!/usr/bin/env bpftrace

// Watch convergence in real-time
usdt:/path/to/example_heat_equation:pde:implicit_iter
{
    if (arg2 < arg3 * 2.0) {  // If error < 2*tolerance
        printf("Step %d, iter %d: error=%.2e (converging)\n",
               arg0, arg1, arg2);
    }
}
```

## Using USDT Probes with SystemTap

```stap
probe process("/path/to/libpde_solver.so").mark("solver_start") {
    printf("Solver starting: t=[%f, %f], dt=%f, steps=%d\n",
           $arg1, $arg2, $arg3, $arg4)
}

probe process("/path/to/libpde_solver.so").mark("implicit_failed") {
    printf("CONVERGENCE FAILURE at step %d, t=%f\n", $arg1, $arg2)
}
```

## Best Practices

### Development and Debugging

1. **Use standard build during development** - The no-op probes have zero overhead
2. **Enable USDT build for debugging** - Switch to `pde_solver_usdt` when you need tracing
3. **Monitor convergence issues** - Use `implicit_failed` probe to catch numerical problems
4. **Profile performance** - Use progress probes to identify slow operations

### Production Monitoring

1. **Deploy with USDT enabled** - The overhead is negligible (<1%)
2. **Selective tracing** - Only enable probes when needed
3. **Automated alerts** - Monitor `implicit_failed` for convergence issues
4. **Performance baselines** - Track average iterations and step times

### Troubleshooting

**Problem**: `bpftrace -l` shows no probes

**Solution**:
- Ensure you built with `//src:pde_solver_usdt` target
- Check that systemtap-sdt-dev was installed during build
- Verify binary has USDT notes: `readelf -n <binary> | grep NT_STAPSDT`

**Problem**: Permission denied when running bpftrace

**Solution**: USDT tracing requires elevated privileges:
```bash
sudo bpftrace <script>
```

**Problem**: Probes fire but show garbage data

**Solution**: Ensure argument types match probe definitions in `pde_trace.h`. Check that you're using the correct `argN` for each parameter.

## Integration with Existing Tools

### perf

```bash
# Record USDT events
sudo perf record -e sdt_pde:* ./example_heat_equation

# Analyze recording
sudo perf script
```

### LTTng (via SDT)

```bash
# List USDT probes
lttng list --userspace

# Enable tracing
lttng create pde-trace
lttng enable-event --userspace sdt_pde:*
lttng start
```

## Further Reading

- [bpftrace documentation](https://github.com/iovisor/bpftrace)
- [SystemTap USDT guide](https://sourceware.org/systemtap/wiki/UserSpaceProbeImplementation)
- [Linux USDT probes](https://lwn.net/Articles/753601/)
- [sys/sdt.h documentation](https://sourceware.org/systemtap/wiki/AddingUserSpaceProbingToApps)

## Support

For issues or questions about USDT tracing in iv_calc:
1. Check that probes are present: `readelf -n <binary> | grep -A4 NT_STAPSDT`
2. Verify bpftrace/systemtap installation
3. Review probe definitions in `src/pde_trace.h` and `src/pde_trace.d`
