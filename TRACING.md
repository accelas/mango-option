# USDT Tracing Guide for mango

This document provides comprehensive documentation for using USDT (User Statically-Defined Tracing) probes in the mango library.

> **Quick Start:** New to tracing? Start with [TRACING_QUICKSTART.md](TRACING_QUICKSTART.md) for a 5-minute introduction.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Common Use Cases](#common-use-cases)
- [Available Probes](#available-probes)
- [Using bpftrace](#using-bpftrace)
- [Helper Tool Reference](#helper-tool-reference)
- [Script Reference](#script-reference)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

## Overview

The mango library uses USDT probe points for zero-overhead runtime tracing. This enables:

- **Zero overhead when disabled**: Probes compile to single NOP instructions
- **Dynamic enablement**: Enable/disable tracing at runtime without recompilation
- **Production-safe**: Can be used in production environments (< 1% overhead when active)
- **Rich context**: Captures solver state, convergence metrics, and performance data
- **Standard tooling**: Works with bpftrace and other eBPF-based tools

### Benefits

| Use Case | Benefit |
|----------|---------|
| **Development** | Debug convergence issues, understand algorithm behavior |
| **Testing** | Validate solver correctness, verify convergence patterns |
| **Performance** | Profile execution time, identify bottlenecks |
| **Production** | Monitor solver health, detect anomalies |

## Getting Started

### Prerequisites

```bash
# Install bpftrace (Ubuntu/Debian)
sudo apt-get install bpftrace

# Install bpftrace (RHEL/Fedora)
sudo yum install bpftrace

# Optional: Install systemtap-sdt-dev for compile-time validation
sudo apt-get install systemtap-sdt-dev
```

**Minimum versions:**
- bpftrace: v0.12+
- Linux kernel: 4.9+ with eBPF support

### Build with USDT

**USDT is enabled by default** - just build normally:

```bash
# Build examples
bazel build //examples:example_heat_equation
bazel build //examples:example_american_option
bazel build //examples:example_implied_volatility

# Build your own program
bazel build //your:target
```

The library gracefully falls back to no-op probes if `sys/sdt.h` is not available.

### Verify USDT Support

```bash
# Using helper tool
sudo ./scripts/mango-trace check ./bazel-bin/examples/example_heat_equation

# Or manually
readelf -n ./bazel-bin/examples/example_heat_equation | grep NT_STAPSDT
sudo bpftrace -l 'usdt:./bazel-bin/examples/example_heat_equation:mango:*'
```

## Common Use Cases

### 1. Debug Convergence Failures

```bash
# Alert on all failures with diagnostics
sudo ./scripts/mango-trace monitor ./my_program --preset=debug

# Or directly with bpftrace
sudo bpftrace scripts/tracing/debug_failures.bt -c './my_program'
```

**What you get:**
- Convergence failure alerts with error values
- Validation error messages
- Last known state before failure
- Suggested fixes based on error codes

### 2. Monitor Convergence Behavior

```bash
# Watch convergence in real-time
sudo bpftrace scripts/tracing/convergence_watch.bt -c './my_program'
```

**What you get:**
- Iteration-by-iteration progress
- Convergence rates and patterns
- Histogram of iterations required
- Success/failure statistics

### 3. Profile Performance

```bash
# Comprehensive performance analysis
sudo bpftrace scripts/tracing/performance_profile.bt -c './my_program'
```

**What you get:**
- Execution time per module
- Per-step timing (PDE solver)
- Iteration counts
- Timing histograms
- Slowest operations identified

### 4. Deep Dive into PDE Solver

```bash
# Detailed PDE solver tracing
sudo bpftrace scripts/tracing/pde_detailed.bt -c './example_heat_equation'
```

**What you get:**
- Time stepping progress (every 10%)
- Convergence per time step
- Slow convergence warnings
- Complete statistics summary

### 5. Analyze Implied Volatility

```bash
# Detailed IV calculation tracing
sudo bpftrace scripts/tracing/iv_detailed.bt -c './example_implied_volatility'
```

**What you get:**
- Input parameters and validation
- Brent's method iteration details
- Convergence analysis
- IV value distribution

## Available Probes

### General-Purpose Probes

These work across all modules (PDE solver, American options, IV, etc.):

#### Algorithm Lifecycle

**`mango:algo_start`**
```c
probe algo_start(int module_id, double param1, double param2, double param3)
```
- Fired when any algorithm begins execution
- `module_id`: 1=PDE, 2=AmOption, 3=IV, 4=Brent, 5=Spline
- `param1-3`: Module-specific parameters

**`mango:algo_progress`**
```c
probe algo_progress(int module_id, size_t current, size_t total, double metric)
```
- Fired periodically during execution (e.g., every 10% for PDE solver)
- `current/total`: Progress counter
- `metric`: Current value (e.g., current time, current iteration)

**`mango:algo_complete`**
```c
probe algo_complete(int module_id, size_t iterations, double final_metric)
```
- Fired when algorithm completes successfully

#### Convergence Tracking

**`mango:convergence_iter`**
```c
probe convergence_iter(int module_id, size_t step, size_t iter, double error, double tolerance)
```
- Fired on each iteration of convergence loop
- `step`: Outer step (time step for PDE, 0 for others)
- `iter`: Current iteration number
- `error`: Current error metric
- `tolerance`: Convergence threshold

**`mango:convergence_success`**
```c
probe convergence_success(int module_id, size_t step, size_t final_iter, double final_error)
```
- Fired when convergence is achieved

**`mango:convergence_failed`**
```c
probe convergence_failed(int module_id, size_t step, size_t max_iter, double final_error)
```
- Fired when convergence fails

#### Validation and Errors

**`mango:validation_error`**
```c
probe validation_error(int module_id, int error_code, double param1, double param2)
```
- Fired when input validation fails
- Error codes vary by module (see module-specific sections)

**`mango:runtime_error`**
```c
probe runtime_error(int module_id, int error_code, double context)
```
- Fired on runtime errors

### Module-Specific Probes

#### Implied Volatility

**`mango:iv_start`**
```c
probe iv_start(double spot, double strike, double time_to_maturity, double market_price)
```

**`mango:iv_complete`**
```c
probe iv_complete(double implied_vol, int iterations, int converged)
```
- `converged`: 1 if successful, 0 if failed

**IV Validation Error Codes:**
- 1: Spot price must be positive
- 2: Strike price must be positive
- 3: Time to maturity must be positive
- 4: Market price must be positive
- 5: Arbitrage bounds violated

#### Brent's Method

**`mango:brent_iter`**
```c
probe brent_iter(int iter, double x, double fx, double interval_width)
```

#### American Options

**`mango:option_start`**
```c
probe option_start(int option_type, double strike, double volatility, double time_to_maturity)
```
- `option_type`: 0=call, 1=put

**`mango:option_complete`**
```c
probe option_complete(int status, int iterations)
```
- `status`: 0=success, -1=failure

### Module IDs

```c
#define MODULE_PDE_SOLVER       1
#define MODULE_AMERICAN_OPTION  2
#define MODULE_IMPLIED_VOL      3
#define MODULE_BRENT_ROOT       4
#define MODULE_CUBIC_SPLINE     5
```

## Using bpftrace

### Basic Syntax

```bash
# Run with program
sudo bpftrace script.bt -c './program'

# Attach to running process
sudo bpftrace script.bt -p <PID>

# One-liner
sudo bpftrace -e 'usdt:./program:mango:convergence_failed { printf("FAIL\n"); }'
```

### Example Scripts

**Monitor all convergence failures:**

```bash
sudo bpftrace -e '
usdt::mango:convergence_failed {
    printf("Module %d failed at step %d after %d iterations (error=%.2e)\n",
           arg0, arg1, arg2, arg3);
}'
-c './my_program'
```

**Count iterations per module:**

```bash
sudo bpftrace -e '
usdt::mango:convergence_success { @iters[arg0] = hist(arg2); }
END { print(@iters); }
'
-c './my_program'
```

**Measure algorithm duration:**

```bash
sudo bpftrace -e '
usdt::mango:algo_start { @start[arg0] = nsecs; }
usdt::mango:algo_complete /@start[arg0]/ {
    $duration_ms = (nsecs - @start[arg0]) / 1000000;
    printf("Module %d: %u ms\n", arg0, $duration_ms);
    delete(@start[arg0]);
}
'
-c './my_program'
```

## Helper Tool Reference

The `mango-trace` helper tool simplifies common tracing tasks.

### Commands

```bash
# List all USDT probes
sudo ./scripts/mango-trace list <binary>

# Validate USDT support
sudo ./scripts/mango-trace check <binary>

# Monitor with preset
sudo ./scripts/mango-trace monitor <binary> --preset=<name>

# Run specific script
sudo ./scripts/mango-trace run <script.bt> <binary>
```

### Monitor Presets

| Preset | Script | Purpose |
|--------|--------|---------|
| `all` | `monitor_all.bt` | High-level overview (default) |
| `convergence` | `convergence_watch.bt` | Convergence tracking |
| `debug` | `debug_failures.bt` | Error diagnostics |
| `performance` | `performance_profile.bt` | Performance analysis |
| `pde` | `pde_detailed.bt` | PDE solver deep dive |
| `iv` | `iv_detailed.bt` | IV calculation deep dive |

### Examples

```bash
# Quick monitoring
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_heat_equation

# Debug convergence
sudo ./scripts/mango-trace monitor ./my_program --preset=debug

# Performance profiling
sudo ./scripts/mango-trace monitor ./my_program --preset=performance

# Run custom script
sudo ./scripts/mango-trace run my_custom.bt ./my_program

# Pass arguments to binary
sudo ./scripts/mango-trace monitor ./my_program -- --my-arg value
```

## Script Reference

All scripts are in `scripts/tracing/`. See [scripts/tracing/README.md](scripts/tracing/README.md) for details.

### Script Summary

| Script | Description | Best For |
|--------|-------------|----------|
| `monitor_all.bt` | Dashboard of all activity | General monitoring |
| `convergence_watch.bt` | Real-time convergence | Tuning parameters |
| `debug_failures.bt` | Alert on errors | Debugging issues |
| `performance_profile.bt` | Timing and statistics | Performance optimization |
| `pde_detailed.bt` | PDE solver details | PDE-specific work |
| `iv_detailed.bt` | IV calculation details | IV-specific work |

## Advanced Topics

### Filtering by Module

```bash
# Only trace PDE solver (module_id == 1)
sudo bpftrace -e '
usdt::mango:convergence_iter /arg0 == 1/ {
    printf("PDE iter %d: error=%.2e\n", arg2, arg3);
}
'
-c './program'
```

### Combining Multiple Probes

```bash
sudo bpftrace -e '
BEGIN { printf("Tracking solver lifecycle...\n"); }

usdt::mango:algo_start {
    @start = nsecs;
    printf("Started\n");
}

usdt::mango:convergence_failed {
    printf("Convergence failed!\n");
}

usdt::mango:algo_complete {
    printf("Completed in %u ms\n", (nsecs - @start) / 1000000);
}
'
-c './program'
```

### Output to JSON

```bash
# Save structured output
sudo bpftrace -e '
usdt::mango:convergence_success {
    printf("{\"module\":%d,\"step\":%d,\"iters\":%d,\"error\":%.2e}\n",
           arg0, arg1, arg2, arg3);
}
'
-c './program' > output.jsonl
```

### Attach to Running Process

```bash
# Find PID
ps aux | grep my_program

# Attach (non-invasive, no restart needed)
sudo bpftrace scripts/tracing/monitor_all.bt -p 12345
```

### Performance Impact

When tracing is **disabled** (no bpftrace attached):
- Overhead: < 0.01% (single NOP instruction per probe)
- Binary size: +few KB for USDT notes
- Runtime: No measurable impact

When tracing is **enabled** (bpftrace attached):
- Overhead: 0.1% - 1% depending on probe frequency
- Memory: ~10MB for bpftrace process
- Safe for production use

## Troubleshooting

### No Probes Found

**Symptom:** `bpftrace -l` shows no probes

**Solutions:**

1. Check if binary has USDT notes:
   ```bash
   readelf -n ./binary | grep NT_STAPSDT
   ```

2. If missing, ensure systemtap-sdt-dev is installed:
   ```bash
   sudo apt-get install systemtap-sdt-dev
   bazel clean
   bazel build //your:target
   ```

3. Verify build includes `-DHAVE_SYSTEMTAP_SDT` (should be automatic)

### Permission Denied

**Symptom:** `Error: Permission denied`

**Solution:** bpftrace requires root:
```bash
sudo bpftrace script.bt -c './program'
```

### Probes Don't Fire

**Symptom:** Script runs but shows no output

**Possible causes:**

1. Program exits too quickly - add delays or increase work
2. Wrong probe names - verify with `bpftrace -l`
3. Filtering too aggressive - remove predicates
4. Program doesn't reach traced code paths

**Debug:**
```bash
# List probes in binary
sudo bpftrace -l 'usdt:./binary:mango:*'

# Add verbose output
sudo bpftrace -v script.bt -c './program'
```

### bpftrace Version Too Old

**Symptom:** Syntax errors or missing features

**Solution:** Upgrade bpftrace:
```bash
# Check version
bpftrace --version

# Need v0.12+
sudo apt-get update
sudo apt-get install bpftrace
```

### Script Shows Garbage Data

**Symptom:** Nonsensical values in output

**Causes:**
- Incorrect `argN` indexing (args are 0-indexed)
- Type mismatch between probe definition and script
- Binary/script version mismatch

**Solution:** Verify probe signatures in `src/ivcalc_trace.h`

## Best Practices

### Development

1. **Start simple**: Use `monitor_all.bt` first
2. **Add detail**: Move to specific scripts as needed
3. **Custom scripts**: Copy and modify existing scripts
4. **Version control**: Save useful custom scripts

### Production

1. **Enable USDT**: Deploy with USDT-enabled binaries (negligible overhead)
2. **Selective tracing**: Only trace when investigating issues
3. **Automated alerts**: Monitor convergence failures
4. **Performance baselines**: Track iteration counts over time

### Performance

1. **Limit output**: Use predicates to filter probes
2. **Sample**: Trace every Nth iteration for high-frequency probes
3. **Short runs**: Trace for limited duration
4. **Export data**: Save to file for offline analysis

## Further Reading

- [TRACING_QUICKSTART.md](TRACING_QUICKSTART.md) - 5-minute getting started
- [scripts/tracing/README.md](scripts/tracing/README.md) - Script reference
- [src/ivcalc_trace.h](src/ivcalc_trace.h) - Probe definitions
- [bpftrace documentation](https://github.com/iovisor/bpftrace)
- [USDT probes](https://lwn.net/Articles/753601/)

## Examples Gallery

Complete working examples for common scenarios.

### Example 1: Find Slowest Time Steps

```bash
sudo bpftrace -e '
usdt::mango:convergence_success /arg0 == 1/ {
    @iters_per_step[arg1] = arg2;
}

END {
    print(@iters_per_step);
}
' -c './example_heat_equation'
```

### Example 2: Alert on Slow Convergence

```bash
sudo bpftrace -e '
usdt::mango:convergence_iter {
    if (arg2 > 50) {  # More than 50 iterations
        printf("SLOW: module=%d, step=%d, iter=%d, error=%.2e\n",
               arg0, arg1, arg2, arg3);
    }
}
' -c './my_program'
```

### Example 3: Measure Time Per Step

```bash
sudo bpftrace -e '
usdt::mango:algo_progress /arg0 == 1/ {
    if (@last_time > 0) {
        $dt = (nsecs - @last_time) / 1000;
        @step_times = hist($dt);
    }
    @last_time = nsecs;
}

END { print(@step_times); }
' -c './example_heat_equation'
```

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Verify probe definitions in `src/ivcalc_trace.h`
3. Review example scripts in `scripts/tracing/`
4. Consult bpftrace documentation

---

**Happy Tracing!** 🔍
