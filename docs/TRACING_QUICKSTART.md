# USDT Tracing Quick Start (5 Minutes)

Get started with zero-overhead tracing of the mango library in under 5 minutes.

## What is USDT Tracing?

USDT (User Statically-Defined Tracing) provides production-safe, zero-overhead instrumentation. When not actively traced, probes are single NOP instructions. When enabled, they capture detailed runtime data without recompiling.

**Benefits:**
- 🚀 Zero overhead when disabled (< 1% when enabled)
- 🔍 Deep visibility into solver behavior
- 🐛 Debug convergence and performance issues
- 📊 Production-safe monitoring

## Prerequisites

```bash
# Install bpftrace (Ubuntu/Debian)
sudo apt-get install bpftrace

# Install bpftrace (RHEL/Fedora)
sudo yum install bpftrace

# Optional: Install systemtap-sdt-dev for compile-time validation
sudo apt-get install systemtap-sdt-dev
```

## Step 1: Build Your Program

**Good news:** USDT is enabled by default! Just build normally:

```bash
# Build an example
bazel build //examples:example_heat_equation

# Or build your own program
bazel build //your:target
```

The binaries automatically include USDT probes (gracefully falls back if systemtap-sdt-dev not installed).

## Step 2: Verify USDT Support (Optional)

```bash
# Check if probes are present
sudo ./scripts/mango-trace check ./bazel-bin/examples/example_heat_equation
```

Expected output:
```
✓ USDT notes found in binary
✓ Found 15 mango probes
✓ bpftrace can attach to probes

USDT support: OK
```

## Step 3: Run Your First Trace

**Option A: Use the helper tool (easiest)**

```bash
# Monitor all library activity
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_heat_equation

# Watch convergence behavior
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_heat_equation --preset=convergence

# Debug failures
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_heat_equation --preset=debug

# Profile performance
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_heat_equation --preset=performance
```

**Option B: Use bpftrace directly**

```bash
# Monitor all activity
sudo bpftrace scripts/tracing/monitor_all.bt -c './bazel-bin/examples/example_heat_equation'

# Watch convergence
sudo bpftrace scripts/tracing/convergence_watch.bt -c './bazel-bin/examples/example_heat_equation'
```

## Common Use Cases

### Debug Convergence Issues

```bash
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_heat_equation --preset=debug
```

Shows:
- Convergence failures with diagnostics
- Iteration counts and errors
- Validation failures

### Monitor PDE Solver Progress

```bash
sudo bpftrace scripts/tracing/pde_detailed.bt -c './bazel-bin/examples/example_heat_equation'
```

Shows:
- Time stepping progress
- Iterations per step
- Convergence statistics

### Profile Performance

```bash
sudo bpftrace scripts/tracing/performance_profile.bt -c './bazel-bin/examples/example_heat_equation'
```

Shows:
- Execution time per module
- Iteration counts
- Timing histograms

### Monitor Implied Volatility Calculations

```bash
sudo bpftrace scripts/tracing/iv_detailed.bt -c './bazel-bin/examples/example_implied_volatility'
```

Shows:
- Input parameters
- Brent's method iterations
- Convergence behavior

## Available Scripts

All scripts are in `scripts/tracing/`:

| Script | Purpose | Use When |
|--------|---------|----------|
| `monitor_all.bt` | High-level overview | General monitoring |
| `convergence_watch.bt` | Convergence tracking | Tuning solver parameters |
| `debug_failures.bt` | Error diagnostics | Something is failing |
| `performance_profile.bt` | Timing analysis | Optimizing performance |
| `pde_detailed.bt` | PDE solver deep dive | PDE-specific debugging |
| `iv_detailed.bt` | IV calculation deep dive | IV-specific debugging |

## Helper Tool Commands

```bash
# List all available probes
sudo ./scripts/mango-trace list ./bazel-bin/examples/example_heat_equation

# Validate USDT support
sudo ./scripts/mango-trace check ./bazel-bin/examples/example_heat_equation

# Monitor with preset
sudo ./scripts/mango-trace monitor <binary> --preset=<name>

# Run specific script
sudo ./scripts/mango-trace run convergence_watch.bt <binary>
```

## Attach to Running Process

If your program is already running:

```bash
# Find the process ID
ps aux | grep my_program

# Attach bpftrace to it
sudo bpftrace scripts/tracing/monitor_all.bt -p <PID>
```

## Save Output

```bash
# Save to file for later analysis
sudo bpftrace scripts/tracing/performance_profile.bt -c './my_program' > trace_output.txt
```

## Troubleshooting

**Problem: "No USDT probes found"**

Solution:
```bash
# Check if probes exist
readelf -n ./bazel-bin/examples/example_heat_equation | grep NT_STAPSDT

# If missing, install systemtap-sdt-dev and rebuild
sudo apt-get install systemtap-sdt-dev
bazel clean
bazel build //examples:example_heat_equation
```

**Problem: "Permission denied"**

Solution: bpftrace requires root privileges
```bash
sudo ./scripts/mango-trace monitor <binary>
```

**Problem: "bpftrace: command not found"**

Solution: Install bpftrace
```bash
# Ubuntu/Debian
sudo apt-get install bpftrace

# RHEL/Fedora
sudo yum install bpftrace
```

## Next Steps

- 📖 Read [TRACING.md](TRACING.md) for comprehensive documentation
- 🔧 Explore scripts in `scripts/tracing/` directory
- 📝 See [scripts/tracing/README.md](scripts/tracing/README.md) for script details
- 🎯 Customize scripts for your specific needs

## Example Session

```bash
# 1. Build
bazel build //examples:example_heat_equation

# 2. Check USDT
sudo ./scripts/mango-trace check ./bazel-bin/examples/example_heat_equation
# ✓ USDT support: OK

# 3. Monitor
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_heat_equation --preset=convergence

# Output shows:
# - Convergence iterations per step
# - Success/failure status
# - Timing information
# - Statistics summary
```

## That's It!

You're now tracing the mango library. Use the scripts to:
- ✅ Debug convergence issues
- ✅ Profile performance
- ✅ Monitor production systems
- ✅ Understand solver behavior

For more advanced usage, see [TRACING.md](TRACING.md).
