# mango Tracing Scripts

This directory contains ready-to-use bpftrace scripts for monitoring and debugging the mango library.

## Quick Start

```bash
# Make scripts executable
chmod +x *.bt

# Monitor all library activity
sudo bpftrace monitor_all.bt -c './bazel-bin/examples/example_heat_equation'

# Watch convergence in real-time
sudo bpftrace convergence_watch.bt -c './bazel-bin/examples/example_heat_equation'

# Alert on failures
sudo bpftrace debug_failures.bt -c './bazel-bin/examples/example_heat_equation'

# Profile performance
sudo bpftrace performance_profile.bt -c './bazel-bin/examples/example_heat_equation'
```

## Available Scripts

### `monitor_all.bt`
**Purpose:** High-level overview of all library activity
**Use when:** You want a dashboard view of everything happening

Shows:
- Algorithm starts and completions
- Convergence failures
- Validation errors
- Execution time statistics

**Example:**
```bash
sudo bpftrace monitor_all.bt -c './bazel-bin/examples/example_american_option'
```

### `convergence_watch.bt`
**Purpose:** Real-time convergence monitoring
**Use when:** Debugging convergence issues or tuning solver parameters

Shows:
- Iteration-by-iteration progress
- Convergence rates
- Iterations required per step
- Success/failure patterns

**Example:**
```bash
sudo bpftrace convergence_watch.bt -c './bazel-bin/examples/example_heat_equation'
```

### `debug_failures.bt`
**Purpose:** Alert on all errors and failures
**Use when:** Something is going wrong and you need diagnostics

Shows:
- Convergence failures with full context
- Validation errors with decoded messages
- Runtime errors
- State before failure

**Example:**
```bash
sudo bpftrace debug_failures.bt -c './bazel-bin/examples/example_heat_equation'
```

### `performance_profile.bt`
**Purpose:** Performance analysis and timing
**Use when:** Optimizing performance or identifying bottlenecks

Shows:
- Execution time per module
- Per-step timing
- Iteration counts
- Throughput metrics

**Example:**
```bash
sudo bpftrace performance_profile.bt -c './bazel-bin/examples/example_american_option'
```

### `pde_detailed.bt`
**Purpose:** Deep dive into PDE solver behavior
**Use when:** Debugging PDE-specific issues

Shows:
- Time stepping progress
- Convergence per time step
- Slow convergence warnings
- Step-by-step statistics

**Example:**
```bash
sudo bpftrace pde_detailed.bt -c './bazel-bin/examples/example_heat_equation'
```

### `iv_detailed.bt`
**Purpose:** Deep dive into implied volatility calculations
**Use when:** Debugging IV calculation issues

Shows:
- Input parameters
- Validation checks
- Brent's method iterations
- Convergence analysis

**Example:**
```bash
sudo bpftrace iv_detailed.bt -c './bazel-bin/examples/example_implied_volatility'
```

## Attaching to Running Processes

If your program is already running, attach by PID:

```bash
# Find the PID
ps aux | grep example

# Attach to it
sudo bpftrace monitor_all.bt -p <PID>
```

## Combining Scripts

You can run multiple scripts simultaneously in different terminals:

```bash
# Terminal 1: High-level monitoring
sudo bpftrace monitor_all.bt -c './example'

# Terminal 2: Detailed convergence
sudo bpftrace convergence_watch.bt -p <PID>
```

## Output to File

Save trace output for later analysis:

```bash
sudo bpftrace performance_profile.bt -c './example' > trace_output.txt
```

## Troubleshooting

**No probes found:**
- Ensure binary was built with USDT support (should be default)
- Verify probes exist: `sudo bpftrace -l 'usdt:./your_binary:mango:*'`

**Permission denied:**
- bpftrace requires root privileges: use `sudo`

**Script syntax error:**
- Ensure bpftrace is up to date: `bpftrace --version` (need v0.12+)

## Requirements

- bpftrace v0.12 or later
- Linux kernel 4.9+ with eBPF support
- Root privileges (sudo)
- Binary built with USDT support (default)

## See Also

- [TRACING_QUICKSTART.md](../../docs/TRACING_QUICKSTART.md) - 5-minute getting started guide
- [TRACING.md](../../TRACING.md) - Comprehensive tracing documentation
- [bpftrace documentation](https://github.com/iovisor/bpftrace)
