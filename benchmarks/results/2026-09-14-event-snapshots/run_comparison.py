#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run prebuilt baseline/candidate binaries serially in alternating order."""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--before", type=Path, required=True)
parser.add_argument("--after", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--rounds", type=int, default=6)
parser.add_argument("--modes", nargs="+", choices=["single", "parallel"], default=["single", "parallel"])
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

for mode, cpus, threads in [("single", "8", "1"), ("parallel", "8-15", "8")]:
    if mode not in args.modes:
        continue
    for repeat in range(args.rounds):
        order = ["before", "after"] if repeat % 2 == 0 else ["after", "before"]
        for variant in order:
            directory = getattr(args, variant)
            cases = [("event_snapshot_impact", "^BM_Event")]
            if mode == "single":
                cases.append(("greek_latency", "^BM_Segmented_(Price|Vega|All)$"))
            else:
                cases = [("event_snapshot_impact", "^BM_Event(Manual|Adaptive)Build")]
            for executable, pattern in cases:
                # The baseline cannot enable an API which it does not have.
                if variant == "before" and mode == "single":
                    if executable == "event_snapshot_impact":
                        pattern = "^BM_Event(DirectSolve|ManualBuild|AdaptiveBuild|FixedGridSolve/0)"
                stem = f"{mode}-{repeat:02d}-{variant}-{executable}"
                result_file = (args.out / f"{stem}.json").resolve()
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS=threads, OMP_DYNAMIC="FALSE",
                           OMP_PROC_BIND="close", OMP_PLACES="cores",
                           OMP_WAIT_POLICY="PASSIVE")
                if mode == "parallel":
                    # In this environment OpenMP's automatic places collapse
                    # to the master's core during runtime initialization.
                    # The taskset mask keeps all workers on eight physical
                    # cores; leave their scheduling inside that mask to Linux.
                    env["OMP_PROC_BIND"] = "false"
                    env.pop("OMP_PLACES", None)
                command = ["taskset", "-c", cpus, str(directory / executable),
                           f"--benchmark_filter={pattern}",
                           "--benchmark_min_time=0.25s",
                           "--benchmark_min_warmup_time=0.05",
                           "--benchmark_repetitions=1",
                           f"--benchmark_out={result_file}",
                           "--benchmark_out_format=json"]
                print(f"{mode} round {repeat + 1}/{args.rounds}: {variant} {executable}", flush=True)
                with (args.out / f"{stem}.log").open("w") as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                data = json.loads(result_file.read_text())
                if not data.get("benchmarks") or any(
                    b.get("error_occurred") for b in data["benchmarks"]
                ):
                    raise RuntimeError(f"Failed or empty benchmark: {result_file}")
