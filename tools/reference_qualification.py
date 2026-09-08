#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reference-only qualification. Never constructs or tunes a price table."""
from __future__ import annotations

import math

PRICE_BUDGET = 0.001
IV_BUDGET = 2e-6
VEGA_FLOOR = 1e-4
TV_RATIO_FLOOR = 1e-4


def convergence(values, roundoff):
    """Conservative empirical Richardson evidence, not an error proof."""
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        return {"stable": False, "uncertainty": None, "reason": "nonfinite-or-incomplete"}
    d0, d1 = values[1] - values[0], values[2] - values[1]
    if max(abs(d0), abs(d1)) <= roundoff:
        return {"stable": True, "uncertainty": roundoff, "reason": "roundoff-limited"}
    if d0 * d1 <= 0 or abs(d1) >= 0.75 * abs(d0):
        return {"stable": False, "uncertainty": None, "reason": "noncontracting-or-oscillating"}
    ratio = abs(d1 / d0)
    error = max(roundoff, 2 * abs(d1) * ratio / (1 - ratio))
    return {"stable": True, "uncertainty": error, "observed_ratio": ratio,
            "reason": "three-level-contracting"}


def price_qualified(price, intrinsic, uncertainty):
    return (price is not None and uncertainty is not None
            and math.isfinite(price) and math.isfinite(uncertainty)
            and 0 <= uncertainty <= PRICE_BUDGET and price + uncertainty >= intrinsic)


def iv_eligibility(price, intrinsic, strike, price_error, vega, vega_error,
                   *, price_budget=PRICE_BUDGET, iv_budget=IV_BUDGET,
                   vega_floor=VEGA_FLOOR):
    """Classify only reference evidence; filtered rows have no IV error value."""
    result = {"status": "oracle-unresolved", "oracle_iv_resolution": None}
    if (price_error is None or not math.isfinite(price_error)
            or price_error > price_budget or not math.isfinite(price)):
        return dict(result, reason="price-budget")
    if vega_error is None or not all(math.isfinite(x) for x in (vega, vega_error)):
        return dict(result, reason="vega-unresolved")
    tv_lo, tv_hi = price - intrinsic - price_error, price - intrinsic + price_error
    cutoff = strike * TV_RATIO_FLOOR
    if tv_hi < 0:
        return dict(result, reason="intrinsic-bound")
    if tv_lo < cutoff <= tv_hi:
        return dict(result, reason="time-value-floor-straddle")
    v_lo, v_hi = vega - vega_error, vega + vega_error
    abs_lo = 0.0 if v_lo <= 0 <= v_hi else min(abs(v_lo), abs(v_hi))
    abs_hi = max(abs(v_lo), abs(v_hi))
    if abs_lo < vega_floor <= abs_hi:
        return dict(result, reason="vega-floor-straddle")
    tv_filtered, v_filtered = tv_hi < cutoff, abs_hi < vega_floor
    if tv_filtered or v_filtered:
        label = "both-filtered" if tv_filtered and v_filtered else (
            "tv-filtered" if tv_filtered else "vega-filtered")
        return dict(result, status=label, reason="reference-floor")
    if v_lo <= 0:
        return dict(result, reason="negative-or-unresolved-vega")
    resolution = price_error / v_lo
    if resolution > iv_budget:
        return dict(result, reason="vega-scaled-price-budget")
    return dict(result, status="reference-measurable", reason="qualified",
                oracle_iv_resolution=resolution)

# Protocol/ledger plumbing is intentionally separate from numerical assessment.
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_manifests(paths):
    rows, manifests, seen = [], [], set()
    for path in map(Path, paths):
        checksum_text = (path / "SHA256SUMS").read_text()
        for line in checksum_text.splitlines():
            expected, name = line.split(maxsplit=1)
            if digest((path / name.strip()).read_bytes()) != expected:
                raise ValueError(f"frozen manifest digest mismatch: {path / name.strip()}")
        metadata = json.loads((path / "metadata.json").read_text())
        builds = {b["id"]: b for b in json.loads((path / "build-requests.json").read_text())}
        for ledger in ("required-pricing", "boundary-admission"):
            selected = [json.loads(line) for line in (path / f"{ledger}.jsonl").read_text().splitlines()]
            if len(selected) != metadata["counts"][ledger]:
                raise ValueError(f"count mismatch: {path}/{ledger}")
            for row in selected:
                if row["id"] in seen:
                    raise ValueError(f"duplicate physical ID: {row['id']}")
                seen.add(row["id"])
                row = dict(row, _ledger=ledger, _build=builds[row["build_id"]],
                           _manifest=metadata["version"])
                audit_schedule(row)
                rows.append(row)
        manifests.append({"path": str(path.resolve()), "metadata": metadata,
                          "checksum_file_sha256": digest(checksum_text.encode())})
    return sorted(rows, key=lambda r: r["id"]), manifests


def audit_schedule(row):
    """Verify physical identity while preserving the frozen binary64 offsets."""
    b = row["_build"]
    elapsed = b["anchor_maturity"] - row["maturity"]
    exact_events = {e["event_index"] for e in row.get("events", []) if e["offset_in_tau"] == 0}
    expected = []
    for i, d in enumerate(b["anchored_dividends"]):
        offset = d["calendar_time"] - elapsed
        if i not in exact_events and 0 < offset < row["maturity"]:
            expected.append((offset, d["amount"]))
    actual = row["rolled_dividends"]
    if len(expected) != len(actual):
        raise ValueError(f"rolled schedule count mismatch: {row['id']}")
    for (offset, amount), d in zip(expected, actual):
        if not 0 < d["calendar_time"] < row["maturity"] or d["amount"] != amount:
            raise ValueError(f"invalid rolled payment: {row['id']}")
        tolerance = 16 * math.ulp(max(1.0, abs(b["anchor_maturity"])))
        if abs(offset - d["calendar_time"]) > tolerance:
            raise ValueError(f"rolled schedule time mismatch: {row['id']}")


def analytic_eligible(row):
    return (row["option_type"] == "CALL" and row["rate"] >= 0
            and row["dividend_yield"] == 0 and not row["rolled_dividends"])


def exact_cash_event(row):
    return any(event["offset_in_tau"] == 0 for event in row.get("events", []))


def should_skip(old, args, row):
    needs_ql = (args.quantlib and not analytic_eligible(row) and day_aligned(row)
                and not row["rolled_dividends"] and not old.get("quantlib_requested"))
    needs_audit = args.audit_analytic and analytic_eligible(row) and (
        "fd_audit" not in old or (old["fd_audit"].get("status") != "qualified"
                                 and old.get("rounds_requested", 0) < args.rounds))
    if needs_ql or needs_audit:
        return False
    return old.get("status") == "qualified" or (
        old.get("status") == "oracle-unresolved" and old.get("rounds_requested", 0) >= args.rounds)


def runtime_libraries(worker):
    """The worker links system QuantLib/libm; binary identity alone is insufficient."""
    linked = subprocess.check_output(["ldd", worker], text=True)
    libraries = {}
    for line in linked.splitlines():
        fields = line.split()
        candidates = [field for field in fields if field.startswith("/")]
        for name in candidates:
            path = Path(name)
            if path.is_file():
                libraries[str(path.resolve())] = digest(path.read_bytes())
    if not libraries:
        raise RuntimeError("cannot fingerprint reference worker's runtime libraries")
    return libraries


def day_aligned(row):
    def aligned(t):
        days = t * 365
        return round(days) > 0 and abs(days - round(days)) <= 32 * sys.float_info.epsilon * max(1, abs(days))
    return aligned(row["maturity"]) and all(aligned(d["calendar_time"]) for d in row["rolled_dividends"])


def mesh_plan(row, round_index):
    """Three independent ladders; domain extension keeps interior sinh spacing.

    x = scale*sinh(u). Extending u's range from +/-2 to +/-2.5 to +/-3
    with the same du extends the grid rather than stretching its interior.
    Space halves du at fixed range; time halves dt on the fixed finest grid.
    """
    x = math.log(row["spot"] / row["strike"])
    t, s = row["maturity"], row["volatility"]
    cash = sum(d["amount"] for d in row["rolled_dividends"])
    radius = (abs(x) + 6 * s * math.sqrt(t)
              + abs(row["rate"] - row["dividend_yield"] - 0.5 * s * s) * t
              + math.log1p(cash / row["strike"]))
    n = 128 * (2 ** round_index)
    nt = 256 * (2 ** round_index)
    def grid(level, domain=2, time_level=2):
        alpha = 4.0 + domain
        width = radius * math.sinh(alpha / 2) / math.sinh(2.0)
        intervals = n * (2 ** level) * (4 + domain) // 4
        return {"kind": "S", "nx": intervals + 1, "nt": nt * (2 ** time_level),
                "radius": width, "alpha": alpha}
    return {"space": [grid(i) for i in range(3)],
            "time": [grid(2, time_level=i) for i in range(3)],
            "domain": [grid(2, domain=i) for i in range(3)]}


def request_line(row, provider, grid=None):
    grid = grid or {"kind": "H", "nx": 0, "nt": 0, "radius": 0, "alpha": 0}
    args = [provider, row["spot"], row["strike"], row["maturity"], row["volatility"],
            row["rate"], row["dividend_yield"], 0 if row["option_type"] == "CALL" else 1,
            grid["kind"], grid["nx"], grid["nt"], grid["radius"], grid["alpha"],
            len(row["rolled_dividends"])]
    for d in row["rolled_dividends"]:
        args.extend([d["calendar_time"], d["amount"]])
    return " ".join(str(a) for a in args)


class Worker:
    def __init__(self, path, cache, lock, binary_hash):
        self.path, self.cache, self.lock, self.binary_hash = path, cache, lock, binary_hash
        self.process = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1, env=dict(os.environ, OMP_NUM_THREADS="2"))
        self.sample_keys = set()

    def query(self, row, provider="fd", grid=None):
        line = request_line(row, provider, grid)
        key = digest((self.binary_hash + "\n" + line).encode())
        self.sample_keys.add(key)
        with self.lock:
            stored = self.cache.execute("SELECT response FROM samples WHERE key=?", (key,)).fetchone()
        if stored:
            return json.loads(stored[0])
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("reference worker exited before returning a complete sample")
        result = json.loads(response)
        with self.lock:
            self.cache.execute("INSERT OR REPLACE INTO samples VALUES(?,?,?)", (key, line, encoded(result)))
            self.cache.commit()  # one durable checkpoint per expensive solve
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait()


def controlled_price(worker, row, plan):
    directions, observations = {}, {}
    floor = 256 * sys.float_info.epsilon * max(1, row["spot"], row["strike"])
    for axis, grids in plan.items():
        samples = [worker.query(row, grid=g) for g in grids]
        observations[axis] = samples
        prices = [sample["price"] for sample in samples if sample.get("ok")]
        directions[axis] = convergence(prices, floor)
    stable = all(d["stable"] for d in directions.values())
    error = sum(d["uncertainty"] for d in directions.values()) if stable else None
    finest = observations["space"][-1]
    return {"price": finest.get("price"), "uncertainty": error, "stable": stable,
            "directions": directions, "observations": observations, "roundoff_floor": floor}


def vega_reference(worker, row, plan):
    hs = [row["volatility"] * 0.01 / (2 ** i) for i in range(3)]
    values, errors, evidence = [], [], []
    for h in hs:
        up = dict(row, volatility=row["volatility"] + h)
        down = dict(row, volatility=row["volatility"] - h)
        axes, axis_errors, finest_value = {}, [], None
        for axis, grids in plan.items():
            samples = [(worker.query(up, grid=g), worker.query(down, grid=g)) for g in grids]
            derivatives = [(a["price"] - b["price"]) / (2 * h)
                           for a, b in samples if a.get("ok") and b.get("ok")]
            floor = 512 * sys.float_info.epsilon * max(1, row["spot"], row["strike"]) / h
            assessed = convergence(derivatives, floor)
            axes[axis] = dict(assessed, values=derivatives)
            if assessed["stable"]:
                axis_errors.append(assessed["uncertainty"])
            if len(derivatives) == 3:
                finest_value = derivatives[-1]
        evidence.append({"h": h, "directions": axes})
        if finest_value is None or len(axis_errors) != 3:
            return {"status": "oracle-unresolved", "value": None, "uncertainty": None,
                    "evidence": evidence, "reason": "vega-mesh-convergence"}
        values.append(finest_value)
        errors.append(sum(axis_errors))
    bump = convergence(values, max(errors))
    if not bump["stable"]:
        return {"status": "oracle-unresolved", "value": values[-1], "uncertainty": None,
                "evidence": evidence, "bump": bump, "reason": "vega-bump-convergence"}
    return {"status": "qualified-numerical-sequence", "value": values[-1],
            "uncertainty": max(errors) + bump["uncertainty"],
            "evidence": evidence, "bump": bump}


def ql_crosscheck(worker, row, round_index, direct):
    if not day_aligned(row):
        return {"status": "not-applicable", "reason": "fractional-day-contract"}
    if row["rolled_dividends"]:
        return {"status": "not-qualified", "reason": "cash-model-equivalence-pending"}
    samples = []
    for level in range(3):
        n = 100 * (2 ** (round_index + level))
        samples.append(worker.query(row, "ql", {"kind": "S", "nx": n, "nt": 2 * n,
                                               "radius": 1, "alpha": 4}))
    values = [s["price"] for s in samples if s.get("ok")]
    assessment = convergence(values, 256 * sys.float_info.epsilon * max(row["spot"], row["strike"]))
    result = {"status": "oracle-unresolved", "samples": samples, "convergence": assessment}
    if assessment["stable"] and direct["uncertainty"] is not None:
        difference = abs(values[-1] - direct["price"])
        agrees = difference <= assessment["uncertainty"] + direct["uncertainty"]
        result.update(status="compatible" if agrees else "independent-disagreement", difference=difference)
    return result


def general_reference(worker, row, rounds, quantlib):
    profiles = {kind: worker.query(row, grid={"kind": kind, "nx": 0, "nt": 0, "radius": 0, "alpha": 0})
                for kind in ("H", "U")}
    history = []
    outcome = {}
    intrinsic = max((row["spot"] - row["strike"]) * (1 if row["option_type"] == "CALL" else -1), 0)
    for iteration in range(rounds):
        plan = mesh_plan(row, iteration)
        price = controlled_price(worker, row, plan)
        profile_compatible = False
        if price["stable"] and all(p.get("ok") for p in profiles.values()):
            profile_noise = 4 * abs(profiles["H"]["price"] - profiles["U"]["price"])
            profile_compatible = abs(profiles["U"]["price"] - price["price"]) <= (
                price["uncertainty"] + profile_noise + price["roundoff_floor"])
        price_ok = (price["stable"] and profile_compatible
                    and price_qualified(price["price"], intrinsic, price["uncertainty"]))
        vega = vega_reference(worker, row, plan) if price_ok else {
            "status": "oracle-unresolved", "value": None, "uncertainty": None, "reason": "price-first"}
        iv = iv_eligibility(price["price"] or 0, intrinsic, row["strike"],
                            price["uncertainty"] if price_ok else None,
                            vega["value"] if vega["value"] is not None else math.nan, vega["uncertainty"])
        ql = ql_crosscheck(worker, row, iteration, price) if quantlib and price_ok else {
            "status": "not-run", "reason": "not-requested-or-price-unresolved"}
        if ql["status"] == "independent-disagreement":
            price_ok = False
            iv = {"status": "oracle-unresolved", "reason": "independent-disagreement",
                  "oracle_iv_resolution": None}
        history.append({"round": iteration, "plan": plan, "price": price,
                        "profile_compatible": profile_compatible, "vega": vega, "quantlib": ql})
        outcome = {"status": "qualified" if price_ok and iv["status"] != "oracle-unresolved" else "oracle-unresolved",
                   "price_status": "qualified" if price_ok else "oracle-unresolved",
                   "price": price["price"], "price_uncertainty": price["uncertainty"],
                   "iv": iv, "vega": vega, "basis": "direct-fde-only", "history": history, "profiles": profiles,
                   "greeks": {g: {"status": "unqualified", "reason": "independent-stencil-and-budget-pending"}
                              for g in ("delta", "gamma", "theta", "rho")}}
        if outcome["status"] == "qualified":
            break
    if exact_cash_event(row):
        outcome["greeks"]["theta"] = {"status": "two-sided-undefined",
                                       "reason": "cash-dividend-instant"}
    return outcome


def qualify(worker, row, args):
    start = time.monotonic()
    worker.sample_keys.clear()
    if analytic_eligible(row):
        a = worker.query(row, "analytic")
        if not a.get("ok"):
            outcome = {"status": "oracle-unresolved", "price_status": "oracle-unresolved", "error": a}
        else:
            intrinsic = max(row["spot"] - row["strike"], 0)
            iv = iv_eligibility(a["price"], intrinsic, row["strike"], a["price_error"], a["vega"], a["vega_error"])
            price_ok = price_qualified(a["price"], intrinsic, a["price_error"])
            outcome = {"status": "qualified" if price_ok and iv["status"] != "oracle-unresolved" else "oracle-unresolved",
                       "price_status": "qualified" if price_ok else "oracle-unresolved", "basis": "analytic-bsm-50-100",
                       "price": a["price"], "price_uncertainty": a["price_error"], "iv": iv,
                       "vega": {"value": a["vega"], "uncertainty": a["vega_error"]},
                       "greeks": {g: {"status": "qualified-analytic", "value": a[g], "uncertainty": a[g + "_error"]}
                                  for g in ("delta", "gamma", "theta", "rho")}}
            if exact_cash_event(row):
                outcome["greeks"]["theta"].update(status="one-sided-analytic",
                    calendar_side="post", two_sided_defined=False)
            if row["rate"] == 0:
                outcome["greeks"]["rho"]["regime_side"] = "nonnegative-rate-side"
            if args.audit_analytic:
                fd = general_reference(worker, row, args.rounds, args.quantlib)
                fd["analytic_error"] = abs(fd["price"] - a["price"]) if fd.get("price") is not None else None
                outcome["fd_audit"] = fd
    else:
        outcome = general_reference(worker, row, args.rounds, args.quantlib)
    return dict(outcome, id=row["id"], row=row, sample_keys=sorted(worker.sample_keys),
                rounds_requested=args.rounds, quantlib_requested=args.quantlib,
                elapsed_seconds=time.monotonic() - start)


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as out:
        out.write(encoded(value) + "\n")
        out.flush()
        os.fsync(out.fileno())
    temp.replace(path)


def summarize(rows, directory, fingerprint):
    counts, families = {}, {}
    greek_counts = {g: {} for g in ("delta", "gamma", "theta", "rho")}
    unique_samples = set()
    for row in rows:
        path = directory / (row["id"] + ".json")
        result = json.loads(path.read_text()) if path.exists() else {}
        if result.get("fingerprint") != fingerprint:
            result = {}
        unique_samples.update(result.get("sample_keys", []))
        for greek, tally in greek_counts.items():
            fallback = "two-sided-undefined" if greek == "theta" and exact_cash_event(row) else "pending"
            state = result.get("greeks", {}).get(greek, {}).get("status", fallback)
            tally[state] = tally.get(state, 0) + 1
        status = result.get("status", "pending")
        price = result.get("price_status", "pending")
        iv = result.get("iv", {}).get("status", "pending" if status == "pending" else "oracle-unresolved")
        ledger = counts.setdefault(row["_ledger"], {"total": 0, "status": {}, "price": {}, "iv": {}})
        ledger["total"] += 1
        for key, value in (("status", status), ("price", price), ("iv", iv)):
            ledger[key][value] = ledger[key].get(value, 0) + 1
        family = families.setdefault(row["build_id"], {})
        family[status] = family.get(status, 0) + 1
    complete = all(ledger["status"].get("qualified", 0) == ledger["total"] for ledger in counts.values())
    return {"primary_count": len(rows), "complete": complete, "ledgers": counts, "families": families,
            "greek_reference_status": greek_counts, "unique_checkpointed_samples": len(unique_samples),
            "acceptance_scope": "references-only; no backend acceptance or Greek acceptance"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="Frozen cohort directory; repeat for D0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--mode", choices=("inventory", "analytic", "general", "all"), default="inventory")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1, help="Each round independently uses three mesh levels")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--expected-count", type=int, default=8628)
    parser.add_argument("--quantlib", action="store_true", help="Date-aligned no-cash supplementary checks")
    parser.add_argument("--audit-analytic", action="store_true", help="Also run FDE convergence at selected analytic anchors")
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 6 or not 1 <= args.workers <= 4 or args.max_cases < 0:
        parser.error("1..6 rounds, 1..4 workers and nonnegative max-cases required")
    rows, manifests = load_manifests(args.manifest)
    if len(rows) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} frozen primary IDs, got {len(rows)}")
    args.output.mkdir(parents=True, exist_ok=True)
    case_dir = args.output / "cases"
    case_dir.mkdir(exist_ok=True)
    worker_path = str(args.worker.resolve())
    binary_hash = digest(Path(worker_path).read_bytes())
    libraries = runtime_libraries(worker_path)
    effective_worker_hash = digest(encoded({"binary": binary_hash, "libraries": libraries}).encode())
    version = json.loads(subprocess.check_output([worker_path, "--version"], text=True))
    policy = {"version": "reference-qualification-v1", "price_budget": PRICE_BUDGET, "iv_budget": IV_BUDGET,
              "vega_floor": VEGA_FLOOR, "tv_ratio_floor": TV_RATIO_FLOOR,
              "driver_sha256": digest(Path(__file__).read_bytes()), "worker_sha256": binary_hash,
              "runtime_libraries": libraries, "effective_worker_hash": effective_worker_hash,
              "manifest_checksums": [m["checksum_file_sha256"] for m in manifests]}
    fingerprint = digest(encoded(policy).encode())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        revision = "unknown"
    atomic_json(args.output / "metadata.json", {"policy": policy, "fingerprint": fingerprint,
                "worker_version": version, "revision": revision, "manifests": manifests, "python": sys.version,
                "rounds_requested": args.rounds, "workers": args.workers, "quantlib": args.quantlib})
    chosen_ids = set(args.ids_file.read_text().split()) if args.ids_file else None
    if chosen_ids is not None and not chosen_ids <= {r["id"] for r in rows}:
        raise ValueError("selection contains an ID absent from the frozen primary population")
    selected = []
    for row in rows:
        path = case_dir / (row["id"] + ".json")
        old = json.loads(path.read_text()) if path.exists() else {}
        if old.get("fingerprint") != fingerprint:
            if old.get("fingerprint"):
                archive = args.output / "history" / old["fingerprint"]
                archive.mkdir(parents=True, exist_ok=True)
                path.replace(archive / path.name)
            atomic_json(path, {"id": row["id"], "row": row, "status": "pending", "fingerprint": fingerprint})
            old = {}
        if args.mode == "inventory" or (chosen_ids is not None and row["id"] not in chosen_ids):
            continue
        if args.mode == "analytic" and not analytic_eligible(row):
            continue
        if args.mode == "general" and analytic_eligible(row):
            continue
        if should_skip(old, args, row):
            continue
        selected.append(row)
    if args.max_cases:
        selected = selected[:args.max_cases]
    atomic_json(args.output / "selection.json", {"ids": [r["id"] for r in selected], "mode": args.mode,
                                               "frozen_before_execution": True})
    cache = sqlite3.connect(args.output / "samples.sqlite", check_same_thread=False)
    cache.execute("PRAGMA journal_mode=WAL")
    cache.execute("CREATE TABLE IF NOT EXISTS samples(key TEXT PRIMARY KEY, request TEXT, response TEXT)")
    cache.commit()
    lock, local, workers = threading.Lock(), threading.local(), []
    completed = 0
    def execute(row):
        if not hasattr(local, "worker"):
            local.worker = Worker(worker_path, cache, lock, effective_worker_hash)
            with lock:
                workers.append(local.worker)
        result = qualify(local.worker, row, args)
        result["fingerprint"] = fingerprint
        atomic_json(case_dir / (row["id"] + ".json"), result)
        return result["status"]
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        for _ in pool.map(execute, selected):
            completed += 1
            if completed % 25 == 0:
                print(f"completed {completed}/{len(selected)} selected references", flush=True)
    except KeyboardInterrupt:
        # Stop only this driver's owned workers; each completed solve is durable.
        with lock:
            for worker in workers:
                if worker.process.poll() is None:
                    worker.process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True)
        for worker in workers:
            worker.close()
        cache.close()
        summary = summarize(rows, case_dir, fingerprint)
        atomic_json(args.output / "summary.json", summary)
    print(encoded(summary), flush=True)
    return 0 if args.mode == "inventory" or summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
