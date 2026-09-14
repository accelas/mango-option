#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarize matched prices and actual IV recovery, retaining failure counts."""
import csv
import json
import math
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
refs = {int(r["id"]): r for r in csv.DictReader((root / "references.csv").open())}
data = {
    v: {(r["fixture"], int(r["id"])): r for r in csv.DictReader((root / f"{v}.csv").open())}
    for v in ("before", "after")
}
assert data["before"].keys() == data["after"].keys()


def stats(values):
    if not values:
        return None
    values = sorted(values)
    x = 0.95 * (len(values) - 1)
    i = int(x)
    p95 = values[i] + (x - i) * (values[min(i + 1, len(values) - 1)] - values[i])
    return {"n": len(values), "median": statistics.median(values), "p95": p95, "max": max(values)}


def eligible(ref):
    strike = float(ref["strike"])
    tv = float(ref["ultra"]) - max(strike - 100.0, 0.0)
    return tv / strike >= 1e-4 and float(ref["vega"]) >= 1e-4


summary = {
    "reference_points": len(refs),
    "table_queries_per_revision": len(data["before"]),
    "reference_price_disagreement_usd": stats([
        abs(float(r["high"]) - float(r["ultra"])) for r in refs.values()
    ]),
    "groups": [],
}
for fixture in ("manual_1y", "adaptive_1y", "adaptive_30d"):
    for region in ("common", "gap", "interior", "edge"):
        keys = [
            k for k in data["before"] if k[0] == fixture and
            ((refs[k[1]]["region"] != "gap") if region == "common"
             else refs[k[1]]["region"] == region)
        ]
        group = {"fixture": fixture, "region": region, "queries": len(keys)}
        for variant in ("before", "after"):
            price_errors, iv_errors, iv_sensitivity = [], [], []
            failure_codes, eligible_failure_codes = {}, {}
            eligible_count = 0
            for key in keys:
                r, ref = data[variant][key], refs[key[1]]
                price, iv, iv_high = (float(r[k]) for k in ("price", "iv", "iv_high"))
                if math.isfinite(price):
                    price_errors.append(abs(price - float(ref["ultra"])))
                if not math.isfinite(iv):
                    code = r["iv_error_code"]
                    failure_codes[code] = failure_codes.get(code, 0) + 1
                if eligible(ref):
                    eligible_count += 1
                    if math.isfinite(iv):
                        iv_errors.append(abs(iv - float(ref["sigma"])) * 1e4)
                        if math.isfinite(iv_high):
                            iv_sensitivity.append(abs(iv - iv_high) * 1e4)
                    else:
                        code = r["iv_error_code"]
                        eligible_failure_codes[code] = eligible_failure_codes.get(code, 0) + 1
            group[variant] = {
                "price_error_usd": stats(price_errors),
                "iv_eligible": eligible_count,
                "iv_error_bps": stats(iv_errors),
                "iv_failures_all": failure_codes,
                "iv_failures_eligible": eligible_failure_codes,
                "iv_reference_sensitivity_bps": stats(iv_sensitivity),
            }
        if region == "common":
            for key in keys:
                assert data["before"][key]["supported"] == data["after"][key]["supported"] == "1"
            for variant in ("before", "after"):
                assert group[variant]["price_error_usd"]["n"] == len(keys)
            old_failures = {k for k in keys if not math.isfinite(float(data["before"][k]["iv"]))}
            new_failures = {k for k in keys if not math.isfinite(float(data["after"][k]["iv"]))}
            group["same_iv_failure_queries"] = old_failures == new_failures
            group["max_price_movement_usd"] = max(
                abs(float(data["after"][k]["price"]) - float(data["before"][k]["price"])) for k in keys
            )
            measured = [k for k in keys if k not in old_failures | new_failures and eligible(refs[k[1]])]
            group["max_eligible_iv_movement_bps"] = max(
                abs(float(data["after"][k]["iv"]) - float(data["before"][k]["iv"])) * 1e4 for k in measured
            )
        else:
            if region == "gap":
                assert all(data["before"][k]["supported"] == "0" for k in keys)
                assert all(data["after"][k]["supported"] == "1" for k in keys)
        summary["groups"].append(group)

(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
