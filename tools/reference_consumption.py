# SPDX-License-Identifier: MIT
"""Reconcile completed direct-FDE evidence without running or rewriting a study.

Input records contain case, policy, source and source_sha256. The caller supplies
all completed records for an ID, including archived prefixes. Source hashes are
provenance labels, not authenticated by this in-memory API. Only matching worker,
frozen row and numerical policy evidence can be combined. v2/v3 IV labels are
reclassified using the current OR rule; raw records are never modified.
"""
from fractions import Fraction
import math

from tools import reference_qualification as rq


POLICY_FIELDS = ("effective_worker_hash", "manifest_checksums", "price_budget",
                 "iv_budget", "vega_floor", "tv_ratio_floor", "vega_bump_fraction")


def _qualified(step, intrinsic):
    price = step["price"]
    return (price["stable"] and step["profile_compatible"]
            and rq.price_qualified(price["price"], intrinsic, price["uncertainty"])
            and step["quantlib"]["status"] != "independent-disagreement")


def _interval(step):
    price = step["price"]
    value, error = map(Fraction, (price["price"], price["uncertainty"]))
    return value - error, value + error


def reconcile_reference(records):
    """Use the latest completed round; retain contradictory earlier evidence.

This is empirical evidence admission, not proof of the limiting PDE price.
An unresolved final round blocks an earlier pass even if its price is close.
Earlier unresolved rounds can recover with later qualified evidence; disjoint
qualified allowances cannot be silently reconciled by choosing the latest one.
No minimum number of rounds or numerical tolerance is added.
"""
    if not records:
        raise ValueError("completed direct-FDE evidence is required")
    first = records[0]
    row, policy = first["case"]["row"], first["policy"]
    policy_key = {k: policy[k] for k in POLICY_FIELDS}
    expected = {"price_budget": rq.PRICE_BUDGET, "iv_budget": rq.IV_BUDGET,
                "vega_floor": rq.VEGA_FLOOR, "tv_ratio_floor": rq.TV_RATIO_FLOOR}
    if any(policy[k] != value for k, value in expected.items()):
        raise ValueError("unsupported numerical policy")
    steps, sources = {}, set()
    for record in records:
        case = record["case"]
        if (case["id"] != first["case"]["id"] or case["row"] != row
                or {k: record["policy"][k] for k in POLICY_FIELDS} != policy_key):
            raise ValueError("cannot combine different contracts, workers or numerical policies")
        if not case.get("history"):
            raise ValueError("completed direct-FDE history is required")
        sources.add((record["source"], record["source_sha256"]))
        for step in case["history"]:
            index = step["round"]
            # Ignore optional raw samples/timing: these do not change assessment.
            assessment = {k: step[k] for k in ("round", "plan", "profile_compatible")}
            assessment["price"] = {k: step["price"][k] for k in ("price", "uncertainty", "stable")}
            assessment["quantlib"] = {"status": step["quantlib"]["status"]}
            assessment["vega"] = {k: step["vega"][k] for k in ("value", "uncertainty")}
            if index in steps and steps[index] != assessment:
                raise ValueError("inconsistent assessments for the same completed round")
            steps[index] = assessment
    intrinsic = max((row["spot"] - row["strike"]) *
                    (1 if row["option_type"] == "CALL" else -1), 0)
    ordered = sorted(steps.values(), key=lambda s: s["round"])
    latest = ordered[-1]
    qualified = [s for s in ordered if _qualified(s, intrinsic)]
    contradictions, disjoint = [], []
    # Exact binary64 rational endpoints avoid cancellation around intrinsic.
    for earlier in qualified:
        lo, hi = _interval(earlier)
        for later in ordered:
            value = later["price"]["price"]
            if later["round"] <= earlier["round"] or value is None or not math.isfinite(value):
                continue
            pair = [earlier["round"], later["round"]]
            if not lo <= Fraction(value) <= hi:
                contradictions.append(pair)
            if _qualified(later, intrinsic):
                later_lo, later_hi = _interval(later)
                if later_hi < lo or later_lo > hi:
                    disjoint.append(pair)
    price_ready = _qualified(latest, intrinsic) and not disjoint
    price, vega = latest["price"], latest["vega"]
    iv = rq.iv_eligibility(price["price"] or 0, intrinsic, row["strike"],
                           price["uncertainty"] if price_ready else None,
                           vega["value"] if vega["value"] is not None else math.nan,
                           vega["uncertainty"])
    iv_ready = price_ready and iv["status"] != "oracle-unresolved"
    reason = ("disjoint-qualified-allowances" if disjoint else
              "latest-price-evidence-unresolved" if not price_ready else
              "qualified-reference" if iv_ready else iv["reason"])
    return {"id": first["case"]["id"], "price_ready": price_ready, "iv_ready": iv_ready,
            "reason": reason, "selected_round": latest["round"],
            "price": price["price"] if price_ready else None,
            "price_uncertainty": price["uncertainty"] if price_ready else None,
            "iv": iv, "observational_contradictions": contradictions,
            "disjoint_qualified_allowances": disjoint,
            "completed_rounds": [s["round"] for s in ordered],
            "sources": [{"source": name, "sha256": sha} for name, sha in sorted(sources)]}
