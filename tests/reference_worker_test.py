# SPDX-License-Identifier: MIT
"""Black-box checks at the worker protocol and resumable CLI seams."""
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest

from tools import reference_qualification as rq


def worker_path():
    if os.environ.get("REFERENCE_WORKER"):
        return os.environ["REFERENCE_WORKER"]
    return str(Path(os.environ["TEST_SRCDIR"]) / os.environ.get("TEST_WORKSPACE", "_main")
               / "benchmarks/reference_oracle_worker")


def row(kind="CALL"):
    return {"spot": 100.0, "strike": 100.0, "maturity": 1.0, "volatility": 0.2,
            "rate": 0.05, "dividend_yield": 0.0, "option_type": kind, "rolled_dividends": []}


class WorkerTest(unittest.TestCase):
    def query(self, value, provider="analytic", grid=None):
        answer = subprocess.check_output([worker_path()],
            input=rq.request_line(value, provider, grid) + "\n", text=True)
        return json.loads(answer)

    def test_independent_call_anchor_and_greeks(self):
        result = self.query(row())
        self.assertTrue(result["ok"])
        self.assertAlmostEqual(result["price"], 10.450583572185565, places=11)
        self.assertAlmostEqual(result["vega"], 37.52403469169379, places=10)
        self.assertLess(result["price_error"], 2e-10)
        self.assertGreater(result["price_error"], 0)

    def test_nonpositive_rate_put_anchors_and_boundary(self):
        deep = self.query(dict(row("PUT"), spot=1.0, rate=-0.05))
        self.assertTrue(deep["ok"])
        self.assertGreater(deep["price"], 100.0)
        self.assertAlmostEqual(deep["price"], 100 * math.exp(0.05) - 1, delta=1e-10)
        put = self.query(dict(row("PUT"), rate=0.0))
        call = self.query(dict(row(), rate=0.0))
        self.assertTrue(put["ok"])
        self.assertAlmostEqual(put["price"], call["price"], places=12)
        self.assertAlmostEqual(call["delta"] - put["delta"], 1.0, places=12)
        self.assertAlmostEqual(put["vega"], call["vega"], places=12)
        self.assertAlmostEqual(call["rho"] - put["rho"], 100.0, places=11)
        self.assertAlmostEqual(call["theta"], put["theta"], places=12)
        negative = dict(row("PUT"), rate=-0.05, dividend_yield=0.02)
        exact = self.query(negative)
        ql = self.query(negative, "ql", {"kind": "S", "nx": 800, "nt": 1600,
                                        "radius": 1, "alpha": 4})
        self.assertTrue(exact["ok"])
        self.assertTrue(ql["ok"])
        self.assertAlmostEqual(exact["price"], ql["price"], delta=0.001)

    def test_analytic_identity_refuses_other_models(self):
        for value in (row("PUT"), dict(row(), rate=-0.01), dict(row(), dividend_yield=0.02),
                      dict(row(), rolled_dividends=[{"calendar_time": 0.5, "amount": 1.0}])):
            result = self.query(value)
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "analytic_regime_not_applicable")

    def test_quantlib_never_rounds_fractional_days_or_drops_cash(self):
        grid = {"kind": "S", "nx": 400, "nt": 800, "radius": 1, "alpha": 4}
        fractional = self.query(dict(row(), maturity=0.1234567), "ql", grid)
        self.assertEqual(fractional["error"], "quantlib_requires_integer_days")
        cash = self.query(dict(row(), rolled_dividends=[{"calendar_time": 0.5, "amount": 1}]), "ql", grid)
        self.assertEqual(cash["error"], "quantlib_cash_model_not_qualified")
        valid = self.query(row(), "ql", grid)
        self.assertTrue(valid["ok"])
        self.assertAlmostEqual(valid["price"], 10.450583572185565, delta=0.001)

    def test_domain_extension_keeps_actual_interior_nodes(self):
        plans = rq.mesh_plan(row(), 0)["domain"]
        nodes = [self.query(row(), "grid", g)["nodes"] for g in plans]
        for wider in nodes[1:]:
            extra = (len(wider) - len(nodes[0])) // 2
            common = wider[extra:extra + len(nodes[0])]
            self.assertLess(max(abs(a-b) for a, b in zip(nodes[0], common)), 1e-12)
        self.assertGreater(nodes[-1][-1], nodes[0][-1])

    def test_direct_worker_preserves_cash_and_reports_actual_time_grid(self):
        grid = {"kind": "S", "nx": 257, "nt": 512, "radius": 1.3, "alpha": 4}
        no_cash = self.query(row("PUT"), "fd", grid)
        cash = self.query(dict(row("PUT"), rolled_dividends=[{"calendar_time": 0.25123, "amount": 3}]), "fd", grid)
        self.assertTrue(cash["ok"])
        self.assertGreater(cash["price"], no_cash["price"] + 0.5)
        self.assertGreaterEqual(cash["nt"], 512)
        self.assertEqual(cash["nx"], 257)


class LedgerTest(unittest.TestCase):
    def fixture(self, root):
        builds = [{"id": "TEST", "anchor_maturity": 1.0,
                   "anchored_dividends": [{"calendar_time": 0.5, "amount": 1.0}]}]
        call = dict(row(), id="a", build_id="TEST", maturity=0.25, events=[])
        put = dict(row("PUT"), id="b", build_id="TEST", events=[],
                   rolled_dividends=[{"calendar_time": 0.5, "amount": 1.0}])
        boundary = dict(row("PUT"), id="c", build_id="TEST", maturity=0.5,
                        events=[{"event_index": 0, "offset_in_tau": 0, "calendar_side": "post"}])
        files = {"build-requests.json": rq.encoded(builds) + "\n",
                 "metadata.json": rq.encoded({"version": "test-fixture", "counts": {
                     "required-pricing": 2, "boundary-admission": 1}}) + "\n",
                 "required-pricing.jsonl": rq.encoded(call) + "\n" + rq.encoded(put) + "\n",
                 "boundary-admission.jsonl": rq.encoded(boundary) + "\n"}
        for name, data in files.items():
            (root / name).write_text(data)
        (root / "SHA256SUMS").write_text("".join(
            f"{rq.digest(data.encode())}  {name}\n" for name, data in files.items()))

    def test_ledger_keeps_pending_rows_and_resumes_without_new_solves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, output = root / "manifest", root / "output"
            manifest.mkdir()
            self.fixture(manifest)
            cmd = ["python3", rq.__file__, "--manifest", str(manifest), "--output", str(output),
                   "--worker", worker_path(), "--expected-count", "3", "--mode", "analytic"]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 2, first.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["primary_count"], 3)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["ledgers"]["required-pricing"]["status"], {"pending": 1, "qualified": 1})
            checkpoint = (output / "cases/a.json").read_bytes()
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(second.returncode, 2, second.stderr)
            self.assertEqual((output / "cases/a.json").read_bytes(), checkpoint)
            with sqlite3.connect(output / "samples.sqlite") as cache:
                self.assertEqual(cache.execute("SELECT count(*) FROM samples").fetchone()[0], 1)
            (manifest / "required-pricing.jsonl").write_text("{}\n")
            invalid = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("digest mismatch", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
