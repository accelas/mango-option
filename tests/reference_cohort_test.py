# SPDX-License-Identifier: MIT
"""Protect the predeclared population independently of backend outcomes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from tools import reference_qualification as rq


class FrozenReferenceCohortTest(unittest.TestCase):
    def test_original_bytes_regenerate_and_cover_the_full_population(self):
        repo = (Path(os.environ["TEST_SRCDIR"]) / os.environ.get("TEST_WORKSPACE", "_main")
                if "TEST_SRCDIR" in os.environ else Path(__file__).resolve().parents[1])
        fixtures = repo / "benchmarks/data/483_reference_cohort"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = []
            for line in (fixtures / "SHA256SUMS").read_text().splitlines():
                expected, filename = line.split()
                archive_path = fixtures / filename
                self.assertEqual(hashlib.sha256(archive_path.read_bytes()).hexdigest(), expected)
                with tarfile.open(archive_path) as archive:
                    for member in archive.getmembers():
                        parts = Path(member.name).parts
                        self.assertTrue(member.isfile())
                        self.assertEqual(len(parts), 2)
                        self.assertNotIn("..", parts)
                        self.assertFalse(Path(member.name).is_absolute())
                        path = root / member.name
                        path.parent.mkdir(exist_ok=True)
                        path.write_bytes(archive.extractfile(member).read())
                manifest = root / filename.removesuffix(".tar.gz")
                manifests.append(manifest)
                for check in (manifest / "SHA256SUMS").read_text().splitlines():
                    file_hash, name = check.split()
                    self.assertEqual(hashlib.sha256((manifest / name).read_bytes()).hexdigest(), file_hash)
                regenerated = root / (manifest.name + "-regenerated")
                subprocess.run([sys.executable, str(manifest / "generate.py"), str(regenerated)],
                               check=True, stdout=subprocess.DEVNULL)
                for path in regenerated.iterdir():
                    self.assertEqual(path.read_bytes(), (manifest / path.name).read_bytes(), path.name)
            rows, metadata = rq.load_manifests(manifests)
            self.assertEqual(len(rows), 8628)
            self.assertEqual(len({r["id"] for r in rows}), 8628)
            self.assertEqual(sum(r["_ledger"] == "required-pricing" for r in rows), 7938)
            self.assertEqual(sum(r["_ledger"] == "boundary-admission" for r in rows), 690)
            self.assertEqual(sum(m["metadata"]["backend_build_requests"] for m in metadata), 32)
            for manifest in manifests:
                declared = json.loads((manifest / "metadata.json").read_text())["counts"]
                for ledger, count in declared.items():
                    self.assertEqual(len((manifest / f"{ledger}.jsonl").read_text().splitlines()), count)


if __name__ == "__main__":
    unittest.main()
