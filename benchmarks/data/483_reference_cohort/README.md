# Frozen interpolation reference population

These archives preserve the exact inputs frozen before #483 backend tuning.
They contain the original generators, numeric ledgers, metadata, source/data
checksums and freeze notes. Archive entry order, timestamps and ownership are
deterministic. No generated input, original source byte, or checksum changed
when the fixtures were packaged for the repository.

| Population | Required prices | Boundary admission | Density study | Exploratory |
|---|---:|---:|---:|---:|
| Original v1 | 6,994 | 690 | 1,088 | 34 |
| Dimensionless addendum v1 | 944 | 0 | 0 | 0 |

The primary population has **8,628 distinct physical IDs**: 7,938 required
prices and 690 separately counted boundary-admission cases. The original
12 physical build requests and four additive requests each specify both
backends, giving 32 backend build requests. Density and exploratory rows
remain separate; neither replaces a failed required case.

To unpack and verify, run from the repository root:

```bash
cohort_dir=$(mktemp -d)
tar -xzf benchmarks/data/483_reference_cohort/cohort-v1.tar.gz -C "$cohort_dir"
tar -xzf benchmarks/data/483_reference_cohort/dimensionless-addendum-v1.tar.gz -C "$cohort_dir"
(cd "$cohort_dir/cohort-v1" && sha256sum -c SHA256SUMS)
(cd "$cohort_dir/dimensionless-addendum-v1" && sha256sum -c SHA256SUMS)
```

Each extracted `generate.py` accepts a separate output directory. Regeneration
must match all original numeric files and `SHA256SUMS` byte for byte. The
repository test verifies archive hashes, extracted file hashes, regeneration,
counts, unique IDs and rolled schedules. Archived generators retain their
original bytes because their source hashes are part of the frozen manifests.

Use the extracted directories with the reference-only harness:

```bash
bazel build -c opt //benchmarks:reference_oracle_worker
python3 tools/reference_qualification.py \
  --manifest "$cohort_dir/cohort-v1" \
  --manifest "$cohort_dir/dimensionless-addendum-v1" \
  --worker bazel-bin/benchmarks/reference_oracle_worker \
  --output "$cohort_dir/reference-results" --mode all \
  --rounds 3 --vega-bump-fraction .04 --workers 4 --quantlib
```

Reference qualification is versioned independently of these immutable inputs.
Pending, unresolved and filtered records remain in the ledger. The historical
freeze metadata correctly says that qualification was pending at the freeze;
it is not a current outcome report. See
[the reference protocol](../../../tools/reference_qualification.md) and
[the approved contract](../../../docs/superpowers/specs/2026-09-06-483-agent-handoffs/contract.md).
Passing the fixture test establishes reproducibility, not backend accuracy,
Greek accuracy, or a monotonicity certificate.
