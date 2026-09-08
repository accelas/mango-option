# Reference qualification for #462

This harness prepares independent price/IV reference evidence. It imports no
price-table builders and never tunes a candidate backend. Production acceptance
and tuning remain gated by #459. The frozen primary and additive D0 manifests
contain 7,938 required prices and 690 boundary-admission rows: all 8,628 IDs
remain in the reference ledgers, including pending and unresolved cases.

## Run

Build the persistent worker and test the protocol/ledger invariants:

```bash
bazel build -c opt //benchmarks:reference_oracle_worker --jobs=4
bazel test -c opt //tests:reference_qualification_test \
  //tests:reference_worker_test --jobs=4 --test_env=OMP_NUM_THREADS=2
```

Pass the two frozen cohort directories explicitly. Keep output on normal disk.
The worker uses two OpenMP threads; the driver defaults to two worker processes.

```bash
python3 tools/reference_qualification.py \
  --manifest /path/to/462-cohort --manifest /path/to/462-cohort-d0 \
  --output /normal-disk/462-references \
  --worker "$PWD/bazel-bin/benchmarks/reference_oracle_worker" \
  --mode inventory
```

Use `--mode analytic` to evaluate every eligible no-future-cash analytic
CALL/PUT reference. Use `--mode all --ids-file pilot-ids.txt
--audit-analytic --quantlib --rounds 1` for a predeclared pilot. The ID file
contains one immutable manifest ID per line. Selection is recorded before
any solve; it changes execution scheduling, never the denominator.

Increase `--rounds` to extend unresolved convergence sequences. Each round has
three levels on each independent axis. A repeated invocation reuses completed
solves and already qualified cases. `--max-cases` is an execution limit; the
remaining cases stay pending. Except for a successful inventory check, exit 2
means the complete reference cohort is still pending/unresolved. Exit 0 is
reserved for a fully qualified primary price/IV reference cohort, not for
backend or Greek acceptance.

The initial FDE pilots use the solver revision recorded in `metadata.json`.
The asymmetric grid-map repair identified during #488 must be qualified before
scaling the general FDE census. Analytic references are unaffected by that
solver-grid defect. Do not interpret the initial pilot as a passing subset.

## Reference methods

**Analytic anchors.** A separate Boost.Multiprecision Black–Scholes calculation
at 50 and 100 decimal digits prices q=0, r>=0 CALLs with no future cash payments.
It also prices no-future-cash PUTs when r<=0 and r<=q, including boundary
equalities, by [Healy, Proposition 2](https://arxiv.org/pdf/2109.15157).
American and European values coincide in these regimes. The worker returns price,
delta, gamma, vega, theta, and rho, with cross-precision disagreement, binary64
conversion error, and a conservative floating-point allowance. The identity is
explicitly refused outside its regime. At an exact cash event, any such theta
is labelled as post-calendar one-sided evidence; an ordinary two-sided theta
across the jump is not defined. Rho at r=0 is labelled on the nonnegative-rate
side of the call regime (nonpositive-rate side for puts).

**General American references.** Public `AmericanOptionSolver::create/solve`
receives the manifest's exact binary64 physical OptionSpec and frozen rolled
payments. The anchored schedule is audited against those offsets; it does not
replace their canonical values. No table or cached table snapshot is an oracle.
High and Ultra automatic profiles are recorded separately.

The controlled grid is sinh-spaced around the strike. Its initial radius is
chosen from the physical query, diffusion length, drift, and cash size before
examining any candidate errors. Three spatial levels halve the sinh-coordinate
step on a fixed domain. Three temporal levels double the step count on the
fixed finest spatial grid. Three domain levels extend the sinh parameter range
from +/-2 to +/-2.5 to +/-3 while holding its coordinate step and physical scale
fixed. Thus wider domains retain the same interior nodes to floating-point
precision; they do not silently coarsen them. The worker test checks actual
GridSpec-generated nodes. Actual Nx, Nt, bounds, work, and timing are retained;
mandatory dividend times can add steps to the nominal count.

Each direction requires three finite levels with contracting, same-sign last
differences, or differences below the stated floating-point floor. The estimate
is twice the empirical Richardson tail, using the observed ratio; ratios >=.75
and unresolved oscillation refuse qualification. The three directional estimates
are summed. This is empirical convergence evidence, not a mathematical error
bound. High/Ultra comparison is an additional consistency check, not a substitute
for independent space/time/domain refinement. Their difference is not a backend
fit error and does not determine a new acceptance tolerance.

**Vega and identifiability.** Vega uses three sigma bump sizes (by default 1%, .5%, .25% of sigma), each evaluated
on all three independent mesh ladders. Mesh and bump-size evidence remain
separate. `--vega-bump-fraction` controls the largest bump; subsequent bumps
halve it. Differences hidden by mesh uncertainty are labelled `mesh-limited`,
not machine roundoff or independently resolved truncation. A wider bump sequence
can expose the truncation trend without changing a price/IV acceptance budget.
The price oracle must resolve <=.001 quote units. An IV-measurable
point must additionally resolve <=2e-6 decimal volatility (0.02 absolute-IV bp)
using the lower numerically supported vega endpoint. A .001 price check alone
cannot establish the IV budget.

TV/K<1e-4 **or** |reference vega|<1e-4 excludes a point from IV measurement.
Either independently established exclusion suffices, even if the other
quantity is unresolved or straddles its threshold. When neither exclusion is
established, a straddled threshold remains unresolved. Price qualification is
still required. Policy v3 records this OR rule explicitly; its new fingerprint
archives earlier classifications while retaining reusable physical solves.
The exclusive states
are reference-measurable, TV-filtered, vega-filtered, both-filtered, unresolved,
and pending. Filtered rows carry no measured-zero IV error. These are numerical
oracle-resolution records, not a market-price uncertainty or confidence API.

**QuantLib.** Supplementary vanilla American FD checks are allowed only when
remaining maturity is an integer number of Actual/365 days, apart from binary64
representation error. Fractional-day inputs are explicitly rejected rather than
truncated. Three meshes are recorded. Converged disagreement is a blocker.
Unconverged or coarser agreement remains labelled as supplementary evidence;
the reference basis remains direct-FDE-only. Cash-model equivalence is not yet
qualified, so cash-dividend rows are explicitly FDE-only and the worker refuses
to substitute a vanilla QuantLib engine. The runtime/library version is recorded.

**Other Greeks.** Analytic evidence is reported where the identity applies.
General delta/gamma/theta/rho remain explicitly unqualified pending independent
bump-stencil and budget work; no numerical acceptance thresholds are invented.
Greek status counters are separate from price/IV qualification. Preserve the
repository's existing tighter Greek tests when these references are integrated.

## Persistence and review

`metadata.json` records the manifest digests, worker binary and linked-library
hashes, compiler/dependency versions, driver hash, numerical policy, revision,
and requested resources. `samples.sqlite` commits each completed worker request.
Its keys include the physical request and effective worker identity, so aliases
can reuse a solve without deleting their manifest IDs. `cases/ID.json` is replaced
atomically after each completed case. Changed qualification fingerprints archive
old records under `history/`; existing physical solves can still be reused when
the worker identity is unchanged. `summary.json` always counts the full cohort.

Worker errors, inconsistent models, unresolved convergence, and pending cases
remain visible. Price qualification, IV measurability, and Greek evidence are
separate results. This preparation does not certify sigma monotonicity or claim
that any interpolation backend meets the frozen targets.
