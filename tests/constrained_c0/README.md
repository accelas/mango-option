# Private C0 constrained construction experiment

This manual research harness does not change public factories or defaults.
It freezes one construction on the original C0 CALL domain and the 772-row
reference snapshot. Read `manifest.json` before running. The manifest digest
is `29038cd255a8b6434c7511320bc5dbc7c1cf2cb6f17cb3326c40ec4347d1fb94`.

The spline has a double interior rate knot at zero and rate coefficients
[B,C,0,0,0,0]. Its left Bezier coefficients are [B,C,0,0], with an identically
zero right branch. This preserves the existing numerical spline type and
EEP floor/European expression. It does not add an intrinsic floor. Tests pin
native price and derivative behavior of the retained knots/coefficients.

At each of the 60×6 existing x/tau sites, a ten-variable sigma/rate block fits
20 raw-dollar EEP samples from four negative-rate planes. 81 inequalities
constrain total European-plus-EEP vega, allowing decreasing EEP. Linear x/tau
assembly would follow only if all blocks succeeded. Whole physical proof
would still be required after that assembly.

The first bounded run stopped: all 20 parameter combinations and 7200 extracted
samples were finite, with no PDE or extraction failure. 227 of 360 blocks solved;
133 reported dependent blocking constraints. The maximum iteration count was 47
under the declared 100 limit. No tensor was assembled and no candidate price,
actual IV, Greek, or whole-domain proof result exists. This is a limitation of
the bounded active-set solver; zero coefficients are feasible for every block,
so it is not a finding that the financial constraints are infeasible.

Read-only least-squares diagnosis found a full-rank design matrix with condition
16.45. The worst unconstrained raw sample fit has RMS .37709 and maximum 1.03141.
This warns about the narrow rate subspace's approximation capacity but is not
a 772-row physical accuracy measurement or an impossibility proof for a floored
expression. No solver/knots/cuts/targets were retuned after this outcome.

Reproduce the private seam checks:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 tests/constrained_c0/fit_blocks_test.py
OMP_NUM_THREADS=2 bazel build -c opt --jobs=4 //tests/constrained_c0:worker //tests/phase_a:constant_shift --linkopt=-lz --linkopt=-lzstd --linkopt=-llz4 --linkopt=-lsnappy
bazel-bin/tests/constrained_c0/worker representation
```

The solver initially returned unconstrained slope -.2, failing the worked
physical-vega example; the constrained solution is intercept .95, slope -.1.
With ordinary generated knots, the native representation test returned raw
EEP .0493827 at zero rate; the explicit repeated-knot representation returns
exact zero and zero rate derivative on the analytic branch.

`run.py /absolute/path/to/main/repo` runs one sampling/fit/composition attempt
within a shared 600-second construction execution limit. It requires the
already frozen Phase-A payload under the main repo's `.cache/483-research/`.
NumPy 2.2.4 is a research-only environment dependency; no library dependency
was added. Do not rerun the fixed candidate as a search loop. A produced tensor
would require the existing whole physical proof under 4096 nodes/depth 24 and
the entire frozen price/actual-IV/Greek panel, within its 180-second research
execution limit. These limits are not user acceptance ceilings.

Detailed artifacts, manifest, stage logs, sample/matrix data, block statuses,
and a 772-row unmeasured ledger are in `.cache/483-research/459-constrained-c0/`
of the main repo. All reference statuses are retained. The source base is
released a3149920, before later public manual/load proof activation. No
uncertified surface from this experiment is published or returned to users.

After stopping that candidate, a separate solver repair received an analytic
RED/GREEN test. Three tight halfspaces include a redundant normal; their
strictly convex objective has a unique feasible optimum. The old routine
mistook a rounded tangential velocity for a new dependent active equality.
The repair recognizes normals in the active row space while retaining every
inequality in the final feasibility audit. The fixed 772-row candidate was
not rerun. All four solver seam tests and the native representation test pass.

The capacity diagnosis is separate from that implementation failure. The
frozen numerical support spans ratios [.67831,1.34157], wider than the unchanged
published [.7,1.3]. 6720 construction samples are inside and 480 are support-only.
The worst inside block still has raw-sample RMS .363682/max 1.030757 at ratio 1.3,
tau 2; its worst row has sigma .05 and rate -.0125. Neither failure is hidden by
changing the published domain.

`basis_audit.py` is a read-only linear capacity calculation on those same
samples. Its predeclared comparison adds a single rate knot at -.0125, once
(C2) or twice (C1), retaining the zero boundary trace/derivative. The C2 option
reduces inside-domain raw sample RMS .08295→.01239 but still has maximum .13538.
The C1 option has four free negative-rate coefficients and interpolates the
four sample planes to rounding accuracy, as expected from its full-rank
20×20 block. That is capacity evidence, not independent accuracy or proof.
The unsampled interval (-.0125,0) remains unresolved; no second candidate was
constructed and no extra PDE sample was taken.
