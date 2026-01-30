## 1D Cubic B-Spline Collocation: Problem, Algorithm, Expected Output

### Problem Statement

Given a strictly increasing 1D grid of nodes \( \{x_i\}_{i=0}^{n-1} \) and observed function values \( f(x_i) \), we seek a clamped cubic B-spline that interpolates the samples, i.e.

\[
\forall i \in [0, n-1]: \quad \sum_{j=0}^{n-1} c_j N_j(x_i) = f(x_i)
\]

where

* \(N_j(\cdot)\) is the \(j\)-th cubic B-spline basis function built on a knot vector that repeats the first and last grid values four times (clamped boundary conditions).
* \(c_j\) are the unknown spline control points (coefficients) we want to recover.

Because cubic splines have compact support, each \(N_j\) overlaps with at most four consecutive grid values. The linear system induced by the equations above is therefore banded with bandwidth \(4\).

### Mathematical Formulation

1. **Grid requirements**
   * \(n \ge 4\) (at least degree + 1 nodes).
   * \(x_0 < x_1 < \dots < x_{n-1}\) and \(x_{i+1} - x_i \ge \varepsilon\) (no duplicates).
   * Domain width \(x_{n-1} - x_0 > 0\).

2. **Knot vector**
   * Degree \(p = 3\) (cubic).
   * Construct knots \(t\) of length \(n + p + 1 = n + 4\).
   * Clamp endpoints: \(t_0 = t_1 = t_2 = t_3 = x_0\) and \(t_{n} = \dots = t_{n+3} = x_{n-1}\).
   * Interior knots use grid points so that the Schoenberg-Whitney conditions are satisfied.

3. **Basis evaluation**
   * For each grid point \(x_i\), locate the knot span index \(s_i\) such that \(t_{s_i} \le x_i < t_{s_i+1}\).
   * Evaluate the four non-zero cubic basis values \( [N_{s_i}, N_{s_i-1}, N_{s_i-2}, N_{s_i-3}] \) via Cox–de Boor recursion.

4. **Collocation matrix**
   * Define \(B \in \mathbb{R}^{n \times n}\) with entries \(B_{i,j} = N_j(x_i)\).
   * Due to locality, each row has at most four non-zero entries, positioned around the diagonal.

5. **Linear system**
   * Solve \(B \mathbf{c} = \mathbf{f}\), where \(\mathbf{f} = [f(x_0), \dots, f(x_{n-1})]^T\).
   * The matrix is banded and diagonally dominant for well-behaved grids, so Gaussian elimination with partial pivoting or a band solver suffices.

### Algorithmic Steps

1. **Input validation**
   * Ensure grid size \(\ge 4\), sorted order, non-zero spacing, and finite function values.

2. **Knot generation**
   * Build clamped knot vector from the grid.

3. **Collocation matrix assembly**
   * For each \(x_i\): find its span, compute four basis values, write them into row \(i\) at columns \(j = s_i, s_i-1, s_i-2, s_i-3\) (as long as \(0 \le j < n\)).

4. **System solve**
   * Copy \(\mathbf{f}\) into a working RHS vector.
   * Run Gaussian elimination with partial pivoting (or a banded solver) to obtain coefficients \(\mathbf{c}\).

5. **Residual verification**
   * Compute \(r_i = \left| \sum_j B_{ij} c_j - f(x_i) \right|\).
   * Report max residual \( \max_i r_i \) and compare against the tolerance (default \(10^{-9}\)).

6. **Condition estimate**
   * As a lightweight diagnostic, estimate \(\kappa(B)\) via the ratio of largest-to-smallest diagonal entries or other inexpensive heuristics.

### Expected Results

* **Success flag** – `true` if the solver finishes without singularities and residuals are within tolerance.
* **Coefficients** – \(n\) control points that can be fed back into spline evaluators (e.g., BSpline4D) to reproduce the samples.
* **Diagnostics**
  * `max_residual`: \( \max_i | (B \mathbf{c})_i - f(x_i) | \).
  * `condition_estimate`: heuristic indicator of numerical conditioning (values \(\lt\!10^2\) for uniform grids, possibly much higher for highly clustered grids).
  * `error_message`: textual description when the solve fails (e.g., singular matrix, invalid input).

When everything works, plugging the computed coefficients into the spline evaluator should interpolate the original data at each grid node and approximate the underlying function between nodes with cubic accuracy.

### Current Observations

Recent test runs (e.g. `//tests:bspline_collocation_1d_test`) reveal:

* The linear solver frequently reports a singular or ill-conditioned system even for benign grids (uniform 10–20 points), preventing coefficient recovery.
* When a solution is produced, spline evaluations deviate drastically from the underlying function (errors ranging from ~10¹ to 10³ in smooth cases where \(10^{-4}\) is expected).
* Condition estimates returned by the code often remain moderate (`~5`), contradicting the enormous condition numbers observed when we explicitly analyze the collocation matrix (`κ(B) ≈ 10⁹`).

### Suspected Root Cause

The collocation matrix appears nearly singular because of how the knot vector and span selection are currently implemented:

* Interior knots are copied directly from the full grid, which violates the Schoenberg–Whitney conditions near the boundaries (several basis functions share identical knot intervals). This creates rows with overlapping support and near-linear dependence.
* Span selection mixes `upper_bound` with manual clamping; for duplicated knots at the clamped ends this can assign the same span to multiple grid points, causing zero rows or missing basis entries.

Together these issues yield a poorly conditioned band matrix and unreliable residual checks. Fixing the knot construction (e.g. using a true open-uniform scheme or Greville abscissae for interior knots) and tightening span/basis evaluation at the boundaries should restore the theoretical guarantees described above.
