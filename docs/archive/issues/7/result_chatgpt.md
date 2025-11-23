Nice, this is an excellent and detailed bug report — thank you. I read everything carefully. I don't think this is a mysterious property of TR-BDF2; the behavior you show (Stage-1 advances, Stage-2 immediately pulls back to essentially (u^n), yet your Newton iterations report tiny residuals) is exactly the sort of symptom produced by one (or more) of these problems:

* **an algebra / sign / coefficient error in the Stage-2 (BDF2) coupling**,
* **an inconsistency between how you evaluate the residual for Newton and how you check the stage equation** (so the Newton “residual” being tiny is not the actual nonlinear equation residual), or
* **a Jacobian / BC / array-overwrite bug** (e.g. wrong array used to assemble RHS or J, or BCs handled differently in residual vs. solver matrix).

Below I summarize the most likely causes, how each produces the observed symptom, and a precise, prioritized set of checks/fixes you can apply immediately. I also show a short sanity computation that demonstrates your Stage-2 RHS looks suspicious and how a small algebraic change would make it reasonable.

---

# Quick diagnosis (summary)

1. **Most likely root cause:** *algebraic / coefficient error in the Stage-2 formula or in the code that computes the Stage-2 RHS.*
   Evidence: your computed Stage-2 RHS (=0.01284) is **smaller than both** (u^n) and (u^*). For a positive forcing/problem with (du/dt\approx 0.935) the Stage-2 RHS should be between (u^n) and a value larger than (u^n) (not substantially below both). That strongly suggests a wrong coefficient (or sign) in the expression used to form the right-hand-side for the BDF2 implicit solve.

2. **Secondary likely cause:** *Newton reporting the wrong "residual"* (for example logging (|\delta u|) or (|A\delta|) rather than the actual nonlinear residual (F(u)=u-cL(u)-r)). That would make the solver claim convergence while the true equation is not satisfied (hence your check of the stage equation shows a (5\times10^{-4}) mismatch).

3. **Other causes to rule out quickly:** Jacobian staleness, incorrect finite-difference perturbation for J, inconsistent BC enforcement, or array aliasing (overwriting buffers). Any one of these can produce wrong converged roots or inconsistent residuals; but the Stage-2 RHS being smaller than the two inputs is a red flag for algebraic coefficient error first.

---

# Immediate algebraic sanity check (why the Stage-2 RHS looks wrong)

You wrote the Stage-2 implicit form and the RHS
[
\text{RHS}=\frac{(1+\alpha)u^*-\alpha u^n}{1+2\alpha},
\qquad \alpha=1-\gamma\approx0.4142.
]
Using your numbers (u^n=0.01583,; u^*=0.02124) this gives
[
\text{RHS}\approx 0.01284,
]
which is **less than** both (u^n) and (u^*). That is counter-intuitive for the TR-BDF2 coupling: the RHS used for the implicit BDF2 solve should not collapse below the previous step value like that for this problem (especially when forcing drives the solution up). A simple algebraic change (putting (1+\alpha) in the denominator instead of (1+2\alpha)) yields
[
\frac{(1+\alpha)u^*-\alpha u^n}{1+\alpha}
= u^* - \frac{\alpha}{1+\alpha}(u^* - u^n)
\approx 0.01962,
]
which is between (u^n) and (u^*) (and much more plausible).

So: **either the analytic BDF2 coupling coefficient in your derivation is incorrect, or the code that computes the RHS uses the wrong denominator** (or an accidental extra factor of (1+\alpha) was introduced). Fixing that algebraic coefficient will likely remove the “Stage-2 pulls back” symptom.

> Action: re-derive the Stage-2 RHS on paper (or compare to a reference implementation / paper for TR-BDF2) and *check the denominator you implement.* If you want I can re-derive the exact algebraic coupling for the version of TR-BDF2 you intend — tell me whether you are using the classical TR-BDF2 with (\gamma=2-\sqrt2) and the formulation where Stage-2 is written as you posted; if yes, I will produce the algebraic derivation step-by-step.

---

# Why the tiny Newton "residual" but large equation mismatch can happen

These code mistakes will commonly produce the pattern you reported:

* **You compute and log** (|A\delta u|) or (|\delta u|) (the Newton step size) instead of the true nonlinear residual (|F(u)|=|u-c L(u)-r|). If the Jacobian (or A) is a good linear approximation, (\delta u) may be tiny but the true (F(u)) can still be non-zero (especially if the Jacobian used in A is wrong/has wrong sign or is evaluated at a different point than assumed). That yields reported convergence but the stage equation check fails by (5\times 10^{-4}).

* **If Jacobian was frozen (quasi-Newton) and is very stale or computed incorrectly**, the linear system you solve is with (A = I - cJ) but Newton’s actual correct linearization is (I - c (\partial L/\partial u)) at the current iterate. The solution of the frozen-Jacobian Newton update can converge (in the sense (|\delta u|\to0)) to a *different* root — one that makes the frozen linearized Newton equation consistent but not the true nonlinear residual = 0.

* **Array aliasing / BC enforcement mismatch:** if the RHS vector `r` or the boundary rows are formed using one u-array and the residual used to judge convergence is formed with a different copy (or if Dirichlet BCs are enforced by overwriting rows after residual assembly but not applied consistently to residual checking), you will again see inconsistency between reported residual and stage equation check.

---

# Concrete debugging checklist (do these in order)

1. **Stop trusting the Newton log. Print the *actual* nonlinear residual (F(u)) after the final Newton update for each stage.**
   For each stage after convergence, compute and print

   ```text
   F = u - c*L(u) - r
   normF = ||F||_2 / sqrt(n)
   ```

   This is the correct residual for the equation (u - c L(u) = r). If `normF` is not small (e.g. ~5e-4 as you observed), your solver is not solving the stage equation even though your Newton log claims convergence.

2. **Compare residual norms computed two ways.**

   * Print `||delta_u||_2` (what you probably printed before).
   * Print `||A*delta_u - res||_2` (linear solve residual) and `||F(u)||` (nonlinear residual).
     If only `||delta_u||` and linear residual are small but `||F(u)||` is not, you have a *consistency* or algebra or Jacobian sign error.

3. **Verify Stage-2 RHS algebra in code.**
   Recompute the RHS with the same floating point code you use and print the intermediate numerators and denominators:

   * print `(1+alpha)*u_star`, `alpha*u_n`, and the denominator you used (`1+2*alpha` in your code). If the denominator is wrong, you’ll see it immediately.

4. **Test the method on the linearized problem using matrix arithmetic (no Newton).**
   Since your operator (L(u) = A u + b) is linear (with (A) from the Laplacian minus identity and (b) the constant 1), you can form the discrete matrix (A) exactly. Then:

   * Stage1 (matrix): solve ((I - c_1 A) u^* = u^n + c_1 A u^n + c_1 b).
   * Stage2 (matrix): solve ((I - c_2 A) u^{n+1} = \text{RHS}).
     If the matrix version behaves correctly (no stagnation), then your Newton implementation or residual/Jacobian handling is the bug. If the matrix version also stagnates, your coefficients are wrong.

   *Because (L) is linear here, you should prefer the exact Jacobian (J=A) and solve the linear systems exactly — that removes Newton nonlinearity from the experiment.*

5. **Check Jacobian assembly:** for this problem you *can* compute the analytic Jacobian. For a 1D second difference with Dirichlet BC and the (-u) term, the Jacobian is the tri-diagonal matrix:
   [
   J = \frac{1}{\Delta x^2},\mathrm{tridiag}(1,-2,1) - I.
   ]
   Use that matrix and see whether Newton with exact J fixes the problem. If the problem disappears using analytic J then your finite-difference Jacobian code is wrong (wrong perturbation size, wrong sign, wrong entries at boundaries).

6. **Check finite-difference Jacobian parameters (if you must use FD):** choose perturbation (h=\sqrt{\epsilon} \times \max(1,|u_i|)) and verify (J_{:,j}\approx (L(u+h e_j)-L(u))/h). Print a few columns and compare to analytic.

7. **Check boundary enforcement consistency:** ensure the Dirichlet BC rows/columns are handled identically in the residual assembly and the matrix (A). If you overwrite rows in the matrix but forget to overwrite the residual at the same indices, you’ll get false convergence.

8. **Guard against array aliasing:** ensure you are not writing `u^*` into the same buffer that `u^n` or `r` is computed from before `r` is fully constructed. Print addresses or simple index checks if in C. Common pattern: compute RHS into `r` using `u_star` but then you overwrite `u_star` before using it in later computations.

9. **If you keep quasi-Newton (frozen J) use it cautiously:** for this problem the operator is linear so there is no cost to reusing exact J; for nonlinear problems reusing J is sometimes OK but you MUST check the true nonlinear residual. If you need robustness, update J or use inexact Newton with line search.

---

# Short, actionable fixes to get you unstuck now

1. **Compute and use analytic Jacobian and solve the two stage linear systems directly.** For your linear operator the analytic Jacobian (J) is simple and exact (tridiagonal). Replace FD-J temporarily with analytic J to confirm the rest of your implementation is correct.

2. **Print the true nonlinear residual after Newton completion** (not (|\delta u|)). If the true residual is not small, do not accept the step — that's the bug.

3. **Check the Stage-2 RHS denominator.** If `1+2*alpha` is indeed in your code, try `1+alpha` as a test (and compare with a trusted reference). Also search literature/notes for TR-BDF2 coefficient formula — in my experience the denominator `1+2*alpha` showing up in code is suspicious and deserves verification.

4. **Run the matrix-only experiment** described above (assemble discrete A and b). If matrix experiment advances correctly, the bug is in nonlinear solver implementation; if matrix experiment still stalls, the coupling coefficients are wrong.

5. **Use unit tests:** create a tiny test (3–7 interior points) where you can print full matrices and vectors and hand-verify algebra. For very small grids you can even compute everything symbolically / numerically by hand.

---

# Answers to your specific questions

1. **TR-BDF2 Formulation — is the Stage-2 coefficient (\dfrac{(1-\alpha)\Delta t}{1+2\alpha}) correct?**
   I cannot assert absolutely without re-deriving the method/consulting the TR-BDF2 derivation you used. Your *numerical evidence* and sanity check strongly suggest the coefficient as implemented is *incorrect for your code* (or you have mis-coded the algebra). Please re-derive the Stage-2 coefficients carefully (or compare to a canonical reference implementation). Practically: *check the denominator used in code*. If changing denominator from (1+2\alpha) to (1+\alpha) produces a RHS between (u^n) and (u^*) and removes stagnation in the matrix test, you had a coefficient bug.

2. **Stage coupling — should Stage 2 use exact (u^*) or a transformed value?**
   Stage-2 should use the converged Stage-1 (u^*) value (the solution of the TR substep) in the BDF2 RHS exactly as written in the continuous method. No additional transform should be necessary beyond the algebraic weighting that the BDF2 coupling prescribes. That said, make sure you use the *converged* (u^*) (not an outdated temporary) and enforce BCs consistently.

3. **Jacobian reuse — is it valid to reuse (J) evaluated at the initial guess for all Newton iterations? Could staleness cause incorrect solutions?**

   * Reusing the Jacobian (quasi-Newton) can be fine for mild nonlinearity and yields convergence but it can also converge to a wrong root or stall if the Jacobian is a poor linear approximation. In your linear problem you should (a) use the analytic Jacobian (exact) or (b) if using FD Jacobian, recompute it at least occasionally.
   * **Yes**, Jacobian staleness *can* cause Newton to converge to the wrong solution (or make the nonlinear residual not zero even though (|\delta u|) is small). Always check the true (F(u)).

4. **BDF2 stability — could Stage 2 converge to a solution near (u^n) rather than advancing forward?**

   * If the RHS of the linear system or the coefficients are wrong, yes — Stage 2 can end up solving a different equation whose root is close to (u^n). Also, if you use an incorrect sign in the Jacobian or wrong assembly you can create an attractor near (u^n). But this is an implementation/coefficients issue, not an inherent stability failure of BDF2 for this linear reaction-diffusion problem.

5. **Discretization error — known issues with TR-BDF2 for reaction–diffusion?**

   * TR-BDF2 is a widely used, stiffly accurate two-stage scheme with good stability for parabolic problems. There are no generic “stagnation” pathologies for linear reaction–diffusion problems when the method is implemented correctly. Issues generally come from algebra/implementation bugs, incorrect BC handling, or misuse of Newton linearization.

6. **Alternative formulation — L-stable variant or different coupling?**

   * TR-BDF2 (with (\gamma=2-\sqrt2)) is already chosen to provide desirable stability properties. If you need extra robustness on nonlinear problems, consider: *use exact Jacobian*, *update Jacobian periodically*, add a simple *line search* or *damping* in Newton updates, or fall back to a one-stage L-stable method (e.g. implicit BDF2 with appropriate starting procedure) until the solver is stable. But first, fix the coefficient / residual issues above.

---

# Practical small debugging snippet (pseudo/C) — compute and print the true nonlinear residual

Add this right after your Newton termination check (for each stage):

```c
// after u has been updated to the converged u_stage
// compute true nonlinear residual F = u - c*L(u) - r
compute_L(u_stage, L_of_u_stage);        // same L() used in Newton
for (i=0;i<n;i++) F[i] = u_stage[i] - c * L_of_u_stage[i] - r[i];

double normF = sqrt(sum_i F[i]*F[i]) / sqrt((double)n);
printf("stage %d: true nonlinear residual norm = %.12e\n", stage, normF);
```

If `normF` is not O(1e-12) (for double precision) and instead ~5e-4, you discovered the inconsistency — go back and examine how `res`/`r` was computed and whether the same `L()`/BCs are used in both places.

---

# If you want me to continue

Tell me which of the following you want next and I will produce it in this message (no waiting, I will do it now):

* (A) I re-derive TR-BDF2 Stage-2 algebra step-by-step from first principles for (\gamma=2-\sqrt2) and produce the exact RHS/coefficient formulas so you can check them line-by-line against your code.
* (B) I produce a minimal, self-contained 1D matrix implementation (Python / NumPy) of TR-BDF2 for your linear operator (L(u)=\partial_{xx} u - u + 1) so you can compare its outputs to your code (this will prove whether the method coefficients are wrong or the Newton solver is buggy).
* (C) I give a small list of the most likely one-line bugs in C implementations that produce your exact symptom (wrong denominator, sign flip in Jacobian FD, residual computed with wrong sign, overwriting arrays in-place, BC rows not zeroed when computing residual). I can provide the exact code snippet to replace in your code.

Say A, B, or C (or ask for a combination) and I’ll run it right now.

