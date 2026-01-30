<!-- SPDX-License-Identifier: MIT -->
Nice, this is a very clear write-up — thank you. You’ve already identified the likely cause (boundary rows / algebraic constraints breaking the telescoping property and allowing drift into the Neumann null space). Below I give a short diagnosis plus several robust fixes (ranked from simplest to most robust), explanations of why they work, and ready-to-run pseudo/Python code you can drop into your solver to test. My recommendation: switch to a flux-form (finite-volume) or ghost-point implementation first — those fix conservation without needing ad-hoc mass fixes. If you need a short-term quick patch, project out the mean after each stage; if you want a clean long-term fix, use flux-form or ghost points and handle the null-space in the linear solver.

---

# Quick diagnosis (why you see +2% mass)

1. Your current enforcement `u_0 = u_1` and `u_{n-1} = u_{n-2}` *as algebraic constraints* changes the discrete matrix and residual so the discrete spatial operator no longer has the telescoping (zero-sum) property.
2. As a consequence the discrete Laplacian no longer satisfies `sum_i L(u)_i = 0`, so the time integrator can produce net mass creation.
3. With pure Neumann the continuous operator has a nullspace (constant modes). If your linear systems or BC enforcement are inconsistent with conservation you will see drift in that nullspace (a constant offset appears).

---

# Correct approaches (with pros/cons)

### 1) **Preferred: Use flux-form (finite-volume) discretization** — *conservative, simple*

Write the operator as discrete divergence of fluxes:
[
L_i = \frac{F_{i+1/2}-F_{i-1/2}}{\Delta x},\qquad F_{i+1/2} = -D\frac{u_{i+1}-u_i}{\Delta x}
]
With zero-flux BCs set (F_{-1/2}=F_{n-1/2}=0). Summing (L_i) telescopes and gives zero exactly (up to rounding). This form is *guaranteed* conservative and is the recommended fix.

**Implementation notes**

* You only need u at grid points (i=0..n-1). Compute fluxes at half points and construct tridiagonal Laplacian such that each row sums to zero.
* This is identical to a second-order finite-volume method; it avoids special algebraic constraint rows.

### 2) **Ghost-point method** — *also conservative if used consistently*

Enforce Neumann by reflecting: (u_{-1}=u_1), (u_{n}=u_{n-2}). Then apply the standard centered second derivative at boundary points:
[
L_0 = \frac{u_{-1}-2u_0+u_1}{\Delta x^2} = \frac{2(u_1-u_0)}{\Delta x^2}
]
Ghost-point implementation preserves telescoping if you *do not* add algebraic constraint rows (i.e., do not overwrite row 0 with `u0 - u1 = 0`). Preferable to strongly enforcing `u0=u1`.

### 3) **Modify boundary rows to maintain divergence form instead of algebraic constraints**

If you must eliminate ghost points, replace row 0 and row n-1 with finite-difference formulas that approximate the Laplacian consistent with Neumann (same as ghost-stencil), not with a constraint `u0=u1`. That keeps the discrete operator row-sum property.

### 4) **Projection / pinning (quick patch)** — *non-invasive but ad-hoc*

After each stage (or after each complete time step) project the solution to have the correct total mass:

```text
mass_error = (sum(u)*dx - M0)
u[:] -= mass_error / domain_length   # subtract uniform offset so mass is exact
```

This works but hides the real discretization inconsistency; use as a last resort or temporary fix in tests.

### 5) **Null-space handling in linear solver** (for implicit solves)

Because the Laplacian with Neumann is singular (constant vector in null space), some solvers may behave poorly. Options:

* Solve the linear system in a subspace orthogonal to constants (project RHS and solution).
* Add a weak pinning constraint (add a tiny regularization ε on diagonal) — but this perturbs solution.
* Use an augmented system with Lagrange multiplier to enforce mass conservation exactly:
  [
  \begin{bmatrix} A & \mathbf{1}\ \mathbf{1}^T & 0\end{bmatrix}
  \begin{bmatrix} u\ \lambda \end{bmatrix}
  =
  \begin{bmatrix} b\ M_0 \end{bmatrix}
  ]
  This keeps mass exact but is more code.

---

# Concrete conservative discretization (flux-form) + TR-BDF2 outline

Below is a compact Python/Numpy example showing how to build a conservative Laplacian (tridiagonal) using flux form, perform TR-BDF2 stages solving the tridiagonal systems with Thomas solver, and optionally project to exact mass (not usually needed if flux-form is used). This is intended as a drop-in reference — adapt to your Newton solver (the Jacobian will then be the discrete Laplacian times D/Δx²).

```python
import numpy as np

def build_flux_laplacian(n, dx, D):
    # returns tridiagonal coefficients a (lower), b (diag), c (upper)
    # for operator L(u) = D * second derivative with zero-flux Neumann,
    # consistent with finite-volume flux form so sum(L) = 0.
    a = np.zeros(n)  # lower diag (a[1] is below diag[1])
    b = np.zeros(n)
    c = np.zeros(n)
    # interior i = 1..n-2
    for i in range(n):
        if i == 0:
            # L0 = 2*(u1 - u0)/dx^2 * D -> row: (-2D/dx^2)*u0 + (2D/dx^2)*u1
            b[i] = -2*D/(dx*dx)
            c[i] =  2*D/(dx*dx)
        elif i == n-1:
            a[i] =  2*D/(dx*dx)
            b[i] = -2*D/(dx*dx)
        else:
            a[i] = D/(dx*dx)
            b[i] = -2*D/(dx*dx)
            c[i] = D/(dx*dx)
    return a, b, c

def thomas_solve(a, b, c, rhs):
    # a[0] unused (or zero), b[0..n-1], c[n-1] unused
    n = len(b)
    # copy
    ac = a.copy()
    bc = b.copy()
    cc = c.copy()
    d = rhs.copy()
    # forward elimination
    for i in range(1, n):
        m = ac[i]/bc[i-1]
        bc[i] = bc[i] - m*cc[i-1]
        d[i]  = d[i] - m*d[i-1]
    # back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / bc[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - cc[i]*x[i+1]) / bc[i]
    return x

# parameters
n = 101
x = np.linspace(0,1,n)
dx = x[1] - x[0]
D = 0.1
dt = 0.01
gamma = 2 - np.sqrt(2)

# initial condition
u0 = np.exp(-50*(x - 0.5)**2)
M0 = u0.sum()*dx

# build Laplacian tridiagonal in flux-consistent form
a, b, c = build_flux_laplacian(n, dx, D)

# For TR-BDF2 we need to solve linear systems (I - c * J) delta = F
# but since L is linear we can form the matrices directly for each stage.
# stage 1: implicit trapezoid with coefficient c1 = gamma*dt/2
# stage 2: BDF2 with coefficient c2 = (1-gamma)*dt/(2-gamma)
c1 = gamma*dt/2.0
c2 = (1-gamma)*dt/(2-gamma)

# build (I - c*L) tridiagonal coefficients for solves:
def build_implicit_tridiag(aL, bL, cL, coeff):
    n = len(bL)
    A_low = np.zeros(n)
    A_diag = np.zeros(n)
    A_up = np.zeros(n)
    A_low[:] = -coeff * aL[:]   # note L was defined so that L(u)=A*u; solve (I - coeff*L)
    A_diag[:] = 1.0 - coeff * bL[:]
    A_up[:] = -coeff * cL[:]
    # enforce a[0] = 0 and c[n-1] = 0 (Thomas solver format)
    A_low[0] = 0.0
    A_up[-1] = 0.0
    return A_low, A_diag, A_up

A1_low, A1_diag, A1_up = build_implicit_tridiag(a,b,c,c1)
A2_low, A2_diag, A2_up = build_implicit_tridiag(a,b,c,c2)

# time stepping
u = u0.copy()
for step in range(100):
    # stage 1 (Trapezoidal)
    rhs1 = u + c1 * (np.dot_linalg_not_required_here := 0)  # L(u) is linear -> compute Lu
    # compute Lu explicitly using tridiagonal multiply
    Lu = np.zeros(n)
    # multiply tridiagonal a,b,c by u: Lu = a*u_{i-1} + b*u_i + c*u_{i+1}
    for i in range(n):
        val = 0.0
        if i-1 >= 0: val += a[i]*u[i-1]
        val += b[i]*u[i]
        if i+1 < n: val += c[i]*u[i+1]
        Lu[i] = val
    rhs1 = u + c1 * Lu
    # solve (I - c1*L) u_star = rhs1
    u_star = thomas_solve(A1_low, A1_diag, A1_up, rhs1)

    # stage 2 (BDF2)
    rhs2 = (1.0/(gamma*(2-gamma))) * u_star - ((1-gamma)**2/(gamma*(2-gamma))) * u
    u_new = thomas_solve(A2_low, A2_diag, A2_up, rhs2)

    # optional small projection to exact mass if you want:
    # mass_err = u_new.sum()*dx - M0
    # u_new -= mass_err / 1.0   # subtract uniform offset (domain length=1)
    u = u_new

print("Final mass ratio:", (u.sum()*dx)/M0)
```

*(Notes: the code above is illustrative — replace the dot-product step with a fast tridiagonal multiply and integrate into your Newton/Jacobian assembly if using Newton iterations. The key is how `a,b,c` were built to preserve flux.)*

---

# Specific answers to your numbered questions

1. **Does current discretization guarantee (\sum_i L(u)_i = 0)?**
   Not if you enforce `u_0 = u_1` and `u_{n-1} = u_{n-2}` as algebraic rows while leaving the interior stencil unchanged. That breaks the telescoping property. A flux-form or ghost-point consistent implementation *does* guarantee sum zero (up to machine precision).

2. **Should boundary (L(u)*0,L(u)*{n-1}) use one-sided differences or ghost points?**
   Use a ghost-point reflection or a finite-volume flux form. Both are conservative if implemented consistently. Avoid replacing the boundary differential operator by a zero row or an algebraic constraint row that is inconsistent with adjacent PDE rows.

3. **Is there inconsistency between enforcing `u_0=u_1` and computing `L(u)_1` that depends on `u_0`?**
   Yes — that is the root of your problem. Row 0 as an algebraic constraint removes one DOF but the stencil in row 1 still treats u0 as an independent DOF; the resulting linear system no longer has the correct discrete divergence structure.

4. **Should we explicitly constrain the null space?**
   It’s good practice to either:

   * Use a discretization that *does not* produce drift (preferred), or
   * If the linear solver cannot handle the singularity, project the solution/RHS to the orthogonal complement of constants (i.e., enforce mass), or pin one DOF (less desirable physically). For time integration, a conservative discretization is the best prevention.

5. **Will ghost points with (u_{-1}=u_1), (u_n = u_{n-2}) help?**
   Yes — if you *then use the standard centered stencil at i=0 and i=n-1* and do not add an algebraic constraint row, the ghost point method provides the right second derivative and preserves conservation.

6. **Weak enforcement vs strong enforcement through matrix constraints?**
   Weak enforcement (natural BCs) via fluxes or finite-element weak forms is generally better for conservation. Strong enforcement via algebraic constraints is brittle unless you also rewrite adjacent PDE rows consistently (which is error-prone).

---

# Practical checklist for you to fix solver

1. Replace boundary algebraic rows (`u0 - u1 = 0`) with either:

   * Ghost-point stencils (use u_{-1}=u1 and u_n = u_{n-2}) and keep centered second derivatives at boundaries; **or**
   * Finite-volume flux form, explicitly compute fluxes at half-nodes and assemble Laplacian so each row is divergence of fluxes.
2. Ensure the Jacobian `J = dL/du` assembled for Newton has row-sum zero (so that `(I - c J)` preserves the constant nullspace structure). That helps the linear solver.
3. If your linear solver can’t handle singular matrices, either augment with constraint to fix mass or project the RHS/solution to remove the constant mode.
4. Validate by checking `sum(L(u))*dx` for random u — it should be ≈ 0. Also monitor mass each time step (or after each Newton solve).

---

# When you should use the projection / Lagrange approach

* Projection (subtracting mean) is a useful test and a short-term band-aid during debugging.
* Lagrange multiplier augmentation is useful when you want the solver to enforce mass exactly inside the linear system (good for iterative solvers that might drift).

---

# Final recommendation

Change the boundary treatment to a flux-conservative form (finite-volume) or ghost-point implementation and remove the algebraic `u0=u1` constraint rows. That single change should remove the ~2% mass drift and make the TR-BDF2 scheme conserve mass to roundoff. If you want, I can (A) adapt the above Python example into code matching your language/data structures (C/Fortran/C++), or (B) show the exact modified tridiagonal entries and Jacobian rows you should assemble for a Newton linear solver — tell me which language or matrix storage (banded tridiagonal arrays) you’re using and I’ll produce the snippet.

