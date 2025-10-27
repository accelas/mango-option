#ifndef BRENT_H
#define BRENT_H

#include <math.h>
#include <stdbool.h>

// Brent's method for root finding (header-only)
//
// Finds a root of f(x) = 0 in the interval [a, b]
// Combines bisection, secant method, and inverse quadratic interpolation
//
// Properties:
// - Guaranteed convergence (if root exists in [a,b])
// - Superlinear convergence rate
// - Does not require derivative
// - More robust than Newton's method
//
// Reference: Brent, R. (1973). "Algorithms for Minimization without Derivatives"

// Function pointer type for the function to find root of
typedef double (*BrentFunction)(double x, void *user_data);

// Result structure
typedef struct {
    double root;          // Found root (if converged)
    double f_root;        // Function value at root
    int iterations;       // Number of iterations performed
    bool converged;       // True if converged, false if max iterations reached
} BrentResult;

// Brent's method root finder
//
// Parameters:
//   f: Function to find root of
//   a, b: Interval endpoints (must bracket root: f(a)*f(b) < 0)
//   tolerance: Absolute tolerance for root (e.g., 1e-6)
//   max_iter: Maximum number of iterations (e.g., 100)
//   user_data: Optional data passed to function f
//
// Returns:
//   BrentResult with root, convergence status, and iterations
static inline BrentResult brent_find_root(BrentFunction f,
                                         double a, double b,
                                         double tolerance,
                                         int max_iter,
                                         void *user_data) {
    BrentResult result = {0.0, 0.0, 0, false};

    double fa = f(a, user_data);
    double fb = f(b, user_data);

    // Check if root is bracketed
    if (fa * fb > 0.0) {
        // Root not bracketed - return failure
        result.root = a;
        result.f_root = fa;
        return result;
    }

    // If one endpoint is already a root
    if (fabs(fa) < tolerance) {
        result.root = a;
        result.f_root = fa;
        result.converged = true;
        return result;
    }
    if (fabs(fb) < tolerance) {
        result.root = b;
        result.f_root = fb;
        result.converged = true;
        return result;
    }

    // Ensure |f(a)| >= |f(b)|
    if (fabs(fa) < fabs(fb)) {
        double tmp = a; a = b; b = tmp;
        tmp = fa; fa = fb; fb = tmp;
    }

    double c = a;     // Previous value of b
    double fc = fa;   // f(c)
    bool mflag = true; // Use bisection on first iteration

    for (int iter = 0; iter < max_iter; iter++) {
        result.iterations = iter + 1;

        double s; // Candidate for next point

        // Try inverse quadratic interpolation if possible
        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Decide whether to accept s or use bisection
        double tmp2 = (3.0 * a + b) / 4.0;
        bool condition1 = !((s > tmp2 && s < b) || (s < tmp2 && s > b));
        bool condition2 = mflag && fabs(s - b) >= fabs(b - c) / 2.0;
        bool condition3 = !mflag && fabs(s - b) >= fabs(c - result.root) / 2.0;
        bool condition4 = mflag && fabs(b - c) < tolerance;
        bool condition5 = !mflag && fabs(c - result.root) < tolerance;

        if (condition1 || condition2 || condition3 || condition4 || condition5) {
            // Use bisection
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        double fs = f(s, user_data);

        // Store previous c value
        result.root = c;
        c = b;
        fc = fb;

        // Update brackets
        if (fa * fs < 0.0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        // Ensure |f(a)| >= |f(b)|
        if (fabs(fa) < fabs(fb)) {
            double tmp = a; a = b; b = tmp;
            tmp = fa; fa = fb; fb = tmp;
        }

        // Check for convergence
        if (fabs(fb) < tolerance || fabs(b - a) < tolerance) {
            result.root = b;
            result.f_root = fb;
            result.converged = true;
            return result;
        }
    }

    // Max iterations reached
    result.root = b;
    result.f_root = fb;
    result.converged = false;
    return result;
}

// Convenience function: Find root with default parameters
// tolerance = 1e-6, max_iter = 100
static inline BrentResult brent_find_root_simple(BrentFunction f,
                                                 double a, double b,
                                                 void *user_data) {
    return brent_find_root(f, a, b, 1e-6, 100, user_data);
}

#endif // BRENT_H
