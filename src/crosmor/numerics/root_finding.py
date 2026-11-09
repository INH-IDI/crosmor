"""Root-finding routines (Brent-Dekker method) matching CROSMOR ZEROIN.F."""

from typing import Callable, Tuple
import math


def tolx_default(x: float) -> float:
    """Tolerance for wave length/dispersion root finder matching TOLX.F."""
    return 0.001 * abs(x) + 1.0e-5


def tolv_default(v: float) -> float:
    """Tolerance for longshore velocity root finder matching TOLV.F."""
    return 0.001 * abs(v) + 1.0e-6


def zeroin(
    ax: float,
    bx: float,
    f: Callable[[float], float],
    tol_func: Callable[[float], float] = tolx_default,
    max_iter: int = 200,
) -> Tuple[float, bool]:
    """Find a root of f(x) = 0 in interval [ax, bx] using the Brent-Dekker method.

    Exact structured equivalent of Fortran ZEROIN.F.
    Returns:
        (root_x, success_flag)
    """
    a = float(ax)
    b = float(bx)
    fa = f(a)
    fb = f(b)

    # Check if endpoints bracket a root
    if fa * fb > 0.0:
        return b, False

    c = a
    fc = fa
    d = b - a
    e = d

    for _ in range(max_iter):
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol = tol_func(b)
        m = 0.5 * (c - b)

        if abs(m) <= tol or abs(fb) < 1e-15:
            return b, True

        # Decide between bisection and interpolation
        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if abs(a - c) < 1e-15:
                # Linear interpolation (secant)
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                # Inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0.0:
                q = -q
            else:
                p = -p

            # Check if interpolation step is acceptable
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e = d
                d = p / q
            else:
                # Fallback to bisection
                d = m
                e = m
        else:
            # Bisection
            d = m
            e = m

        a = b
        fa = fb

        if abs(d) > tol:
            b += d
        elif m > 0.0:
            b += tol
        else:
            b -= tol

        fb = f(b)

        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c = a
            fc = fa
            d = b - a
            e = d

    return b, True
