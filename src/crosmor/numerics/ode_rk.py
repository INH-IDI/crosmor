"""Adaptive 4th-order Runge-Kutta integrator matching CROSMOR RK.F and RK1ST.F."""

from typing import Callable, Tuple
import numpy as np


def comp_err(a: float, b: float, c: float = 0.0) -> float:
    """Relative/absolute error comparison matching COMP.F."""
    if abs(c) < 1.0e-6:
        return abs(a - b)
    else:
        denom = a + c
        if abs(denom) < 1e-12:
            return abs(a - b)
        return abs((a - b) / denom)


def rk1st(
    x: float,
    y: np.ndarray,
    h: float,
    fkt: Callable[[float, np.ndarray], np.ndarray],
) -> Tuple[float, np.ndarray]:
    """Single 4th-order Runge-Kutta step matching RK1ST.F.

    Constants:
        a = [h/2, h/2, h, h, h/2]
    """
    n = len(y)
    ye = np.copy(y)
    w = np.copy(y)
    xe = x

    # Step fractions matching Fortran RK1ST
    a_steps = [h / 2.0, h / 2.0, h, h, h / 2.0]

    for j in range(4):
        z = fkt(xe, w)
        xe = x + a_steps[j]
        for k in range(n):
            w[k] = y[k] + a_steps[j] * z[k]
            ye[k] = ye[k] + a_steps[j + 1] * z[k] / 3.0

    return xe, ye


def rk_adaptive_step(
    x: float,
    y: np.ndarray,
    xe: float,
    fkt: Callable[[float, np.ndarray], np.ndarray],
    eps: float = 0.01,
    eta: float = 0.0,
    first_step: bool = True,
    hs_saved: float = 0.0,
    max_substeps: int = 200,
) -> Tuple[np.ndarray, float]:
    """Adaptive step Runge-Kutta integration from x to xe matching RK.F.

    Returns:
        (y_at_xe, last_step_size_hs)
    """
    n = len(y)
    y_curr = np.copy(y)
    x_curr = x

    if first_step or hs_saved <= 0.0:
        h = (xe - x)
        s = 0.0
    else:
        h = hs_saved
        s = 0.0

    out = False
    substeps = 0

    while substeps < max_substeps:
        substeps += 1

        # Check if approaching end of interval xe
        if (x_curr + 2.01 * h - xe > 0.0) == (h > 0.0):
            hs_saved = h
            out = True
            h = (xe - x_curr) / 2.0

        # Step with 2*h
        _, y1 = rk1st(x_curr, y_curr, 2.0 * h, fkt)

        # Step twice with h
        x2, y2 = rk1st(x_curr, y_curr, h, fkt)
        x3, y3 = rk1st(x2, y2, h, fkt)

        # Check error tolerance on each equation
        max_err = max(comp_err(y1[k], y3[k], eta) for k in range(n))

        if max_err > eps:
            # Error too large: halve step size
            h = 0.5 * h
            out = False
            continue

        # Step accepted
        x_curr = x3
        if out or abs(x_curr - xe) < 1e-12:
            return y3, hs_saved

        y_curr = np.copy(y3)
        if abs(s - 5.0) < 1e-6:
            s = 0.0
            h = 2.0 * h
        s += 1.0

    return y_curr, hs_saved
