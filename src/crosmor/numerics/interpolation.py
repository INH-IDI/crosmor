"""Linear interpolation routines with endpoint clamping matching CROSMOR RINTPL."""

import numpy as np
from numba import njit


@njit(fastmath=True)
def rintpl(x: float, f: np.ndarray, xa: np.ndarray) -> float:
    """1D Linear interpolation with endpoint clamping.

    Matches Fortran RINTPL.F exactly:
    - If x <= xa[0], returns f[0]
    - If x >= xa[-1], returns f[-1]
    - Otherwise linearly interpolates between xa[i] and xa[i+1]
    """
    n = len(xa)
    if n == 0:
        return 0.0
    if n == 1 or x <= xa[0]:
        return float(f[0])
    if x >= xa[n - 1]:
        return float(f[n - 1])

    for i in range(n - 1):
        if xa[i] <= x <= xa[i + 1]:
            dx = xa[i + 1] - xa[i]
            if abs(dx) < 1e-12:
                return float(f[i])
            frac = (x - xa[i]) / dx
            return float(f[i] + frac * (f[i + 1] - f[i]))

    return float(f[n - 1])


def rintpl_vec(x_vec: np.ndarray, f: np.ndarray, xa: np.ndarray) -> np.ndarray:
    """Vectorized linear interpolation with endpoint clamping using numpy."""
    return np.interp(x_vec, xa, f)
