"""Tridiagonal matrix solver (Thomas algorithm / TDMA) matching CROSMOR TRIDIAR.F."""

import numpy as np
from numba import njit


@njit(fastmath=True)
def tridiar(a: np.ndarray, d: np.ndarray, c: np.ndarray, rl: np.ndarray) -> np.ndarray:
    """Solve tridiagonal linear system: A*x_{i-1} + D*x_i + C*x_{i+1} = RL_i.

    Parameters:
        a: Sub-diagonal (length n, index 0..n-2 used, corresponding to 1..nm-1 in 1-based)
        d: Main diagonal (length n)
        c: Super-diagonal (length n, index 0..n-2 used)
        rl: Right-hand side vector (length n)

    Returns:
        Solution vector x (length n).
    """
    n = len(d)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        res = np.empty(1, dtype=np.float64)
        res[0] = rl[0] / d[0] if d[0] != 0 else 0.0
        return res

    # Work copies to prevent mutating inputs
    d_work = np.copy(d).astype(np.float64)
    c_work = np.copy(c).astype(np.float64)
    x = np.copy(rl).astype(np.float64)

    # Forward elimination matching TRIDIAR.F
    x[0] = x[0] / d_work[0]
    r = d_work[0]

    for j in range(1, n):
        d_work[j - 1] = c_work[j - 1] / r
        r = d_work[j] - a[j - 1] * d_work[j - 1]
        if abs(r) < 1e-15:
            r = 1e-15
        x[j] = (x[j] - a[j - 1] * x[j - 1]) / r

    # Backward substitution matching TRIDIAR.F
    for j in range(1, n):
        idx = n - 1 - j
        x[idx] = x[idx] - d_work[idx] * x[idx + 1]

    return x
