"""Spatial smoothing routines matching CROSMOR SMOOTH.F and SMOOP.F."""

import numpy as np
from numba import njit


@njit(fastmath=True)
def smooth(x: np.ndarray, lx: np.ndarray, mean: np.ndarray, i: int, coef: float) -> float:
    """Moving average over seaward distance coef * L_x(i).

    Matches Fortran SMOOTH.F exactly:
    - Distance cutoff: dis = coef * lx[i]
    - Loops backwards from j = i down to 0 while x[i] - x[j] <= dis
    - Returns sum(mean[j]) / n
    """
    if i < 0 or i >= len(x):
        return 0.0

    dis = coef * lx[i]
    sum_val = 0.0
    count = 0

    for j in range(i, -1, -1):
        if (x[i] - x[j]) <= dis:
            sum_val += mean[j]
            count += 1
        else:
            break

    if count == 0:
        return float(mean[i])
    return float(sum_val / count)


@njit(fastmath=True)
def smoop(psi: np.ndarray, nx: int, nfr: int, gamp: float, x: np.ndarray) -> np.ndarray:
    """Smooth fraction percentages along the cross-shore profile matching SMOOP.F.

    Parameters:
        psi: 2D array of shape (nfr, nx)
        nx: number of points to smooth (up to active nxx)
        nfr: number of sand fractions
        gamp: fraction smoothing factor (gamp = facgrain * 0.2)
        x: cross-shore coordinates

    Returns:
        Updated psi array normalized so sum over fractions equals 1.0 at every x.
    """
    hulp = np.zeros(nx, dtype=np.float64)
    out_psi = np.copy(psi)

    for j in range(nfr):
        out_psi[j, 0] = out_psi[j, 1]
        out_psi[j, nx - 1] = out_psi[j, nx - 2]

        for i in range(1, nx - 1):
            sm1 = x[i + 1] - x[i]
            sm2 = x[i] - x[i - 1]
            sm3 = x[i] + 0.5 * sm1 - (x[i - 1] + 0.5 * sm2)
            if abs(sm1 * sm3) < 1e-12 or abs(sm2 * sm3) < 1e-12:
                smoth = 0.0
            else:
                smoth = gamp * (
                    out_psi[j, i + 1] / (sm1 * sm3)
                    - out_psi[j, i] / (sm1 * sm3)
                    - out_psi[j, i] / (sm2 * sm3)
                    + out_psi[j, i - 1] / (sm2 * sm3)
                )
            hulp[i] = out_psi[j, i] + smoth

        for i in range(1, nx - 1):
            out_psi[j, i] = max(0.0, hulp[i])

    # Re-normalize across fractions at each x
    for i in range(nx):
        somp = 0.0
        for j in range(nfr):
            somp += out_psi[j, i]
        if somp > 1e-9:
            for j in range(nfr):
                out_psi[j, i] /= somp

    return out_psi
