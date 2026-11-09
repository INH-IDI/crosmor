"""Sediment properties, settling velocities, Shields parameters and hiding factors."""

import math
from typing import Tuple, List
import numpy as np


def compute_fluid_properties(temp_c: float, salinity: float) -> Tuple[float, float, float]:
    """Compute water density rhow, kinematic viscosity rnu, and sediment relative density del.

    Matches INIT.F:
        cl = (sa - 0.03) / 1.805
        rhow = 1000 + 1.455*cl - 0.0065*(te - 4 + 0.4*cl)^2
        rnu = (4e-5) / (20 + te)
        del = (rhos - rhow) / rhow
    """
    cl = (salinity - 0.03) / 1.805
    rhow = 1000.0 + 1.455 * cl - 0.0065 * ((temp_c - 4.0 + 0.4 * cl) ** 2)
    rhos = 2650.0
    rnu = 4.0e-5 / (20.0 + temp_c)
    del_density = (rhos - rhow) / rhow
    return rhow, rnu, del_density


def compute_settling_velocity(d_grain: float, del_density: float, rnu: float, g: float = 9.81) -> float:
    """Compute grain settling fall velocity ws matching Van Rijn formulas in INIT.F."""
    d_h = 0.01 * g * del_density * (d_grain**3) / (rnu * rnu)
    if d_grain < 0.0001:
        return float((del_density * g * d_grain * d_grain) / (18.0 * rnu))
    elif d_grain < 0.001:
        return float((10.0 * rnu / d_grain) * (math.sqrt(1.0 + d_h) - 1.0))
    else:
        return float(1.1 * math.sqrt(del_density * g * d_grain))


def compute_shields_critical(
    d_grain: float,
    pmud_pct: float,
    del_density: float,
    rnu: float,
    rhow: float,
    rhos: float = 2650.0,
    g: float = 9.81,
) -> Tuple[float, float, float]:
    """Compute critical Shields parameter and bed shear stresses matching FDSTER.F.

    Returns:
        (dster, taucr, thetcr)
    """
    dster = d_grain * ((del_density * g / (rnu**2)) ** (1.0 / 3.0))

    if dster <= 4.0:
        thetcr = 0.115 / (dster**0.5)
    elif dster <= 10.0:
        thetcr = 0.14 * (dster**-0.64)
    elif dster <= 20.0:
        thetcr = 0.04 * (dster**-0.10)
    elif dster <= 150.0:
        thetcr = 0.013 * (dster**0.29)
    else:
        thetcr = 0.055

    taucr = (rhos - rhow) * g * d_grain * thetcr

    dsand = 0.000062
    cmaxs = 0.65
    fch1 = (dsand / d_grain) ** 1.5 if d_grain < dsand else 1.0

    cmax = (d_grain / dsand) * cmaxs
    cmax = min(cmaxs, max(0.05, cmax))
    fpack = cmax / cmaxs
    if fpack < 1.0:
        fpack = 1.0

    fclay = 1.0
    if pmud_pct >= 0.0:
        fclay = min(2.0, (1.0 + pmud_pct / 100.0) ** 3)

    taucr = fpack * fch1 * fclay * taucr
    return dster, taucr, thetcr


def compute_grain_distribution_params(
    dsi: np.ndarray,
    psi: np.ndarray,
) -> Tuple[float, float, float, np.ndarray]:
    """Compute D50, D90, Dm and hiding/exposure factors egfi matching CALC.F.

    Parameters:
        dsi: Grain diameters for each fraction (m)
        psi: Percentage of each fraction (sum = 1.0)

    Returns:
        (d50, d90, dm, egfi)
    """
    nfr = len(dsi)
    if nfr == 1:
        d50 = float(dsi[0])
        return d50, 2.0 * d50, d50, np.array([1.0], dtype=np.float64)

    # Cumulative distribution
    cum = np.cumsum(psi)
    dm = float(np.sum(dsi * psi))

    # D50
    d50 = float(dsi[0])
    for i in range(nfr - 1):
        if cum[i] <= 0.5 <= cum[i + 1]:
            denom = cum[i + 1] - cum[i]
            frac = (0.5 - cum[i]) / denom if abs(denom) > 1e-12 else 0.0
            d50 = float(dsi[i] + frac * (dsi[i + 1] - dsi[i]))
            break

    # D90
    d90 = float(dsi[-1])
    for i in range(nfr - 1):
        if cum[i] <= 0.9 <= cum[i + 1]:
            denom = cum[i + 1] - cum[i]
            frac = (0.9 - cum[i]) / denom if abs(denom) > 1e-12 else 0.0
            d90 = float(dsi[i] + frac * (dsi[i + 1] - dsi[i]))
            break

    # Hiding/exposure factor egfi
    al19 = math.log(19.0)
    egfi = np.zeros(nfr, dtype=np.float64)
    for i in range(nfr):
        denom = al19 + math.log(max(1e-6, dsi[i] / d50))
        val = (al19 / denom) ** 2 if abs(denom) > 1e-6 else 1.0
        egfi[i] = min(10.0, max(0.1, val))

    return d50, d90, dm, egfi
