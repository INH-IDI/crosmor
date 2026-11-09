"""Rayleigh wave spectrum discretization matching CROSMOR RAYL.F."""

from typing import List, Tuple
import math
import numpy as np

from crosmor.models.config import WaveClass, WaveTimePoint
from crosmor.numerics.interpolation import rintpl


def generate_rayleigh_classes(
    time: float,
    nh: int,
    wave_series: List[WaveTimePoint],
) -> List[WaveClass]:
    """Generate discrete wave classes from a Rayleigh distribution matching RAYL.F.

    Parameters:
        time: Current simulation time (s)
        nh: Number of wave classes to discretize into
        wave_series: Time series of (time, hrms, t, teta, w10)

    Returns:
        List of WaveClass objects with (h, t, p, teta, w10).
    """
    if not wave_series:
        return [WaveClass(h=1.0, t=5.0, p=1.0, teta=0.0, w10=0.0)]

    nht = len(wave_series)
    tg = np.array([pt.time for pt in wave_series], dtype=np.float64)
    ht = np.array([pt.hrms for pt in wave_series], dtype=np.float64)
    peri = np.array([pt.t for pt in wave_series], dtype=np.float64)
    tett = np.array([pt.teta for pt in wave_series], dtype=np.float64)
    w10t = np.array([pt.w10 for pt in wave_series], dtype=np.float64)

    tmax = tg[-1] if tg[-1] > 0 else 1.0
    tpol = time % tmax

    hrms = rintpl(tpol, ht, tg)
    tper = rintpl(tpol, peri, tg)
    tet = rintpl(tpol, tett, tg)
    w10 = rintpl(tpol, w10t, tg)

    if nh <= 1:
        return [WaveClass(h=hrms, t=tper, p=1.0, teta=tet, w10=w10)]

    hmax = 2.5 * hrms
    hmin = 0.2 * hrms
    dh = (hmax - hmin) / nh
    somp = 0.0

    classes: List[WaveClass] = [None] * nh  # type: ignore

    for i in range(1, nh):
        hi = hmin + i * dh
        hip = hi + dh
        p_val = math.exp(-((hi / hrms) ** 2)) - math.exp(-((hip / hrms) ** 2))
        somp += p_val
        hs_val = (hi + hip) / 2.0
        ts_val = ((hs_val / hrms) ** 0.2) * tper
        classes[i] = WaveClass(h=hs_val, t=ts_val, p=p_val, teta=tet, w10=w10)

    # Class 0 takes remaining probability
    h1 = hmin
    h2 = hmin + dh
    hs0 = (h1 + h2) / 2.0
    ts0 = ((hs0 / hrms) ** 0.2) * tper
    p0 = max(0.0, 1.0 - somp)
    classes[0] = WaveClass(h=hs0, t=ts0, p=p0, teta=tet, w10=w10)

    return classes
