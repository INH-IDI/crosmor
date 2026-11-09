"""Wave dispersion relation and Snell refraction solver matching FWL.F."""

import math
from typing import Tuple
from crosmor.numerics.root_finding import zeroin, tolx_default


def solve_dispersion_wavelength(
    tp: float,
    hd: float,
    teta0_rad: float,
    c0: float,
    vr: float = 0.0,
    g: float = 9.81,
) -> Tuple[float, float, float, float, float]:
    """Solve local wavelength, phase speed, incidence angle and relative period via dispersion relation and Snell's Law.

    Matches FWL.F and golf.f fallback:
    Parameters:
        tp: Wave period (s)
        hd: Local total water depth (m)
        teta0_rad: Deepwater incidence angle (radians)
        c0: Offshore wave celerity at x=0 (m/s)
        vr: Longshore current (m/s, default 0.0)
        g: Gravity acceleration (m/s2)

    Returns:
        (rls, crel, trel, tetax, c0)
        - rls: wavelength L (m)
        - crel: relative phase speed (m/s)
        - trel: relative wave period (s)
        - tetax: local wave angle w.r.t. shore normal (radians)
        - c0: updated offshore celerity
    """
    pi = math.pi
    alf = 0.5 * pi if vr >= 0.0 else 1.5 * pi

    # Objective function matching FWL.F
    def fwl_obj(crel_val: float) -> float:
        if c0 <= 0.0:
            arg = math.sin(teta0_rad)
        else:
            arg = (crel_val / c0) * math.sin(teta0_rad)
        arg = max(-1.0, min(1.0, arg))
        local_teta = math.asin(arg)
        phii = alf - local_teta
        local_rls = (crel_val + vr * math.cos(phii)) * tp
        if local_rls <= 0.0:
            return 1e6
        w_term = (local_rls / tp - vr * math.cos(phii)) ** 2
        d_term = (g * local_rls / (2.0 * pi)) * math.tanh(2.0 * pi * hd / local_rls)
        return w_term - d_term

    # Solve root between 0.001 and 2.0 * tp
    ccx = 0.001
    ccy = 2.0 * tp
    root_crel, success = zeroin(ccx, ccy, fwl_obj, tolx_default)

    if not success or root_crel <= 0.0:
        # Fallback approximation from golf.f (lines 224-226)
        ccy_approx = 4.02 * hd / (tp * tp)
        rkkh = math.sqrt(ccy_approx) * (1.0 + 0.1666 * ccy_approx + 0.031 * ccy_approx * ccy_approx)
        rls = 2.0 * pi * hd / rkkh
        crel = rls / tp
    else:
        crel = root_crel
        if c0 <= 0.0:
            arg = math.sin(teta0_rad)
        else:
            arg = (crel / c0) * math.sin(teta0_rad)
        arg = max(-1.0, min(1.0, arg))
        local_teta = math.asin(arg)
        phii = alf - local_teta
        rls = (crel + vr * math.cos(phii)) * tp

    if c0 <= 0.0:
        c0 = crel

    arg = (crel / c0) * math.sin(teta0_rad) if c0 > 0.0 else math.sin(teta0_rad)
    arg = max(-1.0, min(1.0, arg))
    tetax = math.asin(arg)
    trel = rls / crel if crel > 0.0 else tp

    return rls, crel, trel, tetax, c0
