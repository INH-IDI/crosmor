"""Longshore current solver (with and without lateral mixing) matching SNELH.F and FVV.F."""

import math
from typing import Tuple, List, Optional
import numpy as np

from crosmor.numerics.root_finding import zeroin, tolv_default
from crosmor.numerics.tridiagonal import tridiar
from crosmor.numerics.smoothing import smooth


def compute_longshore_current(
    x: np.ndarray,
    h3: np.ndarray,
    t3: np.ndarray,
    h: np.ndarray,
    s: np.ndarray,
    lx: np.ndarray,
    tetx_rad: np.ndarray,
    dbr: np.ndarray,
    d50h: np.ndarray,
    vnul: float,
    rhow: float,
    del_density: float,
    w10: float,
    els: float,
    nit: int,
    coef5: float,
    sef1: float,
    hgrens: float,
    arksc_in: Optional[np.ndarray] = None,
    g: float = 9.81,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute longshore current velocity profile matching SNELH.F.

    Returns:
        (v_no_mix, v_mix, c1_chezy, c2_chezy, rksc_out)
    """
    n = len(x)
    pi = math.pi
    rafac = 1.25
    fa = 0.001

    v_no_mix = np.zeros(n, dtype=np.float64)
    v_mix = np.zeros(n, dtype=np.float64)
    c1 = np.zeros(n, dtype=np.float64)
    c2 = np.zeros(n, dtype=np.float64)
    arksc = np.zeros(n, dtype=np.float64)
    mean = np.zeros(n, dtype=np.float64)

    if arksc_in is not None:
        arksc[:] = arksc_in

    h0 = h[0] + s[0]
    h0 = max(1e-4, h0)
    vvv = vnul if abs(vnul) >= 1e-6 else 0.001
    tsy = rafac * fa * w10 * abs(w10)

    # 1. First pass: solve local longshore current without lateral mixing (FVV.F)
    for i in range(n):
        d50 = d50h[i]
        hi = max(1e-4, h[i] + s[i])
        tetxi = tetx_rad[i]

        alf = 0.5 * pi if vvv >= 0.0 else 1.5 * pi
        phii = alf - tetxi
        if phii > pi:
            phii = 2.0 * pi - phii
        gam = 0.8 + phii - 0.3 * (phii**2)

        sinh_arg = 2.0 * pi * hi / max(1e-4, lx[i])
        sinh_val = math.sinh(min(50.0, sinh_arg))
        udel2 = (pi * h3[i] / max(1e-4, t3[i])) / sinh_val if sinh_val > 0 else 0.0

        rksc = arksc[i]
        if d50 > 0.0:
            uwc = math.sqrt(udel2**2 + vvv**2)
            rmob = (uwc**2) / (del_density * g * d50)

            sef = ((1.0 - sef1) / (3.0 * hgrens)) * hi + (5.0 / 3.0 * sef1) - (2.0 / 3.0)
            if hi >= 5.0 * hgrens:
                sef = 1.0
            if hi <= 2.0 * hgrens:
                sef = sef1

            d90 = 2.0 * d50
            dsilt = 0.000032
            dsand = 0.000062
            dgravel = 0.002
            fch2 = min(1.0, max(0.3, d50 / (1.5 * dsand)))
            fcoarse = min(1.0, (0.25 * dgravel / d50) ** 1.5)

            # Ripples in longshore direction
            if rmob <= 50.0:
                rcr = 150.0 * fcoarse * d50
            elif rmob < 250.0:
                rcr = (-0.65 * rmob + 182.5) * fcoarse * d50
            else:
                rcr = 20.0 * fcoarse * d50

            if d50 <= dsilt:
                rcr = 20.0 * dsilt
            if rcr <= d90:
                rcr = d90
            if rcr <= 0.001:
                rcr = 0.001
            if sef > 1.0:
                rcr = 0.02
            if rcr >= 0.1:
                rcr = 0.1

            # Megaripples in longshore direction
            if rmob <= 50.0:
                rcmr = 0.0002 * fch2 * rmob * hi
            elif rmob <= 550.0:
                rcmr = fch2 * (-0.00002 * rmob + 0.011) * hi
            else:
                rcmr = 0.0

            if rcmr >= 0.2:
                rcmr = 0.2
            if rcmr < 0.02 and d50 >= 1.5 * dsand:
                rcmr = 0.02
            if rcmr < 0.02 and d50 < 1.5 * dsand:
                rcmr = 200.0 * d50
            if d50 <= dsilt:
                rcmr = 0.0
            if sef > 1.0:
                rcmr = 0.0

            rksc = math.sqrt(rcr**2 + rcmr**2)

        arksc[i] = rksc
        rka = 10.0 * rksc
        if abs(vvv) > 0.0:
            term = min(udel2 / abs(vvv) * gam, 2.3026)
            rka = math.exp(term) * rksc

        c1[i] = 18.0 * math.log10(max(1.01, 12.0 * hi / max(1e-5, rka)))
        c2[i] = 18.0 * math.log10(max(1.01, 12.0 * hi / max(1e-5, rksc)))

        c0 = c1[0] if c1[0] > 0 else 30.0
        dhdy = -abs(vnul) * vnul / (c0 * c0 * h0)

        cv = lx[i] / max(1e-4, t3[i])
        b1 = (c1[i] ** 2) / (rhow * g)
        dbrr = math.sin(tetxi) * dbr[i] / cv if cv > 0 else 0.0
        mean[i] = dbrr
        if i > 0:
            dbrr = smooth(x, lx, mean, i, coef5)

        vabs = b1 * tsy + b1 * dbrr - (c1[i] ** 2) * hi * dhdy
        if vabs < 0.0:
            vvv = -math.sqrt(-vabs)
        else:
            vvv = math.sqrt(vabs)

        # Refine root using FVV objective function
        def fvv_obj(v_val: float) -> float:
            if abs(v_val) > 0.0:
                t_term = min(udel2 / abs(v_val) * gam, 2.3026)
                rka_loc = math.exp(t_term) * rksc
            else:
                rka_loc = 10.0 * rksc
            cc_loc = 18.0 * math.log10(max(1.01, 12.0 * hi / max(1e-5, rka_loc)))
            b1_loc = (cc_loc**2) / (rhow * g)
            return -v_val * abs(v_val) + b1_loc * tsy + b1_loc * dbrr - (cc_loc**2) * hi * dhdy

        vvx = 0.5 * vvv - 0.05
        vvy = 1.5 * vvv + 0.05
        if vvv < 0.0:
            vvx, vvy = 1.5 * vvv - 0.05, 0.5 * vvv + 0.05

        root_v, ok = zeroin(vvx, vvy, fvv_obj, tolv_default)
        if ok:
            vvv = root_v

        v_no_mix[i] = vvv
        v_mix[i] = vvv

    # 2. Second pass: iterative solution with horizontal eddy viscosity (els)
    a1_diag = np.zeros(n, dtype=np.float64)
    a2_diag = np.zeros(n, dtype=np.float64)
    a3_diag = np.zeros(n, dtype=np.float64)
    rl_rhs = np.zeros(n, dtype=np.float64)

    c0 = c1[0] if c1[0] > 0 else 30.0
    dhdy = -abs(vnul) * vnul / (c0 * c0 * h0)

    for _ in range(nit):
        a1_diag[0] = 0.0
        a2_diag[0] = 1.0
        a3_diag[0] = 0.0
        rl_rhs[0] = v_no_mix[0]

        for i in range(1, n - 1):
            vvv = v_mix[i]
            hi = max(1e-4, h[i] + s[i])
            tetxi = tetx_rad[i]

            alf = 0.5 * pi if vvv >= 0.0 else 1.5 * pi
            phii = alf - tetxi
            if phii > pi:
                phii = 2.0 * pi - phii
            gam = 0.8 + phii - 0.3 * (phii**2)

            sinh_arg = 2.0 * pi * hi / max(1e-4, lx[i])
            sinh_val = math.sinh(min(50.0, sinh_arg))
            udel2 = (pi * h3[i] / max(1e-4, t3[i])) / sinh_val if sinh_val > 0 else 0.0
            rksc = arksc[i]

            rka = 10.0 * rksc
            if abs(vvv) > 0.0:
                term = min(udel2 / abs(vvv) * gam, 2.3026)
                rka = math.exp(term) * rksc

            c_loc = 18.0 * math.log10(max(1.01, 12.0 * hi / max(1e-5, rksc)))
            cv = lx[i] / max(1e-4, t3[i])
            b1 = (c_loc**2) / (rhow * g)

            efac = dbr[i] / rhow
            efac_term = (efac ** (1.0 / 3.0)) if efac > 0 else 0.0
            e_mix = 0.1 * math.sqrt(g) / c_loc * abs(vvv) * hi + 0.025 * hi * efac_term + els
            if e_mix < 0.1:
                e_mix = 0.1

            t1 = (x[i + 1] - x[i]) * (x[i] - x[i - 1])
            t2 = (x[i] - x[i - 1]) ** 2

            dbrr = mean[i]
            if i > 0:
                dbrr = smooth(x, lx, mean, i, coef5)

            inv_geom = -1.0 / t1 - 1.0 / t2
            denom = b1 * rhow * hi * e_mix

            rl_rhs[i] = ((-b1 * tsy - b1 * dbrr + (c_loc**2) * hi * dhdy) / denom) / inv_geom
            a3_diag[i] = (1.0 / t1) / inv_geom
            a1_diag[i - 1] = (1.0 / t2) / inv_geom
            a2_diag[i] = 1.0 - abs(vvv) / denom / inv_geom

        a1_diag[n - 2] = -0.9
        a1_diag[n - 1] = 0.0
        a2_diag[n - 1] = 1.0
        a3_diag[n - 1] = 0.0
        rl_rhs[n - 1] = 0.0

        v_new = tridiar(a1_diag, a2_diag, a3_diag, rl_rhs)
        res = np.linalg.norm(v_new - v_mix)
        v_mix[:] = v_new
        if res < 1e-6:
            break

    return v_no_mix, v_mix, c1, c2, arksc
