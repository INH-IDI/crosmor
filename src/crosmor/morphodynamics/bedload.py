"""Bedload sediment transport formulation matching CROSMOR BODEM.F."""

import math
from typing import Tuple, List, Optional
import numpy as np
from numba import njit


@njit(fastmath=True)
def compute_bedload_cell(
    vr: float,
    ur: float,
    ub: float,
    tp1: float,
    abw: float,
    a_ref: float,
    ubwback: float,
    ubwfor: float,
    rc: float,
    rw: float,
    ra: float,
    delm: float,
    ateta_rad: float,
    phi_rad: float,
    bsly: float,
    tanif: float,
    rhow: float,
    rhos: float,
    d50: float,
    taucr1: float,
    dster: float,
    facbed: float,
    kk1: int,
    kk2: int,
    dsi: np.ndarray,
    psi: np.ndarray,
    dsteri: np.ndarray,
    egfi: np.ndarray,
    fc1i_row: np.ndarray,
    fw1i_row: np.ndarray,
    fc1: float,
    fw1: float,
    fac1: float = 1.0,
    ntime: int = 201,
) -> Tuple[float, float, np.ndarray]:
    """Compute cross-shore and longshore bedload transport for all fractions matching BODEM.F.

    Returns:
        (sbx, sby, dm_by_fraction)
    """
    pi = math.pi
    ns = len(dsi)

    # Velocities at z = a
    if delm > 0.0:
        hulp30 = -1.0 + math.log(max(1e-4, 30.0 * 1.0 / ra))
        udelc = vr * math.log(max(1.01, 30.0 * delm / ra)) / hulp30
        udelw = ur * math.log(max(1.01, 30.0 * delm / ra)) / hulp30
        vra = udelc * math.log(max(1.01, 30.0 * a_ref / rc)) / math.log(max(1.01, 30.0 * delm / rc))
        ura = fac1 * udelw * math.log(max(1.01, 30.0 * a_ref / rc)) / math.log(max(1.01, 30.0 * delm / rc))
    else:
        vra = 0.0
        ura = 0.0

    if a_ref <= ra / 30.0:
        vra = 0.0
        ura = 0.0

    ubtot = ub + ura
    dtt = tp1 / ntime

    profact = 1.0
    acw1 = abs(vr) / (abs(ubwfor) + abs(ubwback) + abs(vr) + 1e-4)
    acw = acw1**1.0

    sbx_tot = 0.0
    sby_tot = 0.0
    dm = np.zeros(ns, dtype=np.float64)

    # Nielsen acceleration phase lead
    pangle = 40.0 if kk2 == 1 else 0.0
    plead1 = math.cos((pangle / 180.0) * pi)
    plead2 = math.sin((pangle / 180.0) * pi)

    # LOOP OVER FRACTIONS
    for is_idx in range(ns):
        d_val = dsi[is_idx]
        psi_val = psi[is_idx]
        dster_val = dsteri[is_idx]
        egfi_val = egfi[is_idx]

        tfor = (ubwback / (ubwfor + ubwback + 1e-6)) * tp1
        tback = max(1e-3, tp1 - tfor)
        d90 = 2.0 * d50

        fw1for = math.exp(-6.0 + 5.2 * ((abw / d90) ** -0.19)) if abw > 0 else 0.02
        fw1back = math.exp(-6.0 + 5.2 * ((abw / d90) ** -0.19)) if abw > 0 else 0.02
        fw1for = min(0.05, fw1for)
        fw1back = min(0.05, fw1back)

        fc11 = 0.25 * fc1i_row[is_idx] * profact
        fw1_base = fw1i_row[is_idx]

        sbx_frac = 0.0
        sby_frac = 0.0

        for nnn in range(ntime):
            time_inst = nnn * dtt

            # Instantaneous wave orbital velocity udt
            if time_inst < tfor:
                udt_base = ubwfor * math.sin(pi * time_inst / tfor)
                udt2 = ubwfor * math.sin(pi * (time_inst + dtt) / tfor)
                udt1 = ubwfor * math.sin(pi * (time_inst - dtt) / tfor)
            else:
                udt_base = -ubwback * math.sin(pi * (time_inst - tfor) / tback)
                udt2 = -ubwback * math.sin(pi * (time_inst + dtt - tfor) / tback)
                udt1 = -ubwback * math.sin(pi * (time_inst - dtt - tfor) / tback)

            dudt = (tp1 / (2.0 * pi)) * (udt2 - udt1) / (2.0 * dtt)
            udt = plead1 * udt_base + dudt * plead2

            # Wave + current velocity vectors
            udtx = udt * math.sin(ateta_rad)
            udty = udt * math.cos(ateta_rad)
            ubtotx = ubtot * math.sin(ateta_rad)
            ubtoty = ubtot * math.cos(ateta_rad)

            uxt = vra + udtx + ubtotx
            uyt = udty + ubtoty
            utvec = math.sqrt(uxt**2 + uyt**2)

            if utvec > 1e-6:
                bsls = (bsly * uyt) / utvec
            else:
                bsls = 0.0

            # Slope factor
            atan_tanif = math.atan(tanif)
            atan_bsls = math.atan(bsls)
            fsl2 = math.sin(atan_tanif + atan_bsls) / math.sin(atan_tanif)

            fcw1 = acw * fc11 + (1.0 - acw) * fw1_base
            tau1t = 0.5 * rhow * fcw1 * (utvec**2)

            # Shields excess parameter
            rrr2 = 0.8 + 0.2 * ((tau1t / taucr1 - 0.8) / 1.2)
            rrr2 = min(1.0, max(0.8, rrr2))

            rlamda = (d_val / d50) ** 0.25
            denom_cr = taucr1 * (d_val / d50)
            if abs(denom_cr) < 1e-6:
                denom_cr = 1e-6
            tt = rlamda * (tau1t - rrr2 * taucr1 * fsl2 * (d_val / d50) * egfi_val) / denom_cr
            tt = max(1.0e-7, tt)

            uster1t = math.sqrt(tau1t / rhow)
            dsand = 0.000062
            fsilti = max(1.0, dsand / d_val)

            denom_dster = dster_val**0.3
            if abs(denom_dster) < 1e-6:
                denom_dster = 1e-6
            sbt = 0.5 * fsilti * d_val * rhos * uster1t * tt / denom_dster * facbed

            if utvec > 1e-6:
                sbx_frac += (uxt / utvec) * sbt * psi_val
                sby_frac += (uyt / utvec) * sbt * psi_val

        # Average over wave period
        count_denom = float(ntime - 1)
        sbx_frac /= count_denom
        sby_frac /= count_denom

        # Slope factor in cross-shore direction
        fsl3 = 1.0
        if abs(sby_frac) > 0.0:
            bsls = bsly * sby_frac / abs(sby_frac)
            fsl3 = 1.0 / (1.0 + 1.0 * bsls / tanif)
            fsl3 = min(2.0, max(0.5, fsl3))

        sby_frac *= fsl3
        dm[is_idx] = sby_frac

        sbx_tot += sbx_frac
        sby_tot += sby_frac

    return sbx_tot, sby_tot, dm
