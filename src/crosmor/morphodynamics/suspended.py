"""Suspended sediment transport formulation matching CROSMOR SUSPEN.F."""

import math
from typing import Tuple, List
import numpy as np


def compute_suspended_cell(
    vr: float,
    ur: float,
    hs: float,
    hd: float,
    tp: float,
    ubw: float,
    ubwr: float,
    a_ref: float,
    rc: float,
    rw: float,
    ra: float,
    delm: float,
    ust: float,
    ca_ref: float,
    cai_fractions: np.ndarray,
    wsi: np.ndarray,
    dsi: np.ndarray,
    psi: np.ndarray,
    d50: float,
    rhos: float,
    rkap: float,
    ateta_rad: float,
    bsly: float,
    tanif: float,
    sef: float,
    faktor: float,
    fac1: float = 1.0,
    rmob: float = 100.0,
    ul_longwave: float = 0.0,
    nn_layers: int = 50,
) -> Tuple[float, float, float, np.ndarray, float]:
    """Integrate vertical concentration and velocity profiles matching SUSPEN.F.

    Returns:
        (ssc_longshore, ssw_cross_shore, ss_asym, dm_by_fraction, ca_actual)
    """
    ns = len(dsi)
    pi = math.pi
    zaa = max(1e-4, ra / 30.0)

    c10 = -1.0 + math.log(max(1.01, hd / zaa))
    alf1 = 0.5
    alf2 = 1.0
    c20 = math.log(max(1.01, alf1 * hd / zaa))
    c30 = -alf1 + alf1 * math.log(max(1.01, alf1 * hd / zaa))

    hulp30 = -1.0 + math.log(max(1.01, 30.0 * hd / ra))
    if delm > 0.0:
        udel = vr * math.log(max(1.01, 30.0 * delm / ra)) / hulp30
        udelw = ur * math.log(max(1.01, 30.0 * delm / ra)) / hulp30
    else:
        udel = 0.0
        udelw = 0.0

    ssc_total = 0.0
    ss_asym_total = 0.0
    dm_frac = np.zeros(ns, dtype=np.float64)
    somdm = 0.0

    # Grid in vertical z from a to hd (logarithmic distribution)
    z_levels = np.zeros(nn_layers, dtype=np.float64)
    for ij in range(nn_layers):
        frac_z = float(ij) / float(nn_layers - 1)
        z_levels[ij] = a_ref * ((hd / a_ref) ** frac_z)

    # LOOP OVER FRACTIONS
    for is_idx in range(ns):
        ca = cai_fractions[is_idx] if ns > 1 else ca_ref
        ws = wsi[is_idx]
        psi_val = psi[is_idx]

        # Wave mixing layer thickness DS
        delw = delm / 2.0
        gambr = 1.0
        if hs / hd > 0.4:
            gambr = 1.0 + math.sqrt(hs / hd - 0.4)
        ds_thick = min(0.2, max(0.05, 2.0 * gambr * delw))

        betawi = min(1.5, 1.0 + 2.0 * ((ws / max(1e-4, ust)) ** 2))
        fdamp = min(1.0, max(0.1, math.sqrt(250.0 / max(1.0, rmob))))
        ubwr2 = math.sqrt(ubwr**2 + ul_longwave**2)
        ebw = 0.018 * sef * fdamp * gambr * betawi * ds_thick * ubwr2

        if tp >= 1.0:
            emaxw = 0.035 * sef * gambr * hd * hs / tp
        else:
            emaxw = 0.0
        emaxw = min(0.05, max(emaxw, ebw))
        emaxc = 0.25 * sef * rkap * ust * hd * betawi

        # Velocity at z=a
        if a_ref < delm:
            uc_a = udel * math.log(max(1.01, 30.0 * a_ref / rc)) / math.log(max(1.01, 30.0 * delm / rc))
            ucw_a = udelw * math.log(max(1.01, 30.0 * a_ref / rc)) / math.log(max(1.01, 30.0 * delm / rc))
        else:
            uc_a = vr * math.log(max(1.01, 30.0 * a_ref / ra)) / hulp30
            ucw_a = ur * math.log(max(1.01, 30.0 * a_ref / ra)) / hulp30

        if a_ref <= rc / 30.0:
            uc_a = 0.0
            ucw_a = 0.0

        # Vertical integration of concentration
        c_prof = np.zeros(nn_layers, dtype=np.float64)
        c_prof[0] = ca

        for iz in range(1, nn_layers):
            z_mid = 0.5 * (z_levels[iz - 1] + z_levels[iz])
            dz = z_levels[iz] - z_levels[iz - 1]

            # Eddy diffusivity Es
            if z_mid <= ds_thick:
                esw = ebw
            elif z_mid <= 0.5 * hd:
                esw = ebw + (emaxw - ebw) * ((z_mid - ds_thick) / (0.5 * hd - ds_thick))
            else:
                esw = emaxw

            if z_mid < 0.5 * hd:
                esc = emaxc - emaxc * ((1.0 - 2.0 * z_mid / hd) ** 2)
            else:
                esc = emaxc

            es = math.sqrt(esw**2 + esc**2)
            es = max(1e-6, es)

            # Hindered settling & damping
            c_curr = max(1e-8, c_prof[iz - 1])
            cmax = 0.65
            fhs = max(0.01, (1.0 - 0.65 * c_curr / cmax) ** 5.0)
            fcc = -ws * c_curr * fhs / es

            # Integrate c to next level
            c_next = c_curr * math.exp(fcc * dz / c_curr) if c_curr > 0 else 0.0
            c_prof[iz] = max(1e-8, min(ca, c_next))

        # Integrate fluxes along vertical
        ssc_i = 0.0
        ssw_i = 0.0
        ssasym_i = 0.0

        ur1 = (c20 / max(1e-4, c10)) * ur * fac1

        for iz in range(1, nn_layers):
            z = z_levels[iz]
            dz = z_levels[iz] - z_levels[iz - 1]
            c_mid = 0.5 * (c_prof[iz] + c_prof[iz - 1]) * rhos * psi_val

            # Current velocity uc(z)
            if z < delm:
                uc = udel * math.log(max(1.01, 30.0 * z / rc)) / math.log(max(1.01, 30.0 * delm / rc))
                ucw = udelw * math.log(max(1.01, 30.0 * z / rc)) / math.log(max(1.01, 30.0 * delm / rc))
            else:
                uc = vr * math.log(max(1.01, 30.0 * z / ra)) / hulp30
                ucw = ur * math.log(max(1.01, 30.0 * z / ra)) / hulp30

            uz2 = ucw * fac1
            if z > alf1 * hd:
                uz2 = ur1 * (1.0 - ((z - alf1 * hd) / ((alf2 - alf1) * hd)) ** 3)
            if z > alf2 * hd:
                uz2 = 0.0

            ssc_i += c_mid * uc * dz
            ssw_i += c_mid * uz2 * dz
            if z <= 0.5:
                ssasym_i += c_mid * dz

        # Cross-shore transport of fraction
        cos_teta = math.cos(ateta_rad)
        dm_val = (ssw_i + faktor * ssasym_i) * cos_teta

        # Slope factor
        fsl3 = 1.0
        if abs(dm_val) > 0.0:
            bsls = bsly * dm_val / abs(dm_val)
            fsl3 = 1.0 / (1.0 + 1.0 * bsls / tanif)
            fsl3 = min(2.0, max(0.5, fsl3))

        dm_val *= fsl3
        dm_frac[is_idx] = dm_val

        ssc_total += ssc_i
        ss_asym_total += ssasym_i
        if abs(cos_teta) > 1e-4:
            somdm += dm_val / (fsl3 * cos_teta)

    return ssc_total, somdm, ss_asym_total, dm_frac, ca_ref
