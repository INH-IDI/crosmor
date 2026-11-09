"""Cross-shore sediment transport, bed level update, sliding and mass conservation matching TRANSP.F."""

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from crosmor.models.config import SimulationConfig, WaveClass
from crosmor.models.results import (
    TransportProfileSummary,
    MorphoProfileSummary,
    VolumeBalanceSummary,
    WaveClassProfile,
)
from crosmor.numerics.tridiagonal import tridiar
from crosmor.numerics.smoothing import smoop
from crosmor.morphodynamics.sediment_props import (
    compute_fluid_properties,
    compute_settling_velocity,
    compute_shields_critical,
    compute_grain_distribution_params,
)
from crosmor.morphodynamics.bedload import compute_bedload_cell
from crosmor.morphodynamics.suspended import compute_suspended_cell
from crosmor.morphodynamics.beach import compute_beach_profile_update


def solve_morphodynamics(
    config: SimulationConfig,
    x_grid: np.ndarray,
    zbn: np.ndarray,
    zbo: np.ndarray,
    water_depth: np.ndarray,
    wave_profiles: List[WaveClassProfile],
    wave_classes: List[WaveClass],
    nxi: int,
    dt: float,
    time: float,
    qcr0_accum: float,
    qcrxr_accum: float,
    psi_grid: np.ndarray,
    d50_grid: np.ndarray,
    d90_grid: np.ndarray,
    arksc: np.ndarray,
    arksw: np.ndarray,
) -> Tuple[
    TransportProfileSummary,
    MorphoProfileSummary,
    VolumeBalanceSummary,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """Execute sediment transport and bed evolution for current time step matching TRANSP.F.

    Returns:
        (transport_summary, morpho_summary, volume_summary, zbn_new, psi_new, qcr0_new, qcrxr_new)
    """
    pi = math.pi
    g = 9.81
    nx = len(x_grid)
    nfr = config.nfr
    rhos = config.morpho.rhos
    por = config.morpho.por
    dl = config.morpho.dl
    tanif = 0.6
    rhow, rnu, del_density = compute_fluid_properties(config.morpho.water_temp, config.morpho.salinity)

    # Initialize transport accumulation
    sbcr = np.zeros(nxi, dtype=np.float64)
    sscr = np.zeros(nxi, dtype=np.float64)
    stcr = np.zeros(nxi, dtype=np.float64)
    sbln = np.zeros(nxi, dtype=np.float64)
    ssln = np.zeros(nxi, dtype=np.float64)
    stln = np.zeros(nxi, dtype=np.float64)
    capr = np.zeros(nxi, dtype=np.float64)

    dmt = np.zeros((nfr, nxi), dtype=np.float64)

    # Sediment metadata along grid
    xs_arr = np.array([p.x for p in config.sediment_points], dtype=np.float64)
    pmud_arr = np.array([p.pmud for p in config.sediment_points], dtype=np.float64)
    pmud_grid = np.interp(x_grid[:nxi], xs_arr, pmud_arr)

    dsi = np.array(config.morpho.grain_diameters, dtype=np.float64)

    # Cross-shore bed slope bsly
    bsly = np.zeros(nxi, dtype=np.float64)
    for i in range(nxi):
        if i == 0:
            bsly[0] = -(zbn[1] - zbn[0]) / (x_grid[1] - x_grid[0])
        elif i == nxi - 1:
            bsly[i] = -(zbn[i] - zbn[i - 1]) / (x_grid[i] - x_grid[i - 1])
        else:
            bsly[i] = -(zbn[i + 1] - zbn[i - 1]) / (x_grid[i + 1] - x_grid[i - 1])

    probs = np.array([wc.p for wc in wave_classes], dtype=np.float64)
    probs /= np.sum(probs)

    # Precalculate settling velocities & Shields for each fraction
    wsi = np.zeros(nfr, dtype=np.float64)
    dsteri = np.zeros(nfr, dtype=np.float64)
    taucri = np.zeros(nfr, dtype=np.float64)
    for j in range(nfr):
        wsi[j] = compute_settling_velocity(dsi[j], del_density, rnu, g)
        dster_val, taucr_val, _ = compute_shields_critical(dsi[j], 0.0, del_density, rnu, rhow, rhos, g)
        dsteri[j] = dster_val
        taucri[j] = taucr_val

    # Friction factors
    fc1i_mat = np.zeros((nfr, nxi), dtype=np.float64)
    fw1i_mat = np.zeros((nfr, nxi), dtype=np.float64)
    for i in range(nxi):
        hd_val = max(0.1, water_depth[i])
        d90_val = max(1e-4, d90_grid[i])
        for j in range(nfr):
            fc1i_mat[j, i] = 0.24 * (math.log10(max(1.01, 12.0 * hd_val / d90_val)) ** -2)
            abw_val = 0.5  # representative
            fw1i_mat[j, i] = min(0.05, math.exp(-6.0 + 5.2 * ((max(1e-3, abw_val) / d90_val) ** -0.19)))

    # LOOP OVER WAVE CLASSES
    for ic, wc in enumerate(wave_classes):
        p_class = probs[ic]
        prof = wave_profiles[ic]

        for ix in range(nxi):
            hd = max(0.1, water_depth[ix])
            d50 = max(1e-5, d50_grid[ix])
            d90 = max(1e-5, d90_grid[ix])
            taucr1 = taucri[0]

            ateta_rad = math.radians(prof.teta[ix])
            ubwfor = prof.ubwf[ix]
            ubwback = prof.ubwb[ix]
            ur = prof.ur[ix]
            ub = prof.ub[ix]
            hs_val = prof.h[ix]
            tp1 = wc.t
            rls_val = prof.rls[ix]
            vr_val = 0.0  # longshore

            kh = max(1e-4, 2.0 * pi / rls_val * hd)
            sinh_kh = math.sinh(min(50.0, kh))
            ubw = pi * hs_val / (tp1 * sinh_kh) if sinh_kh > 0 else 0.0
            abw = ubw * tp1 / (2.0 * pi)

            rc_val = arksc[ix]
            rw_val = arksw[ix]
            ra_val = 10.0 * rc_val
            delm = max(rc_val, 0.36 * abw * ((max(1e-3, abw) / max(1e-4, rw_val)) ** -0.25)) if abw > 0 else rc_val
            a_ref = max(0.01, 0.5 * max(rc_val, rw_val))

            # Shields critical
            dster_val, taucr_val, _ = compute_shields_critical(d50, pmud_grid[ix], del_density, rnu, rhow, rhos, g)

            # Concentration Ca
            ust_val = math.sqrt(g) * abs(vr_val) / 30.0
            ta_val = max(1e-5, (0.5 * rhow * 0.02 * (ubw**2) - taucr_val) / max(1e-4, taucr_val))
            ca_ref = min(0.05, config.morpho.facsusc * 0.015 * (d50 / a_ref) * (dster_val**-0.3) * (ta_val**1.5))
            cai = np.full(nfr, ca_ref)

            # Hiding factor egfi
            _, _, _, egfi_local = compute_grain_distribution_params(dsi, psi_grid[:, ix])

            # 1. Bedload
            sbx, sby, dm_b = compute_bedload_cell(
                vr=vr_val,
                ur=ur,
                ub=ub,
                tp1=tp1,
                abw=abw,
                a_ref=a_ref,
                ubwback=ubwback,
                ubwfor=ubwfor,
                rc=rc_val,
                rw=rw_val,
                ra=ra_val,
                delm=delm,
                ateta_rad=ateta_rad,
                phi_rad=0.5 * pi - ateta_rad,
                bsly=bsly[ix],
                tanif=tanif,
                rhow=rhow,
                rhos=rhos,
                d50=d50,
                taucr1=taucr_val,
                dster=dster_val,
                facbed=config.morpho.facbed,
                kk1=config.hydro.switch1_asymmetry,
                kk2=config.hydro.switch2_nielsen,
                dsi=dsi,
                psi=psi_grid[:, ix],
                dsteri=dsteri,
                egfi=egfi_local,
                fc1i_row=fc1i_mat[:, ix],
                fw1i_row=fw1i_mat[:, ix],
                fc1=fc1i_mat[0, ix],
                fw1=fw1i_mat[0, ix],
            )

            # 2. Suspended load
            faktor = config.morpho.facsusw * 0.1 * (((ubwfor**4 - ubwback**4) / (ubwfor**3 + ubwback**3 + 1e-6)) + ub)
            ssc, ssw, ssasym, dm_s, ca_act = compute_suspended_cell(
                vr=vr_val,
                ur=ur,
                hs=hs_val,
                hd=hd,
                tp=tp1,
                ubw=ubw,
                ubwr=math.sqrt(0.5 * (ubwfor**3) + 0.5 * (ubwback**3)),
                a_ref=a_ref,
                rc=rc_val,
                rw=rw_val,
                ra=ra_val,
                delm=delm,
                ust=max(1e-4, ust_val),
                ca_ref=ca_ref,
                cai_fractions=cai,
                wsi=wsi,
                dsi=dsi,
                psi=psi_grid[:, ix],
                d50=d50,
                rhos=rhos,
                rkap=0.4,
                ateta_rad=ateta_rad,
                bsly=bsly[ix],
                tanif=tanif,
                sef=config.hydro.sef1,
                faktor=faktor,
            )

            # Vectors
            ssx = ssw * math.sin(ateta_rad) + ssc
            ssy = ssw * math.cos(ateta_rad)
            if abs(ssy) > 0.0:
                bsls = bsly[ix] * ssy / abs(ssy)
                fsl3 = min(2.0, max(0.5, 1.0 / (1.0 + 1.0 * bsls / tanif)))
                ssy *= fsl3

            stx = ssx + sbx
            sty = ssy + sby

            # Accumulate across wave classes
            sbcr[ix] += p_class * sby
            sscr[ix] += p_class * ssy
            stcr[ix] += p_class * sty
            sbln[ix] += p_class * sbx
            ssln[ix] += p_class * ssx
            stln[ix] += p_class * stx
            capr[ix] += p_class * ca_act

            # Fractions transport
            for j in range(nfr):
                dmt[j, ix] += (dm_s[j] + dm_b[j]) * p_class

    # Boundary mass influx
    qcr0_new = qcr0_accum + stcr[0] * dt
    qcrxr_new = qcrxr_accum + stcr[nxi - 1] * dt

    # 3. Bed level changes dzb from transport divergence
    # Spatial smoothing of dmt over 5 points matching TRANSP.F (lines 698-723)
    dmt_smooth = np.copy(dmt)
    dms = np.zeros((nfr, nxi), dtype=np.float64)

    for j in range(nfr):
        for i in range(1, nxi - 1):
            if 1 < i < nxi - 2:
                dmt_smooth[j, i] = 0.2 * (
                    dmt[j, i - 2] + dmt[j, i - 1] + dmt[j, i] + dmt[j, i + 1] + dmt[j, i + 2]
                )
                dms[j, i] = -0.5 * (dmt_smooth[j, i + 2] - dmt_smooth[j, i - 2]) * dt
            else:
                dmt_smooth[j, i] = (dmt[j, i - 1] + dmt[j, i] + dmt[j, i + 1]) / 3.0
                dms[j, i] = -0.5 * (dmt_smooth[j, i + 1] - dmt_smooth[j, i - 1]) * dt

        dms[j, 0] = dms[j, 2] if nxi > 2 else dms[j, 1]
        dms[j, 1] = dms[j, 2] if nxi > 2 else dms[j, 0]
        dms[j, nxi - 1] = dms[j, nxi - 2]

    # Total bed level change across fractions
    dmt_tot = np.sum(dms, axis=0)
    dzb = np.zeros(nxi, dtype=np.float64)

    for i in range(1, nxi - 1):
        pmud = pmud_grid[i] / 100.0 if pmud_grid[i] > 1.0 else pmud_grid[i]
        pmud = min(0.9, max(0.0, pmud))
        denom_mud = 1.0 - pmud
        dx_span = (x_grid[i + 2] - x_grid[i - 2]) if (1 < i < nxi - 2) else (x_grid[i + 1] - x_grid[i - 1])
        denom = max(1e-3, 0.5 * dx_span * (1.0 - por) * rhos)
        dzb[i] = (1.0 / denom_mud) * (dmt_tot[i] / denom)

    dzb[nxi - 1] = (1.0 / max(0.1, 1.0 - pmud_grid[nxi - 1] / 100.0)) * (
        dmt_tot[nxi - 1] / max(1e-3, (x_grid[nxi - 1] - x_grid[nxi - 2]) * (1.0 - por) * rhos)
    )

    # 4. Implicit tridiagonal smoothing matching TRANSP.F (lines 830-856)
    gamb = config.morpho.facbed_smooth * 0.005
    odi = np.zeros(nxi, dtype=np.float64)
    hdi = np.ones(nxi, dtype=np.float64)
    udi = np.zeros(nxi, dtype=np.float64)
    rl_diag = np.copy(zbn[:nxi])

    hdi[0] = 1.0
    udi[0] = 0.0
    rl_diag[0] = zbn[0]

    for i in range(1, nxi - 1):
        sm1 = max(1e-3, x_grid[i + 1] - x_grid[i])
        sm2 = max(1e-3, x_grid[i] - x_grid[i - 1])
        sm3 = max(1e-3, x_grid[i] + 0.5 * sm1 - (x_grid[i - 1] + 0.5 * sm2))
        gambx = min(1.0, 5.0 * abs(dzb[i]))
        ganb = max(gamb, gambx)

        odi[i - 1] = -ganb / (sm2 * sm3)
        hdi[i] = 1.0 + ganb * (1.0 / (sm1 * sm3) + 1.0 / (sm2 * sm3))
        udi[i] = -ganb / (sm1 * sm3)
        rl_diag[i] = zbn[i] + dzb[i]

    odi[nxi - 1] = 0.0
    hdi[nxi - 1] = 1.0
    udi[nxi - 1] = 0.0
    rl_diag[nxi - 1] = zbn[nxi - 1]

    zbn_smoothed = tridiar(odi, hdi, udi, rl_diag)
    dzb = zbn_smoothed - zbn[:nxi]

    # Smooth fraction percentages
    gamp = config.morpho.facgrain_smooth * 0.2
    psi_smoothed = smoop(psi_grid[:, :nxi], nxi, nfr, gamp, x_grid[:nxi])

    # 5. Beach module
    zbn_new = np.copy(zbn)
    zbn_new[:nxi] = zbn[:nxi] + dzb

    stlgr = stcr[nxi - 1]
    zm_beach = 0.0
    if config.istrand == 1:
        zbn_new, _, _, zm_beach = compute_beach_profile_update(
            stlgr=stlgr,
            dt=dt,
            por=por,
            rhos=rhos,
            xggr=x_grid[nxi - 1],
            x_grid=x_grid,
            hdg=water_depth,
            nxx=nxi,
            nx=nx,
            zbn=zbn_new,
            sl_degrees=config.hydro.sl_dune_slope,
            d50=d50_grid[0],
            hs0=wave_classes[0].h,
            ls0=wave_profiles[0].rls[0],
            tanbx=float((zbn[1] - zbn[0]) / (x_grid[1] - x_grid[0])),
        )

    # 6. Mass conservation correction Anum matching TRANSP.F (lines 900-928)
    sumdz = 0.0
    samdz = 0.0
    for i in range(1, nxi):
        diff = 0.5 * (zbn_new[i] - zbo[i] + zbn_new[i - 1] - zbo[i - 1])
        dx_val = x_grid[i] - x_grid[i - 1]
        sumdz += diff * dx_val
        samdz += abs(0.5 * dzb[i] + 0.5 * dzb[i - 1]) * dx_val

    anum = sumdz - qcr0_new / ((1.0 - por) * rhos)
    if config.istrand == 0 or abs(zm_beach) < 1e-6:
        anum += qcrxr_new / ((1.0 - por) * rhos)

    if samdz > 0.0:
        vold = 0.0
        for i in range(1, nxi):
            dai = abs(0.5 * dzb[i] + 0.5 * dzb[i - 1]) * (x_grid[i] - x_grid[i - 1])
            daveri = (dai / samdz) * anum
            dzbcor = 2.0 * daveri / (x_grid[i] - x_grid[i - 1]) - vold
            vold = dzbcor
            zbn_new[i] -= dzbcor

    # Fixed boundary depths
    zbn_new[0] = zbo[0]
    zbn_new[-1] = zbo[-1]

    # 7. Sliding procedure for dune avalanche (tanbeta = tan(SL))
    tanbeta = math.tan(math.radians(config.hydro.sl_dune_slope))
    for _ in range(50):
        sum_as = 0.0
        z_pre = np.copy(zbn_new)
        for i in range(3, nx - 2):
            dx_node = x_grid[i + 1] - x_grid[i]
            if dx_node > 0:
                slope = (zbn_new[i + 1] - zbn_new[i]) / dx_node
                if slope > tanbeta:
                    zbn_new[i + 1] = zbn_new[i] + dx_node * tanbeta
                    aslide = 0.5 * (z_pre[i + 1] - zbn_new[i + 1]) * (x_grid[i + 2] - x_grid[i])
                    denom_slide = max(1e-3, (x_grid[i] + x_grid[i - 1] - x_grid[i - 3] - x_grid[i - 4]))
                    zbn_new[i - 1] += 2.0 * aslide / denom_slide
                    zbn_new[i - 2] += 2.0 * aslide / denom_slide
                    zbn_new[i - 3] += 2.0 * aslide / denom_slide
                    sum_as += aslide
        if sum_as <= 1e-5:
            break

    # Integrated longshore transport
    int_sb_long = np.zeros(nxi, dtype=np.float64)
    int_ss_long = np.zeros(nxi, dtype=np.float64)
    int_st_long = np.zeros(nxi, dtype=np.float64)
    for i in range(nxi):
        dx_val = x_grid[i] - (x_grid[i - 1] if i > 0 else x_grid[0])
        int_sb_long[i] = (int_sb_long[i - 1] if i > 0 else 0.0) + sbln[i] * dx_val
        int_ss_long[i] = (int_ss_long[i - 1] if i > 0 else 0.0) + ssln[i] * dx_val
        int_st_long[i] = (int_st_long[i - 1] if i > 0 else 0.0) + stln[i] * dx_val

    # Area balance
    area_balance = float(np.sum(0.5 * (zbn_new[1:] - zbo[1:] + zbn_new[:-1] - zbo[:-1]) * np.diff(x_grid)))

    transport_summary = TransportProfileSummary(
        sb_cross=[float(x) for x in sbcr],
        ss_cross=[float(x) for x in sscr],
        st_cross=[float(x) for x in stcr],
        sb_long=[float(x) for x in sbln],
        ss_long=[float(x) for x in ssln],
        st_long=[float(x) for x in stln],
        int_sb_long=[float(x) for x in int_sb_long],
        int_ss_long=[float(x) for x in int_ss_long],
        int_st_long=[float(x) for x in int_st_long],
        ca_ref=[float(x) for x in capr],
    )

    morpho_summary = MorphoProfileSummary(
        dz_step=[float(x) for x in (zbn_new[:nxi] - zbn[:nxi])],
        dz_total=[float(x) for x in (zbn_new[:nxi] - zbo[:nxi])],
        z_initial=[float(x) for x in zbo[:nxi]],
        z_current=[float(x) for x in zbn_new[:nxi]],
        d50=[float(x) for x in d50_grid[:nxi]],
        d90=[float(x) for x in d90_grid[:nxi]],
        fractions=[[float(psi_smoothed[j, i]) for j in range(nfr)] for i in range(nxi)],
    )

    volume_summary = VolumeBalanceSummary(
        area_balance_m2=area_balance,
        scross_x0_m2=float(qcr0_new / ((1.0 - por) * rhos)),
        scross_xr_m2=float(qcrxr_new / ((1.0 - por) * rhos)),
    )

    # Full grid psi update
    psi_full = np.copy(psi_grid)
    psi_full[:, :nxi] = psi_smoothed

    return (
        transport_summary,
        morpho_summary,
        volume_summary,
        zbn_new,
        psi_full,
        qcr0_new,
        qcrxr_new,
    )
