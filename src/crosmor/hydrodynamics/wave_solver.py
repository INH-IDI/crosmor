"""Cross-shore wave propagation, wave breaking, undertow, roller and streaming solver matching golf.f."""

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from crosmor.models.config import SimulationConfig, WaveClass
from crosmor.models.results import WaveClassProfile, HydroProfileSummary
from crosmor.numerics.interpolation import rintpl
from crosmor.numerics.ode_rk import rk_adaptive_step
from crosmor.numerics.smoothing import smooth
from crosmor.hydrodynamics.dispersion import solve_dispersion_wavelength
from crosmor.hydrodynamics.asymmetry import compute_orbital_asymmetry
from crosmor.hydrodynamics.statistics import compute_wave_statistics
from crosmor.hydrodynamics.longshore import compute_longshore_current


def build_grid(config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray, int]:
    """Discretize the cross-shore bathymetry profile into calculation nodes.

    Matches the grid generation loop in golf.f (lines 40-79).
    Returns:
        (x_grid, hmsl_grid, nxi)
        - x_grid: x coordinates (m)
        - hmsl_grid: initial bed depth to MSL (m, positive downward)
        - nxi: active cutoff index where water depth > hgrens
    """
    x_nodes = []
    h_nodes = []

    sections = config.bathymetry
    for j in range(len(sections) - 1):
        x0 = sections[j].x
        x1 = sections[j + 1].x
        nstap = sections[j].nstap if sections[j].nstap is not None else 10
        dx = (x1 - x0) / nstap
        steps = nstap + 1 if j == len(sections) - 2 else nstap

        for k in range(steps):
            x = x0 + k * dx
            x_nodes.append(x)
            # Linearly interpolate depth to MSL
            # In Fortran: hmsl is positive depth (hmsl = -hmsl)
            # In our BathymetrySection, hmsl is stored as in .inp (negative below MSL)
            h0 = -sections[j].hmsl
            h1 = -sections[j + 1].hmsl
            frac = k / nstap
            h = h0 + frac * (h1 - h0)
            h_nodes.append(h)

    x_arr = np.array(x_nodes, dtype=np.float64)
    h_arr = np.array(h_nodes, dtype=np.float64)

    # Initial cutoff
    nxi = len(x_arr)
    for i in range(len(x_arr)):
        if h_arr[i] <= config.hgrens:
            nxi = i
            break

    return x_arr, h_arr, nxi


def solve_wave_hydrodynamics(
    config: SimulationConfig,
    x_grid: np.ndarray,
    zbn: np.ndarray,
    time: float,
    dt: float,
    htide: float,
    htold: float,
    hstorm: float,
    vnul1: float,
    vnul2: float,
    wave_classes: List[WaveClass],
    nxi: int,
    arksc_in: Optional[np.ndarray] = None,
    arksw_in: Optional[np.ndarray] = None,
    afrip_in: Optional[np.ndarray] = None,
) -> Tuple[List[WaveClassProfile], HydroProfileSummary, np.ndarray, np.ndarray, np.ndarray]:
    """Run cross-shore wave propagation across all wave classes matching golf.f.

    Returns:
        (wave_profiles, hydro_summary, water_depth_hd, rksc, rksw)
    """
    pi = math.pi
    g = 9.81
    nh = len(wave_classes)
    nx = len(x_grid)

    # Fluid density from water temp and salinity
    te = config.morpho.water_temp
    sa = config.morpho.salinity
    cl = (sa - 0.03) / 1.805
    rhow = 1000.0 + 1.455 * cl - 0.0065 * ((te - 4.0 + 0.4 * cl) ** 2)
    rhos = config.morpho.rhos
    del_density = (rhos - rhow) / rhow

    # Breaker index interpolation table
    if nh == 1:
        gammaa = np.array([0.32, 0.37, 0.40, 0.41, 0.42])
    else:
        gammaa = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
    steepf = np.array([0.0, 0.15, 1.0, 3.0, 10.0])

    # Water depth along active grid
    hd_base = np.zeros(nxi, dtype=np.float64)
    for i in range(nxi):
        hd_base[i] = -zbn[i] + hstorm + htide
        if hd_base[i] < config.hgrens:
            hd_base[i] = config.hgrens

    hd0 = hd_base[0]

    # Bed slope
    tanbx = np.zeros(nxi, dtype=np.float64)
    for i in range(nxi):
        if i == 0:
            tanbx[0] = (zbn[1] - zbn[0]) / (x_grid[1] - x_grid[0])
        elif i == nxi - 1:
            tanbx[i] = (zbn[i] - zbn[i - 1]) / (x_grid[i] - x_grid[i - 1])
        else:
            tanbx[i] = (zbn[i + 1] - zbn[i - 1]) / (x_grid[i + 1] - x_grid[i - 1])

    # Sediment properties along grid
    xs_arr = np.array([p.x for p in config.sediment_points], dtype=np.float64)
    rc_arr = np.array([p.rc for p in config.sediment_points], dtype=np.float64)
    rw_arr = np.array([p.rw for p in config.sediment_points], dtype=np.float64)
    frip_arr = np.array([p.frip for p in config.sediment_points], dtype=np.float64)

    rksc_grid = np.zeros(nxi, dtype=np.float64)
    rksw_grid = np.zeros(nxi, dtype=np.float64)
    frip_grid = np.zeros(nxi, dtype=np.float64)
    for i in range(nxi):
        rksc_grid[i] = max(1e-5, rintpl(x_grid[i], rc_arr, xs_arr))
        rksw_grid[i] = max(1e-5, rintpl(x_grid[i], rw_arr, xs_arr))
        frip_grid[i] = rintpl(x_grid[i], frip_arr, xs_arr)

    d50_grid = np.full(nxi, config.morpho.grain_diameters[0] if config.morpho.grain_diameters else 0.0002)

    wave_profiles: List[WaveClassProfile] = []

    # Storage for multi-class integration
    h_all = np.zeros((nxi, nh), dtype=np.float64)
    t_all = np.zeros((nxi, nh), dtype=np.float64)
    s_all = np.zeros((nxi, nh), dtype=np.float64)
    qb_all = np.zeros((nxi, nh), dtype=np.float64)
    dbr_all = np.zeros((nxi, nh), dtype=np.float64)
    dbr_roller_all = np.zeros((nxi, nh), dtype=np.float64)
    l_all = np.zeros((nxi, nh), dtype=np.float64)
    teta_all = np.zeros((nxi, nh), dtype=np.float64)
    uwf_all = np.zeros((nxi, nh), dtype=np.float64)
    uwb_all = np.zeros((nxi, nh), dtype=np.float64)
    ur_all = np.zeros((nxi, nh), dtype=np.float64)
    ub_all = np.zeros((nxi, nh), dtype=np.float64)

    # LOOP OVER WAVE CLASSES
    for ic, wc in enumerate(wave_classes):
        prof = WaveClassProfile(
            class_index=ic,
            h0=wc.h,
            t0=wc.t,
            p=wc.p,
            teta0=wc.teta,
        )

        h0 = wc.h
        tet0_rad = (wc.teta / 180.0) * pi
        tp = wc.t
        y = np.array([h0 * h0, 0.0], dtype=np.float64)  # y[0]=H^2, y[1]=set-up S

        c0 = -1.0
        er = 0.0
        er_old = 0.0
        crel_old = 0.0
        dbr_old = 0.0
        teta_old = 0.0
        hs_rk = 0.0

        for itab in range(nxi):
            x = x_grid[itab]
            dx = (x_grid[itab] - x_grid[itab - 1]) if itab > 0 else (x_grid[1] - x_grid[0])
            hd = hd_base[itab] + y[1]
            hd = max(config.hgrens, hd)

            # 1. Wave dispersion & Snell's law
            rls, crel, trel, tetax, c0 = solve_dispersion_wavelength(tp, hd, tet0_rad, c0, vr=0.0, g=g)

            # Group velocity factor nfac
            kx = 2.0 * pi / rls
            kh = max(1e-4, kx * hd)
            nfac = 0.5 * (1.0 + 2.0 * kh / math.sinh(min(50.0, 2.0 * kh)))

            # Wave breaking limit ymax
            coef3 = 0.25
            x_minus = max(x_grid[0], x - coef3 * rls)
            hmsl_p = rintpl(x, -zbn[:nxi], x_grid[:nxi])
            hmsl_m = rintpl(x_minus, -zbn[:nxi], x_grid[:nxi])
            dxx = max(1e-3, 1.0 * coef3 * rls)
            tanbx_local = (hmsl_m - hmsl_p) / dxx
            steepx = tanbx_local * rls / math.sqrt(max(1e-4, y[0]))
            gammax = rintpl(steepx, gammaa, steepf)
            if config.hydro.gamma > 0.0:
                gammax = config.hydro.gamma

            coef4 = 1.0
            ymax_limit1 = (0.142 * rls * math.tanh(kh)) ** 2
            ymax_limit2 = (coef4 * gammax * hd) ** 2
            ymax = min(ymax_limit1, ymax_limit2)

            qb = 1.0 if y[0] > ymax else 0.0

            # Friction dissipation
            term_adel = math.sqrt(max(1e-4, y[0])) / math.sinh(kh) * (pi / (trel * 2.0 * pi))
            tarm = math.exp(min(-0.0809, -0.19 * math.log(max(1e-4, term_adel / rksw_grid[itab]))))
            fw = math.exp(-6.0 + 5.2 * tarm)
            a2 = 16.0 * (pi**2) / 3.0 * fw / (g * (trel**2) * (math.sinh(kh) ** 3))

            coef1 = 1.5
            a3 = 2.0 * coef1 if y[0] >= ymax else 0.0
            hdd = hd - y[1]

            a1 = math.cos(tetax) * nfac * crel * trel
            a4 = 0.125 * (nfac - 0.5 + nfac * (math.cos(tetax) ** 2))

            if itab == 0:
                da1dx = 0.0
                da4dx = 0.0
            else:
                da1dx = (a1 - a1_old) / dx
                da4dx = (a4 - a4_old) / dx

            a1_old = a1
            a4_old = a4

            # Energy dissipation
            dbr = max(0.0, 0.25 * coef1 * rhow * g * (y[0] - ymax) / trel)

            # Near-bed orbital asymmetry
            ubwf, ubwb, asym, udel1 = compute_orbital_asymmetry(
                config.hydro.switch1_asymmetry,
                math.sqrt(max(1e-4, y[0])),
                hd,
                rls,
                trel,
                g=g,
                hgrens=config.hgrens,
            )

            # Undertow & roller calculation
            hde = hd
            y11 = y[0]
            if itab > 1 and config.hydro.coef6 > 0.0:
                # Average wave height and depth
                prev_h = np.sqrt(np.maximum(1e-4, h_all[:itab, ic]))
                prev_lx = l_all[:itab, ic]
                smooth_h = smooth(x_grid[:itab], prev_lx, prev_h, itab - 1, config.hydro.coef6)
                y11 = smooth_h * smooth_h
                hde = smooth(x_grid[:itab], prev_lx, hd_base[:itab], itab - 1, config.hydro.coef6)

            htrough = hde * (0.95 - 0.35 * (math.sqrt(y[0]) / hd))
            qw_lin = 0.125 * math.sqrt(g) * y11 / math.sqrt(max(1e-4, hde))

            brol = config.hydro.roller_coef
            if itab <= 1:
                er_old = 0.0
                crel_old = crel
                dbr_old = 0.0
                teta_old = 0.0
            else:
                er_old = er
                crel_old = l_all[itab - 1, ic] / trel
                dbr_old = dbr_all[itab - 1, ic]
                teta_old = teta_all[itab - 1, ic]

            dr_old = 2.0 * 0.1 * g * er_old / max(1e-4, crel_old)
            der = -dr_old + (1.0 - brol) * dbr_old
            denom_er = 2.0 * crel * math.cos(tetax)
            if abs(denom_er) < 1e-4:
                denom_er = 1e-4
            er = (2.0 * er_old * math.cos(teta_old) * crel_old + der * dx) / denom_er

            hdhulp = min(1.5, 3.0 * config.hgrens)
            reduc3 = 1.0
            if hd <= hdhulp:
                reduc3 = (hd / hdhulp) ** 1.0

            denom_ut = rhow * crel * htrough
            if abs(denom_ut) < 1e-4:
                denom_ut = 1e-4
            ur = reduc3 * (-qw_lin / max(1e-4, htrough) - 2.0 * er / denom_ut)
            if ur < -0.75:
                ur = -0.75

            # Tide filling/emptying & external cross-shore current
            if dt > 0.0:
                ur += (x_grid[nxi - 1] - x) * (htide - htold) / hd / dt
            v2x = (hd0 / hd) * vnul2
            ur += v2x
            ur = frip_grid[itab] * ur

            # Streaming near bed Ub
            uspar = (((ubwf + ubwb) / 2.0) ** 2) / max(1e-4, rls / trel)
            rrr = trel * (((ubwf + ubwb) / 2.0) / (2.0 * pi)) / max(1e-5, rksw_grid[itab])
            if rrr <= 1.0:
                uhulp = -1.0
            elif rrr <= 100.0:
                uhulp = -1.0 + 0.875 * math.log10(rrr)
            else:
                uhulp = 0.75

            d50 = d50_grid[itab]
            ubore = 1.5 if d50 >= 0.002 else 0.0
            uswash = 0.001
            if hd <= 3.0 * config.hgrens:
                uswash = (1.0 - hd / (3.0 * config.hgrens)) * ubore
            uswash = min(1.5, max(0.001, uswash))
            if tanbx_local >= 0.33:
                uswash = 0.0

            ub = config.hydro.clh * uhulp * uspar + config.hydro.csw * uswash
            if ub <= 0.0:
                ub = 0.0001

            # Dissipation for longshore current (drold + dbrold * brol)
            dbr_roller = dr_old + dbr_old * brol

            # Save in arrays
            h_all[itab, ic] = y[0]
            t_all[itab, ic] = trel
            s_all[itab, ic] = y[1]
            qb_all[itab, ic] = qb
            dbr_all[itab, ic] = dbr
            dbr_roller_all[itab, ic] = dbr_roller
            l_all[itab, ic] = rls
            teta_all[itab, ic] = tetax
            uwf_all[itab, ic] = ubwf
            uwb_all[itab, ic] = ubwb
            ur_all[itab, ic] = ur
            ub_all[itab, ic] = ub

            prof.h.append(float(math.sqrt(max(1e-4, y[0]))))
            prof.s.append(float(y[1]))
            prof.rls.append(float(rls))
            prof.teta.append(float(tetax / pi * 180.0))
            prof.dbr.append(float(dbr))
            prof.qb.append(float(qb))
            prof.ubwf.append(float(ubwf))
            prof.ubwb.append(float(ubwb))
            prof.asym.append(float(asym))
            prof.ur.append(float(ur))
            prof.ub.append(float(ub))

            # Advance to next grid point via RK4
            if itab < nxi - 1:
                xe = x_grid[itab + 1]

                def fkt_ode(x_val: float, y_val: np.ndarray) -> np.ndarray:
                    z_deriv = np.zeros(2, dtype=np.float64)
                    term_y = (y_val[0] ** 1.5) if y_val[0] > 0.0 else 0.0
                    denom_a1 = a1 if abs(a1) > 1e-6 else 1e-6
                    z_deriv[0] = -y_val[0] / denom_a1 * da1dx - a2 / denom_a1 * term_y - a3 / denom_a1 * (y_val[0] - ymax)

                    s_eff = max(-0.1, y_val[1])
                    denom_depth = hdd + s_eff
                    if abs(denom_depth) < 1e-4:
                        denom_depth = 1e-4
                    z_deriv[1] = -a4 / denom_depth * z_deriv[0] - y_val[0] / denom_depth * da4dx + config.hydro.coefsr
                    return z_deriv

                y_next, hs_rk = rk_adaptive_step(
                    x,
                    y,
                    xe,
                    fkt_ode,
                    eps=0.01,
                    first_step=(itab == 0),
                    hs_saved=hs_rk,
                )
                # Cap height by shallow water limit
                y_next[0] = min(hd_base[itab + 1] ** 2, max(1e-6, y_next[0]))
                y = y_next

        wave_profiles.append(prof)

    # 3. Integrate statistics over all wave classes
    probs = np.array([wc.p for wc in wave_classes], dtype=np.float64)
    probs /= np.sum(probs)

    summary = HydroProfileSummary()

    hrms_arr = np.zeros(nxi, dtype=np.float64)
    trms_arr = np.zeros(nxi, dtype=np.float64)
    h13_arr = np.zeros(nxi, dtype=np.float64)
    t13_arr = np.zeros(nxi, dtype=np.float64)
    h110_arr = np.zeros(nxi, dtype=np.float64)
    setup_arr = np.zeros(nxi, dtype=np.float64)
    qb_arr = np.zeros(nxi, dtype=np.float64)
    dbr_arr = np.zeros(nxi, dtype=np.float64)
    dbr_roller_sum = np.zeros(nxi, dtype=np.float64)
    l_sum = np.zeros(nxi, dtype=np.float64)
    teta_sum = np.zeros(nxi, dtype=np.float64)
    uwf_rms_arr = np.zeros(nxi, dtype=np.float64)
    uwb_rms_arr = np.zeros(nxi, dtype=np.float64)
    ucr_mean_arr = np.zeros(nxi, dtype=np.float64)
    ub_mean_arr = np.zeros(nxi, dtype=np.float64)
    ucr_rms_arr = np.zeros(nxi, dtype=np.float64)
    ub_rms_arr = np.zeros(nxi, dtype=np.float64)

    for j in range(nxi):
        h_vals = np.sqrt(np.maximum(1e-6, h_all[j, :]))
        t_vals = t_all[j, :]
        uwf_vals = uwf_all[j, :]
        uwb_vals = uwb_all[j, :]
        ur_vals = ur_all[j, :]
        ub_vals = ub_all[j, :]

        # Stats matching STAT.F
        h3, t3, h10, t10, uwf3, uwb3, urr3, ubb3 = compute_wave_statistics(
            probs, t_vals, h_vals, uwf_vals, uwb_vals, ur_vals, ub_vals
        )

        hrms_val = math.sqrt(np.sum(probs * (h_vals**2)))
        trms_val = math.sqrt(np.sum(probs * (t_vals**2)))
        s_val = float(np.sum(probs * s_all[j, :]))
        qb_val = float(np.sum(probs * qb_all[j, :]))
        dbr_val = float(np.sum(probs * dbr_all[j, :]))
        dbr_roller_val = float(np.sum(probs * dbr_roller_all[j, :]))
        l_val = float(np.sum(probs * l_all[j, :]))
        teta_val = float(np.sum(probs * teta_all[j, :]))

        uwf_rms = math.sqrt(np.sum(probs * (uwf_vals**2)))
        uwb_rms = math.sqrt(np.sum(probs * (uwb_vals**2)))
        ucr_rms = -math.sqrt(np.sum(probs * (ur_vals**2)))
        ub_rms = math.sqrt(np.sum(probs * (ub_vals**2)))

        hrms_arr[j] = hrms_val
        trms_arr[j] = trms_val
        h13_arr[j] = h3
        t13_arr[j] = t3
        h110_arr[j] = h10
        setup_arr[j] = s_val
        qb_arr[j] = qb_val
        dbr_arr[j] = dbr_val
        dbr_roller_sum[j] = dbr_roller_val
        l_sum[j] = l_val
        teta_sum[j] = teta_val
        uwf_rms_arr[j] = uwf_rms
        uwb_rms_arr[j] = uwb_rms
        ucr_mean_arr[j] = urr3
        ub_mean_arr[j] = ubb3
        ucr_rms_arr[j] = ucr_rms
        ub_rms_arr[j] = ub_rms

    # Longshore current via SNELH.F
    # In golf.f (lines 820-827), dissipation passed is dbr_roller_sum (result(1, ivg+24))
    v_no_mix, v_mix, c1_chezy, c2_chezy, arksc_out = compute_longshore_current(
        x_grid[:nxi],
        h13_arr,
        t13_arr,
        hd_base[:nxi],
        setup_arr,
        l_sum,
        teta_sum,
        dbr_roller_sum,
        d50_grid,
        vnul1,
        rhow,
        del_density,
        wave_classes[0].w10,
        config.hydro.els,
        config.hydro.nit,
        config.hydro.coef5,
        config.hydro.sef1,
        config.hgrens,
        arksc_in=rksc_grid,
        g=g,
    )

    summary.hrms = [float(x) for x in hrms_arr]
    summary.trms = [float(x) for x in trms_arr]
    summary.h13 = [float(x) for x in h13_arr]
    summary.t13 = [float(x) for x in t13_arr]
    summary.h110 = [float(x) for x in h110_arr]
    summary.hmax = [float(x) for x in h110_arr]  # representative
    summary.s_setup = [float(x) for x in setup_arr]
    summary.qb_total = [float(x) for x in qb_arr]
    summary.dbr_total = [float(x) for x in dbr_arr]
    summary.v_no_mix = [float(x) for x in v_no_mix]
    summary.v_mix = [float(x) for x in v_mix]
    summary.c1_chezy = [float(x) for x in c1_chezy]
    summary.c2_chezy = [float(x) for x in c2_chezy]
    summary.ubwf_rms = [float(x) for x in uwf_rms_arr]
    summary.ubwb_rms = [float(x) for x in uwb_rms_arr]
    summary.ucr_mean = [float(x) for x in ucr_mean_arr]
    summary.ub_mean = [float(x) for x in ub_mean_arr]
    summary.ucr_rms = [float(x) for x in ucr_rms_arr]
    summary.ub_rms = [float(x) for x in ub_rms_arr]

    water_depth = hd_base[:nxi] + setup_arr

    return wave_profiles, summary, water_depth, arksc_out, rksw_grid
