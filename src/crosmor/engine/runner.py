"""SimulationRunner orchestrating hydrodynamics, morphodynamics and time-stepping matching CROSMOR.F."""

import time as pytime
from typing import Callable, Optional, List
import numpy as np

from crosmor.models.config import SimulationConfig, WaveClass, SedimentProfilePoint
from crosmor.models.results import TimeStepResult, SimulationResults
from crosmor.numerics.interpolation import rintpl
from crosmor.hydrodynamics.wave_solver import build_grid, solve_wave_hydrodynamics
from crosmor.hydrodynamics.rayleigh import generate_rayleigh_classes
from crosmor.morphodynamics.morpho_solver import solve_morphodynamics
from crosmor.morphodynamics.sediment_props import compute_grain_distribution_params


class SimulationRunner:
    """Coordinates the time loop and physical modules of CROSMOR."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._is_paused = False
        self._is_cancelled = False

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False

    def cancel(self) -> None:
        self._is_cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def compute_adaptive_dt(self, h_offshore: float) -> float:
        """Compute time step dt (seconds) matching CROSMOR.F (lines 351-358) or fixed dt."""
        if self.config.morpho.dt_fixed is not None and self.config.morpho.dt_fixed > 0.0:
            return float(self.config.morpho.dt_fixed)

        factime = self.config.morpho.factime
        if 0.0 <= h_offshore <= 1.0:
            return factime * 3600.0
        elif 1.0 < h_offshore <= 2.0:
            return factime * 2400.0
        elif 2.0 < h_offshore <= 3.0:
            return factime * 1200.0
        elif 3.0 < h_offshore <= 3.5:
            return factime * 600.0
        elif 3.5 < h_offshore <= 4.0:
            return factime * 300.0
        elif 4.0 < h_offshore <= 4.5:
            return factime * 200.0
        else:
            return factime * 100.0

    def run(
        self,
        on_step_callback: Optional[Callable[[TimeStepResult, float], None]] = None,
    ) -> SimulationResults:
        """Run full simulation from t=0 to tend.

        Parameters:
            on_step_callback: Callable receiving (TimeStepResult, progress_0_to_1)

        Returns:
            SimulationResults container with all saved time steps.
        """
        config = self.config
        tend = config.morpho.tend

        # 1. Build spatial grid
        x_grid, hmsl_grid, nxi = build_grid(config)
        nx = len(x_grid)
        nxi = max(3, min(nx, nxi))

        # Bed level: in Fortran zbo is initial depth (zbo = -hmsl)
        zbo = -hmsl_grid
        zbn = np.copy(zbo)

        # Sediment fractions along profile
        nfr = config.nfr
        psi_grid = np.zeros((nfr, nx), dtype=np.float64)
        dsi = np.array(config.morpho.grain_diameters, dtype=np.float64)

        # Ensure default sediment points exist spanning the grid if none defined
        if not config.sediment_points:
            config.sediment_points = [
                SedimentProfilePoint(
                    x=float(x_grid[0]),
                    rc=0.0,
                    rw=0.0,
                    pmud=0.0,
                    frip=1.0,
                    fractions=[1.0 / nfr] * nfr,
                ),
                SedimentProfilePoint(
                    x=float(x_grid[-1]),
                    rc=0.0,
                    rw=0.0,
                    pmud=0.0,
                    frip=1.0,
                    fractions=[1.0 / nfr] * nfr,
                ),
            ]
        elif len(config.sediment_points) == 1:
            pt = config.sediment_points[0]
            fracs = list(pt.fractions) if pt.fractions else [1.0 / nfr] * nfr
            config.sediment_points = [
                SedimentProfilePoint(
                    x=float(x_grid[0]),
                    rc=pt.rc,
                    rw=pt.rw,
                    pmud=pt.pmud,
                    frip=pt.frip,
                    fractions=fracs,
                ),
                SedimentProfilePoint(
                    x=float(x_grid[-1]),
                    rc=pt.rc,
                    rw=pt.rw,
                    pmud=pt.pmud,
                    frip=pt.frip,
                    fractions=fracs,
                ),
            ]

        xs_arr = np.array([p.x for p in config.sediment_points], dtype=np.float64)
        for j in range(nfr):
            frac_j_arr = np.array(
                [p.fractions[j] if len(p.fractions) > j else (1.0 / nfr) for p in config.sediment_points],
                dtype=np.float64,
            )
            psi_grid[j, :] = np.interp(x_grid, xs_arr, frac_j_arr)

        d50_grid = np.zeros(nx, dtype=np.float64)
        d90_grid = np.zeros(nx, dtype=np.float64)
        for i in range(nx):
            d50_val, d90_val, _, _ = compute_grain_distribution_params(dsi, psi_grid[:, i])
            d50_grid[i] = d50_val
            d90_grid[i] = d90_val

        # Tidal arrays
        tide_t = np.array([pt.time for pt in config.tide_series], dtype=np.float64)
        tide_v1 = np.array([pt.vnul1 for pt in config.tide_series], dtype=np.float64)
        tide_v2 = np.array([pt.vnul2 for pt in config.tide_series], dtype=np.float64)
        tide_ws = np.array([pt.ws for pt in config.tide_series], dtype=np.float64)
        tijlen = tide_t[-1] if len(tide_t) > 0 and tide_t[-1] > 0 else 1.0

        # Storm surge arrays
        storm_t = np.array([pt.time for pt in config.storm_series], dtype=np.float64)
        storm_h = np.array([pt.hstorm for pt in config.storm_series], dtype=np.float64)
        tstorm_max = storm_t[-1] if len(storm_t) > 0 and storm_t[-1] > 0 else 1.0

        results = SimulationResults()
        time = 0.0
        step_idx = 0
        qcr0_accum = 0.0
        qcrxr_accum = 0.0
        last = False

        arksc_grid = np.zeros(nx, dtype=np.float64)
        arksw_grid = np.zeros(nx, dtype=np.float64)

        while time < tend and not self._is_cancelled:
            # Handle GUI pause
            while self._is_paused and not self._is_cancelled:
                pytime.sleep(0.05)

            step_idx += 1

            # Determine wave classes for current time
            if config.igolf in [2, -2]:
                wave_classes = generate_rayleigh_classes(time, config.nh_wave_classes, config.wave_series)
            else:
                wave_classes = config.wave_classes

            h_offshore = wave_classes[0].h
            dt = self.compute_adaptive_dt(h_offshore)

            if time + dt >= tend:
                dt = tend - time
                last = True

            # Boundary water levels and currents at t and t+dt
            tpol = time % tijlen
            tpols = time % tstorm_max
            vn01 = rintpl(tpol, tide_v1, tide_t)
            vn02 = rintpl(tpol, tide_v2, tide_t)
            htide = rintpl(tpol, tide_ws, tide_t)
            htold = htide
            hstorm = rintpl(tpols, storm_h, storm_t)

            # Recompute active cutoff nxi based on current bed elevation
            nxi = nx
            for i in range(nx):
                current_depth = -zbn[i] + hstorm + htide
                if current_depth <= config.hgrens:
                    nxi = i
                    break
            nxi = max(3, min(nx, nxi))

            # Hydrodynamics
            wave_profs, hydro_summary, water_depth, arksc_out, arksw_out = solve_wave_hydrodynamics(
                config=config,
                x_grid=x_grid,
                zbn=zbn,
                time=time,
                dt=dt,
                htide=htide,
                htold=htold,
                hstorm=hstorm,
                vnul1=vn01,
                vnul2=vn02,
                wave_classes=wave_classes,
                nxi=nxi,
                arksc_in=arksc_grid,
                arksw_in=arksw_grid,
            )
            arksc_grid[:nxi] = arksc_out
            arksw_grid[:nxi] = arksw_out

            # Morphodynamics
            (
                transport_summary,
                morpho_summary,
                vol_summary,
                zbn_new,
                psi_new,
                qcr0_accum,
                qcrxr_accum,
            ) = solve_morphodynamics(
                config=config,
                x_grid=x_grid,
                zbn=zbn,
                zbo=zbo,
                water_depth=water_depth,
                wave_profiles=wave_profs,
                wave_classes=wave_classes,
                nxi=nxi,
                dt=dt,
                time=time,
                qcr0_accum=qcr0_accum,
                qcrxr_accum=qcrxr_accum,
                psi_grid=psi_grid,
                d50_grid=d50_grid,
                d90_grid=d90_grid,
                arksc=arksc_grid,
                arksw=arksw_grid,
            )

            # Update state
            zbn = zbn_new
            psi_grid = psi_new
            for i in range(nx):
                d50_val, d90_val, _, _ = compute_grain_distribution_params(dsi, psi_grid[:, i])
                d50_grid[i] = d50_val
                d90_grid[i] = d90_val

            # Water surface elevation eta(x)
            # In water (i < nxi): eta = zbn + water_depth
            # On dry beach (i >= nxi): float('nan') so water line stops cleanly at shoreline
            eta = [float(zbn[i] + water_depth[i]) if i < nxi else float("nan") for i in range(nx)]

            step_res = TimeStepResult(
                time=time + dt,
                dt=dt,
                step_number=step_idx,
                nxi=nxi,
                x=[float(x) for x in x_grid],
                water_depth=[float(d) for d in water_depth],
                water_level_eta=eta,
                hydro=hydro_summary,
                transport=transport_summary,
                morpho=morpho_summary,
                volume=vol_summary,
                wave_classes=wave_profs if config.iprint == 2 else [],
            )

            results.time_steps.append(step_res)
            time += dt

            progress = min(1.0, time / tend) if tend > 0 else 1.0
            if on_step_callback:
                on_step_callback(step_res, progress)

            if last:
                break

        return results
