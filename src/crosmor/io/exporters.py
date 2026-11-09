"""Export utilities for CROSMOR simulation projects and results."""

import json
from pathlib import Path
from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from crosmor.models.config import SimulationConfig
from crosmor.models.results import SimulationResults, TimeStepResult


def save_project_json(config: SimulationConfig, file_path: Union[str, Path]) -> None:
    """Save simulation configuration to modern JSON project format."""
    path = Path(file_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(config.model_dump_json(indent=2))


def load_project_json(file_path: Union[str, Path]) -> SimulationConfig:
    """Load simulation configuration from modern JSON project format."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SimulationConfig.model_validate(data)


def export_results_to_excel(results: SimulationResults, file_path: Union[str, Path]) -> None:
    """Export all simulation time steps and variables to an Excel workbook (.xlsx)."""
    path = Path(file_path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # 1. Summary of time steps
        ts_rows = []
        for step in results.time_steps:
            ts_rows.append(
                {
                    "step": step.step_number,
                    "time_s": step.time,
                    "dt_s": step.dt,
                    "balance_m2": step.volume.area_balance_m2,
                    "scross_x0_m2": step.volume.scross_x0_m2,
                    "scross_xr_m2": step.volume.scross_xr_m2,
                }
            )
        pd.DataFrame(ts_rows).to_excel(writer, sheet_name="Timeline", index=False)

        def _safe_col(arr, nx):
            if arr is not None:
                if len(arr) == nx:
                    return arr
                elif len(arr) < nx:
                    return list(arr) + [0.0] * (nx - len(arr))
            return [0.0] * nx

        # 2. Latest profile state
        if results.time_steps:
            last = results.time_steps[-1]
            nx_last = len(last.x)
            prof_df = pd.DataFrame(
                {
                    "x_m": last.x,
                    "z_initial_m": _safe_col(last.morpho.z_initial, nx_last),
                    "z_current_m": _safe_col(last.morpho.z_current, nx_last),
                    "dz_total_m": _safe_col(last.morpho.dz_total, nx_last),
                    "water_depth_m": _safe_col(last.water_depth, nx_last),
                    "hrms_m": _safe_col(last.hydro.hrms, nx_last),
                    "h13_m": _safe_col(last.hydro.h13, nx_last),
                    "v_longshore_mix_ms": _safe_col(last.hydro.v_mix, nx_last),
                    "st_cross_kg_s_m": _safe_col(last.transport.st_cross, nx_last),
                    "st_long_kg_s_m": _safe_col(last.transport.st_long, nx_last),
                    "c_vol_m3_m3": _safe_col(last.transport.c_vol or last.transport.ca_ref, nx_last),
                    "c_mean_vol_m3_m3": _safe_col(last.transport.c_mean_vol, nx_last),
                    "d50_m": _safe_col(last.morpho.d50, nx_last),
                }
            )
            prof_df.to_excel(writer, sheet_name="Final_Profile", index=False)

            # 3. Initial profile state
            first = results.time_steps[0]
            nx_first = len(first.x)
            first_df = pd.DataFrame(
                {
                    "x_m": first.x,
                    "z_initial_m": _safe_col(first.morpho.z_initial, nx_first),
                    "hrms_m": _safe_col(first.hydro.hrms, nx_first),
                    "h13_m": _safe_col(first.hydro.h13, nx_first),
                    "v_longshore_mix_ms": _safe_col(first.hydro.v_mix, nx_first),
                    "st_cross_kg_s_m": _safe_col(first.transport.st_cross, nx_first),
                    "st_long_kg_s_m": _safe_col(first.transport.st_long, nx_first),
                }
            )
            first_df.to_excel(writer, sheet_name="Initial_Profile", index=False)


def export_profile_to_csv(step: TimeStepResult, file_path: Union[str, Path]) -> None:
    """Export single time step cross-shore variables to CSV."""
    nx = len(step.x)

    def _col(arr):
        if arr is not None:
            if len(arr) == nx:
                return arr
            elif len(arr) < nx:
                return list(arr) + [0.0] * (nx - len(arr))
        return [0.0] * nx

    df = pd.DataFrame(
        {
            "x_m": step.x,
            "z_initial_m": _col(step.morpho.z_initial),
            "z_current_m": _col(step.morpho.z_current),
            "dz_total_m": _col(step.morpho.dz_total),
            "dz_step_m": _col(step.morpho.dz_step),
            "water_depth_m": _col(step.water_depth),
            "water_level_eta_m": _col(step.water_level_eta),
            "hrms_m": _col(step.hydro.hrms),
            "h13_m": _col(step.hydro.h13),
            "setup_m": _col(step.hydro.s_setup),
            "v_mix_ms": _col(step.hydro.v_mix),
            "v_no_mix_ms": _col(step.hydro.v_no_mix),
            "ucr_mean_ms": _col(step.hydro.ucr_mean),
            "st_cross_kg_s_m": _col(step.transport.st_cross),
            "sb_cross_kg_s_m": _col(step.transport.sb_cross),
            "ss_cross_kg_s_m": _col(step.transport.ss_cross),
            "st_long_kg_s_m": _col(step.transport.st_long),
            "sb_long_kg_s_m": _col(step.transport.sb_long),
            "ss_long_kg_s_m": _col(step.transport.ss_long),
            "c_vol_m3_m3": _col(step.transport.c_vol or step.transport.ca_ref),
            "c_mean_vol_m3_m3": _col(step.transport.c_mean_vol),
            "d50_m": _col(step.morpho.d50),
            "d90_m": _col(step.morpho.d90),
        }
    )
    df.to_csv(file_path, index=False)
