"""Data models for CROSMOR simulation outputs and time step states."""

from typing import List, Dict, Optional
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class WaveClassProfile(BaseModel):
    """Hydrodynamic variables along cross-shore profile for a single wave class."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class_index: int
    h0: float
    t0: float
    p: float
    teta0: float
    h: List[float] = Field(default_factory=list, description="Wave height along profile (m)")
    s: List[float] = Field(default_factory=list, description="Wave set-up / set-down (m)")
    rls: List[float] = Field(default_factory=list, description="Local wave length (m)")
    teta: List[float] = Field(default_factory=list, description="Wave incidence angle (degrees)")
    dbr: List[float] = Field(default_factory=list, description="Wave energy dissipation (N/(m*s))")
    qb: List[float] = Field(default_factory=list, description="Breaking wave flag / fraction")
    ubwf: List[float] = Field(default_factory=list, description="Peak orbital forward velocity near bed (m/s)")
    ubwb: List[float] = Field(default_factory=list, description="Peak orbital backward velocity near bed (m/s)")
    asym: List[float] = Field(default_factory=list, description="Orbital velocity asymmetry factor")
    ur: List[float] = Field(default_factory=list, description="Undertow current (m/s)")
    ub: List[float] = Field(default_factory=list, description="Longuet-Higgins streaming current (m/s)")


class HydroProfileSummary(BaseModel):
    """Integrated/weighted hydrodynamic variables across all wave classes."""
    hrms: List[float] = Field(default_factory=list, description="RMS wave height (m)")
    h13: List[float] = Field(default_factory=list, description="Significant wave height H1/3 (m)")
    h110: List[float] = Field(default_factory=list, description="Mean of highest 10% waves H1/10 (m)")
    hmax: List[float] = Field(default_factory=list, description="Maximum wave height (m)")
    trms: List[float] = Field(default_factory=list, description="RMS wave period (s)")
    t13: List[float] = Field(default_factory=list, description="Significant wave period T1/3 (s)")
    s_setup: List[float] = Field(default_factory=list, description="Wave set-up (m)")
    qb_total: List[float] = Field(default_factory=list, description="Total fraction of breaking waves")
    dbr_total: List[float] = Field(default_factory=list, description="Total wave energy dissipation")
    v_no_mix: List[float] = Field(default_factory=list, description="Longshore current without mixing (m/s)")
    v_mix: List[float] = Field(default_factory=list, description="Longshore current with lateral mixing (m/s)")
    c1_chezy: List[float] = Field(default_factory=list, description="Chezy apparent roughness coefficient")
    c2_chezy: List[float] = Field(default_factory=list, description="Chezy physical roughness coefficient")
    ubwf_rms: List[float] = Field(default_factory=list, description="RMS orbital forward velocity (m/s)")
    ubwb_rms: List[float] = Field(default_factory=list, description="RMS orbital backward velocity (m/s)")
    ucr_mean: List[float] = Field(default_factory=list, description="Mean cross-shore current / undertow (m/s)")
    ub_mean: List[float] = Field(default_factory=list, description="Mean streaming near bed (m/s)")
    ucr_rms: List[float] = Field(default_factory=list, description="RMS cross-shore undertow (m/s)")
    ub_rms: List[float] = Field(default_factory=list, description="RMS streaming (m/s)")


class TransportProfileSummary(BaseModel):
    """Sediment transport rates along the cross-shore profile."""
    sb_cross: List[float] = Field(default_factory=list, description="Cross-shore bedload transport (kg/(s*m))")
    ss_cross: List[float] = Field(default_factory=list, description="Cross-shore suspended transport (kg/(s*m))")
    st_cross: List[float] = Field(default_factory=list, description="Cross-shore total transport (kg/(s*m))")
    sb_long: List[float] = Field(default_factory=list, description="Longshore bedload transport (kg/(s*m))")
    ss_long: List[float] = Field(default_factory=list, description="Longshore suspended transport (kg/(s*m))")
    st_long: List[float] = Field(default_factory=list, description="Longshore total transport (kg/(s*m))")
    int_sb_long: List[float] = Field(default_factory=list, description="Integrated longshore bedload (kg/s)")
    int_ss_long: List[float] = Field(default_factory=list, description="Integrated longshore suspended (kg/s)")
    int_st_long: List[float] = Field(default_factory=list, description="Integrated longshore total (kg/s)")
    ca_ref: List[float] = Field(default_factory=list, description="Reference concentration Ca at z=a (m3 sed / m3 water)")
    c_vol: List[float] = Field(default_factory=list, description="Reference volumetric concentration Ca (m3 sed / m3 water)")
    c_mean_vol: List[float] = Field(default_factory=list, description="Depth-averaged volumetric concentration (m3 sed / m3 water)")


class MorphoProfileSummary(BaseModel):
    """Bed level changes and sediment fraction distribution."""
    dz_step: List[float] = Field(default_factory=list, description="Bed level change during last time step (m)")
    dz_total: List[float] = Field(default_factory=list, description="Cumulative bed level change since t=0 (m)")
    z_initial: List[float] = Field(default_factory=list, description="Initial bed level at t=0 (m)")
    z_current: List[float] = Field(default_factory=list, description="Current bed level (m)")
    d50: List[float] = Field(default_factory=list, description="Median grain diameter D50 along profile (m)")
    d90: List[float] = Field(default_factory=list, description="D90 grain diameter along profile (m)")
    fractions: List[List[float]] = Field(default_factory=list, description="Fraction percentages along profile")


class VolumeBalanceSummary(BaseModel):
    """Mass and volume conservation metrics."""
    area_balance_m2: float = 0.0
    scross_x0_m2: float = 0.0
    scross_xr_m2: float = 0.0
    volume_by_depth_contour: Dict[float, float] = Field(default_factory=dict)
    volume_by_x_contour: Dict[float, float] = Field(default_factory=dict)


class TimeStepResult(BaseModel):
    """Snapshot of the entire system state at a specific simulation time."""
    time: float = Field(..., description="Simulation time (s)")
    dt: float = Field(..., description="Time step duration (s)")
    step_number: int = Field(0, description="Step counter index")
    x: List[float] = Field(default_factory=list, description="Cross-shore coordinates (m)")
    water_depth: List[float] = Field(default_factory=list, description="Total water depth HD (m)")
    water_level_eta: List[float] = Field(default_factory=list, description="Surface water level (m)")
    hydro: HydroProfileSummary = Field(default_factory=HydroProfileSummary)
    transport: TransportProfileSummary = Field(default_factory=TransportProfileSummary)
    morpho: MorphoProfileSummary = Field(default_factory=MorphoProfileSummary)
    volume: VolumeBalanceSummary = Field(default_factory=VolumeBalanceSummary)
    wave_classes: List[WaveClassProfile] = Field(default_factory=list)


class SimulationResults(BaseModel):
    """Collection of time steps and time series for an entire simulation."""
    time_steps: List[TimeStepResult] = Field(default_factory=list)
    time_series_time: List[float] = Field(default_factory=list)
    time_series_volume_depth: Dict[str, List[float]] = Field(default_factory=dict)
    time_series_volume_x: Dict[str, List[float]] = Field(default_factory=dict)
