"""Data models for CROSMOR simulation configuration."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class TidePoint(BaseModel):
    """Point in tidal variation time series."""
    time: float = Field(..., description="Time in seconds (T)")
    vnul1: float = Field(0.0, description="Longshore current at x=0 (m/s)")
    vnul2: float = Field(0.0, description="Cross-shore current at x=0 (m/s, positive landward)")
    ws: float = Field(0.0, description="Tidal water level relative to MSL (m, positive upward)")


class StormPoint(BaseModel):
    """Point in storm surge time series."""
    time: float = Field(..., description="Time in seconds (T)")
    hstorm: float = Field(0.0, description="Storm surge water level set-up (m)")


class BathymetrySection(BaseModel):
    """Bathymetry profile segment defining cross-shore geometry."""
    x: float = Field(..., description="Cross-shore coordinate (m)")
    hmsl: float = Field(..., description="Bed elevation / depth to MSL (negative below MSL, m)")
    nstap: Optional[int] = Field(None, description="Number of grid sub-steps to the next coordinate")


class WaveClass(BaseModel):
    """Wave condition class for discrete wave spectra (igolf = 1)."""
    h: float = Field(..., description="Wave height at seaward boundary x=0 (m)")
    t: float = Field(..., description="Wave period (s)")
    p: float = Field(..., description="Percentage / probability of occurrence (0 to 1)")
    teta: float = Field(0.0, description="Wave angle of incidence w.r.t. shore normal (degrees)")
    w10: float = Field(0.0, description="Wind velocity 10m above surface (m/s)")


class WaveTimePoint(BaseModel):
    """Wave climate time series point (igolf = 2 or -2)."""
    time: float = Field(..., description="Time duration / point in seconds (s)")
    hrms: float = Field(..., description="RMS wave height at x=0 (m)")
    t: float = Field(..., description="Wave period (s)")
    teta: float = Field(0.0, description="Wave angle w.r.t. coast normal (degrees)")
    w10: float = Field(0.0, description="Wind velocity 10m above surface (m/s)")


class SedimentProfilePoint(BaseModel):
    """Bed roughness, mud content and undertow factor along cross-shore coordinate."""
    x: float = Field(..., description="Cross-shore coordinate (m)")
    rc: float = Field(0.0, description="Current-related roughness (m, 0 = compute automatically)")
    rw: float = Field(0.0, description="Wave-related roughness (m, 0 = compute automatically)")
    pmud: float = Field(0.0, description="Percentage of mud in bed (%)")
    frip: float = Field(1.0, description="Multiplication factor to undertow (default = 1.0)")
    fractions: List[float] = Field(default_factory=list, description="Fraction distribution (sum + pmud/100 = 1.0)")


class HydroParams(BaseModel):
    """Hydrodynamic model switches and calibration parameters."""
    switch1_asymmetry: int = Field(1, description="0=Isobe-Horikawa, 1=Ruessink 2008, 2=Ruessink 2010")
    roller_coef: float = Field(0.5, description="Roller coefficient (0=max roller effect, 1=no roller, default 0.5)")
    coef6: float = Field(1.0, description="Distance factor for wave param averaging in undertow (3 field, 1 lab)")
    clh: float = Field(1.0, description="Longuet-Higgins streaming coefficient (0 to 1)")
    csw: float = Field(1.0, description="Swash velocity coefficient for shingle/gravel beaches (0 to 1)")
    switch2_nielsen: int = Field(0, description="0=exclude acceleration effect, 1=include Nielsen acceleration")
    switch3_phaselag: int = Field(0, description="0=exclude phase lag for susp transport, 1=include phase lag")
    sl_dune_slope: float = Field(50.0, description="Maximum slope angle of dune front (degrees) before sliding")
    sef1: float = Field(1.0, description="Suspension enhancement factor (1 without dune erosion, 2 with dune erosion)")
    switch4_longwaves: int = Field(1, description="0=exclude long waves in surf zone, 1=include long waves")
    coefsr: float = Field(0.0, description="Roller shear stress coefficient at water surface (0 = default)")
    gamma: float = Field(0.0, description="Breaking index (0 = computed by model; or specific fixed value)")
    coef5: float = Field(1.0, description="Averaging factor of wave height and depth for longshore current (3 field, 1 lab)")
    els: float = Field(0.5, description="Horizontal mixing coefficient for longshore current (0.1 to 10 m2/s)")
    nit: int = Field(300, description="Max iterations for longshore velocity solver (default 300)")
    sw: float = Field(0.2, description="Extra smoothing near waterline (0.01 to 0.2)")


class MorphoParams(BaseModel):
    """Sediment transport, time stepping and morphology parameters."""
    iprant: int = Field(2, description="Transport output option: 0=no output, 1=summed over classes, 2=per class")
    nout: int = Field(3, description="Output of bed levels every nout time steps")
    dl: float = Field(0.25, description="Thickness of active transport layer (m) (0.2 to 1.0)")
    rhos: float = Field(2650.0, description="Sediment density (kg/m3)")
    por: float = Field(0.4, description="Porosity of bed (0 to 1)")
    facbed: float = Field(1.0, description="Calibration factor bed load transport (default 1.0)")
    facsusc: float = Field(1.0, description="Calibration factor current-related suspended load (default 1.0)")
    facsusw: float = Field(1.0, description="Calibration factor wave-related suspended load (default 1.0)")
    tend: float = Field(86400.0, description="Total computation time (s)")
    factime: float = Field(0.25, description="Time step factor multiplier")
    facbed_smooth: float = Field(0.2, description="Bed smoothing factor (gamb = facbed_smooth * 0.005)")
    facgrain_smooth: float = Field(1.0, description="Fractions smoothing factor (gamp = facgrain_smooth * 0.2)")
    water_temp: float = Field(15.0, description="Water temperature (Celsius)")
    salinity: float = Field(0.0, description="Fluid salinity (promille)")
    grain_diameters: List[float] = Field(default_factory=lambda: [0.0002], description="Grain diameters (m) per fraction")
    depth_contours: List[float] = Field(default_factory=list, description="Depth contours for volume balance (andc, m)")
    x_contours: List[float] = Field(default_factory=list, description="X coordinates for volume balance (xnsp, m)")


class SimulationConfig(BaseModel):
    """Complete CROSMOR simulation configuration."""
    title: str = Field("CROSMOR Simulation", description="Case study or run title")
    strict_fortran_compat: bool = Field(True, description="Emulate original Fortran edge cases / rounding exactly")
    iprint: int = Field(2, description="Hydrodynamics output option: 0=none, 1=weighted summary, 2=per wave class")
    tide_series: List[TidePoint] = Field(default_factory=list, description="Tidal boundary conditions")
    bathymetry: List[BathymetrySection] = Field(default_factory=list, description="Initial bed profile coordinates and steps")
    hgrens: float = Field(0.25, description="Minimum water depth for calculation cutoff (m)")
    istrand: int = Field(0, description="1 = activate kinematic beach/runup model, 0 = no beach module")
    storm_series: List[StormPoint] = Field(default_factory=list, description="Storm surge level series")
    igolf: int = Field(1, description="Wave option: 1=discrete spectrum, 2=Rayleigh table, -2=Rayleigh from file")
    wave_classes: List[WaveClass] = Field(default_factory=list, description="Discrete wave classes (for igolf=1)")
    nh_wave_classes: int = Field(5, description="Number of wave classes for Rayleigh distribution (igolf=2 or -2)")
    wave_file: Optional[str] = Field(None, description="External wave file path (when igolf = -2)")
    wave_series: List[WaveTimePoint] = Field(default_factory=list, description="Wave time series (igolf=2 or from file)")
    hydro: HydroParams = Field(default_factory=HydroParams)
    morpho: MorphoParams = Field(default_factory=MorphoParams)
    sediment_points: List[SedimentProfilePoint] = Field(default_factory=list, description="Cross-shore sediment properties")

    @property
    def nfr(self) -> int:
        """Number of sand fractions."""
        return len(self.morpho.grain_diameters) if self.morpho.grain_diameters else 1

    @property
    def nsed(self) -> int:
        """Number of sediment control points along cross-shore."""
        return len(self.sediment_points)
