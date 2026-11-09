# CROSMOR 2026

Modern, high-performance Python 3.12 implementation of Prof. Leo C. van Rijn's **CROSMOR** coastal morphodynamic model with a native desktop GUI and real-time visualization.

---

## 🌟 Key Features

1. **Modern Numerical Core**:
   - High-speed vectorized computations with NumPy and SciPy.
   - Numba JIT compilation (`@njit(fastmath=True)`) for intra-wave (201 sub-steps) bedload transport and tridiagonal matrix solvers.
   - Complete replacement of legacy Fortran static arrays (`common/room/a(220000)`), common blocks, and unformatted scratch disk files (`units 21, 22, 32, 43, 44`) with in-memory structured data models.

2. **Native Desktop GUI**:
   - Powered by **PyQt6** and **PyQtGraph**.
   - **4 Tabbed Parameter Editors**:
     1. *Grid & Bathymetry*: Interactive coordinate table with live profile preview and runup beach switch.
     2. *Waves & Tide*: Discrete spectra, Rayleigh schematizations, and tidal water level series.
     3. *Hydrodynamics*: Orbital velocity asymmetry (Isobe 2003, Ruessink 2008 & 2010), wave roller, streaming, and mixing coefficients.
     4. *Sediment & Morpho*: Multi-fraction sand grain diameters, active transport layer thickness, calibration factors, and volume balance contours.
   - **4-Panel Synchronized Real-time Dashboard**:
     - *Panel 1*: Bed elevation $Z_b(x, t)$ vs $Z_{b,0}(x)$ and surface water level $\eta(x, t)$.
     - *Panel 2*: Wave height $H_{rms}(x)$ / $H_{1/3}(x)$ and mean currents (longshore $V_{mix}$ and undertow $U_{cr}$).
     - *Panel 3*: Sediment transport rates (cross-shore total $S_t$, bedload $S_b$, suspended $S_s$, and longshore $S_{t,long}$).
     - *Panel 4*: Cumulative bed level change $\Delta z(x, t)$.
   - **Interactive Playback & Scrubbing**: Play, pause, resume, cancel, and a time slider to scrub through computed historical time steps.

3. **100% Backward & Forward Compatibility**:
   - Reads and writes legacy Fortran 77 `.inp` and `.out` files.
   - Reads all 25 original test cases in the repository.
   - Full support for modern JSON project saving/loading, Excel `.xlsx` multi-sheet workbooks, and CSV profile exports.

4. **Rigorous Physical Fidelity & Golden Master Verification**:
   - Preserves all 26 legacy Fortran routines:
     - `golf.f` -> `crosmor.hydrodynamics.wave_solver`
     - `FWL.F` & `dispersion` -> `crosmor.hydrodynamics.dispersion`
     - `RAYL.F` -> `crosmor.hydrodynamics.rayleigh`
     - `SNELH.F` & `FVV.F` -> `crosmor.hydrodynamics.longshore`
     - `STAT.F` & `ORDER.F` -> `crosmor.hydrodynamics.statistics`
     - `BODEM.F` -> `crosmor.morphodynamics.bedload`
     - `SUSPEN.F` -> `crosmor.morphodynamics.suspended`
     - `BEACH.F` -> `crosmor.morphodynamics.beach`
     - `TRANSP.F` -> `crosmor.morphodynamics.morpho_solver`
     - `FDSTER.F` & `CALC.F` -> `crosmor.morphodynamics.sediment_props`
     - `TRIDIAR.F` -> `crosmor.numerics.tridiagonal`
     - `ZEROIN.F` & `TOLX.F` -> `crosmor.numerics.root_finding`
     - `SMOOTH.F` & `SMOOP.F` -> `crosmor.numerics.smoothing`
     - `RK.F`, `RK1ST.F`, `COMP.F`, `FKT.F` -> `crosmor.numerics.ode_rk`

---

## 🚀 Installation

Ensure Python $\ge 3.10$ is installed (Python 3.12 recommended).

From the repository root:
```bash
# Install crosmor in editable mode with development dependencies
python3 -m pip install -e crosmor_app
```

---

## 💻 Running the Application

### 1. Launch the Graphical User Interface (GUI)
```bash
# Launch with empty default case
crosmor-gui

# Or load an existing legacy .inp file directly
crosmor-gui Dgoot1.inp
```

Or via Python module:
```bash
python3 -m crosmor.gui.main_window Dgoot1.inp
```

### 2. Command Line Interface (CLI)
Run batch simulations without a GUI and export results directly:
```bash
crosmor-cli Dgoot1.inp --out-excel results.xlsx --out-csv profile.csv
```

Or via Python module:
```bash
python3 -m crosmor.engine.cli ATEST1.inp --out-excel atest1_results.xlsx
```

---

## 🧪 Automated Test Suite

Run the full pytest suite (18 unit, integration, and golden master regression tests):
```bash
python3 -m pytest crosmor_app/tests -v
```

Tests include:
- `tests/regression/test_golden_master.py`: Comparison against legacy Fortran `dgoot1.out` and `atest1.out`.
- `tests/test_io.py`: Legacy `.inp` / `.out` reading, round-trip writing, JSON, Excel, and CSV export.
- `tests/test_numerics.py`: Linear interpolation, Brent-Dekker root finding, Thomas tridiagonal solver, smoothing filters, Runge-Kutta ODE integrator.
- `tests/test_physics.py`: Wave dispersion, Rayleigh distribution, Ruessink orbital asymmetry, fluid and sediment properties, intra-wave bedload.
- `tests/test_runner_and_gui.py`: Full simulation loop, time stepping, and offscreen PyQt6 GUI parameter collection.

---

## 📂 Project Structure

```
crosmor_app/
├── pyproject.toml                     # Modern PEP 621 package build definition
├── README.md                          # Documentation and instructions
├── src/
│   └── crosmor/
│       ├── models/
│       │   ├── config.py              # Pydantic models for inputs & boundary conditions
│       │   └── results.py             # Pydantic models for simulation outputs & time steps
│       ├── io/
│       │   ├── inp_reader.py          # Robust Fortran 77 list-directed .inp parser
│       │   ├── inp_writer.py          # Formatted legacy .inp writer
│       │   ├── out_reader.py          # Legacy .out block parser
│       │   └── exporters.py          # JSON, Excel (.xlsx) and CSV exporters
│       ├── numerics/
│       │   ├── interpolation.py       # RINTPL linear interpolation & clamping
│       │   ├── root_finding.py        # ZEROIN Brent-Dekker solver
│       │   ├── tridiagonal.py         # TRIDIAR Thomas algorithm (TDMA)
│       │   ├── smoothing.py           # SMOOTH & SMOOP moving averages
│       │   └── ode_rk.py              # Adaptive 4th-order Runge-Kutta integrator
│       ├── hydrodynamics/
│       │   ├── dispersion.py          # FWL dispersion relation & Snell refraction
│       │   ├── rayleigh.py            # RAYL Rayleigh spectrum class generation
│       │   ├── asymmetry.py           # Isobe & Ruessink orbital velocity asymmetry
│       │   ├── statistics.py          # STAT & ORDER wave statistical integrals
│       │   ├── longshore.py           # SNELH & FVV wave-driven longshore currents
│       │   └── wave_solver.py         # GOLF cross-shore wave energy & setup solver
│       ├── morphodynamics/
│       │   ├── sediment_props.py      # Fluid properties, settling velocity, Shields critical
│       │   ├── bedload.py             # BODEM intra-wave bedload transport
│       │   ├── suspended.py           # SUSPEN vertical suspended concentration profile
│       │   ├── beach.py               # BEACH kinematic runup & dune erosion
│       │   └── morpho_solver.py       # TRANSP sediment divergence & bed continuity
│       ├── engine/
│       │   ├── runner.py              # Time-stepping coordinator & volume balance
│       │   └── cli.py                 # Command line runner with progress bar
│       └── gui/
│           ├── worker.py              # Background QThread for asynchronous execution
│           ├── main_window.py         # Main PyQt6 window wiring docks, tabs & runner
│           ├── tabs/
│           │   ├── tab_grid.py        # Bathymetry table & profile preview
│           │   ├── tab_forcing.py     # Waves & tide boundary conditions
│           │   ├── tab_hydro.py       # Switches & hydrodynamic calibration
│           │   └── tab_sediment.py    # Sand fractions, contours & transport options
│           └── plots/
│               ├── dashboard.py       # 4-panel synchronized PyQtGraph real-time canvas
│               └── player_bar.py      # Simulation playback controls & time slider
└── tests/
    ├── regression/
    │   └── test_golden_master.py      # Verification against legacy Fortran .out files
    ├── test_io.py                     # File I/O tests
    ├── test_numerics.py               # Numerical algorithms tests
    ├── test_physics.py                # Physical modules tests
    └── test_runner_and_gui.py         # Runner and GUI integration tests
```
