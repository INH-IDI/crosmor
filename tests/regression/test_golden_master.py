"""Golden Master regression test comparing modernized Python outputs against legacy Fortran .out files."""

from pathlib import Path
import numpy as np
import pytest

from crosmor.io.inp_reader import read_inp
from crosmor.io.out_reader import read_out
from crosmor.engine.runner import SimulationRunner


def _find_file(fname: str) -> Path:
    cand1 = Path(__file__).parent.parent.parent / "examples" / fname
    if cand1.exists():
        return cand1
    return Path(__file__).parent.parent.parent.parent / fname


def test_dgoot1_hydrodynamics_golden_master():
    inp_file = _find_file("Dgoot1.inp")
    out_file = _find_file("dgoot1.out")

    if not inp_file.exists() or not out_file.exists():
        pytest.skip("dgoot1.inp or dgoot1.out not found")

    cfg = read_inp(inp_file)
    cfg.morpho.tend = 10.0
    cfg.morpho.factime = 0.01

    runner = SimulationRunner(cfg)
    res = runner.run()

    step0 = res.time_steps[0]
    out_data = read_out(out_file)

    # Find B001 block (Initial hydrodynamics profile)
    b001 = [b for b in out_data.blocks if b.block_id == "B001"][0]
    fortran_x = b001.data[:, 0]
    fortran_hd = b001.data[:, 1]
    fortran_hrms = b001.data[:, 3]
    fortran_h13 = b001.data[:, 5]

    python_x = np.asarray(step0.x)
    python_hd = np.asarray(step0.water_depth)
    python_hrms = np.asarray(step0.hydro.hrms)
    python_h13 = np.asarray(step0.hydro.h13)

    n_wet = len(fortran_x)
    # 1. Verify grid alignment in wet zone with hydro block B001
    assert np.allclose(python_x[:n_wet], fortran_x, atol=0.01)

    # 2. Verify water depth profile
    assert np.allclose(python_hd[:n_wet], fortran_hd, atol=0.05)

    # 3. Verify wave height (HRMS) matching Fortran 77 within 0.05 m across wet profile
    assert np.allclose(python_hrms[:n_wet], fortran_hrms, atol=0.05)

    # 4. Verify significant wave height (H1/3) within 0.05 m
    assert np.allclose(python_h13[:n_wet], fortran_h13, atol=0.05)

    # 5. Verify full profile grid including dry beach matches Fortran block B011 (651 points)
    b011 = [b for b in out_data.blocks if b.block_id == "B011"][0]
    fortran_x_full = b011.data[:, 0]
    fortran_z_init = b011.data[:, 3]
    assert len(python_x) == len(fortran_x_full)
    assert np.allclose(python_x, fortran_x_full, atol=0.06)
    assert np.allclose(step0.morpho.z_initial, fortran_z_init, atol=0.05)


def test_atest1_wave_height_golden_master():
    inp_file = _find_file("ATEST1.inp")
    out_file = _find_file("atest1.out")

    if not inp_file.exists() or not out_file.exists():
        pytest.skip("ATEST1.inp or atest1.out not found")

    cfg = read_inp(inp_file)
    cfg.morpho.tend = 10.0
    cfg.morpho.factime = 0.01

    runner = SimulationRunner(cfg)
    res = runner.run()

    step0 = res.time_steps[0]
    out_data = read_out(out_file)

    b001 = [b for b in out_data.blocks if b.block_id == "B001"][0]
    fortran_x = b001.data[:, 0]
    fortran_h_class1 = b001.data[:, 4]  # Column 4 is wave height H for wave class 1
    python_x = np.asarray(step0.x)
    python_h_class1 = np.asarray(step0.wave_classes[0].h)

    assert len(python_x) == len(fortran_x)
    assert np.allclose(python_x, fortran_x, atol=0.01)
    assert np.allclose(python_h_class1, fortran_h_class1, atol=0.15)
