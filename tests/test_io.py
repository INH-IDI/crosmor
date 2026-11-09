"""Unit tests for CROSMOR file I/O and exporters."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from crosmor.io.inp_reader import read_inp
from crosmor.io.inp_writer import write_inp
from crosmor.io.out_reader import read_out
from crosmor.io.exporters import (
    save_project_json,
    load_project_json,
    export_results_to_excel,
    export_profile_to_csv,
)
from crosmor.models.config import SimulationConfig, BathymetrySection, WaveClass
from crosmor.models.results import SimulationResults, TimeStepResult, MorphoProfileSummary, HydroProfileSummary, TransportProfileSummary


def _find_data_file(fname: str) -> Path:
    cand1 = Path(__file__).parent.parent / "examples" / fname
    if cand1.exists():
        return cand1
    return Path(__file__).parent.parent.parent / fname


def test_read_legacy_inps():
    files_to_test = ["Dgoot1.inp", "ATEST1.inp", "dms1fr.inp", "dms2.inp", "lip1c.inp"]

    for fname in files_to_test:
        fpath = _find_data_file(fname)
        if fpath.exists():
            cfg = read_inp(fpath)
            assert isinstance(cfg, SimulationConfig)
            assert len(cfg.bathymetry) >= 2
            assert cfg.morpho.tend > 0


def test_roundtrip_inp():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_roundtrip.inp"

        orig_cfg = SimulationConfig(
            title="Roundtrip Test",
            bathymetry=[
                BathymetrySection(x=0.0, hmsl=12.0, nstap=10),
                BathymetrySection(x=500.0, hmsl=6.0, nstap=5),
                BathymetrySection(x=1000.0, hmsl=-2.0, nstap=None),
            ],
            wave_classes=[
                WaveClass(h=2.5, t=8.0, p=0.7, teta=10.0),
                WaveClass(h=1.5, t=6.0, p=0.3, teta=5.0),
            ],
        )
        orig_cfg.morpho.tend = 7200.0

        write_inp(orig_cfg, tmp_path)
        assert tmp_path.exists()

        reloaded_cfg = read_inp(tmp_path)
        assert reloaded_cfg.title == tmp_path.stem
        assert len(reloaded_cfg.bathymetry) == len(orig_cfg.bathymetry)
        assert len(reloaded_cfg.wave_classes) == len(orig_cfg.wave_classes)
        assert np.isclose(reloaded_cfg.morpho.tend, 7200.0)


def test_out_reader():
    for fname in ["dgoot1.out", "atest1.out"]:
        fpath = _find_data_file(fname)
        if fpath.exists():
            out_data = read_out(fpath)
            assert len(out_data.blocks) > 0
            assert any(b.block_id.startswith("A") or b.block_id.startswith("B") for b in out_data.blocks)


def test_exporters():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # JSON project export/import
        cfg = SimulationConfig(title="Export Test")
        json_path = tmp / "case.json"
        save_project_json(cfg, json_path)
        assert json_path.exists()
        loaded = load_project_json(json_path)
        assert loaded.title == "Export Test"

        # Mock results
        x = [0.0, 50.0, 100.0]
        step = TimeStepResult(
            time=100.0,
            dt=100.0,
            step_number=1,
            x=x,
            water_depth=[10.0, 8.0, 5.0],
            water_level_eta=[0.0, 0.05, 0.1],
            hydro=HydroProfileSummary(hrms=[1.5, 1.4, 1.2], h13=[2.0, 1.8, 1.6], v_mix=[0.5, 0.6, 0.4]),
            transport=TransportProfileSummary(st_cross=[0.01, 0.02, 0.01], st_long=[0.05, 0.06, 0.04]),
            morpho=MorphoProfileSummary(
                z_initial=[-10.0, -8.0, -5.0],
                z_current=[-9.99, -8.01, -4.98],
                dz_total=[0.01, -0.01, 0.02],
            ),
        )
        res = SimulationResults(time_steps=[step])

        # Excel export
        excel_path = tmp / "results.xlsx"
        export_results_to_excel(res, excel_path)
        assert excel_path.exists()
        assert excel_path.stat().st_size > 0

        # CSV export
        csv_path = tmp / "profile.csv"
        export_profile_to_csv(step, csv_path)
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0
