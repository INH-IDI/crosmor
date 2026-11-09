"""End-to-end tests for SimulationRunner, QThread worker, and GUI."""

import os
import sys
from pathlib import Path
import numpy as np
import pytest

from crosmor.io.inp_reader import read_inp
from crosmor.engine.runner import SimulationRunner
from crosmor.models.config import SimulationConfig


def _find_file(fname: str) -> Path:
    cand1 = Path(__file__).parent.parent / "examples" / fname
    if cand1.exists():
        return cand1
    return Path(__file__).parent.parent.parent / fname


def test_simulation_runner_execution():
    inp_file = _find_file("Dgoot1.inp")
    if not inp_file.exists():
        pytest.skip("Dgoot1.inp not found")

    cfg = read_inp(inp_file)
    cfg.morpho.tend = 100.0
    cfg.morpho.factime = 0.01  # dt = 36s -> 3 steps for 100s

    runner = SimulationRunner(cfg)
    steps_received = []

    def on_step(res, prog):
        steps_received.append((res, prog))

    results = runner.run(on_step_callback=on_step)
    assert len(results.time_steps) >= 2
    assert len(steps_received) == len(results.time_steps)

    # Check conservation of cross-shore grid
    first_step = results.time_steps[0]
    last_step = results.time_steps[-1]
    assert len(first_step.x) == len(last_step.x)
    assert np.all(np.array(last_step.morpho.z_current) < 5.0)  # Physical bed levels

    # Check volumetric concentration outputs along profile
    assert len(last_step.transport.c_vol) == len(last_step.x)
    assert len(last_step.transport.c_mean_vol) == len(last_step.x)
    assert all(c >= 0.0 for c in last_step.transport.c_vol)
    assert all(c >= 0.0 for c in last_step.transport.c_mean_vol)


def test_fixed_dt_runner():
    inp_file = _find_file("Dgoot1.inp")
    if not inp_file.exists():
        pytest.skip("Dgoot1.inp not found")

    cfg = read_inp(inp_file)
    cfg.morpho.tend = 300.0
    cfg.morpho.dt_fixed = 100.0  # Exactly 100 s per step

    runner = SimulationRunner(cfg)
    assert runner.compute_adaptive_dt(0.5) == 100.0
    assert runner.compute_adaptive_dt(5.0) == 100.0

    res = runner.run()
    assert len(res.time_steps) == 3
    assert np.isclose(res.time_steps[0].dt, 100.0)
    assert np.isclose(res.time_steps[1].dt, 100.0)
    assert np.isclose(res.time_steps[2].dt, 100.0)


def test_gui_offscreen_components():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6.QtWidgets import QApplication
    from crosmor.gui.main_window import MainWindow

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    win = MainWindow()
    inp_file = _find_file("Dgoot1.inp")

    if inp_file.exists():
        win.load_inp_file(inp_file)
        assert win.current_config.title == "Dgoot1"

        # Check tabs state
        assert win.tab_grid.table.rowCount() > 0
        assert win.tab_sediment.spin_tend.value() > 0

        # Collect back config
        cfg = win.collect_config()
        assert cfg.title == "Dgoot1"

        # Check dashboard dynamic panel selection
        win.dashboard.chk_concentration.setChecked(True)
        assert win.dashboard.chk_p_conc.isChecked()
        win.dashboard.combo_presets.setCurrentIndex(1)  # 3 Panels: Profile + Transport + Concentration
        assert win.dashboard.chk_p_conc.isChecked()
        assert win.dashboard.chk_p1.isChecked()
        assert win.dashboard.chk_p3.isChecked()
        assert not win.dashboard.chk_p2.isChecked()

        # Check TabSediment fixed dt UI
        win.tab_sediment.combo_dt_mode.setCurrentIndex(1)
        win.tab_sediment.spin_dt_fixed.setValue(123.0)
        cfg_fixed = win.collect_config()
        assert cfg_fixed.morpho.dt_fixed == 123.0

    win.close()
