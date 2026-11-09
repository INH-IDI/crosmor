"""Main Application Window for CROSMOR 2026."""

import sys
from pathlib import Path
from typing import Optional, List, Union

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QSplitter,
    QScrollArea,
    QFrame,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QLabel,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QIcon

from crosmor.models.config import SimulationConfig, BathymetrySection, WaveClass, SedimentProfilePoint
from crosmor.models.results import SimulationResults, TimeStepResult, MorphoProfileSummary
from crosmor.hydrodynamics.wave_solver import build_grid
from crosmor.io.inp_reader import read_inp
from crosmor.io.inp_writer import write_inp
from crosmor.io.exporters import (
    save_project_json,
    load_project_json,
    export_results_to_excel,
    export_profile_to_csv,
)
from crosmor.gui.tabs.tab_grid import TabGrid
from crosmor.gui.tabs.tab_forcing import TabForcing
from crosmor.gui.tabs.tab_hydro import TabHydro
from crosmor.gui.tabs.tab_sediment import TabSediment
from crosmor.gui.plots.dashboard import MorphoDashboard
from crosmor.gui.plots.player_bar import PlayerBar
from crosmor.gui.worker import SimulationWorker


class MainWindow(QMainWindow):
    """Main application window uniting parameter editors, dashboard and runner."""

    def __init__(self, initial_config_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("CROSMOR 2026 - Coastal Morphodynamic Modeling System")

        # Dynamically size and center window to fit within available screen space
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            target_w = min(1380, max(880, int(avail.width() * 0.90)))
            target_h = min(860, max(600, int(avail.height() * 0.88)))
            self.resize(target_w, target_h)
            x = avail.x() + max(0, (avail.width() - target_w) // 2)
            y = avail.y() + max(0, (avail.height() - target_h) // 2)
            self.move(x, y)
        else:
            self.resize(1200, 750)

        self.current_config: SimulationConfig = self._create_default_config()
        self.current_results: Optional[SimulationResults] = None
        self.worker: Optional[SimulationWorker] = None
        self._current_file_path: Optional[Path] = None

        self._init_ui()
        self._create_menus()
        self._setup_status_bar()

        # Load initial config
        if initial_config_path and Path(initial_config_path).exists():
            self.load_inp_file(Path(initial_config_path))
        else:
            self.load_config(self.current_config)

    def _create_default_config(self) -> SimulationConfig:
        """Create a sensible default simulation configuration."""
        cfg = SimulationConfig()
        cfg.title = "Default CROSMOR Case"
        cfg.bathymetry = [
            BathymetrySection(x=0.0, hmsl=-10.0, nstap=50),
            BathymetrySection(x=1500.0, hmsl=-2.0, nstap=30),
            BathymetrySection(x=1800.0, hmsl=0.0, nstap=20),
            BathymetrySection(x=2000.0, hmsl=3.0, nstap=None),
        ]
        cfg.morpho.grain_diameters = [0.0002]
        cfg.morpho.tend = 3600.0  # 1 hour in seconds
        cfg.morpho.factime = 1.0
        cfg.wave_classes = [WaveClass(h=2.0, t=7.0, p=1.0, teta=15.0)]
        cfg.sediment_points = [
            SedimentProfilePoint(x=0.0, rc=0.0, rw=0.0, pmud=0.0, frip=1.0, fractions=[1.0]),
            SedimentProfilePoint(x=2000.0, rc=0.0, rw=0.0, pmud=0.0, frip=1.0, fractions=[1.0]),
        ]
        return cfg

    def _init_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

        # Left Pane: Tabbed Parameter Editors (wrapped in QScrollArea for smaller displays)
        self.tabs = QTabWidget()
        self.tab_grid = TabGrid()
        self.tab_forcing = TabForcing()
        self.tab_hydro = TabHydro()
        self.tab_sediment = TabSediment()

        def _make_scrollable(widget: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setWidget(widget)
            return scroll

        self.tabs.addTab(_make_scrollable(self.tab_grid), "1. Grid & Bathymetry")
        self.tabs.addTab(_make_scrollable(self.tab_forcing), "2. Waves & Tide")
        self.tabs.addTab(_make_scrollable(self.tab_hydro), "3. Hydrodynamics")
        self.tabs.addTab(_make_scrollable(self.tab_sediment), "4. Sediment & Morpho")
        splitter.addWidget(self.tabs)

        # Right Pane: Real-time Dashboard and Player Bar
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        self.dashboard = MorphoDashboard()
        right_layout.addWidget(self.dashboard, stretch=1)

        self.player_bar = PlayerBar()
        self.player_bar.run_requested.connect(self.start_simulation)
        self.player_bar.pause_requested.connect(self.pause_simulation)
        self.player_bar.stop_requested.connect(self.stop_simulation)
        self.player_bar.step_selected.connect(self._on_scrubber_moved)
        right_layout.addWidget(self.player_bar)

        splitter.addWidget(right_container)

        curr_w = self.width() if self.width() > 0 else 1200
        left_w = max(340, int(curr_w * 0.35))
        right_w = max(550, curr_w - left_w)
        splitter.setSizes([left_w, right_w])

    def _create_menus(self) -> None:
        mb = self.menuBar()

        # --- File Menu ---
        file_menu = mb.addMenu("&File")

        act_new = QAction("&New Case", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self.new_case)
        file_menu.addAction(act_new)

        act_open_inp = QAction("&Open Legacy INP...", self)
        act_open_inp.setShortcut(QKeySequence.StandardKey.Open)
        act_open_inp.triggered.connect(self.open_inp_dialog)
        file_menu.addAction(act_open_inp)

        act_save_inp = QAction("&Save Legacy INP...", self)
        act_save_inp.setShortcut(QKeySequence.StandardKey.Save)
        act_save_inp.triggered.connect(self.save_inp_dialog)
        file_menu.addAction(act_save_inp)

        file_menu.addSeparator()

        act_open_json = QAction("Open Project JSON...", self)
        act_open_json.triggered.connect(self.open_json_dialog)
        file_menu.addAction(act_open_json)

        act_save_json = QAction("Save Project JSON...", self)
        act_save_json.triggered.connect(self.save_json_dialog)
        file_menu.addAction(act_save_json)

        file_menu.addSeparator()

        act_export_excel = QAction("Export Results to Excel (.xlsx)...", self)
        act_export_excel.triggered.connect(self.export_excel_dialog)
        file_menu.addAction(act_export_excel)

        act_export_csv = QAction("Export Final Profile to CSV...", self)
        act_export_csv.triggered.connect(self.export_csv_dialog)
        file_menu.addAction(act_export_csv)

        file_menu.addSeparator()

        act_exit = QAction("E&xit", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # --- Simulation Menu ---
        sim_menu = mb.addMenu("&Simulation")

        act_run = QAction("&Run Simulation", self)
        act_run.setShortcut(Qt.Key.Key_F5)
        act_run.triggered.connect(self.start_simulation)
        sim_menu.addAction(act_run)

        act_pause = QAction("&Pause / Resume", self)
        act_pause.setShortcut(Qt.Key.Key_F6)
        act_pause.triggered.connect(self.pause_simulation)
        sim_menu.addAction(act_pause)

        act_stop = QAction("&Stop Simulation", self)
        act_stop.setShortcut(Qt.Key.Key_Escape)
        act_stop.triggered.connect(self.stop_simulation)
        sim_menu.addAction(act_stop)

        # --- View Menu ---
        view_menu = mb.addMenu("&View")

        act_autorange = QAction("&Auto Range Dashboard", self)
        act_autorange.setShortcut("Ctrl+R")
        act_autorange.triggered.connect(self.dashboard.auto_range)
        view_menu.addAction(act_autorange)

        act_clear = QAction("&Clear Dashboard", self)
        act_clear.triggered.connect(self.dashboard.clear)
        view_menu.addAction(act_clear)

        # --- Help Menu ---
        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About CROSMOR 2026", self)
        act_about.triggered.connect(self.show_about_dialog)
        help_menu.addAction(act_about)

    def _setup_status_bar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_case_status = QLabel("Ready")
        sb.addPermanentWidget(self.lbl_case_status)
        sb.showMessage("CROSMOR 2026 Ready.")

    def collect_config(self) -> SimulationConfig:
        """Collect current values from all tab editors into current_config."""
        self.tab_grid.update_config(self.current_config)
        self.tab_forcing.update_config(self.current_config)
        self.tab_hydro.update_config(self.current_config)
        self.tab_sediment.update_config(self.current_config)
        return self.current_config

    def load_config(self, config: SimulationConfig) -> None:
        """Load configuration into tabs and refresh dashboard."""
        self.current_config = config
        self.tab_grid.load_config(config)
        self.tab_forcing.load_config(config)
        self.tab_hydro.load_config(config)
        self.tab_sediment.load_config(config)

        self.player_bar.reset()
        self.current_results = SimulationResults()

        # Render initial bathymetry on dashboard
        try:
            x_grid, hmsl_grid, _ = build_grid(config)
            nx = len(x_grid)
            init_step = TimeStepResult(
                time=0.0,
                dt=0.0,
                step_number=0,
                x=x_grid.tolist(),
                water_depth=hmsl_grid.tolist(),
                water_level_eta=[0.0] * nx,
                morpho=MorphoProfileSummary(
                    z_initial=(-hmsl_grid).tolist(),
                    z_current=(-hmsl_grid).tolist(),
                    dz_total=[0.0] * nx,
                ),
            )
            self.dashboard.clear()
            self.dashboard.update_data(init_step)
            self.dashboard.auto_range()
        except Exception:
            pass

        title = config.title or "Untitled"
        self.statusBar().showMessage(f"Loaded case: {title}")

    # -------------------------------------------------------------------------
    # Simulation Execution & Real-time Callbacks
    # -------------------------------------------------------------------------
    def start_simulation(self) -> None:
        if self.worker and self.worker.isRunning():
            return

        try:
            cfg = self.collect_config()
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", f"Cannot start simulation:\n{e}")
            return

        self.current_results = SimulationResults()
        self.dashboard.clear()
        self.player_bar.reset()
        self.player_bar.set_running_state(True)
        self.lbl_case_status.setText("Running...")

        self.worker = SimulationWorker(cfg, parent=self)
        self.worker.step_completed.connect(self._on_worker_step)
        self.worker.simulation_finished.connect(self._on_worker_finished)
        self.worker.simulation_error.connect(self._on_worker_error)
        self.worker.status_message.connect(self.statusBar().showMessage)
        self.worker.start()

    def pause_simulation(self) -> None:
        if self.worker and self.worker.isRunning():
            if self.worker.runner._is_paused:
                self.worker.resume()
                self.lbl_case_status.setText("Running...")
            else:
                self.worker.pause()
                self.lbl_case_status.setText("Paused")

    def stop_simulation(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.lbl_case_status.setText("Stopping...")

    def _on_worker_step(self, step_res: TimeStepResult, progress: float) -> None:
        if self.current_results is not None:
            self.current_results.time_steps.append(step_res)
            total_steps = len(self.current_results.time_steps)
            self.dashboard.update_data(step_res)
            self.player_bar.update_step(
                step_idx=total_steps - 1,
                total_steps=total_steps,
                time_s=step_res.time,
                balance_m2=step_res.volume.area_balance_m2,
                progress=progress,
            )

    def _on_worker_finished(self, results: SimulationResults) -> None:
        self.current_results = results
        self.player_bar.set_running_state(False)
        self.lbl_case_status.setText("Finished")
        steps_count = len(results.time_steps)
        self.statusBar().showMessage(f"Simulation finished ({steps_count} steps computed).")
        QMessageBox.information(
            self,
            "Simulation Complete",
            f"Simulation completed successfully with {steps_count} time steps.",
        )

    def _on_worker_error(self, err_msg: str) -> None:
        self.player_bar.set_running_state(False)
        self.lbl_case_status.setText("Error")
        self.statusBar().showMessage(f"Simulation error: {err_msg}")
        QMessageBox.critical(self, "Simulation Error", f"An error occurred during simulation:\n\n{err_msg}")

    def _on_scrubber_moved(self, step_idx: int) -> None:
        """Inspect historical time step when user scrubs time slider."""
        if not self.current_results or not self.current_results.time_steps:
            return
        if 0 <= step_idx < len(self.current_results.time_steps):
            step = self.current_results.time_steps[step_idx]
            self.dashboard.update_data(step)
            hours = step.time / 3600.0
            self.player_bar.lbl_time.setText(f"Time: {step.time:.1f} s ({hours:.2f} h)")
            self.player_bar.lbl_balance.setText(f"Balance: {step.volume.area_balance_m2:+.4f} m²")
            self.player_bar.lbl_step.setText(f"Step: {step_idx + 1} / {len(self.current_results.time_steps)}")

    # -------------------------------------------------------------------------
    # File Operations
    # -------------------------------------------------------------------------
    def new_case(self) -> None:
        self.current_config = self._create_default_config()
        self._current_file_path = None
        self.setWindowTitle("CROSMOR 2026 - [Untitled]")
        self.load_config(self.current_config)

    def open_inp_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Legacy CROSMOR Input File", "", "CROSMOR Input Files (*.inp *.dat);;All Files (*)"
        )
        if path:
            self.load_inp_file(Path(path))

    def load_inp_file(self, path: Union[str, Path]) -> None:
        path = Path(path)
        try:
            cfg = read_inp(path)
            self._current_file_path = path
            self.setWindowTitle(f"CROSMOR 2026 - [{path.name}]")
            self.load_config(cfg)
        except Exception as e:
            QMessageBox.critical(self, "Read Error", f"Could not read input file:\n{path}\n\nError: {e}")

    def save_inp_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Legacy CROSMOR Input File", "", "CROSMOR Input Files (*.inp);;All Files (*)"
        )
        if path:
            try:
                cfg = self.collect_config()
                write_inp(cfg, path)
                self._current_file_path = Path(path)
                self.setWindowTitle(f"CROSMOR 2026 - [{self._current_file_path.name}]")
                self.statusBar().showMessage(f"Saved: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Write Error", f"Could not save input file:\n{path}\n\nError: {e}")

    def open_json_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project JSON", "", "JSON Project Files (*.json);;All Files (*)"
        )
        if path:
            try:
                cfg = load_project_json(path)
                self._current_file_path = Path(path)
                self.setWindowTitle(f"CROSMOR 2026 - [{self._current_file_path.name}]")
                self.load_config(cfg)
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Could not load JSON project:\n{path}\n\nError: {e}")

    def save_json_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project JSON", "", "JSON Project Files (*.json);;All Files (*)"
        )
        if path:
            try:
                cfg = self.collect_config()
                save_project_json(cfg, path)
                self._current_file_path = Path(path)
                self.setWindowTitle(f"CROSMOR 2026 - [{self._current_file_path.name}]")
                self.statusBar().showMessage(f"Saved JSON: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save JSON project:\n{path}\n\nError: {e}")

    def export_excel_dialog(self) -> None:
        if not self.current_results or not self.current_results.time_steps:
            QMessageBox.warning(self, "No Results", "Please run a simulation before exporting to Excel.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results to Excel Workbook", "crosmor_results.xlsx", "Excel Files (*.xlsx);;All Files (*)"
        )
        if path:
            try:
                export_results_to_excel(self.current_results, path)
                self.statusBar().showMessage(f"Exported Excel workbook to: {path}")
                QMessageBox.information(self, "Export Complete", f"Successfully exported results to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export Excel:\n{e}")

    def export_csv_dialog(self) -> None:
        if not self.current_results or not self.current_results.time_steps:
            QMessageBox.warning(self, "No Results", "Please run a simulation before exporting to CSV.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Final Profile to CSV", "crosmor_final_profile.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            try:
                last_step = self.current_results.time_steps[-1]
                export_profile_to_csv(last_step, path)
                self.statusBar().showMessage(f"Exported final profile to: {path}")
                QMessageBox.information(self, "Export Complete", f"Successfully exported profile to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{e}")

    def show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About CROSMOR 2026",
            "<h3>CROSMOR 2026</h3>"
            "<p><b>Coastal Profile Morphodynamic Model</b></p>"
            "<p>Original Fortran 77 model by Prof. Leo C. van Rijn (1997–2010).</p>"
            "<p>Modernized in 2026 with Python 3.12, NumPy, Numba JIT acceleration, "
            "PyQt6 native GUI, and high-performance PyQtGraph real-time visualization.</p>"
            "<p>Features full cross-shore hydrodynamics, orbital asymmetry, intra-wave bedload, "
            "vertical suspended concentration profiles, dune avalanche, and active sediment layer conservation.</p>",
        )


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CROSMOR 2026")

    initial_inp = sys.argv[1] if len(sys.argv) > 1 else None
    window = MainWindow(initial_config_path=initial_inp)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
