"""Tab for editing hydrodynamic forcing: waves, tides, currents and storm surge."""

from typing import List
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QLabel,
    QComboBox,
    QSpinBox,
    QLineEdit,
    QGroupBox,
    QHeaderView,
)

from crosmor.models.config import SimulationConfig, TidePoint, StormPoint, WaveClass, WaveTimePoint


class TabForcing(QWidget):
    """Boundary conditions and forcing editor."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        sub_tabs = QTabWidget()

        # 1. Waves Sub-Tab
        tab_wave = QWidget()
        wave_layout = QVBoxLayout(tab_wave)

        wave_mode_box = QGroupBox("Wave Climate Mode (igolf)")
        wm_layout = QHBoxLayout(wave_mode_box)
        wm_layout.addWidget(QLabel("Mode:"))
        self.combo_igolf = QComboBox()
        self.combo_igolf.addItem("1: Discrete Spectrum (specified classes)", 1)
        self.combo_igolf.addItem("2: Rayleigh distribution from time table", 2)
        self.combo_igolf.addItem("-2: Rayleigh distribution from external file", -2)
        self.combo_igolf.currentIndexChanged.connect(self._on_igolf_changed)
        wm_layout.addWidget(self.combo_igolf)

        wm_layout.addWidget(QLabel("Rayleigh Classes (nh):"))
        self.spin_nh = QSpinBox()
        self.spin_nh.setRange(1, 20)
        self.spin_nh.setValue(5)
        wm_layout.addWidget(self.spin_nh)

        wm_layout.addWidget(QLabel("Wave File:"))
        self.edit_wave_file = QLineEdit()
        self.edit_wave_file.setPlaceholderText("e.g. awavedat.inp")
        wm_layout.addWidget(self.edit_wave_file)

        wave_layout.addWidget(wave_mode_box)

        # Wave table (either discrete classes or time series)
        self.table_waves = QTableWidget()
        self.table_waves.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        wave_layout.addWidget(self.table_waves)

        btn_wave_layout = QHBoxLayout()
        btn_add_w = QPushButton("Add Wave Row")
        btn_add_w.clicked.connect(self._add_wave_row)
        btn_del_w = QPushButton("Delete Wave Row")
        btn_del_w.clicked.connect(self._del_wave_row)
        btn_wave_layout.addWidget(btn_add_w)
        btn_wave_layout.addWidget(btn_del_w)
        wave_layout.addLayout(btn_wave_layout)

        sub_tabs.addTab(tab_wave, "Waves")

        # 2. Tides Sub-Tab
        tab_tide = QWidget()
        tide_layout = QVBoxLayout(tab_tide)
        tide_layout.addWidget(QLabel("<b>Tidal Water Level and Currents at x=0</b>"))

        self.table_tide = QTableWidget()
        self.table_tide.setColumnCount(4)
        self.table_tide.setHorizontalHeaderLabels(["Time (s)", "Vlong (m/s)", "Vcross (m/s)", "Tide Level Ws (m)"])
        self.table_tide.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tide_layout.addWidget(self.table_tide)

        btn_tide_layout = QHBoxLayout()
        btn_add_t = QPushButton("Add Tide Row")
        btn_add_t.clicked.connect(self._add_tide_row)
        btn_del_t = QPushButton("Delete Tide Row")
        btn_del_t.clicked.connect(self._del_tide_row)
        btn_tide_layout.addWidget(btn_add_t)
        btn_tide_layout.addWidget(btn_del_t)
        tide_layout.addLayout(btn_tide_layout)

        sub_tabs.addTab(tab_tide, "Tides & Currents")

        # 3. Storm Surge Sub-Tab
        tab_storm = QWidget()
        storm_layout = QVBoxLayout(tab_storm)
        storm_layout.addWidget(QLabel("<b>Storm Surge Set-up Time Series (Hstorm)</b>"))

        self.table_storm = QTableWidget()
        self.table_storm.setColumnCount(2)
        self.table_storm.setHorizontalHeaderLabels(["Time (s)", "Storm Surge Set-up (m)"])
        self.table_storm.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        storm_layout.addWidget(self.table_storm)

        btn_storm_layout = QHBoxLayout()
        btn_add_s = QPushButton("Add Storm Row")
        btn_add_s.clicked.connect(self._add_storm_row)
        btn_del_s = QPushButton("Delete Storm Row")
        btn_del_s.clicked.connect(self._del_storm_row)
        btn_storm_layout.addWidget(btn_add_s)
        btn_storm_layout.addWidget(btn_del_s)
        storm_layout.addLayout(btn_storm_layout)

        sub_tabs.addTab(tab_storm, "Storm Surge")

        main_layout.addWidget(sub_tabs)
        self._update_wave_table_headers()

    def _on_igolf_changed(self) -> None:
        self._update_wave_table_headers()

    def _update_wave_table_headers(self) -> None:
        mode = self.combo_igolf.currentData()
        if mode == 1:
            self.table_waves.setColumnCount(5)
            self.table_waves.setHorizontalHeaderLabels(["Wave H (m)", "Period T (s)", "Prob P (0-1)", "Angle Teta (deg)", "Wind W10 (m/s)"])
        else:
            self.table_waves.setColumnCount(5)
            self.table_waves.setHorizontalHeaderLabels(["Time (s)", "Hrms (m)", "Period T (s)", "Angle Teta (deg)", "Wind W10 (m/s)"])

    def load_config(self, config: SimulationConfig) -> None:
        # Mode
        idx = self.combo_igolf.findData(config.igolf)
        if idx >= 0:
            self.combo_igolf.setCurrentIndex(idx)
        self.spin_nh.setValue(config.nh_wave_classes)
        self.edit_wave_file.setText(config.wave_file or "")

        self._update_wave_table_headers()

        # Waves
        self.table_waves.blockSignals(True)
        if config.igolf == 1:
            self.table_waves.setRowCount(len(config.wave_classes))
            for r, wc in enumerate(config.wave_classes):
                self.table_waves.setItem(r, 0, QTableWidgetItem(f"{wc.h:.3f}"))
                self.table_waves.setItem(r, 1, QTableWidgetItem(f"{wc.t:.2f}"))
                self.table_waves.setItem(r, 2, QTableWidgetItem(f"{wc.p:.4f}"))
                self.table_waves.setItem(r, 3, QTableWidgetItem(f"{wc.teta:.2f}"))
                self.table_waves.setItem(r, 4, QTableWidgetItem(f"{wc.w10:.2f}"))
        else:
            self.table_waves.setRowCount(len(config.wave_series))
            for r, wt in enumerate(config.wave_series):
                self.table_waves.setItem(r, 0, QTableWidgetItem(f"{wt.time:.1f}"))
                self.table_waves.setItem(r, 1, QTableWidgetItem(f"{wt.hrms:.3f}"))
                self.table_waves.setItem(r, 2, QTableWidgetItem(f"{wt.t:.2f}"))
                self.table_waves.setItem(r, 3, QTableWidgetItem(f"{wt.teta:.2f}"))
                self.table_waves.setItem(r, 4, QTableWidgetItem(f"{wt.w10:.2f}"))
        self.table_waves.blockSignals(False)

        # Tides
        self.table_tide.blockSignals(True)
        self.table_tide.setRowCount(len(config.tide_series))
        for r, tp in enumerate(config.tide_series):
            self.table_tide.setItem(r, 0, QTableWidgetItem(f"{tp.time:.1f}"))
            self.table_tide.setItem(r, 1, QTableWidgetItem(f"{tp.vnul1:.3f}"))
            self.table_tide.setItem(r, 2, QTableWidgetItem(f"{tp.vnul2:.3f}"))
            self.table_tide.setItem(r, 3, QTableWidgetItem(f"{tp.ws:.3f}"))
        self.table_tide.blockSignals(False)

        # Storm
        self.table_storm.blockSignals(True)
        self.table_storm.setRowCount(len(config.storm_series))
        for r, sp in enumerate(config.storm_series):
            self.table_storm.setItem(r, 0, QTableWidgetItem(f"{sp.time:.1f}"))
            self.table_storm.setItem(r, 1, QTableWidgetItem(f"{sp.hstorm:.3f}"))
        self.table_storm.blockSignals(False)

    def update_config(self, config: SimulationConfig) -> None:
        config.igolf = self.combo_igolf.currentData()
        config.nh_wave_classes = self.spin_nh.value()
        config.wave_file = self.edit_wave_file.text().strip() or None

        # Waves
        if config.igolf == 1:
            classes = []
            for r in range(self.table_waves.rowCount()):
                try:
                    h = float(self.table_waves.item(r, 0).text())
                    t = float(self.table_waves.item(r, 1).text())
                    p = float(self.table_waves.item(r, 2).text())
                    teta = float(self.table_waves.item(r, 3).text())
                    w10 = float(self.table_waves.item(r, 4).text()) if self.table_waves.item(r, 4) else 0.0
                    classes.append(WaveClass(h=h, t=t, p=p, teta=teta, w10=w10))
                except Exception:
                    continue
            if classes:
                config.wave_classes = classes
        else:
            series = []
            for r in range(self.table_waves.rowCount()):
                try:
                    time_val = float(self.table_waves.item(r, 0).text())
                    hrms = float(self.table_waves.item(r, 1).text())
                    t = float(self.table_waves.item(r, 2).text())
                    teta = float(self.table_waves.item(r, 3).text())
                    w10 = float(self.table_waves.item(r, 4).text()) if self.table_waves.item(r, 4) else 0.0
                    series.append(WaveTimePoint(time=time_val, hrms=hrms, t=t, teta=teta, w10=w10))
                except Exception:
                    continue
            if series:
                config.wave_series = series

        # Tides
        tides = []
        for r in range(self.table_tide.rowCount()):
            try:
                t = float(self.table_tide.item(r, 0).text())
                v1 = float(self.table_tide.item(r, 1).text())
                v2 = float(self.table_tide.item(r, 2).text())
                ws = float(self.table_tide.item(r, 3).text())
                tides.append(TidePoint(time=t, vnul1=v1, vnul2=v2, ws=ws))
            except Exception:
                continue
        if tides:
            config.tide_series = tides

        # Storm
        storms = []
        for r in range(self.table_storm.rowCount()):
            try:
                t = float(self.table_storm.item(r, 0).text())
                hs = float(self.table_storm.item(r, 1).text())
                storms.append(StormPoint(time=t, hstorm=hs))
            except Exception:
                continue
        if storms:
            config.storm_series = storms

    def _add_wave_row(self) -> None:
        r = self.table_waves.rowCount()
        self.table_waves.insertRow(r)
        self.table_waves.setItem(r, 0, QTableWidgetItem("1.0"))
        self.table_waves.setItem(r, 1, QTableWidgetItem("5.0"))
        self.table_waves.setItem(r, 2, QTableWidgetItem("1.0"))
        self.table_waves.setItem(r, 3, QTableWidgetItem("0.0"))
        self.table_waves.setItem(r, 4, QTableWidgetItem("0.0"))

    def _del_wave_row(self) -> None:
        r = self.table_waves.currentRow()
        if r >= 0:
            self.table_waves.removeRow(r)

    def _add_tide_row(self) -> None:
        r = self.table_tide.rowCount()
        self.table_tide.insertRow(r)
        self.table_tide.setItem(r, 0, QTableWidgetItem("0.0"))
        self.table_tide.setItem(r, 1, QTableWidgetItem("0.5"))
        self.table_tide.setItem(r, 2, QTableWidgetItem("0.0"))
        self.table_tide.setItem(r, 3, QTableWidgetItem("0.0"))

    def _del_tide_row(self) -> None:
        r = self.table_tide.currentRow()
        if r >= 0:
            self.table_tide.removeRow(r)

    def _add_storm_row(self) -> None:
        r = self.table_storm.rowCount()
        self.table_storm.insertRow(r)
        self.table_storm.setItem(r, 0, QTableWidgetItem("0.0"))
        self.table_storm.setItem(r, 1, QTableWidgetItem("0.0"))

    def _del_storm_row(self) -> None:
        r = self.table_storm.currentRow()
        if r >= 0:
            self.table_storm.removeRow(r)
