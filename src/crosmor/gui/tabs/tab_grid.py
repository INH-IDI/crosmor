"""Tab for editing bathymetry profile and computational grid configuration."""

from typing import List, Optional
import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QHeaderView,
)
import pyqtgraph as pg

from crosmor.models.config import SimulationConfig, BathymetrySection
from crosmor.hydrodynamics.wave_solver import build_grid


class TabGrid(QWidget):
    """Bathymetry & Grid Editor with interactive preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)

        # Left Column: Table and options
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)

        # Settings
        grp_settings = QGroupBox("Grid & Boundary Settings")
        set_layout = QVBoxLayout(grp_settings)

        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(QLabel("Cutoff Depth Hgrens (m):"))
        self.spin_hgrens = QDoubleSpinBox()
        self.spin_hgrens.setRange(0.01, 10.0)
        self.spin_hgrens.setValue(0.25)
        self.spin_hgrens.setSingleStep(0.05)
        h_layout1.addWidget(self.spin_hgrens)
        set_layout.addLayout(h_layout1)

        self.chk_beach = QCheckBox("Activate Beach Runup / Dune Module (istrand)")
        self.chk_beach.setChecked(False)
        set_layout.addWidget(self.chk_beach)

        left_layout.addWidget(grp_settings)

        # Table
        grp_table = QGroupBox("Bathymetry Points (X, Depth to MSL, Sub-steps)")
        tab_layout = QVBoxLayout(grp_table)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["X (m)", "HMSL (m)", "Sub-steps (Nstap)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_table_changed)
        tab_layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_add = QPushButton("Add Point")
        btn_add.clicked.connect(self._add_row)
        btn_del = QPushButton("Delete Selected")
        btn_del.clicked.connect(self._delete_row)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_del)
        tab_layout.addLayout(btn_layout)

        left_layout.addWidget(grp_table)
        layout.addWidget(left_box, stretch=1)

        # Right Column: Profile preview plot
        right_box = QWidget()
        right_layout = QVBoxLayout(right_box)
        right_layout.addWidget(QLabel("<b>Initial Bathymetry Profile Preview</b>"))

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Elevation Z (m MSL)")
        self.plot_widget.setLabel("bottom", "Cross-shore X (m)")

        self.curve_bed = self.plot_widget.plot(pen=pg.mkPen(color="saddlebrown", width=3), name="Bed Level")
        self.water_line = pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen(color="dodgerblue", width=1.5, style=pg.QtCore.Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.water_line)

        right_layout.addWidget(self.plot_widget)
        layout.addWidget(right_box, stretch=1)

    def load_config(self, config: SimulationConfig) -> None:
        """Populate widgets from config."""
        self.spin_hgrens.setValue(config.hgrens)
        self.chk_beach.setChecked(config.istrand == 1)

        self.table.blockSignals(True)
        self.table.setRowCount(len(config.bathymetry))
        for r, sec in enumerate(config.bathymetry):
            self.table.setItem(r, 0, QTableWidgetItem(f"{sec.x:.2f}"))
            self.table.setItem(r, 1, QTableWidgetItem(f"{sec.hmsl:.3f}"))
            nstap_str = str(sec.nstap) if sec.nstap is not None else ""
            self.table.setItem(r, 2, QTableWidgetItem(nstap_str))
        self.table.blockSignals(False)
        self._update_plot()

    def update_config(self, config: SimulationConfig) -> None:
        """Write widget state back into config."""
        config.hgrens = self.spin_hgrens.value()
        config.istrand = 1 if self.chk_beach.isChecked() else 0

        sections = []
        n_rows = self.table.rowCount()
        for r in range(n_rows):
            try:
                x = float(self.table.item(r, 0).text())
                hmsl = float(self.table.item(r, 1).text())
                nstap_item = self.table.item(r, 2)
                nstap = int(nstap_item.text()) if nstap_item and nstap_item.text().strip() else None
                sections.append(BathymetrySection(x=x, hmsl=hmsl, nstap=nstap))
            except Exception:
                continue

        if sections:
            config.bathymetry = sections

    def _add_row(self) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem("0.0"))
        self.table.setItem(r, 1, QTableWidgetItem("-10.0"))
        self.table.setItem(r, 2, QTableWidgetItem("10"))
        self._update_plot()

    def _delete_row(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self._update_plot()

    def _on_table_changed(self) -> None:
        self._update_plot()

    def _update_plot(self) -> None:
        x_vals = []
        z_vals = []
        for r in range(self.table.rowCount()):
            try:
                x = float(self.table.item(r, 0).text())
                # In CROSMOR .inp, hmsl is negative below MSL (e.g. -10m is 10m deep)
                hmsl = float(self.table.item(r, 1).text())
                x_vals.append(x)
                z_vals.append(hmsl)
            except Exception:
                continue

        if len(x_vals) >= 2:
            self.curve_bed.setData(x_vals, z_vals)
