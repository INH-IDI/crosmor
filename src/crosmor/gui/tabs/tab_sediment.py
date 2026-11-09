"""Tab for editing sediment fractions, calibration factors, morphology and volume boundaries."""

from typing import List
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QLineEdit,
    QHeaderView,
)

from crosmor.models.config import SimulationConfig, SedimentProfilePoint


class TabSediment(QWidget):
    """Sediment properties and morphological calibration editor."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 1. Physical and Numerical Parameters
        grp_params = QGroupBox("Sediment & Morphodynamic Parameters")
        grid_p = QGridLayout(grp_params)

        grid_p.addWidget(QLabel("Simulation Duration T-END (s):"), 0, 0)
        self.spin_tend = QDoubleSpinBox()
        self.spin_tend.setRange(1.0, 1.0e8)
        self.spin_tend.setValue(86400.0)
        self.spin_tend.setSingleStep(3600.0)
        grid_p.addWidget(self.spin_tend, 0, 1)

        grid_p.addWidget(QLabel("Time Step Factor (factime):"), 0, 2)
        self.spin_factime = QDoubleSpinBox()
        self.spin_factime.setRange(0.01, 10.0)
        self.spin_factime.setValue(0.25)
        self.spin_factime.setSingleStep(0.05)
        grid_p.addWidget(self.spin_factime, 0, 3)

        grid_p.addWidget(QLabel("Active Layer Thickness dl (m):"), 1, 0)
        self.spin_dl = QDoubleSpinBox()
        self.spin_dl.setRange(0.05, 5.0)
        self.spin_dl.setValue(0.25)
        self.spin_dl.setSingleStep(0.05)
        grid_p.addWidget(self.spin_dl, 1, 1)

        grid_p.addWidget(QLabel("Bed Porosity (por):"), 1, 2)
        self.spin_por = QDoubleSpinBox()
        self.spin_por.setRange(0.1, 0.9)
        self.spin_por.setValue(0.4)
        self.spin_por.setSingleStep(0.05)
        grid_p.addWidget(self.spin_por, 1, 3)

        grid_p.addWidget(QLabel("Sediment Density rhos (kg/m3):"), 2, 0)
        self.spin_rhos = QDoubleSpinBox()
        self.spin_rhos.setRange(1000.0, 4000.0)
        self.spin_rhos.setValue(2650.0)
        self.spin_rhos.setSingleStep(50.0)
        grid_p.addWidget(self.spin_rhos, 2, 1)

        grid_p.addWidget(QLabel("Water Temp (C) & Salinity (o/oo):"), 2, 2)
        h_temp_sal = QHBoxLayout()
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(-2.0, 40.0)
        self.spin_temp.setValue(15.0)
        self.spin_sal = QDoubleSpinBox()
        self.spin_sal.setRange(0.0, 45.0)
        self.spin_sal.setValue(0.0)
        h_temp_sal.addWidget(self.spin_temp)
        h_temp_sal.addWidget(self.spin_sal)
        grid_p.addLayout(h_temp_sal, 2, 3)

        # Calibration multipliers
        grid_p.addWidget(QLabel("Calib Factor Bedload (facbed):"), 3, 0)
        self.spin_facbed = QDoubleSpinBox()
        self.spin_facbed.setRange(0.001, 100.0)
        self.spin_facbed.setValue(1.0)
        self.spin_facbed.setSingleStep(0.1)
        grid_p.addWidget(self.spin_facbed, 3, 1)

        grid_p.addWidget(QLabel("Calib Factor Susp-Current (facsusc):"), 3, 2)
        self.spin_facsusc = QDoubleSpinBox()
        self.spin_facsusc.setRange(0.001, 100.0)
        self.spin_facsusc.setValue(1.0)
        self.spin_facsusc.setSingleStep(0.1)
        grid_p.addWidget(self.spin_facsusc, 3, 3)

        grid_p.addWidget(QLabel("Calib Factor Susp-Wave (facsusw):"), 4, 0)
        self.spin_facsusw = QDoubleSpinBox()
        self.spin_facsusw.setRange(0.001, 100.0)
        self.spin_facsusw.setValue(1.0)
        self.spin_facsusw.setSingleStep(0.1)
        grid_p.addWidget(self.spin_facsusw, 4, 1)

        grid_p.addWidget(QLabel("Bed / Grain Smoothing Factors:"), 4, 2)
        h_smooth = QHBoxLayout()
        self.spin_facbed_sm = QDoubleSpinBox()
        self.spin_facbed_sm.setRange(0.0, 100.0)
        self.spin_facbed_sm.setValue(0.2)
        self.spin_facgrain_sm = QDoubleSpinBox()
        self.spin_facgrain_sm.setRange(0.0, 100.0)
        self.spin_facgrain_sm.setValue(1.0)
        h_smooth.addWidget(self.spin_facbed_sm)
        h_smooth.addWidget(self.spin_facgrain_sm)
        grid_p.addLayout(h_smooth, 4, 3)

        layout.addWidget(grp_params)

        # 2. Fractions & Diameters
        grp_fractions = QGroupBox("Grain Diameters (m) & Spatial Sediments Table")
        f_layout = QVBoxLayout(grp_fractions)

        h_diam = QHBoxLayout()
        h_diam.addWidget(QLabel("Grain Diameters (space-separated, m):"))
        self.edit_diams = QLineEdit("0.0002")
        h_diam.addWidget(self.edit_diams)
        f_layout.addLayout(h_diam)

        self.table_sed = QTableWidget()
        self.table_sed.setColumnCount(5)
        self.table_sed.setHorizontalHeaderLabels(["X (m)", "RC (m)", "RW (m)", "Mud (%)", "Undertow Factor (FRIP)"])
        self.table_sed.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        f_layout.addWidget(self.table_sed)

        btn_sed = QHBoxLayout()
        btn_add = QPushButton("Add Sediment Station")
        btn_add.clicked.connect(self._add_sed_row)
        btn_del = QPushButton("Delete Station")
        btn_del.clicked.connect(self._del_sed_row)
        btn_sed.addWidget(btn_add)
        btn_sed.addWidget(btn_del)
        f_layout.addLayout(btn_sed)

        layout.addWidget(grp_fractions)

        # 3. Volume Contours
        grp_vol = QGroupBox("Volume Integration Contours")
        v_layout = QHBoxLayout(grp_vol)

        v_layout.addWidget(QLabel("Depth Contours (m):"))
        self.edit_ndc = QLineEdit("-6. -4. -2. -1. 2.")
        v_layout.addWidget(self.edit_ndc)

        v_layout.addWidget(QLabel("X Contours (m):"))
        self.edit_ndcx = QLineEdit("0. 500. 700. 3000. 4000.")
        v_layout.addWidget(self.edit_ndcx)

        layout.addWidget(grp_vol)

    def load_config(self, config: SimulationConfig) -> None:
        m = config.morpho
        self.spin_tend.setValue(m.tend)
        self.spin_factime.setValue(m.factime)
        self.spin_dl.setValue(m.dl)
        self.spin_por.setValue(m.por)
        self.spin_rhos.setValue(m.rhos)
        self.spin_temp.setValue(m.water_temp)
        self.spin_sal.setValue(m.salinity)

        self.spin_facbed.setValue(m.facbed)
        self.spin_facsusc.setValue(m.facsusc)
        self.spin_facsusw.setValue(m.facsusw)
        self.spin_facbed_sm.setValue(m.facbed_smooth)
        self.spin_facgrain_sm.setValue(m.facgrain_smooth)

        self.edit_diams.setText(" ".join(str(d) for d in m.grain_diameters))

        self.table_sed.blockSignals(True)
        self.table_sed.setRowCount(len(config.sediment_points))
        for r, sp in enumerate(config.sediment_points):
            self.table_sed.setItem(r, 0, QTableWidgetItem(f"{sp.x:.2f}"))
            self.table_sed.setItem(r, 1, QTableWidgetItem(f"{sp.rc:.5f}"))
            self.table_sed.setItem(r, 2, QTableWidgetItem(f"{sp.rw:.5f}"))
            self.table_sed.setItem(r, 3, QTableWidgetItem(f"{sp.pmud:.2f}"))
            self.table_sed.setItem(r, 4, QTableWidgetItem(f"{sp.frip:.2f}"))
        self.table_sed.blockSignals(False)

        self.edit_ndc.setText(" ".join(str(c) for c in m.depth_contours))
        self.edit_ndcx.setText(" ".join(str(x) for x in m.x_contours))

    def update_config(self, config: SimulationConfig) -> None:
        m = config.morpho
        m.tend = self.spin_tend.value()
        m.factime = self.spin_factime.value()
        m.dl = self.spin_dl.value()
        m.por = self.spin_por.value()
        m.rhos = self.spin_rhos.value()
        m.water_temp = self.spin_temp.value()
        m.salinity = self.spin_sal.value()

        m.facbed = self.spin_facbed.value()
        m.facsusc = self.spin_facsusc.value()
        m.facsusw = self.spin_facsusw.value()
        m.facbed_smooth = self.spin_facbed_sm.value()
        m.facgrain_smooth = self.spin_facgrain_sm.value()

        try:
            m.grain_diameters = [float(x) for x in self.edit_diams.text().replace(",", " ").split()]
        except Exception:
            m.grain_diameters = [0.0002]

        nfr = len(m.grain_diameters)
        points = []
        for r in range(self.table_sed.rowCount()):
            try:
                x = float(self.table_sed.item(r, 0).text())
                rc = float(self.table_sed.item(r, 1).text())
                rw = float(self.table_sed.item(r, 2).text())
                pmud = float(self.table_sed.item(r, 3).text())
                frip = float(self.table_sed.item(r, 4).text())
                fracs = [1.0 / nfr] * nfr
                points.append(SedimentProfilePoint(x=x, rc=rc, rw=rw, pmud=pmud, frip=frip, fractions=fracs))
            except Exception:
                continue
        if points:
            config.sediment_points = points

        try:
            m.depth_contours = [float(c) for c in self.edit_ndc.text().replace(",", " ").split()]
        except Exception:
            m.depth_contours = []

        try:
            m.x_contours = [float(x) for x in self.edit_ndcx.text().replace(",", " ").split()]
        except Exception:
            m.x_contours = []

    def _add_sed_row(self) -> None:
        r = self.table_sed.rowCount()
        self.table_sed.insertRow(r)
        self.table_sed.setItem(r, 0, QTableWidgetItem("0.0"))
        self.table_sed.setItem(r, 1, QTableWidgetItem("0.0"))
        self.table_sed.setItem(r, 2, QTableWidgetItem("0.0"))
        self.table_sed.setItem(r, 3, QTableWidgetItem("0.0"))
        self.table_sed.setItem(r, 4, QTableWidgetItem("1.0"))

    def _del_sed_row(self) -> None:
        r = self.table_sed.currentRow()
        if r >= 0:
            self.table_sed.removeRow(r)
