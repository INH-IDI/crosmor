"""Multi-panel synchronized real-time visualization dashboard using PyQtGraph."""

from typing import Optional
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
    QLabel,
    QFrame,
)
from PyQt6.QtCore import Qt
from crosmor.models.results import TimeStepResult


# Configure PyQtGraph styles for high clarity
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOption("antialias", True)


class MorphoDashboard(QWidget):
    """Real-time synchronized 4-panel dashboard displaying:
    1. Cross-shore bed elevation Z_b(x) and water surface eta(x)
    2. Wave height H_rms(x) and currents (Longshore V_mix & Undertow U_cr)
    3. Sediment transport rates (Cross-shore & Longshore)
    4. Cumulative bed level change dz(x) and grain size D50
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._first_render = True
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # Top Control Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        lbl_display = QLabel("Display Layers:")
        lbl_display.setStyleSheet("font-weight: bold; color: #334155;")
        toolbar.addWidget(lbl_display)

        self.chk_init_bed = QCheckBox("Initial Bed (t=0)")
        self.chk_init_bed.setChecked(True)
        self.chk_init_bed.toggled.connect(self._toggle_curves)
        toolbar.addWidget(self.chk_init_bed)

        self.chk_water_level = QCheckBox("Water Level")
        self.chk_water_level.setChecked(True)
        self.chk_water_level.toggled.connect(self._toggle_curves)
        toolbar.addWidget(self.chk_water_level)

        self.chk_transport_components = QCheckBox("Transport Components")
        self.chk_transport_components.setChecked(True)
        self.chk_transport_components.toggled.connect(self._toggle_curves)
        toolbar.addWidget(self.chk_transport_components)

        self.chk_concentration = QCheckBox("Concentration (m³/m³)")
        self.chk_concentration.setChecked(True)
        self.chk_concentration.toggled.connect(self._toggle_curves)
        toolbar.addWidget(self.chk_concentration)

        toolbar.addStretch()

        self.btn_autorange = QPushButton("⟲ Auto Range")
        self.btn_autorange.setToolTip("Reset all plot views to fit full profile data")
        self.btn_autorange.clicked.connect(self.auto_range)
        toolbar.addWidget(self.btn_autorange)

        main_layout.addLayout(toolbar)

        # Graphics Layout Widget with 4 rows
        self.glw = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.glw)

        # ---------------------------------------------------------------------
        # Plot 1: Bed Level & Water Surface
        # ---------------------------------------------------------------------
        self.p1 = self.glw.addPlot(row=0, col=0)
        self.p1.setTitle("<b>Cross-shore Profile & Water Surface</b>", color="#0f172a", size="11pt")
        self.p1.setLabel("left", "Elevation (m NAP)")
        self.p1.showGrid(x=True, y=True, alpha=0.3)
        self.leg1 = self.p1.addLegend(offset=(10, 10))

        pen_init = pg.mkPen(color="#94a3b8", width=1.5, style=Qt.PenStyle.DashLine)
        pen_bed = pg.mkPen(color="#d97706", width=2.5)
        pen_water = pg.mkPen(color="#0284c7", width=1.5)

        self.c_z_init = self.p1.plot([], [], pen=pen_init, name="Initial Bed z_0")
        self.c_z_bed = self.p1.plot([], [], pen=pen_bed, name="Active Bed z(t)")
        self.c_water = self.p1.plot([], [], pen=pen_water, name="Water Level η(t)")

        # ---------------------------------------------------------------------
        # Plot 2: Wave Heights & Currents
        # ---------------------------------------------------------------------
        self.p2 = self.glw.addPlot(row=1, col=0)
        self.p2.setTitle("<b>Hydrodynamics: Wave Height & Mean Currents</b>", color="#0f172a", size="11pt")
        self.p2.setLabel("left", "Wave H (m) / Vel (m/s)")
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        self.p2.setXLink(self.p1)
        self.leg2 = self.p2.addLegend(offset=(10, 10))

        pen_hrms = pg.mkPen(color="#0369a1", width=2.0)
        pen_h13 = pg.mkPen(color="#0284c7", width=1.5, style=Qt.PenStyle.DashLine)
        pen_vmix = pg.mkPen(color="#15803d", width=2.0)
        pen_ucr = pg.mkPen(color="#b91c1c", width=1.5, style=Qt.PenStyle.DotLine)

        self.c_hrms = self.p2.plot([], [], pen=pen_hrms, name="H_rms (m)")
        self.c_h13 = self.p2.plot([], [], pen=pen_h13, name="H_1/3 (m)")
        self.c_vmix = self.p2.plot([], [], pen=pen_vmix, name="Longshore V_mix (m/s)")
        self.c_ucr = self.p2.plot([], [], pen=pen_ucr, name="Undertow U_cr (m/s)")

        # Zero reference line for currents
        zero_line_p2 = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color="#cbd5e1", width=1, style=Qt.PenStyle.DashLine))
        self.p2.addItem(zero_line_p2)

        # ---------------------------------------------------------------------
        # Plot 3: Sediment Transport Rates & Concentration
        # ---------------------------------------------------------------------
        self.p3 = self.glw.addPlot(row=2, col=0)
        self.p3.setTitle("<b>Sediment Transport Rates & Concentration</b>", color="#0f172a", size="11pt")
        self.p3.setLabel("left", "Transport (kg/(s·m))")
        self.p3.showGrid(x=True, y=True, alpha=0.3)
        self.p3.setXLink(self.p1)
        self.leg3 = self.p3.addLegend(offset=(10, 10))

        pen_st_cross = pg.mkPen(color="#b91c1c", width=2.0)
        pen_sb_cross = pg.mkPen(color="#f97316", width=1.2, style=Qt.PenStyle.DashLine)
        pen_ss_cross = pg.mkPen(color="#7e22ce", width=1.2, style=Qt.PenStyle.DashLine)
        pen_st_long = pg.mkPen(color="#047857", width=2.0)

        self.c_st_cross = self.p3.plot([], [], pen=pen_st_cross, name="Cross-shore Total S_t")
        self.c_sb_cross = self.p3.plot([], [], pen=pen_sb_cross, name="Cross-shore Bedload S_b")
        self.c_ss_cross = self.p3.plot([], [], pen=pen_ss_cross, name="Cross-shore Suspended S_s")
        self.c_st_long = self.p3.plot([], [], pen=pen_st_long, name="Longshore Total S_t,long")

        zero_line_p3 = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color="#cbd5e1", width=1, style=Qt.PenStyle.DashLine))
        self.p3.addItem(zero_line_p3)

        # Secondary right axis for volumetric sediment concentration (m³ sed / m³ agua)
        self.vb_conc = pg.ViewBox()
        self.p3.showAxis("right")
        self.p3.scene().addItem(self.vb_conc)
        self.p3.getAxis("right").linkToView(self.vb_conc)
        self.vb_conc.setXLink(self.p3)
        self.p3.getAxis("right").setLabel("Concentration (m³ sed / m³ agua)", color="#7c3aed")

        self.p3.vb.sigResized.connect(self._sync_conc_view)

        pen_conc = pg.mkPen(color="#7c3aed", width=2.2)
        pen_cmean = pg.mkPen(color="#a855f7", width=1.5, style=Qt.PenStyle.DashLine)

        self.c_ca = pg.PlotCurveItem([], [], pen=pen_conc, name="Bed Conc Ca (m³/m³)")
        self.c_cmean = pg.PlotCurveItem([], [], pen=pen_cmean, name="Mean Conc C_mean (m³/m³)")
        self.vb_conc.addItem(self.c_ca)
        self.vb_conc.addItem(self.c_cmean)
        self.leg3.addItem(self.c_ca, "Bed Conc Ca (m³/m³)")
        self.leg3.addItem(self.c_cmean, "Mean Conc C_mean (m³/m³)")

        # ---------------------------------------------------------------------
        # Plot 4: Cumulative Bed Level Change dz
        # ---------------------------------------------------------------------
        self.p4 = self.glw.addPlot(row=3, col=0)
        self.p4.setTitle("<b>Morphodynamics: Cumulative Bed Level Change Δz</b>", color="#0f172a", size="11pt")
        self.p4.setLabel("left", "Δz (m)")
        self.p4.setLabel("bottom", "Cross-shore Distance x (m)")
        self.p4.showGrid(x=True, y=True, alpha=0.3)
        self.p4.setXLink(self.p1)
        self.leg4 = self.p4.addLegend(offset=(10, 10))

        pen_dz = pg.mkPen(color="#0284c7", width=2.0)
        self.c_dz = self.p4.plot([], [], pen=pen_dz, name="Cumulative Δz(t)")

        zero_line_p4 = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color="#cbd5e1", width=1, style=Qt.PenStyle.DashLine))
        self.p4.addItem(zero_line_p4)

    def _sync_conc_view(self) -> None:
        """Keep right secondary ViewBox perfectly synchronized with Plot 3 geometry."""
        self.vb_conc.setGeometry(self.p3.vb.sceneBoundingRect())
        self.vb_conc.linkedViewChanged(self.p3.vb, self.vb_conc.XAxis)

    def _toggle_curves(self) -> None:
        """Handle layer visibility toggles."""
        self.c_z_init.setVisible(self.chk_init_bed.isChecked())
        self.c_water.setVisible(self.chk_water_level.isChecked())
        show_trans_parts = self.chk_transport_components.isChecked()
        self.c_sb_cross.setVisible(show_trans_parts)
        self.c_ss_cross.setVisible(show_trans_parts)
        show_conc = self.chk_concentration.isChecked()
        self.c_ca.setVisible(show_conc)
        self.c_cmean.setVisible(show_conc)
        self.p3.showAxis("right", show_conc)

    def update_data(self, step_res: TimeStepResult) -> None:
        """Update all 4 panels with data from the given simulation time step."""
        if not step_res.x:
            return

        x = np.asarray(step_res.x, dtype=np.float64)

        # Plot 1: Profile & Water level
        z_init = np.asarray(step_res.morpho.z_initial, dtype=np.float64)
        z_curr = np.asarray(step_res.morpho.z_current, dtype=np.float64)
        eta = np.asarray(step_res.water_level_eta, dtype=np.float64)

        if len(z_init) == len(x):
            self.c_z_init.setData(x, z_init)
        if len(z_curr) == len(x):
            self.c_z_bed.setData(x, z_curr)
        if len(eta) == len(x):
            self.c_water.setData(x, eta)

        # Plot 2: Hydrodynamics
        hrms = np.asarray(step_res.hydro.hrms, dtype=np.float64)
        h13 = np.asarray(step_res.hydro.h13, dtype=np.float64)
        vmix = np.asarray(step_res.hydro.v_mix, dtype=np.float64)
        ucr = np.asarray(step_res.hydro.ucr_mean, dtype=np.float64)

        if len(hrms) == len(x):
            self.c_hrms.setData(x, hrms)
        if len(h13) == len(x):
            self.c_h13.setData(x, h13)
        if len(vmix) == len(x):
            self.c_vmix.setData(x, vmix)
        if len(ucr) == len(x):
            self.c_ucr.setData(x, ucr)

        # Plot 3: Transport
        st_c = np.asarray(step_res.transport.st_cross, dtype=np.float64)
        sb_c = np.asarray(step_res.transport.sb_cross, dtype=np.float64)
        ss_c = np.asarray(step_res.transport.ss_cross, dtype=np.float64)
        st_l = np.asarray(step_res.transport.st_long, dtype=np.float64)

        if len(st_c) == len(x):
            self.c_st_cross.setData(x, st_c)
        if len(sb_c) == len(x):
            self.c_sb_cross.setData(x, sb_c)
        if len(ss_c) == len(x):
            self.c_ss_cross.setData(x, ss_c)
        if len(st_l) == len(x):
            self.c_st_long.setData(x, st_l)

        # Plot 3 secondary axis: Volumetric concentration (m³ sed / m³ agua)
        c_ca = np.asarray(step_res.transport.c_vol or step_res.transport.ca_ref, dtype=np.float64)
        c_mean = np.asarray(step_res.transport.c_mean_vol, dtype=np.float64)

        if len(c_ca) == len(x):
            self.c_ca.setData(x, c_ca)
        if len(c_mean) == len(x):
            self.c_cmean.setData(x, c_mean)

        self._sync_conc_view()
        if len(c_ca) == len(x) or len(c_mean) == len(x):
            self.vb_conc.enableAutoRange(axis=pg.ViewBox.YAxis)

        # Plot 4: Bed Change
        dz = np.asarray(step_res.morpho.dz_total, dtype=np.float64)
        if len(dz) == len(x):
            self.c_dz.setData(x, dz)

        if self._first_render:
            self.auto_range()
            self._first_render = False

    def auto_range(self) -> None:
        """Autorange all subplots and secondary views."""
        self.p1.autoRange()
        self.p2.autoRange()
        self.p3.autoRange()
        self.vb_conc.autoRange()
        self.p4.autoRange()
        self._sync_conc_view()

    def clear(self) -> None:
        """Clear all curve data."""
        self.c_z_init.clear()
        self.c_z_bed.clear()
        self.c_water.clear()
        self.c_hrms.clear()
        self.c_h13.clear()
        self.c_vmix.clear()
        self.c_ucr.clear()
        self.c_st_cross.clear()
        self.c_sb_cross.clear()
        self.c_ss_cross.clear()
        self.c_st_long.clear()
        self.c_ca.clear()
        self.c_cmean.clear()
        self.c_dz.clear()
        self._first_render = True
