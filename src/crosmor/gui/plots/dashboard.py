"""Multi-panel synchronized real-time visualization dashboard using PyQtGraph."""

from typing import Optional, List
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
    QComboBox,
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
    """Real-time synchronized multi-panel dashboard displaying:
    1. Cross-shore bed elevation Z_b(x) and water surface eta(x)
    2. Wave height H_rms(x) and currents (Longshore V_mix & Undertow U_cr)
    3. Sediment transport rates (Cross-shore & Longshore)
    4. Volumetric sediment concentration Ca(x) and C_mean(x)
    5. Cumulative bed level change dz(x) and morphological evolution
    
    Supports dynamic panel selection (e.g. 3 of 5) to prevent vertical distortion.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._first_render = True
        self._block_preset_signal = False
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # ---------------------------------------------------------------------
        # Top Control Toolbar: Panel Selection and Layer Toggles
        # ---------------------------------------------------------------------
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        lbl_panels = QLabel("Panels:")
        lbl_panels.setStyleSheet("font-weight: bold; color: #1e293b;")
        toolbar.addWidget(lbl_panels)

        self.combo_presets = QComboBox()
        self.combo_presets.addItems([
            "3 Panels: Profile + Hydro + Transport",
            "3 Panels: Profile + Transport + Concentration",
            "3 Panels: Profile + Transport + Bed Δz",
            "3 Panels: Profile + Hydro + Bed Δz",
            "3 Panels: Hydro + Transport + Concentration",
            "4 Panels: Profile + Hydro + Transport + Bed Δz",
            "4 Panels: Profile + Transport + Conc + Bed Δz",
            "All 5 Panels",
            "Custom Selection",
        ])
        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)
        toolbar.addWidget(self.combo_presets)

        # Individual panel toggle checkboxes
        self.chk_p1 = QCheckBox("Profile")
        self.chk_p1.setChecked(True)
        self.chk_p1.toggled.connect(self._on_panel_checkbox_toggled)
        toolbar.addWidget(self.chk_p1)

        self.chk_p2 = QCheckBox("Hydro")
        self.chk_p2.setChecked(True)
        self.chk_p2.toggled.connect(self._on_panel_checkbox_toggled)
        toolbar.addWidget(self.chk_p2)

        self.chk_p3 = QCheckBox("Transport")
        self.chk_p3.setChecked(True)
        self.chk_p3.toggled.connect(self._on_panel_checkbox_toggled)
        toolbar.addWidget(self.chk_p3)

        self.chk_p_conc = QCheckBox("Concentration")
        self.chk_p_conc.setChecked(False)
        self.chk_p_conc.toggled.connect(self._on_panel_checkbox_toggled)
        toolbar.addWidget(self.chk_p_conc)
        # Compatibility alias
        self.chk_concentration = self.chk_p_conc

        self.chk_p4 = QCheckBox("Bed Δz")
        self.chk_p4.setChecked(False)
        self.chk_p4.toggled.connect(self._on_panel_checkbox_toggled)
        toolbar.addWidget(self.chk_p4)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        toolbar.addWidget(sep)

        # Internal layer toggles
        self.chk_init_bed = QCheckBox("Init Bed")
        self.chk_init_bed.setChecked(True)
        self.chk_init_bed.toggled.connect(self._toggle_curves)
        toolbar.addWidget(self.chk_init_bed)

        self.chk_water_level = QCheckBox("Water Level")
        self.chk_water_level.setChecked(True)
        self.chk_water_level.toggled.connect(self._toggle_curves)
        toolbar.addWidget(self.chk_water_level)

        self.chk_transport_components = QCheckBox("Transport Parts")
        self.chk_transport_components.setChecked(True)
        self.chk_transport_components.toggled.connect(self._toggle_curves)
        toolbar.addWidget(self.chk_transport_components)

        toolbar.addStretch()

        self.btn_autorange = QPushButton("⟲ Auto Range")
        self.btn_autorange.setToolTip("Reset visible plot views to fit full data")
        self.btn_autorange.clicked.connect(self.auto_range)
        toolbar.addWidget(self.btn_autorange)

        main_layout.addLayout(toolbar)

        # Graphics Layout Widget
        self.glw = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.glw)

        # ---------------------------------------------------------------------
        # Plot 1: Bed Level & Water Surface
        # ---------------------------------------------------------------------
        self.p1 = pg.PlotItem()
        self.p1.setTitle("<b>Cross-shore Profile & Water Surface</b>", color="#0f172a", size="10pt")
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
        self.p2 = pg.PlotItem()
        self.p2.setTitle("<b>Hydrodynamics: Wave Height & Mean Currents</b>", color="#0f172a", size="10pt")
        self.p2.setLabel("left", "Wave H (m) / Vel (m/s)")
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        self.leg2 = self.p2.addLegend(offset=(10, 10))

        pen_hrms = pg.mkPen(color="#0369a1", width=2.0)
        pen_h13 = pg.mkPen(color="#0284c7", width=1.5, style=Qt.PenStyle.DashLine)
        pen_vmix = pg.mkPen(color="#15803d", width=2.0)
        pen_ucr = pg.mkPen(color="#b91c1c", width=1.5, style=Qt.PenStyle.DotLine)

        self.c_hrms = self.p2.plot([], [], pen=pen_hrms, name="H_rms (m)")
        self.c_h13 = self.p2.plot([], [], pen=pen_h13, name="H_1/3 (m)")
        self.c_vmix = self.p2.plot([], [], pen=pen_vmix, name="Longshore V_mix (m/s)")
        self.c_ucr = self.p2.plot([], [], pen=pen_ucr, name="Undertow U_cr (m/s)")

        zero_line_p2 = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color="#cbd5e1", width=1, style=Qt.PenStyle.DashLine))
        self.p2.addItem(zero_line_p2)

        # ---------------------------------------------------------------------
        # Plot 3: Sediment Transport Rates (Pure transport in kg/(s*m))
        # ---------------------------------------------------------------------
        self.p3 = pg.PlotItem()
        self.p3.setTitle("<b>Sediment Transport Rates</b>", color="#0f172a", size="10pt")
        self.p3.setLabel("left", "Transport (kg/(s·m))")
        self.p3.showGrid(x=True, y=True, alpha=0.3)
        self.leg3 = self.p3.addLegend(offset=(10, 10))

        pen_st_cross = pg.mkPen(color="#b91c1c", width=2.0)
        pen_sb_cross = pg.mkPen(color="#f97316", width=1.2, style=Qt.PenStyle.DashLine)
        pen_ss_cross = pg.mkPen(color="#7e22ce", width=1.2, style=Qt.PenStyle.DashLine)
        pen_st_long = pg.mkPen(color="#047857", width=2.0)

        self.c_st_cross = self.p3.plot([], [], pen=pen_st_cross, name="Cross Total S_t")
        self.c_sb_cross = self.p3.plot([], [], pen=pen_sb_cross, name="Cross Bedload S_b")
        self.c_ss_cross = self.p3.plot([], [], pen=pen_ss_cross, name="Cross Suspended S_s")
        self.c_st_long = self.p3.plot([], [], pen=pen_st_long, name="Longshore Total S_t,long")

        zero_line_p3 = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color="#cbd5e1", width=1, style=Qt.PenStyle.DashLine))
        self.p3.addItem(zero_line_p3)

        # ---------------------------------------------------------------------
        # Plot 4: Sediment Concentration (Dedicated plot in m3/m3)
        # ---------------------------------------------------------------------
        self.p_conc = pg.PlotItem()
        self.p_conc.setTitle("<b>Sediment Concentration (Volumetric)</b>", color="#0f172a", size="10pt")
        self.p_conc.setLabel("left", "Concentration (m³ sed / m³ agua)")
        self.p_conc.showGrid(x=True, y=True, alpha=0.3)
        self.leg_conc = self.p_conc.addLegend(offset=(10, 10))

        pen_ca = pg.mkPen(color="#7c3aed", width=2.2)
        pen_cmean = pg.mkPen(color="#c026d3", width=1.8, style=Qt.PenStyle.DashLine)

        self.c_ca = self.p_conc.plot([], [], pen=pen_ca, name="Bed Conc Ca (m³/m³)")
        self.c_cmean = self.p_conc.plot([], [], pen=pen_cmean, name="Mean Conc C_mean (m³/m³)")

        zero_line_conc = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color="#cbd5e1", width=1, style=Qt.PenStyle.DashLine))
        self.p_conc.addItem(zero_line_conc)

        # ---------------------------------------------------------------------
        # Plot 5: Cumulative Bed Level Change dz
        # ---------------------------------------------------------------------
        self.p4 = pg.PlotItem()
        self.p4.setTitle("<b>Morphodynamics: Cumulative Bed Level Change Δz</b>", color="#0f172a", size="10pt")
        self.p4.setLabel("left", "Δz (m)")
        self.p4.showGrid(x=True, y=True, alpha=0.3)
        self.leg4 = self.p4.addLegend(offset=(10, 10))

        pen_dz = pg.mkPen(color="#0284c7", width=2.0)
        self.c_dz = self.p4.plot([], [], pen=pen_dz, name="Cumulative Δz(t)")

        zero_line_p4 = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color="#cbd5e1", width=1, style=Qt.PenStyle.DashLine))
        self.p4.addItem(zero_line_p4)

        # Initial layout build
        self._rebuild_plot_layout()

    # -------------------------------------------------------------------------
    # Dynamic Layout Management
    # -------------------------------------------------------------------------
    def _on_preset_changed(self, idx: int) -> None:
        """Apply a standard panel preset."""
        if self._block_preset_signal:
            return

        presets_map = {
            0: [True, True, True, False, False],   # Profile + Hydro + Transport (3)
            1: [True, False, True, True, False],   # Profile + Transport + Conc (3)
            2: [True, False, True, False, True],   # Profile + Transport + Bed dz (3)
            3: [True, True, False, False, True],   # Profile + Hydro + Bed dz (3)
            4: [False, True, True, True, False],   # Hydro + Transport + Conc (3)
            5: [True, True, True, False, True],    # Profile + Hydro + Trans + Bed dz (4)
            6: [True, False, True, True, True],    # Profile + Trans + Conc + Bed dz (4)
            7: [True, True, True, True, True],     # All 5
        }

        if idx in presets_map:
            states = presets_map[idx]
            self.chk_p1.blockSignals(True)
            self.chk_p2.blockSignals(True)
            self.chk_p3.blockSignals(True)
            self.chk_p_conc.blockSignals(True)
            self.chk_p4.blockSignals(True)

            self.chk_p1.setChecked(states[0])
            self.chk_p2.setChecked(states[1])
            self.chk_p3.setChecked(states[2])
            self.chk_p_conc.setChecked(states[3])
            self.chk_p4.setChecked(states[4])

            self.chk_p1.blockSignals(False)
            self.chk_p2.blockSignals(False)
            self.chk_p3.blockSignals(False)
            self.chk_p_conc.blockSignals(False)
            self.chk_p4.blockSignals(False)

            self._rebuild_plot_layout()

    def _on_panel_checkbox_toggled(self) -> None:
        """Handle individual panel checkbox toggle and update preset combo."""
        self._block_preset_signal = True
        self.combo_presets.setCurrentIndex(8)  # 'Custom Selection'
        self._block_preset_signal = False
        self._rebuild_plot_layout()

    def _rebuild_plot_layout(self) -> None:
        """Dynamically add only selected plots to GraphicsLayoutWidget to maintain ideal vertical scale."""
        self.glw.clear()

        panels_to_show = []
        if self.chk_p1.isChecked():
            panels_to_show.append(self.p1)
        if self.chk_p2.isChecked():
            panels_to_show.append(self.p2)
        if self.chk_p3.isChecked():
            panels_to_show.append(self.p3)
        if self.chk_p_conc.isChecked():
            panels_to_show.append(self.p_conc)
        if self.chk_p4.isChecked():
            panels_to_show.append(self.p4)

        if not panels_to_show:
            self.chk_p1.setChecked(True)
            panels_to_show.append(self.p1)

        first_p = panels_to_show[0]
        n_plots = len(panels_to_show)

        for row_idx, p in enumerate(panels_to_show):
            self.glw.addItem(p, row=row_idx, col=0)
            if row_idx > 0:
                p.setXLink(first_p)

            is_bottom = (row_idx == n_plots - 1)
            p.showAxis("bottom", is_bottom)
            if is_bottom:
                p.setLabel("bottom", "Cross-shore Distance x (m)")
            else:
                p.setLabel("bottom", "")

        self.glw.ci.layout.setSpacing(6)

    def _toggle_curves(self) -> None:
        """Handle internal curve layer visibility toggles."""
        self.c_z_init.setVisible(self.chk_init_bed.isChecked())
        self.c_water.setVisible(self.chk_water_level.isChecked())
        show_trans_parts = self.chk_transport_components.isChecked()
        self.c_sb_cross.setVisible(show_trans_parts)
        self.c_ss_cross.setVisible(show_trans_parts)

    # -------------------------------------------------------------------------
    # Data Streaming & Plotting
    # -------------------------------------------------------------------------
    def update_data(self, step_res: TimeStepResult) -> None:
        """Update all curves with data from the simulation time step."""
        if not step_res.x:
            return

        x = np.asarray(step_res.x, dtype=np.float64)

        def _safe_set(curve, arr):
            if arr is not None and len(arr) > 0:
                n = min(len(x), len(arr))
                curve.setData(x[:n], arr[:n])

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

        _safe_set(self.c_hrms, hrms)
        _safe_set(self.c_h13, h13)
        _safe_set(self.c_vmix, vmix)
        _safe_set(self.c_ucr, ucr)

        # Plot 3: Transport Rates
        st_c = np.asarray(step_res.transport.st_cross, dtype=np.float64)
        sb_c = np.asarray(step_res.transport.sb_cross, dtype=np.float64)
        ss_c = np.asarray(step_res.transport.ss_cross, dtype=np.float64)
        st_l = np.asarray(step_res.transport.st_long, dtype=np.float64)

        _safe_set(self.c_st_cross, st_c)
        _safe_set(self.c_sb_cross, sb_c)
        _safe_set(self.c_ss_cross, ss_c)
        _safe_set(self.c_st_long, st_l)

        # Plot 4: Concentration (m3/m3)
        c_ca = np.asarray(step_res.transport.c_vol or step_res.transport.ca_ref, dtype=np.float64)
        c_mean = np.asarray(step_res.transport.c_mean_vol, dtype=np.float64)

        _safe_set(self.c_ca, c_ca)
        _safe_set(self.c_cmean, c_mean)

        # Plot 5: Bed Change
        dz = np.asarray(step_res.morpho.dz_total, dtype=np.float64)
        _safe_set(self.c_dz, dz)

        if self._first_render:
            self.auto_range()
            self._first_render = False

    def auto_range(self) -> None:
        """Autorange all instantiated plots."""
        for p in [self.p1, self.p2, self.p3, self.p_conc, self.p4]:
            p.autoRange()

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
