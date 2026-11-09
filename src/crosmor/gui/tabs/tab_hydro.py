"""Tab for editing hydrodynamic model coefficients, switches and calibration factors."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
)

from crosmor.models.config import SimulationConfig, HydroParams


class TabHydro(QWidget):
    """Hydrodynamic switches and tuning parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 1. Physics Switches
        grp_switches = QGroupBox("Non-linear Waves & Orbital Velocity Switches")
        grid_sw = QGridLayout(grp_switches)

        grid_sw.addWidget(QLabel("Orbital Asymmetry Formulation:"), 0, 0)
        self.combo_asym = QComboBox()
        self.combo_asym.addItem("0: Isobe and Horikawa (2003)", 0)
        self.combo_asym.addItem("1: Ruessink et al. (2008)", 1)
        self.combo_asym.addItem("2: Ruessink et al. (2010)", 2)
        grid_sw.addWidget(self.combo_asym, 0, 1)

        self.chk_nielsen = QCheckBox("Switch 2: Nielsen acceleration phase lead on bed shear stress")
        grid_sw.addWidget(self.chk_nielsen, 1, 0, 1, 2)

        self.chk_phaselag = QCheckBox("Switch 3: Include phase lag effects for suspended load")
        grid_sw.addWidget(self.chk_phaselag, 2, 0, 1, 2)

        self.chk_longwaves = QCheckBox("Switch 4: Include infragravity long waves in surf zone")
        grid_sw.addWidget(self.chk_longwaves, 3, 0, 1, 2)

        layout.addWidget(grp_switches)

        # 2. Wave Breaking & Roller Parameters
        grp_roller = QGroupBox("Wave Breaking & Current Calibration")
        grid_r = QGridLayout(grp_roller)

        grid_r.addWidget(QLabel("Roller Coefficient (0=max roller, 1=no roller):"), 0, 0)
        self.spin_brol = QDoubleSpinBox()
        self.spin_brol.setRange(0.0, 1.0)
        self.spin_brol.setValue(0.5)
        self.spin_brol.setSingleStep(0.05)
        grid_r.addWidget(self.spin_brol, 0, 1)

        grid_r.addWidget(QLabel("Breaking Index Gamma (0=auto computed by model):"), 0, 2)
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.0, 1.5)
        self.spin_gamma.setValue(0.0)
        self.spin_gamma.setSingleStep(0.05)
        grid_r.addWidget(self.spin_gamma, 0, 3)

        grid_r.addWidget(QLabel("Undertow Averaging Factor (coef6):"), 1, 0)
        self.spin_coef6 = QDoubleSpinBox()
        self.spin_coef6.setRange(0.0, 10.0)
        self.spin_coef6.setValue(1.0)
        grid_r.addWidget(self.spin_coef6, 1, 1)

        grid_r.addWidget(QLabel("Longshore Averaging Factor (coef5):"), 1, 2)
        self.spin_coef5 = QDoubleSpinBox()
        self.spin_coef5.setRange(0.0, 10.0)
        self.spin_coef5.setValue(1.0)
        grid_r.addWidget(self.spin_coef5, 1, 3)

        grid_r.addWidget(QLabel("Longuet-Higgins Streaming (clh, 0-1):"), 2, 0)
        self.spin_clh = QDoubleSpinBox()
        self.spin_clh.setRange(0.0, 1.0)
        self.spin_clh.setValue(1.0)
        grid_r.addWidget(self.spin_clh, 2, 1)

        grid_r.addWidget(QLabel("Shingle Beach Swash Velocity (csw, 0-1):"), 2, 2)
        self.spin_csw = QDoubleSpinBox()
        self.spin_csw.setRange(0.0, 1.0)
        self.spin_csw.setValue(1.0)
        grid_r.addWidget(self.spin_csw, 2, 3)

        grid_r.addWidget(QLabel("Longshore Lateral Mixing els (m2/s):"), 3, 0)
        self.spin_els = QDoubleSpinBox()
        self.spin_els.setRange(0.01, 20.0)
        self.spin_els.setValue(0.5)
        grid_r.addWidget(self.spin_els, 3, 1)

        grid_r.addWidget(QLabel("Longshore Max Iterations (nit):"), 3, 2)
        self.spin_nit = QSpinBox()
        self.spin_nit.setRange(1, 2000)
        self.spin_nit.setValue(300)
        grid_r.addWidget(self.spin_nit, 3, 3)

        grid_r.addWidget(QLabel("Waterline Smoothing (SW):"), 4, 0)
        self.spin_sw = QDoubleSpinBox()
        self.spin_sw.setRange(0.0, 1.0)
        self.spin_sw.setValue(0.2)
        grid_r.addWidget(self.spin_sw, 4, 1)

        grid_r.addWidget(QLabel("Max Dune Slope SL (deg):"), 4, 2)
        self.spin_sl = QDoubleSpinBox()
        self.spin_sl.setRange(10.0, 90.0)
        self.spin_sl.setValue(50.0)
        grid_r.addWidget(self.spin_sl, 4, 3)

        layout.addWidget(grp_roller)
        layout.addStretch()

    def load_config(self, config: SimulationConfig) -> None:
        h = config.hydro
        idx = self.combo_asym.findData(h.switch1_asymmetry)
        if idx >= 0:
            self.combo_asym.setCurrentIndex(idx)

        self.chk_nielsen.setChecked(h.switch2_nielsen == 1)
        self.chk_phaselag.setChecked(h.switch3_phaselag == 1)
        self.chk_longwaves.setChecked(h.switch4_longwaves == 1)

        self.spin_brol.setValue(h.roller_coef)
        self.spin_gamma.setValue(h.gamma)
        self.spin_coef6.setValue(h.coef6)
        self.spin_coef5.setValue(h.coef5)
        self.spin_clh.setValue(h.clh)
        self.spin_csw.setValue(h.csw)
        self.spin_els.setValue(h.els)
        self.spin_nit.setValue(h.nit)
        self.spin_sw.setValue(h.sw)
        self.spin_sl.setValue(h.sl_dune_slope)

    def update_config(self, config: SimulationConfig) -> None:
        h = config.hydro
        h.switch1_asymmetry = self.combo_asym.currentData()
        h.switch2_nielsen = 1 if self.chk_nielsen.isChecked() else 0
        h.switch3_phaselag = 1 if self.chk_phaselag.isChecked() else 0
        h.switch4_longwaves = 1 if self.chk_longwaves.isChecked() else 0

        h.roller_coef = self.spin_brol.value()
        h.gamma = self.spin_gamma.value()
        h.coef6 = self.spin_coef6.value()
        h.coef5 = self.spin_coef5.value()
        h.clh = self.spin_clh.value()
        h.csw = self.spin_csw.value()
        h.els = self.spin_els.value()
        h.nit = self.spin_nit.value()
        h.sw = self.spin_sw.value()
        h.sl_dune_slope = self.spin_sl.value()
