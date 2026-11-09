"""Simulation player bar with play, pause, step controls, time slider and status indicators."""

from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal


class PlayerBar(QWidget):
    """Control toolbar for running, scrubbing and monitoring simulations."""

    run_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    step_selected = pyqtSignal(int)  # Emits step index when slider moves

    def __init__(self, parent=None):
        super().__init__(parent)
        self._max_steps = 0
        self._is_running = False
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Row 1: Slider and progress bar
        h_slider_layout = QHBoxLayout()
        h_slider_layout.addWidget(QLabel("Time Scrubber:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider_moved)
        h_slider_layout.addWidget(self.slider)

        self.lbl_step = QLabel("Step: 0 / 0")
        h_slider_layout.addWidget(self.lbl_step)
        main_layout.addLayout(h_slider_layout)

        # Row 2: Action buttons and live stats
        h_btn_layout = QHBoxLayout()

        self.btn_run = QPushButton("▶ Run Simulation")
        self.btn_run.setStyleSheet("font-weight: bold; background-color: #e1f5fe; color: #0277bd; padding: 6px 14px;")
        self.btn_run.clicked.connect(self._on_run_clicked)
        h_btn_layout.addWidget(self.btn_run)

        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        h_btn_layout.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        h_btn_layout.addWidget(self.btn_stop)

        h_btn_layout.addSpacing(15)

        self.lbl_time = QLabel("Time: 0.0 s")
        self.lbl_time.setStyleSheet("font-weight: bold; font-size: 13px;")
        h_btn_layout.addWidget(self.lbl_time)

        self.lbl_balance = QLabel("Balance: 0.000 m²")
        h_btn_layout.addWidget(self.lbl_balance)

        h_btn_layout.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumWidth(80)
        self.progress_bar.setMaximumWidth(160)
        h_btn_layout.addWidget(self.progress_bar)

        main_layout.addLayout(h_btn_layout)

    def set_running_state(self, is_running: bool) -> None:
        self._is_running = is_running
        self.btn_run.setEnabled(not is_running)
        self.btn_pause.setEnabled(is_running)
        self.btn_stop.setEnabled(is_running)
        if not is_running:
            self.btn_pause.setText("⏸ Pause")

    def reset(self) -> None:
        self._max_steps = 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.lbl_step.setText("Step: 0 / 0")
        self.lbl_time.setText("Time: 0.0 s")
        self.lbl_balance.setText("Balance: 0.000 m²")
        self.progress_bar.setValue(0)
        self.set_running_state(False)

    def update_step(self, step_idx: int, total_steps: int, time_s: float, balance_m2: float, progress: float) -> None:
        self._max_steps = max(self._max_steps, total_steps)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, self._max_steps - 1))
        self.slider.setValue(step_idx)
        self.slider.blockSignals(False)

        hours = time_s / 3600.0
        self.lbl_step.setText(f"Step: {step_idx + 1} / {self._max_steps}")
        self.lbl_time.setText(f"Time: {time_s:.1f} s ({hours:.2f} h)")
        self.lbl_balance.setText(f"Balance: {balance_m2:+.4f} m²")
        self.progress_bar.setValue(int(progress * 100))

    def _on_run_clicked(self) -> None:
        self.run_requested.emit()

    def _on_pause_clicked(self) -> None:
        if self.btn_pause.text().startswith("⏸"):
            self.btn_pause.setText("▶ Resume")
            self.pause_requested.emit()
        else:
            self.btn_pause.setText("⏸ Pause")
            self.pause_requested.emit()

    def _on_stop_clicked(self) -> None:
        self.stop_requested.emit()

    def _on_slider_moved(self, value: int) -> None:
        self.step_selected.emit(value)
