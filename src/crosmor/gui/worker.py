"""QThread worker for executing CROSMOR simulations asynchronously in GUI."""

from PyQt6.QtCore import QThread, pyqtSignal
from crosmor.models.config import SimulationConfig
from crosmor.models.results import TimeStepResult, SimulationResults
from crosmor.engine.runner import SimulationRunner


class SimulationWorker(QThread):
    """Executes the simulation runner in a background thread to maintain GUI responsiveness."""

    step_completed = pyqtSignal(object, float)  # (TimeStepResult, progress_0_to_1)
    simulation_finished = pyqtSignal(object)   # SimulationResults
    simulation_error = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(self, config: SimulationConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.runner = SimulationRunner(config)
        self._results: SimulationResults = None  # type: ignore

    def run(self) -> None:
        try:
            self.status_message.emit("Simulation running...")

            def on_step(step_res: TimeStepResult, progress: float):
                self.step_completed.emit(step_res, progress)

            self._results = self.runner.run(on_step_callback=on_step)
            if not self.runner.is_cancelled:
                self.status_message.emit("Simulation completed.")
                self.simulation_finished.emit(self._results)
            else:
                self.status_message.emit("Simulation cancelled by user.")
        except Exception as e:
            self.simulation_error.emit(str(e))
            self.status_message.emit(f"Error: {e}")

    def pause(self) -> None:
        self.runner.pause()
        self.status_message.emit("Simulation paused.")

    def resume(self) -> None:
        self.runner.resume()
        self.status_message.emit("Simulation resumed.")

    def stop(self) -> None:
        self.runner.cancel()
        self.status_message.emit("Cancelling simulation...")
