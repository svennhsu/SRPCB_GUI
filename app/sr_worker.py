"""QThread worker for super-resolution processing."""

from PyQt6.QtCore import QThread, pyqtSignal


class SRWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, engine, image_path: str, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._image_path = image_path

    def run(self) -> None:
        try:
            self.progress.emit(0, 1)

            def on_tile(current: int, total: int) -> None:
                self.progress.emit(current, total)

            sr_path = self._engine.run(self._image_path, progress_callback=on_tile)
            self.finished.emit(sr_path)
        except Exception as exc:
            self.error.emit(f"Super-resolution error: {exc}")
