from PyQt6.QtCore import QThread, pyqtSignal


class InferenceWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        engine,
        image_path: str,
        confidence_threshold: float,
        pass_a_only: bool,
        inference_mode: str = "CPU + GPU",
        use_vram_strategy: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._engine = engine
        self._image_path = image_path
        self._confidence_threshold = confidence_threshold
        self._pass_a_only = pass_a_only
        self._inference_mode = inference_mode
        self._use_vram_strategy = use_vram_strategy

    def run(self) -> None:
        try:
            self.progress.emit(f"Running {self._inference_mode}…")
            result = self._engine.infer(
                self._image_path,
                confidence_threshold=self._confidence_threshold,
                pass_a_only=self._pass_a_only,
                use_vram_strategy=self._use_vram_strategy,
            )
            if result.error:
                self.error.emit(result.error)
            else:
                self.finished.emit(result)
        except Exception as exc:
            self.error.emit(f"Inference worker error: {exc}")
