from __future__ import annotations

import csv
import logging
import platform
from pathlib import Path
from shutil import copyfile

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableView,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.image_viewer import ImageViewer
from app.paths import SR_PROXY_CACHE
from app.results_model import ResultsTableModel, result_to_rows
from app.sr_worker import SRWorker
from app.status_line import StatusLineController
from app.worker import InferenceWorker
from history.history_manager import HistoryManager
from inference.aoi_inference_engine import AOIInferenceEngine
from inference.detection_result import DetectionResult
from inference.sr_engine import SuperResolutionEngine
from reports.pdf_exporter import export_pdf
from reports.report_builder import build_operator_report


APP_TITLE = "Super Resolution & PCB Component Counting"
APP_VERSION = "1.0.0"
logger = logging.getLogger(__name__)


class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._toggle = QToolButton()
        self._toggle.setObjectName("sectionHeader")
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow)
        self._content = content
        self._toggle.toggled.connect(self._set_expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._toggle)
        layout.addWidget(self._content)

    def _set_expanded(self, expanded: bool) -> None:
        self._content.setVisible(expanded)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} v{APP_VERSION}")

        self._image_path: Path | None = None
        self._original_size: tuple[int, int] | None = None
        self._current_result: DetectionResult | None = None
        self._worker: InferenceWorker | None = None
        self._sr_worker: SRWorker | None = None
        self._sr_fullres_path: str | None = None
        self._sr_detection_proxy_path: str | None = None
        self._detect_use_sr: bool = False
        self._fullscreen = False

        self._detector = AOIInferenceEngine()
        self._sr_engine = SuperResolutionEngine()
        self._history = HistoryManager()
        self._results_model = ResultsTableModel()
        self._status = StatusLineController()

        self._create_actions()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_central_ui()
        self._status.attach(self.statusBar())

        self.device_combo.currentIndexChanged.connect(self._sync_device_menu_checks)
        self.detection_source_combo.currentIndexChanged.connect(self._sync_source_menu_checks)
        self._sync_device_menu_checks(self.device_combo.currentIndex())
        self._sync_source_menu_checks(self.detection_source_combo.currentIndex())

        self._update_action_states()
        self._refresh_history_list()
        self._try_auto_load_model()
        self._try_auto_load_sr_model()

    def _action(self, text: str, slot, shortcut=None) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut) if isinstance(shortcut, str) else shortcut)
        action.triggered.connect(slot)
        return action

    def _create_actions(self) -> None:
        self.open_action = self._action("Open Image", self.open_image, QKeySequence.StandardKey.Open)
        self.open_folder_action = self._action("Open Folder", self.open_folder)
        self.run_detection_action = self._action("Run Inference", self.run_detection, "F5")
        self.save_result_action = self._action("Save Result Image", self.save_result_image, QKeySequence.StandardKey.Save)
        self.export_csv_action = self._action("Export Counts CSV", self.export_counts_csv)
        self.export_pdf_action = self._action("Export PDF", self.export_pdf_report, "Ctrl+P")
        self.clear_current_action = self._action("Clear Current", self.clear_current)
        self.clear_unpinned_history_action = self._action("Clear Unpinned History", self.clear_unpinned_history)
        self.exit_action = self._action("Exit", self.close, QKeySequence.StandardKey.Quit)
        self.toggle_sidebar_action = self._action("Toggle Sidebar", self._toggle_sidebar, "Ctrl+B")
        self.toggle_report_action = self._action("Toggle Report Panel", self._toggle_report, "Ctrl+R")
        self.toggle_fullscreen_action = self._action("Toggle Full Screen", self._toggle_fullscreen, "F11")
        self.reset_layout_action = self._action("Reset Layout", self._reset_layout)
        self.fit_to_window_action = self._action("Fit to Window", self._fit_to_window)
        self.about_action = self._action("About This Application", self.show_about)
        self.run_sr_action = self._action("Run Super Resolution", self.run_sr, "Ctrl+Shift+R")
        self.save_sr_action = self._action("Save SR Image", self.save_sr_image, "Ctrl+Shift+S")

    def _create_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_folder_action)
        file_menu.addSeparator()
        file_menu.addAction(self.run_sr_action)
        file_menu.addAction(self.run_detection_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_sr_action)
        file_menu.addAction(self.save_result_action)
        file_menu.addAction(self.export_csv_action)
        file_menu.addAction(self.export_pdf_action)
        file_menu.addSeparator()
        file_menu.addAction(self.clear_current_action)
        file_menu.addAction(self.clear_unpinned_history_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.fit_to_window_action)
        view_menu.addSeparator()
        view_menu.addAction(self.toggle_sidebar_action)
        view_menu.addAction(self.toggle_report_action)
        view_menu.addSeparator()
        view_menu.addAction(self.toggle_fullscreen_action)
        view_menu.addSeparator()
        view_menu.addAction(self.reset_layout_action)

        settings_menu = self.menuBar().addMenu("Settings")

        self._device_menu = settings_menu.addMenu("Detection Device")
        self._device_actions = []
        for idx, label in enumerate(["CUDA if available", "Auto", "CPU"]):
            act = self._device_menu.addAction(label)
            act.setCheckable(True)
            act.triggered.connect(lambda checked, i=idx: self._on_device_menu_triggered(i))
            self._device_actions.append(act)

        self._source_menu = settings_menu.addMenu("Detection Input Source")
        self._source_actions = []
        for idx, label in enumerate(["Original Image", "SR-Restored Image (Original Size)"]):
            act = self._source_menu.addAction(label)
            act.setCheckable(True)
            act.triggered.connect(lambda checked, i=idx: self._on_source_menu_triggered(i))
            self._source_actions.append(act)

        about_menu = self.menuBar().addMenu("About")
        about_menu.addAction(self.about_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setObjectName("mainToolbar")
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.open_folder_action)
        toolbar.addSeparator()
        toolbar.addAction(self.run_sr_action)
        toolbar.addAction(self.run_detection_action)
        toolbar.addSeparator()
        toolbar.addAction(self.save_sr_action)
        toolbar.addAction(self.save_result_action)
        toolbar.addAction(self.export_pdf_action)
        toolbar.addSeparator()
        toolbar.addAction(self.clear_current_action)
        self.addToolBar(toolbar)

    def _create_central_ui(self) -> None:
        root_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_splitter.setObjectName("rootSplitter")

        self._sidebar_scroll = QScrollArea()
        self._sidebar_scroll.setObjectName("sidebarScroll")
        self._sidebar_scroll.setWidgetResizable(True)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._sidebar_scroll.setMinimumWidth(220)
        self._sidebar_scroll.setMaximumWidth(440)
        self._sidebar_scroll.setWidget(self._build_sidebar())

        root_splitter.addWidget(self._sidebar_scroll)
        root_splitter.addWidget(self._build_workspace())
        root_splitter.setStretchFactor(0, 0)
        root_splitter.setStretchFactor(1, 1)
        root_splitter.setSizes([360, 1000])
        self._root_splitter = root_splitter
        self.setCentralWidget(root_splitter)

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")

        title = QLabel(APP_TITLE)
        title.setObjectName("appTitle")

        self.path_label = QLabel("No image loaded")
        self.path_label.setObjectName("mutedValue")
        self.path_label.setWordWrap(True)
        self.path_label.setToolTip("Full path of the current image")
        self.size_label = QLabel("—")
        self.size_label.setObjectName("mutedValue")
        self.model_path_label = QLabel("No model loaded")
        self.model_path_label.setObjectName("mutedValue")
        self.model_path_label.setWordWrap(True)
        self.gpu_status_label = QLabel("GPU not checked")
        self.gpu_status_label.setObjectName("mutedValue")
        self.gpu_status_label.setWordWrap(True)
        self.total_components_label = QLabel("0")
        self.total_components_label.setObjectName("totalCount")

        self.device_combo = QComboBox()
        self.device_combo.addItems(["CUDA if available", "Auto", "CPU"])

        self.inference_mode_combo = QComboBox()
        self.inference_mode_combo.addItems([
            "CPU + GPU",
            "GPU only",
            "Fast Preview",
        ])

        self.detection_source_combo = QComboBox()
        self.detection_source_combo.addItems(["Original Image", "SR-Restored Image (Original Size)"])
        self.detection_source_combo.setToolTip(
            "Uses the super-resolved image downsampled back to the original "
            "dimensions before detection. Preserves restoration benefits "
            "while reducing GPU memory usage."
        )

        self.sr_status_label = QLabel("Not run")
        self.sr_status_label.setObjectName("mutedValue")
        self.sr_status_label.setWordWrap(True)

        image_input = self._form_section([
            ("File", self.path_label),
            ("Size", self.size_label),
        ])
        model = self._form_section([
            ("Status", self.model_path_label),
            ("GPU", self.gpu_status_label),
            ("Device", self.device_combo),
            ("Mode", self.inference_mode_combo),
            ("Detect on", self.detection_source_combo),
        ])
        sr_status = self._form_section([
            ("SR Status", self.sr_status_label),
        ])
        detection = self._form_section([
            ("Total Components", self.total_components_label),
        ])
        self.summary_label = QLabel(self._summary_text())
        self.summary_label.setObjectName("mutedValue")
        self.summary_label.setWordWrap(True)
        results = self._plain_group(self.summary_label)

        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(80)
        self.history_list.setAlternatingRowColors(True)
        self.history_list.itemClicked.connect(self._on_history_item_clicked)
        self.history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self._on_history_context_menu)
        history_section = self._plain_group(self.history_list)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(CollapsibleSection("Image", image_input))
        layout.addWidget(CollapsibleSection("Model & Settings", model))
        layout.addWidget(CollapsibleSection("Super Resolution", sr_status))
        layout.addWidget(CollapsibleSection("Detection", detection))
        layout.addWidget(CollapsibleSection("Results Summary", results))
        layout.addWidget(CollapsibleSection("Processing History", history_section))
        layout.addStretch(1)
        return sidebar

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(6, 8, 8, 8)
        workspace_layout.setSpacing(12)

        image_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.input_viewer = ImageViewer("Original PCB Image", "Open a PCB image to begin")
        self.sr_viewer = ImageViewer("Super Resolution", "Run super resolution to view result")
        self.result_viewer = ImageViewer("Detection Result", "Run inference to generate a result view")
        image_splitter.addWidget(self.input_viewer)
        image_splitter.addWidget(self.sr_viewer)
        image_splitter.addWidget(self.result_viewer)
        image_splitter.setSizes([320, 320, 320])

        self._report_panel = QGroupBox("Inspection Report")
        report_layout = QVBoxLayout(self._report_panel)
        report_layout.setContentsMargins(10, 14, 10, 10)
        report_layout.setSpacing(8)

        self.results_table = QTableView()
        self.results_table.setModel(self._results_model)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.results_table.setMinimumWidth(340)

        header = self.results_table.horizontalHeader()
        header.setMinimumSectionSize(60)
        header.resizeSection(0, 160)
        header.resizeSection(1, 80)
        header.resizeSection(2, 140)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText("Run inference to generate an inspection report.")
        self.report_text.setMinimumWidth(300)

        report_splitter = QSplitter(Qt.Orientation.Horizontal)
        report_splitter.addWidget(self.results_table)
        report_splitter.addWidget(self.report_text)
        report_splitter.setStretchFactor(0, 0)
        report_splitter.setStretchFactor(1, 1)
        report_splitter.setSizes([420, 460])
        report_layout.addWidget(report_splitter, 1)

        report_wrapper = QWidget()
        wrapper_layout = QVBoxLayout(report_wrapper)
        wrapper_layout.setContentsMargins(0, 10, 0, 0)
        wrapper_layout.addWidget(self._report_panel)
        self._report_wrapper = report_wrapper

        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        vertical_splitter.addWidget(image_splitter)
        vertical_splitter.addWidget(report_wrapper)
        vertical_splitter.setStretchFactor(0, 2)
        vertical_splitter.setStretchFactor(1, 1)
        vertical_splitter.setSizes([460, 340])
        workspace_layout.addWidget(vertical_splitter)
        return workspace

    def _form_section(self, rows: list[tuple[str, QWidget]]) -> QWidget:
        container = QWidget()
        layout = QFormLayout(container)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(7)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        for label, widget in rows:
            if label:
                layout.addRow(label, widget)
            else:
                layout.addRow(widget)
        return container

    def _plain_group(self, child: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.addWidget(child)
        return container

    def _device_preference(self) -> str:
        text = self.device_combo.currentText().lower()
        if text.startswith("cuda"):
            return "cuda"
        if text.startswith("cpu"):
            return "cpu"
        return "auto"

    def _update_action_states(self) -> None:
        has_image = self._image_path is not None and self.input_viewer.pixmap() is not None
        has_result = self._current_result is not None
        model_loaded = self._detector.loaded
        sr_loaded = self._sr_engine.loaded
        has_sr = self._sr_fullres_path is not None

        self.run_sr_action.setEnabled(has_image and sr_loaded)
        self.save_sr_action.setEnabled(has_sr)

        use_sr = self.detection_source_combo.currentText() == "SR-Restored Image (Original Size)"
        can_detect = has_image and model_loaded and (not use_sr or bool(self._sr_detection_proxy_path))
        self.run_detection_action.setEnabled(can_detect)
        self.save_result_action.setEnabled(has_result and bool(self._current_result and self._current_result.annotated_image_path))
        self.export_pdf_action.setEnabled(has_result)
        self.export_csv_action.setEnabled(has_result)
        self.clear_current_action.setEnabled(has_image or has_result or has_sr)

    def _try_auto_load_model(self) -> None:
        if self._detector.load(device_preference=self._device_preference()):
            self._sync_model_label()
            self._status.success(f"Model ready on {self._detector.device_label}")
            logger.info("Model auto-loaded on %s", self._detector.device_label)
        else:
            self.model_path_label.setText("No model loaded")
            self.gpu_status_label.setText(self._cuda_status_text())
            self._status.warning("Model not loaded. Install PyTorch with a CUDA build for your GPU.")
        self._update_action_states()

    def _try_auto_load_sr_model(self) -> None:
        if self._sr_engine.load(self._device_preference()):
            self.sr_status_label.setText(f"Ready on {self._sr_engine.device_label}")
            self._status.info(f"SR model ready on {self._sr_engine.device_label}")
            logger.info("SR model auto-loaded on %s", self._sr_engine.device_label)
        else:
            self.sr_status_label.setText(self._sr_engine.load_error or "Not loaded")
            logger.warning("SR model not loaded: %s", self._sr_engine.load_error)
        self._update_action_states()

    def _sync_model_label(self) -> None:
        warning = f" — {self._detector.warnings[-1]}" if self._detector.warnings else ""
        self.model_path_label.setText(
            f"Loaded (best_model.pth)\n"
            f"{self._detector.pipeline_source}\n"
            f"{self._detector.device_label}{warning}"
        )
        self.model_path_label.setToolTip(self._detector.device_label)
        self.gpu_status_label.setText(self._cuda_status_text())

    def _cuda_status_text(self) -> str:
        if self._detector.cuda_block_reason:
            return "CUDA blocked. Re-run with a compatible PyTorch env."
        return self._detector.device_label

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open PCB Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self._load_image_path(Path(path))

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open PCB Image Folder")
        if not folder:
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        images = sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in exts)
        if not images:
            QMessageBox.information(self, "Open Folder", "No supported image files found.")
            return
        self._load_image_path(images[0])
        self._status.info(f"Loaded first of {len(images)} images from folder.")

    def _load_image_path(self, path: Path) -> None:
        try:
            pixmap = self.input_viewer.load_image(path)
        except ValueError as exc:
            self._status.error("Error loading image")
            QMessageBox.critical(self, "Error Loading Image", str(exc))
            return
        self._image_path = path
        try:
            from PIL import Image as PILImage
            PILImage.MAX_IMAGE_PIXELS = None
            with PILImage.open(path) as pil_img:
                self._original_size = (pil_img.width, pil_img.height)
        except OSError as exc:
            logger.debug("Could not read original image size with PIL: %s", exc)
            self._original_size = (pixmap.width(), pixmap.height())
        self._current_result = None
        self._sr_fullres_path = None
        self._sr_detection_proxy_path = None
        self.sr_viewer.clear("Run super resolution to view result")
        self.result_viewer.clear("Run inference to generate a result view")
        self.sr_status_label.setText("Not run")
        self.detection_source_combo.setCurrentIndex(0)
        self._results_model.set_rows([])
        self.report_text.clear()
        self.path_label.setText(str(path.name))
        self.path_label.setToolTip(str(path))
        self.size_label.setText(f"{pixmap.width()} × {pixmap.height()} px")
        self._refresh_summary()
        self._update_action_states()
        self._status.info("Image loaded — ready for inference.")

    def _on_device_menu_triggered(self, idx: int) -> None:
        self.device_combo.setCurrentIndex(idx)

    def _on_source_menu_triggered(self, idx: int) -> None:
        self.detection_source_combo.setCurrentIndex(idx)

    def _sync_device_menu_checks(self, idx: int) -> None:
        for i, act in enumerate(self._device_actions):
            act.setChecked(i == idx)

    def _sync_source_menu_checks(self, idx: int) -> None:
        for i, act in enumerate(self._source_actions):
            act.setChecked(i == idx)

    def run_detection(self) -> None:
        use_sr = self.detection_source_combo.currentText() == "SR-Restored Image (Original Size)"
        if use_sr and not self._sr_detection_proxy_path:
            self._status.warning("No SR detection proxy available")
            QMessageBox.warning(self, "Run Inference", "Run super resolution first, or switch detection source to Original Image.")
            return

        self._detect_use_sr = use_sr

        detect_path = Path(self._sr_detection_proxy_path) if use_sr else self._image_path
        if detect_path is None or not detect_path.exists():
            self._status.warning("No image available for detection")
            return

        if self.input_viewer.pixmap() is None:
            self._status.warning("Open an image before running inference")
            QMessageBox.warning(self, "Run Inference", "Open a PCB image before running inference.")
            return
        if not self._detector.loaded:
            self._status.warning("Model not loaded")
            detail = self._detector.load_error or self._detector.cuda_block_reason or "Load the model before running inference."
            QMessageBox.warning(self, "Run Inference", detail)
            return

        mode_name = self.inference_mode_combo.currentText()
        pass_a_only = mode_name == "Fast Preview"
        use_vram_strategy = mode_name == "CPU + GPU"
        self.run_button_set_enabled(False)
        source_label = "SR-Restored Image (Original Size)" if use_sr else "Original Image"
        self._status.info(f"Running {mode_name} on {source_label}…")
        logger.info(
            "Inference start: mode=%s input=%s source=%s sr_proxy=%s",
            mode_name, detect_path, source_label, use_sr,
        )

        self._worker = InferenceWorker(
            self._detector,
            str(detect_path),
            0.5,
            pass_a_only,
            mode_name,
            use_vram_strategy=use_vram_strategy,
        )
        self._worker.finished.connect(self._on_inference_finished)
        self._worker.error.connect(self._on_inference_error)
        self._worker.progress.connect(self._status.info)
        self._worker.start()

    def run_button_set_enabled(self, enabled: bool) -> None:
        self.run_sr_action.setEnabled(enabled)
        self.run_detection_action.setEnabled(enabled)

    def run_sr(self) -> None:
        if self._image_path is None:
            self._status.warning("Open an image before running super resolution")
            return
        if not self._sr_engine.loaded:
            if not self._sr_engine.load(self._device_preference()):
                QMessageBox.critical(self, "SR Model Error", self._sr_engine.load_error or "Failed to load SR model")
                return
            self.sr_status_label.setText(f"Ready on {self._sr_engine.device_label}")

        self.run_button_set_enabled(False)
        self._status.info("Running super resolution…")
        logger.info("SR start: %s", self._image_path)

        self._sr_worker = SRWorker(self._sr_engine, str(self._image_path))
        self._sr_worker.finished.connect(self._on_sr_finished)
        self._sr_worker.error.connect(self._on_sr_error)
        self._sr_worker.progress.connect(self._on_sr_progress)
        self._sr_worker.start()

    def _on_sr_progress(self, current: int, total: int) -> None:
        self._status.info(f"Super resolution: tile {current}/{total}")

    def _on_sr_finished(self, sr_path: str) -> None:
        self._sr_fullres_path = sr_path
        self.run_button_set_enabled(True)
        self.detection_source_combo.setCurrentIndex(1)
        sr_file = Path(sr_path)
        if not sr_file.exists():
            self._status.error(f"SR output file missing: {sr_path}")
            self.sr_status_label.setText("Error: output file missing")
            logger.error("SR output file not found: %s", sr_path)
            self._update_action_states()
            return
        file_size_mb = sr_file.stat().st_size / (1024 * 1024)
        logger.info("SR output: %s (%.1f MB)", sr_path, file_size_mb)

        if self._original_size:
            self._sr_detection_proxy_path = self._create_sr_detection_proxy(sr_path)

        try:
            self.sr_viewer.load_image(sr_path)
            self.sr_status_label.setText("Complete")
            self._status.success(f"Super resolution complete — {sr_file.name} ({file_size_mb:.1f} MB)")
        except ValueError as exc:
            self._status.warning(f"SR image view failed: {exc}")
            self.sr_status_label.setText(f"Complete (view failed — {file_size_mb:.1f} MB)")
        self._update_action_states()

    def _on_sr_error(self, error_msg: str) -> None:
        self.run_button_set_enabled(True)
        self._update_action_states()
        self.sr_status_label.setText(f"Error: {error_msg}")
        self._status.error(error_msg)
        logger.error("SR error: %s", error_msg)
        QMessageBox.critical(self, "Super Resolution Error", error_msg)

    def _create_sr_detection_proxy(self, sr_path: str) -> str | None:
        if self._original_size is None or self._image_path is None:
            return None
        try:
            from PIL import Image as PILImage

            PILImage.MAX_IMAGE_PIXELS = None
            SR_PROXY_CACHE.mkdir(parents=True, exist_ok=True)
            proxy_path = SR_PROXY_CACHE / f"{self._image_path.stem}_SR_detection_input.png"
            with PILImage.open(sr_path) as sr_img:
                sr_size = sr_img.size
                sr_img.resize(self._original_size, PILImage.LANCZOS).save(proxy_path, "PNG")
            logger.info(
                "SR detection proxy created — full=%dx%d effective_detection=%dx%d",
                sr_size[0], sr_size[1], self._original_size[0], self._original_size[1],
            )
            return str(proxy_path)
        except Exception as exc:
            logger.exception("Failed to create SR detection proxy: %s", exc)
            return None

    def _save_copy(self, source: str | Path, title: str, default_name: str) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            title,
            default_name,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )
        if not path:
            return
        label = title.removeprefix("Save ")
        try:
            copyfile(source, path)
            self._status.success(f"Saved {label}: {path}")
        except OSError as exc:
            self._status.error(f"Error saving {label}")
            QMessageBox.critical(self, title, str(exc))

    def save_sr_image(self) -> None:
        if not self._sr_fullres_path:
            self._status.warning("No SR result to save")
            QMessageBox.information(self, "Save SR Image", "Run super resolution before saving.")
            return
        stem = Path(self._image_path).stem if self._image_path else "pcb"
        self._save_copy(self._sr_fullres_path, "Save SR Image", f"{stem}_SR.png")

    def _on_inference_finished(self, result: DetectionResult) -> None:
        self._current_result = result
        self.run_button_set_enabled(True)

        result.detection_source = "sr_restored" if self._detect_use_sr else "original"
        result.detection_source_label = (
            "SR-Restored Image (Original Size)" if self._detect_use_sr else "Original Image"
        )
        if self._detect_use_sr and self._sr_fullres_path:
            result.sr_image_path = self._sr_fullres_path
            result.sr_detection_proxy_path = self._sr_detection_proxy_path
            if self._sr_detection_proxy_path:
                result.effective_image_path = self._sr_detection_proxy_path
            if self._original_size:
                result.effective_image_size = self._original_size
                logger.info(
                    "Detection on SR-restored original-size proxy — original=%dx%d full-SR=%s proxy=%s",
                    self._original_size[0], self._original_size[1],
                    Path(self._sr_fullres_path).name,
                    Path(self._sr_detection_proxy_path).name if self._sr_detection_proxy_path else "N/A",
                )
        elif result.effective_image_size is None and self._original_size:
            result.effective_image_size = self._original_size

        if result.annotated_image_path:
            try:
                self.result_viewer.load_image(result.annotated_image_path)
            except ValueError as exc:
                self._status.warning(f"Annotated image unavailable: {exc}")
                self.result_viewer.clear("Annotation failed — counts are still valid")
        else:
            self.result_viewer.clear(
                "Detection complete but annotated image not available. "
                "Counts and table are valid."
            )

        self._results_model.set_rows(result_to_rows(result))
        self._refresh_summary()

        report = build_operator_report(result, self.model_path_label.text())
        self.report_text.setPlainText(report)

        self._history.add(result)
        self._refresh_history_list()
        self._update_action_states()

        msg = (
            f"{result.inference_mode} complete — {result.total_count} components "
            f"on {result.device} in {result.processing_time:.2f}s"
        )
        self._status.success(msg) if not result.warnings else self._status.warning(msg)
        logger.info("Inference complete: count=%s time=%.2fs", result.total_count, result.processing_time)

    def _on_inference_error(self, error_msg: str) -> None:
        self.run_button_set_enabled(True)
        self._update_action_states()
        self._status.error(error_msg)
        logger.error("Inference error: %s", error_msg)
        title = "CUDA Blocked" if "CUDA is blocked" in error_msg else "Inference Error"
        QMessageBox.critical(self, title, error_msg)

    def save_result_image(self) -> None:
        if self._current_result is None or not self._current_result.annotated_image_path:
            self._status.warning("No result image to save")
            QMessageBox.information(self, "Save Result Image", "Run inference before saving a result image.")
            return
        stem = Path(self._current_result.image_path).stem
        self._save_copy(self._current_result.annotated_image_path, "Save Result Image", f"{stem}_V4_Result.png")

    def export_counts_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Counts CSV",
            "pcb_component_counts.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(ResultsTableModel.HEADERS)
                for row in self._results_model.rows():
                    writer.writerow([row.component_class, row.count, row.mean_confidence, row.notes])
        except OSError as exc:
            self._status.error("Error exporting CSV")
            QMessageBox.critical(self, "Export Counts CSV", str(exc))
            return
        self._status.info("Counts CSV exported")

    def export_pdf_report(self) -> None:
        if self._current_result is None:
            self._status.warning("No detection result to export")
            QMessageBox.information(self, "Export PDF", "Run inference before exporting a PDF report.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export PDF Report",
            self._default_pdf_name(),
            "PDF Files (*.pdf);;All Files (*)",
        )
        if not path:
            return
        try:
            export_pdf(self._current_result, path)
            self._status.success(f"PDF report saved to {path}")
            logger.info("PDF exported: %s", path)
        except Exception as exc:
            self._status.error("Error exporting PDF")
            logger.exception("PDF export failed")
            QMessageBox.critical(self, "Export PDF", str(exc))

    def clear_current(self) -> None:
        self._image_path = None
        self._original_size = None
        self._current_result = None
        self._sr_fullres_path = None
        self._sr_detection_proxy_path = None
        self.input_viewer.clear("Open a PCB image to begin")
        self.sr_viewer.clear("Run super resolution to view result")
        self.result_viewer.clear("Run inference to generate a result view")
        self.sr_status_label.setText("Not run")
        self._results_model.set_rows([])
        self.report_text.clear()
        self.path_label.setText("No image loaded")
        self.path_label.setToolTip("")
        self.size_label.setText("—")
        self._refresh_summary()
        self._update_action_states()
        self._status.ready()

    def show_about(self) -> None:
        try:
            import torch
            import torchvision
            from PyQt6.QtCore import QT_VERSION_STR
            qt_ver = QT_VERSION_STR
            torch_ver = torch.__version__
            tv_ver = torchvision.__version__
            cuda_ver = torch.version.cuda or "N/A"
            cuda_avail = "Available" if torch.cuda.is_available() else "Not available"
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        except Exception as exc:
            logger.debug("Runtime version probe failed: %s", exc)
            qt_ver = "—"
            torch_ver = "—"
            tv_ver = "—"
            cuda_ver = "—"
            cuda_avail = "No"
            gpu = "N/A"

        dlg = QDialog(self)
        dlg.setWindowTitle("About")
        dlg.setMinimumWidth(660)
        dlg.setMaximumWidth(800)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(12)

        app_name = QLabel(APP_TITLE)
        app_name.setStyleSheet(
            "font-size: 16px; font-weight: 800; color: #1E1E1E; padding: 0;"
        )
        layout.addWidget(app_name)

        content = QTextEdit()
        content.setReadOnly(True)
        content.setFrameShape(QFrame.Shape.NoFrame)
        content.setStyleSheet(
            "QTextEdit { background: transparent; color: #1E1E1E; "
            "font-size: 12px; padding: 0; line-height: 150%; }"
        )

        det_name = Path(self._detector.checkpoint_name).name
        try:
            sr_ckpt = Path(self._sr_engine.checkpoint_path).name
        except Exception as exc:
            logger.debug("Could not read SR checkpoint name: %s", exc)
            sr_ckpt = "best_model.pth"

        html = (
            f"<p><b>Course</b><br>Automated Optical Inspection</p>"
            f"<p><b>Project</b><br>Machine Learning-Based PCB Image Restoration "
            f"and Quality Assurance for Automated Optical Inspection</p>"
            f"<p><b>Developed by</b><br>Steven Jones &amp; Isabella Scalia</p>"
            f"<p><b>Contributions</b><br>"
            f"Steven Jones — PCB component counting, Faster R-CNN integration, "
            f"VRAM-aware inference, GUI and reporting.<br>"
            f"Isabella Scalia — PCB super-resolution, image restoration, "
            f"SR model integration.</p>"
            f"<p><b>Runtime</b><br>"
            f"Python {platform.python_version()} &ensp; PyQt {qt_ver} &ensp; "
            f"PyTorch {torch_ver} &ensp; torchvision {tv_ver}<br>"
            f"CUDA {cuda_ver} &ensp; {cuda_avail}<br>"
            f"GPU: {gpu}<br>"
            f"Detection model: {det_name}<br>"
            f"SR model: {sr_ckpt}</p>"
        )
        content.setHtml(html)

        content.setMinimumHeight(340)
        content.setMaximumHeight(520)
        layout.addWidget(content)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setMinimumWidth(90)
        ok_btn.clicked.connect(dlg.accept)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

        dlg.exec()

    def _toggle_sidebar(self) -> None:
        visible = self._sidebar_scroll.isVisible()
        self._sidebar_scroll.setVisible(not visible)

    def _toggle_report(self) -> None:
        visible = self._report_wrapper.isVisible()
        self._report_wrapper.setVisible(not visible)

    def _toggle_fullscreen(self) -> None:
        if self._fullscreen:
            self.showNormal()
        else:
            self.showFullScreen()
        self._fullscreen = not self._fullscreen

    def _fit_to_window(self) -> None:
        self.input_viewer.fit()
        self.sr_viewer.fit()
        self.result_viewer.fit()

    def _reset_layout(self) -> None:
        self._sidebar_scroll.setVisible(True)
        self._report_wrapper.setVisible(True)
        self._root_splitter.setSizes([360, 1000])
        if self._fullscreen:
            self.showNormal()
            self._fullscreen = False
        self._status.info("Layout reset")

    def _refresh_summary(self) -> None:
        self.total_components_label.setText(str(self._results_model.total_count()))
        self.summary_label.setText(self._summary_text())

    def _summary_text(self) -> str:
        lines = [f"{row.component_class}: {row.count}" for row in self._results_model.rows() if row.count > 0]
        return "\n".join(lines) if lines else "No components detected"

    def _default_pdf_name(self) -> str:
        if self._current_result is None:
            return "pcb_report.pdf"
        stem = Path(self._current_result.image_path).stem
        stamp = self._current_result.timestamp.replace("-", "").replace(":", "").replace(" ", "_")
        return f"pcb_report_{stem}_{stamp}.pdf"

    def _refresh_history_list(self) -> None:
        if not hasattr(self, "history_list"):
            return
        self.history_list.clear()
        for record in self._history.records():
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, record.id)
            self._refresh_history_item_style(item, record)
            self.history_list.addItem(item)

    def _refresh_history_item_style(self, item: QListWidgetItem, record) -> None:
        prefix = "★ " if record.pinned else "  "
        mode = record.result.inference_mode
        if mode == "CPU + GPU":
            mode = "CPU+GPU"
        elif mode == "GPU only":
            mode = "GPU"
        elif mode == "Fast Preview":
            mode = "Prev"
        sr_tag = "[SR]" if record.result.sr_image_path else ""
        item.setText(
            f"{prefix}{record.filename}  |  {record.total_count} comps  |  {mode} {sr_tag}  |  {record.timestamp}"
        )
        item.setToolTip(
            f"{record.image_path}\n"
            f"Mode: {record.result.inference_mode}\n"
            f"Source: {record.result.detection_source}\n"
            f"Count: {record.total_count}  |  Status: {record.status}"
        )

    def _on_history_item_clicked(self, item: QListWidgetItem) -> None:
        record_id = item.data(Qt.ItemDataRole.UserRole)
        record = self._history.get(record_id)
        if record is None:
            return
        result = record.result
        self._current_result = result
        self._image_path = Path(result.image_path)
        self._sr_fullres_path = result.sr_image_path
        self._sr_detection_proxy_path = result.sr_detection_proxy_path
        self.path_label.setText(str(self._image_path.name))
        self.path_label.setToolTip(str(self._image_path))
        mode_idx = self.inference_mode_combo.findText(result.inference_mode)
        if mode_idx >= 0:
            self.inference_mode_combo.setCurrentIndex(mode_idx)
        src_idx = self.detection_source_combo.findText(
            "SR-Restored Image (Original Size)" if result.detection_source == "sr_restored" else "Original Image"
        )
        if src_idx >= 0:
            self.detection_source_combo.setCurrentIndex(src_idx)

        try:
            pixmap = self.input_viewer.load_image(result.image_path)
            self.size_label.setText(f"{pixmap.width()} × {pixmap.height()} px")
        except (ValueError, OSError) as exc:
            self._status.warning(f"Cannot reload original image: {exc}")
            return

        if result.annotated_image_path and Path(result.annotated_image_path).exists():
            self.result_viewer.load_image(result.annotated_image_path)
        else:
            self.result_viewer.clear("Cached annotated image is missing")

        if result.sr_image_path and Path(result.sr_image_path).exists():
            try:
                self.sr_viewer.load_image(result.sr_image_path)
                self.sr_status_label.setText("Complete")
            except (ValueError, OSError):
                self.sr_viewer.clear("Cached SR image is missing")
                self.sr_status_label.setText("Not run")
                self._sr_fullres_path = None
        else:
            self.sr_viewer.clear("No SR result in this record")
            self.sr_status_label.setText("Not run")

        self._results_model.set_rows(result_to_rows(result))
        self._refresh_summary()
        self.report_text.setPlainText(build_operator_report(result, self.model_path_label.text()))
        self._update_action_states()
        self._status.info(f"Restored: {Path(result.image_path).name}")

    def _on_history_context_menu(self, pos) -> None:
        item = self.history_list.itemAt(pos)
        if item is None:
            return
        record_id = item.data(Qt.ItemDataRole.UserRole)
        record = self._history.get(record_id)
        if record is None:
            return

        menu = QMenu(self)
        load_action = menu.addAction("Load Result")
        load_action.triggered.connect(lambda: self._on_history_item_clicked(item))
        pin_action = menu.addAction("Unpin" if record.pinned else "Pin")
        pin_action.triggered.connect(lambda: self._handle_pin_toggle(record_id))
        remove_action = menu.addAction("Remove from History")
        remove_action.triggered.connect(lambda: self._handle_delete_history(record_id))
        menu.exec(self.history_list.mapToGlobal(pos))

    def _handle_pin_toggle(self, record_id: str) -> None:
        pinned = self._history.toggle_pin(record_id)
        self._refresh_history_list()
        self._status.info("Record pinned" if pinned else "Record unpinned")

    def _handle_delete_history(self, record_id: str) -> None:
        record = self._history.get(record_id)
        if record and record.pinned:
            self._status.warning("Cannot remove pinned item — unpin first.")
            return
        answer = QMessageBox.question(self, "Remove History Record", "Remove this record from GUI history?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._history.remove(record_id)
        self._refresh_history_list()
        self._status.info("History item removed")

    def clear_unpinned_history(self) -> None:
        removed = self._history.clear_unpinned()
        self._refresh_history_list()
        self._status.info(f"Removed {removed} unpinned history record{'s' if removed != 1 else ''}")
