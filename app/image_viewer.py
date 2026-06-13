import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QRectF, QSize
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QImageReader,
    QPainter,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

ZOOM_FACTOR = 1.25
ZOOM_FACTOR_INV = 1.0 / ZOOM_FACTOR
logger = logging.getLogger(__name__)


class _InspectionGraphicsView(QGraphicsView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setOptimizationFlags(
            QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing
            | QGraphicsView.OptimizationFlag.DontSavePainterState
        )

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        factor = ZOOM_FACTOR if event.angleDelta().y() > 0 else ZOOM_FACTOR_INV
        self.scale(factor, factor)


class ImageViewer(QWidget):
    def __init__(self, title: str, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder

        self._title_label = QLabel(title)
        self._title_label.setObjectName("panelTitle")

        self._scene = QGraphicsScene(self)
        self._view = _InspectionGraphicsView(self)
        self._view.setScene(self._scene)
        self._view.setMinimumSize(200, 160)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._fit_btn = QToolButton()
        self._fit_btn.setObjectName("viewerZoomBtn")
        self._fit_btn.setText("Fit")
        self._fit_btn.setToolTip("Fit image to window")
        self._fit_btn.clicked.connect(self.fit)

        self._actual_btn = QToolButton()
        self._actual_btn.setObjectName("viewerZoomBtn")
        self._actual_btn.setText("1:1")
        self._actual_btn.setToolTip("Actual size (100%)")
        self._actual_btn.clicked.connect(self.actual_size)

        self._zoom_out_btn = QToolButton()
        self._zoom_out_btn.setObjectName("viewerZoomBtn")
        self._zoom_out_btn.setText("\u2212")
        self._zoom_out_btn.setToolTip("Zoom out")
        self._zoom_out_btn.clicked.connect(self.zoom_out)

        self._zoom_in_btn = QToolButton()
        self._zoom_in_btn.setObjectName("viewerZoomBtn")
        self._zoom_in_btn.setText("+")
        self._zoom_in_btn.setToolTip("Zoom in")
        self._zoom_in_btn.clicked.connect(self.zoom_in)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 4, 0, 4)
        controls.setSpacing(6)
        controls.addWidget(self._fit_btn)
        controls.addWidget(self._actual_btn)
        controls.addWidget(self._zoom_out_btn)
        controls.addWidget(self._zoom_in_btn)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)
        layout.addWidget(self._title_label)
        layout.addLayout(controls)
        layout.addWidget(self._view, 1)

        self._show_placeholder()

    def load_image(self, path: str | Path) -> QPixmap:
        path = str(path)
        MAX_DISPLAY_DIM = 4096

        try:
            from PIL import Image as PILImage

            PILImage.MAX_IMAGE_PIXELS = None
            pil_img = PILImage.open(path)
            if pil_img.mode not in ("RGB", "RGBA"):
                pil_img = pil_img.convert("RGB")
            w, h = pil_img.size
            if max(w, h) > MAX_DISPLAY_DIM:
                scale = MAX_DISPLAY_DIM / max(w, h)
                pil_img = pil_img.resize(
                    (int(w * scale), int(h * scale)), PILImage.LANCZOS)
            if pil_img.mode == "RGBA":
                fmt = QImage.Format.Format_RGBA8888
                stride = pil_img.width * 4
                data = pil_img.tobytes("raw", "RGBA")
            else:
                fmt = QImage.Format.Format_RGB888
                stride = pil_img.width * 3
                data = pil_img.tobytes("raw", "RGB")
            qimage = QImage(data, pil_img.width, pil_img.height, stride, fmt)
            if not qimage.isNull():
                pixmap = QPixmap.fromImage(qimage)
                self._set_pixmap_internal(pixmap)
                self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                return pixmap
        except Exception as exc:
            logger.debug("PIL image load failed for %s; trying QImageReader: %s", path, exc)

        reader = QImageReader(path)
        reader.setAutoTransform(True)
        native_size = reader.size()
        if native_size.isValid():
            w, h = native_size.width(), native_size.height()
            if max(w, h) > MAX_DISPLAY_DIM:
                scale = MAX_DISPLAY_DIM / max(w, h)
                reader.setScaledSize(QSize(int(w * scale), int(h * scale)))
        qimage = reader.read()
        if qimage.isNull():
            raise ValueError(f"Could not load image: {path}")
        pixmap = QPixmap.fromImage(qimage)
        self._set_pixmap_internal(pixmap)
        self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        return pixmap

    def pixmap(self) -> QPixmap | None:
        return self._pixmap

    def clear(self, placeholder: str) -> None:
        self._pixmap = None
        self._placeholder = placeholder
        self._show_placeholder()

    def fit(self) -> None:
        if self._pixmap is not None:
            self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self) -> None:
        self._view.scale(ZOOM_FACTOR, ZOOM_FACTOR)

    def zoom_out(self) -> None:
        self._view.scale(ZOOM_FACTOR_INV, ZOOM_FACTOR_INV)

    def actual_size(self) -> None:
        self._view.resetTransform()

    def _set_pixmap_internal(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._scene.clear()
        self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

    def _show_placeholder(self) -> None:
        self._scene.clear()
        text = self._scene.addSimpleText(self._placeholder)
        text.setBrush(QBrush(QColor("#87909E")))
        font = QFont()
        font.setPointSize(12)
        text.setFont(font)
        text_rect = text.boundingRect()
        scene_rect = self._scene.sceneRect()
        text.setPos(
            (scene_rect.width() - text_rect.width()) / 2,
            (scene_rect.height() - text_rect.height()) / 2,
        )
