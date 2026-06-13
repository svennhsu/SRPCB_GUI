from enum import Enum

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy, QStatusBar


class StatusSeverity(Enum):
    NEUTRAL = "neutral"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class StatusLineController:

    _COLORS = {
        StatusSeverity.NEUTRAL: ("#F3F3F3", "#1E1E1E"),
        StatusSeverity.INFO: ("#F3F3F3", "#1E1E1E"),
        StatusSeverity.SUCCESS: ("#F0F5F0", "#3B7D3B"),
        StatusSeverity.WARNING: ("#FAF6F0", "#8B6914"),
        StatusSeverity.ERROR: ("#FAF0F0", "#A33"),
    }

    def __init__(self) -> None:
        self._status_bar: QStatusBar | None = None
        self._label: QLabel | None = None

    def attach(self, status_bar: QStatusBar) -> None:
        self._status_bar = status_bar
        self._status_bar.clearMessage()
        self._status_bar.setContentsMargins(2, 0, 8, 0)
        self._status_bar.setSizeGripEnabled(False)

        self._label = QLabel()
        self._label.setObjectName("operatorStatusLine")
        self._label.setMinimumHeight(22)
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self._status_bar.addPermanentWidget(self._label, 1)
        self.set_status("Ready", StatusSeverity.NEUTRAL)

    def set_status(self, message: str, severity: StatusSeverity = StatusSeverity.INFO) -> None:
        if self._status_bar is None or self._label is None:
            return
        bg, fg = self._COLORS[severity]
        self._status_bar.setStyleSheet(
            "QStatusBar { "
            f"background: {bg}; border-top: 1px solid #C7D0DD; "
            "} "
            "QLabel#operatorStatusLine { "
            f"background: transparent; color: {fg}; font-weight: 600; "
            "padding: 1px 4px; font-size: 10px; "
            "}"
        )
        self._label.setText(message)

    def ready(self) -> None:
        self.set_status("Ready", StatusSeverity.NEUTRAL)

    def info(self, message: str) -> None:
        self.set_status(message, StatusSeverity.INFO)

    def success(self, message: str) -> None:
        self.set_status(message, StatusSeverity.SUCCESS)

    def warning(self, message: str) -> None:
        self.set_status(message, StatusSeverity.WARNING)

    def error(self, message: str) -> None:
        self.set_status(message, StatusSeverity.ERROR)
