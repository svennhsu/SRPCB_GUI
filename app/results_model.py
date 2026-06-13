from dataclasses import dataclass

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt

from inference.aoi_inference_engine import CLASS_NAMES
from inference.detection_result import DetectionResult


@dataclass(frozen=True)
class DetectionRow:
    component_class: str
    count: int
    mean_confidence: str
    notes: str


def result_to_rows(result: DetectionResult) -> list[DetectionRow]:
    rows = []
    det_by_class: dict[str, list[float]] = {}
    for det in result.detections:
        det_by_class.setdefault(det.class_name, []).append(det.score)

    for class_name in CLASS_NAMES[1:]:
        scores = det_by_class.get(class_name, [])
        rows.append(
            DetectionRow(
                class_name.capitalize(),
                len(scores),
                f"{sum(scores) / len(scores):.2f}" if scores else "-",
                "",
            )
        )
    return rows


class ResultsTableModel(QAbstractTableModel):
    HEADERS = ("Component Class", "Count", "Mean Confidence", "Notes")

    def __init__(self, rows: list[DetectionRow] | None = None) -> None:
        super().__init__()
        self._rows = rows or []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.HEADERS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.TextAlignmentRole):
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if index.column() == 1:
                return Qt.AlignmentFlag.AlignCenter
            return Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft

        row = self._rows[index.row()]
        values = (row.component_class, row.count, row.mean_confidence, row.notes)
        return values[index.column()]

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def set_rows(self, rows: list[DetectionRow]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rows(self) -> list[DetectionRow]:
        return list(self._rows)

    def total_count(self) -> int:
        return sum(row.count for row in self._rows)
