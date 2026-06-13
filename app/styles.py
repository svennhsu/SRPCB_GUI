def application_stylesheet() -> str:
    return """
QMainWindow {
    background-color: #F3F3F3;
}
QWidget {
    color: #1E1E1E;
    font-family: "Segoe UI", "Ubuntu", "Noto Sans", sans-serif;
    font-size: 11px;
}

QMenuBar {
    background: #2D2D30;
    color: #CCCCCC;
    border-bottom: 1px solid #1E1E1E;
    padding: 1px 2px;
}
QMenuBar::item {
    background: transparent;
    padding: 5px 12px;
    border-radius: 2px;
}
QMenuBar::item:selected {
    background: #3E3E42;
}
QMenuBar::item:pressed {
    background: #007ACC;
}

QMenu {
    background: #FFFFFF;
    color: #1E1E1E;
    border: 1px solid #C7D0DD;
    padding: 4px 0;
}
QMenu::item {
    padding: 6px 32px 6px 22px;
}
QMenu::item:selected {
    background: #DDEEFF;
    color: #1E1E1E;
}
QMenu::item:disabled {
    color: #A0AAB4;
}
QMenu::separator {
    height: 1px;
    background: #D6DEE8;
    margin: 4px 8px;
}

QToolBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #C7D0DD;
    spacing: 6px;
    padding: 5px 10px;
}
QToolBar QToolButton {
    color: #1E1E1E;
    background: #F6F8FA;
    border: 1px solid #C7D0DD;
    border-radius: 3px;
    padding: 6px 16px;
    font-size: 11px;
}
QToolBar QToolButton:hover {
    background: #DDEEFF;
    border: 1px solid #007ACC;
}
QToolBar QToolButton:pressed {
    background: #007ACC;
    color: #FFFFFF;
}
QToolBar QToolButton:disabled {
    background: #F3F3F3;
    color: #A0AAB4;
    border-color: #D6DEE8;
}

QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #C7D0DD;
    color: #1E1E1E;
    padding: 5px 10px;
    border-radius: 3px;
}
QPushButton:hover {
    background-color: #DDEEFF;
    border: 1px solid #007ACC;
}
QPushButton:pressed {
    background-color: #007ACC;
    color: #FFFFFF;
}
QPushButton:disabled {
    background: #F3F3F3;
    color: #A0AAB4;
    border: 1px solid #D6DEE8;
}
QPushButton#primaryButton {
    background-color: #007ACC;
    border: 1px solid #005A9E;
    color: #FFFFFF;
    font-weight: 700;
    min-height: 24px;
}
QPushButton#primaryButton:hover {
    background-color: #1A8CE8;
    border: 1px solid #007ACC;
}
QPushButton#primaryButton:pressed {
    background-color: #005A9E;
}

QLabel#appTitle {
    color: #1E1E1E;
    font-size: 15px;
    font-weight: 800;
    padding: 0 0 6px 0;
}
QLabel#mutedValue {
    color: #5F6B7A;
    font-size: 11px;
    padding: 1px 0;
}
QLabel#totalCount {
    color: #007ACC;
    font-size: 26px;
    font-weight: 800;
}
QLabel#panelTitle {
    color: #1E1E1E;
    font-weight: 700;
    padding: 2px 4px;
}
QLabel#imageCanvas {
    background: #FFFFFF;
    border: 1px solid #C7D0DD;
    border-radius: 2px;
    color: #87909E;
    font-size: 12px;
}

QGroupBox {
    background: #FFFFFF;
    border: 1px solid #C7D0DD;
    border-radius: 4px;
    margin-top: 18px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}

QFrame#sidebar {
    background: #F6F8FA;
}
QScrollArea#sidebarScroll {
    background: #F6F8FA;
    border-right: 1px solid #C7D0DD;
}

QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTableView, QTextEdit, QPlainTextEdit {
    background-color: #FFFFFF;
    color: #1E1E1E;
    border: 1px solid #C7D0DD;
    padding: 3px 5px;
    selection-background-color: #007ACC;
    selection-color: #FFFFFF;
}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #007ACC;
}

QComboBox {
    padding: 3px 6px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background: #FFFFFF;
    color: #1E1E1E;
    border: 1px solid #C7D0DD;
    selection-background-color: #DDEEFF;
    selection-color: #1E1E1E;
}

QTableView {
    alternate-background-color: #F6F8FA;
    gridline-color: #D6DEE8;
    border: 1px solid #C7D0DD;
}
QHeaderView::section {
    background: #E7ECF3;
    color: #1E1E1E;
    border: 0;
    border-right: 1px solid #C7D0DD;
    border-bottom: 1px solid #C7D0DD;
    padding: 6px;
    font-weight: 700;
}

QTextEdit, QPlainTextEdit {
    background-color: #FFFFFF;
    color: #1E1E1E;
    border: 1px solid #C7D0DD;
    padding: 8px 10px;
    line-height: 150%;
}

QScrollArea {
    background: transparent;
    border: none;
}
QScrollBar:horizontal {
    background: #F3F3F3;
    height: 10px;
    border: 1px solid #C7D0DD;
}
QScrollBar::handle:horizontal {
    background: #C7D0DD;
    min-width: 30px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #A0AAB4;
}
QScrollBar:vertical {
    background: #F3F3F3;
    width: 10px;
    border: 1px solid #C7D0DD;
}
QScrollBar::handle:vertical {
    background: #C7D0DD;
    min-height: 30px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #A0AAB4;
}
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
}

QListWidget {
    background: #FFFFFF;
    border: 1px solid #C7D0DD;
    alternate-background-color: #F6F8FA;
    padding: 1px;
}
QListWidget::item {
    padding: 8px 10px;
    border-bottom: 1px solid #E7ECF3;
}
QListWidget::item:selected {
    background: #DDEEFF;
    color: #1E1E1E;
}

QSplitter::handle:horizontal, QSplitter::handle:vertical {
    background-color: #C7D0DD;
    width: 4px;
    height: 4px;
}

QToolButton#sectionHeader {
    background: #E7ECF3;
    border: 1px solid #C7D0DD;
    border-radius: 3px;
    color: #1E1E1E;
    font-weight: 700;
    padding: 7px 10px;
    text-align: left;
    margin-top: 2px;
}

QStatusBar {
    background: #F3F3F3;
    color: #1E1E1E;
    border-top: 1px solid #C7D0DD;
    font-size: 10px;
    padding: 1px 6px;
}
QStatusBar QLabel#operatorStatusLine {
    background: transparent;
    color: #1E1E1E;
    font-weight: 600;
    padding: 1px 4px;
    font-size: 10px;
}

QGraphicsView {
    background: #FFFFFF;
    border: 1px solid #C7D0DD;
    border-radius: 2px;
}

QToolButton#viewerZoomBtn {
    background: #FFFFFF;
    border: 1px solid #C7D0DD;
    border-radius: 2px;
    padding: 3px 8px;
    font-size: 11px;
    color: #1E1E1E;
}
QToolButton#viewerZoomBtn:hover {
    background: #DDEEFF;
    border: 1px solid #007ACC;
}
QToolButton#viewerZoomBtn:pressed {
    background: #007ACC;
    color: #FFFFFF;
}

QDialog {
    background-color: #FFFFFF;
    color: #1E1E1E;
}
QMessageBox {
    background-color: #FFFFFF;
    color: #1E1E1E;
}
QMessageBox QLabel {
    color: #1E1E1E;
    font-size: 11px;
}
QMessageBox QPushButton {
    min-width: 70px;
    padding: 5px 16px;
}

QTabWidget::pane {
    background: #FFFFFF;
    border: 1px solid #C7D0DD;
    border-top: none;
}
QTabBar::tab {
    background: #E7ECF3;
    color: #1E1E1E;
    border: 1px solid #C7D0DD;
    border-bottom: none;
    padding: 6px 16px;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
}
QTabBar::tab:selected {
    background: #FFFFFF;
    color: #007ACC;
    font-weight: 700;
}
QTabBar::tab:hover {
    background: #DDEEFF;
}
QTabBar::tab:!selected {
    margin-top: 2px;
}

QToolTip {
    background-color: #FFFFFF;
    color: #1E1E1E;
    border: 1px solid #C7D0DD;
    padding: 3px 6px;
}
"""
