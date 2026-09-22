import sys
import cv2

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFrame,
    QSizePolicy,
    QSpacerItem,
)
from PySide6.QtCore import Qt
from vision.detection import detect_component, draw_component
from vision.measurement import calculate_width, calculate_height
from calibration.calibration import calculate_scale, pixels_to_real


# ----------------------------------------------------------------------
# Dark professional palette — kept in one place so re-theming is easy.
# ----------------------------------------------------------------------
BG_MAIN = "#0f172a"        # window background
BG_TOPBAR = "#111827"      # top bar / toolbar
BG_PANEL = "#1e293b"       # right panel + group boxes
BG_CANVAS = "#0b1220"      # image workspace background
BORDER = "#334155"
BORDER_LIGHT = "#475569"

TEXT_PRIMARY = "#f1f5f9"
TEXT_MUTED = "#94a3b8"
TEXT_DIM = "#64748b"

ACCENT = "#3b82f6"
ACCENT_HOVER = "#2563eb"
ACCENT_DIM = "#1e3a5f"

CHIP_IDLE = {"bg": "#334155", "fg": "#cbd5e1"}
CHIP_SUCCESS = {"bg": "#064e3b", "fg": "#34d399"}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Visual Measuring System")
        self.setMinimumSize(1150, 720)
        self.setStyleSheet(f"QMainWindow {{ background-color: {BG_MAIN}; }}")

        # ------------------------------------------------------------
        # Workflow state flags — GUI-only, used to enable/disable
        # buttons and update status chips. Replace with real results
        # once vision/detection.py, calibration/calibration.py and
        # vision/measurement.py are wired in.
        # ------------------------------------------------------------
        self.image_loaded = False
        self.border_detected = False
        self.corners_selected = 0
        self.point_coords = []          # placeholder coordinates
        self.calibrated = False
        

        self.image = None
        self.detection_result = None
        self.pixels_per_mm = None


        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_topbar())
        root_layout.addWidget(self._build_body(), 1)

        # Status bar
        self.statusBar().setStyleSheet(
            f"""
            QStatusBar {{
                background-color: {BG_TOPBAR};
                color: {TEXT_MUTED};
                border-top: 1px solid {BORDER};
            }}
            """
        )
        self.status_chip = QLabel("  Ready  ")
        self._style_chip(self.status_chip, CHIP_SUCCESS)
        self.statusBar().addPermanentWidget(self.status_chip)
        self.statusBar().showMessage("Ready")

        self._connect_signals()
        self._refresh_button_states()

    # ------------------------------------------------------------------
    # TOP BAR — app name + compact action toolbar in a single slim row
    # ------------------------------------------------------------------
    def _build_topbar(self):
        bar = QFrame()
        bar.setFixedHeight(58)
        bar.setStyleSheet(
            f"background-color: {BG_TOPBAR}; border-bottom: 1px solid {BORDER};"
        )

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(10)

        title = QLabel("Visual Measuring System")
        title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 15px; font-weight: 600; border: none;"
        )
        layout.addWidget(title)
        layout.addSpacing(24)
        layout.addWidget(self._vertical_separator())
        layout.addSpacing(8)

        self.load_button = self._toolbar_button("Open Image")
        self.detect_button = self._toolbar_button("Detect Border")
        self.select_points_button = self._toolbar_button("Select Points")
        self.calibrate_button = self._toolbar_button("Calibrate")
        self.measure_button = self._toolbar_button("Measure", primary=True)

        for btn in (
            self.load_button,
            self.detect_button,
            self.select_points_button,
            self.calibrate_button,
            self.measure_button,
        ):
            layout.addWidget(btn)

        layout.addStretch(1)

        layout.addWidget(self._vertical_separator())
        layout.addSpacing(8)
        self.reset_button = self._toolbar_button("Reset", ghost=True)
        layout.addWidget(self.reset_button)

        return bar

    def _vertical_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFixedHeight(28)
        line.setStyleSheet(f"color: {BORDER};")
        return line

    def _toolbar_button(self, text, primary=False, ghost=False):
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(self._toolbar_button_style(primary=primary, ghost=ghost))
        return btn

    def _toolbar_button_style(self, primary=False, ghost=False):
        if primary:
            return f"""
                QPushButton {{
                    background-color: {ACCENT};
                    color: #ffffff;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-size: 12px;
                    font-weight: 600;
                }}
                QPushButton:hover {{ background-color: {ACCENT_HOVER}; }}
                QPushButton:disabled {{ background-color: #1e293b; color: {TEXT_DIM}; }}
            """
        if ghost:
            return f"""
                QPushButton {{
                    background-color: transparent;
                    color: {TEXT_MUTED};
                    border: 1px solid {BORDER};
                    border-radius: 5px;
                    padding: 7px 14px;
                    font-size: 12px;
                }}
                QPushButton:hover {{ color: {TEXT_PRIMARY}; border-color: {BORDER_LIGHT}; }}
            """
        return f"""
            QPushButton {{
                background-color: transparent;
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER};
                border-radius: 5px;
                padding: 8px 14px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{ background-color: {ACCENT_DIM}; border-color: {ACCENT}; }}
            QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; }}
        """

    # ------------------------------------------------------------------
    # BODY — large image workspace (main focus) + compact right panel
    # ------------------------------------------------------------------
    def _build_body(self):
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        layout.addWidget(self._build_workspace(), 4)
        layout.addWidget(self._build_side_panel(), 1)

        return wrapper

    def _build_workspace(self):
        panel = QFrame()
        panel.setStyleSheet(
            f"QFrame {{ background-color: {BG_PANEL}; border: 1px solid {BORDER}; "
            "border-radius: 8px; }}"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Status strip: quick-glance state of the pipeline, docked to the
        # workspace itself rather than a separate section.
        strip = QHBoxLayout()
        strip.setSpacing(8)

        self.image_chip = QLabel("  Image: Not Loaded  ")
        self.border_chip = QLabel("  Border: Not Detected  ")
        self.points_chip = QLabel("  Points: 0 / 4  ")
        for chip in (self.image_chip, self.border_chip, self.points_chip):
            self._style_chip(chip, CHIP_IDLE)
            strip.addWidget(chip)
        strip.addStretch(1)

        layout.addLayout(strip)

        # The canvas itself
        self.image_label = QLabel(
            "No image loaded\n\nUse \u201cOpen Image\u201d in the toolbar to begin"
        )
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet(
            f"""
            QLabel {{
                background-color: {BG_CANVAS};
                border: 1px dashed {BORDER_LIGHT};
                border-radius: 6px;
                color: {TEXT_DIM};
                font-size: 13px;
            }}
            """
        )
        layout.addWidget(self.image_label, 1)

        return panel

    def _build_side_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(270)
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        layout.addWidget(self._build_calibration_group())
        layout.addWidget(self._build_measurement_group())
        layout.addStretch(1)

        return panel

    def _groupbox_style(self):
        return f"""
            QGroupBox {{
                background-color: {BG_PANEL};
                border: 1px solid {BORDER};
                border-radius: 8px;
                margin-top: 10px;
                font-weight: 600;
                font-size: 12px;
                color: {TEXT_PRIMARY};
                padding: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: {TEXT_MUTED};
            }}
        """

    def _panel_button_style(self):
        return f"""
            QPushButton {{
                background-color: {ACCENT};
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background-color: {ACCENT_HOVER}; }}
            QPushButton:disabled {{ background-color: #1e293b; color: {TEXT_DIM}; }}
        """

    # ---- Calibration -----------------------------------------------
    def _build_calibration_group(self):
        group = QGroupBox("CALIBRATION")
        group.setStyleSheet(self._groupbox_style())
        layout = QVBoxLayout()
        layout.setSpacing(8)

        ref_row = QHBoxLayout()
        ref_label = QLabel("Known length (mm)")
        ref_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; border: none;")
        self.reference_input = QLineEdit()
        self.reference_input.setPlaceholderText("25.00")
        self.reference_input.setStyleSheet(
            f"""
            QLineEdit {{
                background-color: {BG_CANVAS};
                border: 1px solid {BORDER};
                border-radius: 4px;
                padding: 5px;
                color: {TEXT_PRIMARY};
                font-size: 12px;
            }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
            """
        )
        layout.addWidget(ref_label)
        layout.addWidget(self.reference_input)

        self.calibration_chip = QLabel("  Not Calibrated  ")
        self._style_chip(self.calibration_chip, CHIP_IDLE)
        layout.addWidget(self.calibration_chip)

        group.setLayout(layout)
        return group

    # ---- Measurement -------------------------------------------------
    def _build_measurement_group(self):
        group = QGroupBox("MEASUREMENT RESULT")
        group.setStyleSheet(self._groupbox_style())
        layout = QVBoxLayout()
        layout.setSpacing(8)

        points_label = QLabel("Selected Points")
        points_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; border: none;")
        layout.addWidget(points_label)

        self.points_list_label = QLabel("No points selected yet")
        self.points_list_label.setWordWrap(True)
        self.points_list_label.setStyleSheet(
            f"""
            QLabel {{
                background-color: {BG_CANVAS};
                border: 1px solid {BORDER};
                border-radius: 4px;
                color: {TEXT_MUTED};
                font-size: 11px;
                padding: 6px;
            }}
            """
        )
        layout.addWidget(self.points_list_label)

        self.result_label = QLabel("-- mm")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            f"""
            QLabel {{
                background-color: {ACCENT_DIM};
                color: #93c5fd;
                border: 1px solid {ACCENT};
                border-radius: 6px;
                font-size: 22px;
                font-weight: 700;
                padding: 12px;
            }}
            """
        )
        layout.addWidget(self.result_label)

        group.setLayout(layout)
        return group

    # ------------------------------------------------------------------
    # style helpers
    # ------------------------------------------------------------------
    def _style_chip(self, label, palette):
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(
            f"""
            QLabel {{
                background-color: {palette['bg']};
                color: {palette['fg']};
                border-radius: 9px;
                font-size: 11px;
                font-weight: 600;
                padding: 3px 8px;
                border: none;
            }}
            """
        )

    # ------------------------------------------------------------------
    # Signals — placeholder handlers only. Replace bodies with real
    # calls into vision/detection.py, calibration/calibration.py and
    # vision/measurement.py once that logic is ready.
    # ------------------------------------------------------------------
    def _connect_signals(self):
        self.load_button.clicked.connect(self.on_load_image)
        self.reset_button.clicked.connect(self.on_clear)
        self.detect_button.clicked.connect(self.on_detect_border)
        self.select_points_button.clicked.connect(self.on_select_points)
        self.calibrate_button.clicked.connect(self.on_calibrate)
        self.measure_button.clicked.connect(self.on_measure)

    def on_load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return

        image = cv2.imread(file_path)

        if image is None:
            self._set_status("Could not load image", CHIP_IDLE)
            return

        self.image = image
        self.image_loaded = True

        # Convert OpenCV BGR image to RGB for Qt
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width

        q_image = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        self._style_chip(self.image_chip, CHIP_SUCCESS)
        self.image_chip.setText("  Image: Loaded  ")
        self._set_status("Image loaded", CHIP_SUCCESS)
        self._refresh_button_states()

    def on_clear(self):
        self.image_loaded = False
        self.border_detected = False
        self.corners_selected = 0
        self.point_coords = []
        self.calibrated = False

        self.image_label.setText(
            "No image loaded\n\nUse \u201cOpen Image\u201d in the toolbar to begin"
        )
        self._style_chip(self.image_chip, CHIP_IDLE)
        self.image_chip.setText("  Image: Not Loaded  ")
        self._style_chip(self.border_chip, CHIP_IDLE)
        self.border_chip.setText("  Border: Not Detected  ")
        self._style_chip(self.points_chip, CHIP_IDLE)
        self.points_chip.setText("  Points: 0 / 4  ")
        self._style_chip(self.calibration_chip, CHIP_IDLE)
        self.calibration_chip.setText("  Not Calibrated  ")
        self.points_list_label.setText("No points selected yet")
        self.result_label.setText("-- mm")

        self._set_status("Cleared", CHIP_IDLE)
        self._refresh_button_states()

    def on_detect_border(self):
        # TODO: call vision/detection.py and draw the contour over the
        # image currently shown in self.image_label.
        self.border_detected = True
        self._style_chip(self.border_chip, CHIP_SUCCESS)
        self.border_chip.setText("  Border: Detected  ")
        self._set_status("Border detected", CHIP_SUCCESS)
        self._refresh_button_states()

    def on_select_points(self):
        # TODO: replace with real click-to-select points on the canvas
        # (e.g. via a custom mousePressEvent on the image widget).
        if self.corners_selected < 4:
            self.corners_selected += 1
            self.point_coords.append((self.corners_selected * 40, self.corners_selected * 30))

        self.points_chip.setText(f"  Points: {self.corners_selected} / 4  ")
        self._style_chip(
            self.points_chip,
            CHIP_SUCCESS if self.corners_selected >= 2 else CHIP_IDLE,
        )
        lines = [f"P{i+1}: {pt}" for i, pt in enumerate(self.point_coords)]
        self.points_list_label.setText("\n".join(lines) if lines else "No points selected yet")

        self._set_status(f"{self.corners_selected} point(s) selected", CHIP_IDLE)
        self._refresh_button_states()

    def on_calibrate(self):
        # TODO: call calibration/calibration.py using the value typed
        # into self.reference_input.
        self.calibrated = True
        self._style_chip(self.calibration_chip, CHIP_SUCCESS)
        self.calibration_chip.setText("  Calibrated  ")
        self._set_status("Calibration complete", CHIP_SUCCESS)
        self._refresh_button_states()

    def on_measure(self):
        # TODO: call vision/measurement.py using the selected points and
        # the calibration factor, then display the real result.
        self.result_label.setText("25.40 mm")
        self._set_status("Measurement complete", CHIP_SUCCESS)

    def _set_status(self, message, palette):
        self.statusBar().showMessage(message)
        self.status_chip.setText(f"  {message}  ")
        self._style_chip(self.status_chip, palette)

    def _refresh_button_states(self):
        """Enable/disable toolbar buttons so the workflow order is obvious."""
        self.detect_button.setEnabled(self.image_loaded)
        self.select_points_button.setEnabled(self.border_detected)
        self.calibrate_button.setEnabled(self.border_detected)
        self.measure_button.setEnabled(self.calibrated and self.corners_selected >= 2)


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()