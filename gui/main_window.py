import sys
import cv2

from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
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
from PySide6.QtCore import Qt, Signal, QPoint, QRectF
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


# ----------------------------------------------------------------------
# ImageCanvas — a QLabel subclass that displays a cv2/numpy image and
# lets the user click points directly on it.
#
# The pixmap shown is scaled to fit the widget (keeping aspect ratio)
# and centered, so a raw click position is NOT the same as a pixel
# position in the original image. This widget does that conversion so
# the rest of the app can always work in ORIGINAL image coordinates,
# regardless of window size or image resolution.
# ----------------------------------------------------------------------
class ImageCanvas(QLabel):

    point_clicked = Signal(int, int)  # emits (x, y) in original image pixels

    POINT_LABELS = ["1: Top-Left", "2: Top-Right", "3: Bottom-Left", "4: Bottom-Right"]

    # Layout constants for the point markers + their labels. Kept as
    # class constants so the drawing code and the "does this click land
    # near a point" logic (if ever needed) stay in sync.
    DOT_RADIUS = 5
    LABEL_GAP = 8        # space between the dot's edge and the label box
    LABEL_H_PAD = 6      # horizontal padding inside the label box
    LABEL_V_PAD = 4      # vertical padding inside the label box

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)

        self._cv_image = None       # original numpy/cv2 image
        self._base_pixmap = None    # full-resolution QPixmap of _cv_image

        self.selecting_points = False
        self.max_points = 4
        self.points = []            # [(x, y), ...] in ORIGINAL image pixels

    # ---- loading / clearing ------------------------------------------
    def set_image(self, cv_image):
        """Load a new cv2/numpy image and display it. Clears old points."""
        self._cv_image = cv_image
        self._base_pixmap = self._cv_to_pixmap(cv_image)
        self.points = []
        self.selecting_points = False
        self._refresh_pixmap()

    def clear_image(self):
        self._cv_image = None
        self._base_pixmap = None
        self.points = []
        self.selecting_points = False
        self.setPixmap(QPixmap())
        self.setText("No image loaded\n\nUse \u201cOpen Image\u201d in the toolbar to begin")

    # ---- point-selection mode -----------------------------------------
    def start_point_selection(self):
        self.points = []
        self.selecting_points = True
        self._refresh_pixmap()

    def reset_points(self):
        self.points = []
        self._refresh_pixmap()

    def set_points(self, points, selecting=None):
        """Restore a previously selected set of points without discarding
        them — used after the base pixmap is replaced (e.g. by Detect
        Border) so manual selection isn't silently thrown away.

        `points` are ORIGINAL image coordinates, same format as self.points.
        `selecting`, if given, updates whether selection mode stays active
        (e.g. resume selecting if the user hadn't picked all 4 yet).
        """
        self.points = list(points)
        if selecting is not None:
            self.selecting_points = selecting
        self._refresh_pixmap()

    # ---- mouse handling -------------------------------------------------
    def mousePressEvent(self, event):
        if not self.selecting_points or self._cv_image is None:
            return
        if event.button() != Qt.LeftButton:
            return
        if len(self.points) >= self.max_points:
            return

        original_coords = self._widget_to_image_coords(event.position().toPoint())
        if original_coords is None:
            return  # clicked in the empty letterbox area, not on the image

        self.points.append(original_coords)
        self.point_clicked.emit(*original_coords)
        self._refresh_pixmap()

        if len(self.points) >= self.max_points:
            self.selecting_points = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    # ---- coordinate mapping ----------------------------------------------
    def _display_geometry(self):
        """(scaled_width, scaled_height, x_offset, y_offset, scale) of the
        image as it's actually drawn inside this widget right now."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return None

        label_w, label_h = self.width(), self.height()
        pix_w, pix_h = self._base_pixmap.width(), self._base_pixmap.height()
        if label_w <= 0 or label_h <= 0 or pix_w <= 0 or pix_h <= 0:
            return None

        scale = min(label_w / pix_w, label_h / pix_h)
        scaled_w, scaled_h = pix_w * scale, pix_h * scale
        x_offset = (label_w - scaled_w) / 2
        y_offset = (label_h - scaled_h) / 2
        return scaled_w, scaled_h, x_offset, y_offset, scale

    def _widget_to_image_coords(self, pos):
        geometry = self._display_geometry()
        if geometry is None:
            return None

        scaled_w, scaled_h, x_offset, y_offset, scale = geometry
        x = pos.x() - x_offset
        y = pos.y() - y_offset

        if x < 0 or y < 0 or x > scaled_w or y > scaled_h:
            return None

        return int(x / scale), int(y / scale)

    # ---- drawing -------------------------------------------------------------
    def _refresh_pixmap(self):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return

        scaled = self._base_pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        if self.points:
            scaled = self._draw_points(scaled)
        self.setPixmap(scaled)

    def _draw_points(self, pixmap):
        geometry = self._display_geometry()
        if geometry is None:
            return pixmap

        _, _, _, _, scale = geometry
        painted = QPixmap(pixmap)

        # Use the ACTUAL pixmap's own dimensions as the boundary for label
        # clamping (not the recomputed geometry, which can differ from the
        # real pixmap by rounding). This keeps every label's bounding box
        # inside [0, canvas_w] x [0, canvas_h].
        canvas_w, canvas_h = painted.width(), painted.height()

        painter = QPainter(painted)
        painter.setRenderHint(QPainter.Antialiasing)

        dot_pen = QPen(QColor(ACCENT))
        dot_pen.setWidth(2)
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        for i, (orig_x, orig_y) in enumerate(self.points):
            x, y = int(orig_x * scale), int(orig_y * scale)

            # -- the point marker itself --
            painter.setPen(dot_pen)
            painter.setBrush(QColor(ACCENT))
            painter.drawEllipse(QPoint(x, y), self.DOT_RADIUS, self.DOT_RADIUS)

            # -- the label, positioned so it always stays inside the canvas --
            label = self.POINT_LABELS[i] if i < len(self.POINT_LABELS) else str(i + 1)
            text_rect = metrics.boundingRect(label)
            box_w = text_rect.width() + self.LABEL_H_PAD * 2
            box_h = text_rect.height() + self.LABEL_V_PAD * 2

            # Prefer the point's right side; flip to the left if placing it
            # on the right would spill past the right edge of the canvas.
            offset = self.DOT_RADIUS + self.LABEL_GAP
            if x + offset + box_w <= canvas_w:
                box_x = x + offset
            else:
                box_x = x - offset - box_w
            box_x = max(0.0, min(box_x, canvas_w - box_w))

            box_y = y - box_h / 2
            box_y = max(0.0, min(box_y, canvas_h - box_h))

            label_rect = QRectF(box_x, box_y, box_w, box_h)
            painter.fillRect(label_rect, QColor(BG_TOPBAR))
            painter.setPen(QColor(TEXT_PRIMARY))
            painter.drawText(label_rect, Qt.AlignCenter, label)

        painter.end()
        return painted

    @staticmethod
    def _cv_to_pixmap(cv_image):
        if cv_image.ndim == 2:
            # Grayscale image (2-D array, no channel dimension). Convert to
            # RGB first so the rest of the logic below doesn't need to
            # special-case it, and loading a grayscale file doesn't crash.
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            h, w, ch = rgb_image.shape
            q_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888).copy()
        elif cv_image.shape[2] == 4:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)
            h, w, ch = rgb_image.shape
            q_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGBA8888).copy()
        else:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            q_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(q_image)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Visual Measuring System")
        self.setMinimumSize(1150, 720)
        self.setStyleSheet(f"QMainWindow {{ background-color: {BG_MAIN}; }}")

        # ------------------------------------------------------------
        # Workflow state flags — GUI-only, used to enable/disable
        # buttons and update status chips.
        # ------------------------------------------------------------
        self.image_loaded = False
        self.border_detected = False
        self.corners_selected = 0
        self.point_coords = []          # manually selected points (original image px)
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

        # The canvas itself — a clickable ImageCanvas instead of a plain QLabel
        self.image_label = ImageCanvas()
        self.image_label.setText(
            "No image loaded\n\nUse \u201cOpen Image\u201d in the toolbar to begin"
        )
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
    # Signals
    # ------------------------------------------------------------------
    def _connect_signals(self):
        self.load_button.clicked.connect(self.on_load_image)
        self.reset_button.clicked.connect(self.on_clear)
        self.detect_button.clicked.connect(self.on_detect_border)
        self.select_points_button.clicked.connect(self.on_select_points)
        self.calibrate_button.clicked.connect(self.on_calibrate)
        self.measure_button.clicked.connect(self.on_measure)
        self.image_label.point_clicked.connect(self._on_canvas_point_clicked)

    def _display_image(self, image):
        """Show a cv2/numpy image on the canvas — scaling, centering and
        coordinate conversion are handled inside ImageCanvas itself."""
        self.image_label.set_image(image)

    # ------------------------------------------------------------------
    # NEW: converts the manually clicked points into the dict shape that
    # vision/measurement.py expects. Returns None if the user hasn't
    # finished selecting all 4 points yet.
    # ------------------------------------------------------------------
    def _get_selected_points_dict(self):
        """Convert self.point_coords (a list of 4 (x, y) tuples, collected
        in Top-Left -> Top-Right -> Bottom-Left -> Bottom-Right click
        order) into the dict shape calculate_width()/calculate_height()
        expect. Returns None if exactly 4 points haven't been selected."""
        if len(self.point_coords) != 4:
            return None

        return {
            "top_left": self.point_coords[0],
            "top_right": self.point_coords[1],
            "bottom_left": self.point_coords[2],
            "bottom_right": self.point_coords[3],
        }

    def on_load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return

        image = cv2.imread(
            file_path,
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            self._set_status(
                "Could not load image",
                CHIP_IDLE
            )
            return

        self.image = image

        self.image_loaded = True
        self.border_detected = False
        self.corners_selected = 0
        self.point_coords = []
        self.detection_result = None
        self.calibrated = False
        self.pixels_per_mm = None

        self._display_image(self.image)

        self._style_chip(
            self.image_chip,
            CHIP_SUCCESS
        )

        self.image_chip.setText(
            "  Image: Loaded  "
        )

        self._style_chip(
            self.border_chip,
            CHIP_IDLE
        )

        self.border_chip.setText(
            "  Border: Not Detected  "
        )

        self._style_chip(
            self.calibration_chip,
            CHIP_IDLE
        )

        self.calibration_chip.setText(
            "  Not Calibrated  "
        )

        self._style_chip(
            self.points_chip,
            CHIP_IDLE
        )

        self.points_chip.setText(
            "  Points: 0 / 4  "
        )

        self.points_list_label.setText(
            "No points selected yet"
        )

        self.result_label.setText(
            "-- mm"
        )

        self._set_status(
            "Image loaded",
            CHIP_SUCCESS
        )

        self._refresh_button_states()

    def on_clear(self):
        self.image = None
        self.image_loaded = False
        self.border_detected = False
        self.corners_selected = 0
        self.point_coords = []
        self.detection_result = None
        self.calibrated = False
        self.pixels_per_mm = None

        self.image_label.clear_image()

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
        if self.image is None:
            self._set_status(
                "No image loaded",
                CHIP_IDLE
            )
            return

        try:
            # Run actual detection
            result = detect_component(self.image)

            self.detection_result = result

            contour = result["contour"]
            info = result["info"]

            self.border_detected = True

            # Draw the border outline only — this is NOT the same as
            # manually selecting points, so we do not touch point_coords
            # with the detected corners here.
            output = draw_component(
                self.image,
                contour,
                info
            )

            # _display_image() reloads the canvas via set_image(), which
            # would normally also wipe out any manually selected points.
            # We explicitly restore whatever the user had already picked
            # (in original-image coordinates, so they're still valid on
            # the new border-drawn image) so Detect Border never silently
            # throws away manual selection. If the user hadn't finished
            # picking all 4 points yet, selection mode resumes too.
            self._display_image(output)
            if self.point_coords:
                still_selecting = self.corners_selected < 4
                self.image_label.set_points(self.point_coords, selecting=still_selecting)

            self.border_chip.setText(
                "  Border: Detected  "
            )
            self._style_chip(
                self.border_chip,
                CHIP_SUCCESS
            )

            if self.point_coords:
                self._set_status(
                    "Border detected \u2014 your selected points were kept",
                    CHIP_SUCCESS
                )
            else:
                self._set_status(
                    "Border detected \u2014 click \u201cSelect Points\u201d to mark corners manually",
                    CHIP_SUCCESS
                )

            self._refresh_button_states()

        except ValueError as error:

            self.border_detected = False
            self.detection_result = None

            self._style_chip(
                self.border_chip,
                CHIP_IDLE
            )

            self.border_chip.setText(
                "  Border: Not Detected  "
            )

            self._set_status(
                str(error),
                CHIP_IDLE
            )

            self._refresh_button_states()

    def on_select_points(self):
        if self.image is None:
            self._set_status("Load an image first", CHIP_IDLE)
            return

        # Starting (or restarting) selection always clears old points —
        # this doubles as the "reset/reselect" control.
        self.point_coords = []
        self.corners_selected = 0

        # A new set of points invalidates any existing calibration —
        # the old pixels-per-mm scale was computed from the previous
        # points and no longer applies.
        self.calibrated = False
        self.pixels_per_mm = None
        self.calibration_chip.setText("  Not Calibrated  ")
        self._style_chip(self.calibration_chip, CHIP_IDLE)
        self.result_label.setText("-- mm")

        self.image_label.start_point_selection()

        self.points_chip.setText("  Points: 0 / 4  ")
        self._style_chip(self.points_chip, CHIP_IDLE)
        self.points_list_label.setText(
            "Click 4 points on the image:\n"
            "Top-Left \u2192 Top-Right \u2192 Bottom-Left \u2192 Bottom-Right"
        )

        self._set_status("Click 4 points on the image (Top-Left first)", CHIP_IDLE)
        self._refresh_button_states()

    def _on_canvas_point_clicked(self, x, y):
        self.point_coords = list(self.image_label.points)
        self.corners_selected = len(self.point_coords)

        self.points_chip.setText(f"  Points: {self.corners_selected} / 4  ")
        self._style_chip(
            self.points_chip,
            CHIP_SUCCESS if self.corners_selected == 4 else CHIP_IDLE,
        )

        labels = ["Point 1 (Top-Left)", "Point 2 (Top-Right)",
                  "Point 3 (Bottom-Left)", "Point 4 (Bottom-Right)"]
        lines = [f"{labels[i]}: {pt}" for i, pt in enumerate(self.point_coords)]
        self.points_list_label.setText("\n".join(lines))

        if self.corners_selected < 4:
            remaining = 4 - self.corners_selected
            self._set_status(
                f"Point {self.corners_selected} selected \u2014 click {remaining} more",
                CHIP_IDLE,
            )
        else:
            self._set_status("All 4 points selected", CHIP_SUCCESS)

        self._refresh_button_states()

    def on_calibrate(self):
        # CHANGED: calibration now reads the four MANUALLY selected
        # points instead of self.detection_result["points"].
        points = self._get_selected_points_dict()

        if points is None:
            self._set_status(
                "Select all 4 points before calibrating",
                CHIP_IDLE
            )
            return

        try:
            known_length = float(
                self.reference_input.text()
            )

            if known_length <= 0:
                raise ValueError(
                    "Known length must be greater than zero."
                )

            # Use the distance between the manually selected
            # top-left and top-right points as the calibration
            # pixel distance.
            pixel_distance = calculate_width(points)

            if pixel_distance <= 0:
                raise ValueError(
                    "Top-Left and Top-Right points are the same spot — "
                    "re-select 4 distinct points before calibrating."
                )

            self.pixels_per_mm = calculate_scale(
                pixel_distance,
                known_length
            )

            self.calibrated = True

            self._style_chip(
                self.calibration_chip,
                CHIP_SUCCESS
            )

            self.calibration_chip.setText(
                f"  Calibrated: "
                f"{self.pixels_per_mm:.2f} px/mm  "
            )

            self._set_status(
                f"Calibration complete: "
                f"{self.pixels_per_mm:.2f} px/mm",
                CHIP_SUCCESS
            )

            self._refresh_button_states()

        except ValueError as error:

            self.calibrated = False

            self._style_chip(
                self.calibration_chip,
                CHIP_IDLE
            )

            self.calibration_chip.setText(
                "  Not Calibrated  "
            )

            self._set_status(
                str(error),
                CHIP_IDLE
            )

            self._refresh_button_states()

    def on_measure(self):

        if not self.calibrated:
            self._set_status(
                "Calibrate first",
                CHIP_IDLE
            )
            return

        # CHANGED: measurement now reads the four MANUALLY selected
        # points instead of self.detection_result["points"].
        points = self._get_selected_points_dict()

        if points is None:
            self._set_status(
                "Select all 4 points before measuring",
                CHIP_IDLE
            )
            return

        try:
            pixel_width = calculate_width(points)
            pixel_height = calculate_height(points)

            width_mm = pixels_to_real(
                pixel_width,
                self.pixels_per_mm
            )

            height_mm = pixels_to_real(
                pixel_height,
                self.pixels_per_mm
            )

            self.result_label.setText(
                f"{width_mm:.2f} \u00d7 {height_mm:.2f} mm"
            )

            self._set_status(
                f"Measurement complete: "
                f"{width_mm:.2f} \u00d7 {height_mm:.2f} mm",
                CHIP_SUCCESS
            )

        except ValueError as error:

            self._set_status(
                str(error),
                CHIP_IDLE
            )

    def _set_status(self, message, palette):
        self.statusBar().showMessage(message)
        self.status_chip.setText(f"  {message}  ")
        self._style_chip(self.status_chip, palette)

    def _refresh_button_states(self):
        """Enable/disable toolbar buttons so the workflow order is obvious."""
        self.detect_button.setEnabled(self.image_loaded)
        self.select_points_button.setEnabled(self.image_loaded)   # Detect Border is optional
        # CHANGED: Calibrate now only needs 4 manually selected points.
        # Calibration and measurement read from self.point_coords, not
        # self.detection_result, so completing Detect Border is no longer
        # a prerequisite for calibrating — matching the "Detect Border
        # (optional)" step in the intended workflow.
        self.calibrate_button.setEnabled(self.corners_selected == 4)
        self.measure_button.setEnabled(self.calibrated)


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()