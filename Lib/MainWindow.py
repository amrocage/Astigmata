from time import process_time

# GUI
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea,
                             QFileDialog, QSpacerItem)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from torch.distributed.tensor import empty

# INTERNAL LIBRARIES

from Lib.ImageDropZone import ImageDropZone
from Lib.LoadingAnimation import LoadingAnimation
from Lib.DataVisualizer import CornealAnalysisWidget

from HeatmapModel import CornealAnalyzer


class MainWindow(QMainWindow):

    current_astigmatism_results = []

    def __init__(self):
        super().__init__()
        self.Analyzer = CornealAnalyzer()
        # Initialize QSettings

        self.setWindowTitle("Team Astigmata")
        self.setMinimumSize(1100, 800)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #e0e0e0; font-family: Segoe UI, Arial, sans-serif; }
            QPushButton { background-color: #0078d4; color: white; border: none; border-radius: 4px; padding: 10px 18px; font-size: 13px; font-weight: bold; outline: none; }
            QPushButton:hover { background-color: #005a9e; } QPushButton:pressed { background-color: #004c87; }
            QScrollArea { border: 1px solid #333; border-radius: 5px; background-color: #252525; }
            QLabel { color: #e0e0e0; }
            QLabel#header_label { color: #ffffff; font-size: 20px; font-weight: bold; margin-bottom: 15px; padding-top: 5px; }
            QLabel#results_title_label { color: #ffffff; font-size: 18px; font-weight: bold; margin-bottom: 10px; }
            QLabel#notice_label { font-size: 11px; color: #b0b0b0; margin-top: 10px; }
            QFrame#separator_line { background-color: #444; min-width: 1px; max-width: 1px; }
            QWidget#title_bar { background-color: #2d2d2d; border-bottom: 1px solid #444; }
            QLabel#title_bar_label { color: #e0e0e0; font-weight: bold; padding-left: 5px; }
            QPushButton.title_bar_button { background-color: transparent; color: #e0e0e0; border: none; padding: 6px; margin: 0px; font-size: 14px; font-weight: bold; min-width: 35px; }
            QPushButton.title_bar_button:hover { background-color: #555; }
            QPushButton#close_button.title_bar_button:hover { background-color: #e81123; color: white; }
            QPushButton.title_bar_button:pressed { background-color: #666; }
            QWidget.measurement_result_card { border: 1px solid #4a4a4a; border-radius: 6px; background-color: #2c2c2c; margin-bottom: 15px; padding: 12px; }
            QLabel.patient_info_label { font-weight: bold; font-size: 15px; margin-bottom: 5px; }
            QLabel.measurement_detail_label { font-size: 13px; color: #c8c8c8; }
            QLabel.astigmatism_status_label_detected { font-weight: bold; color: #66BB6A; font-size: 14px; }
            QLabel.astigmatism_status_label_not_detected { color: #BDBDBD; font-size: 14px; }
            QLabel.map_title_label { font-size: 11px; color: #dadada; text-align: center; margin-bottom: 3px; font-weight:bold;}
            QLabel.heatmap_display_label { border: 1px solid #555; background-color: #383838; } 
            QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 1ex; font-weight: bold; color: #e0e0e0;}
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; background-color: #1e1e1e;}
            QLineEdit { background-color: #2c2c2c; border: 1px solid #555; border-radius: 3px; padding: 4px; color: #e0e0e0;}
            QRadioButton { color: #e0e0e0; }
            QTableWidget { background-color: #2c2c2c; border: 1px solid #555; gridline-color: #444; selection-background-color: #0078d4; selection-color: white;}
            QHeaderView::section { background-color: #3a3a3a; color: white; padding: 4px; border: 1px solid #555; font-weight: bold; }
        """)

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 10, 15, 15);
        main_layout.setSpacing(15)
        self.title_bar = QWidget();
        self.title_bar.setObjectName("title_bar")
        title_bar_layout = QHBoxLayout(self.title_bar);
        title_bar_layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel("Team Astigmata Analyzer");
        title_label.setObjectName("title_bar_label")
        min_button = QPushButton("–");
        min_button.setFixedSize(40, 30);
        min_button.clicked.connect(self.showMinimized);
        min_button.setProperty("class", "title_bar_button")
        self.max_button = QPushButton("□");
        self.max_button.setFixedSize(40, 30);
        self.max_button.clicked.connect(self.toggle_maximize);
        self.max_button.setProperty("class", "title_bar_button")
        close_button = QPushButton("✕");
        close_button.setFixedSize(40, 30);
        close_button.clicked.connect(self.close);
        close_button.setObjectName("close_button");
        close_button.setProperty("class", "title_bar_button")
        title_bar_layout.addWidget(title_label);
        title_bar_layout.addStretch();
        title_bar_layout.addWidget(min_button);
        title_bar_layout.addWidget(self.max_button);
        title_bar_layout.addWidget(close_button)
        main_layout.addWidget(self.title_bar)
        header_label = QLabel("Eye Condition & Astigmatism Analyzer");
        header_label.setAlignment(Qt.AlignCenter);
        header_label.setObjectName("header_label");
        main_layout.addWidget(header_label)
        content_layout = QHBoxLayout();
        content_layout.setSpacing(15)
        left_widget = QWidget();
        left_layout = QVBoxLayout(left_widget);
        left_layout.setContentsMargins(0, 0, 0, 0);
        left_layout.setSpacing(12)

        input_section_label = QLabel("Upload Eye Image or Load Data:")
        input_section_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        left_layout.addWidget(input_section_label)

        self.drop_zone = ImageDropZone();
        self.drop_zone.imageDropped.connect(self.process_image);
        left_layout.addWidget(self.drop_zone)


        file_actions_layout = QHBoxLayout()
        browse_image_button = QPushButton("Browse Image"); browse_image_button.clicked.connect(self.browse_image)
        file_actions_layout.addWidget(browse_image_button)
        left_layout.addLayout(file_actions_layout)

        self.notice_label = QLabel(
            "Note: Image analysis provides a general classification. Measurement data provides specific astigmatism values. Neither should substitute professional medical advice.");
        self.notice_label.setAlignment(Qt.AlignLeft);
        self.notice_label.setWordWrap(True);
        self.notice_label.setObjectName("notice_label");
        left_layout.addWidget(self.notice_label)

        left_layout.addStretch();
        content_layout.addWidget(left_widget)
        line = QFrame();
        line.setFrameShape(QFrame.VLine);
        line.setFrameShadow(QFrame.Sunken);
        line.setObjectName("separator_line");
        content_layout.addWidget(line)

        self.result_display = CornealAnalysisWidget()

        content_layout.addWidget(self.result_display)
        content_layout.setStretchFactor(left_widget, 3);
        content_layout.setStretchFactor(self.result_display, 7)
        main_layout.addLayout(content_layout);
        self.setCentralWidget(central_widget);
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self._drag_pos = None;

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.title_bar.underMouse():
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft();
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos);
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None; super().mouseReleaseEvent(event)

    def changeEvent(self, event):
        if event.type() == event.WindowStateChange: self.max_button.setText("❐" if self.isMaximized() else "□")
        super().changeEvent(event)

    def toggle_maximize(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if (file_path) :
            self.process_image(file_path)
            self.drop_zone.show_dropped_image(file_path)

    def process_image(self, image_path):

        self.result_display.display(self.Analyzer.analyze_eye_image(image_path))

        self.drop_zone.setAlignment(Qt.AlignCenter)


