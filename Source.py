import sys
import os
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea,
                           QFileDialog, QSizePolicy, QMessageBox, QSpacerItem, 
                           QGridLayout, QLineEdit, QRadioButton, QButtonGroup, QGroupBox,
                           QTableWidget, QTableWidgetItem, QDialog, QDialogButtonBox,
                           QHeaderView, QAbstractItemView) 
from PyQt5.QtCore import Qt, QMimeData, QTimer, QSize, pyqtSignal, QThread, QObject, QSettings # Added QSettings
from PyQt5.QtGui import QDrag, QPixmap, QPainter, QPen, QColor, QFont, QPalette, QMovie, QFontDatabase

import torch
from torchvision import transforms
from PIL import Image as PILImage 
import random
import sqlite3 
from datetime import datetime 
import math 

# --- For PDF Generation ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors as reportlab_colors
    REPORTLAB_AVAILABLE = True
    print("ReportLab imported successfully.")
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: ReportLab library not found. PDF export will be disabled. Install with: pip install reportlab")
# --- End PDF Generation Imports ---

# Import the professional heatmap generation function
try:
    from professional_astigmatism_heatmap import generate_corneal_map 
    print("Successfully imported generate_corneal_map from professional_astigmatism_heatmap.py.")
except ImportError:
    print("Error: Could not import generate_corneal_map from professional_astigmatism_heatmap.py. Heatmap visualization will be disabled.")
    def generate_corneal_map(*args, **kwargs):
        print("Dummy professional_astigmatism_heatmap.generate_corneal_map function called: import failed.")
        return None

try:
    from Main import CNN 
except ImportError:
    print("Warning: Could not import CNN from Main.py. Image processing will be disabled.")
    class CNN(torch.nn.Module): 
        def __init__(self, num_classes=4):
            super().__init__()
            self.dummy_layer = torch.nn.Linear(1, 1) 
            self.fc = torch.nn.Linear(1, num_classes) 
        def forward(self, x):
            print("Warning: Dummy CNN forward called - image model not loaded.")
            return torch.randn(x.size(0), self.fc.out_features if hasattr(self, 'fc') else 4)

# --- Worker Thread for Image Processing ---
class ImageProcessingWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result_ready = pyqtSignal(str, float)
    def __init__(self, image_path, model, transform, class_names, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.model = model
        self.transform = transform
        self.class_names = class_names
    def run_processing(self):
        if self.model is None:
            self.error.emit("Model not loaded.")
            self.finished.emit()
            return
        try:
            image = PILImage.open(self.image_path).convert("RGB") 
            image_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                device = next(self.model.parameters()).device
                image_tensor = image_tensor.to(device)
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class_index = torch.max(probabilities, 1)
            predicted_class_name = self.class_names[predicted_class_index.item()]
            confidence_score = confidence.item() * 100
            self.result_ready.emit(predicted_class_name, confidence_score)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

# --- ImageDropZone ---
class ImageDropZone(QLabel):
    imageDropped = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #555; border-radius: 10px; background-color: #2a2a2a;
                color: #ccc; padding: 20px; font-size: 14px;
            }
            QLabel:hover { border: 2px dashed #777; background-color: #3a3a3a; }
        """)
        self.initial_pixmap = QPixmap(200, 200)
        self.initial_pixmap.fill(Qt.transparent)
        painter = QPainter(self.initial_pixmap)
        painter.setPen(QPen(Qt.white, 2))
        font_families = QFontDatabase().families()
        font_name = "Arial" 
        if "Liberation Sans" in font_families: font_name = "Liberation Sans"
        elif "DejaVu Sans" in font_families: font_name = "DejaVu Sans"
        elif not "Arial" in font_families and font_families: font_name = font_families[0]
        painter.setFont(QFont(font_name, 12))
        painter.drawText(self.initial_pixmap.rect(), Qt.AlignCenter, "Drop Eye Image Here\n(.jpg, .jpeg, .png)")
        painter.end()
        self.setPixmap(self.initial_pixmap)
        self.setMinimumSize(200, 200)
        self.setMaximumHeight(300) 
        self.dropped_pixmap_path = None
    def resizeEvent(self, event):
        current_pixmap_is_initial = (self.pixmap() is self.initial_pixmap or self.dropped_pixmap_path is None or not os.path.exists(self.dropped_pixmap_path))
        if current_pixmap_is_initial and hasattr(self, 'initial_pixmap') and not self.initial_pixmap.isNull():
            self.setPixmap(self.initial_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif self.dropped_pixmap_path and os.path.exists(self.dropped_pixmap_path):
            loaded_pixmap = QPixmap(self.dropped_pixmap_path)
            if not loaded_pixmap.isNull(): self.setPixmap(loaded_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else: self.setPixmap(self.initial_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif hasattr(self, 'initial_pixmap') and not self.initial_pixmap.isNull():
             self.setPixmap(self.initial_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setAlignment(Qt.AlignCenter)
        super().resizeEvent(event)
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and event.mimeData().urls()[0].toLocalFile().lower().endswith(('.jpg', '.jpeg', '.png')):
            event.acceptProposedAction()
            self.setStyleSheet("QLabel { border: 2px dashed #aaa; border-radius: 10px; background-color: rgba(255, 255, 255, 0.1); color: #ccc; font-size: 14px; }")
        else: event.ignore()
    def dragLeaveEvent(self, event):
        self.setStyleSheet(""" QLabel { border: 2px dashed #555; border-radius: 10px; background-color: #2a2a2a; color: #ccc; padding: 20px; font-size: 14px; } QLabel:hover { border: 2px dashed #777; background-color: #3a3a3a; } """)
    def dropEvent(self, event):
        self.dropped_pixmap_path = None
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                dropped_pixmap = QPixmap(file_path)
                if not dropped_pixmap.isNull():
                    self.dropped_pixmap_path = file_path
                    self.setPixmap(dropped_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    self.imageDropped.emit(file_path)
                    event.acceptProposedAction()
                else:
                    self.setPixmap(self.initial_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)); event.ignore()
            else: self.setPixmap(self.initial_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)); event.ignore()
        else: self.setPixmap(self.initial_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)); event.ignore()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(""" QLabel { border: 2px dashed #555; border-radius: 10px; background-color: #2a2a2a; color: #ccc; padding: 20px; font-size: 14px; } QLabel:hover { border: 2px dashed #777; background-color: #3a3a3a; } """)

# --- LoadingAnimation ---
class LoadingAnimation(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 60); self.angle = 0
        self.timer = QTimer(self); self.timer.timeout.connect(self.rotate)
        self.destroyed.connect(self.timer.stop)
    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width()/2, self.height()/2); painter.rotate(self.angle)
        pen = QPen(Qt.white, 2); painter.setPen(pen)
        for i in range(8):
            line_length = 25 if i % 2 == (self.angle // 45) % 2 else 15
            painter.drawLine(0, 10, 0, line_length); painter.rotate(45)
    def rotate(self): self.angle = (self.angle + 10) % 360; self.update()
    def showEvent(self, event): self.timer.start(50); super().showEvent(event)
    def hideEvent(self, event): self.timer.stop(); super().hideEvent(event)

# --- Settings Dialog ---
class SettingsDialog(QDialog):
    settings_updated_signal = pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings # QSettings object
        self.setWindowTitle("Application Settings")
        self.setMinimumWidth(400)
        if parent: self.setStyleSheet(parent.styleSheet())

        layout = QVBoxLayout(self)
        form_layout = QGridLayout()

        # K Mean
        self.k_mean_edit = QLineEdit(str(self.settings.value("heatmap/k_mean", 44.5, type=float)))
        form_layout.addWidget(QLabel("Default K Mean (D):"), 0, 0)
        form_layout.addWidget(self.k_mean_edit, 0, 1)

        # BFS Front
        self.bfs_front_edit = QLineEdit(str(self.settings.value("heatmap/bfs_front", 7.15, type=float)))
        form_layout.addWidget(QLabel("Default BFS Front (mm):"), 1, 0)
        form_layout.addWidget(self.bfs_front_edit, 1, 1)
        
        # BFS Back
        self.bfs_back_edit = QLineEdit(str(self.settings.value("heatmap/bfs_back", 5.98, type=float)))
        form_layout.addWidget(QLabel("Default BFS Back (mm):"), 2, 0)
        form_layout.addWidget(self.bfs_back_edit, 2, 1)

        # Central Thickness
        self.cct_edit = QLineEdit(str(self.settings.value("heatmap/central_thickness_sim", 540, type=float)))
        form_layout.addWidget(QLabel("Default Central Thickness (µm):"), 3, 0)
        form_layout.addWidget(self.cct_edit, 3, 1)
        
        # Astigmatism Threshold
        self.astig_threshold_edit = QLineEdit(str(self.settings.value("general/astigmatism_threshold", 0.25, type=float)))
        form_layout.addWidget(QLabel("Astigmatism Threshold (D) for Text:"), 4, 0)
        form_layout.addWidget(self.astig_threshold_edit, 4, 1)

        # Output Directory
        self.output_dir_edit = QLineEdit(self.settings.value("paths/output_directory", "outputs", type=str))
        browse_output_dir_button = QPushButton("Browse...")
        browse_output_dir_button.clicked.connect(self.browse_output_directory)
        form_layout.addWidget(QLabel("Base Output Directory:"), 5, 0)
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(browse_output_dir_button)
        form_layout.addLayout(output_dir_layout, 5, 1)

        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept_settings)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Base Output Directory", self.output_dir_edit.text())
        if directory:
            self.output_dir_edit.setText(directory)

    def accept_settings(self):
        try:
            # Validate and save settings
            k_mean = float(self.k_mean_edit.text())
            bfs_front = float(self.bfs_front_edit.text())
            bfs_back = float(self.bfs_back_edit.text())
            cct = float(self.cct_edit.text())
            astig_threshold = float(self.astig_threshold_edit.text())
            output_dir = self.output_dir_edit.text().strip()

            if not output_dir:
                QMessageBox.warning(self, "Input Error", "Output directory cannot be empty.")
                return
            if astig_threshold < 0:
                QMessageBox.warning(self, "Input Error", "Astigmatism threshold must be non-negative.")
                return

            self.settings.setValue("heatmap/k_mean", k_mean)
            self.settings.setValue("heatmap/bfs_front", bfs_front)
            self.settings.setValue("heatmap/bfs_back", bfs_back)
            self.settings.setValue("heatmap/central_thickness_sim", cct)
            self.settings.setValue("general/astigmatism_threshold", astig_threshold)
            self.settings.setValue("paths/output_directory", output_dir)
            
            self.settings_updated_signal.emit() # Emit signal that settings were updated
            self.accept()

        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numerical values for heatmap parameters and threshold.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save settings: {e}")


# --- View Records Dialog ---
class ViewRecordsDialog(QDialog):
    record_selected_signal = pyqtSignal(int) 

    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.setWindowTitle("Patient Records")
        self.setMinimumSize(800, 500)
        if parent: 
            self.setStyleSheet(parent.styleSheet()) 

        layout = QVBoxLayout(self)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(9) 
        self.table_widget.setHorizontalHeaderLabels(["Rec ID", "Patient ID", "Eye", "Date", "Sphere", "Cylinder", "Axis", "J0", "J45"])
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers) 
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) 
        self.table_widget.itemDoubleClicked.connect(self.handle_load_selected) 

        layout.addWidget(self.table_widget)

        button_layout = QHBoxLayout()
        refresh_button = QPushButton("Refresh List")
        refresh_button.clicked.connect(self.load_records)
        button_layout.addWidget(refresh_button)

        load_button = QPushButton("Load Selected Record")
        load_button.clicked.connect(self.handle_load_selected)
        button_layout.addWidget(load_button)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)
        
        layout.addLayout(button_layout)
        self.load_records()

    def load_records(self):
        self.table_widget.setRowCount(0) 
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT record_id, patient_id, eye, measurement_date, sphere, cylinder, axis, j0, j45 FROM patient_records ORDER BY measurement_date DESC")
            records = cursor.fetchall()

            for row_idx, record in enumerate(records):
                self.table_widget.insertRow(row_idx)
                for col_idx, data in enumerate(record):
                    if col_idx in [7, 8] and isinstance(data, (int, float)): 
                        item = QTableWidgetItem(f"{data:.2f}")
                    else:
                        item = QTableWidgetItem(str(data))
                    if col_idx == 0: 
                         item.setData(Qt.UserRole, record[0]) 
                    self.table_widget.setItem(row_idx, col_idx, item)
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Database Error", f"Could not load records: {e}")
        finally:
            if conn: conn.close()

    def handle_load_selected(self):
        selected_rows = self.table_widget.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select a record to load.")
            return
        selected_row_index = selected_rows[0].row() 
        record_id_item = self.table_widget.item(selected_row_index, 0) 
        if record_id_item:
            record_id = int(record_id_item.text()) 
            self.record_selected_signal.emit(record_id)
            self.accept() 

class MainWindow(QMainWindow):
    DB_PATH = "astigmata_analyzer.db" 
    current_astigmatism_results = [] 

    def __init__(self):
        super().__init__()
        
        # Initialize QSettings
        self.settings = QSettings("MyCompany", "AstigmataAnalyzer") # Org name, App name
        self.load_app_settings() # Load settings or set defaults

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
        main_layout.setContentsMargins(15, 10, 15, 15); main_layout.setSpacing(15)
        self.title_bar = QWidget(); self.title_bar.setObjectName("title_bar")
        title_bar_layout = QHBoxLayout(self.title_bar); title_bar_layout.setContentsMargins(0,0,0,0)
        title_label = QLabel("Team Astigmata Analyzer"); title_label.setObjectName("title_bar_label")
        min_button = QPushButton("–"); min_button.setFixedSize(40,30); min_button.clicked.connect(self.showMinimized); min_button.setProperty("class", "title_bar_button")
        self.max_button = QPushButton("□"); self.max_button.setFixedSize(40,30); self.max_button.clicked.connect(self.toggle_maximize); self.max_button.setProperty("class", "title_bar_button")
        close_button = QPushButton("✕"); close_button.setFixedSize(40,30); close_button.clicked.connect(self.close); close_button.setObjectName("close_button"); close_button.setProperty("class", "title_bar_button")
        title_bar_layout.addWidget(title_label); title_bar_layout.addStretch(); title_bar_layout.addWidget(min_button); title_bar_layout.addWidget(self.max_button); title_bar_layout.addWidget(close_button)
        main_layout.addWidget(self.title_bar)
        header_label = QLabel("Eye Condition & Astigmatism Analyzer"); header_label.setAlignment(Qt.AlignCenter); header_label.setObjectName("header_label"); main_layout.addWidget(header_label)
        content_layout = QHBoxLayout(); content_layout.setSpacing(15)
        left_widget = QWidget(); left_layout = QVBoxLayout(left_widget); left_layout.setContentsMargins(0,0,0,0); left_layout.setSpacing(12)
        
        input_section_label = QLabel("Upload Eye Image or Load Data:") 
        input_section_label.setFont(QFont("Segoe UI", 12, QFont.Bold)) 
        left_layout.addWidget(input_section_label)

        self.drop_zone = ImageDropZone(); self.drop_zone.imageDropped.connect(self.start_image_processing_thread); left_layout.addWidget(self.drop_zone)
        
        manual_entry_groupbox = QGroupBox("Manual Data Entry")
        manual_entry_layout = QGridLayout(manual_entry_groupbox) 
        manual_entry_layout.setSpacing(8)
        self.patient_id_edit = QLineEdit(); self.patient_id_edit.setPlaceholderText("e.g., P001")
        manual_entry_layout.addWidget(QLabel("Patient ID:"), 0, 0); manual_entry_layout.addWidget(self.patient_id_edit, 0, 1)
        self.eye_os_radio = QRadioButton("OS (Left Eye)"); self.eye_os_radio.setChecked(True)
        self.eye_od_radio = QRadioButton("OD (Right Eye)")
        self.eye_button_group = QButtonGroup(self) 
        self.eye_button_group.addButton(self.eye_os_radio); self.eye_button_group.addButton(self.eye_od_radio)
        eye_radio_layout = QHBoxLayout(); eye_radio_layout.addWidget(self.eye_os_radio); eye_radio_layout.addWidget(self.eye_od_radio); eye_radio_layout.addStretch()
        manual_entry_layout.addWidget(QLabel("Eye:"), 1, 0); manual_entry_layout.addLayout(eye_radio_layout, 1, 1)
        self.sphere_edit = QLineEdit(); self.sphere_edit.setPlaceholderText("e.g., -1.75")
        manual_entry_layout.addWidget(QLabel("Sphere (D):"), 2, 0); manual_entry_layout.addWidget(self.sphere_edit, 2, 1)
        self.cylinder_edit = QLineEdit(); self.cylinder_edit.setPlaceholderText("e.g., -0.50")
        manual_entry_layout.addWidget(QLabel("Cylinder (D):"), 3, 0); manual_entry_layout.addWidget(self.cylinder_edit, 3, 1)
        self.axis_edit = QLineEdit(); self.axis_edit.setPlaceholderText("e.g., 90 (0-180)")
        manual_entry_layout.addWidget(QLabel("Axis (°):"), 4, 0); manual_entry_layout.addWidget(self.axis_edit, 4, 1)
        self.process_manual_button = QPushButton("Process Manual Data"); self.process_manual_button.clicked.connect(self.process_manual_data_entry)
        manual_entry_layout.addWidget(self.process_manual_button, 5, 0, 1, 2) 
        left_layout.addWidget(manual_entry_groupbox)

        file_actions_layout = QHBoxLayout()
        browse_image_button = QPushButton("Browse Image"); browse_image_button.clicked.connect(self.browse_image)
        load_data_button = QPushButton("Load Measurements File"); load_data_button.clicked.connect(self.load_measurement_data)
        file_actions_layout.addWidget(browse_image_button)
        file_actions_layout.addWidget(load_data_button)
        left_layout.addLayout(file_actions_layout)

        db_export_actions_layout = QVBoxLayout() 
        db_export_actions_layout.setSpacing(8) 
        self.view_records_button = QPushButton("View Patient Records"); self.view_records_button.clicked.connect(self.show_view_records_dialog)
        self.export_pdf_button = QPushButton("Export Current to PDF"); self.export_pdf_button.clicked.connect(self.generate_pdf_report)
        self.export_csv_button = QPushButton("Export All Results to CSV"); self.export_csv_button.clicked.connect(self.export_results_to_csv)
        self.settings_button = QPushButton("Settings"); self.settings_button.clicked.connect(self.show_settings_dialog) # New Settings button
        
        db_export_actions_layout.addWidget(self.view_records_button) 
        db_export_actions_layout.addWidget(self.export_pdf_button)
        db_export_actions_layout.addWidget(self.export_csv_button)
        db_export_actions_layout.addWidget(self.settings_button) # Add settings button to layout
        left_layout.addLayout(db_export_actions_layout) 
        
        self.notice_label = QLabel("Note: Image analysis provides a general classification. Measurement data provides specific astigmatism values. Neither should substitute professional medical advice."); self.notice_label.setAlignment(Qt.AlignLeft); self.notice_label.setWordWrap(True); self.notice_label.setObjectName("notice_label"); left_layout.addWidget(self.notice_label)
        
        left_layout.addStretch(); content_layout.addWidget(left_widget)
        line = QFrame(); line.setFrameShape(QFrame.VLine); line.setFrameShadow(QFrame.Sunken); line.setObjectName("separator_line"); content_layout.addWidget(line)
        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget); right_layout.setContentsMargins(0,0,0,0); right_layout.setSpacing(10)
        self.results_scroll = QScrollArea(); self.results_scroll.setWidgetResizable(True); self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.results_content = QWidget(); self.results_layout = QVBoxLayout(self.results_content); self.results_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft); self.results_layout.setContentsMargins(15,15,15,15); self.results_layout.setSpacing(10)
        self.results_scroll.setWidget(self.results_content)
        self.loading_container = QWidget(); loading_layout = QVBoxLayout(self.loading_container); loading_layout.setContentsMargins(0,0,0,0); self.loading_animation = LoadingAnimation(); loading_layout.addWidget(self.loading_animation, 0, Qt.AlignCenter); self.loading_container.setVisible(False); self.results_layout.addWidget(self.loading_container, 0, Qt.AlignCenter)
        self.initial_results_label = QLabel("Load an image or measurement data to see results."); self.initial_results_label.setAlignment(Qt.AlignCenter); self.initial_results_label.setWordWrap(True); self.results_layout.addWidget(self.initial_results_label, 0, Qt.AlignCenter | Qt.AlignTop)
        self.image_results_container = QWidget(); self.image_results_layout = QVBoxLayout(self.image_results_container); self.image_results_layout.setContentsMargins(0,0,0,0); self.image_results_container.setVisible(False)
        self.detection_label = QLabel(); self.confidence_label = QLabel(); self.confidence_label.setStyleSheet("color: #66BB6A;"); self.confidence_label.setAlignment(Qt.AlignRight)
        detection_layout = QHBoxLayout(); detection_layout.addWidget(self.detection_label); detection_layout.addWidget(self.confidence_label); self.image_results_layout.addLayout(detection_layout)
        self.symptoms_label = QLabel("Possible Symptoms:"); self.image_results_layout.addWidget(self.symptoms_label)
        self.symptoms_content = QWidget(); self.symptoms_layout = QVBoxLayout(self.symptoms_content); self.symptoms_layout.setContentsMargins(10,5,10,10)
        self.symptoms_content.setStyleSheet("QWidget { background-color: #2c2c2c; border-radius: 5px; border: 1px solid #444; } QLabel { color: #d0d0d0; font-size: 12px; padding-bottom: 3px; }")
        self.image_results_layout.addWidget(self.symptoms_content); self.results_layout.addWidget(self.image_results_container)
        self.numerical_results_container = QWidget(); self.numerical_results_layout = QVBoxLayout(self.numerical_results_container); self.numerical_results_layout.setContentsMargins(0,0,0,0); self.numerical_results_layout.setSpacing(0); self.numerical_results_container.setVisible(False)
        self.numerical_results_title_label = QLabel("Astigmatism Measurement Results:"); self.numerical_results_title_label.setObjectName("results_title_label")
        self.results_layout.addWidget(self.numerical_results_container); self.results_layout.addStretch(); right_layout.addWidget(self.results_scroll); content_layout.addWidget(right_widget)
        content_layout.setStretchFactor(left_widget, 3); content_layout.setStretchFactor(right_widget, 7) 
        main_layout.addLayout(content_layout); self.setCentralWidget(central_widget); self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self._drag_pos = None; self.thread = None; self.worker = None
        self.image_model = None; model_path = "cnn_model.pth"
        if os.path.exists(model_path):
            try:
                self.image_model = CNN(num_classes=4); self.image_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))); self.image_model.eval(); self.image_model.to(torch.device('cpu'))
                print(f"Image classification model loaded: {model_path}")
            except Exception as e: print(f"Error loading model: {e}"); self.image_model = None
        else: print(f"Warning: Model not found: {model_path}"); self.image_model = None
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.image_class_names = ['Astigmatism', 'Normal', 'Cataract', 'Diabetic Retinopathy']
        # self.heatmap_output_dir is now set in load_app_settings
        self.init_database() 
        self.current_astigmatism_results = [] 

    def load_app_settings(self):
        """Loads application settings or sets defaults."""
        self.settings_k_mean = self.settings.value("heatmap/k_mean", 44.5, type=float)
        self.settings_bfs_front = self.settings.value("heatmap/bfs_front", 7.15, type=float)
        self.settings_bfs_back = self.settings.value("heatmap/bfs_back", 5.98, type=float)
        self.settings_central_thickness_sim = self.settings.value("heatmap/central_thickness_sim", 540, type=float)
        self.settings_astigmatism_threshold = self.settings.value("general/astigmatism_threshold", 0.25, type=float)
        
        default_output_dir = os.path.join(os.getcwd(), "outputs") # Default to 'outputs' in current dir
        self.settings_base_output_dir = self.settings.value("paths/output_directory", default_output_dir, type=str)
        
        self.heatmap_output_dir = os.path.join(self.settings_base_output_dir, "corneal_maps")
        os.makedirs(self.heatmap_output_dir, exist_ok=True)
        print(f"Settings loaded. Heatmap output directory: {self.heatmap_output_dir}")


    def show_settings_dialog(self):
        dialog = SettingsDialog(self.settings, self)
        dialog.settings_updated_signal.connect(self.load_app_settings) # Reload settings if changed
        dialog.exec_()


    def init_database(self):
        try:
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_records (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    eye TEXT CHECK(eye IN ('OS', 'OD')) NOT NULL,
                    measurement_date TEXT NOT NULL,
                    sphere REAL,
                    cylinder REAL,
                    axis INTEGER,
                    j0 REAL, 
                    j45 REAL,
                    image_class TEXT,
                    image_confidence REAL,
                    map_axial_path TEXT,
                    map_elevation_front_path TEXT,
                    map_thickness_path TEXT,
                    map_elevation_back_path TEXT,
                    notes TEXT,
                    UNIQUE(patient_id, eye, measurement_date) 
                )
            """) 
            conn.commit()
            print(f"Database initialized successfully at {self.DB_PATH}")
        except sqlite3.Error as e:
            print(f"Error initializing database: {e}")
            QMessageBox.critical(self, "Database Error", f"Could not initialize database: {e}")
        finally:
            if conn: conn.close()

    def save_record_to_db(self, record_data):
        print(f"Attempting to save record to DB for Patient: {record_data.get('PatientID')}, Eye: {record_data.get('Eye')}")
        try:
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            measurement_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            map_paths = record_data.get('map_paths', {})
            cursor.execute("""
                INSERT INTO patient_records (
                    patient_id, eye, measurement_date, sphere, cylinder, axis, j0, j45,
                    image_class, image_confidence,
                    map_axial_path, map_elevation_front_path, 
                    map_thickness_path, map_elevation_back_path, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_data.get('PatientID', 'N/A'), record_data.get('Eye', 'N/A'),
                measurement_date, record_data.get('Sphere (Diopter)'), 
                record_data.get('Raw Cylinder'), record_data.get('Axis (Degrees)'), 
                record_data.get('J0'), record_data.get('J45'), 
                record_data.get('image_classification_result'), 
                record_data.get('image_classification_confidence'), 
                map_paths.get('axial_curvature'), map_paths.get('elevation_front'),
                map_paths.get('corneal_thickness'), map_paths.get('elevation_back'),
                None 
            ))
            conn.commit()
            print(f"Record saved successfully for {record_data.get('PatientID')} {record_data.get('Eye')}")
            return True
        except sqlite3.Error as e:
            print(f"Error saving record to database: {e}")
            return False 
        finally:
            if conn: conn.close()

    def show_view_records_dialog(self):
        dialog = ViewRecordsDialog(self.DB_PATH, self)
        dialog.record_selected_signal.connect(self.load_record_from_db_and_display)
        dialog.exec_()

    def load_record_from_db_and_display(self, record_id):
        print(f"Attempting to load record ID: {record_id} from database.")
        try:
            conn = sqlite3.connect(self.DB_PATH)
            conn.row_factory = sqlite3.Row 
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM patient_records WHERE record_id = ? 
            """, (record_id,)) 
            record = cursor.fetchone()

            if record:
                self.clear_results() 
                self.loading_container.setVisible(True) 
                
                cylinder = record['cylinder'] if record['cylinder'] is not None else 0.0
                axis_degrees = record['axis'] if record['axis'] is not None else 0
                
                j0 = record['j0'] if record['j0'] is not None else 0.0
                j45 = record['j45'] if record['j45'] is not None else 0.0

                result_entry = {
                    'PatientID': record['patient_id'], 
                    'Eye': record['eye'],
                    'Sphere (Diopter)': record['sphere'], 
                    'Raw Cylinder': cylinder,
                    'Degree (Diopter)': abs(cylinder), 
                    'Axis (Degrees)': axis_degrees,
                    'Astigmatism Detected': abs(cylinder) >= self.settings_astigmatism_threshold, # Use loaded setting
                    'J0': j0,
                    'J45': j45,
                    'map_paths': {
                        'axial_curvature': record['map_axial_path'],
                        'elevation_front': record['map_elevation_front_path'],
                        'corneal_thickness': record['map_thickness_path'],
                        'elevation_back': record['map_elevation_back_path']
                    }
                }
                self.current_astigmatism_results = [result_entry] 
                self.display_measurement_results(self.current_astigmatism_results) 
                self.loading_container.setVisible(False)
                print(f"Successfully loaded and displayed record ID: {record_id}")

            else:
                QMessageBox.warning(self, "Load Error", f"Could not find record with ID: {record_id}")
        except sqlite3.Error as e:
            print(f"Error loading record from database: {e}")
            QMessageBox.critical(self, "Database Error", f"Could not load record: {e}")
        finally:
            if conn: conn.close()


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.title_bar.underMouse(): 
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft(); event.accept()
        else: super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos); event.accept()
        else: super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event): self._drag_pos = None; super().mouseReleaseEvent(event)
    def changeEvent(self, event):
        if event.type() == event.WindowStateChange: self.max_button.setText("❐" if self.isMaximized() else "□")
        super().changeEvent(event)
    def toggle_maximize(self): self.showNormal() if self.isMaximized() else self.showMaximized()
    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path: self.clear_results(); self.start_image_processing_thread(file_path)
    def start_image_processing_thread(self, image_path):
        if self.image_model is None:
            self.clear_results_content(); error_label = QLabel("Error: Image model not available."); error_label.setStyleSheet("color: #FF5252;"); self.results_layout.addWidget(error_label)
            self.loading_container.setVisible(False); self.initial_results_label.setVisible(False); self.numerical_results_container.setVisible(False); self.image_results_container.setVisible(False)
            return
        self.clear_results_content(); self.loading_container.setVisible(True); self.initial_results_label.setVisible(False); self.numerical_results_container.setVisible(False); self.image_results_container.setVisible(False)
        self.drop_zone.dropped_pixmap_path = image_path; loaded_pixmap = QPixmap(image_path)
        if not loaded_pixmap.isNull(): self.drop_zone.setPixmap(loaded_pixmap.scaled(self.drop_zone.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else: self.drop_zone.setText("Failed to load preview")
        self.drop_zone.setAlignment(Qt.AlignCenter)
        self.thread = QThread(); self.worker = ImageProcessingWorker(image_path, self.image_model, self.transform, self.image_class_names)
        self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run_processing); self.worker.result_ready.connect(self.display_image_results, Qt.QueuedConnection)
        self.worker.error.connect(self.handle_image_processing_error, Qt.QueuedConnection); self.worker.finished.connect(self.thread.quit); self.thread.finished.connect(self.cleanup_image_thread)
        self.thread.start()
    def cleanup_image_thread(self):
        if self.worker: self.worker.deleteLater(); self.worker = None
        if self.thread: self.thread.deleteLater(); self.thread = None
    def display_image_results(self, predicted_class_name, confidence_score):
        self.loading_container.setVisible(False); self.numerical_results_container.setVisible(False); self.image_results_container.setVisible(True)
        self.detection_label.setText(f"Detected Condition: <b>{predicted_class_name}</b>"); self.confidence_label.setText(f"Confidence: <b>{confidence_score:.2f}%</b>")
        symptoms_map = {"Astigmatism": ["Blurred vision.", "Eyestrain.", "Headaches."], "Cataract": ["Clouded vision.", "Night difficulty.", "Light sensitivity."], "Diabetic Retinopathy": ["Floaters.", "Blurred vision.", "Vision loss."], "Normal": ["No specific symptoms."]}
        current_symptoms = symptoms_map.get(predicted_class_name, ["No info."])
        self.clear_layout(self.symptoms_layout)
        if current_symptoms:
            self.symptoms_label.setVisible(True); self.symptoms_content.setVisible(True)
            for symptom in current_symptoms: lbl = QLabel(f"• {symptom}"); lbl.setWordWrap(True); self.symptoms_layout.addWidget(lbl)
        else: self.symptoms_label.setVisible(False); self.symptoms_content.setVisible(False)
    def handle_image_processing_error(self, error_message):
        self.loading_container.setVisible(False); self.numerical_results_container.setVisible(False); self.image_results_container.setVisible(False); self.initial_results_label.setVisible(False)
        self.clear_results_content(keep_persistent=True)
        lbl = QLabel(f"<b>Error:</b><br>{error_message}"); lbl.setStyleSheet("color: #FF5252;"); lbl.setWordWrap(True); lbl.setAlignment(Qt.AlignCenter); self.results_layout.insertWidget(1, lbl)
    
    def process_manual_data_entry(self):
        print("--- Processing Manual Data Entry ---")
        patient_id = self.patient_id_edit.text().strip()
        sphere_str = self.sphere_edit.text().strip()
        cylinder_str = self.cylinder_edit.text().strip()
        axis_str = self.axis_edit.text().strip()
        eye = "OS" if self.eye_os_radio.isChecked() else "OD"

        if not patient_id:
            QMessageBox.warning(self, "Input Error", "Patient ID cannot be empty.")
            return
        
        try:
            sphere = float(sphere_str) if sphere_str else 0.0
            cylinder = float(cylinder_str) if cylinder_str else 0.0
            axis_degrees = int(axis_str) if axis_str else 0
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Sphere, Cylinder, and Axis must be valid numbers if provided, or left empty for 0.")
            return

        if not (0 <= axis_degrees <= 180):
            QMessageBox.warning(self, "Input Error", "Axis must be between 0 and 180 degrees.")
            return
        
        self.clear_results()
        self.loading_container.setVisible(True)
        self.initial_results_label.setVisible(False)
        self.image_results_container.setVisible(False)
        self.numerical_results_container.setVisible(False) 

        map_types_to_generate = ["axial_curvature", "elevation_front", "corneal_thickness", "elevation_back"]
        
        axis_radians = math.radians(axis_degrees)
        j0 = (-cylinder / 2) * math.cos(2 * axis_radians)
        j45 = (-cylinder / 2) * math.sin(2 * axis_radians)

        result_entry = {
            'PatientID': patient_id, 'Eye': eye, 
            'Sphere (Diopter)': sphere, 'Raw Cylinder': cylinder, 
            'Degree (Diopter)': abs(cylinder), 'Axis (Degrees)': axis_degrees, 
            'Astigmatism Detected': abs(cylinder) >= self.settings_astigmatism_threshold, # Use loaded setting
            'J0': j0, 'J45': j45,
            'map_paths': {} 
        }
        
        print(f"Generating maps for manual entry: {patient_id} {eye}...")
        for map_type in map_types_to_generate:
            safe_pid = "".join(c if c.isalnum() else "_" for c in patient_id)
            heatmap_filename = f"{safe_pid}_{eye}_{map_type}_cyl{cylinder:.2f}_axis{axis_degrees}.png"
            output_heatmap_file = os.path.join(self.heatmap_output_dir, heatmap_filename) # Use configured output dir
            
            map_path = generate_corneal_map( 
                cylinder_power=cylinder, axis_degrees=axis_degrees, map_type=map_type, 
                output_filename=output_heatmap_file, patient_id=patient_id, eye_type=eye,
                k_mean=self.settings_k_mean, bfs_front=self.settings_bfs_front,
                bfs_back=self.settings_bfs_back, central_thickness_sim=self.settings_central_thickness_sim
            )
            result_entry['map_paths'][map_type] = map_path 
        
        if self.save_record_to_db(result_entry):
             QMessageBox.information(self, "Data Saved", f"Record for {patient_id} ({eye}) saved to database.")
        else:
             QMessageBox.warning(self, "Save Failed", f"Could not save record for {patient_id} ({eye}) to database.")

        self.current_astigmatism_results = [result_entry] 
        self.display_measurement_results(self.current_astigmatism_results) 
        self.loading_container.setVisible(False)
        print("--- Manual Data Entry Processing Complete ---")

    def load_measurement_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Measurement Data", "", "CSV (*.csv);;Text (*.txt)")
        if file_path: 
            self.clear_results()
            self.loading_container.setVisible(True)
            self.initial_results_label.setVisible(False)
            self.numerical_results_container.setVisible(False)
            self.image_results_container.setVisible(False)
            self.process_measurement_data(file_path)
    
    def process_measurement_data(self, file_path):
        self.current_astigmatism_results = [] 
        map_types_to_generate = ["axial_curvature", "elevation_front", "corneal_thickness", "elevation_back"]
        processed_count = 0
        saved_count = 0

        try:
            with open(file_path, mode='r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                required_cols = ['PatientID', 'Eye', 'Sphere', 'Cylinder', 'Axis']
                if not reader.fieldnames or not all(col in reader.fieldnames for col in required_cols):
                    self.handle_measurement_data_error(f"Incorrect CSV header. Expected: {', '.join(required_cols)}"); return
                
                for row_num, row in enumerate(reader, 1):
                    try:
                        processed_count +=1
                        patient_id = row.get('PatientID', f'P{row_num}').strip()
                        eye = row.get('Eye', 'N/A').strip().upper()
                        cylinder = float(row.get('Cylinder', '0.0').strip())
                        axis_degrees = int(float(row.get('Axis', '0').strip()))
                        sphere = float(row.get('Sphere', '0.0').strip())
                        
                        if not (0 <= axis_degrees <= 180): 
                            print(f"Warning: Invalid Axis {axis_degrees} for {patient_id} {eye}. Defaulting to 0.")
                            axis_degrees = 0 

                        axis_radians = math.radians(axis_degrees)
                        j0 = (-cylinder / 2) * math.cos(2 * axis_radians)
                        j45 = (-cylinder / 2) * math.sin(2 * axis_radians)

                        result_entry = {
                            'PatientID': patient_id, 'Eye': eye, 
                            'Sphere (Diopter)': sphere, 'Raw Cylinder': cylinder, 
                            'Degree (Diopter)': abs(cylinder), 'Axis (Degrees)': axis_degrees, 
                            'Astigmatism Detected': abs(cylinder) >= self.settings_astigmatism_threshold, # Use loaded setting
                            'J0': j0, 'J45': j45,
                            'map_paths': {} 
                        }
                        
                        for map_type in map_types_to_generate:
                            safe_pid = "".join(c if c.isalnum() else "_" for c in patient_id)
                            heatmap_filename = f"{safe_pid}_{eye}_{map_type}_cyl{cylinder:.2f}_axis{axis_degrees}.png"
                            output_heatmap_file = os.path.join(self.heatmap_output_dir, heatmap_filename) # Use configured output dir
                            
                            print(f"Generating {map_type} map for {patient_id} {eye}...")
                            map_path = generate_corneal_map( 
                                cylinder_power=cylinder, axis_degrees=axis_degrees, map_type=map_type, 
                                output_filename=output_heatmap_file, patient_id=patient_id, eye_type=eye,
                                k_mean=self.settings_k_mean, bfs_front=self.settings_bfs_front,
                                bfs_back=self.settings_bfs_back, central_thickness_sim=self.settings_central_thickness_sim
                            )
                            result_entry['map_paths'][map_type] = map_path 
                        
                        if self.save_record_to_db(result_entry): 
                            saved_count +=1
                        self.current_astigmatism_results.append(result_entry)

                    except ValueError as ve: 
                        print(f"Skipping row {row_num} due to data conversion error: {ve}. Row: {row}")
                    except Exception as e_row: 
                        print(f"An unexpected error occurred processing row {row_num}: {e_row}. Row: {row}")
            
            if processed_count > 0:
                QMessageBox.information(self, "Processing Complete", f"{processed_count} rows processed. {saved_count} records saved to database.")

            self.display_measurement_results(self.current_astigmatism_results)

        except FileNotFoundError: 
            self.handle_measurement_data_error(f"File not found: {file_path}")
        except Exception as e: 
            self.handle_measurement_data_error(f"An error occurred while processing the file: {e}")
        finally: 
            self.loading_container.setVisible(False)

    def handle_measurement_data_error(self, error_message):
        self.loading_container.setVisible(False); self.initial_results_label.setVisible(False); self.numerical_results_container.setVisible(False); self.image_results_container.setVisible(False)
        self.clear_results_content(keep_persistent=True)
        lbl = QLabel(f"<b>Data Error:</b><br>{error_message}"); lbl.setStyleSheet("color: #FF5252;"); lbl.setWordWrap(True); lbl.setAlignment(Qt.AlignCenter); self.results_layout.insertWidget(1, lbl)

    def display_measurement_results(self, results):
        self.clear_layout(self.numerical_results_layout) 
        if not results: 
            self.initial_results_label.setVisible(True)
            self.numerical_results_container.setVisible(False)
            print("No measurement results to display.")
            return
        
        self.initial_results_label.setVisible(False)
        self.numerical_results_container.setVisible(True)
        self.image_results_container.setVisible(False) 
        
        self.numerical_results_layout.addWidget(self.numerical_results_title_label)
        self.numerical_results_layout.addSpacing(10)

        map_display_size = QSize(180, 180) 
        map_titles = {
            "axial_curvature": "Axial Curvature (Front)", 
            "elevation_front": "Elevation (Front)",
            "corneal_thickness": "Corneal Thickness", 
            "elevation_back": "Elevation (Back)"
        }
        map_order = ["axial_curvature", "elevation_front", "corneal_thickness", "elevation_back"]

        for res in results:
            result_card = QWidget()
            result_card.setProperty("class", "measurement_result_card")
            card_main_layout = QVBoxLayout(result_card) 
            card_main_layout.setContentsMargins(12, 12, 12, 12)
            card_main_layout.setSpacing(12) 
            
            text_details_widget = QWidget()
            text_details_layout = QVBoxLayout(text_details_widget)
            text_details_layout.setContentsMargins(0,0,0,0)
            text_details_layout.setSpacing(5) 
            
            patient_lbl = QLabel(f"Patient {res.get('PatientID','N/A')}, Eye: {res.get('Eye','N/A')}")
            patient_lbl.setProperty("class","patient_info_label")
            text_details_layout.addWidget(patient_lbl)

            status_text = "ASTIGMATISM DETECTED" if res.get('Astigmatism Detected') else f"No significant astigmatism detected (Cylinder < {self.settings_astigmatism_threshold}D)."
            status_lbl_class = "astigmatism_status_label_detected" if res.get('Astigmatism Detected') else "astigmatism_status_label_not_detected"
            status_lbl = QLabel(status_text)
            status_lbl.setProperty("class", status_lbl_class)
            text_details_layout.addWidget(status_lbl)
            
            degree_label = QLabel(f"Degree (Cylinder): {res.get('Degree (Diopter)',0.0):.2f} D")
            degree_label.setProperty("class", "measurement_detail_label")
            text_details_layout.addWidget(degree_label)

            axis_label = QLabel(f"Axis: {res.get('Axis (Degrees)','N/A')}°")
            axis_label.setProperty("class", "measurement_detail_label")
            text_details_layout.addWidget(axis_label)
            
            if abs(res.get('Sphere (Diopter)',0.0)) > 0.01 or res.get('Astigmatism Detected'):
                sphere_label = QLabel(f"Sphere: {res.get('Sphere (Diopter)',0.0):.2f} D")
                sphere_label.setProperty("class", "measurement_detail_label")
                text_details_layout.addWidget(sphere_label)

            j0_val = res.get('J0', 0.0)
            j45_val = res.get('J45', 0.0)
            j0_label = QLabel(f"J0: {j0_val:.2f} D")
            j0_label.setProperty("class", "measurement_detail_label")
            text_details_layout.addWidget(j0_label)
            j45_label = QLabel(f"J45: {j45_val:.2f} D")
            j45_label.setProperty("class", "measurement_detail_label")
            text_details_layout.addWidget(j45_label)

            text_details_layout.addStretch(1) 
            card_main_layout.addWidget(text_details_widget)

            maps_grid_widget = QWidget()
            maps_layout = QGridLayout(maps_grid_widget) 
            maps_layout.setSpacing(8) 
            
            map_paths_dict = res.get('map_paths', {})
            grid_positions = [(0,0), (0,1), (1,0), (1,1)] 

            for i, map_key in enumerate(map_order):
                map_path = map_paths_dict.get(map_key)
                
                map_item_container = QWidget()
                map_item_layout = QVBoxLayout(map_item_container)
                map_item_layout.setContentsMargins(0,0,0,0)
                map_item_layout.setSpacing(2) 
                map_item_layout.setAlignment(Qt.AlignCenter)

                title_label = QLabel(map_titles.get(map_key, "Corneal Map"))
                title_label.setProperty("class", "map_title_label") 
                title_label.setAlignment(Qt.AlignCenter)
                map_item_layout.addWidget(title_label)

                heatmap_lbl = QLabel()
                heatmap_lbl.setFixedSize(map_display_size) 
                heatmap_lbl.setAlignment(Qt.AlignCenter)
                heatmap_lbl.setProperty("class", "heatmap_display_label") 

                if map_path and os.path.exists(map_path):
                    pixmap = QPixmap(map_path)
                    if not pixmap.isNull(): 
                        heatmap_lbl.setPixmap(pixmap.scaled(map_display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    else: 
                        heatmap_lbl.setText("Map Error"); heatmap_lbl.setWordWrap(True)
                else: 
                    heatmap_lbl.setText("Map N/A"); heatmap_lbl.setWordWrap(True) 
                
                map_item_layout.addWidget(heatmap_lbl)
                maps_layout.addWidget(map_item_container, grid_positions[i][0], grid_positions[i][1])
            
            card_main_layout.addWidget(maps_grid_widget)
            self.numerical_results_layout.addWidget(result_card)
        
        self.numerical_results_layout.addStretch()

    def generate_pdf_report(self):
        if not REPORTLAB_AVAILABLE:
            QMessageBox.warning(self, "PDF Export Error", "ReportLab library is not installed. PDF export is disabled.\nPlease install it using: pip install reportlab")
            return

        if not self.current_astigmatism_results:
            QMessageBox.information(self, "No Data", "No measurement data is currently loaded to generate a PDF report.")
            return
        
        record_to_report = self.current_astigmatism_results[0]

        default_pdf_dir = os.path.join(self.settings_base_output_dir, "reports")
        os.makedirs(default_pdf_dir, exist_ok=True)
        default_pdf_filename = f"{record_to_report.get('PatientID', 'report')}_{record_to_report.get('Eye', 'EYE')}_report.pdf"
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", os.path.join(default_pdf_dir, default_pdf_filename) , "PDF Files (*.pdf)")
        if not file_path:
            return

        try:
            doc = SimpleDocTemplate(file_path, pagesize=letter,
                                    rightMargin=72, leftMargin=72,
                                    topMargin=72, bottomMargin=18)
            styles = getSampleStyleSheet()
            story = []

            title_style = styles['h1']
            title_style.alignment = Qt.AlignCenter
            story.append(Paragraph("Astigmata Analyzer - Patient Report", title_style))
            story.append(Spacer(1, 0.25 * inch))

            p_style = styles['Normal']
            p_style_bold = ParagraphStyle('Bold', parent=styles['Normal'], fontName='Helvetica-Bold')

            story.append(Paragraph(f"Patient ID: {record_to_report.get('PatientID', 'N/A')}", p_style_bold))
            story.append(Paragraph(f"Eye: {record_to_report.get('Eye', 'N/A')}", p_style_bold))
            story.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", p_style))
            story.append(Spacer(1, 0.2 * inch))

            story.append(Paragraph("Measurement Data:", styles['h2']))
            data_text = (f"Sphere: {record_to_report.get('Sphere (Diopter)', 0.0):.2f} D<br/>"
                         f"Cylinder: {record_to_report.get('Raw Cylinder', 0.0):.2f} D<br/>"
                         f"Axis: {record_to_report.get('Axis (Degrees)', 0)}°<br/>"
                         f"J0: {record_to_report.get('J0', 0.0):.2f} D<br/>"
                         f"J45: {record_to_report.get('J45', 0.0):.2f} D")
            story.append(Paragraph(data_text, p_style))
            story.append(Spacer(1, 0.2 * inch))

            status_text = "ASTIGMATISM DETECTED" if record_to_report.get('Astigmatism Detected') else f"No significant astigmatism detected (Cylinder < {self.settings_astigmatism_threshold}D)."
            story.append(Paragraph(f"Status: {status_text}", p_style_bold))
            story.append(Spacer(1, 0.2 * inch))

            story.append(Paragraph("Corneal Maps:", styles['h2']))
            map_paths = record_to_report.get('map_paths', {})
            map_order = ["axial_curvature", "elevation_front", "corneal_thickness", "elevation_back"]
            map_titles_pdf = {
                "axial_curvature": "Axial Curvature (Front)", "elevation_front": "Elevation (Front)",
                "corneal_thickness": "Corneal Thickness", "elevation_back": "Elevation (Back)"
            }
            
            table_data = []
            row_images = []
            row_titles = []
            img_width = 2.5 * inch 

            for i, key in enumerate(map_order):
                path = map_paths.get(key)
                title = map_titles_pdf.get(key, "Map")
                if path and os.path.exists(path):
                    img = ReportLabImage(path, width=img_width, height=img_width) 
                    row_images.append(img)
                    row_titles.append(Paragraph(title, styles['Normal']))
                else:
                    row_images.append(Paragraph("(Map N/A)", styles['Italic']))
                    row_titles.append(Paragraph(title, styles['Normal']))
                
                if len(row_images) == 2: 
                    table_data.append(row_titles)
                    table_data.append(row_images)
                    row_images = []
                    row_titles = []
            
            if row_images: 
                table_data.append(row_titles)
                table_data.append(row_images)

            if table_data:
                num_cols = 2 
                available_width = doc.width
                col_width = available_width / num_cols
                
                pdf_table = Table(table_data, colWidths=[col_width] * num_cols)
                pdf_table.setStyle(TableStyle([
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('GRID', (0,0), (-1,-1), 0.5, reportlab_colors.grey),
                    ('LEFTPADDING', (0,0), (-1,-1), 6), ('RIGHTPADDING', (0,0), (-1,-1), 6),
                    ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(pdf_table)

            doc.build(story)
            QMessageBox.information(self, "PDF Exported", f"Report saved to: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "PDF Export Error", f"Could not generate PDF: {e}")
            print(f"Error generating PDF: {e}")


    def export_results_to_csv(self):
        if not self.current_astigmatism_results:
            QMessageBox.information(self, "No Data", "No measurement data is currently loaded to export.")
            return

        default_csv_dir = os.path.join(self.settings_base_output_dir, "csv_exports")
        os.makedirs(default_csv_dir, exist_ok=True)
        default_csv_filename = f"astigmata_analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results to CSV", os.path.join(default_csv_dir, default_csv_filename), "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            headers = [
                'PatientID', 'Eye', 'Sphere (Diopter)', 'Raw Cylinder', 
                'Degree (Diopter)', 'Axis (Degrees)', 'Astigmatism Detected',
                'J0', 'J45', 'Map Axial Path', 'Map Elevation Front Path', 
                'Map Thickness Path', 'Map Elevation Back Path'
            ]
            with open(file_path, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=headers)
                writer.writeheader()
                for record in self.current_astigmatism_results:
                    row_to_write = {
                        'PatientID': record.get('PatientID'),
                        'Eye': record.get('Eye'),
                        'Sphere (Diopter)': record.get('Sphere (Diopter)'),
                        'Raw Cylinder': record.get('Raw Cylinder'),
                        'Degree (Diopter)': record.get('Degree (Diopter)'),
                        'Axis (Degrees)': record.get('Axis (Degrees)'),
                        'Astigmatism Detected': record.get('Astigmatism Detected'),
                        'J0': f"{record.get('J0', 0.0):.2f}", 
                        'J45': f"{record.get('J45', 0.0):.2f}", 
                        'Map Axial Path': record.get('map_paths', {}).get('axial_curvature'),
                        'Map Elevation Front Path': record.get('map_paths', {}).get('elevation_front'),
                        'Map Thickness Path': record.get('map_paths', {}).get('corneal_thickness'),
                        'Map Elevation Back Path': record.get('map_paths', {}).get('elevation_back')
                    }
                    writer.writerow(row_to_write)
            QMessageBox.information(self, "CSV Exported", f"Results saved to: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "CSV Export Error", f"Could not export to CSV: {e}")
            print(f"Error exporting to CSV: {e}")


    def clear_results(self):
        self.loading_container.setVisible(False); self.initial_results_label.setVisible(True); self.notice_label.setVisible(True)
        self.numerical_results_container.setVisible(False); self.image_results_container.setVisible(False)
        self.clear_results_content(keep_persistent=False)
        self.drop_zone.dropped_pixmap_path = None
        if hasattr(self.drop_zone, 'initial_pixmap') and not self.drop_zone.initial_pixmap.isNull():
            self.drop_zone.setPixmap(self.drop_zone.initial_pixmap.scaled(self.drop_zone.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else: self.drop_zone.setPixmap(QPixmap()); self.drop_zone.setText("Drop Eye Image Here\n(.jpg, .jpeg, .png)")
        self.drop_zone.setAlignment(Qt.AlignCenter)
        self.current_astigmatism_results = [] 

    def clear_results_content(self, keep_persistent=False):
        persistent_widgets = [self.loading_container, self.initial_results_label, self.image_results_container, self.numerical_results_container]
        self.clear_layout(self.symptoms_layout); self.clear_layout(self.numerical_results_layout)
        if not keep_persistent:
            items_to_remove = []
            for i in reversed(range(self.results_layout.count())):
                item = self.results_layout.itemAt(i); widget = item.widget()
                is_persistent = widget in persistent_widgets
                is_stretch = isinstance(item, QSpacerItem) and (i == self.results_layout.count() -1) 
                if not is_persistent and not is_stretch: items_to_remove.append(item)
            for item in items_to_remove:
                if item.widget(): self.results_layout.removeWidget(item.widget()); item.widget().deleteLater()
    def clear_layout(self, layout):
        if layout:
            while layout.count():
                item = layout.takeAt(0); widget = item.widget()
                if widget: widget.deleteLater()
                elif item.layout(): self.clear_layout(item.layout()); item.layout().deleteLater()

if __name__ == "__main__":
    app = QApplication.instance(); 
    if not app: app = QApplication(sys.argv)
    window = MainWindow(); window.show()
    sys.exit(app.exec_())
