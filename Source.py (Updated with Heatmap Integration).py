import sys
import os
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea,
                           QFileDialog, QSizePolicy, QMessageBox)
from PyQt5.QtCore import Qt, QMimeData, QTimer, QSize, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QDrag, QPixmap, QPainter, QPen, QColor, QFont, QPalette, QMovie, QFontDatabase

import torch
from torchvision import transforms
from PIL import Image
import random

# Import the heatmap generation function from heatmap.py
# Ensure heatmap.py is in the same directory as Source.py
try:
    from heatmap import generate_astigmatism_heatmap_viz
    print("Successfully imported generate_astigmatism_heatmap_viz from heatmap.py")
except ImportError:
    print("Error: Could not import generate_astigmatism_heatmap_viz from heatmap.py. Heatmap visualization will be disabled.")
    # Define a dummy function if import fails, to prevent crashes
    def generate_astigmatism_heatmap_viz(*args, **kwargs):
        print("Dummy heatmap function called: heatmap.py not found or import failed.")
        return None


try:
    from Main import CNN
    from torchvision.models import ResNet50_Weights
except ImportError:
    print("Warning: Could not import CNN or ResNet50_Weights. Image processing will be disabled.")
    class CNN(torch.nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.dummy_layer = torch.nn.Linear(1, 1)
            self.fc = torch.nn.Linear(1, num_classes)

        def forward(self, x):
            print("Warning: Dummy CNN forward called - image model not loaded.")
            return torch.randn(x.size(0), self.fc.out_features if hasattr(self, 'fc') else 4)
    ResNet50_Weights = None


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
        print(f"Worker: Initialized for image: {self.image_path}")

    def run_processing(self):
        """This method runs in the separate thread."""
        print(f"Worker: run_processing method started for {self.image_path}")

        if self.model is None:
             error_msg = "Worker: Image classification model not loaded. Cannot process image."
             print(error_msg)
             self.error.emit(error_msg)
             self.finished.emit()
             print("Worker: run_processing finished due to no model.")
             return

        try:
            print("Worker: Attempting to load image...")
            image = Image.open(self.image_path).convert("RGB")
            print("Worker: Image loaded. Applying transforms...")
            image = self.transform(image)
            image = image.unsqueeze(0)
            print("Worker: Transforms applied. Image tensor shape:", image.shape)

            print("Worker: Running model inference...")
            with torch.no_grad():
                device = next(self.model.parameters()).device
                print(f"Worker: Moving image to device: {device}")
                image = image.to(device)
                print("Worker: Calling model forward pass...")
                outputs = self.model(image)
                print("Worker: Model forward pass complete. Calculating probabilities...")
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                print("Worker: Probabilities calculated. Finding predicted class...")
                confidence, predicted_class_index = torch.max(probabilities, 1)
                print(f"Worker: Predicted class index: {predicted_class_index.item()}, Confidence: {confidence.item():.4f}")


            predicted_class_name = self.class_names[predicted_class_index.item()]
            confidence_score = confidence.item() * 100

            print(f"Worker: ML Processing Complete for {self.image_path}")
            print(f"Worker: Predicted Class: {predicted_class_name}")
            print(f"Worker: Confidence: {confidence_score:.2f}%")

            print("Worker: Emitting result_ready signal...")
            self.result_ready.emit(predicted_class_name, confidence_score)
            print("Worker: result_ready signal emitted.")

        except Exception as e:
            error_msg = f"Worker: Error during processing: {e}"
            print(f"!!! {error_msg}")
            self.error.emit(error_msg)

        finally:
            print("Worker: run_processing finally block entered. Emitting finished signal...")
            self.finished.emit()
            print("Worker: finished signal emitted.")


class ImageDropZone(QLabel):
    imageDropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed white;
                border-radius: 10px;
                background-color: transparent;
                padding: 10px;
            }
        """)

        self.initial_pixmap = QPixmap(200, 200)
        self.initial_pixmap.fill(Qt.transparent)
        painter = QPainter(self.initial_pixmap)
        painter.setPen(QPen(Qt.black, 2))

        if "Arial" in QFontDatabase().families():
             painter.setFont(QFont("Arial", 12))
        elif "Liberation Sans" in QFontDatabase().families():
             painter.setFont(QFont("Liberation Sans", 12))
        else:
             painter.setFont(QFont("Sans Serif", 12))

        painter.drawText(self.initial_pixmap.rect(), Qt.AlignCenter, "Drop Image Here\n(.jpg, .jpeg)")
        painter.end()

        self.setPixmap(self.initial_pixmap)
        self.setMinimumSize(200, 200)

    def resizeEvent(self, event):
        if hasattr(self, 'initial_pixmap') and not self.initial_pixmap.isNull():
            scaled_pixmap = self.initial_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            self.setAlignment(Qt.AlignCenter)
        super().resizeEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and event.mimeData().urls()[0].toLocalFile().lower().endswith(('.jpg', '.jpeg')):
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #aaa;
                    border-radius: 10px;
                    background-color: rgba(255, 255, 255, 0.1);
                    padding: 0px;
                }
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed white;
                border-radius: 10px;
                background-color: transparent;
                padding: 10px;
            }
        """)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if file_path.lower().endswith(('.jpg', '.jpeg')):
                dropped_pixmap = QPixmap(file_path)
                if not dropped_pixmap.isNull():
                     scaled_pixmap = dropped_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                     self.setPixmap(scaled_pixmap)
                     self.setAlignment(Qt.AlignCenter)
                     self.imageDropped.emit(file_path)
                     event.acceptProposedAction()
                else:
                    print(f"Failed to load image from {file_path}")
                    self.setPixmap(self.initial_pixmap)
                    self.setAlignment(Qt.AlignCenter)
                    event.ignore()
            else:
                self.setPixmap(self.initial_pixmap)
                self.setAlignment(Qt.AlignCenter)
                event.ignore()
        else:
            self.setPixmap(self.initial_pixmap)
            self.setAlignment(Qt.AlignCenter)
            event.ignore()

        self.setStyleSheet("""
            QLabel {
                border: 2px dashed white;
                border-radius: 10px;
                background-color: transparent;
                padding: 10px;
            }
        """)


class LoadingAnimation(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 60)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.angle)

        pen = QPen(Qt.black, 2)
        painter.setPen(pen)

        for i in range(8):
            line_length = 25 if i % 2 == (self.angle // 45) % 2 else 15
            painter.drawLine(0, 10, 0, line_length)
            painter.rotate(45)

    def rotate(self):
        self.angle = (self.angle + 10) % 360
        self.update()

    def showEvent(self, event):
        self.timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Team Astigmata")
        self.setMinimumSize(800, 500)
        # --- TEMPORARILY COMMENT OUT THE ENTIRE STYLESHEET (Main Window) ---
        # self.setStyleSheet("""
        #     QMainWindow, QWidget {
        #         background-color: #1a1a1a;
        #         color: white;
        #     }
        #     QPushButton {
        #         background-color: #333;
        #         color: white;
        #         border: 1px solid white;
        #         border-radius: 5px;
        #         padding: 8px 15px;
        #     }
        #     QPushButton:hover {
        #         background-color: #444;
        #     }
        #      QPushButton:pressed { /* Added pressed state */
        #         background-color: #555;
        #     }
        #     QScrollArea {
        #         border: 1px solid white;
        #         border-radius: 5px;
        #         background-color: #222;
        #     }
        #     QLabel {
        #         color: white;
        #     }
        # """)
        # --- END TEMPORARY COMMENT OUT ---


        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        header_label = QLabel("Eye Condition & Astigmatism Analyzer")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(header_label)

        content_layout = QHBoxLayout()

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        input_label = QLabel("Upload an Eye Image or Load Measurement Data:")
        input_label.setFont(QFont("Arial", 12))
        input_label.setAlignment(Qt.AlignLeft)
        left_layout.addWidget(input_label)

        self.drop_zone = ImageDropZone()
        self.drop_zone.imageDropped.connect(self.start_image_processing_thread)
        left_layout.addWidget(self.drop_zone)

        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0,10,0,0)
        button_layout.setAlignment(Qt.AlignCenter)

        browse_image_button = QPushButton("Browse Image")
        browse_image_button.clicked.connect(self.browse_image)
        browse_image_button.setFixedWidth(150)
        button_layout.addWidget(browse_image_button)

        load_data_button = QPushButton("Load Measurement Data")
        load_data_button.clicked.connect(self.load_measurement_data)
        load_data_button.setFixedWidth(150)
        button_layout.addWidget(load_data_button)

        left_layout.addWidget(button_container)

        self.notice_label = QLabel("Note: Image analysis provides a general classification.\nMeasurement data provides specific astigmatism values.\nNeither should substitute professional medical advice.")
        self.notice_label.setFont(QFont("Arial", 10))
        self.notice_label.setAlignment(Qt.AlignLeft)
        self.notice_label.setWordWrap(True)
        left_layout.addWidget(self.notice_label)

        left_layout.addStretch()
        content_layout.addWidget(left_widget)

        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        # --- Temporarily remove stylesheet here too ---
        # line.setStyleSheet("background-color: white;")
        # --- End temporary removal ---
        content_layout.addWidget(line)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        results_label = QLabel("Analysis Results")
        results_label.setFont(QFont("Arial", 12))
        right_layout.addWidget(results_label)

        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.results_content = QWidget()
        self.results_layout = QVBoxLayout(self.results_content)
        self.results_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # --- Temporarily remove stylesheet here too ---
        # self.results_content.setStyleSheet("padding: 10px;")
        # --- End temporary removal ---
        self.results_scroll.setWidget(self.results_content)

        # self.loading_container and initial_results_label are added to results_layout in __init__
        self.loading_container = QWidget()
        loading_layout = QVBoxLayout(self.loading_container)
        loading_layout.setContentsMargins(0,0,0,0)
        self.loading_animation = LoadingAnimation()
        loading_layout.addWidget(self.loading_animation, 0, Qt.AlignCenter)
        self.loading_container.setVisible(False)
        self.results_layout.addWidget(self.loading_container, 0, Qt.AlignCenter) # Add loading container initially


        self.detection_label = None
        self.confidence_label = None
        self.symptoms_label = None
        self.symptoms_content = None
        self.symptoms_layout = None

        # Initial label is also added to the layout initially
        self.initial_results_label = QLabel("Load an image or a measurement data file to see results.")
        self.initial_results_label.setAlignment(Qt.AlignCenter)
        self.initial_results_label.setFont(QFont("Arial", 11, QFont.Weight.Normal, True))
        self.initial_results_label.setWordWrap(True)
        self.results_layout.addWidget(self.initial_results_label, 0, Qt.AlignCenter | Qt.AlignTop) # Add initial label initially

        self.results_layout.addStretch() # Add stretch at the end


        right_layout.addWidget(self.results_scroll)
        content_layout.addWidget(right_widget)

        content_layout.setStretchFactor(left_widget, 4)
        content_layout.setStretchFactor(right_widget, 6)

        main_layout.addLayout(content_layout)
        self.setCentralWidget(central_widget)

        self.setWindowFlags(Qt.FramelessWindowHint)

        title_bar = QWidget()
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(10, 5, 10, 5)

        title_label = QLabel("Team Astigmata")
        title_label.setFont(QFont("Arial", 10, QFont.Bold))

        min_button = QPushButton("-")
        min_button.setFixedSize(30, 30)
        min_button.clicked.connect(self.showMinimized)

        max_button = QPushButton("□")
        max_button.setFixedSize(30, 30)
        max_button.clicked.connect(self.toggle_maximize)

        close_button = QPushButton("x")
        close_button.setFixedSize(30, 30)
        close_button.clicked.connect(self.close)

        title_bar_layout.addWidget(title_label)
        title_bar_layout.addStretch()
        title_bar_layout.addWidget(min_button)
        title_bar_layout.addWidget(max_button)
        title_bar_layout.addWidget(close_button)

        main_layout.insertWidget(0, title_bar)

        self._drag_pos = None

        self.thread = None
        self.worker = None

        self.image_model = None
        model_path = "cnn_model.pth"
        if os.path.exists(model_path):
            try:
                self.image_model = CNN(num_classes = 4)
                self.image_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.image_model.eval()
                print(f"Image classification model loaded successfully from {model_path}")
                self.image_model.to(torch.device('cpu'))
            except Exception as e:
                print(f"Error loading image classification model from {model_path}: {e}")
                self.image_model = None
        else:
             print(f"Warning: Image classification model file not found at {model_path}. Image processing will be disabled.")
             self.image_model = None

        self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                            std  = [0.229, 0.224, 0.225])
])
        self.image_class_names = ['Astigmatism', 'Normal', 'Cataract', 'Diabetic Retinopathy']


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def browse_image(self):
        """Handles Browse and processing an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "JPEG Images (*.jpg *.jpeg)"
        )
        if file_path:
            print(f"Selected image: {file_path}")
            # Clear results and trigger threaded image processing
            self.clear_results() # Clear old results and reset drop zone state
            self.start_image_processing_thread(file_path)


    # --- Trigger Threaded Image Processing ---
    def start_image_processing_thread(self, image_path):
        """Starts a new thread to process the image."""
        print("Main Thread: Preparing to start image processing thread...")
        if self.image_model is None:
            print("Main Thread: Cannot start image processing thread: Model not loaded.")
            self.clear_results_content() # Clear any previous content
            error_label = QLabel("Error: Image classification model not available. Cannot analyze image.")
            error_label.setStyleSheet("color: red;")
            error_label.setWordWrap(True)
            self.results_layout.addWidget(error_label)
            self.results_layout.addStretch()
            self.loading_container.setVisible(False)
            self.initial_results_label.setVisible(False) # Hide initial message
            print("Main Thread: Image processing thread not started.")
            return

        # Clear previous results content BEFORE showing loading/starting thread
        print("Main Thread: Clearing results content...")
        self.clear_results_content()

        print("Main Thread: Showing loading animation...")
        self.loading_container.setVisible(True)
        self.initial_results_label.setVisible(False) # Hide initial message
        self.notice_label.setVisible(True)

        print("Main Thread: Displaying selected image in drop zone...")
        self.drop_zone.setPixmap(QPixmap())
        loaded_pixmap = QPixmap(image_path)
        if not loaded_pixmap.isNull():
             scaled_pixmap = loaded_pixmap.scaled(self.drop_zone.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
             self.drop_zone.setPixmap(scaled_pixmap)
             self.drop_zone.setAlignment(Qt.AlignCenter)
        else:
             print(f"Main Thread: Failed to load image for display in drop zone: {image_path}")
             self.drop_zone.setText("Failed to load image")
             self.drop_zone.setPixmap(QPixmap())
             self.drop_zone.setAlignment(Qt.AlignCenter)


        print("Main Thread: Creating QThread and ImageProcessingWorker...")
        self.thread = QThread()
        self.worker = ImageProcessingWorker(
            image_path=image_path,
            model=self.image_model,
            transform=self.transform,
            class_names=self.image_class_names
        )

        print("Main Thread: Moving worker to thread...")
        self.worker.moveToThread(self.thread)

        print("Main Thread: Connecting signals and slots...")
        self.thread.started.connect(self.worker.run_processing)
        self.worker.result_ready.connect(self.display_image_results, Qt.QueuedConnection)
        self.worker.error.connect(self.handle_image_processing_error, Qt.QueuedConnection)

        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_image_thread)


        print("Main Thread: Starting the thread...")
        self.thread.start()
        print("Main Thread: Image processing thread started.")


    # --- Slot for Cleanup ---
    def cleanup_image_thread(self):
        """Cleans up the image processing worker and thread after they finish."""
        print("Main Thread: Cleaning up image processing thread and worker.")
        if self.worker:
            print("Main Thread: Deleting worker...")
            self.worker.deleteLater()
            self.worker = None
            print("Main Thread: Worker deleted.")
        if self.thread:
            print("Main Thread: Deleting thread...")
            self.thread.deleteLater()
            self.thread = None
            print("Main Thread: Thread deleted.")
        print("Main Thread: Cleanup complete.")

    # --- END Slot for Cleanup ---


    def display_image_results(self, predicted_class_name, confidence_score):
        """Displays the results for image classification (called from worker signal)."""
        print("Main Thread: Received image processing results. Displaying...")
        print("Main Thread: Attempting to hide loading animation.")
        self.loading_container.setVisible(False)
        print("Main Thread: Loading animation visibility set.")

        # clear_results_content() is now called BEFORE the thread starts

        # Now, add the new result widgets to the layout
        self.detection_label = QLabel(f"Detection: {predicted_class_name}")
        self.detection_label.setFont(QFont("Arial", 11))

        self.confidence_label = QLabel(f"{confidence_score:.2f}%")
        self.confidence_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.confidence_label.setStyleSheet("color: lightgreen;")
        self.confidence_label.setAlignment(Qt.AlignRight)

        detection_layout_container = QWidget()
        detection_layout = QHBoxLayout(detection_layout_container)
        detection_layout.setContentsMargins(0,0,0,0)
        detection_layout.addWidget(self.detection_label)
        detection_layout.addWidget(self.confidence_label)
        self.results_layout.addWidget(detection_layout_container)

        symptoms = []
        AstigmatsmSymtoms = [
            "Blurred or distorted vision.",
            "Eyestrain or discomfort.",
            "Headaches"
        ]
        CataractSymptoms = [
            "Clouded, blurred, or dim vision.",
            "Difficulty seeing at night.",
            "Sensitivity to light and glare.",
            "Seeing 'halos' around lights.",
            "Frequent changes in eyeglass or contact lens prescription.",
            "Fading or yellowing of colors.",
            "Double vision in a single eye."
        ]
        DiabeticRetinopathySymptoms = [
            "Spots or dark strings (floaters) in vision.",
            "Blurred vision.",
            "Fluctuating vision.",
            "Impaired color vision.",
            "Dark or empty areas in your vision.",
            "Vision loss."
        ]
        NormalSymptoms = [
            "No specific symptoms commonly associated with eye diseases or significant refractive errors are detected from the image."
        ]

        if "Astigmatism" in predicted_class_name:
            symptoms = AstigmatsmSymtoms
        elif "Cataract" in predicted_class_name:
             symptoms = CataractSymptoms
        elif "Diabetic Retinopathy" in predicted_class_name:
             symptoms = DiabeticRetinopathySymptoms
        elif "Normal" in predicted_class_name:
             symptoms = NormalSymptoms

        if symptoms:
            self.symptoms_label = QLabel("Symptoms:")
            self.symptoms_label.setFont(QFont("Arial", 11, QFont.Bold))
            self.results_layout.addWidget(self.symptoms_label)

            self.symptoms_content = QWidget()
            self.symptoms_layout = QVBoxLayout(self.symptoms_content)
            self.symptoms_layout.setContentsMargins(10, 10, 10, 10)
            # --- Temporarily remove stylesheet here too ---
            # self.symptoms_content.setStyleSheet("""
            #     QWidget {
            #         background-color: #222;
            #         border-radius: 5px;
            #         border: 1px solid #555;
            #     }
            # """)
            # --- End temporary removal ---
            self.results_layout.addWidget(self.symptoms_content)

            for symptom in symptoms:
                symptom_label = QLabel(f"• {symptom}")
                symptom_label.setFont(QFont("Arial", 10))
                symptom_label.setWordWrap(True)
                self.symptoms_layout.addWidget(symptom_label)
        else:
             pass

        self.results_layout.addStretch() # Ensure stretch is at the end
        print("Main Thread: Finished displaying image results.")

    def handle_image_processing_error(self, error_message):
        """Displays an error message if image processing fails in the worker thread."""
        print(f"Main Thread: Received error from worker: {error_message}")
        print("Main Thread: Attempting to hide loading animation on error.")
        self.loading_container.setVisible(False)
        print("Main Thread: Loading animation visibility set on error.")

        # clear_results_content() is now called BEFORE the thread starts

        self.clear_results_content() # Clear whatever was in the layout (e.g., the initial message)

        error_label = QLabel(f"Error during image analysis:<br>{error_message}")
        error_label.setStyleSheet("color: red;")
        error_label.setWordWrap(True)
        self.results_layout.addWidget(error_label)
        self.results_layout.addStretch() # Ensure stretch is at the end
        # Optionally show a QMessageBox for critical errors
        # QMessageBox.critical(self, "Processing Error", error_message)


    def load_measurement_data(self):
        """Handles loading the simulated numerical autorefractor data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Measurement Data File",
            "",
            "Data Files (*.txt *.csv);;All Files (*.*)"
        )
        if file_path:
            print(f"Loading measurement data from: {file_path}")
            self.clear_results() # Clear old results and reset drop zone state
            self.loading_container.setVisible(True)
            self.initial_results_label.setVisible(False) # Hide initial message
            self.notice_label.setVisible(True)

            self.process_measurement_data(file_path)


    def process_measurement_data(self, file_path):
        """Reads the numerical data file, detects astigmatism, and prepares results."""
        print("--- Start Processing Measurement Data ---")
        print(f"Attempting to process file: {file_path}")

        astigmatism_results = []
        ASTIGMATISM_THRESHOLD = 0.25

        try:
            print("Opening file...")
            with open(file_path, mode='r') as infile:
                print("File opened. Creating CSV reader.")
                reader = csv.DictReader(infile)
                required_cols = ['PatientID', 'Eye', 'Sphere', 'Cylinder', 'Axis']
                print(f"Checking header, expected: {required_cols}")
                if not reader.fieldnames or not all(col in reader.fieldnames for col in required_cols):
                     print(f"Error: File {file_path} does not have the expected header {required_cols}.")
                     self.clear_results_content()
                     error_label = QLabel(f"Error: Incorrect file format.<br>Expected header: {', '.join(required_cols)}")
                     error_label.setStyleSheet("color: red;")
                     error_label.setWordWrap(True)
                     self.results_layout.addWidget(error_label)
                     self.results_layout.addStretch()
                     self.loading_container.setVisible(False)
                     self.initial_results_label.setVisible(False)
                     print("--- End Processing Measurement Data (Header Error) ---")
                     return

                print(f"Header check passed. Found fields: {reader.fieldnames}. Starting row iteration.")
                row_count = 0
                for row in reader:
                    row_count += 1
                    try:
                        patient_id = row.get('PatientID', 'N/A').strip()
                        eye = row.get('Eye', 'N/A').strip()
                        cylinder_str = row.get('Cylinder', '0.0').strip()
                        axis_str = row.get('Axis', '0').strip()
                        sphere_str = row.get('Sphere', '0.0').strip()

                        try:
                           cylinder = float(cylinder_str)
                        except ValueError:
                           print(f"Warning: Invalid Cylinder value '{cylinder_str}' for {patient_id} {eye}. Using 0.0.")
                           cylinder = 0.0

                        try:
                           axis = int(float(axis_str))
                           if not (0 <= axis <= 180):
                               print(f"Warning: Invalid Axis value '{axis_str}' for {patient_id} {eye}. Axis should be between 0 and 180. Using 0.")
                               axis = 0
                        except ValueError:
                           print(f"Warning: Invalid Axis value '{axis_str}' for {patient_id} {eye}. Using 0.")
                           axis = 0

                        try:
                           sphere = float(sphere_str)
                        except ValueError:
                           print(f"Warning: Invalid Sphere value '{sphere_str}' for {patient_id} {eye}. Using 0.0.")
                           sphere = 0.0

                        # --- Heatmap Generation Call ---
                        heatmap_path = None
                        # Only attempt to generate heatmap if cylinder is above threshold
                        if abs(cylinder) >= ASTIGMATISM_THRESHOLD:
                             # Call the heatmap generation function from heatmap.py
                             heatmap_path = generate_astigmatism_heatmap_viz(
                                 cylinder=cylinder,
                                 axis=axis,
                                 patient_id=patient_id,
                                 eye=eye
                             )
                        # --- End Heatmap Generation Call ---


                        if abs(cylinder) >= ASTIGMATISM_THRESHOLD:
                            astigmatism_results.append({
                                'PatientID': patient_id,
                                'Eye': eye,
                                'Astigmatism Detected': True,
                                'Degree (Diopter)': abs(cylinder),
                                'Axis (Degrees)': axis,
                                'Sphere (Diopter)': sphere,
                                'Raw Cylinder': cylinder,
                                'HeatmapImagePath': heatmap_path # Store the path (will be None if generation failed or skipped)
                            })
                        else:
                             astigmatism_results.append({
                                'PatientID': patient_id,
                                'Eye': eye,
                                'Astigmatism Detected': False,
                                'Degree (Diopter)': 0.00,
                                'Axis (Degrees)': None,
                                'Sphere (Diopter)': sphere,
                                'Raw Cylinder': cylinder,
                                'HeatmapImagePath': heatmap_path # Store the path (will be None)
                             })
                    except Exception as e:
                        print(f"An unexpected error occurred processing row {row_count}: {row}. Error: {e}")

                print(f"Finished processing all {row_count} rows.")

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            self.clear_results_content()
            error_label = QLabel(f"Error: File not found at<br>{file_path}")
            error_label.setStyleSheet("color: red;")
            error_label.setWordWrap(True)
            self.results_layout.addWidget(error_label)
            self.results_layout.addStretch()
            self.loading_container.setVisible(False)
            self.initial_results_label.setVisible(False)
            print("--- End Processing Measurement Data (File Not Found) ---")
            return

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            self.clear_results_content()
            error_label = QLabel(f"Error reading file:<br>{e}")
            error_label.setStyleSheet("color: red;")
            error_label.setWordWrap(True)
            self.results_layout.addWidget(error_label)
            self.results_layout.addStretch()
            self.loading_container.setVisible(False)
            self.initial_results_label.setVisible(False)
            print("--- End Processing Measurement Data (File Reading Error) ---")
            return

        print(f"Successfully processed data. Results collected: {len(astigmatism_results)} items.")
        print("Calling display_measurement_results...")

        try:
            self.display_measurement_results(astigmatism_results)
            print("display_measurement_results called successfully.")
        except Exception as e:
            print(f"!!! ERROR during display_measurement_results: {e}")
            self.clear_results_content()
            error_label = QLabel(f"Error displaying results:<br>{e}")
            error_label.setStyleSheet("color: red;")
            error_label.setWordWrap(True)
            self.results_layout.addWidget(error_label)
            self.results_layout.addStretch()


        self.loading_container.setVisible(False)
        self.initial_results_label.setVisible(False) # Hide initial message after processing
        print("--- End Processing Measurement Data (Complete) ---")


    def display_measurement_results(self, results):
        """Displays the astigmatism detection results from numerical data."""
        print("--- Start Displaying Measurement Results ---")

        self.clear_results_content() # Clear previous content

        if not results:
            print("No results to display.")
            # If no results, show the initial message
            self.initial_results_label.setVisible(True)
            self.loading_container.setVisible(False) # Ensure loading is hidden
            print("--- End Displaying Measurement Results (No Results) ---")
            return

        # Hide the initial message if results are being displayed
        self.initial_results_label.setVisible(False)
        self.loading_container.setVisible(False) # Ensure loading is hidden

        print(f"Displaying {len(results)} results.")
        results_title = QLabel("Astigmatism Measurement Results:")
        results_title.setFont(QFont("Arial", 12, QFont.Bold))
        self.results_layout.addWidget(results_title)
        self.results_layout.addSpacing(10)
        print("Title and spacing added.")

        item_count = 0
        for res in results:
            item_count += 1
            try:
                # --- Create a container for this result item ---
                result_item_container = QWidget()
                # Use QHBoxLayout to place text and image side-by-side
                result_item_layout = QHBoxLayout(result_item_container)
                result_item_layout.setContentsMargins(8, 8, 8, 8) # Add some padding inside the container
                result_item_layout.setSpacing(10) # Space between text and image

                # --- Create Text Label ---
                result_text = ""
                if res.get('Astigmatism Detected', False): # Use .get with default for safety
                    result_text = f"<font color='lightgreen'>Patient {res.get('PatientID', 'N/A')}, {res.get('Eye', 'N/A')}:</font> <b>ASTIGMATISM Detected!</b>"
                    result_text += f"<br>&nbsp;&nbsp;Degree (Cylinder): {res.get('Degree (Diopter)', 0.0):.2f} D"
                    axis_val = res.get('Axis (Degrees)')
                    result_text += f"<br>&nbsp;&nbsp;Axis: {axis_val if axis_val is not None else 'N/A'} degrees"
                    sphere_val = res.get('Sphere (Diopter)', 0.0)
                    if abs(sphere_val) > 0.01:
                         result_text += f"<br>&nbsp;&nbsp;Sphere: {sphere_val:.2f} D"

                else:
                    result_text = f"<font color='white'>Patient {res.get('PatientID', 'N/A')}, {res.get('Eye', 'N/A')}:</font> No significant astigmatism detected."
                    raw_cylinder_val = res.get('Raw Cylinder', 0.0)
                    result_text += f"<br>&nbsp;&nbsp;Cylinder: {raw_cylinder_val:.2f} D (below threshold)"
                    sphere_val = res.get('Sphere (Diopter)', 0.0)
                    if abs(sphere_val) > 0.01:
                         result_text += f"<br>&nbsp;&nbsp;Sphere: {sphere_val:.2f} D"


                text_label = QLabel(result_text)
                text_label.setFont(QFont("Arial", 10))
                text_label.setWordWrap(True)
                text_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
                # Add text label to the container layout
                result_item_layout.addWidget(text_label, 1) # Give text label a stretch factor


                # --- Add Heatmap Image if available ---
                heatmap_path = res.get('HeatmapImagePath')
                if heatmap_path and os.path.exists(heatmap_path):
                    print(f"  Attempting to load heatmap image from {heatmap_path}")
                    heatmap_pixmap = QPixmap(heatmap_path)
                    if not heatmap_pixmap.isNull():
                        heatmap_label = QLabel()
                        # Scale the image to a fixed size for display (adjust as needed)
                        display_size = QSize(80, 80) # Example display size
                        scaled_heatmap_pixmap = heatmap_pixmap.scaled(display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        heatmap_label.setPixmap(scaled_heatmap_pixmap)
                        heatmap_label.setAlignment(Qt.AlignCenter)
                        heatmap_label.setFixedSize(display_size) # Fix the label size
                        # Add heatmap label to the container layout
                        result_item_layout.addWidget(heatmap_label, 0) # Give image label a fixed size (stretch 0)
                        print(f"  Added heatmap image for {res.get('PatientID', 'N/A')} {res.get('Eye', 'N/A')}.")
                    else:
                        print(f"  Failed to load heatmap image from {heatmap_path}")
                        # Optionally add a placeholder or error message
                        placeholder_label = QLabel("Image Failed")
                        placeholder_label.setFont(QFont("Arial", 8))
                        placeholder_label.setAlignment(Qt.AlignCenter)
                        placeholder_label.setFixedSize(QSize(80, 80))
                        result_item_layout.addWidget(placeholder_label, 0)

                # Add styling to the container widget for each result item
                result_item_container.setStyleSheet("""
                    QWidget {
                        border: 1px solid #555;
                        border-radius: 5px;
                        background-color: #2a2a2a;
                        margin-bottom: 8px; /* Add space between items */
                    }
                """)

                # Add the container widget for this result item to the main results layout
                self.results_layout.addWidget(result_item_container)


            except Exception as e:
                 print(f"!!! ERROR displaying result item {item_count}: {e}")
                 error_label = QLabel(f"<font color='red'>Error displaying result for Patient {res.get('PatientID', 'N/A')}, {res.get('Eye', 'N/A')}: {e}</font>")
                 error_label.setFont(QFont("Arial", 10))
                 error_label.setWordWrap(True)
                 self.results_layout.addWidget(error_label)


        self.results_layout.addStretch() # Ensure stretch is at the end
        print("Stretch added.")
        print("--- End Displaying Measurement Results ---")


    def clear_results(self):
        """Clears all displayed results, hides loading, and resets drop zone."""
        print("Clearing results...")
        self.loading_container.setVisible(False)
        self.initial_results_label.setVisible(True) # Show initial message
        self.notice_label.setVisible(True) # Keep notice visible

        self.clear_results_content() # Clear the content area

        # Reset drop zone appearance
        drop_zone_size = self.drop_zone.size()
        if drop_zone_size.width() <= 0 or drop_zone_size.height() <= 0:
             drop_zone_size = QSize(200, 200)

        self.drop_zone.initial_pixmap = QPixmap(drop_zone_size)
        self.drop_zone.initial_pixmap.fill(Qt.transparent)
        painter = QPainter(self.drop_zone.initial_pixmap)
        painter.setPen(QPen(Qt.black, 2))

        if "Arial" in QFontDatabase().families():
             painter.setFont(QFont("Arial", 12))
        elif "Liberation Sans" in QFontDatabase().families():
             painter.setFont(QFont("Liberation Sans", 12))
        else:
             painter.setFont(QFont("Sans Serif", 12))

        painter.drawText(self.drop_zone.initial_pixmap.rect(), Qt.AlignCenter, "Drop Image Here\n(.jpg, .jpeg)")
        painter.end()
        self.drop_zone.setPixmap(self.drop_zone.initial_pixmap)
        self.drop_zone.setAlignment(Qt.AlignCenter)


        print("Results area cleared, initial state restored.")


    def clear_results_content(self):
         """Helper function to clear all widgets currently in the results_layout EXCEPT persistent ones."""
         print("Clearing results content layout (protecting persistent widgets)...")

         # Identify persistent widgets that should NOT be deleted
         widgets_to_keep = [self.loading_container, self.initial_results_label]
         # Identify the final stretch item
         stretch_item = None
         if self.results_layout.count() > 0:
             last_item = self.results_layout.itemAt(self.results_layout.count() - 1)
             if hasattr(last_item, 'spacerItem') and last_item.spacerItem() is not None:
                  stretch_item = last_item.spacerItem()
                  print("Found stretch item to keep.")

         items_to_delete = []

         # Iterate through the layout items and decide what to delete
         # Iterate in reverse order to safely remove items
         for i in reversed(range(self.results_layout.count())):
              item = self.results_layout.itemAt(i)
              widget = item.widget()
              layout = item.layout()
              spacer = None
              if hasattr(item, 'spacerItem') and item.spacerItem() is not None:
                   spacer = item.spacerItem()

              # Check if the item is one of the persistent widgets or the main stretch item
              if (widget and widget in widgets_to_keep) or (spacer and spacer is stretch_item):
                   print(f"  Keeping persistent item: {type(item).__name__}")
                   continue # Skip items we want to keep

              # If it's not a persistent item, mark it for deletion
              items_to_delete.append(item)


         # Now, remove and delete the marked items
         for item in items_to_delete:
            widget = item.widget()
            layout = item.layout()
            if widget:
                 print(f"  Deleting dynamic widget: {type(widget).__name__}")
                 # Remove from layout first
                 self.results_layout.removeWidget(widget)
                 widget.deleteLater()
            elif layout:
                 print("  Deleting dynamic layout item...")
                 # Recursively clear nested layouts first
                 self.clear_layout(layout)
                 # Then remove the layout item from the parent layout
                 self.results_layout.removeItem(layout)
                 # Finally, delete the layout object itself
                 layout.deleteLater()
            elif hasattr(item, 'spacerItem') and item.spacerItem() is not None:
                 print("  Removing dynamic spacerItem.")
                 # Spacer items are removed directly from the layout
                 self.results_layout.removeItem(item.spacerItem())


         print("Results content layout cleared (dynamic content removed).")


    def clear_layout(self, layout):
        """Recursive helper to clear all items from a layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    self.clear_layout(item.layout())
                    item.layout().deleteLater()
                elif hasattr(item, 'spacerItem') and item.spacerItem() is not None:
                    pass


if __name__ == "__main__":
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
