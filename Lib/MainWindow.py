from time import process_time

# GUI
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea,
                             QFileDialog, QSpacerItem, QSizePolicy)

from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve
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
        
        self.setWindowTitle("Team Astigmata - Advanced Corneal Analysis")
        self.setMinimumSize(1400, 900)
        
        # Enhanced modern styling
        self.setStyleSheet("""
            QMainWindow, QWidget { 
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #0f0f23, stop: 1 #1a1a2e);
                color: #e8e9f3; 
                font-family: 'Segoe UI', 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif; 
            }
            
            QPushButton { 
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #667eea, stop: 1 #764ba2);
                color: white; 
                border: none; 
                border-radius: 8px; 
                padding: 12px 24px; 
                font-size: 14px; 
                font-weight: 600; 
                outline: none;
                min-width: 120px;
            }
            
            QPushButton:hover { 
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #5a67d8, stop: 1 #6b46c1);
                transform: translateY(-1px);
            } 
            
            QPushButton:pressed { 
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4c51bf, stop: 1 #553c9a);
            }
            
            QPushButton#browse_button {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #48bb78, stop: 1 #38a169);
            }
            
            QPushButton#browse_button:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #38a169, stop: 1 #2f855a);
            }
            
            QScrollArea { 
                border: 1px solid rgba(255, 255, 255, 0.1); 
                border-radius: 12px; 
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
            }
            
            QLabel { color: #e8e9f3; }
            
            QLabel#header_label { 
                color: #ffffff; 
                font-size: 28px; 
                font-weight: 700; 
                margin-bottom: 20px; 
                padding: 15px;
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(102, 126, 234, 0.1), stop: 1 rgba(118, 75, 162, 0.1));
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            QLabel#section_header { 
                color: #a78bfa; 
                font-size: 16px; 
                font-weight: 600; 
                margin-bottom: 15px;
                padding: 8px 0px;
            }
            
            QLabel#notice_label { 
                font-size: 12px; 
                color: #cbd5e0; 
                margin-top: 15px;
                padding: 15px;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 8px;
                border-left: 3px solid #667eea;
            }
            
            QFrame#separator_line { 
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 rgba(255, 255, 255, 0.1), 
                    stop: 0.5 rgba(255, 255, 255, 0.2), 
                    stop: 1 rgba(255, 255, 255, 0.1));
                min-width: 2px; 
                max-width: 2px;
                border-radius: 1px;
            }
            
            QWidget#title_bar { 
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(15, 15, 35, 0.95), stop: 1 rgba(26, 26, 46, 0.95));
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
            }
            
            QLabel#title_bar_label { 
                color: #e8e9f3; 
                font-weight: 600; 
                padding-left: 15px;
                font-size: 14px;
            }
            
            QPushButton.title_bar_button { 
                background-color: transparent; 
                color: #e8e9f3; 
                border: none; 
                padding: 8px; 
                margin: 0px; 
                font-size: 16px; 
                font-weight: bold; 
                min-width: 45px;
                border-radius: 6px;
            }
            
            QPushButton.title_bar_button:hover { 
                background-color: rgba(255, 255, 255, 0.1); 
            }
            
            QPushButton#close_button.title_bar_button:hover { 
                background-color: #e53e3e; 
                color: white; 
            }
            
            QPushButton.title_bar_button:pressed { 
                background-color: rgba(255, 255, 255, 0.2); 
            }
            
            QWidget#left_panel {
                background: rgba(255, 255, 255, 0.02);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
            
            QWidget#results_panel {
                background: rgba(255, 255, 255, 0.02);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
        """)

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 10, 20, 20)
        main_layout.setSpacing(20)
        
        # Title bar
        self.create_title_bar(main_layout)
        
        # Header
        header_label = QLabel("Advanced Corneal Analysis & Astigmatism Detection")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setObjectName("header_label")
        main_layout.addWidget(header_label)
        
        # Main content layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(25)
        
        # Left panel (Image upload) - Increased size
        left_widget = self.create_left_panel()
        content_layout.addWidget(left_widget)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setObjectName("separator_line")
        content_layout.addWidget(line)
        
        # Right panel (Results)
        self.result_display = CornealAnalysisWidget()
        results_container = QWidget()
        results_container.setObjectName("results_panel")
        results_layout = QVBoxLayout(results_container)
        results_layout.setContentsMargins(20, 20, 20, 20)
        results_layout.addWidget(self.result_display)
        content_layout.addWidget(results_container)
        
        # Adjust stretch factors to give more space to image
        content_layout.setStretchFactor(left_widget, 6)  # Increased from 3
        content_layout.setStretchFactor(results_container, 4)  # Decreased from 7
        
        main_layout.addLayout(content_layout)
        self.setCentralWidget(central_widget)
        
        # Window settings
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self._drag_pos = None

    def create_title_bar(self, main_layout):
        """Create the custom title bar"""
        self.title_bar = QWidget()
        self.title_bar.setObjectName("title_bar")
        self.title_bar.setFixedHeight(45)
        
        title_bar_layout = QHBoxLayout(self.title_bar)
        title_bar_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel("Team Astigmata - Corneal Analysis Suite")
        title_label.setObjectName("title_bar_label")
        
        # Window control buttons
        min_button = QPushButton("−")
        min_button.setFixedSize(45, 35)
        min_button.clicked.connect(self.showMinimized)
        min_button.setProperty("class", "title_bar_button")
        
        self.max_button = QPushButton("□")
        self.max_button.setFixedSize(45, 35)
        self.max_button.clicked.connect(self.toggle_maximize)
        self.max_button.setProperty("class", "title_bar_button")
        
        close_button = QPushButton("✕")
        close_button.setFixedSize(45, 35)
        close_button.clicked.connect(self.close)
        close_button.setObjectName("close_button")
        close_button.setProperty("class", "title_bar_button")
        
        title_bar_layout.addWidget(title_label)
        title_bar_layout.addStretch()
        title_bar_layout.addWidget(min_button)
        title_bar_layout.addWidget(self.max_button)
        title_bar_layout.addWidget(close_button)
        
        main_layout.addWidget(self.title_bar)

    def create_left_panel(self):
        """Create the enhanced left panel with larger image area"""
        left_widget = QWidget()
        left_widget.setObjectName("left_panel")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(25, 25, 25, 25)
        left_layout.setSpacing(20)

        # Section header
        input_section_label = QLabel("📸 Upload Corneal Image")
        input_section_label.setObjectName("section_header")
        left_layout.addWidget(input_section_label)

        # Enhanced drop zone - Removed height constraint and made it expand
        self.drop_zone = ImageDropZone()
        self.drop_zone.imageDropped.connect(self.process_image)
        
        # Allow the drop zone to expand vertically
        self.drop_zone.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.drop_zone.setMinimumSize(400, 400)  # Increased minimum size
        
        left_layout.addWidget(self.drop_zone)

        # File actions
        file_actions_layout = QHBoxLayout()
        file_actions_layout.setSpacing(15)
        
        browse_image_button = QPushButton("📁 Browse Images")
        browse_image_button.setObjectName("browse_button")
        browse_image_button.clicked.connect(self.browse_image)
        
        clear_button = QPushButton("🗑️ Clear Image")
        clear_button.clicked.connect(self.clear_image)
        
        file_actions_layout.addWidget(browse_image_button)
        file_actions_layout.addWidget(clear_button)
        left_layout.addLayout(file_actions_layout)

        # Notice
        self.notice_label = QLabel(
            "💡 Upload high-quality corneal topography images (JPG, PNG) for accurate analysis. "
            "This tool provides preliminary analysis and should not replace professional medical consultation.")
        self.notice_label.setAlignment(Qt.AlignLeft)
        self.notice_label.setWordWrap(True)
        self.notice_label.setObjectName("notice_label")
        left_layout.addWidget(self.notice_label)

        return left_widget

    def clear_image(self):
        """Clear the current image"""
        self.drop_zone.clear_image()
        self.result_display.clear()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.title_bar.underMouse():
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def changeEvent(self, event):
        if event.type() == event.WindowStateChange:
            self.max_button.setText("❐" if self.isMaximized() else "□")
        super().changeEvent(event)

    def toggle_maximize(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Corneal Image", 
            "", 
            "Image Files (*.jpg *.jpeg *.png);;All Files (*)"
        )
        if file_path:
            self.process_image(file_path)
            self.drop_zone.show_dropped_image(file_path)

    def process_image(self, image_path):
        """Process the uploaded image"""
        try:
            # Show loading state (you could add a loading animation here)
            self.result_display.clear()
            
            # Analyze the image
            analysis_results = self.Analyzer.analyze_eye_image(image_path)
            
            # Display results
            self.result_display.display(analysis_results)
            
        except Exception as e:
            # Handle errors gracefully
            error_msg = f"Error processing image: {str(e)}"
            print(error_msg)  # You could show this in a status bar or dialog