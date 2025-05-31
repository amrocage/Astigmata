import os
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont, QFontDatabase


class ImageDropZone(QLabel):
    imageDropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumSize(200, 200)
        self.setMaximumHeight(300)

        # Initialize style
        self.normal_style = """
            QLabel {
                border: 2px dashed #555; 
                border-radius: 10px; 
                background-color: #2a2a2a;
                color: #ccc; 
                padding: 20px; 
                font-size: 14px;
            }
            QLabel:hover { 
                border: 2px dashed #777; 
                background-color: #3a3a3a; 
            }
        """

        self.drag_style = """
            QLabel { 
                border: 2px dashed #aaa; 
                border-radius: 10px; 
                background-color: rgba(255, 255, 255, 0.1); 
                color: #ccc; 
                font-size: 14px; 
            }
        """

        self.setStyleSheet(self.normal_style)

        # Track the current image
        self.dropped_image_path = None
        self.initial_pixmap = None

        # Create and set initial pixmap
        self.create_initial_pixmap()
        self.show_initial_state()

    def create_initial_pixmap(self):
        """Create the initial placeholder pixmap"""
        self.initial_pixmap = QPixmap(200, 200)
        self.initial_pixmap.fill(Qt.transparent)

        painter = QPainter(self.initial_pixmap)
        painter.setPen(QPen(Qt.white, 2))

        # Font selection
        font_families = QFontDatabase().families()
        font_name = "Arial"
        if "Liberation Sans" in font_families:
            font_name = "Liberation Sans"
        elif "DejaVu Sans" in font_families:
            font_name = "DejaVu Sans"
        elif not "Arial" in font_families and font_families:
            font_name = font_families[0]

        painter.setFont(QFont(font_name, 12))
        painter.drawText(self.initial_pixmap.rect(), Qt.AlignCenter,
                         "Drop Eye Image Here\n(.jpg, .jpeg, .png)")
        painter.end()

    def show_initial_state(self):
        """Show the initial placeholder state"""
        self.dropped_image_path = None
        if self.initial_pixmap:
            scaled_pixmap = self.initial_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)

    def show_dropped_image(self, image_path):
        """Display the dropped image"""
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                self.dropped_image_path = image_path
                scaled_pixmap = pixmap.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.setPixmap(scaled_pixmap)
                return True
        return False

    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)

        if self.dropped_image_path and os.path.exists(self.dropped_image_path):
            # Reload and scale the dropped image
            pixmap = QPixmap(self.dropped_image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.setPixmap(scaled_pixmap)
            else:
                self.show_initial_state()
        elif self.initial_pixmap:
            # Show scaled initial pixmap
            scaled_pixmap = self.initial_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)

        self.setAlignment(Qt.AlignCenter)

    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        if (event.mimeData().hasUrls() and
                event.mimeData().urls()[0].toLocalFile().lower().endswith(('.jpg', '.jpeg', '.png'))):
            event.acceptProposedAction()
            self.setStyleSheet(self.drag_style)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave events"""
        self.setStyleSheet(self.normal_style)

    def dropEvent(self, event):
        """Handle drop events"""
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()

            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Try to display the dropped image
                if self.show_dropped_image(file_path):
                    self.imageDropped.emit(file_path)
                    event.acceptProposedAction()
                else:
                    # Failed to load image, show initial state
                    self.show_initial_state()
                    event.ignore()
            else:
                # Not a valid image file
                self.show_initial_state()
                event.ignore()
        else:
            # No URLs in drop
            self.show_initial_state()
            event.ignore()

        # Reset styling
        self.setStyleSheet(self.normal_style)
        self.setAlignment(Qt.AlignCenter)

    def clear_image(self):
        self.show_initial_state()

    def get_current_image_path(self):
        return self.dropped_image_path
