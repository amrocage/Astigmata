import os
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect, QSize
from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont, QFontDatabase, QBrush, QColor


class ImageDropZone(QLabel):
    imageDropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 400)  # Increased minimum size
        # Removed setMaximumHeight to allow expansion

        # Enhanced modern styling
        self.normal_style = """
            QLabel {
                border: 3px dashed rgba(102, 126, 234, 0.4); 
                border-radius: 20px; 
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(255, 255, 255, 0.02), 
                    stop: 1 rgba(255, 255, 255, 0.05));
                color: #a78bfa; 
                padding: 30px; 
                font-size: 16px;
                font-weight: 500;
                backdrop-filter: blur(10px);
            }
            QLabel:hover { 
                border: 3px dashed rgba(167, 139, 250, 0.6); 
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(255, 255, 255, 0.05), 
                    stop: 1 rgba(255, 255, 255, 0.08));
                color: #c4b5fd;
            }
        """

        self.drag_style = """
            QLabel { 
                border: 3px dashed rgba(72, 187, 120, 0.8); 
                border-radius: 20px; 
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(72, 187, 120, 0.1), 
                    stop: 1 rgba(56, 161, 105, 0.15));
                color: #68d391; 
                font-size: 18px;
                font-weight: 600;
                backdrop-filter: blur(15px);
            }
        """

        self.image_style = """
            QLabel {
                border: 2px solid rgba(102, 126, 234, 0.3);
                border-radius: 15px;
                background: rgba(255, 255, 255, 0.02);
                padding: 10px;
            }
        """

        self.setStyleSheet(self.normal_style)

        # Track the current image
        self.dropped_image_path = None
        self.initial_pixmap = None
        self.has_image = False

        # Create and set initial pixmap
        self.create_initial_pixmap()
        self.show_initial_state()

    def create_initial_pixmap(self):
        """Create an enhanced initial placeholder pixmap"""
        self.initial_pixmap = QPixmap(400, 400)  # Larger initial size
        self.initial_pixmap.fill(Qt.transparent)

        painter = QPainter(self.initial_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create gradient background
        gradient_brush = QBrush(QColor(102, 126, 234, 30))
        painter.setBrush(gradient_brush)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, 400, 400, 20, 20)

        # Draw upload icon (simplified cloud with arrow)
        painter.setPen(QPen(QColor(167, 139, 250), 4))
        painter.setBrush(Qt.NoBrush)
        
        # Cloud shape (simplified)
        cloud_rect = QRect(150, 120, 100, 60)
        painter.drawEllipse(cloud_rect)
        painter.drawEllipse(cloud_rect.adjusted(-20, 10, -30, 20))
        painter.drawEllipse(cloud_rect.adjusted(30, 10, 20, 20))
        
        # Upload arrow
        painter.setPen(QPen(QColor(167, 139, 250), 6))
        painter.drawLine(200, 200, 200, 260)  # Vertical line
        painter.drawLine(185, 215, 200, 200)  # Left arrow part
        painter.drawLine(215, 215, 200, 200)  # Right arrow part

        # Enhanced text
        font_families = QFontDatabase().families()
        font_name = "Segoe UI"
        if "SF Pro Display" in font_families:
            font_name = "SF Pro Display"
        elif "Helvetica Neue" in font_families:
            font_name = "Helvetica Neue"
        elif not "Segoe UI" in font_families and font_families:
            font_name = font_families[0]

        # Main text
        painter.setPen(QColor(167, 139, 250))
        painter.setFont(QFont(font_name, 16, QFont.Bold))
        main_text_rect = QRect(50, 280, 300, 40)
        painter.drawText(main_text_rect, Qt.AlignCenter, "Drop Corneal Image Here")
        
        # Secondary text
        painter.setFont(QFont(font_name, 12))
        painter.setPen(QColor(203, 213, 224))
        sub_text_rect = QRect(50, 320, 300, 60)
        painter.drawText(sub_text_rect, Qt.AlignCenter | Qt.TextWordWrap, 
                        "Supports JPG, JPEG, PNG formats\nDrag & drop or click browse")
        
        painter.end()

    def show_initial_state(self):
        """Show the enhanced initial placeholder state"""
        self.dropped_image_path = None
        self.has_image = False
        self.setStyleSheet(self.normal_style)
        
        if self.initial_pixmap:
            scaled_pixmap = self.initial_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)

    def get_available_size(self):
        """Get available size for image display accounting for margins"""
        try:
            margins = self.getContentsMargins()
            available_width = self.width() - margins[0] - margins[2]
            available_height = self.height() - margins[1] - margins[3]
        except (TypeError, AttributeError):
            # Fallback if getContentsMargins() doesn't work as expected
            padding = 40  # Default padding
            available_width = self.width() - padding
            available_height = self.height() - padding
        
        return QSize(max(1, available_width), max(1, available_height))

    def show_dropped_image(self, image_path):
        """Display the dropped image with enhanced styling"""
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                self.dropped_image_path = image_path
                self.has_image = True
                
                # Apply image styling
                self.setStyleSheet(self.image_style)
                
                # Scale image to fit the available space while maintaining aspect ratio
                available_size = self.get_available_size()
                scaled_pixmap = pixmap.scaled(
                    available_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.setPixmap(scaled_pixmap)
                return True
        return False

    def resizeEvent(self, event):
        """Handle resize events with better image scaling"""
        super().resizeEvent(event)

        if self.has_image and self.dropped_image_path and os.path.exists(self.dropped_image_path):
            # Reload and scale the dropped image
            pixmap = QPixmap(self.dropped_image_path)
            if not pixmap.isNull():
                # Calculate available space for image (accounting for padding)
                available_size = self.get_available_size()
                scaled_pixmap = pixmap.scaled(
                    available_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
        """Handle drag enter events with enhanced visual feedback"""
        if (event.mimeData().hasUrls() and
                event.mimeData().urls()[0].toLocalFile().lower().endswith(('.jpg', '.jpeg', '.png'))):
            event.acceptProposedAction()
            self.setStyleSheet(self.drag_style)
            # Update text for drag state
            self.setText("🎯 Release to upload image")
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave events"""
        if self.has_image:
            self.setStyleSheet(self.image_style)
        else:
            self.setStyleSheet(self.normal_style)
            if self.initial_pixmap:
                scaled_pixmap = self.initial_pixmap.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.setPixmap(scaled_pixmap)

    def dropEvent(self, event):
        """Handle drop events with enhanced feedback"""
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

        # Reset styling based on current state
        if self.has_image:
            self.setStyleSheet(self.image_style)
        else:
            self.setStyleSheet(self.normal_style)
        
        self.setAlignment(Qt.AlignCenter)

    def clear_image(self):
        """Clear the current image with animation"""
        self.show_initial_state()

    def get_current_image_path(self):
        """Get the path of the currently displayed image"""
        return self.dropped_image_path

    def mousePressEvent(self, event):
        """Handle mouse clicks to trigger file browser"""
        if event.button() == Qt.LeftButton and not self.has_image:
            # Emit a signal that can be connected to the browse function
            # This provides an alternative way to upload images
            pass
        super().mousePressEvent(event)

    def enterEvent(self, event):
        """Handle mouse enter for hover effects"""
        if not self.has_image:
            self.setStyleSheet(self.normal_style.replace(
                "border: 3px dashed rgba(102, 126, 234, 0.4)", 
                "border: 3px dashed rgba(167, 139, 250, 0.6)"
            ))
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave"""
        if not self.has_image:
            self.setStyleSheet(self.normal_style)
        super().leaveEvent(event)