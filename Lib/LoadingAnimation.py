from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QLinearGradient, QFont
import math


class LoadingAnimation(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 80)
        
        # Animation properties
        self.angle = 0
        self.opacity = 1.0
        self.scale_factor = 1.0
        
        # Timer for rotation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        
        # Animation for pulsing effect
        self.pulse_animation = QPropertyAnimation(self, b"scale_factor")
        self.pulse_animation.setDuration(1000)
        self.pulse_animation.setStartValue(0.8)
        self.pulse_animation.setEndValue(1.2)
        self.pulse_animation.setEasingCurve(QEasingCurve.InOutSine)
        self.pulse_animation.setLoopCount(-1)  # Infinite loop
        
        # Clean up when destroyed
        self.destroyed.connect(self.stop_animation)

    @pyqtProperty(float)
    def scale_factor(self):
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value):
        self._scale_factor = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set up the coordinate system
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(self.scale_factor, self.scale_factor)
        
        # Create gradient colors
        primary_color = QColor(102, 126, 234)
        secondary_color = QColor(167, 139, 250)
        
        # Draw modern spinner with gradient effect
        for i in range(12):
            # Calculate opacity based on position and current angle
            angle_offset = (self.angle + i * 30) % 360
            opacity = max(0.1, 1.0 - (angle_offset / 360.0))
            
            # Create gradient for each line
            gradient = QLinearGradient(0, 0, 0, 25)
            
            # Interpolate between colors based on position
            color_factor = i / 12.0
            r = int(primary_color.red() * (1 - color_factor) + secondary_color.red() * color_factor)
            g = int(primary_color.green() * (1 - color_factor) + secondary_color.green() * color_factor)
            b = int(primary_color.blue() * (1 - color_factor) + secondary_color.blue() * color_factor)
            
            line_color = QColor(r, g, b, int(255 * opacity))
            
            gradient.setColorAt(0, line_color)
            gradient.setColorAt(1, QColor(r, g, b, int(255 * opacity * 0.3)))
            
            # Set up pen with gradient
            pen = QPen(QBrush(gradient), 3, Qt.RoundCap)
            painter.setPen(pen)
            
            # Draw the line
            painter.drawLine(0, 8, 0, 25)
            
            # Rotate for next line
            painter.rotate(30)

    def rotate(self):
        """Update rotation angle"""
        self.angle = (self.angle + 12) % 360
        self.update()

    def start_animation(self):
        """Start the loading animation"""
        self.timer.start(50)  # 50ms for smooth animation
        self.pulse_animation.start()

    def stop_animation(self):
        """Stop the loading animation"""
        self.timer.stop()
        self.pulse_animation.stop()

    def showEvent(self, event):
        """Start animation when widget becomes visible"""
        self.start_animation()
        super().showEvent(event)

    def hideEvent(self, event):
        """Stop animation when widget is hidden"""
        self.stop_animation()
        super().hideEvent(event)


class LoadingOverlay(QWidget):
    """A more comprehensive loading overlay with text"""
    
    def __init__(self, parent=None, message="Analyzing..."):
        super().__init__(parent)
        self.message = message
        self.setupUI()
        
    def setupUI(self):
        """Set up the loading overlay UI"""
        self.setStyleSheet("""
            QWidget {
                background: rgba(15, 15, 35, 0.9);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)
        
        # Loading animation
        self.loading_animation = LoadingAnimation()
        layout.addWidget(self.loading_animation, 0, Qt.AlignCenter)
        
        # Loading text
        self.loading_label = QLabel(self.message)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("""
            QLabel {
                color: #e8e9f3;
                font-size: 16px;
                font-weight: 600;
                background: transparent;
            }
        """)
        layout.addWidget(self.loading_label)
        
        # Status text
        self.status_label = QLabel("Please wait...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #a78bfa;
                font-size: 12px;
                background: transparent;
            }
        """)
        layout.addWidget(self.status_label)
        
    def update_message(self, message):
        """Update the loading message"""
        self.loading_label.setText(message)
        
    def update_status(self, status):
        """Update the status text"""
        self.status_label.setText(status)
        
    def show_overlay(self):
        """Show the overlay and start animation"""
        self.show()
        self.loading_animation.start_animation()
        
    def hide_overlay(self):
        """Hide the overlay and stop animation"""
        self.loading_animation.stop_animation()
        self.hide()


# Backwards compatibility with original LoadingAnimation interface
class EnhancedLoadingAnimation(LoadingAnimation):
    """Enhanced version that maintains compatibility with original interface"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Keep original size for compatibility
        self.setFixedSize(60, 60)


# Export the enhanced version as the default
LoadingAnimation = EnhancedLoadingAnimation