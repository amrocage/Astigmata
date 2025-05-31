from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import  QPainter, QPen

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