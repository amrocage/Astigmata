# SYSTEM
import sys


# GUI
from PyQt5.QtWidgets import QApplication

REPORTLAB_AVAILABLE = True

# INTERNAL LIBRARIES
from Lib.MainWindow import MainWindow


if __name__ == "__main__":
    app = QApplication.instance(); 
    if not app: app = QApplication(sys.argv)
    window = MainWindow(); window.show()
    sys.exit(app.exec_())
