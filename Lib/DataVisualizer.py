import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QFrame, QScrollArea, QGroupBox, QGridLayout,
                             QApplication, QPushButton)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette


class CornealAnalysisWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        """Initialize the UI components"""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        self.title_label = QLabel("Corneal Analysis Results")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("QLabel { color: #2c3e50; margin: 10px; }")

        # Scroll area for content
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        # Content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(15)

        # Initialize group boxes
        self.init_group_boxes()

        # Set up scroll area
        self.scroll_area.setWidget(self.content_widget)

        # Add to main layout
        self.main_layout.addWidget(self.title_label)
        self.main_layout.addWidget(self.scroll_area)

        # Initially hide all content
        self.clear()

    def init_group_boxes(self):
        """Initialize all group boxes for different data sections"""
        # AI Classification Group
        self.ai_group = QGroupBox("AI Classification")
        self.ai_layout = QGridLayout(self.ai_group)
        self.ai_prediction_label = QLabel()
        self.ai_confidence_label = QLabel()
        self.ai_probabilities_label = QLabel()

        # Basic Measurements Group
        self.basic_group = QGroupBox("Basic Measurements")
        self.basic_layout = QGridLayout(self.basic_group)
        self.cylinder_power_label = QLabel()
        self.astigmatism_axis_label = QLabel()
        self.astigmatism_type_label = QLabel()

        # Regularity Group
        self.regularity_group = QGroupBox("Corneal Regularity")
        self.regularity_layout = QGridLayout(self.regularity_group)
        self.regularity_type_label = QLabel()
        self.irregularity_measure_label = QLabel()

        # SimK Values Group
        self.simk_group = QGroupBox("SimK Values")
        self.simk_layout = QGridLayout(self.simk_group)
        self.avg_keratometry_label = QLabel()
        self.k1_central_label = QLabel()
        self.k2_central_label = QLabel()

        # Irregularity Indices Group
        self.irregularity_group = QGroupBox("Irregularity Indices")
        self.irregularity_layout = QVBoxLayout(self.irregularity_group)
        self.irregularity_indices_label = QLabel()

        # Corneal Topographic Astigmatism Group
        self.cort_group = QGroupBox("Corneal Topographic Astigmatism")
        self.cort_layout = QVBoxLayout(self.cort_group)
        self.cort_values_label = QLabel()

        # Posterior Corneal Analysis Group
        self.posterior_group = QGroupBox("Posterior Corneal Analysis")
        self.posterior_layout = QVBoxLayout(self.posterior_group)
        self.posterior_values_label = QLabel()

        # Style the group boxes
        self.style_group_boxes()

        # Add to content layout
        self.content_layout.addWidget(self.ai_group)
        self.content_layout.addWidget(self.basic_group)
        self.content_layout.addWidget(self.regularity_group)
        self.content_layout.addWidget(self.simk_group)
        self.content_layout.addWidget(self.irregularity_group)
        self.content_layout.addWidget(self.cort_group)
        self.content_layout.addWidget(self.posterior_group)

        # Add stretch to push content to top
        self.content_layout.addStretch()

    def style_group_boxes(self):
        """Apply styling to group boxes and labels"""
        group_style = """
        QGroupBox {
            font-weight: bold;
            font-size: 12px;
            border: 2px solid #bdc3c7;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            color: #2c3e50;
        }
        """

        label_style = """
        QLabel {
            font-size: 11px;
            padding: 5px;
            background-color: #2c3e50;
            color: white;
            border: 1px solid #34495e;
            border-radius: 4px;
            margin: 2px;
        }
        """

        for group in [self.ai_group, self.basic_group, self.regularity_group,
                      self.simk_group, self.irregularity_group, self.cort_group,
                      self.posterior_group]:
            group.setStyleSheet(group_style)

        # Apply label styling
        all_labels = [
            self.ai_prediction_label, self.ai_confidence_label, self.ai_probabilities_label,
            self.cylinder_power_label, self.astigmatism_axis_label, self.astigmatism_type_label,
            self.regularity_type_label, self.irregularity_measure_label,
            self.avg_keratometry_label, self.k1_central_label, self.k2_central_label,
            self.irregularity_indices_label, self.cort_values_label, self.posterior_values_label
        ]

        for label in all_labels:
            label.setStyleSheet(label_style)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def display(self, data):
        """Display the corneal analysis data"""
        try:
            # Show all group boxes
            for group in [self.ai_group, self.basic_group, self.regularity_group,
                          self.simk_group, self.irregularity_group, self.cort_group,
                          self.posterior_group]:
                group.show()

            # AI Classification
            ai_data = data.get("ai_classification", {})
            self.ai_prediction_label.setText(f"<b>Prediction:</b> {ai_data.get('prediction', 'N/A')}")
            self.ai_confidence_label.setText(
                f"<b>Confidence:</b> {ai_data.get('confidence', 'N/A'):.4f}" if isinstance(ai_data.get('confidence'),
                                                                                           (int,
                                                                                            float)) else f"<b>Confidence:</b> {ai_data.get('confidence', 'N/A')}")

            # Format probabilities
            probs = ai_data.get('all_probabilities', {})
            if probs:
                prob_text = "<b>All Probabilities:</b><br>"
                for class_name, prob in probs.items():
                    prob_text += f"• {class_name}: {prob:.4f}<br>"
                self.ai_probabilities_label.setText(prob_text)
            else:
                self.ai_probabilities_label.setText("<b>All Probabilities:</b> N/A")

            # Layout AI group
            self.ai_layout.addWidget(QLabel("Prediction:"), 0, 0)
            self.ai_layout.addWidget(self.ai_prediction_label, 0, 1)
            self.ai_layout.addWidget(QLabel("Confidence:"), 1, 0)
            self.ai_layout.addWidget(self.ai_confidence_label, 1, 1)
            self.ai_layout.addWidget(self.ai_probabilities_label, 2, 0, 1, 2)

            # Basic Measurements
            self.cylinder_power_label.setText(f"<b>Value:</b> {data.get('cylinder_power', 'N/A')}")
            self.astigmatism_axis_label.setText(f"<b>Axis:</b> {data.get('astigmatism_axis', 'N/A')}")
            self.astigmatism_type_label.setText(f"<b>Type:</b> {data.get('astigmatism_type', 'N/A')}")

            self.basic_layout.addWidget(QLabel("Cylinder Power:"), 0, 0)
            self.basic_layout.addWidget(self.cylinder_power_label, 0, 1)
            self.basic_layout.addWidget(QLabel("Astigmatism Axis:"), 1, 0)
            self.basic_layout.addWidget(self.astigmatism_axis_label, 1, 1)
            self.basic_layout.addWidget(QLabel("Astigmatism Type:"), 2, 0)
            self.basic_layout.addWidget(self.astigmatism_type_label, 2, 1)

            # Regularity
            regularity_data = data.get("regularity", {})
            self.regularity_type_label.setText(f"<b>Type:</b> {regularity_data.get('type', 'N/A')}")
            irreg_measure = regularity_data.get('irregularity_measure', 'N/A')
            if isinstance(irreg_measure, (int, float)):
                self.irregularity_measure_label.setText(f"<b>Measure:</b> {irreg_measure:.4f}")
            else:
                self.irregularity_measure_label.setText(f"<b>Measure:</b> {irreg_measure}")

            self.regularity_layout.addWidget(QLabel("Regularity Type:"), 0, 0)
            self.regularity_layout.addWidget(self.regularity_type_label, 0, 1)
            self.regularity_layout.addWidget(QLabel("Irregularity Measure:"), 1, 0)
            self.regularity_layout.addWidget(self.irregularity_measure_label, 1, 1)

            # SimK Values
            simk_data = data.get("simk_values", {})
            self.avg_keratometry_label.setText(f"<b>Value:</b> {simk_data.get('average_keratometry', 'N/A')}")
            self.k1_central_label.setText(f"<b>K1:</b> {simk_data.get('k1_central', 'N/A')}")
            self.k2_central_label.setText(f"<b>K2:</b> {simk_data.get('k2_central', 'N/A')}")

            self.simk_layout.addWidget(QLabel("Average Keratometry:"), 0, 0)
            self.simk_layout.addWidget(self.avg_keratometry_label, 0, 1)
            self.simk_layout.addWidget(QLabel("K1 Central:"), 1, 0)
            self.simk_layout.addWidget(self.k1_central_label, 1, 1)
            self.simk_layout.addWidget(QLabel("K2 Central:"), 2, 0)
            self.simk_layout.addWidget(self.k2_central_label, 2, 1)

            # Irregularity Indices
            irreg_indices = data.get("irregularity_indices", {})
            if irreg_indices:
                indices_text = "<b>Irregularity Indices:</b><br>"
                for key, value in irreg_indices.items():
                    indices_text += f"• {key}: {value}<br>"
                self.irregularity_indices_label.setText(indices_text)
            else:
                self.irregularity_indices_label.setText("<b>Irregularity Indices:</b> N/A")

            self.irregularity_layout.addWidget(self.irregularity_indices_label)

            # Corneal Topographic Astigmatism
            cort_data = data.get("corneal_topographic_astigmatism", {})
            if cort_data:
                cort_text = "<b>Corneal Topographic Astigmatism:</b><br>"
                for key, value in cort_data.items():
                    cort_text += f"• {key}: {value}<br>"
                self.cort_values_label.setText(cort_text)
            else:
                self.cort_values_label.setText("<b>Corneal Topographic Astigmatism:</b> N/A")

            self.cort_layout.addWidget(self.cort_values_label)

            # Posterior Corneal Analysis
            posterior_data = data.get("posterior_corneal_analysis", {})
            if posterior_data:
                posterior_text = "<b>Posterior Corneal Analysis:</b><br>"
                for key, value in posterior_data.items():
                    posterior_text += f"• {key}: {value}<br>"
                self.posterior_values_label.setText(posterior_text)
            else:
                self.posterior_values_label.setText("<b>Posterior Corneal Analysis:</b> N/A")

            self.posterior_layout.addWidget(self.posterior_values_label)

        except Exception as e:
            # Handle any display errors gracefully
            error_label = QLabel(f"Error displaying data: {str(e)}")
            error_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            self.content_layout.addWidget(error_label)

    def clear(self):
        """Clear all displayed data and hide the content"""
        # Hide all group boxes
        for group in [self.ai_group, self.basic_group, self.regularity_group,
                      self.simk_group, self.irregularity_group, self.cort_group,
                      self.posterior_group]:
            group.hide()

        # Clear all labels
        all_labels = [
            self.ai_prediction_label, self.ai_confidence_label, self.ai_probabilities_label,
            self.cylinder_power_label, self.astigmatism_axis_label, self.astigmatism_type_label,
            self.regularity_type_label, self.irregularity_measure_label,
            self.avg_keratometry_label, self.k1_central_label, self.k2_central_label,
            self.irregularity_indices_label, self.cort_values_label, self.posterior_values_label
        ]

        for label in all_labels:
            label.clear()


# Example usage and test
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create main window for testing
    main_widget = QWidget()
    main_layout = QVBoxLayout(main_widget)

    # Create the corneal analysis widget
    corneal_widget = CornealAnalysisWidget()

    # Create test buttons
    button_layout = QHBoxLayout()
    display_button = QPushButton("Display Test Data")
    clear_button = QPushButton("Clear Data")

    # Sample test data
    test_data = {
        "ai_classification": {
            "prediction": "Normal",
            "confidence": 0.8765,
            "all_probabilities": {
                "Normal": 0.8765,
                "Keratoconus": 0.1235,
                "Astigmatism": 0.0000
            }
        },
        "cylinder_power": -2.5,
        "astigmatism_axis": 90,
        "astigmatism_type": "With-the-rule",
        "regularity": {
            "type": "Regular",
            "irregularity_measure": 0.234
        },
        "simk_values": {
            "average_keratometry": 43.5,
            "k1_central": 42.0,
            "k2_central": 45.0
        },
        "irregularity_indices": {
            "ISV": 23,
            "IVA": 0.15,
            "KI": 1.05,
            "CKI": 1.02
        },
        "corneal_topographic_astigmatism": {
            "magnitude": 2.5,
            "axis": 90
        },
        "posterior_corneal_analysis": {
            "elevation": 5.2,
            "curvature": 6.1
        }
    }


    def display_test():
        corneal_widget.display(test_data)


    def clear_test():
        corneal_widget.clear()


    display_button.clicked.connect(display_test)
    clear_button.clicked.connect(clear_test)

    button_layout.addWidget(display_button)
    button_layout.addWidget(clear_button)

    main_layout.addLayout(button_layout)
    main_layout.addWidget(corneal_widget)

    main_widget.setWindowTitle("Corneal Analysis Widget Test")
    main_widget.resize(600, 800)
    main_widget.show()

    sys.exit(app.exec_())