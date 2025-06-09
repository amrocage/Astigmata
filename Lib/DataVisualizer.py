import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QFrame, QScrollArea, QGroupBox, QGridLayout,
                             QApplication, QPushButton, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette


class CornealAnalysisWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        """Initialize the enhanced UI components"""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Enhanced title with status indicator
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 10)
        
        self.title_label = QLabel("📊 Analysis Results")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("""
            QLabel { 
                color: #e8e9f3; 
                padding: 15px 20px;
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(102, 126, 234, 0.1), stop: 1 rgba(118, 75, 162, 0.1));
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
        
        self.status_indicator = QLabel("⏳ Ready for analysis")
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: #a78bfa;
                font-size: 12px;
                font-weight: 500;
                padding: 5px 10px;
                background: rgba(167, 139, 250, 0.1);
                border-radius: 15px;
                border: 1px solid rgba(167, 139, 250, 0.3);
            }
        """)
        
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.status_indicator)

        # Scroll area for content
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.05);
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(167, 139, 250, 0.3);
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(167, 139, 250, 0.5);
            }
        """)

        # Content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(20)
        self.content_layout.setContentsMargins(10, 10, 10, 10)

        # Initialize group boxes
        self.init_group_boxes()

        # Set up scroll area
        self.scroll_area.setWidget(self.content_widget)

        # Add to main layout
        self.main_layout.addWidget(title_container)
        self.main_layout.addWidget(self.scroll_area)

        # Initially hide all content
        self.clear()

    def init_group_boxes(self):
        """Initialize all group boxes with enhanced styling"""
        
        # AI Classification Group
        self.ai_group = self.create_group_box("🤖 AI Classification", "AI_PRIMARY")
        self.ai_layout = QVBoxLayout(self.ai_group)
        
        self.ai_prediction_widget = self.create_info_card()
        self.ai_confidence_widget = self.create_info_card()
        self.ai_probabilities_widget = self.create_info_card()
        
        self.ai_layout.addWidget(self.ai_prediction_widget)
        self.ai_layout.addWidget(self.ai_confidence_widget)
        self.ai_layout.addWidget(self.ai_probabilities_widget)

        # Basic Measurements Group
        self.basic_group = self.create_group_box("📏 Basic Measurements", "MEASUREMENT")
        self.basic_layout = QVBoxLayout(self.basic_group)
        
        self.cylinder_power_widget = self.create_info_card()
        self.astigmatism_axis_widget = self.create_info_card()
        self.astigmatism_type_widget = self.create_info_card()
        
        self.basic_layout.addWidget(self.cylinder_power_widget)
        self.basic_layout.addWidget(self.astigmatism_axis_widget)
        self.basic_layout.addWidget(self.astigmatism_type_widget)

        # Regularity Group
        self.regularity_group = self.create_group_box("🔄 Corneal Regularity", "REGULARITY")
        self.regularity_layout = QVBoxLayout(self.regularity_group)
        
        self.regularity_type_widget = self.create_info_card()
        self.irregularity_measure_widget = self.create_info_card()
        
        self.regularity_layout.addWidget(self.regularity_type_widget)
        self.regularity_layout.addWidget(self.irregularity_measure_widget)

        # SimK Values Group
        self.simk_group = self.create_group_box("📊 SimK Values", "SIMK")
        self.simk_layout = QVBoxLayout(self.simk_group)
        
        self.avg_keratometry_widget = self.create_info_card()
        self.k1_central_widget = self.create_info_card()
        self.k2_central_widget = self.create_info_card()
        
        self.simk_layout.addWidget(self.avg_keratometry_widget)
        self.simk_layout.addWidget(self.k1_central_widget)
        self.simk_layout.addWidget(self.k2_central_widget)

        # Advanced Analysis Groups
        self.irregularity_group = self.create_group_box("📈 Irregularity Indices", "ADVANCED")
        self.irregularity_layout = QVBoxLayout(self.irregularity_group)
        self.irregularity_indices_widget = self.create_info_card()
        self.irregularity_layout.addWidget(self.irregularity_indices_widget)

        self.cort_group = self.create_group_box("🎯 Corneal Topographic Astigmatism", "ADVANCED")
        self.cort_layout = QVBoxLayout(self.cort_group)
        self.cort_values_widget = self.create_info_card()
        self.cort_layout.addWidget(self.cort_values_widget)

        self.posterior_group = self.create_group_box("🔬 Posterior Corneal Analysis", "ADVANCED")
        self.posterior_layout = QVBoxLayout(self.posterior_group)
        self.posterior_values_widget = self.create_info_card()
        self.posterior_layout.addWidget(self.posterior_values_widget)

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

    def create_group_box(self, title, category):
        """Create a styled group box based on category"""
        group = QGroupBox(title)
        
        category_colors = {
            "AI_PRIMARY": ("102, 126, 234", "118, 75, 162"),
            "MEASUREMENT": ("72, 187, 120", "56, 161, 105"),
            "REGULARITY": ("237, 137, 54", "221, 107, 32"),
            "SIMK": ("139, 92, 246", "124, 58, 237"),
            "ADVANCED": ("245, 101, 101", "229, 62, 62")
        }
        
        primary_color, secondary_color = category_colors.get(category, ("102, 126, 234", "118, 75, 162"))
        
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: 600;
                font-size: 14px;
                color: #e8e9f3;
                border: 2px solid rgba({primary_color}, 0.3);
                border-radius: 15px;
                margin-top: 15px;
                padding-top: 20px;
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba({primary_color}, 0.05), 
                    stop: 1 rgba({secondary_color}, 0.08));
                backdrop-filter: blur(10px);
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                top: 5px;
                padding: 5px 15px;
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba({primary_color}, 0.8), 
                    stop: 1 rgba({secondary_color}, 0.9));
                border-radius: 12px;
                color: white;
                font-weight: 700;
            }}
        """)
        
        return group

    def create_info_card(self):
        """Create a styled information card"""
        card = QLabel()
        card.setWordWrap(True)
        card.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        card.setStyleSheet("""
            QLabel {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
                color: #e8e9f3;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        return card

    def display(self, data):
        """Display the corneal analysis data with enhanced formatting"""
        try:
            # Update status indicator
            self.status_indicator.setText("✅ Analysis Complete")
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: #68d391;
                    font-size: 12px;
                    font-weight: 500;
                    padding: 5px 10px;
                    background: rgba(72, 187, 120, 0.1);
                    border-radius: 15px;
                    border: 1px solid rgba(72, 187, 120, 0.3);
                }
            """)

            # Show all group boxes
            for group in [self.ai_group, self.basic_group, self.regularity_group,
                          self.simk_group, self.irregularity_group, self.cort_group,
                          self.posterior_group]:
                group.show()

            # AI Classification
            ai_data = data.get("ai_classification", {})
            
            prediction = ai_data.get('prediction', 'N/A')
            confidence = ai_data.get('confidence', 'N/A')
            
            self.ai_prediction_widget.setText(f"""
                <div style='margin-bottom: 8px;'>
                    <span style='color: #a78bfa; font-weight: 600;'>Prediction:</span><br>
                    <span style='font-size: 16px; font-weight: 700; color: #68d391;'>{prediction}</span>
                </div>
            """)
            
            if isinstance(confidence, (int, float)):
                confidence_color = "#68d391" if confidence > 0.8 else "#fbbf24" if confidence > 0.6 else "#f87171"
                self.ai_confidence_widget.setText(f"""
                    <div style='margin-bottom: 8px;'>
                        <span style='color: #a78bfa; font-weight: 600;'>Confidence Score:</span><br>
                        <span style='font-size: 16px; font-weight: 700; color: {confidence_color};'>{confidence:.1%}</span>
                    </div>
                """)
            else:
                self.ai_confidence_widget.setText(f"""
                    <div>
                        <span style='color: #a78bfa; font-weight: 600;'>Confidence:</span><br>
                        <span style='color: #cbd5e0;'>{confidence}</span>
                    </div>
                """)

            # Format probabilities
            probs = ai_data.get('all_probabilities', {})
            if probs:
                prob_html = "<span style='color: #a78bfa; font-weight: 600;'>All Probabilities:</span><br><br>"
                for class_name, prob in probs.items():
                    bar_width = int(prob * 100)
                    prob_html += f"""
                        <div style='margin-bottom: 8px;'>
                            <span style='color: #e8e9f3;'>{class_name}:</span> 
                            <span style='color: #68d391; font-weight: 600;'>{prob:.1%}</span><br>
                            <div style='background: rgba(255,255,255,0.1); border-radius: 3px; height: 6px; margin-top: 2px;'>
                                <div style='background: linear-gradient(90deg, #667eea, #764ba2); width: {bar_width}%; height: 100%; border-radius: 3px;'></div>
                            </div>
                        </div>
                    """
                self.ai_probabilities_widget.setText(prob_html)
            else:
                self.ai_probabilities_widget.setText("<span style='color: #a78bfa; font-weight: 600;'>Probabilities:</span><br><span style='color: #cbd5e0;'>N/A</span>")

            # Basic Measurements with better formatting
            self.format_measurement_card(self.cylinder_power_widget, "Cylinder Power", 
                                       data.get('cylinder_power', 'N/A'), "D")
            self.format_measurement_card(self.astigmatism_axis_widget, "Astigmatism Axis", 
                                       data.get('astigmatism_axis', 'N/A'), "°")
            self.format_measurement_card(self.astigmatism_type_widget, "Astigmatism Type", 
                                       data.get('astigmatism_type', 'N/A'))

            # Regularity
            regularity_data = data.get("regularity", {})
            self.format_measurement_card(self.regularity_type_widget, "Regularity Type", 
                                       regularity_data.get('type', 'N/A'))
            
            irreg_measure = regularity_data.get('irregularity_measure', 'N/A')
            if isinstance(irreg_measure, (int, float)):
                self.format_measurement_card(self.irregularity_measure_widget, "Irregularity Measure", 
                                           f"{irreg_measure:.4f}")
            else:
                self.format_measurement_card(self.irregularity_measure_widget, "Irregularity Measure", 
                                           str(irreg_measure))

            # SimK Values
            simk_data = data.get("simk_values", {})
            self.format_measurement_card(self.avg_keratometry_widget, "Average Keratometry", 
                                       simk_data.get('average_keratometry', 'N/A'), "D")
            self.format_measurement_card(self.k1_central_widget, "K1 Central", 
                                       simk_data.get('k1_central', 'N/A'), "D")
            self.format_measurement_card(self.k2_central_widget, "K2 Central", 
                                       simk_data.get('k2_central', 'N/A'), "D")

            # Advanced measurements
            self.format_advanced_data(self.irregularity_indices_widget, "Irregularity Indices", 
                                    data.get("irregularity_indices", {}))
            self.format_advanced_data(self.cort_values_widget, "Corneal Topographic Astigmatism", 
                                    data.get("corneal_topographic_astigmatism", {}))
            self.format_advanced_data(self.posterior_values_widget, "Posterior Corneal Analysis", 
                                    data.get("posterior_corneal_analysis", {}))

        except Exception as e:
            # Handle any display errors gracefully
            self.status_indicator.setText("❌ Display Error")
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: #f87171;
                    font-size: 12px;
                    font-weight: 500;
                    padding: 5px 10px;
                    background: rgba(248, 113, 113, 0.1);
                    border-radius: 15px;
                    border: 1px solid rgba(248, 113, 113, 0.3);
                }
            """)
            
            error_widget = self.create_info_card()
            error_widget.setText(f"""
                <span style='color: #f87171; font-weight: 600;'>Error displaying data:</span><br>
                <span style='color: #cbd5e0;'>{str(e)}</span>
            """)
            self.content_layout.addWidget(error_widget)

    def format_measurement_card(self, widget, label, value, unit=""):
        """Format a measurement card with consistent styling"""
        if isinstance(value, (int, float)):
            if unit:
                formatted_value = f"{value:.2f} {unit}"
            else:
                formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)
            
        widget.setText(f"""
            <div>
                <span style='color: #a78bfa; font-weight: 600; font-size: 12px;'>{label.upper()}</span><br>
                <span style='color: #e8e9f3; font-size: 16px; font-weight: 700;'>{formatted_value}</span>
            </div>
        """)

    def format_advanced_data(self, widget, title, data):
        """Format advanced data with structured display"""
        if not data:
            widget.setText(f"""
                <span style='color: #a78bfa; font-weight: 600;'>{title}:</span><br>
                <span style='color: #cbd5e0;'>No data available</span>
            """)
            return
            
        html = f"<span style='color: #a78bfa; font-weight: 600; font-size: 12px;'>{title.upper()}</span><br><br>"
        
        for key, value in data.items():
            display_key = key.replace('_', ' ').title()
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)
                
            html += f"""
                <div style='margin-bottom: 6px; display: flex; justify-content: space-between;'>
                    <span style='color: #cbd5e0;'>{display_key}:</span>
                    <span style='color: #e8e9f3; font-weight: 600;'>{formatted_value}</span>
                </div>
            """
            
        widget.setText(html)

    def clear(self):
        """Clear all displayed data and reset to initial state"""
        # Update status
        self.status_indicator.setText("⏳ Ready for analysis")
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: #a78bfa;
                font-size: 12px;
                font-weight: 500;
                padding: 5px 10px;
                background: rgba(167, 139, 250, 0.1);
                border-radius: 15px;
                border: 1px solid rgba(167, 139, 250, 0.3);
            }
        """)
        
        # Hide all group boxes
        for group in [self.ai_group, self.basic_group, self.regularity_group,
                      self.simk_group, self.irregularity_group, self.cort_group,
                      self.posterior_group]:
            group.hide()

        # Clear all widgets
        all_widgets = [
            self.ai_prediction_widget, self.ai_confidence_widget, self.ai_probabilities_widget,
            self.cylinder_power_widget, self.astigmatism_axis_widget, self.astigmatism_type_widget,
            self.regularity_type_widget, self.irregularity_measure_widget,
            self.avg_keratometry_widget, self.k1_central_widget, self.k2_central_widget,
            self.irregularity_indices_widget, self.cort_values_widget, self.posterior_values_widget
        ]

        for widget in all_widgets:
            widget.clear()


# Example usage and test
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create main window for testing
    main_widget = QWidget()
    main_widget.setStyleSheet("""
        QWidget { 
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 #0f0f23, stop: 1 #1a1a2e);
        }
    """)
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
            "SAI": 23.4,
            "SRI": 0.15,
            "Vertical_Asymmetry": 1.05,
            "Horizontal_Asymmetry": 1.02
        },
        "corneal_topographic_astigmatism": {
            "CorT_Magnitude": 2.5,
            "J0": 1.2,
            "J45": -0.8,
            "Axis": 90
        },
        "posterior_corneal_analysis": {
            "Posterior_Cylinder": 0.25,
            "Posterior_Axis": 180,
            "Note": "Estimated - requires actual posterior elevation data"
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

    main_widget.setWindowTitle("Enhanced Corneal Analysis Widget")
    main_widget.resize(800, 1000)
    main_widget.show()

    sys.exit(app.exec_())