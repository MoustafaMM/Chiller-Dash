import sys
import os
import matplotlib.pyplot as plt
from functools import partial
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QListWidget, QFrame, QLabel, QScrollArea,
    QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class HVAC_GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window settings
        self.setWindowTitle("HVAC Data Processing")
        self.setGeometry(100, 100, 1000, 600)

        # Set Light Theme (Baby Blue)
        self.set_light_theme()

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QHBoxLayout()

        # Sidebar Panel (Fixed at 300px width) - Now 25% of Window
        self.sidebar = self.create_sidebar()
        self.sidebar.setFixedWidth(300)

        # Graph Area (75% of window) with Title Label
        self.graph_widget = GraphWidget()

        # Main Layout: Sidebar + Graph
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.graph_widget)

        central_widget.setLayout(main_layout)

        # Track the currently selected button
        self.selected_button = None

    def create_sidebar(self):
        """ Create a sidebar with optimized button & text spacing """
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(5)

        # File Upload Section
        self.file_button = QPushButton("Load Data File")
        self.file_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.file_button.clicked.connect(self.load_file)

        self.file_list = QListWidget()
        self.file_list.setFixedHeight(60)

        sidebar_layout.addWidget(self.file_button)
        sidebar_layout.addWidget(self.file_list)

        # Processing Sections
        self.sections = {
            "Data Pre-processing": [``
                "Voltage (V)", "Current (I)", "Power (P)",
                "Temp (T)", "Vibration", "Frequency", "Flow Rate"
            ],
            "Features Extraction": ["Time (T)", "Frequency (f)", "T-f"],
            "Features Selection": ["AI", "Hybrid"],
            "Forecasting": [
                "Linear Regression", "Neural Networks", "Decision Tree"
            ],
            "Fault Diagnosis": ["Anomaly Detection", "Predictive Maintenance"]
        }

        # Scroll Area for Sidebar
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.buttons = {}

        for section, options in self.sections.items():
            frame = QFrame()
            frame_layout = QVBoxLayout()
            frame_layout.setSpacing(3)

            label = QLabel(section)
            label.setFont(QFont("Times New Roman", 25, QFont.Bold))
            frame_layout.addWidget(label)

            for option in options:
                btn = QPushButton(option)
                btn.setFont(QFont("Times New Roman", 20))
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                btn.setMinimumHeight(35)
                btn.setStyleSheet(self.default_button_style())

                # 🔹 Fix: Use functools.partial to correctly bind variables
                btn.clicked.connect(partial(self.sidebar_button_clicked, btn,
                                            option, section))

                self.buttons[option] = btn
                frame_layout.addWidget(btn)

            frame.setLayout(frame_layout)
            scroll_layout.addWidget(frame)

        scroll_area.setWidget(scroll_widget)
        sidebar_layout.addWidget(scroll_area)

        sidebar_widget.setLayout(sidebar_layout)
        return sidebar_widget

    def sidebar_button_clicked(self, button, option, section):
        """ Handle Sidebar Button Click - Toggle Selection & Update Graph Title """

        print(f"Button Clicked: {section} - {option}")  # Debugging

        # If the same button is clicked again, reset everything
        if self.selected_button == button:
            self.reset_buttons()
            self.graph_widget.update_title("Select an option to display")
            self.selected_button = None
            button.style().unpolish(button)  # 🔹 Force UI Refresh
            button.style().polish(button)  # 🔹 Refresh after resetting
            return

        # Reset all buttons before selecting a new one
        self.reset_buttons()

        # Update Graph Title
        self.graph_widget.update_title(f"{section} - {option}")

        # Highlight Selected Button
        button.setStyleSheet(self.selected_button_style())
        self.selected_button = button

        button.style().unpolish(button)  # 🔹 Force UI Refresh
        button.style().polish(button)  # 🔹 Refresh after applying style

    def reset_buttons(self):
        """ Reset all sidebar buttons to default style """
        for btn in self.buttons.values():
            btn.setStyleSheet(self.default_button_style())

    def default_button_style(self):
        """ Returns the default button style (Unselected) """
        return (
            "background-color: #F0F0F0; color: black;"
            " text-align: left; padding: 8px; font-size: 14px;"
            " border-radius: 5px;"
        )

    def selected_button_style(self):
        """ Returns the selected button style (Clicked) """
        return (
            "background-color: #B0BEC5; color: white;"
            " text-align: left; padding: 8px; font-size: 14px;"
            " border-radius: 5px;"
        )

    def set_light_theme(self):
        """ Set Light Mode (Baby Blue Theme) """
        self.setStyleSheet(
            "QMainWindow { background-color: #B3E5FC; }"
            " QPushButton { font-size: 14px; padding: 10px; border-radius: 5px;"
            " background-color: #F0F0F0; color: black; }"
            " QPushButton:hover { background-color: #D6D6D6; }"
            " QLabel { font-size: 16px; font-weight: bold; color: black;"
            " padding: 5px; }"
            " QFrame { background-color: white; border-radius: 8px; padding: 8px; }"
        )

    def load_file(self):
        """ Open file dialog to select a data file """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx);;"
            "All Files (*)"
        )

        if file_path:
            file_name = os.path.basename(file_path)
            self.file_list.clear()
            self.file_list.addItem(file_name)
            self.graph_widget.plot_data_from_file(file_path)


class GraphWidget(QWidget):
    """ Widget for displaying Matplotlib graph with a title """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title Label
        self.title_label = QLabel("Select an option to display")
        self.title_label.setFont(QFont("Times New Roman", 25, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Matplotlib Figure (Takes 95% of Space)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=95)

        self.ax.set_facecolor('white')  
        self.canvas.draw()

    def update_title(self, title):
        """ Update the label above the graph """
        self.title_label.setText(title)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = HVAC_GUI()
    main_window.show()
    sys.exit(app.exec_())
