import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget
import PyQt5

PyQt5.QApplic

file_path = "output_data/test_data.xlsx"

all_sheets = pd.read_excel(file_path, sheet_name=None)

print(all_sheets.keys())  # Print sheet names to verify they are read correctly

data = all_sheets["Data"]
params = all_sheets["Parameters"]
data_3d = all_sheets["3D Data"]
seam_angles = all_sheets["Seam Angles"]
main_calibration = all_sheets["Main Calibrations"]
side_calibration = all_sheets["Side Calibration"]



# Only needed for access to command line arguments
import sys

# You need one (and only one) QApplication instance per application.
# Pass in sys.argv to allow command line arguments for your app.
# If you know you won't use command line arguments QApplication([]) works too.
app = QApplication(sys.argv)

# Create a Qt widget, which will be our window.
window = QWidget()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec()


# Your application won't reach here until you exit and the event
# loop has stopped.