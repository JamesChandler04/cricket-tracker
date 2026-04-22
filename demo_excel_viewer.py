from cricket_ball_tracker import display_excel
import os

test_file = "test_data.xlsx"
if os.path.exists(test_file):
    print(f"Opening {test_file}")
    display_excel(test_file)
else:
    raise FileNotFoundError(f"Test file {test_file} not found.")
