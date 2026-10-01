from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Results the programs write.
OUTPUT_DIR = ROOT / "output"
DELIVERY_DIR = OUTPUT_DIR / "demo_delivery"
TOP_DOWN_TRACKING_DIR = OUTPUT_DIR / "top_down_tracking"
SIDE_ON_TRACKING_DIR = OUTPUT_DIR / "side_on_tracking"

# Settings and inputs, kept in the project folder.
CONFIG_PATH = ROOT / "config.yml"
CALIBRATION_PATH = ROOT / "camera_calibration.npz"
REFERENCE_CALIBRATION_PATH = ROOT / "reference_swingless_calibration.npz"
REPORT_DATA_DIR = ROOT / "report_data"
