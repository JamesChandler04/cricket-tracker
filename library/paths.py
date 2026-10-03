"""Folder and file paths used by the programs, all anchored to the project folder."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
"""The project folder, which holds library/ and main/."""

# Results the programs write.
OUTPUT_DIR = ROOT / "output"
"""Folder that everything the programs write goes into."""
DELIVERY_DIR = OUTPUT_DIR / "demo_delivery"
"""Default save folder for one delivery's results."""
TOP_DOWN_TRACKING_DIR = OUTPUT_DIR / "top_down_tracking"
"""Folder for the images saved while detecting the ball in the top-down video."""
SIDE_ON_TRACKING_DIR = OUTPUT_DIR / "side_on_tracking"
"""Folder for the images saved while detecting the ball in the side-on video."""

# Settings and inputs, kept in the project folder.
CONFIG_PATH = ROOT / "config.yml"
"""The settings file, config.yml."""
CALIBRATION_PATH = ROOT / "camera_calibration.npz"
"""The side-on camera calibration."""
REFERENCE_CALIBRATION_PATH = ROOT / "reference_swingless_calibration.npz"
"""Calibration for the swingless reference delivery."""
REPORT_DATA_DIR = ROOT / "report_data"
"""Folder of data collected for the thesis report."""
