"""Tests for library/calculators.py."""

import pytest
import math
from library.calculators import Calculators

@pytest.fixture
def calc():
    """Return a new Calculators for each test."""
    return Calculators()

class TestCalculateMetersPerPixel:
    """Tests for Calculators._calculate_meters_per_pixel."""
    def test_calculate_meters_per_pixel_normal(self, calc):
        """Check that a valid calibration returns the calibrations and an m/px value."""
        calibrations = [(1, [(10, 20), (30, 40)])]
        ball_diameter_m = 0.1
        meters_per_pixel = None
        result = calc._calculate_meters_per_pixel(calibrations, ball_diameter_m, meters_per_pixel)
        assert result is not None
        assert len(result[0]) == 1
        assert isinstance(result[1], float)

    def test_calculate_meters_per_pixel_too_close(self, calc):
        """Check that calibration points under 1 px apart give None."""
        calibrations = [(1, [(10, 20), (10.5, 20.5)])]
        ball_diameter_m = 0.1
        meters_per_pixel = None
        result = calc._calculate_meters_per_pixel(calibrations, ball_diameter_m, meters_per_pixel)
        assert result is None

    def test_calculate_meters_per_pixel_empty(self, calc):
        """Check that an empty calibration list gives None."""
        calibrations = []
        ball_diameter_m = 0.1
        meters_per_pixel = None
        result = calc._calculate_meters_per_pixel(calibrations, ball_diameter_m, meters_per_pixel)
        assert result is None

class TestCalculateSideFocalLength:
    """Tests for Calculators._calculate_side_focal_length."""
    def test_calculate_side_focal_length_normal(self, calc):
        """Check that a valid side-on calibration is kept with a focal length in px."""
        y_positions = [1.0]
        side_calibration = (1, [(10, 20), (30, 40)])
        side_frame_for_main_frame1 = 1
        ball_diameter_m = 0.1
        side_focal_length_px = None
        result = calc._calculate_side_focal_length(y_positions, side_calibration, side_frame_for_main_frame1, ball_diameter_m, side_focal_length_px)
        assert result[0] is not None
        assert isinstance(result[1], float)

    def test_calculate_side_focal_length_too_close(self, calc):
        """Check that side-on calibration points under 1 px apart give (None, None)."""
        y_positions = [1.0]
        side_calibration = (1, [(10, 20), (10.5, 20.5)])
        side_frame_for_main_frame1 = 1
        ball_diameter_m = 0.1
        side_focal_length_px = None
        result = calc._calculate_side_focal_length(y_positions, side_calibration, side_frame_for_main_frame1, ball_diameter_m, side_focal_length_px)
        assert result[0] is None
        assert result[1] is None

    def test_calculate_side_focal_length_invalid_frame(self, calc):
        """Check that a calibration frame with no y position gives (None, None)."""
        y_positions = [1.0]
        side_calibration = (5, [(10, 20), (30, 40)])
        side_frame_for_main_frame1 = 1
        ball_diameter_m = 0.1
        side_focal_length_px = None
        result = calc._calculate_side_focal_length(y_positions, side_calibration, side_frame_for_main_frame1, ball_diameter_m, side_focal_length_px)
        assert result[0] is None
        assert result[1] is None

class TestCalculateSeamAngle:
    """Tests for Calculators._calculate_seam_angle."""
    def test_calculate_seam_angle_normal(self, calc):
        """Check that two seam points add one measurement and clear the points."""
        seam_points = [(10, 20), (30, 40)]
        seam_measurements = []
        current_frame = 1
        result = calc._calculate_seam_angle(seam_points, seam_measurements, current_frame)
        assert len(result[0]) == 1
        assert result[1] == []

    def test_calculate_seam_angle_insufficient_points(self, calc):
        """Check that one seam point adds no measurement and is kept."""
        seam_points = [(10, 20)]
        seam_measurements = []
        current_frame = 1
        result = calc._calculate_seam_angle(seam_points, seam_measurements, current_frame)
        assert result[0] == []
        assert result[1] == [(10, 20)]

class CalculateInitialTrajectory:
    """Tests for Calculators._calculate_initial_trajectory."""
    def test_calculate_initial_trajectory_normal(self, calc):
        """Check that two ball positions give the trajectory angle as a float."""
        frame_positions = [(0, 10, 20, 0), (1, 30, 40, 0)]
        frame_height = 100
        meters_per_pixel = 0.01
        result = calc._calculate_initial_trajectory(frame_positions, frame_height, meters_per_pixel)
        assert isinstance(result, float)

    def test_calculate_initial_trajectory_insufficient_positions(self, calc):
        """Check that one ball position gives None."""
        frame_positions = [(0, 10, 20, 0)]
        frame_height = 100
        meters_per_pixel = 0.01
        result = calc._calculate_initial_trajectory(frame_positions, frame_height, meters_per_pixel)
        assert result is None