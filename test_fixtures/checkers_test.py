"""Tests for library/checkers.py."""

import pytest
import math
from library.checkers import Checker

@pytest.fixture
def checker():
    """Return a new Checker for each test."""
    return Checker()

class TestCheckCalibrationDifference:
    """Tests for Checker._check_calibration_difference."""
    def test_invalid_length_not_two(self, checker):
        """Check that one calibration instead of two gives (None, False)."""
        calibrations = [("", ((0, 0), (1, 1)))]
        result = checker._check_calibration_difference(calibrations, 0.1)
        assert result == (None, False)

    def test_invalid_pixel_distance_less_than_one(self, checker):
        """Check that calibration points under 1 px apart give (None, False)."""
        calibrations = [
            ("", ((0, 0), (0.5, 0))),
            ("", ((0, 0), (1, 1)))
        ]
        result = checker._check_calibration_difference(calibrations, 0.1)
        assert result == (None, False)

    def test_valid_calibration_low_percent_diff(self, checker, capsys):
        """Check that two calibrations under 25% apart are reported as valid."""
        calibrations = [
            ("", ((0, 0), (10, 0))),
            ("", ((0, 0), (10.1, 0)))
        ]
        ball_diameter_m = 1.0
        result = checker._check_calibration_difference(calibrations, ball_diameter_m)
        percent_diff, is_invalid = result
        assert not is_invalid
        assert percent_diff < 25.0
        captured = capsys.readouterr()
        assert "Main calibration valid" in captured.out

    def test_invalid_calibration_high_percent_diff(self, checker, capsys):
        """Check that two calibrations 25% or more apart are reported as invalid."""
        calibrations = [
            ("", ((0, 0), (10, 0))),
            ("", ((0, 0), (20, 0)))
        ]
        ball_diameter_m = 1.0
        result = checker._check_calibration_difference(calibrations, ball_diameter_m)
        percent_diff, is_invalid = result
        assert is_invalid
        assert percent_diff >= 25.0
        captured = capsys.readouterr()
        assert "Invalid Main Calibration" in captured.out

    def test_edge_case_zero_avg_mpp(self, checker):
        """Check that a zero average m/px gives a 0% difference and counts as valid."""
        calibrations = [
            ("", ((0, 0), (1, 0))),
            ("", ((0, 0), (1, 0)))
        ]
        ball_diameter_m = 0.0
        result = checker._check_calibration_difference(calibrations, ball_diameter_m)
        percent_diff, is_invalid = result
        assert percent_diff == 0
        assert not is_invalid

class TestCheckSeamWobble:
    """Tests for Checker._check_seam_wobble."""
    def test_empty_measurements(self, checker):
        """Check that no seam measurements give (None, False, 0.0)."""
        result = checker._check_seam_wobble([])
        assert result == (None, False, 0.0)

    def test_single_measurement(self, checker):
        """Check that one measurement gives back its angle with no wobble."""
        measurements = [(1, 45.0)]
        result = checker._check_seam_wobble(measurements)
        assert result == (45.0, False, 0.0)

    def test_multiple_no_wobble(self, checker):
        """Check that changes under 10 deg give the average angle and no wobble."""
        measurements = [(1, 10.0), (2, 15.0), (3, 12.0)]
        result = checker._check_seam_wobble(measurements)
        avg_angle, is_wobble, max_diff = result
        assert not is_wobble
        assert avg_angle == pytest.approx(12.333, rel=1e-3)
        assert max_diff <= 10.0

    def test_multiple_with_wobble(self, checker, capsys):
        """Check that a change of over 10 deg is flagged as a wobble seam."""
        measurements = [(1, 0.0), (2, 15.0), (3, 30.0)]
        result = checker._check_seam_wobble(measurements)
        assert result == ("Wobble Seam", True, 15.0)
        captured = capsys.readouterr()
        assert "Wobble Seam detected" in captured.out

    def test_angle_difference_wrapping_around_180(self, checker):
        """Check that angles wrap at 180 deg, so 170 and 10 deg are 20 deg apart."""
        measurements = [(1, 170.0), (2, 10.0)]
        result = checker._check_seam_wobble(measurements)
        assert result == ("Wobble Seam", True, 20.0)

    def test_unsorted_frames(self, checker):
        """Check that measurements are put in frame order before comparing them."""
        measurements = [(3, 10.0), (1, 5.0), (2, 15.0)]
        result = checker._check_seam_wobble(measurements)
        avg_angle, is_wobble, max_diff = result
        assert not is_wobble
        assert avg_angle == pytest.approx(10.0, rel=1e-3)
        assert max_diff == 10.0