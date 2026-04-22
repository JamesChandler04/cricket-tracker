import sys

sys.path.append("../")

import pytest
from unittest.mock import patch, MagicMock
import cv2
from drawers import Drawers

@pytest.fixture
def drawers():
    return Drawers()

class TestDrawText:
    def test_draw_text_default_parameters(self, drawers):
        frame = MagicMock()
        with patch('cv2.putText') as mock_puttext:
            drawers.draw_text(frame, "test text", (10, 20))
            assert mock_puttext.call_count == 2
            # Check white outline call
            white_call = mock_puttext.call_args_list[0]
            args, kwargs = white_call
            assert args[0] == frame
            assert args[1] == "test text"
            assert args[2] == (12, 22)  # x+2, y+2
            assert args[3] == cv2.FONT_HERSHEY_SIMPLEX
            assert args[4] == 1.2  # font_scale
            assert args[5] == (255, 255, 255)  # Color.WHITE.value
            assert args[6] == 5  # thickness + 2
            assert args[7] == cv2.LINE_AA
            # Check black text call
            black_call = mock_puttext.call_args_list[1]
            args, kwargs = black_call
            assert args[2] == (10, 20)  # original position
            assert args[5] == (0, 0, 0)  # Color.BLACK.value
            assert args[6] == 3  # thickness

    def test_draw_text_custom_parameters(self, drawers):
        frame = MagicMock()
        with patch('cv2.putText') as mock_puttext:
            drawers.draw_text(frame, "custom", (5, 15), font_scale=2.0, thickness=2)
            assert mock_puttext.call_count == 2
            # Check white outline call
            white_call = mock_puttext.call_args_list[0]
            args, kwargs = white_call
            assert args[4] == 2.0  # font_scale
            assert args[6] == 4  # thickness + 2
            # Check black text call
            black_call = mock_puttext.call_args_list[1]
            args, kwargs = black_call
            assert args[6] == 2  # thickness

class TestDrawMainTrajectory:
    def test_draw_main_trajectory_no_positions(self, drawers):
        frame = MagicMock()
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_main_trajectory(frame, [], 0, 1920, [], [], [])
            # Should not draw anything
            mock_line.assert_not_called()
            mock_circle.assert_not_called()
            mock_puttext.assert_not_called()

    def test_draw_main_trajectory_single_position(self, drawers):
        frame = MagicMock()
        frame_positions = [(5, 100, 200, 1.0)]
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_main_trajectory(frame, frame_positions, 5, 1920, [], [], [])
            # Should draw one circle and one text
            mock_line.assert_not_called()  # No lines with single position
            assert mock_circle.call_count == 1
            assert mock_puttext.call_count == 1
            # Check circle call (current frame, so red)
            circle_call = mock_circle.call_args
            args, kwargs = circle_call
            assert args[0] == frame
            assert args[1] == (100, 200)
            assert args[2] == 5  # radius
            assert args[3] == (0, 0, 255)  # red for current frame
            assert args[4] == -1  # filled

    def test_draw_main_trajectory_multiple_positions(self, drawers):
        frame = MagicMock()
        frame_positions = [
            (1, 100, 200, 0.0),
            (2, 110, 210, 0.033),
            (3, 120, 220, 0.066)
        ]
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_main_trajectory(frame, frame_positions, 2, 1920, [], [], [])
            # Should draw 2 lines, 3 circles, 3 texts
            assert mock_line.call_count == 2
            assert mock_circle.call_count == 3
            assert mock_puttext.call_count == 3
            # Check first line
            line_call = mock_line.call_args_list[0]
            args, kwargs = line_call
            assert args[1] == (100, 200)
            assert args[2] == (110, 210)
            assert args[3] == (0, 255, 0)  # green
            assert args[4] == 2  # thickness

    def test_draw_main_trajectory_seam_points_single(self, drawers):
        frame = MagicMock()
        frame_positions = [(1, 100, 200, 0.0)]
        seam_points = [(150, 250)]
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_main_trajectory(frame, frame_positions, 1, 1920, seam_points, [], [])
            # Should draw position circle/text + seam circle/text, no seam line
            assert mock_circle.call_count == 2
            assert mock_puttext.call_count == 2
            mock_line.assert_not_called()

    def test_draw_main_trajectory_seam_points_two(self, drawers):
        frame = MagicMock()
        frame_positions = [(1, 100, 200, 0.0)]
        seam_points = [(150, 250), (160, 260)]
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_main_trajectory(frame, frame_positions, 1, 1920, seam_points, [], [])
            # Should draw position circle/text + 2 seam circles/text + 1 seam line
            assert mock_circle.call_count == 3
            assert mock_puttext.call_count == 3
            assert mock_line.call_count == 1
            # Check seam line
            line_call = mock_line.call_args
            args, kwargs = line_call
            assert args[1] == (150, 250)
            assert args[2] == (160, 260)
            assert args[3] == (255, 255, 0)  # yellow

    def test_draw_main_trajectory_current_seam_angle(self, drawers):
        frame = MagicMock()
        frame_positions = [(5, 100, 200, 0.0)]
        seam_points = [(150, 250)]  # Only 1 point, so should show angle
        seam_measurements = [(5, 45.0)]
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_main_trajectory(frame, frame_positions, 5, 1920, seam_points, seam_measurements, [])
            # Should include seam angle text
            assert mock_puttext.call_count == 3  # position + seam + angle
            angle_call = mock_puttext.call_args_list[2]
            args, kwargs = angle_call
            assert "Seam Angle: 45.00°" in args[1]
            assert args[2] == (960, 30)  # frame_width // 2 = 1920 // 2 = 960

    def test_draw_main_trajectory_calibrations(self, drawers):
        frame = MagicMock()
        frame_positions = [(1, 100, 200, 0.0)]
        calibrations = [
            (1, [(50, 100), (60, 100)]),  # 2 points, should draw line
            (2, [(70, 120)])  # 1 point, no line
        ]
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_main_trajectory(frame, frame_positions, 1, 1920, [], [], calibrations)
            # Should draw 3 circles (1 position + 3 calibration), 2 texts, 1 calibration line
            assert mock_circle.call_count == 4  # 1 position + 3 calibration
            assert mock_puttext.call_count == 4  # 1 position + 3 calibration
            assert mock_line.call_count == 1  # 1 calibration line
            # Check calibration line
            line_call = mock_line.call_args
            args, kwargs = line_call
            assert args[1] == (50, 100)
            assert args[2] == (60, 100)
            assert args[3] == (0, 255, 255)  # cyan

class TestDrawSideTrajectory:
    def test_draw_side_trajectory_no_positions(self, drawers):
        frame = MagicMock()
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_side_trajectory(frame, [], 0, None)
            mock_line.assert_not_called()
            mock_circle.assert_not_called()
            mock_puttext.assert_not_called()

    def test_draw_side_trajectory_single_position(self, drawers):
        frame = MagicMock()
        side_positions = [(5, 100, 200, 1.0)]
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_side_trajectory(frame, side_positions, 5, None)
            mock_line.assert_not_called()  # No lines with single position
            assert mock_circle.call_count == 1
            assert mock_puttext.call_count == 1
            # Check circle call (smaller radius for side view)
            circle_call = mock_circle.call_args
            args, kwargs = circle_call
            assert args[2] == 2  # smaller radius

    def test_draw_side_trajectory_multiple_positions(self, drawers):
        frame = MagicMock()
        side_positions = [
            (1, 100, 200, 0.0),
            (2, 110, 210, 0.033),
            (3, 120, 220, 0.066)
        ]
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_side_trajectory(frame, side_positions, 2, None)
            assert mock_line.call_count == 2
            assert mock_circle.call_count == 3
            assert mock_puttext.call_count == 3

    def test_draw_side_trajectory_with_calibration(self, drawers):
        frame = MagicMock()
        side_positions = [(1, 100, 200, 0.0)]
        side_calibration = (1, [(50, 150), (60, 160)])
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_side_trajectory(frame, side_positions, 1, side_calibration)
            # Should draw position + 2 calibration circles/text + 1 calibration line
            assert mock_circle.call_count == 3
            assert mock_puttext.call_count == 3
            assert mock_line.call_count == 1

    def test_draw_side_trajectory_calibration_partial(self, drawers):
        frame = MagicMock()
        side_positions = [(1, 100, 200, 0.0)]
        side_calibration = (1, [(50, 150)])  # Only 1 point
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_side_trajectory(frame, side_positions, 1, side_calibration)
            # Should draw position + 1 calibration circle/text, no line
            assert mock_circle.call_count == 2
            assert mock_puttext.call_count == 2
            mock_line.assert_not_called()

    def test_draw_side_trajectory_no_calibration(self, drawers):
        frame = MagicMock()
        side_positions = [(1, 100, 200, 0.0)]
        with patch('cv2.line') as mock_line, \
             patch('cv2.circle') as mock_circle, \
             patch('cv2.putText') as mock_puttext:
            drawers.draw_side_trajectory(frame, side_positions, 1, None)
            # Should only draw position
            assert mock_circle.call_count == 1
            assert mock_puttext.call_count == 1
            mock_line.assert_not_called()
