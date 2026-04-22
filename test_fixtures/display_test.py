import sys

sys.path.append("../")

import pytest
from unittest.mock import patch, MagicMock
import cv2
from display import Display


class TestDisplay:
    def test_transform_frame_90(self):
        display = Display()
        frame = MagicMock()
        with patch('cv2.rotate') as mock_rotate:
            mock_rotate.return_value = 'rotated_90'
            result = display.transform_frame(frame, 90)
            mock_rotate.assert_called_once_with(frame, cv2.ROTATE_90_CLOCKWISE)
            assert result == 'rotated_90'

    def test_transform_frame_180(self):
        display = Display()
        frame = MagicMock()
        with patch('cv2.rotate') as mock_rotate:
            mock_rotate.return_value = 'rotated_180'
            result = display.transform_frame(frame, 180)
            mock_rotate.assert_called_once_with(frame, cv2.ROTATE_180)
            assert result == 'rotated_180'

    def test_transform_frame_270(self):
        display = Display()
        frame = MagicMock()
        with patch('cv2.rotate') as mock_rotate:
            mock_rotate.return_value = 'rotated_270'
            result = display.transform_frame(frame, 270)
            mock_rotate.assert_called_once_with(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            assert result == 'rotated_270'

    def test_transform_frame_default(self):
        display = Display()
        frame = 'original_frame'
        result = display.transform_frame(frame, 0)
        assert result == frame

    def test_advance_frame_increment(self):
        display = Display()
        result = display.advance_frame(5, 10)
        assert result == 6

    def test_advance_frame_at_end(self):
        display = Display()
        result = display.advance_frame(9, 10)
        assert result == 9

    def test_previous_frame_decrement(self):
        display = Display()
        result = display.previous_frame(5)
        assert result == 4

    def test_previous_frame_at_zero(self):
        display = Display()
        result = display.previous_frame(0)
        assert result == 0

    def test_load_main_video_success(self):
        display = Display()
        with patch('builtins.input') as mock_input, \
             patch('cv2.VideoCapture') as mock_cap_class:
            mock_input.return_value = 'path/to/video.mp4'
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FPS: 30
            }.get(prop, 0)
            mock_cap_class.return_value = mock_cap
            cap, total_frames, width, height, fps = display.load_main_video()
            assert cap == mock_cap
            assert total_frames == 100
            assert width == 1920
            assert height == 1080
            assert fps == 30.0
            mock_input.assert_called_once_with("Enter the path to your main cricket video file (bird's eye view): ")

    def test_load_main_video_fps_zero_valid_input(self):
        display = Display()
        with patch('builtins.input') as mock_input, \
             patch('cv2.VideoCapture') as mock_cap_class:
            mock_input.side_effect = ['path/to/video.mp4', '30.5']
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FPS: 0
            }.get(prop, 0)
            mock_cap_class.return_value = mock_cap
            cap, total_frames, width, height, fps = display.load_main_video()
            assert fps == 30.5

    def test_load_main_video_fps_zero_invalid_then_valid(self):
        display = Display()
        with patch('builtins.input') as mock_input, \
             patch('cv2.VideoCapture') as mock_cap_class:
            mock_input.side_effect = ['path/to/video.mp4', 'invalid', '-5', '25.0']
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FPS: 0
            }.get(prop, 0)
            mock_cap_class.return_value = mock_cap
            cap, total_frames, width, height, fps = display.load_main_video()
            assert fps == 25.0

    def test_load_main_video_not_opened(self):
        display = Display()
        with patch('builtins.input') as mock_input, \
             patch('cv2.VideoCapture') as mock_cap_class:
            mock_input.return_value = 'bad/path'
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_class.return_value = mock_cap
            with pytest.raises(ValueError, match="Error: Could not open main video file bad/path"):
                display.load_main_video()

    def test_load_side_video_success(self):
        display = Display()
        with patch('builtins.input') as mock_input, \
             patch('cv2.VideoCapture') as mock_cap_class:
            mock_input.return_value = 'path/to/side.mp4'
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 200,
                cv2.CAP_PROP_FRAME_WIDTH: 1280,
                cv2.CAP_PROP_FRAME_HEIGHT: 720
            }.get(prop, 0)
            mock_cap_class.return_value = mock_cap
            cap_side, total_frames_side, width_side, height_side = display.load_side_video()
            assert cap_side == mock_cap
            assert total_frames_side == 200
            assert width_side == 1280
            assert height_side == 720
            mock_input.assert_called_once_with("Enter the path to the side view video file: ")

    def test_load_side_video_empty_path(self):
        display = Display()
        with patch('builtins.input') as mock_input:
            mock_input.return_value = ''
            with pytest.raises(ValueError, match="No side view video path provided."):
                display.load_side_video()

    def test_load_side_video_not_opened(self):
        display = Display()
        with patch('builtins.input') as mock_input, \
             patch('cv2.VideoCapture') as mock_cap_class:
            mock_input.return_value = 'bad/path'
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_class.return_value = mock_cap
            with pytest.raises(ValueError, match="Error: Could not open side view video file bad/path"):
                display.load_side_video()