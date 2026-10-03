"""Calculations on clicked points: metres per pixel, side-on focal length, seam angle
and initial direction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.helpers import Calibration, FramePosition

class Calculators:
    """Calculations on the points clicked in the tracker windows."""
    def _calculate_meters_per_pixel(self, calibrations: list[Calibration], ball_diameter_m: float, meters_per_pixel: float | None) -> tuple[list[Calibration], float] | None:
        """Return the calibrations and mean m/px, or None if a pair is unusable."""
        if not calibrations or len(calibrations[-1][1]) != 2:
            raise ValueError(f"Invalid number of calibrations points: {len(calibrations[-1][1]) if calibrations else 0}")
        meters_per_pixel_values = []
        for frame_num, points in calibrations:
            (x1, y1), (x2, y2) = points
            pixel_distance = math.hypot(x2 - x1, y2 - y1)
            if pixel_distance < 1.0:
                raise ValueError(f"Calibration points in frame {frame_num} are too close.")
            mpp = ball_diameter_m / pixel_distance
            meters_per_pixel_values.append(mpp)
            print(f"Main calibration in frame {frame_num}: 1 px = {mpp:.6f} m")
        meters_per_pixel = sum(meters_per_pixel_values) / len(meters_per_pixel_values)
        print(f"Main average meters per pixel: {meters_per_pixel:.6f} m")
        return calibrations, meters_per_pixel
    
    def _calculate_side_focal_length(self, y_positions: list[float], side_calibration: Calibration | None, side_frame_for_main_frame1: int | None, ball_diameter_m: float, side_focal_length_px: float | None) -> tuple[Calibration | None, float | None]:
        '''
        Returns optional[side calibration], optional[side focal length pixels]
        '''
        
        if not side_calibration or len(side_calibration[1]) != 2:
            return side_calibration, None
        frame_num, points = side_calibration
        (x1, z1), (x2, z2) = points
        pixel_distance = math.hypot(x2 - x1, z2 - z1)
        if pixel_distance < 1.0:
            raise ValueError(f"Side calibration points in frame {frame_num} are too close.")
        assert side_frame_for_main_frame1 is not None
        frame_idx = frame_num - side_frame_for_main_frame1 + 1
        if frame_idx < 1 or frame_idx > len(y_positions):
            raise ValueError(f"Side calibration frame {frame_num} has no corresponding Y-position.")
        y_m = y_positions[frame_idx - 1]
        if y_m <= 0:
            raise ValueError(f"Invalid Y-position {y_m:.3f}m for side calibration frame {frame_num}.")
        side_focal_length_px = pixel_distance * y_m / ball_diameter_m
        print(f"Side calibration in frame {frame_num}: Focal length = {side_focal_length_px:.2f} px (y_m={y_m:.3f}m, d_px={pixel_distance:.2f}px)")
        return side_calibration, side_focal_length_px
    
    def _calculate_seam_angle(self, seam_points: list[tuple[int, int]], seam_measurements: list[tuple[int, float]], current_frame: int) -> tuple[list[tuple[int, float]], list[tuple[int, int]]]:
        """Append a frame's seam angle from two clicked points, in degrees clockwise
        from image up, and return the measurements and an empty point list.
        """
        (x1, y1), (x2, y2) = seam_points
        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dx, -dy)) % 360
        seam_measurements.append((current_frame, angle))
        seam_points = []
        print(f"Seam angle calculated for frame {current_frame}: {angle:.2f} degrees")
        return seam_measurements, seam_points
    
    def _calculate_initial_trajectory(self, frame_positions: list[FramePosition], frame_height: int, meters_per_pixel: float | None) -> float | None:
        """Return the initial direction of travel, in degrees from the image vertical (0
        to 180), or None with under two points.
        """
        if len(frame_positions) < 2:
            return None
        _, x1, y1, _ = frame_positions[0]
        _, x2, y2, _ = frame_positions[1]
        assert meters_per_pixel is not None
        dx = (x2 - x1) * meters_per_pixel
        dy: float = (frame_height - y2) - (frame_height - y1)
        dy *= meters_per_pixel
        angle = math.degrees(math.atan2(dx, -dy)) % 360
        if angle > 180:
            angle -= 180
        return angle