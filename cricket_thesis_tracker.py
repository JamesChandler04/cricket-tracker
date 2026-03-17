import cv2
import numpy as np
import pandas as pd
import math
from datetime import datetime
import os
import typing as t
if not t.TYPE_CHECKING:
    import xlsxwriter # Used when saving to excel

class CricketBallTracker:
    def __init__(self):
        self.video_path = None
        self.video_path_side = None
        self.cap = None
        self.cap_side = None
        self.frame_positions = []  # (frame_number, x_px, y_px, time_s) for main view
        self.side_positions = []   # (frame_number, x_px, z_px, time_s) for side view
        self.current_frame = 0
        self.total_frames = 0
        self.total_frames_side = 0
        self.fps = 30.0
        self.frame_width = 0
        self.frame_height = 0
        self.frame_width_side = 0
        self.frame_height_side = 0
        self.meters_per_pixel = None
        self.side_focal_length_px = None
        self.tracking_active = False
        self.seam_angle_active = False
        self.calibration_active = False
        self.side_calibration_active = False
        self.seam_points = []
        self.seam_measurements = []  # (frame_number, angle)
        self.calibrations = []  # Main view: (frame_number, [(x1, y1), (x2, y2)])
        self.side_calibration = None  # Side view: (frame_number, [(x1, z1), (x2, z2)])
        self.window_name = "Cricket Ball Tracker - Main View"
        self.window_name_side = "Cricket Ball Tracker - Side View"
        self.initial_velocity = None
        self.deceleration = None
        self.BALL_DIAMETER_M = 0.072  # Cricket ball diameter in meters
        self.first_frame_main = None
        self.side_frame_for_main_frame1 = None
        self.main_rotation = 0  # 0, 90, 180, or 270 degrees
        self.side_rotation = 0  # 0, 90, 180, or 270 degrees

    def transform_frame(self, frame, rotation):
        """Apply rotation to a frame"""
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    # ---------------------- Setup ---------------------- #
    def load_main_video(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open main video file {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Main video loaded: {self.total_frames} frames")
        print(f"Main resolution: {self.frame_width}x{self.frame_height}")

        while True:
            try:
                self.fps = float(input("Enter the frame rate (FPS) of the video: ").strip())
                if self.fps <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Please enter a positive number for FPS.")

    def load_side_video(self):
        self.video_path_side = input("Enter the path to the side view video file: ").strip().strip('"')
        if not self.video_path_side:
            raise ValueError("No side view video path provided.")
        self.cap_side = cv2.VideoCapture(self.video_path_side)
        if not self.cap_side.isOpened():
            raise ValueError(f"Error: Could not open side view video file {self.video_path_side}")

        self.total_frames_side = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width_side = int(self.cap_side.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height_side = int(self.cap_side.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Side video loaded: {self.total_frames_side} frames")
        print(f"Side resolution: {self.frame_width_side}x{self.frame_height_side}")

    # ---------------------- Calibration ---------------------- #
    def _calculate_meters_per_pixel(self):
        if not self.calibrations or len(self.calibrations[-1][1]) != 2:
            return
        meters_per_pixel_values = []
        for frame_num, points in self.calibrations:
            (x1, y1), (x2, y2) = points
            pixel_distance = math.hypot(x2 - x1, y2 - y1)
            if pixel_distance < 1.0:
                print(f"Error: Calibration points in frame {frame_num} are too close.")
                self.calibrations.pop()
                return
            mpp = self.BALL_DIAMETER_M / pixel_distance
            meters_per_pixel_values.append(mpp)
            print(f"Main calibration in frame {frame_num}: 1 px = {mpp:.6f} m")
        self.meters_per_pixel = sum(meters_per_pixel_values) / len(meters_per_pixel_values)
        print(f"Main average meters per pixel: {self.meters_per_pixel:.6f} m")

    def _calculate_side_focal_length(self, y_positions):
        if not self.side_calibration or len(self.side_calibration[1]) != 2:
            return
        frame_num, points = self.side_calibration
        (x1, z1), (x2, z2) = points
        pixel_distance = math.hypot(x2 - x1, z2 - z1)
        if pixel_distance < 1.0:
            print(f"Error: Side calibration points in frame {frame_num} are too close.")
            self.side_calibration = None
            return
        frame_idx = frame_num - self.side_frame_for_main_frame1 + 1
        if frame_idx < 1 or frame_idx > len(y_positions):
            print(f"Error: Side calibration frame {frame_num} has no corresponding Y-position.")
            self.side_calibration = None
            return
        y_m = y_positions[frame_idx - 1]
        if y_m <= 0:
            print(f"Error: Invalid Y-position {y_m:.3f}m for side calibration frame {frame_num}.")
            self.side_calibration = None
            return
        self.side_focal_length_px = pixel_distance * y_m / self.BALL_DIAMETER_M
        print(f"Side calibration in frame {frame_num}: Focal length = {self.side_focal_length_px:.2f} px (y_m={y_m:.3f}m, d_px={pixel_distance:.2f}px)")

    # ---------------------- Calibration Validation ---------------------- #
    def _check_calibration_difference(self):
        if len(self.calibrations) != 2:
            return None, False
        meters_per_pixel_values = []
        for frame_num, points in self.calibrations:
            (x1, y1), (x2, y2) = points
            pixel_distance = math.hypot(x2 - x1, y2 - y1)
            if pixel_distance < 1.0:
                return None, False
            mpp = self.BALL_DIAMETER_M / pixel_distance
            meters_per_pixel_values.append(mpp)
        mpp1, mpp2 = meters_per_pixel_values
        avg_mpp = (mpp1 + mpp2) / 2
        percent_diff = 100 * abs(mpp1 - mpp2) / avg_mpp if avg_mpp > 0 else 0
        is_invalid = percent_diff >= 25.0
        if is_invalid:
            print(f"Invalid Main Calibration: Percentage difference {percent_diff:.2f}% exceeds 25% threshold")
        else:
            print(f"Main calibration valid: Percentage difference {percent_diff:.2f}%")
        return percent_diff, is_invalid

    # ---------------------- Mouse ---------------------- #
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.calibration_active:
                if not self.calibrations or len(self.calibrations[-1][1]) == 2:
                    self.calibrations.append((self.current_frame, []))
                self.calibrations[-1][1].append((x, y))
                print(f"Main diameter point added in frame {self.current_frame}: ({x}, {y})")
                if len(self.calibrations[-1][1]) == 2:
                    self._calculate_meters_per_pixel()
                    if len(self.calibrations) >= 2:
                        self.calibration_active = False
                        print("Main second calibration completed.")
                    else:
                        print("Main first calibration completed. Navigate to another frame and press 'C'.")
            elif self.tracking_active and self.meters_per_pixel is not None:
                timestamp = self.current_frame / self.fps
                if not self.first_frame_main:
                    self.first_frame_main = self.current_frame
                    print(f"Main view Frame 1 set to raw frame {self.first_frame_main}")
                self._add_or_replace_point_for_frame(self.current_frame, x, y, timestamp, is_side=False)
                print(f"Main View - Frame {self.current_frame}: Ball at (x={x}, y={y}) - Time: {timestamp:.3f}s")
            elif self.seam_angle_active and self.meters_per_pixel is not None:
                self.seam_points.append((x, y))
                print(f"Seam point added: ({x}, {y})")
                if len(self.seam_points) == 2:
                    self._calculate_seam_angle()
                    self.seam_angle_active = False
                    print(f"Seam angle tracking stopped for frame {self.current_frame}. Angle: {self.seam_measurements[-1][1]:.2f} degrees")

    def mouse_callback_side(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.side_calibration_active:
                if self.side_frame_for_main_frame1 is None:
                    print(f"Error: Track the first ball position before calibrating.")
                    return
                if self.current_frame != self.side_frame_for_main_frame1:
                    print(f"Error: Side calibration must occur in first tracked frame ({self.side_frame_for_main_frame1}).")
                    return
                if self.side_calibration is None or len(self.side_calibration[1]) == 2:
                    self.side_calibration = (self.current_frame, [])
                self.side_calibration[1].append((x, y))
                print(f"Side diameter point added in frame {self.current_frame}: ({x}, {y})")
                if len(self.side_calibration[1]) == 2:
                    self.side_calibration_active = False
                    print(f"Side calibration completed for frame {self.current_frame}.")
            elif self.tracking_active:
                timestamp = self.current_frame / self.fps
                if not self.side_positions and self.side_frame_for_main_frame1 is None:
                    self.side_frame_for_main_frame1 = self.current_frame
                    print(f"Side view Frame 1 set to raw frame {self.side_frame_for_main_frame1}, corresponding to main view Frame 1")
                self._add_or_replace_point_for_frame(self.current_frame, x, y, timestamp, is_side=True)
                print(f"Side View - Frame {self.current_frame}: Ball at (x={x}, z={y}) - Time: {timestamp:.3f}s")

    # ---------------------- Seam Angle Calculation ---------------------- #
    def _calculate_seam_angle(self):
        if len(self.seam_points) != 2:
            return
        (x1, y1), (x2, y2) = self.seam_points
        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dx, -dy)) % 360
        self.seam_measurements.append((self.current_frame, angle))
        self.seam_points = []
        print(f"Seam angle calculated for frame {self.current_frame}: {angle:.2f} degrees")

    # ---------------------- Seam Wobble Check ---------------------- #
    def _check_seam_wobble(self):
        if not self.seam_measurements:
            return None, False, 0.0
        if len(self.seam_measurements) == 1:
            return self.seam_measurements[0][1], False, 0.0

        sorted_measurements = sorted(self.seam_measurements, key=lambda x: x[0])
        angles = [m[1] for m in sorted_measurements]
        avg_angle = sum(angles) / len(angles)
        max_diff = 0.0

        for i in range(1, len(angles)):
            diff = abs(angles[i] - angles[i-1])
            diff = min(diff, 180 - diff)
            max_diff = max(max_diff, diff)
            if diff > 10.0:
                print(f"Wobble Seam detected: Angle difference {diff:.2f} degrees between frames {sorted_measurements[i-1][0]} and {sorted_measurements[i][0]}")
                return "Wobble Seam", True, max_diff
        return avg_angle, False, max_diff

    # ---------------------- Initial Trajectory Calculation ---------------------- #
    def _calculate_initial_trajectory(self):
        if len(self.frame_positions) < 2:
            return None
        _, x1, y1, _ = self.frame_positions[0]
        _, x2, y2, _ = self.frame_positions[1]
        dx = (x2 - x1) * self.meters_per_pixel
        dy = (self.frame_height - y2) - (self.frame_height - y1)
        dy *= self.meters_per_pixel
        angle = math.degrees(math.atan2(dx, -dy)) % 360
        if angle > 180:
            angle -= 180
        return angle

    # ---------------------- Navigation ---------------------- #
    def advance_frame(self, is_side=False):
        total = self.total_frames_side if is_side else self.total_frames
        if self.current_frame < total - 1:
            self.current_frame += 1

    def previous_frame(self, is_side=False):
        if self.current_frame > 0:
            self.current_frame -= 1

    # ---------------------- Drawing ---------------------- #
    def draw_text(self, frame, text, position, font_scale=1.2, thickness=3):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = position
        cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    def draw_trajectory(self, frame, is_side=False):
        positions = self.side_positions if is_side else self.frame_positions
        height = self.frame_height_side if is_side else self.frame_height
        marker_radius = 2 if is_side else 5  # Smaller marker for side view
        if len(positions) > 1:
            for i in range(1, len(positions)):
                prev = positions[i - 1]
                curr = positions[i]
                cv2.line(frame, (prev[1], prev[2]), (curr[1], curr[2]), (0, 255, 0), 2)

        for i, (frame_num, x, y, timestamp) in enumerate(positions):
            color = (0, 0, 255) if frame_num == self.current_frame else (255, 0, 0)
            cv2.circle(frame, (x, y), marker_radius, color, -1)
            cv2.putText(frame, f"{i}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if not is_side:
            if len(self.seam_points) >= 1:
                for i, (x, y) in enumerate(self.seam_points):
                    cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)
                    cv2.putText(frame, f"S{i+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if len(self.seam_points) == 2:
                    cv2.line(frame, self.seam_points[0], self.seam_points[1], (255, 255, 0), 2)

            current_seam_angle = None
            for frame_num, angle in self.seam_measurements:
                if frame_num == self.current_frame and len(self.seam_points) < 2:
                    current_seam_angle = angle
                    break
            if current_seam_angle is not None and len(self.seam_points) < 2:
                cv2.putText(frame, f"Seam Angle: {current_seam_angle:.2f}°",
                            (self.frame_width // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            for i, (frame_num, points) in enumerate(self.calibrations):
                for j, (x, y) in enumerate(points):
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                    cv2.putText(frame, f"D{i+1}{j+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if len(points) == 2:
                    cv2.line(frame, points[0], points[1], (0, 255, 255), 2)
        else:
            if self.side_calibration and len(self.side_calibration[1]) >= 1:
                for j, (x, z) in enumerate(self.side_calibration[1]):
                    cv2.circle(frame, (x, z), 5, (0, 255, 255), -1)
                    cv2.putText(frame, f"D{j+1}", (x + 10, z - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if len(self.side_calibration[1]) == 2:
                    cv2.line(frame, self.side_calibration[1][0], self.side_calibration[1][1], (0, 255, 255), 2)

    # ---------------------- Data Helpers ---------------------- #
    def _add_or_replace_point_for_frame(self, frame_no, x, y, t, is_side=False):
        positions = self.side_positions if is_side else self.frame_positions
        for i, (f, *_rest) in enumerate(positions):
            if f == frame_no:
                positions[i] = (frame_no, int(x), int(y), t)
                return
        positions.append((frame_no, int(x), int(y), t))
        positions.sort(key=lambda z: z[0])

    def reset(self):
        self.frame_positions = []
        self.side_positions = []
        self.seam_points = []
        self.seam_measurements = []
        self.calibrations = []
        self.side_calibration = None
        self.meters_per_pixel = None
        self.side_focal_length_px = None
        self.initial_velocity = None
        self.deceleration = None
        self.first_frame_main = None
        self.side_frame_for_main_frame1 = None
        print("Tracking data, seam angle, calibrations, and parameters reset.")

    # ---------------------- 3D Data ---------------------- #
    def _build_3d_data(self):
        FX = 2877.72
        FZ = 2877.72
        X0 = 1920.0
        Z0 = 1080.0
        scale_factor = 175

        if not self.frame_positions:
            print("Cannot build 3D Data: No tracking data in main view.")
            return None
        if self.meters_per_pixel is None:
            print("Cannot build 3D Data: Main calibration not completed.")
            return None
        if len(self.frame_positions) < 2:
            print("Cannot build 3D Data: Need at least two points to calculate velocity.")
            return None

        t0 = self.frame_positions[0][3]
        records = []
        y_positions = []
        cum_dist = 0.0

        # Calculate initial velocity and deceleration
        if self.initial_velocity is None or self.deceleration is None:
            initial_speeds = []
            decelerations = []
            for i in range(1, len(self.frame_positions)):
                frame_num, x_px, y_px, t = self.frame_positions[i]
                prev_frame_num, px_prev, py_prev, t_prev = self.frame_positions[i-1]
                dt = max(t - t_prev, 1.0 / self.fps)
                dist_px = abs(y_px - py_prev)
                dist_m = dist_px * self.meters_per_pixel
                speed_ms = dist_m / dt if dt > 0 else 0.0
                initial_speeds.append(speed_ms)
                if i > 1 and initial_speeds[-2] > speed_ms:
                    decel = (initial_speeds[-2] - speed_ms) / dt
                    decelerations.append(decel)
            self.initial_velocity = sum(initial_speeds) / len(initial_speeds) if initial_speeds else 0.0
            self.deceleration = sum(decelerations) / len(decelerations) if decelerations else 0.0

        if self.initial_velocity == 0.0:
            print("Cannot build 3D Data: Initial velocity is zero.")
            return None

        # Initial trajectory angle (used for projected X only)
        initial_trajectory = self._calculate_initial_trajectory()
        if initial_trajectory is None:
            print("Cannot build 3D Data: Initial trajectory not available.")
            return None
        theta = math.radians(initial_trajectory)
        vx0 = self.initial_velocity * math.sin(theta)  # X-velocity component
        ax = self.deceleration or 0.0  # Assume same deceleration as Y

        # Process tracked frames
        prev = None
        for i, (frame_num, x_px, y_px, t) in enumerate(self.frame_positions):
            adj_frame = i + 1
            adj_time = t - t0
            y_m = y_px * self.meters_per_pixel
            y_positions.append(y_m)
            if prev is not None:
                dist_px = abs(y_px - prev[2])
                dist_m = dist_px * self.meters_per_pixel
                cum_dist += dist_m
            # Projected X-position (kept)
            x_m_projected = vx0 * adj_time - 0.5 * ax * adj_time**2
            # Build: Frame, Time, Projected X, Y, Z (Z filled later), then placeholders
            records.append([adj_frame, adj_time, x_m_projected, y_m, None])  # Z to be filled later
            prev = (frame_num, x_px, y_px, t)

        # Extrapolate to 17m (for Y-Position path length along flight)
        if records:
            last_row = records[-1]
            last_frame = last_row[0]
            last_time = last_row[1]
            z0 = cum_dist
            v0 = self.initial_velocity
            a = self.deceleration or 0.0
            t_step = 1.0 / self.fps
            t = last_time + t_step
            frame_counter = last_frame + 1

            print(f"3D Data: Starting extrapolation with z0={z0:.3f}m, v0={v0:.3f}m/s, a={a:.3f}m/s^2, vx0={vx0:.3f}m/s, theta={initial_trajectory:.2f}°")

            while cum_dist < 17.0:
                delta_t = t - last_time
                v = max(0, v0 - a * delta_t)
                if v <= 0:
                    print(f"Warning: Velocity reached zero at t={t:.3f}s, cum_dist={cum_dist:.3f}m")
                    break
                cum_dist = z0 + v0 * delta_t - 0.5 * a * delta_t**2
                x_m_projected = vx0 * (t - t0) - 0.5 * ax * (t - t0)**2
                if cum_dist >= 17.0:
                    A = 0.5 * a
                    B = -v0
                    C = 17.0 - z0
                    discriminant = B**2 - 4*A*C
                    if discriminant < 0:
                        print(f"Warning: Cannot reach 17 meters with discriminant={discriminant:.3f}")
                        break
                    delta_t = (-B - math.sqrt(discriminant)) / (2*A) if a != 0 else (17.0 - z0) / v0
                    if delta_t <= 0:
                        print(f"Warning: Invalid delta_t={delta_t:.3f}s for final point")
                        break
                    t = last_time + delta_t
                    cum_dist = 17.0
                    x_m_projected = vx0 * (t - t0) - 0.5 * ax * (t - t0)**2
                y_positions.append(cum_dist)
                records.append([frame_counter, t, x_m_projected, cum_dist, None])
                t += t_step
                frame_counter += 1
                if self.side_frame_for_main_frame1 is not None and frame_counter - 1 + self.side_frame_for_main_frame1 >= self.total_frames_side:
                    print(f"Reached end of side video at frame {frame_counter - 1 + self.side_frame_for_main_frame1}")
                    break
                if cum_dist >= 17.0:
                    break

        # Calculate side view focal length (still used in Parameters)
        if self.side_calibration:
            self._calculate_side_focal_length(y_positions)

        # Side/back clicks mapping (we only use Z from old mapping for this sheet)
        x0_side, z0_side = None, None
        if self.side_positions:
            x0_side, z0_side = self.side_positions[0][1], self.side_positions[0][2]

        side_frame_map = {}  # frame -> (x_m_old, z_m_old) [we keep z only]
        side_px_map   = {}  # frame -> (x_px, z_px)

        if self.side_positions and self.side_frame_for_main_frame1 is not None:
            for frame_num, x, z, t in self.side_positions:
                frame_idx = frame_num - self.side_frame_for_main_frame1 + 1
                if frame_idx < 1 or frame_idx > len(y_positions):
                    continue
                y_m = y_positions[frame_idx - 1]
                if self.side_focal_length_px and y_m > 0:
                    mpp = y_m / self.side_focal_length_px
                    x_m_old = ((x - x0_side) * mpp) / FX * 175 if x0_side is not None else (x * mpp) / FX * 175
                    z_m_old = ((z - z0_side) * mpp) / FZ * 175 if z0_side is not None else (z * mpp) / FZ * 175
                else:
                    x_m_old = (x - x0_side) * 175 if x0_side is not None else x * 175
                    z_m_old = (z - z0_side) * 175 if z0_side is not None else z * 175
                side_frame_map[frame_num] = (x_m_old, z_m_old)
                side_px_map[frame_num] = (x, z)

        # ---- Back-adjusted X (meters) & Actual Data arrays (side/back camera) ----
        # Use first-frame side calibration diameter as metres-per-pixel
        m_per_px_first = None
        if self.side_calibration and len(self.side_calibration[1]) == 2:
            (sx1, sz1), (sx2, sz2) = self.side_calibration[1]
            px_diam = math.hypot(sx2 - sx1, sz2 - sz1)
            if px_diam >= 1.0:
                m_per_px_first = self.BALL_DIAMETER_M / px_diam

        # Side X/Z pixels by adjusted frame (1-based to match records)
        side_px_by_adjframe = {}
        if self.side_positions and self.side_frame_for_main_frame1 is not None:
            for frame_num, x, z, _ in self.side_positions:
                adj_idx = frame_num - self.side_frame_for_main_frame1 + 1
                if adj_idx >= 1:
                    side_px_by_adjframe[adj_idx] = (x, z)

        x_m_adj = []
        actual_adj = []
        side_x_px_col = []
        side_z_px_col = []

        first_x_px = side_px_by_adjframe.get(1, (None, None))[0]
        prev_xm = None
        prev_actual = 0.0

        for i, rec in enumerate(records):
            adj_frame = rec[0]
            sx_px, sz_px = side_px_by_adjframe.get(adj_frame, (None, None))
            side_x_px_col.append(sx_px)
            side_z_px_col.append(sz_px)

            if m_per_px_first is not None and sx_px is not None and first_x_px is not None:
                xm = (sx_px - first_x_px) * m_per_px_first
            else:
                xm = None

            if xm is not None and prev_xm is not None:
                prev_actual = prev_actual + (xm - prev_xm) * 0.1 * adj_frame
            elif prev_xm is None and xm is not None:
                prev_actual = 0.0

            x_m_adj.append(xm)
            actual_adj.append(prev_actual if xm is not None else None)
            prev_xm = xm if xm is not None else prev_xm

        # ---- Assemble final 3D Data rows (without Initial Path / Swing yet) ----
        final_records = []
        for i, rec in enumerate(records):
            frame_no = rec[0]
            time_s = rec[1]
            proj_x = rec[2]
            y_pos = rec[3]
            # z position from side_frame_map (if available)
            side_frame = self.side_frame_for_main_frame1 + (frame_no - 1) if self.side_frame_for_main_frame1 is not None else None
            _, z_m_old = side_frame_map.get(side_frame, (None, None))

            final_records.append([
                frame_no,
                time_s,
                proj_x,
                y_pos,
                z_m_old,
                side_x_px_col[i],
                x_m_adj[i],
                actual_adj[i],
                side_z_px_col[i]
            ])

            print(f"3D Data: Frame {frame_no}, Time {time_s:.3f}s, Y={y_pos:.3f}m, "
                  f"Z={z_m_old}, SideXpx={side_x_px_col[i]}, x(m)={x_m_adj[i]}, Actual={actual_adj[i]}")

        columns = [
            "Frame Number",
            "Time (s)",
            "Projected X-Position (m)",
            "Y-Position (m)",
            "Z-Position (m)",
            "Side X (px)",
            "x (m)",
            "Actual Data",
            "Side Z (px)"
        ]
        df = pd.DataFrame(final_records, columns=columns)

        # --- Initial Path & Swing (computed AFTER DataFrame is built) ---
        actual = df["Actual Data"]
        valid_mask = actual.notna()
        if valid_mask.any():
            valid_df = df[valid_mask]
            first_frame = int(valid_df["Frame Number"].iloc[0])
            target_frame = int(valid_df["Frame Number"].iloc[19]) if len(valid_df) >= 20 else int(valid_df["Frame Number"].iloc[-1])

            a1 = float(df.loc[df["Frame Number"] == first_frame, "Actual Data"].iloc[0])
            a20 = float(df.loc[df["Frame Number"] == target_frame, "Actual Data"].iloc[0])
            denom = (target_frame - first_frame)
            m = (a20 - a1) / denom if denom != 0 else 0.0
            b = a1 - m * first_frame

            df["Initial Path (m)"] = m * df["Frame Number"] + b
            df["Swing (m)"] = df["Actual Data"] - df["Initial Path (m)"]
        else:
            df["Initial Path (m)"] = np.nan
            df["Swing (m)"] = np.nan

        print(f"3D Data: Generated {len(df)} rows")
        return df

    # ---------------------- DataFrame ---------------------- #
    def _build_dataframe(self):
        if not self.frame_positions or self.meters_per_pixel is None:
            return None, None

        records = []
        decelerations = []

        t0 = self.frame_positions[0][3]
        x0_px, y0_px = self.frame_positions[0][1], self.frame_positions[0][2]
        x0_m = x0_px * self.meters_per_pixel
        y0_m = (self.frame_height - y0_px) * self.meters_per_pixel
        initial_speed = None

        initial_speeds = []
        n_initial_frames = min(3, len(self.frame_positions))
        for i in range(1, n_initial_frames):
            if i < len(self.frame_positions):
                frame_num, x_px, y_px, t = self.frame_positions[i]
                prev_frame_num, px_prev, py_prev, t_prev = self.frame_positions[i-1]
                dt = max(t - t_prev, 1.0 / self.fps)
                dist_px = math.hypot(x_px - px_prev, y_px - py_prev)
                dist_m = dist_px * self.meters_per_pixel
                speed_ms = dist_m / dt if dt > 0 else 0.0
                initial_speeds.append(speed_ms)
                if i == 1:
                    initial_speed = speed_ms
        self.initial_velocity = sum(initial_speeds) / len(initial_speeds) if initial_speeds else 0.0

        prev = None
        for i, (frame_num, x_px, y_px, t) in enumerate(self.frame_positions):
            x_m = x_px * self.meters_per_pixel - x0_m
            y_m = (self.frame_height - y_px) * self.meters_per_pixel - y0_m
            adj_frame = i + 1
            adj_time = t - t0
            if prev is None:
                dt = 0.0
                speed_ms = None
                speed_kmh = None
            else:
                _, _, _, t_prev = prev
                dt = max(t - t_prev, 1.0 / self.fps)
                dist_px = math.hypot(x_px - prev[1], y_px - prev[2])
                dist_m = dist_px * self.meters_per_pixel
                speed_ms = dist_m / dt if dt > 0 else 0.0
                speed_kmh = speed_ms * 3.6 if speed_ms is not None else None

                if len(records) > 0 and records[-1][4] is not None:
                    prev_speed = records[-1][4]
                    if speed_ms is not None and prev_speed > speed_ms:
                        decel = (prev_speed - speed_ms) / dt
                        decelerations.append(decel)

            records.append([
                adj_frame,
                adj_time,
                x_m,
                y_m,
                dt,
                speed_ms,
                speed_kmh
            ])
            prev = (frame_num, x_px, y_px, t)

        self.deceleration = sum(decelerations) / len(decelerations) if decelerations else 0.0

        columns = [
            "Frame Number",
            "Time (s)",
            "X-Position (m)",
            "Y-Position (m)",
            "Delta T (s)",
            "Speed (m/s)",
            "Speed (km/h)"
        ]
        return pd.DataFrame(records, columns=columns), initial_speed * 3.6 if initial_speed is not None else None

    # ---------------------- Save to Excel ---------------------- #
    def save_to_excel(self):
        if not self.frame_positions:
            print("No tracking data to save.")
            return

        df, initial_speed_kmh = self._build_dataframe()
        if df is None:
            print("No data to save.")
            return

        df_3d_data = self._build_3d_data()
        initial_trajectory = self._calculate_initial_trajectory()
        seam_angle, is_wobble, max_seam_diff = self._check_seam_wobble()
        percent_diff, is_invalid_cal = self._check_calibration_difference()

        # Seam Angles DataFrame
        seam_data = [[frame_num, f"{angle:.2f}"] for frame_num, angle in self.seam_measurements]
        seam_df = pd.DataFrame(seam_data, columns=["Frame Number", "Seam Angle (degrees)"])
        seam_summary = ["Average" if not is_wobble else "Wobble Seam",
                        f"{seam_angle:.2f}" if not is_wobble and seam_angle is not None else "Wobble Seam"]
        seam_diff_row = ["Max Consecutive Difference", f"{max_seam_diff:.2f} (Threshold: 10.00 degrees)"]
        seam_df = pd.concat([seam_df,
                             pd.DataFrame([seam_summary], columns=["Frame Number", "Seam Angle (degrees)"]),
                             pd.DataFrame([seam_diff_row], columns=["Frame Number", "Seam Angle (degrees)"])],
                            ignore_index=True)

        # Main Calibrations DataFrame
        cal_data = []
        for frame_num, points in self.calibrations:
            (x1, y1), (x2, y2) = points
            pixel_distance = math.hypot(x2 - x1, y2 - y1)
            mpp = self.BALL_DIAMETER_M / pixel_distance if pixel_distance >= 1.0 else None
            if mpp is not None:
                cal_data.append([frame_num, f"{mpp:.6f}"])
        cal_df = pd.DataFrame(cal_data, columns=["Frame Number", "Meters per Pixel"])
        if self.meters_per_pixel is not None:
            cal_summary = ["Average", f"{self.meters_per_pixel:.6f}"]
            cal_diff_row = ["Percentage Difference",
                            f"{percent_diff:.2f}% (Threshold: 25.00%)" if percent_diff is not None else "N/A"]
            cal_df = pd.concat([cal_df,
                                pd.DataFrame([cal_summary], columns=["Frame Number", "Meters per Pixel"]),
                                pd.DataFrame([cal_diff_row], columns=["Frame Number", "Meters per Pixel"])],
                               ignore_index=True)

        # Side Calibration DataFrame
        side_cal_data = []
        if df_3d_data is not None and self.side_calibration:
            frame_num, points = self.side_calibration
            (x1, z1), (x2, z2) = points
            pixel_distance = math.hypot(x2 - x1, z2 - z1)
            if pixel_distance >= 1.0:
                # find corresponding Y in df_3d_data (same indexing as frames in that sheet)
                idx = None
                if "Frame Number" in df_3d_data.columns:
                    # use first occurrence
                    idx = df_3d_data.index[df_3d_data["Frame Number"] == (frame_num - self.side_frame_for_main_frame1 + 1)].tolist()
                    if idx:
                        y_m = df_3d_data.loc[idx[0], "Y-Position (m)"]
                        f = pixel_distance * y_m / self.BALL_DIAMETER_M if y_m > 0 else None
                        if f is not None:
                            side_cal_data.append([frame_num, f"{pixel_distance:.2f}", f"{y_m:.3f}", f"{f:.2f}"])
        side_cal_df = pd.DataFrame(side_cal_data, columns=["Frame Number", "Pixel Diameter", "Y-Position (m)", "Focal Length (px)"])

        while True:
            folder_path = input("Enter the folder path to save the Excel file (leave blank for current directory): ").strip().strip('"')
            if not folder_path:
                folder_path = os.getcwd()
                break
            if os.path.isdir(folder_path):
                break
            print("Invalid folder path. Please enter a valid directory or leave blank for current directory.")

        filename = input("Enter Excel filename (without extension): ").strip()
        if not filename:
            filename = datetime.now().strftime("cricket_tracking_%Y%m%d_%H%M%S")
        filename = f"{filename}.xlsx"
        full_path = os.path.join(folder_path, filename)

        try:
            with pd.ExcelWriter(full_path, engine='xlsxwriter') as writer:
                # Main Data sheet
                df.to_excel(writer, sheet_name='Data', index=False)

                # Parameters (with Final Swing)
                final_swing_val = "N/A"
                if df_3d_data is not None and not df_3d_data.empty:
                    if "Actual Data" in df_3d_data.columns and "Initial Path (m)" in df_3d_data.columns:
                        valid = df_3d_data["Actual Data"].dropna()
                        if len(valid) >= 1:
                            last5 = valid.tail(5).mean()
                            # initial path at final frame present in the same last row
                            last_row_idx = df_3d_data.index[-1]
                            init_at_end = df_3d_data.loc[last_row_idx, "Initial Path (m)"]
                            if pd.notna(init_at_end):
                                final_swing_val = f"{(last5 - init_at_end):.6f}"

                param_df = pd.DataFrame({
                    'Parameter': [
                        'Seam Angle (degrees)',
                        'Initial Trajectory (degrees)',
                        'Initial Speed (km/h)',
                        'Main Meters per Pixel',
                        'Side Focal Length (px)',
                        'Side Frame for Main Frame 1',
                        'Final Swing (m)'
                    ],
                    'Value': [
                        "Wobble Seam" if is_wobble else f"{seam_angle:.2f}" if seam_angle is not None else 'N/A',
                        f"{initial_trajectory:.2f}" if initial_trajectory is not None else 'N/A',
                        f"{initial_speed_kmh:.2f}" if initial_speed_kmh is not None else 'N/A',
                        "Invalid Calibration" if is_invalid_cal else f"{self.meters_per_pixel:.6f}" if self.meters_per_pixel is not None else 'N/A',
                        f"{self.side_focal_length_px:.2f}" if self.side_focal_length_px is not None else 'N/A',
                        f"{self.side_frame_for_main_frame1}" if self.side_frame_for_main_frame1 is not None else 'Not set',
                        final_swing_val
                    ]
                })
                param_df.to_excel(writer, sheet_name='Parameters', index=False)

                # 3D Data (includes Side X px, x(m), Actual Data, Side Z px, plus Initial Path & Swing)
                if df_3d_data is not None and not df_3d_data.empty:
                    print("Saving 3D Data sheet with {} rows".format(len(df_3d_data)))
                    df_3d_data.to_excel(writer, sheet_name='3D Data', index=False)

                if not seam_df.empty:
                    print("Saving Seam Angles sheet with {} rows".format(len(seam_df)))
                    seam_df.to_excel(writer, sheet_name='Seam Angles', index=False)

                if not cal_df.empty:
                    print("Saving Main Calibrations sheet with {} rows".format(len(cal_df)))
                    cal_df.to_excel(writer, sheet_name='Main Calibrations', index=False)

                if not side_cal_df.empty:
                    print("Saving Side Calibration sheet with {} rows".format(len(side_cal_df)))
                    side_cal_df.to_excel(writer, sheet_name='Side Calibration', index=False)

            print(f"Tracking data saved to {full_path}")
        except Exception as e:
            print(f"Error saving file: {e}")

    # ---------------------- Main Tracker ---------------------- #
    def run_main_tracker(self, video_path):
        try:
            self.load_main_video(video_path)
        except ValueError as e:
            print(e)
            return False

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n=== CRICKET BALL TRACKER - MAIN VIEW (BIRD'S EYE) ===")
        print("Controls:")
        print("- C: Start/Stop Calibration Mode (click two points across ball diameter, repeat in another frame)")
        print("- SPACE: Start/Stop Ball Tracking Mode (after both calibrations)")
        print("- T: Start/Stop Seam Angle Tracking Mode (after both calibrations)")
        print("- A/D or ←/→: Move frame back/forward")
        print("- Click: Mark calibration points, ball position, or seam points")
        print("- S: Proceed to side view tracking")
        print("- R: Reset tracking and calibration")
        print("- Q or ESC: Quit")
        print("- O: Rotate video 90 degrees clockwise")

        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                print("Frame read failed or end of main video.")
                break

            display_frame = self.transform_frame(frame.copy(), self.main_rotation)
            self.draw_trajectory(display_frame, is_side=False)

            if self.calibration_active:
                cal_num = len(self.calibrations) + 1 if not self.calibrations or len(self.calibrations[-1][1]) == 2 else len(self.calibrations)
                points = len(self.calibrations[-1][1]) if self.calibrations else 0
                status = f"{'First' if cal_num == 1 else 'Second'} Calibration - Click point {points + 1}/2 for ball diameter"
            elif self.tracking_active:
                status = "BALL TRACKING ACTIVE - Click on ball"
            elif self.seam_angle_active:
                status = f"SEAM ANGLE TRACKING ACTIVE - Click point {len(self.seam_points) + 1}/2"
            else:
                status = f"PAUSED - Press C for {'first' if not self.calibrations else 'second'} calibration, SPACE for ball, T for seam"

            seam_angle, is_wobble, _ = self._check_seam_wobble()
            seam_display = "Wobble Seam" if is_wobble else f"{seam_angle:.2f}°" if seam_angle is not None else "Not set"
            self.draw_text(display_frame, status, (10, 40), font_scale=1.1)
            self.draw_text(display_frame, f"Frame: {self.current_frame}/{self.total_frames-1}", (10, 80), font_scale=0.9)
            self.draw_text(display_frame, f"Main Points: {len(self.frame_positions)}", (10, 110), font_scale=0.9)
            self.draw_text(display_frame, f"Seam Angle: {seam_display}", (10, 140), font_scale=0.9)
            self.draw_text(display_frame, f"Deceleration: {'{:.3f} m/s^2'.format(self.deceleration) if self.deceleration is not None else 'Not set'}", (10, 170), font_scale=0.9)
            self.draw_text(display_frame, f"Meters/Pixel: {'{:.6f} m/px'.format(self.meters_per_pixel) if self.meters_per_pixel is not None else 'Not set'}", (10, 200), font_scale=0.9)
            self.draw_text(display_frame, "S = Proceed to Side View", (10, 230), font_scale=0.9)

            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(10) & 0xFF

            if key in (ord('q'), 27):
                self.cap.release()
                cv2.destroyAllWindows()
                return False
            elif key == ord('c'):
                if self.tracking_active or self.seam_angle_active:
                    print("Cannot start calibration while tracking or seam angle mode is active.")
                elif len(self.calibrations) >= 2:
                    print("Main calibrations already completed.")
                else:
                    self.calibration_active = not self.calibration_active
                    if self.calibration_active:
                        if not self.calibrations or len(self.calibrations[-1][1]) == 2:
                            print(f"Starting {'first' if not self.calibrations else 'second'} main calibration.")
                        else:
                            print(f"Continuing {'first' if len(self.calibrations) == 1 else 'second'} main calibration.")
                    else:
                        print("Main calibration paused.")
            elif key == ord(' '):
                if self.meters_per_pixel is None or len(self.calibrations) < 2:
                    print("Cannot start ball tracking until main calibrations are complete.")
                elif self.seam_angle_active or self.calibration_active:
                    print("Cannot start ball tracking while seam angle or calibration mode is active.")
                else:
                    self.tracking_active = not self.tracking_active
                    print(f"{'Ball tracking started' if self.tracking_active else 'Ball tracking paused'}")
            elif key == ord('t'):
                if self.meters_per_pixel is None or len(self.calibrations) < 2:
                    print("Cannot start seam angle tracking until main calibrations are complete.")
                elif self.tracking_active or self.calibration_active:
                    print("Cannot start seam angle tracking while ball tracking or calibration mode is active.")
                else:
                    self.seam_angle_active = not self.seam_angle_active
                    if self.seam_angle_active:
                        self.seam_points = []
                        print("Seam angle tracking started.")
                    else:
                        print("Seam angle tracking stopped.")
            elif key in (ord('a'), 81):
                self.previous_frame()
            elif key in (ord('d'), 83):
                self.advance_frame()
            elif key == ord('s'):
                if self.meters_per_pixel is None or len(self.calibrations) < 2:
                    print("Cannot proceed until main calibrations are complete.")
                elif not self.frame_positions:
                    print("Cannot proceed without tracking at least one point.")
                else:
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return True
            elif key == ord('o'):  # 'o' for rotate
                self.main_rotation = (self.main_rotation + 90) % 360
                print(f"Main view rotated to {self.main_rotation} degrees")
            elif key == ord('r'):
                self.reset()

        self.cap.release()
        cv2.destroyAllWindows()
        return False

    # ---------------------- Side Tracker ---------------------- #
    def run_side_tracker(self):
        try:
            self.load_side_video()
        except ValueError as e:
            print(e)
            return

        self.current_frame = 0
        cv2.namedWindow(self.window_name_side, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name_side, self.mouse_callback_side)

        print("\n=== CRICKET BALL TRACKER - SIDE VIEW ===")
        print("First click to track the ball will set this as Frame 1, corresponding to main view Frame 1.")
        print("Calibrate ball diameter in the first tracked frame by pressing 'C' (can be done while tracking).")
        print("Controls:")
        print("- C: Start/Stop Calibration Mode (click two points across ball diameter in first tracked frame)")
        print("- SPACE: Start/Stop Ball Tracking Mode")
        print("- A/D or ←/→: Move frame back/forward")
        print("- Click: Mark calibration points or ball position (X, Z)")
        print("- S: Save data to Excel")
        print("- R: Reset side view tracking and calibration")
        print("- Q or ESC: Quit")
        print("- O: Rotate video 90 degrees clockwise")

        while True:
            self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap_side.read()
            if not ret:
                print("Frame read failed or end of side video.")
                break

            display_frame = self.transform_frame(frame.copy(), self.side_rotation)
            self.draw_trajectory(display_frame, is_side=True)

            if self.side_calibration_active:
                points = len(self.side_calibration[1]) if self.side_calibration else 0
                status = f"Side Calibration - Click point {points + 1}/2 for ball diameter in Frame {self.side_frame_for_main_frame1}"
            elif self.tracking_active:
                status = "BALL TRACKING ACTIVE - Click on ball, press C to calibrate in first frame"
            else:
                status = "PAUSED - Press SPACE to track, C to calibrate after first point"

            self.draw_text(display_frame, status, (10, 40), font_scale=1.1)
            self.draw_text(display_frame, f"Frame: {self.current_frame}/{self.total_frames_side-1}", (10, 80), font_scale=0.9)
            self.draw_text(display_frame, f"Side Points: {len(self.side_positions)}", (10, 110), font_scale=0.9)
            self.draw_text(display_frame, f"Main Frame 1 = Side Frame {'Not set' if self.side_frame_for_main_frame1 is None else self.side_frame_for_main_frame1}", (10, 140), font_scale=0.9)
            self.draw_text(display_frame, f"Focal Length: {'{:.2f} px'.format(self.side_focal_length_px) if self.side_focal_length_px is not None else 'Not set'}", (10, 170), font_scale=0.9)
            self.draw_text(display_frame, "S = Save Excel", (10, 200), font_scale=0.9)

            cv2.imshow(self.window_name_side, display_frame)
            key = cv2.waitKey(10) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                if self.side_frame_for_main_frame1 is None:
                    print("Error: Track the first ball position before calibrating.")
                elif self.current_frame != self.side_frame_for_main_frame1:
                    print(f"Error: Side calibration must occur in first tracked frame ({self.side_frame_for_main_frame1}).")
                elif self.side_calibration and len(self.side_calibration[1]) == 2:
                    print("Side calibration already completed.")
                else:
                    self.side_calibration_active = not self.side_calibration_active
                    if self.side_calibration_active:
                        if self.side_calibration is None or len(self.side_calibration[1]) == 2:
                            print(f"Starting side calibration in frame {self.side_frame_for_main_frame1}.")
                        else:
                            print("Continuing side calibration.")
                    else:
                        print("Side calibration paused.")
            elif key == ord(' '):
                self.tracking_active = not self.tracking_active
                print(f"{'Ball tracking started' if self.tracking_active else 'Ball tracking paused'}")
            elif key in (ord('a'), 81):
                self.previous_frame(is_side=True)
            elif key in (ord('d'), 83):
                self.advance_frame(is_side=True)
            elif key == ord('s'):
                self.save_to_excel()
            elif key == ord('o'):  # 'o' for rotate
                self.side_rotation = (self.side_rotation + 90) % 360
                print(f"Side view rotated to {self.side_rotation} degrees")
            elif key == ord('r'):
                self.side_positions = []
                self.side_calibration = None
                self.side_focal_length_px = None
                self.side_frame_for_main_frame1 = None
                print("Side view tracking, calibration, and frame mapping reset.")

        self.cap_side.release()
        cv2.destroyAllWindows()

    # ---------------------- Main Entry ---------------------- #
    def run_tracker(self, video_path):
        if self.run_main_tracker(video_path):
            self.run_side_tracker()

def main():
    tracker = CricketBallTracker()
    video_path = input("Enter the path to your main cricket video file (bird's eye view): ").strip().strip('"')
    if not video_path:
        print("No main video path provided. Exiting.")
        return
    tracker.run_tracker(video_path)

if __name__ == "__main__":
    main()