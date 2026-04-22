import cv2
from enum import Enum

class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

class Drawers:
    def draw_text(self, frame, text, position, font_scale=1.2, thickness=3):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = position
        cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, Color.WHITE.value, thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, font_scale, Color.BLACK.value, thickness, cv2.LINE_AA)

    def draw_main_trajectory(self, frame, frame_positions, current_frame, frame_width, seam_points, seam_measurements, calibrations):
        if len(frame_positions) > 1:
            for i in range(1, len(frame_positions)):
                prev = frame_positions[i - 1]
                curr = frame_positions[i]
                cv2.line(frame, (prev[1], prev[2]), (curr[1], curr[2]), (0, 255, 0), 2)

        for i, (frame_num, x, y, _) in enumerate(frame_positions):
            color = (0, 0, 255) if frame_num == current_frame else (255, 0, 0)
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.putText(frame, f"{i}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if len(seam_points) >= 1:
            for i, (x, y) in enumerate(seam_points):
                cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)
                cv2.putText(frame, f"S{i+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if len(seam_points) == 2:
                cv2.line(frame, seam_points[0], seam_points[1], (255, 255, 0), 2)

        current_seam_angle = None
        for frame_num, angle in seam_measurements:
            if frame_num == current_frame and len(seam_points) < 2:
                current_seam_angle = angle
                break
        if current_seam_angle is not None and len(seam_points) < 2:
            cv2.putText(frame, f"Seam Angle: {current_seam_angle:.2f}°",
                        (frame_width // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        for i, (frame_num, points) in enumerate(calibrations):
            for j, (x, y) in enumerate(points):
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(frame, f"D{i+1}{j+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if len(points) == 2:
                cv2.line(frame, points[0], points[1], (0, 255, 255), 2)
        
    def draw_side_trajectory(self, frame, side_positions, current_frame, side_calibration):
        if len(side_positions) > 1:
            for i in range(1, len(side_positions)):
                prev = side_positions[i - 1]
                curr = side_positions[i]
                cv2.line(frame, (prev[1], prev[2]), (curr[1], curr[2]), (0, 255, 0), 2)

        for i, (frame_num, x, y, _) in enumerate(side_positions):
            color = (0, 0, 255) if frame_num == current_frame else (255, 0, 0)
            cv2.circle(frame, (x, y), 2, color, -1)
            cv2.putText(frame, f"{i}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if side_calibration and len(side_calibration[1]) >= 1:
            for j, (x, z) in enumerate(side_calibration[1]):
                cv2.circle(frame, (x, z), 5, (0, 255, 255), -1)
                cv2.putText(frame, f"D{j+1}", (x + 10, z - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if len(side_calibration[1]) == 2:
                cv2.line(frame, side_calibration[1][0], side_calibration[1][1], (0, 255, 255), 2)