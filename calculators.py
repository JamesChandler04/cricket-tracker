import math

class Calculators:
    def _calculate_meters_per_pixel(self, calibrations, ball_diameter_m, meters_per_pixel):
        if not calibrations or len(calibrations[-1][1]) != 2:
            return
        meters_per_pixel_values = []
        for frame_num, points in calibrations:
            (x1, y1), (x2, y2) = points
            pixel_distance = math.hypot(x2 - x1, y2 - y1)
            if pixel_distance < 1.0:
                print(f"Error: Calibration points in frame {frame_num} are too close.")
                calibrations.pop()
                return
            mpp = ball_diameter_m / pixel_distance
            meters_per_pixel_values.append(mpp)
            print(f"Main calibration in frame {frame_num}: 1 px = {mpp:.6f} m")
        meters_per_pixel = sum(meters_per_pixel_values) / len(meters_per_pixel_values)
        print(f"Main average meters per pixel: {meters_per_pixel:.6f} m")
        return calibrations, meters_per_pixel
    
    def _calculate_side_focal_length(self, y_positions, side_calibration, side_frame_for_main_frame1, ball_diameter_m, side_focal_length_px):
        '''
        Returns optional[side calibration], optional[side focal length pixels]
        '''
        
        if not side_calibration or len(side_calibration[1]) != 2:
            return side_calibration, None
        frame_num, points = side_calibration
        (x1, z1), (x2, z2) = points
        pixel_distance = math.hypot(x2 - x1, z2 - z1)
        if pixel_distance < 1.0:
            print(f"Error: Side calibration points in frame {frame_num} are too close.")
            return None, None
        frame_idx = frame_num - side_frame_for_main_frame1 + 1
        if frame_idx < 1 or frame_idx > len(y_positions):
            print(f"Error: Side calibration frame {frame_num} has no corresponding Y-position.")
            return None, None
        y_m = y_positions[frame_idx - 1]
        if y_m <= 0:
            print(f"Error: Invalid Y-position {y_m:.3f}m for side calibration frame {frame_num}.")
            return None, None
        side_focal_length_px = pixel_distance * y_m / ball_diameter_m
        print(f"Side calibration in frame {frame_num}: Focal length = {side_focal_length_px:.2f} px (y_m={y_m:.3f}m, d_px={pixel_distance:.2f}px)")
        return side_calibration, side_focal_length_px
    
    def _calculate_seam_angle(self, seam_points, seam_measurements, current_frame):
        if len(seam_points) != 2:
            return seam_measurements, seam_points
        (x1, y1), (x2, y2) = seam_points
        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dx, -dy)) % 360
        seam_measurements.append((current_frame, angle))
        seam_points = []
        print(f"Seam angle calculated for frame {current_frame}: {angle:.2f} degrees")
        return seam_measurements, seam_points
    
    def _calculate_initial_trajectory(self, frame_positions, frame_height, meters_per_pixel):
        if len(frame_positions) < 2:
            return None
        _, x1, y1, _ = frame_positions[0]
        _, x2, y2, _ = frame_positions[1]
        dx = (x2 - x1) * meters_per_pixel
        dy = (frame_height - y2) - (frame_height - y1)
        dy *= meters_per_pixel
        angle = math.degrees(math.atan2(dx, -dy)) % 360
        if angle > 180:
            angle -= 180
        return angle