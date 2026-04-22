import math

class Checker:
    def _check_calibration_difference(self, calibrations, ball_diameter_m):
        if len(calibrations) != 2:
            return None, False
        meters_per_pixel_values = []
        for _, points in calibrations:
            (x1, y1), (x2, y2) = points
            pixel_distance = math.hypot(x2 - x1, y2 - y1)
            if pixel_distance < 1.0:
                return None, False
            mpp = ball_diameter_m / pixel_distance
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
    
    def _check_seam_wobble(self, seam_measurements):
        if not seam_measurements:
            return None, False, 0.0
        if len(seam_measurements) == 1:
            return seam_measurements[0][1], False, 0.0

        sorted_measurements = sorted(seam_measurements, key=lambda x: x[0])
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