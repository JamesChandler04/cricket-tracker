import cv2
import numpy as np
import os
import math
from typing import Optional
from helpers import BallData, Coord

tracking_folder = "tracking_output"

AVERAGE_BALL_RADIUS = 90

class BallFinder:
    def find_ball(self, frame, background_frame) -> Optional[BallData]:
        if frame is None or background_frame is None:
            return None

        if frame.shape != background_frame.shape:
            background_frame = cv2.resize(background_frame, (frame.shape[1], frame.shape[0]))

        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        blurred_bg = cv2.GaussianBlur(background_frame, (7, 7), 0)

        grey = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        background_grey = cv2.cvtColor(blurred_bg, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        background_hsv = cv2.cvtColor(blurred_bg, cv2.COLOR_BGR2HSV)

        diff_v = cv2.absdiff(hsv[:, :, 2], background_hsv[:, :, 2])
        _, motion_mask = cv2.threshold(diff_v, 15, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

        lower_red1 = np.array([0, 140, 100])
        upper_red1 = np.array([8, 255, 255])
        lower_red2 = np.array([172, 140, 100])
        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        combined_mask = cv2.bitwise_and(motion_mask, red_mask)

        h, w = frame.shape[:2]
        combined_mask[int(h * 0.75):, :] = 0

        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        cv2.imwrite(f"{tracking_folder}/motion_mask.jpg", motion_mask)
        cv2.imwrite(f"{tracking_folder}/red_mask.jpg", red_mask)
        cv2.imwrite(f"{tracking_folder}/combined_mask.jpg", combined_mask)

        best_circle = None
        grey_blurred = cv2.medianBlur(grey, 5)
        circles = cv2.HoughCircles(
            grey_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=200,
            param2=25,
            minRadius=25,
            maxRadius=180
        )

        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            for x, y, radius in circles:
                if x - radius < 0 or y - radius < 0 or x + radius >= w or y + radius >= h:
                    continue

                if radius < 20:
                    continue

                circle_mask = np.zeros_like(combined_mask)
                cv2.circle(circle_mask, (x, y), radius, 255, -1)
                overlap = cv2.countNonZero(cv2.bitwise_and(circle_mask, combined_mask))
                overlap_ratio = overlap / (np.pi * radius * radius)

                if overlap_ratio < 0.25:
                    continue

                score = overlap_ratio - abs(radius - AVERAGE_BALL_RADIUS) / 90.0
                if best_circle is None or score > best_circle[3]:
                    best_circle = (x, y, radius, score)

        if best_circle is not None:
            centre_x, centre_y, radius, _ = best_circle
        else:
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500 or area > 20000:
                    continue

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                radius = float(radius)
                if radius < 20 or radius > 180:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter <= 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.68:
                    continue

                mask = np.zeros_like(combined_mask)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                overlap = cv2.countNonZero(cv2.bitwise_and(mask, combined_mask))
                fill_ratio = overlap / area
                if fill_ratio < 0.55:
                    continue

                score = fill_ratio + circularity * 1.5 - abs(radius - AVERAGE_BALL_RADIUS) / 70.0
                candidates.append((cnt, x, y, radius, score))

            if not candidates:
                return None

            _, x, y, radius, _ = max(candidates, key=lambda c: c[4])
            centre_x, centre_y = int(round(x)), int(round(y))
            radius = int(round(radius))

        centre_x, centre_y, radius = int(round(centre_x)), int(round(centre_y)), int(round(radius))

        original_frame = frame.copy()
        cv2.circle(frame, (centre_x, centre_y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (centre_x, centre_y), 2, (0, 0, 255), 3)
        cv2.imwrite(f"{tracking_folder}/detected_ball.jpg", frame)

        os.makedirs(tracking_folder, exist_ok=True)
        margin = int(radius * 0.25)
        x0 = max(0, centre_x - radius - margin)
        y0 = max(0, centre_y - radius - margin)
        x1 = min(original_frame.shape[1], centre_x + radius + margin)
        y1 = min(original_frame.shape[0], centre_y + radius + margin)
        cropped_ball = original_frame[y0:y1, x0:x1].copy()

        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.circle(mask, (centre_x - x0, centre_y - y0), radius, 255, -1)
        cropped_ball[mask == 0] = 0

        cv2.imwrite(f"{tracking_folder}/cropped_ball.jpg", cropped_ball)

        return BallData(
            top_left=Coord(centre_x - radius, centre_y - radius),
            bottom_right=Coord(centre_x + radius, centre_y + radius),
            centre=Coord(centre_x, centre_y),
            seam_start=Coord(-1, -1),
            seam_end=Coord(-1, -1),
            seam_angle=-1.0
        )

    def find_seam(self, ball_data: BallData, frame) -> Optional[BallData]:
        if frame is None or ball_data is None:
            return None

        x0 = max(0, ball_data.top_left.x)
        y0 = max(0, ball_data.top_left.y)
        x1 = min(frame.shape[1], ball_data.bottom_right.x)
        y1 = min(frame.shape[0], ball_data.bottom_right.y)

        if x1 <= x0 or y1 <= y0:
            return None

        roi = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # White seam mask: bright pixels with low saturation
        # Widened ranges to capture fainter / slightly off-white seams
        lower_white = np.array([0, 0, 140])   # allow slightly darker/brighter range
        upper_white = np.array([180, 160, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        os.makedirs(tracking_folder, exist_ok=True)
        cv2.imwrite(f"{tracking_folder}/seam_white_mask.jpg", white_mask)

        edges = cv2.Canny(white_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=15, maxLineGap=8)

        if lines is None:
            return None

        best_line = None
        best_length = 0
        for x_start, y_start, x_end, y_end in lines.reshape(-1, 4):
            length = np.hypot(x_end - x_start, y_end - y_start)
            if length > best_length:
                best_length = length
                best_line = (x_start, y_start, x_end, y_end)

        if best_line is None:
            return None

        x_start, y_start, x_end, y_end = best_line

        # Compute seam angle from the found line (ROI coordinates)
        seam_angle = math.degrees(math.atan2(y_end - y_start, x_end - x_start))

        # Determine circle centre and radius in full-image coordinates
        try:
            centre_full = ball_data.centre
            radius_full = int((ball_data.bottom_right.x - ball_data.top_left.x) / 2)
        except Exception:
            # Fallback: estimate from bounding box
            centre_full = Coord(x=(x0 + x1) // 2, y=(y0 + y1) // 2)
            radius_full = int(min((x1 - x0), (y1 - y0)) / 2)

        # Convert circle centre to ROI coordinates
        cx = centre_full.x - x0
        cy = centre_full.y - y0
        r = radius_full

        # Parametric line: p(t) = p1 + t*(d), p1=(x_start,y_start), d=(dx,dy)
        dx = x_end - x_start
        dy = y_end - y_start
        a = dx * dx + dy * dy
        b = 2 * (dx * (x_start - cx) + dy * (y_start - cy))
        c = (x_start - cx) ** 2 + (y_start - cy) ** 2 - r * r

        seam_full_start = None
        seam_full_end = None

        disc = b * b - 4 * a * c
        if a != 0 and disc >= 0:
            sqrt_disc = math.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)

            p1x = x_start + t1 * dx
            p1y = y_start + t1 * dy
            p2x = x_start + t2 * dx
            p2y = y_start + t2 * dy

            # Map to full-image coords
            seam_full_pt1 = (int(round(x0 + p1x)), int(round(y0 + p1y)))
            seam_full_pt2 = (int(round(x0 + p2x)), int(round(y0 + p2y)))

            # Assign seam endpoints across the ball (both intersection points)
            seam_full_start = Coord(x=seam_full_pt1[0], y=seam_full_pt1[1])
            seam_full_end = Coord(x=seam_full_pt2[0], y=seam_full_pt2[1])

        # If intersections not found fall back to original short line (map to full coords)
        if seam_full_start is None or seam_full_end is None:
            seam_full_start = Coord(x=x0 + x_start, y=y0 + y_start)
            seam_full_end = Coord(x=x0 + x_end, y=y0 + y_end)

        ball_data.seam_start = seam_full_start
        ball_data.seam_end = seam_full_end
        ball_data.seam_angle = seam_angle
        # --- Save images: cropped (masked) with seam line, and full image with circle+seam ---
        try:
            os.makedirs(tracking_folder, exist_ok=True)

            # Cropped masked image (outside circle black) with seam line
            cropped = roi.copy()
            # compute radius from bounding box
            try:
                centre = ball_data.centre
                radius = int((ball_data.bottom_right.x - ball_data.top_left.x) / 2)
            except Exception:
                centre = centre_full
                radius = radius_full

            mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.circle(mask, (centre.x - x0, centre.y - y0), radius, 255, -1)
            cropped_masked = cropped.copy()
            cropped_masked[mask == 0] = 0
            # draw the extended seam on the cropped masked image (map full-image seam pts back to ROI coords)
            s1 = (int(ball_data.seam_start.x - x0), int(ball_data.seam_start.y - y0))
            s2 = (int(ball_data.seam_end.x - x0), int(ball_data.seam_end.y - y0))
            cv2.line(cropped_masked, s1, s2, (0, 255, 255), 2)
            cv2.imwrite(f"{tracking_folder}/cropped_ball_with_seam.jpg", cropped_masked)

            # Full image with detected circle and seam
            full_with_seam = frame.copy()
            # draw circle (use computed centre/radius)
            cv2.circle(full_with_seam, (centre.x, centre.y), radius, (0, 255, 0), 2)
            # draw seam on full image (yellow)
            cv2.line(full_with_seam, (ball_data.seam_start.x, ball_data.seam_start.y), (ball_data.seam_end.x, ball_data.seam_end.y), (0, 255, 255), 2)
            cv2.imwrite(f"{tracking_folder}/detected_ball_with_seam.jpg", full_with_seam)
        except Exception:
            pass

        return ball_data



tester = BallFinder()

curr_frame = cv2.imread("current_frame.jpg")
background_frame = cv2.imread("background_frame.jpg")

ball_data = tester.find_ball(curr_frame, background_frame)
if ball_data:
    print(tester.find_seam(ball_data, curr_frame))
else:
    print("No ball found.")