import cv2
import numpy as np
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

        diff_grey = cv2.absdiff(grey, background_grey)
        _, motion_mask = cv2.threshold(diff_grey, 25, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
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

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
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

        cv2.circle(frame, (centre_x, centre_y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (centre_x, centre_y), 2, (0, 0, 255), 3)
        cv2.imwrite(f"{tracking_folder}/detected_ball.jpg", frame)

        return BallData(
            top_left=Coord(centre_x - radius, centre_y - radius),
            bottom_right=Coord(centre_x + radius, centre_y + radius),
            centre=Coord(centre_x, centre_y),
            seam_start=Coord(-1, -1),
            seam_end=Coord(-1, -1),
            seam_angle=-1.0
        )



tester = BallFinder()

curr_frame = cv2.imread("current_frame.jpg")
background_frame = cv2.imread("background_frame.jpg")

print(tester.find_ball(curr_frame, background_frame))