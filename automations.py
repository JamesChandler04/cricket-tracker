import cv2
import numpy as np
import os
import math
import time
from typing import Optional
from helpers import BallData, Coord, Video
import sys
from enum import Enum
from dataclasses import dataclass
import yaml
import multiprocessing as mp

import log_bridge

'''
TODO:
- Downsample frames for faster processing.
- Only save images on ball found.
'''

# Output folders for debug and tracking images
top_down_tracking_folder = "top_down_tracking_output"
side_on_tracking_folder = "side_on_tracking_output"

CONFIG_PATH = "config.yml"

# Set to 1 to process every frame. Skips frames until ball is found, then backtracks and processes every frame until the last frame with the ball.
TOP_DOWN_FRAME_SKIP = 1
SIDE_ON_FRAME_SKIP = 1

# Ball size (radius in pixels)
TOP_DOWN_BALL_RADIUS = 69   # measured: locked-on ball is 138 px across -> r = 69
SIDE_ON_BALL_RADIUS  = 30

# Strictness knob for the top-down size gate. Smaller = stricter.
# Both detection paths (Hough + contour) only accept radii within
# TOP_DOWN_BALL_RADIUS +/- TOP_DOWN_MAX_DIFFERENCE.
TOP_DOWN_MAX_DIFFERENCE = 15
TOP_DOWN_MIN_RADIUS = TOP_DOWN_BALL_RADIUS - TOP_DOWN_MAX_DIFFERENCE   # 54
TOP_DOWN_MAX_RADIUS = TOP_DOWN_BALL_RADIUS + TOP_DOWN_MAX_DIFFERENCE   # 84

# Thickness of the detection circle drawn on debug images (pixels)
DETECTION_CIRCLE_THICKNESS = 4

# Ball colour HSV ranges (red). Kept LOOSE on purpose so the full ball is captured
# even when motion blur desaturates its edges. Skin (red in hue but pale) is NOT
# rejected here any more -- that happens per-candidate via the blob-level median
# saturation test (TOP_DOWN_MIN_MEDIAN_SAT) in find_ball.
TOP_DOWN_BALL_COLOR_RANGES = [
    (np.array([0,   70, 20]), np.array([14,  255, 255])),
    (np.array([166, 70, 20]), np.array([180, 255, 255]))
]

# A real ball blob is vividly saturated THROUGHOUT (median S ~140-190); the bowler's
# skin is dull (median S <=117). A candidate whose median saturation is below this is
# rejected as skin. This survives motion blur -- blur lowers individual pixels' S, but
# not the blob median.
TOP_DOWN_MIN_MEDIAN_SAT = 120

# Seam colour HSV range (white)
SEAM_COLOUR_RANGE = (np.array([0, 0, 100]), np.array([180, 70, 255]))


SIDE_ON_BALL_COLOR_RANGES = [
    (np.array([0,   120,  80]), np.array([10,  255, 255])),
    (np.array([170, 120,  80]), np.array([180, 255, 255]))
]


def _load_bounding_box(view: str) -> Optional[tuple[int, int, int, int]]:
    """
    Read bounding box for 'top_down' or 'side_on' from config.yml.
    Returns (x_start, y_start, x_end, y_end) or None if not fully set.
    """
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}
    bb = config.get("bounding_boxes", {}).get(view, {})
    tl = bb.get("top_left", {})
    br = bb.get("bottom_right", {})
    xs, ys = tl.get("x"), tl.get("y")
    xe, ye = br.get("x"), br.get("y")
    if None in (xs, ys, xe, ye):
        return None
    return int(xs), int(ys), int(xe), int(ye)


def _apply_bounding_box(frame, background_frame, bbox):
    """
    Crop both frames to the bounding box region.
    Returns (cropped_frame, cropped_bg, x_offset, y_offset).
    Offsets are used to map detected coords back to full-image space.
    """
    xs, ys, xe, ye = bbox
    h, w = frame.shape[:2]
    xs = max(0, min(xs, w - 1))
    ys = max(0, min(ys, h - 1))
    xe = max(xs + 1, min(xe, w))
    ye = max(ys + 1, min(ye, h))
    return frame[ys:ye, xs:xe], background_frame[ys:ye, xs:xe], xs, ys


@dataclass
class TrackedBallDataPoint:
    frame_number: int
    data: BallData

class BallPosition(Enum):
    BEFORE_FRAME = 0
    IN_FRAME = 1
    AFTER_FRAME = 2

class TopDownBallFinder:
    def new_get_ball_data(self, video: Video) -> list[TrackedBallDataPoint]:
        workers = mp.Pool(processes=mp.cpu_count() - 1)
        return []

    def get_ball_data(self, video: Video) -> list[TrackedBallDataPoint]:
        ball_data_points: list[TrackedBallDataPoint] = []

        # Skip frames until first ball is found
        while video.get_current_frame_number() < video.total_frames - 1:
            video.change_frame(TOP_DOWN_FRAME_SKIP - 1)
            st = time.time()
            background_frame = video.get_current_frame()
            video.change_frame(1)
            current_frame = video.get_current_frame()
            ball_data = self.find_ball(current_frame, background_frame)
            print(f"Checking frame {video.current_frame} for ball ({(time.time() - st)*1000:.2f}ms)")
            if ball_data:
                # Backtrack to last frame before ball was found to start processing from there
                video.change_frame(-(TOP_DOWN_FRAME_SKIP + 1))
                print(f"Ball found in frame {video.current_frame}, starting processing from frame {video.current_frame - TOP_DOWN_FRAME_SKIP - 1}.")
                break
            print(f"No ball found in frame {video.current_frame}, skipping {TOP_DOWN_FRAME_SKIP - 1} frames.")

        ball_position = BallPosition.BEFORE_FRAME

        # Process every frame until ball is no longer found, then stop alltogether.
        while ball_position is not BallPosition.AFTER_FRAME and video.get_current_frame_number() < video.total_frames - 1:
            st = time.time()
            background_frame = video.get_current_frame()
            video.change_frame(1)
            current_frame = video.get_current_frame()
            ball_data = self.find_ball(current_frame, background_frame)
            if ball_data:
                ball_data = self.find_seam(ball_data, current_frame)
            if ball_data:
                ball_data_points.append(TrackedBallDataPoint(frame_number=video.current_frame, data=ball_data))
                cv2.imwrite(f"{top_down_tracking_folder}/frame_{video.current_frame:04d}.jpg", current_frame)
                ball_position = BallPosition.IN_FRAME
            else:
                if ball_position == BallPosition.IN_FRAME:
                    continue
                    ball_position = BallPosition.AFTER_FRAME
                    print(f"Ball lost after frame {video.current_frame}, stopping processing.")
            et = time.time()
            print(f"Top Down Frame {video.current_frame} ({(et - st)*1000:.2f}ms): {ball_data}")
        return ball_data_points

    def find_ball(self, frame, background_frame) -> Optional[BallData]:
        if frame is None or background_frame is None:
            return None

        if frame.shape != background_frame.shape:
            background_frame = cv2.resize(background_frame, (frame.shape[1], frame.shape[0]))

        # ── Apply bounding box crop if configured ─────────────────────────────
        x_offset, y_offset = 0, 0
        bbox = _load_bounding_box("top_down")
        if bbox:
            frame, background_frame, x_offset, y_offset = _apply_bounding_box(
                frame, background_frame, bbox)

        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        blurred_bg = cv2.GaussianBlur(background_frame, (7, 7), 0)

        grey = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        background_hsv = cv2.cvtColor(blurred_bg, cv2.COLOR_BGR2HSV)

        diff_v = cv2.absdiff(hsv[:, :, 2], background_hsv[:, :, 2])
        _, motion_mask = cv2.threshold(diff_v, 15, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

        # Build colour mask from configurable ranges
        color_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in TOP_DOWN_BALL_COLOR_RANGES:
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))

        combined_mask = cv2.bitwise_and(motion_mask, color_mask)

        h, w = frame.shape[:2]
        combined_mask[int(h * 0.75):, :] = 0

        # The tight maroon range excludes the bright white seam, leaving a gap
        # across the middle of the ball. A wide close (sized to the seam band)
        # bridges it so the ball stays one round blob; the 9x9 open won't re-split it.
        seam_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, seam_kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        cv2.imwrite(f"{top_down_tracking_folder}/motion_mask.jpg", motion_mask)
        cv2.imwrite(f"{top_down_tracking_folder}/color_mask.jpg", color_mask)
        cv2.imwrite(f"{top_down_tracking_folder}/combined_mask.jpg", combined_mask)

        best_circle = None
        grey_blurred = cv2.medianBlur(grey, 5)
        circles = cv2.HoughCircles(
            grey_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=200,
            param2=15,
            minRadius=TOP_DOWN_MIN_RADIUS,
            maxRadius=TOP_DOWN_MAX_RADIUS
        )

        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            for x, y, radius in circles:
                if x - radius < 0 or y - radius < 0 or x + radius >= w or y + radius >= h:
                    continue
                if abs(radius - TOP_DOWN_BALL_RADIUS) > TOP_DOWN_MAX_DIFFERENCE:
                    continue
                circle_mask = np.zeros_like(combined_mask)
                cv2.circle(circle_mask, (x, y), radius, 255, -1)
                overlap = cv2.countNonZero(cv2.bitwise_and(circle_mask, combined_mask))
                overlap_ratio = overlap / (np.pi * radius * radius)
                if overlap_ratio < 0.25:
                    continue
                # Skin reject: the ball is vividly saturated, skin is pale.
                ball_px = cv2.bitwise_and(circle_mask, combined_mask)
                if np.median(hsv[:, :, 1][ball_px > 0]) < TOP_DOWN_MIN_MEDIAN_SAT:
                    continue
                score = overlap_ratio - abs(radius - TOP_DOWN_BALL_RADIUS) / 90.0
                if best_circle is None or score > best_circle[3]:
                    best_circle = (x, y, radius, score)

        if best_circle is not None:
            centre_x, centre_y, radius, _ = best_circle
        else:
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < np.pi * TOP_DOWN_MIN_RADIUS ** 2 * 0.5 or area > np.pi * TOP_DOWN_MAX_RADIUS ** 2 * 1.3:
                    continue
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                radius = float(radius)
                if abs(radius - TOP_DOWN_BALL_RADIUS) > TOP_DOWN_MAX_DIFFERENCE:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter <= 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.45:
                    continue
                mask = np.zeros_like(combined_mask)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                overlap = cv2.countNonZero(cv2.bitwise_and(mask, combined_mask))
                fill_ratio = overlap / area
                if fill_ratio < 0.55:
                    continue
                # Skin reject: the ball is vividly saturated, skin is pale.
                if np.median(hsv[:, :, 1][cv2.bitwise_and(mask, combined_mask) > 0]) < TOP_DOWN_MIN_MEDIAN_SAT:
                    continue
                score = fill_ratio + circularity * 1.5 - abs(radius - TOP_DOWN_BALL_RADIUS) / 70.0
                candidates.append((cnt, x, y, radius, score))
            if not candidates:
                return None
            _, x, y, radius, _ = max(candidates, key=lambda c: c[4])
            centre_x, centre_y = int(round(x)), int(round(y))
            radius = int(round(radius))

        centre_x, centre_y, radius = int(round(centre_x)), int(round(centre_y)), int(round(radius))

        original_frame = frame.copy()
        cv2.circle(frame, (centre_x, centre_y), radius, (0, 255, 0), DETECTION_CIRCLE_THICKNESS)
        cv2.circle(frame, (centre_x, centre_y), 2, (0, 0, 255), 3)
        cv2.imwrite(f"{top_down_tracking_folder}/detected_ball.jpg", frame)

        margin = int(radius * 0.25)
        x0 = max(0, centre_x - radius - margin)
        y0 = max(0, centre_y - radius - margin)
        x1 = min(original_frame.shape[1], centre_x + radius + margin)
        y1 = min(original_frame.shape[0], centre_y + radius + margin)
        cropped_ball = original_frame[y0:y1, x0:x1].copy()
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.circle(mask, (centre_x - x0, centre_y - y0), radius, 255, -1)
        cropped_ball[mask == 0] = 0
        cv2.imwrite(f"{top_down_tracking_folder}/cropped_ball.jpg", cropped_ball)

        # Map local coords back to full-image space
        centre_x += x_offset
        centre_y += y_offset

        return BallData(
            top_left=Coord(centre_x - radius, centre_y - radius),
            bottom_right=Coord(centre_x + radius, centre_y + radius),
            centre=Coord(centre_x, centre_y),
            seam_start=Coord(-1, -1),
            seam_end=Coord(-1, -1),
            seam_angle=-1.0
        )

    def find_seam(self, ball_data: BallData, frame) -> Optional[BallData]:
        x0 = max(0, ball_data.top_left.x)
        y0 = max(0, ball_data.top_left.y)
        x1 = min(frame.shape[1], ball_data.bottom_right.x)
        y1 = min(frame.shape[0], ball_data.bottom_right.y)

        if x1 <= x0 or y1 <= y0:
            return None

        roi = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, SEAM_COLOUR_RANGE[0], SEAM_COLOUR_RANGE[1])

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        cv2.imwrite(f"{top_down_tracking_folder}/seam_white_mask.jpg", white_mask)

        edges = cv2.Canny(white_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=15, maxLineGap=8)

        if lines is None:
            ball_data.seam_angle = -1.0
            return ball_data

        best_line = None
        best_length = 0
        for x_start, y_start, x_end, y_end in lines.reshape(-1, 4):
            length = np.hypot(x_end - x_start, y_end - y_start)
            if length > best_length:
                best_length = length
                best_line = (x_start, y_start, x_end, y_end)

        if best_line is None:
            ball_data.seam_angle = -1.0
            return ball_data

        x_start, y_start, x_end, y_end = best_line
        seam_angle = math.degrees(math.atan2(y_end - y_start, x_end - x_start))

        try:
            centre_full = ball_data.centre
            radius_full = int((ball_data.bottom_right.x - ball_data.top_left.x) / 2)
        except Exception:
            centre_full = Coord(x=(x0 + x1) // 2, y=(y0 + y1) // 2)
            radius_full = int(min((x1 - x0), (y1 - y0)) / 2)

        cx = centre_full.x - x0
        cy = centre_full.y - y0
        r = radius_full

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
            seam_full_start = Coord(x=int(round(x0 + p1x)), y=int(round(y0 + p1y)))
            seam_full_end   = Coord(x=int(round(x0 + p2x)), y=int(round(y0 + p2y)))

        if seam_full_start is None or seam_full_end is None:
            seam_full_start = Coord(x=x0 + x_start, y=y0 + y_start)
            seam_full_end   = Coord(x=x0 + x_end,   y=y0 + y_end)

        ball_data.seam_start = seam_full_start
        ball_data.seam_end   = seam_full_end
        ball_data.seam_angle = seam_angle

        try:
            cropped = roi.copy()
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
            s1 = (int(ball_data.seam_start.x - x0), int(ball_data.seam_start.y - y0))
            s2 = (int(ball_data.seam_end.x   - x0), int(ball_data.seam_end.y   - y0))
            cv2.line(cropped_masked, s1, s2, (0, 255, 255), 2)
            cv2.imwrite(f"{top_down_tracking_folder}/cropped_ball_with_seam.jpg", cropped_masked)

            full_with_seam = frame.copy()
            cv2.circle(full_with_seam, (centre.x, centre.y), radius, (0, 255, 0), DETECTION_CIRCLE_THICKNESS)
            cv2.line(full_with_seam,
                     (ball_data.seam_start.x, ball_data.seam_start.y),
                     (ball_data.seam_end.x,   ball_data.seam_end.y),
                     (0, 255, 255), 2)
            cv2.imwrite(f"{top_down_tracking_folder}/detected_ball_with_seam.jpg", full_with_seam)
        except Exception:
            pass

        return ball_data


class SideOnBallFinder:
    def get_ball_data(self, video: Video) -> list[TrackedBallDataPoint]:
        ball_data_points: list[TrackedBallDataPoint] = []

        while video.get_current_frame_number() < video.total_frames - 1:
            video.change_frame(SIDE_ON_FRAME_SKIP - 1)
            st = time.time()
            background_frame = video.get_current_frame()
            video.change_frame(1)
            current_frame = video.get_current_frame()
            ball_data = self.find_ball(current_frame, background_frame)
            print(f"Checking frame {video.current_frame} for ball ({(time.time() - st)*1000:.2f}ms)")
            if ball_data:
                video.change_frame(-(SIDE_ON_FRAME_SKIP + 1))
                print(f"Ball found in frame {video.current_frame}, starting processing from frame {video.current_frame - SIDE_ON_FRAME_SKIP - 1}.")
                break
            print(f"No ball found in frame {video.current_frame}, skipping {SIDE_ON_FRAME_SKIP - 1} frames.")

        ball_position = BallPosition.BEFORE_FRAME

        while ball_position is not BallPosition.AFTER_FRAME and video.get_current_frame_number() < video.total_frames - 1:
            st = time.time()
            background_frame = video.get_current_frame()
            video.change_frame(1)
            current_frame = video.get_current_frame()
            ball_data = self.find_ball(current_frame, background_frame)
            if ball_data:
                ball_data_points.append(TrackedBallDataPoint(frame_number=video.current_frame, data=ball_data))
                cv2.imwrite(f"{side_on_tracking_folder}/frame_{video.current_frame:04d}.jpg", current_frame)
                ball_position = BallPosition.IN_FRAME
            else:
                if ball_position == BallPosition.IN_FRAME:
                    ball_position = BallPosition.AFTER_FRAME
                    print(f"Ball lost after frame {video.current_frame}, stopping processing.")
            et = time.time()
            print(f"Side On Frame {video.current_frame} ({(et - st)*1000:.2f}ms): {ball_data}")
        return ball_data_points

    def find_ball(self, current_frame, background_frame) -> Optional[BallData]:
        if current_frame is None or background_frame is None:
            return None

        if current_frame.shape != background_frame.shape:
            background_frame = cv2.resize(background_frame, (current_frame.shape[1], current_frame.shape[0]))

        # ── Apply bounding box crop if configured ─────────────────────────────
        x_offset, y_offset = 0, 0
        bbox = _load_bounding_box("side_on")
        if bbox:
            current_frame, background_frame, x_offset, y_offset = _apply_bounding_box(
                current_frame, background_frame, bbox)

        h, w = current_frame.shape[:2]

        # ── 1. Downscale for faster processing ────────────────────────────────
        PROCESS_WIDTH = 1280
        scale = PROCESS_WIDTH / w if w > PROCESS_WIDTH else 1.0
        if scale < 1.0:
            proc_w = PROCESS_WIDTH
            proc_h = int(h * scale)
            small    = cv2.resize(current_frame,   (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            small_bg = cv2.resize(background_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            small    = current_frame
            small_bg = background_frame
            proc_h, proc_w = h, w

        # ── 2. Motion mask ────────────────────────────────────────────────────
        blurred    = cv2.GaussianBlur(small,    (5, 5), 0)
        blurred_bg = cv2.GaussianBlur(small_bg, (5, 5), 0)

        hsv    = cv2.cvtColor(blurred,    cv2.COLOR_BGR2HSV)
        hsv_bg = cv2.cvtColor(blurred_bg, cv2.COLOR_BGR2HSV)

        diff_v = cv2.absdiff(hsv[:, :, 2], hsv_bg[:, :, 2])
        _, motion_mask = cv2.threshold(diff_v, 20, 255, cv2.THRESH_BINARY)

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        motion_mask = cv2.dilate(motion_mask, kernel_small, iterations=1)

        # ── 3. Colour mask ────────────────────────────────────────────────────
        color_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
        for lower, upper in SIDE_ON_BALL_COLOR_RANGES:
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))

        # ── 4. Combined mask ──────────────────────────────────────────────────
        combined_mask = cv2.bitwise_and(motion_mask, color_mask)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  kernel_small, iterations=1)

        cv2.imwrite(f"{side_on_tracking_folder}/motion_mask.jpg",   motion_mask)
        cv2.imwrite(f"{side_on_tracking_folder}/color_mask.jpg",    color_mask)
        cv2.imwrite(f"{side_on_tracking_folder}/combined_mask.jpg", combined_mask)

        # ── 5. Hough circle detection ─────────────────────────────────────────
        grey        = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        grey_median = cv2.medianBlur(grey, 5)

        scaled_radius = SIDE_ON_BALL_RADIUS * scale
        min_r = max(3, int(scaled_radius - 10 * scale))
        max_r = int(scaled_radius + 15 * scale)

        best_circle = None
        circles = cv2.HoughCircles(
            grey_median,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=100,
            param2=15,
            minRadius=min_r,
            maxRadius=max_r,
        )

        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            for x, y, radius in circles:
                if x - radius < 0 or y - radius < 0 or x + radius >= proc_w or y + radius >= proc_h:
                    continue
                circle_mask = np.zeros_like(combined_mask)
                cv2.circle(circle_mask, (x, y), radius, 255, -1)
                overlap       = cv2.countNonZero(cv2.bitwise_and(circle_mask, combined_mask))
                overlap_ratio = overlap / (np.pi * radius * radius)
                if overlap_ratio < 0.20:
                    continue
                score = overlap_ratio - abs(radius - scaled_radius) / max(scaled_radius, 1)
                if best_circle is None or score > best_circle[3]:
                    best_circle = (x, y, radius, score)

        # ── 6. Contour fallback ───────────────────────────────────────────────
        if best_circle is None:
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            min_area = np.pi * (min_r ** 2) * 0.4
            max_area = np.pi * (max_r ** 2) * 1.6
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                radius = float(radius)
                if radius < min_r or radius > max_r:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter <= 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.55:
                    continue
                mask = np.zeros_like(combined_mask)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                overlap    = cv2.countNonZero(cv2.bitwise_and(mask, combined_mask))
                fill_ratio = overlap / area
                if fill_ratio < 0.45:
                    continue
                score = fill_ratio + circularity * 1.5 - abs(radius - scaled_radius) / max(scaled_radius, 1)
                candidates.append((cnt, x, y, radius, score))
            if not candidates:
                return None
            _, x, y, radius, _ = max(candidates, key=lambda c: c[4])
            best_circle = (int(round(x)), int(round(y)), int(round(radius)), 0.0)

        # ── 7. Map from downscaled → crop-local → full-image coords ──────────
        sx, sy, sr, _ = best_circle
        if scale < 1.0:
            local_x = int(round(sx / scale))
            local_y = int(round(sy / scale))
            radius  = int(round(sr / scale))
        else:
            local_x, local_y, radius = int(round(sx)), int(round(sy)), int(round(sr))

        centre_x = local_x + x_offset
        centre_y = local_y + y_offset

        # ── 8. Debug output (drawn on the cropped frame at local coords) ──────
        debug_frame = current_frame.copy()
        cv2.circle(debug_frame, (local_x, local_y), radius, (0, 255, 0), DETECTION_CIRCLE_THICKNESS)
        cv2.circle(debug_frame, (local_x, local_y), 2,      (0, 0, 255), 3)
        cv2.imwrite(f"{side_on_tracking_folder}/detected_ball.jpg", debug_frame)

        margin = int(radius * 0.3)
        x0 = max(0, local_x - radius - margin)
        y0 = max(0, local_y - radius - margin)
        x1 = min(w, local_x + radius + margin)
        y1 = min(h, local_y + radius + margin)
        cropped = current_frame[y0:y1, x0:x1].copy()
        crop_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.circle(crop_mask, (local_x - x0, local_y - y0), radius, 255, -1)
        cropped[crop_mask == 0] = 0
        cv2.imwrite(f"{side_on_tracking_folder}/cropped_ball.jpg", cropped)

        return BallData(
            top_left=Coord(centre_x - radius, centre_y - radius),
            bottom_right=Coord(centre_x + radius, centre_y + radius),
            centre=Coord(centre_x, centre_y),
            seam_start=Coord(-1, -1),
            seam_end=Coord(-1, -1),
            seam_angle=-1.0,
        )


# tester = TopDownBallFinder()

# background = cv2.imread("background_frame.jpg")
# current = cv2.imread("current_frame.jpg")

# ball_data = tester.find_ball(current, background)
# print(ball_data)

# if ball_data:
#     print(tester.find_seam(ball_data, current))