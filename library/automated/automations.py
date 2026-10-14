"""Earlier automatic ball detection for both videos (colour masks, background
subtraction, Hough circles and a saturation test to reject skin), with a top-down seam
finder.

Only the old programs use it; the current program has the ball clicked by hand.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import os
import math
import time
from typing import TYPE_CHECKING, Optional
from library.helpers import TopDownBallData, TopDownBallDataPoint, SideOnBallData, SideOnBallDataPoint, Coord, Video
import sys
from enum import Enum
from dataclasses import dataclass
import yaml
import multiprocessing as mp

from library import log_bridge # Lets print statements to be redirected to console in window.
from library import paths

if TYPE_CHECKING:
    from cv2.typing import MatLike

# Output folders for debug and tracking images
top_down_tracking_folder = str(paths.TOP_DOWN_TRACKING_DIR)
"""Folder for the top-down detector's debug images and the frames with the ball."""
side_on_tracking_folder = str(paths.SIDE_ON_TRACKING_DIR)
"""Folder for the side-on detector's debug images and the frames with the ball."""

FloatArray = npt.NDArray[np.float64]
"""Type of the float64 numpy arrays used in this module's type annotations."""

CONFIG_PATH = str(paths.CONFIG_PATH)
"""Path of config.yml, read for the optional search bounding boxes."""

# Set to 1 to process every frame. Skips frames until ball is found, then backtracks and processes every frame until the ball is lost
# (top-down: missed in TOP_DOWN_MAX_MISSED_FRAMES frames in a row; side-on: missed once).
TOP_DOWN_FRAME_SKIP = 1
"""Frame step while searching the top-down video for the ball; 1 checks every frame."""
SIDE_ON_FRAME_SKIP = 1
"""Frame step while searching the side-on video for the ball; 1 checks every frame."""
TOP_DOWN_MAX_MISSED_FRAMES = 5
"""Number of frames in a row without the ball, once it has been found, that ends the
top-down search.
"""

# Checks on one delivery's detections as a whole (check_track): the ball crosses the
# top-down view in a straight line, moving the same way at a steady speed.
TOP_DOWN_LINE_TOLERANCE = 0.6
"""Largest distance a detection's centre may lie from the ball's line, as a fraction of
the ball's radius. If it is further off then it is removed as a false detection.
"""
TOP_DOWN_STEP_TOLERANCE = 0.3
"""Largest fraction a detection's step from its neighbours, per frame, may differ from
the rest of the track's typical step before it is removed.
"""
TOP_DOWN_MIN_TRACK_DETECTIONS = 3
"""Fewest detections check_track needs to check them against each other. With fewer,
they are all kept.
"""

# Ball size (radius in pixels)
TOP_DOWN_BALL_RADIUS = 69
"""Expected top-down ball radius, in px."""
SIDE_ON_BALL_RADIUS  = 30
"""Expected side-on ball radius at full resolution, in px."""

CURCULARITY_THRESHOLD = 0.45
"""How circular a candidate must be to be accepted as the ball, in 0-1. Larger is stricter."""
FILL_RATIO_THRESHOLD = 0.55
"""Minimum fill ratio of a candidate to be accepted as the ball, in 0-1. Larger is stricter."""

TOP_DOWN_MAX_DIFFERENCE = 25
"""How far a top-down candidate's radius may be from TOP_DOWN_BALL_RADIUS, in px, in
both the Hough and contour paths. Smaller is stricter.
"""
TOP_DOWN_MIN_RADIUS = TOP_DOWN_BALL_RADIUS - TOP_DOWN_MAX_DIFFERENCE
"""Smallest top-down ball radius accepted, in px."""
TOP_DOWN_MAX_RADIUS = TOP_DOWN_BALL_RADIUS + TOP_DOWN_MAX_DIFFERENCE
"""Largest top-down ball radius accepted, in px."""

DETECTION_CIRCLE_THICKNESS = 4
"""Line thickness of the circle drawn round the found ball on debug images, in px."""

TOP_DOWN_BALL_COLOR_RANGES = [
    (np.array([0,   70, 20]), np.array([14,  255, 255])),
    (np.array([166, 70, 20]), np.array([180, 255, 255]))
]
"""Loose HSV ranges for the red ball in the top-down video, so edges desaturated by
motion blur are kept. Skin is rejected later, per candidate, by TOP_DOWN_MIN_MEDIAN_SAT.
"""

TOP_DOWN_MIN_MEDIAN_SAT = 120
"""Lowest median HSV saturation a top-down candidate may have before it is rejected as
skin. The ball measures about 140-190 and skin 117 or less.
"""

SEAM_COLOUR_RANGE = (np.array([0, 0, 100]), np.array([180, 70, 255]))
"""HSV range for the white seam inside the top-down ball, as (lower, upper)."""


SIDE_ON_BALL_COLOR_RANGES = [
    (np.array([0,   120,  80]), np.array([10,  255, 255])),
    (np.array([170, 120,  80]), np.array([180, 255, 255]))
]
"""HSV ranges for the red ball in the side-on video."""


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


def _apply_bounding_box(frame: MatLike, background_frame: MatLike, bbox: tuple[int, int, int, int]) -> tuple[MatLike, MatLike, int, int]:
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
class TrackCheck:
    """Which of one delivery's detections check_track kept, and why it removed the rest."""

    kept: list[int]
    """Indices of the detections kept, in frame order."""
    removed: dict[int, str]
    """Reason each removed detection was removed, by its index."""


def check_track(detections: list[tuple[int, float, float, float]]) -> TrackCheck:
    """Check one delivery's top-down detections, given as (frame, centre x, centre y,
    radius) in px, against each other, and return which to keep.

    A detection is removed if its centre is more than TOP_DOWN_LINE_TOLERANCE of a radius
    from the straight line most of the detections lie on, if it is out of order along
    that line, or if its step from its neighbours per frame differs from the rest of the
    track's by more than TOP_DOWN_STEP_TOLERANCE. With fewer than
    TOP_DOWN_MIN_TRACK_DETECTIONS detections, all are kept.
    """
    count = len(detections)
    if count < TOP_DOWN_MIN_TRACK_DETECTIONS:
        return TrackCheck(list(range(count)), {})
    frames = np.array([d[0] for d in detections], dtype=np.float64)
    points = np.array([(d[1], d[2]) for d in detections], dtype=np.float64)
    limit = TOP_DOWN_LINE_TOLERANCE * float(np.median([d[3] for d in detections]))
    removed: dict[int, str] = {}

    # The line: the one through two of the detections that the most others lie close to,
    # then refitted to all that lie close to it.
    on_line = np.zeros(count, dtype=bool)
    best_score = (-1, 0.0)
    for first in range(count):
        for second in range(first + 1, count):
            distances = np.abs(_distances_from_line(points, points[first], points[second] - points[first]))
            close = distances <= limit
            score = (int(close.sum()), -float(distances[close].sum()))
            if score > best_score:
                best_score, on_line = score, close
    for _ in range(3):
        centre, direction = _fitted_line(points[on_line])
        distances = _distances_from_line(points, centre, direction)
        refitted = np.abs(distances) <= limit
        if np.array_equal(refitted, on_line) or refitted.sum() < 2:
            break
        on_line = refitted
    for off_line in np.flatnonzero(~on_line):
        removed[int(off_line)] = f"{abs(distances[off_line]):.0f} px off the ball's line (limit {limit:.0f} px)"

    # Order: the ball moves one way along the line, so the longest run of detections
    # that moves steadily forward frame by frame is kept.
    along = (points - centre) @ direction
    candidates = [int(i) for i in np.flatnonzero(on_line)]
    candidates.sort(key=lambda i: frames[i])
    if len(candidates) >= 2 and np.corrcoef(frames[candidates], along[candidates])[0, 1] < 0:
        along = -along
    in_order = _longest_forward_run(candidates, along)
    for index in candidates:
        if index not in in_order:
            removed[index] = "out of order along the ball's line"

    # Steps: while a step per frame between neighbouring detections is too far from the
    # track's typical step, remove the detection that leaves the most even track.
    kept = in_order
    while len(kept) >= TOP_DOWN_MIN_TRACK_DETECTIONS and _unevenness(kept, frames, points) > TOP_DOWN_STEP_TOLERANCE:
        options = [kept[:position] + kept[position + 1:] for position in range(len(kept))]
        best = min(range(len(kept)), key=lambda position: _unevenness(options[position], frames, points))
        worst = kept[best]
        typical = float(np.median(_steps(options[best], frames, points)))
        neighbours = kept[max(best - 1, 0):best + 2]
        change = max(abs(step / typical - 1.0) for step in _steps(neighbours, frames, points))
        removed[worst] = (f"its step from its neighbours is {change:.0%} off the rest of the track's "
                          f"(limit {TOP_DOWN_STEP_TOLERANCE:.0%})")
        kept = options[best]
    return TrackCheck(kept, removed)


def _distances_from_line(points: FloatArray, centre: FloatArray, direction: FloatArray) -> FloatArray:
    """Return each point's signed distance from the line through centre along direction, in px."""
    unit = direction / np.linalg.norm(direction)
    offsets = points - centre
    return np.asarray(offsets[:, 1] * unit[0] - offsets[:, 0] * unit[1], dtype=np.float64)


def _fitted_line(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the centre and unit direction of the straight line closest to the points."""
    centre = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - centre)
    return np.asarray(centre, dtype=np.float64), np.asarray(axes[0], dtype=np.float64)


def _longest_forward_run(indices: list[int], along: FloatArray) -> list[int]:
    """Return the longest run of indices, in the order given, whose positions along the
    line keep increasing.
    """
    best: list[list[int]] = []
    for position, index in enumerate(indices):
        run = [index]
        for earlier in range(position):
            if along[indices[earlier]] < along[index] and len(best[earlier]) + 1 > len(run):
                run = best[earlier] + [index]
        best.append(run)
    return max(best, key=len) if best else []


def _unevenness(indices: list[int], frames: FloatArray, points: FloatArray) -> float:
    """Return how far the step per frame between neighbouring detections strays from the
    track's typical (median) step at most, as a fraction of it; 0 for one step or none.
    """
    steps = _steps(indices, frames, points)
    if len(steps) < 2:
        return 0.0
    typical = float(np.median(steps))
    return max(abs(step / typical - 1.0) for step in steps)


def _track_point(point: TopDownBallDataPoint) -> tuple[int, float, float, float]:
    """Return a found ball as (frame, centre x, centre y, radius) in px, for check_track."""
    data = point.data
    assert data.centre is not None and data.top_left is not None and data.bottom_right is not None
    return point.frame_number, data.centre.x, data.centre.y, (data.bottom_right.x - data.top_left.x) / 2


def _steps(indices: list[int], frames: FloatArray, points: FloatArray) -> list[float]:
    """Return the distance moved per frame between each detection and the next, in px."""
    return [float(np.linalg.norm(points[b] - points[a]) / (frames[b] - frames[a]))
            for a, b in zip(indices, indices[1:])]


class BallPosition(Enum):
    """Stage of a video scan: ball not found yet, found, or lost again."""
    BEFORE_FRAME = 0
    """Ball not found in any frame yet."""
    IN_FRAME = 1
    """Ball found in the latest frame."""
    AFTER_FRAME = 2
    """Ball lost after being found, which ends the scan."""

class TopDownBallFinder:
    """Automatic finder for the ball and its seam in the top-down video."""

    ball_found: bool
    """Whether track_ball has found the ball since the search started."""
    missed_in_a_row: int
    """Frames in a row track_ball has not found the ball in since it last found it."""

    def __init__(self) -> None:
        """Set up a finder that has not found the ball yet."""
        self.reset()

    def reset(self) -> None:
        """Start a new search: the ball not found yet and no frames missed."""
        self.ball_found = False
        self.missed_in_a_row = 0

    @property
    def ball_lost(self) -> bool:
        """Return whether the ball has been found and then missed in
        TOP_DOWN_MAX_MISSED_FRAMES frames in a row, which ends the search.
        """
        return self.ball_found and self.missed_in_a_row >= TOP_DOWN_MAX_MISSED_FRAMES

    def track_ball(self, frame: MatLike | None, background_frame: MatLike | None) -> Optional[TopDownBallData]:
        """Find the ball in the next frame of a search with find_ball, keeping count of the
        frames in a row it has been missed in since it was found, for ball_lost.
        """
        ball_data = self.find_ball(frame, background_frame)
        if ball_data is not None:
            self.ball_found = True
            self.missed_in_a_row = 0
        elif self.ball_found:
            self.missed_in_a_row += 1
        return ball_data

    def get_ball_data(self, video: Video) -> list[TopDownBallDataPoint]:
        """Scan the video from its current frame and return the ball and seam in each
        frame it is found in, until it has been missed in TOP_DOWN_MAX_MISSED_FRAMES
        frames in a row, saving those frames as images. Detections that do not fit the
        ball's track (check_track) are left out.
        """
        ball_data_points: list[TopDownBallDataPoint] = []

        # Skip frames until first ball is found
        while video.get_current_frame_number() < video.total_frames - 1:
            video.change_frame(TOP_DOWN_FRAME_SKIP - 1)
            st = time.time()
            background_frame = video.get_current_frame()
            video.change_frame(1)
            current_frame = video.get_current_frame()
            ball_data = self.find_ball(current_frame, background_frame)
            #print(f"Checking frame {video.current_frame} for ball ({(time.time() - st)*1000:.2f}ms)")
            if ball_data:
                # Backtrack to last frame before ball was found to start processing from there
                video.change_frame(-(TOP_DOWN_FRAME_SKIP + 1))
                #print(f"Ball found in frame {video.current_frame}, starting processing from frame {video.current_frame - TOP_DOWN_FRAME_SKIP - 1}.")
                break
            #print(f"No ball found in frame {video.current_frame}, skipping {TOP_DOWN_FRAME_SKIP - 1} frames.")

        # Process every frame until the ball has been missed in TOP_DOWN_MAX_MISSED_FRAMES frames in a row, then stop alltogether.
        self.reset()
        while not self.ball_lost and video.get_current_frame_number() < video.total_frames - 1:
            st = time.time()
            background_frame = video.get_current_frame()
            video.change_frame(1)
            current_frame = video.get_current_frame()
            ball_data = self.track_ball(current_frame, background_frame)
            if ball_data:
                assert current_frame is not None
                ball_data = self.find_seam(ball_data, current_frame)
            if ball_data:
                ball_data_points.append(TopDownBallDataPoint(frame_number=video.current_frame, data=ball_data))
                assert current_frame is not None
                cv2.imwrite(f"{top_down_tracking_folder}/frame_{video.current_frame:04d}.jpg", current_frame)
            #if self.ball_lost:
            #    print(f"Ball lost after frame {video.current_frame}, stopping processing.")
            et = time.time()
            #print(f"Top Down Frame {video.current_frame} ({(et - st)*1000:.2f}ms): {ball_data}")
        
        check = check_track([_track_point(point) for point in ball_data_points])
        for index, reason in sorted(check.removed.items()):
            print(f"Ball found in frame {ball_data_points[index].frame_number} left out: {reason}")
        ball_data_points = [ball_data_points[index] for index in check.kept]

        for point in ball_data_points:
            print(point)

        return ball_data_points

    def find_ball(self, frame: MatLike | None, background_frame: MatLike | None) -> Optional[TopDownBallData]:
        """Find the ball by motion against the background frame, red colour and circle
        shape, and return its box and centre in px (seam fields set to -1), or None.
        """
        if frame is None or background_frame is None:
            return None

        if frame.shape != background_frame.shape:
            background_frame = cv2.resize(background_frame, (frame.shape[1], frame.shape[0]))

        # Apply bounding box crop if configured
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
                if circularity < CURCULARITY_THRESHOLD:
                    continue
                mask = np.zeros_like(combined_mask)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                overlap = cv2.countNonZero(cv2.bitwise_and(mask, combined_mask))
                fill_ratio = overlap / area
                if fill_ratio < FILL_RATIO_THRESHOLD:
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

        return TopDownBallData(
            top_left=Coord(centre_x - radius, centre_y - radius),
            bottom_right=Coord(centre_x + radius, centre_y + radius),
            centre=Coord(centre_x, centre_y),
            seam_start=Coord(-1, -1),
            seam_end=Coord(-1, -1),
            seam_angle=-1.0
        )

    def find_seam(self, ball_data: TopDownBallData, frame: MatLike) -> Optional[TopDownBallData]:
        """Return the ball data with the seam ends and angle in degrees set from the
        longest white line in the ball, or with the angle -1 if there is none.
        """
        assert ball_data.top_left is not None and ball_data.bottom_right is not None
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
        assert centre_full is not None

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
            assert centre is not None

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
    """Automatic ball finder for the side-on video."""
    def get_ball_data(self, video: Video) -> list[SideOnBallDataPoint]:
        """Scan the video from its current frame and return the ball in each frame until
        it is lost, saving those frames as images.
        """
        ball_data_points: list[SideOnBallDataPoint] = []

        while video.get_current_frame_number() < video.total_frames - 1:
            video.change_frame(SIDE_ON_FRAME_SKIP - 1)
            st = time.time()
            background_frame = video.get_current_frame()
            video.change_frame(1)
            current_frame = video.get_current_frame()
            ball_data = self.find_ball(current_frame, background_frame)
            #print(f"Checking frame {video.current_frame} for ball ({(time.time() - st)*1000:.2f}ms)")
            if ball_data:
                video.change_frame(-(SIDE_ON_FRAME_SKIP + 1))
                #print(f"Ball found in frame {video.current_frame}, starting processing from frame {video.current_frame - SIDE_ON_FRAME_SKIP - 1}.")
                break
            #print(f"No ball found in frame {video.current_frame}, skipping {SIDE_ON_FRAME_SKIP - 1} frames.")

        ball_position = BallPosition.BEFORE_FRAME

        while ball_position is not BallPosition.AFTER_FRAME and video.get_current_frame_number() < video.total_frames - 1:
            st = time.time()
            background_frame = video.get_current_frame()
            video.change_frame(1)
            current_frame = video.get_current_frame()
            ball_data = self.find_ball(current_frame, background_frame)
            if ball_data:
                ball_data_points.append(SideOnBallDataPoint(frame_number=video.current_frame, data=ball_data))
                assert current_frame is not None
                cv2.imwrite(f"{side_on_tracking_folder}/frame_{video.current_frame:04d}.jpg", current_frame)
                ball_position = BallPosition.IN_FRAME
            else:
                if ball_position == BallPosition.IN_FRAME:
                    ball_position = BallPosition.AFTER_FRAME
                    #print(f"Ball lost after frame {video.current_frame}, stopping processing.")
            et = time.time()
            #print(f"Side On Frame {video.current_frame} ({(et - st)*1000:.2f}ms): {ball_data}")
        return ball_data_points

    def find_ball(self, current_frame: MatLike | None, background_frame: MatLike | None) -> Optional[SideOnBallData]:
        """Find the ball by motion against the background frame, red colour and circle
        shape, and return its box and centre in px, or None.
        """
        if current_frame is None or background_frame is None:
            return None

        if current_frame.shape != background_frame.shape:
            background_frame = cv2.resize(background_frame, (current_frame.shape[1], current_frame.shape[0]))

        # Apply bounding box crop if configured
        x_offset, y_offset = 0, 0
        bbox = _load_bounding_box("side_on")
        if bbox:
            current_frame, background_frame, x_offset, y_offset = _apply_bounding_box(
                current_frame, background_frame, bbox)

        h, w = current_frame.shape[:2]

        # Downscale for faster processing
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

        # Motion mask
        blurred    = cv2.GaussianBlur(small,    (5, 5), 0)
        blurred_bg = cv2.GaussianBlur(small_bg, (5, 5), 0)

        hsv    = cv2.cvtColor(blurred,    cv2.COLOR_BGR2HSV)
        hsv_bg = cv2.cvtColor(blurred_bg, cv2.COLOR_BGR2HSV)

        diff_v = cv2.absdiff(hsv[:, :, 2], hsv_bg[:, :, 2])
        _, motion_mask = cv2.threshold(diff_v, 20, 255, cv2.THRESH_BINARY)

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        motion_mask = cv2.dilate(motion_mask, kernel_small, iterations=1)

        # Colour mask
        color_mask: np.ndarray = np.zeros((proc_h, proc_w), dtype=np.uint8)
        for lower, upper in SIDE_ON_BALL_COLOR_RANGES:
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))

        # Combined mask
        combined_mask = cv2.bitwise_and(motion_mask, color_mask)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  kernel_small, iterations=1)

        cv2.imwrite(f"{side_on_tracking_folder}/motion_mask.jpg",   motion_mask)
        cv2.imwrite(f"{side_on_tracking_folder}/color_mask.jpg",    color_mask)
        cv2.imwrite(f"{side_on_tracking_folder}/combined_mask.jpg", combined_mask)

        # Hough circle detection
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

        # Contour fallback
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

        # Get full-image coords
        sx, sy, sr, _ = best_circle
        if scale < 1.0:
            local_x = int(round(sx / scale))
            local_y = int(round(sy / scale))
            radius  = int(round(sr / scale))
        else:
            local_x, local_y, radius = int(round(sx)), int(round(sy)), int(round(sr))

        centre_x = local_x + x_offset
        centre_y = local_y + y_offset

        # Debug output
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

        return SideOnBallData(
            top_left=Coord(centre_x - radius, centre_y - radius),
            bottom_right=Coord(centre_x + radius, centre_y + radius),
            centre=Coord(centre_x, centre_y)
        )


# tester = TopDownBallFinder()

# background = cv2.imread(str(paths.OUTPUT_DIR / "background_frame.jpg"))
# current = cv2.imread(str(paths.OUTPUT_DIR / "current_frame.jpg"))

# ball_data = tester.find_ball(current, background)
# print(ball_data)

# if ball_data:
#     print(tester.find_seam(ball_data, current))
