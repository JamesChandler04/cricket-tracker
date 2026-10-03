"""Shared building blocks: OpenCV key codes, pixel coordinates, per-frame ball data, a
video reader and a JSON settings loader.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, cast
import cv2
import numpy as np

if TYPE_CHECKING:
    from cv2.typing import MatLike

FramePosition = tuple[int, int, int, float]
"""A tracked ball position: (frame number, x in px, y in px, time in s)."""
Calibration = tuple[int, list[tuple[int, int]]]
"""A ball-diameter calibration: (frame number, [(x1, y1), (x2, y2)] in px)."""

class Key(Enum):
    """Key codes from cv2.waitKey for the keys the OpenCV windows use."""
    esc = 27
    """Escape key, which quits."""
    q = ord('q')
    """Q key, which quits."""
    c = ord('c')
    """C key, which starts or pauses calibration."""
    space = ord(' ')
    """Space bar, which starts or pauses ball tracking."""
    a = ord('a')
    """Lowercase a, which steps back one frame."""
    d = ord('d')
    """Lowercase d, which steps forward one frame."""
    s = ord('s')
    """S key, which saves or moves on to the next step."""
    o = ord('o')
    """O key, which rotates the video 90 degrees clockwise."""
    r = ord('r')
    """R key, which resets the tracking."""
    t = ord('t')
    """T key, which starts or stops seam angle mode."""
    A = ord('A')
    """Capital A, which steps back 10 frames."""
    D = ord('D')
    """Capital D, which steps forward 10 frames."""
    f = ord('f')
    """F key, which looks for the ball automatically in the automatic ball tracker."""
    b = ord('b')
    """B key, which sets the background frame for ball finding in the automatic ball tracker."""
    z = ord('z')
    """Z key, which toggles the side-on zoom."""

@dataclass
class Coord:
    """Pixel position (x, y) in a frame, with simple vector arithmetic."""
    x: int
    y: int

    def distance_to(self, other: 'Coord') -> float:
        """Return the straight-line distance to another point, in px."""
        return cast(float, ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5)

    def __str__(self) -> str:
        """Return the point as (x, y) text."""
        return f"({self.x}, {self.y})"
    
    def __add__(self, other: 'Coord') -> Coord:
        """Return the sum of two points."""
        return Coord(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Coord') -> Coord:
        """Return this point minus another."""
        return Coord(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Coord:
        """Return the point scaled by a number, truncated to whole pixels."""
        return Coord(int(self.x * scalar), int(self.y * scalar))

    def __truediv__(self, scalar: float) -> Coord:
        """Return the point divided by a number, truncated to whole pixels."""
        return Coord(int(self.x / scalar), int(self.y / scalar))

    def __floor__(self, scalar: float) -> Coord:
        """Return the point divided by a number, truncated; the same as /."""
        return Coord(int(self.x / scalar), int(self.y / scalar))    

@dataclass
class TopDownBallData:
    """Ball and seam measurements for one top-down frame, in px and degrees."""
    top_left: Coord | None
    bottom_right: Coord | None
    centre: Coord | None
    seam_start: Coord | None
    seam_end: Coord | None
    seam_angle: float | None

    def __str__(self) -> str:
        """Return the ball data as one line of text."""
        return (f"BallData(top_left={self.top_left}, "
                f"bottom_right={self.bottom_right}, "
                f"centre={self.centre}, "
                f"seam_start={self.seam_start}, "
                f"seam_end={self.seam_end}, "
                f"seam_angle={self.seam_angle} degrees)")

    def calc_seam_angle(self) -> None:
        """Set seam_angle to the seam's image angle in degrees, clockwise from +x."""
        assert self.seam_end is not None and self.seam_start is not None
        self.seam_angle = math.degrees(math.atan2(self.seam_end.y - self.seam_start.y, self.seam_end.x - self.seam_start.x))

    def calc_centre(self) -> None:
        """Set centre to the midpoint of top_left and bottom_right."""
        assert self.top_left is not None and self.bottom_right is not None
        self.centre = Coord(
            x=(self.top_left.x + self.bottom_right.x) // 2,
            y=(self.top_left.y + self.bottom_right.y) // 2
        )

@dataclass
class SideOnBallData:
    """Ball measurements for one side-on frame, in px."""
    top_left: Coord
    bottom_right: Coord
    centre: Coord

    def __str__(self) -> str:
        """Return the ball data as one line of text."""
        return (f"BallData(top_left={self.top_left}, "
                f"bottom_right={self.bottom_right}, "
                f"centre={self.centre})")
    
    def calc_centre(self) -> None:
        """Set centre to the midpoint of top_left and bottom_right."""
        self.centre = Coord(
            x=(self.top_left.x + self.bottom_right.x) // 2,
            y=(self.top_left.y + self.bottom_right.y) // 2
        )

class Video:
    """Video file read one frame at a time, with caching and 90 degree rotation."""
    def __init__(self, path: str) -> None:
        """Open the video at path and read its frame count, size and frame rate."""
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open main video file {path}")
        self._get_frame_data()

        self.current_frame = 0
        self.rotation = 0
        self._cached_frame: np.ndarray | None = None
        self._cached_frame_index = -1

    def _get_frame_data(self) -> None:
        """Read the frame count, size and fps (asked for in the console if missing)."""
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Main video loaded: {self.total_frames} frames")
        print(f"Main resolution: {self.frame_width}x{self.frame_height}")

        if self.fps == 0:
            print("Warning: Could not determine FPS from video metadata.")
            while True:
                try:
                    self.fps = float(input("Enter the frame rate (FPS) of the video: ").strip())
                    if self.fps <= 0:
                        raise ValueError
                    break
                except ValueError:
                    print("Please enter a positive number for FPS.")
        
        print(f"Main video FPS: {self.fps:.2f}")
    
    def get_current_frame(self) -> MatLike | None:
        """Return the current frame, rotated, or None if it cannot be read."""
        if self._cached_frame is not None and self._cached_frame_index == self.current_frame:
            return self._rotate_frame(self._cached_frame.copy())

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return None

        self._cached_frame = frame.copy()
        self._cached_frame_index = self.current_frame
        return self._rotate_frame(frame)
    
    def get_current_frame_number(self) -> int:
        """Return the index of the current frame."""
        return self.current_frame

    def change_frame(self, offset: int) -> None:
        """Move offset frames, staying within the video, and cache the new frame."""
        new_frame = self.current_frame + offset
        if new_frame < 0:
            new_frame = 0
        elif new_frame >= self.total_frames:
            new_frame = self.total_frames - 1

        if new_frame == self.current_frame:
            return

        if offset == 1 and self._cached_frame is not None and self._cached_frame_index == self.current_frame:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = new_frame
                self._cached_frame = frame.copy()
                self._cached_frame_index = self.current_frame
                return

        self.current_frame = new_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            self._cached_frame = frame.copy()
            self._cached_frame_index = self.current_frame
        else:
            self._cached_frame = None
            self._cached_frame_index = -1

    def rotate(self) -> None:
        """Turn the video a further 90 degrees clockwise."""
        self.rotation = (self.rotation + 90) % 360

    def _rotate_frame(self, frame: MatLike) -> MatLike:
        """Return the frame turned by the current rotation."""
        match self.rotation:
            case 90:
                return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            case 180:
                return cv2.rotate(frame, cv2.ROTATE_180)
            case 270:
                return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            case _:
                return frame
            
class Config:
    """Settings loaded from a JSON file."""
    def __init__(self, path: str) -> None:
        """Load the settings from the JSON file at path."""
        self.path = path
        self.data = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Return the file's JSON contents, or an empty dict if it cannot be read."""
        import json
        try:
            with open(self.path, 'r') as f:
                return cast(dict[str, Any], json.load(f))
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}

@dataclass
class TopDownBallDataPoint:
    """Top-down ball data with the frame number it belongs to."""
    frame_number: int
    data: TopDownBallData

    def __str__(self) -> str:
        """Return the frame number and ball data as one line of text."""
        return f"Frame {self.frame_number}: {self.data}"

@dataclass
class SideOnBallDataPoint:
    """Side-on ball data with the frame number it belongs to."""
    frame_number: int
    data: SideOnBallData

    def __str__(self) -> str:
        """Return the frame number and ball data as one line of text."""
        return f"Frame {self.frame_number}: {self.data}"