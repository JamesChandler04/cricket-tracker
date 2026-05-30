from enum import Enum
from dataclasses import dataclass
import math
import cv2

class Key(Enum):
    esc = 27
    q = ord('q')
    c = ord('c')
    space = ord(' ')
    a = ord('a')
    d = ord('d')
    s = ord('s')
    o = ord('o')
    r = ord('r')
    t = ord('t')
    A = ord('A')
    D = ord('D')
    f = ord('f')
    b = ord('b')

@dataclass
class Coord:
    x: int
    y: int

    def __str__(self):
        return f"({self.x}, {self.y})"

@dataclass
class BallData:
    top_left: Coord
    bottom_right: Coord
    centre: Coord
    seam_start: Coord
    seam_end: Coord
    seam_angle: float

    def __str__(self):
        return (f"BallData(top_left={self.top_left}, "
                f"bottom_right={self.bottom_right}, "
                f"centre={self.centre}, "
                f"seam_start={self.seam_start}, "
                f"seam_end={self.seam_end}, "
                f"seam_angle={self.seam_angle:.2f} degrees)")

    def calc_seam_angle(self):
        self.seam_angle = math.degrees(math.atan2(self.seam_end.y - self.seam_start.y, self.seam_end.x - self.seam_start.x))

    def calc_centre(self):
        self.centre = Coord(
            x=(self.top_left.x + self.bottom_right.x) // 2,
            y=(self.top_left.y + self.bottom_right.y) // 2
        )

class Video:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open main video file {path}")
        self._get_frame_data()

        self.current_frame = 0
        self.rotation = 0
        self._cached_frame = None
        self._cached_frame_index = -1

    def _get_frame_data(self):
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
    
    def get_current_frame(self):
        if self._cached_frame is not None and self._cached_frame_index == self.current_frame:
            return self._rotate_frame(self._cached_frame.copy())

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return None

        self._cached_frame = frame.copy()
        self._cached_frame_index = self.current_frame
        return self._rotate_frame(frame)
    
    def get_current_frame_number(self):
        return self.current_frame

    def change_frame(self, offset: int):
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

    def rotate(self):
        self.rotation = (self.rotation + 90) % 360

    def _rotate_frame(self, frame):
        match self.rotation:
            case 90:
                return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            case 180:
                return cv2.rotate(frame, cv2.ROTATE_180)
            case 270:
                return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            case _:
                return frame