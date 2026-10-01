from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from library import paths
from library.helpers import Coord, TopDownBallDataPoint, Video


DEFAULT_SAVE_DIR = paths.DELIVERY_DIR
BALL_IMAGE_FOLDER = "ball_images"
INDEX_FILE_NAME = "ball_images.yaml"
IMAGE_NAME = "frame_{frame:06d}.png"
SEAM_FILE_NAME = "seam_angle_{method}.yaml"

# Side of each square crop, in ball diameters. The margin round the ball allows
# for a slightly off-centre click and for motion blur.
CROP_SCALE = 1.5

# Seam picker window layout, in window pixels, top to bottom: a line with the
# frame and seam angle, the enlarged ball image, a strip of every ball image,
# and a line of controls.
MARGIN = 16
VIEW_SIZE = 560                                  # the enlarged ball image
CANVAS_WIDTH = VIEW_SIZE + 2 * MARGIN
INFO_Y = MARGIN + 16                             # baseline of the frame / seam angle line
VIEW_X = MARGIN
VIEW_Y = INFO_Y + 14
THUMB_SIZE = 64
THUMB_GAP = 8
STRIP_Y = VIEW_Y + VIEW_SIZE + 12                # top of the thumbnail strip
HINT_Y = STRIP_Y + THUMB_SIZE + 24               # baseline of the controls line
CANVAS_HEIGHT = HINT_Y + 14

# Colours, BGR. The seam colour matches the seam points in drawers.py.
BACKGROUND = (255, 255, 255)
TEXT = (40, 40, 40)
HINT_TEXT = (130, 130, 130)
THUMB_BORDER = (205, 205, 205)
SEAM_COLOUR = (255, 255, 0)
OUTLINE = (0, 0, 0)

# waitKeyEx codes for the arrow keys on Windows, Linux (GTK and Qt) and macOS.
LEFT_KEYS = {0x250000, 0xFF51, 0xF702}
RIGHT_KEYS = {0x270000, 0xFF53, 0xF703}
ESC, ENTER, LINE_FEED = 27, 13, 10


# ---------------------------------------------------------------- geometry

def fold_line_angle(angle_deg: float) -> float:
    """
    Fold an angle into (-90, 90].

    A line through the ball looks the same turned through 180 degrees, so an
    angle and the same angle plus or minus 180 describe one seam.
    """
    folded = math.fmod(angle_deg, 180.0)
    if folded <= -90.0:
        folded += 180.0
    elif folded > 90.0:
        folded -= 180.0
    return folded + 0.0  # turns -0.0 into 0.0


def _direction_deg(start: Coord, end: Coord) -> float | None:
    dx, dy = end.x - start.x, end.y - start.y
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, dx))


def travel_directions(points: list[TopDownBallDataPoint]) -> dict[int, float | None]:
    """
    Direction of travel at every frame with a ball centre, in degrees, by frame.

    Taken from that frame's centre to the next one, as
    TopDownPhysicsEngine.calculate_seam_angle does, and from the previous
    centre for the last frame.
    """
    centred = [point for point in points if point.data.centre is not None]
    directions: dict[int, float | None] = {}
    for index, point in enumerate(centred):
        centre = point.data.centre
        assert centre is not None
        direction = None
        if index + 1 < len(centred):
            following = centred[index + 1].data.centre
            assert following is not None
            direction = _direction_deg(centre, following)
        if direction is None and index > 0:
            previous = centred[index - 1].data.centre
            assert previous is not None
            direction = _direction_deg(previous, centre)
        directions[point.frame_number] = direction
    return directions


# ------------------------------------------------------------- data types

@dataclass
class BallImage:
    """One saved crop of the ball, and where it sits in the full top-down frame."""

    frame_number: int
    path: Path
    centre: tuple[float, float]          # ball centre in the full frame, pixels
    diameter_px: float                   # ball diameter the crop was sized from
    origin: tuple[int, int]              # full-frame pixel at the crop's top-left
    size_px: int                         # the crop is size_px square
    travel_direction_deg: float | None   # direction of travel at this frame

    def load(self) -> np.ndarray:
        return _read_image(self.path)

    def to_frame(self, x: float, y: float) -> tuple[float, float]:
        """Crop pixels to full-frame pixels."""
        return self.origin[0] + x, self.origin[1] + y

    def to_crop(self, x: float, y: float) -> tuple[float, float]:
        """Full-frame pixels to crop pixels."""
        return x - self.origin[0], y - self.origin[1]


@dataclass
class SeamMeasurement:
    """A line along the seam on one ball image, with its ends in full-frame pixels."""

    image: BallImage
    start: tuple[float, float]
    end: tuple[float, float]
    method: str

    @property
    def frame_number(self) -> int:
        return self.image.frame_number

    @property
    def length_px(self) -> float:
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    @property
    def raw_angle_deg(self) -> float:
        """Angle of the seam line in the image, folded into (-90, 90]."""
        return fold_line_angle(math.degrees(math.atan2(self.end[1] - self.start[1],
                                                       self.end[0] - self.start[0])))

    @property
    def seam_angle_deg(self) -> float | None:
        """Seam angle relative to the direction of travel, folded into (-90, 90]."""
        travel = self.image.travel_direction_deg
        if travel is None:
            return None
        return fold_line_angle(self.raw_angle_deg - travel)

    def details(self) -> dict[str, object]:
        """Anything more a detector wants saved with the seam. Nothing by default."""
        return {}


# ------------------------------------------------------------ ball images

def save_ball_images(video_path: str, points: list[TopDownBallDataPoint], save_dir: str,
                     rotation: int = 0, crop_scale: float = CROP_SCALE) -> list[BallImage]:
    """
    Crop the ball out of every top-down frame with a centre and save the crops
    to save_dir/ball_images, replacing any from an earlier run.

    Each crop is a square crop_scale ball diameters across, centred on the
    ball. The diameter comes from the diameter clicks, interpolated by frame
    number between the frames where it was clicked. Any part of a crop that
    falls off the frame is left black, so the ball always sits in the middle
    of its image.

    rotation has to match the rotation used while tracking, because the clicks
    were made on the rotated frame.
    """
    diameters = _clicked_diameters(points)
    if not diameters:
        raise ValueError("No top-down frame has a clicked ball diameter, "
                         "so the ball images cannot be sized.")
    clicked_frames = sorted(diameters)
    clicked_diameters = [diameters[frame] for frame in clicked_frames]
    directions = travel_directions(points)

    folder = Path(save_dir) / BALL_IMAGE_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    _clear_earlier_run(folder)

    print(f"\nSaving ball images to {folder}")
    video = Video(str(video_path))
    video.rotation = rotation
    ball_images = []
    try:
        for point in points:
            centre = point.data.centre
            if centre is None:
                continue
            frame = _read_frame(video, point.frame_number)
            if frame is None:
                print(f"Could not read frame {point.frame_number}, so it has no ball image.")
                continue
            diameter = float(np.interp(point.frame_number, clicked_frames, clicked_diameters))
            size = max(1, round(crop_scale * diameter))
            origin = (int(centre.x) - size // 2, int(centre.y) - size // 2)
            path = folder / IMAGE_NAME.format(frame=point.frame_number)
            _write_image(path, _crop(frame, origin, size))
            ball_images.append(BallImage(
                frame_number=point.frame_number,
                path=path,
                centre=(float(centre.x), float(centre.y)),
                diameter_px=diameter,
                origin=origin,
                size_px=size,
                travel_direction_deg=directions.get(point.frame_number),
            ))
    finally:
        video.cap.release()

    _write_index(folder, video_path, rotation, crop_scale, ball_images)
    print(f"Saved {len(ball_images)} ball images.")
    return ball_images


def load_ball_images(folder: str | Path) -> list[BallImage]:
    """Read back the ball images save_ball_images wrote to folder."""
    folder = Path(folder)
    with open(folder / INDEX_FILE_NAME) as handle:
        index = yaml.safe_load(handle)
    images = []
    for entry in index["images"]:
        travel = entry["travel_direction_deg"]
        images.append(BallImage(
            frame_number=int(entry["frame"]),
            path=folder / entry["file"],
            centre=(float(entry["centre_px"][0]), float(entry["centre_px"][1])),
            diameter_px=float(entry["diameter_px"]),
            origin=(int(entry["origin_px"][0]), int(entry["origin_px"][1])),
            size_px=int(entry["size_px"]),
            travel_direction_deg=None if travel is None else float(travel),
        ))
    return images


def save_seam_measurement(seam: SeamMeasurement, folder: str | Path | None = None) -> Path:
    """Write the seam line and its angles next to the ball images, as seam_angle_<method>.yaml."""
    directory = seam.image.path.parent if folder is None else Path(folder)
    path = directory / SEAM_FILE_NAME.format(method=seam.method)
    with open(path, "w") as handle:
        yaml.safe_dump({
            "method": seam.method,
            "frame": seam.frame_number,
            "image": seam.image.path.name,
            "seam_start_px": [float(seam.start[0]), float(seam.start[1])],
            "seam_end_px": [float(seam.end[0]), float(seam.end[1])],
            "raw_angle_deg": seam.raw_angle_deg,
            "travel_direction_deg": seam.image.travel_direction_deg,
            "seam_angle_deg": seam.seam_angle_deg,
            **seam.details(),
        }, handle, sort_keys=False, default_flow_style=None)
    return path


def _clicked_diameters(points: list[TopDownBallDataPoint]) -> dict[int, float]:
    """
    Ball diameter in pixels at every frame where it was clicked. In manual
    tracking top_left and bottom_right are the two ends of the clicked diameter.
    """
    diameters = {}
    for point in points:
        top_left, bottom_right = point.data.top_left, point.data.bottom_right
        if top_left is not None and bottom_right is not None:
            diameter = top_left.distance_to(bottom_right)
            if diameter >= 1.0:
                diameters[point.frame_number] = diameter
    return diameters


def _read_frame(video: Video, frame_number: int) -> np.ndarray | None:
    """Step the video to frame_number the same way the tracker does, and return it rotated."""
    if not 0 <= frame_number < video.total_frames:
        return None
    video.change_frame(frame_number - video.current_frame)
    return video.get_current_frame()


def _crop(frame: np.ndarray, origin: tuple[int, int], size: int) -> np.ndarray:
    """size by size crop with its top-left at origin. Anything off the frame is black."""
    height, width = frame.shape[:2]
    crop = np.zeros((size, size) + frame.shape[2:], dtype=frame.dtype)
    x0, y0 = origin
    left, top = max(x0, 0), max(y0, 0)
    right, bottom = min(x0 + size, width), min(y0 + size, height)
    if right > left and bottom > top:
        crop[top - y0:bottom - y0, left - x0:right - x0] = frame[top:bottom, left:right]
    return crop


def _clear_earlier_run(folder: Path) -> None:
    """Delete the ball images and seam files an earlier run left in folder, and nothing else."""
    for pattern in ("frame_*.png", INDEX_FILE_NAME, SEAM_FILE_NAME.format(method="*")):
        for path in folder.glob(pattern):
            path.unlink()


def _write_index(folder: Path, video_path: str, rotation: int, crop_scale: float,
                 ball_images: list[BallImage]) -> None:
    index = {
        "video": str(video_path),
        "rotation_deg": int(rotation),
        "crop_scale": float(crop_scale),
        "images": [{
            "frame": image.frame_number,
            "file": image.path.name,
            "centre_px": [image.centre[0], image.centre[1]],
            "diameter_px": image.diameter_px,
            "origin_px": [image.origin[0], image.origin[1]],
            "size_px": image.size_px,
            "travel_direction_deg": image.travel_direction_deg,
        } for image in ball_images],
    }
    with open(folder / INDEX_FILE_NAME, "w") as handle:
        yaml.safe_dump(index, handle, sort_keys=False, default_flow_style=None)


def _write_image(path: Path, image: np.ndarray) -> None:
    # imencode and write_bytes instead of cv2.imwrite, which fails on Windows
    # when the folder name has characters outside the system code page.
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {path}")
    return image


# -------------------------------------------------------------- detectors

class SeamAngleDetector(ABC):
    """
    One way of finding the seam on the ball images.

    detect() gets every saved ball image in frame order and returns the seam
    as a line on one of them, or None if there is no seam to use. Work in crop
    pixels from image.load() and convert the seam's ends with image.to_frame().
    Manual and automatic detectors return the same SeamMeasurement, so the
    rest of the pipeline does not need to know which one ran.
    """

    method = ""

    @abstractmethod
    def detect(self, ball_images: list[BallImage]) -> SeamMeasurement | None:
        """Find the seam on one of ball_images."""


class ManualSeamAngleDetector(SeamAngleDetector):
    """Opens a window to scroll through the ball images and click the seam on one of them."""

    method = "manual"

    def detect(self, ball_images: list[BallImage]) -> SeamMeasurement | None:
        if not ball_images:
            print("There are no ball images to pick the seam from.")
            return None
        return SeamPicker(ball_images, self.method).run()


# ------------------------------------------------------------ seam picker

class SeamPicker:
    """
    The window ManualSeamAngleDetector opens: the ball image on show enlarged,
    a strip of every ball image underneath, and two clicks along the seam.

    Seam points are kept in crop pixels of the image they were clicked on.
    Clicking on a different image starts a new seam there, so there is only
    ever one seam, on one image.
    """

    window_name = "Cricket Ball Tracker - Seam Angle"

    def __init__(self, ball_images: list[BallImage], method: str = "manual"):
        self.images = ball_images
        self.method = method
        self.crops = [image.load() for image in ball_images]
        self.thumbnails = [cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
                           for crop in self.crops]
        self.index = 0
        self.seam_index: int | None = None
        self.seam_points: list[tuple[float, float]] = []
        self.mouse: tuple[int, int] | None = None
        self._thumbnail_slots: list[tuple[int, int]] = []  # (left edge, image index)
        self._seen_window = False

    # ----------------------------------------------------------------- loop

    def run(self) -> SeamMeasurement | None:
        """Show the window until a seam is used (returned) or the window is closed (None)."""
        self._print_controls()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, CANVAS_WIDTH, CANVAS_HEIGHT)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            while True:
                cv2.imshow(self.window_name, self.render())
                code = cv2.waitKeyEx(15)
                if self._window_closed():
                    print("Seam angle window closed without a seam angle.")
                    return None
                action = self._on_key(code)
                if action == "use":
                    seam = self.measurement()
                    assert seam is not None
                    self._print_seam("Using the seam", seam)
                    return seam
                if action == "close":
                    print("Closed without a seam angle.")
                    return None
        finally:
            try:
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass

    def _print_controls(self) -> None:
        print("\n=== CRICKET BALL TRACKER - SEAM ANGLE ===")
        print(f"{len(self.images)} ball images. Scroll to the one where the seam is clearest "
              "and click two points along the seam.")
        print("Controls:")
        print("- A/D, Left/Right arrows or mouse wheel: Previous/next image")
        print("- Click a thumbnail: Show that image")
        print("- Click on the ball: Seam point (two clicks, a third click starts again)")
        print("- U: Undo the last seam point")
        print("- R: Reset the seam")
        print("- S or Enter: Use this seam angle")
        print("- Q or ESC: Close without a seam angle")

    def _window_closed(self) -> bool:
        """True once the window has been shown and then closed with its X button."""
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return self._seen_window
        if visible >= 1:
            self._seen_window = True
            return False
        return self._seen_window

    # --------------------------------------------------------------- input

    def _on_key(self, code: int) -> str | None:
        """Handle one waitKeyEx code. Returns "use", "close" or None."""
        if code == -1:
            return None
        if code in LEFT_KEYS:
            self.show(self.index - 1)
            return None
        if code in RIGHT_KEYS:
            self.show(self.index + 1)
            return None

        key = code & 0xFF
        if key in (ord("q"), ord("Q"), ESC):
            return "close"
        if key in (ord("s"), ord("S"), ENTER, LINE_FEED):
            if self.seam_ready():
                return "use"
            print("Click two points along the seam first.")
        elif key in (ord("a"), ord("A")):
            self.show(self.index - 1)
        elif key in (ord("d"), ord("D")):
            self.show(self.index + 1)
        elif key in (ord("u"), ord("U")):
            self.undo()
        elif key in (ord("r"), ord("R")):
            self.reset()
        return None

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        # x and y are canvas pixels: OpenCV scales them back if the window is resized.
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.show(self.index - 1 if _wheel_delta(flags) > 0 else self.index + 1)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse = (x, y)
            if _inside_view(x, y):
                self.add_seam_point(*self._canvas_to_crop(x, y))
            else:
                clicked = self._thumbnail_at(x, y)
                if clicked is not None:
                    self.show(clicked)

    # --------------------------------------------------------------- state

    def show(self, index: int) -> None:
        self.index = min(max(index, 0), len(self.images) - 1)

    def add_seam_point(self, x: float, y: float) -> None:
        """Add a seam point, in crop pixels of the image on show."""
        frame = self.images[self.index].frame_number
        if self.seam_index != self.index or len(self.seam_points) == 2:
            if self.seam_points and self.seam_index is not None and self.seam_index != self.index:
                dropped = self.images[self.seam_index].frame_number
                print(f"Seam on frame {dropped} dropped, starting a new one on frame {frame}.")
            self.seam_points = []
            self.seam_index = self.index
        self.seam_points.append((x, y))
        if self.seam_ready():
            seam = self.measurement()
            assert seam is not None
            self._print_seam("Seam", seam)
            print("Press S to use it, or click again to start over.")

    def undo(self) -> None:
        if self.seam_points:
            self.seam_points.pop()
        if not self.seam_points:
            self.seam_index = None

    def reset(self) -> None:
        self.seam_points = []
        self.seam_index = None

    def seam_ready(self) -> bool:
        seam = self.measurement()
        return seam is not None and seam.length_px > 0.5

    def measurement(self) -> SeamMeasurement | None:
        """The clicked seam, or None until both points are in."""
        if self.seam_index is None or len(self.seam_points) < 2:
            return None
        image = self.images[self.seam_index]
        return SeamMeasurement(image, image.to_frame(*self.seam_points[0]),
                               image.to_frame(*self.seam_points[1]), self.method)

    def _live_measurement(self) -> SeamMeasurement | None:
        """From the first seam point to the mouse, while the second point is being placed."""
        if (self.seam_index != self.index or len(self.seam_points) != 1
                or self.mouse is None or not _inside_view(*self.mouse)):
            return None
        image = self.images[self.index]
        end = image.to_frame(*self._canvas_to_crop(*self.mouse))
        live = SeamMeasurement(image, image.to_frame(*self.seam_points[0]), end, self.method)
        return live if live.length_px > 0.5 else None

    @staticmethod
    def _print_seam(prefix: str, seam: SeamMeasurement) -> None:
        print(f"{prefix} on frame {seam.frame_number}: seam angle {_format_angle(seam.seam_angle_deg)} "
              f"(seam line {_format_angle(seam.raw_angle_deg)}, "
              f"direction of travel {_format_angle(seam.image.travel_direction_deg)})")

    # --------------------------------------------------------- coordinates

    def _canvas_to_crop(self, x: int, y: int) -> tuple[float, float]:
        """Canvas pixel inside the view to crop pixel of the image on show."""
        crop_per_view = self.images[self.index].size_px / VIEW_SIZE
        # cv2.resize puts pixel centres at half-pixel positions, so map through those.
        return ((x - VIEW_X + 0.5) * crop_per_view - 0.5,
                (y - VIEW_Y + 0.5) * crop_per_view - 0.5)

    @staticmethod
    def _crop_to_view(image: BallImage, x: float, y: float) -> tuple[int, int]:
        """Crop pixel to pixel of the enlarged view, for drawing."""
        view_per_crop = VIEW_SIZE / image.size_px
        return (int(round((x + 0.5) * view_per_crop - 0.5)),
                int(round((y + 0.5) * view_per_crop - 0.5)))

    def _thumbnail_at(self, x: int, y: int) -> int | None:
        if not STRIP_Y <= y < STRIP_Y + THUMB_SIZE:
            return None
        for left, index in self._thumbnail_slots:
            if left <= x < left + THUMB_SIZE:
                return index
        return None

    # -------------------------------------------------------------- drawing

    def render(self) -> np.ndarray:
        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), BACKGROUND, dtype=np.uint8)
        view = cv2.resize(self.crops[self.index], (VIEW_SIZE, VIEW_SIZE),
                          interpolation=cv2.INTER_LINEAR)
        self._draw_seam(view, self.images[self.index])
        canvas[VIEW_Y:VIEW_Y + VIEW_SIZE, VIEW_X:VIEW_X + VIEW_SIZE] = view
        self._draw_info(canvas)
        self._draw_strip(canvas)
        _text(canvas, "a/d: change image    r: reset    s: save    q: skip",
              VIEW_X, HINT_Y, 0.45, HINT_TEXT)
        return canvas

    def _draw_seam(self, view: np.ndarray, image: BallImage) -> None:
        if self.seam_index != self.index or not self.seam_points:
            return
        points = [self._crop_to_view(image, x, y) for x, y in self.seam_points]
        # Lines get a dark outline so they still show on top of a white seam.
        if len(points) == 2:
            cv2.line(view, points[0], points[1], OUTLINE, 4, cv2.LINE_AA)
            cv2.line(view, points[0], points[1], SEAM_COLOUR, 2, cv2.LINE_AA)
        elif self.mouse is not None and _inside_view(*self.mouse):
            mouse = (self.mouse[0] - VIEW_X, self.mouse[1] - VIEW_Y)
            cv2.line(view, points[0], mouse, OUTLINE, 3, cv2.LINE_AA)
            cv2.line(view, points[0], mouse, SEAM_COLOUR, 1, cv2.LINE_AA)
        for point in points:
            cv2.circle(view, point, 6, OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(view, point, 5, SEAM_COLOUR, -1, cv2.LINE_AA)

    def _draw_info(self, canvas: np.ndarray) -> None:
        """Frame on the left; the seam angle, or what to do next, on the right."""
        image = self.images[self.index]
        _text(canvas, f"Frame {image.frame_number}  ({self.index + 1}/{len(self.images)})",
              VIEW_X, INFO_Y, 0.6, TEXT)

        seam = self.measurement() if self.seam_ready() else self._live_measurement()
        if seam is None:
            message = "Click two points on the seam"
        elif seam.seam_angle_deg is None:
            message = "Seam angle -"
        else:
            message = f"Seam angle {seam.seam_angle_deg:+.1f} deg"
        width = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        _text(canvas, message, VIEW_X + VIEW_SIZE - width, INFO_Y, 0.6, TEXT)

    def _draw_strip(self, canvas: np.ndarray) -> None:
        """Every ball image, centred under the view. The one with the seam shows its seam."""
        count = len(self.images)
        fits = max(1, (VIEW_SIZE + THUMB_GAP) // (THUMB_SIZE + THUMB_GAP))
        visible = min(count, fits)
        first = min(max(self.index - visible // 2, 0), count - visible)
        strip_width = visible * THUMB_SIZE + (visible - 1) * THUMB_GAP
        start = VIEW_X + (VIEW_SIZE - strip_width) // 2
        self._thumbnail_slots = []
        for slot in range(visible):
            index = first + slot
            left = start + slot * (THUMB_SIZE + THUMB_GAP)
            thumbnail = self.thumbnails[index].copy()
            if index == self.seam_index and self.seam_ready():
                ends = [self._crop_to_thumbnail(self.images[index], x, y)
                        for x, y in self.seam_points]
                cv2.line(thumbnail, ends[0], ends[1], OUTLINE, 3, cv2.LINE_AA)
                cv2.line(thumbnail, ends[0], ends[1], SEAM_COLOUR, 1, cv2.LINE_AA)
            canvas[STRIP_Y:STRIP_Y + THUMB_SIZE, left:left + THUMB_SIZE] = thumbnail
            if index == self.index:
                cv2.rectangle(canvas, (left - 2, STRIP_Y - 2),
                              (left + THUMB_SIZE + 1, STRIP_Y + THUMB_SIZE + 1), TEXT, 2)
            else:
                cv2.rectangle(canvas, (left - 1, STRIP_Y - 1),
                              (left + THUMB_SIZE, STRIP_Y + THUMB_SIZE), THUMB_BORDER, 1)
            self._thumbnail_slots.append((left, index))

    @staticmethod
    def _crop_to_thumbnail(image: BallImage, x: float, y: float) -> tuple[int, int]:
        thumb_per_crop = THUMB_SIZE / image.size_px
        return (int(round((x + 0.5) * thumb_per_crop - 0.5)),
                int(round((y + 0.5) * thumb_per_crop - 0.5)))


# ---------------------------------------------------------------- helpers

def _inside_view(x: int, y: int) -> bool:
    return VIEW_X <= x < VIEW_X + VIEW_SIZE and VIEW_Y <= y < VIEW_Y + VIEW_SIZE


def _wheel_delta(flags: int) -> int:
    """
    The signed wheel step OpenCV packs into the top 16 bits of flags. This is
    what cv::getMouseWheelDelta does, which the Python bindings do not include.
    """
    delta = (flags >> 16) & 0xFFFF
    return delta - 0x10000 if delta & 0x8000 else delta


def _text(canvas: np.ndarray, text: str, x: int, y: int, scale: float,
          colour: tuple[int, int, int]) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def _format_angle(angle: float | None) -> str:
    return "-" if angle is None else f"{angle:+.2f} deg"


def main(argv: list[str] | None = None) -> None:
    """
    Pick the seam again on a folder of ball images from an earlier run. Only
    the seam file in that folder is written; top_down_analysis.yaml is not.
    """
    arguments = sys.argv[1:] if argv is None else argv
    folder = Path(arguments[0]) if arguments else Path(DEFAULT_SAVE_DIR) / BALL_IMAGE_FOLDER
    seam = ManualSeamAngleDetector().detect(load_ball_images(folder))
    if seam is None:
        return
    print(f"Calculated seam angle: {_format_angle(seam.seam_angle_deg)}")
    print(f"Seam saved to {save_seam_measurement(seam)}")


if __name__ == "__main__":
    main()
