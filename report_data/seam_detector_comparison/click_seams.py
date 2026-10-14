"""Shows each sample ball image in turn for clicking two points along its seam, saving every
seam to manual_seam_angles.csv as soon as it is clicked.

No seam angle is shown, so the clicking stays blind, and running it again carries on from
the first image without a seam.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    # Started as a plain script (python report_data/seam_detector_comparison/<name>.py,
    # or the Run button in VS Code), so put the project folder on the path for the
    # imports. python -m report_data.seam_detector_comparison.<name> does not need this.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library.detect_seam_angle_file import (BACKGROUND, CANVAS_WIDTH, ENTER, ESC, HINT_TEXT, INDEX_FILE_NAME,
                                            INFO_Y, LEFT_KEYS, LINE_FEED, OUTLINE, RIGHT_KEYS, SEAM_COLOUR,
                                            TEXT, THUMB_BORDER, VIEW_SIZE, VIEW_X, VIEW_Y, BallImage,
                                            SeamMeasurement, _inside_view, _text, _wheel_delta,
                                            load_ball_images)
from report_data.seam_detector_comparison.make_seam_images import IMAGE_FOLDER, MANUAL_PATH

# Window layout below the enlarged image, in window pixels. Above it, the window
# matches the GUI's seam picker, so the seams are clicked the same way.
PROGRESS_Y = VIEW_Y + VIEW_SIZE + 12
"""Top edge of the progress bar, in px."""
PROGRESS_HEIGHT = 6
"""Height of the progress bar, in px."""
COUNT_Y = PROGRESS_Y + PROGRESS_HEIGHT + 20
"""Baseline of the line saying how many images are done, in px."""
HINT_Y = COUNT_Y + 24
"""Baseline of the first line of controls, in px."""
HINT_GAP = 20
"""Distance between the baselines of the two lines of controls, in px."""
CANVAS_HEIGHT = HINT_Y + HINT_GAP + 14
"""Height of the window, in px."""

MIN_SEAM_PX = 0.5
"""Shortest seam line accepted, in image px."""
COLUMNS = ["image", "seam_seen", "start_x_px", "start_y_px", "end_x_px", "end_y_px", "seam_line_deg",
           "travel_direction_deg", "seam_angle_deg", "clicked_at"]
"""Column headings of manual_seam_angles.csv."""


@dataclass
class ManualSeam:
    """The seam clicked on one image, or a note that no seam could be seen on it."""

    image: str
    start: tuple[float, float] | None   # ends of the seam line, in image px
    end: tuple[float, float] | None
    seam_line_deg: float | None         # angle of the seam line in the image
    travel_direction_deg: float | None
    seam_angle_deg: float | None        # angle to the direction of travel
    clicked_at: str

    @property
    def seen(self) -> bool:
        """Return whether a seam was clicked, rather than marked as not visible."""
        return self.start is not None and self.end is not None


def clicked_seam(image: BallImage, start: tuple[float, float], end: tuple[float, float]) -> ManualSeam:
    """Return the seam through two points clicked on image, with its angles worked out as the
    GUI's seam picker does.
    """
    seam = SeamMeasurement(image, image.to_frame(*start), image.to_frame(*end), "manual")
    return ManualSeam(image.path.name, start, end, seam.raw_angle_deg, image.travel_direction_deg,
                      seam.seam_angle_deg, _now())


def no_seam(image: BallImage) -> ManualSeam:
    """Return a note that no seam could be seen on image."""
    return ManualSeam(image.path.name, None, None, None, image.travel_direction_deg, None, _now())


def load_manual_seams(path: Path) -> dict[str, ManualSeam]:
    """Read the seams saved in path, by image file name, or return none if it does not exist."""
    if not path.exists():
        return {}
    seams = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            seen = row["seam_seen"] == "yes"
            seams[row["image"]] = ManualSeam(
                row["image"],
                (float(row["start_x_px"]), float(row["start_y_px"])) if seen else None,
                (float(row["end_x_px"]), float(row["end_y_px"])) if seen else None,
                _number(row["seam_line_deg"]), _number(row["travel_direction_deg"]),
                _number(row["seam_angle_deg"]), row["clicked_at"])
    return seams


def save_manual_seams(seams: list[ManualSeam], path: Path) -> None:
    """Write seams to path, replacing the file in one step so it is never left half written."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for seam in seams:
            start = (None, None) if seam.start is None else seam.start
            end = (None, None) if seam.end is None else seam.end
            writer.writerow([seam.image, "yes" if seam.seen else "no",
                             _format(start[0], 2), _format(start[1], 2), _format(end[0], 2), _format(end[1], 2),
                             _format(seam.seam_line_deg, 3), _format(seam.travel_direction_deg, 3),
                             _format(seam.seam_angle_deg, 3), seam.clicked_at])
    temporary.replace(path)


def _number(text: str) -> float | None:
    """Return the number in a CSV cell, or None if the cell is blank."""
    return float(text) if text.strip() else None


def _format(value: float | None, decimals: int) -> str:
    """Format a number to decimals places, or leave it blank when there is none."""
    return "" if value is None else f"{value:.{decimals}f}"


def _now() -> str:
    """Return the date and time now, to the second."""
    return datetime.now().isoformat(timespec="seconds")


class SeamClicker:
    """Window that shows the sample images one at a time, enlarged, for clicking the seam on each."""

    window_name = "Seam Detector Comparison - Click the Seam"
    """Title of the window."""

    def __init__(self, images: list[BallImage], path: Path) -> None:
        """Set up the window for images, carrying on from the seams already saved in path."""
        self.images = images
        self.path = path
        self.crops = [image.load() for image in images]
        self.seams = load_manual_seams(path)  # kept even for images no longer in the folder
        self.points: list[tuple[float, float]] = []  # a seam being clicked, in image px
        self.mouse: tuple[int, int] | None = None
        self.index = 0
        first = self._first_to_do(0)
        if first is not None:
            self.index = first
        self._save_pending = False
        self._seen_window = False

    # ----------------------------------------------------------------- loop

    def run(self) -> None:
        """Show the window until Q or Esc is pressed or it is closed, then print what was done."""
        self._print_controls()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, CANVAS_WIDTH, CANVAS_HEIGHT)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            while True:
                cv2.imshow(self.window_name, self.render())
                code = cv2.waitKeyEx(15)
                if self._window_closed() or self._on_key(code) == "quit":
                    break
        finally:
            try:
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass
            self._print_summary(self._save_on_close())

    def _print_controls(self) -> None:
        """Print the instructions and controls for the window."""
        done, _ = self.counts()
        print("\n=== SEAM DETECTOR COMPARISON - CLICK THE SEAM ===")
        print(f"{len(self.images)} sample images, {done} done already. On each, click two points along "
              "the seam, one near each end. The seam angle is not shown, so the clicks stay blind.")
        print("Controls:")
        print("- Click on the ball: Seam point (the second click saves the seam; clicking again starts a new one)")
        print("- Space or Enter: Next image without a seam")
        print("- A/D, Left/Right arrows or mouse wheel: Previous/next image")
        print("- N: No seam visible on this image")
        print("- U: Undo the last seam point")
        print("- R: Remove this image's seam")
        print("- Q or ESC: Finish (every seam is saved as soon as it is clicked)")

    def _save_on_close(self) -> Path:
        """Save any seams not saved yet, to a new file if the CSV file is still open in another
        program, and return the file they are in.
        """
        if self._save_pending:
            self.save()
        if not self._save_pending:
            return self.path
        saved_to = self.path.with_name(f"{self.path.stem}_unsaved_{datetime.now():%Y%m%d_%H%M%S}.csv")
        save_manual_seams(self._ordered_seams(), saved_to)
        print(f"\n{self.path.name} is still open in another program, so the seams were saved to {saved_to.name} "
              f"instead. Close the other program and rename that file to {self.path.name}.")
        return saved_to

    def _print_summary(self, saved_to: Path) -> None:
        """Print how many images are done, where their seams were saved, and what to do next."""
        done, unseen = self.counts()
        total = len(self.images)
        print(f"\nSeams saved for {done} of {total} images in {saved_to}"
              + (f" ({unseen} with no seam visible)." if unseen else "."))
        if done < total:
            print(f"{total - done} still to do: run this again to carry on.")
        else:
            print("Next, compare them with the automatic detector: "
                  "python -m report_data.seam_detector_comparison.compare_seam_angles")

    def _window_closed(self) -> bool:
        """Return True once the window has been shown and then closed with its X button."""
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return self._seen_window
        if visible >= 1:
            self._seen_window = True
            return False
        return self._seen_window

    # ---------------------------------------------------------------- input

    def _on_key(self, code: int) -> str | None:
        """Handle one waitKeyEx code. Returns "quit" to close the window, otherwise None."""
        if code == -1:
            return None
        if code in LEFT_KEYS:
            self.show(self.index - 1)
            return None
        if code in RIGHT_KEYS:
            self.show(self.index + 1)
            return None
        if code > 0xFF and code & 0xFFFF >= 0xFF00:
            return None  # another special key on Linux, such as up or down

        key = code & 0xFF
        if key in (ord("q"), ord("Q"), ESC):
            return "quit"
        if key in (ord(" "), ENTER, LINE_FEED):
            self.next_to_do()
        elif key in (ord("a"), ord("A")):
            self.show(self.index - 1)
        elif key in (ord("d"), ord("D")):
            self.show(self.index + 1)
        elif key in (ord("n"), ord("N")):
            self.mark_no_seam()
        elif key in (ord("u"), ord("U")):
            self.undo()
        elif key in (ord("r"), ord("R")):
            self.reset()
        return None

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        """Handle mouse moves, the wheel and clicks on the image."""
        # x and y are window pixels: OpenCV scales them back if the window is resized.
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.show(self.index - 1 if _wheel_delta(flags) > 0 else self.index + 1)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse = (x, y)
            if _inside_view(x, y):
                self.add_point(*self._canvas_to_image(x, y))

    # ---------------------------------------------------------------- state

    def show(self, index: int) -> None:
        """Show the image at index, stopping at the first and last, and drop any half-clicked seam."""
        index = min(max(index, 0), len(self.images) - 1)
        if index != self.index:
            self.points = []
        self.index = index

    def add_point(self, x: float, y: float) -> None:
        """Add a point along the seam of the image on show, saving the seam at the second point."""
        self.points.append((x, y))
        if len(self.points) < 2:
            return
        start, end = self.points
        if math.dist(start, end) < MIN_SEAM_PX:
            self.points = [start]  # the same point twice: wait for the other end
            return
        self.points = []
        image = self.images[self.index]
        self.seams[image.path.name] = clicked_seam(image, start, end)
        self.save()

    def mark_no_seam(self) -> None:
        """Note that no seam can be seen on the image on show, then go to the next one to do."""
        image = self.images[self.index]
        self.points = []
        self.seams[image.path.name] = no_seam(image)
        self.save()
        self.next_to_do()

    def undo(self) -> None:
        """Remove the last seam point, turning a saved seam back into a half-clicked one."""
        if self.points:
            self.points.pop()
            return
        seam = self.seams.get(self.images[self.index].path.name)
        if seam is not None and seam.start is not None:
            self.points = [seam.start]
            del self.seams[seam.image]
            self.save()

    def reset(self) -> None:
        """Remove the seam of the image on show, whether saved or half-clicked."""
        self.points = []
        if self.seams.pop(self.images[self.index].path.name, None) is not None:
            self.save()

    def next_to_do(self) -> None:
        """Go to the next image without a seam, going round to the start, if there is one."""
        upcoming = self._first_to_do(self.index + 1)
        if upcoming is not None:
            self.show(upcoming)

    def _first_to_do(self, start: int) -> int | None:
        """Return the first image from start on without a seam, going round to the start, or None."""
        count = len(self.images)
        for step in range(count):
            index = (start + step) % count
            if self.images[index].path.name not in self.seams:
                return index
        return None

    def counts(self) -> tuple[int, int]:
        """Return how many images have a seam saved, and how many of those had no seam visible."""
        saved = [self.seams[image.path.name] for image in self.images if image.path.name in self.seams]
        return len(saved), sum(1 for seam in saved if not seam.seen)

    def save(self) -> None:
        """Write every seam to the CSV file. If the file is open in another program, it is
        tried again at the next change and when the window closes.
        """
        try:
            save_manual_seams(self._ordered_seams(), self.path)
        except PermissionError:
            if not self._save_pending:
                print(f"Could not save {self.path.name}: close it if it is open in another program. "
                      "It is saved again at the next click.")
            self._save_pending = True
            return
        self._save_pending = False

    def _ordered_seams(self) -> list[ManualSeam]:
        """Return every seam in image order, then any for images no longer in the folder."""
        names = [image.path.name for image in self.images]
        known = set(names)
        return ([self.seams[name] for name in names if name in self.seams]
                + [seam for name, seam in self.seams.items() if name not in known])

    # ---------------------------------------------------------- coordinates

    def _canvas_to_image(self, x: int, y: int) -> tuple[float, float]:
        """Convert a window pixel on the enlarged image to a pixel of the image on show."""
        image_per_view = self.images[self.index].size_px / VIEW_SIZE
        # cv2.resize puts pixel centres at half-pixel positions, so map through those.
        return ((x - VIEW_X + 0.5) * image_per_view - 0.5,
                (y - VIEW_Y + 0.5) * image_per_view - 0.5)

    def _image_to_view(self, x: float, y: float) -> tuple[int, int]:
        """Convert a pixel of the image on show to a pixel of the enlarged view, for drawing."""
        view_per_image = VIEW_SIZE / self.images[self.index].size_px
        return (int(round((x + 0.5) * view_per_image - 0.5)),
                int(round((y + 0.5) * view_per_image - 0.5)))

    # -------------------------------------------------------------- drawing

    def render(self) -> np.ndarray:
        """Draw the whole window and return it as an image."""
        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), BACKGROUND, dtype=np.uint8)
        view = cv2.resize(self.crops[self.index], (VIEW_SIZE, VIEW_SIZE), interpolation=cv2.INTER_LINEAR)
        self._draw_seam(view)
        canvas[VIEW_Y:VIEW_Y + VIEW_SIZE, VIEW_X:VIEW_X + VIEW_SIZE] = view
        self._draw_info(canvas)
        self._draw_progress(canvas)
        _text(canvas, "click: seam point    u: undo    r: reset    n: no seam visible",
              VIEW_X, HINT_Y, 0.45, HINT_TEXT)
        _text(canvas, "space/enter: next to do    a/d: back/forward    q: finish",
              VIEW_X, HINT_Y + HINT_GAP, 0.45, HINT_TEXT)
        return canvas

    def _draw_seam(self, view: np.ndarray) -> None:
        """Draw the saved seam, or the points of a half-clicked one with a line to the mouse."""
        if self.points:
            points = [self._image_to_view(x, y) for x, y in self.points]
            if self.mouse is not None and _inside_view(*self.mouse):
                mouse = (self.mouse[0] - VIEW_X, self.mouse[1] - VIEW_Y)
                cv2.line(view, points[0], mouse, OUTLINE, 3, cv2.LINE_AA)
                cv2.line(view, points[0], mouse, SEAM_COLOUR, 1, cv2.LINE_AA)
        else:
            seam = self.seams.get(self.images[self.index].path.name)
            if seam is None or seam.start is None or seam.end is None:
                return
            points = [self._image_to_view(*seam.start), self._image_to_view(*seam.end)]
            # The line gets a dark outline so it still shows on top of a white seam.
            cv2.line(view, points[0], points[1], OUTLINE, 4, cv2.LINE_AA)
            cv2.line(view, points[0], points[1], SEAM_COLOUR, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(view, point, 6, OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(view, point, 5, SEAM_COLOUR, -1, cv2.LINE_AA)

    def _draw_info(self, canvas: np.ndarray) -> None:
        """Image number on the left; what has been clicked, or what to do next, on the right."""
        image = self.images[self.index]
        _text(canvas, f"Image {self.index + 1} of {len(self.images)}", VIEW_X, INFO_Y, 0.6, TEXT)
        seam = self.seams.get(image.path.name)
        if self.points:
            message = "Click the other end of the seam"
        elif seam is None:
            message = "Click two points along the seam"
        elif seam.seen:
            message = "Seam saved"
        else:
            message = "No seam visible"
        width = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        _text(canvas, message, VIEW_X + VIEW_SIZE - width, INFO_Y, 0.6, TEXT)

    def _draw_progress(self, canvas: np.ndarray) -> None:
        """Draw the progress bar, how many images are done and the file name of the one on show."""
        done, unseen = self.counts()
        total = len(self.images)
        bottom = PROGRESS_Y + PROGRESS_HEIGHT - 1
        cv2.rectangle(canvas, (VIEW_X, PROGRESS_Y), (VIEW_X + VIEW_SIZE - 1, bottom), THUMB_BORDER, -1)
        filled = round(VIEW_SIZE * done / total) if total else 0
        if filled:
            cv2.rectangle(canvas, (VIEW_X, PROGRESS_Y), (VIEW_X + filled - 1, bottom), TEXT, -1)

        if self._save_pending:
            line = f"Not saved: close {self.path.name} in other programs"
        elif done == total:
            line = f"All {total} done - press Q to finish"
        else:
            line = f"{done} of {total} done" + (f", {unseen} with no seam visible" if unseen else "")
        _text(canvas, line, VIEW_X, COUNT_Y, 0.5, TEXT)
        if not self._save_pending:  # the warning needs the whole line
            name = self.images[self.index].path.name
            width = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            _text(canvas, name, VIEW_X + VIEW_SIZE - width, COUNT_Y, 0.5, HINT_TEXT)


def main(argv: list[str] | None = None) -> None:
    """Open the window on the sample images, carrying on from any seams already clicked."""
    parser = argparse.ArgumentParser(description="Click the seam on each sample ball image, for comparing "
                                                 "with the automatic seam angle detector.")
    parser.parse_args(argv)
    if not (IMAGE_FOLDER / INDEX_FILE_NAME).exists():
        print(f"There are no sample images in {IMAGE_FOLDER} yet. Make them first: "
              "python -m report_data.seam_detector_comparison.make_seam_images")
        raise SystemExit(1)
    SeamClicker(load_ball_images(IMAGE_FOLDER), MANUAL_PATH).run()


if __name__ == "__main__":
    main()
