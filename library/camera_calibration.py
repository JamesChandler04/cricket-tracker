"""Calibrates the side-on camera: the user clicks the calibration cubes' ring corners in
one frame, and the camera is solved from them by DLT and saved as an .npz file.
"""

from __future__ import annotations

import sys
import json
import numpy as np
import cv2
from tkinter import Tk, filedialog
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from library import paths
from library.helpers import Coord

MAX_DISPLAY_WIDTH = 1400
"""Largest width the frame is shown at, in px; wider frames are scaled down."""
MAX_DISPLAY_HEIGHT = 900
"""Largest height the frame is shown at, in px; taller frames are scaled down."""

ZOOM_FACTOR = 10
"""How many times the zoom view, toggled with 'z', magnifies the normal view."""
ZOOM_INTERPOLATION = cv2.INTER_NEAREST
"""Zoom interpolation: nearest-neighbour, so each raw pixel shows as a sharp block."""

RING_FORWARD_DISTANCES_M = [0, 3, 6, 9]
"""Distance from release to each calibration ring, in m (4 rings, 3 chained cubes)."""
TUNNEL_HALF_WIDTH_M = 1.5
"""Half the width of each calibration ring, in m."""
TUNNEL_HALF_HEIGHT_M = 1.5
"""Half the height of each calibration ring, in m."""

CALIBRATION_DEFAULT_NAME = "camera_calibration"
"""File name the calibration is saved under when none is given (np.savez adds .npz)."""
CORNER_LABELS = ["top_left", "top_right", "bottom_right", "bottom_left"]
"""Names of each ring's four corners, in the order they are clicked."""


def resection_camera(
    ring_forward_distances_m: Sequence[float],
    ring_corners_px: Sequence[Sequence[Coord]],
    tunnel_half_width_m: float = 1.5,
    tunnel_half_height_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recovers the camera's full intrinsics + pose from the ring corners.
    Returns (K_inv, R_T, t_std) - used directly by pixel_to_xyz.
    """
    corner_list: list[tuple[float, float, float]] = []
    for depth in ring_forward_distances_m:
        corner_list += [
            (-tunnel_half_width_m, depth, -tunnel_half_height_m),  # top_left
            (tunnel_half_width_m, depth, -tunnel_half_height_m),   # top_right
            (tunnel_half_width_m, depth, tunnel_half_height_m),    # bottom_right
            (-tunnel_half_width_m, depth, tunnel_half_height_m),   # bottom_left
        ]
    object_points = np.array(corner_list, dtype=np.float64)

    image_points = np.array(
        [(c.x, c.y) for corners in ring_corners_px for c in corners], dtype=np.float64
    )

    if len(object_points) < 6:
        raise ValueError(f"Need at least 6 corner correspondences (2+ full rings), got {len(object_points)}.")

    rows = []
    for (X, Y, Z), (u, v) in zip(object_points, image_points):
        rows.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        rows.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    A = np.array(rows)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    K, R, t_hom, *_ = cv2.decomposeProjectionMatrix(P)
    K = K / K[2, 2]
    cam_center = (t_hom[:3] / t_hom[3]).flatten()
    t_std = -R @ cam_center

    K_inv = np.linalg.inv(K)
    return K_inv, R.T, t_std


def save_calibration(K_inv: np.ndarray, R_T: np.ndarray, t_std: np.ndarray) -> str:
    """Ask the user for a folder and file name, save the calibration there with np.savez
    (which adds .npz if missing) and return that path.
    """
    dir = input("What folder do you want to save this to? (Hit enter for default)\n")
    name = input("What do you want to call the calibration file? (Hit enter for default)\n")
    if not name:
        name = CALIBRATION_DEFAULT_NAME

    if dir:
        os.makedirs(dir, exist_ok=True)

    # Hitting enter for the folder saves into the project folder.
    path = (Path(dir) if dir else paths.ROOT) / name

    np.savez(path, K_inv=K_inv, R_T=R_T, t_std=t_std)

    return path.__str__()


def load_calibration(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the K_inv, R_T and t_std arrays from a saved calibration file."""
    data = np.load(path)
    return data["K_inv"], data["R_T"], data["t_std"]

def choose_file_path() -> str | None:
    """Ask the user to pick a file and return its path, or None if cancelled."""
    root = Tk()
    root.withdraw()
    
    absolute_path = filedialog.askopenfilename(
        title="Select a File",
        filetypes=[("All Files", "*.*")]
    )
    
    if not absolute_path:
        return None
    
    return absolute_path

def main(video_path: str) -> None:
    """Ask which frame of the video to use, let the user click every ring corner on it,
    then solve the camera and save the calibration.
    """
    frame_number = int(input("What frame do you want to calibrate on?\n"))
    num_rings = len(RING_FORWARD_DISTANCES_M)
    num_points_needed = num_rings * 4

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read a frame from {video_path}")

    frame_h, frame_w = frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / frame_w, MAX_DISPLAY_HEIGHT / frame_h)
    display_frame = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale))) if scale < 1.0 else frame.copy()
    display_h, display_w = display_frame.shape[:2]

    points: list[tuple[int, int]] = []
    mouse_pos: tuple[int, int] | None = None
    zoom_active = False
    zoom_centre: tuple[int, int] | None = None
    zoom_transform = None
    window_name = "Click ring corners in order - 'z' zoom, 'u' undo, 's' save+calibrate, 'q'/Esc quit"

    def next_label() -> str:
        """Return a label for the next corner to click, or a save prompt once done."""
        ring_idx = len(points) // 4
        corner_idx = len(points) % 4
        if ring_idx >= num_rings:
            return "done - press 's' to calibrate"
        return f"ring {ring_idx + 1}/{num_rings}, {CORNER_LABELS[corner_idx]} (corner {len(points) + 1}/{num_points_needed})"

    def apply_zoom() -> np.ndarray:
        """Crop a ZOOM_FACTOR-smaller region of the raw frame and blow it back up to the display size."""
        nonlocal zoom_transform
        crop_w = max(1, int(round(frame_w / ZOOM_FACTOR)))
        crop_h = max(1, int(round(frame_h / ZOOM_FACTOR)))

        assert zoom_centre is not None  # only called while zoom is on
        centre_x, centre_y = zoom_centre
        x0 = min(max(int(centre_x) - crop_w // 2, 0), max(frame_w - crop_w, 0))
        y0 = min(max(int(centre_y) - crop_h // 2, 0), max(frame_h - crop_h, 0))

        crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        zoomed = cv2.resize(crop, (display_w, display_h), interpolation=ZOOM_INTERPOLATION)

        # Store the actual scales rather than deriving them from ZOOM_FACTOR: display_w / crop_w
        # is only exactly ZOOM_FACTOR * scale when the frame size divides evenly.
        zoom_transform = (x0, y0, display_w / crop_w, display_h / crop_h, crop_w, crop_h)
        return zoomed

    def view_to_frame_coords(x: int, y: int) -> tuple[int, int]:
        """
        Map a coordinate from the displayed view back to raw frame pixels.

        With zoom off this is the plain display-scale inverse. With zoom on it inverts the
        crop-and-resize exactly, so a click resolves to the raw pixel actually drawn under
        the cursor.
        """
        if not zoom_active or zoom_transform is None:
            return round(x / scale), round(y / scale)

        x0, y0, scale_x, scale_y, crop_w, crop_h = zoom_transform
        # cv2.resize with INTER_NEAREST maps dst pixel j -> src pixel floor(j * (1 / scale)),
        # clamped to the source size. Multiplying by the reciprocal rather than dividing
        # matters: at exact pixel boundaries the two disagree by one pixel.
        col = min(int(max(0.0, x) * (1.0 / scale_x)), crop_w - 1)
        row = min(int(max(0.0, y) * (1.0 / scale_y)), crop_h - 1)
        return x0 + col, y0 + row

    def frame_to_view_coords(x: int, y: int) -> tuple[int, int]:
        """Map raw frame pixels to the displayed view, for drawing the corner markers."""
        if not zoom_active or zoom_transform is None:
            return int(x * scale), int(y * scale)

        x0, y0, scale_x, scale_y, _crop_w, _crop_h = zoom_transform
        # +0.5 puts the marker in the centre of the magnified pixel block.
        return int(round((x - x0 + 0.5) * scale_x)), int(round((y - y0 + 0.5) * scale_y))

    def toggle_zoom() -> None:
        """Turn the zoom view on around the mouse, or back off."""
        nonlocal zoom_active, zoom_centre, zoom_transform
        if zoom_active:
            zoom_active = False
            zoom_centre = None
            zoom_transform = None
            print("Zoom off.")
            return

        if mouse_pos is None:
            print("Move the mouse over the window before pressing 'z'.")
            return

        zoom_active = True
        zoom_centre = mouse_pos
        print(f"Zoom on: {ZOOM_FACTOR}x around ({zoom_centre[0]}, {zoom_centre[1]}). Press 'z' again to zoom out.")

    def redraw() -> None:
        """Redraw the frame (zoomed if on), the numbered corners and the title."""
        # Zoom is applied to the raw frame first, then markers are drawn on top at normal
        # size so they stay readable at any ZOOM_FACTOR.
        img = apply_zoom() if zoom_active and zoom_centre is not None else display_frame.copy()
        for i, (x, y) in enumerate(points):
            dx, dy = frame_to_view_coords(x, y)
            cv2.circle(img, (dx, dy), 4, (0, 0, 255), -1)
            cv2.putText(img, str(i), (dx + 8, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        zoom_status = f"{ZOOM_FACTOR}x @ ({zoom_centre[0]}, {zoom_centre[1]})" if zoom_active and zoom_centre is not None else "Off"
        cv2.setWindowTitle(window_name, f"Next: {next_label()}  |  Zoom: {zoom_status}")
        cv2.imshow(window_name, img)

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Track the mouse and add a corner, in raw frame pixels, on each left click."""
        nonlocal mouse_pos
        # Everything below works in raw frame coordinates, regardless of zoom state.
        x, y = view_to_frame_coords(x, y)

        if event == cv2.EVENT_MOUSEMOVE:
            mouse_pos = (x, y)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pos = (x, y)
            if len(points) < num_points_needed:
                points.append((x, y))
                print(f"[{len(points) - 1}] x={x}, y={y}  ({next_label()})")
                redraw()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    print(f"Frame size: {frame_w}x{frame_h} (displayed at {scale:.2f}x scale)")
    print(f"Click {num_points_needed} corners total, in order: for each ring, {' -> '.join(CORNER_LABELS)}.")
    print(f"'z' = toggle {ZOOM_FACTOR}x zoom centred on the mouse, 'u' = undo last, 's' = save + calibrate, 'q'/Esc = quit without saving.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('z'):
            toggle_zoom()
            redraw()
        elif key == ord('u') and points:
            removed = points.pop()
            print(f"Removed point {removed}  ({next_label()})")
            redraw()
        elif key == ord('s'):
            if len(points) != num_points_needed:
                print(f"Need exactly {num_points_needed} points, have {len(points)} - keep clicking or undo.")
                continue

            cv2.destroyAllWindows()

            ring_corners_px = []
            for i in range(num_rings):
                corners = [Coord(x=px, y=py) for px, py in points[i*4:(i+1)*4]]
                ring_corners_px.append(tuple(corners))

            K_inv, R_T, t_std = resection_camera(
                RING_FORWARD_DISTANCES_M, ring_corners_px,
                TUNNEL_HALF_WIDTH_M, TUNNEL_HALF_HEIGHT_M,
            )
            save_path = save_calibration(K_inv, R_T, t_std)

            print(f"Saved calibration matrices to {save_path}")
            break
        elif key == ord('q') or key == 27:
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Choose a video to calibrate on.")
    file_path = choose_file_path()
    if not file_path:
        raise FileNotFoundError("No file chosen.")
    main(file_path)