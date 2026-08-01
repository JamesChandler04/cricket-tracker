import sys
import json
import numpy as np
import cv2

from helpers import Coord

# ---------------------------------------------------------------------------
# CONFIG - edit these for your rig
# ---------------------------------------------------------------------------
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 900

# Distance in metres from release to each ring's vertical face (4 rings from 3 chained cubes)
RING_FORWARD_DISTANCES_M = [4.2, 7.2, 10.2, 13.2]
TUNNEL_HALF_WIDTH_M = 1.5
TUNNEL_HALF_HEIGHT_M = 1.5

CALIBRATION_OUTPUT_PATH = "camera_calibration.npz"
CORNER_LABELS = ["top_left", "top_right", "bottom_right", "bottom_left"]


# ---------------------------------------------------------------------------
# Camera resectioning (Direct Linear Transform)
# ---------------------------------------------------------------------------
def resection_camera(
    ring_forward_distances_m: list[float],
    ring_corners_px: list[tuple[Coord, Coord, Coord, Coord]],
    tunnel_half_width_m: float = 1.5,
    tunnel_half_height_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Recovers the camera's full intrinsics + pose from the ring corners.
    Returns (K_inv, R_T, t_std) - used directly by pixel_to_xyz.
    '''
    object_points = []
    for depth in ring_forward_distances_m:
        object_points += [
            (-tunnel_half_width_m, depth, -tunnel_half_height_m),  # top_left
            (tunnel_half_width_m, depth, -tunnel_half_height_m),   # top_right
            (tunnel_half_width_m, depth, tunnel_half_height_m),    # bottom_right
            (-tunnel_half_width_m, depth, tunnel_half_height_m),   # bottom_left
        ]
    object_points = np.array(object_points, dtype=np.float64)

    image_points = np.array(
        [(c.x, c.y) for corners in ring_corners_px for c in corners], dtype=np.float64
    )

    if len(object_points) < 6:
        raise ValueError(f"Need at least 6 corner correspondences (2+ full rings), got {len(object_points)}.")

    A = []
    for (X, Y, Z), (u, v) in zip(object_points, image_points):
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    K, R, t_hom, *_ = cv2.decomposeProjectionMatrix(P)
    K = K / K[2, 2]
    cam_center = (t_hom[:3] / t_hom[3]).flatten()
    t_std = -R @ cam_center

    K_inv = np.linalg.inv(K)
    return K_inv, R.T, t_std


def save_calibration(path: str, K_inv: np.ndarray, R_T: np.ndarray, t_std: np.ndarray) -> None:
    np.savez(path, K_inv=K_inv, R_T=R_T, t_std=t_std)


def load_calibration(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Use this in your main tracking program to load the saved calibration.'''
    data = np.load(path)
    return data["K_inv"], data["R_T"], data["t_std"]


# ---------------------------------------------------------------------------
# Click tool
# ---------------------------------------------------------------------------
def main(video_path: str):
    num_rings = len(RING_FORWARD_DISTANCES_M)
    num_points_needed = num_rings * 4

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read a frame from {video_path}")

    frame_h, frame_w = frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / frame_w, MAX_DISPLAY_HEIGHT / frame_h)
    display_frame = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale))) if scale < 1.0 else frame.copy()

    points: list[tuple[int, int]] = []
    window_name = "Click ring corners in order - 'z' undo, 's' save+calibrate, 'q'/Esc quit"

    def next_label() -> str:
        ring_idx = len(points) // 4
        corner_idx = len(points) % 4
        if ring_idx >= num_rings:
            return "done - press 's' to calibrate"
        return f"ring {ring_idx + 1}/{num_rings}, {CORNER_LABELS[corner_idx]} (corner {len(points) + 1}/{num_points_needed})"

    def redraw():
        img = display_frame.copy()
        for i, (x, y) in enumerate(points):
            dx, dy = int(x * scale), int(y * scale)
            cv2.circle(img, (dx, dy), 4, (0, 0, 255), -1)
            cv2.putText(img, str(i), (dx + 8, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.setWindowTitle(window_name, f"Next: {next_label()}")
        cv2.imshow(window_name, img)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points_needed:
            orig_x, orig_y = round(x / scale), round(y / scale)
            points.append((orig_x, orig_y))
            print(f"[{len(points) - 1}] x={orig_x}, y={orig_y}  ({next_label()})")
            redraw()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)
    redraw()

    print(f"Frame size: {frame_w}x{frame_h} (displayed at {scale:.2f}x scale)")
    print(f"Click {num_points_needed} corners total, in order: for each ring, {' -> '.join(CORNER_LABELS)}.")
    print("'z' = undo last, 's' = save + calibrate, 'q'/Esc = quit without saving.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('z') and points:
            removed = points.pop()
            print(f"Removed point {removed}  ({next_label()})")
            redraw()
        elif key == ord('s'):
            if len(points) != num_points_needed:
                print(f"Need exactly {num_points_needed} points, have {len(points)} - keep clicking or undo.")
                continue

            ring_corners_px = []
            for i in range(num_rings):
                corners = [Coord(x=px, y=py) for px, py in points[i*4:(i+1)*4]]
                ring_corners_px.append(tuple(corners))

            with open("ring_corners.json", "w") as f:
                json.dump(points, f, indent=2)

            K_inv, R_T, t_std = resection_camera(
                RING_FORWARD_DISTANCES_M, ring_corners_px,
                TUNNEL_HALF_WIDTH_M, TUNNEL_HALF_HEIGHT_M,
            )
            save_calibration(CALIBRATION_OUTPUT_PATH, K_inv, R_T, t_std)
            print(f"Saved raw corners to ring_corners.json")
            print(f"Saved calibration matrices to {CALIBRATION_OUTPUT_PATH}")
            break
        elif key == ord('q') or key == 27:
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python calibrate.py <video_path>")
    main(sys.argv[1])