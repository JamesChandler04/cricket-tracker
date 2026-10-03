"""Utility script that steps through a video and saves chosen frames as PNG images."""

import sys
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library import paths
from library.drawers import Drawers
from library.helpers import Key, Video

FRAMES_DIR = paths.OUTPUT_DIR / "frames"
"""Folder the saved frames go in, in a subfolder named after the video."""
WINDOW_NAME = "Frame Saver"
"""Title of the frame saver window."""


def choose_video() -> str:
    """Ask for a video with a file dialog. Returns an empty string if none is chosen."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select a video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def ask_start_frame(total_frames: int) -> int:
    """Ask for the starting frame, kept within the video; 0 if blank or not a number."""
    answer = input("Enter the starting frame number (default 0): ").strip()
    if not answer:
        return 0
    try:
        return min(max(int(answer), 0), total_frames - 1)
    except ValueError:
        print(f"'{answer}' is not a frame number, starting at frame 0.")
        return 0


def main(argv: list[str] | None = None) -> None:
    """Run the frame saver on the video path given, or one picked in a file dialog."""
    arguments = sys.argv[1:] if argv is None else argv
    video_path = arguments[0] if arguments else choose_video()
    if not video_path:
        print("No video chosen.")
        return

    video = Video(video_path)
    video.current_frame = ask_start_frame(video.total_frames)
    save_dir = FRAMES_DIR / Path(video_path).stem
    saved: set[int] = set()
    drawers = Drawers()

    print("\n=== FRAME SAVER ===")
    print(f"Saved frames go to {save_dir}")
    print("Controls:")
    print("- a/d: Move back/forward 1 frame (A/D: 10 frames)")
    print("- s: Save the frame on screen")
    print("- o: Rotate the video 90 degrees clockwise")
    print("- q or ESC: Quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    seen_window = False
    try:
        while True:
            frame = video.get_current_frame()
            readable = frame is not None
            if frame is None:
                frame = np.zeros((video.frame_height, video.frame_width, 3), dtype=np.uint8)

            # Draw on a copy, so what gets saved is the clean frame.
            shown = frame.copy()
            status = f"Frame {video.current_frame}/{video.total_frames - 1}"
            if not readable:
                status += " - could not read this frame"
            elif video.current_frame in saved:
                status += " - SAVED"
            drawers.draw_text(shown, status, (10, 40), font_scale=1.1)
            drawers.draw_text(shown, f"Saved: {len(saved)}", (10, 80), font_scale=0.9)
            drawers.draw_text(shown, "a/d: 1 frame  A/D: 10 frames  s: save  o: rotate  q: quit",
                              (10, 120), font_scale=0.9)
            cv2.imshow(WINDOW_NAME, shown)

            key_code = int(cv2.waitKey(10) & 0xFF)

            # Stop if the window was closed with its X button.
            try:
                visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                visible = -1
            if visible >= 1:
                seen_window = True
            elif seen_window:
                break

            try:
                key = Key(key_code)
            except ValueError:
                key = None

            match key:
                case Key.q | Key.esc:
                    break
                case Key.a:
                    video.change_frame(-1)
                case Key.d:
                    video.change_frame(1)
                case Key.A:
                    video.change_frame(-10)
                case Key.D:
                    video.change_frame(10)
                case Key.o:
                    video.rotate()
                    print(f"Rotated to {video.rotation} degrees")
                case Key.s:
                    if not readable:
                        print(f"Frame {video.current_frame} could not be read, so it was not saved.")
                        continue
                    save_dir.mkdir(parents=True, exist_ok=True)
                    path = save_dir / f"frame_{video.current_frame:06d}.png"
                    if cv2.imwrite(str(path), frame):
                        saved.add(video.current_frame)
                        print(f"Saved {path}")
                    else:
                        print(f"Could not save {path}")
    finally:
        video.cap.release()
        cv2.destroyAllWindows()

    print(f"\n{len(saved)} frames saved to {save_dir}" if saved else "\nNo frames saved.")


if __name__ == "__main__":
    main()