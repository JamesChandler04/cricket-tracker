import cv2
import os
import tkinter as tk
from tkinter import filedialog
from helpers import Video

class Display:
    def load_main_video(self) -> Video:
        print("Please select the main cricket video file (bird's eye view).")

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        abs_path = filedialog.askopenfilename(
            title="Select the main cricket video file (bird's eye view)",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        root.destroy()

        if not abs_path:
            raise ValueError("No top down view video path provided.")

        rel_path = os.path.relpath(abs_path, os.path.dirname(__file__))
        return Video(rel_path)

    def load_side_video(self) -> Video:
        print("Please select the side view video file (side view).")

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        abs_path = filedialog.askopenfilename(
            title="Select the side view video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        root.destroy()

        if not abs_path:
            raise ValueError("No side on view video path provided.")

        rel_path = os.path.relpath(abs_path, os.path.dirname(__file__))
        return Video(rel_path)