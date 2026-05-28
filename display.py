import cv2
from helpers import Video

class Display:
    def load_main_video(self) -> Video:
        top_down_video_path = input("Enter the path to your main cricket video file (bird's eye view): ").strip().strip('"')
        if not top_down_video_path:
            raise ValueError("No top down view video path provided.")
        top_down_video = Video(top_down_video_path)
        
        return top_down_video

    def load_side_video(self) -> Video:
        side_on_video_path = input("Enter the path to the side view video file: ").strip().strip('"')
        if not side_on_video_path:
            raise ValueError("No side on view video path provided.")
        side_video = Video(side_on_video_path)
        
        return side_video