import cv2

class Display:
    def transform_frame(self, frame, rotation):
        """Apply rotation to a frame"""
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    
    def advance_frame(self, current_frame, total_frames):
        if current_frame < total_frames - 1:
            current_frame += 1
        
        return current_frame

    def previous_frame(self, current_frame):
        if current_frame > 0:
            current_frame -= 1
        return current_frame

    def load_main_video(self):
        video_path = input("Enter the path to your main cricket video file (bird's eye view): ").strip().strip('"')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open main video file {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Main video loaded: {total_frames} frames")
        print(f"Main resolution: {frame_width}x{frame_height}")

        if fps == 0:
            print("Warning: Could not determine FPS from video metadata.")
            while True:
                try:
                    fps = float(input("Enter the frame rate (FPS) of the video: ").strip())
                    if fps <= 0:
                        raise ValueError
                    break
                except ValueError:
                    print("Please enter a positive number for FPS.")
        
        print(f"Main video FPS: {fps:.2f}")
        
        return cap, total_frames, frame_width, frame_height, fps

    def load_side_video(self):
        video_path_side = input("Enter the path to the side view video file: ").strip().strip('"')
        if not video_path_side:
            raise ValueError("No side view video path provided.")
        cap_side = cv2.VideoCapture(video_path_side)
        if not cap_side.isOpened():
            raise ValueError(f"Error: Could not open side view video file {video_path_side}")

        total_frames_side = int(cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width_side = int(cap_side.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height_side = int(cap_side.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Side video loaded: {total_frames_side} frames")
        print(f"Side resolution: {frame_width_side}x{frame_height_side}")

        return cap_side, total_frames_side, frame_width_side, frame_height_side