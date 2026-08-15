import sys

import cv2

import checkers
import drawers
import display
import calculators
from helpers import Key
from automations import TopDownBallDataPoint, SideOnBallDataPoint


ZOOM_FACTOR = 10
ZOOM_INTERPOLATION = cv2.INTER_NEAREST


class TopDownTracker:
    def __init__(self):
        self.top_down_video = None
        self.frame_positions = []
        self.seam_points = []
        self.seam_measurements = []
        self.calibrations = []
        self.meters_per_pixel = None
        self.deceleration = None
        self.calibration_active = False
        self.tracking_active = False
        self.seam_angle_active = False
        self.window_name = "Cricket Ball Tracker - Main View"
        self.ball_diameter_m = 0.073
        self.first_frame_main = None

        self.display = display.Display()
        self.drawers = drawers.Drawers()
        self.checkers = checkers.Checker()
        self.calculators = calculators.Calculators()

    def _add_or_replace_point_for_frame(self, frame_no, x, y, t):
            positions = self.frame_positions
            for i, (f, *_rest) in enumerate(positions):
                if f == frame_no:
                    positions[i] = (frame_no, int(x), int(y), t)
                    return
            positions.append((frame_no, int(x), int(y), t))
            positions.sort(key=lambda z: z[0])

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.calibration_active:
                if not self.calibrations or len(self.calibrations[-1][1]) == 2:
                    self.calibrations.append((self.top_down_video.get_current_frame_number(), []))
                self.calibrations[-1][1].append((x, y))
                print(f"Main diameter point added in frame {self.top_down_video.get_current_frame_number()}: ({x}, {y})")
                if len(self.calibrations[-1][1]) == 2:
                    self.calibrations, self.meters_per_pixel = self.calculators._calculate_meters_per_pixel(self.calibrations, self.ball_diameter_m, self.meters_per_pixel)
                    if len(self.calibrations) >= 2:
                        self.calibration_active = False
                        print("Main second calibration completed.")
                    else:
                        print("Main first calibration completed. Navigate to another frame and press 'C'.")
            elif self.tracking_active and self.meters_per_pixel is not None:
                timestamp = self.top_down_video.get_current_frame_number() / self.top_down_video.fps
                if not self.first_frame_main:
                    self.first_frame_main = self.top_down_video.get_current_frame_number()
                    print(f"Main view Frame 1 set to raw frame {self.first_frame_main}")
                self._add_or_replace_point_for_frame(self.top_down_video.get_current_frame_number(), x, y, timestamp)
                print(f"Main View - Frame {self.top_down_video.get_current_frame_number()}: Ball at (x={x}, y={y}) - Time: {timestamp:.5f}s")
            elif self.seam_angle_active and self.meters_per_pixel is not None:
                self.seam_points.append((x, y))
                print(f"Seam point added: ({x}, {y})")
                if len(self.seam_points) == 2:
                    self.seam_measurements, self.seam_points = self.calculators._calculate_seam_angle(self.seam_points, self.seam_measurements, self.top_down_video.get_current_frame_number())
                    self.seam_angle_active = False
                    print(f"Seam angle tracking stopped for frame {self.top_down_video.get_current_frame_number()}. Angle: {self.seam_measurements[-1][1]:.2f} degrees")
    

    def run(self):
        self.top_down_video = self.display.load_main_video()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n=== CRICKET BALL TRACKER - MAIN VIEW (BIRD'S EYE) ===")
        print("Controls:")
        print("- C: Start/Stop Calibration Mode (click two points across ball diameter, repeat in another frame)")
        print("- SPACE: Start/Stop Ball Tracking Mode (after both calibrations)")
        print("- T: Start/Stop Seam Angle Tracking Mode (after both calibrations)")
        print("- A/D or ←/→: Move frame back/forward")
        print("- Click: Mark calibration points, ball position, or seam points")
        print("- F: Find ball in current frame")
        print("- B: Set current frame to background frame (for ball finding)")
        print("- S: Proceed to side view tracking")
        print("- R: Reset tracking and calibration")
        print("- Q or ESC: Quit")
        print("- O: Rotate video 90 degrees clockwise")

        while True:
            frame = self.top_down_video.get_current_frame()
            self.drawers.draw_main_trajectory(frame, self.frame_positions, self.top_down_video.current_frame, self.top_down_video.frame_width, self.seam_points, self.seam_measurements, self.calibrations)

            if self.calibration_active:
                cal_num = len(self.calibrations) + 1 if not self.calibrations or len(self.calibrations[-1][1]) == 2 else len(self.calibrations)
                points = len(self.calibrations[-1][1]) if self.calibrations else 0
                status = f"{'First' if cal_num == 1 else 'Second'} Calibration - Click point {points + 1}/2 for ball diameter"
            elif self.tracking_active:
                status = "BALL TRACKING ACTIVE - Click on ball"
            elif self.seam_angle_active:
                status = f"SEAM ANGLE TRACKING ACTIVE - Click point {len(self.seam_points) + 1}/2"
            else:
                status = f"PAUSED - Press C for {'first' if not self.calibrations else 'second'} calibration, SPACE for ball, T for seam"

            seam_angle, is_wobble, _ = self.checkers._check_seam_wobble(self.seam_measurements)
            seam_display = "Wobble Seam" if is_wobble else f"{seam_angle:.2f}°" if seam_angle is not None else "Not set"
            self.drawers.draw_text(frame, status, (10, 40), font_scale=1.1)
            self.drawers.draw_text(frame, f"Frame: {self.top_down_video.current_frame}/{self.top_down_video.total_frames-1}", (10, 80), font_scale=0.9)
            self.drawers.draw_text(frame, f"Main Points: {len(self.frame_positions)}", (10, 110), font_scale=0.9)
            self.drawers.draw_text(frame, f"Seam Angle: {seam_display}", (10, 140), font_scale=0.9)
            self.drawers.draw_text(frame, f"Deceleration: {'{:.3f} m/s^2'.format(self.deceleration) if self.deceleration is not None else 'Not set'}", (10, 170), font_scale=0.9)
            self.drawers.draw_text(frame, f"Meters/Pixel: {'{:.6f} m/px'.format(self.meters_per_pixel) if self.meters_per_pixel is not None else 'Not set'}", (10, 200), font_scale=0.9)
            self.drawers.draw_text(frame, "S = Proceed to Side View", (10, 230), font_scale=0.9)

            cv2.imshow(self.window_name, frame)
            key_code = int(cv2.waitKey(10) & 0xFF)
            try:
                key = Key(key_code)
            except ValueError:
                key = None

            match key:
                case Key.q | Key.esc:
                    self.top_down_video.cap.release()
                    cv2.destroyAllWindows()
                    sys.exit()
                case Key.c:
                    if self.tracking_active or self.seam_angle_active:
                        print("Cannot start calibration while tracking or seam angle mode is active.")
                    elif len(self.calibrations) >= 2:
                        print("Main calibrations already completed.")
                    else:
                        self.calibration_active = not self.calibration_active
                        if self.calibration_active:
                            if not self.calibrations or len(self.calibrations[-1][1]) == 2:
                                print(f"Starting {'first' if not self.calibrations else 'second'} main calibration.")
                            else:
                                print(f"Continuing {'first' if len(self.calibrations) == 1 else 'second'} main calibration.")
                        else:
                            print("Main calibration paused.")
                case Key.space:
                    if self.meters_per_pixel is None or len(self.calibrations) < 2:
                        print("Cannot start ball tracking until main calibrations are complete.")
                    elif self.seam_angle_active or self.calibration_active:
                        print("Cannot start ball tracking while seam angle or calibration mode is active.")
                    else:
                        self.tracking_active = not self.tracking_active
                        print(f"{'Ball tracking started' if self.tracking_active else 'Ball tracking paused'}")
                case Key.t:
                    if self.meters_per_pixel is None or len(self.calibrations) < 2:
                        print("Cannot start seam angle tracking until main calibrations are complete.")
                    elif self.tracking_active or self.calibration_active:
                        print("Cannot start seam angle tracking while ball tracking or calibration mode is active.")
                    else:
                        self.seam_angle_active = not self.seam_angle_active
                        if self.seam_angle_active:
                            self.seam_points = []
                            print("Seam angle tracking started.")
                        else:
                            print("Seam angle tracking stopped.")
                case Key.a:
                    self.top_down_video.change_frame(-1)
                case Key.d:
                    self.top_down_video.change_frame(1)
                case Key.A:
                    self.top_down_video.change_frame(-10)
                case Key.D:
                    self.top_down_video.change_frame(10)
                case Key.s:
                    if self.meters_per_pixel is None or len(self.calibrations) < 2:
                        print("Cannot proceed until main calibrations are complete.")
                    elif not self.frame_positions:
                        print("Cannot proceed without tracking at least one point.")
                    else:
                        self.top_down_video.cap.release()
                        cv2.destroyAllWindows()
                        return self.frame_positions, self.seam_points, self.seam_measurements, self.calibrations, self.meters_per_pixel, self.deceleration, self.first_frame_main
                case Key.o:
                    self.top_down_video.rotate()
                    print(f"Main view rotated to {self.top_down_video.rotation} degrees")

class SideOnTracker:
    def __init__(self):
        self.side_on_video = None
        self.side_positions = []
        self.side_calibration = None
        self.side_focal_length_px = None
        self.tracking_active = False
        self.side_calibration_active = False
        self.side_frame_for_main_frame1 = None
        self.side_zoom_active = False
        self.side_zoom_centre = None
        self.side_mouse_pos = None
        self._side_zoom_transform = None
        self.window_name_side = "Cricket Ball Tracker - Side View"

        self.display = display.Display()
        self.drawers = drawers.Drawers()

    def _add_or_replace_point_for_frame(self, frame_no, x, y, t):
        positions = self.side_positions
        for i, (f, *_rest) in enumerate(positions):
            if f == frame_no:
                positions[i] = (frame_no, int(x), int(y), t)
                return
        positions.append((frame_no, int(x), int(y), t))
        positions.sort(key=lambda z: z[0])

    def _toggle_side_zoom(self):
        if self.side_zoom_active:
            self.side_zoom_active = False
            self.side_zoom_centre = None
            self._side_zoom_transform = None
            print("Side view zoom off.")
            return

        if self.side_mouse_pos is None:
            print("Move the mouse over the side view before pressing 'Z'.")
            return

        self.side_zoom_active = True
        self.side_zoom_centre = self.side_mouse_pos
        print(f"Side view zoom on: {ZOOM_FACTOR}x around ({self.side_zoom_centre[0]}, {self.side_zoom_centre[1]}). Press 'Z' again to zoom out.")

    def _side_overlays_for_display(self):
        """Return (positions, calibration) with coordinates mapped into display space."""
        if not self.side_zoom_active or self._side_zoom_transform is None:
            return self.side_positions, self.side_calibration

        positions = []
        for frame_num, x, z, t in self.side_positions:
            vx, vz = self._frame_to_side_view_coords(x, z)
            positions.append((frame_num, vx, vz, t))

        calibration = None
        if self.side_calibration:
            calibration = (
                self.side_calibration[0],
                [self._frame_to_side_view_coords(x, z) for x, z in self.side_calibration[1]],
            )
        return positions, calibration

    def _side_view_to_frame_coords(self, x, y):
        """
        Map a coordinate from the displayed side-on view back to raw frame pixels.

        When zoom is off this is the identity. When zoom is on it inverts the
        crop-and-resize exactly, so a click always resolves to the same raw pixel
        it would have resolved to unzoomed.
        """
        if not self.side_zoom_active or self._side_zoom_transform is None:
            return int(x), int(y)

        x0, y0, scale_x, scale_y, crop_w, crop_h = self._side_zoom_transform
        # cv2.resize with INTER_NEAREST maps dst pixel j -> src pixel floor(j / scale),
        # so flooring here is the exact inverse of what is on screen.
        col = int(max(0.0, x) / scale_x)
        row = int(max(0.0, y) / scale_y)
        col = min(col, crop_w - 1)
        row = min(row, crop_h - 1)
        return x0 + col, y0 + row

    def _frame_to_side_view_coords(self, x, y):
        """Map raw frame pixels to the displayed side-on view (for drawing overlays)."""
        if not self.side_zoom_active or self._side_zoom_transform is None:
            return int(x), int(y)

        x0, y0, scale_x, scale_y, _crop_w, _crop_h = self._side_zoom_transform
        # +0.5 puts the marker in the centre of the magnified pixel block.
        return int(round((x - x0 + 0.5) * scale_x)), int(round((y - y0 + 0.5) * scale_y))

    def _apply_side_zoom(self, frame):
        """Crop a ZOOM_FACTOR-smaller region around the zoom centre and blow it back up to full size."""
        height, width = frame.shape[:2]
        crop_w = max(1, int(round(width / ZOOM_FACTOR)))
        crop_h = max(1, int(round(height / ZOOM_FACTOR)))

        centre_x, centre_y = self.side_zoom_centre
        x0 = min(max(int(centre_x) - crop_w // 2, 0), max(width - crop_w, 0))
        y0 = min(max(int(centre_y) - crop_h // 2, 0), max(height - crop_h, 0))

        crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        zoomed = cv2.resize(crop, (width, height), interpolation=ZOOM_INTERPOLATION)

        # Store the actual scales rather than ZOOM_FACTOR itself: width / crop_w is
        # only exactly ZOOM_FACTOR when the frame size divides evenly.
        self._side_zoom_transform = (x0, y0, width / crop_w, height / crop_h, crop_w, crop_h)
        return zoomed

    def _mouse_callback(self, event, x, y, flags, param):
        # Everything below works in raw frame coordinates, regardless of zoom state.
        x, y = self._side_view_to_frame_coords(x, y)

        if event == cv2.EVENT_MOUSEMOVE:
            self.side_mouse_pos = (x, y)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.side_mouse_pos = (x, y)
            print(f"Clicked at point ({x}, {y}).")
            if self.side_calibration_active:
                if self.side_frame_for_main_frame1 is None:
                    print(f"Error: Track the first ball position before calibrating.")
                    return
                if self.side_on_video.get_current_frame_number() != self.side_frame_for_main_frame1:
                    print(f"Error: Side calibration must occur in first tracked frame ({self.side_frame_for_main_frame1}).")
                    return
                if self.side_calibration is None or len(self.side_calibration[1]) == 2:
                    self.side_calibration = (self.side_on_video.get_current_frame_number(), [])
                self.side_calibration[1].append((x, y))
                print(f"Side diameter point added in frame {self.side_on_video.get_current_frame_number()}: ({x}, {y})")
                if len(self.side_calibration[1]) == 2:
                    self.side_calibration_active = False
                    print(f"Side calibration completed for frame {self.side_on_video.get_current_frame_number()}.")
            elif self.tracking_active:
                timestamp = self.side_on_video.get_current_frame_number() / self.side_on_video.fps
                if not self.side_positions and self.side_frame_for_main_frame1 is None:
                    self.side_frame_for_main_frame1 = self.side_on_video.get_current_frame_number()
                    print(f"Side view Frame 1 set to raw frame {self.side_frame_for_main_frame1}, corresponding to main view Frame 1")
                self._add_or_replace_point_for_frame(self.side_on_video.get_current_frame_number(), x, y, timestamp, is_side=True)
                print(f"Side View - Frame {self.side_on_video.get_current_frame_number()}: Ball at (x={x}, z={y}) - Time: {timestamp:.3f}s")

    def run(self):
        self.side_on_video = self.display.load_side_video()

        cv2.namedWindow(self.window_name_side, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name_side, self._mouse_callback)

        print("\n=== CRICKET BALL TRACKER - SIDE VIEW ===")
        print("First click to track the ball will set this as Frame 1, corresponding to main view Frame 1.")
        print("Calibrate ball diameter in the first tracked frame by pressing 'C' (can be done while tracking).")
        print("Controls:")
        print("- C: Start/Stop Calibration Mode (click two points across ball diameter in first tracked frame)")
        print("- SPACE: Start/Stop Ball Tracking Mode")
        print("- a/d or A/D: Move frame back/forward (capitals for 10 frame jumps)")
        print("- Click: Mark calibration points or ball position (X, Z)")
        print(f"- Z: Toggle {ZOOM_FACTOR}x zoom centred on the mouse (press again to zoom out)")
        print("- S: Save data to Excel")
        print("- R: Reset side view tracking and calibration")
        print("- Q or ESC: Quit")
        print("- O: Rotate video 90 degrees clockwise")

        while True:
            frame = self.side_on_video.get_current_frame()

            # Zoom is applied to the raw frame first, then overlays and HUD text are
            # drawn on top at normal size so they stay readable at any ZOOM_FACTOR.
            if self.side_zoom_active and self.side_zoom_centre is not None and frame is not None:
                frame = self._apply_side_zoom(frame)
            display_positions, display_calibration = self._side_overlays_for_display()

            self.drawers.draw_side_trajectory(frame, display_positions, self.side_on_video.current_frame, display_calibration)

            if self.side_calibration_active:
                points = len(self.side_calibration[1]) if self.side_calibration else 0
                status = f"Side Calibration - Click point {points + 1}/2 for ball diameter in Frame {self.side_frame_for_main_frame1}"
            elif self.tracking_active:
                status = "BALL TRACKING ACTIVE - Click on ball, press C to calibrate in first frame"
            else:
                status = "PAUSED - Press SPACE to track, C to calibrate after first point"

            self.drawers.draw_text(frame, status, (10, 40), font_scale=1.1)
            self.drawers.draw_text(frame, f"Frame: {self.side_on_video.current_frame}/{self.side_on_video.total_frames-1}", (10, 80), font_scale=0.9)
            self.drawers.draw_text(frame, f"Side Points: {len(self.side_positions)}", (10, 110), font_scale=0.9)
            self.drawers.draw_text(frame, f"Main Frame 1 = Side Frame {'Not set' if self.side_frame_for_main_frame1 is None else self.side_frame_for_main_frame1}", (10, 140), font_scale=0.9)
            self.drawers.draw_text(frame, f"Focal Length: {'{:.2f} px'.format(self.side_focal_length_px) if self.side_focal_length_px is not None else 'Not set'}", (10, 170), font_scale=0.9)
            zoom_status = f"{ZOOM_FACTOR}x @ ({self.side_zoom_centre[0]}, {self.side_zoom_centre[1]})" if self.side_zoom_active and self.side_zoom_centre is not None else "Off"
            self.drawers.draw_text(frame, f"Zoom: {zoom_status}  (Z to toggle at mouse)", (10, 200), font_scale=0.9)
            self.drawers.draw_text(frame, "S = Save Excel", (10, 230), font_scale=0.9)

            cv2.imshow(self.window_name_side, frame)
            key_code = int(cv2.waitKey(10) & 0xFF)
            try:
                key = Key(key_code)
            except ValueError:
                key = None

            match key:
                case Key.q | Key.esc:
                    self.side_on_video.cap.release()
                    cv2.destroyAllWindows()
                    return
                case Key.c:
                    if self.side_frame_for_main_frame1 is None:
                        print("Error: Track the first ball position before calibrating.")
                    elif self.side_on_video.current_frame != self.side_frame_for_main_frame1:
                        print(f"Error: Side calibration must occur in first tracked frame ({self.side_frame_for_main_frame1}).")
                    elif self.side_calibration and len(self.side_calibration[1]) == 2:
                        print("Side calibration already completed.")
                    else:
                        self.side_calibration_active = not self.side_calibration_active
                        if self.side_calibration_active:
                            if self.side_calibration is None or len(self.side_calibration[1]) == 2:
                                print(f"Starting side calibration in frame {self.side_frame_for_main_frame1}.")
                            else:
                                print("Continuing side calibration.")
                        else:
                            print("Side calibration paused.")
                case Key.space:
                    self.tracking_active = not self.tracking_active
                    print(f"{'Ball tracking started' if self.tracking_active else 'Ball tracking paused'}")
                case Key.a:
                    self.side_on_video.change_frame(-1)
                case Key.d:
                    self.side_on_video.change_frame(1)
                case Key.A:
                    self.side_on_video.change_frame(-10)
                case Key.D:
                    self.side_on_video.change_frame(10)
                case Key.z:
                    self._toggle_side_zoom()
                case Key.s:
                    return []
                case Key.o:
                    self.side_on_video.rotate()
                    # Frame dimensions change on rotation, so the old zoom centre is meaningless.
                    if self.side_zoom_active:
                        self.side_zoom_active = False
                        self.side_zoom_centre = None
                        self._side_zoom_transform = None
                        print("Side view zoom off (video rotated).")
                    print(f"Side view rotated to {self.side_on_video.rotation} degrees")
                case Key.r:
                    self.side_positions = []
                    self.side_calibration = None
                    self.side_focal_length_px = None
                    self.side_frame_for_main_frame1 = None
                    print("Side view tracking, calibration, and frame mapping reset.")

tracker = TopDownTracker()
x = tracker.run()

for i in x:
    print(i)