import os
import math
import pandas as pd
from datetime import datetime

import calculators
import checkers

class Exporter:
    def __init__(self):
        self.calculators = calculators.Calculators()
        self.checkers = checkers.Checker()
    
    def excel(self):
        if not self.frame_positions:
            print("No tracking data to save.")
            return

        df, initial_speed_kmh = self._build_dataframe()
        if df is None:
            print("No data to save.")
            return

        df_3d_data = self._build_3d_data()
        initial_trajectory = self.calculators._calculate_initial_trajectory(self.frame_positions, self.frame_height, self.meters_per_pixel)
        seam_angle, is_wobble, max_seam_diff = self.checkers._check_seam_wobble(self.seam_measurements)
        percent_diff, is_invalid_cal = self.checkers._check_calibration_difference(self.calibrations, self.BALL_DIAMETER_M)

        # Seam Angles DataFrame
        seam_data = [[frame_num, f"{angle:.2f}"] for frame_num, angle in self.seam_measurements]
        seam_df = pd.DataFrame(seam_data, columns=["Frame Number", "Seam Angle (degrees)"])
        seam_summary = ["Average" if not is_wobble else "Wobble Seam",
                        f"{seam_angle:.2f}" if not is_wobble and seam_angle is not None else "Wobble Seam"]
        seam_diff_row = ["Max Consecutive Difference", f"{max_seam_diff:.2f} (Threshold: 10.00 degrees)"]
        seam_df = pd.concat([seam_df,
                             pd.DataFrame([seam_summary], columns=["Frame Number", "Seam Angle (degrees)"]),
                             pd.DataFrame([seam_diff_row], columns=["Frame Number", "Seam Angle (degrees)"])],
                            ignore_index=True)

        # Main Calibrations DataFrame
        cal_data = []
        for frame_num, points in self.calibrations:
            (x1, y1), (x2, y2) = points
            pixel_distance = math.hypot(x2 - x1, y2 - y1)
            mpp = self.BALL_DIAMETER_M / pixel_distance if pixel_distance >= 1.0 else None
            if mpp is not None:
                cal_data.append([frame_num, f"{mpp:.6f}"])
        cal_df = pd.DataFrame(cal_data, columns=["Frame Number", "Meters per Pixel"])
        if self.meters_per_pixel is not None:
            cal_summary = ["Average", f"{self.meters_per_pixel:.6f}"]
            cal_diff_row = ["Percentage Difference",
                            f"{percent_diff:.2f}% (Threshold: 25.00%)" if percent_diff is not None else "N/A"]
            cal_df = pd.concat([cal_df,
                                pd.DataFrame([cal_summary], columns=["Frame Number", "Meters per Pixel"]),
                                pd.DataFrame([cal_diff_row], columns=["Frame Number", "Meters per Pixel"])],
                               ignore_index=True)

        # Side Calibration DataFrame
        side_cal_data = []
        if df_3d_data is not None and self.side_calibration:
            frame_num, points = self.side_calibration
            (x1, z1), (x2, z2) = points
            pixel_distance = math.hypot(x2 - x1, z2 - z1)
            if pixel_distance >= 1.0:
                # find corresponding Y in df_3d_data (same indexing as frames in that sheet)
                idx = None
                if "Frame Number" in df_3d_data.columns:
                    # use first occurrence
                    idx = df_3d_data.index[df_3d_data["Frame Number"] == (frame_num - self.side_frame_for_main_frame1 + 1)].tolist()
                    if idx:
                        y_m = df_3d_data.loc[idx[0], "Y-Position (m)"]
                        f = pixel_distance * y_m / self.BALL_DIAMETER_M if y_m > 0 else None
                        if f is not None:
                            side_cal_data.append([frame_num, f"{pixel_distance:.2f}", f"{y_m:.3f}", f"{f:.2f}"])
        side_cal_df = pd.DataFrame(side_cal_data, columns=["Frame Number", "Pixel Diameter", "Y-Position (m)", "Focal Length (px)"])

        while True:
            folder_path = input("Enter the folder path to save the Excel file (leave blank for current directory): ").strip().strip('"')
            if not folder_path:
                folder_path = os.getcwd()
                break
            if os.path.isdir(folder_path):
                break
            print("Invalid folder path. Please enter a valid directory or leave blank for current directory.")

        filename = input("Enter Excel filename (without extension): ").strip()
        if not filename:
            filename = datetime.now().strftime("cricket_tracking_%Y%m%d_%H%M%S")
        filename = f"{filename}.xlsx"
        full_path = os.path.join(folder_path, filename)

        try:
            with pd.ExcelWriter(full_path, engine='xlsxwriter') as writer:
                # Main Data sheet
                df.to_excel(writer, sheet_name='Data', index=False)

                # Parameters (with Final Swing)
                final_swing_val = "N/A"
                if df_3d_data is not None and not df_3d_data.empty:
                    if "Actual Data" in df_3d_data.columns and "Initial Path (m)" in df_3d_data.columns:
                        valid = df_3d_data["Actual Data"].dropna()
                        if len(valid) >= 1:
                            last5 = valid.tail(5).mean()
                            # initial path at final frame present in the same last row
                            last_row_idx = df_3d_data.index[-1]
                            init_at_end = df_3d_data.loc[last_row_idx, "Initial Path (m)"]
                            if pd.notna(init_at_end):
                                final_swing_val = f"{(last5 - init_at_end):.6f}"

                param_df = pd.DataFrame({
                    'Parameter': [
                        'Seam Angle (degrees)',
                        'Initial Trajectory (degrees)',
                        'Initial Speed (km/h)',
                        'Main Meters per Pixel',
                        'Side Focal Length (px)',
                        'Side Frame for Main Frame 1',
                        'Final Swing (m)'
                    ],
                    'Value': [
                        "Wobble Seam" if is_wobble else f"{seam_angle:.2f}" if seam_angle is not None else 'N/A',
                        f"{initial_trajectory:.2f}" if initial_trajectory is not None else 'N/A',
                        f"{initial_speed_kmh:.2f}" if initial_speed_kmh is not None else 'N/A',
                        "Invalid Calibration" if is_invalid_cal else f"{self.meters_per_pixel:.6f}" if self.meters_per_pixel is not None else 'N/A',
                        f"{self.side_focal_length_px:.2f}" if self.side_focal_length_px is not None else 'N/A',
                        f"{self.side_frame_for_main_frame1}" if self.side_frame_for_main_frame1 is not None else 'Not set',
                        final_swing_val
                    ]
                })
                param_df.to_excel(writer, sheet_name='Parameters', index=False)

                # 3D Data (includes Side X px, x(m), Actual Data, Side Z px, plus Initial Path & Swing)
                if df_3d_data is not None and not df_3d_data.empty:
                    print(f"Saving 3D Data sheet with {len(df_3d_data)} rows")
                    df_3d_data.to_excel(writer, sheet_name='3D Data', index=False)

                if not seam_df.empty:
                    print(f"Saving Seam Angles sheet with {len(seam_df)} rows")
                    seam_df.to_excel(writer, sheet_name='Seam Angles', index=False)

                if not cal_df.empty:
                    print(f"Saving Main Calibrations sheet with {len(cal_df)} rows")
                    cal_df.to_excel(writer, sheet_name='Main Calibrations', index=False)

                if not side_cal_df.empty:
                    print(f"Saving Side Calibration sheet with {len(side_cal_df)} rows")
                    side_cal_df.to_excel(writer, sheet_name='Side Calibration', index=False)

            print(f"Tracking data saved to {full_path}")
        except Exception as e:
            print(f"Error saving file: {e}")