import math
import pandas as pd
import numpy as np

class Builder:
    def _build_3d_data(self, frame_positions, meters_per_pixel, initial_velocity, deceleration):
        FX = 2877.72
        FZ = 2877.72

        if not frame_positions:
            print("Cannot build 3D Data: No tracking data in main view.")
            return None
        if meters_per_pixel is None:
            print("Cannot build 3D Data: Main calibration not completed.")
            return None
        if len(frame_positions) < 2:
            print("Cannot build 3D Data: Need at least two points to calculate velocity.")
            return None

        t0 = frame_positions[0][3]
        records = []
        y_positions = []
        cum_dist = 0.0

        # Calculate initial velocity and deceleration
        if initial_velocity is None or deceleration is None:
            initial_speeds = []
            decelerations = []
            for i in range(1, len(frame_positions)):
                frame_num, x_px, y_px, t = frame_positions[i]
                prev_frame_num, px_prev, py_prev, t_prev = frame_positions[i-1]
                dt = max(t - t_prev, 1.0 / self.fps)
                dist_px = abs(y_px - py_prev)
                dist_m = dist_px * meters_per_pixel
                speed_ms = dist_m / dt if dt > 0 else 0.0
                initial_speeds.append(speed_ms)
                if i > 1 and initial_speeds[-2] > speed_ms:
                    decel = (initial_speeds[-2] - speed_ms) / dt
                    decelerations.append(decel)
            initial_velocity = sum(initial_speeds) / len(initial_speeds) if initial_speeds else 0.0
            deceleration = sum(decelerations) / len(decelerations) if decelerations else 0.0

        if self.initial_velocity == 0.0:
            print("Cannot build 3D Data: Initial velocity is zero.")
            return None

        # Initial trajectory angle (used for projected X only)
        initial_trajectory = self.calculators._calculate_initial_trajectory(self.frame_positions, self.frame_height, self.meters_per_pixel)
        if initial_trajectory is None:
            print("Cannot build 3D Data: Initial trajectory not available.")
            return None
        theta = math.radians(initial_trajectory)
        vx0 = self.initial_velocity * math.sin(theta)  # X-velocity component
        ax = self.deceleration or 0.0  # Assume same deceleration as Y

        # Process tracked frames
        prev = None
        for i, (frame_num, x_px, y_px, t) in enumerate(self.frame_positions):
            adj_frame = i + 1
            adj_time = t - t0
            y_m = y_px * self.meters_per_pixel
            y_positions.append(y_m)
            if prev is not None:
                dist_px = abs(y_px - prev[2])
                dist_m = dist_px * self.meters_per_pixel
                cum_dist += dist_m
            # Projected X-position (kept)
            x_m_projected = vx0 * adj_time - 0.5 * ax * adj_time**2
            # Build: Frame, Time, Projected X, Y, Z (Z filled later), then placeholders
            records.append([adj_frame, adj_time, x_m_projected, y_m, None])  # Z to be filled later
            prev = (frame_num, x_px, y_px, t)

        # Extrapolate to 17m (for Y-Position path length along flight)
        if records:
            last_row = records[-1]
            last_frame = last_row[0]
            last_time = last_row[1]
            z0 = cum_dist
            v0 = self.initial_velocity
            a = self.deceleration or 0.0
            t_step = 1.0 / self.fps
            t = last_time + t_step
            frame_counter = last_frame + 1

            print(f"3D Data: Starting extrapolation with z0={z0:.3f}m, v0={v0:.3f}m/s, a={a:.3f}m/s^2, vx0={vx0:.3f}m/s, theta={initial_trajectory:.2f}°")

            while cum_dist < 17.0:
                delta_t = t - last_time
                v = max(0, v0 - a * delta_t)
                if v <= 0:
                    print(f"Warning: Velocity reached zero at t={t:.3f}s, cum_dist={cum_dist:.3f}m")
                    break
                cum_dist = z0 + v0 * delta_t - 0.5 * a * delta_t**2
                x_m_projected = vx0 * (t - t0) - 0.5 * ax * (t - t0)**2
                if cum_dist >= 17.0:
                    A = 0.5 * a
                    B = -v0
                    C = 17.0 - z0
                    discriminant = B**2 - 4*A*C
                    if discriminant < 0:
                        print(f"Warning: Cannot reach 17 meters with discriminant={discriminant:.3f}")
                        break
                    delta_t = (-B - math.sqrt(discriminant)) / (2*A) if a != 0 else (17.0 - z0) / v0
                    if delta_t <= 0:
                        print(f"Warning: Invalid delta_t={delta_t:.3f}s for final point")
                        break
                    t = last_time + delta_t
                    cum_dist = 17.0
                    x_m_projected = vx0 * (t - t0) - 0.5 * ax * (t - t0)**2
                y_positions.append(cum_dist)
                records.append([frame_counter, t, x_m_projected, cum_dist, None])
                t += t_step
                frame_counter += 1
                if self.side_frame_for_main_frame1 is not None and frame_counter - 1 + self.side_frame_for_main_frame1 >= self.total_frames_side:
                    print(f"Reached end of side video at frame {frame_counter - 1 + self.side_frame_for_main_frame1}")
                    break
                if cum_dist >= 17.0:
                    break

        # Calculate side view focal length (still used in Parameters)
        if self.side_calibration:
            self.side_calibration, tmp_side_focal_length_px = self.calculators._calculate_side_focal_length(y_positions, self.side_calibration, self.side_frame_for_main_frame1, self.BALL_DIAMETER_M, self.side_focal_length_px)
            if tmp_side_focal_length_px is not None:
                self.side_focal_length_px = tmp_side_focal_length_px

        # Side/back clicks mapping (we only use Z from old mapping for this sheet)
        x0_side, z0_side = None, None
        if self.side_positions:
            x0_side, z0_side = self.side_positions[0][1], self.side_positions[0][2]

        side_frame_map = {}  # frame -> (x_m_old, z_m_old) [we keep z only]
        side_px_map   = {}  # frame -> (x_px, z_px)

        if self.side_positions and self.side_frame_for_main_frame1 is not None:
            for frame_num, x, z, t in self.side_positions:
                frame_idx = frame_num - self.side_frame_for_main_frame1 + 1
                if frame_idx < 1 or frame_idx > len(y_positions):
                    continue
                y_m = y_positions[frame_idx - 1]
                if self.side_focal_length_px and y_m > 0:
                    mpp = y_m / self.side_focal_length_px
                    x_m_old = ((x - x0_side) * mpp) / FX * 175 if x0_side is not None else (x * mpp) / FX * 175
                    z_m_old = ((z - z0_side) * mpp) / FZ * 175 if z0_side is not None else (z * mpp) / FZ * 175
                else:
                    x_m_old = (x - x0_side) * 175 if x0_side is not None else x * 175
                    z_m_old = (z - z0_side) * 175 if z0_side is not None else z * 175
                side_frame_map[frame_num] = (x_m_old, z_m_old)
                side_px_map[frame_num] = (x, z)

        # ---- Back-adjusted X (meters) & Actual Data arrays (side/back camera) ----
        # Use first-frame side calibration diameter as metres-per-pixel
        m_per_px_first = None
        if self.side_calibration and len(self.side_calibration[1]) == 2:
            (sx1, sz1), (sx2, sz2) = self.side_calibration[1]
            px_diam = math.hypot(sx2 - sx1, sz2 - sz1)
            if px_diam >= 1.0:
                m_per_px_first = self.BALL_DIAMETER_M / px_diam

        # Side X/Z pixels by adjusted frame (1-based to match records)
        side_px_by_adjframe = {}
        if self.side_positions and self.side_frame_for_main_frame1 is not None:
            for frame_num, x, z, _ in self.side_positions:
                adj_idx = frame_num - self.side_frame_for_main_frame1 + 1
                if adj_idx >= 1:
                    side_px_by_adjframe[adj_idx] = (x, z)

        x_m_adj = []
        actual_adj = []
        side_x_px_col = []
        side_z_px_col = []

        first_x_px = side_px_by_adjframe.get(1, (None, None))[0]
        prev_xm = None
        prev_actual = 0.0

        for i, rec in enumerate(records):
            adj_frame = rec[0]
            sx_px, sz_px = side_px_by_adjframe.get(adj_frame, (None, None))
            side_x_px_col.append(sx_px)
            side_z_px_col.append(sz_px)

            if m_per_px_first is not None and sx_px is not None and first_x_px is not None:
                xm = (sx_px - first_x_px) * m_per_px_first
            else:
                xm = None

            if xm is not None and prev_xm is not None:
                prev_actual = prev_actual + (xm - prev_xm) * 0.1 * adj_frame
            elif prev_xm is None and xm is not None:
                prev_actual = 0.0

            x_m_adj.append(xm)
            actual_adj.append(prev_actual if xm is not None else None)
            prev_xm = xm if xm is not None else prev_xm

        # ---- Assemble final 3D Data rows (without Initial Path / Swing yet) ----
        final_records = []
        for i, rec in enumerate(records):
            frame_no = rec[0]
            time_s = rec[1]
            proj_x = rec[2]
            y_pos = rec[3]
            # z position from side_frame_map (if available)
            side_frame = self.side_frame_for_main_frame1 + (frame_no - 1) if self.side_frame_for_main_frame1 is not None else None
            _, z_m_old = side_frame_map.get(side_frame, (None, None))

            final_records.append([
                frame_no,
                time_s,
                proj_x,
                y_pos,
                z_m_old,
                side_x_px_col[i],
                x_m_adj[i],
                actual_adj[i],
                side_z_px_col[i]
            ])

            print(f"3D Data: Frame {frame_no}, Time {time_s:.3f}s, Y={y_pos:.3f}m, "
                  f"Z={z_m_old}, SideXpx={side_x_px_col[i]}, x(m)={x_m_adj[i]}, Actual={actual_adj[i]}")

        columns = [
            "Frame Number",
            "Time (s)",
            "Projected X-Position (m)",
            "Y-Position (m)",
            "Z-Position (m)",
            "Side X (px)",
            "x (m)",
            "Actual Data",
            "Side Z (px)"
        ]
        df = pd.DataFrame(final_records, columns=columns)

        # --- Initial Path & Swing (computed AFTER DataFrame is built) ---
        actual = df["Actual Data"]
        valid_mask = actual.notna()
        if valid_mask.any():
            valid_df = df[valid_mask]
            first_frame = int(valid_df["Frame Number"].iloc[0])
            target_frame = int(valid_df["Frame Number"].iloc[19]) if len(valid_df) >= 20 else int(valid_df["Frame Number"].iloc[-1])

            a1 = float(df.loc[df["Frame Number"] == first_frame, "Actual Data"].iloc[0])
            a20 = float(df.loc[df["Frame Number"] == target_frame, "Actual Data"].iloc[0])
            denom = (target_frame - first_frame)
            m = (a20 - a1) / denom if denom != 0 else 0.0
            b = a1 - m * first_frame

            df["Initial Path (m)"] = m * df["Frame Number"] + b
            df["Swing (m)"] = df["Actual Data"] - df["Initial Path (m)"]
        else:
            df["Initial Path (m)"] = np.nan
            df["Swing (m)"] = np.nan

        print(f"3D Data: Generated {len(df)} rows")
        return df