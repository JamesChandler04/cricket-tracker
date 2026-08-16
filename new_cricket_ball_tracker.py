from manual_tracker import TopDownTracker
from top_down_physics_engine import SelectionType, TopDownPhysicsEngine
from side_on_physics_engine import SideOnPhysicsEngine

# Click points on top-down video
tracker = TopDownTracker()
top_down_clicked_points = tracker.get_top_down_points()


top_down_engine = TopDownPhysicsEngine()
fps = tracker.top_down_video.fps

# Calculate velocity using the clicked points and fps
velocity = top_down_engine.calculate_velocity(top_down_clicked_points, fps=fps, type=SelectionType.MEAN)
print(f"Calculated velocity: {velocity:.2f} km/h")

# Calculate seam angle
seam_angle = top_down_engine.calculate_seam_angle(top_down_clicked_points)
print(f"Calculated seam angle: {seam_angle}")
