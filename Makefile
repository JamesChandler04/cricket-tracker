.PHONY: venv install docs-install docs docs-serve

VENV=.venv\Scripts

setup:
	python -m venv .venv

install:
	$(VENV)\activate && pip install -r requirements.txt

clean:
	del /Q .venv

# Run code

run-text:
	$(VENV)\python -m main.new_cricket_ball_tracker

run-gui:
	$(VENV)\python -m main.GUI_new_cricket_ball_tracker

# Generate and view documentation

docs-install:
	$(VENV)\python -m pip install -r requirements-docs.txt

docs:
	$(VENV)\python -m mkdocs build

docs-serve:
	$(VENV)\python -m mkdocs serve

# Generate manual vs automatic seam angle comparison

seam-images:
	$(VENV)\python -m report_data.seam_detector_comparison.make_seam_images

seam-compare:
	$(VENV)\python -m report_data.seam_detector_comparison.click_seams
	$(VENV)\python -m report_data.seam_detector_comparison.compare_seam_angles
	
# Generate automatic ball detection data

ball-detection:
	$(VENV)\python -m report_data.ball_detection.detect_ball_frames

ball-compare:
	$(VENV)\python -m report_data.ball_detection.confusion_matrix

# Generate radar vs manual speed comparison data

speed-compare:
	$(VENV)\python -m report_data.speed_comparison.compare_speeds --balls 1

# Generate swing comparison data

swing-record:
	$(VENV)\python -m report_data.swing_seam_comparison.record_delivery

swing-compare:
	$(VENV)\python -m report_data.swing_seam_comparison.calculate_swing --plots

swing-plot:
	$(VENV)\python -m report_data.swing_seam_comparison.plot_swing_seam