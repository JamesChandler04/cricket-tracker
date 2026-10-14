# Automatic Cricket Ball Tracker

Measures a cricket delivery's speed, seam angle and swing from two videos: a top-down camera looking down on the pitch and a side-on camera behind the bowler. It was developed on Windows with Python 3.13.

## Installation

Clone the repo and go into its folder:

```sh
git clone https://github.com/JamesChandler04/cricket-tracker.git
cd cricket-tracker
```

On Windows with make installed, `make setup` and then `make install` create a virtual environment in `.venv` and install the packages. The [make guide](./MAKE_GUIDE.md) shows how to install make and lists the other commands.

Without make, create the virtual environment and install the packages in [requirements.txt](./requirements.txt) yourself:

```sh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate it with `source .venv/bin/activate` instead.

## Running the Tracker

Run everything from the project folder, with the virtual environment active:

```sh
python -m main.GUI_new_cricket_ball_tracker
```

`make run-gui` does the same. Running the file directly, for example with the Run button in VS Code, also works.

In the window:

1. Choose the top-down and side-on videos, and the frame to start each one from.
2. Choose the side-on camera's calibration file, or make one with **Calibrate camera...** by clicking the 16 ring corners of the calibration cubes on one side-on frame.
3. **1 Track top-down**: click the ball through the top-down video to get the delivery speed. The seam angle is then measured on the ball images, by clicking the seam yourself, or automatically if **Automatic seam angle detection** is ticked.
4. **2 Track side-on**: click the ball through the side-on video. Its path is reconstructed at the delivery speed to give the swing. To run this step on its own, type the speed into **Delivery speed** first.

**Run both stages** runs steps 3 and 4 in one go, asking in between for the speed to reconstruct with. The results are saved to the **Save to** folder (`output/demo_delivery` by default) and shown in the Swing, Trajectory, Results and Coordinates tabs.

The delivery videos and calibration files are not in the repository, and neither are the scripts and data behind the results in the thesis.

## Other Tools

These also run from the project folder:

- `python -m library.camera_calibration` makes a side-on calibration file from a video.
- `python -m library.detect_seam_angle_file` picks the seam again on the ball images of an earlier run.
- `python frame_save.py` saves chosen frames of a video as images.
- `python generate_seam_training_data.py` makes seam angle training images from labelled ball images.
- `python evaluate_seam_detector.py` tests how accurately the automatic seam angle detector finds known seam angles.

## Documentation

The code reference is built from the docstrings with MkDocs. Run `make docs-install` once, then `make docs` to build it into `site/` and open `site/index.html`, or `make docs-serve` to view it at http://127.0.0.1:8000 while you work. Without make, run `pip install -r requirements-docs.txt` and then `python -m mkdocs build`.

## Project Layout

```
main/                   the programs; GUI_new_cricket_ball_tracker.py is the tracker
library/                modules the programs use
    physics_engines/    top-down and side-on physics engines
    automated/          automatic seam angle detection, and the earlier automatic ball detection
docs/                   documentation pages for MkDocs
output/                 where results are saved, made when first needed
```

Folder and file locations are set in `library/paths.py`. The other programs in `main/` (`new_cricket_ball_tracker.py`, `cricket_ball_tracker.py` and `automated_cricket_ball_tracker.py`) are earlier versions, kept for reference only.
