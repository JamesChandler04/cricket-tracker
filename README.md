# Automatic Cricket Ball Tracker

## Installation And Usage

To clone the repo, run the following command.

```sh
git clone https://github.com/JamesChandler04/cricket-tracker.git
```

To run the code, refer to the instructions on how to use make [here](./MAKE_GUIDE.md) to set up everything automatically. Otherwise, manually install all packages in the [requirements](./requirements.txt) file by running this command.

```sh
pip install -r requirements.txt
```

Run everything from the project folder. You can run the text-based version of the program with this command:

```sh
python -m main.new_cricket_ball_tracker
```

And the graphics-based version with this command:

```sh
python -m main.GUI_new_cricket_ball_tracker
```

`make run-text` and `make run-gui` do the same. Running a file in `main/` directly, for example with the Run button in VS Code, also works.

## Project Layout

```
main/                   the programs you run
library/                helper modules used by the programs
    physics_engines/    top-down and side-on physics engines
    automated/          automatic ball detection and seam angle
output/                 everything the programs write
test_fixtures/          tests
report_data/            radar speeds and data for the report
vids/                   delivery videos
```

Folder locations are set in one place, `library/paths.py`. The tools in `library/` run the same way, for example:

```sh
python -m library.camera_calibration
python -m library.detect_seam_angle_file
python -m pytest test_fixtures
```
