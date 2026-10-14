# Make Guide

The Makefile runs the setup, run and documentation commands for you. Its commands are written for Windows, so on macOS or Linux use the commands in the [README](./README.md) instead.

## Installing Make on Windows

Run the following command, then open a new terminal so that `make` is found:

```
winget install ezwinports.make
```

## Using the Makefile

Run these from the project folder:

- `make setup`: Create the virtual environment in `.venv`, with the `python` on your PATH. It needs Python 3.10 or newer, so check `python --version` first. If it is older, run `py -3.13 -m venv .venv` instead.
- `make install`: Install the packages in [requirements.txt](./requirements.txt) into the virtual environment.
- `make run-gui`: Run the tracker.
- `make run-text`: Run the old text-based program, kept for reference only.
- `make docs-install`: Install the packages for the documentation, listed in [requirements-docs.txt](./requirements-docs.txt).
- `make docs`: Build the documentation into `site/`. Open `site/index.html` to read it.
- `make docs-serve`: Show the documentation at http://127.0.0.1:8000, rebuilding it when the code changes.

To set up a new environment, run `make setup` and then `make install`, and `make docs-install` if you want the documentation. While you still have the `.venv` folder, `make run-gui` runs the tracker. You only have to rerun `make install` if you update requirements.txt.
