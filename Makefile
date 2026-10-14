.PHONY: venv install docs-install docs docs-serve

VENV=.venv\Scripts

setup:
	python -m venv .venv

install:
	$(VENV)\activate && pip install -r requirements.txt

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