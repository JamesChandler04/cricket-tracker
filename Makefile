.PHONY: venv install

VENV=.venv\Scripts

setup:
	python -m venv .venv

install:
	$(VENV)\activate && pip install -r requirements.txt

run-text:
	$(VENV)\python -m main.new_cricket_ball_tracker

run-gui:
	$(VENV)\python -m main.GUI_new_cricket_ball_tracker

test:
	$(VENV)\activate && pip list

clean:
	del /Q .venv