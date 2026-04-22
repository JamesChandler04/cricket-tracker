.PHONY: venv install

VENV=.venv\Scripts

setup:
	python -m venv .venv

install:
	$(VENV)\activate && pip install -r requirements.txt

run:
	$(VENV)\python cricket_ball_tracker.py

test:
	$(VENV)\activate && pip list

clean:
	del /Q .venv