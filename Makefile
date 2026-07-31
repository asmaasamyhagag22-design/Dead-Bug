# Dead Bug AQA -- the commands, in the order they are meant to be run.
#
# PY / PYA point at the two environments. They are separate by necessity, not by
# preference: aeon pins numpy<2.5, pandas<2.4 and scipy<1.18, all below what
# MediaPipe and OpenCV need. Installing both into one venv downgrades a working
# stack.
#
# Override them if your venvs live elsewhere -- which they should if the repo is
# inside a cloud-synced folder, where installing tens of thousands of small
# files fails with OS error 1450:
#
#     make test PY=C:/Users/you/venvs/deadbug/Scripts/python.exe

PY  ?= ./venv/Scripts/python.exe
PYA ?= ./venv-a/Scripts/python.exe

.DEFAULT_GOAL := help
.PHONY: help venv venv-a test gate0 triage build qc band pipeline \
        coach demo masar-a masar-sanity clean-cache clean-interim

help:
	@echo "setup    venv venv-a"
	@echo "check    test gate0"
	@echo "data     triage build qc band  (or: pipeline)"
	@echo "app      coach demo"
	@echo "bench    masar-sanity masar-a"
	@echo "clean    clean-cache clean-interim"

# -- setup -----------------------------------------------------------------

venv:
	uv venv venv --python 3.13
	uv pip install --python venv -r requirements.txt

venv-a:
	uv venv venv-a --python 3.13
	uv pip install --python venv-a -r requirements-a.txt

# -- checks ----------------------------------------------------------------

test:
	$(PY) -m pytest tests/ -q

# Gate 0 is normalization and skeleton invariance. If it fails, stop: every
# number produced after it is meaningless.
gate0:
	$(PY) -m pytest tests/test_normalize.py tests/test_skeleton.py -q

# -- data ------------------------------------------------------------------

# Measure every clip and draft data/clips.csv. person_id still needs a human.
triage:
	$(PY) scripts/run_triage.py --write-manifest

build:
	$(PY) -m deadbug.cli build

qc:
	$(PY) -m deadbug.cli qc

band:
	$(PY) -m deadbug.cli band

pipeline: gate0 build qc band

# -- app -------------------------------------------------------------------

coach:
	$(PY) scripts/run_live.py --source 0

# A rehearsable demo: the same code path as the camera, on a file, so it can be
# debugged and practised without one.
demo:
	$(PY) scripts/run_live.py --source "data/clips/videoplayback (1).mp4"

# -- benchmark -------------------------------------------------------------

masar-sanity:
	$(PYA) scripts/run_masar_a.py --sanity

masar-a:
	$(PYA) scripts/run_masar_a.py --all --cheap

# -- clean -----------------------------------------------------------------

clean-cache:
	$(PY) -c "import pathlib,shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	rm -rf .pytest_cache

# The extraction cache only. Rebuilding it costs minutes per clip, so it is
# never part of `clean`.
clean-interim:
	rm -f data/interim/*.npz
