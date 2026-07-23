PYTHON ?= python3
VENV ?= .venv

.PHONY: setup
setup:
	@$(PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else "Python 3.10 or newer is required (selected: %s)" % sys.version.split()[0])'
	$(PYTHON) -m venv "$(VENV)"
	"$(VENV)/bin/python" -m pip install --upgrade pip
	"$(VENV)/bin/python" -m pip install -e '.[notebook,test]'
	"$(VENV)/bin/python" -c 'from esm.models.esmc import ESMC; print("ESM-C environment ready")'
