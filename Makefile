PYTHON ?= python3
VENV ?= .venv
TORCH_VERSION ?= 2.11.0
TORCHVISION_VERSION ?= 0.26.0
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu128

.PHONY: setup
setup:
	@$(PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else "Python 3.10 or newer is required (selected: %s)" % sys.version.split()[0])'
	$(PYTHON) -m venv "$(VENV)"
	"$(VENV)/bin/python" -m pip install --upgrade pip
	"$(VENV)/bin/python" -m pip install "torch==$(TORCH_VERSION)" "torchvision==$(TORCHVISION_VERSION)" --index-url "$(TORCH_INDEX_URL)"
	"$(VENV)/bin/python" -m pip install -e '.[notebook,test]'
	"$(VENV)/bin/python" -c 'import torch; from esm.models.esmc import ESMC; print(f"ESM-C environment ready (torch={torch.__version__}, CUDA={torch.version.cuda})")'
