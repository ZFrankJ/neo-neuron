.PHONY: check test docs-check mlx-parity mps-probe cuda-probe torch-validate help

PYTHON ?= python3

check: docs-check test

test:
	$(PYTHON) -m pytest -q

docs-check:
	@test -f AGENTS.md
	@test -f README.md
	@test -f justfile
	@test -f Makefile
	@test -f .github/workflows/tests.yml
	@test -f docs/ROADMAP.md
	@test -f docs/PROGRESS.md
	@test -f docs/IMPLEMENTATION_PLAN.md
	@test -f docs/DEVLOG.md
	@test -f docs/DOCUMENTATION_STRUCTURE.md
	@test -f docs/CONTEXT_BOOTSTRAP.md

mlx-parity:
	$(PYTHON) -m pytest -q tests/test_mlx_reference_parity.py

mps-probe:
	NEO_RUN_MPS_PROBE=1 $(PYTHON) -m pytest -q tests/test_mps_no_checkpoint_probe.py

cuda-probe:
	NEO_RUN_CUDA_PROBE=1 $(PYTHON) -m pytest -q tests/test_cuda_parity_harness.py

torch-validate:
	bash scripts/validate_torch_path.sh

help:
	@echo "Targets:"
	@echo "  check       - docs contract plus full pytest suite"
	@echo "  test        - full pytest suite"
	@echo "  docs-check  - required harness docs/workflow files exist"
	@echo "  mlx-parity  - MLX reference parity tests"
	@echo "  mps-probe   - opt-in tiny PyTorch MPS no-checkpoint probe"
	@echo "  cuda-probe  - opt-in tiny PyTorch CUDA no-checkpoint probe"
	@echo "  torch-validate - Torch CPU checks plus MLX/MPS parity preflights"
