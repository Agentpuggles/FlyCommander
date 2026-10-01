FORGE_DIR    := /home/flynn/Downloads/mtg forge
FORGE_JAR    := $(FORGE_DIR)/forge-gui-desktop-2.0.15-jar-with-dependencies.jar
PATCH_CLASSES := $(FORGE_DIR)/forge-agent-patch/classes
PATCH_SRC    := forge_patch/src/fly/agent

.PHONY: help test compile run train physical physical-setup physical-check clean

help:
	@echo "FlyCommander targets:"
	@echo "  make test     - run the Python test suite"
	@echo "  make compile  - compile the Java agent patch against Forge"
	@echo "  make run      - one demo Commander match, fly vs Forge AI"
	@echo "  make train    - short training run (5 games)"
	@echo "  make physical-setup - install camera/reference scanner dependencies"
	@echo "  make physical-check - check OpenCV and optional OCR"
	@echo "  make physical - physical-table mode (camera + fly UI on :8795)"
	@echo "  make clean    - remove logs/checkpoints and compiled classes"

test:
	python3 -m pytest tests/ -q

compile:
	mkdir -p "$(PATCH_CLASSES)"
	javac -cp "$(FORGE_JAR)" -d "$(PATCH_CLASSES)" $(PATCH_SRC)/*.java

run:
	python3 scripts/run_match.py --games 1

train:
	python3 scripts/train.py --episodes 5 --games-per-vm 5

physical-setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements-physical.txt

physical-check:
	.venv/bin/python -c 'import cv2; from physical.card_scan import tesseract_ready; print("Artwork scanner:", "ready" if hasattr(cv2, "SIFT_create") else "SIFT unavailable"); print("Optional OCR:", tesseract_ready())'

physical:
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python scripts/physical_table.py; \
	else \
		echo "[warn] .venv missing — camera OCR disabled (manual registration only)"; \
		python3 scripts/physical_table.py; \
	fi

clean:
	rm -rf logs checkpoints
	rm -rf $(PATCH_CLASSES)
