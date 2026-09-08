export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME=.hf_cache
export HF_HUB_OFFLINE=0

install:
	@echo " + Installing dependencies..."
	@echo "   - Installing python3.13"
	@uv python install 3.13
	@echo "   - Creating a virtual env"
	@uv venv --python 3.13
	@echo "   - Installing dependencies"
	@uv -n sync

run:
	@echo " + Launching the project..."
	@echo "   - Initializing env var"
	@uv run python3.13 -m src

debug:
	@echo " + Debugging the project..."
	@uv run python3.13 -m pdb -m src

clean:
	@echo " + Cleaning the project..."
	@find . -type d \( \
		-name "__pycache__" \
		-o -name ".mypy_cache" \
		-o -name ".pytest_cache" \
		-o -name ".ruff_cache" \
		-o -name "*.egg-info" \
		-o -name "dist" \
		-o -name "build" \
	\) -prune -exec rm -rf {} +
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

lint:
	@echo " + Checking norme..."
	@uv -n tool run flake8 src/
	@echo " + Checking type hint..."
	@uv -n tool run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@echo " + Checking norme..."
	@uv -n tool run flake8 src/
	@echo " + Checking type hint..."
	@uv -n tool run mypy src/ --strict
