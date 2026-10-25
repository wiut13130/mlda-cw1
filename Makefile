# Convenience targets (Unix shells and Git Bash). Windows users can run the uv commands directly.
.PHONY: sync lock export lint test test-all notebook app train download docker-build docker-run

sync:            ## create/update the virtual environment from uv.lock
	uv sync --locked --all-groups

lock:            ## re-resolve dependencies and refresh the exported requirement files
	uv lock
	$(MAKE) export

export:          ## export pinned requirements for pip users and for Streamlit Community Cloud
	uv export --format requirements-txt --all-groups --no-hashes --no-emit-project -o requirements.txt
	uv export --format requirements-txt --no-dev --no-hashes --no-emit-project -o app/requirements.txt

lint:
	uv run ruff check src tests app notebooks

test:            ## fast unit and app smoke tests
	uv run pytest -m "not slow"

test-all:        ## includes notebook execution (slow)
	uv run pytest

download:        ## fetch the MSHA source files and verify their SHA-256
	uv run mlda-download

train:           ## rebuild data/processed, models/ and reports/
	uv run mlda-train

notebook:
	uv run jupyter lab notebooks/01_end_to_end.ipynb

app:
	uv run streamlit run app/streamlit_app.py

docker-build:
	docker build -t mlda-cw1 .

docker-run:
	docker run --rm -p 8501:8501 mlda-cw1
