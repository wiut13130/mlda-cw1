# Container for the Streamlit app (CW1, student ID 13130).
# Build:  docker build -t mlda-cw1 .
# Run:    docker run --rm -p 8501:8501 mlda-cw1
# Hosts that assign the port through PORT are served on it; without PORT the app listens on 8501.
# Add -e MLDA_LOW_MEMORY=1 on a host with 512 MB of RAM.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

# Dependency layer first so code changes do not invalidate the install cache.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# Application code and the precomputed artefacts the app serves.
COPY .streamlit ./.streamlit
COPY app ./app
COPY data/processed ./data/processed
COPY data/raw/SNAPSHOT.json data/raw/*_Definition_File.txt ./data/raw/
COPY models ./models
COPY reports ./reports

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os, sys, urllib.request; url = 'http://localhost:%s/_stcore/health' % os.environ.get('PORT', '8501'); sys.exit(0 if urllib.request.urlopen(url).status == 200 else 1)"

# Shell form so that ${PORT:-8501} is expanded; exec makes Streamlit PID 1, so it receives SIGTERM on a redeploy.
CMD exec streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
