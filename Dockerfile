FROM python:3.11-slim

# HuggingFace Spaces requires non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY --chown=user requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY --chown=user environment/ environment/
COPY --chown=user server/ server/
COPY --chown=user inference.py .
COPY --chown=user openenv.yaml .

# Install the local package so `server` and `environment` are in site-packages.
# --no-deps avoids reinstalling everything from requirements.txt.
RUN pip install --no-cache-dir --no-deps .

# Demo dataset — bundled so the Space works without external downloads
COPY --chown=user mimic-iv-clinical-database-demo-2.2/ mimic-iv-clinical-database-demo-2.2/

ENV PORT=7860
ENV MIMIC_DATA_PATH=/app/mimic-iv-clinical-database-demo-2.2
# Belt-and-suspenders: ensures /app is always on sys.path regardless of
# how the interpreter is invoked (e.g. uvicorn worker subprocesses).
ENV PYTHONPATH=/app

EXPOSE 7860

# /health returns 200 once environment tables are loaded
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "\
import urllib.request, sys; \
r = urllib.request.urlopen('http://localhost:7860/health'); \
import json; d = json.loads(r.read()); \
sys.exit(0 if d.get('ready') else 1)"

CMD ["python", "-m", "server.app","--port", "7860"]