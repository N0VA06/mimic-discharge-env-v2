# ──────────────────────────────────────────────────────────────────────────────
# MIMIC Discharge Planning — OpenEnv  (v2)
#
# Build:  docker build -t mimic-discharge-env .
# Run:    docker run -p 7860:7860 mimic-discharge-env
#
# With HF Router (primary) + Bedrock fallback:
#   docker run -p 7860:7860 \
#     -e HF_TOKEN=hf_... \
#     -e MODEL_NAME=Qwen/Qwen2.5-14B:featherless-ai \
#     -e API_BASE_URL=https://router.huggingface.co/v1 \
#     -e LLM_PRIORITY=hf_router,bedrock \
#     -e AWS_ACCESS_KEY_ID=... \
#     -e AWS_SECRET_ACCESS_KEY=... \
#     -e AWS_REGION=us-east-1 \
#     mimic-discharge-env
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# HuggingFace Spaces requires non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY --chown=user environment/ environment/
COPY --chown=user server/ server/
COPY --chown=user inference.py .
COPY --chown=user openenv.yaml .

# Demo dataset — bundled so the Space works without external downloads
COPY --chown=user mimic-iv-clinical-database-demo-2.2/ mimic-iv-clinical-database-demo-2.2/

ENV PORT=7860
ENV MIMIC_DATA_PATH=/app/mimic-iv-clinical-database-demo-2.2

EXPOSE 7860

# /health returns 200 once environment tables are loaded
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "\
import urllib.request, sys; \
r = urllib.request.urlopen('http://localhost:7860/health'); \
import json; d = json.loads(r.read()); \
sys.exit(0 if d.get('ready') else 1)"

CMD ["python", "-m", "server.app"]