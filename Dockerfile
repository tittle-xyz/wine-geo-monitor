# wine-geo-monitor — Dagster code-location image.
#
# Secrets (ANTHROPIC_API_KEY / OPENAI_API_KEY) arrive as env vars from a Doppler-synced
# k8s Secret — the same operator pattern home-infra uses for track (a DopplerSecret CRD
# → managed Secret → secretKeyRef in the pod). Nothing secret lives in the image or repo.
FROM python:3.11-slim

ENV DAGSTER_HOME=/opt/dagster/dagster_home \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[dagster,anthropic,openai,viz]" \
 && mkdir -p "$DAGSTER_HOME"

EXPOSE 4000
# The Dagster Helm user-code deployment runs this gRPC server serving wine_geo.definitions.
CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4000", "-m", "wine_geo.definitions"]
