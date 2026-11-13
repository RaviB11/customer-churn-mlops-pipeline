# Telco churn prediction API
# Build:  docker build -t telco-churn .
# Run:    docker run -p 8000:8000 -v ${PWD}/monitoring:/app/monitoring telco-churn
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHURN_PROJECT_ROOT=/app \
    CHURN_DISABLE_MLFLOW=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY churn_pipeline/ churn_pipeline/
COPY scripts/ scripts/
COPY data/raw/ data/raw/
COPY data/sample/ data/sample/

# Train inside the image so the pickled model always matches the image's
# library versions (avoids scikit-learn/xgboost version-mismatch errors).
RUN python scripts/train.py

RUN useradd --create-home appuser && mkdir -p monitoring && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "churn_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]
