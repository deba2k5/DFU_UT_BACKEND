FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY weights/*.onnx weights/
# TORCH-free image (~350 MB). WEB_CONCURRENCY: workers (~250 MB RAM each).
# DATA_DIR should point at a persistent disk on your host.
ENV MODEL_THREADS=2 WEB_CONCURRENCY=1 PORT=8000 DATA_DIR=/srv/data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY}"]
