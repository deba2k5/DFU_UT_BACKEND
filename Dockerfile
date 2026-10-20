FROM python:3.11-slim
WORKDIR /srv
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY weights/ weights/
# TORCH_THREADS: CPU threads per worker. WEB_CONCURRENCY: workers (each loads the ~8 MB model,
# ~500 MB RAM). DATA_DIR should point at a persistent disk on your host.
ENV TORCH_THREADS=2 WEB_CONCURRENCY=1 PORT=8000 DATA_DIR=/srv/data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY}"]
