# Serving image: ONNX Runtime + FastAPI only. No PyTorch, no model code — just the exported graphs.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve/ serve/
COPY artifacts/ artifacts/
ENV ARTIFACTS=/app/artifacts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
