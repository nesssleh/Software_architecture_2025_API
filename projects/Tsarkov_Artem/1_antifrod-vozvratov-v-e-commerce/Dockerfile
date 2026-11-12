FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY returnguard ./returnguard
COPY services ./services
COPY scripts ./scripts
COPY models ./models
COPY tests ./tests
EXPOSE 8000
CMD ["uvicorn", "services.investigation_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
