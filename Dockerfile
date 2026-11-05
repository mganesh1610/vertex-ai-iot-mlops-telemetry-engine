FROM python:3.11-slim
WORKDIR /app
COPY src/ ./src/
COPY modules/ ./modules/
EXPOSE 8080
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
