FROM python:3.11.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# Cài thêm sentencepiece & protobuf
RUN pip install --no-cache-dir fastapi uvicorn[standard] pandas openpyxl httpx apscheduler

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
