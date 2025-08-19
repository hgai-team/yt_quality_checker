FROM python:3.11.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# Cài thêm sentencepiece & protobuf
RUN pip install --no-cache-dir fastapi uvicorn[standard] torch torchvision transformers pillow sentencepiece protobuf python-multipart

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
