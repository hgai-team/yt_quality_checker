# --- Builder stage ---
FROM python:3.11.13-slim AS builder
WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential libssl-dev libffi-dev libpq-dev python3-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --disable-pip-version-check \
      --prefix /install -r requirements.txt \
 # Remove dataclasses backport from the installed location
 && rm -rf /install/lib/python3.11/site-packages/dataclasses.py \
 && rm -rf /install/lib/python3.11/site-packages/dataclasses-*.dist-info \
 # Ensure typing-extensions is up to date
 && pip install --upgrade --prefix /install typing-extensions

# --- Runtime stage ---
FROM python:3.11.13-slim AS runtime
WORKDIR /app

# Install runtime libraries needed for compiled packages (e.g., psycopg2)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/install/bin:$PATH \
    PYTHONPATH=/install/lib/python3.11/site-packages

COPY --from=builder /install /install

# # Copy your application code
# COPY ./app /app

# # Set the command
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
