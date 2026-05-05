FROM python:3.11-slim

# System packages:
#   curl  — used by docker-compose healthchecks
#   gcc + libpq-dev — required to compile psycopg2-binary on slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (separate layer so Docker can cache it)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command — overridden per-service in docker-compose.yml
CMD ["python", "train.py"]
