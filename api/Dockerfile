FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging files and install dependencies
COPY pyproject.toml setup.py /app/
RUN pip install --no-cache-dir -e .

# Copy pipeline engine modules and api code
COPY engine /app/engine
COPY api /app/api

# Open API port
EXPOSE 8000

# Run FastAPI backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
