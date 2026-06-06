FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging files and install dependencies
COPY pyproject.toml setup.py /app/
RUN pip install --no-cache-dir -e .

# Copy Streamlit dashboard portal code
COPY dashboard /app/dashboard

# Open Streamlit port
EXPOSE 8501

# Run Streamlit frontend application
CMD ["streamlit", "run", "dashboard/main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
