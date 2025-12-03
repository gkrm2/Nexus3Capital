# Use Python 3.11 slim image (arm64)
FROM --platform=linux/arm64 python:3.11-slim

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make run script executable
RUN chmod +x run_pipeline.sh

# Run the pipeline
CMD ["./run_pipeline.sh"]
