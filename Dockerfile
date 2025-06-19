FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Expose the metrics port
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Default command (override as needed)
CMD ["python", "scripts/run_morphogenetic_experiment.py", "--problem_type", "spirals", "--n_samples", "2000", "--warm_up_epochs", "20", "--adaptation_epochs", "40"]
