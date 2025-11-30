FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY bitcoin_api_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r bitcoin_api_requirements.txt

# Copy application code
COPY *.py ./
COPY models/ ./models/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 starlight && chown -R starlight:starlight /app
USER starlight

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "bitcoin_api:app", "--host", "0.0.0.0", "--port", "8080"]