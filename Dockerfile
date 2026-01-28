# ---- Builder Stage ----
FROM python:3.12-slim AS builder

# Set working directory for the virtual environment
WORKDIR /opt/venv

# Create a virtual environment
RUN python3 -m venv .

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the API-specific requirements file
COPY requirements.api.txt .

# Install CPU-only PyTorch (much smaller) and other dependencies
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    && pip install --no-cache-dir -r requirements.api.txt


# ---- Final Stage ----
FROM python:3.12-slim

# Set working directory in the final image
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies including nodejs for opencode
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install OpenCode
RUN curl -fsSL https://opencode.ai/install | bash \
    && mv /root/.opencode /opt/opencode \
    && ln -s /opt/opencode/bin/opencode /usr/local/bin/opencode \
    && chmod -R 755 /opt/opencode

# Create non-root user
RUN useradd -m -u 1000 starlight

# Copy application code
COPY *.py ./
COPY scripts/ ./scripts/
COPY starlight/ ./starlight/

# Copy models and metadata
COPY models_dist/ ./models/

# Ensure permissions for starlight user on home, app and opencode opts
RUN chown -R starlight:starlight /app /home/starlight /opt/opencode

USER starlight

# Set home environment variable for opencode
ENV HOME=/home/starlight

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "bitcoin_api:app", "--host", "0.0.0.0", "--port", "8080"]