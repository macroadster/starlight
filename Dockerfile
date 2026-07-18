# Starlight training / export image (no API serve)
FROM python:3.12-slim AS builder

WORKDIR /opt/venv
RUN python3 -m venv .

ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt .

# CPU-only PyTorch (smaller image) + training/export deps
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

# ---- Final stage ----
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN useradd -m -u 1000 starlight

# Training / export surface
COPY trainer.py data_generator.py diag.py scanner.py ./
COPY scripts/ ./scripts/
COPY starlight/ ./starlight/
COPY models/ ./models/
COPY requirements.txt ./

RUN chown -R starlight:starlight /app
USER starlight

# Neutral default: show GGUF export help (train/export via make or override CMD)
CMD ["python", "scripts/export_starlight_gguf.py", "--help"]
