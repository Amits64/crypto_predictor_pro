############################################
# 1) Builder stage: install build deps, build wheels
############################################
FROM python:3.10-slim AS builder

# Prevent Python buffering & fix PyTorch warning
ENV PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# Install system build-time dependencies for TA-Lib and wheel creation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libta-lib0 \
      libta-lib-dev \
      gcc \
      g++ \
      && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage Docker cache
COPY requirements.txt .

# Build wheels for all Python deps
RUN pip install --upgrade pip wheel && \
    pip wheel --wheel-dir=/wheels --no-cache-dir -r requirements.txt

############################################
# 2) Final stage: only runtime deps + app
############################################
FROM python:3.10-slim AS runtime

# Streamlit config
ENV STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER=false \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install only the runtime system library for TA-Lib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libta-lib0 \
    && rm -rf /var/lib/apt/lists/*

# Copy in built wheels and install them
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Launch Streamlit
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
