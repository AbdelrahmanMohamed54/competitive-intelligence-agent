FROM python:3.11-slim

# System dependencies required by ChromaDB / sentence-transformers / onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source so the layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project source files
COPY agents/       ./agents/
COPY app/          ./app/
COPY evaluation/   ./evaluation/
COPY observability/ ./observability/
COPY schemas/      ./schemas/
COPY tools/        ./tools/
COPY agents/__init__.py ./agents/__init__.py

# Create ChromaDB persistence directory
RUN mkdir -p /app/.chroma

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Default: run the FastAPI backend
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
