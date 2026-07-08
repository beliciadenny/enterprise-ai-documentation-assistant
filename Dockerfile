# Enterprise AI Documentation Assistant
# Builds a single image containing the FastAPI backend and Streamlit
# frontend. Ollama (the LLM runtime) is run as a separate container/host
# service — see docker-compose.yml and the README for details.

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout, and set the
# working directory for the whole application.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies required by unstructured/pypdf and faiss-cpu.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application source.
COPY backend/ backend/
COPY frontend/ frontend/
COPY start.sh .
COPY .env.example .env.example

RUN mkdir -p docs vectorstore screenshots \
    && chmod +x start.sh

# FastAPI backend port and Streamlit frontend port.
EXPOSE 8000 8501

CMD ["./start.sh"]
