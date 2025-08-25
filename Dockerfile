# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    libxrender1 \
    libxext6 \
    libsm6 \
    libice6 \
    libglib2.0-0 \
    libfontconfig1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY docker-requirements.txt .

# Install Python dependencies
# Using simplified requirements for containerized deployment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r docker-requirements.txt

# Create compatibility layer for mmpdblib (REINVENT4 expects old API)
RUN cd /usr/local/lib/python3.10/site-packages/mmpdblib && \
    echo "# Compatibility layer for REINVENT4" > do_fragment.py && \
    echo "from mmpdblib.cli.fragment import fragment as fragment_command" >> do_fragment.py && \
    echo "__all__ = ['fragment_command']" >> do_fragment.py && \
    echo "# Compatibility layer for REINVENT4" > do_index.py && \
    echo "from mmpdblib.cli.index import index as index_command" >> do_index.py && \
    echo "__all__ = ['index_command']" >> do_index.py

# Copy the application code
COPY . .

# Copy and make startup script executable
COPY start.sh .
RUN chmod +x start.sh

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/outputs

# Expose the port that Streamlit runs on
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the application
CMD ["./start.sh"]
