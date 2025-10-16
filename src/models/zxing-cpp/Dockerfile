# ZXing-CPP Docker Container
# Fast barcode detection using ZXing-CPP

FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies in smaller chunks to avoid timeout
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install ZXing-CPP with timeout handling
RUN git clone --depth 1 https://github.com/nu-book/zxing-cpp.git /tmp/zxing-cpp && \
    cd /tmp/zxing-cpp && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON && \
    make -j2 && \
    make install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/zxing-cpp

# Set working directory
WORKDIR /app

# Copy model files
COPY model_parallel_pattern.py .
COPY api_server.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["python3", "api_server.py"]





