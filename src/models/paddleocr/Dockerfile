# PaddleOCR Docker Container
# Using official PaddleOCR image as base

FROM paddlecloud/paddleocr:2.6-cpu-latest

# Set working directory
WORKDIR /app

# Copy our custom model files
COPY model.py .
COPY api_server.py .

# Copy requirements file
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PaddleOCR (not included in base image)
RUN pip install --no-cache-dir paddleocr

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["python", "api_server.py"]