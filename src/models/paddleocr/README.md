# PaddleOCR Model Directory
# Self-contained model with Docker configuration

## Files in this directory:
- `model.py` - PaddleOCR model implementation
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
- `README.md` - Model documentation
- `test.py` - Model test script (direct testing)
- `build_container.py` - Build Docker container
- `test_container.py` - Test Docker container

## Usage:
```bash
# Test model directly (no Docker)
python test.py

# Build Docker container
python build_container.py

# Test Docker container
python test_container.py
```

## Features:
- High-performance OCR with PaddleOCR
- Text extraction with confidence scores
- Barcode detection using Pyzbar
- Package-focused analysis
- Docker-ready deployment