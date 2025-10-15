# BLIP Model Directory
# Self-contained vision-language model with Docker configuration

## Files in this directory:
- `model.py` - BLIP model implementation
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
- `README.md` - Model documentation
- `test.py` - Model-specific tests

## Usage:
```bash
# Build container
docker build -t vision-models/blip .

# Run container
docker run -p 8003:8000 vision-models/blip

# Test model
python test.py
```

## Features:
- Better vision-language model than current MiniGPT-4 implementation
- Simple dependencies (~1GB)
- Good for text extraction and image description





