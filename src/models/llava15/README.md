# LLaVA-1.5 Model Directory
# Self-contained vision-language model with Docker configuration

## Files in this directory:
- `model.py` - LLaVA-1.5 model implementation
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
- `README.md` - Model documentation
- `test.py` - Model-specific tests

## Usage:
```bash
# Build container
docker build -t vision-models/llava15 .

# Run container
docker run -p 8002:8000 vision-models/llava15

# Test model
python test.py
```

## Features:
- Large Language and Vision Assistant (7B/13B variants)
- Strong vision-language capabilities
- Memory optimized for Docker deployment





