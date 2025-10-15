# MobileSAM Model Directory
# Self-contained segmentation model with Docker configuration

## Files in this directory:
- `model.py` - MobileSAM model implementation
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
- `README.md` - Model documentation
- `test.py` - Model-specific tests

## Usage:
```bash
# Build container
docker build -t vision-models/mobilesam .

# Run container
docker run -p 8004:8000 vision-models/mobilesam

# Test model
python test.py
```

## Features:
- Lightweight segmentation model
- Good for object detection and segmentation
- Mobile-optimized performance





