# ZXing-CPP Model

Fast barcode detection using the ZXing-CPP library.

## Model Type
**Barcode-Only Container** - No prompts, no text extraction, just barcode detection.

## Features
- Fast barcode detection using ZXing-CPP
- Supports multiple barcode formats (QR, Code128, Code39, etc.)
- JSON output format
- Docker containerized for easy deployment

## API Endpoints

### `GET /`
Returns basic information about the API.

### `GET /health`
Health check endpoint.

### `POST /detect_barcodes`
Detect barcodes in an uploaded image.

**Request**: Multipart form with image file
**Response**: JSON with detected barcodes

```json
{
  "success": true,
  "model": "ZXing-CPP",
  "model_type": "barcode-only",
  "barcode_results": [
    {
      "bbox": [x, y, width, height],
      "value": "barcode_value",
      "barcode_type": "CODE128",
      "confidence": 0.95
    }
  ],
  "processing_time": 0.123,
  "image_size": [width, height],
  "image_format": "JPEG"
}
```

## Usage

### Build Container
```bash
python build_container.py
```

### Test Container
```bash
python test.py
```

### Run Container
```bash
docker run -p 8001:8000 vision-models/zxing-cpp
```

## Dependencies
- ZXing-CPP library
- OpenCV
- FastAPI
- Python 3.8+

## Notes
- This is a barcode-only model - it does not extract text or provide image descriptions
- Use this as the first step in a pipeline for fast barcode detection
- If barcodes are found, skip OCR models for faster processing





