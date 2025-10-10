# 7 Free Vision Models Test Suite

This project implements comprehensive tests for 7 free vision models that can run locally on your laptop, as described in the article ["Forget GPT-4o: 7 Free Vision Models That Crush It on Your Laptop"](https://iamdgarcia.medium.com/forget-gpt-4o-7-free-vision-models-that-crush-it-on-your-laptop-4baeb8287925).

## Models Tested

1. **YOLOv9** - Advanced object detection with text and barcode detection capabilities
2. **MobileSAM** - Lightweight segmentation model optimized for mobile devices
3. **LLaVA-1.5** - Large Language and Vision Assistant for image understanding
4. **MiniGPT-4** - Compact vision-language model for image description
5. **Qwen-VL** - Versatile vision-language model for multimodal tasks
6. **CogVLM** - Vision-language model for image understanding
7. **Mobile-tuned LLaVA** - Mobile-optimized version of LLaVA

## Test Capabilities

For each model, the test suite evaluates:

1. **Image Description** - Detailed description of objects, people, text, and activities
2. **Text Extraction** - OCR with bounding boxes for all text in the image
3. **Barcode Detection** - Detection and decoding of barcodes and QR codes

## Quick Start

1. **Setup the environment:**
   ```bash
   python setup.py
   ```

2. **Add test images:**
   - Place your test images in the `samples/` folder
   - Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp

3. **Run all tests:**
   ```bash
   python main_test_runner.py
   ```

4. **Run a specific model:**
   ```bash
   python main_test_runner.py --model yolov9
   ```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Tesseract OCR
- Various model-specific dependencies (see requirements.txt)

## Project Structure

```
├── main_test_runner.py          # Main test runner
├── vision_test_framework.py     # Core testing framework
├── setup.py                     # Setup and installation script
├── requirements.txt             # Python dependencies
├── models/                      # Individual model implementations
│   ├── yolov9_model.py
│   ├── mobilesam_model.py
│   ├── llava15_model.py
│   ├── minigpt4_model.py
│   ├── qwen_vl_model.py
│   ├── cogvlm_model.py
│   └── mobile_llava_model.py
├── samples/                     # Test images directory
└── results/                     # Test results (JSON files)
```

## Model Details

### YOLOv9
- **Purpose**: Object detection, text detection, barcode detection
- **Strengths**: Fast, accurate object detection
- **Use Case**: General object detection with specialized text/barcode detection

### MobileSAM
- **Purpose**: Image segmentation
- **Strengths**: Lightweight, mobile-optimized
- **Use Case**: Object segmentation and region detection

### LLaVA-1.5
- **Purpose**: Vision-language understanding
- **Strengths**: Excellent image description and text extraction
- **Use Case**: Detailed image analysis and OCR

### MiniGPT-4
- **Purpose**: Compact vision-language model
- **Strengths**: Good balance of performance and efficiency
- **Use Case**: Image description and text extraction

### Qwen-VL
- **Purpose**: Versatile vision-language model
- **Strengths**: Multilingual support, good accuracy
- **Use Case**: Comprehensive image understanding

### CogVLM
- **Purpose**: Vision-language model
- **Strengths**: Strong visual understanding
- **Use Case**: Image analysis and description

### Mobile-tuned LLaVA
- **Purpose**: Mobile-optimized vision-language model
- **Strengths**: Efficient, optimized for mobile devices
- **Use Case**: Fast image processing on limited hardware

## Output Format

Results are saved as JSON files in the `results/` directory with the following structure:

```json
{
  "model_name": "YOLOv9",
  "image_path": "samples/test_image.jpg",
  "description": "Detailed image description...",
  "text_detections": [
    {
      "text": "Sample text",
      "bbox": [x, y, width, height],
      "confidence": 0.95
    }
  ],
  "barcode_detections": [
    {
      "bbox": [x, y, width, height],
      "value": "123456789",
      "barcode_type": "CODE128",
      "confidence": 1.0
    }
  ],
  "processing_time": 1.23,
  "error": null
}
```

## Performance Notes

- **GPU Acceleration**: Models will automatically use GPU if available
- **Memory Requirements**: Some models require significant RAM/VRAM
- **Processing Time**: Varies by model and image complexity
- **Accuracy**: Results depend on image quality and model capabilities

## Troubleshooting

1. **Model Loading Errors**: Check that all dependencies are installed
2. **Memory Issues**: Try running models individually or reduce image size
3. **OCR Issues**: Ensure Tesseract is properly installed and in PATH
4. **Barcode Detection**: Some barcodes may require specific image preprocessing

## Contributing

Feel free to contribute improvements, additional models, or bug fixes. The framework is designed to be easily extensible for new vision models.

## License

This project is open source. Please check individual model licenses for commercial use restrictions.