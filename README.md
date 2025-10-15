# Vision Models Test Suite

A comprehensive test suite for 14 vision models, designed for package analysis, text extraction, and barcode detection. All models run as Docker containers for consistent deployment across Mac, Azure, and AWS.

## 🎯 **Project Overview**

This project tests and compares 14 vision models from two sources:
- **Medium Article**: 7 free vision models that work on laptops
- **Labellerr Article**: 5 top open-source vision-language models of 2025
- **User Suggestions**: BLIP and PaddleOCR for better performance

## 📁 **Project Structure**

```
test-7-vision-models/
├── 📁 src/                          # Source code
│   ├── 📁 models/                   # Model implementations
│   ├── 📁 api/                      # API server code
│   └── 📁 utils/                    # Utility functions
├── 📁 dockerfiles/                  # Docker configurations
│   ├── 📁 base/                    # Base images
│   ├── 📁 containers/              # Model-specific containers
│   └── 📄 docker-compose.yml       # Local development
├── 📁 docs/                         # Documentation
│   ├── 📁 architecture/             # Architecture docs
│   └── 📁 deployment/              # Deployment guides
├── 📁 scripts/                     # Automation scripts
├── 📁 tests/                       # Test suites
├── 📁 test_images/                     # Test images
└── 📁 results/                     # Test results
```

## 🚀 **Quick Start**

### **Local Development (Mac)**
```bash
# 1. Test PaddleOCR directly (no Docker)
cd src/models/paddleocr
python test.py

# 2. Build PaddleOCR container
cd src/models/paddleocr
python build_container.py

# 3. Test PaddleOCR container
cd src/models/paddleocr
python test_container.py
```

### **Azure Deployment**
```bash
# Deploy to Azure Container Instances
cd scripts
./deploy_azure.sh
```

### **AWS Deployment**
```bash
# Deploy to AWS ECS Fargate
cd scripts
./deploy_aws.sh
```

## 🐳 **Docker Architecture**

### **Why All-Docker?**
- **Consistency**: Same deployment process for all models
- **Predictability**: Uniform resource management and scaling
- **Maintainability**: One deployment pipeline, not multiple
- **Future-proof**: Easy to add new models
- **No edge cases**: No "this model works serverless, that one doesn't"

### **Model Categories**
All 14 models run as Docker containers:

| **Model** | **Size** | **Dependencies** | **Docker Benefit** |
|-----------|----------|------------------|-------------------|
| **PaddleOCR** | ~500MB | Complex C++ libs | **Essential** - Only way to deploy reliably |
| **YOLOv9** | ~200MB | OpenCV, PyTorch | **High** - Consistent OpenCV versions |
| **LLaVA-1.5** | ~7GB | Transformers, CUDA | **High** - GPU memory management |
| **Qwen-VL** | ~4GB | Transformers | **High** - Memory optimization |
| **CogVLM** | ~17GB | Complex dependencies | **Essential** - Too large for serverless |
| **BLIP** | ~1GB | Transformers only | **Medium** - Consistency matters |
| **Phi-4** | ~2GB | Lightweight | **Medium** - Docker for consistency |

## 📊 **Models Tested**

### **From Medium Article**
1. **YOLOv9** - Advanced object detection
2. **MobileSAM** - Lightweight segmentation
3. **LLaVA-1.5** - Large Language and Vision Assistant
4. **MiniGPT-4** - Compact vision-language model
5. **Qwen-VL** - Versatile vision-language model
6. **CogVLM** - Vision-language model
7. **Mobile-Enhanced LLaVA** - Mobile-optimized LLaVA

### **From Labellerr Article**
8. **Gemma 3** - High-res vision, multilingual OCR
9. **Qwen 2.5 VL** - Superior document understanding
10. **LLaMA 3.2 Vision** - Strong OCR, document VQA
11. **DeepSeek-VL** - Strong reasoning, scientific tasks
12. **Phi-4 Multimodal** - Lightweight, on-device potential

### **User Suggestions**
13. **BLIP** - Better vision-language model
14. **PaddleOCR** - Good OCR performance

## 🧪 **Testing**

### **Test Image**
Primary test image: `test_images/Complete Labels/3_J18CBEP8CCN070812400095N_90_rot.jpeg`
- Package shipping label
- Multiple barcodes
- Text in various orientations
- Real-world complexity

### **Test Categories**
1. **Image Description** - What the model sees
2. **Text Extraction** - All text found with confidence scores
3. **Barcode Detection** - Barcodes found with values and positions

### **Run Tests**
```bash
# Quick model loading test
python scripts/quick_model_test.py

# Full model comparison
python scripts/test_models.py

# Test specific model
python scripts/main_test_runner.py --model PaddleOCR
```

## 🔧 **Development**

### **Adding New Models**
1. Create model implementation in `src/models/new_model.py`
2. Create Dockerfile in `dockerfiles/containers/Dockerfile.new_model`
3. Add to `dockerfiles/docker-compose.yml`
4. Update `scripts/test_models.py`

### **API Endpoints**
- `GET /` - API information
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /analyze/{model_name}` - Complete analysis
- `POST /text/{model_name}` - Text extraction only
- `POST /barcodes/{model_name}` - Barcode detection only

## 📈 **Performance**

### **Model Loading Times**
- **YOLOv9**: ~0.8s
- **MobileSAM**: ~0.9s
- **BLIP**: ~5.0s
- **Qwen-VL**: ~8.1s
- **CogVLM**: ~4.4s
- **LLaVA-1.5**: ~72s
- **Mobile-LLaVA**: ~229s

### **Processing Times**
- **Text Extraction**: 0.5-5s per image
- **Barcode Detection**: 0.1-1s per image
- **Image Description**: 1-10s per image

## 🚀 **Deployment**

### **Local (Mac)**
```bash
docker-compose up
```

### **Azure**
- Azure Container Instances (ACI)
- Azure App Service
- Azure Kubernetes Service (AKS)

### **AWS**
- ECS Fargate
- Lambda (for lightweight models)
- SageMaker

## 📚 **Documentation**

- [Architecture Overview](docs/architecture/docker_architecture_plan.md)
- [Project Structure](docs/architecture/project_structure.md)
- [Azure Deployment](docs/deployment/azure_deployment.md)
- [AWS Deployment](docs/deployment/aws_deployment.md)

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Add your model implementation
4. Create corresponding Dockerfile
5. Add tests
6. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- [Medium Article](https://iamdgarcia.medium.com/forget-gpt-4o-7-free-vision-models-that-crush-it-on-your-laptop-4baeb8287925) - Original 7 models
- [Labellerr Article](https://www.labellerr.com/blog/top-open-source-vision-language-models/) - Additional 5 models
- All the open-source model developers and contributors