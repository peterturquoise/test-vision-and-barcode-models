# Vision Models Project Structure
# Organized folder hierarchy for Docker-based deployment

# Vision Models Project Structure
# Organized folder hierarchy with per-model directories

```
test-7-vision-models/
├── 📁 src/                          # Source code
│   ├── 📁 models/                   # Model implementations (per-model directories)
│   │   ├── 📁 paddleocr/            # PaddleOCR model directory
│   │   │   ├── 📄 model.py          # Model implementation
│   │   │   ├── 📄 Dockerfile        # Container configuration
│   │   │   ├── 📄 requirements.txt  # Python dependencies
│   │   │   ├── 📄 README.md         # Model documentation
│   │   │   └── 📄 test.py           # Model-specific tests
│   │   ├── 📁 yolov9/               # YOLOv9 model directory
│   │   │   ├── 📄 model.py
│   │   │   ├── 📄 Dockerfile
│   │   │   ├── 📄 requirements.txt
│   │   │   ├── 📄 README.md
│   │   │   └── 📄 test.py
│   │   ├── 📁 llava15/              # LLaVA-1.5 model directory
│   │   │   ├── 📄 model.py
│   │   │   ├── 📄 Dockerfile
│   │   │   ├── 📄 requirements.txt
│   │   │   ├── 📄 README.md
│   │   │   └── 📄 test.py
│   │   └── 📁 ... (one directory per model)
│   ├── 📁 api/                      # API server code
│   │   ├── 📄 main.py              # FastAPI main server
│   │   ├── 📄 routes/              # API route modules
│   │   └── 📄 middleware/          # Custom middleware
│   └── 📁 utils/                    # Utility functions
│       ├── 📄 image_processing.py
│       ├── 📄 model_loader.py
│       └── 📄 response_formatter.py
│
├── 📁 docs/                         # Documentation
│   ├── 📁 architecture/             # Architecture docs
│   │   ├── 📄 docker_architecture_plan.md
│   │   ├── 📄 model_comparison.md
│   │   ├── 📄 api_design.md
│   │   └── 📄 project_structure.md
│   ├── 📁 deployment/              # Deployment guides
│   │   ├── 📄 azure_deployment.md
│   │   ├── 📄 aws_deployment.md
│   │   └── 📄 local_setup.md
│   └── 📄 README.md               # Main project README
│
├── 📁 config/                      # Configuration files
│   ├── 📄 model_config.yaml       # Model configurations
│   ├── 📄 docker_config.yaml      # Docker settings
│   └── 📄 api_config.yaml         # API settings
│
├── 📁 scripts/                     # Automation scripts
│   ├── 📄 build_all.sh            # Build all containers
│   ├── 📄 deploy_azure.sh         # Azure deployment
│   ├── 📄 deploy_aws.sh           # AWS deployment
│   ├── 📄 test_models.py          # Model testing
│   ├── 📄 main_test_runner.py     # Main test runner
│   └── 📄 quick_model_test.py     # Quick model test
│
├── 📁 tests/                       # Test suites
│   ├── 📁 unit/                   # Unit tests
│   │   ├── 📄 test_paddleocr.py
│   │   └── 📄 test_yolov9.py
│   └── 📁 integration/            # Integration tests
│       ├── 📄 test_api_endpoints.py
│       └── 📄 test_docker_containers.py
│
├── 📁 test_images/                     # Test images (existing)
│   ├── 📁 Complete Labels/
│   ├── 📁 Fragment Labels/
│   └── 📁 Broken Barcode/
│
├── 📁 results/                     # Test results (existing)
│   ├── 📄 YOLOv9_results.json
│   └── 📄 LLaVA-1.5_results.json
│
├── 📄 docker-compose.yml          # Local development
├── 📄 requirements.txt            # Main requirements
└── 📄 .gitignore                  # Git ignore rules
```

## 🎯 **Per-Model Directory Benefits:**

### **📁 src/models/{model_name}/** - Self-contained model packages
- **model.py**: Model implementation
- **Dockerfile**: Container configuration
- **requirements.txt**: Model-specific dependencies
- **README.md**: Model documentation
- **test.py**: Model-specific tests

### **✅ Advantages:**
1. **Self-contained**: Each model has everything it needs
2. **Easy to manage**: Clear separation between models
3. **Independent deployment**: Each model can be deployed separately
4. **Easy to add new models**: Just create a new directory
5. **Version control**: Easy to track changes per model
6. **Testing**: Each model can be tested independently

### **🔧 Usage Examples:**

```bash
# Test individual model
cd src/models/paddleocr
python test.py

# Build individual model
cd src/models/paddleocr
docker build -t vision-models/paddleocr .

# Run individual model
docker run -p 8000:8000 vision-models/paddleocr

# Test all models
cd scripts
python test_models.py

# Run all models locally
docker-compose up
```

## 🚀 **Migration Plan:**

1. **✅ Create per-model directories** - Done
2. **✅ Move model implementations** - Done
3. **✅ Create individual Dockerfiles** - Done
4. **✅ Update docker-compose.yml** - Done
5. **🔄 Create deployment scripts** - In progress
6. **🔄 Add comprehensive testing** - In progress

## 📋 **Next Steps:**

1. **Complete model directories** for all 14 models
2. **Create deployment scripts** for Azure/AWS
3. **Add comprehensive testing** suite
4. **Create API gateway** for model routing
5. **Add monitoring and logging**
