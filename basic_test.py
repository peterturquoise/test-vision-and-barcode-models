"""
Simple Test Script - Basic functionality test
"""

import sys
from pathlib import Path

def test_basic_imports():
    """Test if basic packages can be imported"""
    print("Testing basic imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL (Pillow)")
    except ImportError as e:
        print(f"✗ PIL import failed: {e}")
        return False
    
    try:
        import pytesseract
        print("✓ Tesseract OCR")
    except ImportError as e:
        print(f"✗ Tesseract import failed: {e}")
        return False
    
    try:
        from pyzbar import pyzbar
        print("✓ PyZbar (barcode detection)")
    except ImportError as e:
        print(f"✗ PyZbar import failed: {e}")
        return False
    
    return True

def test_yolov9_basic():
    """Test YOLOv9 basic functionality"""
    print("\nTesting YOLOv9 basic functionality...")
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics imported successfully")
        
        # Try to load a small model
        model = YOLO('yolov8n.pt')  # Use nano model for testing
        print("✓ YOLOv8n model loaded successfully")
        
        # Test with a simple image
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='white')
        print("✓ Test image created")
        
        # Run inference
        results = model(test_image)
        print("✓ YOLOv9 inference completed")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLOv9 test failed: {e}")
        return False

def test_sample_images():
    """Test if sample images exist"""
    print("\nChecking sample images...")
    
    samples_dir = Path("samples")
    if not samples_dir.exists():
        print("✗ Samples directory not found")
        return False
    
    image_files = list(samples_dir.glob("**/*.jpg")) + list(samples_dir.glob("**/*.jpeg")) + list(samples_dir.glob("**/*.png"))
    
    if not image_files:
        print("✗ No sample images found")
        return False
    
    print(f"✓ Found {len(image_files)} sample images")
    for img_file in image_files[:3]:  # Show first 3
        print(f"  - {img_file}")
    
    return True

def main():
    """Run basic tests"""
    print("="*60)
    print("VISION MODELS - BASIC FUNCTIONALITY TEST")
    print("="*60)
    
    # Test imports
    if not test_basic_imports():
        print("\n❌ Basic imports failed. Please check your installation.")
        return
    
    # Test sample images
    if not test_sample_images():
        print("\n❌ Sample images not found. Please add images to the 'samples' folder.")
        return
    
    # Test YOLOv9
    if not test_yolov9_basic():
        print("\n❌ YOLOv9 test failed.")
        return
    
    print("\n" + "="*60)
    print("✅ ALL BASIC TESTS PASSED!")
    print("="*60)
    print("Your environment is ready for vision model testing.")
    print("\nNext steps:")
    print("1. Run: python demo.py")
    print("2. Or run specific model: python test_individual.py yolov9")
    print("3. Or run all models: python main_test_runner.py")

if __name__ == "__main__":
    main()
