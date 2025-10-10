"""
Demo Script - Quick Test of Vision Models
This script demonstrates the capabilities of the vision models test suite
"""

import sys
from pathlib import Path

# Add models directory to path
sys.path.append(str(Path(__file__).parent / "models"))

def demo_yolov9():
    """Demo YOLOv9 capabilities"""
    print("="*60)
    print("YOLOv9 DEMO")
    print("="*60)
    
    try:
        from yolov9_model import YOLOv9Model
        from vision_test_framework import VisionModelTester
        
        # Initialize model
        model = YOLOv9Model()
        
        # Get sample images
        samples_dir = Path("samples")
        image_files = list(samples_dir.glob("**/*.jpeg")) + list(samples_dir.glob("**/*.jpg")) + list(samples_dir.glob("**/*.png"))
        
        if not image_files:
            print("No sample images found. Please add images to the 'samples' folder.")
            return
        
        # Test with first image
        test_image = str(image_files[0])
        print(f"Testing with: {test_image}")
        
        # Process image
        result = model.process_image(test_image)
        
        print(f"\nDescription: {result.description}")
        print(f"\nText Detections ({len(result.text_detections)}):")
        for i, text_det in enumerate(result.text_detections):
            print(f"  {i+1}. '{text_det.text}' (confidence: {text_det.confidence:.2f})")
        
        print(f"\nBarcode Detections ({len(result.barcode_detections)}):")
        for i, barcode_det in enumerate(result.barcode_detections):
            print(f"  {i+1}. Value: '{barcode_det.value}' Type: {barcode_det.barcode_type} (confidence: {barcode_det.confidence:.2f})")
        
        print(f"\nProcessing Time: {result.processing_time:.2f} seconds")
        
        if result.error:
            print(f"\nError: {result.error}")
            
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure to install dependencies first: python setup.py")

def demo_llava():
    """Demo LLaVA-1.5 capabilities"""
    print("\n" + "="*60)
    print("LLaVA-1.5 DEMO")
    print("="*60)
    
    try:
        from llava15_model import LLaVA15Model
        
        # Initialize model
        model = LLaVA15Model()
        
        # Get sample images
        samples_dir = Path("samples")
        image_files = list(samples_dir.glob("**/*.jpeg")) + list(samples_dir.glob("**/*.jpg")) + list(samples_dir.glob("**/*.png"))
        
        if not image_files:
            print("No sample images found.")
            return
        
        # Test with first image
        test_image = str(image_files[0])
        print(f"Testing with: {test_image}")
        
        # Process image
        result = model.process_image(test_image)
        
        print(f"\nDescription: {result.description}")
        print(f"\nText Detections ({len(result.text_detections)}):")
        for i, text_det in enumerate(result.text_detections):
            print(f"  {i+1}. '{text_det.text}' (confidence: {text_det.confidence:.2f})")
        
        print(f"\nProcessing Time: {result.processing_time:.2f} seconds")
        
        if result.error:
            print(f"\nError: {result.error}")
            
    except Exception as e:
        print(f"Demo failed: {e}")
        print("LLaVA-1.5 requires significant GPU memory. Try YOLOv9 demo instead.")

def main():
    """Run demos"""
    print("VISION MODELS DEMO")
    print("This demo shows the capabilities of the 7 free vision models")
    print("\nNote: Some models require significant GPU memory and may fail on CPU-only systems")
    
    # Try YOLOv9 first (most likely to work)
    demo_yolov9()
    
    # Try LLaVA-1.5 (may require GPU)
    demo_llava()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("To run full tests on all models:")
    print("  python main_test_runner.py")
    print("\nTo run tests on a specific model:")
    print("  python main_test_runner.py --model yolov9")
    print("  python main_test_runner.py --model llava")
    print("\nTo run individual model tests:")
    print("  python test_individual.py yolov9")

if __name__ == "__main__":
    main()
