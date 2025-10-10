"""
Main Test Runner for 7 Free Vision Models
Runs comprehensive tests on all models with sample images
"""

import os
import sys
import argparse
from pathlib import Path

# Add models directory to path
sys.path.append(str(Path(__file__).parent / "models"))

from vision_test_framework import VisionModelTester
from yolov9_model import YOLOv9Model
from mobilesam_model import MobileSAMModel
from llava15_model import LLaVA15Model
from minigpt4_model import MiniGPT4Model
from qwen_vl_model import QwenVLModel
from cogvlm_model import CogVLMModel
from mobile_llava_model import MobileLLaVAModel

def create_sample_images():
    """Create some sample images for testing if none exist"""
    samples_dir = Path("samples")
    if not samples_dir.exists():
        samples_dir.mkdir()
    
    # Check if there are any images
    image_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpeg"))
    
    if not image_files:
        print("No sample images found in the 'samples' directory.")
        print("Please add some test images to the 'samples' folder.")
        print("Supported formats: .jpg, .jpeg, .png")
        return False
    
    return True

def run_all_tests():
    """Run tests on all 7 vision models"""
    print("="*80)
    print("7 FREE VISION MODELS TEST SUITE")
    print("="*80)
    print("Testing models:")
    print("1. YOLOv9 - Advanced object detection")
    print("2. MobileSAM - Lightweight segmentation")
    print("3. LLaVA-1.5 - Large Language and Vision Assistant")
    print("4. MiniGPT-4 - Compact vision-language model")
    print("5. Qwen-VL - Versatile vision-language model")
    print("6. CogVLM - Vision-language model")
    print("7. Mobile-tuned LLaVA - Mobile-optimized LLaVA")
    print("="*80)
    
    # Check for sample images
    if not create_sample_images():
        return
    
    # Initialize tester
    tester = VisionModelTester()
    
    # Add all models
    models = [
        ("YOLOv9", YOLOv9Model),
        ("MobileSAM", MobileSAMModel),
        ("LLaVA-1.5", LLaVA15Model),
        ("MiniGPT-4", MiniGPT4Model),
        ("Qwen-VL", QwenVLModel),
        ("CogVLM", CogVLMModel),
        ("Mobile-tuned LLaVA", MobileLLaVAModel)
    ]
    
    print(f"\nLoading {len(models)} models...")
    
    for name, model_class in models:
        try:
            print(f"Loading {name}...")
            model = model_class()
            tester.add_model(model)
            print(f"✓ {name} loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
            print(f"  Skipping {name} for this test run")
    
    if not tester.models:
        print("\nNo models could be loaded. Please check your dependencies.")
        return
    
    print(f"\nRunning tests with {len(tester.models)} models...")
    
    # Run tests
    results = tester.run_tests()
    
    # Save results
    tester.save_results(results)
    
    # Print summary
    tester.print_summary(results)
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)
    print("Results saved in the 'results' directory")
    print("Check individual JSON files for detailed results")

def run_single_model(model_name: str):
    """Run tests on a single model"""
    print(f"Running tests for {model_name} only...")
    
    # Check for sample images
    if not create_sample_images():
        return
    
    # Initialize tester
    tester = VisionModelTester()
    
    # Model mapping
    model_classes = {
        "yolov9": YOLOv9Model,
        "mobilesam": MobileSAMModel,
        "llava": LLaVA15Model,
        "minigpt4": MiniGPT4Model,
        "qwen-vl": QwenVLModel,
        "cogvlm": CogVLMModel,
        "mobile-llava": MobileLLaVAModel
    }
    
    if model_name.lower() not in model_classes:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(model_classes.keys())}")
        return
    
    try:
        model_class = model_classes[model_name.lower()]
        model = model_class()
        tester.add_model(model)
        
        print(f"Running tests with {model_name}...")
        results = tester.run_tests()
        tester.save_results(results)
        tester.print_summary(results)
        
    except Exception as e:
        print(f"Error running {model_name}: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test 7 Free Vision Models")
    parser.add_argument(
        "--model", 
        type=str, 
        help="Run tests for a specific model only",
        choices=["yolov9", "mobilesam", "llava", "minigpt4", "qwen-vl", "cogvlm", "mobile-llava"]
    )
    parser.add_argument(
        "--list-models", 
        action="store_true", 
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        print("1. yolov9 - YOLOv9 object detection")
        print("2. mobilesam - MobileSAM segmentation")
        print("3. llava - LLaVA-1.5 vision-language")
        print("4. minigpt4 - MiniGPT-4 vision-language")
        print("5. qwen-vl - Qwen-VL vision-language")
        print("6. cogvlm - CogVLM vision-language")
        print("7. mobile-llava - Mobile-tuned LLaVA")
        return
    
    if args.model:
        run_single_model(args.model)
    else:
        run_all_tests()

if __name__ == "__main__":
    main()
