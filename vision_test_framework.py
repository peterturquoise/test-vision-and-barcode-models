"""
Vision Models Test Suite
Tests 7 free vision models for image description, text extraction, and barcode detection
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
from PIL import Image
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class TextDetection:
    """Represents detected text with bounding box and content"""
    text: str
    bbox: List[int]  # [x, y, width, height]
    confidence: float

@dataclass
class BarcodeDetection:
    """Represents detected barcode with bounding box and decoded value"""
    bbox: List[int]  # [x, y, width, height]
    value: Optional[str]
    barcode_type: Optional[str]
    confidence: float

@dataclass
class ModelResult:
    """Complete result from a vision model"""
    model_name: str
    image_path: str
    description: str
    text_detections: List[TextDetection]
    barcode_detections: List[BarcodeDetection]
    processing_time: float
    error: Optional[str] = None

class VisionModel(ABC):
    """Abstract base class for vision models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model"""
        pass
    
    @abstractmethod
    def describe_image(self, image: Image.Image) -> str:
        """Generate detailed description of the image"""
        pass
    
    @abstractmethod
    def extract_text(self, image: Image.Image) -> List[TextDetection]:
        """Extract all text from the image with bounding boxes"""
        pass
    
    @abstractmethod
    def detect_barcodes(self, image: Image.Image) -> List[BarcodeDetection]:
        """Detect barcodes and decode their values"""
        pass
    
    def process_image(self, image_path: str) -> ModelResult:
        """Process a single image and return complete results"""
        start_time = time.time()
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Process image
            description = self.describe_image(image)
            text_detections = self.extract_text(image)
            barcode_detections = self.detect_barcodes(image)
            
            processing_time = time.time() - start_time
            
            return ModelResult(
                model_name=self.model_name,
                image_path=image_path,
                description=description,
                text_detections=text_detections,
                barcode_detections=barcode_detections,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ModelResult(
                model_name=self.model_name,
                image_path=image_path,
                description="",
                text_detections=[],
                barcode_detections=[],
                processing_time=processing_time,
                error=str(e)
            )

class VisionModelTester:
    """Main test runner for all vision models"""
    
    def __init__(self, samples_dir: str = "samples", output_dir: str = "results"):
        self.samples_dir = Path(samples_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.models: List[VisionModel] = []
        
    def add_model(self, model: VisionModel):
        """Add a model to the test suite"""
        self.models.append(model)
    
    def get_image_files(self) -> List[Path]:
        """Get all image files from samples directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        # Search recursively in subdirectories
        for ext in image_extensions:
            image_files.extend(self.samples_dir.glob(f"**/*{ext}"))
            image_files.extend(self.samples_dir.glob(f"**/*{ext.upper()}"))
        
        return sorted(image_files)
    
    def run_tests(self) -> Dict[str, List[ModelResult]]:
        """Run tests on all images with all models"""
        image_files = self.get_image_files()
        
        if not image_files:
            print(f"No image files found in {self.samples_dir}")
            return {}
        
        print(f"Found {len(image_files)} images to process")
        print(f"Testing with {len(self.models)} models")
        
        results = {}
        
        for model in self.models:
            print(f"\nTesting {model.model_name}...")
            model_results = []
            
            for image_path in image_files:
                print(f"  Processing {image_path.name}...")
                result = model.process_image(str(image_path))
                model_results.append(result)
                
                if result.error:
                    print(f"    Error: {result.error}")
                else:
                    print(f"    Completed in {result.processing_time:.2f}s")
            
            results[model.model_name] = model_results
        
        return results
    
    def save_results(self, results: Dict[str, List[ModelResult]]):
        """Save results to JSON files"""
        for model_name, model_results in results.items():
            # Convert results to serializable format
            serializable_results = []
            for result in model_results:
                serializable_results.append({
                    'model_name': result.model_name,
                    'image_path': result.image_path,
                    'description': result.description,
                    'text_detections': [
                        {
                            'text': td.text,
                            'bbox': td.bbox,
                            'confidence': td.confidence
                        } for td in result.text_detections
                    ],
                    'barcode_detections': [
                        {
                            'bbox': bd.bbox,
                            'value': bd.value,
                            'barcode_type': bd.barcode_type,
                            'confidence': bd.confidence
                        } for bd in result.barcode_detections
                    ],
                    'processing_time': result.processing_time,
                    'error': result.error
                })
            
            # Save to file
            output_file = self.output_dir / f"{model_name}_results.json"
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"Results saved to {output_file}")
    
    def print_summary(self, results: Dict[str, List[ModelResult]]):
        """Print a summary of test results"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        for model_name, model_results in results.items():
            print(f"\n{model_name}:")
            print(f"  Total images processed: {len(model_results)}")
            
            successful = [r for r in model_results if r.error is None]
            failed = [r for r in model_results if r.error is not None]
            
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {len(failed)}")
            
            if successful:
                avg_time = sum(r.processing_time for r in successful) / len(successful)
                print(f"  Average processing time: {avg_time:.2f}s")
                
                total_text = sum(len(r.text_detections) for r in successful)
                total_barcodes = sum(len(r.barcode_detections) for r in successful)
                print(f"  Total text detections: {total_text}")
                print(f"  Total barcode detections: {total_barcodes}")
            
            if failed:
                print(f"  Errors:")
                for result in failed:
                    print(f"    {Path(result.image_path).name}: {result.error}")

if __name__ == "__main__":
    # This will be populated with actual model implementations
    tester = VisionModelTester()
    
    # Add models here once implemented
    # tester.add_model(YOLOv9Model())
    # tester.add_model(MobileSAMModel())
    # etc.
    
    print("Vision Models Test Suite")
    print("Add your test images to the 'samples' folder")
    print("Run individual model tests or use the main runner")
