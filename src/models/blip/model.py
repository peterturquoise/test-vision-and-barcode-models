"""
BLIP Model Implementation
Better vision-language model than current MiniGPT-4 implementation
"""

import cv2
import numpy as np
from PIL import Image
import torch
from pyzbar import pyzbar
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src" / "utils"))
from vision_test_framework import VisionModel, TextDetection, BarcodeDetection

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
    print("BLIP available")
except ImportError:
    BLIP_AVAILABLE = False
    print("Warning: BLIP not available. Install transformers package.")

class BLIPModel(VisionModel):
    """BLIP implementation for vision-language tasks"""
    
    def __init__(self):
        super().__init__("BLIP")
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load BLIP model"""
        if not BLIP_AVAILABLE:
            raise ImportError("BLIP not available. Install transformers package.")
        
        try:
            # Use BLIP model
            model_name = "Salesforce/blip-image-captioning-large"
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32  # Use float32 to avoid Half precision issues
            )
            self.model.to(self.device)
            
            print(f"BLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading BLIP model: {e}")
            raise
    
    def describe_image(self, image: Image.Image) -> str:
        """Generate detailed description using BLIP"""
        if self.model is None:
            self.load_model()
        
        try:
            # Ask BLIP to describe what it sees
            prompt = "This is a package or shipping label image. Describe what you see: package details, sender/recipient information, addresses, tracking numbers, postal codes, and any shipping labels or markings. Focus on text content and package characteristics."
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=200, num_beams=5)
            
            description = self.processor.decode(out[0], skip_special_tokens=True)
            return description if description else "Unable to generate description"
            
        except Exception as e:
            print(f"Error in image description: {e}")
            return f"Error generating description: {e}"
    
    def extract_text(self, image: Image.Image) -> list[TextDetection]:
        """Extract text using BLIP's capabilities"""
        text_detections = []
        
        if self.model is None:
            self.load_model()
        
        try:
            # Ask BLIP to extract all text from the image
            prompt = "This is a package/shipping label. Extract all text you can see including: addresses, names, tracking numbers, postal codes, package dimensions, weight, sender/recipient info, and any other text on labels or the package itself."
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=256, num_beams=5)
            
            response = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Simple parsing of the response to extract text
            if response:
                # Split by common separators and clean up
                text_pieces = [t.strip() for t in response.split('\n') if t.strip()]
                for text in text_pieces:
                    # BLIP doesn't provide bounding boxes directly, so approximate
                    text_detections.append(TextDetection(
                        text=text,
                        bbox=[0, 0, image.width, image.height],  # Placeholder bbox
                        confidence=0.7  # Placeholder confidence
                    ))
            
        except Exception as e:
            print(f"Error in text extraction: {e}")
        
        return text_detections
    
    def detect_barcodes(self, image: Image.Image) -> list[BarcodeDetection]:
        """Detect and decode barcodes using traditional methods, with BLIP for context"""
        barcode_detections = []
        
        try:
            # First, ask BLIP to identify barcodes
            if self.model is not None:
                try:
                    prompt = "Do you see any barcodes, QR codes, or similar machine-readable codes in this package image? If yes, describe their location and type."
                    inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        out = self.model.generate(**inputs, max_length=100, num_beams=5)
                    
                    blip_response = self.processor.decode(out[0], skip_special_tokens=True)
                    
                    # Check if BLIP detected barcodes
                    if any(keyword in blip_response.lower() for keyword in ['barcode', 'qr code', 'code', 'machine readable']):
                        print(f"BLIP detected potential barcodes: {blip_response}")
                
                except Exception as e:
                    print(f"BLIP barcode detection failed: {e}")
            
            # Use traditional barcode detection (Pyzbar)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            barcodes = pyzbar.decode(cv_image)
            
            for barcode in barcodes:
                x, y, w, h = barcode.rect
                barcode_data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                
                barcode_detections.append(BarcodeDetection(
                    bbox=[x, y, w, h],
                    value=barcode_data,
                    barcode_type=barcode_type,
                    confidence=1.0
                ))
            
        except Exception as e:
            print(f"Error in barcode detection: {e}")
        
        return barcode_detections

def test_blip():
    """Test BLIP model"""
    model = BLIPModel()
    return model

if __name__ == "__main__":
    # Quick test
    model = BLIPModel()
    model.load_model()
    print("✅ BLIP model loaded successfully!")






