"""
Qwen-VL Model Implementation
Versatile vision-language model for image understanding and text extraction
"""

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from pyzbar import pyzbar
from vision_test_framework import VisionModel, TextDetection, BarcodeDetection

class QwenVLModel(VisionModel):
    """Qwen-VL implementation for vision-language tasks"""
    
    def __init__(self):
        super().__init__("Qwen-VL")
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load Qwen-VL model"""
        try:
            # Load Qwen-VL model
            model_name = "Qwen/Qwen2-VL-2B-Instruct"  # or Qwen2-VL-7B-Instruct for better performance
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"Qwen-VL model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading Qwen-VL model: {e}")
            # Try smaller model as fallback
            try:
                model_name = "Qwen/Qwen2-VL-2B-Instruct"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                self.model.to(self.device)
            except Exception as e2:
                print(f"Fallback model loading failed: {e2}")
                raise
    
    def describe_image(self, image: Image.Image) -> str:
        """Generate detailed description using Qwen-VL"""
        if self.model is None:
            self.load_model()
        
        try:
            # Prepare query
            query = self.tokenizer.from_list_format([
                {'image': image},
                {'text': 'Describe this image in detail. What objects, people, text, and activities do you see?'}
            ])
            
            # Generate response
            with torch.no_grad():
                response, history = self.model.chat(
                    self.tokenizer,
                    query=query,
                    history=None,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=False
                )
            
            return response if response else "Unable to generate description"
            
        except Exception as e:
            print(f"Error in image description: {e}")
            return f"Error generating description: {e}"
    
    def extract_text(self, image: Image.Image) -> list[TextDetection]:
        """Extract text using Qwen-VL's OCR capabilities"""
        text_detections = []
        
        if self.model is None:
            self.load_model()
        
        try:
            # Ask Qwen-VL to extract all text from the image
            query = self.tokenizer.from_list_format([
                {'image': image},
                {'text': 'Extract all the text you can see in this image. List each piece of text separately.'}
            ])
            
            with torch.no_grad():
                response, history = self.model.chat(
                    self.tokenizer,
                    query=query,
                    history=None,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Parse the response to extract individual text pieces
            if response and response != "Unable to generate description":
                # Split by common separators and clean up
                text_pieces = []
                for line in response.split('\n'):
                    line = line.strip()
                    if line and not line.startswith(('I can see', 'The text', 'There is', 'Here')):
                        # Remove common prefixes
                        for prefix in ['- ', '* ', '• ', '1. ', '2. ', '3. ', '4. ', '5. ']:
                            if line.startswith(prefix):
                                line = line[len(prefix):].strip()
                                break
                        if line and len(line) > 1:
                            text_pieces.append(line)
                
                # Create text detections (without precise bounding boxes)
                for i, text in enumerate(text_pieces):
                    text_detections.append(TextDetection(
                        text=text,
                        bbox=[0, i*20, len(text)*10, 20],  # Approximate bounding box
                        confidence=0.8
                    ))
            
            # Fallback: Use traditional OCR if Qwen-VL didn't extract much text
            if len(text_detections) < 2:
                try:
                    import pytesseract
                    ocr_text = pytesseract.image_to_string(image)
                    if ocr_text.strip():
                        text_detections.append(TextDetection(
                            text=ocr_text.strip(),
                            bbox=[0, 0, image.width, image.height],
                            confidence=0.6
                        ))
                except ImportError:
                    pass
        
        except Exception as e:
            print(f"Error in text extraction: {e}")
        
        return text_detections
    
    def detect_barcodes(self, image: Image.Image) -> list[BarcodeDetection]:
        """Detect barcodes using Qwen-VL + traditional barcode detection"""
        barcode_detections = []
        
        try:
            # First, ask Qwen-VL to identify barcodes
            if self.model is not None:
                try:
                    query = self.tokenizer.from_list_format([
                        {'image': image},
                        {'text': 'Do you see any barcodes, QR codes, or similar machine-readable codes in this image? If yes, describe their location.'}
                    ])
                    
                    with torch.no_grad():
                        response, history = self.model.chat(
                            self.tokenizer,
                            query=query,
                            history=None,
                            max_new_tokens=128,
                            temperature=0.1,
                            do_sample=False
                        )
                    
                    # Check if Qwen-VL detected barcodes
                    if any(keyword in response.lower() for keyword in ['barcode', 'qr code', 'code', 'machine readable']):
                        print(f"Qwen-VL detected potential barcodes: {response}")
                
                except Exception as e:
                    print(f"Qwen-VL barcode detection failed: {e}")
            
            # Use traditional barcode detection
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
            
            # If no barcodes found, try on different scales
            if not barcode_detections:
                scales = [0.5, 1.5, 2.0]
                for scale in scales:
                    if scale != 1.0:
                        scaled_image = cv2.resize(cv_image, None, fx=scale, fy=scale)
                        scaled_barcodes = pyzbar.decode(scaled_image)
                        
                        for barcode in scaled_barcodes:
                            x, y, w, h = barcode.rect
                            # Scale back to original coordinates
                            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                            barcode_data = barcode.data.decode('utf-8')
                            barcode_type = barcode.type
                            
                            barcode_detections.append(BarcodeDetection(
                                bbox=[x, y, w, h],
                                value=barcode_data,
                                barcode_type=barcode_type,
                                confidence=0.9
                            ))
        
        except Exception as e:
            print(f"Error in barcode detection: {e}")
        
        return barcode_detections

# Test function
def test_qwen_vl():
    """Test Qwen-VL model with sample images"""
    from vision_test_framework import VisionModelTester
    
    tester = VisionModelTester()
    tester.add_model(QwenVLModel())
    
    results = tester.run_tests()
    tester.save_results(results)
    tester.print_summary(results)

if __name__ == "__main__":
    test_qwen_vl()
