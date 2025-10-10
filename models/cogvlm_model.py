"""
CogVLM Model Implementation
Vision-language model for image understanding and text extraction
"""

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyzbar import pyzbar
from vision_test_framework import VisionModel, TextDetection, BarcodeDetection

class CogVLMModel(VisionModel):
    """CogVLM implementation for vision-language tasks"""
    
    def __init__(self):
        super().__init__("CogVLM")
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load CogVLM model"""
        try:
            # Load CogVLM model
            model_name = "THUDM/cogvlm-chinese-hf"  # or cogvlm-english-hf for English
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"CogVLM model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading CogVLM model: {e}")
            # Try alternative model
            try:
                model_name = "THUDM/cogvlm-english-hf"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                self.model.to(self.device)
            except Exception as e2:
                print(f"Fallback model loading failed: {e2}")
                raise
    
    def describe_image(self, image: Image.Image) -> str:
        """Generate detailed description using CogVLM"""
        if self.model is None:
            self.load_model()
        
        try:
            # Prepare query
            query = "Describe this image in detail. What objects, people, text, and activities do you see?"
            
            # Process image and text
            inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
            inputs.update({'images': [image]})
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=False,
                    use_cache=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the input query)
            if query in response:
                response = response.split(query)[-1].strip()
            
            return response if response else "Unable to generate description"
            
        except Exception as e:
            print(f"Error in image description: {e}")
            return f"Error generating description: {e}"
    
    def extract_text(self, image: Image.Image) -> list[TextDetection]:
        """Extract text using CogVLM's OCR capabilities"""
        text_detections = []
        
        if self.model is None:
            self.load_model()
        
        try:
            # Ask CogVLM to extract all text from the image
            query = "Extract all the text you can see in this image. List each piece of text separately."
            
            inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
            inputs.update({'images': [image]})
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                    use_cache=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if query in response:
                response = response.split(query)[-1].strip()
            
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
            
            # Fallback: Use traditional OCR if CogVLM didn't extract much text
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
        """Detect barcodes using CogVLM + traditional barcode detection"""
        barcode_detections = []
        
        try:
            # First, ask CogVLM to identify barcodes
            if self.model is not None:
                try:
                    query = "Do you see any barcodes, QR codes, or similar machine-readable codes in this image? If yes, describe their location."
                    
                    inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
                    inputs.update({'images': [image]})
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            temperature=0.1,
                            do_sample=False,
                            use_cache=True
                        )
                    
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract only the generated part
                    if query in response:
                        response = response.split(query)[-1].strip()
                    
                    # Check if CogVLM detected barcodes
                    if any(keyword in response.lower() for keyword in ['barcode', 'qr code', 'code', 'machine readable']):
                        print(f"CogVLM detected potential barcodes: {response}")
                
                except Exception as e:
                    print(f"CogVLM barcode detection failed: {e}")
            
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
def test_cogvlm():
    """Test CogVLM model with sample images"""
    from vision_test_framework import VisionModelTester
    
    tester = VisionModelTester()
    tester.add_model(CogVLMModel())
    
    results = tester.run_tests()
    tester.save_results(results)
    tester.print_summary(results)

if __name__ == "__main__":
    test_cogvlm()
