"""
MiniGPT-4 Model Implementation
Compact vision-language model for image understanding and text extraction
"""

import cv2
import numpy as np
from PIL import Image
import torch
from pyzbar import pyzbar
from vision_test_framework import VisionModel, TextDetection, BarcodeDetection

try:
    from minigpt4.models import MiniGPT4
    from minigpt4.conversation import CONV_VISION
    from minigpt4.utils import disable_torch_init
    MINIGPT4_AVAILABLE = True
except ImportError:
    MINIGPT4_AVAILABLE = False
    print("Warning: MiniGPT-4 not available. Install minigpt4 package.")

class MiniGPT4Model(VisionModel):
    """MiniGPT-4 implementation for vision-language tasks"""
    
    def __init__(self):
        super().__init__("MiniGPT-4")
        self.model = None
        self.conv_mode = None
        
    def load_model(self):
        """Load MiniGPT-4 model"""
        if not MINIGPT4_AVAILABLE:
            raise ImportError("MiniGPT-4 not available. Install minigpt4 package.")
        
        try:
            disable_torch_init()
            
            # Load MiniGPT-4 model
            model_path = "minigpt4-7b"  # or minigpt4-13b for better performance
            
            self.model = MiniGPT4.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            self.conv_mode = CONV_VISION
            
            print(f"MiniGPT-4 model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading MiniGPT-4 model: {e}")
            # Try alternative loading method
            try:
                self.model = MiniGPT4.from_pretrained(
                    "minigpt4-7b",
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                self.model.to(self.device)
            except Exception as e2:
                print(f"Fallback model loading failed: {e2}")
                raise
    
    def describe_image(self, image: Image.Image) -> str:
        """Generate detailed description using MiniGPT-4"""
        if self.model is None:
            self.load_model()
        
        try:
            # Prepare conversation
            conv = self.conv_mode.copy()
            conv.append_message(conv.roles[0], "Describe this image in detail. What objects, people, text, and activities do you see?")
            conv.append_message(conv.roles[1], None)
            
            # Process image and generate response
            with torch.no_grad():
                response = self.model.generate(
                    image,
                    conv,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=False
                )
            
            # Extract response
            if response:
                # Remove the conversation history and get only the response
                response_text = response.split("###")[-1].strip()
                return response_text if response_text else "Unable to generate description"
            else:
                return "No response generated"
            
        except Exception as e:
            print(f"Error in image description: {e}")
            return f"Error generating description: {e}"
    
    def extract_text(self, image: Image.Image) -> list[TextDetection]:
        """Extract text using MiniGPT-4's OCR capabilities"""
        text_detections = []
        
        if self.model is None:
            self.load_model()
        
        try:
            # Ask MiniGPT-4 to extract all text from the image
            conv = self.conv_mode.copy()
            conv.append_message(conv.roles[0], "Extract all the text you can see in this image. List each piece of text separately.")
            conv.append_message(conv.roles[1], None)
            
            with torch.no_grad():
                response = self.model.generate(
                    image,
                    conv,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Parse the response to extract individual text pieces
            if response:
                response_text = response.split("###")[-1].strip()
                
                if response_text and response_text != "Unable to generate description":
                    # Split by common separators and clean up
                    text_pieces = []
                    for line in response_text.split('\n'):
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
            
            # Fallback: Use traditional OCR if MiniGPT-4 didn't extract much text
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
        """Detect barcodes using MiniGPT-4 + traditional barcode detection"""
        barcode_detections = []
        
        try:
            # First, ask MiniGPT-4 to identify barcodes
            if self.model is not None:
                try:
                    conv = self.conv_mode.copy()
                    conv.append_message(conv.roles[0], "Do you see any barcodes, QR codes, or similar machine-readable codes in this image? If yes, describe their location.")
                    conv.append_message(conv.roles[1], None)
                    
                    with torch.no_grad():
                        response = self.model.generate(
                            image,
                            conv,
                            max_new_tokens=128,
                            temperature=0.1,
                            do_sample=False
                        )
                    
                    if response:
                        response_text = response.split("###")[-1].strip()
                        # Check if MiniGPT-4 detected barcodes
                        if any(keyword in response_text.lower() for keyword in ['barcode', 'qr code', 'code', 'machine readable']):
                            print(f"MiniGPT-4 detected potential barcodes: {response_text}")
                
                except Exception as e:
                    print(f"MiniGPT-4 barcode detection failed: {e}")
            
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
def test_minigpt4():
    """Test MiniGPT-4 model with sample images"""
    from vision_test_framework import VisionModelTester
    
    tester = VisionModelTester()
    tester.add_model(MiniGPT4Model())
    
    results = tester.run_tests()
    tester.save_results(results)
    tester.print_summary(results)

if __name__ == "__main__":
    test_minigpt4()
