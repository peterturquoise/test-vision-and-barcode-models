"""
LLaVA-1.5 Model Implementation
Large Language and Vision Assistant for image understanding and text extraction
"""

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from pyzbar import pyzbar
from vision_test_framework import VisionModel, TextDetection, BarcodeDetection

class LLaVA15Model(VisionModel):
    """LLaVA-1.5 implementation for vision-language tasks"""
    
    def __init__(self):
        super().__init__("LLaVA-1.5")
        self.processor = None
        self.model = None
        
    def load_model(self):
        """Load LLaVA-1.5 model"""
        try:
            # Load LLaVA-1.5 model
            model_name = "llava-hf/llava-v1.6-mistral-7b-hf"  # Updated to working model
            
            self.processor = LlavaNextProcessor.from_pretrained(model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            
            print(f"LLaVA-1.5 model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading LLaVA-1.5 model: {e}")
            # Try smaller model as fallback
            try:
                model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
                self.processor = LlavaNextProcessor.from_pretrained(model_name)
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                self.model.to(self.device)
            except Exception as e2:
                print(f"Fallback model loading failed: {e2}")
                raise
    
    def describe_image(self, image: Image.Image) -> str:
        """Generate detailed description using LLaVA-1.5"""
        if self.model is None:
            self.load_model()
        
        try:
            # Prepare conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This is a package or shipping label image. Describe what you see: package details, sender/recipient information, addresses, tracking numbers, postal codes, and any shipping labels or markings. Focus on text content and package characteristics."},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            # Process conversation
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.2,
                    use_cache=True
                )
            
            # Decode response
            generated_text = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text
            
            return response if response else "Unable to generate description"
            
        except Exception as e:
            print(f"Error in image description: {e}")
            return f"Error generating description: {e}"
    
    def extract_text(self, image: Image.Image) -> list[TextDetection]:
        """Extract text using LLaVA-1.5's OCR capabilities"""
        text_detections = []
        
        if self.model is None:
            self.load_model()
        
        try:
            # Ask LLaVA to extract all text from the image
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This is a package/shipping label. Extract all text you can see including: addresses, names, tracking numbers, postal codes, package dimensions, weight, sender/recipient info, and any other text on labels or the package itself."},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.1,
                    use_cache=True
                )
            
            generated_text = self.processor.decode(output[0], skip_special_tokens=True)
            
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text
            
            # Parse the response to extract individual text pieces
            if response and response != "Unable to generate description":
                # Split by common separators and clean up
                text_pieces = []
                for line in response.split('\n'):
                    line = line.strip()
                    if line and not line.startswith(('I can see', 'The text', 'There is')):
                        # Remove common prefixes
                        for prefix in ['- ', '* ', '• ', '1. ', '2. ', '3. ', '4. ', '5. ']:
                            if line.startswith(prefix):
                                line = line[len(prefix):].strip()
                                break
                        if line:
                            text_pieces.append(line)
                
                # Create text detections (without precise bounding boxes)
                for i, text in enumerate(text_pieces):
                    if len(text) > 1:  # Filter out single characters
                        text_detections.append(TextDetection(
                            text=text,
                            bbox=[0, i*20, len(text)*10, 20],  # Approximate bounding box
                            confidence=0.8
                        ))
            
            # Fallback: Use traditional OCR if LLaVA didn't extract much text
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
        """Detect barcodes using LLaVA-1.5 + traditional barcode detection"""
        barcode_detections = []
        
        try:
            # First, ask LLaVA to identify barcodes
            if self.model is not None:
                try:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Do you see any barcodes, QR codes, or similar machine-readable codes in this image? If yes, describe their location."},
                                {"type": "image", "image": image}
                            ]
                        }
                    ]
                    
                    prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            temperature=0.1,
                            use_cache=True
                        )
                    
                    generated_text = self.processor.decode(output[0], skip_special_tokens=True)
                    
                    if "ASSISTANT:" in generated_text:
                        response = generated_text.split("ASSISTANT:")[-1].strip()
                    else:
                        response = generated_text
                    
                    # Check if LLaVA detected barcodes
                    if any(keyword in response.lower() for keyword in ['barcode', 'qr code', 'code', 'machine readable']):
                        print(f"LLaVA detected potential barcodes: {response}")
                
                except Exception as e:
                    print(f"LLaVA barcode detection failed: {e}")
            
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
                # Try different image scales
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
def test_llava15():
    """Test LLaVA-1.5 model with sample images"""
    from vision_test_framework import VisionModelTester
    
    tester = VisionModelTester()
    tester.add_model(LLaVA15Model())
    
    results = tester.run_tests()
    tester.save_results(results)
    tester.print_summary(results)

if __name__ == "__main__":
    test_llava15()
