"""
Mobile-tuned LLaVA Model Implementation
Mobile-optimized version of LLaVA for efficient vision-language tasks
"""

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from pyzbar import pyzbar
from vision_test_framework import VisionModel, TextDetection, BarcodeDetection

class MobileLLaVAModel(VisionModel):
    """Mobile-tuned LLaVA implementation for efficient vision-language tasks"""
    
    def __init__(self):
        super().__init__("Mobile-tuned LLaVA")
        self.processor = None
        self.model = None
        
    def load_model(self):
        """Load Mobile-tuned LLaVA model"""
        try:
            # Load mobile-optimized LLaVA model
            # This could be a quantized or smaller version of LLaVA
            model_name = "llava-hf/llava-v1.5-7b-hf"  # Using standard model as mobile version
            
            self.processor = LlavaNextProcessor.from_pretrained(model_name)
            
            # Load with optimizations for mobile deployment
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision for memory efficiency
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            
            # Apply additional optimizations
            if hasattr(self.model, 'half'):
                self.model = self.model.half()
            
            print(f"Mobile-tuned LLaVA model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading Mobile-tuned LLaVA model: {e}")
            # Try with even more aggressive optimizations
            try:
                model_name = "llava-hf/llava-v1.5-7b-hf"
                self.processor = LlavaNextProcessor.from_pretrained(model_name)
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                self.model.to(self.device)
                
                # Apply quantization if available
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    self.model = LlavaNextForConditionalGeneration.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto"
                    )
                except ImportError:
                    pass
                    
            except Exception as e2:
                print(f"Fallback model loading failed: {e2}")
                raise
    
    def describe_image(self, image: Image.Image) -> str:
        """Generate concise description using Mobile-tuned LLaVA"""
        if self.model is None:
            self.load_model()
        
        try:
            # Prepare conversation with shorter prompt for efficiency
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Briefly describe this image."},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            # Process conversation
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            # Generate response with shorter max tokens for efficiency
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Shorter response for mobile efficiency
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
        """Extract text using Mobile-tuned LLaVA's OCR capabilities"""
        text_detections = []
        
        if self.model is None:
            self.load_model()
        
        try:
            # Ask Mobile-tuned LLaVA to extract text (shorter prompt for efficiency)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "List all text in this image."},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=128,  # Shorter response for mobile efficiency
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
                        if line and len(line) > 1:
                            text_pieces.append(line)
                
                # Create text detections (without precise bounding boxes)
                for i, text in enumerate(text_pieces):
                    text_detections.append(TextDetection(
                        text=text,
                        bbox=[0, i*20, len(text)*10, 20],  # Approximate bounding box
                        confidence=0.7  # Slightly lower confidence for mobile model
                    ))
            
            # Fallback: Use traditional OCR if Mobile-tuned LLaVA didn't extract much text
            if len(text_detections) < 2:
                try:
                    import pytesseract
                    ocr_text = pytesseract.image_to_string(image)
                    if ocr_text.strip():
                        text_detections.append(TextDetection(
                            text=ocr_text.strip(),
                            bbox=[0, 0, image.width, image.height],
                            confidence=0.5
                        ))
                except ImportError:
                    pass
        
        except Exception as e:
            print(f"Error in text extraction: {e}")
        
        return text_detections
    
    def detect_barcodes(self, image: Image.Image) -> list[BarcodeDetection]:
        """Detect barcodes using Mobile-tuned LLaVA + traditional barcode detection"""
        barcode_detections = []
        
        try:
            # First, ask Mobile-tuned LLaVA to identify barcodes (short prompt)
            if self.model is not None:
                try:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Any barcodes or QR codes?"},
                                {"type": "image", "image": image}
                            ]
                        }
                    ]
                    
                    prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=64,  # Very short response for mobile efficiency
                            do_sample=False,
                            temperature=0.1,
                            use_cache=True
                        )
                    
                    generated_text = self.processor.decode(output[0], skip_special_tokens=True)
                    
                    if "ASSISTANT:" in generated_text:
                        response = generated_text.split("ASSISTANT:")[-1].strip()
                    else:
                        response = generated_text
                    
                    # Check if Mobile-tuned LLaVA detected barcodes
                    if any(keyword in response.lower() for keyword in ['barcode', 'qr code', 'code', 'yes']):
                        print(f"Mobile-tuned LLaVA detected potential barcodes: {response}")
                
                except Exception as e:
                    print(f"Mobile-tuned LLaVA barcode detection failed: {e}")
            
            # Use traditional barcode detection (more reliable for mobile)
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
            
            # If no barcodes found, try on different scales (limited for mobile efficiency)
            if not barcode_detections:
                scales = [0.8, 1.2]  # Fewer scales for mobile efficiency
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
                                confidence=0.8
                            ))
        
        except Exception as e:
            print(f"Error in barcode detection: {e}")
        
        return barcode_detections

# Test function
def test_mobile_llava():
    """Test Mobile-tuned LLaVA model with sample images"""
    from vision_test_framework import VisionModelTester
    
    tester = VisionModelTester()
    tester.add_model(MobileLLaVAModel())
    
    results = tester.run_tests()
    tester.save_results(results)
    tester.print_summary(results)

if __name__ == "__main__":
    test_mobile_llava()
