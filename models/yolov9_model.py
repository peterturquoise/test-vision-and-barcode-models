"""
YOLOv9 Model Implementation
Advanced object detection with text and barcode detection capabilities
"""

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import pytesseract
from pyzbar import pyzbar
from vision_test_framework import VisionModel, TextDetection, BarcodeDetection

class YOLOv9Model(VisionModel):
    """YOLOv9 implementation for object detection, text extraction, and barcode detection"""
    
    def __init__(self):
        super().__init__("YOLOv9")
        self.yolo_model = None
        self.text_model = None
        
    def load_model(self):
        """Load YOLOv9 models"""
        try:
            # Load YOLOv9 for general object detection
            self.yolo_model = YOLO('yolov9c.pt')  # or yolov9e.pt for better accuracy
            
            # Load specialized text detection model if available
            # You can fine-tune YOLOv9 on text datasets or use a pre-trained text detection model
            try:
                self.text_model = YOLO('yolov9-text.pt')  # Custom text detection model
            except:
                print("Warning: Custom text detection model not found, using general YOLOv9")
                self.text_model = self.yolo_model
                
        except Exception as e:
            print(f"Error loading YOLOv9 model: {e}")
            raise
    
    def describe_image(self, image: Image.Image) -> str:
        """Generate detailed description focused on package analysis"""
        if self.yolo_model is None:
            self.load_model()
        
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        description_parts = []
        
        # Focus on package-relevant analysis
        description_parts.append("Package/Label Analysis:")
        
        # Check for rectangular shapes (packages)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_objects = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Rectangular shape
                    rectangular_objects += 1
        
        if rectangular_objects > 0:
            description_parts.append(f"Detected {rectangular_objects} rectangular package-like objects")
        
        # Check for text regions
        try:
            import pytesseract
            # Use OCR to detect text regions
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            text_regions = 0
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30 and data['text'][i].strip():
                    text_regions += 1
            
            if text_regions > 0:
                description_parts.append(f"Detected {text_regions} text regions")
        except:
            pass
        
        # Check for barcodes
        try:
            from pyzbar import pyzbar
            barcodes = pyzbar.decode(cv_image)
            if barcodes:
                description_parts.append(f"Detected {len(barcodes)} barcodes")
        except:
            pass
        
        # Add image dimensions
        width, height = image.size
        description_parts.append(f"Image dimensions: {width}x{height} pixels")
        
        return ". ".join(description_parts)
    
    def extract_text(self, image: Image.Image) -> list[TextDetection]:
        """Extract text using OCR optimized for package labels"""
        text_detections = []
        
        try:
            # Use Tesseract with package-optimized settings
            import pytesseract
            
            # Method 1: Use Tesseract with detailed output for better text detection
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                # Filter out low confidence and empty text
                if conf > 30 and text and len(text) > 1:
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    text_detections.append(TextDetection(
                        text=text,
                        bbox=[x, y, w, h],
                        confidence=conf / 100.0  # Convert to 0-1 scale
                    ))
            
            # Method 2: Try different OCR modes for better results
            if len(text_detections) < 3:  # If we didn't find much text
                # Try different page segmentation modes
                for psm in [6, 7, 8, 13]:  # Different OCR modes
                    try:
                        text = pytesseract.image_to_string(image, config=f'--psm {psm}')
                        if text.strip() and len(text.strip()) > 10:
                            # Split into lines and add as separate detections
                            lines = text.strip().split('\n')
                            for j, line in enumerate(lines):
                                if line.strip():
                                    text_detections.append(TextDetection(
                                        text=line.strip(),
                                        bbox=[0, j*20, len(line)*10, 20],  # Approximate position
                                        confidence=0.6
                                    ))
                            break  # Use first successful mode
                    except:
                        continue
            
            # Method 3: Try EasyOCR as fallback
            if len(text_detections) < 2:
                try:
                    import easyocr
                    reader = easyocr.Reader(['en'])
                    results = reader.readtext(np.array(image))
                    
                    for (bbox, text, confidence) in results:
                        if confidence > 0.3 and text.strip():  # Lower threshold for packages
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            x1, x2 = min(x_coords), max(x_coords)
                            y1, y2 = min(y_coords), max(y_coords)
                            
                            text_detections.append(TextDetection(
                                text=text.strip(),
                                bbox=[int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                confidence=confidence
                            ))
                except ImportError:
                    pass
        
        except Exception as e:
            print(f"Error in text extraction: {e}")
        
        return text_detections
    
    def detect_barcodes(self, image: Image.Image) -> list[BarcodeDetection]:
        """Detect and decode 1D and 2D barcodes optimized for packages"""
        barcode_detections = []
        
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Method 1: Direct barcode detection using pyzbar
            barcodes = pyzbar.decode(cv_image)
            
            for barcode in barcodes:
                # Extract bounding box
                x, y, w, h = barcode.rect
                
                # Decode barcode data
                barcode_data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                
                barcode_detections.append(BarcodeDetection(
                    bbox=[x, y, w, h],
                    value=barcode_data,
                    barcode_type=barcode_type,
                    confidence=1.0  # pyzbar doesn't provide confidence scores
                ))
            
            # Method 2: Try different image preprocessing for better detection
            if not barcode_detections:
                # Convert to grayscale
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                
                # Try different preprocessing techniques
                preprocessing_methods = [
                    gray,  # Original grayscale
                    cv2.GaussianBlur(gray, (3, 3), 0),  # Blurred
                    cv2.medianBlur(gray, 3),  # Median blur
                    cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Binary
                ]
                
                for processed_img in preprocessing_methods:
                    processed_barcodes = pyzbar.decode(processed_img)
                    for barcode in processed_barcodes:
                        x, y, w, h = barcode.rect
                        barcode_data = barcode.data.decode('utf-8')
                        barcode_type = barcode.type
                        
                        barcode_detections.append(BarcodeDetection(
                            bbox=[x, y, w, h],
                            value=barcode_data,
                            barcode_type=barcode_type,
                            confidence=0.9
                        ))
            
            # Method 3: Try different scales
            if not barcode_detections:
                scales = [0.5, 0.8, 1.2, 1.5, 2.0]
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
            
            # Method 4: Try rotating the image
            if not barcode_detections:
                angles = [90, 180, 270]  # Try different rotations
                for angle in angles:
                    # Rotate image
                    height, width = cv_image.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated_image = cv2.warpAffine(cv_image, rotation_matrix, (width, height))
                    
                    rotated_barcodes = pyzbar.decode(rotated_image)
                    for barcode in rotated_barcodes:
                        x, y, w, h = barcode.rect
                        barcode_data = barcode.data.decode('utf-8')
                        barcode_type = barcode.type
                        
                        barcode_detections.append(BarcodeDetection(
                            bbox=[x, y, w, h],
                            value=barcode_data,
                            barcode_type=barcode_type,
                            confidence=0.7
                        ))
        
        except Exception as e:
            print(f"Error in barcode detection: {e}")
        
        return barcode_detections

# Test function
def test_yolov9():
    """Test YOLOv9 model with sample images"""
    from vision_test_framework import VisionModelTester
    
    tester = VisionModelTester()
    tester.add_model(YOLOv9Model())
    
    results = tester.run_tests()
    tester.save_results(results)
    tester.print_summary(results)

if __name__ == "__main__":
    test_yolov9()
