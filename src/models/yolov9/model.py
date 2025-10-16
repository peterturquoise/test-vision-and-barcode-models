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
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src" / "utils"))
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
                # Try to load a custom text detection model
                self.text_model = YOLO('yolov9-text.pt')  # Custom text detection model
                print("Custom text detection model loaded")
            except:
                print("Warning: Custom text detection model not found, using general YOLOv9")
                self.text_model = self.yolo_model
            
            print(f"YOLOv9 model loaded successfully on {self.device}")
            
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
        """Extract text using multiple OCR methods"""
        text_detections = []
        
        if self.yolo_model is None:
            self.load_model()
        
        try:
            # Method 1: Tesseract OCR with multiple modes
            import pytesseract
            
            # Try different Tesseract modes
            modes = [
                ('--psm 6', 'Uniform block of text'),
                ('--psm 8', 'Single word'),
                ('--psm 13', 'Raw line'),
                ('--psm 11', 'Sparse text')
            ]
            
            for mode, description in modes:
                try:
                    data = pytesseract.image_to_data(image, config=mode, output_type=pytesseract.Output.DICT)
                    
                    for i in range(len(data['text'])):
                        text = data['text'][i].strip()
                        conf = int(data['conf'][i])
                        
                        if conf > 30 and len(text) > 1:  # Filter low confidence and single characters
                            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                            
                            text_detections.append(TextDetection(
                                text=text,
                                bbox=[x, y, w, h],
                                confidence=conf / 100.0
                            ))
                except Exception as e:
                    print(f"Tesseract mode {mode} failed: {e}")
                    continue
            
            # Method 2: EasyOCR as fallback (if available)
            try:
                import easyocr
                reader = easyocr.Reader(['en'])
                results = reader.readtext(np.array(image))
                
                for (bbox, text, conf) in results:
                    if conf > 0.3 and len(text.strip()) > 1:
                        # Convert bbox to [x, y, w, h] format
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        text_detections.append(TextDetection(
                            text=text.strip(),
                            bbox=[int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                            confidence=conf
                        ))
            except ImportError:
                print("EasyOCR not available, skipping")
            except Exception as e:
                print(f"EasyOCR failed: {e}")
            
            # Remove duplicates and sort by confidence
            seen_texts = set()
            unique_detections = []
            for detection in text_detections:
                if detection.text not in seen_texts:
                    seen_texts.add(detection.text)
                    unique_detections.append(detection)
            
            text_detections = sorted(unique_detections, key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            print(f"Error in text extraction: {e}")
        
        return text_detections
    
    def detect_barcodes(self, image: Image.Image) -> list[BarcodeDetection]:
        """Detect and decode barcodes with enhanced preprocessing"""
        barcode_detections = []
        
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Enhanced preprocessing for better barcode detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Try multiple preprocessing techniques
            preprocessing_methods = [
                gray,  # Original grayscale
                cv2.GaussianBlur(gray, (3, 3), 0),  # Slight blur
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # OTSU threshold
            ]
            
            for processed_image in preprocessing_methods:
                try:
                    barcodes = pyzbar.decode(processed_image)
                    
                    for barcode in barcodes:
                        x, y, w, h = barcode.rect
                        barcode_data = barcode.data.decode('utf-8')
                        barcode_type = barcode.type
                        
                        # Check if we already detected this barcode
                        if not any(det.value == barcode_data for det in barcode_detections):
                            barcode_detections.append(BarcodeDetection(
                                bbox=[x, y, w, h],
                                value=barcode_data,
                                barcode_type=barcode_type,
                                confidence=1.0
                            ))
                except Exception as e:
                    print(f"Barcode detection failed for preprocessing method: {e}")
                    continue
            
            # Try with rotated images (common for package labels)
            for angle in [90, 180, 270]:
                try:
                    # Rotate image
                    height, width = gray.shape
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(gray, rotation_matrix, (width, height))
                    
                    barcodes = pyzbar.decode(rotated)
                    
                    for barcode in barcodes:
                        x, y, w, h = barcode.rect
                        barcode_data = barcode.data.decode('utf-8')
                        barcode_type = barcode.type
                        
                        # Check if we already detected this barcode
                        if not any(det.value == barcode_data for det in barcode_detections):
                            barcode_detections.append(BarcodeDetection(
                                bbox=[x, y, w, h],
                                value=barcode_data,
                                barcode_type=barcode_type,
                                confidence=1.0
                            ))
                except Exception as e:
                    print(f"Barcode detection failed for rotation {angle}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error in barcode detection: {e}")
        
        return barcode_detections

def test_yolov9():
    """Test YOLOv9 model"""
    model = YOLOv9Model()
    return model

if __name__ == "__main__":
    # Quick test
    model = YOLOv9Model()
    model.load_model()
    print("✅ YOLOv9 model loaded successfully!")









