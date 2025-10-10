"""
MobileSAM Model Implementation
Lightweight segmentation model optimized for mobile devices
"""

import cv2
import numpy as np
from PIL import Image
import torch
import pytesseract
from pyzbar import pyzbar
from vision_test_framework import VisionModel, TextDetection, BarcodeDetection

try:
    from segment_anything import sam_model_registry, SamPredictor
    from segment_anything.utils.transforms import ResizeLongestSide
    MOBILESAM_AVAILABLE = True
except ImportError:
    MOBILESAM_AVAILABLE = False
    print("Warning: MobileSAM not available. Install segment-anything package.")

class MobileSAMModel(VisionModel):
    """MobileSAM implementation for segmentation-based object detection"""
    
    def __init__(self):
        super().__init__("MobileSAM")
        self.sam_predictor = None
        self.sam_model = None
        
    def load_model(self):
        """Load MobileSAM model"""
        if not MOBILESAM_AVAILABLE:
            raise ImportError("MobileSAM not available. Install segment-anything package.")
        
        try:
            # Load MobileSAM model
            sam_checkpoint = "mobile_sam.pt"  # You need to download this
            model_type = "vit_t"  # MobileSAM uses ViT-Tiny
            
            # Try to load the model
            try:
                self.sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                self.sam_model.to(device=self.device)
                self.sam_predictor = SamPredictor(self.sam_model)
            except FileNotFoundError:
                print("MobileSAM checkpoint not found. Using fallback object detection.")
                # Fallback to OpenCV-based object detection
                self.sam_model = None
                self.sam_predictor = None
                
        except Exception as e:
            print(f"Error loading MobileSAM model: {e}")
            # Fallback to basic OpenCV detection
            self.sam_model = None
            self.sam_predictor = None
    
    def describe_image(self, image: Image.Image) -> str:
        """Generate description using segmentation"""
        description_parts = []
        
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if self.sam_predictor is not None:
            try:
                # Use MobileSAM for segmentation
                self.sam_predictor.set_image(cv_image)
                
                # Generate automatic masks
                masks, scores, logits = self.sam_predictor.predict()
                
                description_parts.append(f"Detected {len(masks)} segmented regions")
                
                # Analyze segmented regions
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if score > 0.5:  # High confidence segments
                        # Calculate region properties
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if contours:
                            area = cv2.contourArea(contours[0])
                            description_parts.append(f"Region {i+1}: area={area:.0f} pixels, confidence={score:.2f}")
                
            except Exception as e:
                description_parts.append(f"Segmentation failed: {e}")
                # Fallback to basic description
                description_parts.append("Using fallback object detection")
        
        # Fallback: Use OpenCV for basic object detection
        if not description_parts or "fallback" in description_parts[-1]:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            description_parts.append(f"Detected {len(contours)} edge contours")
            
            # Detect shapes
            shapes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small contours
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    if len(approx) == 3:
                        shapes.append("triangle")
                    elif len(approx) == 4:
                        shapes.append("rectangle")
                    elif len(approx) > 8:
                        shapes.append("circle")
            
            if shapes:
                description_parts.append(f"Detected shapes: {', '.join(set(shapes))}")
        
        # Add image dimensions
        width, height = image.size
        description_parts.append(f"Image dimensions: {width}x{height} pixels")
        
        return ". ".join(description_parts)
    
    def extract_text(self, image: Image.Image) -> list[TextDetection]:
        """Extract text using segmentation-guided OCR"""
        text_detections = []
        
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            if self.sam_predictor is not None:
                try:
                    # Use MobileSAM to segment potential text regions
                    self.sam_predictor.set_image(cv_image)
                    masks, scores, logits = self.sam_predictor.predict()
                    
                    for mask, score in zip(masks, scores):
                        if score > 0.3:  # Lower threshold for text detection
                            # Convert mask to bounding box
                            coords = np.where(mask)
                            if len(coords[0]) > 0:
                                y_min, y_max = coords[0].min(), coords[0].max()
                                x_min, x_max = coords[1].min(), coords[1].max()
                                
                                # Crop text region
                                text_region = image.crop((x_min, y_min, x_max, y_max))
                                
                                # Apply OCR
                                text = pytesseract.image_to_string(text_region, config='--psm 8')
                                text = text.strip()
                                
                                if text and len(text) > 1:  # Filter out single characters
                                    text_detections.append(TextDetection(
                                        text=text,
                                        bbox=[x_min, y_min, x_max-x_min, y_max-y_min],
                                        confidence=float(score)
                                    ))
                
                except Exception as e:
                    print(f"MobileSAM text extraction failed: {e}")
            
            # Fallback: Use EasyOCR for better text detection
            if not text_detections:
                try:
                    import easyocr
                    reader = easyocr.Reader(['en'])
                    results = reader.readtext(np.array(image))
                    
                    for (bbox, text, confidence) in results:
                        if confidence > 0.5:
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            x1, x2 = min(x_coords), max(x_coords)
                            y1, y2 = min(y_coords), max(y_coords)
                            
                            text_detections.append(TextDetection(
                                text=text,
                                bbox=[int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                confidence=confidence
                            ))
                except ImportError:
                    # Final fallback to Tesseract on full image
                    text = pytesseract.image_to_string(image)
                    if text.strip():
                        text_detections.append(TextDetection(
                            text=text.strip(),
                            bbox=[0, 0, image.width, image.height],
                            confidence=0.5
                        ))
        
        except Exception as e:
            print(f"Error in text extraction: {e}")
        
        return text_detections
    
    def detect_barcodes(self, image: Image.Image) -> list[BarcodeDetection]:
        """Detect barcodes using segmentation-guided detection"""
        barcode_detections = []
        
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            if self.sam_predictor is not None:
                try:
                    # Use MobileSAM to segment potential barcode regions
                    self.sam_predictor.set_image(cv_image)
                    masks, scores, logits = self.sam_predictor.predict()
                    
                    for mask, score in zip(masks, scores):
                        if score > 0.3:
                            # Convert mask to bounding box
                            coords = np.where(mask)
                            if len(coords[0]) > 0:
                                y_min, y_max = coords[0].min(), coords[0].max()
                                x_min, x_max = coords[1].min(), coords[1].max()
                                
                                # Crop barcode region
                                barcode_region = cv_image[y_min:y_max, x_min:x_max]
                                
                                # Try to decode barcode
                                barcodes = pyzbar.decode(barcode_region)
                                
                                if barcodes:
                                    barcode_data = barcodes[0].data.decode('utf-8')
                                    barcode_type = barcodes[0].type
                                else:
                                    barcode_data = None
                                    barcode_type = None
                                
                                barcode_detections.append(BarcodeDetection(
                                    bbox=[x_min, y_min, x_max-x_min, y_max-y_min],
                                    value=barcode_data,
                                    barcode_type=barcode_type,
                                    confidence=float(score)
                                ))
                
                except Exception as e:
                    print(f"MobileSAM barcode detection failed: {e}")
            
            # Fallback: Direct barcode detection on full image
            if not barcode_detections:
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

# Test function
def test_mobilesam():
    """Test MobileSAM model with sample images"""
    from vision_test_framework import VisionModelTester
    
    tester = VisionModelTester()
    tester.add_model(MobileSAMModel())
    
    results = tester.run_tests()
    tester.save_results(results)
    tester.print_summary(results)

if __name__ == "__main__":
    test_mobilesam()
