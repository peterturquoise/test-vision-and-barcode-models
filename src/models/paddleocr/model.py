#!/usr/bin/env python3
"""
PaddleOCR Model Implementation
High-performance OCR with Docker deployment
"""

import cv2
import numpy as np
from PIL import Image
import sys
import re
from pathlib import Path

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    print("PaddleOCR available")
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("Warning: PaddleOCR not available. Install paddlepaddle and paddleocr packages.")

class TextDetection:
    """Text detection result"""
    def __init__(self, text, bbox, confidence):
        self.text = text
        self.bbox = bbox
        self.confidence = confidence

class PaddleOCRModel:
    """PaddleOCR implementation for high-performance text extraction"""
    
    def __init__(self):
        self.device = "cpu"  # PaddleOCR handles GPU internally
        self.ocr = None
        self.load_model()
    
    def detect_barcode_regions(self, image_cv):
        """
        Specialized function to detect barcode regions using image processing
        Focuses on finding long numeric sequences that PaddleOCR might miss
        """
        print("DEBUG: detect_barcode_regions - Starting barcode detection...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold to enhance text
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        barcode_candidates = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter for potential barcode regions (long horizontal rectangles)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # Look for long horizontal regions that could contain barcodes
            if (aspect_ratio > 3 and  # Long horizontal shape
                area > 1000 and       # Minimum area
                h > 15 and h < 50 and # Reasonable height for text
                w > 100):             # Minimum width for long numbers
                
                # Extract the region
                roi = thresh[y:y+h, x:x+w]
                
                # Try to extract text from this region using PaddleOCR
                try:
                    # Convert ROI back to BGR for PaddleOCR
                    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    
                    # Run PaddleOCR on this specific region
                    result = self.ocr.ocr(roi_bgr, cls=False)
                    
                    if result and result[0]:
                        for line in result[0]:
                            if len(line) >= 2:
                                bbox, text_info = line
                                if isinstance(text_info, tuple):
                                    text = text_info[0]
                                    confidence = text_info[1]
                                else:
                                    text = str(text_info)
                                    confidence = 0.9
                                
                                # Check if this looks like a barcode (long numeric sequence)
                                if (len(text) >= 15 and 
                                    re.match(r'^[0-9]+$', text) and 
                                    confidence > 0.3):
                                    
                                    print(f"🔍 BARCODE CANDIDATE FOUND: '{text}' (length: {len(text)}, confidence: {confidence:.3f})")
                                    
                                    # Convert local coordinates to global coordinates
                                    global_bbox = [
                                        [x + bbox[0][0], y + bbox[0][1]],
                                        [x + bbox[1][0], y + bbox[1][1]],
                                        [x + bbox[2][0], y + bbox[2][1]],
                                        [x + bbox[3][0], y + bbox[3][1]]
                                    ]
                                    
                                    barcode_candidates.append({
                                        'text': text,
                                        'bbox': global_bbox,
                                        'confidence': confidence
                                    })
                
                except Exception as e:
                    print(f"DEBUG: detect_barcode_regions - Error processing ROI: {e}")
                    continue
        
        print(f"DEBUG: detect_barcode_regions - Found {len(barcode_candidates)} barcode candidates")
        return barcode_candidates
    
    def load_model(self):
        """Load PaddleOCR model"""
        try:
            if PADDLEOCR_AVAILABLE:
                # Initialize PaddleOCR with optimized settings for shipping labels (based on working code)
                self.ocr = PaddleOCR(
                    use_angle_cls=True,  # Enable text angle classification
                    lang='en',  # English language
                    show_log=True,  # Enable logging to debug
                    use_gpu=False,  # Use CPU for consistency
                    det_limit_side_len=1216,  # Higher resolution processing
                    det_limit_type='min',  # Process at higher resolution
                    use_space_char=True,  # Enable space recognition
                    det_box_thresh=0.1,  # Much lower threshold for detection
                    det_unclip_ratio=3.0,  # Expand text boxes even more
                    drop_score=0.01,  # Very low drop score to catch more detections
                    det_db_thresh=0.1,  # Lower DB threshold
                    det_db_box_thresh=0.1,  # Lower DB box threshold
                    max_text_length=50  # Allow longer text sequences
                )
                print(f"PaddleOCR model loaded successfully on {self.device}")
            else:
                print("Error: PaddleOCR not available")
        except Exception as e:
            print(f"Error loading PaddleOCR model: {e}")
    
    def describe_image(self, image):
        """Describe the image content focusing on package details"""
        try:
            if not self.ocr:
                return "PaddleOCR model not loaded"
            
            # Convert PIL to OpenCV format
            if isinstance(image, Image.Image):
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                image_cv = image
            
            # Apply image enhancement (from working code)
            enhanced = cv2.convertScaleAbs(image_cv, alpha=1.5, beta=0)
            
            # Extract text using PaddleOCR
            print("DEBUG: Starting PaddleOCR extraction...")
            results = self.ocr.ocr(enhanced, cls=True)
            
            print(f"DEBUG: PaddleOCR raw results type: {type(results)}")
            print(f"DEBUG: PaddleOCR raw results: {results}")
            
            if not results or not results[0]:
                print("DEBUG: No results from PaddleOCR")
                return "No text detected in the image"
            
            # Build description focusing on package details
            description_parts = []
            description_parts.append("Package/shipping label detected with the following text content:")
            
            print(f"DEBUG: Results[0] type: {type(results[0])}")
            print(f"DEBUG: Results[0] length: {len(results[0])}")
            print(f"DEBUG: Results[0] content: {results[0]}")
            
            for i, line in enumerate(results[0]):
                print(f"DEBUG: Processing detection {i+1}: {line}")
                print(f"DEBUG: Detection {i+1} type: {type(line)}")
                print(f"DEBUG: Detection {i+1} length: {len(line) if line else 'None'}")
                
                if line and len(line) >= 2:
                    print(f"DEBUG: Detection {i+1} line[1]: {line[1]}")
                    print(f"DEBUG: Detection {i+1} line[1] type: {type(line[1])}")
                    
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    confidence = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.9
                    
                    print(f"DEBUG: Detection {i+1} - EXTRACTED TEXT: '{text}' (confidence: {confidence:.3f})")
                    
                    # Include ALL text for debugging
                    description_parts.append(f"- {text} (confidence: {confidence:.2f})")
                else:
                    print(f"DEBUG: Detection {i+1} - INVALID FORMAT: {line}")
            
            return "\n".join(description_parts)
            
        except Exception as e:
            return f"Error describing image: {str(e)}"
    
    def extract_text(self, image):
        """Extract text from image with bounding boxes"""
        try:
            if not self.ocr:
                return []
            
            # Convert PIL to OpenCV format
            if isinstance(image, Image.Image):
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                image_cv = image
            
            # Apply image enhancement (from working code)
            enhanced = cv2.convertScaleAbs(image_cv, alpha=1.5, beta=0)
            
            # Extract text using PaddleOCR
            print("DEBUG: Starting PaddleOCR extraction in extract_text...")
            results = self.ocr.ocr(enhanced, cls=True)
            
            print(f"DEBUG: extract_text - PaddleOCR raw results: {results}")
            
            text_detections = []
            
            if results and results[0]:
                print(f"DEBUG: extract_text - Processing {len(results[0])} detections")
                for i, line in enumerate(results[0]):
                    print(f"DEBUG: extract_text - Processing detection {i+1}: {line}")
                    if line and len(line) >= 2:
                        # Extract bounding box coordinates
                        bbox_coords = line[0]
                        if len(bbox_coords) >= 4:
                            # Convert to [x, y, width, height] format
                            x_coords = [point[0] for point in bbox_coords]
                            y_coords = [point[1] for point in bbox_coords]
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                        else:
                            bbox = [0, 0, 0, 0]
                        
                        # Extract text and confidence
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                        else:
                            text = str(text_info)
                            confidence = 0.9
                        
                        # Include ALL detections for debugging
                        print(f"DEBUG: extract_text - EXTRACTED TEXT: '{text}' with confidence {confidence:.3f}")
                        
                        # Check if this might be a barcode (long numeric sequence)
                        if len(text) >= 20 and text.isdigit():
                            print(f"🔍 POTENTIAL BARCODE DETECTED: '{text}' (length: {len(text)})")
                        
                        text_detections.append(TextDetection(text, bbox, confidence))
                    else:
                        print(f"DEBUG: extract_text - INVALID FORMAT detection {i+1}: {line}")
            else:
                print("DEBUG: extract_text - No results or empty results")
            
            # Try specialized barcode detection for long numeric sequences
            print("DEBUG: extract_text - Running specialized barcode detection...")
            barcode_candidates = self.detect_barcode_regions(enhanced)
            
            # Add barcode candidates to results
            for candidate in barcode_candidates:
                print(f"DEBUG: extract_text - Adding barcode candidate: '{candidate['text']}'")
                # Convert bbox format to match TextDetection format
                bbox_coords = candidate['bbox']
                x_coords = [point[0] for point in bbox_coords]
                y_coords = [point[1] for point in bbox_coords]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                
                text_detections.append(TextDetection(
                    candidate['text'], 
                    bbox, 
                    candidate['confidence']
                ))
            
            return text_detections
            
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return []
    